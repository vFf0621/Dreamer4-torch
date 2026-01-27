import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from utils import *
from tqdm import tqdm
from typing import Optional, Dict, Any
import cv2
from collections import deque
import numpy as np
import math
class Policy(nn.Module):
    def __init__(self, action_dim: int = 3, latent_dim: int = 128, mtp: int = 8, num_bins: int = 50):
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.mtp = mtp
        
        # Helper for binning math
        self.binner = ActionBinning(bins=num_bins)

        # 1. Main Network
        # Output size: action_dim * num_bins
        self.network = build_network(latent_dim, latent_dim*2, 2, "SwiGLU", latent_dim*4)
        
        # BC Head (Multi-Step / MTP)
        # Output: [MTP, ActionDim, Bins]
        self.head = nn.Sequential(
                SwiGLU(), 
                nn.RMSNorm(latent_dim*2), 
                nn.Linear(latent_dim*2, action_dim * num_bins * mtp)
            )
        self.agent_token = nn.Parameter(0.02*torch.randn(1,1,1,latent_dim))
        self.eps = 1e-2

    def forward(self, posterior: torch.Tensor, sample=True):
        """
        posterior: [B, L, D]
        Returns:
            action: [B, L, A] (Continuous value)
            logits: [B, L, A, Bins]
            dist: Categorical Distribution object
        """
        B, L, D = posterior.shape
        
        # Features
        x = self.network(posterior)

        # --- 1. Main Policy Head ---
        logits = self.head(x)
        logits = logits.reshape(B, L,self.mtp, self.action_dim, self.num_bins)

        probs = F.softmax(logits, dim=-1)
        probs = (1 - self.eps) * probs + self.eps / probs.size(-1)

        # --- 2. MTP Head (Auxiliary) ---
        # [B, L, MTP, A, Bins]

        # --- 3. Sampling ---
        dist = td.Categorical(probs=probs)
        # For training stability, we often use expectation or straight-through sampling
        dist_act = td.Categorical(probs=probs[:,:,0])

        idx = dist_act.sample() # [B, L, A]
        action = self.binner.centers[idx]
       
        log_prob = dist_act.log_prob(idx).sum(dim=-1, keepdim=True) # Sum over action dims
        
        return action, log_prob, dist, dist_act, idx

# ---- Highway-gated residual wrapper ---
class GQA(nn.Module):
    def __init__(self, embed_dim=16, num_heads=8, num_kv_heads=4, dropout=0.1, causal=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.norm = nn.RMSNorm(embed_dim)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        if causal:
            self.rope = RoPE1D(self.head_dim)
        self.dropout = dropout

    def forward(self, x, x_k=None, attn_mask=None, key_padding_mask=None):
        B, Tq, D = x.shape
        xq = self.norm(x)
        xk_in = x_k if x_k is not None else xq
        Tk = xk_in.shape[1]

        q = self.q_proj(xq)    
        k = self.k_proj(xk_in) 
        v = self.v_proj(xk_in) 

        q = q.view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)      
        k = k.view(B, Tk, self.num_kv_heads, self.head_dim).transpose(1, 2)   
        v = v.view(B, Tk, self.num_kv_heads, self.head_dim).transpose(1, 2)   

        if self.causal:
            cos, sin = self.rope(q, seq_len=Tq)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_mask_ = None
        if attn_mask is not None:
            attn_mask_ = attn_mask
            if attn_mask_.dim() == 2:
                attn_mask_ = attn_mask_[None, None, :, :]   
            elif attn_mask_.dim() == 3:
                attn_mask_ = attn_mask_[:, None, :, :]      

        if self.causal:
            causal = torch.ones((Tq, Tk), dtype=torch.bool, device=q.device).triu(1)  
            causal = causal[None, None, :, :]  
            attn_mask_ = causal if attn_mask_ is None else (attn_mask_ | causal)

        if key_padding_mask is not None:
            kpm = key_padding_mask[:, None, None, :]  
            attn_mask_ = kpm if attn_mask_ is None else (attn_mask_ | kpm)
        if attn_mask is not None:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask_,
                dropout_p=0.,
                is_causal=False,
                enable_gqa=True,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.,
                is_causal=self.causal,
                enable_gqa=True,
            )

        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        return self.out_proj(out)


class CausalSTBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, num_reserved=2, time_attn=True, cap_value=50,  device="cuda"):
        super().__init__()
        self.time_attn_enabled = time_attn
        self.d_model = d_model
        self.ln_space = nn.RMSNorm(d_model)

        if not self.time_attn_enabled:
            self.space_attn = GQA(d_model, n_heads, n_heads // 2, dropout, )
        else:
            self.time_attn = GQA(d_model, n_heads, n_heads // 2, dropout, causal=True)

        self.num_reserved = num_reserved
        self.ln_time = nn.RMSNorm(d_model)
        self.reserved_tokens = nn.Parameter(torch.zeros(1, num_reserved, int(d_model)))
        self.mlp = build_network(d_model, d_model * 2, 3, "SwiGLU", d_model, True)
        self.device = device
        self.to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        token_pad_mask: torch.Tensor | None = None,
        *,
        agent_idx: int | None = None,   
        mask = None
    ) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(2)  # [B,T,1,D]

        reserved = self.reserved_tokens
        B, T, N, D = x.shape
        R = reserved.shape[1]

        if token_pad_mask is not None:
            assert token_pad_mask.shape[:2] == (B, T), (token_pad_mask.shape, (B, T))
            if token_pad_mask.size(2) >= N:
                token_pad_mask = token_pad_mask[:, :, :N]
            else:
                pad = torch.zeros(B, T, N - token_pad_mask.size(2), dtype=torch.bool, device=x.device)
                token_pad_mask = torch.cat([token_pad_mask.to(x.device), pad], dim=2)

        if agent_idx is not None and not (0 <= agent_idx < N):
            raise ValueError(f"agent_idx={agent_idx} out of range for N={N}")

        if self.time_attn_enabled:
            x_time = self.ln_time(x).permute(0, 2, 1, 3).reshape(B * N, T, D)

            time_kpm = None
            if token_pad_mask is not None:
                time_kpm = token_pad_mask.permute(0, 2, 1).reshape(B * N, T)
            if mask is None:
        

                xt_out = self.time_attn(x_time, attn_mask=None, key_padding_mask=time_kpm)  
            else:
                xt_out = self.time_attn(x_time, attn_mask=mask, key_padding_mask=time_kpm)  


            xt_out = xt_out.reshape(B, N, T, D).permute(0, 2, 1, 3)

            
            x = x + xt_out

            x = x + self.mlp(x)
            return x.squeeze(2) if x.shape[2] == 1 else x

        x_space = self.ln_space(x)
        xt = x_space.reshape(B * T, N, D)  

        space_kpm = None
        if token_pad_mask is not None:
            space_kpm = token_pad_mask.reshape(B * T, N)  
            space_kpm = torch.cat(
                [space_kpm, torch.zeros(B * T, R, dtype=torch.bool, device=x.device)],
                dim=1,
            ) 

        
        xcat = torch.cat([xt, reserved.expand(B * T, R, D)], dim=1)  # [B*T, N+R, D]

        # 1) start with caller-provided mask (e.g., modality mask)
        # --- build final attention mask over xcat = [tokens (N) | reserved (R)] ---
        Ncat = N + R
        final_mask = None

        if mask is not None:
            if mask.shape[-2:] == (N, N):
                m = torch.zeros((Ncat, Ncat), dtype=torch.bool, device=x.device)
                m[:N, :N] = mask  # embed caller mask on real tokens
                # by default: allow attention to/from reserved tokens (block False)
                final_mask = m
            elif mask.shape[-2:] == (Ncat, Ncat):
                final_mask = mask
            else:
                raise ValueError(
                    f"mask has shape {mask.shape}, expected ({N},{N}) or ({Ncat},{Ncat})"
                )
        else:
            # no mask from caller: default allow-all (block nothing)
            final_mask = torch.zeros((Ncat, Ncat), dtype=torch.bool, device=x.device)

        # --- one-way agent rule: agent can't read others; others can read agent ---
        #
      #  save_attention_mask(final_mask, "attn_mask.png")
        xs = self.space_attn(xcat, attn_mask=final_mask, key_padding_mask=space_kpm)[:, :N]

        x = x + xs.reshape(B, T, N, D)
    
        x = x + self.mlp(x)
        return x.squeeze(2) if x.shape[2] == 1 else x

class Encoder(nn.Module):
    def __init__(
        self,
        img_channels: int = 3,
        h:int = 96,
        w:int = 96,
        patch: int = 16,
        d_model: int = 256,
        n_heads: int = 8,
        depth: int = 6,
        latent_tokens: int = 64,
        time_every: int = 2,
        dropout: float = 0.05,
        out_dim: int = 16,     # Dz
        max_T: int = 256,
        pool: str = "first",      # "mean" or "first"
    ):
        super().__init__()
        assert (h % patch == 0) and (w % patch == 0)
        assert pool in ("mean", "first")
        self.pool = pool
        self.z_dim = out_dim
        self.patch = patch
        self.latent_tokens = latent_tokens
        self.d_model = d_model
        grid = (h // patch, w//patch)
        self.num_patches = grid[0] * grid[1]
        N = self.num_patches + latent_tokens
        patch_dim = img_channels * patch * patch
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.latent_tok = nn.Parameter(torch.randn(1, 1, latent_tokens, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.pos_emb_lat = nn.Parameter(torch.randn(1, 1, self.latent_tokens + self.num_patches, d_model) * 0.02)
        blocks = []
        for i in range(depth):
            use_time = ((i+1) % time_every == 0)
            blocks.append(CausalSTBlock(d_model, n_heads, dropout=dropout, time_attn=use_time))
        self.blocks = nn.ModuleList(blocks)

        self.ln_out = nn.RMSNorm(d_model)
        self.readout = nn.Linear(d_model, out_dim)

    def forward(self, frames: torch.Tensor, return_tokens= True) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        p = self.patch
        assert H % p == 0 and W % p == 0, (H, W, p)

        x = frames.reshape(B * T, C, H, W)                                   # (B*T,C,H,W)
        patches = F.unfold(x, kernel_size=p, stride=p)                       # (B*T, C*p*p, Np)
        patches = patches.transpose(1, 2).contiguous()                       # (B*T, Np, patch_dim)
        patches = patches.view(B, T, self.num_patches, -1)                   # (B,T,Np,patch_dim)
        space_mask = modality_mask(
            L=self.latent_tokens,
            modality_sizes=[self.num_patches],
            device=frames.device
        ) 
        
        proj = self.patch_proj(patches)                             # (B,T,Np,D)
        lat = self.latent_tok.view(1, 1, self.latent_tokens, self.d_model).expand(B, T, -1, -1) 
        x = torch.cat([lat, proj], dim=2) +self.pos_emb_lat                             # (B,T,S,D) with S=L+Np      

        x = self.drop(x)
        for blk in self.blocks:
            if blk.time_attn_enabled:
                x = blk(x, mask=None,)          # no space mask here
            else:
                x = blk(x, mask=space_mask)    # modality mask only here
        x = self.ln_out(x)  

        patch_tok = x[:, :,: self.latent_tokens, :]
        pre = (self.readout(patch_tok))
        ztok = torch.tanh(pre)   # [B,T,Np,Dz]
        return ztok


class TokenDynamics(nn.Module):
    """
    Token-based dynamics with discrete signal embeddings + discrete action embeddings.

    Inputs:
      z_tokens: [B, T, Nz, Dz]
      signals:  [B, T, Nz, 2]   (signals[...,0]=level_idx (tau_idx), signals[...,1]=step_idx (k_idx))  (int/long)
                (Legacy accepted: [B,T,Nz] => interpreted as level_idx, step_idx=0)
      actions:  [B, T-1, A] or [B, T, A] continuous in [-1, 1]

    Outputs:
      z_pred:      [B, T, Nz, Dz]
      policy_feat: [B, 1, D]  (agent token at last timestep, if injected)
    """

    def __init__(
        self,
        Dz: int = 16,
        action_dim: int = 2,
        action_bins: int = 100,

        level_vocab: int = 128,
        level_dim: int = 16,

        step_vocab: int = 128,
        num_tasks: int = 10,
        d_model: int = 512,
        n_heads: int = 4,
        depth: int = 16,
        time_every: int = 4,
        dropout: float = 0.1,
        Sa: int = 64,
        max_T: int = 255,
        use_agent_token: bool = True,
        latent_tokens: int = 32,
        device: str = "cuda",
        action_lookup: bool = True,

        # behavior toggles
        mask_last_action: bool = True,      # usually correct for "predict next"
        clamp_signal_indices: bool = False, # set True if you’d rather clamp than crash
    ):
        super().__init__()
        self.max_T = max_T
        self.use_agent_token = use_agent_token
        self.device = device
        self.Sa = Sa
        self.action_dim = action_dim
        self.action_bins = action_bins
        self.d_model = d_model
        self.Nz = latent_tokens
        self.num_task = num_tasks
        self.mask_last_action = mask_last_action
        self.clamp_signal_indices = clamp_signal_indices
        # --- Discrete signal embeddings (tau_idx + k_idx) ---
        self.level_vocab = int(level_vocab)
        self.step_vocab = int(step_vocab)
        self.space_pos_embed = nn.Parameter(0.02 * torch.randn(1, 1, self.Sa + self.Nz+2, d_model))
        self.z_proj = nn.Sequential(nn.RMSNorm(Dz), nn.Linear(Dz, d_model))
        self.sig_proj = nn.Sequential(nn.RMSNorm(level_dim), nn.Linear(level_dim, d_model))

        self.level_emb = nn.Embedding(self.level_vocab, level_dim)
        self.step_emb  = nn.Embedding(self.step_vocab, level_dim)
        self.action_conditioner = nn.Parameter(0.02 * torch.randn(1, 1, self.Sa, d_model))

        self.action_pad = nn.Parameter( torch.zeros(1, 1, 1, d_model))
        self.action_lookup = action_lookup

        # --- Action embedding (true lookup, no one-hot materialization) ---
        self.action_embs = nn.ModuleList([nn.Embedding(action_bins, self.d_model) for _ in range(self.action_dim)] )
        self.action_normalize = nn.RMSNorm(d_model)
        # --- z + signal -> model dim --

        # Agent token (learned)
        self.agent_token = nn.Parameter(0.02 * torch.randn(1, 1, num_tasks, d_model))

        # Blocks
        blocks = []
        for i in range(depth):
            use_time = ((i+1) % time_every == (0))
            blocks.append(
                CausalSTBlock(
                    d_model, n_heads,
                    dropout=dropout,
                    time_attn=use_time,
                    device=device,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Output head: model dim -> Dz
        self.out = build_network(d_model, 2*d_model, 3, "SwiGLU", Dz)

        self.to(device)
    def interleave_obs_and_actions(self, z_emb, a_emb):
            """
            Interleaves the first Sa observation tokens with actions.
            Appends the remaining N-Sa tokens at the end.
            """
            B, T, N, D = z_emb.shape
            Sa = self.Sa
            assert self.Sa <= N

            # 1. Ensure action has the 'N' dim
            if a_emb.dim() == 3:
                a_emb = a_emb.unsqueeze(2)  # (B, T, 1, D)

            # 2. Split z_emb into "paired" and "unpaired"
            z_paired = z_emb[:, :, :Sa]      # Shape: (B, T, Sa, D)
            z_rest   = z_emb[:, :, Sa:]      # Shape: (B, T, N-Sa, D)

            # 3. Expand Actions to match Sa
            # Note: We expand to 'z_paired', not the full 'z_emb'
            a_emb = a_emb.expand_as(z_paired) + self.action_conditioner

            # 4. Stack and Flatten the Paired Section
            # stack -> (B, T, Sa, 2, D)
            # flatten -> (B, T, 2*Sa, D) -> [z0, a0, z1, a1, ... z_Sa, a_Sa]
            paired = torch.stack((z_paired, a_emb), dim=3).flatten(2, 3)

            # 5. Concatenate the Rest
            # Result -> [z0, a0, ... z_Sa, a_Sa, z_Sa+1, z_Sa+2 ...]
            interleaved = torch.cat([paired, z_rest], dim=2)
            
            return interleaved

# Forwar 
    # -----------------------------
    # Discretization helpers
    # -----------------------------
    def discretize_actions_to_indices(self, actions: torch.Tensor) -> torch.Tensor:
        """
        actions: [B, T, A] in [-1, 1]
        returns: [B, T, A] long in [0, action_bins-1]
        """
        
        normed = (actions.clamp(-1,1) + 1.0) * 0.5  # [0,1]
        idx = (normed * (self.action_bins-1)).round().long()
        return idx

    def align_actions(self, actions, T, B, dtype, device):
        if actions is None:
            return torch.zeros((B, T-1, self.action_dim), device=device, dtype=dtype)

        if actions.dim() != 3:
            raise ValueError(...)
        Ba, Ta, A = actions.shape
        if Ba != B or A != self.action_dim:
            raise ValueError(...)

        if Ta == T:
            return actions[:, :-1]
        if Ta == T-1:
            return actions

    def _validate_or_clamp_index(self, idx: torch.Tensor, vocab: int, name: str) -> torch.Tensor:
        if self.clamp_signal_indices:
            return idx.clamp(0, vocab - 1)

        # strict mode: crash early with an informative message
        if (idx.min() < 0) or (idx.max() >= vocab):            
            raise ValueError(
                f"{name} out of range: min={int(idx.min())}, max={int(idx.max())}, vocab={int(vocab)}"
            )
        return idx

    def agent_mask(self, S: int, agent_idx: int, device=None) -> torch.Tensor:
        allow = torch.ones((S, S), dtype=torch.bool, device=device)
        allow[:, agent_idx] = False
        allow[agent_idx, :] = True
        allow[agent_idx, agent_idx] = True        
        return ~allow

            # Forwar
    # -----------------------------
    def forward(
        self,
        z_tokens: torch.Tensor,                 # [B, T, Nz, Dz]
        actions: torch.Tensor | None,           # [B, T-1, A] or [B, T, A]
        signals: torch.Tensor,                  # [B, T, Nz, 2] (level_idx, step_idx)
        policy_tok_in: torch.Tensor | None = None,  # [1,1,1,D] or [B,1,1,D] or [B,T,1,D]
        detach_agent: bool = False,
        last_z = None,
        task_id = 0, 
    ):
        device = z_tokens.device
        if z_tokens.dim() != 4:
            raise ValueError(f"z_tokens must be [B,T,Nz,Dz], got {tuple(z_tokens.shape)}")
        if last_z is not None:
            z_tokens = torch.cat([z_tokens, last_z], dim=1)
        B, T, Nz, Dz = z_tokens.shape

        if signals.size(1) != T:
            raise ValueError(f"signals has T={signals.size(1)} but z_tokens has T={T}.")
        acts_bt = self.align_actions(actions, T, B, z_tokens.dtype, device)   # [B, T-1, A]
        act_idx = self.discretize_actions_to_indices(acts_bt)                   # [B, T-1, A]

        a = 0
        for i, emb in enumerate(self.action_embs):
            a = a + emb(act_idx[..., i])                                        # [B, T-1, D]
        a = self.action_normalize(a) 
        a_emb = a[:, :,None, :]                                          # [B, T-1, 1, D]
        pad = self.action_pad.expand(B,1,1,-1)         # [B, 1, 1, D]
        a_emb = torch.cat([pad, a_emb], dim=1)                                  # [B, T, 1, D]
        signals = signals.long()
        if signals.dim() == 4 and signals.size(-1) == 2:
            level_idx, step_idx = signals[..., 0], signals[..., 1]
        elif signals.dim() == 3:
            level_idx, step_idx = signals, torch.zeros_like(signals)
        else:
            raise ValueError(f"signals must be [B,T,Nz,2] or [B,T,Nz], got {tuple(signals.shape)}")

        level_idx = self._validate_or_clamp_index(level_idx, self.level_vocab, "level_idx")
        step_idx  = self._validate_or_clamp_index(step_idx,  self.step_vocab,  "step_idx")
        level_t = level_idx[:, :, 0]                 # [B,T]
        step_t  = step_idx[:, :, 0]                  # [B,T]        
        lev_feat  = self.level_emb(level_t)
        step_feat = self.step_emb(step_t)
        z_inp = self.z_proj(z_tokens)
        # 2. Stack at dim=3 (immediately after N)
        # This places z and a side-by-side for every N element.
        # New Shape: (B, T, N, 2, D)
        x =self.interleave_obs_and_actions(z_inp, a_emb)
        # 3. Flatten the N dimension (dim 2) and the new stack dimension (dim 3)
        # New Shape: (B, T, N * 2, D)
        x = torch.cat([x, self.sig_proj(lev_feat + step_feat).unsqueeze(-2), ], dim=2)

        Nmain = x.size(2)

        token_pad_mask = torch.zeros(B, T, Nmain-1, device=device, dtype=torch.bool)  # True = PAD
        token_pad_mask[:, 0, 1:2*self.Sa:2] = True
        agent = None
        x_aug = x
        if policy_tok_in is not None:
            ag = policy_tok_in
            if ag.size(0) == 1 and B > 1: ag = ag.expand(B, -1, -1, -1)
            if ag.size(1) == 1 and T > 1: ag = ag.expand(B, T, -1, -1)
            if ag.size(2) != 1: raise ValueError("policy_tok_in token dim must be 1 at dim=2")
            agent = ag
            agent_in = agent.detach() if detach_agent else agent
        elif self.use_agent_token:
            assert task_id < self.num_task
            agent = self.agent_token[:, :, task_id].expand(B, T, 1, self.d_model)
            agent_in = agent.detach() if detach_agent else agent
        x_aug = torch.cat([x, agent_in], dim=2)
        agent_out_bt = None
        x_aug = x_aug + self.space_pos_embed.expand(B,1,-1,self.d_model)
        for blk in self.blocks:
            S = x_aug.size(2)
            agent_idx = S - 1
            tok_mask = self.agent_mask(S, agent_idx, device=device)
            pad_agent = torch.zeros(B, T, 2, dtype=torch.bool, device=device)
            token_pad_mask_aug = torch.cat([token_pad_mask, pad_agent], dim=2)
            
            if blk.time_attn_enabled:
                x_aug = blk(x_aug, token_pad_mask=token_pad_mask_aug, agent_idx=agent_idx)
            else:
                x_aug = blk(
                    x_aug,
                    token_pad_mask=token_pad_mask_aug,
                    mask=tok_mask,
                    agent_idx=agent_idx,
                )

            agent = x_aug[:, :, agent_idx:agent_idx+1, :]
            x = x_aug[:, :, :agent_idx, :]
        agent_out_bt = agent[:, :, 0, :]
        h_hist = torch.cat([x[:, :, 0:2*self.Sa:2, :], x[:,:,2*self.Sa:Nz+self.Sa]], 2)

        z_pred = self.out(h_hist)
        policy_feat = agent_out_bt[:, :, :] if agent_out_bt is not None else None
        return z_pred, policy_feat

class Dreamer4(nn.Module):
    def __init__(self,agent_id, ch=3, h=96, w=96, patch = 16, latent_tokens=32, z_dim=16, action_dim=2, latent_dim=512, 
                 rep_depth = 8, rep_d_model=256, dyn_d_model=256, num_heads=8, dropout=0.1, k_max=8, mtp=8, 
                 policy_bins = 100, reward_bins = 100, pretrain=False, reward_clamp=6,level_vocab = 16, level_embed_dim = 16,
                 batch_lens = (45, 65), batch_size=16, accum=1, max_imag_len=128, ckpt=None, rep_lr=1e-4, rep_decay=1e-3,Sa = 64,
                 dyn_lr=1e-4, dyn_decay=1e-3, ac_lr = 1e-4, ac_decay=1e-3, policy_lr=1e-4, policy_decay=1e-3, num_tasks=30, task_id = 0):
        super(Dreamer4, self).__init__()
        self.encoder =  Encoder(img_channels=ch, h=h, w=w, patch=patch, 
                                n_heads=num_heads, depth=rep_depth, latent_tokens=latent_tokens, time_every=2,
                                out_dim=z_dim, dropout=dropout, max_T=max_imag_len)
 
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.pretrain = False
        self.agent_id = agent_id

        # --- TokenDynamics ---
        # level_vocab, step_vocab match
        self.transformer = TokenDynamics(
            device=self.device, 
            use_agent_token=True, 
            level_vocab=level_vocab,
            level_dim=level_embed_dim,
            dropout=dropout,
            n_heads=num_heads,
            d_model=dyn_d_model,
            Sa = Sa, 
            max_T = max_imag_len,
            num_tasks=num_tasks,
            time_every=4,
            latent_tokens=latent_tokens
        )
        
        self.decoder = Decoder(img_channels=ch, w = w, h=h, patch=patch, z_dim=z_dim, d_model=rep_d_model, n_heads=num_heads,
                               depth=rep_depth, latent_tokens=latent_tokens, time_every=2, dropout=dropout, max_T=max_imag_len)
        self.imagination_steps = max_imag_len - 1
        self.rminv = -reward_clamp
        self.rmaxv = reward_clamp
        self.reward_bins = reward_bins
        self.task_id = task_id
        self.batch_lengths= batch_lens
        self.horizon_length = max_imag_len
        self.grad_accum = accum       
        self.batch_size = batch_size
        self.action_dim=action_dim
        self.shortcut_kmax = k_max
        self.z_buffer = None
        self.aux_horizon = mtp+1
        self.policy_num_bins = policy_bins
        self.bin_num = reward_bins
        self.steps = 0
        self.policy = Policy(action_dim=action_dim, latent_dim=dyn_d_model, mtp=self.aux_horizon, num_bins=self.policy_num_bins)
        self.t_policy = Policy(action_dim=action_dim, latent_dim=dyn_d_model, mtp=self.aux_horizon, num_bins=self.policy_num_bins)
        self.reward = Reward(bin_num=self.bin_num, latent_dim=dyn_d_model, mtp=self.aux_horizon, r_max=reward_clamp)
        self.train_policy = False
        self.value = Value(bin_num=self.bin_num, latent_dim=dyn_d_model, hidden_dim=latent_dim*2, r_max=reward_clamp)

        self.encoder.load_state_dict(torch.load("enc.pt"))
       
        self.decoder.load_state_dict(torch.load("dec.pt"))
        
        self.lpips = LPIPSLoss(reduction="none",)
        self.to(self.device)
        self.rep_optim = torch.optim.AdamW(
            [{'params': self.encoder.parameters()}, 
            {'params': self.decoder.parameters()},]
        , lr=rep_lr, capturable=True, weight_decay=rep_decay)
        self.reset()
        self.dyn_optim = torch.optim.AdamW(self.parameters(), lr=dyn_lr, capturable=True,weight_decay=dyn_decay)
        self.ac_optim = torch.optim.AdamW(
                [{'params': self.transformer.agent_token}, 
                {"params": self.policy.parameters()},
                {"params": self.reward.parameters()},]        , lr=ac_lr, capturable=True,weight_decay=ac_decay)

        self.t_policy.load_state_dict(self.policy.state_dict(), )

        self.policy_optim= torch.optim.AdamW([
            {'params': self.policy.parameters()}, 
            {'params': self.value.parameters()},
        ], lr=policy_lr, capturable=True,weight_decay=policy_decay)
        if ckpt:
            self.load_state_dict(torch.load(ckpt))

    def reset(self):
        self.state_buffer = None
        self.action_buffer = None


    def action_step(self, s):
        """
        One environment step -> one action.
        Uses clean-level signals for the transformer context (tau_idx = N-1, k_idx = 0).
        """
        with torch.no_grad():
            s = s.to(self.device)

            if self.state_buffer is None:
                self.state_buffer = s[None, None]
            else:
                self.state_buffer = torch.cat([self.state_buffer, s[None, None]], dim=1)

            z = self.encoder(self.state_buffer)  # [B,T,Nz,Dz]
            B, T, Nz, _ = z.shape

            # clean level for context
            N = int(getattr(self, "shortcut_kmax", 64))
            sigs = self.make_signals_indices(B, T, Nz, tau_idx=N-1, k_idx=0)

            _, h = self.transformer(z, self.action_buffer, signals=sigs)

            a, log_p, entropy, base_p, idx = self.policy(h[:, -1:])

            if self.action_buffer is None:
                self.action_buffer = a
            else:
                self.action_buffer = torch.cat([self.action_buffer, a], dim=1)

            # keep a rolling window
            W = 15
            if self.state_buffer.size(1) > W:
                self.state_buffer = self.state_buffer[:, -W:]
                if self.action_buffer is not None:
                    self.action_buffer = self.action_buffer[:, -W:]

            return a[0, 0].detach().cpu().numpy()


    def evaluate(self, buffer, steps=4): 
        s, a = buffer.sample_seq( 1, 64)[:2]
        s = torch.from_numpy(s).to(self.device).float()
        a = torch.from_numpy(a).to(self.device).float()
        
        with torch.no_grad():
            z = self.encoder(s)
            z_eval = self.latent_imagination(z[:,:], a[:,:], ctx_len=a.size(1), eval_=False,random=False, forced=True)[0]
            decoded = self.decode(z_eval).detach().cpu().numpy()
            for i in range(decoded.shape[1]):
                cv2.imwrite(f"./eval_imgs/imag_{i}.png", 255 * decoded[0, i].transpose(1, 2, 0)[..., ::-1])
        return
    def make_signals_indices(self, B, T, Nz, tau_idx=0, k_idx=0):
        """
        Create indices tensor [B, T, Nz, 2]:
        signals[..., 0] = tau_idx  (signal / level index)
        signals[..., 1] = k_idx    (step / skip index)
        Backwards compatible: existing callers that only pass k_idx still work (tau_idx defaults to 0).
        """
        device = self.device

        def _to_long_btNz(x):
            if torch.is_tensor(x):
                t = x.to(device=device, dtype=torch.long)
            else:
                t = torch.tensor(x, device=device, dtype=torch.long)

            if t.dim() == 0:
                return t.view(1, 1, 1).expand(B, T, Nz)
            if t.shape == (B, T, Nz):
                return t
            # allow [1,1,1] style broadcast
            if t.dim() == 3 and all((s == 1 or s == d) for s, d in zip(t.shape, (B, T, Nz))):
                return t.expand(B, T, Nz)
            raise ValueError(f"Index must be scalar or [B,T,Nz]; got {tuple(t.shape)}")

        tau_idx_t = _to_long_btNz(tau_idx)
        k_idx_t   = _to_long_btNz(k_idx)

        sigs = torch.empty((B, T, Nz, 2), dtype=torch.long, device=device)
        sigs[..., 0] = tau_idx_t
        sigs[..., 1] = k_idx_t
        return sigs
    @torch.no_grad()
    def shortcut_generate(
        self,
        z_prev: torch.Tensor,
        a: torch.Tensor | None = None,
        steps: int = 4,  # Fixed K=4 steps
        clamp: bool = True,
    ):
        device = z_prev.device
        dtype = z_prev.dtype
        
        # --- 1. Input Prep ---
        if z_prev.dim() == 3:
            z_prev = z_prev.unsqueeze(2)
        B, T, Nz, Dz = z_prev.shape
        
        N = int(getattr(self, "shortcut_kmax", 8))
        stride = N // steps
        k_step_idx = int(math.log2(stride)) 

        # --- Context Corruption (The "Robustness" Trick) ---
        # "We slightly corrupt the past inputs... to signal level 0.1"
        tau_ctx = 0.1
        
        # 1. Calculate the signal index for 0.1 (e.g., if N=8, 0.1*8 ≈ index 1)
        # We clamp it to be at least 1 (some noise) but less than N.
        ctx_signal_idx =int((1-tau_ctx)*(N))
        
        # 2. Create the noisy context Input (Do not overwrite z_prev!)
        # We assume z_prev is "clean" (level 1.0). We mix it down to level 0.1.
        noise = torch.randn_like(z_prev)
        z_prev_noisy = (1 - tau_ctx) * z_prev + tau_ctx * noise

        # Initialize current token 'z' with pure noise
        z = torch.randn((B, 1, Nz, Dz), device=device, dtype=dtype)
        
        # --- 2. Generation Loop (4 Steps) ---
        for i in range(steps):
            curr_tau_idx = i * stride
            
            # --- Signal Construction ---
            # CONTEXT: Signal that it is at level 0.1 (ctx_signal_idx)
            tau_signals = torch.full((B, T + 1, Nz), ctx_signal_idx, device=device, dtype=torch.long)
            
            # TARGET: Signal that it is at current noise level
            tau_signals[:, -1] = curr_tau_idx
            
            # K (Stride): Standard stride signal
            k_signals = torch.full((B, T + 1, Nz), k_step_idx, device=device, dtype=torch.long)

            sigs = torch.stack([tau_signals, k_signals], dim=-1)

            # --- Forward Pass ---
            # Feed the NOISY context, but keep refining 'z'
            packed_seq = torch.cat([z_prev_noisy, z], dim=1)
            
            # Get prediction for the WHOLE sequence
            full_pred = self.transformer(
                packed_seq, a, sigs, detach_agent=False, task_id=self.task_id
            )[0]

            # Extract prediction for the NEW token only
            z_clean_pred = full_pred[:, -1:] 
            
            # --- Linear Update (Euler) ---
            # Update only the new token 'z'
            steps_remaining = steps - i
            step_size = 1.0 / steps_remaining
            z = z + (z_clean_pred - z) * step_size

        return z.clamp(-1, 1) if clamp else z
    def shortcut_forcing(self, z_t, actions, mask: torch.Tensor | None = None):
        """
        Eq (7)-style shortcut forcing / consistency training with x-prediction.

        Supports endpoints:
        - tau can be 1 (by sampling tau_idx = N)
        - k can hit kmax (k_steps = N) when k_pow = log2(N)

        Notes on signals:
        - If signals want k in [0, log2(N)] (power index): use k_pow
        - If signals want k in [1, N] (step-count): use k_steps
        - If signals want tau_idx in [0, N], you must have embeddings sized N+1.
            Otherwise we clamp tau_idx for signals while keeping real tau float (incl. (N-1)/N).
        """
        device = z_t.device
        seq_mask = mask  # DO NOT overwrite caller mask

        if z_t.dim() == 3:
            z_t = z_t.unsqueeze(2)
        B, T, Nz, Dz = z_t.shape

        z1_clean = z_t
        z0 = torch.randn_like(z1_clean)

        N = int(self.shortcut_kmax)
        if N <= 1:
            raise ValueError(f"shortcut_kmax (N) must be > 1, got {N}")
        if (N & (N - 1)) != 0:
            raise ValueError(f"shortcut_kmax (N) should be a power-of-two, got {N}")

        max_pow = int(math.log2(N))

        # --- sample k as power-index (0..log2N), and also materialize k as steps (1..N) ---
        k_pow = torch.randint(0, max_pow + 1, (B, T, Nz), device=device, dtype=torch.long)
        s = (1 << k_pow)                           # {1,2,4,...,N}
        k_steps = s                                # explicit "k = kmax" happens when k_steps == N

        # n_cells = number of tau-grid cells for this step size
        n_cells = (N // s).clamp(min=1)            # >= 1

        # --- allow tau_idx == N (=> tau == 1) by sampling j in {0,...,n_cells} inclusive ---
        # We need per-element "high", so do it via uniform u and floor(u*(n_cells+1)).
        u = torch.rand(B, T, Nz, device=device)
        j = torch.floor(u * (n_cells.float() + 1.0)).long()   # in [0, n_cells]
        j = torch.minimum(j, n_cells)                          # safety

        tau_idx = j * s                                        # in {0, s, ..., N}  (endpoint included)
        tau = (tau_idx.float() / float(N)).unsqueeze(-1).clamp(max=((N-1)/N)) # can be exactly 1.0

        # Build noisy sample between z0 and clean x1
        z_targ = (1.0 - tau) * z0 + tau * z1_clean


        # --- signals safety: if your embedding only supports [0..N-1], clamp for signals ---
        # Keep true tau (float) for math/weighting.
        tau_idx_sig = tau_idx.clamp(max=N-1)

        # Choose which k to feed into signals:
        #   k_sig = k_pow   # if signals expect 0..log2(N)
        #   k_sig = k_steps # if signals expect 1..N  (so "kidx=kmax" means k_sig==N)
        k_sig = k_pow  # <-- change to k_steps if your signals are step-count based

        # Student x1 prediction
      #  x1_hat = self.transformer(
       #     z_targ,
        #    act_denoise,
         #   signals=self.make_signals_indices(B, T, Nz, tau_idx_sig, k_sig),
          #  detach_agent=False,
        #)[0][:, -T:]
        x1_hat = self.transformer(z_targ, actions[:,:-1], signals = self.make_signals_indices(B, T, Nz, tau_idx_sig, k_sig), task_id=self.task_id, detach_agent=False)[0]
        # Endpoint mask (tau == 1)
        #is_end = (tau_idx == N).float().unsqueeze(-1)          # [B,T,Nz,1]

        # Base vs non-base: treat tau==1 as base-like (don’t bootstrap velocities at endpoint)
        is_base = (k_pow == 0).unsqueeze(-1).float() #| (is_end > 0)).float()
        nonbase = 1.0 - is_base

        # ---- teacher bootstrap (2 half steps) ----
        with torch.no_grad():
            # For endpoint tau==1, we will set valid_mid=0 and zero velocities.
            k_half = (k_pow - 1).clamp(min=0)
            s_half = (1 << k_half)
            d_half = (s_half.float() / float(N)).unsqueeze(-1)

            tau_f = tau  # already [B,T,Nz,1]
            tau_mid_idx = tau_idx + s_half
            valid_mid = (tau_mid_idx < N).float().unsqueeze(-1)  # endpoint will be 0 (since tau_idx==N)

            tau_mid_idx_safe = tau_mid_idx.clamp(max=N - 1)

            # Teacher pass at tau_f
            z1_prime = self.transformer(
                z_targ,
                actions[:,:-1],
                task_id=self.task_id,
                signals=self.make_signals_indices(B, T, Nz, tau_idx_sig, k_half),  # k_half still power-index
            )[0][:, -T:]

            # Avoid nonsense at tau==1 by masking the velocity computation
            denom1 = (1.0 - tau_f).clamp(min=1e-5)
            v1 = (z1_prime - z_targ) / denom1
              # zero out endpoint velocities

            z_mid = z_targ + v1 * d_half

            # Teacher pass at tau_mid
            tau_mid = (tau_mid_idx_safe.float() / float(N)).unsqueeze(-1)
            z1_mid = self.transformer(
                z_mid,
                actions[:,:-1],
                task_id=self.task_id,
                signals=self.make_signals_indices(B, T, Nz, tau_mid_idx_safe, k_half),
            )[0][:, -T:]

            denom2 = (1.0 - tau_mid).clamp(min=1e-5)
            v2 = (z1_mid - z_mid) / denom2
            v2 = v2 * (valid_mid)  # only valid, non-end contribute

            avg_vel = 0.5 * (v1 + v2)

        # ---- losses ----
        loss_base = F.mse_loss(x1_hat, z1_clean.detach(), reduction="none")

        # Student velocity (mask endpoint so it doesn't matter)
        denom_s = (1.0 - tau).clamp(min=1e-5)
        v_student = (x1_hat - z_targ) / denom_s

        loss_boot = (1.0 - tau).pow(2) * F.mse_loss(v_student, avg_vel, reduction="none")
        loss_boot = loss_boot * valid_mid  # endpoint already has valid_mid=0

        loss = is_base * loss_base + nonbase * loss_boot
        loss = loss.mean(dim=-1)  # [B,T,Nz]

        w_tau = (0.9 * tau.squeeze(-1) + 0.1)  # [B,T,Nz]

        if seq_mask is None:
            m = torch.ones_like(w_tau)
        else:
            m = seq_mask
            if m.dim() == 4 and m.size(-1) == 1:
                m = m.squeeze(-1)
            m = m.float()

        num = (loss * w_tau * m).sum()
        den = (w_tau * m).sum().clamp_min(1e-8)
        loss = num / den

        debug = {
            "N": N,
            "tau_mean": tau.mean().item(),
            "tau1_frac": (tau_idx == N).float().mean().item(),
            "k_pow_mean": k_pow.float().mean().item(),
            "k_steps_mean": k_steps.float().mean().item(),
            "kmax_frac": (k_steps == N).float().mean().item(),
            "base_frac": is_base.mean().item(),
            "valid_mid_frac": valid_mid.mean().item(),
        }
        return loss, debug




    def sample_tau_d(self, Kmax: int, batch_size: int, device):
        k_choices = 2 ** torch.randint(0, int(math.log2(Kmax)) + 1, (batch_size,), device=device)
        d = 1.0 / k_choices.float().unsqueeze(1).unsqueeze(1)
        tau_grid = torch.rand(batch_size, 1, 1, device=device)
        tau_grid = torch.floor(tau_grid / d) * d
        tau = tau_grid.clamp(0.0, 1.0 - d)
        return tau, d

    def save_checkpoint(self, name):
        torch.save(self.state_dict(), f"./ckpts/Agent-{self.agent_id}-Task-{self.task_id}-" + name)
    def save_rep(self, name):
        torch.save(self.encoder.state_dict(), "./ckpts/" + "enc.pt")
        torch.save(self.decoder.state_dict(), "./ckpts/"+"dec.pt")
    def latent_imagination(
            self,
            initial_latent,
            actions,
            ctx_len,
            eval_=True,
            mtp=False,
            detach=False,
            random=False,
            steps=4,
            sig_inp=None,
            forced=False,
        ):
            device = initial_latent.device
            
            # FIX 1: Do NOT slice [:1]. Keep the full context history provided.
            z_all = initial_latent.detach()[:,:]
            # Initialize executed actions. 
            # If actions are provided (Reward training), we use them. 
            # If None (Policy), we start empty.
            a_exec = None
            kl_list, h_list, lp_list, a_list = [], [], [], []
            N = int(getattr(self, "shortcut_kmax", 8)) 
            z_inp=z_all
            a_exec = actions
            if not eval_:
                z_inp = z_all[:,:1]
                a_exec = None
            for i in range(ctx_len):
                # Current state input
                
                B, T_curr, Nz, _ = z_inp.shape
                
                # Add noise for robustness
                # --- Dynamics Step (Next State) ---
                    # Training: Use Ground Truth (Teacher Forcing)
                    # If initial_latent contains the full sequence, we just advance the window

                # --- Feature Extraction ---
                if eval_:
                    # Policy Mode: Run transformer on accumulated history
                    with torch.no_grad():
                        _, policy_feat = self.transformer(
                            z_inp, 
                            a_exec, 
                            signals=self.make_signals_indices(B, T_curr, Nz, N-1, 0), 
                         task_id=self.task_id,
                            detach_agent=False
                        )
                elif not eval_ and forced:

                    policy_feat = torch.zeros(B,1,self.transformer.d_model,device=self.device)
                else:
                    # FIX 2: Train Mode (Reward/BC)
                    # PREVIOUSLY: policy_feat = torch.zeros_like(...)  <-- THIS KILLED TRAINING
                    # NOW: Actually run the transformer to get features 'h'
                    
                    # We need features given the history up to 'i'. 
                    # In train mode, we usually have the full actions available in 'actions'.
                    
                    # Slice actions to match the current timestep
                    curr_actions = actions[:, :i] if actions is not None else a_exec
                    
                    _, policy_feat = self.transformer(
                        z_inp, 
                        curr_actions, 
                        signals=self.make_signals_indices(B, T_curr, Nz, N-1, 0), 
                        detach_agent=detach,
                        task_id=self.task_id
                    )

                # Get the features for the LAST step to feed into Policy/Value heads
                h_last = policy_feat[:, -1:, :]

                # --- Action Selection ---
                if not eval_:
                    # Teacher Forcing / BC: Use the ground truth action provided
                    a = actions[:, i:i+1]
                    log_p = torch.zeros((a.size(0), 1, 1), device=device)
                    kl = torch.zeros((a.size(0), 1, 1), device=device)
                elif random:
                    a = torch.tanh(torch.randn(1, 1, self.action_dim, device=device))
                    log_p = torch.zeros((a.size(0), 1, 1), device=device)
                    kl = torch.zeros((a.size(0), 1, 1), device=device)
                else:
                    # Policy Sampling
                    a, log_p, _, base_p, idx = self.policy(h_last)
                    a_list.append(a)
                    lp_list.append(log_p)
                    # Compute KL divergence for auxiliary loss
                    with torch.no_grad():
                        _, log_q, _, base_q, _ = self.t_policy(h_last)
                    kl_step = td.kl_divergence(base_p, base_q)
                    if kl_step.dim() >= 2:
                        kl_step = kl_step.sum(-1, keepdim=True)
                    kl = kl_step

                kl_list.append(kl)

                # Accumulate actions
                if a_exec is None: a_exec = a
                else: a_exec = torch.cat([a_exec, a], dim=1)

                h_list.append(h_last)
                if eval_ or (not eval_ and forced):
                    # Imagination: Use the model to predict next z
                    with torch.no_grad():
                        z_next = self.shortcut_generate(z_inp, a_exec, steps=steps)[:, -1:]
                    z_inp = torch.cat([z_inp, z_next], dim=1)

                
            h = torch.cat(h_list, dim=1)
            kl = torch.cat(kl_list, dim=1)

            if eval_ and not random:
                lp = torch.cat(lp_list, dim=1)
                imagined_actions = torch.cat(a_list, dim=1)
            else:
                lp = torch.zeros_like(kl)
                imagined_actions = actions # Return GT actions in train mode
            # Return relevant slice (exclude initial context from output if desired, or keep all)
            return z_inp, h, lp, kl, imagined_actions
    def decode(self, latents):
        return self.decoder(latents)

    def multistep_aux_losses(self, feat, actions, rewards):
        B, Lf, D = feat.shape
        K = int(self.aux_horizon)
        device = feat.device
        T_a = actions.size(1)
        T_r = rewards.size(1)
        Lp = feat.size(1)

        L_use = min(Lf, Lp, T_a - K - 1, T_r - K - 1)
        if L_use <= 0:
            zero = torch.tensor(0.0, device=device)
            return zero, zero, None

        feat = feat[:, :L_use]
        feat_acts = feat[:, :L_use]
 
        a_src = actions[:,  : (L_use + K)-1]              
        a_win = a_src.unfold(dimension=1, size=K, step=1)    
        a_tgt_all = a_win.permute(0, 1, 3, 2).contiguous()   
        a_tgt_bins = self.policy.binner.to_logits(a_tgt_all).float()

        out = self.policy(feat_acts, sample=False)
        _, _, dist, _, _ = out
        logits_bc = dist.logits

        logits_bc = logits_bc[:, :L_use]
        K_pred = logits_bc.shape[2]
        K_use = min(K, K_pred)
        logits_bc = logits_bc[:, :, :K_use]
        a_tgt_bins = a_tgt_bins[:, :, :K_use]
        act_loss = soft_ce(logits_bc, a_tgt_all, self.policy_num_bins, -10, 10).mean()

        r_src = rewards[:, : (L_use + K)-1]              
        r_win = r_src.unfold(dimension=1, size=K, step=1).unsqueeze(-1)   
        r_tgt_all = r_win.permute(0, 1, 3, 2).contiguous()   

        _, _, rew_logits, _ = self.reward(feat)
        rew_logits = rew_logits[:, :L_use]
        K_pred_r = rew_logits.shape[2]
        K_use_r = min(K_use, K_pred_r)
        rew_logits = rew_logits[:, :, :K_use_r]
        r_tgt_all = r_tgt_all[:, :, 0]
        r_tgt_idx =r_tgt_all          
        rew_loss =(soft_ce(
            rew_logits,
            r_tgt_idx,
            self.reward_bins,
            self.rminv, self.rmaxv, 
        )).mean()

        return act_loss, rew_loss, dist

    def unfreeze_agent_token(self):
    
        self.transformer.agent_token.requires_grad_(True)

    def freeze_agent_token(self):
        for p in self.transformer.parameters():
            p.requires_grad_(True)
        self.transformer.agent_token.requires_grad_(False)

    def update_ema(self):
        with torch.no_grad():
            for ema_param, param in zip(self.t_policy.parameters(), self.policy.parameters()):
                ema_param.mul_(0.98).add_(param * (1 - 0.98))
    def freeze_agent_token(self):
        for p in self.transformer.parameters():
            p.requires_grad_(True)
        self.transformer.agent_token.requires_grad_(False)

    def train_one_epoch(self, logger,buffer, model=False, policy=False, train_reward=False):
        self.transformer.train()
        epoch_loss = 0.0
        action_loss = 0.
        model_gn = 0.
        dyn_loss = 0.
        actor_gn = 0.
        kl = 0.
        self.steps += 1
        k = buffer.rng.integers(self.batch_lengths[0], self.batch_lengths[1])
        s = buffer.sample_seq(self.batch_size, k)

        states, actions, reward, termination, = s
        full_sequence = states[0].transpose(0,2,3,1)

        states = torch.from_numpy(states).to(self.device).float()
        actions = torch.from_numpy(actions).to(self.device).float()
        reward = torch.from_numpy(reward).to(self.device).float()
        termination = torch.from_numpy(termination).to(self.device).float()

        for i in tqdm(range(self.grad_accum)):
            if not model and not policy and not train_reward:
                self.rep_optim.zero_grad()

                self.dyn_optim = None
                self.ac_optim = None
                self.policy_optim = None
                self.ac_optim=  None
                self.encoder.train()
                self.decoder.train()

                z_t   = self.encoder(apply_random_patch_mask(states, max_mask_ratio=0.9, patch_size=self.encoder.patch)[0])
                reconst = self.decode(z_t)
                
                full_sequence = reconst[0].clone().detach().cpu().numpy().transpose(0,2,3,1)
                if self.steps % 100 ==0 and i ==0:
                    for i in range(full_sequence.shape[0]):
                        frame_bgr = full_sequence[i][..., ::-1] *255
                        cv2.imwrite(f"./eval_imgs/reconst_{i}.png", frame_bgr.astype(np.uint8))        
                reconst = 2 * reconst - 1
                targ_state = 2 * states - 1
                mse = ((F.mse_loss(reconst, targ_state, reduction="none").squeeze(-1))).mean()
                lp  = (self.lpips(reconst, targ_state))
                reconst_loss = (mse + 0.2 * lp.mean() )

                
                (reconst_loss*10).backward()

                encoder_gn = adaptive_grad_clip(self.encoder, 0.3)
                decoder_gn = adaptive_grad_clip(self.decoder, 0.3)

                if i == self.grad_accum-1:

                    (self.rep_optim).step()
            if model:

                self.rep_optim = None
                self.policy_optim = None
                self.dyn_optim.zero_grad()
                    # Create the full trajectory first
                    # shape: (Batch, Sequence_Length + 1, ...)
                full_sequence = states


                    # Apply mask and encode once
                    # This ensures s_{t+1} has the same embedding whether it's looked at as "current" or "next"
                z_t = self.encoder(full_sequence).detach()
                    # Split into current and next steps
                self.freeze_agent_token()
                kl_loss, _ = self.shortcut_forcing( z_t, actions)
                kl = kl_loss.detach().item()
                dyn_loss = kl_loss
                self.dyn_optim.zero_grad()
                        # Scale the loss before backward
                (dyn_loss/self.grad_accum).backward()
                  # [B,T,D]
                                        # Unscale gradients before clipping (important!)
                model_gn = adaptive_grad_clip(self, 0.3)
                        
                if i == self.grad_accum-1:
                    (self.dyn_optim.step())
            if train_reward:
                clean_latents = self.encoder((states))
                noised = clean_latents * 0.9 + torch.randn_like(clean_latents)*0.1
                N = self.shortcut_kmax
                signals=self.make_signals_indices(noised.shape[0], noised.shape[1], noised.shape[2], N, 0)
                _, h_grad = self.transformer(noised, actions[:,:-1],signals=signals, detach_agent=False )[:2]

                action_loss, reward_loss, _ = self.multistep_aux_losses(h_grad[:,:], actions[:,:], reward[:,:])
                                        
                term_target = termination[:, :].float()
                if term_target.dim()==3: term_target = term_target.squeeze(-1)
                rew_pred = self.reward(h_grad[:,:])
                term_logits = concat_mtp(rew_pred[-1].logits, self.aux_horizon)
                pos_w = torch.tensor(15.0, device=term_logits.device)
                term_loss = F.binary_cross_entropy_with_logits(
                                term_logits[:,:], term_target[:,:],
                                pos_weight=pos_w
                            ) 
                ac_loss =  action_loss+reward_loss+term_loss
                self.ac_optim.zero_grad()
                self.unfreeze_agent_token()
                (ac_loss.mean()).backward()
                reward_policy_model_gn = adaptive_grad_clip(self, 0.3)
                (self.ac_optim).step()
                self.train()
            if policy:
                self.dyn_optim = None
                self.ac_optim = None
                self.ac_optim=  None

                with torch.amp.autocast( "cuda",dtype=torch.float16, enabled=False):
                    initial_latent = self.encoder(states[:, :])
                    H = torch.randint(7, self.horizon_length - 1, size=(1,))[0].item()
                    actions= actions[:,:0]
                    z_0 = initial_latent[:,:1].detach()
                    
                    imag_z, h_t, _, lp, kl_prior, imagined_actions = self.latent_imagination(
                                z_0, actions[:, :], H)
                    with torch.no_grad():
                        r_full, _, _, termination = self.reward(h_t[:, :])
                        r_seq = concat_mtp(r_full)  
                        term_probs = termination.probs
                        term_probs = concat_mtp(term_probs)

                        cont_state =1-(term_probs).float()    
                        cont_t = cont_state[:, :-1]                    
                        cont_tp1 = cont_state[:, 1:]                    
                        V_full = self.value(h_t)[0]  

                    if r_seq.size(1) < 2+1:
                        continue
                    else:
                        bs = lambda_returns(r_seq, cont_tp1, V_full).squeeze(-1)  

                        adv = bs - self.value(h_t[:,:-1])[0].squeeze(-1).detach()
                        cont_t = cont_t > 0.5
                        value_loss = (soft_ce(self.value(h_t[:,:-1].detach())[1], bs, self.reward_bins, self.rminv, self.rmaxv).squeeze(-1) * cont_t).mean()
                        lp_t = lp.squeeze(-1)  * cont_t
                        pos_mask = (adv >= 0)
                        neg_mask = (adv < 0)
                        avg_imag_len = cont_t.float().sum(1).mean() / cont_t.shape[1]
                        pos_n = pos_mask.float().sum().clamp(min=1.0)
                        neg_n = neg_mask.float().sum().clamp(min=1.0)
                        pos_term = lp_t[pos_mask].sum() / pos_n
                        neg_term = lp_t[neg_mask].sum() / neg_n
                        actor_loss = 0.5*(-pos_term + neg_term) + 3e-2*(kl_prior.mean())

                        with FreezeParameters([
                                self.encoder,
                                self.transformer,
                                self.reward,
                                self.decoder,
                            ]):
                            self.policy_optim.zero_grad(set_to_none=True)
                            (actor_loss + value_loss).backward()
                            actor_gn =adaptive_grad_clip(self, 0.3)
                            self.policy_optim.step()

        if model:   
            logger["model_gn"] = model_gn 
            logger["shortcut_loss"] = kl

        if train_reward:
            logger["finetune_bc_loss"] = action_loss.mean().detach().item()
            logger["reward_loss"] = reward_loss.detach().item()
            logger["termination_loss"] = term_loss.detach().item()

            logger["reward_policy_gn"] = reward_policy_model_gn
        elif policy:
            logger["value_loss"] = value_loss.detach().item()
            logger["avg_imag_len"]= avg_imag_len.mean().detach().item()
            logger["kl_prior"]= kl_prior.mean().detach().item()
            logger["r_min"]= r_seq.min().detach().item()
            logger["r_max"]= r_seq.max().detach().item()
            logger["log_pi_mean"]= lp_t.mean().detach().item()
            logger["log_pi_std"]= lp_t.std().detach().item()
            logger["rollout_mean"]= bs.mean().detach().item()
            logger["rollout_max"]= bs.max().detach().item()
            logger["rollout_min"]= bs.min().detach().item()
            logger["actor_gn"] = actor_gn
            logger["actor_loss"] = actor_loss.detach().item()
        else:
            logger["reconst"] = reconst_loss.detach().item()
            logger["encoder_gn"] = encoder_gn 
            logger["decoder_gn"] = decoder_gn 

        return logger

class Reward(nn.Module):
    def __init__(self, bin_num: int, latent_dim: int, mtp: int = 1,r_max: float = 6):
        super().__init__()
        self.mtp = int(mtp)
        self.bin_num = int(bin_num)
        self.latent_dim = int(latent_dim)

        self.network = build_network(latent_dim, 2*latent_dim, 2, "SwiGLU", latent_dim*2)
        self.r_max = r_max
        self.reward_head = nn.Sequential(
            SwiGLU(),
            nn.RMSNorm(latent_dim),
            nn.Linear(latent_dim, self.mtp * self.bin_num),
        )

        self.term_head = nn.Sequential(
            SwiGLU(),
            nn.RMSNorm(latent_dim),
            nn.Linear(latent_dim, self.mtp),
        )

    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape
        h = self.network(x)  

        r_logits_all = self.reward_head(h).view(B, L, self.mtp, self.bin_num)  
        r_logits_1 = r_logits_all[:, :, 0]                                     
        r_mean = two_hot_inv(r_logits_all, self.bin_num, -self.r_max, self.r_max)                                       

        term_logits = self.term_head(h)                                        
        term_logits = term_logits.squeeze(-1)                              

        term_dist = td.Bernoulli(logits=term_logits)

        return r_mean, r_logits_1, r_logits_all, term_dist

class Value(nn.Module):
    def __init__(self,latent_dim, hidden_dim,bin_num, num_layers=4, r_max=6):
        super().__init__()
        self.network = build_network(latent_dim, hidden_dim, num_layers, "SwiGLU",bin_num)
        self.bin_num = bin_num
        self.r_max=r_max
    def forward(self, x):
        x = self.network(x)
        return two_hot_inv(x, self.bin_num, -self.r_max, self.r_max), x

class Decoder(nn.Module):
    def __init__(
        self,
        img_channels: int = 3,
        h: int = 96,
        w: int = 96,
        patch: int = 16,
        z_dim: int = 16,
        d_model: int = 256,
        n_heads: int = 4,
        depth: int = 8,
        latent_tokens: int = 64,
        time_every: int = 2,
        dropout: float = 0.05,
        max_T: int = 256,
        output_range: str = "0_1",
    ):
        super().__init__()
        assert (h % patch == 0) and (w % patch == 0)
        assert output_range in ("0_1", "minus1_1")
        self.output_range = output_range

        self.img_channels = img_channels
        self.patch = patch
        self.latent_tokens = latent_tokens
        self.d_model = d_model
        self.h = h
        self.w = w
        g1 = w // patch
        g2 = h // patch
        self.grid = (g1, g2)
        self.num_patches = g1 * g2

        self.patch_queries = nn.Parameter(torch.randn(1, 1, self.num_patches, d_model) * 0.02)

        self.z_to_latents = nn.Linear(z_dim, latent_tokens * d_model)
        self.z_tok_proj   = nn.Linear(z_dim, d_model)
        self.drop = nn.Dropout(dropout)
        blocks = []
        for i in range(depth):
            use_time = ((i+1) % time_every == 0)
            blocks.append(CausalSTBlock(d_model, n_heads, dropout=dropout, time_attn=use_time))
        self.blocks = nn.ModuleList(blocks)
        self.pos_embed_lat = nn.Parameter(torch.randn(1, 1, self.latent_tokens+self.num_patches, d_model) * 0.02)
        self.ln_out = nn.RMSNorm(d_model)
        self.to_patch = nn.Linear(d_model, img_channels * patch * patch)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        BT, P, Dp = patches.shape
        C = self.img_channels
        p = self.patch
        g1, g2 = self.grid
        assert P == g1 * g2, (P, g1 * g2)

        x = patches.view(BT, g1, g2, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(BT, C, g2 * p, g1 * p)
        return x
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # ---- project z to latent tokens ----
        if z.dim() == 3:
            B, T, Dz = z.shape
            zlat = self.z_to_latents(z).view(B, T, self.latent_tokens, self.d_model)
            L = self.latent_tokens
        elif z.dim() == 4:
            B, T, K, Dz = z.shape
            zlat = self.z_tok_proj(z)  # (B,T,K,D)
            L = K
        else:
            raise ValueError(f"Expected z dim 3 or 4, got {tuple(z.shape)}")
        # ---- patch queries ----
        pq = self.patch_queries.expand(B, T, self.num_patches, self.d_model)
        # IMPORTANT: put latents FIRST, then patch queries
        x = torch.cat([zlat, pq, ], dim=2) + self.pos_embed_lat  # (B,T,L+Np,D)
        x = self.drop(x)
        space_mask = modality_mask(L, [self.num_patches], encoder=False, device=x.device)

        # ---- transformer ----
        for blk in self.blocks:
            if blk.time_attn_enabled:
                x = blk(x)          # no space mask here
            else:
                x = blk(x,  mask=space_mask)    # modality mask only here
        x = self.ln_out(x)

        # ---- decode ONLY patch tokens ----
        patch_tok = x[:, :, L:, :]           # (B,T,Np,D)
        # sanity check
        assert patch_tok.shape[2] == self.num_patches, (patch_tok.shape, self.num_patches)

        patches = self.to_patch(patch_tok).contiguous().view(B * T, self.num_patches, -1)
        frames = self.unpatchify(patches).view(B, T, self.img_channels, self.h, self.w)

        return torch.sigmoid(frames) if self.output_range == "0_1" else torch.tanh(frames)