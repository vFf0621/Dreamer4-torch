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
from torch.cuda.amp import autocast, GradScaler

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
    def __init__(self, embed_dim=16, num_heads=8, num_kv_heads=4, dropout=0.1, causal=False,device="cuda"):
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
        self.device=device
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

        # =====================================================
        # BUILD FINAL ATTENTION MASK (SDPA bool mask)
        # =====================================================

        # ---- key padding mask (block padded keys) ----
        final_bias = None  # additive float bias

        # ---- user-provided attn_mask ----
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # bool allow mask (True=allow) -> additive bias
                bias = torch.zeros_like(attn_mask, dtype=q.dtype, device=q.device)
                bias.masked_fill_(~attn_mask, float("-inf"))
            else:
                # already additive bias (0 / -inf)
                bias = attn_mask.to(device=q.device, dtype=q.dtype)

            # normalize shape to be broadcastable
            if bias.dim() == 2:
                bias = bias[None, None, :, :]       # [1,1,Tq,Tk]
            elif bias.dim() == 3:
                bias = bias[:, None, :, :]          # [B,1,Tq,Tk]

            final_bias = bias

        # ---- key padding mask (block padded KEYS) ----
        if key_padding_mask is not None:
            # key_padding_mask: [B, Tk], True = PAD
            if key_padding_mask.dim() != 2 or key_padding_mask.shape != (B, Tk):
                raise ValueError(f"key_padding_mask must be [B,Tk]=[{B},{Tk}], got {tuple(key_padding_mask.shape)}")

            kpm_bias = torch.zeros((B, 1, 1, Tk), dtype=q.dtype, device=q.device)
            kpm_bias.masked_fill_(key_padding_mask[:, None, None, :], float("-inf"))

            final_bias = kpm_bias if final_bias is None else (final_bias + kpm_bias)

        # ---- causal mask (ONLY if time attention) ----
        if self.causal:
            # block attending to future keys: j > i
            causal_bool = torch.ones((Tq, Tk), dtype=torch.bool, device=q.device).tril(diagonal=0)
            causal_bias = torch.zeros((Tq, Tk), dtype=q.dtype, device=q.device)
            causal_bias.masked_fill_(~causal_bool, float("-inf"))
            causal_bias = causal_bias[None, None, :, :]  # [1,1,Tq,Tk]

            final_bias = causal_bias if final_bias is None else (final_bias + causal_bias)

        # =====================================================
        # SDPA
        # =====================================================
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=final_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,   # IMPORTANT: we already applied causal bias
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
            self.space_attn = GQA(d_model, n_heads, n_heads // 2, dropout, device=device)
        else:
            self.time_attn = GQA(d_model, n_heads, n_heads // 2, dropout, causal=True, device=device)

        self.num_reserved = num_reserved
        self.ln_time = nn.RMSNorm(d_model)
        self.reserved_tokens = nn.Parameter(torch.zeros(1, num_reserved, int(d_model)))
        self.mlp = build_network(d_model, d_model * 2, 3, "SwiGLU", d_model, True)
        self.device = device
        self.to(self.device)

    def forward(
            self,
            x: torch.Tensor,
            token_pad_mask: torch.Tensor | None = None,   # [B,T,N] True = PAD
            *,
            agent_idx: int | None = None,
            mask=None,                                    # either None, or [N,N] or [N+R, N+R] additive float (-inf)
            ) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(2)  # [B,T,1,D]

        reserved = self.reserved_tokens
        B, T, N, D = x.shape
        R = reserved.shape[1]
        Ncat = N + R

        # ---- normalize token_pad_mask to [B,T,N] ----
        if token_pad_mask is not None:
            assert token_pad_mask.shape[:2] == (B, T), (token_pad_mask.shape, (B, T))
            if token_pad_mask.size(2) >= N:
                token_pad_mask = token_pad_mask[:, :, :N]
            else:
                pad = torch.zeros(B, T, N - token_pad_mask.size(2),
                                dtype=torch.bool, device=x.device)
                token_pad_mask = torch.cat([token_pad_mask.to(x.device), pad], dim=2)

        if agent_idx is not None and not (0 <= agent_idx < N):
            raise ValueError(f"agent_idx={agent_idx} out of range for N={N}")

        # =========================
        # TIME ATTENTION
        # =========================
        if self.time_attn_enabled:
            x_time = self.ln_time(x).permute(0, 2, 1, 3).reshape(B * N, T, D)

            time_kpm = None
            if token_pad_mask is not None:
                # token_pad_mask: [B,T,N] -> [B,N,T] -> [B*N, T]
                time_kpm = token_pad_mask.permute(0, 2, 1).reshape(B * N, T)

            # Prefer is_causal=True inside your GQA/attn module if possible; otherwise pass a causal float mask.
            xt_out = self.time_attn(
                x_time,
                attn_mask=None,              # or causal_mask(T) if your impl needs it
                key_padding_mask=time_kpm,
            )

            xt_out = xt_out.reshape(B, N, T, D).permute(0, 2, 1, 3)
            x = x + xt_out
            x = x + self.mlp(x)
            return x.squeeze(2) if x.shape[2] == 1 else x

        # =========================
        # SPACE ATTENTION
        # =========================
        x_space = self.ln_space(x)
        xt = x_space.reshape(B * T, N, D)  # [B*T, N, D]
        xcat = torch.cat([xt, reserved.expand(B * T, R, D)], dim=1)  # [B*T, N+R, D]

        # ---- build key_padding_mask for space attn: [B*T, N+R] ----
        space_kpm = None
        if token_pad_mask is not None:
            space_kpm = token_pad_mask.reshape(B * T, N)  # [B*T, N]
            # reserved tokens are never PAD
            space_kpm = torch.cat(
                [space_kpm, torch.zeros(B * T, R, dtype=torch.bool, device=x.device)],
                dim=1,
            )  # [B*T, N+R]

        # ---- build final additive attn mask over [N+R, N+R] ----
        final_mask = None
        if mask is not None:
            if mask.shape[-2:] == (N, N):
                # embed into top-left; allow everything involving reserved by default
                m = torch.zeros((Ncat, Ncat), dtype=torch.float, device=x.device)
                m[:N, :N] = mask
                m[N:, :N] = float("-inf")
                final_mask = m
            elif mask.shape[-2:] == (Ncat, Ncat):
                final_mask = mask
            else:
                raise ValueError(f"mask has shape {mask.shape}, expected ({N},{N}) or ({Ncat},{Ncat})")

        xs = self.space_attn(
            xcat,
            attn_mask=final_mask,          # additive float mask (0 / -inf)
            key_padding_mask=space_kpm,    # True = PAD keys
        )[:, :N]  # drop reserved outputs

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
        self.patch_proj = nn.Sequential(nn.RMSNorm(patch_dim), nn.Linear(patch_dim, d_model))
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
        Nr: int = 4,
        # behavior toggles
        mask_last_action: bool = True,      # usually correct for "predict next"
        clamp_signal_indices: bool = False, # set True if you’d rather clamp than crash
    ):
        super().__init__()
        self.max_T = max_T
        self.use_agent_token = use_agent_token
        self.device = device
        self.Sa = Sa
        self.Nr = Nr
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
        self.space_pos_embed = nn.Parameter(0.02 * torch.randn(1, 1, self.Sa + self.Nz+2 + Nr, d_model))
        self.z_proj = nn.Sequential(nn.RMSNorm(Dz), nn.Linear(Dz, d_model))
        self.sig_proj = nn.Sequential(nn.RMSNorm(level_dim), nn.Linear(level_dim, d_model))

        self.level_emb = nn.Embedding(self.level_vocab, level_dim)
        self.step_emb  = nn.Embedding(self.step_vocab, level_dim)
        self.action_conditioner = nn.Parameter(0.02 * torch.randn(1, 1, self.Sa, d_model))
        self.reserved = nn.Parameter(0.02 * torch.randn(1, 1, self.Nr, d_model))

        self.action_pad = nn.Parameter(0.02*torch.randn(1, 1, 1, d_model, device=self.device))
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
                    num_reserved=0,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Output head: model dim -> Dz
        self.out = build_network(d_model, 2*d_model, 3, "SwiGLU", Dz)

        self.to(device)
    def recover_z(self, out: torch.Tensor):
        """
        out: [B,T,Nz+Sa,D] created by interleave_gb_1a
        returns:
        a_rec: [B,T,Sa,D]
        b_rec: [B,T,Nz,D]
        """
        Sa = self.Sa
        B, T, L, D = out.shape
        assert L % Sa == 0, "Token length must be divisible by Sa"
        block = L // Sa                 # block = g+1
        g = block - 1
        assert g >= 1

        blocks = out.reshape(B, T, Sa, g+1, D)   # [B,T,Sa,g+1,D]
        z_rec = blocks[:, :, :, :g, :].reshape(B, T, Sa*g, D)  # [B,T,Nz,D]
       # a_rec = blocks[:, :, :,  g, :]                          # [B,T,Sa,D]
        return z_rec
    
    def interleave_z_a(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        1 action token per (Nz/Sa) latent tokens
        z: [B,T,Nz,D]
        a: [B,T,Sa,D]
        returns: [B,T,Nz+Sa,D]
        """
        B, T, Nz, D = z.shape
        assert a.shape[0] == B and a.shape[1] == T and a.shape[3] == D

        Sa = a.shape[2]
        assert Nz % Sa == 0, f"Nz={Nz} must be divisible by Sa={Sa}"
        g = Nz // Sa

        z_grp = z.reshape(B, T, Sa, g, D)     # [B,T,Sa,g,D]
        a_grp = a.unsqueeze(3)               # [B,T,Sa,1,D]

        out = torch.cat([z_grp, a_grp], dim=3)  # [B,T,Sa,g+1,D]
        return out.reshape(B, T, Nz + Sa, D)
    def make_kpm_for_actions(self, B, T, S, Nz, *, pad_action_t: int | None, device):
        """
        key_padding_mask: [B, T, S] with True = PAD (blocked as key).
        pad_action_t:
        - None  : no padded action timestep
        - 0     : you PREPEND pad (a_{-1} unknown) -> t=0 action slots are padded
        - T-1   : you APPEND pad (a_{T-1} unknown) -> last timestep action slots are padded
        """
        kpm = torch.zeros((B, T, S), dtype=torch.bool, device=device)
        if pad_action_t is not None:
            kpm[:, pad_action_t, Nz] = True
        return kpm
  
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
        else:
            raise ValueError("Incorrect action shape")



    def agent_mask(self, S: int, agent_idx: int, device=None, Nr: int = 0, *, dtype=torch.float32):
        if not (0 <= agent_idx < S):
            raise ValueError(f"agent_idx={agent_idx} out of range for S={S}")
        if Nr < 0 or Nr > S:
            raise ValueError(f"Nr={Nr} must satisfy 0 <= Nr <= S={S}")

        allow = torch.ones((S, S), dtype=torch.bool, device=device)

        # Agent rule: others cannot read agent; agent reads everyone
        allow[:, agent_idx] = False
        allow[agent_idx, :] = True
        allow[agent_idx, agent_idx] = True

        # Reserved rule: reserved can read everyone EXCEPT agent
        if Nr > 0:
            rs = S - Nr  # start index of reserved tokens
            allow[rs:, agent_idx] = False  # reserved queries can't read agent
            allow[rs:, rs:] = True         # reserved can read themselves

        # Convert allow-mask -> additive bias (0 keep, -inf block)
        attn_bias = torch.zeros((S, S), dtype=dtype, device=device)
        attn_bias.masked_fill_(~allow, float("-inf"))

        # Optional sanity checks (safe ones)
        assert attn_bias.shape == (S, S)
        assert torch.isfinite(attn_bias[allow]).all()

        return attn_bias

                                # Forwar
    # -----------------------------
# -----------------------------
    def forward(
        self,
        z_tokens: torch.Tensor,                 # [B, T, Nz, Dz]
        actions: torch.Tensor | None,           # [B, T-1, A] or [B, T, A]
        signals: torch.Tensor,                  # [B, T, Nz, 2] (level_idx, step_idx)
        policy_tok_in: torch.Tensor | None = None,  # [1,1,1,D] or [B,1,1,D] or [B,T,1,D]
        detach_agent: bool = False,
        last_z=None,
        task_id=0,
    ):
        assert (actions.size(1) == z_tokens.size(1) - 1) or ( actions.size(1) == z_tokens.size(1) )
        device = z_tokens.device
        if z_tokens.dim() != 4:
            raise ValueError(f"z_tokens must be [B,T,Nz,Dz], got {tuple(z_tokens.shape)}")
        if last_z is not None:
            z_tokens = torch.cat([z_tokens, last_z], dim=1)

        B, T, Nz_in, Dz = z_tokens.shape

        if signals.size(1) != T:
            raise ValueError(f"signals has T={signals.size(1)} but z_tokens has T={T}.")

        acts_bt = self.align_actions(actions, T, B, z_tokens.dtype, device)          # [B, T-1, A]
        act_idx = self.discretize_actions_to_indices(acts_bt)                        # [B, T-1, A]

        a = 0
        for i, emb in enumerate(self.action_embs):
            a = a + emb(act_idx[..., i])                                             # [B, T-1, D]
        a = self.action_normalize(a)

        a_emb = a[:, :, None, :]                                                     # [B, T-1, 1, D]
        pad = self.action_pad.expand(B, 1, 1, -1)                                     # [B, 1, 1, D]
        a_emb = torch.cat([a_emb, pad], dim=1)                                        # [B, T, 1, D]
        a_tokens = a_emb.expand(-1, -1, self.Sa, -1) + self.action_conditioner  # [B,T,Sa,D]
        signals = signals.long()
        if signals.dim() == 4 and signals.size(-1) == 2:
            level_idx, step_idx = signals[..., 0], signals[..., 1]
        elif signals.dim() == 3:
            level_idx, step_idx = signals, torch.zeros_like(signals)
        else:
            raise ValueError(f"signals must be [B,T,Nz,2] or [B,T,Nz], got {tuple(signals.shape)}")

        level_t = level_idx[:, :, 0]                                                 # [B,T]
        step_t  = step_idx[:, :, 0]                                                  # [B,T]
        lev_feat  = self.level_emb(level_t)
        step_feat = self.step_emb(step_t)

        z_inp = self.z_proj(z_tokens)

        # Interleave z + a, then append signal token
        x = self.interleave_z_a(z_inp, a_tokens )                 # x: [B,T,?,D]
        Nmain = x.size(2)
        x = torch.cat([x, self.sig_proj(lev_feat + step_feat).unsqueeze(-2)], dim=2)
        token_pad_mask = self.make_kpm_for_actions(B, T, S=x.size(2), Nz=Nz_in, pad_action_t=T-1, device=device)
        pad_extra = torch.zeros(B, T, 1 + self.Nr, dtype=torch.bool, device=device)
        token_pad_mask_aug = torch.cat([token_pad_mask, pad_extra], dim=2)  # [B,T,Nmain+1+Nr]
        agent = None
        if policy_tok_in is not None:
            ag = policy_tok_in
            if ag.size(0) == 1 and B > 1: ag = ag.expand(B, -1, -1, -1)
            if ag.size(1) == 1 and T > 1: ag = ag.expand(B, T, -1, -1)
            if ag.size(2) != 1:
                raise ValueError("policy_tok_in token dim must be 1 at dim=2")
            agent = ag
            agent_in = agent.detach() if detach_agent else agent
        elif self.use_agent_token:
            assert task_id < self.num_task
            agent = self.agent_token[:, :, task_id].expand(B, T, 1, self.d_model)
            agent_in = agent.detach() if detach_agent else agent

        agent_idx = x.size(2)
        reserved  = self.reserved.expand(B, T, self.Nr, -1)

        x_aug = torch.cat([x, agent_in, reserved], dim=2)
        x_aug = x_aug + self.space_pos_embed.expand(B, 1, -1, self.d_model)

        for blk in self.blocks:
            S = x_aug.size(2)
            tok_mask = self.agent_mask(S, agent_idx, device=device, Nr=self.Nr)  # keep agent mask

            if blk.time_attn_enabled:
                x_aug = blk(x_aug, token_pad_mask=token_pad_mask_aug, agent_idx=None)
            else:
                x_aug = blk(x_aug, token_pad_mask=token_pad_mask_aug, mask=tok_mask, agent_idx=agent_idx)

        agent = x_aug[:, :, agent_idx:agent_idx+1, :]
        x     = x_aug[:, :, :agent_idx, :]
        agent_out_bt = agent[:, :, 0, :]


        hist = self.recover_z(x[:,:,:Nmain])     

        hist = hist.reshape(B, T*Nz_in, self.d_model)     # [B,T*Nz,D]
        z = self.out(hist)                              # [B,T*Nz,Dz]
        z_pred = z.view(B, T, Nz_in, Dz)            # [B,T,Nz,Dz]                        # [B,T,Nz,D]

        policy_feat = agent_out_bt if agent_out_bt is not None else None
        return z_pred, policy_feat


class Dreamer4(nn.Module):
    def __init__(self,agent_id, ch=3, h=96, w=96, patch = 16, latent_tokens=32, z_dim=16, action_dim=2, latent_dim=512, 
                 rep_depth = 8, rep_d_model=256, dyn_d_model=256, num_heads=8, dropout=0.1, k_max=8, mtp=8, 
                 policy_bins = 100, reward_bins = 100, pretrain=False, reward_clamp=6,level_vocab = 16, level_embed_dim = 16,
                 batch_lens = (45, 65), batch_size=16, accum=1, max_imag_len=128, ckpt=None, rep_lr=1e-4, rep_decay=1e-3,Sa = 64,eval_context_len=15,
                 dyn_lr=1e-4, dyn_decay=1e-3, ac_lr = 1e-4, ac_decay=1e-3, policy_lr=1e-4, policy_decay=1e-3, num_tasks=30, task_id = 0, Nr = 4,
                kmax_prob=0.1):
        super(Dreamer4, self).__init__()
        self.encoder =  Encoder(img_channels=ch, h=h, w=w, patch=patch, d_model=rep_d_model,
                                n_heads=num_heads, depth=rep_depth, latent_tokens=latent_tokens, time_every=2,
                                out_dim=z_dim, dropout=dropout, max_T=max_imag_len)
 
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.pretrain = False
        self.agent_id = agent_id
        self.eval_ctx = eval_context_len
        # --- TokenDynamics --      -
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
            Nr = Nr, 
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
        self.kmax_prob = kmax_prob
       # self.encoder.load_state_dict(torch.load("enc.pt"))
       
        # self.decoder.load_state_dict(torch.load("dec.pt"))
        
        self.lpips = LPIPSLoss(reduction="none",)
        self.to(self.device)
        self.rep_optim = torch.optim.AdamW(
            [{'params': self.encoder.parameters()}, 
            {'params': self.decoder.parameters()},]
        , lr=rep_lr, capturable=True, weight_decay=rep_decay)
        self.reset()
        self.dyn_optim = torch.optim.AdamW(            
            [{'params': self.policy.parameters()}, 
            {'params': self.reward.parameters()}, 
            {'params': self.transformer.parameters()}, ],

            lr=dyn_lr, capturable=True,weight_decay=dyn_decay)

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
            sigs = self.make_signals_indices(B, T, Nz, tau_idx=N - 1, k_idx=0)

            _, h = self.transformer(z, self.action_buffer, signals=sigs)

            a, log_p, entropy, base_p, idx = self.policy(h[:, -1:])

            if self.action_buffer is None:
                self.action_buffer = a
            else:
                self.action_buffer = torch.cat([self.action_buffer, a], dim=1)

            # keep a rolling window
            W = self.eval_ctx
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
            z_eval = self.latent_imagination(z[:,:], None, ctx_len=a.size(1), eval_=False,random=True, forced=False)[0]
            decoded = self.decode(z_eval).detach().cpu().numpy()
            for i in range(decoded.shape[1]):
                cv2.imwrite(f"./eval_imgs/random_{i}.png", 255 * decoded[0, i].transpose(1, 2, 0)[..., ::-1])

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
    def shortcut_generate(self, z_prev, actions, steps=4, clamp=True):
        import math
        device = z_prev.device
        dtype = z_prev.dtype

        if z_prev.dim() == 3:
            z_prev = z_prev.unsqueeze(2)
        B, T, Nz, Dz = z_prev.shape

        N = int(self.shortcut_kmax)
        assert steps <= N and (N % steps == 0)

        stride = N // steps
        k_pow = int(math.log2(stride))

        # ============================
        # τ LOOKUP TABLE (same as training)
        # ============================
        noise_min = 0.1
        tau_max = 1.0 - noise_min

        tau_lut = torch.arange(N + 1, device=device, dtype=torch.float32) / float(N)
        tau_lut[-1] = tau_max
        # context is "cleanest" (tau_idx=N)
        tau_ctx = torch.full((B, T, 1), N - 1, device=device, dtype=torch.long)
        k_ctx = torch.zeros_like(tau_ctx)

        z = torch.randn((B, 1, Nz, Dz), device=device, dtype=dtype)
        tau_idx_schedule = [i * stride for i in range(steps)] + [N]

        for i in range(steps):
            tau_i = tau_idx_schedule[i]
            tau_next = tau_idx_schedule[i + 1]

            tau = tau_lut[tau_i].item()
            tau_nextf = tau_lut[tau_next].item()
            d_tau = tau_nextf - tau

            tau_sig = torch.cat(
                [tau_ctx, torch.full((B, 1, 1), tau_i, device=device, dtype=torch.long)], dim=1
            )
            k_sig = torch.cat(
                [k_ctx, torch.full((B, 1, 1), k_pow, device=device, dtype=torch.long)], dim=1
            )
            sigs = torch.stack([tau_sig, k_sig], dim=-1)

            packed = torch.cat([z_prev, z], dim=1)

            x1_hat = self.transformer(
                packed, actions, sigs, detach_agent=False, task_id=self.task_id
            )[0][:, -1:]

            v = (x1_hat - z) / max(1e-5, 1.0 - tau)
            z = z + v * d_tau
        return z.clamp(-1, 1) if clamp else z

    def shortcut_forcing(self, z_t, actions, mask: torch.Tensor | None = None):


        device = z_t.device
        seq_mask = mask

        if z_t.dim() == 3:
            z_t = z_t.unsqueeze(2)
        B, T, Nz, Dz = z_t.shape

        z1_clean = z_t
        z0 = torch.randn_like(z1_clean)

        N = int(self.shortcut_kmax)
        assert N > 1 and (N & (N - 1)) == 0
        max_pow = int(math.log2(N))
        assert (z_t.size(1) == actions.size(1) -1 ) or (z_t.size(1) == actions.size(1))
        # ============================
        # τ LOOKUP TABLE (min noise 0.1)
        # ============================
        noise_min = 0
        tau_max = 1.0 - noise_min  # 0.9

        tau_lut = torch.arange(N + 1, device=device, dtype=torch.float32) / float(N)
        tau_lut[-1] = tau_max

        # ============================
        # sample k
        # ============================
        k_pow = torch.randint(1, max_pow + 1, (B, T, 1), device=device)
        base = torch.rand(B, T, 1, device=device) <= self.kmax_prob
        k_pow[base] = 0

        s = (1 << k_pow)
        k_half = (k_pow - 1).clamp(min=0)
        s_half = (1 << k_half)

        u = torch.rand(B, T, 1, device=device)
        tau_idx_base = torch.floor(u * (N + 1)).long().clamp(0, N)

        max_tau_idx = (N - 1) - s_half
        max_j = torch.div(max_tau_idx, s, rounding_mode="floor").clamp(min=0)
        u2 = torch.rand(B, T, 1, device=device)
        tau_idx_nb = (torch.floor(u2 * (max_j + 1).float()).long()) * s
        tau_idx = torch.where(base, tau_idx_base, tau_idx_nb)

        is_base = ((k_pow == 0) | (tau_idx == N - 1))


        # ============================
        # sample tau_idx
        # ============================
    
        assert (tau_idx >= 0).all()
        assert (tau_idx[is_base] <= N).all()
        assert (tau_idx[~is_base] <= N - 1).all()

        # ============================
        # τ via LUT
        # ============================
        tau = tau_lut[tau_idx].to(z_t.dtype).unsqueeze(-1)

        z_targ = (1.0 - tau) * z0 + tau * z1_clean

        # ============================
        # student prediction
        # ============================
        x1_hat = self.latent_imagination(
            z_targ,
            actions[:, :],
            eval_=False,
            signals=self.make_signals_indices(B, T, 1, tau_idx, k_pow),
            get_feat=True,
        )[0]

        is_base_f = is_base.unsqueeze(-1).float()
        valid_mid = (~is_base).float().unsqueeze(-1)

        # ============================
        # teacher bootstrap
        # ============================
        with torch.no_grad():
            d_half = (s_half.float() / float(N)).unsqueeze(-1)

            tau_mid_idx = torch.where(is_base, torch.zeros_like(tau_idx), tau_idx + s_half)
            tau_mid = tau_lut[tau_mid_idx].to(z_t.dtype).unsqueeze(-1)

            z1_prime = self.latent_imagination(
                z_targ,
                actions[:, :],
                signals=self.make_signals_indices(B, T, 1, tau_idx, k_half),
                get_feat=True
            )[0][:, :]

            v1 = (z1_prime - z_targ) / (1.0 - tau).clamp(min=1e-5)
            v1 = v1 * valid_mid

            z_mid = z_targ + v1 * d_half

            z1_mid = self.latent_imagination(
                z_mid,
                actions[:, :],
                signals=self.make_signals_indices(B, T, 1, tau_mid_idx, k_half),
                get_feat=True,
            )[0][:, :]

            v2 = (z1_mid - z_mid) / (1.0 - tau_mid).clamp(min=1e-5)
            v2 = v2 * valid_mid

            avg_vel = 0.5 * (v1 + v2)

        # ============================
        # losses
        # ============================
        loss_base = F.mse_loss(x1_hat, z1_clean.detach(), reduction="none")

        v_student = (x1_hat - z_targ) / (1.0 - tau).clamp(min=1e-5)
        loss_boot = (1.0 - tau).pow(2) * F.mse_loss(v_student, avg_vel, reduction="none")
        loss_boot = loss_boot * valid_mid

        loss = is_base_f * loss_base + (1.0 - is_base_f) * loss_boot

        w_tau = (0.9 * tau + 0.1)

        loss = (loss * w_tau ).mean()

        return loss, {
            "tau_mean": tau.mean().item(),
            "tauN_frac": (tau_idx == N).float().mean().item(),
            "base_frac": is_base.float().mean().item(),
        }



    def save_checkpoint(self, name):
        torch.save(self.state_dict(), f"./ckpts/Agent-{self.agent_id}-Task-{self.task_id}-" + name)
    def save_rep(self, name):
        torch.save(self.encoder.state_dict(), "./ckpts/" + "enc.pt")
        torch.save(self.decoder.state_dict(), "./ckpts/"+"dec.pt")
    def latent_imagination(
            self,
            initial_latent,
            actions,
            ctx_len = None,
            eval_=True,
            signals=None,
            detach=False,
            random=False,
            steps=4,
            get_feat=False,
            forced=False,
        ):
            device = initial_latent.device
            if not ctx_len:
                ctx_len = initial_latent.size(1)
            # FIX 1: Do NOT slice [:1]. Keep the full context history provided.
            z_all = initial_latent.detach()[:,:]
            # Initialize executed actions. 
            # If actions are provided (Reward training), we use them. 
            # If None (Policy), we start empty.
            a_exec = None
            kl_list, h_list, lp_list, a_list = [], [], [], []
            z_list =[]
            N = int(getattr(self, "shortcut_kmax", 8)) 
            z_inp=z_all

            a_exec = actions
            if not eval_ and not get_feat:
                z_inp = z_all[:,:1]
                a_exec = None
            for i in range(ctx_len):
                # Current state input
                
                B, T_curr, Nz, _ = z_inp.shape
                T_curr = min(i+1, T_curr)
                # Add noise for robustness
                # --- Dynamics Step (Next State) ---
                    # Training: Use Ground Truth (Teacher Forcing)
                    # If initial_latent contains the full sequence, we just advance the window

                # --- Feature Extraction ---
                if get_feat:
                    z_feat, policy_feat = self.transformer(
                            initial_latent[:,:i+1].detach(), 
                            actions[:,:i], 
                            signals=signals[:,:i+1], 
                            task_id=self.task_id,
                            detach_agent=False
                        )
                    z_list.append(z_feat[:,-1:])
                    h_list.append(policy_feat[:,-1:])
                    if i == ctx_len - 1:
                        return torch.cat(z_list,1), torch.cat(h_list, 1)
                    else:
                        continue
                elif eval_ and not random:
                    # Policy Mode: Run transformer on accumulated history
                    with torch.no_grad():
                        _, policy_feat = self.transformer(
                            z_inp, 
                            a_exec, 
                            signals=self.make_signals_indices(B, T_curr, Nz, N - 1, 0), 
                         task_id=self.task_id,
                            detach_agent=False
                        )
                elif random or (not eval_ and forced):

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
                        signals=self.make_signals_indices(B, T_curr, Nz, N - 1, 0), 
                        detach_agent=detach,
                        task_id=self.task_id
                    )

                # Get the features for the LAST step to feed into Policy/Value heads
                h_last = policy_feat[:, -1:, :]

                # --- Action Selection ---
                if not eval_ and not random:
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

    def update_ema(self):
        with torch.no_grad():
            for ema_param, param in zip(self.t_policy.parameters(), self.policy.parameters()):
                ema_param.mul_(0.98).add_(param * (1 - 0.98))
    def freeze_agent_token(self):
        for p in self.transformer.parameters():
            p.requires_grad_(True)
        self.transformer.agent_token.requires_grad_(False)

    def train_step(self, logger,buffer, model=False, policy=False, train_reward=False):
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
        self.dyn_optim.zero_grad()

        states, actions, reward, termination, = s
        full_sequence = states[0].transpose(0,2,3,1)

        states = torch.from_numpy(states).to(self.device).float()
        actions = torch.from_numpy(actions).to(self.device).float()
        reward = torch.from_numpy(reward).to(self.device).float()
        termination = torch.from_numpy(termination).to(self.device).float()
        self.rep_optim.zero_grad()
        self.policy_optim.zero_grad(set_to_none=True)

        for i in tqdm(range(self.grad_accum)):
            if not model and not policy and not train_reward:
                self.encoder.train()
                self.decoder.train()

                z_t   = self.encoder(apply_random_patch_mask(states, max_mask_ratio=0.9, patch_size=self.encoder.patch)[0])
                reconst = self.decode(z_t)
                
                full_sequence = reconst[0].clone().detach().cpu().numpy().transpose(0,2,3,1)
                if self.step%100 ==0 and i ==0:
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
                    self.rep_optim.zero_grad()
            if model:
                    
                    # Create the full trajectory first
                    # shape: (Batch, Sequence_Length + 1, ...)
                full_sequence = states


                        # Apply mask and encode once
                        # This ensures s_{t+1} has the same embedding whether it's looked at as "current" or "next"
                z_t = self.encoder(full_sequence).detach()
                        # Split into current and next steps
               # self.freeze_agent_token()
                kl_loss, _ = self.shortcut_forcing( z_t, actions)
                kl = kl_loss.detach().item()
                dyn_loss = kl_loss
                                                # Scale the loss before backward
                (dyn_loss/self.grad_accum).backward()
                    # [B,T,D]
                                            # Unscale gradients before clipping (important!)
                model_gn = adaptive_grad_clip(self, 0.3)
                            
                if i == self.grad_accum-1 and not train_reward:
                    (self.dyn_optim).step()
                    #self.dyn_optim.zero_grad()
            if train_reward:
                clean_latents = self.encoder((states)).detach()
                B, T, Nz, _ = clean_latents.shape

                noised = clean_latents * 0.9 + torch.randn_like(clean_latents)*0.1
                N = self.shortcut_kmax
                h_with_grad = self.latent_imagination(noised, actions[:,:-1], get_feat=True, 
                                                      signals=self.make_signals_indices(B, T, Nz, N, 0)
                                                      , ctx_len=noised.size(1))[1]

                action_loss, reward_loss, _ = self.multistep_aux_losses(h_with_grad[:,:], actions[:,:], reward[:,:])
                                            
                term_target = termination[:, :].float()
                if term_target.dim()==3: term_target = term_target[:,:].squeeze(-1)
                rew_pred = self.reward(h_with_grad[:,:])
                term_logits = concat_mtp(rew_pred[-1].logits, self.aux_horizon)

                pos_w = torch.tensor(15.0, device=term_logits.device)
                term_loss = F.binary_cross_entropy_with_logits(
                                    term_logits[:,:], term_target[:,:],
                                    pos_weight=pos_w
                                ) 
                ac_loss =  action_loss+reward_loss+term_loss
                with FreezeParameters([self.transformer]):
                    self.unfreeze_agent_token()
                    (ac_loss.mean()/self.grad_accum).backward()
                    reward_policy_model_gn = adaptive_grad_clip(self, 0.3)
                    if i == self.grad_accum-1:

                        (self.dyn_optim).step()
                        self.dyn_optim.zero_grad()

            if policy:
 

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
                            scaler.scale((actor_loss + value_loss)/self.grad_accum).backward()
                            actor_gn =adaptive_grad_clip(self, 0.3)
                            if i==self.grad_accum - 1:
                                (self.policy_optim).step()

        if model:   
            logger["model_gn"] = model_gn 
            logger["shortcut_loss"] = kl
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