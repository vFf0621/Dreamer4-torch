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
from torchmetrics.image import PeakSignalNoiseRatio

import math
class Policy(nn.Module):
    def __init__(self, action_dim: int = 3, latent_dim: int = 128,hidden_dim: int  = 256, mtp: int = 8, num_bins: int = 50,action_low = -1, action_high=1):
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.mtp = mtp
        self.action_low = action_low
        self.action_high = action_high
        # Helper for binning math

        # 1. Main Network
        # Output size: action_dim * num_bins
        self.network = build_network(latent_dim, hidden_dim, 2, "SwiGLU", hidden_dim*4, )
        
        # BC Head (Multi-Step / MTP)
        # Output: [MTP, ActionDim, Bins]
        self.head = nn.Sequential(
                SwiGLU(), 
                nn.RMSNorm(hidden_dim*2), 
                nn.Linear(hidden_dim*2, action_dim * num_bins * mtp)
            )
      #  self.agent_token = nn.Parameter(0.02*torch.randn(1,1,1,latent_dim))

    def forward(self, posterior: torch.Tensor, sample=False):
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

        # --- 2. MTP Head (Auxiliary) ---
        # [B, L, MTP, A, Bins]
        # prob = unimix_probs(prob)
        # --- 3. Sampling ---
        prob = F.softmax(logits, -1)
        dist = td.Independent(td.Categorical(probs=(prob)),3)
        # For training stability, we often use expectation or straight-through sampling
        dist_act =td.Independent(td.Categorical(probs=prob[:,:,0]), 2)
        idx = dist_act.base_dist.probs.argmax(-1)
        sample_idx = dist_act.sample()
        action = two_hot_inv(logits[:,:,0],self.num_bins, self.action_low, self.action_high, temp=1.0)
        # logp of the sampled bins

        logp = dist_act.log_prob(sample_idx)
        if sample:
            action = idx_to_bin_center(sample_idx, self.num_bins, self.action_low, self.action_high)
                 # [B,L]        logp = logp.sum(-1) 
        
        action = action.squeeze(-1)
        return action, logp, dist, dist_act, idx
class GQA(nn.Module):
    def __init__(self, embed_dim=16, num_heads=8, dropout=0.1, causal=False, softcap=50.0,
                 gqa_ratio=2, device="cuda"):
        super().__init__()
        assert embed_dim % num_heads == 0
        # gqa_ratio = query heads per kv head: 1 -> MHA, num_heads -> MQA
        assert num_heads % gqa_ratio == 0, f"num_heads={num_heads} not divisible by gqa_ratio={gqa_ratio}"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_heads // gqa_ratio
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.softcap = float(softcap)

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.norm = nn.RMSNorm(embed_dim)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.device=device
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
        # MANUAL ATTENTION WITH LOGIT SOFT CAPPING
        # =====================================================
        # grow kv heads to match query heads (GQA)
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # soft cap the raw logits BEFORE masking so -inf mask entries stay -inf
        scores = self.softcap * torch.tanh(scores / self.softcap)
        if final_bias is not None:
            scores = scores + final_bias

        # a fully masked query row (e.g. the padded action at t=0 in causal
        # time attention) would softmax to NaN and poison the backward pass;
        # neutralize such rows before softmax and zero their output, like SDPA
        dead_row = torch.isinf(scores).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(dead_row, 0.0)
        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(dead_row, 0.0)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        return self.out_proj(out)


class CausalSTBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, time_attn=True, cap_value=50,
                 mlp_layers=3, modality_sizes=None, gqa_ratio=2,
                 temporal_interleave=False, device="cuda"):
        super().__init__()
        self.time_attn_enabled = time_attn
        # temporal_interleave: time attention runs over the flattened token
        # stream a_1, z_1, (t,d)_1, ... under GQA's standard causal mask,
        # instead of per-channel independent causal attention.
        self.temporal_interleave = temporal_interleave
        self.d_model = d_model
        self.ln_space = nn.RMSNorm(d_model)

        if not self.time_attn_enabled:
            self.space_attn = GQA(d_model, n_heads,  dropout, softcap=cap_value,
                                  gqa_ratio=gqa_ratio, device=device)
        else:
            self.time_attn = GQA(d_model, n_heads, dropout, causal=True, softcap=cap_value,
                                 gqa_ratio=gqa_ratio, device=device)

        self.ln_time = nn.RMSNorm(d_model)
        # modality_sizes splits the token dim into groups (e.g. [Sa, Nz, 1, Nr])
        # that each get their own independent feed-forward; None shares one MLP.
        self.modality_sizes = list(modality_sizes) if modality_sizes is not None else None
        if self.modality_sizes is None:
            self.mlp = build_network(d_model, d_model * 2, mlp_layers, "SwiGLU", d_model, True)
        else:
            self.mlps = nn.ModuleList([
                build_network(d_model, d_model * 2, mlp_layers, "SwiGLU", d_model, True)
                for _ in self.modality_sizes
            ])
        self.device = device
        self.to(self.device)

    def _ff(self, x):
        # x: [B,T,N,D]; token-wise, so modality splitting never mixes tokens
        if self.modality_sizes is None:
            return self.mlp(x)
        assert x.size(2) == sum(self.modality_sizes), (x.size(2), self.modality_sizes)
        outs, i = [], 0
        for size, mlp in zip(self.modality_sizes, self.mlps):
            outs.append(mlp(x[:, :, i:i + size]))
            i += size
        return torch.cat(outs, dim=2)

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

        B, T, N, D = x.shape
        Ncat = N

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
            if self.temporal_interleave:
                # Flatten the per-timestep blocks into the single stream
                # a_1, z_1, (t,d)_1, a_2, z_2, (t,d)_2, ... and apply the
                # standard causal mask over it: a token attends to every token
                # at an earlier position in the stream, plus itself. Within a
                # block this makes the modality order strict (z sees a, not the
                # signal token that follows it); the space layers, which stay
                # bidirectional, are what let z read its own (t,d) token.
                x_time = self.ln_time(x).reshape(B, T * N, D)

                time_kpm = None
                if token_pad_mask is not None:
                    time_kpm = token_pad_mask.reshape(B, T * N)

                xt_out = self.time_attn(
                    x_time,
                    attn_mask=None,      # GQA applies the standard causal tril
                    key_padding_mask=time_kpm,
                ).reshape(B, T, N, D)
            else:
                # Per-channel independent: each of the N channels attends
                # causally over its own history only.
                x_time = self.ln_time(x).permute(0, 2, 1, 3).reshape(B * N, T, D)

                time_kpm = None
                if token_pad_mask is not None:
                    # token_pad_mask: [B,T,N] -> [B,N,T] -> [B*N, T]
                    time_kpm = token_pad_mask.permute(0, 2, 1).reshape(B * N, T)

                xt_out = self.time_attn(
                    x_time,
                    attn_mask=None,
                    key_padding_mask=time_kpm,
                ).reshape(B, N, T, D).permute(0, 2, 1, 3)

            x = x + xt_out
            x = x + self._ff(x)
            return x.squeeze(2) if x.shape[2] == 1 else x

        # =========================
        # SPACE ATTENTION
        # =========================
        x_space = self.ln_space(x)
        x_space = x_space.reshape(B * T, N, D)  # [B*T, N, D]

        # ---- build key_padding_mask for space attn: [B*T, N+R] ----
        space_kpm = None
        if token_pad_mask is not None:
            space_kpm = token_pad_mask.reshape(B * T, N)  # [B*T, N]
            # reserved tokens are never PAD
          

        # ---- build final additive attn mask over [N+R, N+R] ----
        final_mask = None
        if mask is not None:
            if mask.shape[-2:] == (N, N):
                # Expand (N,N) -> (Ncat,Ncat) with "reserved are keys-only":
                # - Everyone may attend to reserved keys (cols N:)
                # - Reserved queries (rows N:) may NOT attend to non-reserved keys (:N)
                # - Optionally: reserved queries only attend to themselves (recommended)

               
                    # additive mask: 0 = keep, -inf = block
                m = torch.zeros((Ncat, Ncat), dtype=mask.dtype, device=x.device)

                    # normal->normal from provided mask (should contain 0/-inf or similar)
                m[:N, :N] = mask

                 
                final_mask = m

            elif mask.shape[-2:] == (Ncat, Ncat):
                final_mask = mask
            else:
                raise ValueError(f"mask has shape {mask.shape}, expected ({N},{N}) or ({Ncat},{Ncat})")
           
        xs = self.space_attn(
            x_space,
            attn_mask=final_mask,          # additive float mask (0 / -inf)
            key_padding_mask=space_kpm,    # True = PAD keys
        )[:, :N]  # drop reserved outputs

        x = x + xs.reshape(B, T, N, D)
        x = x + self._ff(x)
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
        num_reserved = 4,
        gqa_ratio: int = 2,
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
        self.reserved = nn.Parameter(torch.randn(1, 1, num_reserved, d_model) * 0.02)
        self.num_reserved=num_reserved
        blocks = []
        for i in range(depth):
            use_time = ((i+1) % time_every == 0)
            blocks.append(CausalSTBlock(d_model, n_heads, dropout=dropout, time_attn=use_time, gqa_ratio=gqa_ratio))
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
            modality_sizes=[self.num_patches, self.num_reserved],
            device=frames.device
        ) 
        
        proj = self.patch_proj(patches)                             # (B,T,Np,D)
        lat = self.latent_tok.view(1, 1, self.latent_tokens, self.d_model).expand(B, T, -1, -1) 
        x = torch.cat([lat, proj, self.reserved.expand(B,T,-1,self.d_model)], dim=2)                     # (B,T,S,D) with S=L+Np      

        x = self.drop(x)
        for blk in self.blocks:
            if blk.time_attn_enabled:
                x = blk(x, mask=None,)          # no space mask here
            else:
                x = blk(x, mask=space_mask)    # modality mask only here
        x = self.ln_out(x[:,:,:self.latent_tokens])  

        pre = (self.readout(x))
        ztok = torch.tanh(pre)   # [B,T,Np,Dz]
        return ztok

class ReadoutAttention(nn.Module):
    """
    Cross-attention readout: learned query tokens read the processed stream but
    never write into it, so the backbone's z predictions are structurally
    unaffected by the agent while gradients from the readout heads flow back
    through the entire transformer.
    """
    def __init__(self, d_model, n_heads, dropout=0.0, softcap=50.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.num_heads = n_heads
        self.head_dim = d_model // n_heads
        self.softcap = float(softcap)
        self.norm_q = nn.RMSNorm(d_model)
        self.norm_k = nn.RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp = build_network(d_model, d_model * 2, 3, "SwiGLU", d_model, True)
        self.dropout = dropout

    def forward(self, q_tok, kv, attn_bias=None, key_padding_mask=None):
        # q_tok: [B, Tq, D], kv: [B, Tk, D]
        B, Tq, D = q_tok.shape
        Tk = kv.shape[1]
        q = self.q_proj(self.norm_q(q_tok)).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(self.norm_k(kv)).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)

        bias = None
        if attn_bias is not None:
            bias = attn_bias[None, None].to(dtype=q.dtype, device=q.device)  # [1,1,Tq,Tk]
        if key_padding_mask is not None:
            kpm = torch.zeros((B, 1, 1, Tk), dtype=q.dtype, device=q.device)
            kpm.masked_fill_(key_padding_mask[:, None, None, :], float("-inf"))
            bias = kpm if bias is None else bias + kpm

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # soft cap the raw logits BEFORE masking so -inf mask entries stay -inf
        scores = self.softcap * torch.tanh(scores / self.softcap)
        if bias is not None:
            scores = scores + bias
        # neutralize fully masked query rows before softmax (see GQA.forward)
        dead_row = torch.isinf(scores).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(dead_row, 0.0)
        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(dead_row, 0.0)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        out = q_tok + self.out_proj(out)
        return out + self.mlp(out)


class Dynamics(nn.Module):
    """
    Token-based dynamics with discrete signal embeddings + discrete action embeddings.

    Tokens are temporally interleaved: each timestep contributes the block
    [a (Sa), z (Nz), (t,d) (1), reserved (Nr)]. Temporal attention runs over the
    flattened stream a_1, z_1, (t,d)_1, a_2, ... under a standard causal mask,
    so ordering is strict at token granularity; the space attention layers stay
    bidirectional within each timestep and are what let z read its own (t,d)
    signal token.

    The agent token is a pure readout: after the backbone, one learned query
    per timestep cross-attends block-causally over the processed stream to
    produce the policy features. It never enters the stream, so z predictions
    cannot depend on it, while readout gradients flow through the entire
    transformer.

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
        mlp_layers: int = 4,
        gqa_ratio: int = 2,
        Sa: int = 64,
        max_T: int = 255,
        use_agent_token: bool = True,
        latent_tokens: int = 32,
        device: str = "cuda",
        action_low: int = -1,
        action_high: int= 1,
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
        self.Dz = Dz
        self.Nz = latent_tokens
        self.num_task = num_tasks
        self.mask_last_action = mask_last_action
        self.clamp_signal_indices = clamp_signal_indices
        # --- Discrete signal embeddings (tau_idx + k_idx) ---
        self.level_vocab = int(level_vocab)
        self.step_vocab = int(step_vocab)
        self.z_proj = nn.Sequential(nn.RMSNorm(Dz), nn.Linear(Dz, d_model))
        self.sig_proj = nn.Sequential(nn.RMSNorm(level_dim), nn.Linear(level_dim, d_model))
        self.action_low = action_low
        self.action_high = action_high
        self.level_emb = nn.Embedding(self.level_vocab, level_dim)
        self.step_emb  = nn.Embedding(self.level_vocab, level_dim)
        self.action_conditioner = nn.Parameter(0.02 * torch.randn(1, 1, self.Sa, d_model))
        self.reserved = nn.Parameter(0.02 * torch.randn(1, 1, self.Nr, d_model))

        self.action_pad = nn.Parameter(0.02*torch.randn(1, 1, 1, d_model, device=self.device))
        self.action_lookup = action_lookup
        act_embed_dim = self.action_bins*self.action_dim
        # --- Action embedding (true lookup, no one-hot materialization) ---
        self.action_embs = nn.Sequential(nn.RMSNorm(act_embed_dim), nn.Linear(act_embed_dim, self.d_model), nn.RMSNorm(d_model))

        # Agent token (learned readout query, one per task)
        self.agent_token = nn.Parameter(0.02 * torch.randn(1, 1, num_tasks, d_model))
        self.readout_attn = ReadoutAttention(d_model, n_heads, dropout)

        # Blocks: deeper feed-forward than the tokenizer, and one independent
        # MLP per modality (a, z, (t,d), reserved).
        blocks = []
        for i in range(depth):
            use_time = ((i+1) % time_every == (0))
            blocks.append(
                CausalSTBlock(
                    d_model, n_heads,
                    dropout=dropout,
                    time_attn=use_time,
                    mlp_layers=mlp_layers,
                    modality_sizes=[self.Sa, self.Nz, 1, self.Nr],
                    gqa_ratio=gqa_ratio,
                    temporal_interleave=True,
                    device=device,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Output head: model dim -> Dz
        self.out = build_network(d_model, 2*d_model, 3, "SwiGLU", Dz)

        self.to(device)

    def align_actions(self, actions, T, B):
        if actions.dim() != 3:
            raise ValueError(...)
        Ba, Ta, A = actions.shape
        if Ba != B or A != self.action_dim:
            raise ValueError(...)

        if Ta == T:
            return actions[:,:T-1]
        if Ta == T-1:
            return actions
        else:
            raise ValueError("Incorrect action shape")

    def make_kpm_for_actions(self, B, T, S, Nz, *, pad_action_t: int | None, device):
        kpm = torch.zeros((B, T, S), dtype=torch.bool, device=device)
        if pad_action_t is None:
            return kpm
        Sa = self.Sa
        # actions occupy the first Sa positions of each timestep's [a, z, (t,d)] block;
        # do NOT include the signal/agent/reserved tail
        kpm[:, pad_action_t, :Sa] = True
        return kpm

    def forward(
        self,
        z_tokens: torch.Tensor,                 # [B, T, Nz, Dz]
        actions: torch.Tensor | None,           # [B, T-1, A] or [B, T, A]
        signals: torch.Tensor,                  # [B, T, Nz, 2] (level_idx, step_idx)
        policy_tok_in: torch.Tensor | None = None,  # [1,1,1,D] or [B,1,1,D] or [B,T,1,D]
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
        z_tokens = z_tokens.reshape(B,T,self.Nz, self.Dz)
        if signals.size(1) != T:
            raise ValueError(f"signals has T={signals.size(1)} but z_tokens has T={T}.")

        acts_bt = self.align_actions(actions, T, B)          # [B, T-1, A]
        act_two_hot = two_hot(acts_bt, self.action_low, self.action_high, self.action_bins).reshape(B, -1, 1, self.action_bins * self.action_dim)                     # [B, T-1, A]

                                                    # [B, T-1, D]
        a_emb = self.action_embs(act_two_hot[:, :, :])                                                     # [B, T-1, 1, D]
        if a_emb.size(1) == z_tokens.size(1)-1:
            pad = self.action_pad.expand(B, 1, 1, -1)                                     # [B, 1, 1, D]
            a_emb = torch.cat([pad, a_emb ], dim=1)                                        # [B, T, 1, D]
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

        # Temporal interleaving: each timestep contributes the block [a, z, (t,d)].
        # Time layers attend per channel independently and causally; space
        # layers mix the channels bidirectionally within each timestep.
        x = torch.cat([a_tokens, z_inp], dim=2)                  # [B,T,Sa+Nz,D]
        x = torch.cat([x, self.sig_proj(lev_feat + step_feat).unsqueeze(-2)], dim=2)
        token_pad_mask = self.make_kpm_for_actions(B, T, S=x.size(2), Nz=Nz_in, pad_action_t=0, device=device)
        pad_extra = torch.zeros(B, T, self.Nr, dtype=torch.bool, device=device)
        token_pad_mask_aug = torch.cat([token_pad_mask, pad_extra], dim=2)  # [B,T,S]

        reserved = self.reserved.expand(B, T, self.Nr, -1)
        x_aug = torch.cat([x, reserved], dim=2)
        S = x_aug.size(2)

        for blk in self.blocks:
            x_aug = blk(x_aug, token_pad_mask=token_pad_mask_aug)

        # ---- agent readout ----
        # One learned query per timestep cross-attends block-causally over the
        # processed stream. The agent never enters the stream, so the backbone's
        # z predictions cannot depend on it, while gradients from the readout
        # heads flow through the entire transformer.
        if policy_tok_in is not None:
            q = policy_tok_in
            if q.size(0) == 1 and B > 1: q = q.expand(B, -1, -1, -1)
            if q.size(1) == 1 and T > 1: q = q.expand(B, T, -1, -1)
            if q.size(2) != 1:
                raise ValueError("policy_tok_in token dim must be 1 at dim=2")
            q = q[:, :, 0]                                            # [B,T,D]
        else:
            assert task_id < self.num_task
            q = self.agent_token[:, :, task_id].expand(B, T, self.d_model)

        stream = x_aug.reshape(B, T * S, self.d_model)
        stream_kpm = token_pad_mask_aug.reshape(B, T * S)
        t_k = torch.arange(T, device=device).repeat_interleave(S)     # [T*S]
        readout_bias = torch.zeros((T, T * S), dtype=x_aug.dtype, device=device)
        readout_bias.masked_fill_(
            torch.arange(T, device=device).unsqueeze(1) < t_k.unsqueeze(0), float("-inf")
        )
        policy_feat = self.readout_attn(q, stream, attn_bias=readout_bias,
                                        key_padding_mask=stream_kpm)  # [B,T,D]

        # z tokens sit at positions [Sa, Sa+Nz) of every timestep block
        hist = x_aug[:, :, self.Sa:self.Sa + self.Nz, :]

        hist = hist.reshape(B, T*self.Nz, self.d_model)     # [B,T*Nz,D]
        z = self.out(hist)                              # [B,T*Nz,Dz]
        z_pred = z.view(B, T, Nz_in, Dz)            # [B,T,Nz,Dz]

        return z_pred, policy_feat



class Dreamer4(nn.Module):
    # Default sizes: ~430M tokenizer (encoder+decoder at d=1024, depth 12)
    # + 1.56B dynamics (d=1024, depth 16, 4-layer per-modality MLPs),
    # ~1.99B world model total.
    def __init__(self,agent_id, ch=3, h=96, w=96, patch = 16, latent_tokens=32, z_dim=16, action_dim=2, latent_dim=512,
                 rep_depth = 12, rep_d_model=1024, dyn_d_model=1024, dyn_depth=16, num_heads=16, gqa_ratio=2, dropout=0.1, k_max=8, mtp=8, action_low = -1, action_high = 1,
                 policy_bins = 100, reward_bins = 100, pretrain=False, reward_clamp=6,level_vocab = 16, level_embed_dim = 16,
                 batch_lens = (45, 65), batch_size=16, accum=1, max_imag_len=128, ckpt=None, rep_lr=1e-4, rep_decay=1e-3,Sa = 64,eval_context_len=15,
                 dyn_lr=1e-4, dyn_decay=1e-3, policy_lr=1e-4, policy_decay=1e-3, num_tasks=30, task_id = 0, Nr = 4,lambda_=0.8, symlog_for_reward=True, symlog_for_value=True,
                kmax_prob=0.1):
        super(Dreamer4, self).__init__()
        self.encoder =  Encoder(img_channels=ch, h=h, w=w, patch=patch, d_model=rep_d_model,
                                n_heads=num_heads, depth=rep_depth, latent_tokens=latent_tokens, time_every=2,
                                out_dim=z_dim, dropout=dropout, max_T=max_imag_len, num_reserved=Nr,
                                gqa_ratio=gqa_ratio)
        self.ema = 0.98
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.pretrain = False
        self.agent_id = agent_id
        self.eval_ctx = eval_context_len
        # --- TokenDynamics --      -
        # level_vocab, step_vocab match
        self.sym_rew = symlog_for_reward

        self.lambda_=lambda_
        self.decoder = Decoder(img_channels=ch, w = w, h=h, patch=patch, z_dim=z_dim, d_model=rep_d_model, n_heads=num_heads,
                               depth=rep_depth, latent_tokens=latent_tokens, time_every=2, dropout=dropout, max_T=max_imag_len, num_reserved=Nr,
                               gqa_ratio=gqa_ratio)
        self.imagination_steps = max_imag_len - 1
        self.rminv = -reward_clamp
        self.rmaxv = reward_clamp
        self.reward_bins = reward_bins
        self.task_id = task_id
        self.sym_val = symlog_for_value
        self.batch_lengths= batch_lens
        self.horizon_length = max_imag_len
        self.grad_accum = accum       
        self.batch_size = batch_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_dim=action_dim
        self.shortcut_kmax = k_max
        self.z_buffer = None
        self.aux_horizon = mtp+1
        self.policy_num_bins = policy_bins
        self.bin_num = reward_bins
        self.steps = 0
        self.kmax_prob = kmax_prob
        self.disc = 0.997
        self.expanding_ratio = 1

      #  self.encoder.load_state_dict(torch.load("enc.pt"))
        self.psnr = PeakSignalNoiseRatio((-1,1))

       # self.decoder.load_state_dict(torch.load("dec.pt"))
        self.tau_ctx = 0.1
        self.lpips = LPIPSLoss(reduction="none",)

    
        self.policy = Policy(action_low = action_low,action_high = action_high,action_dim=action_dim,hidden_dim = latent_dim,latent_dim=dyn_d_model, mtp=self.aux_horizon, num_bins=self.policy_num_bins)
        self.t_policy = Policy(action_low = action_low,action_high = action_high, action_dim=action_dim, hidden_dim = latent_dim, latent_dim=dyn_d_model, mtp=self.aux_horizon, num_bins=self.policy_num_bins)
        self.reward = Reward(bin_num=self.bin_num, latent_dim=dyn_d_model, mtp=self.aux_horizon, hidden_dim = latent_dim, r_max=reward_clamp, sym=self.sym_rew)
        self.train_policy = False
        self.value = Value(bin_num=self.bin_num, latent_dim=dyn_d_model, hidden_dim=latent_dim, r_max=reward_clamp,sym= self.sym_val)
        self.t_value = Value(bin_num=self.bin_num, latent_dim=dyn_d_model, hidden_dim=latent_dim, r_max=reward_clamp, sym=self.sym_val)

        self.transformer = Dynamics(
            device=self.device, 
            Dz = int(z_dim / self.expanding_ratio),
            use_agent_token=True, 
            level_vocab=self.shortcut_kmax + 1,
            level_dim=level_embed_dim,
            dropout=dropout,
            action_low = action_low,
            action_high = action_high,
            n_heads=num_heads,
            action_dim =action_dim,
            d_model=dyn_d_model,
            depth=dyn_depth,
            gqa_ratio=gqa_ratio,
            Sa = Sa,
            Nr = Nr,
            max_T = max_imag_len,
            num_tasks=num_tasks,
            time_every=4,
            latent_tokens=latent_tokens * self.expanding_ratio
        )


        if ckpt:
            self.load_state_dict(torch.load(ckpt), strict=False)

        self.t_policy.load_state_dict(self.policy.state_dict(), )
        self.t_value.load_state_dict(self.value.state_dict(), )

        self.lpips = LPIPSLoss(reduction="none",)
        self.rep_optim = torch.optim.AdamW(
            [{'params': self.encoder.parameters()}, 

            {'params': self.decoder.parameters()},

            ]
        , lr=rep_lr, capturable=True, weight_decay=rep_decay)
        self.reset()

        self.dyn_optim = torch.optim.AdamW(            
            [{'params': self.policy.parameters()}, 
            {'params': self.t_policy.parameters()}, 

            {'params': self.reward.parameters()}, 
            {'params': self.transformer.parameters()}, ],

            lr=dyn_lr, capturable=True,weight_decay=dyn_decay)

        self.value_optim= torch.optim.AdamW([
            {'params': self.value.parameters()},
            ], lr=policy_lr, capturable=True,weight_decay=policy_decay)
   
        self.to(self.device)

    def reset(self):
        self.state_buffer = None
        self.action_buffer = torch.zeros(1, 0, self.action_dim, device=self.device)

    def mix_tau_ctx(self, src):
        return src*(1-self.tau_ctx) + self.tau_ctx*torch.randn_like(src)
    def action_step(self, s):
        """
        One environment step -> one action.

        Uses "clean-level" signals for the transformer context.
        NOTE: whether "clean" is tau_idx=N or tau_idx=N-1 depends on your LUT convention.
            In your earlier code, tau_lut has length N+1 and uses tau_idx==N as the special max/clean entry.
            So we use tau_idx=N here.
        """
        with torch.no_grad():
            device = self.device
            s = s.to(device)

            # -----------------------------
            # init buffers
            # -----------------------------
            if self.state_buffer is None:
                # [B=1, T=1, ...]
                self.state_buffer = s.unsqueeze(0).unsqueeze(0)
            else:
                self.state_buffer = torch.cat(
                    [self.state_buffer, s.unsqueeze(0).unsqueeze(0)], dim=1
                )

            # Encode full state context: [B, T, Nz, Dz]
            z = self.encoder(self.state_buffer)
            B, T, Nz, _ = z.shape

            # Make sure action_buffer exists and is correctly aligned:
            # transformer typically expects actions of length T-1 (transitions) or T depending on your impl.
            if self.action_buffer is None:
                # We'll build actions as [B, 0, A] initially, then append one per step.
                # You MUST know action_dim. If you have it stored, use it; otherwise infer from policy output later.
                A = int(getattr(self, "action_dim", 2))
                self.action_buffer = torch.zeros((B, 0, A), device=device, dtype=z.dtype)

            # -----------------------------
            # Signals: clean context
            # -----------------------------
            N = int(getattr(self, "shortcut_kmax", 64))
            # If your convention is tau_idx in [0..N] with tau_idx==N being the special "clean/max" entry:
            sigs = self.make_signals_indices(B, T, Nz, tau_idx=N, k_idx=N-1)

            # -----------------------------
            # Optional: small eval noise for robustness
            # -----------------------------
            z_in = self.mix_tau_ctx(z) 
            # -----------------------------
            # Align actions to context length
            # -----------------------------
            # If we have T states in buffer, we have (T-1) actions that lead between them.
            # So feed at most T-1 actions.
           # [B, T-1, A]
            # Forward world model / transformer
            # Expected signature (based on your earlier code):
            if self.action_buffer.size(1) == z_in.size(1):
                self.action_buffer = self.action_buffer[:, 1:]
 
            actions_ctx = self.action_buffer
            assert actions_ctx.size(1)  == z_in.size(1) -1 
            z_pred, h = self.transformer(z_in, actions_ctx[:,:], signals=sigs)

            a, *_ = self.policy(h[:, -1:])

            # --- append then re-enforce invariant ---
            self.action_buffer = torch.cat([self.action_buffer, a], dim=1)

            self.state_buffer = self.state_buffer[:,-self.eval_ctx:]
            return a[0, 0].detach().cpu().numpy()


    def evaluate(self, buffer, steps=4): 
        s, a = buffer.sample_seq( 1, 64)[:2]
        s = torch.from_numpy(s).to(self.device).float()
        a = torch.from_numpy(a).to(self.device).float()
        
        with torch.no_grad():
            z = self.encoder(s)
            z_eval = self.latent_imagination(z[:,:], a[:,:], num_iter=a.size(1) - 1, eval_=False,random=False, forced=True)[0]
            self.decode_and_save(z_eval, "single_frame")
            z_eval = self.latent_imagination(z[:,:], a[:], num_iter=a.size(1) - 1, eval_=False,random=True, forced=False)[0]
            self.decode_and_save(z_eval, 'random')
        return
    

    def decode_and_save(self, z_in, name):
            decoded = self.decode(z_in).detach().cpu().numpy()
            for i in range(decoded.shape[1]):
                cv2.imwrite(f"./eval_imgs/{name}_{i}.png", 255 * decoded[0, i].transpose(1, 2, 0)[..., ::-1])
                
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
        noise_min = self.tau_ctx
        tau_max = 1.0 - noise_min
        assert actions.size(1) == z_prev.size(1)
        tau_lut = torch.arange(N + 1, device=device, dtype=torch.float32) / float(N)
        tau_lut[-1] = tau_max
        # context is "cleanest" (tau_idx=N)
        tau_ctx = torch.full((B, T, 1), N, device=device, dtype=torch.long)
        k_ctx = torch.ones_like(tau_ctx) * stride

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
                packed, actions[:,:], sigs, task_id=self.task_id
            )[0][:, -1:]

            v = (x1_hat - z) / max(1e-5, 1.0 - tau)
            z = z + v * d_tau
        return z.clamp(-1, 1) if clamp else z

    def shortcut_forcing(self, z_t, actions, mask: torch.Tensor | None = None):


        device = z_t.device
        assert z_t.size(1) == actions.size(1) or (z_t.size(1)-1) == actions.size(1)

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
        noise_min = self.tau_ctx
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
        x1_hat = self.transformer(
            z_targ,
            actions[:, :-1],
            signals=self.make_signals_indices(B, T, 1, tau_idx, k_pow),
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

            z1_prime = self.transformer(
                z_targ,
                actions[:, :-1],
                signals=self.make_signals_indices(B, T, 1, tau_idx, k_half),
            )[0][:, :]

            v1 = (z1_prime - z_targ) / (1.0 - tau).clamp(min=1e-5)
            v1 = v1 * valid_mid

            z_mid = z_targ + v1 * d_half

            z1_mid = self.transformer(
                z_mid,
                actions[:, :-1],
                signals=self.make_signals_indices(B, T, 1, tau_mid_idx, k_half),
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
        return loss, self.psnr(x1_hat[is_base[...,None].expand_as(x1_hat)], z1_clean.detach()[is_base[...,None].expand_as(x1_hat)])


    def save_checkpoint(self, name):
        torch.save(self.state_dict(), f"./ckpts/Agent-{self.agent_id}-Task-{self.task_id}-" + name)
    def save_rep(self, name):
        torch.save(self.encoder.state_dict(), "./ckpts/" + "enc.pt")
        torch.save(self.decoder.state_dict(), "./ckpts/"+"dec.pt")
    def latent_imagination(
            self,
            initial_latent,
            actions,
            num_iter = None,
            eval_=True,
            detach=False,
            offset = 1,
            random=False,
            steps=4,
            get_feat=False,
            forced=False,
        ):
            device = initial_latent.device
            if not num_iter:
                num_iter = initial_latent.size(1)
            # FIX 1: Do NOT slice [:1]. Keep the full context history provided.
            z_all = initial_latent.detach()[:,:]
            # Initialize executed actions. 
            # If actions are provided (Reward training), we use them. 
            # If None (Policy), we start empty.
            a_exec = None
            kl_list, h_list, lp_list, a_list = [], [], [], []
            z_list =[]
            N = int(getattr(self, "shortcut_kmax", 8)) 
            if not get_feat:
                z_inp=z_all[:,:offset]
            else:
                z_inp = z_all

            a_exec = actions[:,:z_inp.size(1)-1]
            if not eval_ and not get_feat:
                z_inp = z_all[:,:1]
                a_exec = actions[:,:0]
        

            for i in range(num_iter):
                # Current state input
                
                B, T_curr, Nz, _ = z_inp.shape
                T_curr = min(i+1, T_curr)
                # Add noise for robustness
                # --- Dynamics Step (Next State) ---
                    # Training: Use Ground Truth (Teacher Forcing)
                    # If initial_latent contains the full sequence, we just advance the window

                # --- Feature Extraction ---
                if get_feat:
                    z_act = z_all[:,:i+1]
                    z_act = self.mix_tau_ctx(z_act)
                    # Policy Mode: Run transformer on accumulated history
                    z_last, policy_feat = self.transformer(
                            z_act, 
                            actions[:,:i], 
                            signals=self.make_signals_indices(B,z_act.size(1), 1, N, N-1), 
                            task_id=self.task_id,
                        )
                    z_list.append(z_last[:,-1:])
                    h_list.append(policy_feat[:,-1:])
                    if i == num_iter-1:
                        return torch.cat(z_list, 1), torch.cat(h_list, 1)
                    else:
                        continue
                elif eval_ and not random:
                    z_act = z_inp[:,:]
                    z_act = self.mix_tau_ctx(z_act)
                    # Policy Mode: Run transformer on accumulated history
                    assert z_act.size(1) == a_exec.size(1) + 1
                    with torch.no_grad():
                        _, policy_feat = self.transformer(
                            z_act, 
                            a_exec[:,:], 
                            signals=self.make_signals_indices(B,z_act.size(1), 1, N, N-1), 
                            task_id=self.task_id,
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
                        signals=self.make_signals_indices(B, T_curr, 1, N, N-1), 
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
                    a, log_p, _, base_p, idx = self.policy(h_last, sample=True)
                    a_list.append(a)
                    lp_list.append(log_p.unsqueeze(-1))
                    # Compute KL divergence for auxiliary loss
                    with torch.no_grad():
                        _, log_q, _, base_q, _ = self.t_policy(h_last)
                    kl_step = kl_div(base_p.base_dist.logits, base_q.base_dist.logits).sum(-1)
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
            return z_inp, h, lp, imagined_actions
    def soft_update(self, target_net: nn.Module, online_net: nn.Module):
        """
        Updates the target network parameters using an exponential moving average.
        tau: The soft update coefficient (usually between 0.001 and 0.05).
        """
        tau=1-self.ema
        with torch.no_grad():
            for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
                # target = target + tau * (online - target)
                # which is mathematically identical to: target = (1-tau)*target + tau*online
                target_param.data.lerp_(online_param.data, tau)
    def decode(self, latents):
        return self.decoder(latents)
    def multistep_aux_losses(self, feat, actions, rewards, termination, t_policy=False):
        """
        Auxiliary Loss:
        Predict NEXT K steps for both Action and Reward.
        Target at step t:
        - Actions: a_t, ..., a_{t+K-1}
        - Rewards: r_t, ..., r_{t+K-1}
        """

        B, T_feat, D = feat.shape
        K = int(self.aux_horizon)
        device = feat.device

        # -----------------------
        # Normalize rewards shape
        # -----------------------
        if rewards.dim() == 3 and rewards.size(-1) == 1:
            rewards_1d = rewards[..., 0]
        elif rewards.dim() == 2:
            rewards_1d = rewards
        else:
            raise ValueError(f"rewards shape unexpected: {tuple(rewards.shape)}")

        # -----------------------
        # Normalize actions shape
        # -----------------------
        if actions.dim() == 2:
            actions_ = actions.unsqueeze(-1)           # [B, T, 1]
        elif actions.dim() == 3:
            actions_ = actions                         # expect [B, T, Da]
        else:
            raise ValueError(f"actions shape unexpected: {tuple(actions.shape)}")

        # If user accidentally passed [B, Da, T], fix it.
        action_dim = getattr(self, "action_dim", actions_.size(-1))
        if actions_.size(-1) != action_dim and actions_.size(1) == action_dim:
            actions_ = actions_.transpose(1, 2).contiguous()  # [B, T, Da]

        T_a = actions_.size(1)
        T_r = rewards_1d.size(1)

        # -----------------------
        # Useable length: Now covering the ENTIRE sequence (no -1 limits)
        # -----------------------
        L_use = min(T_feat, T_a, T_r)
        L_use = max(0, L_use)

        feat = feat[:, :L_use]

        # ======================================================================
        # ACTION HEAD
        # ======================================================================
        a_t = actions_[:, :]            # [B, T_a, Da]
        req_len_a = L_use + K - 1
        
        # Calculate padding needed to ensure the window can slide to the end
        pad_len_a = max(0, req_len_a - T_a)
        if pad_len_a > 0:
            a_padded = F.pad(a_t, (0, 0, 0, pad_len_a), mode="constant", value=0.0)
        else:
            a_padded = a_t

        # PyTorch unfold on dim 1 always creates [B, Windows, Da, K]
        a_win = a_padded.unfold(dimension=1, size=K, step=1)   
        a_tgt = a_win[:, :L_use].contiguous()

        # Forward
        out = self.policy(feat) if not t_policy else self.t_policy(feat)
        _, _, dist, _, _ = out
        logits_bc = dist.base_dist.logits[:, :L_use]            # [B, L_use, K_pred, Da, bins]

        # Align horizons
        K_pred = logits_bc.shape[2]
        K_use = min(K, K_pred)
        logits_bc = logits_bc[:, :, :K_use]                     

        # ======================================================================
        # PREPARE TERMINATION TARGETS & DONE MASK
        # ======================================================================
        t_tp1 = termination[:, :]         # [B, T_r]
        req_len_t = L_use + K - 1
        pad_len_t = max(0, req_len_t - T_r)
        
        if pad_len_t > 0:
            t_padded = F.pad(t_tp1, (0, pad_len_t), mode="constant", value=0.0)
        else:
            t_padded = t_tp1

        t_win = t_padded.unfold(dimension=1, size=K, step=1)      # [B, L_use, K]
        t_tgt = t_win[:, :L_use].contiguous()[:, :, :K_use]       # [B, L_use, K_use]

        # Create done_mask: True for everything up to and including the first 'done'.
        done_mask = (torch.cumsum(t_tgt.float(), dim=-1) - t_tgt.float()) <= 0  

        # ======================================================================
        # FIX TARGET SHAPES & MASKS
        # ======================================================================
        # Permute from [B, L_use, Da, K] -> [B, L_use, K, Da] 
        a_tgt = a_tgt.permute(0, 1, 3, 2).contiguous()
        a_tgt = a_tgt[:, :, :K_use]

        # Valid Index Mask: Ensure the target isn't pulling from the padded zero zone
        t_idx = torch.arange(L_use, device=device).view(1, L_use, 1)
        k_idx = torch.arange(K_use, device=device).view(1, 1, K_use)
        
        # FIX: Strict less-than T_a prevents the off-by-one padding leak
        a_mask = (t_idx + k_idx) < T_a                                # [1, L, K_use]
        a_mask = a_mask & done_mask                                   # [B, L, K_use]

        raw_act_loss = soft_ce(
            logits_bc,              
            a_tgt,                  
            self.policy_num_bins,
            self.policy.action_low,self.policy.action_high,
        ).mean([-2,-1])  
        
        a_mask_exp = a_mask.expand_as(raw_act_loss)                   
        act_loss = (raw_act_loss * a_mask_exp).sum() / (a_mask_exp.sum() + 1e-6)

        # ======================================================================
        # REWARD HEAD
        # ======================================================================
        r_tp1 = rewards_1d[:, :]         
        req_len_r = L_use + K - 1
        pad_len_r = max(0, req_len_r - T_r)
        
        if pad_len_r > 0:
            r_padded = F.pad(r_tp1, (0, pad_len_r), mode="constant", value=0.0)
        else:
            r_padded = r_tp1

        r_win = r_padded.unfold(dimension=1, size=K, step=1)      
        r_tgt = r_win[:, :L_use].contiguous()[:, :, :K_use]       
        
        r_mask = (t_idx + k_idx) < T_r                               # [1, L, K_use]
        r_mask = r_mask & done_mask                                  

        rew_pred, _, rew_logits, term_dist = self.reward(feat)
        rew_logits = rew_logits[:, :L_use, :K_use]                
        
        raw_rew_loss = soft_ce(
            rew_logits,
            r_tgt,
            self.reward_bins,
            self.rminv, self.rmaxv, self.sym_rew
        ).mean([-1]) 

        r_mask_exp = r_mask.expand_as(raw_rew_loss)
        rew_loss = (raw_rew_loss * r_mask_exp).sum() / (r_mask_exp.sum() + 1e-6)

        # ======================================================================
        # Termination HEAD 
        # ======================================================================
        term_logits = term_dist.logits
        term_logits = term_logits[:, :L_use, :K_use]                

        term_loss = F.binary_cross_entropy_with_logits(
            term_logits,
            t_tgt.float(), 
            reduction='none'
        ) 
        
        t_mask = r_mask.clone() 
        t_mask_exp = t_mask.expand_as(term_loss)

        term_loss = (t_mask_exp * term_loss).sum() / (t_mask_exp.sum() + 1e-6)

        return act_loss, rew_loss, term_loss

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
        self.rep_optim.zero_grad()
        self.dyn_optim.zero_grad()

        self.value_optim.zero_grad(set_to_none=True)
        for i in tqdm(range(self.grad_accum)):
            s = buffer.sample_seq(self.batch_size, k)

            states, actions, reward, termination = s
       
            states = torch.from_numpy(states).to(self.device).float()
            actions = torch.from_numpy(actions).to(self.device).float()
            reward = torch.from_numpy(reward).to(self.device).float()
            termination = torch.from_numpy(termination).to(self.device).float()

            if not model and not policy and not train_reward:
                self.encoder.train()
                self.decoder.train()
              #  self.canonical_decoder.train()
               # self.canonical_decoder1.train()

                z_t  = self.encoder(states)

                B,T = z_t.shape[:2]
  


                reconst = self.decode(z_t)
                reconst = 2 * reconst - 1
                states = 2 * states - 1

                mse = F.mse_loss(reconst, states, reduction="mean")
                lp = (self.lpips(reconst, states)).mean()
                reconst_loss = mse+0.2 * lp 
                total_loss = reconst_loss 
                (total_loss/self.grad_accum).backward()
                psnr=self.psnr(reconst,states)
              #  decoder1_gn = adaptive_grad_clip(self.canonical_decoder, 0.3)
             #   decoder2_gn = adaptive_grad_clip(self.canonical_decoder1, 0.3)

                if i == self.grad_accum-1:
                    encoder_gn = adaptive_grad_clip(self.encoder, 0.3)
                    decoder_gn = adaptive_grad_clip(self.decoder, 0.3)

                    (self.rep_optim).step()
                    self.rep_optim.zero_grad()
            if model or policy:
                    
                    # Create the full trajectory first
                    # shape: (Batch, Sequence_Length + 1, ...)
                full_sequence = states


                        # Apply mask and encode once
                        # This ensures s_{t+1} has the same embedding whether it's looked at as "current" or "next"
                z_t = self.encoder(full_sequence, ).detach()
                        # Split into current and next steps
                kl_loss, psnr = self.shortcut_forcing( z_t, actions)
                kl = kl_loss.detach().item()
                dyn_loss = kl_loss
                if train_reward or policy:
                    pass
                else:                        # Scale the loss before backward
                    (dyn_loss/self.grad_accum).backward()
                        # [B,T,D]
                                                # Unscale gradients before clipping (important!)
                    model_gn = adaptive_grad_clip(self, 0.3)
                            
                if i == self.grad_accum-1 and not train_reward:
                    (self.dyn_optim).step()
                    self.dyn_optim.zero_grad()
        
            if train_reward or policy:
                clean_latents = self.encoder(states).detach()
                B, T, Nz, _ = clean_latents.shape
                
                noised = self.mix_tau_ctx(clean_latents)

                N = self.shortcut_kmax
                h_with_grad = self.transformer(noised, actions[:,:-1], signals=self.make_signals_indices(B, noised.size(1), 1, N, N-1))[1]
            
                action_loss, reward_loss, term_loss = self.multistep_aux_losses(h_with_grad[:,:], actions[:,:], reward[:,:], termination.float(), policy)
                                                
             
                ac_loss =  0.1*(action_loss+reward_loss+term_loss) + 2 *  dyn_loss
                if not policy:
                    (ac_loss.mean()/self.grad_accum).backward()
                if i == self.grad_accum-1 and not policy:
                    model_gn = adaptive_grad_clip(self, 0.3)
                    self.dyn_optim.step()
                    self.dyn_optim.zero_grad()
            if policy:
                with FreezeParameters([
                                            self.encoder,
                                            self.transformer,
                                            self.decoder,

                                        ]): 

                    rollout_states = states

                    initial_latent = self.encoder(states[:, :])
                    B, T, Nz, _ = initial_latent.shape

                    H = torch.randint(low=self.batch_lengths[0],high=self.horizon_length, size=(1,))[0].item()
                    z_0 = initial_latent[:,:].detach()
                    rollout_z = self.encoder(rollout_states)
                    imag_z, h_t, lp, imagined_actions = self.latent_imagination(
                                rollout_z[:,:1], actions[:, :0], num_iter=self.imagination_steps) 
                    mixed_z = self.mix_tau_ctx(z_0)
                    N = self.shortcut_kmax

              
                    disc= self.disc

                    with torch.no_grad():
                                                                     
                        r_full, _, _, term = self.reward(h_t[:, :])
                        r_seq = r_full[:,:]
                        term_probs = term.probs
                        term_probs = (term_probs[:,:,0])
                        cont_state =1-(term_probs).float()   
                        r_seq = (r_seq[...,0] ) 
                        cont_t = cont_state[:, :]         
                        cont_t = mask_after_done(cont_t, cont_t)
                        r_rp, term_rp = reward, termination
                        V_rp = self.value(h_with_grad[:,:])[0] 
                        cont_rp = (1-term_rp[:,:])

                        cont_rp = mask_after_done(cont_rp, cont_rp)

                        V_full = self.value(h_t[:,:])[0] 
                        r_rp_seq = r_rp * mask_after_done(r_rp, cont_rp) 
                        r_seq = r_seq * mask_after_done(r_seq, cont_t) 
                        bs = lambda_returns(r_seq[:,:], cont_t[:,:],  lambda_=self.lambda_, discount=self.disc, boot=V_full[:, :]).squeeze(-1) 
                        rp_targ = lambda_returns(r_rp_seq, cont_rp,  lambda_=self.lambda_, discount=self.disc, boot=V_rp[:, :]).squeeze(-1) 
                        weight_rp = torch.cumprod(disc * (cont_rp[:,:-1] > 0.5), 1) /disc 
                    rp_targ=torch.cat([rp_targ, 0 * rp_targ[:, -1:]], 1)

                    V_rp_pred = self.value(h_with_grad[:,:].detach())[1] 
                    rploss = (soft_ce(V_rp_pred,rp_targ , self.reward_bins, self.rminv, self.rmaxv, self.sym_val)[:,:-1, 0] * weight_rp).mean()
                    weight = torch.cumprod(disc * (cont_t[:,:] > 0.5), 1)/disc 
                    
                    adv = (bs - self.value(h_t[:,:-1])[0].squeeze(-1).detach()) * weight[:,:-1]                    
                    V_pred = self.value(h_t[:,:].detach())[1]
                    bs_padded=torch.cat([bs, 0 * bs[:, -1:]], 1)
                    V_targ = self.t_value(h_t)[0].detach()
                    value_loss = (weight[:,:-1] * soft_ce(V_pred[:,:], V_targ, self.reward_bins, self.rminv, self.rmaxv, self.sym_val)[:,:-1, 0]).mean()
                    value_loss = value_loss + (weight[:,:-1]  * soft_ce(V_pred, bs_padded, self.reward_bins, self.rminv, self.rmaxv, self.sym_val)[:,:-1, 0]).mean()
                    value_loss = value_loss + rploss

                    lp_t = mask_after_done(cont_t, cont_t) * lp

                    base = torch.ones_like(adv)
                    _,_,_,p,_ = self.policy(h_with_grad.detach())
                    with torch.no_grad():
                        _,_,_, q,_ = self.t_policy(h_with_grad)
                    kl = td.kl_divergence(p, q).mean()
                    pos = -(((adv>=0) * lp_t[:,:-1])*weight[:,:-1]).sum() / (base[adv >= 0]).sum().clamp(min=1)

                    neg =  (((adv<0) * lp_t[:,:-1])*weight[:,:-1]).sum()/ (base[adv < 0]).sum().clamp(min=1)
                    actor_loss = 0.5*(pos + neg).mean()+ 0.3*((kl))
                    ((actor_loss + value_loss +ac_loss.mean())/self.grad_accum).backward()
                    if i==self.grad_accum - 1:
                        model_gn =adaptive_grad_clip(self, 0.3)
                        self.dyn_optim.step()

                        (self.value_optim).step()
                        self.soft_update(self.t_value, self.value)


        if model:   
            logger["model_gn"] = model_gn 
            logger["shortcut_loss"] = dyn_loss.item()
            logger["psnr"] = psnr.item()
        if train_reward:
            logger["finetune_bc_loss"] = action_loss.mean().detach().item()
            logger["reward_loss"] = reward_loss.detach().item()
            logger["termination_loss"] = term_loss.mean().detach().item()
            logger["model_gn"] = model_gn

        if policy:
            logger["value_loss"] = value_loss.detach().item()
            logger["kl_prior"]= kl.mean().detach().item()
            logger["r_min"]= r_seq.min().detach().item()
            logger["r_max"]= r_seq.max().detach().item()
            logger["log_pi_mean"]= lp_t.mean().detach().item()
            logger["log_pi_std"]= lp_t.std().detach().item()
            logger["rollout_mean"]= bs.mean().detach().item()
            logger["rollout_max"]= bs.max().detach().item()
            logger["rollout_min"]= bs.min().detach().item()
            if not isinstance(actor_loss, float):
                logger["actor_loss"] = actor_loss.detach().item()
        elif not model and not train_reward and not policy:
            logger["reconst"] = reconst_loss.detach().item()
            logger["encoder_gn"] = encoder_gn 
            logger["decoder_gn"] = decoder_gn 
            logger["psnr"] = psnr.item()

        return logger

class Reward(nn.Module):
    def __init__(self, bin_num: int, latent_dim: int, hidden_dim: int = 256, mtp: int = 1,r_max: float = 6, sym=False,binary=True):
        super().__init__()
        self.mtp = int(mtp)
        self.bin_num = int(bin_num)
        self.latent_dim = int(latent_dim)

        self.network = build_network(latent_dim, 2*hidden_dim, 2, "SwiGLU", hidden_dim*2)
        self.r_max = r_max
        self.reward_head = nn.Sequential(
            SwiGLU(),
            nn.RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, self.mtp * self.bin_num),
        )

        self.term_head = nn.Sequential(
            SwiGLU(),
            nn.RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, self.mtp),
        )
        self.sym=sym
    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape
        h = self.network(x)  

        r_logits_all = self.reward_head(h).view(B, L, self.mtp, self.bin_num)  
        r_logits_1 = r_logits_all[:, :, 0]                                     
        r_mean = two_hot_inv(r_logits_1, self.bin_num, -self.r_max, self.r_max, self.sym)                                       

        term_logits = self.term_head(h)                                        
        term_logits = term_logits.squeeze(-1)                              

        term_dist = td.Bernoulli(logits=term_logits)
        return r_mean, r_logits_1, r_logits_all, term_dist

class Value(nn.Module):
    def __init__(self,latent_dim, hidden_dim,bin_num, num_layers=4, r_max=6,sym=True ):
        super().__init__()
        self.network = build_network(latent_dim, hidden_dim, num_layers, "SwiGLU",bin_num)
        self.bin_num = bin_num
        self.sym=sym
        self.r_max=r_max
    def forward(self, x):
        x = self.network(x)
        return two_hot_inv(x, self.bin_num, -self.r_max, self.r_max, self.sym), x

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
        num_reserved = 4,
        gqa_ratio: int = 2,
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
        self.reserved = nn.Parameter(torch.randn(1, 1, num_reserved, d_model) * 0.02)
        self.num_reserved=num_reserved

        self.patch_queries = nn.Parameter(torch.randn(1, 1, self.num_patches, d_model) * 0.02)

        self.z_to_latents = nn.Linear(z_dim, latent_tokens * d_model)
        self.z_tok_proj   = nn.Linear(z_dim, d_model)
        self.drop = nn.Dropout(dropout)
        blocks = []
        for i in range(depth):
            use_time = ((i+1) % time_every == 0)
            blocks.append(CausalSTBlock(d_model, n_heads, dropout=dropout, time_attn=use_time, gqa_ratio=gqa_ratio))
        self.blocks = nn.ModuleList(blocks)
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
        x = torch.cat([zlat, pq, self.reserved.expand(B, T, -1, self.d_model)], dim=2)   # (B,T,L+Np,D)
        x = self.drop(x)
        space_mask = modality_mask(L, [self.num_patches, self.num_reserved], encoder=False, device=x.device)

        # ---- transformer ----
        for blk in self.blocks:
            if blk.time_attn_enabled:
                x = blk(x)          # no space mask here
            else:
                x = blk(x,  mask=space_mask)    # modality mask only here
        x = self.ln_out(x)

        # ---- decode ONLY patch tokens ----
        patch_tok = x[:, :, L:-self.num_reserved, :]           # (B,T,Np,D)
        # sanity check
        assert patch_tok.shape[2] == self.num_patches, (patch_tok.shape, self.num_patches)

        patches = self.to_patch(patch_tok).contiguous().view(B * T, self.num_patches, -1)
        frames = self.unpatchify(patches).view(B, T, self.img_channels, self.h, self.w)

        return torch.sigmoid(frames) if self.output_range == "0_1" else torch.tanh(frames)
