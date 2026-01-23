import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.distributions.transforms import TanhTransform
import math
from collections.abc import Iterable

def soft_ce(pred, target, bins, minv, maxv):
    """Computes the cross entropy loss between predictions and soft targets."""
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, minv, maxv, bins).squeeze(-2)
    return -(target * pred).sum(-1, keepdim=True)


def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def two_hot(x, minv, maxv, bins):
    """Converts scalars (in last dim) to soft two-hot over NUM_BINS bins."""
    MAX_VAL=maxv
    MIN_VAL=minv
    NUM_BINS=bins
    x_clamped = torch.clamp(symlog(x), MIN_VAL, MAX_VAL)
    orig_shape = x_clamped.shape                      
    flat_x = x_clamped.reshape(-1)                    

    step = (MAX_VAL - MIN_VAL) / (NUM_BINS - 1)
    pos = (flat_x - MIN_VAL) / step                   

    bin_idx = torch.floor(pos)                        
    bin_offset = pos - bin_idx                        

    bin_idx = bin_idx.clamp(0, NUM_BINS - 1)
    soft_two_hot = torch.zeros(flat_x.shape[0], NUM_BINS, device=x.device, dtype=x.dtype)

    bin_idx = bin_idx.long().unsqueeze(-1)            
    bin_offset = bin_offset.unsqueeze(-1)             

    soft_two_hot = soft_two_hot.scatter(1, bin_idx, 1 - bin_offset)
    upper_idx = (bin_idx + 1).clamp(max=NUM_BINS - 1)
    soft_two_hot = soft_two_hot.scatter(1, upper_idx, bin_offset)

    new_shape = (*orig_shape, NUM_BINS)               
    return soft_two_hot.view(*new_shape)

def two_hot_inv(x, bin_num, minv, maxv):
    """Converts a batch of soft two-hot encoded vectors to scalars."""
    dreg_bins = torch.linspace(-minv, maxv, bin_num, device=x.device, dtype=x.dtype)
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
    return symexp(x)

def build_network(input_size, hidden_size, num_layers, activation, output_size, rms=True):

    layers = []
    in_dim = input_size
    
    # hidden blocks (num_layers-1 of them)
    for _ in range(num_layers - 1):
        layers.append(nn.RMSNorm(in_dim))

        layers.append(nn.Linear(in_dim, 2 * hidden_size))
        layers.append(SwiGLU())
        in_dim = hidden_size
    layers.append(nn.RMSNorm(hidden_size))

    # output projection
    layers.append(nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers)

def apply_random_patch_mask(
    images: torch.Tensor,
    patch_size: int = 16,
    mask_ratio: float | None = None,   # if None -> per-frame random in [0, max_mask_ratio]
    max_mask_ratio: float = 0.9,
    return_ratios: bool = False,
):
    """
    images: (B,C,H,W) or (B,T,C,H,W)
    Returns:
      - masked images with same shape as input
      - mask_pixel: (B,T,Hc,Wc) (or (B,Hc,Wc) if input was 4D)
      - optionally ratios: (B,T) (or (B,) if input was 4D)
    mask_pixel is 1 where kept, 0 where masked.
    """
    assert images.ndim in (4, 5)
    orig_4d = (images.ndim == 4)

    if orig_4d:
        B, C, H, W = images.shape
        T = 1
        images_5d = images[:, None]  # (B,1,C,H,W)
    else:
        B, T, C, H, W = images.shape
        images_5d = images

    # crop to patch grid
    Hc = (H // patch_size) * patch_size
    Wc = (W // patch_size) * patch_size
    x = images_5d[..., :Hc, :Wc]  # (B,T,C,Hc,Wc)

    gh, gw = Hc // patch_size, Wc // patch_size
    P = gh * gw

    device = images.device

    # --- choose mask ratio per frame ---
    if mask_ratio is None:
        ratios = torch.rand(B, T, device=device) * max_mask_ratio  # (B,T) in [0, max_mask_ratio]
    else:
        # constant ratio everywhere
        ratios = torch.full((B, T), float(mask_ratio), device=device).clamp(0.0, max_mask_ratio)

    # keep count per frame (at least 1 patch)
    keep_counts = torch.floor(P * (1.0 - ratios)).to(torch.long).clamp(min=1, max=P)  # (B,T)

    # --- sample per-frame random patch ordering ---
    noise = torch.rand(B, T, P, device=device)               # (B,T,P)
    ids = noise.argsort(dim=-1)                              # permute patches low->high
    ranks = ids.argsort(dim=-1)                              # inverse perm: rank of each patch in sorted order
    keep_mask_flat = (ranks < keep_counts[..., None])        # (B,T,P) boolean

    mask_flat = keep_mask_flat.to(dtype=x.dtype)             # (B,T,P) 1 keep / 0 drop
    mask_2d = mask_flat.view(B, T, gh, gw)                   # (B,T,gh,gw)
    mask_pixel = mask_2d.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)  # (B,T,Hc,Wc)

    masked = x * mask_pixel.unsqueeze(2)  # (B,T,C,Hc,Wc)

    out = images_5d.clone()
    out[..., :Hc, :Wc] = masked

    # return in same ndim as input
    if orig_4d:
        out = out[:, 0]            # (B,C,H,W)
        mask_pixel_out = mask_pixel[:, 0]  # (B,Hc,Wc)
        ratios_out = ratios[:, 0]  # (B,)
    else:
        mask_pixel_out = mask_pixel        # (B,T,Hc,Wc)
        ratios_out = ratios                # (B,T)

    if return_ratios:
        return out, mask_pixel_out, ratios_out
    return out, mask_pixel_out
class LPIPSLoss(nn.Module):
    def __init__(self, net: str = "vgg", reduction: str = "none", device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

        import lpips  
        self.lpips = lpips.LPIPS(net=net).to(device)
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        detach_target: bool = True,
    ) -> torch.Tensor:
        if detach_target:
            target = target.detach()

        if pred.ndim == 5:
            B, T, C, H, W = pred.shape
            pred = pred.reshape(B * T, C, H, W)
            target = target.reshape(B * T, C, H, W)

        d = self.lpips(pred, target)

        if self.reduction == "none":
            return d
        if self.reduction == "sum":
            return d.sum()
        return d.mean()

def adaptive_grad_clip(model, clip=0.01, eps=1e-3):
    total_grad_norm = 0.0

    with torch.no_grad():
        for param in model.parameters():
            if param.grad is None:
                continue

            param_norm = param.data.norm(2)
            grad_norm = param.grad.data.norm(2)
            max_norm = max(param_norm, eps)
            ratio = grad_norm / max_norm

            if ratio > clip:
                param.grad.data.mul_(clip / ratio)

            total_grad_norm += grad_norm.item() ** 2

    return total_grad_norm ** 0.5

def check_shape(t):
    if len(t.shape) == 3:
        t = t.squeeze(-1)
    return t
def lambda_returns(reward, cont, value, lambda_=0.95, discount=0.997):
    reward = check_shape(reward).squeeze(-1)   # [B,T]
    cont   = check_shape(cont).squeeze(-1)     # [B,T]
    value  = check_shape(value).squeeze(-1)    # [B,T+1]

    B, T = reward.shape
    assert value.shape[1] == T + 1, (reward.shape, value.shape)

    returns = torch.zeros(B, T, device=reward.device, dtype=reward.dtype)
    next_ret = value[:, -1]  # bootstrap V_{T}

    for t in reversed(range(T)):
        disc = discount * cont[:, t]
        next_val = value[:, t + 1]
        target = reward[:, t] + disc * ((1 - lambda_) * next_val + lambda_ * next_ret)
        returns[:, t] = target
        next_ret = target

    return returns
def get_parameters(modules: Iterable[nn.Module]):
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

class FreezeParameters:
    def __init__(self, modules: Iterable[nn.Module]):
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


class SwiGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.silu(b)
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.size(-1)
    if d % 2 != 0:
        raise ValueError(f"RoPE needs even last dim, got {d}.")
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


# --- PASTE INTO utils.py ---

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: [..., D] where D is even
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # interleave [-x2, x1]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(-2)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    q,k: [B, H, T, Dh]
    cos,sin: broadcastable to [B, H, T, Dh] (typically [1,1,T,Dh])
    """
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

class RoPE1D(nn.Module):
    """
    Real RoPE using cos/sin caches (no complex numbers, no torch.polar).
    Cache is stored in fp32 and cast to q/k dtype at use-time.
    """
    def __init__(self, head_dim: int, base: float = 10000.0, max_seq_len: int = 4096):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")

        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._cached_len = 0

    @torch.no_grad()
    def _build_cache(self, seq_len: int, device):
        # Build in fp32 for numerical stability
        t = torch.arange(seq_len, device=device, dtype=torch.float32)                      # [T]
        freqs = torch.einsum("t,d->td", t, self.inv_freq.to(device=device))                # [T, Dh/2]
        cos = freqs.cos()
        sin = freqs.sin()

        # Expand to full Dh by interleaving (cos0, cos0, cos1, cos1, ...)
        cos = torch.repeat_interleave(cos, repeats=2, dim=-1)  # [T, Dh]
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1)  # [T, Dh]

        # Cache as [1,1,T,Dh] for broadcasting over [B,H,T,Dh]
        self.cos_cached = cos[None, None, :, :]  # fp32
        self.sin_cached = sin[None, None, :, :]
        self._cached_len = seq_len

    def forward(self, x: torch.Tensor, seq_len: int | None = None):
        """
        x: [B, H, T, Dh] (or anything with last dim Dh)
        Returns cos,sin shaped [1,1,T,Dh] cast to x.dtype.
        """
        device = x.device
        T = seq_len if seq_len is not None else x.shape[-2]

        if (self._cached_len < T) or (self.cos_cached.device != device):
            # Grow cache (use max_seq_len if you like; here we grow to T)
            self._build_cache(T, device=device)

        cos = self.cos_cached[..., :T, :].to(dtype=x.dtype)
        sin = self.sin_cached[..., :T, :].to(dtype=x.dtype)
        return cos, sin
def decoder_modality_mask(L: int, modality_sizes: list[int], device=None) -> torch.Tensor:
    """
    Layout: [z | m1 | m2 | ...]
    Decoder rule:
      - z queries attend only to z keys
      - modality_i queries attend to z keys + modality_i keys
    Returns boolean mask [S,S] where True blocks attention.
    """
    S = L + sum(modality_sizes)
    allow = torch.zeros((S, S), dtype=torch.bool, device=device)

    # z attends only to z
    allow[:L, :L] = True

    # each modality attends to (z + itself)
    start = L
    for n in modality_sizes:
        allow[start:start+n, :L] = True               # to z
        allow[start:start+n, start:start+n] = True    # within itself
        start += n

    return ~allow

def causal_mask(T: int, device=None) -> torch.Tensor:
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

class ActionBinning:
    def __init__(self, bins=50, low=-1.0, high=1.0, device="cuda"):
        self.bins = bins
        self.low = low
        self.high = high
        self.bin_step = (high - low) / (bins - 1)
        self.centers = torch.linspace(low, high, bins, device=device)

    def to_logits(self, actions):
        x = actions.clamp(self.low, self.high)
        pos = (x - self.low) / self.bin_step
        
        floor_idx = pos.floor().long().clamp(0, self.bins - 1)
        ceil_idx = (floor_idx + 1).clamp(0, self.bins - 1)
        
        ceil_weight = pos - floor_idx.float()
        floor_weight = 1.0 - ceil_weight
        
        shape = actions.shape
        target = torch.zeros(*shape, self.bins, device=actions.device)
        
        target.scatter_(-1, floor_idx.unsqueeze(-1), floor_weight.unsqueeze(-1))
        target.scatter_(-1, ceil_idx.unsqueeze(-1), ceil_weight.unsqueeze(-1))
        
        return target

    def from_logits(self, logits, deterministic=False):
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            indices = torch.argmax(probs, dim=-1)
            return self.centers[indices]
        else:
            return (probs * self.centers).sum(dim=-1)
            
    def sample(self, logits):
        dist = td.OneHotCategorical(logits=logits)
        sample_one_hot = dist.sample() 
        return (sample_one_hot * self.centers).sum(dim=-1)
    
def encoder_modality_mask(
    L: int,                   # number of latent (z) tokens
    modality_sizes: list[int], # [n1, n2, ...]
    device=None
) -> torch.Tensor:
    """
    Returns a boolean mask [S, S] where True = BLOCK attention, False = ALLOW.
    Layout: [z | m1 | m2 | ...]
    Encoder rule:
      - z queries attend to all keys
      - modality queries attend only within same modality
    """
    S = L + sum(modality_sizes)
    allow = torch.zeros((S, S), dtype=torch.bool, device=device)

    # z attends to everything
    allow[:L, :] = True

    # each modality attends only to itself
    start = L
    for n in modality_sizes:
        allow[start:start+n, start:start+n] = True
        start += n

    # Convert allow->mask-out
    attn_mask = ~allow  # True means "block"
    return attn_mask