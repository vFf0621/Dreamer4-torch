import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.distributions.transforms import TanhTransform
from collections.abc import Iterable
import torch.nn.init as init
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
    dreg_bins = torch.linspace(minv, maxv, bin_num, device=x.device, dtype=x.dtype)
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
    return symexp(x)

def concat_mtp(x, mtp):
    if len(x.shape) < 3:
        x = x[None]
    last= x[:,-mtp]
    prev = x[:,:-mtp, 0]   
    seq = torch.cat([prev, last], 1)   
    return seq

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Apply Xavier Uniform to Linear Layers
        init.orthogonal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
            
            
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
def int_to_one_hot(x, num_classes):
	"""
	Converts an integer tensor to a one-hot tensor.
	Supports batched inputs.
	"""
	one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
	one_hot.scatter_(-1, x.unsqueeze(-1), 1)
	return one_hot
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
def mask_after_done(inp, contprob, thresh=0.5):
    """
    inp:   [B, T] or [B, T, 1]
    contprob: [B, T] or [B, T, 1]   (continuation probability in [0,1])

    Masks reward so that once contprob < thresh occurs,
    reward at that timestep and all future timesteps become 0.
    """
    inp = inp.squeeze(-1)
    contprob = contprob.squeeze(-1)

    B, T = inp.shape

    # done mask: True where episode is considered ended
    done = contprob  >= thresh   # [B, T]

    # cumulative "has ended already" mask
    ended = torch.cumsum(done.int(), dim=1) > 0   # [B, T], True after first done
    ended = torch.cat([torch.zeros_like(ended[:,:1]), ended], 1)[:,:-1]

    # mask rewards
    out = inp.masked_fill(ended, 0.0)

    return out


def kl_div(p, q):
    logprob = nn.functional.log_softmax(p, -1)
    logother = nn.functional.log_softmax(q, -1)
    prob = torch.softmax(p, -1)
    return (prob * (logprob - logother)).sum(-1)
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

def lambda_returns(reward, contprob, boot,lambda_=0.95, discount=0.997):
    # reward, contprob, value: [B,T] (or [B,T,1] -> squeeze)
    reward   = reward.squeeze(-1)
    contprob = contprob.squeeze(-1)      # continuation probability in [0,1]
    boot    = boot.squeeze(-1)
    B, T = reward.shape

    # Effective discount per step: gamma_t = discount * contprob_t
    
    # Shift value to get V_{t+1}. For last step, bootstrap with V_{T-1} (or provide an explicit boot value)
    rets = [boot[:, -1]]
    live = (contprob[:, 1:] > 0.5).float() * discount
    cont = lambda_
    interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont * rets[-1])
    return torch.stack(list(reversed(rets))[:-1], 1)
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


def causal_mask(T: int, device=None) -> torch.Tensor:
    return ~torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

def unimix_probs(probs, eps=0.01, dim=-1):
    # probs: [..., num_bins]
    num_bins = probs.size(dim)
    uniform = torch.full_like(probs, 1.0 / num_bins)
    return (1.0 - eps) * probs + eps * uniform
def gumbel_softmax(
    prob: torch.Tensor,
    tau: float = 1.0,
    hard: bool = False,
    dim: int = -1,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Gumbel-Softmax sampling from probabilities.

    Args:
      prob: [..., num_bins] (Probabilities, must sum to 1)
      tau: temperature (lower -> more one-hot, higher -> uniform)
      hard: if True, returns one-hot but allows gradient flow (straight-through)
      dim: bins dimension
      eps: numerical stability

    Returns:
      y: same shape as prob
    """
    # 1. Convert Probabilities to Logits
    #    We clamp to avoid log(0) = -inf
    logits = torch.log(prob.clamp_min(eps))

    # 2. Sample Gumbel Noise
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u.clamp_min(eps)).clamp_min(eps))

    # 3. Softmax with Temperature
    #    (log(p) + g) / tau
    y = F.softmax((logits + g) / tau, dim=dim)

    if not hard:
        return y

    # 4. Straight-through: Hard One-Hot forward, Soft backward
    #    index of max value
    index = y.argmax(dim, keepdim=True)
    
    #    create hard one-hot tensor
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    
    #    detach hard so gradients flow through 'y' (the soft version)
    return y_hard - y.detach() + y
def modality_mask(
    L: int,                   
    modality_sizes: list[int], 
    device=None,
    encoder=True  # <--- You must pass False for the Decoder!
) -> torch.Tensor:
    
    Np = sum(modality_sizes)
    S  = L + Np
    allow = torch.zeros((S, S), dtype=torch.bool, device=device)
    
    if not encoder:
        # --- ENCODER LOGIC ---
        # 1. Latents read from everything (to encode)
        allow[:L, :] = True 
        
        # 2. Patches read from Patches (spatial context)
        #    BUT Patches DO NOT read from Latents (prevent leakage)
        allow[L:, L:] = True

    else:
        # --- DECODER LOGIC ---
        # 1. Latents read ONLY from Latents (keep source pure)
        allow[:L, :L] = True
        
        # 2. Patches read from Patches (to form image) 
        #    AND Patches read from Latents (CRITICAL: this is the gradient path)
        allow[L:, :] = True  # <--- This fixes the zero grad

    return ~allow
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
def save_attention_mask(mask, filename, title="Attention Mask"):
    """
    Saves a 2D attention mask as an image file.
    
    Args:
        mask (Tensor): 2D Tensor of shape (H, W)
        filename (str): Path to save (e.g., 'plots/mask_epoch_1.png')
    """
    # 1. Prepare Data: Detach, move to CPU, convert to numpy
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
        
    # 2. Create Figure
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='viridis', interpolation='nearest')
    
    # 3. Styling
    plt.colorbar(label='Weight')
    plt.title(title)
    plt.xlabel("Key")
    plt.ylabel("Query")
    
    # 4. Save and Close (Crucial for memory management in loops)
    # bbox_inches='tight' prevents labels from being cut off
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
