"""
Tests for the temporally interleaved dynamics stream a_1, z_1, (t,d)_1, a_2, ...

Invariants:
  1. Dynamics time layers flatten the blocks into one stream and apply the
     standard causal mask over it, so ordering is strict at token
     granularity; the tokenizer keeps per-channel independent time attention.
     Space layers stay bidirectional within a timestep, which is what lets z
     read its own (t,d) signal token despite the strict temporal order.
  2. The agent token is a pure readout: block-causal over the processed
     stream, never written back into it, with gradient reaching the entire
     transformer.

Run with `pytest tests/` or `python tests/test_temporal_interleaving.py`.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GQA, CausalSTBlock, Decoder, Dynamics, Encoder
from utils import modality_mask


def make_dynamics():
    torch.manual_seed(0)
    return Dynamics(
        Dz=8, action_dim=2, d_model=64, n_heads=4, depth=4, time_every=2,
        Sa=4, latent_tokens=8, Nr=2, num_tasks=2, dropout=0.0, device="cpu",
    ).eval()


def test_interleaved_time_attention_is_standard_causal():
    """With temporal interleaving, a time layer flattens the blocks into one
    stream and applies the STANDARD causal mask: a token attends to strictly
    earlier stream positions plus itself, so ordering is strict even inside a
    block (z sees a, but not the (t,d) token that follows it)."""
    torch.manual_seed(0)
    blk = CausalSTBlock(32, 4, dropout=0.0, time_attn=True,
                        time_mode="stream_causal", device="cpu").eval()
    B, T, N = 1, 4, 5
    x = torch.randn(B, T, N, 32, requires_grad=True)
    out = blk(x)

    # query at block t=2, slot 3 -> flat position 2*N + 3
    qpos = 2 * N + 3
    g = torch.autograd.grad(out[:, 2, 3].sum(), x)[0]
    flat = g.abs().sum(-1).reshape(-1)  # [T*N]

    assert flat[:qpos + 1].sum() > 0, "token cannot see its own prefix"
    assert flat[qpos + 1:].sum() == 0, "standard causal mask leaks later tokens"
    # strictly inside the block: earlier slots visible, later slots not
    assert flat[2 * N:qpos + 1].sum() > 0, "cannot see earlier slots of its block"
    assert flat[qpos + 1:3 * N].sum() == 0, "sees later slots of its own block"


def test_per_channel_time_attention_still_available():
    """per_channel remains selectable (no caller uses it by default now)."""
    torch.manual_seed(0)
    blk = CausalSTBlock(32, 4, dropout=0.0, time_attn=True, device="cpu").eval()
    B, T, N = 1, 4, 5
    x = torch.randn(B, T, N, 32, requires_grad=True)
    out = blk(x)

    g = torch.autograd.grad(out[:, 2, 3].sum(), x)[0]
    per_channel = g.abs().sum(-1)[0]  # [T, N]
    assert per_channel[:3, 3].sum() > 0, "channel cannot see its own history"
    assert per_channel[3:, :].sum() == 0, "time layer leaks the future"
    other = [n for n in range(N) if n != 3]
    assert per_channel[:, other].sum() == 0, "time layer mixes channels"


def test_z_reads_signal_token_both_ways_in_block():
    """(t,d) comes AFTER z in the stream; z_t must still read its own block's
    signal token, while future blocks' signals stay invisible."""
    dyn = make_dynamics()
    B, T, Nz = 2, 5, 8
    z = torch.randn(B, T, Nz, 8)
    a = torch.rand(B, T - 1, 2) * 2 - 1
    # distinct tau index per timestep so embedding rows identify timesteps
    sigs = torch.zeros(B, T, Nz, 2, dtype=torch.long)
    sigs[..., 0] = torch.arange(T).view(1, T, 1)

    z_pred, _ = dyn(z, a, sigs, task_id=0)
    grad = torch.autograd.grad(z_pred[:, 1].sum(), dyn.level_emb.weight)[0]
    row_norms = grad.abs().sum(-1)

    assert row_norms[1] > 0, "z_1 cannot read its own (t,d) token"
    assert row_norms[0] > 0, "z_1 cannot read past (t,d) tokens"
    assert row_norms[2:T].sum() == 0, "z_1 reads future (t,d) tokens"


def test_block_causality_of_z_and_actions():
    dyn = make_dynamics()
    B, T, Nz = 2, 5, 8
    z = torch.randn(B, T, Nz, 8, requires_grad=True)
    a = (torch.rand(B, T - 1, 2) * 2 - 1).requires_grad_(True)
    sigs = torch.zeros(B, T, Nz, 2, dtype=torch.long)

    z_pred, _ = dyn(z, a, sigs, task_id=0)
    g_z, g_a = torch.autograd.grad(z_pred[:, 2].sum(), [z, a])

    assert g_z[:, :3].abs().sum() > 0
    assert g_z[:, 3:].abs().sum() == 0, "z leaks across time"
    # actions are front-padded: block t carries a_{t-1}, so z_2 sees a_0, a_1
    assert g_a[:, :2].abs().sum() > 0
    assert g_a[:, 2:].abs().sum() == 0, "action leaks across time"


def test_agent_is_pure_readout():
    """The agent token reads the stream but never writes into it: z_pred is
    structurally independent of it, while the readout query itself trains."""
    dyn = make_dynamics()
    B, T, Nz = 2, 4, 8
    z = torch.randn(B, T, Nz, 8, requires_grad=True)
    a = torch.rand(B, T - 1, 2) * 2 - 1
    sigs = torch.zeros(B, T, Nz, 2, dtype=torch.long)

    z_pred, feat = dyn(z, a, sigs, task_id=0)
    g_agent = torch.autograd.grad(
        z_pred.sum(), dyn.agent_token, retain_graph=True, allow_unused=True
    )[0]
    assert g_agent is None or g_agent.abs().sum() == 0, "z_pred reads agent token"
    assert torch.autograd.grad(feat[:, -1].sum(), z, retain_graph=True)[0].abs().sum() > 0
    assert torch.autograd.grad(
        feat.sum(), dyn.agent_token, retain_graph=True
    )[0].abs().sum() > 0, "readout query gets no gradient"


def test_readout_grad_flows_through_entire_transformer():
    """Gradients from the policy features must reach every backbone layer,
    not stop at the readout."""
    dyn = make_dynamics()
    B, T, Nz = 2, 4, 8
    z = torch.randn(B, T, Nz, 8)
    a = torch.rand(B, T - 1, 2) * 2 - 1
    sigs = torch.zeros(B, T, Nz, 2, dtype=torch.long)

    _, feat = dyn(z, a, sigs, task_id=0)
    feat.sum().backward()

    # each CausalSTBlock owns both norms but uses only one, depending on its type
    dead = set()
    for i, blk in enumerate(dyn.blocks):
        dead.add(f"blocks.{i}." + ("ln_space" if blk.time_attn_enabled else "ln_time"))

    for name, p in dyn.named_parameters():
        if name.startswith(("out.",)) or "agent_token" in name:
            continue  # output head is z-only; agent covered above
        if "action_pad" in name:
            continue  # key-padding-masked everywhere by design, never receives grad
        if any(name.startswith(d) for d in dead):
            continue  # unused norm of the other attention type
        assert p.grad is not None and p.grad.abs().sum() > 0, \
            f"no gradient reaches {name}"


def test_readout_is_block_causal():
    dyn = make_dynamics()
    B, T, Nz = 2, 5, 8
    z = torch.randn(B, T, Nz, 8, requires_grad=True)
    a = torch.rand(B, T - 1, 2) * 2 - 1
    sigs = torch.zeros(B, T, Nz, 2, dtype=torch.long)

    _, feat = dyn(z, a, sigs, task_id=0)
    g = torch.autograd.grad(feat[:, 1].sum(), z)[0]
    assert g[:, :2].abs().sum() > 0
    assert g[:, 2:].abs().sum() == 0, "readout sees future timesteps"


def test_attention_soft_capping():
    """As softcap -> 0, capped logits -> 0, so attention becomes uniform over
    the allowed keys and the output no longer depends on the query content."""
    torch.manual_seed(0)
    g = GQA(32, 4, dropout=0.0, causal=False, device="cpu").eval()
    kv = torch.randn(2, 6, 32)
    q1, q2 = torch.randn(2, 6, 32), torch.randn(2, 6, 32)

    g.softcap = 1e-6
    out_uniform_1 = g(q1, x_k=kv)
    out_uniform_2 = g(q2, x_k=kv)
    assert torch.allclose(out_uniform_1, out_uniform_2, atol=1e-4), \
        "tiny cap should erase query dependence"

    g.softcap = 50.0
    assert not torch.allclose(g(q1, x_k=kv), g(q2, x_k=kv), atol=1e-4), \
        "normal cap should keep content-dependent attention"

    # capping must not corrupt masking: fully valid rows stay finite
    out = g(q1, x_k=kv, key_padding_mask=torch.tensor([[False]*5 + [True]] * 2))
    assert torch.isfinite(out).all()


def test_tokenizer_time_attention_is_block_causal():
    """Tokenizer time layers flatten the frames and apply a BLOCK-causal mask:
    each frame is internally bidirectional, the future is blocked."""
    blk = CausalSTBlock(32, 4, dropout=0.0, time_attn=True,
                        time_mode="block_causal", device="cpu")
    T, N = 3, 4
    allow = torch.isfinite(blk._block_causal_bias(T, N, "cpu", torch.float32))
    for tq in range(T):
        for tk in range(T):
            sub = allow[tq * N:(tq + 1) * N, tk * N:(tk + 1) * N]
            if tk > tq:
                assert not sub.any(), f"future frame {tk} visible from {tq}"
            else:
                assert sub.all(), f"frame {tk} not fully visible from {tq}"


def test_block_causal_tiles_the_modality_restriction():
    """Flattening must not let a token bypass the encoder's one-directional
    modality mask by reading the same group in an earlier frame."""
    blk = CausalSTBlock(32, 4, dropout=0.0, time_attn=True,
                        time_mode="block_causal", device="cpu")
    T, N, L = 3, 4, 2                      # 2 latents + 2 patches per frame
    mod = torch.zeros((N, N), dtype=torch.bool)
    mod[:L, :] = True                      # latents read everything
    mod[L:, L:] = True                     # patches read only patches
    allow = torch.isfinite(
        blk._block_causal_bias(T, N, "cpu", torch.float32, modality=mod))

    patch_q = 1 * N + L                    # a patch token in frame 1
    for tk in (0, 1):
        assert not allow[patch_q, tk * N:tk * N + L].any(), \
            f"patch reads latents of frame {tk}"
        assert allow[patch_q, tk * N + L:(tk + 1) * N].all(), \
            f"patch cannot read patches of frame {tk}"
    assert allow[1 * N, :2 * N].all(), "latent query lost its past"


def test_encoder_and_decoder_are_block_causal_in_time():
    torch.manual_seed(0)
    enc = Encoder(h=32, w=32, patch=16, d_model=64, n_heads=4, depth=4,
                  latent_tokens=4, time_every=2, out_dim=8, num_reserved=2)
    dec = Decoder(h=32, w=32, patch=16, z_dim=8, d_model=64, n_heads=4, depth=4,
                  latent_tokens=4, time_every=2, num_reserved=2)
    assert {b.time_mode for b in enc.blocks} == {"block_causal"}
    assert {b.time_mode for b in dec.blocks} == {"block_causal"}

    frames = torch.rand(2, 4, 3, 32, 32, requires_grad=True)
    z = enc(frames)
    assert torch.isfinite(z).all()
    assert torch.isfinite(dec(z)).all()

    g = torch.autograd.grad(z[:, 1].sum(), frames)[0]
    per_t = g.abs().sum((0, 2, 3, 4))
    assert per_t[:2].sum() > 0, "encoder cannot see current/past frames"
    assert per_t[2:].sum() == 0, "encoder leaks future frames"


def test_modality_mask_is_bidirectional_except_latents():
    """Within a timestep every modality attends every other modality both
    ways; the only restriction is the latent tokens, which are
    one-directional (encoder: nothing may read them; decoder: the mirror)."""
    L, Np, Nr = 4, 6, 2
    enc = modality_mask(L=L, modality_sizes=[Np, Nr])
    assert enc[:L, :].all(), "latents must read everything"
    assert enc[L:, L:].all(), "non-latent modalities must be bidirectional"
    assert not enc[L:, :L].any(), "nothing but a latent may read a latent"

    dec = modality_mask(L=L, modality_sizes=[Np, Nr], encoder=False)
    assert dec[:L, :L].all() and not dec[:L, L:].any(), "decoder latents must stay pure"
    assert dec[L:, :].all(), "decoder non-latents must read everything"

    # the restriction must survive being tiled across the time staircase
    T, N = 3, L + Np + Nr
    blk = CausalSTBlock(32, 4, dropout=0.0, time_attn=True,
                        time_mode="block_causal", device="cpu")
    allow = torch.isfinite(
        blk._block_causal_bias(T, N, "cpu", torch.float32, modality=enc))
    for tq in range(T):
        for tk in range(tq + 1):           # within and past frames
            sub = allow[tq * N:(tq + 1) * N, tk * N:(tk + 1) * N]
            assert sub[:L, :].all(), (tq, tk, "latent row lost context")
            assert sub[L:, L:].all(), (tq, tk, "modalities not bidirectional")
            assert not sub[L:, :L].any(), (tq, tk, "latent leaked across frames")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"{fn.__name__} ok")
    print(f"{len(tests)} tests passed")
