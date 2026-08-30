"""
Tests for the temporally interleaved dynamics stream a_1, z_1, (t,d)_1, a_2, ...

Invariants:
  1. Time layers are axial: per-channel independent and causal; space layers
     mix a, z, (t,d) bidirectionally within each timestep, so information
     still flows both ways inside a block and causally across blocks.
  2. The agent token is a pure readout: block-causal over the processed
     stream, never written back into it, with gradient reaching the entire
     transformer.

Run with `pytest tests/` or `python tests/test_temporal_interleaving.py`.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GQA, CausalSTBlock, Dynamics


def make_dynamics():
    torch.manual_seed(0)
    return Dynamics(
        Dz=8, action_dim=2, d_model=64, n_heads=4, depth=4, time_every=2,
        Sa=4, latent_tokens=8, Nr=2, num_tasks=2, dropout=0.0, device="cpu",
    ).eval()


def test_time_attention_is_per_channel_independent():
    """A time layer must attend causally within each channel and never mix
    channels; cross-channel mixing belongs to the space layers."""
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
    q1, q2 = torch.randn(2, 3, 32), torch.randn(2, 3, 32)

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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"{fn.__name__} ok")
    print(f"{len(tests)} tests passed")
