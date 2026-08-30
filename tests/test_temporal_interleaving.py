"""
Tests for the temporally interleaved dynamics stream a_1, z_1, (t,d)_1, a_2, ...

Invariants:
  1. Within a timestep block, attention among a, z, (t,d) is fully
     bidirectional (token order in the stream does not restrict it).
  2. Across timesteps, attention is causal at block granularity.
  3. Only agent queries may read agent keys; z predictions never depend on
     the agent token.

Run with `pytest tests/` or `python tests/test_temporal_interleaving.py`.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CausalSTBlock, Dynamics


def make_dynamics():
    torch.manual_seed(0)
    return Dynamics(
        Dz=8, action_dim=2, d_model=64, n_heads=4, depth=4, time_every=2,
        Sa=4, latent_tokens=8, Nr=2, num_tasks=2, dropout=0.0, device="cpu",
    ).eval()


def test_block_causal_mask_bidirectional_within_block():
    blk = CausalSTBlock(32, 4, time_attn=True, temporal_interleave=True, device="cpu")
    T, N, agent_idx = 3, 7, 5  # per block: a(2) z(2) sig(1) agent(1) reserved(1)
    bias = blk._block_causal_bias(T, N, agent_idx, "cpu", torch.float32)
    allow = torch.isfinite(bias)

    non_agent = [i for i in range(N) if i != agent_idx]
    for t in range(T):
        base = t * N
        # bidirectional among a, z, (t,d), reserved within the block
        for i in non_agent:
            for j in non_agent:
                assert allow[base + i, base + j], (t, i, j)
        # agent reads everyone; nobody else reads agent
        for i in range(N):
            assert allow[base + agent_idx, base + i]
            if i != agent_idx:
                assert not allow[base + i, base + agent_idx]

    # causal at block granularity: past fully visible (minus agent), future blocked
    for tq in range(T):
        for tk in range(T):
            sub = allow[tq * N:(tq + 1) * N, tk * N:(tk + 1) * N]
            if tk > tq:
                assert not sub.any(), (tq, tk)
            else:
                assert sub[non_agent][:, non_agent].all(), (tq, tk)
                assert not sub[non_agent][:, agent_idx].any(), (tq, tk)


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


def test_agent_isolation():
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
    assert torch.autograd.grad(feat[:, -1].sum(), z)[0].abs().sum() > 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"{fn.__name__} ok")
    print(f"{len(tests)} tests passed")
