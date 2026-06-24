"""Gating tests for Phase 79 AOT-readout (delta/aot_readout.py).

The load-bearing one is the K=1 IDENTITY: at 1p neither AOT-readout nor JIT-pec traverses, so with a
shared readout head they MUST produce identical scores. The pre-registration makes this the go/no-go
plumbing gate — if it fails, the 3p comparison is void (a bug, not a finding).
"""
import torch
import pytest

from delta.path_compose import PathComposerPF, build_csr, build_R2E
from delta.aot_readout import AOTReadout, copy_readout_head


def _toy(seed=0, N=12, d_node=48, d_edge=24):
    g = torch.Generator().manual_seed(seed)
    nf = torch.randn(N, d_node, generator=g)
    R, E = 5, 30
    src = torch.randint(0, N, (E,), generator=g)
    tgt = torch.randint(0, N, (E,), generator=g)
    et = torch.randint(0, R, (E,), generator=g)
    ei = torch.stack([src, tgt])
    ef0 = torch.randn(E, d_edge, generator=g)
    return nf, ef0, ei, et, R


def _find_2hop(nf, ei, et, R, csr):
    """An (anchor, [r1,r2]) whose AOT frontier is nonempty."""
    aot = AOTReadout(R)
    for a in range(nf.shape[0]):
        for r1 in range(R):
            for r2 in range(R):
                anchors = torch.tensor([a]); rel_mat = torch.tensor([[r1, r2]])
                fnode, _ = aot._frontier_batch(anchors, rel_mat, 2, nf, csr)
                if fnode.numel() > 0:
                    return a, [r1, r2]
    return None


# ── (i) THE GATE: K=1 AOT-readout == JIT-pec, given a shared readout head ──
def test_1p_identity_with_pec():
    nf, ef0, ei, et, R = _toy()
    csr = build_csr(ei, et, R, nf.device)
    pec = PathComposerPF(R, mode='pec')
    aot = AOTReadout(R)
    copy_readout_head(pec, aot)                       # share the seed+readout submodule weights
    qs = [(a, [r]) for a in range(nf.shape[0]) for r in range(R)]   # all 1p queries
    s_pec = pec.score_batch_vec(qs, nf, ef0, ei, csr)
    s_aot = aot.score_batch_vec(qs, nf, ef0, ei, csr)
    assert torch.allclose(s_pec, s_aot, atol=1e-5), \
        f"1p AOT-readout must be bit-identical to JIT-pec (max diff {(s_pec - s_aot).abs().max():.2e})"


# ── (ii) edges never participate: AOT-readout is invariant to ef0 ──
def test_aot_readout_ignores_ef0():
    nf, ef0, ei, et, R = _toy(seed=3)
    csr = build_csr(ei, et, R, nf.device)
    aot = AOTReadout(R)
    q2 = _find_2hop(nf, ei, et, R, csr)
    qs = [(a, [0]) for a in range(nf.shape[0])] + ([q2] if q2 else [])
    s1 = aot.score_batch_vec(qs, nf, ef0, ei, csr)
    s2 = aot.score_batch_vec(qs, nf, ef0 + 5.0 * torch.randn_like(ef0), ei, csr)
    assert torch.equal(s1, s2), "AOT-readout must not depend on ef0 — edges never participate"


# ── (iii) frontier reachability matches JIT-pec's traversal frontier (same support) ──
def test_frontier_matches_pec_reachability():
    nf, ef0, ei, et, R = _toy(seed=7, N=14)
    csr = build_csr(ei, et, R, nf.device)
    pec = PathComposerPF(R, mode='pec'); aot = AOTReadout(R)
    q = _find_2hop(nf, ei, et, R, csr)
    assert q is not None, "toy graph produced no 2-hop path; bump seed"
    a, rc = q
    anchors = torch.tensor([a]); rel_mat = torch.tensor([rc])
    ej = pec.build_ej(nf, ef0, ei)
    fnode_p, _, _ = pec._traverse_batch(anchors, rel_mat, 2, nf, ej, csr)
    fnode_a, _ = aot._frontier_batch(anchors, rel_mat, 2, nf, csr)
    assert set(fnode_p.tolist()) == set(fnode_a.tolist()), \
        "AOT-readout must pool the SAME train-reachable penult frontier JIT traverses to"


# ── (iv) at 2-hop, node-only z genuinely differs from edge-composed z (the contrast is real) ──
def test_aot_differs_from_pec_at_2hop():
    nf, ef0, ei, et, R = _toy(seed=11, N=14)
    csr = build_csr(ei, et, R, nf.device)
    pec = PathComposerPF(R, mode='pec'); aot = AOTReadout(R)
    copy_readout_head(pec, aot)                       # identical head -> any diff is purely the z-state
    q = _find_2hop(nf, ei, et, R, csr)
    assert q is not None
    s_pec = pec.score_batch_vec([q], nf, ef0, ei, csr)
    s_aot = aot.score_batch_vec([q], nf, ef0, ei, csr)
    assert not torch.allclose(s_pec, s_aot, atol=1e-5), \
        "at 2-hop, edge-composed z (pec) must differ from node-only z (aot) — else no edge axis to test"


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
