"""Gating unit tests for Phase 77 PEC-pf (delta/path_compose.py).

These encode the red-team's must-fix correctness checks. They run on a tiny synthetic graph and must
all pass before any training/sweep. Failures here mean confidently-wrong numbers later.
"""
import math
import torch
import torch.nn as nn
import pytest

from delta.graph import DeltaGraph
from delta.attention import EdgeAttention
from delta.path_compose import PathComposerPF, build_R2E, build_csr, build_hr2t, MODES


def _toy(seed=0, N=10, d_node=48, d_edge=24):
    g = torch.Generator().manual_seed(seed)
    nf = torch.randn(N, d_node, generator=g)
    # a small typed multigraph; relations in [0, R)
    R = 5
    E = 24
    src = torch.randint(0, N, (E,), generator=g)
    tgt = torch.randint(0, N, (E,), generator=g)
    et = torch.randint(0, R, (E,), generator=g)
    ei = torch.stack([src, tgt])
    ef0 = torch.randn(E, d_edge, generator=g)
    return nf, ef0, ei, et, R


# ── (i) operator faithfulness: compose_scores reproduces EdgeAttention's score block exactly ──
def test_compose_scores_bit_identical_to_edge_attention():
    torch.manual_seed(1)
    N, E, d_node, d_edge, H = 12, 20, 16, 8, 4
    nf = torch.randn(N, d_node); ef = torch.randn(E, d_edge)
    ei = torch.randint(0, N, (2, E))
    g = DeltaGraph(node_features=nf, edge_features=ef, edge_index=ei)
    att = EdgeAttention(d_edge, d_node, num_heads=H); att.eval()
    adj = g.build_edge_adjacency()
    src, tgt = adj
    d_h = d_edge // H
    # reference: inline original math (pre-route_prob, pre-softmax)
    Q = att.W_q(ef).view(E, H, d_h); K = att.W_k(ef).view(E, H, d_h)
    ref = (Q[tgt] * K[src]).sum(-1) / math.sqrt(d_h)
    ctx = torch.cat([nf[ei[0, src]], nf[ei[1, src]]], -1)
    ref = (ref + att.W_ctx(ctx)) * att._log_temp.exp()
    got = EdgeAttention.compose_scores(att.W_q, att.W_k, att.W_ctx, att._log_temp,
                                       ef[tgt], ef[src], nf, ei[0, src], ei[1, src], H, d_h)
    assert torch.allclose(got, ref, atol=1e-6)


# ── (ii) 1p readout is a real all-N distribution (NOT ~1/N collapse) ──
def test_1p_readout_is_all_N_and_discriminative():
    nf, ef0, ei, et, R = _toy()
    N = nf.shape[0]
    m = PathComposerPF(R, mode='pec')
    R2E = build_R2E(ei, et, R, nf.device)
    s = m.forward_query(2, [1], nf, m.build_ej(nf, ef0, ei), R2E)
    assert s.shape == (N,)
    assert torch.isfinite(s).all()
    assert s.std() > 1e-4, "1p readout must vary across entities (uses nf[t]), not collapse to ~1/N"


# ── (iii) traversal only ever touches TRAIN edges (gold test edge never traversed) ──
def test_traverse_only_uses_train_edges():
    nf, ef0, ei, et, R = _toy()
    R2E = build_R2E(ei, et, R, nf.device)
    # a "test" edge with a relation id outside the train R2E -> cannot be traversed
    fake_rel = R  # not present in R2E
    assert fake_rel not in R2E
    m = PathComposerPF(R + 1, mode='pec')
    frontier, z = m.traverse(0, [fake_rel, 0], nf, m.build_ej(nf, ef0, ei), R2E)
    assert frontier.shape[0] == 0, "a relation absent from the train R2E must yield an empty frontier"


def _find_2hop(nf, ei, R2E, R, seedm=None):
    m = seedm or PathComposerPF(R, mode='pec')
    ej0 = torch.zeros(ei.shape[1], m.d_edge)
    for a in range(nf.shape[0]):
        for r1 in R2E:
            for r2 in R2E:
                if m.traverse(a, [r1, r2], nf, ej0, R2E)[0].shape[0] > 0:
                    return a, [r1, r2]
    return None


# ── (iv) rrprior is invariant to EDGE-instance features (ef0); pec is NOT ──
def test_rrprior_invariant_to_edge_instance_pec_sensitive():
    nf, ef0, ei, et, R = _toy(seed=3)
    R2E = build_R2E(ei, et, R, nf.device)
    q = _find_2hop(nf, ei, R2E, R)
    assert q is not None, "toy graph produced no 2-hop path; bump seed"
    a, rc = q
    ef2 = ef0 + 0.7 * torch.randn_like(ef0)   # perturb ONLY edge features (keep nf fixed)
    for mode, should_change in [('rrprior', False), ('pec', True)]:
        m = PathComposerPF(R, mode=mode)
        f1, z1 = m.traverse(a, rc, nf, m.build_ej(nf, ef0, ei), R2E)
        f2, z2 = m.traverse(a, rc, nf, m.build_ej(nf, ef2, ei), R2E)
        assert torch.equal(f1, f2), f"{mode}: frontier must be structural"
        same = torch.allclose(z1, z2, atol=1e-6)
        if should_change:
            assert not same, "pec penult-state MUST depend on edge-instance features (ej)"
        else:
            assert same, "rrprior must NOT depend on edge-instance features (relation-type messages only)"


# ── (iv-b) pec genuinely composes the path into the message: pec != static (shared weights) ──
def test_pec_differs_from_static_message_composition():
    nf, ef0, ei, et, R = _toy(seed=11)
    R2E = build_R2E(ei, et, R, nf.device)
    q = _find_2hop(nf, ei, R2E, R)
    assert q is not None
    a, rc = q
    pec = PathComposerPF(R, mode='pec')
    static = PathComposerPF(R, mode='static')
    static.load_state_dict(pec.state_dict())   # identical weights -> only the mechanism differs
    s_pec = pec.forward_query(a, rc, nf, pec.build_ej(nf, ef0, ei), R2E)
    s_static = static.forward_query(a, rc, nf, static.build_ej(nf, ef0, ei), R2E)
    assert not torch.allclose(s_pec, s_static, atol=1e-5), \
        "pec (z_src⊙W_v(ej)) must differ from static (W_v(ej) only) — composition must matter"


# ── (v) anchor-sensitivity at EVERY hop (catches dropped per-hop conditioning) ──
def test_anchor_sensitivity_propagates_through_hops():
    nf, ef0, ei, et, R = _toy(seed=5)
    R2E = build_R2E(ei, et, R, nf.device)
    m = PathComposerPF(R, mode='pec')
    ej = m.build_ej(nf, ef0, ei)
    # find two distinct anchors that both reach a nonempty penult frontier for the SAME 2-hop chain
    chain = None; anchors = []
    for r1 in R2E:
        for r2 in R2E:
            good = [a for a in range(nf.shape[0]) if m.traverse(a, [r1, r2], nf, ej, R2E)[0].shape[0] > 0]
            if len(good) >= 2:
                chain = [r1, r2]; anchors = good[:2]; break
        if chain: break
    assert chain is not None, "need two anchors sharing a 2-hop chain; bump seed"
    s0 = m.forward_query(anchors[0], chain, nf, ej, R2E)
    s1 = m.forward_query(anchors[1], chain, nf, ej, R2E)
    assert not torch.allclose(s0, s1, atol=1e-5), "different anchors must yield different scores (hop conditioning)"


# ── (vii) all four arms share the SAME frontier sets + all-N denominator; differ only in scores ──
def test_all_arms_identical_frontier_and_denominator():
    nf, ef0, ei, et, R = _toy(seed=7)
    R2E = build_R2E(ei, et, R, nf.device)
    # pick a 2-hop query with nonempty frontier
    probe = PathComposerPF(R, mode='pec'); ej = probe.build_ej(nf, ef0, ei)
    q = None
    for a in range(nf.shape[0]):
        for r1 in R2E:
            for r2 in R2E:
                if probe.traverse(a, [r1, r2], nf, ej, R2E)[0].shape[0] > 0:
                    q = (a, [r1, r2]); break
            if q: break
        if q: break
    assert q is not None
    a, rc = q
    frontiers = {}; scores = {}
    for mode in MODES:
        m = PathComposerPF(R, mode=mode)
        fr, _ = m.traverse(a, rc, nf, m.build_ej(nf, ef0, ei), R2E)
        frontiers[mode] = set(fr.tolist())
        s = m.forward_query(a, rc, nf, m.build_ej(nf, ef0, ei), R2E)
        assert s.shape == (nf.shape[0],) and torch.isfinite(s).all()  # same all-N denominator
        scores[mode] = s
    base = frontiers['pec']
    for mode in MODES:
        assert frontiers[mode] == base, f"{mode} frontier differs — arms must isolate only the mechanism"


# ── (vi) train-only hr2t differs from full and does not leak val/test answers ──
def test_train_only_hr2t_no_leak():
    train = torch.tensor([[0, 1, 2], [0, 0, 0], [1, 2, 3]])  # (h,r,t): (0,0,1),(1,0,2),(2,0,3)
    val = torch.tensor([[3], [0], [4]])                       # (3,0,4) val-only
    test = torch.tensor([[2], [0], [5]])                      # (2,0,5) test-only
    train_hr2t = build_hr2t(train)
    full = build_hr2t(torch.cat([train, val, test], dim=1))
    assert (2, 0) in train_hr2t and 3 in train_hr2t[(2, 0)]
    assert 5 not in train_hr2t.get((2, 0), set()), "test-only tail must NOT be in train_hr2t"
    assert 5 in full[(2, 0)], "full adjacency should contain the test tail"
    assert (3, 0) not in train_hr2t and 4 in full[(3, 0)], "val-only edge leaks into train_hr2t"


# ── batched scorer must equal the per-query reference (all arms, mixed K) ──
def test_batched_equals_per_query():
    nf, ef0, ei, et, R = _toy(seed=13, N=14)
    R2E = build_R2E(ei, et, R, nf.device)
    csr = build_csr(ei, et, R, nf.device)
    # assemble a mixed-K query batch that includes nonempty 2p/3p paths
    probe = PathComposerPF(R, mode='pec'); ej0 = torch.zeros(ei.shape[1], probe.d_edge)
    qs = []
    for a in range(nf.shape[0]):
        qs.append((a, [list(R2E)[0]]))                      # 1p
    twohop = _find_2hop(nf, ei, R2E, R)
    if twohop:
        qs.append(twohop)
    for mode in MODES:
        m = PathComposerPF(R, mode=mode)
        ej = m.build_ej(nf, ef0, ei)
        ref = torch.stack([m.forward_query(int(a), list(rc), nf, ej, R2E) for (a, rc) in qs], 0)
        vec = m.score_batch_vec(qs, nf, ef0, ei, csr)
        assert torch.allclose(ref, vec, atol=1e-5), f"{mode}: batched != per-query (max {(ref-vec).abs().max():.2e})"


# ── capacity arm: operator weights frozen, the rest trainable ──
def test_capacity_freezes_operator_only():
    m = PathComposerPF(5, mode='capacity')
    for mod in (m.W_q, m.W_k, m.W_v, m.W_out, m.W_ctx):
        assert all(not p.requires_grad for p in mod.parameters())
    assert not m._log_temp.requires_grad
    for mod in (m.W_ein, m.W_seed, m.W_read, m.rel_emb, m.W_poolattn):
        assert any(p.requires_grad for p in mod.parameters())


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
