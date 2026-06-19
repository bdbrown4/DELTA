"""Phase 74b — SUPERVISED UPPER-BOUND GATE for composability routing.

Decisive cheap gate (red-team's #1 recommendation) BEFORE building the full hardened
pipeline. Question: can asymmetric M-bucket anchor routing, optimized DIRECTLY for
composability (the best case it could ever reach), produce an edge adjacency whose
composable-fraction substantially exceeds structural 2-hop's 0.216? If even this
supervised CEILING can't clear the bar with room, an LP-gradient-trained router
certainly won't -> refute the thesis in hours, not days.

Design (relation-only routing = the clean, circularity-free variant):
  - Learn per-relation HEAD-bucket and TAIL-bucket soft assignments (logits [R, M]).
  - Middle-consistency objective: for composable (r_i, r_j) [r_i can chain into r_j],
    maximize P(tail_bucket(r_i) == head_bucket(r_j)); for non-composable pairs, minimize.
    (This is the strongest supervision the mechanism could get — it directly teaches the
    composition table.)
  - Then HARD-assign buckets, build the content_comp edge adjacency on the real edges
    (edge t attends to edge s iff tail_bucket(rel_t)==head_bucket(rel_s), capped K/target),
    and measure: composable-fraction (vs struct-2hop 0.216) and shared-node-overlap (vs
    struct-1hop -> is it secretly just 1-hop?).
  - Sweep M. Report the ceiling.

GATE: if max-over-M supervised composable_frac < ~0.40 (comfortably above 0.216), or if
shared-node-overlap ~= struct-1hop, REFUTE before building the full pipeline.

Usage: python experiments/phase74b_supervised_ceiling.py --sample_frac 0.10 --Ms 48,128,256
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
from experiments.phase46c_link_prediction import load_lp_data, build_train_graph_tensors
from experiments.phase74_content_probe import composable_relpairs, frac_composable, cap_per_target
from delta.graph import DeltaGraph


def shared_node_overlap(adj, edge_index):
    """Fraction of kept (src=s, tgt=t) pairs where tail_entity(t) == head_entity(s)
    (i.e. s structurally continues t) — if ~= struct-1hop's, content is just 1-hop."""
    if adj.shape[1] == 0:
        return 0.0
    s, t = adj[0], adj[1]
    return float((edge_index[1][t] == edge_index[0][s]).float().mean())


def learn_relation_buckets(comp, R, M, steps=400, lr=0.1, neg_per_pos=5, seed=0, device='cpu'):
    """Optimize per-relation head/tail bucket logits to maximize composable middle-consistency.
    Returns hard head_bucket[R], tail_bucket[R]."""
    torch.manual_seed(seed)
    Hh = torch.randn(R, M, device=device, requires_grad=True)   # head-bucket logits
    Ht = torch.randn(R, M, device=device, requires_grad=True)   # tail-bucket logits
    opt = torch.optim.Adam([Hh, Ht], lr=lr)
    pos = torch.tensor(list(comp), device=device) if comp else torch.zeros(0, 2, dtype=torch.long)
    if pos.numel() == 0:
        return Hh.argmax(-1).detach(), Ht.argmax(-1).detach()
    for it in range(steps):
        ph = torch.softmax(Hh, -1); pt = torch.softmax(Ht, -1)
        # composable (r_i -> r_j): tail_bucket(r_i) should match head_bucket(r_j)
        agree_pos = (pt[pos[:, 0]] * ph[pos[:, 1]]).sum(-1).clamp(1e-6, 1)   # P(share middle)
        # negatives: random relation pairs (mostly non-composable)
        ni = torch.randint(0, R, (pos.shape[0] * neg_per_pos,), device=device)
        nj = torch.randint(0, R, (pos.shape[0] * neg_per_pos,), device=device)
        agree_neg = (pt[ni] * ph[nj]).sum(-1).clamp(1e-6, 1)
        loss = -(agree_pos.log().mean()) - ((1 - agree_neg).log().mean())
        # mild load-balance (entropy of usage) to avoid one-bucket collapse
        usage = (torch.softmax(Hh, -1).mean(0) + torch.softmax(Ht, -1).mean(0)) / 2
        loss = loss + 0.01 * (usage * usage.clamp(1e-6).log()).sum()  # -entropy penalty
        opt.zero_grad(); loss.backward(); opt.step()
    return Hh.argmax(-1).detach(), Ht.argmax(-1).detach()


def build_rel_routed_adj(edge_rel, head_b, tail_b, K, seed=0):
    """content_comp adjacency from relation-bucket assignments: edge t attends to edge s
    iff tail_bucket(rel_t) == head_bucket(rel_s). Capped K per target. Vectorized-ish."""
    E = edge_rel.shape[0]
    bh = head_b[edge_rel]   # each edge's head-bucket (as potential source)
    bt = tail_b[edge_rel]   # each edge's tail-bucket (as potential target)
    from collections import defaultdict
    src_by_bucket = defaultdict(list)
    for e in range(E):
        src_by_bucket[int(bh[e])].append(e)
    rows, cols = [], []
    for t in range(E):
        for s in src_by_bucket.get(int(bt[t]), ()):
            if s != t:
                rows.append(t); cols.append(s)
    if not rows:
        return torch.zeros(2, 0, dtype=torch.long)
    adj = torch.stack([torch.tensor(cols), torch.tensor(rows)])
    return cap_per_target(adj, K, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample_frac', type=float, default=0.10)
    ap.add_argument('--sample_seed', type=int, default=42)
    ap.add_argument('--Ms', type=str, default='48,128,256')
    ap.add_argument('--K', type=int, default=32)
    args = ap.parse_args()

    print("Phase 74b — SUPERVISED UPPER-BOUND GATE for composability routing")
    data = load_lp_data('fb15k-237', 'data', sample_mode='edge',
                        sample_frac=args.sample_frac, sample_seed=args.sample_seed, use_cache=True)
    ei, et = build_train_graph_tensors(data['train'])
    E, R = ei.shape[1], data['num_relations']
    print(f"  ents={data['num_entities']} E={E} R={R} deg={2*ei.shape[1]/data['num_entities']:.2f}")
    comp = composable_relpairs(data['train'])
    print(f"  composable relation-pairs: {len(comp)}")

    # structural baselines for the bar
    g = DeltaGraph(node_features=torch.zeros(data['num_entities'], 1),
                   edge_features=torch.zeros(E, 1), edge_index=ei)
    a1 = g.build_edge_adjacency(hops=1); g._edge_adj_cache = None
    a2 = g.build_edge_adjacency(hops=2)
    f1, _ = frac_composable(a1, et, comp); o1 = shared_node_overlap(a1, ei)
    f2, _ = frac_composable(a2, et, comp); o2 = shared_node_overlap(a2, ei)
    print(f"\n  BAR (structural): 1-hop composable={f1:.3f} (node-overlap={o1:.3f}); "
          f"2-hop composable={f2:.3f} (node-overlap={o2:.3f})")

    print(f"\n  SUPERVISED CEILING — best relation-only composability routing per M:")
    print(f"  {'M':>5} {'comp_frac':>10} {'vs 2hop':>9} {'node_overlap':>13} {'pairs':>10}")
    best = 0.0
    for M in [int(x) for x in args.Ms.split(',')]:
        hb, tb = learn_relation_buckets(comp, R, M, steps=400)
        adj = build_rel_routed_adj(et, hb, tb, args.K)
        fc, n = frac_composable(adj, et, comp); ov = shared_node_overlap(adj, ei)
        best = max(best, fc)
        print(f"  {M:>5} {fc:>10.3f} {fc-f2:>+9.3f} {ov:>13.3f} {n:>10,}")

    print(f"\n  VERDICT: supervised ceiling composable_frac = {best:.3f}  (struct-2hop bar = {f2:.3f})")
    if best > f2 + 0.15:
        print(f"  -> PROCEED: large headroom ({best-f2:+.3f} over 2-hop). Composability is separable;")
        print(f"     worth building the full LP-trained hardened pipeline.")
    elif best > f2 + 0.05:
        print(f"  -> MARGINAL: some headroom ({best-f2:+.3f}). Build, but expect a tight/possible-null result.")
    else:
        print(f"  -> REFUTE CHEAPLY: even directly-supervised routing barely beats struct-2hop")
        print(f"     ({best-f2:+.3f}); an LP-trained router will be worse. The composability signal")
        print(f"     is not separable enough at this scale — do NOT build the full pipeline.")


if __name__ == '__main__':
    main()
