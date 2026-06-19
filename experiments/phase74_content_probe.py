"""Phase 74 — offline probe for content-dependent edge adjacency (no training).

Two cheap, decisive offline checks before investing in the full router/STE/training:

(1) QUANTIFY THE DILUTION. A relation-pair (r_i, r_j) is "composable" if some node y
    has an incoming r_i and an outgoing r_j in train (r_i can chain into r_j). Measure
    what fraction of each adjacency's edge-pairs connect a composable relation-pair:
      - structural 1-hop, structural 2-hop, random baseline.
    Hypothesis: structural 2-hop's composable-fraction is ~random (lots of non-composable
    co-incident pairs) -> this is *why* 2-hop dilutes (Phases 66/71/72/73). It sets the bar
    that learned content_comp routing must beat.

(2) PROVE THE ROUTING MECHANISM IS SOUND & LINEAR. Build a content adjacency via
    asymmetric head/tail anchor routing (composability variant) with (untrained) anchors,
    and verify: E_adj is O(E*K) (linear, not E^2), buckets don't collapse (usage entropy),
    and it returns a valid [2,E_adj]. Untrained anchors => composable-fraction at base rate
    (expected; composability is LEARNED) — this check is about soundness/cost, not the win.

Usage:
  python experiments/phase74_content_probe.py --sample_frac 0.10 --M 48 --K 32
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, time
import torch
from collections import defaultdict
from experiments.phase46c_link_prediction import load_lp_data, build_train_graph_tensors
from delta.graph import DeltaGraph


def composable_relpairs(train):
    """Set of (r_i, r_j) such that some node y has incoming r_i and outgoing r_j in train."""
    h, r, t = train[0].tolist(), train[1].tolist(), train[2].tolist()
    inc = defaultdict(set)   # node -> {relations arriving at it}
    out = defaultdict(set)   # node -> {relations leaving it}
    for i in range(len(h)):
        inc[t[i]].add(r[i]); out[h[i]].add(r[i])
    comp = set()
    for y in set(inc) & set(out):
        for ri in inc[y]:
            for rj in out[y]:
                comp.add((ri, rj))
    return comp


def frac_composable(adj, edge_rel, comp):
    """Fraction of edge-pairs (i->j) whose (rel_i, rel_j) is a composable relation-pair."""
    if adj.shape[1] == 0:
        return 0.0, 0
    ri = edge_rel[adj[0]].tolist(); rj = edge_rel[adj[1]].tolist()
    hit = sum(1 for a, b in zip(ri, rj) if (a, b) in comp)
    return hit / adj.shape[1], adj.shape[1]


def cap_per_target(adj, K, seed=0):
    """Keep <=K random source neighbors per target edge (bound E_adj to E*K)."""
    if adj.shape[1] == 0:
        return adj
    src, tgt = adj[0], adj[1]
    E = int(tgt.max()) + 1
    cnt = torch.zeros(E, dtype=torch.long); cnt.scatter_add_(0, tgt, torch.ones_like(tgt))
    if cnt.max().item() <= K:
        return adj
    g = torch.Generator().manual_seed(seed)
    pri = torch.rand(adj.shape[1], generator=g)
    order = torch.argsort(tgt.double() + pri)        # group by tgt, random within
    st = torch.zeros(E + 1, dtype=torch.long); st[1:] = cnt.cumsum(0)
    rank = torch.arange(adj.shape[1]) - st[tgt[order]]
    keep = order[rank < K]
    return adj[:, keep.sort().values]


def content_comp_adjacency(edge_feats, node_feats, edge_index, M, K, d_r=16, seed=0):
    """Composability routing: each edge gets a HEAD-bucket and TAIL-bucket via asymmetric
    (untrained) anchors; target t attends to source s iff t.tail_bucket == s.head_bucket
    (t-then-s composes through a shared latent middle). Returns [2,E_adj] capped to E*K.
    Anchors/projections are random here — this validates the MECHANISM (linear, balanced),
    not learned composability."""
    torch.manual_seed(seed)
    E = edge_index.shape[1]
    d_e, d_n = edge_feats.shape[1], node_feats.shape[1]
    Wh = torch.randn(d_e + d_n, d_r) / (d_e + d_n) ** 0.5
    Wt = torch.randn(d_e + d_n, d_r) / (d_e + d_n) ** 0.5
    Ah = torch.randn(M, d_r) / d_r ** 0.5      # head anchors
    At = torch.randn(M, d_r) / d_r ** 0.5      # tail anchors
    head_in = torch.cat([edge_feats, node_feats[edge_index[0]]], dim=-1)   # edge + its head node
    tail_in = torch.cat([edge_feats, node_feats[edge_index[1]]], dim=-1)   # edge + its tail node
    bh = (head_in @ Wh @ Ah.t()).argmax(-1)    # [E] head bucket of each (potential source)
    bt = (tail_in @ Wt @ At.t()).argmax(-1)    # [E] tail bucket of each (potential target)
    # bucket -> source edges with that head bucket
    src_by_bucket = defaultdict(list)
    for e in range(E):
        src_by_bucket[bh[e].item()].append(e)
    rows, cols = [], []
    for t in range(E):
        for s in src_by_bucket.get(bt[t].item(), ()):
            if s != t:
                cols.append(s); rows.append(t)   # (tgt=t, src=s)
    if not rows:
        return torch.zeros(2, 0, dtype=torch.long), bh, bt
    adj = torch.stack([torch.tensor(cols), torch.tensor(rows)])  # [src, tgt]
    adj = cap_per_target(adj, K, seed=seed)
    return adj, bh, bt


def entropy_frac(bucket_ids, M):
    cnt = torch.zeros(M); cnt.scatter_add_(0, bucket_ids, torch.ones(len(bucket_ids)))
    p = cnt / cnt.sum(); p = p[p > 0]
    import math
    return float(-(p * p.log()).sum() / math.log(M)), float(cnt.max() / cnt.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample_frac', type=float, default=0.10)
    ap.add_argument('--sample_seed', type=int, default=42)
    ap.add_argument('--M', type=int, default=48)
    ap.add_argument('--K', type=int, default=32)
    args = ap.parse_args()

    print("Phase 74 content-adjacency offline probe")
    data = load_lp_data('fb15k-237', 'data', sample_mode='edge',
                        sample_frac=args.sample_frac, sample_seed=args.sample_seed, use_cache=True)
    ei, et = build_train_graph_tensors(data['train'])
    E = ei.shape[1]; N = data['num_entities']
    print(f"  ents={N} E={E} test={data['test'].shape[1]} deg={2*E/N:.2f}")

    t0 = time.time(); comp = composable_relpairs(data['train']);
    print(f"  composable relation-pairs: {len(comp)} (of {data['num_relations']}^2={data['num_relations']**2}) [{time.time()-t0:.1f}s]")

    g = DeltaGraph(node_features=torch.zeros(N, 1), edge_features=torch.zeros(E, 1), edge_index=ei)
    a1 = g.build_edge_adjacency(hops=1); g._edge_adj_cache = None
    a2 = g.build_edge_adjacency(hops=2)
    # random edge-pair baseline (same count as 1-hop)
    rng = torch.Generator().manual_seed(0)
    rnd = torch.stack([torch.randint(0, E, (a1.shape[1],), generator=rng),
                       torch.randint(0, E, (a1.shape[1],), generator=rng)])

    print("\n(1) DILUTION QUANTIFIED — fraction of edge-pairs that are composable relation-types:")
    for name, adj in [('random', rnd), ('struct-1hop', a1), ('struct-2hop', a2)]:
        f, n = frac_composable(adj, et, comp)
        print(f"    {name:<12} composable_frac={f:.4f}  ({n:,} pairs)")

    # (2) mechanism soundness with random anchors (need real edge/node feats; use random embeddings)
    print("\n(2) ROUTING MECHANISM (untrained anchors) — soundness/linearity/balance:")
    d_e, d_n = 24, 48
    ef = torch.randn(E, d_e); nf = torch.randn(N, d_n)
    t1 = time.time()
    cadj, bh, bt = content_comp_adjacency(ef, nf, ei, args.M, args.K)
    dt = time.time() - t1
    ent_h, max_h = entropy_frac(bh, args.M); ent_t, max_t = entropy_frac(bt, args.M)
    fc, nc = frac_composable(cadj, et, comp)
    print(f"    content_comp pairs={cadj.shape[1]:,}  (E*K bound={E*args.K:,})  [{dt:.1f}s]")
    print(f"    head-bucket entropy={ent_h:.2f} (max-frac {max_h:.2f}); tail-bucket entropy={ent_t:.2f} (max-frac {max_t:.2f})  [1.0=balanced, low=collapse]")
    print(f"    content_comp composable_frac={fc:.4f} (untrained -> ~base rate; LEARNED routing must beat struct-2hop)")
    print(f"    linear? E_adj/E = {cadj.shape[1]/E:.1f} (should be <= K={args.K})")


if __name__ == '__main__':
    main()
