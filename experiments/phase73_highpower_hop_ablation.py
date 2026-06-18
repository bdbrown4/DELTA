"""Phase 73: High-Power Hop Ablation on an Edge-Sampled Sparse Graph (1p-5p).

Phases 66/71/72 could not robustly settle the hop-depth question: dense subgraphs
(deg ~20) saturate 1-hop and force a *random* 5.3% cap on hops=2; small random-node
subgraphs (deg ~5) are sparse but have tiny test sets (200-600 queries) -> high noise.

This phase gets BOTH sparsity AND statistical power by EDGE-sampling the full graph:
  sample_mode='edge', sample_frac=0.10 -> ~11.5K entities, ~27K train edges,
  mean degree ~4.7 (Phase 67's regime), and ~18K test triples (10k+ multi-hop queries
  with real eval power, per Phase 54).

The hops=2 neighborhood (~17.5M pairs) does not fit 12GB through edge attention, so we
apply a FAIR per-target-edge cap (cap_hops2_fair): every edge keeps its FULL 1-hop
neighborhood plus a per-edge random sample of 2-hop-only neighbors up to K total. This
fixes Phase 66's flaw (a *global* random 5.3% cap that left many edges with zero
neighbors) — here every edge keeps a complete local neighborhood, bounded to K.

Hypothesis (falsifiable):
  (a) hops=1 beats node_only on 3p by > 0.010 across 5 seeds (edge attention helps in
      the sparse regime — confirm the Phase 72 trend at power), AND
  (b) hops=2 (fair-capped) beats hops=1 on 3p by > 0.010 (the 2-hop claim, fair test).
REJECTED for whichever clause fails. Expectation from Phases 71/72: (a) holds, (b) fails.

Usage:
  # smoke
  python experiments/phase73_highpower_hop_ablation.py --sample_frac 0.03 --epochs 6 \
      --eval_every 3 --patience 0 --seeds 42 --cap_k 128
  # full
  python experiments/phase73_highpower_hop_ablation.py --sample_frac 0.10 --epochs 600 \
      --eval_every 25 --patience 0 --seeds 42,123,456,7,99 --cap_k 128
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import torch
from delta.graph import DeltaGraph
from experiments.phase46c_link_prediction import load_lp_data, build_train_graph_tensors
from experiments.phase71_sparse_hop_ablation import run_condition, print_summary, QTYPES
from experiments.phase44_depth import generate_extended_queries, audit_extended_queries
from experiments.phase42_multihop import build_full_adjacency


def cap_hops2_fair(adj1, adj2, K, E, seed=0):
    """Per-target-edge fair cap: keep every target edge's FULL 1-hop neighborhood plus
    a per-edge random sample of 2-hop-only neighbors, up to K neighbors per target.

    adj1: [2, E1] 1-hop pairs; adj2: [2, E2] combined (1-hop u 2-hop) pairs.
    Returns a [2, <=E*K] capped adjacency. Unlike a global random cap, no edge is starved.
    """
    if adj2.shape[1] == 0:
        return adj2
    dev = adj2.device
    src2, tgt2 = adj2[0].long(), adj2[1].long()
    # which adj2 pairs are 1-hop?
    flat2 = src2 * E + tgt2
    flat1 = adj1[0].long() * E + adj1[1].long()
    is1 = torch.isin(flat2, flat1)

    cnt = torch.zeros(E, dtype=torch.long, device=dev)
    cnt.scatter_add_(0, tgt2, torch.ones_like(tgt2))
    if cnt.max().item() <= K:
        return adj2

    g = torch.Generator(device=dev); g.manual_seed(seed)
    pri = torch.rand(adj2.shape[1], generator=g, device=dev)
    pri = torch.where(is1, torch.full_like(pri, -1.0), pri)  # 1-hop sorts first
    # sort key: tgt (integer) groups; (pri+1)/3 in [0, 0.67) orders within group
    keys = tgt2.double() + (pri.double() + 1.0) / 3.0
    order = torch.argsort(keys)
    st = torch.zeros(E + 1, dtype=torch.long, device=dev); st[1:] = cnt.cumsum(0)
    pos = torch.arange(adj2.shape[1], device=dev)
    rank = pos - st[tgt2[order]]
    keep = order[rank < K]
    return adj2[:, keep.sort().values]


def build_adjs(edge_index, device, cap_k, seed=0):
    N = edge_index.max().item() + 1
    E = edge_index.shape[1]
    tmp = DeltaGraph(node_features=torch.zeros(N, 1, device=device),
                     edge_features=torch.zeros(E, 1, device=device),
                     edge_index=edge_index.to(device))
    t0 = time.time()
    adj1 = tmp.build_edge_adjacency(hops=1)
    t1 = time.time()
    print(f"    hops=1: {adj1.shape[1]:,} pairs [{t1-t0:.1f}s]")
    tmp._edge_adj_cache = None
    adj2_full = tmp.build_edge_adjacency(hops=2)
    t2 = time.time()
    adj2 = cap_hops2_fair(adj1, adj2_full, cap_k, E, seed=seed)
    note = f" [fair-capped K={cap_k} from {adj2_full.shape[1]:,}]" if adj2.shape[1] < adj2_full.shape[1] else ""
    print(f"    hops=2: {adj2.shape[1]:,} pairs [{t2-t1:.1f}s]{note}")
    empty = torch.zeros(2, 0, dtype=torch.long, device=device)
    return {'node_only': (empty, 1), 'hops1': (adj1, 1), 'hops2': (adj2, 2)}, adj1.shape[1], adj2.shape[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=600)
    p.add_argument('--eval_every', type=int, default=25)
    p.add_argument('--patience', type=int, default=0)
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--batch_size', type=int, default=4096)
    p.add_argument('--seeds', type=str, default='42,123,456,7,99')
    p.add_argument('--conditions', type=str, default='node_only,hops1,hops2')
    p.add_argument('--sample_frac', type=float, default=0.10)
    p.add_argument('--sample_seed', type=int, default=42)
    p.add_argument('--cap_k', type=int, default=128)
    p.add_argument('--max_queries', type=int, default=10000)
    p.add_argument('--out', type=str, default='phase73_output.json')
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    conditions = [c.strip() for c in args.conditions.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Phase 73: High-Power Edge-Sampled Hop Ablation")
    print(f"  device={device} epochs={args.epochs} seeds={seeds} frac={args.sample_frac} cap_k={args.cap_k}")

    print("\n[1/4] Loading edge-sampled FB15k-237...")
    data = load_lp_data('fb15k-237', 'data', sample_mode='edge',
                        sample_frac=args.sample_frac, sample_seed=args.sample_seed,
                        use_cache=True)
    edge_index, edge_types = build_train_graph_tensors(data['train'])
    E_train = edge_index.shape[1]
    mean_deg = 2 * E_train / data['num_entities']
    print(f"  ents={data['num_entities']} E_train={E_train} test={data['test'].shape[1]} "
          f"mean_degree={mean_deg:.2f}")

    print("\n[2/4] Generating 1p-5p queries...")
    queries = generate_extended_queries(data, max_queries_per_type=args.max_queries, seed=42)
    print("  " + ", ".join(f"{qt}={len(queries.get(qt, []))}" for qt in QTYPES))
    issues = audit_extended_queries(queries, data)
    print(f"  Leakage audit: {'PASSED' if not issues else str(len(issues)) + ' ISSUES'}")
    full_hr2t = build_full_adjacency(data)

    print("\n[3/4] Building adjacencies (hops=2 FAIR per-edge cap)...")
    all_adjs, n1, n2 = build_adjs(edge_index, device, args.cap_k, seed=args.sample_seed)
    meta = {'sample_mode': 'edge', 'sample_frac': args.sample_frac,
            'num_entities': data['num_entities'], 'E_train': E_train,
            'test': data['test'].shape[1], 'mean_deg': mean_deg,
            'n1': n1, 'n2': n2, 'cap_k': args.cap_k, 'capped': True,
            'max_entities': f"edge_frac_{args.sample_frac}"}

    print("\n[4/4] Running conditions x seeds...")
    all_results = []
    t_total = time.time()
    for cond_name in conditions:
        adj_tensor, cache_hops = all_adjs[cond_name]
        print(f"\n{'='*60}\nCondition: {cond_name} (pairs={adj_tensor.shape[1]:,})\n{'='*60}")
        for seed in seeds:
            try:
                res = run_condition(cond_name, adj_tensor, cache_hops, data, queries,
                                    full_hr2t, args.epochs, args.lr, device,
                                    args.batch_size, seed, args.eval_every, args.patience)
                all_results.append(res)
                with open(os.path.join(os.path.dirname(__file__), '..', args.out), 'w') as f:
                    json.dump({'meta': meta, 'results': all_results}, f, indent=2)
            except Exception as e:
                import traceback
                print(f"  FAILED ({cond_name}, seed={seed}): {e}"); traceback.print_exc()

    elapsed = time.time() - t_total
    print_summary(all_results, meta)
    print(f"Total time: {elapsed:.0f}s ({elapsed/3600:.2f}h)")
    with open(os.path.join(os.path.dirname(__file__), '..', args.out), 'w') as f:
        json.dump({'meta': meta, 'results': all_results}, f, indent=2)
    print(f"Results saved to {args.out}")


if __name__ == '__main__':
    main()
