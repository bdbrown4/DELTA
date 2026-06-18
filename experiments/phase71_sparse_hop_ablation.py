"""Phase 71: Clean UNCAPPED Hop-Depth Ablation in the SPARSE Regime (1p-5p).

Motivation
----------
Phase 66 (dense top-500 subgraph, mean degree ~20) REJECTED the 2-hop edge
adjacency claim — but with two confounds the doc itself flags:
  1. hops=2 (28.2M pairs) had to be RANDOMLY subsampled to 1.5M (5.3%) to fit
     12GB VRAM, so it tested "a random 5.3% of the 2-hop neighborhood vs the
     full 1-hop neighborhood" — not a clean comparison (docs/phase_66.md obs 5).
  2. The dense subgraph saturates 1-hop adjacency (~20 neighbors -> ~400 edge
     pairs/edge); 2-hop adds a near-complete, diluted edge graph (obs 6).

Phase 67 (full sparse FB15k-237, mean degree ~4.1, H200) CONFIRMED a +0.017 3p
benefit for hops=2 — but no docs/phase_67.md or phase67_output.json exists in
the repo, so the project's central confirmed claim is currently unreproducible
locally.

This phase closes both gaps CHEAPLY and LOCALLY by running the Phase 66 ablation
in the SPARSE regime, fully UNCAPPED, extended to 4p/5p:

  * Subgraph: random N entities (sample_mode='random') -> mean degree ~7,
    where the full 2-hop neighborhood is only ~3.3M pairs (fits 12GB uncapped).
    This is the regime where 2-hop is hypothesized (and Phase 67-confirmed) to
    help, unlike the dense top-N subgraph (which is ~degree 20-40 at any N).
  * Conditions: node_only / hops=1 / hops=2 (identical to Phase 66).
  * hops=2 uses the FULL 2-hop adjacency (no random subsample) thanks to the
    sparse-matrix-power fix in delta/graph.py (no longer densifies [E,E]).
  * Evaluation extended to 1p-5p (Phase 44 generator) at 3 seeds.

Hypothesis (falsifiable, directional, with thresholds)
------------------------------------------------------
On the sparse subgraph, edge-to-edge attention provides a multi-hop advantage
ABSENT on the dense subgraph:
  (a) hops=1 and/or hops=2 beat node_only on 3p by > 0.010, AND
  (b) hops=2 beats hops=1 on 3p by > 0.010 (reproducing Phase 67 locally), AND
  (c) the hops=2-vs-hops=1 gap is non-decreasing with depth (3p <= 4p <= 5p).
REJECTED if none of (a)/(b) clears 0.010 across 3 seeds.

Usage
-----
  # Smoke test (fast, 1 seed, tiny)
  python experiments/phase71_sparse_hop_ablation.py --epochs 6 --seeds 42 \
      --max_entities 1500 --max_queries 1000

  # Full run (3 seeds, sparse N=2500, uncapped 2-hop)
  python experiments/phase71_sparse_hop_ablation.py --epochs 500 \
      --eval_every 25 --patience 10 --seeds 42,123,456 --max_entities 2500
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import numpy as np
import torch
from collections import defaultdict

from experiments.phase46c_link_prediction import (
    load_lp_data,
    build_train_graph_tensors,
)
# Reuse Phase 66's controlled-adjacency harness verbatim
from experiments.phase66_hop_ablation import (
    build_condition_adjs,
    train_with_controlled_adj,
    _evaluate_with_adj,
    _encode_with_adj,
)
# Reuse Phase 44's leak-free 1p-5p query generation/audit
from experiments.phase44_depth import (
    generate_extended_queries,
    audit_extended_queries,
)
from experiments.phase42_multihop import (
    build_full_adjacency,
    compute_valid_answers,
)

QTYPES = ['1p', '2p', '3p', '4p', '5p']


# ═══════════════════════════════════════════════════════════════════════════
# Extended (1p-5p) multi-hop evaluation under a controlled edge adjacency
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_multihop_extended_with_adj(model, queries_by_type, edge_index,
                                        edge_types, full_hr2t, device, adj,
                                        cache_hops, temperature=1.0,
                                        batch_size=256):
    """Soft-traversal multi-hop eval (1p-5p) with a pre-built adjacency injected.

    Identical scoring to phase42/phase44, but encodes node features ONCE under
    the controlled adjacency (so the condition's edge-attention is the only
    thing that varies). Query tuples are (anchor, rel_chain, answer, *rest);
    only the first three are used.
    """
    model.eval()
    node_feats = _encode_with_adj(model, edge_index, edge_types,
                                  adj, cache_hops, device)

    results = {}
    for qtype in QTYPES:
        queries = queries_by_type.get(qtype, [])
        if not queries:
            results[qtype] = {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0,
                              'Hits@10': 0.0, 'count': 0}
            continue

        num_hops = len(queries[0][1])
        all_ranks = []

        for start in range(0, len(queries), batch_size):
            batch = queries[start:start + batch_size]
            B = len(batch)

            anchors = torch.tensor([q[0] for q in batch], device=device)
            current_emb = node_feats[anchors]

            for hop in range(num_hops):
                rels = torch.tensor([q[1][hop] for q in batch], device=device)
                hr = current_emb * model.decoder_rel_emb(rels)
                scores = hr @ node_feats.t()
                if hop < num_hops - 1:
                    weights = torch.softmax(scores / temperature, dim=-1)
                    current_emb = weights @ node_feats

            # Vectorized filtered ranking (single scatter + one comparison per batch).
            answers = torch.tensor([q[2] for q in batch], device=device)
            rows = []; cols = []
            for i in range(B):
                ans = batch[i][2]
                for va in compute_valid_answers(batch[i][0], batch[i][1], full_hr2t):
                    if va != ans:
                        rows.append(i); cols.append(va)
            if rows:
                scores[torch.tensor(rows, device=device),
                       torch.tensor(cols, device=device)] = float('-inf')
            tgt = scores[torch.arange(B, device=device), answers]
            all_ranks.append((scores >= tgt.unsqueeze(1)).sum(dim=1).clamp_(min=1))

        ranks = torch.cat(all_ranks).double()
        results[qtype] = {
            'MRR': float((1.0 / ranks).mean()),
            'Hits@1': float((ranks <= 1).double().mean()),
            'Hits@3': float((ranks <= 3).double().mean()),
            'Hits@10': float((ranks <= 10).double().mean()),
            'count': len(queries),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Single (condition, seed) run
# ═══════════════════════════════════════════════════════════════════════════

def run_condition(condition_name, adj_tensor, cache_hops, data, queries,
                  full_hr2t, epochs, lr, device, batch_size, seed,
                  eval_every, patience):
    model, best_val, edge_index, edge_types, elapsed, n_params = \
        train_with_controlled_adj(
            condition_name, adj_tensor, cache_hops, data, epochs, lr, device,
            batch_size, seed, eval_every, patience)

    adj_dev = adj_tensor.to(device)

    lp_test = _evaluate_with_adj(
        model, data['test'], edge_index, edge_types,
        data['hr_to_tails'], data['rt_to_heads'], device, adj_dev, cache_hops)
    print(f"    LP test: MRR={lp_test['MRR']:.4f}  H@10={lp_test['Hits@10']:.4f}")

    mh = evaluate_multihop_extended_with_adj(
        model, queries, edge_index, edge_types, full_hr2t, device,
        adj_dev, cache_hops)

    for qt in QTYPES:
        r = mh[qt]
        if r['count'] > 0:
            print(f"    {qt}: MRR={r['MRR']:.4f}  H@10={r['Hits@10']:.4f}"
                  f"  (n={r['count']})")

    out = {
        'condition': condition_name,
        'seed': seed,
        'params': n_params,
        'train_time_s': elapsed,
        'best_val_MRR': best_val,
        'lp_MRR': lp_test['MRR'],
        'lp_H1': lp_test['Hits@1'],
        'lp_H3': lp_test['Hits@3'],
        'lp_H10': lp_test['Hits@10'],
    }
    for qt in QTYPES:
        out[f'{qt}_MRR'] = mh[qt]['MRR']
        out[f'{qt}_H10'] = mh[qt]['Hits@10']
        out[f'{qt}_count'] = mh[qt]['count']
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(all_results, meta):
    if not all_results:
        return
    by_cond = defaultdict(list)
    for r in all_results:
        by_cond[r['condition']].append(r)

    print("\n" + "=" * 104)
    print("PHASE 71: SPARSE-REGIME UNCAPPED HOP ABLATION (1p-5p)")
    print(f"  Subgraph: random N={meta['max_entities']} -> {meta['num_entities']} ents, "
          f"{meta['E_train']} train edges, mean degree {meta['mean_deg']:.1f}")
    print(f"  hops=1 pairs={meta['n1']:,}  hops=2 pairs={meta['n2']:,} "
          f"(capped={meta['capped']})")
    print("=" * 104)

    W = 13  # wide enough for "0.123±0.045"
    hdr = (f"{'Condition':<11}{'Sd':>3}" + "".join(
        f"{c:>{W}}" for c in ['LP MRR', '1p', '2p', '3p', '4p', '5p',
                              '2p->3p', '3p->5p']))
    print("\n" + hdr)
    print("-" * len(hdr))

    def agg(results, key):
        vals = [r[key] for r in results]
        m = float(np.mean(vals))
        s = float(np.std(vals)) if len(vals) > 1 else 0.0
        return m, s

    def fmt(results, key):
        m, s = agg(results, key)
        return f"{m:.4f}" if s == 0 else f"{m:.3f}±{s:.3f}"

    means = {}
    for cond in ['node_only', 'hops1', 'hops2']:
        results = by_cond.get(cond, [])
        if not results:
            continue
        n = len(results)
        m = {qt: agg(results, f'{qt}_MRR')[0] for qt in QTYPES}
        means[cond] = m
        d23 = m['3p'] - m['2p']
        d35 = m['5p'] - m['3p']
        cells = [fmt(results, f'{q}_MRR') for q in
                 ['lp', '1p', '2p', '3p', '4p', '5p']]
        cells += [f"{d23:+.4f}", f"{d35:+.4f}"]
        print(f"{cond:<11}{n:>3}" + "".join(f"{c:>{W}}" for c in cells))
    print("-" * len(hdr))

    # Gap analysis vs thresholds (0.010 meaningfulness threshold)
    def gap_line(a, b, label):
        if a not in means or b not in means:
            return
        print(f"\n  {label}:")
        for qt in QTYPES:
            g = means[a][qt] - means[b][qt]
            flag = "  <-- >0.010" if abs(g) > 0.010 else ""
            print(f"    {qt} gap = {g:+.4f}{flag}")

    gap_line('hops2', 'hops1', 'hops=2 vs hops=1 (Phase 67 reproduction: expect 3p > +0.010)')
    gap_line('hops1', 'node_only', 'hops=1 vs node_only (does edge attention help at all?)')
    gap_line('hops2', 'node_only', 'hops=2 vs node_only (full edge-attention benefit)')
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--eval_every', type=int, default=25)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--batch_size', type=int, default=4096)
    p.add_argument('--seeds', type=str, default='42,123,456')
    p.add_argument('--conditions', type=str, default='node_only,hops1,hops2')
    p.add_argument('--max_entities', type=int, default=2500)
    p.add_argument('--sample_mode', type=str, default='random',
                   choices=['random', 'degree'])
    p.add_argument('--sample_seed', type=int, default=42)
    p.add_argument('--max_queries', type=int, default=10000)
    p.add_argument('--max_adj_pairs', type=int, default=None,
                   help='Cap hops=2 adj (random subsample). None = UNCAPPED '
                        '(the whole point of this phase).')
    p.add_argument('--out', type=str, default='phase71_output.json')
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    conditions = [c.strip() for c in args.conditions.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Phase 71: Sparse-Regime Uncapped Hop Ablation (1p-5p)")
    print(f"  device={device}  epochs={args.epochs}  seeds={seeds}")
    print(f"  conditions={conditions}")
    print(f"  sample_mode={args.sample_mode} N={args.max_entities} "
          f"seed={args.sample_seed}  max_adj_pairs={args.max_adj_pairs}")

    print("\n[1/4] Loading FB15k-237 (sparse subgraph)...")
    data = load_lp_data('fb15k-237', 'data', max_entities=args.max_entities,
                        sample_mode=args.sample_mode, sample_seed=args.sample_seed,
                        use_cache=True)
    edge_index, edge_types = build_train_graph_tensors(data['train'])
    E_train = edge_index.shape[1]
    mean_deg = 2 * E_train / data['num_entities']
    print(f"  ents={data['num_entities']} E_train={E_train} "
          f"mean_degree={mean_deg:.1f}")

    print("\n[2/4] Generating 1p-5p queries...")
    queries = generate_extended_queries(
        data, max_queries_per_type=args.max_queries, seed=42)
    print("  " + ", ".join(f"{qt}={len(queries.get(qt, []))}" for qt in QTYPES))
    issues = audit_extended_queries(queries, data)
    print(f"  Leakage audit: {'PASSED' if not issues else str(len(issues)) + ' ISSUES'}")
    for issue in issues[:5]:
        print(f"    {issue}")
    full_hr2t = build_full_adjacency(data)

    print("\n[3/4] Building edge adjacencies (hops=2 UNCAPPED unless --max_adj_pairs)...")
    all_adjs = build_condition_adjs(edge_index, device,
                                    max_adj_pairs=args.max_adj_pairs)
    meta = {
        'max_entities': args.max_entities,
        'num_entities': data['num_entities'],
        'E_train': E_train,
        'mean_deg': mean_deg,
        'n1': all_adjs['hops1'][0].shape[1],
        'n2': all_adjs['hops2'][0].shape[1],
        'capped': args.max_adj_pairs is not None,
        'sample_mode': args.sample_mode,
        'sample_seed': args.sample_seed,
    }

    print("\n[4/4] Running conditions x seeds...")
    all_results = []
    t_total = time.time()
    for cond_name in conditions:
        if cond_name not in all_adjs:
            print(f"  Unknown condition: {cond_name}; skipping.")
            continue
        adj_tensor, cache_hops = all_adjs[cond_name]
        print(f"\n{'='*60}\nCondition: {cond_name} "
              f"(cache_hops={cache_hops}, pairs={adj_tensor.shape[1]:,})\n{'='*60}")
        for seed in seeds:
            try:
                res = run_condition(
                    cond_name, adj_tensor, cache_hops, data, queries,
                    full_hr2t, args.epochs, args.lr, device, args.batch_size,
                    seed, args.eval_every, args.patience)
                all_results.append(res)
                # Incremental save so a late OOM doesn't lose prior results
                with open(os.path.join(os.path.dirname(__file__), '..', args.out), 'w') as f:
                    json.dump({'meta': meta, 'results': all_results}, f, indent=2)
            except Exception as e:
                import traceback
                print(f"  FAILED ({cond_name}, seed={seed}): {e}")
                traceback.print_exc()

    elapsed = time.time() - t_total
    print_summary(all_results, meta)
    print(f"Total time: {elapsed:.0f}s ({elapsed/3600:.2f}h)")

    out_path = os.path.join(os.path.dirname(__file__), '..', args.out)
    with open(out_path, 'w') as f:
        json.dump({'meta': meta, 'results': all_results}, f, indent=2)
    print(f"Results saved to {args.out}")


if __name__ == '__main__':
    main()
