"""Phase 68: Random-Subgraph Control for Hop-Depth Ablation

Reviewer concern (Round 2, Weakness 2):
  "Re-running the multi-hop sweep on a random 500-entity subgraph (matched
   node count, not matched degree) would tell the reader whether the
   depth-monotonic pattern is a property of the model or a property of the
   graph density regime."

Phase 66 used a TOP-DEGREE subgraph (top-500 highest-degree entities), which
creates a dense graph (mean degree ~19.7). The reviewer correctly notes this
may inflate multi-hop performance and that the depth-monotonic pattern could
be a density artifact rather than a model property.

This phase uses a RANDOM subgraph (500 entities sampled uniformly at random),
which produces a sparser, more realistic graph. We re-run the same hops=1 vs
hops=2 ablation under identical training conditions.

Hypothesis: The 2-hop edge adjacency advantage on multi-hop chain queries
(2p→3p improvement, hops=2 > hops=1) persists on a random subgraph, confirming
it is a model property rather than a dense-graph artifact.
Expected: hops=2 3p MRR > hops=1 3p MRR by at least +0.005 on the random subgraph.

Three possible outcomes:
  A) Advantage persists (hops=2 > hops=1) → depth-monotonic pattern is robust;
     paper's subgraph claim is substantially strengthened.
  B) Advantage disappears (hops=2 ≈ hops=1, as in Phase 66) → consistent with
     density-saturation story; both graphs are too easy for the mechanism to matter.
  C) Advantage reverses (hops=1 > hops=2) → would be surprising; likely indicates
     the random graph is too sparse for edge adjacency to be informative.

Comparison baseline: Phase 66 degree-biased results are printed alongside.

Usage:
  # Quick smoke test (3 epochs, 1 seed)
  python experiments/phase68_random_subgraph_control.py --epochs 5 --seeds 42

  # Standard run — default N=2500 random entities (~1250 survive, mean degree ~6)
  #   Expect ~6-10h total on RTX 3080 Ti for 2 conditions × 3 seeds
  python experiments/phase68_random_subgraph_control.py \\
      --epochs 500 --eval_every 25 --patience 10 --seeds 42,123,456

  # Larger subgraph for higher query counts (slower)
  python experiments/phase68_random_subgraph_control.py \\
      --max_entities 3500 --epochs 500 --patience 10 --seeds 42,123,456

N-vs-density notes (FB15k-237 power-law degree distribution):
  --max_entities 500  → ~250 entities survive, ~275 triples, mean deg ~2.4 (too sparse)
  --max_entities 1500 → ~760 entities survive, ~2.5k triples, mean deg ~6.6
  --max_entities 2500 → ~1270 entities survive, ~7k triples, mean deg ~11  ← DEFAULT
  --max_entities 3500 → ~1775 entities survive, ~14k triples, mean deg ~15.8
Contrast with Phase 66 degree-biased N=500: ~494 entities, ~9.7k triples, mean deg ~19.7
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import random
import time
import numpy as np
import torch
from collections import defaultdict

# ── Phase 66 infrastructure (re-used wholesale) ────────────────────────────
from experiments.phase66_hop_ablation import (
    HopsControlledEncoder,
    _cap_adj,
    build_condition_adjs,
    train_with_controlled_adj,
    _evaluate_with_adj,
    _evaluate_multihop_with_adj,
    run_condition,
)
from experiments.phase46c_link_prediction import (
    build_train_graph_tensors,
    download_dataset,
)
from experiments.phase42_multihop import (
    generate_multihop_queries,
    audit_queries,
    build_full_adjacency,
)
from delta.datasets import _load_triples


# ═══════════════════════════════════════════════════════════════════════════
# Random-subgraph data loader
# ═══════════════════════════════════════════════════════════════════════════

def load_lp_data_random(name='fb15k-237', data_dir='data',
                        max_entities=500, subgraph_seed=0):
    """Load KG with RANDOM (not degree-biased) entity subsampling.

    Identical to phase46c.load_lp_data() except:
      - entity selection is uniform random (seeded by subgraph_seed)
      - preserves strict train/val/test split separation
      - mean degree of result is ~4-6 (vs ~19.7 for top-degree)

    The subgraph_seed is separate from the training seed so the graph
    structure is the same across all 3 training seeds.
    """
    from experiments.phase46c_link_prediction import download_dataset

    dataset_dir = download_dataset(name, data_dir)

    train_raw = _load_triples(os.path.join(dataset_dir, 'train.txt'))
    val_raw = _load_triples(os.path.join(dataset_dir, 'valid.txt'))
    test_raw = _load_triples(os.path.join(dataset_dir, 'test.txt'))

    all_raw = train_raw + val_raw + test_raw

    all_entities = sorted({e for h, r, t in all_raw for e in (h, t)})
    all_relations = sorted({r for _, r, _ in all_raw})

    num_entities = len(all_entities)

    if max_entities and max_entities < num_entities:
        # RANDOM selection — uniform, seeded for reproducibility
        rng = random.Random(subgraph_seed)
        keep = set(rng.sample(all_entities, max_entities))

        train_raw = [(h, r, t) for h, r, t in train_raw if h in keep and t in keep]
        val_raw   = [(h, r, t) for h, r, t in val_raw   if h in keep and t in keep]
        test_raw  = [(h, r, t) for h, r, t in test_raw  if h in keep and t in keep]

        alive_ents = sorted({e for h, r, t in (train_raw + val_raw + test_raw)
                             for e in (h, t)})
        alive_rels = sorted({r for _, r, _ in (train_raw + val_raw + test_raw)})
        all_entities = alive_ents
        all_relations = alive_rels
        num_entities = len(all_entities)
        num_relations = len(all_relations)
    else:
        num_relations = len(all_relations)

    ent2id = {e: i for i, e in enumerate(all_entities)}
    rel2id = {r: i for i, r in enumerate(all_relations)}

    def encode(triples):
        out = []
        for h, r, t in triples:
            if h in ent2id and t in ent2id and r in rel2id:
                out.append((ent2id[h], rel2id[r], ent2id[t]))
        return out

    train_enc = encode(train_raw)
    val_enc   = encode(val_raw)
    test_enc  = encode(test_raw)

    all_enc = train_enc + val_enc + test_enc
    hr_to_tails = defaultdict(set)
    rt_to_heads = defaultdict(set)
    for h, r, t in all_enc:
        hr_to_tails[(h, r)].add(t)
        rt_to_heads[(r, t)].add(h)

    def to_tensor(triples):
        if not triples:
            return torch.zeros(3, 0, dtype=torch.long)
        arr = torch.tensor(triples, dtype=torch.long)
        return arr.t().contiguous()

    return {
        'train':         to_tensor(train_enc),
        'val':           to_tensor(val_enc),
        'test':          to_tensor(test_enc),
        'num_entities':  num_entities,
        'num_relations': num_relations,
        'hr_to_tails':   dict(hr_to_tails),
        'rt_to_heads':   dict(rt_to_heads),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Summary printing
# ═══════════════════════════════════════════════════════════════════════════

# Phase 66 degree-biased results (3 seeds each) — for direct comparison
PHASE66_REFERENCE = {
    'hops1': {'lp_MRR': 0.498, 'lp_std': 0.008,
              '2p_MRR': 0.721, '2p_std': 0.007,
              '3p_MRR': 0.729, '3p_std': 0.007},
    'hops2': {'lp_MRR': 0.496, 'lp_std': 0.003,
              '2p_MRR': 0.726, '2p_std': 0.014,
              '3p_MRR': 0.731, '3p_std': 0.006},
}


def print_summary(all_results, data_stats):
    """Print Phase 68 summary table with Phase 66 comparison."""
    if not all_results:
        return

    conditions_order = ['hops1', 'hops2']
    by_condition = defaultdict(list)
    for r in all_results:
        by_condition[r['condition']].append(r)

    sep = "=" * 105
    print(f"\n{sep}")
    print("PHASE 68: RANDOM-SUBGRAPH HOP-DEPTH CONTROL")
    print("  Hypothesis: hops=2 > hops=1 multi-hop MRR persists on random (non-degree-biased) subgraph")
    print(f"  Random subgraph: N={data_stats['num_entities']} entities, "
          f"{data_stats['num_relations']} relations, "
          f"{data_stats['num_train']} train triples, "
          f"mean degree ~{data_stats['mean_degree']:.1f}")
    print(f"  Phase 66 (degree-biased): N~494 entities, mean degree ~19.7")
    print(sep)

    header = (f"{'Condition':<10} {'Seeds':>5} {'LP MRR':>10} {'2p MRR':>10} "
              f"{'3p MRR':>10} {'3p H@10':>10} {'2p→3p':>8}")
    print(f"\n{'Phase 68 — RANDOM subgraph':}")
    print(f"{header}")
    print("-" * 75)

    p68_means = {}
    for cond in conditions_order:
        results = by_condition.get(cond, [])
        if not results:
            continue
        n = len(results)

        def mean_std(key):
            vals = [r[key] for r in results]
            m = np.mean(vals)
            s = np.std(vals) if n > 1 else 0.0
            return m, s

        def fmt(key):
            m, s = mean_std(key)
            return f"{m:.3f}±{s:.3f}" if s > 0 else f"{m:.4f}"

        lp_m, _ = mean_std('lp_MRR')
        p2_m, _ = mean_std('2p_MRR')
        p3_m, _ = mean_std('3p_MRR')
        p3h10_m, _ = mean_std('3p_H10')
        delta_23 = p3_m - p2_m
        p68_means[cond] = {'lp': lp_m, '2p': p2_m, '3p': p3_m}

        print(f"{cond:<10} {n:>5} {fmt('lp_MRR'):>10} {fmt('2p_MRR'):>10} "
              f"{fmt('3p_MRR'):>10} {fmt('3p_H10'):>10} {delta_23:>+8.4f}")

    # Phase 66 reference
    print(f"\n{'Phase 66 — DEGREE-BIASED subgraph (reference)':}")
    print(f"{header}")
    print("-" * 75)
    for cond in conditions_order:
        ref = PHASE66_REFERENCE.get(cond)
        if not ref:
            continue
        lp_s = f"{ref['lp_MRR']:.3f}±{ref['lp_std']:.3f}"
        p2_s = f"{ref['2p_MRR']:.3f}±{ref['2p_std']:.3f}"
        p3_s = f"{ref['3p_MRR']:.3f}±{ref['3p_std']:.3f}"
        delta_23 = ref['3p_MRR'] - ref['2p_MRR']
        print(f"{cond:<10} {'3':>5} {lp_s:>10} {p2_s:>10} {p3_s:>10} {'---':>10} {delta_23:>+8.4f}")

    # Cross-condition analysis
    print(f"\n{sep}")
    print("ANALYSIS:")
    h1 = by_condition.get('hops1', [])
    h2 = by_condition.get('hops2', [])
    if h1 and h2:
        g_2p = np.mean([r['2p_MRR'] for r in h2]) - np.mean([r['2p_MRR'] for r in h1])
        g_3p = np.mean([r['3p_MRR'] for r in h2]) - np.mean([r['3p_MRR'] for r in h1])
        print(f"\n  Phase 68 (random):  hops=2 vs hops=1  2p gap={g_2p:+.4f}  3p gap={g_3p:+.4f}")

    ref_h1 = PHASE66_REFERENCE.get('hops1')
    ref_h2 = PHASE66_REFERENCE.get('hops2')
    if ref_h1 and ref_h2:
        g_2p_66 = ref_h2['2p_MRR'] - ref_h1['2p_MRR']
        g_3p_66 = ref_h2['3p_MRR'] - ref_h1['3p_MRR']
        print(f"  Phase 66 (degree):  hops=2 vs hops=1  2p gap={g_2p_66:+.4f}  3p gap={g_3p_66:+.4f}")

    if h1 and h2:
        g_3p = np.mean([r['3p_MRR'] for r in h2]) - np.mean([r['3p_MRR'] for r in h1])
        print()
        if g_3p >= 0.010:
            print("  OUTCOME A: Advantage PERSISTS on random subgraph.")
            print("  → Depth-monotonic pattern is a model property, not a density artifact.")
            print("  → Paper's subgraph claim is substantially strengthened.")
        elif g_3p >= -0.005:
            print("  OUTCOME B: Advantage ABSENT on random subgraph (as in Phase 66 degree).")
            print("  → Consistent with density-saturation story: both subgraphs easy for 1-hop.")
            print("  → Full-graph ablation (Phase 67) remains the primary mechanistic evidence.")
        else:
            print("  OUTCOME C: Advantage REVERSED on random subgraph (hops=1 > hops=2).")
            print("  → Random subgraph may be too sparse for edge adjacency to be informative.")
            print("  → Investigate mean degree; may need larger random subgraph.")

    print(sep)
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Arg parsing + main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Phase 68: Random-subgraph hop-depth control experiment')
    p.add_argument('--epochs',       type=int,   default=500)
    p.add_argument('--eval_every',   type=int,   default=25)
    p.add_argument('--patience',     type=int,   default=10)
    p.add_argument('--lr',           type=float, default=0.003)
    p.add_argument('--batch_size',   type=int,   default=4096)
    p.add_argument('--seeds',        type=str,   default='42,123,456',
                   help='Comma-separated training seeds')
    p.add_argument('--conditions',   type=str,   default='hops1,hops2',
                   help='Comma-separated conditions (hops1, hops2)')
    p.add_argument('--max_entities', type=int,   default=2500,
                   help='Entities to request in random subgraph. Due to FB15k-237 power-law '
                        'degree distribution, ~51%% survive after filtering. '
                        'Default 2500 → ~1250 surviving entities, ~7k triples, mean degree ~11. '
                        'Contrast with Phase 66 degree-biased N=500: mean degree ~19.7.')
    p.add_argument('--subgraph_seed', type=int,  default=0,
                   help='Seed for random entity selection (fixed; independent of training seed)')
    p.add_argument('--max_queries',  type=int,   default=10000)
    p.add_argument('--max_adj_pairs', type=int,  default=None,
                   help='Cap hops=2 adj pairs for smoke tests. Use None for full run.')
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    conditions = [c.strip() for c in args.conditions.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 75)
    print("Phase 68: Random-Subgraph Hop-Depth Control")
    print(f"  device={device}  epochs={args.epochs}  seeds={seeds}")
    print(f"  conditions={conditions}")
    print(f"  subgraph: random N={args.max_entities} (seed={args.subgraph_seed})")
    print("=" * 75)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading random FB15k-237 subgraph...")
    data = load_lp_data_random(
        'fb15k-237', 'data',
        max_entities=args.max_entities,
        subgraph_seed=args.subgraph_seed,
    )

    # Compute mean degree for reporting
    if data['train'].shape[1] > 0:
        num_train = data['train'].shape[1]
        degree = defaultdict(int)
        for i in range(num_train):
            h = data['train'][0, i].item()
            t = data['train'][2, i].item()
            degree[h] += 1
            degree[t] += 1
        mean_deg = np.mean(list(degree.values())) if degree else 0.0
    else:
        num_train = 0
        mean_deg = 0.0

    data_stats = {
        'num_entities':  data['num_entities'],
        'num_relations': data['num_relations'],
        'num_train':     num_train,
        'mean_degree':   mean_deg,
    }
    print(f"  Loaded: N={data['num_entities']} entities, "
          f"{data['num_relations']} relations, "
          f"{num_train} train triples")
    print(f"  Mean entity degree (train): {mean_deg:.1f}")

    if num_train < 2000:
        print(f"\n  WARNING: Only {num_train} training triples. Random subgraph is too sparse "
              f"for meaningful multi-hop evaluation.")
        print(f"  FB15k-237 has a power-law degree distribution: random N=500 requests yield "
              f"only ~250 surviving entities and ~275 triples.")
        print(f"  Recommendation: --max_entities 2500 (yields ~1250 entities, ~7k triples, "
              f"mean degree ~11).")
        print(f"  Aborting to avoid running a scientifically invalid experiment.")
        return

    # ── Multi-hop queries ─────────────────────────────────────────────────
    print("\n[2/4] Generating multi-hop queries...")
    queries = generate_multihop_queries(
        data, max_queries_per_type=args.max_queries, seed=42)
    print(f"  1p: {len(queries['1p'])}, 2p: {len(queries['2p'])}, "
          f"3p: {len(queries['3p'])}")

    if len(queries['2p']) < 100 or len(queries['3p']) < 100:
        print(f"\n  WARNING: Very few multi-hop queries ({len(queries['2p'])} 2p, "
              f"{len(queries['3p'])} 3p). Results will be statistically meaningless.")
        print("  Increase --max_entities (recommended: 2500-3500) for a valid experiment.")

    issues = audit_queries(queries, data)
    if issues:
        print(f"  WARNING: {len(issues)} leakage issues found!")
        for issue in issues[:5]:
            print(f"    {issue}")
    else:
        print("  Leakage audit: PASSED")

    full_hr2t = build_full_adjacency(data)

    # ── Edge adjacencies ──────────────────────────────────────────────────
    print("\n[3/4] Pre-building edge adjacencies...")
    edge_index, edge_types = build_train_graph_tensors(data['train'])
    if args.max_adj_pairs:
        print(f"  NOTE: hops=2 adj capped at {args.max_adj_pairs:,} pairs (smoke test mode)")
    all_adjs = build_condition_adjs(edge_index, device,
                                    max_adj_pairs=args.max_adj_pairs)

    # ── Run conditions × seeds ────────────────────────────────────────────
    print("\n[4/4] Running ablation conditions...")
    all_results = []
    t_total = time.time()

    for cond_name in conditions:
        if cond_name not in all_adjs:
            print(f"  Unknown condition: {cond_name}. "
                  f"Valid: {list(all_adjs.keys())}. Skipping.")
            continue
        adj_tensor, cache_hops = all_adjs[cond_name]
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} (cache_hops={cache_hops}, "
              f"adj_pairs={adj_tensor.shape[1]:,})")
        print('=' * 60)

        for seed in seeds:
            print(f"\n  Seed {seed}:")
            t_seed = time.time()
            result = run_condition(
                cond_name, adj_tensor, cache_hops,
                data, queries, full_hr2t,
                args.epochs, args.lr, device, args.batch_size, seed,
                args.eval_every, args.patience,
            )
            elapsed = time.time() - t_seed
            print(f"  Seed {seed} done in {elapsed:.0f}s")
            all_results.append(result)

    total_elapsed = time.time() - t_total
    print(f"\nTotal elapsed: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(all_results, data_stats)


if __name__ == '__main__':
    main()
