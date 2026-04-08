"""Phase 54: High-Power Multi-Hop Evaluation (10k queries)
==========================================================

Motivation (from Phase 53):
  Phase 53 multi-seed validation revealed that 500-query multi-hop
  evaluation is too noisy for reliable single-seed conclusions:
    K: 3p=0.3699±0.0200 (BELOW baseline A=0.3725)
    N: 4p=0.2354±0.0618, 5p=0.2665±0.0738 (HUGE variance)
  
  LP MRR IS robust: K=0.4832±0.0052, N=0.4842±0.0089.
  
  Question: Is the multi-hop variance from evaluation noise (small
  query set) or model noise (different seeds → different representations)?
  
  This phase re-runs K and N with 10k-query evaluation (20x more queries)
  to separate evaluation noise from model noise.

Hypothesis (falsifiable):
  "With 10k-query multi-hop evaluation, the standard deviation of 3p/4p/5p
  MRR across 3 seeds will be at least 50% smaller than Phase 53's 500-query
  results. If so, evaluation noise was the primary source of variance."

Design:
  Same 2 conditions × 3 seeds = 6 runs as Phase 53
  ONLY change: cross_depth_analysis uses 10k queries instead of 500

Usage:
  # Smoke test (5 epochs, 1 seed)
  python experiments/phase54_highpower_multihop.py --epochs 5 --seeds 42

  # Full run
  python experiments/phase54_highpower_multihop.py --epochs 500 --eval_every 25 --patience 10 --seeds 42,123,456
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch

from delta.model import DELTAModel

from experiments.phase46c_link_prediction import (
    load_lp_data,
    evaluate_lp,
)

from experiments.phase46_capacity_signal import (
    AttentionCollector,
    GateCollector,
    get_learned_temperatures,
    _serialize_checkpoints,
)

from experiments.phase44_depth import (
    generate_extended_queries,
)

from experiments.phase47_layer_specific_temp import (
    train_with_temp_override,
)

from experiments.phase50_temp_anneal import (
    train_with_anneal,
)


def evaluate_multihop_highpower(model, data, device, edge_index, edge_types,
                                 max_queries=10000):
    """Evaluate 1p-5p with HIGH query count (default 10k vs 500 in Phase 53)."""
    print(f"\n  Multi-hop evaluation (max {max_queries} queries per depth)...")

    queries = generate_extended_queries(data, max_queries_per_type=max_queries, seed=42)

    ei = edge_index.to(device)
    et = edge_types.to(device)
    full_hr2t = data['hr_to_tails']

    model.eval()
    with torch.no_grad():
        node_feats = model.encode(ei, et)

    depth_results = {}

    for depth_label in ['1p', '2p', '3p', '4p', '5p']:
        qs = queries.get(depth_label, [])
        if not qs:
            print(f"    {depth_label}: no queries available")
            continue

        qs = qs[:max_queries]
        num_hops = int(depth_label[0])
        all_ranks = []

        for q in qs:
            anchor, rel_chain, answer = q[0], q[1], q[2]
            current_emb = node_feats[anchor].unsqueeze(0)

            for hop in range(num_hops):
                rel = torch.tensor([rel_chain[hop]], device=device)
                hr = current_emb * model.decoder_rel_emb(rel)
                scores = hr @ node_feats.t()

                if hop < num_hops - 1:
                    weights = torch.softmax(scores, dim=-1)
                    current_emb = weights @ node_feats
                else:
                    rank = int((scores[0] >= scores[0, answer]).sum().item())
                    all_ranks.append(max(rank, 1))

        ranks = np.array(all_ranks, dtype=float)
        mrr = float(np.mean(1.0 / ranks))
        h10 = float(np.mean(ranks <= 10))

        depth_results[depth_label] = {
            'MRR': mrr,
            'Hits@10': h10,
            'count': len(qs),
        }

        print(f"    {depth_label}: MRR={mrr:.4f}  H@10={h10:.4f}  (n={len(qs)})")

    return depth_results


def run_single_seed(cond_name, cond_type, data, args, device, seed, max_queries):
    """Train and evaluate a single condition with a single seed."""

    if cond_type == 'anneal':
        init_temps = {0: (4.0, 4.0), 1: (4.0, 6.0), 2: (4.0, 6.0)}
        anneal_epochs = int(args.epochs * 0.50)

        (model, edge_index, edge_types, checkpoint_stats,
         best_val_mrr, attn_collector, gate_collector) = train_with_anneal(
            'delta_full', data, args.epochs, args.lr, device,
            args.batch_size, seed, args.eval_every, args.patience,
            init_temps, anneal_epochs, node_start=4.0, node_end=2.0)

    elif cond_type == 'static':
        layer_temps = {0: (4.0, 4.0), 1: (2.6, 6.0), 2: (2.6, 6.0)}

        (model, edge_index, edge_types, checkpoint_stats,
         best_val_mrr, attn_collector, gate_collector) = train_with_temp_override(
            'delta_full', data, args.epochs, args.lr, device,
            args.batch_size, seed, args.eval_every, args.patience,
            layer_temps)

    # Evaluate LP
    lp_test = evaluate_lp(model, data['test'], edge_index, edge_types,
                          data['hr_to_tails'], data['rt_to_heads'], device)
    print(f"\n  [{cond_name} seed={seed}] LP: MRR={lp_test['MRR']:.4f}  "
          f"H@10={lp_test['Hits@10']:.4f}")

    # Evaluate multi-hop with HIGH query count
    depth_results = evaluate_multihop_highpower(
        model, data, device, edge_index, edge_types, max_queries=max_queries)

    hop_mrrs = {}
    for hp in ['1p', '2p', '3p', '4p', '5p']:
        hop_mrrs[hp] = depth_results.get(hp, {}).get('MRR', 0.0)

    # Attention health
    attn_stats = attn_collector.get_stats()
    dead, total = 0, 0
    for li in attn_stats:
        for at in ['node_attn', 'edge_attn']:
            heads = attn_stats[li].get(at, {})
            H = len(heads.get('per_head_norm_entropy', []))
            dead += int(heads.get('dead_head_frac', 0) * H)
            total += H

    result = {
        'seed': seed,
        'best_val_mrr': best_val_mrr,
        'lp_mrr': lp_test['MRR'],
        'lp_h10': lp_test['Hits@10'],
        'hop_mrrs': hop_mrrs,
        'hop_counts': {h: depth_results.get(h, {}).get('count', 0) for h in ['1p','2p','3p','4p','5p']},
        'dead_heads': dead,
        'total_heads': total,
    }

    print(f"  [{cond_name} seed={seed}] 3p={hop_mrrs.get('3p',0):.4f}  "
          f"4p={hop_mrrs.get('4p',0):.4f}  5p={hop_mrrs.get('5p',0):.4f}  "
          f"dead={dead}/{total}")

    return result


def print_multiseed_summary(cond_name, seed_results):
    """Print mean ± std across seeds."""
    seeds = [r['seed'] for r in seed_results]

    lp_mrrs = [r['lp_mrr'] for r in seed_results]
    lp_h10s = [r['lp_h10'] for r in seed_results]

    hop_keys = ['1p', '2p', '3p', '4p', '5p']
    hop_arrays = {h: [r['hop_mrrs'].get(h, 0) for r in seed_results] for h in hop_keys}

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │ {cond_name:^47} │")
    print(f"  │ Seeds: {seeds}{'':>{39-len(str(seeds))}} │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │ LP MRR:   {np.mean(lp_mrrs):.4f} ± {np.std(lp_mrrs):.4f}"
          f"{'':>{22-len(f'{np.std(lp_mrrs):.4f}')}} │")
    print(f"  │ LP H@10:  {np.mean(lp_h10s):.4f} ± {np.std(lp_h10s):.4f}"
          f"{'':>{22-len(f'{np.std(lp_h10s):.4f}')}} │")
    print(f"  ├─────────────────────────────────────────────────┤")
    for h in hop_keys:
        arr = hop_arrays[h]
        if all(v == 0 for v in arr):
            continue
        print(f"  │ {h} MRR:   {np.mean(arr):.4f} ± {np.std(arr):.4f}"
              f"{'':>{22-len(f'{np.std(arr):.4f}')}} │")
    print(f"  └─────────────────────────────────────────────────┘")

    # Per-seed breakdown
    print(f"\n  Per-seed breakdown:")
    print(f"  {'Seed':>6} {'LP MRR':>8} {'LP H@10':>8} {'3p MRR':>8} "
          f"{'4p MRR':>8} {'5p MRR':>8} {'Dead':>6}")
    print(f"  {'-'*56}")
    for r in seed_results:
        print(f"  {r['seed']:>6} {r['lp_mrr']:8.4f} {r['lp_h10']:8.4f} "
              f"{r['hop_mrrs'].get('3p',0):8.4f} "
              f"{r['hop_mrrs'].get('4p',0):8.4f} "
              f"{r['hop_mrrs'].get('5p',0):8.4f} "
              f"{r['dead_heads']:>3}/{r['total_heads']}")


def main():
    parser = argparse.ArgumentParser(
        description='Phase 54: High-power multi-hop evaluation')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_every', type=int, default=None)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seeds', type=str, default='42')
    parser.add_argument('--max_queries', type=int, default=10000,
                        help='Max queries per hop depth (default: 10000)')
    parser.add_argument('--max_entities', type=int, default=500)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    if args.eval_every is None:
        args.eval_every = max(1, args.epochs // 10)

    max_ent = None if args.max_entities == 0 else args.max_entities
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("PHASE 54: HIGH-POWER MULTI-HOP EVALUATION")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, eval_every: {args.eval_every}")
    print(f"  Seeds: {seeds}")
    print(f"  Max queries per hop: {args.max_queries}")
    print(f"  Max entities: {max_ent or 'full dataset'}")

    print("\n  Loading data...")
    data = load_lp_data('fb15k-237', max_entities=max_ent)

    # ── Reference ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  REFERENCE (Phase 53, 500 queries, 3 seeds)")
    print("=" * 70)
    print("  K: LP=0.4832±0.0052, 3p=0.3699±0.0200, 4p=0.2292±0.0221, 5p=0.2481±0.0391")
    print("  N: LP=0.4842±0.0089, 3p=0.3488±0.0472, 4p=0.2354±0.0618, 5p=0.2665±0.0738")
    print("  A baseline: LP=0.4744, 3p=0.3725")

    conditions = {
        'K_anneal_fast': 'anneal',
        'N_static_2.6': 'static',
    }

    all_condition_results = {}

    for cond_name, cond_type in conditions.items():
        print(f"\n{'='*70}")
        print(f"  CONDITION: {cond_name} ({cond_type})")
        print(f"  {len(seeds)} seeds × {args.max_queries} queries per hop")
        print(f"{'='*70}")

        seed_results = []
        for seed in seeds:
            print(f"\n  --- Seed {seed} ---")
            result = run_single_seed(cond_name, cond_type, data, args,
                                     device, seed, args.max_queries)
            seed_results.append(result)

        all_condition_results[cond_name] = seed_results
        print_multiseed_summary(cond_name, seed_results)

    # ═══════════════════════════════════════════════════════════════════
    # Variance comparison: 500q (Phase 53) vs 10kq (Phase 54)
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  VARIANCE COMPARISON: 500q (Phase 53) vs {args.max_queries}q (Phase 54)")
    print(f"{'='*70}")

    p53_stds = {
        'K_anneal_fast': {'3p': 0.0200, '4p': 0.0221, '5p': 0.0391},
        'N_static_2.6': {'3p': 0.0472, '4p': 0.0618, '5p': 0.0738},
    }

    print(f"\n  {'Condition':<22} {'Metric':>6} {'P53 std (500q)':>14} "
          f"{'P54 std ({args.max_queries}q)':>14} {'Reduction':>10}")
    print(f"  {'-'*70}")

    for cond_name, seed_results in all_condition_results.items():
        for hp in ['3p', '4p', '5p']:
            arr = [r['hop_mrrs'].get(hp, 0) for r in seed_results]
            p54_std = np.std(arr)
            p53_std = p53_stds.get(cond_name, {}).get(hp, 0)
            if p53_std > 0:
                reduction = (1 - p54_std / p53_std) * 100
                print(f"  {cond_name:<22} {hp:>6} {p53_std:14.4f} "
                      f"{p54_std:14.4f} {reduction:9.1f}%")
            else:
                print(f"  {cond_name:<22} {hp:>6} {'—':>14} "
                      f"{p54_std:14.4f} {'—':>10}")

    # ═══════════════════════════════════════════════════════════════════
    # Cross-condition comparison
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  CROSS-CONDITION COMPARISON (mean ± std, {args.max_queries}q)")
    print(f"{'='*70}")

    print(f"\n  {'Condition':<22} {'LP MRR':>14} {'3p MRR':>14} "
          f"{'4p MRR':>14} {'5p MRR':>14}")
    print(f"  {'-'*78}")

    print(f"  {'A (baseline, ref)':<22} {'0.4744':>14} {'0.3725':>14} "
          f"{'—':>14} {'—':>14}")

    for cond_name, seed_results in all_condition_results.items():
        lp_mrrs = [r['lp_mrr'] for r in seed_results]
        mrr_3p = [r['hop_mrrs'].get('3p', 0) for r in seed_results]
        mrr_4p = [r['hop_mrrs'].get('4p', 0) for r in seed_results]
        mrr_5p = [r['hop_mrrs'].get('5p', 0) for r in seed_results]

        def fmt(arr):
            return f"{np.mean(arr):.4f}±{np.std(arr):.4f}"

        print(f"  {cond_name:<22} {fmt(lp_mrrs):>14} {fmt(mrr_3p):>14} "
              f"{fmt(mrr_4p):>14} {fmt(mrr_5p):>14}")

    # ═══════════════════════════════════════════════════════════════════
    # Hypothesis
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  HYPOTHESIS EVALUATION")
    print(f"{'='*70}")

    for cond_name, seed_results in all_condition_results.items():
        p53_std_vals = p53_stds.get(cond_name, {})
        current_stds = {}
        for hp in ['3p', '4p', '5p']:
            arr = [r['hop_mrrs'].get(hp, 0) for r in seed_results]
            current_stds[hp] = np.std(arr)

        reductions = []
        for hp in ['3p', '4p', '5p']:
            if p53_std_vals.get(hp, 0) > 0 and current_stds.get(hp, 0) >= 0:
                red = (1 - current_stds[hp] / p53_std_vals[hp]) * 100
                reductions.append(red)

        avg_red = np.mean(reductions) if reductions else 0
        print(f"\n  {cond_name}: avg variance reduction = {avg_red:.1f}%")
        print(f"    Target: ≥50% reduction → "
              f"{'CONFIRMED' if avg_red >= 50 else 'FAILED'}")

    # ═══════════════════════════════════════════════════════════════════
    # Save JSON
    # ═══════════════════════════════════════════════════════════════════

    output = {
        'phase': 54,
        'title': 'High-power multi-hop evaluation (10k queries)',
        'max_queries': args.max_queries,
        'seeds': seeds,
        'epochs': args.epochs,
        'conditions': {},
    }
    for cond_name, seed_results in all_condition_results.items():
        lp_mrrs = [r['lp_mrr'] for r in seed_results]
        lp_h10s = [r['lp_h10'] for r in seed_results]
        hop_keys = ['1p', '2p', '3p', '4p', '5p']
        hop_arrays = {h: [r['hop_mrrs'].get(h, 0) for r in seed_results]
                      for h in hop_keys}

        output['conditions'][cond_name] = {
            'seeds': [r['seed'] for r in seed_results],
            'per_seed': seed_results,
            'summary': {
                'lp_mrr': {'mean': float(np.mean(lp_mrrs)),
                           'std': float(np.std(lp_mrrs))},
                'lp_h10': {'mean': float(np.mean(lp_h10s)),
                           'std': float(np.std(lp_h10s))},
                **{f'{h}_mrr': {'mean': float(np.mean(hop_arrays[h])),
                                'std': float(np.std(hop_arrays[h]))}
                   for h in hop_keys},
            },
        }

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'phase54_output.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    print(f"\n{'='*70}")
    print(f"  PHASE 54 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
