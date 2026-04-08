"""Phase 53: Multi-Seed Validation of K and N
==============================================

Motivation (from Phase 52 and advisor consensus):
  After 7 phases of temperature investigation (46-52), three distinct
  operating modes are identified:
    - LP-optimized: P/Q (LP≥0.4890, sharp edges)
    - Balanced-3p: K (3p=0.4148, fast anneal 50%)
    - Deep-reasoning: N (4p=0.3426, 5p=0.3788, static node=2.6)

  K and N are the two most scientifically important configurations:
    K: First to break D's 3p ceiling, best 3p MRR ever
    N: Best 4p and 5p MRR ever, potential deep reasoning mode

  All results so far are single-seed (seed=42). Multi-seed validation
  is required to confirm statistical robustness before publication.

Hypothesis (falsifiable):
  "K's 3p advantage (≥0.4018) and N's deep-hop advantage (4p≥0.30,
  5p≥0.30) are statistically robust across 3 seeds (42, 123, 456),
  with non-overlapping standard deviation bars vs baseline A."

Design:
  2 conditions × 3 seeds = 6 training runs

  K (anneal):  L0=(4,4), L1+L2=(4,6), anneal node 4→2 over 50%
  N (static):  L0=(4,4), L1+L2=(2.6,6)

  Seeds: [42, 123, 456]

Measurements:
  Per seed: LP MRR, LP H@10, 1p-5p MRR, dead heads, learned temps
  Across seeds: mean ± std for each metric

Regression safety:
  LP MRR must stay >= 0.47. 3p MRR must stay >= 0.35.

Usage:
  # Smoke test (5 epochs, 1 seed)
  python experiments/phase53_multiseed_validation.py --epochs 5 --seeds 42

  # Full run (3 seeds)
  python experiments/phase53_multiseed_validation.py --epochs 500 --eval_every 25 --patience 10 --seeds 42,123,456
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
    build_train_graph_tensors,
    evaluate_lp,
)

from experiments.phase46_capacity_signal import (
    AttentionCollector,
    GateCollector,
    get_learned_temperatures,
    print_attention_report,
    print_temperature_report,
    cross_depth_analysis,
    _serialize_stats,
    _serialize_checkpoints,
)

from experiments.phase47_layer_specific_temp import (
    set_layer_temps,
    describe_temp_config,
    train_with_temp_override,
)

from experiments.phase50_temp_anneal import (
    train_with_anneal,
)


def run_single_seed(cond_name, cond_type, data, args, device, seed):
    """Train and evaluate a single condition with a single seed.
    Returns dict of results."""

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

    # Evaluate multi-hop 1p-5p
    cross_depth = cross_depth_analysis(
        model, data, attn_collector, device, edge_index, edge_types)

    # Extract per-hop metrics
    depth_metrics = cross_depth.get('depth_metrics', {})
    hop_mrrs = {}
    for hp in ['1p', '2p', '3p', '4p', '5p']:
        hop_mrrs[hp] = depth_metrics.get(hp, {}).get('MRR', 0.0)

    # Attention health
    attn_stats = attn_collector.get_stats()
    dead, total = 0, 0
    for li in attn_stats:
        for at in ['node_attn', 'edge_attn']:
            heads = attn_stats[li].get(at, {})
            H = len(heads.get('per_head_norm_entropy', []))
            dead += int(heads.get('dead_head_frac', 0) * H)
            total += H

    final_temps = get_learned_temperatures(model)

    result = {
        'seed': seed,
        'best_val_mrr': best_val_mrr,
        'lp_mrr': lp_test['MRR'],
        'lp_h10': lp_test['Hits@10'],
        'hop_mrrs': hop_mrrs,
        'dead_heads': dead,
        'total_heads': total,
        'final_temperatures': final_temps,
        'checkpoint_stats': _serialize_checkpoints(checkpoint_stats),
    }

    print(f"  [{cond_name} seed={seed}] 3p={hop_mrrs.get('3p',0):.4f}  "
          f"4p={hop_mrrs.get('4p',0):.4f}  5p={hop_mrrs.get('5p',0):.4f}  "
          f"dead={dead}/{total}")

    return result


def print_multiseed_summary(cond_name, seed_results):
    """Print mean ± std across seeds for key metrics."""
    seeds = [r['seed'] for r in seed_results]
    n = len(seed_results)

    lp_mrrs = [r['lp_mrr'] for r in seed_results]
    lp_h10s = [r['lp_h10'] for r in seed_results]
    val_mrrs = [r['best_val_mrr'] for r in seed_results]

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
    print(f"  │ Val MRR:  {np.mean(val_mrrs):.4f} ± {np.std(val_mrrs):.4f}"
          f"{'':>{22-len(f'{np.std(val_mrrs):.4f}')}} │")
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
        description='Phase 53: Multi-seed validation of K and N')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs (default: 5 smoke test)')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_every', type=int, default=None,
                        help='Evaluate every N epochs (default: auto)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--seeds', type=str, default='42',
                        help='Comma-separated seeds (default: 42)')
    parser.add_argument('--max_entities', type=int, default=500)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    if args.eval_every is None:
        args.eval_every = max(1, args.epochs // 10)

    max_ent = None if args.max_entities == 0 else args.max_entities
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("PHASE 53: MULTI-SEED VALIDATION OF K AND N")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, eval_every: {args.eval_every}")
    print(f"  Seeds: {seeds}")
    print(f"  Max entities: {max_ent or 'full dataset'}")

    print("\n  Loading data...")
    data = load_lp_data('fb15k-237', max_entities=max_ent)

    # ── Reference results ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  REFERENCE (single seed=42, from Phases 50-51)")
    print("=" * 70)
    print("  K (anneal 4→2, 50%):  LP=0.4819, 3p=0.4148, 4p=0.3107, 5p=0.2811")
    print("  N (static 2.6):       LP=0.4746, 3p=0.4001, 4p=0.3426, 5p=0.3788")
    print("  A (baseline, temp=1): LP=0.4744, 3p=0.3725")

    # ── Condition K: anneal node 4→2, 50% ─────────────────────────────
    conditions = {
        'K_anneal_fast': 'anneal',
        'N_static_2.6': 'static',
    }

    all_condition_results = {}

    for cond_name, cond_type in conditions.items():
        print(f"\n{'='*70}")
        print(f"  CONDITION: {cond_name} ({cond_type})")
        print(f"  Running {len(seeds)} seeds: {seeds}")
        print(f"{'='*70}")

        seed_results = []
        for seed in seeds:
            print(f"\n  --- Seed {seed} ---")
            result = run_single_seed(cond_name, cond_type, data, args,
                                     device, seed)
            seed_results.append(result)

        all_condition_results[cond_name] = seed_results

        print_multiseed_summary(cond_name, seed_results)

    # ═══════════════════════════════════════════════════════════════════
    # Cross-condition comparison
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  CROSS-CONDITION COMPARISON (mean ± std)")
    print(f"{'='*70}")

    baseline_ref = {
        'lp_mrr': 0.4744, '3p_mrr': 0.3725,
        '4p_mrr': 0.0, '5p_mrr': 0.0,
    }

    print(f"\n  {'Condition':<22} {'LP MRR':>14} {'3p MRR':>14} "
          f"{'4p MRR':>14} {'5p MRR':>14}")
    print(f"  {'-'*78}")

    # Baseline (single seed)
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
    # Statistical significance check
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  HYPOTHESIS EVALUATION")
    print(f"{'='*70}")

    for cond_name, seed_results in all_condition_results.items():
        lp_mrrs = [r['lp_mrr'] for r in seed_results]
        mrr_3p = [r['hop_mrrs'].get('3p', 0) for r in seed_results]
        mrr_4p = [r['hop_mrrs'].get('4p', 0) for r in seed_results]
        mrr_5p = [r['hop_mrrs'].get('5p', 0) for r in seed_results]

        print(f"\n  {cond_name}:")

        if cond_name == 'K_anneal_fast':
            # K target: 3p consistently ≥ 0.4018
            threep_mean = np.mean(mrr_3p)
            threep_std = np.std(mrr_3p)
            threep_min = np.min(mrr_3p)
            passes_3p = threep_min >= 0.4018
            passes_3p_mean = threep_mean >= 0.4018
            print(f"    3p ≥ 0.4018?  mean={threep_mean:.4f}±{threep_std:.4f}, "
                  f"min={threep_min:.4f}  "
                  f"{'CONFIRMED (all seeds)' if passes_3p else 'CONFIRMED (mean)' if passes_3p_mean else 'FAILED'}")

            # Non-overlapping with A baseline (3p=0.3725)
            gap = threep_mean - 0.3725
            print(f"    vs A baseline: gap={gap:+.4f}, "
                  f"lower_bound={threep_mean - threep_std:.4f} "
                  f"{'> 0.3725 (non-overlapping)' if (threep_mean - threep_std) > 0.3725 else '≤ 0.3725 (overlapping!)'}")

        elif cond_name == 'N_static_2.6':
            # N targets: 4p ≥ 0.30, 5p ≥ 0.30
            fourp_mean = np.mean(mrr_4p)
            fourp_std = np.std(mrr_4p)
            fivep_mean = np.mean(mrr_5p)
            fivep_std = np.std(mrr_5p)
            fourp_min = np.min(mrr_4p)
            fivep_min = np.min(mrr_5p)

            passes_4p = fourp_min >= 0.30
            passes_5p = fivep_min >= 0.30
            print(f"    4p ≥ 0.30?  mean={fourp_mean:.4f}±{fourp_std:.4f}, "
                  f"min={fourp_min:.4f}  "
                  f"{'CONFIRMED (all seeds)' if passes_4p else 'PARTIAL'}")
            print(f"    5p ≥ 0.30?  mean={fivep_mean:.4f}±{fivep_std:.4f}, "
                  f"min={fivep_min:.4f}  "
                  f"{'CONFIRMED (all seeds)' if passes_5p else 'PARTIAL'}")

            # 3p still reasonable
            threep_mean = np.mean(mrr_3p)
            threep_std = np.std(mrr_3p)
            print(f"    3p:  mean={threep_mean:.4f}±{threep_std:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # Save JSON
    # ═══════════════════════════════════════════════════════════════════

    output = {
        'phase': 53,
        'title': 'Multi-seed validation of K and N',
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
                            'phase53_output.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    print(f"\n{'='*70}")
    print(f"  PHASE 53 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
