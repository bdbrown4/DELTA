"""
Phase 41: Generalization Gap Investigation

Phase 40 revealed a striking pattern in val→test MRR gaps:

  Model               Val MRR   Test MRR   Gap
  ─────────────────── ────────  ─────────  ────────
  GRIT                0.4827    0.4390     0.044  ← worst
  DELTA-Matched       0.5300    0.4950     0.035  ← focus
  SelfBootstrap       0.5104    0.4891     0.021
  GraphGPS            0.5303    0.5126     0.018
  DistMult            0.5016    0.4841     0.017
  SelfBootstrapHybrid 0.5171    0.5089     0.008  ← best encoder
  DELTA-Full          0.4909    0.4938    ≈ 0.000

Key observations:
  1. DELTA-Matched and GraphGPS have near-identical val peaks (0.5300 vs 0.5303)
     but DELTA-Matched's test gap is 2× larger (0.035 vs 0.018).
  2. DELTA-Full practically zero-gaps despite being LARGER than DELTA-Matched.
  3. SelfBootstrapHybrid generalizes best of all encoder models (0.008 gap).
  4. The Phase 40 optimizer used no weight decay (Adam, wd=0.0).

Hypotheses:
  H1. Weight decay (L2 regularization) — primary lever. DELTA's edge attention
      can memorize specific val-set co-occurrence patterns; L2 penalizes large
      weight norms and should reduce this memorization.
  H2. DELTA-Full's zero gap may reflect: (a) lower peak → less overfitting,
      or (b) deeper networks produce more averaged, generalizable representations.
  H3. SBH's 2-stage architecture acts as implicit regularization (information
      bottleneck at the bridge MLP between stages).

Experiment design:
  - Models: graphgps, delta_matched, self_bootstrap_hybrid (the three most
    informative based on Phase 40)
  - Conditions:
      wd=0.0   (Phase 40 baseline — reproduced for comparison)
      wd=1e-4  (standard KGE weight decay, e.g. RotatE default)
      wd=1e-3  (stronger regularization)
  - Epochs: 300 (all models peak by ep 200; cutting 200 epochs saves 40% time)
  - Eval: every 25, patience 15
  - Seeds: 1 (single seed for speed; extend if results are close)
  - Metric of interest: test_MRR AND val-test gap (both reported)

Time estimates (GPU, top-500 entities):
  graphgps:            ~7s/ep × 300ep = ~35 min per condition (3 × 35 ≈ 1.75h)
  delta_matched:     ~270s/ep × 300ep =  ~22h per condition (very slow)
  self_bootstrap_hybrid: ~510s/ep × 300ep = ~42h per condition

  ─── QUICK MODE (--quick) ──────────────────────────────────────────────────
  Uses max_entities=200; estimated delta_matched ~105s/ep (≈ 40% of 500-entity
  time). Quick mode: 200 epochs, patience=10.
  graphgps:    ~3s/ep × 200ep ≈ 10 min per condition (3 × 10 ≈ 30 min)
  delta_matched: ~105s/ep × 200ep ≈ 6h per condition (3 × 6 ≈ 18h)

  Recommendation: Run graphgps-only first (~1.75h) to validate the hypothesis.
  If weight_decay improves graphgps gap, proceed with delta_matched overnight.

Usage:
  # GraphGPS only, all three WD conditions (~1.75h on GPU)
  python experiments/phase41_generalization_gap.py --models graphgps

  # All three target models, all conditions (multi-day on GPU)
  python experiments/phase41_generalization_gap.py

  # Quick mode (smaller entity subset, faster per-epoch time)
  python experiments/phase41_generalization_gap.py --quick --models graphgps,delta_matched

  # Single condition (e.g. just test wd=1e-4 on delta_matched)
  python experiments/phase41_generalization_gap.py --models delta_matched --conditions 1e-4
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import torch

# Reuse Phase 40 infrastructure
from experiments.phase46c_link_prediction import (
    load_lp_data,
    build_train_graph_tensors,
    create_lp_model,
    run_single,
)

# ═══════════════════════════════════════════════════════════════════════════
# Phase 40 baseline results (for comparison in summary)
# ═══════════════════════════════════════════════════════════════════════════

PHASE40_BASELINES = {
    'graphgps':            {'val_MRR': 0.5303, 'test_MRR': 0.5126, 'test_H10': 0.8128},
    'delta_matched':       {'val_MRR': 0.5300, 'test_MRR': 0.4950, 'test_H10': 0.8035},
    'self_bootstrap_hybrid': {'val_MRR': 0.5171, 'test_MRR': 0.5089, 'test_H10': 0.8158},
    'delta_full':          {'val_MRR': 0.4909, 'test_MRR': 0.4938, 'test_H10': 0.7922},
    'self_bootstrap':      {'val_MRR': 0.5104, 'test_MRR': 0.4891, 'test_H10': 0.7912},
    'grit':                {'val_MRR': 0.4827, 'test_MRR': 0.4390, 'test_H10': 0.7603},
    'distmult':            {'val_MRR': 0.5016, 'test_MRR': 0.4841, 'test_H10': 0.7634},
}


# ═══════════════════════════════════════════════════════════════════════════
# Condition runner
# ═══════════════════════════════════════════════════════════════════════════

def run_condition(model_type, data, args, device, weight_decay, seed=1):
    """Run a single (model, weight_decay) condition. Returns result dict."""
    result = run_single(
        model_type=model_type,
        data=data,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        batch_size=args.batch_size,
        seed=seed,
        eval_every=args.eval_every,
        patience=args.patience,
        weight_decay=weight_decay,
    )
    result['weight_decay'] = weight_decay
    result['val_gap'] = result['best_val_MRR'] - result['test_MRR']
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(all_results, models, conditions, show_baseline=True):
    """Print val/test gap comparison across all conditions."""
    print("\n" + "=" * 100)
    print("PHASE 41: GENERALIZATION GAP — VAL/TEST MRR COMPARISON")
    print("=" * 100)

    # Header
    print(f"\n{'Model':<25} {'WD':>8} {'Val MRR':>10} {'Test MRR':>10} "
          f"{'Gap':>8} {'H@10':>8} {'ΔTest vs P40':>14}")
    print("-" * 100)

    for model in models:
        model_results = [r for r in all_results if r['model'] == model]
        p40 = PHASE40_BASELINES.get(model)

        if show_baseline and p40:
            gap = p40['val_MRR'] - p40['test_MRR']
            print(f"  {'[P40 baseline]':<23} {'0.0':>8}  {p40['val_MRR']:>9.4f}  "
                  f"{p40['test_MRR']:>9.4f}  {gap:>+8.4f}  {p40['test_H10']:>8.4f}  "
                  f"{'(reference)':>14}  ← {model}")

        for r in sorted(model_results, key=lambda x: x['weight_decay']):
            wd_str = f"{r['weight_decay']:.0e}" if r['weight_decay'] > 0 else "0.0"
            delta_test = (r['test_MRR'] - p40['test_MRR']) if p40 else 0
            delta_str = f"{delta_test:+.4f}" if p40 else "—"
            print(f"  {model:<23}  {wd_str:>8}  {r['best_val_MRR']:>9.4f}  "
                  f"{r['test_MRR']:>9.4f}  {r['val_gap']:>+8.4f}  "
                  f"{r['test_Hits@10']:>8.4f}  {delta_str:>14}")
        print()

    print("=" * 100)

    # Key takeaways
    print("\nKey findings:")
    for model in models:
        model_results = [r for r in all_results if r['model'] == model]
        if len(model_results) < 2:
            continue
        best = min(model_results, key=lambda x: x['val_gap'])
        worst = max(model_results, key=lambda x: x['val_gap'])
        gap_reduction = worst['val_gap'] - best['val_gap']
        best_test = max(model_results, key=lambda x: x['test_MRR'])
        print(f"  {model}: best gap {best['val_gap']:+.4f} at wd={best['weight_decay']:.0e}, "
              f"worst {worst['val_gap']:+.4f} — gap reduction {gap_reduction:.4f}. "
              f"Best test MRR: {best_test['test_MRR']:.4f} at wd={best_test['weight_decay']:.0e}")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_MODELS = ['graphgps', 'delta_matched', 'self_bootstrap_hybrid']
DEFAULT_CONDITIONS = [0.0, 1e-4, 1e-3]


def main():
    parser = argparse.ArgumentParser(
        description='Phase 41: Generalization gap — weight decay investigation')
    parser.add_argument('--models', type=str,
                        default=','.join(DEFAULT_MODELS),
                        help='Comma-separated model names')
    parser.add_argument('--conditions', type=str, default=None,
                        help='Comma-separated weight_decay values '
                             f'(default: {DEFAULT_CONDITIONS})')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 200 entities, 200 epochs (faster per-epoch time)')
    parser.add_argument('--max_entities', type=int, default=500,
                        help='Entity subset size (overridden by --quick)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_every', type=int, default=25)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='fb15k-237')
    args = parser.parse_args()

    if args.quick:
        args.max_entities = 200
        args.epochs = 200
        args.patience = 10

    models = [m.strip() for m in args.models.split(',')]
    if args.conditions:
        conditions = [float(c.strip()) for c in args.conditions.split(',')]
    else:
        conditions = DEFAULT_CONDITIONS

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Phase 41: Generalization Gap Investigation")
    print(f"  Hypothesis: weight decay reduces val-test overfitting in DELTA")
    print(f"  Dataset:    {args.dataset}, top-{args.max_entities} entities")
    print(f"  Models:     {models}")
    print(f"  Conditions: weight_decay ∈ {conditions}")
    print(f"  Epochs:     {args.epochs} (eval every {args.eval_every}, patience {args.patience})")
    print(f"  Seed:       {args.seed}")
    print(f"  Device:     {device}")

    # Time estimates
    print("\n  Estimated time per condition (rough):")
    per_ep = {'graphgps': 7, 'delta_matched': 270, 'self_bootstrap_hybrid': 510,
              'delta_full': 468, 'grit': 11, 'self_bootstrap': 382, 'distmult': 2}
    for m in models:
        spe = per_ep.get(m, 60)
        if args.quick:
            spe = max(2, spe // 5)  # rough estimate for smaller entity count
        total_min = spe * args.epochs / 60
        total_cond = total_min * len(conditions)
        print(f"    {m:<30} ~{total_min:.0f} min/condition × {len(conditions)} = ~{total_cond:.0f} min")

    print()
    data = load_lp_data(args.dataset, max_entities=args.max_entities)

    all_results = []
    for model_type in models:
        for wd in conditions:
            print(f"\n{'─' * 70}")
            print(f"  Model: {model_type}  |  weight_decay: {wd}")
            print(f"{'─' * 70}")
            try:
                result = run_condition(model_type, data, args, device,
                                       weight_decay=wd, seed=args.seed)
                all_results.append(result)
                print(f"    → val_MRR={result['best_val_MRR']:.4f}  "
                      f"test_MRR={result['test_MRR']:.4f}  "
                      f"gap={result['val_gap']:+.4f}")
            except Exception as e:
                import traceback
                print(f"    FAILED: {e}")
                traceback.print_exc()

    print_summary(all_results, models, conditions)


if __name__ == '__main__':
    main()
