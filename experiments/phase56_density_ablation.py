"""Phase 56: Constructor Density Ablation — Precision vs Recall Trade-off
=============================================================================

Motivation (from Phase 55 results):
  brain_hybrid at target_density=0.02 achieved:
    MRR=0.4773 (-0.002 below 0.475 threshold)
    H@10=0.7973 (+3.7% over delta_full — best recall ever)
    H@1=0.3282 (-1.4% vs delta_full)
    4870 constructed edges (~50% of 9703 train triples)

  Key insight: constructed edges improve recall but add noise that hurts
  precision. The precision/recall trade-off suggests the constructor is
  too aggressive — too many low-confidence edges dilute signal.

  Additionally, brain_hybrid was still improving monotonically at epoch 150,
  suggesting undertraining. We extend to 300 epochs to remove this confound.

Hypothesis (falsifiable):
  "Reducing constructor density from 0.02 to 0.01 improves brain_hybrid LP
  MRR to >= 0.480 by constructing fewer, higher-quality edges."

  Expected direction: lower density → fewer edges → less noise → higher
  H@1/MRR at mild H@10 cost. The optimal density balances recall gain
  against precision loss.

Design:
  3 density levels + baseline = 4 conditions:
    A. delta_full         — baseline (no construction)
    B. brain_hybrid@0.005 — conservative: ~1210 edges (12.5% of train KG)
    C. brain_hybrid@0.01  — moderate:     ~2430 edges (25% of train KG)
    D. brain_hybrid@0.02  — aggressive:   ~4870 edges (50% of train KG, = Phase 55)

  ONE primary variable: target_density
  Epochs: 300 (extended from P55's 150 to avoid undertraining confound)
  Seeds: 42 (single seed for initial ablation)

Measurements:
  LP: filtered MRR, H@1, H@3, H@10
  Constructor: actual edges, training time
  Key delta: H@1 improvement from density reduction

Regression safety:
  All conditions must achieve MRR >= 0.40 (above distmult baseline).

Usage:
  python experiments/phase56_density_ablation.py --seeds 42 --epochs 300

  # Quick test
  python experiments/phase56_density_ablation.py --seeds 42 --epochs 5 --eval_every 5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np
import torch

from experiments.phase55_brain_port import (
    create_phase55_model,
    run_single,
    build_train_graph_tensors,
)
from experiments.phase46c_link_prediction import load_lp_data


DENSITIES = [0.005, 0.01, 0.02]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 56: Constructor Density Ablation")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_every', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seeds', type=str, default='42')
    parser.add_argument('--max_entities', type=int, default=500)
    parser.add_argument('--sparsity_weight', type=float, default=0.01)
    parser.add_argument('--densities', type=str, default=None,
                        help='Comma-separated densities (default: 0.005,0.01,0.02)')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    densities = ([float(d) for d in args.densities.split(',')]
                 if args.densities else DENSITIES)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 95)
    print("PHASE 56: Constructor Density Ablation — Precision vs Recall Trade-off")
    print("=" * 95)
    print(f"  Device: {device}")
    print(f"  Seeds: {seeds}")
    print(f"  Densities: {densities}")
    print(f"  Epochs: {args.epochs}, eval_every: {args.eval_every}, "
          f"patience: {args.patience}")
    print(f"  Sparsity weight: {args.sparsity_weight}")

    # Load data
    max_ent = args.max_entities
    data = load_lp_data('fb15k-237', max_entities=max_ent)

    all_results = []

    # --- Condition A: delta_full baseline ---
    print(f"\n{'='*60}")
    print(f"  CONDITION A: delta_full (baseline, no construction)")
    print(f"{'='*60}")
    for seed in seeds:
        r = run_single('delta_full', data, args, device, seed)
        r['condition'] = 'A_delta_full'
        r['target_density'] = 0.0
        all_results.append(r)

    # --- Conditions B, C, D: brain_hybrid at different densities ---
    for i, density in enumerate(densities):
        label = chr(ord('B') + i)
        print(f"\n{'='*60}")
        print(f"  CONDITION {label}: brain_hybrid @ density={density}")
        print(f"{'='*60}")

        # Override target_density for this condition
        args.target_density = density

        for seed in seeds:
            r = run_single('brain_hybrid', data, args, device, seed)
            r['condition'] = f'{label}_brain_d{density}'
            r['target_density'] = density
            all_results.append(r)

    # === SUMMARY ===
    print("\n" + "=" * 100)
    print("PHASE 56: DENSITY ABLATION — RESULTS")
    print("=" * 100)
    print(f"  Data: FB15k-237, {data['num_entities']} entities, "
          f"{data['num_relations']} relations, {data['train'].shape[1]} train")
    print(f"  Epochs: {args.epochs}, seeds: {seeds}")

    header = (f"{'Condition':<25} {'Density':>8} {'MRR':>8} {'H@1':>8} "
              f"{'H@3':>8} {'H@10':>8} {'Edges':>7} {'Time':>7}")
    print(f"\n{header}")
    print("-" * 100)

    # Aggregate by condition
    conditions = {}
    for r in all_results:
        cond = r['condition']
        if cond not in conditions:
            conditions[cond] = []
        conditions[cond].append(r)

    summary_rows = []
    for cond, runs in conditions.items():
        mrr = np.mean([r['test_MRR'] for r in runs])
        h1 = np.mean([r['test_Hits@1'] for r in runs])
        h3 = np.mean([r['test_Hits@3'] for r in runs])
        h10 = np.mean([r['test_Hits@10'] for r in runs])
        edges = np.mean([r.get('constructed_edges', 0) for r in runs])
        t = np.mean([r['time_s'] for r in runs])
        density = runs[0]['target_density']

        row = {
            'condition': cond, 'density': density,
            'MRR': mrr, 'H@1': h1, 'H@3': h3, 'H@10': h10,
            'edges': edges, 'time': t, 'num_seeds': len(runs),
        }
        summary_rows.append(row)

        d_str = f"{density:.3f}" if density > 0 else "—"
        e_str = f"{edges:.0f}" if edges > 0 else "—"
        print(f"{cond:<25} {d_str:>8} {mrr:>8.4f} {h1:>8.4f} "
              f"{h3:>8.4f} {h10:>8.4f} {e_str:>7} {t:>6.0f}s")

    print("-" * 100)

    # --- Analysis ---
    baseline = next((r for r in summary_rows if 'delta_full' in r['condition']), None)
    brain_rows = [r for r in summary_rows if 'brain' in r['condition']]

    if baseline and brain_rows:
        print("\n  ANALYSIS:")
        best_brain = max(brain_rows, key=lambda r: r['MRR'])
        print(f"\n  Best brain_hybrid: {best_brain['condition']} "
              f"(density={best_brain['density']}, MRR={best_brain['MRR']:.4f})")
        print(f"  vs delta_full baseline: MRR={baseline['MRR']:.4f} "
              f"(Δ={best_brain['MRR'] - baseline['MRR']:+.4f})")

        # Precision/recall analysis
        print(f"\n  Precision (H@1) by density:")
        for r in sorted(brain_rows, key=lambda x: x['density']):
            delta_h1 = r['H@1'] - baseline['H@1']
            delta_h10 = r['H@10'] - baseline['H@10']
            print(f"    density={r['density']:.3f}: H@1={r['H@1']:.4f} "
                  f"(Δ={delta_h1:+.4f})  H@10={r['H@10']:.4f} "
                  f"(Δ={delta_h10:+.4f})  edges={r['edges']:.0f}")

        # Verdict
        print(f"\n  HYPOTHESIS: density 0.01 achieves MRR >= 0.480")
        d01 = next((r for r in brain_rows if abs(r['density'] - 0.01) < 0.001), None)
        if d01:
            if d01['MRR'] >= 0.480:
                print(f"    --> CONFIRMED: MRR={d01['MRR']:.4f} >= 0.480")
            elif d01['MRR'] >= 0.475:
                print(f"    --> PARTIAL: MRR={d01['MRR']:.4f} passes 0.475 but not 0.480")
            else:
                print(f"    --> REJECTED: MRR={d01['MRR']:.4f} < 0.475")

        # Overall best
        print(f"\n  OVERALL BEST brain_hybrid density: {best_brain['density']}")
        if best_brain['MRR'] >= 0.490:
            print("  VERDICT: Strong — Brain matches temperature-tuned DELTA-Full range")
        elif best_brain['MRR'] >= 0.480:
            print("  VERDICT: Promising — exceeds baseline, proceed to temperature tuning")
        elif best_brain['MRR'] >= 0.475:
            print("  VERDICT: Viable — matches baseline threshold, needs further optimization")
        else:
            print("  VERDICT: Below threshold — density alone insufficient, investigate architecture")

    # Save output
    output = {
        'phase': 56,
        'title': 'Constructor Density Ablation — Precision vs Recall Trade-off',
        'hypothesis': 'Reducing constructor density from 0.02 to 0.01 improves brain_hybrid LP MRR to >= 0.480',
        'data': {
            'entities': data['num_entities'],
            'relations': data['num_relations'],
            'train_triples': int(data['train'].shape[1]),
        },
        'config': {
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'eval_every': args.eval_every,
            'patience': args.patience,
            'seeds': seeds,
            'densities': densities,
            'sparsity_weight': args.sparsity_weight,
        },
        'results': all_results,
        'summary': summary_rows,
    }

    out_json = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'phase56_output.json')
    try:
        with open(out_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {out_json}")
    except Exception as e:
        print(f"\n  Warning: Could not save results: {e}")

    print("\n  Phase 56 complete.")


if __name__ == "__main__":
    main()
