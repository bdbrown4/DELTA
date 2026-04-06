"""Phase 48: Asymmetric Node/Edge Temperature at L1+L2
====================================================

Motivation (from Phase 47):
  Phase 47 found that layer-specific sharpening (B: L0 soft, L1+L2 sharp)
  achieves the best LP MRR (0.4783) of all 4 conditions while reducing dead
  heads to 38%.  However, B's 3p MRR (0.3908) didn't match D's (0.4018).

  Key clues from Phase 47 learned temperature evolution:
  - Node temps at L1+L2 drift DOWN from 4.0 -> 3.5-3.7
    -> Optimal node temp is ~2-3, not 4.0
  - Edge temps at L1+L2 drift UP from 4.0 -> 4.4-4.5
    -> Edge attention wants MORE sharpness than 4.0
  - Node attention NEEDS some sharpening to activate (C showed edge-only
    sharpening keeps node heads dead)
  - L0 is structurally dead regardless of temperature

  This suggests initializing node and edge temperatures SEPARATELY,
  following each type's drift direction.

Hypothesis (falsifiable):
  "Condition E (node=2.0, edge=6.0 at L1+L2) or Condition F (node=3.0,
  edge=5.0 at L1+L2) achieves BOTH:
    - LP MRR >= 0.4783 (Phase 47 B, best so far)
    - 3p MRR >= 0.4018 (Phase 46 D, best multi-hop)
  by matching the learned drift direction from Phase 47."

Design:
  6 conditions total (4 reference, 2 new runs):
    A. all temp=1.0          (Phase 46 reference)
    B. L0=1, L1+L2=4        (Phase 47 reference, best LP)
    D. all temp=4.0          (Phase 46 reference, best 3p)
    E. L0=(1,1), L1+L2=(2,6)  (node moderate, edge very sharp) NEW
    F. L0=(1,1), L1+L2=(3,5)  (node medium, edge sharp) NEW
    G. L0=(1,1), L1+L2=(2.5,5)  (midpoint between E and F) NEW

Measurements:
  Same as Phase 46/47: per-head entropy, dead heads, learned temperatures,
  LP MRR/H@10, multi-hop 1p-5p MRR.

Regression safety:
  LP MRR must stay >= 0.47 (Phase 46 DELTA-Full temp=1.0 baseline).
  3p MRR must stay >= 0.35 (regression safety floor from Phase 45).

Usage:
  # Smoke test (5 epochs)
  python experiments/phase48_asymmetric_temp.py --epochs 5

  # Full run
  python experiments/phase48_asymmetric_temp.py --epochs 500 --eval_every 25 --patience 10
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
    print_cross_depth_report,
    print_training_dynamics,
    cross_depth_analysis,
    _serialize_stats,
    _serialize_checkpoints,
)

from experiments.phase47_layer_specific_temp import (
    set_layer_temps,
    describe_temp_config,
    train_with_temp_override,
)


def main():
    parser = argparse.ArgumentParser(
        description='Phase 48: Asymmetric Node/Edge Temperature at L1+L2')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs (default: 5 smoke test)')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_every', type=int, default=None,
                        help='Evaluate every N epochs (default: auto)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_entities', type=int, default=500)
    args = parser.parse_args()

    if args.eval_every is None:
        args.eval_every = max(1, args.epochs // 10)

    max_ent = None if args.max_entities == 0 else args.max_entities
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("PHASE 48: ASYMMETRIC NODE/EDGE TEMPERATURE AT L1+L2")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, eval_every: {args.eval_every}")
    print(f"  Seed: {args.seed}")
    print(f"  Max entities: {max_ent or 'full dataset'}")

    # Load data
    print("\n  Loading data...")
    data = load_lp_data('fb15k-237', max_entities=max_ent)

    # ── Reference results ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  REFERENCE RESULTS (Phases 46-47, not re-run)")
    print("=" * 70)
    print("  A (all temp=1.0):        LP MRR=0.4744, 3p MRR=0.3725, dead=20/24 (83%)")
    print("  B (L0=1, L1+L2=4):      LP MRR=0.4783, 3p MRR=0.3908, dead= 9/24 (38%)")
    print("  D (all temp=4.0):        LP MRR=0.4729, 3p MRR=0.4018, dead= 9/24 (38%)")

    # ── New conditions: E, F, G ────────────────────────────────────────
    # DELTA-Full has 3 layers (L0, L1, L2)
    # node_temp, edge_temp at each layer
    conditions = {
        'E_node2_edge6': {
            # Node moderate (matching drift DOWN from 4->3.5),
            # Edge very sharp (matching drift UP from 4->4.5)
            0: (1.0, 1.0),
            1: (2.0, 6.0),
            2: (2.0, 6.0),
        },
        'F_node3_edge5': {
            # Node medium, edge sharp — bracketing expected optimum
            0: (1.0, 1.0),
            1: (3.0, 5.0),
            2: (3.0, 5.0),
        },
        'G_node2p5_edge5': {
            # Midpoint between E and F
            0: (1.0, 1.0),
            1: (2.5, 5.0),
            2: (2.5, 5.0),
        },
    }

    all_results = {}

    for cond_name, layer_temps in conditions.items():
        print(f"\n{'='*70}")
        print(f"  TRAINING: Condition {cond_name}")
        print(f"  Config: {describe_temp_config(layer_temps)}")
        print(f"{'='*70}")

        (model, edge_index, edge_types, checkpoint_stats,
         best_val_mrr, attn_collector, gate_collector) = train_with_temp_override(
            'delta_full', data, args.epochs, args.lr, device,
            args.batch_size, args.seed, args.eval_every, args.patience,
            layer_temps)

        # Standard LP evaluation
        lp_test = evaluate_lp(model, data['test'], edge_index, edge_types,
                              data['hr_to_tails'], data['rt_to_heads'], device)
        print(f"\n  Standard LP: test_MRR={lp_test['MRR']:.4f}  "
              f"test_H@10={lp_test['Hits@10']:.4f}")

        # Final attention stats
        attn_stats = attn_collector.get_stats()
        gate_stats = gate_collector.get_stats()
        final_temps = get_learned_temperatures(model)

        print_attention_report(cond_name, attn_stats, gate_stats)
        print_temperature_report(cond_name, final_temps, checkpoint_stats)
        print_training_dynamics(checkpoint_stats, cond_name)

        # Cross-depth analysis (multi-hop 1p-5p)
        cross_depth = cross_depth_analysis(
            model, data, attn_collector, device, edge_index, edge_types)
        print_cross_depth_report(cross_depth)

        all_results[cond_name] = {
            'model_type': 'delta_full',
            'layer_temps': {str(k): v for k, v in layer_temps.items()},
            'best_val_mrr': best_val_mrr,
            'lp_test': lp_test,
            'attention_stats': _serialize_stats(attn_stats),
            'gate_stats': gate_stats,
            'final_temperatures': final_temps,
            'checkpoint_stats': _serialize_checkpoints(checkpoint_stats),
            'cross_depth': cross_depth,
        }

    # ═══════════════════════════════════════════════════════════════════
    # Comparative analysis: all 6 conditions
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  COMPARATIVE ANALYSIS: ALL 6 CONDITIONS")
    print(f"{'='*70}")

    # Phase 46+47 reference data
    p_ref = {
        'A_all_temp1': {
            'lp_mrr': 0.4744, 'lp_h10': 0.7860, 'best_val': 0.5030,
            '3p_mrr': 0.3725, 'dead': '20/24 (83%)',
        },
        'B_layer_sharp': {
            'lp_mrr': 0.4783, 'lp_h10': 0.7757, 'best_val': 0.5075,
            '3p_mrr': 0.3908, 'dead': '9/24 (38%)',
        },
        'D_all_temp4': {
            'lp_mrr': 0.4729, 'lp_h10': 0.7901, 'best_val': 0.5106,
            '3p_mrr': 0.4018, 'dead': '9/24 (38%)',
        },
    }

    print(f"\n  {'Condition':<20} {'LP MRR':>8} {'LP H@10':>8} {'3p MRR':>8} {'Dead Heads':>12} {'Best Val':>10}")
    print(f"  {'-'*76}")

    # Print references
    for ref_name, ref in p_ref.items():
        print(f"  {ref_name:<20} {ref['lp_mrr']:8.4f} {ref['lp_h10']:8.4f} "
              f"{ref['3p_mrr']:8.4f} {ref['dead']:>12} {ref['best_val']:10.4f}")

    # Print new conditions
    for cond_name, result in all_results.items():
        lp = result['lp_test']
        attn = result.get('attention_stats', {})
        dead, total = 0, 0
        for li in attn:
            for at in ['node_attn', 'edge_attn']:
                heads = attn[li].get(at, {})
                H = len(heads.get('per_head_norm_entropy', []))
                dead += int(heads.get('dead_head_frac', 0) * H)
                total += H

        cross = result.get('cross_depth', {})
        mrr_3p = cross.get('depth_metrics', {}).get('3p', {}).get('MRR', 0)

        dead_str = f"{dead}/{total} ({dead/max(total,1)*100:.0f}%)"
        print(f"  {cond_name:<20} {lp['MRR']:8.4f} {lp['Hits@10']:8.4f} "
              f"{mrr_3p:8.4f} {dead_str:>12} {result['best_val_mrr']:10.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # Hypothesis evaluation
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  HYPOTHESIS EVALUATION")
    print(f"{'='*70}")

    for cond_name, result in all_results.items():
        lp_mrr = result['lp_test']['MRR']
        cross = result.get('cross_depth', {})
        mrr_3p = cross.get('depth_metrics', {}).get('3p', {}).get('MRR', 0)

        # Target: LP MRR >= 0.4783 (Phase 47 B, best LP) AND 3p MRR >= 0.4018 (Phase 46 D)
        lp_pass = lp_mrr >= 0.4783
        mrr3p_pass = mrr_3p >= 0.4018
        regression_pass = mrr_3p >= 0.35

        print(f"\n  {cond_name}:")
        print(f"    LP MRR = {lp_mrr:.4f}  {'PASS' if lp_pass else 'FAIL'} (>= 0.4783, Phase 47 B)")
        print(f"    3p MRR = {mrr_3p:.4f}  {'PASS' if mrr3p_pass else 'FAIL'} (>= 0.4018, Phase 46 D)")
        print(f"    Regression safety = {'PASS' if regression_pass else 'FAIL'} (3p >= 0.35)")

        if lp_pass and mrr3p_pass:
            print(f"    >>> HYPOTHESIS CONFIRMED: Asymmetric temp beats both B and D <<<")
        elif lp_mrr >= 0.4744 and mrr_3p >= 0.3908:
            print(f"    >>> PROMISING: Beats A on both LP and 3p, approaches B/D <<<")
        elif regression_pass:
            print(f"    >>> PARTIAL: Regression-safe but did not beat Phase 47 best <<<")
        else:
            print(f"    >>> FAILED: Regression safety violated <<<")

    # Temperature evolution summary
    print(f"\n{'='*70}")
    print(f"  TEMPERATURE EVOLUTION SUMMARY")
    print(f"{'='*70}")

    for cond_name, result in all_results.items():
        temps = result.get('final_temperatures', {})
        if temps:
            print(f"\n  {cond_name} final temperatures:")
            for key in sorted(temps.keys()):
                vals = temps[key]
                mean_t = np.mean(vals)
                print(f"    {key}: {' '.join(f'{t:.3f}' for t in vals)}  (mean={mean_t:.3f})")

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'phase48_output.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    print(f"\n{'='*70}")
    print(f"  PHASE 48 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
