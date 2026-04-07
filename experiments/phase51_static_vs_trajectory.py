"""Phase 51: Static vs Trajectory Temperature Optimization
==========================================================

Motivation (from Phase 50):
  K (anneal node 4.0→2.0 fast) broke the 3p ceiling: 3p MRR=0.4148
  (first to beat D's 0.4018, +0.013). But LP MRR=0.4819 missed target
  (0.4856, gap=-0.004).

  K's best checkpoint (ep 175) had node temp=2.6 — NOT the anneal target
  of 2.0. After ep 175, the continued push toward 2.0 degraded validation.

  M (anneal 4→3 fast) tied H's LP record (0.4887) but 3p only 0.3803.
  K and M bracket the trade-off — the optimal lies between them.

  Three questions:
  1. Is K's 3p advantage due to the VALUE 2.6, or the 4.0→2.6 TRAJECTORY?
  2. Is there a better static node temp (3.2 = L/M checkpoint value)?
  3. Is there a better anneal endpoint between K's 2.0 and M's 3.0?

Hypothesis (falsifiable):
  "Condition P (anneal node 4.0→2.5 over 50% of training, edge=6.0)
  achieves BOTH:
    - LP MRR >= 0.4856 (Phase 48 E / Phase 49 H record)
    - 3p MRR >= 0.4018 (Phase 46 D record)
  by stopping anneal at a less aggressive endpoint than K."

Design:
  Reference conditions (not re-run):
    A. all temp=1.0              (Phase 46 ref)
    D. all temp=4.0              (Phase 46 ref, best 3p until K)
    E. L0=1, L1+L2=(2,6)        (Phase 48 ref)
    H. L0=4, L1+L2=(2,6)        (Phase 49 ref, best LP)
    K. anneal 4→2 fast           (Phase 50 ref, best 3p)

  3 new conditions:
    N. static_2.6: L0=(4,4), L1+L2=(2.6, 6.0) — static, no annealing
       Tests: does the VALUE 2.6 alone explain K's 3p advantage?
    O. static_3.2: L0=(4,4), L1+L2=(3.2, 6.0) — static, no annealing
       Tests: is 3.2 (L/M best checkpoint value) even better?
    P. anneal_moderate: node 4.0→2.5 over 50%, edge learnable from 6.0
       Tests: optimal anneal endpoint between K's 2.0 and M's 3.0

  Predictions:
    If N matches K's 3p (≥0.41): static 2.6 is sufficient, trajectory irrelevant
    If N << K (stays ~H's 0.3930): training trajectory is essential
    P's less aggressive endpoint may improve LP while retaining 3p benefit

Measurements:
  Same as Phase 46-50: per-head entropy, dead heads, learned temperatures,
  LP MRR/H@10, multi-hop 1p-5p MRR.

Regression safety:
  LP MRR must stay >= 0.47. 3p MRR must stay >= 0.35.

Usage:
  # Smoke test (5 epochs)
  python experiments/phase51_static_vs_trajectory.py --epochs 5

  # Full run
  python experiments/phase51_static_vs_trajectory.py --epochs 500 --eval_every 25 --patience 10
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

from experiments.phase50_temp_anneal import (
    train_with_anneal,
)


def main():
    parser = argparse.ArgumentParser(
        description='Phase 51: Static vs Trajectory Temperature')
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
    print("PHASE 51: STATIC VS TRAJECTORY TEMPERATURE OPTIMIZATION")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, eval_every: {args.eval_every}")
    print(f"  Seed: {args.seed}")
    print(f"  Max entities: {max_ent or 'full dataset'}")

    print("\n  Loading data...")
    data = load_lp_data('fb15k-237', max_entities=max_ent)

    # ── Reference results ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  REFERENCE RESULTS (Phases 46-50, not re-run)")
    print("=" * 70)
    print("  A (all temp=1.0):        LP MRR=0.4744, 3p MRR=0.3725, dead=20/24 (83%)")
    print("  D (all temp=4.0):        LP MRR=0.4729, 3p MRR=0.4018, dead= 9/24 (38%)")
    print("  E (L0=1, L1+L2 n=2,e=6):LP MRR=0.4856, 3p MRR=0.3872, dead= 9/24 (38%)")
    print("  H (L0=4, L1+L2 n=2,e=6):LP MRR=0.4887, 3p MRR=0.3930, dead= 9/24 (38%)")
    print("  K (anneal 4→2 fast):     LP MRR=0.4819, 3p MRR=0.4148, dead= 8/24 (33%)")

    # ── Static conditions (N, O) ──────────────────────────────────────
    static_conditions = {
        'N_static_2.6': {
            0: (4.0, 4.0),   # L0: same as H/K
            1: (2.6, 6.0),   # L1: node=2.6 (K's best checkpoint), edge=6.0
            2: (2.6, 6.0),   # L2: same
        },
        'O_static_3.2': {
            0: (4.0, 4.0),   # L0: same
            1: (3.2, 6.0),   # L1: node=3.2 (L/M best checkpoint), edge=6.0
            2: (3.2, 6.0),   # L2: same
        },
    }

    # ── Annealing condition (P) ────────────────────────────────────────
    anneal_conditions = {
        'P_anneal_moderate': (0.50, 4.0, 2.5),  # 50% of training, 4.0 → 2.5
    }
    anneal_init_temps = {
        0: (4.0, 4.0),
        1: (4.0, 6.0),
        2: (4.0, 6.0),
    }

    all_results = {}

    # ── Run static conditions ──────────────────────────────────────────
    for cond_name, layer_temps in static_conditions.items():
        print(f"\n{'='*70}")
        print(f"  TRAINING: Condition {cond_name}")
        print(f"  Config: {describe_temp_config(layer_temps)}")
        print(f"{'='*70}")

        (model, edge_index, edge_types, checkpoint_stats,
         best_val_mrr, attn_collector, gate_collector) = train_with_temp_override(
            'delta_full', data, args.epochs, args.lr, device,
            args.batch_size, args.seed, args.eval_every, args.patience,
            layer_temps)

        lp_test = evaluate_lp(model, data['test'], edge_index, edge_types,
                              data['hr_to_tails'], data['rt_to_heads'], device)
        print(f"\n  Standard LP: test_MRR={lp_test['MRR']:.4f}  "
              f"test_H@10={lp_test['Hits@10']:.4f}")

        attn_stats = attn_collector.get_stats()
        gate_stats = gate_collector.get_stats()
        final_temps = get_learned_temperatures(model)

        print_attention_report(cond_name, attn_stats, gate_stats)
        print_temperature_report(cond_name, final_temps, checkpoint_stats)
        print_training_dynamics(checkpoint_stats, cond_name)

        cross_depth = cross_depth_analysis(
            model, data, attn_collector, device, edge_index, edge_types)
        print_cross_depth_report(cross_depth)

        all_results[cond_name] = {
            'model_type': 'delta_full',
            'config': 'static',
            'init_layer_temps': {str(k): v for k, v in layer_temps.items()},
            'best_val_mrr': best_val_mrr,
            'lp_test': lp_test,
            'attention_stats': _serialize_stats(attn_stats),
            'gate_stats': gate_stats,
            'final_temperatures': final_temps,
            'checkpoint_stats': _serialize_checkpoints(checkpoint_stats),
            'cross_depth': cross_depth,
        }

    # ── Run annealing condition ────────────────────────────────────────
    for cond_name, (frac, node_start, node_end) in anneal_conditions.items():
        anneal_epochs = int(args.epochs * frac)

        print(f"\n{'='*70}")
        print(f"  TRAINING: Condition {cond_name}")
        print(f"  Init: L0=(4.0,4.0), L1+L2=(4.0,6.0)")
        print(f"  Anneal: node {node_start:.1f} → {node_end:.1f} over {anneal_epochs} epochs")
        print(f"{'='*70}")

        (model, edge_index, edge_types, checkpoint_stats,
         best_val_mrr, attn_collector, gate_collector) = train_with_anneal(
            'delta_full', data, args.epochs, args.lr, device,
            args.batch_size, args.seed, args.eval_every, args.patience,
            anneal_init_temps, anneal_epochs, node_start, node_end)

        lp_test = evaluate_lp(model, data['test'], edge_index, edge_types,
                              data['hr_to_tails'], data['rt_to_heads'], device)
        print(f"\n  Standard LP: test_MRR={lp_test['MRR']:.4f}  "
              f"test_H@10={lp_test['Hits@10']:.4f}")

        attn_stats = attn_collector.get_stats()
        gate_stats = gate_collector.get_stats()
        final_temps = get_learned_temperatures(model)

        print_attention_report(cond_name, attn_stats, gate_stats)
        print_temperature_report(cond_name, final_temps, checkpoint_stats)
        print_training_dynamics(checkpoint_stats, cond_name)

        cross_depth = cross_depth_analysis(
            model, data, attn_collector, device, edge_index, edge_types)
        print_cross_depth_report(cross_depth)

        all_results[cond_name] = {
            'model_type': 'delta_full',
            'config': 'anneal',
            'init_layer_temps': {str(k): v for k, v in anneal_init_temps.items()},
            'anneal_config': {
                'frac': frac,
                'anneal_epochs': anneal_epochs,
                'node_start': node_start,
                'node_end': node_end,
            },
            'best_val_mrr': best_val_mrr,
            'lp_test': lp_test,
            'attention_stats': _serialize_stats(attn_stats),
            'gate_stats': gate_stats,
            'final_temperatures': final_temps,
            'checkpoint_stats': _serialize_checkpoints(checkpoint_stats),
            'cross_depth': cross_depth,
        }

    # ═══════════════════════════════════════════════════════════════════
    # Comparative analysis
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  COMPARATIVE ANALYSIS: ALL CONDITIONS")
    print(f"{'='*70}")

    p_ref = {
        'A_all_temp1': {
            'lp_mrr': 0.4744, 'lp_h10': 0.7860, 'best_val': 0.5030,
            '3p_mrr': 0.3725, 'dead': '20/24 (83%)',
        },
        'D_all_temp4': {
            'lp_mrr': 0.4729, 'lp_h10': 0.7901, 'best_val': 0.5106,
            '3p_mrr': 0.4018, 'dead': '9/24 (38%)',
        },
        'E_node2_edge6': {
            'lp_mrr': 0.4856, 'lp_h10': 0.8004, 'best_val': 0.4889,
            '3p_mrr': 0.3872, 'dead': '9/24 (38%)',
        },
        'H_l0_4_e_asym': {
            'lp_mrr': 0.4887, 'lp_h10': 0.7973, 'best_val': 0.4925,
            '3p_mrr': 0.3930, 'dead': '9/24 (38%)',
        },
        'K_anneal_fast': {
            'lp_mrr': 0.4819, 'lp_h10': 0.7901, 'best_val': 0.5046,
            '3p_mrr': 0.4148, 'dead': '8/24 (33%)',
        },
    }

    print(f"\n  {'Condition':<22} {'LP MRR':>8} {'LP H@10':>8} {'3p MRR':>8} "
          f"{'Dead Heads':>12} {'Best Val':>10}")
    print(f"  {'-'*78}")

    for ref_name, ref in p_ref.items():
        print(f"  {ref_name:<22} {ref['lp_mrr']:8.4f} {ref['lp_h10']:8.4f} "
              f"{ref['3p_mrr']:8.4f} {ref['dead']:>12} {ref['best_val']:10.4f}")

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
        print(f"  {cond_name:<22} {lp['MRR']:8.4f} {lp['Hits@10']:8.4f} "
              f"{mrr_3p:8.4f} {dead_str:>12} {result['best_val_mrr']:10.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # Hypothesis evaluation
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  HYPOTHESIS EVALUATION")
    print(f"{'='*70}")

    for cond_name, result in all_results.items():
        lp = result['lp_test']
        cross = result.get('cross_depth', {})
        mrr_3p = cross.get('depth_metrics', {}).get('3p', {}).get('MRR', 0)

        lp_pass = lp['MRR'] >= 0.4856
        threep_pass = mrr_3p >= 0.4018
        both = lp_pass and threep_pass

        status = "CONFIRMED" if both else "PARTIAL" if (lp_pass or threep_pass) else "REJECTED"
        print(f"\n  {cond_name}:")
        print(f"    LP MRR:  {lp['MRR']:.4f} (target >= 0.4856) → "
              f"{'PASS' if lp_pass else 'FAIL'}")
        print(f"    3p MRR:  {mrr_3p:.4f} (target >= 0.4018) → "
              f"{'PASS' if threep_pass else 'FAIL'}")
        print(f"    Overall: {status}")

    # ═══════════════════════════════════════════════════════════════════
    # Trajectory vs static comparison
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  TRAJECTORY VS STATIC ANALYSIS")
    print(f"{'='*70}")

    n_cross = all_results.get('N_static_2.6', {}).get('cross_depth', {})
    n_3p = n_cross.get('depth_metrics', {}).get('3p', {}).get('MRR', 0) if n_cross else 0
    n_lp = all_results.get('N_static_2.6', {}).get('lp_test', {}).get('MRR', 0)

    print(f"\n  K (anneal 4→2, checkpoint node=2.6): LP={0.4819:.4f}, 3p={0.4148:.4f}")
    print(f"  N (static node=2.6):                  LP={n_lp:.4f}, 3p={n_3p:.4f}")
    diff_lp = n_lp - 0.4819
    diff_3p = n_3p - 0.4148
    print(f"  N vs K delta:                         LP={diff_lp:+.4f}, 3p={diff_3p:+.4f}")

    if n_3p >= 0.41:
        print(f"\n  → N matches K's 3p: TRAJECTORY IS NOT ESSENTIAL")
        print(f"    The value 2.6 alone explains the 3p advantage.")
    elif n_3p < 0.40:
        print(f"\n  → N is significantly below K: TRAJECTORY IS ESSENTIAL")
        print(f"    High node temp during early training creates representations")
        print(f"    that cannot be achieved with static initialization.")
    else:
        print(f"\n  → N partially matches K: TRAJECTORY HAS MODEST EFFECT")

    # ═══════════════════════════════════════════════════════════════════
    # Final temperature comparison
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  FINAL TEMPERATURES (BEST MODEL CHECKPOINT)")
    print(f"{'='*70}")

    for cond_name, result in all_results.items():
        temps = result.get('final_temperatures', {})
        if temps:
            print(f"\n  {cond_name}:")
            for key in sorted(temps.keys()):
                vals = temps[key]
                mean_t = np.mean(vals)
                print(f"    {key}: {' '.join(f'{t:.3f}' for t in vals)}  "
                      f"(mean={mean_t:.3f})")

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'phase51_output.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    print(f"\n{'='*70}")
    print(f"  PHASE 51 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
