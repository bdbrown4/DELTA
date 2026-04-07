"""Phase 52: Closing K's LP Gap
=================================

Motivation (from Phases 50-51):
  K (anneal node 4→2 fast) is closest to the combined target:
    - LP MRR = 0.4819 (gap −0.004 from target 0.4856)
    - 3p MRR = 0.4148 (exceeds target 0.4018 by +0.013)
  
  P (anneal 4→2.5 moderate) achieved new LP record (0.4890) but 3p=0.3823.
  Phase 51 confirmed trajectory matters: K (annealed to 2.6) beats N
  (static 2.6) by +0.015 on 3p.
  
  K's LP gap analysis:
    K checkpoint (ep 175): L1_edge=6.210, L2_edge=6.533
    H checkpoint (ep 200): L1_edge=6.266, L2_edge=6.660
    P checkpoint (ep 200): L1_edge=6.218, L2_edge=6.578
  K's L2_edge (6.53) is the lowest among high-LP configs. Edge sharpness
  at L2 is consistently correlated with LP MRR across all phases.
  
  Two strategies:
  1. Higher edge init: Start edge at 7.0 instead of 6.0, so by ep 175
     the edge temps are already high (~7.2) without needing drift time.
  2. Faster anneal: Complete node anneal by ep 175 (35% schedule) instead
     of ep 250 (50%). This makes node temps learnable sooner, giving the
     model more post-anneal optimization time for edge sharpness.

Hypothesis (falsifiable):
  "Condition Q (K's anneal 4→2 over 50% + edge init 7.0) achieves BOTH:
    - LP MRR >= 0.4856 (E's record, Phase 48)
    - 3p MRR >= 0.4018 (D's record, Phase 46)
  by boosting edge sharpness at K's best checkpoint while preserving
  K's 3p-building node trajectory."

Design:
  Reference conditions (not re-run):
    A. all temp=1.0              (Phase 46 ref)
    D. all temp=4.0              (Phase 46 ref, best 3p until K)
    E. L0=(1,1), L1+L2=(2,6)    (Phase 48 ref, LP target)
    H. L0=(4,4), L1+L2=(2,6)    (Phase 49 ref, 2nd best LP)
    K. anneal 4→2 fast, edge=6   (Phase 50, best 3p, LP target gap −0.004)
    P. anneal 4→2.5, edge=6      (Phase 51, best LP 0.4890)

  3 new conditions:
    Q. K_edge7:     anneal node 4→2 over 50%, edge init 7.0 (from 6.0)
       Tests: does higher edge init close K's LP gap?
    R. K_faster:    anneal node 4→2 over 35% (175ep), edge init 6.0
       Tests: does earlier anneal completion + more learnable time help LP?
    S. K_edge7_faster: anneal node 4→2 over 35%, edge init 7.0
       Tests: combined approach (if Q and R both improve, S should be best)

Measurements:
  Same as Phase 46-51: per-head entropy, dead heads, learned temperatures,
  LP MRR/H@10, multi-hop 1p-5p MRR.

Regression safety:
  LP MRR must stay >= 0.47. 3p MRR must stay >= 0.35.

Usage:
  # Smoke test (5 epochs)
  python experiments/phase52_closing_lp_gap.py --epochs 5

  # Full run
  python experiments/phase52_closing_lp_gap.py --epochs 500 --eval_every 25 --patience 10
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
)

from experiments.phase50_temp_anneal import (
    train_with_anneal,
)


def main():
    parser = argparse.ArgumentParser(
        description='Phase 52: Closing K\'s LP Gap')
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
    print("PHASE 52: CLOSING K'S LP GAP")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, eval_every: {args.eval_every}")
    print(f"  Seed: {args.seed}")
    print(f"  Max entities: {max_ent or 'full dataset'}")

    print("\n  Loading data...")
    data = load_lp_data('fb15k-237', max_entities=max_ent)

    # ── Reference results ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  REFERENCE RESULTS (Phases 46-51, not re-run)")
    print("=" * 70)
    print("  A (all temp=1.0):        LP MRR=0.4744, 3p MRR=0.3725, dead=20/24 (83%)")
    print("  D (all temp=4.0):        LP MRR=0.4729, 3p MRR=0.4018, dead= 9/24 (38%)")
    print("  E (L0=1, L1+L2 n=2,e=6):LP MRR=0.4856, 3p MRR=0.3872, dead= 9/24 (38%)")
    print("  H (L0=4, L1+L2 n=2,e=6):LP MRR=0.4887, 3p MRR=0.3930, dead= 9/24 (38%)")
    print("  K (anneal 4→2, edge=6):  LP MRR=0.4819, 3p MRR=0.4148, dead= 8/24 (33%)")
    print("  P (anneal 4→2.5, edge=6):LP MRR=0.4890, 3p MRR=0.3823, dead= 8/24 (33%)")

    # ── Annealing conditions (Q, R, S) ────────────────────────────────
    # All three use K's core: anneal node from 4.0 to 2.0
    # Vary: edge init (6.0 vs 7.0) and anneal fraction (50% vs 35%)
    anneal_conditions = {
        'Q_K_edge7': {
            'frac': 0.50,       # Same as K: 50% of training
            'node_start': 4.0,
            'node_end': 2.0,
            'edge_init': 7.0,   # Higher than K's 6.0
            'description': 'K anneal (4→2 over 50%) + edge init 7.0',
        },
        'R_K_faster': {
            'frac': 0.35,       # Faster: 35% of training (175 ep at 500 total)
            'node_start': 4.0,
            'node_end': 2.0,
            'edge_init': 6.0,   # Same as K
            'description': 'faster anneal (4→2 over 35%) + edge 6.0',
        },
        'S_K_edge7_faster': {
            'frac': 0.35,       # Faster: 35%
            'node_start': 4.0,
            'node_end': 2.0,
            'edge_init': 7.0,   # Higher edge
            'description': 'faster anneal (4→2 over 35%) + edge 7.0',
        },
    }

    all_results = {}

    for cond_name, cond in anneal_conditions.items():
        anneal_epochs = int(args.epochs * cond['frac'])
        edge_init = cond['edge_init']

        # Init temps: L0=(4,4), L1+L2 node=start, edge=edge_init
        init_temps = {
            0: (4.0, 4.0),
            1: (cond['node_start'], edge_init),
            2: (cond['node_start'], edge_init),
        }

        print(f"\n{'='*70}")
        print(f"  TRAINING: Condition {cond_name}")
        print(f"  {cond['description']}")
        print(f"  Init: L0=(4.0,4.0), L1+L2=(node={cond['node_start']:.1f}, edge={edge_init:.1f})")
        print(f"  Anneal: node {cond['node_start']:.1f} → {cond['node_end']:.1f} "
              f"over {anneal_epochs} epochs ({cond['frac']*100:.0f}% of training)")
        print(f"{'='*70}")

        (model, edge_index, edge_types, checkpoint_stats,
         best_val_mrr, attn_collector, gate_collector) = train_with_anneal(
            'delta_full', data, args.epochs, args.lr, device,
            args.batch_size, args.seed, args.eval_every, args.patience,
            init_temps, anneal_epochs, cond['node_start'], cond['node_end'])

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
            'init_layer_temps': {str(k): v for k, v in init_temps.items()},
            'anneal_config': {
                'frac': cond['frac'],
                'anneal_epochs': anneal_epochs,
                'node_start': cond['node_start'],
                'node_end': cond['node_end'],
                'edge_init': edge_init,
            },
            'description': cond['description'],
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
        'K_anneal_4to2_edge6': {
            'lp_mrr': 0.4819, 'lp_h10': 0.7901, 'best_val': 0.5046,
            '3p_mrr': 0.4148, 'dead': '8/24 (33%)',
        },
        'P_anneal_4to2.5_edge6': {
            'lp_mrr': 0.4890, 'lp_h10': 0.8014, 'best_val': 0.5039,
            '3p_mrr': 0.3823, 'dead': '8/24 (33%)',
        },
    }

    print(f"\n  {'Condition':<25} {'LP MRR':>8} {'LP H@10':>8} {'3p MRR':>8} "
          f"{'Dead Heads':>12} {'Best Val':>10}")
    print(f"  {'-'*81}")

    for ref_name, ref in p_ref.items():
        print(f"  {ref_name:<25} {ref['lp_mrr']:8.4f} {ref['lp_h10']:8.4f} "
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
        print(f"  {cond_name:<25} {lp['MRR']:8.4f} {lp['Hits@10']:8.4f} "
              f"{mrr_3p:8.4f} {dead_str:>12} {result['best_val_mrr']:10.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # Hypothesis evaluation
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  HYPOTHESIS EVALUATION")
    print(f"{'='*70}")

    lp_target = 0.4856
    threep_target = 0.4018

    for cond_name, result in all_results.items():
        lp = result['lp_test']
        cross = result.get('cross_depth', {})
        mrr_3p = cross.get('depth_metrics', {}).get('3p', {}).get('MRR', 0)

        lp_pass = lp['MRR'] >= lp_target
        threep_pass = mrr_3p >= threep_target
        both = lp_pass and threep_pass

        status = "CONFIRMED" if both else "PARTIAL" if (lp_pass or threep_pass) else "REJECTED"
        print(f"\n  {cond_name}:")
        print(f"    LP MRR:  {lp['MRR']:.4f} (target >= {lp_target}) → "
              f"{'PASS' if lp_pass else 'FAIL'}")
        print(f"    3p MRR:  {mrr_3p:.4f} (target >= {threep_target}) → "
              f"{'PASS' if threep_pass else 'FAIL'}")
        print(f"    Overall: {status}")

        if both:
            print(f"\n  *** COMBINED TARGET ACHIEVED! ***")
            print(f"  LP gap from K: {lp['MRR'] - 0.4819:+.4f}")
            print(f"  3p gap from K: {mrr_3p - 0.4148:+.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # Edge temperature analysis (LP driver)
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  EDGE TEMPERATURE VS LP ANALYSIS")
    print(f"{'='*70}")

    print(f"\n  Reference L2_edge at best checkpoint:")
    print(f"    K (LP=0.4819): L2_edge=6.533")
    print(f"    P (LP=0.4890): L2_edge=6.578")
    print(f"    H (LP=0.4887): L2_edge=6.660")

    for cond_name, result in all_results.items():
        temps = result.get('final_temperatures', {})
        lp = result['lp_test']
        l2_edge = temps.get('L2_edge', [])
        l2_node = temps.get('L2_node', [])
        if l2_edge:
            print(f"    {cond_name} (LP={lp['MRR']:.4f}): L2_edge={np.mean(l2_edge):.3f}, "
                  f"L2_node={np.mean(l2_node):.3f}")

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

    # ═══════════════════════════════════════════════════════════════════
    # Best checkpoint epoch analysis
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  BEST CHECKPOINT ANALYSIS")
    print(f"{'='*70}")

    for cond_name, result in all_results.items():
        cps = result.get('checkpoint_stats', [])
        if cps:
            best_cp = max(cps, key=lambda x: x.get('val_MRR', 0))
            sched = best_cp.get('scheduled_node_temp')
            sched_str = f"node_sched={sched:.2f}" if sched else "learnable"
            print(f"\n  {cond_name}: best ep={best_cp['epoch']}, "
                  f"val_MRR={best_cp['val_MRR']:.4f}, {sched_str}")
            temps = best_cp.get('temperatures', {})
            for key in ['L1_edge', 'L2_edge', 'L1_node', 'L2_node']:
                vals = temps.get(key, [])
                if vals:
                    print(f"    {key}: {np.mean(vals):.3f}")

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'phase52_output.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    print(f"\n{'='*70}")
    print(f"  PHASE 52 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
