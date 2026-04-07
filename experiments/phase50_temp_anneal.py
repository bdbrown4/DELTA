"""Phase 50: Temperature Annealing (Node Temp Schedule)
======================================================

Motivation (from Phase 49):
  Phases 46-49 tested 10+ static temperature configurations. Key result:
    - D (uniform temp=4.0): Best 3p MRR = 0.4018, LP MRR = 0.4729
    - H (L0=4,4 + L1+L2 node=2,edge=6): Best LP MRR = 0.4887, 3p = 0.3930
  
  The LP/3p trade-off is fundamental to STATIC temperature initialization.
  D's 3p advantage may come from its training TRAJECTORY: early epochs with
  high node temp (4.0) create compositional representations, then the optimizer
  naturally decays node temps toward ~3.5-3.7.
  
  Phase 48 G (node=2.5, edge=5) had the closest 3p to D (0.3970 vs 0.4018),
  suggesting higher node temp correlates with better 3p. But high node temp
  hurts LP. The solution: ANNEAL — start with D's high node temp for early
  representation formation, then decay to E's low node temp for LP performance.

Hypothesis (falsifiable):
  "Condition K (anneal node temp from 4.0 to 2.0 over first 50% of training,
  edge temp learnable from 6.0) achieves BOTH:
    - LP MRR >= 0.4856 (Phase 48 E / Phase 49 H record)
    - 3p MRR >= 0.4018 (Phase 46 D record)
  by combining D's early training dynamics with E's optimal final temperatures."

Design:
  4 reference conditions (not re-run):
    A. all temp=1.0              (Phase 46 ref)
    D. all temp=4.0              (Phase 46 ref, best 3p)
    E. L0=(1,1), L1+L2=(2,6)    (Phase 48 ref, 2nd best LP)
    H. L0=(4,4), L1+L2=(2,6)    (Phase 49 ref, best LP)

  3 new conditions:
    K. anneal_fast: node 4.0→2.0 over first 50%, edge learnable from 6.0
    L. anneal_slow: node 4.0→2.0 over 100% training, edge learnable from 6.0
    M. anneal_partial: node 4.0→3.0 over first 50%, edge learnable from 6.0

  Temperature annealing mechanics:
    - At each eval checkpoint, compute scheduled node temp via linear decay
    - Reset _log_temp for node attention to scheduled value
    - Edge temps remain fully learnable (optimizer adjusts freely)
    - L0 temps: edge learnable from 4.0, node fixed at 4.0 (irrelevant per P49)
    - After anneal period ends, node temps become fully learnable

Measurements:
  Same as Phase 46-49: per-head entropy, dead heads, learned temperatures,
  LP MRR/H@10, multi-hop 1p-5p MRR.

Regression safety:
  LP MRR must stay >= 0.47 (Phase 46 baseline).
  3p MRR must stay >= 0.35 (Phase 45 regression floor).

Usage:
  # Smoke test (5 epochs)
  python experiments/phase50_temp_anneal.py --epochs 5

  # Full run
  python experiments/phase50_temp_anneal.py --epochs 500 --eval_every 25 --patience 10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import time
import numpy as np
import torch

from delta.model import DELTAModel

from experiments.phase46c_link_prediction import (
    load_lp_data,
    build_train_graph_tensors,
    evaluate_lp,
    create_lp_model,
    train_epoch,
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


def anneal_node_temps(model, epoch, anneal_epochs, start_temp, end_temp):
    """Apply linear temperature annealing to node attention heads.

    During the anneal period (epoch 1..anneal_epochs), node temp decays
    linearly from start_temp to end_temp. After the anneal period, node
    temps become fully learnable (no more resets).

    Only affects L1 and L2 node attention. L0 is left alone (100% dead).
    Edge temps are never touched (remain fully learnable).

    Returns the current scheduled node temp (for logging).
    """
    if epoch > anneal_epochs:
        return None  # No more annealing; temps are learnable

    # Linear interpolation
    progress = epoch / anneal_epochs  # 0.0 -> 1.0
    scheduled_temp = start_temp + (end_temp - start_temp) * progress

    encoder = model.encoder
    assert isinstance(encoder, DELTAModel)

    with torch.no_grad():
        for layer_idx in [1, 2]:  # Only L1 and L2
            layer = encoder.layers[layer_idx]
            layer.dual_attn.node_attn._log_temp.fill_(math.log(scheduled_temp))
            if hasattr(layer, 'global_node_attn'):
                layer.global_node_attn._log_temp.fill_(math.log(scheduled_temp))

    return scheduled_temp


def train_with_anneal(model_type, data, epochs, lr, device, batch_size,
                      seed, eval_every, patience, init_layer_temps,
                      anneal_epochs, node_start, node_end):
    """Train with temperature annealing on node attention.

    Args:
        init_layer_temps: Initial temperature config (like set_layer_temps)
        anneal_epochs: Number of epochs over which to anneal
        node_start: Starting node temperature for L1+L2
        node_end: Target node temperature for L1+L2 at end of anneal
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = create_lp_model(model_type,
                            data['num_entities'], data['num_relations'],
                            init_temp=1.0)

    set_layer_temps(model, init_layer_temps)

    temps = get_learned_temperatures(model)
    print(f"\n  Initial temperatures: {temps}")
    print(f"  Anneal: node {node_start:.1f} → {node_end:.1f} "
          f"over {anneal_epochs} epochs (L1+L2 only)")

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = (sum(p.numel() for p in model.encoder.parameters())
             if model.encoder is not None else 0)
    print(f"\n  [{model_type}] seed={seed}, {n_params:,} params "
          f"({n_enc:,} encoder), device={device}")

    attn_collector = AttentionCollector()
    gate_collector = GateCollector()
    is_delta = model.encoder is not None and isinstance(model.encoder, DELTAModel)
    if is_delta:
        attn_collector.instrument_attention(model)
        gate_collector.instrument(model)

    edge_index, edge_types = build_train_graph_tensors(data['train'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_mrr = 0.0
    best_state = None
    evals_no_improve = 0
    checkpoint_stats = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data['train'], edge_index, edge_types,
                           optimizer, device, batch_size)

        if epoch % eval_every == 0 or epoch == epochs:
            # Apply annealing BEFORE evaluation (so checkpoint temps reflect schedule)
            sched_temp = anneal_node_temps(model, epoch, anneal_epochs,
                                           node_start, node_end)

            val = evaluate_lp(model, data['val'], edge_index, edge_types,
                              data['hr_to_tails'], data['rt_to_heads'], device)
            elapsed = time.time() - t0

            anneal_str = f"  node_sched={sched_temp:.2f}" if sched_temp is not None else "  [learnable]"
            print(f"    Ep {epoch:4d}  loss={loss:.4f}  "
                  f"val_MRR={val['MRR']:.4f}  val_H@10={val['Hits@10']:.4f}"
                  f"{anneal_str}  [{elapsed:.0f}s]")

            if is_delta:
                model.eval()
                attn_collector.start()
                gate_collector.start()
                with torch.no_grad():
                    ei = edge_index.to(device)
                    et = edge_types.to(device)
                    model.encode(ei, et)
                attn_collector.stop()
                gate_collector.stop()

                cp = {
                    'epoch': epoch,
                    'val_MRR': val['MRR'],
                    'attention': attn_collector.get_stats(),
                    'gates': gate_collector.get_stats(),
                    'temperatures': get_learned_temperatures(model),
                    'scheduled_node_temp': sched_temp,
                }
                checkpoint_stats.append(cp)

            if val['MRR'] > best_val_mrr:
                best_val_mrr = val['MRR']
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}
                evals_no_improve = 0
            else:
                evals_no_improve += 1
                if patience > 0 and evals_no_improve >= patience:
                    print(f"    Early stopping at epoch {epoch}")
                    break

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"    Training done: best_val_MRR={best_val_mrr:.4f} [{elapsed:.0f}s]")

    if is_delta:
        model.eval()
        attn_collector.start()
        gate_collector.start()
        with torch.no_grad():
            ei = edge_index.to(device)
            et = edge_types.to(device)
            model.encode(ei, et)
        attn_collector.stop()
        gate_collector.stop()

    return (model, edge_index, edge_types, checkpoint_stats,
            best_val_mrr, attn_collector, gate_collector)


def main():
    parser = argparse.ArgumentParser(
        description='Phase 50: Temperature Annealing')
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
    print("PHASE 50: TEMPERATURE ANNEALING (NODE TEMP SCHEDULE)")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, eval_every: {args.eval_every}")
    print(f"  Seed: {args.seed}")
    print(f"  Max entities: {max_ent or 'full dataset'}")

    print("\n  Loading data...")
    data = load_lp_data('fb15k-237', max_entities=max_ent)

    # ── Reference results ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  REFERENCE RESULTS (Phases 46-49, not re-run)")
    print("=" * 70)
    print("  A (all temp=1.0):        LP MRR=0.4744, 3p MRR=0.3725, dead=20/24 (83%)")
    print("  D (all temp=4.0):        LP MRR=0.4729, 3p MRR=0.4018, dead= 9/24 (38%)")
    print("  E (L0=1, L1+L2 n=2,e=6):LP MRR=0.4856, 3p MRR=0.3872, dead= 9/24 (38%)")
    print("  H (L0=4, L1+L2 n=2,e=6):LP MRR=0.4887, 3p MRR=0.3930, dead= 9/24 (38%)")

    # ── Annealing conditions ───────────────────────────────────────────
    # All start with L0=(4.0,4.0), L1+L2=(node_start, 6.0)
    # Node temps anneal; edge temps remain fully learnable from 6.0 (like E/H)
    anneal_frac_of_epochs = {
        'K_anneal_fast': (0.50, 4.0, 2.0),   # 50% of training, 4.0 → 2.0
        'L_anneal_slow': (1.00, 4.0, 2.0),   # 100% of training, 4.0 → 2.0
        'M_anneal_partial': (0.50, 4.0, 3.0), # 50% of training, 4.0 → 3.0
    }

    # Initial temperature config: all start at L0=(4,4), L1+L2=(4,6)
    # The node temp will be overridden by the annealing schedule
    init_temps = {
        0: (4.0, 4.0),   # L0: fixed at 4.0 (irrelevant, 100% dead)
        1: (4.0, 6.0),   # L1: node starts at 4.0 (annealed), edge at 6.0 (learnable)
        2: (4.0, 6.0),   # L2: same as L1
    }

    all_results = {}

    for cond_name, (frac, node_start, node_end) in anneal_frac_of_epochs.items():
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
            init_temps, anneal_epochs, node_start, node_end)

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
            'init_layer_temps': {str(k): v for k, v in init_temps.items()},
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
    # Annealing trajectory analysis
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  ANNEALING TRAJECTORY ANALYSIS")
    print(f"{'='*70}")

    for cond_name, result in all_results.items():
        cfg = result['anneal_config']
        checkpoints = result.get('checkpoint_stats', [])
        print(f"\n  {cond_name}: node {cfg['node_start']:.1f} → "
              f"{cfg['node_end']:.1f} over {cfg['anneal_epochs']} epochs")
        print(f"    {'Epoch':>6}  {'val_MRR':>8}  {'sched_node':>11}  "
              f"{'L1_node':>8}  {'L1_edge':>8}  {'L2_node':>8}  {'L2_edge':>8}")

        for cp in checkpoints:
            epoch = cp.get('epoch', 0)
            val_mrr = cp.get('val_MRR', 0)
            sched = cp.get('scheduled_node_temp')
            temps = cp.get('temperatures', {})

            l1n = np.mean(temps.get('L1_node', [0])) if temps.get('L1_node') else 0
            l1e = np.mean(temps.get('L1_edge', [0])) if temps.get('L1_edge') else 0
            l2n = np.mean(temps.get('L2_node', [0])) if temps.get('L2_node') else 0
            l2e = np.mean(temps.get('L2_edge', [0])) if temps.get('L2_edge') else 0

            sched_str = f"{sched:.2f}" if sched is not None else "learnable"
            print(f"    {epoch:6d}  {val_mrr:8.4f}  {sched_str:>11}  "
                  f"{l1n:8.3f}  {l1e:8.3f}  {l2n:8.3f}  {l2e:8.3f}")

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
        'phase50_output.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    print(f"\n{'='*70}")
    print(f"  PHASE 50 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
