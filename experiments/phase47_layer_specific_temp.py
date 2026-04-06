"""Phase 47: Layer-Specific Temperature Initialization
======================================================

Motivation (from Phase 46):
  Temperature sharpening activates DELTA-Full's excess capacity on 3p
  (MRR +0.029) but the effect is blunt — uniform temp=4.0 applied to
  all layers and attention types.  Phase 46 discovered that:

  1. Layer 0 is always 100% dead regardless of temperature
  2. Edge temps drift UP (wants sharper), node temps drift DOWN (softer)
  3. DELTA-Full Layer 2 comes alive with temp=4.0 (0% dead from ep 100)

  This suggests targeted temperature initialization:
  - Layer 0 doesn't need high temperature (it averages anyway)
  - Upper layers (L1, L2) benefit from temperature sharpening
  - Edge attention benefits more from sharpness than node attention

Hypothesis (falsifiable):
  "Condition B (L0 temp=1.0, L1+L2 temp=4.0) or Condition C (node temp=1.0,
  edge temp=4.0) achieves DELTA-Full temp=4.0's 3p improvement without LP
  degradation.  Expected: LP MRR >= 0.4744 (Phase 46 full temp=1.0) AND
  3p MRR >= 0.4018 (Phase 46 full temp=4.0)."

Design:
  4 conditions (all delta_full):
    A. all temp=1.0          (control — reuse Phase 46 data)
    B. L0 temp=1.0, L1+L2 temp=4.0   (targeted layer sharpening)
    C. node temp=1.0, edge temp=4.0   (edge-only sharpening)
    D. all temp=4.0          (control — reuse Phase 46 data)

  Only B and C are new runs.  A and D reference Phase 46 results.

Measurements:
  Same as Phase 46: per-head entropy, dead heads, learned temperatures,
  LP MRR/H@10, multi-hop 1p-5p MRR.

Regression safety:
  LP MRR must stay >= 0.47 (Phase 46 DELTA-Full temp=1.0 baseline).
  3p MRR must stay >= 0.35 (regression safety floor from Phase 45).

Usage:
  # Smoke test (5 epochs)
  python experiments/phase47_layer_specific_temp.py --epochs 5

  # Full run
  python experiments/phase47_layer_specific_temp.py --epochs 500 --eval_every 25 --patience 10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import time
import json
import numpy as np
import torch

from delta.model import DELTAModel, DELTALayer

from experiments.phase46c_link_prediction import (
    load_lp_data,
    create_lp_model,
    LinkPredictionModel,
)

# Reuse all instrumentation from Phase 46
from experiments.phase46_capacity_signal import (
    AttentionCollector,
    GateCollector,
    get_learned_temperatures,
    train_instrumented,
    print_attention_report,
    print_temperature_report,
    print_cross_depth_report,
    print_training_dynamics,
    cross_depth_analysis,
    _serialize_stats,
    _serialize_checkpoints,
)

from experiments.phase46c_link_prediction import (
    build_train_graph_tensors,
    evaluate_lp,
)


# ═══════════════════════════════════════════════════════════════════════════
# Temperature manipulation helpers
# ═══════════════════════════════════════════════════════════════════════════

def set_layer_temps(model, layer_temps):
    """Override per-layer temperature initialization after model creation.

    Args:
        model: LinkPredictionModel with a DELTAModel encoder
        layer_temps: dict mapping layer_idx -> (node_temp, edge_temp)
            e.g. {0: (1.0, 1.0), 1: (1.0, 4.0), 2: (1.0, 4.0)}
    """
    encoder = model.encoder
    assert isinstance(encoder, DELTAModel), "Only works with DELTAModel encoder"

    for layer_idx, (node_temp, edge_temp) in layer_temps.items():
        layer = encoder.layers[layer_idx]
        with torch.no_grad():
            layer.dual_attn.node_attn._log_temp.fill_(math.log(node_temp))
            layer.dual_attn.edge_attn._log_temp.fill_(math.log(edge_temp))
            # Also set global_node_attn if it exists
            if hasattr(layer, 'global_node_attn'):
                layer.global_node_attn._log_temp.fill_(math.log(node_temp))


def describe_temp_config(layer_temps):
    """Human-readable description of a temperature configuration."""
    parts = []
    for li in sorted(layer_temps.keys()):
        nt, et = layer_temps[li]
        parts.append(f"L{li}(node={nt:.1f},edge={et:.1f})")
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Custom training with temperature override
# ═══════════════════════════════════════════════════════════════════════════

def train_with_temp_override(model_type, data, epochs, lr, device, batch_size,
                              seed, eval_every, patience, layer_temps):
    """Create model, override temperatures, then train with instrumentation.

    Like train_instrumented but with a temperature override step.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model with default temp=1.0
    model = create_lp_model(model_type,
                            data['num_entities'], data['num_relations'],
                            init_temp=1.0)

    # Override temperatures per layer
    set_layer_temps(model, layer_temps)

    # Verify the override
    temps = get_learned_temperatures(model)
    print(f"\n  Temperature config: {describe_temp_config(layer_temps)}")
    print(f"  After override: {temps}")

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = (sum(p.numel() for p in model.encoder.parameters())
             if model.encoder is not None else 0)
    print(f"\n  [{model_type}] seed={seed}, {n_params:,} params "
          f"({n_enc:,} encoder), device={device}")

    # Set up instrumentation
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

    from experiments.phase46c_link_prediction import train_epoch

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data['train'], edge_index, edge_types,
                           optimizer, device, batch_size)

        if epoch % eval_every == 0 or epoch == epochs:
            val = evaluate_lp(model, data['val'], edge_index, edge_types,
                              data['hr_to_tails'], data['rt_to_heads'], device)
            elapsed = time.time() - t0
            print(f"    Ep {epoch:4d}  loss={loss:.4f}  "
                  f"val_MRR={val['MRR']:.4f}  val_H@10={val['Hits@10']:.4f}  "
                  f"[{elapsed:.0f}s]")

            # Collect attention/gate stats
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

    # Final instrumentation pass with best model
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


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Phase 47: Layer-Specific Temperature Initialization')
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
    print("PHASE 47: LAYER-SPECIFIC TEMPERATURE INITIALIZATION")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, eval_every: {args.eval_every}")
    print(f"  Seed: {args.seed}")
    print(f"  Max entities: {max_ent or 'full dataset'}")

    # Load data
    print("\n  Loading data...")
    data = load_lp_data('fb15k-237', max_entities=max_ent)

    # ── Phase 46 reference results (A and D) ───────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 46 REFERENCE RESULTS (not re-run)")
    print("=" * 70)
    print("  Condition A (all temp=1.0): LP MRR=0.4744, 3p MRR=0.3725, dead=20/24 (83%)")
    print("  Condition D (all temp=4.0): LP MRR=0.4729, 3p MRR=0.4018, dead=9/24  (38%)")

    # ── New conditions: B and C ────────────────────────────────────────
    # DELTA-Full has 3 layers (L0, L1, L2)
    conditions = {
        'B_layer_sharp': {
            # L0 soft (it's always dead anyway), L1+L2 sharp
            0: (1.0, 1.0),
            1: (4.0, 4.0),
            2: (4.0, 4.0),
        },
        'C_edge_sharp': {
            # Node attention stays soft, edge attention gets sharp
            0: (1.0, 4.0),
            1: (1.0, 4.0),
            2: (1.0, 4.0),
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
    # Comparative analysis: all 4 conditions
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  COMPARATIVE ANALYSIS: ALL 4 CONDITIONS")
    print(f"{'='*70}")

    # Phase 46 reference data embedded
    p46_ref = {
        'A_all_temp1': {
            'lp_mrr': 0.4744, 'lp_h10': 0.7860, 'best_val': 0.5030,
            '3p_mrr': 0.3725, 'dead': '20/24 (83%)',
            'description': 'All temp=1.0 (Phase 46 control)',
        },
        'D_all_temp4': {
            'lp_mrr': 0.4729, 'lp_h10': 0.7901, 'best_val': 0.5106,
            '3p_mrr': 0.4018, 'dead': '9/24 (38%)',
            'description': 'All temp=4.0 (Phase 46 treatment)',
        },
    }

    print(f"\n  {'Condition':<20} {'LP MRR':>8} {'LP H@10':>8} {'3p MRR':>8} {'Dead Heads':>12} {'Best Val':>10}")
    print(f"  {'-'*76}")

    # Print Phase 46 references
    for ref_name, ref in p46_ref.items():
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

        # Target: LP MRR >= 0.4744 (Phase 46 full temp=1.0) AND 3p MRR >= 0.4018
        lp_pass = lp_mrr >= 0.4744
        mrr3p_pass = mrr_3p >= 0.4018
        regression_pass = mrr_3p >= 0.35

        print(f"\n  {cond_name}:")
        print(f"    LP MRR = {lp_mrr:.4f}  {'PASS' if lp_pass else 'FAIL'} (>= 0.4744)")
        print(f"    3p MRR = {mrr_3p:.4f}  {'PASS' if mrr3p_pass else 'FAIL'} (>= 0.4018)")
        print(f"    Regression safety = {'PASS' if regression_pass else 'FAIL'} (3p >= 0.35)")

        if lp_pass and mrr3p_pass:
            print(f"    >>> HYPOTHESIS CONFIRMED: Targeted temperature beats uniform <<<")
        elif regression_pass:
            print(f"    >>> PARTIAL: Regression-safe but did not beat Phase 46 best <<<")
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
                               'phase47_output.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    print(f"\n{'='*70}")
    print(f"  PHASE 47 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
