"""Phase 58: Multi-seed Brain Density Validation
=================================================

Motivation (from Phases 56-57):
  Phase 57 showed brain_hybrid @ d=0.01 exceeds 0.480 MRR:
    A (baseline, 200ep): MRR=0.4808, H@10=0.8076  (seed=42)
    B (K-anneal):        MRR=0.4818, H@10=0.7613  (seed=42)
  Temperature annealing provides no benefit — baseline is optimal.

  Phase 56 showed d=0.01 strictly dominates d=0.02:
    d=0.01: MRR=0.4794, H@10=0.8076, 2435 edges
    d=0.02: MRR=0.4678, H@10=0.7500, 4870 edges

  ALL results are single-seed (42). Phase 53 showed single-seed conclusions
  are unreliable (K's 3p=0.4148 was a single-seed outlier, mean=0.3699).
  LP MRR IS robust across seeds for delta_full (K=0.4832±0.0052), but
  brain_hybrid multi-seed validation has never been done.

  Open question: "Does density=0.005 continue the improvement trend?"

Hypothesis (falsifiable):
  "brain_hybrid @ d=0.01 achieves mean LP MRR >= 0.480 across 3 seeds
  (statistically robust), and d=0.005 achieves mean LP MRR >= d=0.01's mean
  (continuing the density improvement trend)."

Design:
  2 densities × 3 seeds = 6 runs:
    A. brain_hybrid @ d=0.01, seeds=[42, 123, 456]  — multi-seed validation
    B. brain_hybrid @ d=0.005, seeds=[42, 123, 456] — density exploration

  ONE primary change: test density=0.005 as the next point on the density curve.
  Multi-seed methodology validates both densities simultaneously.

  Config: 200 epochs, eval_every=30, patience=10, temp=1.0, no annealing
  (Phase 57 established baseline as optimal for brain_hybrid)

  Data: FB15k-237, 500-entity dense subset (same as Phases 46-57)

Measurements:
  LP: filtered MRR, Hits@1, H@3, H@10 (per seed + mean±std)
  Brain-specific: constructed edges, sparsity loss

Regression safety:
  LP MRR must not regress below 0.45 (brain_hybrid floor from Phase 55).

Usage:
  # Smoke test (5 epochs, 1 seed)
  python experiments/phase58_multiseed_density.py --epochs 5 --seeds 42

  # Full run
  python experiments/phase58_multiseed_density.py --epochs 200 --eval_every 30 --patience 10 --seeds 42,123,456

  # Run specific density only
  python experiments/phase58_multiseed_density.py --epochs 200 --seeds 42,123,456 --conditions A
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np
import torch

from delta.brain import BrainEncoder

from experiments.phase46c_link_prediction import (
    load_lp_data,
    build_train_graph_tensors,
    evaluate_lp,
    LinkPredictionModel,
)

from experiments.phase55_brain_port import (
    train_epoch_brain,
    build_train_graph_tensors,
)


# ═══════════════════════════════════════════════════════════════════════════
# Model creation
# ═══════════════════════════════════════════════════════════════════════════

def create_brain_model(num_entities, num_relations, d_node=64, d_edge=32,
                       target_density=0.01):
    """Create brain_hybrid model with baseline temp=1.0 (Phase 57 optimal)."""
    enc = BrainEncoder(
        d_node=d_node, d_edge=d_edge,
        bootstrap_layers=1, delta_layers=2, num_heads=4,
        target_density=target_density, hybrid=True, init_temp=1.0)
    return LinkPredictionModel(enc, num_entities, num_relations, d_node, d_edge)


# ═══════════════════════════════════════════════════════════════════════════
# Conditions
# ═══════════════════════════════════════════════════════════════════════════

CONDITIONS = {
    'A': {
        'name': 'brain_d001',
        'desc': 'brain_hybrid @ d=0.01, temp=1.0, no anneal (P57 optimal)',
        'target_density': 0.01,
    },
    'B': {
        'name': 'brain_d0005',
        'desc': 'brain_hybrid @ d=0.005, temp=1.0, no anneal (sparser construction)',
        'target_density': 0.005,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Training loop (no annealing — Phase 57 established baseline as optimal)
# ═══════════════════════════════════════════════════════════════════════════

def run_condition(cond_key, cond, data, args, device, seed):
    """Train + evaluate one condition on one seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    density = cond['target_density']
    model = create_brain_model(
        data['num_entities'], data['num_relations'],
        target_density=density)
    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.encoder.parameters())

    print(f"\n  [{cond['name']}] seed={seed}, d={density}, "
          f"{n_total:,} total params ({n_encoder:,} encoder), device={device}")

    edge_index, edge_types = build_train_graph_tensors(data['train'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_mrr = 0.0
    best_test = None
    best_brain_stats = {}
    evals_no_improve = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        loss, sp_loss = train_epoch_brain(
            model, data['train'], edge_index, edge_types,
            optimizer, device, args.batch_size,
            sparsity_weight=args.sparsity_weight)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate_lp(model, data['val'], edge_index, edge_types,
                              data['hr_to_tails'], data['rt_to_heads'], device)
            elapsed = time.time() - t0

            n_new = 0
            if hasattr(model.encoder, 'last_num_constructed_edges'):
                n_new = model.encoder.last_num_constructed_edges

            print(f"    Ep {epoch:4d}  loss={loss:.4f}  "
                  f"val_MRR={val['MRR']:.4f}  val_H@10={val['Hits@10']:.4f}"
                  f"  new_edges={n_new}  sp_loss={sp_loss:.4f}  [{elapsed:.0f}s]")

            if val['MRR'] > best_val_mrr:
                best_val_mrr = val['MRR']
                evals_no_improve = 0
                best_test = evaluate_lp(
                    model, data['test'], edge_index, edge_types,
                    data['hr_to_tails'], data['rt_to_heads'], device)
                best_brain_stats = {
                    'constructed_edges': n_new,
                    'sparsity_loss': sp_loss,
                }
            else:
                evals_no_improve += 1
                if args.patience > 0 and evals_no_improve >= args.patience:
                    print(f"    Early stopping at epoch {epoch}")
                    break

    elapsed = time.time() - t0
    if best_test is None:
        best_test = {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0}

    print(f"    Done: val_MRR={best_val_mrr:.4f}  "
          f"test_MRR={best_test['MRR']:.4f}  "
          f"test_H@10={best_test['Hits@10']:.4f}  [{elapsed:.0f}s]")

    return {
        'model': 'brain_hybrid',
        'condition': f"{cond_key}_{cond['name']}",
        'seed': seed,
        'params_total': n_total,
        'params_encoder': n_encoder,
        'best_val_MRR': best_val_mrr,
        'test_MRR': best_test['MRR'],
        'test_Hits@1': best_test['Hits@1'],
        'test_Hits@3': best_test['Hits@3'],
        'test_Hits@10': best_test['Hits@10'],
        'time_s': elapsed,
        'target_density': density,
        **best_brain_stats,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Phase 58: Multi-seed Brain Density Validation')
    parser.add_argument('--seeds', type=str, default='42,123,456',
                        help='Comma-separated seeds (default: 42,123,456)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_every', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--sparsity_weight', type=float, default=0.01)
    parser.add_argument('--max_entities', type=int, default=500,
                        help='Max entities for dense subset (default: 500)')
    parser.add_argument('--conditions', type=str, default=None,
                        help='Comma-separated condition letters to run (e.g. A,B). Default: all')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    run_conditions = set(args.conditions.split(',')) if args.conditions else set(CONDITIONS.keys())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("PHASE 58: MULTI-SEED BRAIN DENSITY VALIDATION")
    print("=" * 80)
    print(f"  Device: {device}")
    print(f"  Seeds: {seeds}")
    print(f"  Epochs: {args.epochs}, eval_every: {args.eval_every}, patience: {args.patience}")
    print(f"  Running conditions: {sorted(run_conditions)}")

    data = load_lp_data('fb15k-237', max_entities=args.max_entities)
    print(f"  Loaded fb15k-237 for link prediction:")
    print(f"    {data['num_entities']} entities, {data['num_relations']} relations")
    print(f"    {data['train'].shape[1]} train / {data['val'].shape[1]} val / "
          f"{data['test'].shape[1]} test")

    all_results = []

    for cond_key in sorted(CONDITIONS.keys()):
        cond = CONDITIONS[cond_key]

        if cond_key not in run_conditions:
            print(f"\n  SKIPPING Condition {cond_key} (not in {sorted(run_conditions)})\n")
            continue

        print(f"\n{'=' * 60}")
        print(f"  CONDITION {cond_key}: {cond['desc']}")
        print(f"{'=' * 60}")

        for seed in seeds:
            result = run_condition(cond_key, cond, data, args, device, seed)
            all_results.append(result)

    # ═══════════════════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print(f"PHASE 58: MULTI-SEED BRAIN DENSITY VALIDATION — RESULTS")
    print(f"{'=' * 80}")
    print(f"  Data: FB15k-237, {data['num_entities']} entities, "
          f"{data['num_relations']} relations, {data['train'].shape[1]} train")
    print(f"  Epochs: {args.epochs}, seeds: {seeds}")

    # Per-run table
    header = (f"{'Condition':<20s} {'Seed':>5s} {'Density':>8s} "
              f"{'MRR':>8s} {'H@1':>8s} {'H@3':>8s} {'H@10':>8s} "
              f"{'Edges':>7s} {'Time':>7s}")
    print(f"\n{header}")
    print("-" * len(header))

    for r in all_results:
        cond_name = r.get('condition', '?')
        density = r.get('target_density', 0)
        edges = r.get('constructed_edges', 0)
        time_s = r.get('time_s', 0)

        print(f"{cond_name:<20s} {r['seed']:>5d} {density:>8.3f} "
              f"{r['test_MRR']:>8.4f} {r['test_Hits@1']:>8.4f} "
              f"{r['test_Hits@3']:>8.4f} {r['test_Hits@10']:>8.4f} "
              f"{edges:>7d} {time_s:>6.0f}s")

    print("-" * len(header))

    # Per-condition aggregation (mean ± std)
    print(f"\n  AGGREGATED (mean ± std):")
    print(f"  {'Condition':<20s} {'Density':>8s} {'MRR':>14s} {'H@1':>14s} "
          f"{'H@3':>14s} {'H@10':>14s}")
    print(f"  " + "-" * 80)

    for cond_key in sorted(CONDITIONS.keys()):
        cond = CONDITIONS[cond_key]
        cond_results = [r for r in all_results
                        if r['condition'].startswith(f"{cond_key}_")]
        if not cond_results:
            continue

        mrrs = [r['test_MRR'] for r in cond_results]
        h1s = [r['test_Hits@1'] for r in cond_results]
        h3s = [r['test_Hits@3'] for r in cond_results]
        h10s = [r['test_Hits@10'] for r in cond_results]

        def fmt(vals):
            if len(vals) == 1:
                return f"{vals[0]:.4f}"
            return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"

        print(f"  {cond_key}_{cond['name']:<17s} {cond['target_density']:>8.3f} "
              f"{fmt(mrrs):>14s} {fmt(h1s):>14s} "
              f"{fmt(h3s):>14s} {fmt(h10s):>14s}")

    # Hypothesis evaluation
    cond_a_results = [r for r in all_results if r['condition'].startswith('A_')]
    cond_b_results = [r for r in all_results if r['condition'].startswith('B_')]

    print(f"\n  HYPOTHESIS EVALUATION:")
    if cond_a_results:
        a_mrrs = [r['test_MRR'] for r in cond_a_results]
        a_mean = np.mean(a_mrrs)
        a_std = np.std(a_mrrs)
        a_min = np.min(a_mrrs)
        print(f"    A (d=0.01): mean MRR={a_mean:.4f}±{a_std:.4f}, min={a_min:.4f}")
        if a_mean >= 0.480:
            print(f"    ✓ d=0.01 mean MRR >= 0.480 CONFIRMED (robust across {len(a_mrrs)} seeds)")
        else:
            print(f"    ✗ d=0.01 mean MRR < 0.480 REJECTED (gap={0.480-a_mean:.4f})")

    if cond_b_results:
        b_mrrs = [r['test_MRR'] for r in cond_b_results]
        b_mean = np.mean(b_mrrs)
        b_std = np.std(b_mrrs)
        b_min = np.min(b_mrrs)
        print(f"    B (d=0.005): mean MRR={b_mean:.4f}±{b_std:.4f}, min={b_min:.4f}")
        if cond_a_results:
            if b_mean > a_mean:
                print(f"    ✓ d=0.005 > d=0.01 CONFIRMED (delta=+{b_mean-a_mean:.4f})")
            else:
                print(f"    ✗ d=0.005 <= d=0.01 REJECTED (delta={b_mean-a_mean:.4f})")

    print(f"\n  Phase 58 complete.")

    # Save results
    output = {
        'phase': 58,
        'title': 'Multi-seed Brain Density Validation',
        'hypothesis': 'brain_hybrid @ d=0.01 achieves mean LP MRR >= 0.480 across 3 seeds; d=0.005 continues improvement trend',
        'data': {
            'entities': data['num_entities'],
            'relations': data['num_relations'],
            'train_triples': data['train'].shape[1],
        },
        'config': {
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'eval_every': args.eval_every,
            'patience': args.patience,
            'seeds': seeds,
            'sparsity_weight': args.sparsity_weight,
        },
        'results': all_results,
    }

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'phase58_output.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
