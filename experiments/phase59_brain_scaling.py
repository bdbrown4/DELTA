"""Phase 59: Brain Hybrid Scaling — Medium-Scale Evaluation
=============================================================

Motivation (from Phases 55–58):
  brain_hybrid @ d=0.01 is validated on the 500-entity subset:
    mean MRR=0.4844±0.0097, H@10=0.7994±0.0058 (3 seeds, Phase 58)
    Seed=456 achieves MRR=0.4956 (brain_hybrid record)
    Density optimization CLOSED: d=0.01 is the sweet spot.

  Gap 1 (highest priority): ALL results use 494-entity subset (3.4% of
  full FB15k-237). We have never tested ANY model beyond 500 entities.

  BrainConstructor computes all N² entity pairs for edge scoring.
  N=500: 250K pairs — trivial.
  N=14,541 (full): 211M pairs — OOM on all GPUs (>50GB).
  N=2000: 4M pairs — feasible (~1GB, well within 98GB VRAM).
  N=5000: 25M pairs — tight but feasible (~6GB).

  This phase tests brain_hybrid vs delta_full at N=2000 (4× current scale)
  to characterize scaling behavior before committing to full-scale
  infrastructure changes.

Hypothesis (falsifiable):
  "brain_hybrid @ d=0.01 achieves LP MRR >= 0.40 on the 2000-entity subset
  and maintains its H@10 advantage over delta_full (H@10 delta >= +0.02)."

  Note: MRR target lowered to 0.40 because the 2000-entity subset is harder
  (less dense, more entities to rank against). delta_full's MRR on the 500
  subset is ~0.48; scaling typically causes ~10-20% MRR drop.

Design:
  2 models × 1 seed = 2 runs (quick scaling test):
    A. delta_full (baseline) @ N=2000, seed=42
    B. brain_hybrid @ d=0.01, N=2000, seed=42

  ONE primary change: scale from N=500 to N=2000 entities.
  Baseline (delta_full) needed because we have NO reference at N=2000.

  We also test brain_hybrid @ N=5000 if N=2000 succeeds (stretch goal).

  Config: 200 epochs, eval_every=30, patience=10
  Data: FB15k-237, 2000-entity dense subset

Measurements:
  LP: filtered MRR, Hits@1, H@3, H@10
  Brain-specific: constructed edges, time per epoch
  Scaling comparison: metrics at N=500 vs N=2000

Regression safety:
  LP MRR must be > 0.0 (model converges at new scale).
  Training must complete within 8 hours per run.

Scaling note:
  At N=2000, training has 62K triples. With batch_size=512, that's 123
  batches/epoch, each requiring a full 3-layer DELTA GNN encode (expensive
  edge-to-edge attention on 62K edges). This makes per-batch encoding
  ~77× slower than N=500. Solution: use --fullbatch to process all triples
  in one batch per epoch (1 GNN encode/epoch instead of 123). The scoring
  tensor [62K, 1991] ≈ 500 MB fits easily in A100 80GB VRAM.

  CRITICAL: fullbatch requires LR scaling. Going from bs=512 to bs=62K is
  a 122× batch increase. With Adam, the effective step size shrinks because
  gradients are averaged over 122× more samples. Without LR compensation,
  the model converges to a low-loss but non-discriminative solution
  (MRR ≈ 0.002 = random). Use --lr 0.01 (10× base) for fullbatch mode.
  The linear scaling rule suggests 0.12 but Adam is less sensitive.

  brain_hybrid OOMs at fullbatch on A100-80GB because brain-constructed
  edges increase total edges from ~62K to ~102K, causing edge-to-edge
  attention to try allocating 24+ GB for a single tensor. brain_hybrid
  must use mini-batch training with appropriately scaled LR.

Usage:
  # Smoke test (5 epochs, N=500)
  python experiments/phase59_brain_scaling.py --epochs 5 --max_entities 500

  # Condition A (delta_full) at N=2000: fullbatch + scaled LR
  python experiments/phase59_brain_scaling.py --epochs 200 --eval_every 30 --patience 10 --max_entities 2000 --fullbatch --lr 0.01 --conditions A

  # Condition B (brain_hybrid) at N=2000: mini-batch + scaled LR (OOMs at fullbatch)
  python experiments/phase59_brain_scaling.py --epochs 200 --eval_every 30 --patience 10 --max_entities 2000 --batch_size 4096 --lr 0.003 --conditions B
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gc
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
    create_lp_model,
    train_epoch,
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
        'name': 'delta_full',
        'desc': 'delta_full baseline @ N=2000 (no brain constructor)',
        'model_type': 'delta_full',
        'use_brain': False,
    },
    'B': {
        'name': 'brain_d001',
        'desc': 'brain_hybrid @ d=0.01, temp=1.0 (Phase 58 validated config)',
        'model_type': 'brain_hybrid',
        'use_brain': True,
        'target_density': 0.01,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def run_condition(cond_key, cond, data, args, device, seed):
    """Train + evaluate one condition on one seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if cond['use_brain']:
        model = create_brain_model(
            data['num_entities'], data['num_relations'],
            target_density=cond['target_density'])
    else:
        model = create_lp_model(
            cond['model_type'], data['num_entities'], data['num_relations'])

    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.encoder.parameters()) if model.encoder else 0

    print(f"\n  [{cond['name']}] seed={seed}, "
          f"{n_total:,} total params ({n_encoder:,} encoder), device={device}")

    edge_index, edge_types = build_train_graph_tensors(data['train'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_mrr = 0.0
    best_test = None
    best_brain_stats = {}
    evals_no_improve = 0
    t0 = time.time()

    # Determine effective batch size
    effective_bs = data['train'].shape[1] if args.fullbatch else args.batch_size
    if args.fullbatch:
        print(f"  Using full-batch training: batch_size={effective_bs} "
              f"(1 GNN encode/epoch)")

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        if cond['use_brain']:
            loss, sp_loss = train_epoch_brain(
                model, data['train'], edge_index, edge_types,
                optimizer, device, effective_bs,
                sparsity_weight=args.sparsity_weight)
        else:
            loss = train_epoch(
                model, data['train'], edge_index, edge_types,
                optimizer, device, effective_bs)
            sp_loss = 0.0
        ep_elapsed = time.time() - ep_t0

        # Per-epoch progress (lightweight, no eval)
        if epoch % args.eval_every != 0 and epoch != args.epochs:
            if epoch <= 3 or epoch % 10 == 0:
                elapsed = time.time() - t0
                print(f"    Ep {epoch:4d}  loss={loss:.4f}  [{ep_elapsed:.1f}s/ep, {elapsed:.0f}s total]")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate_lp(model, data['val'], edge_index, edge_types,
                              data['hr_to_tails'], data['rt_to_heads'], device)
            elapsed = time.time() - t0

            n_new = 0
            if hasattr(model.encoder, 'last_num_constructed_edges'):
                n_new = model.encoder.last_num_constructed_edges

            extra = f"  new_edges={n_new}  sp_loss={sp_loss:.4f}" if cond['use_brain'] else ""
            print(f"    Ep {epoch:4d}  loss={loss:.4f}  "
                  f"val_MRR={val['MRR']:.4f}  val_H@10={val['Hits@10']:.4f}"
                  f"{extra}  [{elapsed:.0f}s]")

            if val['MRR'] > best_val_mrr:
                best_val_mrr = val['MRR']
                evals_no_improve = 0
                best_test = evaluate_lp(
                    model, data['test'], edge_index, edge_types,
                    data['hr_to_tails'], data['rt_to_heads'], device)
                best_brain_stats = {
                    'constructed_edges': n_new,
                    'sparsity_loss': sp_loss,
                } if cond['use_brain'] else {}
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
        'model': cond['model_type'],
        'condition': f"{cond_key}_{cond['name']}",
        'seed': seed,
        'max_entities': args.max_entities,
        'params_total': n_total,
        'params_encoder': n_encoder,
        'best_val_MRR': best_val_mrr,
        'test_MRR': best_test['MRR'],
        'test_Hits@1': best_test['Hits@1'],
        'test_Hits@3': best_test['Hits@3'],
        'test_Hits@10': best_test['Hits@10'],
        'time_s': elapsed,
        **best_brain_stats,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Phase 59: Brain Hybrid Scaling — Medium-Scale Evaluation')
    parser.add_argument('--seeds', type=str, default='42',
                        help='Comma-separated seeds (default: 42)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_every', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--sparsity_weight', type=float, default=0.01)
    parser.add_argument('--max_entities', type=int, default=2000,
                        help='Max entities for dense subset (default: 2000)')
    parser.add_argument('--conditions', type=str, default=None,
                        help='Comma-separated condition letters to run (e.g. A,B). Default: all')
    parser.add_argument('--fullbatch', action='store_true',
                        help='Use full-batch training (1 GNN encode/epoch). Essential for N>=2000.')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    run_conditions = set(args.conditions.split(',')) if args.conditions else set(CONDITIONS.keys())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("PHASE 59: BRAIN HYBRID SCALING — MEDIUM-SCALE EVALUATION")
    print("=" * 80)
    print(f"  Device: {device}")
    print(f"  Seeds: {seeds}")
    print(f"  Max entities: {args.max_entities}")
    print(f"  Epochs: {args.epochs}, eval_every: {args.eval_every}, patience: {args.patience}")
    print(f"  Running conditions: {sorted(run_conditions)}")

    data = load_lp_data('fb15k-237', max_entities=args.max_entities)
    print(f"  Loaded fb15k-237 for link prediction:")
    print(f"    {data['num_entities']} entities, {data['num_relations']} relations")
    print(f"    {data['train'].shape[1]} train / {data['val'].shape[1]} val / "
          f"{data['test'].shape[1]} test")

    # BrainConstructor memory estimate
    N = data['num_entities']
    mem_gb = N * N * 128 * 4 / (1024**3)  # N² pairs × 2*d_node × float32
    print(f"    BrainConstructor estimate: {N}² = {N*N:,} pairs, ~{mem_gb:.1f} GB")

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

        # Free GPU memory between conditions to avoid OOM on subsequent runs
        if device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            free_mb = torch.cuda.mem_get_info()[0] / (1024**2)
            print(f"  GPU cleanup: {free_mb:.0f} MB free")

    # ═══════════════════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print(f"PHASE 59: BRAIN HYBRID SCALING — RESULTS (N={data['num_entities']})")
    print(f"{'=' * 80}")
    print(f"  Data: FB15k-237, {data['num_entities']} entities, "
          f"{data['num_relations']} relations, {data['train'].shape[1]} train")
    print(f"  Epochs: {args.epochs}, seeds: {seeds}")

    header = (f"{'Condition':<20s} {'Seed':>5s} "
              f"{'MRR':>8s} {'H@1':>8s} {'H@3':>8s} {'H@10':>8s} "
              f"{'Edges':>7s} {'Time':>7s}")
    print(f"\n{header}")
    print("-" * len(header))

    for r in all_results:
        cond_name = r.get('condition', '?')
        edges = r.get('constructed_edges', 0)
        time_s = r.get('time_s', 0)

        print(f"{cond_name:<20s} {r['seed']:>5d} "
              f"{r['test_MRR']:>8.4f} {r['test_Hits@1']:>8.4f} "
              f"{r['test_Hits@3']:>8.4f} {r['test_Hits@10']:>8.4f} "
              f"{edges:>7d} {time_s:>6.0f}s")

    print("-" * len(header))

    # Reference comparison to N=500 results
    print(f"\n  REFERENCE (N=500, Phase 58):")
    print(f"    delta_full:       MRR~0.4796, H@10~0.7603")
    print(f"    brain_hybrid d01: MRR=0.4844±0.0097, H@10=0.7994±0.0058")
    print(f"    brain_hybrid best: MRR=0.4956 (seed=456)")

    # Hypothesis evaluation
    cond_a = [r for r in all_results if r['condition'].startswith('A_')]
    cond_b = [r for r in all_results if r['condition'].startswith('B_')]

    print(f"\n  HYPOTHESIS EVALUATION:")
    if cond_a and cond_b:
        a_mrr = np.mean([r['test_MRR'] for r in cond_a])
        b_mrr = np.mean([r['test_MRR'] for r in cond_b])
        a_h10 = np.mean([r['test_Hits@10'] for r in cond_a])
        b_h10 = np.mean([r['test_Hits@10'] for r in cond_b])

        print(f"    delta_full  MRR={a_mrr:.4f}, H@10={a_h10:.4f}")
        print(f"    brain_hybrid MRR={b_mrr:.4f}, H@10={b_h10:.4f}")
        h10_delta = b_h10 - a_h10
        mrr_delta = b_mrr - a_mrr
        print(f"    H@10 delta: {h10_delta:+.4f} ({'≥+0.02 CONFIRMED' if h10_delta >= 0.02 else '<+0.02 REJECTED'})")
        print(f"    MRR delta: {mrr_delta:+.4f}")

        if b_mrr >= 0.40:
            print(f"    ✓ brain_hybrid MRR >= 0.40 CONFIRMED at N={args.max_entities}")
        else:
            print(f"    ✗ brain_hybrid MRR < 0.40 REJECTED at N={args.max_entities}")

    print(f"\n  Phase 59 complete.")

    # Save results
    output = {
        'phase': 59,
        'title': 'Brain Hybrid Scaling — Medium-Scale Evaluation',
        'hypothesis': 'brain_hybrid @ d=0.01 achieves LP MRR >= 0.40 at N=2000 and maintains H@10 advantage >= +0.02 over delta_full',
        'data': {
            'entities': data['num_entities'],
            'relations': data['num_relations'],
            'train_triples': data['train'].shape[1],
            'max_entities_arg': args.max_entities,
        },
        'config': {
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'fullbatch': args.fullbatch,
            'effective_batch_size': data['train'].shape[1] if args.fullbatch else args.batch_size,
            'eval_every': args.eval_every,
            'patience': args.patience,
            'seeds': seeds,
            'sparsity_weight': args.sparsity_weight,
        },
        'results': all_results,
    }

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'phase59_output.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
