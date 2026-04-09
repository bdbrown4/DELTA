"""Phase 55: Brain Architecture Port — Differentiable Graph Construction for LP
=============================================================================

Motivation (from Phase 54 closure + Brain architecture):
  The DELTA-Full multi-hop/temperature investigation is CLOSED (Phases 46-54).
  Temperature reliably improves LP MRR but has no statistically supported
  effect on multi-hop reasoning depth.

  Three agent analyses (GPT, Gemini Pro, Claude Opus 4.6) converged on the
  next direction: the Brain architecture — porting Phase 46b's self-bootstrap
  with differentiable graph construction into the core package.

  The Brain vision (https://bdbrown4.github.io/DELTA/the-brain/):
    "DELTA constructs its own relational graphs and reasons over them,
    without relying on pre-defined topology or transformer scaffolding."

  Phase 46b proved the concept on a toy task: SelfBootstrapHybrid at +57%
  over FixedChain. Phase 40 achieved MRR 0.5089 on FB15k-237 (within 0.004
  of GraphGPS). But those results used an experiment-only implementation.

  This phase ports the architecture to delta/brain.py and validates on
  FB15k-237 LP using the standard evaluation infrastructure.

Hypothesis (falsifiable):
  "BrainEncoder (Stage 1: bootstrap DELTALayer on KG → Stage 2: BrainConstructor
  learns new edges via Gumbel-sigmoid → Stage 3: DELTALayers on augmented graph)
  achieves LP MRR >= 0.475 on FB15k-237, demonstrating that learned graph
  augmentation adds value for link prediction."

  If LP MRR >= 0.490 (temperature-tuned DELTA-Full range), the Brain
  architecture shows clear promise for replacing static KG topology.

Design:
  4 conditions × 3 seeds = 12 runs:
    A. delta_full         — DELTA-Full 3 layers (Phase 46+ baseline)
    B. self_bootstrap     — SelfBootstrapDELTAEncoder 1+2 layers (phase46c)
    C. brain_hybrid       — BrainEncoder 1+2 layers + constructor, keeps KG edges
    D. brain_pure         — BrainEncoder 1+2 layers + constructor, learned edges only

  ONE primary change: New delta/brain.py (BrainConstructor + BrainEncoder)

  Seeds: 42, 123, 456
  Data: FB15k-237, 500-entity dense subset (same as Phases 46-54)

Measurements:
  LP: filtered MRR, Hits@1, H@3, H@10
  Brain-specific: number of constructed edges, sparsity loss, actual density

Regression safety:
  LP MRR must not regress below 0.40 for any condition (below distmult baseline).

Usage:
  # Smoke test (5 epochs, 1 seed) — ~2 min CPU
  python experiments/phase55_brain_port.py --epochs 5 --seeds 42

  # Full run (500 epochs, 3 seeds)
  python experiments/phase55_brain_port.py --epochs 500 --eval_every 25 --patience 10 --seeds 42,123,456
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.model import DELTAModel
from delta.graph import DeltaGraph
from delta.brain import BrainEncoder

from experiments.phase46c_link_prediction import (
    load_lp_data,
    build_train_graph_tensors,
    evaluate_lp,
    create_lp_model,
    train_epoch,
    LinkPredictionModel,
    SelfBootstrapDELTAEncoder,
)


# ═══════════════════════════════════════════════════════════════════════════
# Model Factories
# ═══════════════════════════════════════════════════════════════════════════

PHASE55_MODELS = ['delta_full', 'self_bootstrap', 'brain_hybrid', 'brain_pure']


def create_phase55_model(model_type, num_entities, num_relations,
                          d_node=64, d_edge=32, target_density=0.05):
    """Create models for Phase 55 comparison."""

    if model_type == 'delta_full':
        enc = DELTAModel(d_node=d_node, d_edge=d_edge,
                         num_layers=3, num_heads=4, init_temp=1.0)
        return LinkPredictionModel(enc, num_entities, num_relations,
                                   d_node, d_edge)

    elif model_type == 'self_bootstrap':
        enc = SelfBootstrapDELTAEncoder(
            d_node=d_node, d_edge=d_edge,
            bootstrap_layers=1, delta_layers=2, num_heads=4)
        return LinkPredictionModel(enc, num_entities, num_relations,
                                   d_node, d_edge)

    elif model_type == 'brain_hybrid':
        enc = BrainEncoder(
            d_node=d_node, d_edge=d_edge,
            bootstrap_layers=1, delta_layers=2, num_heads=4,
            target_density=target_density, hybrid=True, init_temp=1.0)
        return LinkPredictionModel(enc, num_entities, num_relations,
                                   d_node, d_edge)

    elif model_type == 'brain_pure':
        enc = BrainEncoder(
            d_node=d_node, d_edge=d_edge,
            bootstrap_layers=1, delta_layers=2, num_heads=4,
            target_density=target_density, hybrid=False, init_temp=1.0)
        return LinkPredictionModel(enc, num_entities, num_relations,
                                   d_node, d_edge)

    else:
        raise ValueError(f"Unknown model: {model_type}. Choose from {PHASE55_MODELS}")


# ═══════════════════════════════════════════════════════════════════════════
# Training — standard LP + sparsity loss for Brain models
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch_brain(model, train_triples, edge_index, edge_types,
                      optimizer, device, batch_size=512, label_smoothing=0.1,
                      sparsity_weight=0.01):
    """Train one epoch for Brain models — per-batch encoding with sparsity loss.

    Same as phase46c train_epoch but adds sparsity regularization
    from BrainConstructor when present. Per-batch encoding gives the
    constructor fresh Gumbel noise each batch for exploration.
    """
    model.train()
    n = train_triples.shape[1]
    perm = torch.randperm(n)
    total_loss = 0.0
    total_sparsity = 0.0
    num_batches = 0

    ei = edge_index.to(device)
    et = edge_types.to(device)

    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        h = train_triples[0, idx].to(device)
        r = train_triples[1, idx].to(device)
        t = train_triples[2, idx].to(device)
        B = h.shape[0]
        N = model.num_entities

        # Encode (gradients flow through GNN + constructor)
        node_feats = model.encode(ei, et)

        # --- Tail prediction ---
        scores_t = model.score_all_tails(node_feats, h, r)
        targets_t = torch.zeros(B, N, device=device)
        targets_t[torch.arange(B, device=device), t] = 1.0
        if label_smoothing > 0:
            targets_t = targets_t * (1 - label_smoothing) + label_smoothing / N
        loss_t = F.binary_cross_entropy_with_logits(scores_t, targets_t)

        # --- Head prediction ---
        scores_h = model.score_all_heads(node_feats, r, t)
        targets_h = torch.zeros(B, N, device=device)
        targets_h[torch.arange(B, device=device), h] = 1.0
        if label_smoothing > 0:
            targets_h = targets_h * (1 - label_smoothing) + label_smoothing / N
        loss_h = F.binary_cross_entropy_with_logits(scores_h, targets_h)

        lp_loss = (loss_t + loss_h) / 2

        # Add sparsity regularization for Brain models
        sp_loss = torch.tensor(0.0, device=device)
        if hasattr(model, 'encoder') and model.encoder is not None:
            if hasattr(model.encoder, 'last_sparsity_loss'):
                sp_loss = model.encoder.last_sparsity_loss

        loss = lp_loss + sparsity_weight * sp_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += lp_loss.item()
        total_sparsity += sp_loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_sparsity = total_sparsity / max(num_batches, 1)
    return avg_loss, avg_sparsity


# ═══════════════════════════════════════════════════════════════════════════
# Single model run
# ═══════════════════════════════════════════════════════════════════════════

def run_single(model_type, data, args, device, seed):
    """Train + evaluate one model on one seed. Returns metrics dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = create_phase55_model(
        model_type,
        data['num_entities'], data['num_relations'],
        target_density=args.target_density)
    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_encoder = (sum(p.numel() for p in model.encoder.parameters())
                 if model.encoder is not None else 0)
    is_brain = model_type.startswith('brain')

    print(f"\n  [{model_type}] seed={seed}, "
          f"{n_total:,} total params ({n_encoder:,} encoder), device={device}")

    edge_index, edge_types = build_train_graph_tensors(data['train'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_mrr = 0.0
    best_test = None
    best_brain_stats = {}
    evals_no_improve = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        if is_brain:
            loss, sp_loss = train_epoch_brain(
                model, data['train'], edge_index, edge_types,
                optimizer, device, args.batch_size,
                sparsity_weight=args.sparsity_weight)
        else:
            loss = train_epoch(
                model, data['train'], edge_index, edge_types,
                optimizer, device, args.batch_size)
            sp_loss = 0.0

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate_lp(model, data['val'], edge_index, edge_types,
                              data['hr_to_tails'], data['rt_to_heads'], device)
            elapsed = time.time() - t0

            # Brain-specific stats
            brain_info = ""
            if is_brain and hasattr(model.encoder, 'last_num_constructed_edges'):
                n_new = model.encoder.last_num_constructed_edges
                brain_info = f"  new_edges={n_new}  sp_loss={sp_loss:.4f}"

            print(f"    Ep {epoch:4d}  loss={loss:.4f}  "
                  f"val_MRR={val['MRR']:.4f}  val_H@10={val['Hits@10']:.4f}"
                  f"{brain_info}  [{elapsed:.0f}s]")

            if val['MRR'] > best_val_mrr:
                best_val_mrr = val['MRR']
                evals_no_improve = 0
                best_test = evaluate_lp(
                    model, data['test'], edge_index, edge_types,
                    data['hr_to_tails'], data['rt_to_heads'], device)
                if is_brain and hasattr(model.encoder, 'last_num_constructed_edges'):
                    best_brain_stats = {
                        'constructed_edges': model.encoder.last_num_constructed_edges,
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

    result = {
        'model': model_type,
        'seed': seed,
        'params_total': n_total,
        'params_encoder': n_encoder,
        'best_val_MRR': best_val_mrr,
        **{f'test_{k}': v for k, v in best_test.items()},
        'time_s': elapsed,
    }
    if best_brain_stats:
        result.update(best_brain_stats)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Multi-seed runner + summary
# ═══════════════════════════════════════════════════════════════════════════

def run_multi_seed(model_type, data, args, device, seeds):
    """Run one model across multiple seeds."""
    results = []
    for seed in seeds:
        try:
            r = run_single(model_type, data, args, device, seed)
            results.append(r)
        except Exception as e:
            print(f"    FAILED seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        return None

    metrics = ['test_MRR', 'test_Hits@1', 'test_Hits@3', 'test_Hits@10']
    agg = {
        'model': model_type,
        'params_total': results[0]['params_total'],
        'params_encoder': results[0]['params_encoder'],
        'num_seeds': len(results),
    }
    for m in metrics:
        vals = [r[m] for r in results]
        agg[f'{m}_mean'] = float(np.mean(vals))
        agg[f'{m}_std'] = float(np.std(vals)) if len(vals) > 1 else 0.0
    agg['time_mean'] = float(np.mean([r['time_s'] for r in results]))

    # Brain-specific aggregation
    if any('constructed_edges' in r for r in results):
        edges = [r.get('constructed_edges', 0) for r in results]
        agg['constructed_edges_mean'] = float(np.mean(edges))

    return agg


def print_summary(all_results, data):
    """Print Phase 55 comparison table."""
    valid = [r for r in all_results if r is not None]
    if not valid:
        print("\nNo results to summarize.")
        return

    N_ent = data['num_entities']
    N_train = data['train'].shape[1]

    print("\n" + "=" * 95)
    print("PHASE 55: BRAIN ARCHITECTURE PORT — LINK PREDICTION RESULTS")
    print("=" * 95)
    print(f"  Data: FB15k-237, {N_ent} entities, {data['num_relations']} relations, "
          f"{N_train} train triples")

    header = (f"{'Model':<18} {'Total':>8} {'Enc':>7} "
              f"{'MRR':>14} {'H@1':>14} {'H@3':>14} {'H@10':>14}")
    print(f"\n{header}")
    print("-" * 95)

    for r in valid:
        def fmt(key):
            m = r[f'{key}_mean']
            s = r[f'{key}_std']
            return f"{m:.4f}±{s:.3f}" if s > 0 else f"{m:.4f}      "

        line = (f"{r['model']:<18} {r['params_total']:>8,} {r['params_encoder']:>7,} "
                f"{fmt('test_MRR'):>14} {fmt('test_Hits@1'):>14} "
                f"{fmt('test_Hits@3'):>14} {fmt('test_Hits@10'):>14}")
        if 'constructed_edges_mean' in r:
            line += f"  [new_edges≈{r['constructed_edges_mean']:.0f}]"
        print(line)

    print("-" * 95)

    # Reference values
    print("\n  Phase 46+ references (same data, temp=1.0):")
    print("    DELTA-Full (A):  LP MRR ~0.474")
    print("    SelfBootstrap:   LP MRR ~0.509  (Phase 40)")
    print("    GraphGPS:        LP MRR ~0.501")
    print()

    # KEY QUESTIONS
    df = next((r for r in valid if r['model'] == 'delta_full'), None)
    bh = next((r for r in valid if r['model'] == 'brain_hybrid'), None)
    bp = next((r for r in valid if r['model'] == 'brain_pure'), None)
    sb = next((r for r in valid if r['model'] == 'self_bootstrap'), None)

    print("KEY QUESTIONS:")
    print()

    # Q1: Does Brain match DELTA-Full baseline?
    if bh and df:
        diff = bh['test_MRR_mean'] - df['test_MRR_mean']
        print(f"  1. Does Brain-Hybrid match DELTA-Full baseline (≥0.475)?")
        print(f"     Brain-Hybrid MRR: {bh['test_MRR_mean']:.4f}  "
              f"DELTA-Full MRR: {df['test_MRR_mean']:.4f}  "
              f"Δ={diff:+.4f}")
        if bh['test_MRR_mean'] >= 0.475:
            print(f"     --> YES, Brain matches baseline threshold")
        else:
            print(f"     --> NO, Brain below baseline threshold")
        print()

    # Q2: Does graph construction help vs same-topology self-bootstrap?
    if bh and sb:
        diff = bh['test_MRR_mean'] - sb['test_MRR_mean']
        print(f"  2. Does graph construction add value vs same-topology self-bootstrap?")
        print(f"     Brain-Hybrid MRR: {bh['test_MRR_mean']:.4f}  "
              f"SelfBootstrap MRR: {sb['test_MRR_mean']:.4f}  "
              f"Δ={diff:+.4f}")
        if diff > 0.005:
            print(f"     --> YES, graph construction provides +{diff:.4f} MRR")
        elif diff > -0.005:
            print(f"     --> COMPARABLE (within ±0.005)")
        else:
            print(f"     --> NO, construction hurts ({diff:+.4f})")
        print()

    # Q3: Is KG topology essential or can Brain learn from scratch?
    if bh and bp:
        diff = bh['test_MRR_mean'] - bp['test_MRR_mean']
        print(f"  3. Is KG topology essential (hybrid vs pure)?")
        print(f"     Brain-Hybrid MRR: {bh['test_MRR_mean']:.4f}  "
              f"Brain-Pure MRR: {bp['test_MRR_mean']:.4f}  "
              f"Δ={diff:+.4f}")
        if diff > 0.01:
            print(f"     --> YES, KG edges are essential (+{diff:.4f})")
        elif diff > -0.01:
            print(f"     --> Surprisingly comparable — Brain can discover structure")
        else:
            print(f"     --> KG edges HURT — original topology confuses the model")
        print()

    # Overall verdict
    print("VERDICT:")
    if bh and bh['test_MRR_mean'] >= 0.490:
        print("  Brain architecture shows STRONG promise for LP.")
        print("  Proceed to Phase 56: temperature tuning for Brain + multi-hop eval.")
    elif bh and bh['test_MRR_mean'] >= 0.475:
        print("  Brain architecture MATCHES baseline — viable but needs optimization.")
        print("  Proceed to Phase 56: constructor tuning (density, tau schedule).")
    elif bh and bh['test_MRR_mean'] >= 0.450:
        print("  Brain architecture UNDERPERFORMS — constructor may be adding noise.")
        print("  Investigate: reduce target_density, add edge filtering, try curriculum.")
    else:
        print("  Brain architecture FAILS baseline threshold.")
        print("  Diagnose: check gradient flow through constructor, verify edge quality.")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 55: Brain Architecture Port")
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_every', type=int, default=25)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seeds', type=str, default='42,123,456')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated models (default: all)')
    parser.add_argument('--max_entities', type=int, default=500)
    parser.add_argument('--target_density', type=float, default=0.005)
    parser.add_argument('--sparsity_weight', type=float, default=0.01)
    parser.add_argument('--full', action='store_true',
                        help='Full FB15k-237 (no entity cap)')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    models = (args.models.split(',') if args.models
              else PHASE55_MODELS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 95)
    print("PHASE 55: Brain Architecture Port — Differentiable Graph Construction")
    print("=" * 95)
    print(f"  Device: {device}")
    print(f"  Seeds: {seeds}")
    print(f"  Models: {models}")
    print(f"  Epochs: {args.epochs}, eval_every: {args.eval_every}, "
          f"patience: {args.patience}")
    print(f"  Target density: {args.target_density}, "
          f"sparsity weight: {args.sparsity_weight}")

    # Load data
    max_ent = None if args.full else args.max_entities
    data = load_lp_data('fb15k-237', max_entities=max_ent)

    # Run all models
    all_results = []
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_type}")
        print(f"{'='*60}")

        agg = run_multi_seed(model_type, data, args, device, seeds)
        all_results.append(agg)

    # Summary
    print_summary(all_results, data)

    # Save results
    output = {
        'phase': 55,
        'title': 'Brain Architecture Port — Differentiable Graph Construction',
        'hypothesis': ('BrainEncoder achieves LP MRR >= 0.475 on FB15k-237'),
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
            'target_density': args.target_density,
            'sparsity_weight': args.sparsity_weight,
        },
        'results': [r for r in all_results if r is not None],
    }

    out_json = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'DELTA', 'phase55_output.json')
    # Handle path for running from DELTA/ directory
    if not os.path.isdir(os.path.dirname(out_json)):
        out_json = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'phase55_output.json')

    try:
        with open(out_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {out_json}")
    except Exception as e:
        print(f"\n  Warning: Could not save results: {e}")

    print("\n  Phase 55 complete.")


if __name__ == "__main__":
    main()
