"""Phase 57: Temperature Annealing on Brain Architecture
=======================================================

Motivation (from Phase 56):
  Phase 56 showed density=0.01 strictly dominates density=0.02 for brain_hybrid:
    C (d=0.01): MRR=0.4794, H@10=0.8076, 2435 edges
    D (d=0.02): MRR=0.4678, H@10=0.7500, 4870 edges

  brain_hybrid @ d=0.01 matches delta_full MRR (0.4794 vs 0.4796) while adding
  +4.7% H@10.  The 0.480 MRR target was missed by only 0.0006.

  Temperature annealing (Phases 50-52) reliably improved delta_full LP MRR:
    K (node anneal 4→2, edge=6.0): LP=0.4819 (+0.009 over baseline)
    Q (K + edge=7.0):              LP=0.4905 (+0.018 over baseline)

  Applying proven temperature annealing to brain_hybrid @ optimal density
  should close the 0.0006 gap. BrainEncoder has 3 DELTA layers
  (1 bootstrap + 2 delta) — annealing targets the Stage 3 delta layers.

Hypothesis (falsifiable):
  "Applying K-style node temperature annealing (4→2 over 50% of training,
  edge init=6.0) to brain_hybrid @ density=0.01 achieves LP MRR >= 0.480."

  If MRR >= 0.490 (temperature-tuned delta_full range), brain_hybrid with
  density+temperature optimization shows clear superiority.

Design:
  3 conditions × 1 seed = 3 runs:
    A. brain_hybrid @ d=0.01, temp=1.0, no anneal (Phase 56 C baseline)
    B. brain_hybrid @ d=0.01, K-style anneal (node 4→2, edge=6.0, Stage 3 only)
    C. brain_hybrid @ d=0.01, Q-style anneal (node 4→2, edge=7.0, Stage 3 only)

  ONE primary change: Temperature annealing applied to BrainEncoder delta layers.
  Bootstrap layer (Stage 1) uses init_temp=1.0 (low complexity, L0-equivalent).

  Seeds: [42]
  Data: FB15k-237, 500-entity dense subset (same as Phases 46-56)
  Epochs: 200, eval_every=30, patience=10 (brain_hybrid peaks ~ep150, overtains after)

Measurements:
  LP: filtered MRR, Hits@1, H@3, H@10
  Brain-specific: number of constructed edges, sparsity loss

Regression safety:
  LP MRR must not regress below 0.45 (brain_hybrid floor from Phase 55).

Usage:
  # Smoke test (5 epochs)
  python experiments/phase57_brain_temp_anneal.py --epochs 5 --seeds 42

  # Full run
  python experiments/phase57_brain_temp_anneal.py --epochs 200 --eval_every 30 --patience 10 --seeds 42

  # Run specific conditions
  python experiments/phase57_brain_temp_anneal.py --epochs 200 --eval_every 30 --patience 10 --conditions B,C
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
import torch.nn as nn
import torch.nn.functional as F

from delta.model import DELTAModel
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
# Temperature utilities for BrainEncoder
# ═══════════════════════════════════════════════════════════════════════════

def set_brain_temps(model, stage3_node_temp, stage3_edge_temp,
                    stage1_temp=1.0):
    """Set temperature values on BrainEncoder layers.

    Args:
        model: LinkPredictionModel with BrainEncoder as encoder
        stage3_node_temp: Node attention temperature for delta layers (Stage 3)
        stage3_edge_temp: Edge attention temperature for delta layers (Stage 3)
        stage1_temp: Temperature for bootstrap layers (Stage 1, default=1.0)
    """
    enc = model.encoder
    assert isinstance(enc, BrainEncoder), f"Expected BrainEncoder, got {type(enc)}"

    with torch.no_grad():
        # Stage 1: Bootstrap layers — set to low temp (like L0 in standard model)
        for layer in enc.bootstrap_layers:
            for attn_name in ['dual_attn', 'global_node_attn']:
                if not hasattr(layer, attn_name):
                    continue
                attn = getattr(layer, attn_name)
                if hasattr(attn, 'node_attn') and hasattr(attn.node_attn, '_log_temp'):
                    attn.node_attn._log_temp.fill_(math.log(stage1_temp))
                if hasattr(attn, 'edge_attn') and hasattr(attn.edge_attn, '_log_temp'):
                    attn.edge_attn._log_temp.fill_(math.log(stage1_temp))
                if hasattr(attn, '_log_temp'):
                    attn._log_temp.fill_(math.log(stage1_temp))

        # Stage 3: Delta layers — set asymmetric node/edge temps
        for layer in enc.delta_layers:
            for attn_name in ['dual_attn', 'global_node_attn']:
                if not hasattr(layer, attn_name):
                    continue
                attn = getattr(layer, attn_name)
                if hasattr(attn, 'node_attn') and hasattr(attn.node_attn, '_log_temp'):
                    attn.node_attn._log_temp.fill_(math.log(stage3_node_temp))
                if hasattr(attn, 'edge_attn') and hasattr(attn.edge_attn, '_log_temp'):
                    attn.edge_attn._log_temp.fill_(math.log(stage3_edge_temp))
                if hasattr(attn, '_log_temp'):
                    attn._log_temp.fill_(math.log(stage3_node_temp))


def anneal_brain_node_temps(model, epoch, anneal_epochs, start_temp, end_temp):
    """Apply linear node temperature annealing to BrainEncoder Stage 3 layers.

    During the anneal period (epoch 1..anneal_epochs), node temp decays
    linearly from start_temp to end_temp.  After the anneal period, node
    temps become fully learnable (no more resets).

    Only affects Stage 3 delta layers' node attention.
    Edge temps and Stage 1 bootstrap temps are never touched.

    Returns the current scheduled node temp (for logging), or None after
    annealing ends.
    """
    if epoch > anneal_epochs:
        return None  # Post-anneal: temps are learnable

    progress = epoch / anneal_epochs  # 0.0 → 1.0
    scheduled_temp = start_temp + (end_temp - start_temp) * progress

    enc = model.encoder
    assert isinstance(enc, BrainEncoder)

    with torch.no_grad():
        for layer in enc.delta_layers:
            for attn_name in ['dual_attn', 'global_node_attn']:
                if not hasattr(layer, attn_name):
                    continue
                attn = getattr(layer, attn_name)
                if hasattr(attn, 'node_attn') and hasattr(attn.node_attn, '_log_temp'):
                    attn.node_attn._log_temp.fill_(math.log(scheduled_temp))
                if hasattr(attn, '_log_temp'):
                    attn._log_temp.fill_(math.log(scheduled_temp))

    return scheduled_temp


def get_brain_temps(model):
    """Extract current learned temperatures from BrainEncoder layers."""
    enc = model.encoder
    temps = {'stage1': {}, 'stage3': {}}

    for i, layer in enumerate(enc.bootstrap_layers):
        key = f'bootstrap_{i}'
        temps['stage1'][key] = _extract_layer_temps(layer)

    for i, layer in enumerate(enc.delta_layers):
        key = f'delta_{i}'
        temps['stage3'][key] = _extract_layer_temps(layer)

    return temps


def _extract_layer_temps(layer):
    """Extract node/edge temps from a DELTALayer."""
    result = {}
    if hasattr(layer, 'dual_attn'):
        da = layer.dual_attn
        if hasattr(da, 'node_attn') and hasattr(da.node_attn, '_log_temp'):
            result['node'] = da.node_attn._log_temp.exp().detach().cpu().tolist()
        if hasattr(da, 'edge_attn') and hasattr(da.edge_attn, '_log_temp'):
            result['edge'] = da.edge_attn._log_temp.exp().detach().cpu().tolist()
    if hasattr(layer, 'global_node_attn') and hasattr(layer.global_node_attn, '_log_temp'):
        result['global_node'] = layer.global_node_attn._log_temp.exp().detach().cpu().tolist()
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Model creation
# ═══════════════════════════════════════════════════════════════════════════

def create_brain_model(num_entities, num_relations, d_node=64, d_edge=32,
                       target_density=0.01, init_temp=1.0):
    """Create brain_hybrid model with configurable init_temp."""
    enc = BrainEncoder(
        d_node=d_node, d_edge=d_edge,
        bootstrap_layers=1, delta_layers=2, num_heads=4,
        target_density=target_density, hybrid=True, init_temp=init_temp)
    return LinkPredictionModel(enc, num_entities, num_relations, d_node, d_edge)


# ═══════════════════════════════════════════════════════════════════════════
# Conditions
# ═══════════════════════════════════════════════════════════════════════════

CONDITIONS = {
    'A': {
        'name': 'brain_baseline',
        'desc': 'brain_hybrid @ d=0.01, temp=1.0, no anneal (P56 C reference)',
        'init_temp': 1.0,
        'stage3_node_temp': 1.0,
        'stage3_edge_temp': 1.0,
        'stage1_temp': 1.0,
        'anneal': False,
    },
    'B': {
        'name': 'brain_K_anneal',
        'desc': 'brain_hybrid @ d=0.01, K-style: node anneal 4→2, edge=6.0',
        'init_temp': 4.0,
        'stage3_node_temp': 4.0,
        'stage3_edge_temp': 6.0,
        'stage1_temp': 1.0,
        'anneal': True,
        'node_start': 4.0,
        'node_end': 2.0,
        'anneal_frac': 0.5,
    },
    'C': {
        'name': 'brain_Q_anneal',
        'desc': 'brain_hybrid @ d=0.01, Q-style: node anneal 4→2, edge=7.0',
        'init_temp': 4.0,
        'stage3_node_temp': 4.0,
        'stage3_edge_temp': 7.0,
        'stage1_temp': 1.0,
        'anneal': True,
        'node_start': 4.0,
        'node_end': 2.0,
        'anneal_frac': 0.5,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Training loop with annealing
# ═══════════════════════════════════════════════════════════════════════════

def run_condition(cond_key, cond, data, args, device, seed):
    """Train + evaluate one condition on one seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    model = create_brain_model(
        data['num_entities'], data['num_relations'],
        target_density=args.target_density,
        init_temp=cond['init_temp'])

    # Set per-stage temperatures
    set_brain_temps(model,
                    stage3_node_temp=cond['stage3_node_temp'],
                    stage3_edge_temp=cond['stage3_edge_temp'],
                    stage1_temp=cond['stage1_temp'])

    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.encoder.parameters())

    anneal_epochs = int(args.epochs * cond.get('anneal_frac', 0.5)) if cond['anneal'] else 0

    print(f"\n  [{cond['name']}] seed={seed}, "
          f"{n_total:,} total params ({n_encoder:,} encoder), device={device}")
    if cond['anneal']:
        print(f"    Anneal: node {cond['node_start']:.1f}→{cond['node_end']:.1f} "
              f"over {anneal_epochs} epochs, edge init={cond['stage3_edge_temp']:.1f}")
    else:
        print(f"    No annealing, all temps={cond['init_temp']:.1f}")

    edge_index, edge_types = build_train_graph_tensors(data['train'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_mrr = 0.0
    best_test = None
    best_brain_stats = {}
    evals_no_improve = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # Apply temperature annealing before training
        if cond['anneal']:
            sched_temp = anneal_brain_node_temps(
                model, epoch, anneal_epochs,
                cond['node_start'], cond['node_end'])

        # Train one epoch
        loss, sp_loss = train_epoch_brain(
            model, data['train'], edge_index, edge_types,
            optimizer, device, args.batch_size,
            sparsity_weight=args.sparsity_weight)

        # Evaluate
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate_lp(model, data['val'], edge_index, edge_types,
                              data['hr_to_tails'], data['rt_to_heads'], device)
            elapsed = time.time() - t0

            # Brain-specific stats
            n_new = 0
            if hasattr(model.encoder, 'last_num_constructed_edges'):
                n_new = model.encoder.last_num_constructed_edges

            # Temperature info (read-only, don't re-anneal)
            anneal_info = ""
            if cond['anneal']:
                if epoch <= anneal_epochs:
                    progress = epoch / anneal_epochs
                    cur_temp = cond['node_start'] + (cond['node_end'] - cond['node_start']) * progress
                    anneal_info = f"  node_t={cur_temp:.2f}"
                else:
                    anneal_info = "  node_t=learnable"

            print(f"    Ep {epoch:4d}  loss={loss:.4f}  "
                  f"val_MRR={val['MRR']:.4f}  val_H@10={val['Hits@10']:.4f}"
                  f"  new_edges={n_new}  sp_loss={sp_loss:.4f}"
                  f"{anneal_info}  [{elapsed:.0f}s]")

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
        'target_density': args.target_density,
        'stage3_node_temp_init': cond['stage3_node_temp'],
        'stage3_edge_temp_init': cond['stage3_edge_temp'],
        'anneal': cond['anneal'],
        **best_brain_stats,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Phase 57: Brain Temperature Annealing')
    parser.add_argument('--seeds', type=str, default='42',
                        help='Comma-separated seeds (default: 42)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_every', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--target_density', type=float, default=0.01,
                        help='Constructor density (default: 0.01, optimal from P56)')
    parser.add_argument('--sparsity_weight', type=float, default=0.01)
    parser.add_argument('--max_entities', type=int, default=500,
                        help='Max entities for dense subset (default: 500)')
    parser.add_argument('--conditions', type=str, default=None,
                        help='Comma-separated condition letters to run (e.g. B,C). Default: all')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to prior phase57_output.json to merge results')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    run_conditions = set(args.conditions.split(',')) if args.conditions else set(CONDITIONS.keys())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("PHASE 57: Brain Temperature Annealing — Closing the 0.480 MRR Gap")
    print("=" * 80)
    print(f"  Device: {device}")
    print(f"  Seeds: {seeds}")
    print(f"  Target density: {args.target_density}")
    print(f"  Epochs: {args.epochs}, eval_every: {args.eval_every}, patience: {args.patience}")
    print(f"  Running conditions: {sorted(run_conditions)}")

    # Load data (500-entity dense subset, same as Phases 46-56)
    data = load_lp_data('fb15k-237', max_entities=args.max_entities)
    print(f"  Loaded fb15k-237 for link prediction:")
    print(f"    {data['num_entities']} entities, {data['num_relations']} relations")
    print(f"    {data['train'].shape[1]} train / {data['val'].shape[1]} val / "
          f"{data['test'].shape[1]} test")

    # Load prior results if resuming
    prior_results = []
    if args.resume and os.path.exists(args.resume):
        with open(args.resume) as f:
            prior = json.load(f)
            prior_results = prior.get('results', [])
        print(f"  Loaded {len(prior_results)} prior results from {args.resume}")

    all_results = list(prior_results)

    # Run conditions
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

    # Summary
    print(f"\n{'=' * 80}")
    print(f"PHASE 57: BRAIN TEMPERATURE ANNEALING — RESULTS")
    print(f"{'=' * 80}")
    print(f"  Data: FB15k-237, {data['num_entities']} entities, "
          f"{data['num_relations']} relations, {data['train'].shape[1]} train")
    print(f"  Epochs: {args.epochs}, seeds: {seeds}")
    print(f"  Target density: {args.target_density}")

    # Summary table
    header = f"{'Condition':<28s} {'NodeT':>6s} {'EdgeT':>6s} {'Anneal':>6s} " \
             f"{'MRR':>8s} {'H@1':>8s} {'H@3':>8s} {'H@10':>8s} {'Edges':>7s} {'Time':>7s}"
    print(f"\n{header}")
    print("-" * len(header))

    for r in all_results:
        cond_name = r.get('condition', r.get('model', '?'))
        node_t = r.get('stage3_node_temp_init', 1.0)
        edge_t = r.get('stage3_edge_temp_init', 1.0)
        anneal_str = 'yes' if r.get('anneal', False) else 'no'
        edges = r.get('constructed_edges', 0)
        time_s = r.get('time_s', 0)

        print(f"{cond_name:<28s} {node_t:>6.1f} {edge_t:>6.1f} {anneal_str:>6s} "
              f"{r['test_MRR']:>8.4f} {r['test_Hits@1']:>8.4f} "
              f"{r['test_Hits@3']:>8.4f} {r['test_Hits@10']:>8.4f} "
              f"{edges:>7d} {time_s:>6.0f}s")

    print("-" * len(header))

    # Save results
    output = {
        'phase': 57,
        'title': 'Brain Temperature Annealing — Closing the 0.480 MRR Gap',
        'hypothesis': 'Applying K-style node annealing (4→2) to brain_hybrid @ d=0.01 achieves LP MRR >= 0.480',
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
            'target_density': args.target_density,
            'sparsity_weight': args.sparsity_weight,
        },
        'results': all_results,
    }

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'phase57_output.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    print(f"\n  Phase 57 complete.")


if __name__ == '__main__':
    main()
