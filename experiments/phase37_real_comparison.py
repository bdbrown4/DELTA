"""
Phase 37: Real FB15k-237 Parameter-Matched Comparison

Phase 34 showed DELTA dominating synthetic benchmarks, but with 2× more
parameters than baselines (60,594 vs ~30,000). This confounds the result —
any model with double the parameters should do better on small tasks.

This phase removes the parameter confound and tests on real data:

  4 models (all from existing code):
    1. DELTA-Full     (d_node=64, d_edge=32, num_layers=3, num_heads=4) → ~60K params
    2. DELTA-Matched  (d_node=48, d_edge=24, num_layers=2, num_heads=4) → ~30K params
    3. GraphGPS       (existing config)                                 → ~33K params
    4. GRIT           (existing config)                                 → ~28K params

  Real FB15k-237:
    - 14,541 entities, 237 relations, 310,116 triples
    - Uses Phase 31 mini-batching for full-scale training
    - 5 random seeds, report mean ± std

  If DELTA-Matched (30K) beats GraphGPS (33K) and GRIT (28K), the
  architecture genuinely wins — not just the parameter budget.

Success criteria:
    - DELTA-Matched > GraphGPS by ≥ 2% (mean test accuracy, 5 seeds)
    - DELTA-Full ≥ DELTA-Matched (more params should help, not hurt)

Requirements:
    - Phase 31 mini-batching (NeighborSampler)
    - GPU recommended (6-10 hours on H100 for all seeds)
    - pip install torch numpy

Usage:
    python experiments/phase37_real_comparison.py [--entities 500]
    python experiments/phase37_real_comparison.py --full
    python experiments/phase37_real_comparison.py --num_seeds 5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.model import DELTAModel
from delta.baselines import GraphGPSModel, GRITModel
from delta.utils import create_realistic_kg_benchmark
from delta.datasets import load_real_kg
from experiments.phase31_mini_batching import NeighborSampler


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def create_delta_full(num_relations, d_node=64, d_edge=32):
    """DELTA with full parameter budget (~60K params)."""
    return DELTAModel(
        d_node=d_node, d_edge=d_edge, num_layers=3, num_heads=4,
        num_classes=num_relations,
    )


def create_delta_matched(num_relations, d_node=48, d_edge=24):
    """DELTA with parameter budget matched to baselines (~30K params)."""
    return DELTAModel(
        d_node=d_node, d_edge=d_edge, num_layers=2, num_heads=4,
        num_classes=num_relations,
    )


def create_graphgps(num_relations, d_node=64, d_edge=32):
    """GraphGPS baseline (~33K params)."""
    return GraphGPSModel(
        d_node=d_node, d_edge=d_edge, num_layers=3, num_heads=4,
        num_classes=num_relations,
    )


def create_grit(num_relations, d_node=64, d_edge=32):
    """GRIT baseline (~28K params)."""
    return GRITModel(
        d_node=d_node, d_edge=d_edge, num_layers=3, num_heads=4,
        num_classes=num_relations,
    )


# ---------------------------------------------------------------------------
# Feature projections for parameter-matched DELTA
# ---------------------------------------------------------------------------

class FeatureProjector(nn.Module):
    """Projects fixed-dimension graph features to a different dimension.

    Used when DELTA-Matched has d_node=48, d_edge=24 but the graph was
    generated with d_node=64, d_edge=32.
    """

    def __init__(self, in_d_node, in_d_edge, out_d_node, out_d_edge):
        super().__init__()
        self.node_proj = nn.Linear(in_d_node, out_d_node)
        self.edge_proj = nn.Linear(in_d_edge, out_d_edge)

    def forward(self, graph):
        return DeltaGraph(
            node_features=self.node_proj(graph.node_features),
            edge_features=self.edge_proj(graph.edge_features),
            edge_index=graph.edge_index,
            node_tiers=graph.node_tiers,
            node_importance=graph.node_importance,
            edge_importance=graph.edge_importance,
        )


class ProjectedDELTA(nn.Module):
    """DELTA with input projection for dimension mismatch."""

    def __init__(self, in_d_node, in_d_edge, model_d_node, model_d_edge,
                 num_layers, num_heads, num_classes):
        super().__init__()
        self.projector = FeatureProjector(in_d_node, in_d_edge,
                                          model_d_node, model_d_edge)
        self.model = DELTAModel(
            d_node=model_d_node, d_edge=model_d_edge,
            num_layers=num_layers, num_heads=num_heads,
            num_classes=num_classes,
        )

    def forward(self, graph):
        return self.model(self.projector(graph))

    def classify_edges(self, encoded_graph):
        return self.model.classify_edges(encoded_graph)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_and_evaluate(model, graph, labels, epochs, lr, device, label='model',
                       log_every=50, sampler=None, batch_size=64, accum_steps=4,
                       train_idx=None, val_idx=None, test_idx=None,
                       patience=0):
    """Train and evaluate a single model. Returns dict with metrics.

    Args:
        patience: Early stopping patience in evaluation intervals. 0 = disabled.
            E.g. patience=4 with log_every=25 stops after 100 epochs without improvement.
    """
    model = model.to(device)

    E = labels.shape[0]
    if train_idx is None or val_idx is None or test_idx is None:
        perm = torch.randperm(E)
        train_idx = perm[:int(E * 0.7)]
        val_idx = perm[int(E * 0.7):int(E * 0.85)]
        test_idx = perm[int(E * 0.85):]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_train_acc = 0.0
    evals_without_improvement = 0
    stopped_epoch = epochs
    start = time.time()

    if sampler is None:
        # Small graph: full-graph training
        graph = graph.to(device)
        labels = labels.to(device)
        train_idx_d = train_idx.to(device)
        val_idx_d = val_idx.to(device)
        test_idx_d = test_idx.to(device)

        for epoch in range(epochs):
            model.train()
            out = model(graph)
            logits = model.classify_edges(out)
            loss = F.cross_entropy(logits[train_idx_d], labels[train_idx_d])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
                model.eval()
                with torch.no_grad():
                    out = model(graph)
                    logits = model.classify_edges(out)
                    train_acc = (logits[train_idx_d].argmax(-1) == labels[train_idx_d]).float().mean().item()
                    val_acc = (logits[val_idx_d].argmax(-1) == labels[val_idx_d]).float().mean().item()
                    test_acc = (logits[test_idx_d].argmax(-1) == labels[test_idx_d]).float().mean().item()

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_test_acc = test_acc
                        best_train_acc = train_acc
                        evals_without_improvement = 0
                    else:
                        evals_without_improvement += 1

                print(f"    [{label}] Epoch {epoch+1:3d}  "
                      f"Loss: {loss.item():.4f}  "
                      f"Train: {train_acc:.3f}  "
                      f"Val: {val_acc:.3f}  "
                      f"Test: {test_acc:.3f}")

                if patience > 0 and evals_without_improvement >= patience:
                    print(f"    [{label}] Early stopping at epoch {epoch+1} "
                          f"(no improvement for {patience} evals)")
                    stopped_epoch = epoch + 1
                    break
    else:
        # Large graph: mini-batch training
        import random as _random
        train_idx_list = train_idx.tolist()
        val_idx_list = val_idx.tolist()
        test_idx_list = test_idx.tolist()

        def _eval_mb(idx_list):
            correct = 0
            model.eval()
            with torch.no_grad():
                for i in range(0, len(idx_list), batch_size * 2):
                    batch = idx_list[i:i + batch_size * 2]
                    mg, ml, li = sampler.sample_subgraph(
                        batch, graph.node_features, graph.edge_features, labels)
                    if mg is None:
                        continue
                    mg, ml = mg.to(device), ml.to(device)
                    out = model(mg)
                    logits = model.classify_edges(out)
                    correct += (logits[li].argmax(-1) == ml[li]).sum().item()
            return correct / max(len(idx_list), 1)

        for epoch in range(epochs):
            model.train()
            _random.shuffle(train_idx_list)
            optimizer.zero_grad()

            for i in range(0, len(train_idx_list), batch_size):
                batch = train_idx_list[i:i + batch_size]
                mg, ml, li = sampler.sample_subgraph(
                    batch, graph.node_features, graph.edge_features, labels)
                if mg is None:
                    continue
                mg, ml, li = mg.to(device), ml.to(device), li.to(device)
                out = model(mg)
                logits = model.classify_edges(out)
                loss = F.cross_entropy(logits[li], ml[li])
                (loss / accum_steps).backward()
                step_num = (i // batch_size) + 1
                if step_num % accum_steps == 0 or (i + batch_size) >= len(train_idx_list):
                    optimizer.step()
                    optimizer.zero_grad()

            if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
                train_acc = _eval_mb(train_idx_list[:min(len(train_idx_list), 500)])
                val_acc = _eval_mb(val_idx_list)
                test_acc = _eval_mb(test_idx_list)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_train_acc = train_acc
                    evals_without_improvement = 0
                else:
                    evals_without_improvement += 1

                print(f"    [{label}] Epoch {epoch+1:3d}  "
                      f"Train: {train_acc:.3f}  "
                      f"Val: {val_acc:.3f}  "
                      f"Test: {test_acc:.3f}")

                if patience > 0 and evals_without_improvement >= patience:
                    print(f"    [{label}] Early stopping at epoch {epoch+1} "
                          f"(no improvement for {patience} evals)")
                    stopped_epoch = epoch + 1
                    break

    training_time = time.time() - start
    total_params = sum(p.numel() for p in model.parameters())

    return {
        'best_test_acc': best_test_acc,
        'best_val_acc': best_val_acc,
        'best_train_acc': best_train_acc,
        'training_time_s': training_time,
        'total_params': total_params,
        'stopped_epoch': stopped_epoch,
    }


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_multi_seed(model_factory, graph, labels, num_seeds, epochs, lr,
                   device, label='model', log_every=50,
                   sampler=None, batch_size=64, accum_steps=4,
                   train_idx=None, val_idx=None, test_idx=None,
                   patience=0, results_dir=None):
    """Run training across multiple seeds, return aggregated results.

    If results_dir is set, saves per-seed results incrementally and
    skips seeds that already have saved results (crash recovery).
    """
    all_results = []

    # Check for previously completed seeds (resume support)
    completed_seeds = {}
    if results_dir:
        seed_file = os.path.join(results_dir, f'{label}_seeds.json')
        if os.path.exists(seed_file):
            with open(seed_file, 'r') as f:
                completed_seeds = json.load(f)
            print(f"  [Resume] Found {len(completed_seeds)} completed seed(s) for {label}")

    for seed_idx in range(num_seeds):
        seed = 42 + seed_idx * 100
        seed_key = str(seed_idx)

        # Skip already-completed seeds
        if seed_key in completed_seeds:
            print(f"\n  --- Seed {seed_idx + 1}/{num_seeds} (seed={seed}) --- SKIPPED (already complete)")
            print(f"    Cached: test={completed_seeds[seed_key]['best_test_acc']:.3f}, "
                  f"val={completed_seeds[seed_key]['best_val_acc']:.3f}")
            all_results.append(completed_seeds[seed_key])
            continue

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"\n  --- Seed {seed_idx + 1}/{num_seeds} (seed={seed}) ---")
        model = model_factory()
        result = train_and_evaluate(
            model, graph, labels, epochs, lr, device,
            label=f'{label}-s{seed_idx}', log_every=log_every,
            sampler=sampler, batch_size=batch_size, accum_steps=accum_steps,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            patience=patience)
        all_results.append(result)

        # Save per-seed result immediately (crash recovery)
        if results_dir:
            completed_seeds[seed_key] = result
            with open(seed_file, 'w') as f:
                json.dump(completed_seeds, f, indent=2)
            print(f"    [Saved seed {seed_idx+1} to {seed_file}]")

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate
    test_accs = [r['best_test_acc'] for r in all_results]
    val_accs = [r['best_val_acc'] for r in all_results]
    times = [r['training_time_s'] for r in all_results]

    return {
        'test_mean': np.mean(test_accs),
        'test_std': np.std(test_accs),
        'val_mean': np.mean(val_accs),
        'val_std': np.std(val_accs),
        'time_mean': np.mean(times),
        'total_params': all_results[0]['total_params'],
        'individual': all_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 37: Real FB15k-237 Parameter-Matched Comparison")
    parser.add_argument('--entities', type=int, default=500,
                        help='Number of entities (default: 500 for quick test)')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_seeds', type=int, default=3,
                        help='Number of random seeds (5 for publication)')
    parser.add_argument('--full', action='store_true',
                        help='Full FB15k-237 scale (14541 entities, 5 seeds)')
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--skip_full_delta', action='store_true',
                        help='Skip DELTA-Full (save time if only checking param match)')
    parser.add_argument('--skip_models', type=str, default='',
                        help='Comma-separated model names to skip (e.g. "DELTA-Full,GraphGPS")')
    parser.add_argument('--patience', type=int, default=0,
                        help='Early stopping patience in eval intervals. 0=disabled. '
                             'Recommended: 4 (=100 epochs with log_every=25)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to write results log (in addition to stdout)')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory for results/checkpoints. On Colab, defaults to '
                             '/content/drive/MyDrive/DELTA_results/phase37')
    parser.add_argument('--max_neighbors', type=int, default=50,
                        help='Max nodes per mini-graph subgraph (for large graphs)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Edge batch size for mini-batch training')
    parser.add_argument('--accum_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    d_node, d_edge = 64, 32  # Graph generation dimensions

    if args.full:
        args.num_seeds = 5
        if args.log_every == 50:
            args.log_every = 25
        if args.patience == 0:
            args.patience = 4  # Default early stopping for --full (4 evals × 25 = 100 epochs)

    # --- Set up results directory (Google Drive on Colab) ---
    is_colab = os.path.exists('/content')
    if args.results_dir is None and is_colab:
        # Drive must be mounted in the notebook cell before running this script.
        # Check if it's already mounted before using it.
        drive_path = '/content/drive/MyDrive/DELTA_results/phase37'
        if os.path.exists('/content/drive/MyDrive'):
            args.results_dir = drive_path
            print(f"  Google Drive detected — using {drive_path}")
        else:
            # Fall back to local Colab storage (survives within session, not across restarts)
            args.results_dir = '/content/DELTA_results/phase37'
            print(f"  [Warning] Google Drive not mounted. Saving to local Colab storage: {args.results_dir}")
            print(f"  To persist across restarts, run in a notebook cell first:")
            print(f"    from google.colab import drive; drive.mount('/content/drive')")

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        print(f"  Results directory: {args.results_dir}")
        # Default log file to results dir if not specified
        if args.log_file is None:
            args.log_file = os.path.join(args.results_dir, 'phase37_output.txt')

    # Set up file logging (tee to both stdout and file)
    if args.log_file:
        import io

        class TeeWriter:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, data):
                for s in self.streams:
                    s.write(data)
                    s.flush()

            def flush(self):
                for s in self.streams:
                    s.flush()

        log_fh = open(args.log_file, 'a')
        sys.stdout = TeeWriter(sys.__stdout__, log_fh)
    else:
        log_fh = None

    # Parse skip list
    skip_models = set()
    if args.skip_full_delta:
        skip_models.add('DELTA-Full')
    if args.skip_models:
        skip_models.update(name.strip() for name in args.skip_models.split(','))

    use_real_data = args.full  # --full uses real FB15k-237

    print("=" * 70)
    print("PHASE 37: Real FB15k-237 Parameter-Matched Comparison")
    print("=" * 70)
    print(f"  Epochs: {args.epochs}, Seeds: {args.num_seeds}, Device: {device}")
    print(f"  Data: {'REAL FB15k-237' if use_real_data else f'synthetic ({args.entities} entities)'}")
    print()

    # --- Load or generate data ---
    train_idx, val_idx, test_idx = None, None, None
    if use_real_data:
        print("Loading real FB15k-237 dataset...")
        graph, labels, metadata = load_real_kg(
            'fb15k-237', d_node, d_edge)
        num_relations = metadata['num_relations']
        train_idx = metadata['train_idx']
        val_idx = metadata['val_idx']
        test_idx = metadata['test_idx']
    else:
        print("Creating FB15k-237-like synthetic benchmark...")
        num_triples = args.entities * 21
        graph, labels, metadata = create_realistic_kg_benchmark(
            num_entities=args.entities,
            num_triples=num_triples,
            d_node=d_node, d_edge=d_edge,
            seed=42,
        )
        num_relations = metadata['num_relations']
    print(f"  {graph.num_nodes} nodes, {graph.num_edges} edges, "
          f"{num_relations} relations")
    print(f"  Random baseline: {1.0 / num_relations:.3f}")
    print()

    # Create mini-batch sampler for large graphs (avoids CUDA OOM on 304K-edge graphs)
    sampler = None
    if graph.num_edges > 50000:
        print(f"  Large graph detected ({graph.num_edges} edges) — "
              f"using mini-batch training (max_neighbors={args.max_neighbors})")
        sampler = NeighborSampler(
            graph.edge_index, graph.num_nodes,
            k_hops=2, max_neighbors=args.max_neighbors,
        )
        print()

    # --- Model configurations ---
    matched_d_node, matched_d_edge = 48, 24

    model_configs = {}

    if 'DELTA-Full' not in skip_models:
        model_configs['DELTA-Full'] = {
            'factory': lambda: create_delta_full(num_relations, d_node, d_edge),
            'desc': f'd_node={d_node}, d_edge={d_edge}, layers=3',
        }

    model_configs['DELTA-Matched'] = {
        'factory': lambda: ProjectedDELTA(
            d_node, d_edge, matched_d_node, matched_d_edge,
            num_layers=2, num_heads=4, num_classes=num_relations),
        'desc': f'd_node={matched_d_node}, d_edge={matched_d_edge}, layers=2 (projected)',
    }

    model_configs['GraphGPS'] = {
        'factory': lambda: create_graphgps(num_relations, d_node, d_edge),
        'desc': f'd_node={d_node}, d_edge={d_edge}, layers=3',
    }

    model_configs['GRIT'] = {
        'factory': lambda: create_grit(num_relations, d_node, d_edge),
        'desc': f'd_node={d_node}, d_edge={d_edge}, layers=3',
    }

    # Remove skipped models
    for name in skip_models:
        model_configs.pop(name, None)

    if skip_models:
        print(f"  Skipping models: {', '.join(sorted(skip_models))}")
        print()

    # Print param counts before training
    print("  Model Parameter Counts:")
    for name, cfg in model_configs.items():
        m = cfg['factory']()
        params = sum(p.numel() for p in m.parameters())
        print(f"    {name:20s}: {params:,d} params — {cfg['desc']}")
        del m
    print()

    # --- Run all models ---
    results = {}
    if args.results_dir:
        results_file = os.path.join(args.results_dir, 'phase37_results.json')
    elif args.log_file:
        results_file = args.log_file.replace('.txt', '.json')
    else:
        results_file = 'phase37_results.json'
    for name, cfg in model_configs.items():
        print(f"\n{'='*70}")
        print(f"Training: {name}")
        print(f"{'='*70}")
        results[name] = run_multi_seed(
            cfg['factory'], graph, labels, args.num_seeds,
            args.epochs, args.lr, device,
            label=name, log_every=args.log_every,
            sampler=sampler, batch_size=args.batch_size,
            accum_steps=args.accum_steps,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            patience=args.patience,
            results_dir=args.results_dir)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save incremental results after each model completes
        _save = {}
        for rname, rdata in results.items():
            _save[rname] = {
                'test_mean': rdata['test_mean'], 'test_std': rdata['test_std'],
                'val_mean': rdata['val_mean'], 'val_std': rdata['val_std'],
                'time_mean': rdata['time_mean'], 'total_params': rdata['total_params'],
                'per_seed': [{'test': r['best_test_acc'], 'val': r['best_val_acc'],
                              'time': r['training_time_s'], 'stopped_epoch': r['stopped_epoch']}
                             for r in rdata['individual']],
            }
        with open(results_file, 'w') as f:
            json.dump(_save, f, indent=2)
        print(f"\n  [Saved incremental results to {results_file}]")

    # ==================================================================
    # RESULTS
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 37 RESULTS")
    print("=" * 70)
    print(f"\n  {'Model':20s}  {'Params':>8s}  "
          f"{'Test Acc (mean±std)':>22s}  "
          f"{'Val Acc (mean±std)':>22s}  "
          f"{'Time (s)':>10s}")
    print("  " + "-" * 90)

    for name, res in results.items():
        print(f"  {name:20s}  {res['total_params']:>8,d}  "
              f"{res['test_mean']:.3f} ± {res['test_std']:.3f}"
              f"{'':>10s}"
              f"{res['val_mean']:.3f} ± {res['val_std']:.3f}"
              f"{'':>6s}"
              f"{res['time_mean']:>8.1f}")

    # --- Statistical comparison ---
    print(f"\n  Key Comparisons:")

    matched = results.get('DELTA-Matched')
    gps = results.get('GraphGPS')
    grit = results.get('GRIT')
    full = results.get('DELTA-Full')

    if matched and gps:
        delta_vs_gps = matched['test_mean'] - gps['test_mean']
        print(f"    DELTA-Matched vs GraphGPS:  {delta_vs_gps:+.3f}  "
              f"({'✓ DELTA wins' if delta_vs_gps > 0.02 else '✗ No clear winner'})")

    if matched and grit:
        delta_vs_grit = matched['test_mean'] - grit['test_mean']
        print(f"    DELTA-Matched vs GRIT:      {delta_vs_grit:+.3f}  "
              f"({'✓ DELTA wins' if delta_vs_grit > 0.02 else '✗ No clear winner'})")

    if full and matched:
        full_vs_matched = full['test_mean'] - matched['test_mean']
        print(f"    DELTA-Full vs DELTA-Matched: {full_vs_matched:+.3f}  "
              f"({'Params help' if full_vs_matched > 0.01 else 'Matched is sufficient'})")

    # Param efficiency
    print(f"\n  Parameter Efficiency (acc / 10K params):")
    for name, res in results.items():
        eff = res['test_mean'] / (res['total_params'] / 10000)
        print(f"    {name:20s}: {eff:.4f}")

    # Per-seed detail
    print(f"\n  Per-Seed Test Accuracies:")
    for name, res in results.items():
        accs = [f"{r['best_test_acc']:.3f}" for r in res['individual']]
        print(f"    {name:20s}: {', '.join(accs)}")

    # Verdict
    print(f"\n  Publication Criteria:")
    if matched and gps:
        delta_vs_gps = matched['test_mean'] - gps['test_mean']
        passed = delta_vs_gps >= 0.02
        print(f"    DELTA-Matched > GraphGPS by ≥ 2%:  "
              f"{'✓ PASSED' if passed else '✗ FAILED'} ({delta_vs_gps:+.3f})")
    if full and matched:
        full_vs_matched = full['test_mean'] - matched['test_mean']
        passed = full_vs_matched >= 0.0
        print(f"    DELTA-Full ≥ DELTA-Matched:         "
              f"{'✓ PASSED' if passed else '✗ FAILED'} ({full_vs_matched:+.3f})")

    # Cleanup
    if log_fh:
        sys.stdout = sys.__stdout__
        log_fh.close()
        print(f"\nResults saved to {args.log_file} and {results_file}")


if __name__ == '__main__':
    main()
