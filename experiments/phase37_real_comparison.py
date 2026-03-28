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
                       train_idx=None, val_idx=None, test_idx=None):
    """Train and evaluate a single model. Returns dict with metrics."""
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

                print(f"    [{label}] Epoch {epoch+1:3d}  "
                      f"Loss: {loss.item():.4f}  "
                      f"Train: {train_acc:.3f}  "
                      f"Val: {val_acc:.3f}  "
                      f"Test: {test_acc:.3f}")
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

                print(f"    [{label}] Epoch {epoch+1:3d}  "
                      f"Train: {train_acc:.3f}  "
                      f"Val: {val_acc:.3f}  "
                      f"Test: {test_acc:.3f}")

    training_time = time.time() - start
    total_params = sum(p.numel() for p in model.parameters())

    return {
        'best_test_acc': best_test_acc,
        'best_val_acc': best_val_acc,
        'best_train_acc': best_train_acc,
        'training_time_s': training_time,
        'total_params': total_params,
    }


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_multi_seed(model_factory, graph, labels, num_seeds, epochs, lr,
                   device, label='model', log_every=50,
                   sampler=None, batch_size=64, accum_steps=4,
                   train_idx=None, val_idx=None, test_idx=None):
    """Run training across multiple seeds, return aggregated results."""
    all_results = []

    for seed_idx in range(num_seeds):
        seed = 42 + seed_idx * 100
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
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        all_results.append(result)

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
                        help='Full FB15k-237 scale (14505 entities, 5 seeds)')
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--skip_full_delta', action='store_true',
                        help='Skip DELTA-Full (save time if only checking param match)')
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

    if not args.skip_full_delta:
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
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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


if __name__ == '__main__':
    main()
