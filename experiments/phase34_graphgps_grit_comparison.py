"""
Phase 34: DELTA vs GraphGPS vs GRIT — Controlled Comparison

Core question: How does DELTA's dual parallel attention compare against
state-of-the-art graph transformers (GraphGPS 2022, GRIT 2023)?

This is the critical experiment for publication readiness — Gap 1 in the
research agenda. The community will ask: "Why not just use GraphGPS?"

Comparison protocol:
- Same data (synthetic KG, noisy KG, path composition, and — when compute
  allows — real FB15k-237)
- Same number of layers, same hidden dimensions
- Same optimizer, learning rate, and training epochs
- Same train/test split
- Multiple random seeds for statistical validity

Tasks:
1. Edge classification on synthetic KG (relation type prediction)
2. Edge classification under noise (robustness test)
3. Path composition (multi-hop relational reasoning)
4. Link prediction (when run on GPU with full FB15k-237)

Expected outcome based on prior phases:
- DELTA should excel on edge classification (edge-first attention is purpose-built)
- DELTA should show noise robustness advantage (Phase 28: +24% at extreme noise)
- GraphGPS may be competitive on node tasks (MPNN + global attn is strong)
- GRIT may struggle on edge tasks (no explicit edge computation)

Usage:
    python experiments/phase34_graphgps_grit_comparison.py [--seeds 5] [--epochs 300]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.model import DELTAModel
from delta.baselines import GraphGPSModel, GRITModel
from delta.utils import (
    create_analogy_task,
    create_noisy_kg_benchmark,
    create_multi_relational_reasoning_task,
    create_knowledge_graph,
)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(model, graph, labels, task_type='edge',
                       epochs=200, lr=1e-3, seed=42, device='cpu'):
    """Train a model and return best test accuracy + training time.

    Args:
        model: any model with forward(), classify_nodes(), classify_edges()
        graph: DeltaGraph input
        labels: ground-truth labels
        task_type: 'edge' or 'node'
        epochs: training epochs
        lr: learning rate
        seed: random seed for train/test split
        device: 'cpu' or 'cuda'

    Returns:
        dict with best_test_acc, final_train_acc, training_time_s
    """
    torch.manual_seed(seed)

    # Move model and data to device
    model = model.to(device)
    graph = graph.to(device)
    labels = labels.to(device)

    N_items = labels.shape[0]

    perm = torch.randperm(N_items, device=device)
    train_end = int(N_items * 0.7)
    train_idx = perm[:train_end]
    test_idx = perm[train_end:]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_test_acc = 0.0
    final_train_acc = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        out = model(graph)

        if task_type == 'edge':
            logits = model.classify_edges(out)
        else:
            logits = model.classify_nodes(out)

        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                out = model(graph)
                if task_type == 'edge':
                    logits = model.classify_edges(out)
                else:
                    logits = model.classify_nodes(out)

                train_acc = (logits[train_idx].argmax(-1) == labels[train_idx]).float().mean().item()
                test_acc = (logits[test_idx].argmax(-1) == labels[test_idx]).float().mean().item()
                best_test_acc = max(best_test_acc, test_acc)
                final_train_acc = train_acc

    training_time = time.time() - start_time
    return {
        'best_test_acc': best_test_acc,
        'final_train_acc': final_train_acc,
        'training_time_s': training_time,
    }


def create_models(d_node, d_edge, num_classes, num_layers=3, num_heads=4):
    """Create all three architectures with matched hyperparameters."""
    models = {
        'DELTA': DELTAModel(
            d_node=d_node, d_edge=d_edge, num_layers=num_layers,
            num_heads=num_heads, num_classes=num_classes,
        ),
        'GraphGPS': GraphGPSModel(
            d_node=d_node, d_edge=d_edge, num_layers=num_layers,
            num_heads=num_heads, num_classes=num_classes,
        ),
        'GRIT': GRITModel(
            d_node=d_node, d_edge=d_edge, num_layers=num_layers,
            num_heads=num_heads, walk_length=8, d_pe=16,
            num_classes=num_classes,
        ),
    }
    return models


# ---------------------------------------------------------------------------
# Benchmark tasks
# ---------------------------------------------------------------------------

def benchmark_edge_classification(num_seeds=3, epochs=200, d_node=64, d_edge=32,
                                  device='cpu'):
    """Task 1: Relation type classification on synthetic KG."""
    print("=" * 70)
    print("TASK 1: Edge Classification (Synthetic KG)")
    print("=" * 70)
    print("  Classify relation types — DELTA's edge-first attention should shine.")
    print()

    num_patterns = 6
    results = {name: [] for name in ['DELTA', 'GraphGPS', 'GRIT']}

    for seed in range(num_seeds):
        graph, labels = create_analogy_task(
            num_patterns=num_patterns, instances_per_pattern=8,
            d_node=d_node, d_edge=d_edge, seed=seed,
        )
        print(f"  Seed {seed}: {graph.num_nodes} nodes, {graph.num_edges} edges")

        models = create_models(d_node, d_edge, num_patterns)
        for name, model in models.items():
            r = train_and_evaluate(model, graph, labels, 'edge', epochs,
                                   seed=seed, device=device)
            results[name].append(r['best_test_acc'])
            print(f"    {name:<10s}  Test Acc: {r['best_test_acc']:.3f}  "
                  f"({r['training_time_s']:.1f}s)")

    return results


def benchmark_noise_robustness(num_seeds=3, epochs=200, d_node=64, d_edge=32,
                               device='cpu'):
    """Task 2: Edge classification under increasing noise levels."""
    print()
    print("=" * 70)
    print("TASK 2: Noise Robustness (Edge Classification Under Noise)")
    print("=" * 70)
    print("  Phase 28 showed DELTA +24% at extreme noise. Replicate vs baselines.")
    print()

    noise_levels = [0.0, 0.2, 0.5, 0.8]
    all_results = {}

    for noise in noise_levels:
        print(f"  --- Noise level: {noise:.1f} ---")
        results = {name: [] for name in ['DELTA', 'GraphGPS', 'GRIT']}

        for seed in range(num_seeds):
            graph, labels, metadata = create_noisy_kg_benchmark(
                num_entities=50, num_relations=6,
                num_triples=200,
                d_node=d_node, d_edge=d_edge,
                noise_ratio=noise, seed=seed,
            )
            num_classes = labels.max().item() + 1
            models = create_models(d_node, d_edge, num_classes)

            for name, model in models.items():
                r = train_and_evaluate(model, graph, labels, 'edge', epochs,
                                       seed=seed, device=device)
                results[name].append(r['best_test_acc'])

            print(f"    Seed {seed}: " + "  ".join(
                f"{n}: {results[n][-1]:.3f}" for n in results))

        all_results[noise] = results

    return all_results


def benchmark_path_composition(num_seeds=3, epochs=200, d_node=64, d_edge=32,
                               device='cpu'):
    """Task 3: Multi-hop relational reasoning (compositional rules)."""
    print()
    print("=" * 70)
    print("TASK 3: Compositional Relational Reasoning")
    print("=" * 70)
    print("  Phase 27b: fixed graph +4.4% over transformer. Test vs GraphGPS/GRIT.")
    print()

    results = {name: [] for name in ['DELTA', 'GraphGPS', 'GRIT']}

    for seed in range(num_seeds):
        graph, labels, metadata = create_multi_relational_reasoning_task(
            num_entities=60, num_base_relations=4, num_derived_rules=3,
            d_node=d_node, d_edge=d_edge, seed=seed,
        )
        num_classes = metadata['num_total_relations']
        print(f"  Seed {seed}: {graph.num_nodes} nodes, {graph.num_edges} edges, "
              f"{num_classes} classes ({metadata['n_base']} base + {metadata['n_derived']} derived)")

        models = create_models(d_node, d_edge, num_classes)
        for name, model in models.items():
            r = train_and_evaluate(model, graph, labels, 'edge', epochs,
                                   seed=seed, device=device)
            results[name].append(r['best_test_acc'])
            print(f"    {name:<10s}  Test Acc: {r['best_test_acc']:.3f}  "
                  f"({r['training_time_s']:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# Summary and analysis
# ---------------------------------------------------------------------------

def summarize_results(task_name, results):
    """Print mean ± std for each model on a task."""
    print(f"\n  {task_name}:")
    for name in ['DELTA', 'GraphGPS', 'GRIT']:
        accs = results[name]
        if accs:
            mean = sum(accs) / len(accs)
            if len(accs) > 1:
                std = (sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)) ** 0.5
                print(f"    {name:<10s}  {mean:.3f} ± {std:.3f}  (n={len(accs)})")
            else:
                print(f"    {name:<10s}  {mean:.3f}  (n=1)")


def main():
    parser = argparse.ArgumentParser(description="Phase 34: DELTA vs GraphGPS vs GRIT")
    parser.add_argument('--seeds', type=int, default=3, help='Number of random seeds')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs per run')
    parser.add_argument('--d_node', type=int, default=64, help='Node feature dimension')
    parser.add_argument('--d_edge', type=int, default=32, help='Edge feature dimension')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu/cuda). Auto-detects if not set.')
    args = parser.parse_args()

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 70)
    print("PHASE 34: DELTA vs GraphGPS vs GRIT — Controlled Comparison")
    print("=" * 70)
    print(f"Config: seeds={args.seeds}, epochs={args.epochs}, "
          f"d_node={args.d_node}, d_edge={args.d_edge}, device={device}")
    print()

    start = time.time()

    # Task 1: Edge classification
    edge_results = benchmark_edge_classification(
        args.seeds, args.epochs, args.d_node, args.d_edge, device=device)

    # Task 2: Noise robustness
    noise_results = benchmark_noise_robustness(
        args.seeds, args.epochs, args.d_node, args.d_edge, device=device)

    # Task 3: Path composition
    path_results = benchmark_path_composition(
        args.seeds, args.epochs, args.d_node, args.d_edge, device=device)

    total_time = time.time() - start

    # --- Summary ---
    print()
    print("=" * 70)
    print("PHASE 34 RESULTS SUMMARY")
    print("=" * 70)

    summarize_results("Edge Classification (Synthetic KG)", edge_results)

    for noise, results in noise_results.items():
        summarize_results(f"Noise Robustness (noise={noise:.1f})", results)

    summarize_results("Path Composition (Multi-Hop)", path_results)

    print(f"\n  Total time: {total_time:.1f}s")
    print()

    # --- Analysis ---
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Check if DELTA leads on edge tasks
    delta_edge_mean = sum(edge_results['DELTA']) / len(edge_results['DELTA'])
    gps_edge_mean = sum(edge_results['GraphGPS']) / len(edge_results['GraphGPS'])
    grit_edge_mean = sum(edge_results['GRIT']) / len(edge_results['GRIT'])

    print(f"\n  Edge classification advantage:")
    print(f"    DELTA vs GraphGPS: {delta_edge_mean - gps_edge_mean:+.3f}")
    print(f"    DELTA vs GRIT:     {delta_edge_mean - grit_edge_mean:+.3f}")

    if delta_edge_mean > gps_edge_mean and delta_edge_mean > grit_edge_mean:
        print("  >> DELTA leads on edge classification ✓")
    elif delta_edge_mean > gps_edge_mean or delta_edge_mean > grit_edge_mean:
        print("  >> DELTA beats one baseline but not both — investigate.")
    else:
        print("  >> DELTA does not lead on edge classification — reframe contribution.")

    print()
    print("  NOTE: This uses synthetic data. For publication, rerun on full FB15k-237")
    print("  using Google Colab Pro (see COLAB_SETUP.md and notebooks/).")
    print()
    print("  Next steps:")
    print("    Phase 31: Mini-batching for full-scale FB15k-237")
    print("    Phase 34b: Rerun this comparison on real data with A100 GPU")


if __name__ == '__main__':
    main()
