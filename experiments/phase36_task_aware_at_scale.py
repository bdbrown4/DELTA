"""
Phase 36: Task-Aware Construction at Scale

Phase 33 showed no improvement from TaskAwareConstructor on 60-node graphs
(too small, already near-complete — new edges add noise, not signal).

This phase scales to 500-2000 nodes with deliberately sparse/missing edges:

  Experiment A — Sparse Path Graphs:
    Create path-composition KGs at 500/1000/2000 nodes with every 3rd edge
    removed (33% sparsity). The constructor must discover which missing
    edges are useful for multi-hop path reasoning.

  Experiment B — Cross-Cluster Graphs:
    5-10 clusters of 100-200 nodes each with sparse inter-cluster bridges
    (1-3 edges per cluster pair). Task requires cross-cluster inference.
    The constructor must create inter-cluster shortcuts.

  Experiment C — Edge Threshold Sweep:
    Test edge_threshold at {0.3, 0.1, 0.05} — Phase 33 used only 0.3.
    At 500+ nodes, 0.05 may be needed to generate enough new edges.

All experiments use Phase 31 mini-batching for large graphs.
Success criteria: AugmentedDELTA beats FixedTopologyDELTA by > 3% accuracy.

Requirements:
    - Phase 31 mini-batching (NeighborSampler)
    - Phase 33 TaskAwareConstructor
    - GPU recommended (2-4 hours on H100)
    - pip install torch numpy

Usage:
    python experiments/phase36_task_aware_at_scale.py [--min_nodes 500]
    python experiments/phase36_task_aware_at_scale.py --full
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.model import DELTAModel
from delta.utils import create_realistic_kg_benchmark
from experiments.phase31_mini_batching import NeighborSampler
from experiments.phase33_task_aware_construction import TaskAwareConstructor


# ---------------------------------------------------------------------------
# Fixed Topology DELTA (control)
# ---------------------------------------------------------------------------

class FixedTopologyDELTA(nn.Module):
    """DELTA with no constructor — uses the input graph topology as-is."""

    def __init__(self, d_node, d_edge, num_layers, num_heads, num_classes):
        super().__init__()
        self.model = DELTAModel(
            d_node=d_node, d_edge=d_edge, num_layers=num_layers,
            num_heads=num_heads, num_classes=num_classes,
        )

    def forward(self, graph):
        return self.model(graph)

    def classify_edges(self, encoded_graph):
        return self.model.classify_edges(encoded_graph)


# ---------------------------------------------------------------------------
# Augmented DELTA (TaskAwareConstructor + DELTA)
# ---------------------------------------------------------------------------

class AugmentedDELTA(nn.Module):
    """DELTA preceded by TaskAwareConstructor to learn new edges."""

    def __init__(self, d_node, d_edge, num_layers, num_heads, num_classes,
                 edge_threshold=0.3):
        super().__init__()
        self.constructor = TaskAwareConstructor(
            d_node=d_node, d_edge=d_edge,
            edge_threshold=edge_threshold,
        )
        self.model = DELTAModel(
            d_node=d_node, d_edge=d_edge, num_layers=num_layers,
            num_heads=num_heads, num_classes=num_classes,
        )

    def forward(self, graph):
        return self.model(self.constructor(graph))

    def classify_edges(self, encoded_graph):
        return self.model.classify_edges(encoded_graph)


# ---------------------------------------------------------------------------
# Graph Generators
# ---------------------------------------------------------------------------

def create_sparse_path_graph(num_nodes, num_relations, d_node, d_edge,
                             sparsity=0.33, seed=42):
    """Create path-composition KG with deliberate sparsity.

    Every `1 / sparsity`-th edge is removed, forcing the constructor
    to discover which missing edges aid multi-hop reasoning.
    """
    graph, labels, metadata = create_realistic_kg_benchmark(
        num_entities=num_nodes,
        num_triples=num_nodes * 5,
        d_node=d_node, d_edge=d_edge,
        seed=seed,
    )

    # Remove every Kth edge to introduce sparsity
    E = graph.edge_index.shape[1]
    K = max(1, int(1.0 / sparsity))
    keep_mask = torch.ones(E, dtype=torch.bool)
    keep_mask[::K] = False

    sparse_graph = DeltaGraph(
        node_features=graph.node_features,
        edge_features=graph.edge_features[keep_mask],
        edge_index=graph.edge_index[:, keep_mask],
        node_tiers=graph.node_tiers,
        node_importance=graph.node_importance,
        edge_importance=graph.edge_importance[keep_mask] if graph.edge_importance is not None else None,
    )

    sparse_labels = labels[keep_mask]
    removed_count = E - keep_mask.sum().item()
    num_rel = metadata['num_relations']

    return sparse_graph, sparse_labels, num_rel, removed_count


def create_cross_cluster_graph(num_clusters, nodes_per_cluster, num_relations,
                               d_node, d_edge, bridges_per_pair=2, seed=42):
    """Create multi-cluster KG with sparse inter-cluster bridges.

    Each cluster is a small dense KG. Clusters are connected by a few
    bridge edges. Task requires cross-cluster relational inference.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    all_node_feats = []
    all_edge_feats = []
    all_edge_index = []
    all_labels = []
    node_offset = 0

    # Create intra-cluster graphs
    for c in range(num_clusters):
        graph, labels, metadata = create_realistic_kg_benchmark(
            num_entities=nodes_per_cluster,
            num_triples=nodes_per_cluster * 3,
            d_node=d_node, d_edge=d_edge,
            seed=seed + c * 100,
        )
        all_node_feats.append(graph.node_features)
        all_edge_feats.append(graph.edge_features)
        all_edge_index.append(graph.edge_index + node_offset)
        all_labels.append(labels % num_relations)
        node_offset += graph.num_nodes

    total_nodes = node_offset

    # Create sparse inter-cluster bridges
    bridge_edge_index = []
    bridge_edge_feats = []
    bridge_labels = []

    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            for _ in range(bridges_per_pair):
                src_node = random.randint(
                    i * nodes_per_cluster,
                    min((i + 1) * nodes_per_cluster - 1, total_nodes - 1))
                tgt_node = random.randint(
                    j * nodes_per_cluster,
                    min((j + 1) * nodes_per_cluster - 1, total_nodes - 1))
                bridge_edge_index.append([src_node, tgt_node])
                feat = torch.randn(d_edge)
                bridge_edge_feats.append(feat)
                bridge_labels.append(random.randint(0, num_relations - 1))

    if bridge_edge_index:
        bridge_ei = torch.tensor(bridge_edge_index, dtype=torch.long).t()
        bridge_ef = torch.stack(bridge_edge_feats)
        bridge_lbl = torch.tensor(bridge_labels, dtype=torch.long)

        all_edge_index.append(bridge_ei)
        all_edge_feats.append(bridge_ef)
        all_labels.append(bridge_lbl)

    # Combine
    combined_node_feats = torch.cat(all_node_feats, dim=0)
    combined_edge_feats = torch.cat(all_edge_feats, dim=0)
    combined_edge_index = torch.cat(all_edge_index, dim=1)
    combined_labels = torch.cat(all_labels, dim=0)

    combined_graph = DeltaGraph(
        node_features=combined_node_feats,
        edge_features=combined_edge_feats,
        edge_index=combined_edge_index,
        node_tiers=None,
        node_importance=None,
        edge_importance=None,
    )

    num_bridges = len(bridge_edge_index)
    return combined_graph, combined_labels, num_relations, num_bridges


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(model, graph, labels, epochs, lr, device, label='model',
                       log_every=50):
    """Train model and report accuracy. Returns dict with metrics."""
    model = model.to(device)
    graph = graph.to(device)
    labels = labels.to(device)

    E = labels.shape[0]
    perm = torch.randperm(E, device=device)
    train_idx = perm[:int(E * 0.7)]
    val_idx = perm[int(E * 0.7):int(E * 0.85)]
    test_idx = perm[int(E * 0.85):]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    best_test_acc = 0.0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        out = model(graph)
        logits = model.classify_edges(out)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                out = model(graph)
                logits = model.classify_edges(out)
                val_acc = (logits[val_idx].argmax(-1) == labels[val_idx]).float().mean().item()
                test_acc = (logits[test_idx].argmax(-1) == labels[test_idx]).float().mean().item()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc

                print(f"    [{label}] Epoch {epoch+1:3d}  "
                      f"Loss: {loss.item():.4f}  "
                      f"Val: {val_acc:.3f}  "
                      f"Test: {test_acc:.3f}")

    training_time = time.time() - start
    total_params = sum(p.numel() for p in model.parameters())

    return {
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc,
        'training_time_s': training_time,
        'total_params': total_params,
    }


# ---------------------------------------------------------------------------
# Experiment Runners
# ---------------------------------------------------------------------------

def experiment_a_sparse_paths(node_sizes, num_relations, d_node, d_edge,
                              epochs, lr, device, edge_threshold=0.1):
    """Experiment A: Sparse path graphs at increasing scale."""
    print("=" * 70)
    print("EXPERIMENT A: Sparse Path Graphs")
    print("=" * 70)
    print(f"  Sizes: {node_sizes}")
    print(f"  33% edges removed, edge_threshold={edge_threshold}")
    print()

    results = {}
    for N in node_sizes:
        print(f"  --- {N} nodes ---")
        graph, labels, num_rel, removed = create_sparse_path_graph(
            N, num_relations, d_node, d_edge, sparsity=0.33, seed=42)
        print(f"  {graph.num_nodes} nodes, {graph.num_edges} edges "
              f"({removed} removed)")

        # Fixed topology baseline
        fixed = FixedTopologyDELTA(d_node, d_edge, 3, 4, num_rel)
        fixed_result = train_and_evaluate(
            fixed, graph, labels, epochs, lr, device, label=f'Fixed-{N}')

        # Augmented DELTA
        aug = AugmentedDELTA(d_node, d_edge, 3, 4, num_rel,
                             edge_threshold=edge_threshold)
        aug_result = train_and_evaluate(
            aug, graph, labels, epochs, lr, device, label=f'Aug-{N}')

        delta = aug_result['best_test_acc'] - fixed_result['best_test_acc']
        results[N] = {
            'fixed': fixed_result,
            'augmented': aug_result,
            'delta': delta,
        }
        print(f"  Fixed: {fixed_result['best_test_acc']:.3f}  "
              f"Aug: {aug_result['best_test_acc']:.3f}  "
              f"Δ: {delta:+.3f}")
        print()

    return results


def experiment_b_cross_cluster(cluster_configs, num_relations, d_node, d_edge,
                               epochs, lr, device, edge_threshold=0.1):
    """Experiment B: Cross-cluster reasoning."""
    print("=" * 70)
    print("EXPERIMENT B: Cross-Cluster Reasoning")
    print("=" * 70)

    results = {}
    for cfg in cluster_configs:
        nc, npc = cfg['num_clusters'], cfg['nodes_per_cluster']
        bridges = cfg.get('bridges_per_pair', 2)
        total = nc * npc
        key = f'{nc}c_{npc}n'
        print(f"  --- {nc} clusters × {npc} nodes ({total} total), "
              f"{bridges} bridges ---")

        graph, labels, num_rel, num_bridges = create_cross_cluster_graph(
            nc, npc, num_relations, d_node, d_edge,
            bridges_per_pair=bridges, seed=42)
        print(f"  {graph.num_nodes} nodes, {graph.num_edges} edges, "
              f"{num_bridges} bridge edges")

        # Fixed topology baseline
        fixed = FixedTopologyDELTA(d_node, d_edge, 3, 4, num_rel)
        fixed_result = train_and_evaluate(
            fixed, graph, labels, epochs, lr, device, label=f'Fixed-{key}')

        # Augmented DELTA
        aug = AugmentedDELTA(d_node, d_edge, 3, 4, num_rel,
                             edge_threshold=edge_threshold)
        aug_result = train_and_evaluate(
            aug, graph, labels, epochs, lr, device, label=f'Aug-{key}')

        delta = aug_result['best_test_acc'] - fixed_result['best_test_acc']
        results[key] = {
            'fixed': fixed_result,
            'augmented': aug_result,
            'delta': delta,
        }
        print(f"  Fixed: {fixed_result['best_test_acc']:.3f}  "
              f"Aug: {aug_result['best_test_acc']:.3f}  "
              f"Δ: {delta:+.3f}")
        print()

    return results


def experiment_c_threshold_sweep(num_nodes, num_relations, d_node, d_edge,
                                 epochs, lr, device,
                                 thresholds=(0.3, 0.1, 0.05)):
    """Experiment C: Edge threshold sweep."""
    print("=" * 70)
    print("EXPERIMENT C: Edge Threshold Sweep")
    print("=" * 70)
    print(f"  {num_nodes} nodes, thresholds: {thresholds}")
    print()

    graph, labels, num_rel, removed = create_sparse_path_graph(
        num_nodes, num_relations, d_node, d_edge, sparsity=0.33, seed=42)
    print(f"  {graph.num_nodes} nodes, {graph.num_edges} edges "
          f"({removed} removed)")

    # Fixed baseline (run once)
    fixed = FixedTopologyDELTA(d_node, d_edge, 3, 4, num_rel)
    fixed_result = train_and_evaluate(
        fixed, graph, labels, epochs, lr, device, label='Fixed')
    print(f"  Fixed baseline: {fixed_result['best_test_acc']:.3f}")
    print()

    results = {'fixed': fixed_result}
    for t in thresholds:
        print(f"  --- threshold={t} ---")
        aug = AugmentedDELTA(d_node, d_edge, 3, 4, num_rel, edge_threshold=t)
        aug_result = train_and_evaluate(
            aug, graph, labels, epochs, lr, device, label=f'Aug-t{t}')
        delta = aug_result['best_test_acc'] - fixed_result['best_test_acc']
        results[t] = {
            'augmented': aug_result,
            'delta': delta,
        }
        print(f"  threshold={t}: {aug_result['best_test_acc']:.3f}  "
              f"Δ: {delta:+.3f}")
        print()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 36: Task-Aware Construction at Scale")
    parser.add_argument('--min_nodes', type=int, default=500)
    parser.add_argument('--max_nodes', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--full', action='store_true',
                        help='Full-scale (500 → 5000 nodes)')
    parser.add_argument('--skip_a', action='store_true')
    parser.add_argument('--skip_b', action='store_true')
    parser.add_argument('--skip_c', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    d_node, d_edge = 64, 32
    num_relations = 20

    if args.full:
        node_sizes = [500, 1000, 2000, 5000]
        cluster_configs = [
            {'num_clusters': 5, 'nodes_per_cluster': 100, 'bridges_per_pair': 2},
            {'num_clusters': 5, 'nodes_per_cluster': 200, 'bridges_per_pair': 2},
            {'num_clusters': 10, 'nodes_per_cluster': 200, 'bridges_per_pair': 3},
        ]
    else:
        node_sizes = [args.min_nodes, min(args.min_nodes * 2, args.max_nodes)]
        cluster_configs = [
            {'num_clusters': 5, 'nodes_per_cluster': 100, 'bridges_per_pair': 2},
        ]

    print("=" * 70)
    print("PHASE 36: Task-Aware Construction at Scale")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}")
    print(f"  Node sizes: {node_sizes}")
    print()

    all_results = {}

    # Experiment A
    if not args.skip_a:
        all_results['A'] = experiment_a_sparse_paths(
            node_sizes, num_relations, d_node, d_edge,
            args.epochs, args.lr, device, edge_threshold=0.1)

    # Experiment B
    if not args.skip_b:
        all_results['B'] = experiment_b_cross_cluster(
            cluster_configs, num_relations, d_node, d_edge,
            args.epochs, args.lr, device, edge_threshold=0.1)

    # Experiment C
    if not args.skip_c:
        sweep_nodes = node_sizes[0]
        all_results['C'] = experiment_c_threshold_sweep(
            sweep_nodes, num_relations, d_node, d_edge,
            args.epochs, args.lr, device,
            thresholds=(0.3, 0.1, 0.05))

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("=" * 70)
    print("PHASE 36 RESULTS SUMMARY")
    print("=" * 70)

    if 'A' in all_results:
        print("\n  Experiment A: Sparse Path Graphs")
        for N, res in all_results['A'].items():
            print(f"    {N:5d} nodes — Fixed: {res['fixed']['best_test_acc']:.3f}  "
                  f"Aug: {res['augmented']['best_test_acc']:.3f}  "
                  f"Δ: {res['delta']:+.3f}"
                  f"{'  ✓' if res['delta'] > 0.03 else ''}")

    if 'B' in all_results:
        print("\n  Experiment B: Cross-Cluster Reasoning")
        for key, res in all_results['B'].items():
            print(f"    {key:10s} — Fixed: {res['fixed']['best_test_acc']:.3f}  "
                  f"Aug: {res['augmented']['best_test_acc']:.3f}  "
                  f"Δ: {res['delta']:+.3f}"
                  f"{'  ✓' if res['delta'] > 0.03 else ''}")

    if 'C' in all_results:
        print("\n  Experiment C: Edge Threshold Sweep")
        fixed_acc = all_results['C']['fixed']['best_test_acc']
        print(f"    Fixed baseline: {fixed_acc:.3f}")
        for t in [0.3, 0.1, 0.05]:
            if t in all_results['C']:
                res = all_results['C'][t]
                print(f"    threshold={t:.2f}  — Aug: {res['augmented']['best_test_acc']:.3f}  "
                      f"Δ: {res['delta']:+.3f}"
                      f"{'  ✓' if res['delta'] > 0.03 else ''}")

    # Overall verdict
    a_wins = sum(1 for r in all_results.get('A', {}).values() if r['delta'] > 0.03)
    b_wins = sum(1 for r in all_results.get('B', {}).values() if r['delta'] > 0.03)
    total_configs = len(all_results.get('A', {})) + len(all_results.get('B', {}))
    print(f"\n  Constructor wins (> 3% improvement): {a_wins + b_wins}/{total_configs}")

    if a_wins + b_wins > total_configs // 2:
        print("  >> TaskAwareConstructor is effective at scale. ✓")
    else:
        print("  >> Constructor shows limited benefit. Consider:")
        print("     - Lower edge thresholds (see Experiment C)")
        print("     - Constructor architecture changes")
        print("     - The graph topology may already be sufficient")


if __name__ == '__main__':
    main()
