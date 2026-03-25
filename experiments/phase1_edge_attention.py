"""
Phase 1: Validate Edge-to-Edge Attention

Core question: Does edge-to-edge attention help the model discover relational
patterns that node-only attention misses?

Task: Relation type classification on a synthetic knowledge graph.
Given edges with noisy features, classify each edge's relation type.

Comparison:
- Baseline: MLP on raw edge features (no graph structure)
- Node-only: GAT-style node attention, then classify edges from endpoint features
- Edge-only: DELTA edge-to-edge attention, then classify edges
- Combined: Both node and edge attention

If edge attention shows meaningful improvement on relation classification,
the core DELTA thesis (edges as first-class citizens) holds.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import NodeAttention, EdgeAttention
from delta.utils import create_analogy_task


# --- Models ---

class BaselineMLP(nn.Module):
    """Baseline: classify edges from raw features only (no graph structure)."""
    def __init__(self, d_edge, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_edge, d_edge * 2),
            nn.GELU(),
            nn.Linear(d_edge * 2, num_classes),
        )

    def forward(self, graph):
        return self.mlp(graph.edge_features)


class NodeOnlyModel(nn.Module):
    """Node attention only — classify edges from updated endpoint features."""
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.node_attn = NodeAttention(d_node, d_edge, num_heads)
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * d_node + d_edge, d_edge),
            nn.GELU(),
            nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph):
        new_nodes = self.node_attn(graph)
        src, tgt = graph.edge_index
        edge_repr = torch.cat([
            new_nodes[src], new_nodes[tgt], graph.edge_features
        ], dim=-1)
        return self.edge_classifier(edge_repr)


class EdgeOnlyModel(nn.Module):
    """Edge attention only — DELTA's core novelty in isolation."""
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge * 2),
            nn.GELU(),
            nn.Linear(d_edge * 2, num_classes),
        )

    def forward(self, graph):
        edge_adj = graph.build_edge_adjacency()
        new_edges = self.edge_attn(graph, edge_adj=edge_adj)
        return self.classifier(new_edges)


class CombinedModel(nn.Module):
    """Both node and edge attention — preview of dual parallel."""
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.node_attn = NodeAttention(d_node, d_edge, num_heads)
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_node + d_edge, d_edge),
            nn.GELU(),
            nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph):
        new_nodes = self.node_attn(graph)
        edge_adj = graph.build_edge_adjacency()
        new_edges = self.edge_attn(graph, edge_adj=edge_adj)
        src, tgt = graph.edge_index
        combined = torch.cat([new_nodes[src], new_nodes[tgt], new_edges], dim=-1)
        return self.classifier(combined)


# --- Training ---

def train_and_evaluate(model, graph, labels, model_name, epochs=200, lr=1e-3):
    """Train a model and report accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train/test split on edges
    E = labels.shape[0]
    perm = torch.randperm(E)
    train_end = int(E * 0.7)
    train_idx = perm[:train_end]
    test_idx = perm[train_end:]

    best_test_acc = 0.0
    for epoch in range(epochs):
        model.train()
        logits = model(graph)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                logits = model(graph)
                train_acc = (logits[train_idx].argmax(-1) == labels[train_idx]).float().mean()
                test_acc = (logits[test_idx].argmax(-1) == labels[test_idx]).float().mean()
                best_test_acc = max(best_test_acc, test_acc.item())
                print(f"  [{model_name}] Epoch {epoch+1:4d}  "
                      f"Loss: {loss.item():.4f}  "
                      f"Train Acc: {train_acc.item():.3f}  "
                      f"Test Acc: {test_acc.item():.3f}")

    return best_test_acc


def main():
    print("=" * 70)
    print("PHASE 1: Edge-to-Edge Attention Validation")
    print("=" * 70)
    print()
    print("Task: Classify relation types in a synthetic knowledge graph.")
    print("Question: Does edge-to-edge attention discover relational patterns?")
    print()

    # Create dataset
    d_node, d_edge = 64, 32
    num_patterns = 6
    instances = 8

    graph, labels = create_analogy_task(
        num_patterns=num_patterns,
        instances_per_pattern=instances,
        d_node=d_node, d_edge=d_edge,
        seed=42,
    )
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges, "
          f"{num_patterns} relation types")
    print()

    results = {}

    # 1. Baseline MLP
    print("--- Baseline: MLP (no graph structure) ---")
    model = BaselineMLP(d_edge, num_patterns)
    results['MLP Baseline'] = train_and_evaluate(model, graph, labels, 'MLP')
    print()

    # 2. Node attention only
    print("--- Node Attention Only (GAT-style) ---")
    model = NodeOnlyModel(d_node, d_edge, num_patterns)
    results['Node Attention'] = train_and_evaluate(model, graph, labels, 'NodeAttn')
    print()

    # 3. Edge attention only (DELTA's novelty)
    print("--- Edge-to-Edge Attention (DELTA core) ---")
    model = EdgeOnlyModel(d_node, d_edge, num_patterns)
    results['Edge Attention'] = train_and_evaluate(model, graph, labels, 'EdgeAttn')
    print()

    # 4. Combined
    print("--- Combined Node + Edge Attention ---")
    model = CombinedModel(d_node, d_edge, num_patterns)
    results['Combined'] = train_and_evaluate(model, graph, labels, 'Combined')
    print()

    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for name, acc in results.items():
        bar = "#" * int(acc * 40)
        print(f"  {name:<20s}  Best Test Acc: {acc:.3f}  {bar}")
    print()

    edge_vs_baseline = results['Edge Attention'] - results['MLP Baseline']
    edge_vs_node = results['Edge Attention'] - results['Node Attention']
    print(f"Edge attention vs baseline:       {edge_vs_baseline:+.3f}")
    print(f"Edge attention vs node attention:  {edge_vs_node:+.3f}")
    print()

    if edge_vs_baseline > 0.05:
        print(">> Edge-to-edge attention shows meaningful improvement over baseline.")
        print(">> The core DELTA thesis appears to hold at small scale.")
    elif edge_vs_baseline > 0:
        print(">> Edge attention shows marginal improvement. May need larger graph or more epochs.")
    else:
        print(">> Edge attention did not outperform baseline on this task.")
        print(">> Consider: larger graph, different task, or architecture tuning.")


if __name__ == '__main__':
    main()
