"""
Phase 2: Validate Dual Parallel Attention

Core question: Does running node and edge attention in parallel (with
reconciliation) outperform either one alone?

Uses the same relation classification task as Phase 1, but now compares:
- Node attention only (from Phase 1)
- Edge attention only (from Phase 1)
- Sequential: node then edge (no parallelism)
- Dual parallel + reconciliation (full DELTA mechanism)

If dual parallel outperforms sequential, it validates that the reconciliation
step adds value — updated nodes inform edges and vice versa simultaneously.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import NodeAttention, EdgeAttention, DualParallelAttention
from delta.utils import create_analogy_task


class SequentialModel(nn.Module):
    """Node attention → edge attention → classify. No reconciliation."""
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.node_attn = NodeAttention(d_node, d_edge, num_heads)
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge * 2),
            nn.GELU(),
            nn.Linear(d_edge * 2, num_classes),
        )

    def forward(self, graph):
        # Sequential: node attention updates nodes, then edge attention uses updated nodes
        new_nodes = self.node_attn(graph)
        updated_graph = DeltaGraph(
            node_features=new_nodes,
            edge_features=graph.edge_features,
            edge_index=graph.edge_index,
        )
        edge_adj = updated_graph.build_edge_adjacency()
        new_edges = self.edge_attn(updated_graph, edge_adj=edge_adj)
        return self.classifier(new_edges)


class DualParallelModel(nn.Module):
    """Full DELTA dual parallel attention with reconciliation."""
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.dual_attn = DualParallelAttention(d_node, d_edge, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge * 2),
            nn.GELU(),
            nn.Linear(d_edge * 2, num_classes),
        )

    def forward(self, graph):
        edge_adj = graph.build_edge_adjacency()
        updated = self.dual_attn(graph, edge_adj=edge_adj)
        return self.classifier(updated.edge_features)


class StackedDualModel(nn.Module):
    """Multiple layers of dual parallel attention — deeper processing."""
    def __init__(self, d_node, d_edge, num_classes, num_layers=2, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([
            DualParallelAttention(d_node, d_edge, num_heads)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge * 2),
            nn.GELU(),
            nn.Linear(d_edge * 2, num_classes),
        )

    def forward(self, graph):
        edge_adj = graph.build_edge_adjacency()
        for layer in self.layers:
            graph = layer(graph, edge_adj=edge_adj)
        return self.classifier(graph.edge_features)


def train_and_evaluate(model, graph, labels, model_name, epochs=200, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
                      f"Train: {train_acc.item():.3f}  "
                      f"Test: {test_acc.item():.3f}")

    return best_test_acc


def main():
    print("=" * 70)
    print("PHASE 2: Dual Parallel Attention Validation")
    print("=" * 70)
    print()
    print("Task: Relation classification — same as Phase 1.")
    print("Question: Does dual parallel attention + reconciliation beat sequential?")
    print()

    d_node, d_edge = 64, 32
    num_patterns = 6
    instances = 8

    graph, labels = create_analogy_task(
        num_patterns=num_patterns,
        instances_per_pattern=instances,
        d_node=d_node, d_edge=d_edge, seed=42,
    )
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges\n")

    results = {}

    print("--- Sequential: Node → Edge (no reconciliation) ---")
    model = SequentialModel(d_node, d_edge, num_patterns)
    results['Sequential'] = train_and_evaluate(model, graph, labels, 'Sequential')
    print()

    print("--- Dual Parallel + Reconciliation (1 layer) ---")
    model = DualParallelModel(d_node, d_edge, num_patterns)
    results['Dual (1 layer)'] = train_and_evaluate(model, graph, labels, 'Dual-1')
    print()

    print("--- Dual Parallel + Reconciliation (2 layers) ---")
    model = StackedDualModel(d_node, d_edge, num_patterns, num_layers=2)
    results['Dual (2 layers)'] = train_and_evaluate(model, graph, labels, 'Dual-2')
    print()

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for name, acc in results.items():
        bar = "#" * int(acc * 40)
        print(f"  {name:<20s}  Best Test Acc: {acc:.3f}  {bar}")
    print()

    dual_vs_seq = results['Dual (1 layer)'] - results['Sequential']
    print(f"Dual parallel vs sequential: {dual_vs_seq:+.3f}")
    if dual_vs_seq > 0.02:
        print(">> Reconciliation adds value — bidirectional node-edge updates help.")
    else:
        print(">> Marginal difference — reconciliation may need tuning or a harder task.")


if __name__ == '__main__':
    main()
