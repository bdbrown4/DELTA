"""
Phase 3: Validate Importance Router

Core question: Can the learned router enable sparse attention (attending to
fewer nodes/edges) without degrading accuracy?

If yes: DELTA can scale to larger graphs because the router reduces the
effective N before attention fires — the biological "only 1-5% of neurons
fire at once" principle, validated computationally.

Test: Train on full attention, then progressively reduce the active fraction
via the router and measure accuracy degradation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import DualParallelAttention
from delta.router import ImportanceRouter
from delta.utils import create_knowledge_graph


class RouterModel(nn.Module):
    """DELTA with importance routing — sparse attention."""
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.router = ImportanceRouter(d_node, d_edge)
        self.dual_attn = DualParallelAttention(d_node, d_edge, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge * 2),
            nn.GELU(),
            nn.Linear(d_edge * 2, num_classes),
        )

    def forward(self, graph, sparse_ratio=1.0):
        node_scores, edge_scores = self.router(graph)
        graph.node_importance = node_scores
        graph.edge_importance = edge_scores

        # Apply sparse attention if ratio < 1
        node_mask = None
        if sparse_ratio < 1.0:
            node_mask, _ = self.router.apply_top_k(
                graph, node_scores, edge_scores,
                node_k_ratio=sparse_ratio,
                edge_k_ratio=sparse_ratio,
            )

        edge_adj = graph.build_edge_adjacency()
        updated = self.dual_attn(graph, edge_adj=edge_adj, node_mask=node_mask)
        return self.classifier(updated.edge_features)


class FullAttentionModel(nn.Module):
    """Baseline: dual attention without routing (attend to everything)."""
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.dual_attn = DualParallelAttention(d_node, d_edge, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge * 2),
            nn.GELU(),
            nn.Linear(d_edge * 2, num_classes),
        )

    def forward(self, graph, sparse_ratio=1.0):
        edge_adj = graph.build_edge_adjacency()
        updated = self.dual_attn(graph, edge_adj=edge_adj)
        return self.classifier(updated.edge_features)


def train_model(model, graph, labels, epochs=200, lr=1e-3, sparse_ratio=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    E = labels.shape[0]
    perm = torch.randperm(E)
    train_idx = perm[:int(E * 0.7)]
    test_idx = perm[int(E * 0.7):]

    best_test_acc = 0.0
    for epoch in range(epochs):
        model.train()
        logits = model(graph, sparse_ratio=sparse_ratio)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(graph, sparse_ratio=sparse_ratio)
                test_acc = (logits[test_idx].argmax(-1) == labels[test_idx]).float().mean()
                best_test_acc = max(best_test_acc, test_acc.item())

    return best_test_acc


def main():
    print("=" * 70)
    print("PHASE 3: Importance Router Validation")
    print("=" * 70)
    print()
    print("Question: Can sparse attention (via learned routing) maintain accuracy?")
    print()

    d_node, d_edge = 64, 32

    graph, metadata = create_knowledge_graph(
        num_entities=30, num_relation_types=5,
        edges_per_entity=4, d_node=d_node, d_edge=d_edge, seed=42,
    )
    labels = torch.tensor(metadata['edge_labels'], dtype=torch.long)
    num_classes = metadata['num_relation_types']
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges, "
          f"{num_classes} relation types\n")

    # Train full attention baseline
    print("--- Full Attention Baseline ---")
    baseline = FullAttentionModel(d_node, d_edge, num_classes)
    full_acc = train_model(baseline, graph, labels, epochs=200)
    print(f"  Full attention accuracy: {full_acc:.3f}\n")

    # Train with router at various sparsity levels
    sparsity_levels = [1.0, 0.8, 0.6, 0.4, 0.2]
    results = {'Full (no router)': full_acc}

    for ratio in sparsity_levels:
        name = f"Router ({int(ratio*100)}%)"
        print(f"--- {name} ---")
        model = RouterModel(d_node, d_edge, num_classes)
        acc = train_model(model, graph, labels, epochs=200, sparse_ratio=ratio)
        results[name] = acc
        print(f"  Accuracy: {acc:.3f}  "
              f"(delta vs full: {acc - full_acc:+.3f})\n")

    print("=" * 70)
    print("SPARSITY vs ACCURACY")
    print("=" * 70)
    for name, acc in results.items():
        bar = "#" * int(acc * 40)
        print(f"  {name:<20s}  {acc:.3f}  {bar}")
    print()

    # Find the sweet spot
    for ratio in [0.6, 0.4, 0.2]:
        key = f"Router ({int(ratio*100)}%)"
        if key in results:
            drop = full_acc - results[key]
            print(f"At {int(ratio*100)}% active: accuracy drops {drop:.3f} "
                  f"({int((1-ratio)*100)}% compute saved)")


if __name__ == '__main__':
    main()
