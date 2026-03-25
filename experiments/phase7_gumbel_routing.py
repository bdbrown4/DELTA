"""
Phase 7: Differentiable Routing via Gumbel-Softmax

Core question: Does making the router differentiable (via Gumbel-softmax
straight-through estimator) let the router learn better selection policies
than the non-differentiable top-k baseline?

Compared:
1. Hard top-k (non-differentiable) — Phase 3 baseline
2. Gumbel-softmax with temperature annealing
3. Gumbel-softmax without annealing (fixed temperature)

Key metric: gradient flow through router params + accuracy at high sparsity.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import EdgeAttention
from delta.router import ImportanceRouter
from delta.utils import create_knowledge_graph


class RoutedEdgeClassifier(nn.Module):
    """Edge classifier with importance routing — supports both hard and Gumbel."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4, sparse_ratio=0.4):
        super().__init__()
        self.router = ImportanceRouter(d_node, d_edge)
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge),
            nn.GELU(),
            nn.Linear(d_edge, num_classes),
        )
        self.sparse_ratio = sparse_ratio

    def forward(self, graph, mode='hard', temperature=1.0):
        node_scores, edge_scores = self.router(graph)

        if mode == 'hard':
            # Non-differentiable top-k
            node_mask, edge_mask = self.router.apply_top_k(
                graph, node_scores, edge_scores,
                node_k_ratio=self.sparse_ratio,
                edge_k_ratio=self.sparse_ratio,
            )
            # Weight features by mask (no gradient to router)
            weighted_edge_feats = graph.edge_features * edge_mask.float().unsqueeze(-1)
        elif mode == 'gumbel':
            # Differentiable Gumbel-softmax
            node_weights, edge_weights = self.router.apply_top_k_gumbel(
                graph, node_scores, edge_scores,
                node_k_ratio=self.sparse_ratio,
                edge_k_ratio=self.sparse_ratio,
                temperature=temperature,
                hard=True,
            )
            weighted_edge_feats = graph.edge_features * edge_weights.unsqueeze(-1)
        else:
            # No routing — full attention baseline
            weighted_edge_feats = graph.edge_features

        # Create weighted graph
        weighted_graph = DeltaGraph(
            node_features=graph.node_features,
            edge_features=weighted_edge_feats,
            edge_index=graph.edge_index,
            node_tiers=graph.node_tiers,
        )

        edge_adj = weighted_graph.build_edge_adjacency()
        updated_edge_feats = self.edge_attn(weighted_graph, edge_adj=edge_adj)
        return self.classifier(updated_edge_feats)


def count_router_gradients(model):
    """Count how many router parameters received gradients."""
    has_grad = 0
    total = 0
    for name, param in model.named_parameters():
        if 'router' in name:
            total += 1
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad += 1
    return has_grad, total


def train_and_evaluate(model, graph, labels, train_mask, test_mask,
                       mode='hard', epochs=200, lr=1e-3,
                       anneal=False, temp_start=2.0, temp_end=0.1):
    """Train with specified routing mode, return test accuracy and gradient stats."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    grad_counts = []

    for epoch in range(epochs):
        model.train()

        # Temperature annealing schedule
        if anneal and mode == 'gumbel':
            progress = epoch / max(1, epochs - 1)
            temperature = temp_start * (temp_end / temp_start) ** progress
        else:
            temperature = 1.0

        logits = model(graph, mode=mode, temperature=temperature)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()

        # Track router gradients
        has_grad, total = count_router_gradients(model)
        grad_counts.append(has_grad)

        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(graph, mode=mode, temperature=temperature)
                acc = (logits[test_mask].argmax(-1) == labels[test_mask]).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}  "
                      f"Test Acc={acc:.3f}  "
                      f"Router grads: {has_grad}/{total}  "
                      f"{'τ=' + f'{temperature:.3f}' if mode == 'gumbel' else ''}")

    avg_grad = sum(grad_counts) / len(grad_counts) if grad_counts else 0
    return best_acc, avg_grad, grad_counts[-1] if grad_counts else 0


def main():
    print("=" * 70)
    print("PHASE 7: Differentiable Routing (Gumbel-Softmax)")
    print("=" * 70)
    print()
    print("Question: Does Gumbel-softmax let the router learn from task loss?")
    print(f"Sparsity: 60% pruned (only 40% of elements active)")
    print()

    d_node, d_edge = 64, 32
    num_classes = 6
    torch.manual_seed(42)

    graph, metadata = create_knowledge_graph(
        num_entities=40, num_relation_types=num_classes,
        edges_per_entity=4, d_node=d_node, d_edge=d_edge,
    )
    labels = torch.tensor(metadata['edge_labels'], dtype=torch.long)

    # Train/test split
    n_edges = graph.num_edges
    perm = torch.randperm(n_edges)
    train_end = int(n_edges * 0.7)
    train_mask = torch.zeros(n_edges, dtype=torch.bool)
    train_mask[perm[:train_end]] = True
    test_mask = ~train_mask

    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges, "
          f"{num_classes} relation types")
    print(f"Train: {train_mask.sum()}, Test: {test_mask.sum()}\n")

    results = {}

    # --- Full attention (no sparsity) ---
    print("--- Full Attention (no routing, baseline) ---")
    torch.manual_seed(42)
    model_full = RoutedEdgeClassifier(d_node, d_edge, num_classes, sparse_ratio=1.0)
    acc, avg_g, final_g = train_and_evaluate(model_full, graph, labels,
                                              train_mask, test_mask, mode='full')
    results['Full (no routing)'] = (acc, final_g)

    # --- Hard top-k ---
    print("\n--- Hard Top-K (non-differentiable routing) ---")
    torch.manual_seed(42)
    model_hard = RoutedEdgeClassifier(d_node, d_edge, num_classes, sparse_ratio=0.4)
    acc, avg_g, final_g = train_and_evaluate(model_hard, graph, labels,
                                              train_mask, test_mask, mode='hard')
    results['Hard top-k'] = (acc, final_g)

    # --- Gumbel (fixed temperature) ---
    print("\n--- Gumbel-Softmax (fixed τ=1.0) ---")
    torch.manual_seed(42)
    model_gumbel_fixed = RoutedEdgeClassifier(d_node, d_edge, num_classes, sparse_ratio=0.4)
    acc, avg_g, final_g = train_and_evaluate(model_gumbel_fixed, graph, labels,
                                              train_mask, test_mask, mode='gumbel',
                                              anneal=False)
    results['Gumbel (τ=1.0)'] = (acc, final_g)

    # --- Gumbel (annealed temperature) ---
    print("\n--- Gumbel-Softmax (annealed τ: 2.0 → 0.1) ---")
    torch.manual_seed(42)
    model_gumbel_anneal = RoutedEdgeClassifier(d_node, d_edge, num_classes, sparse_ratio=0.4)
    acc, avg_g, final_g = train_and_evaluate(model_gumbel_anneal, graph, labels,
                                              train_mask, test_mask, mode='gumbel',
                                              anneal=True, temp_start=2.0, temp_end=0.1)
    results['Gumbel (annealed)'] = (acc, final_g)

    # --- Summary ---
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    max_acc = max(v[0] for v in results.values())
    for name, (acc, grads) in results.items():
        bar = '#' * int(acc * 40)
        grad_info = f"  Router grads: {grads}" if 'routing' not in name.lower() else ""
        print(f"  {name:30s} Acc: {acc:.3f}  {bar}{grad_info}")

    hard_acc = results['Hard top-k'][0]
    gumbel_acc = results['Gumbel (annealed)'][0]
    print(f"\nGumbel (annealed) vs Hard top-k: {gumbel_acc - hard_acc:+.3f}")

    hard_grads = results['Hard top-k'][1]
    gumbel_grads = results['Gumbel (annealed)'][1]
    print(f"Router params with gradients — Hard: {hard_grads}, Gumbel: {gumbel_grads}")

    if gumbel_grads > hard_grads:
        print("\n>> Gumbel-softmax enables gradient flow through the router.")
        print(">> The router can now learn selection policies from task loss.")
    if gumbel_acc > hard_acc:
        print(">> Differentiable routing improves accuracy at same sparsity level.")
    elif gumbel_acc == hard_acc:
        print(">> Same accuracy — but Gumbel provides a path for learning at scale.")


if __name__ == '__main__':
    main()
