"""
Phase 16: Post-Attention Pruning vs Pre-Attention Routing

Core question: Does pruning based on OBSERVED attention weights outperform
the old pre-attention Gumbel router that predicted importance before seeing
how elements actually participate in attention?

This directly addresses Pitfall #1 (router chicken-and-egg) and Pitfall #6
(Gumbel-softmax not improving accuracy).

Benchmark: Same FB15k-237-style KG as Phase 15, comparing:
1. Old pre-attention router (ImportanceRouter) at 50% sparsity — was ~65-75%
2. New post-attention pruner (PostAttentionPruner) at 50% sparsity
3. Post-attention pruner at 30% sparsity (aggressive)
4. Full attention baseline (no pruning) — should stay 100%

Key metric: accuracy at 50% sparsity should improve over Phase 15's result.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import EdgeAttention, DualParallelAttention
from delta.router import PostAttentionPruner, ImportanceRouter
from delta.utils import create_synthetic_kg_benchmark


class OldRouterModel(nn.Module):
    """Phase 15-style model: pre-attention scoring, then masked attention."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.router = ImportanceRouter(d_node, d_edge)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, k_ratio=0.5):
        node_scores, edge_scores = self.router(graph)
        _, edge_mask = self.router.apply_top_k(
            graph, node_scores, edge_scores,
            node_k_ratio=k_ratio, edge_k_ratio=k_ratio,
        )
        ef = graph.edge_features * edge_mask.float().unsqueeze(-1)
        g = DeltaGraph(node_features=graph.node_features, edge_features=ef,
                       edge_index=graph.edge_index, node_tiers=graph.node_tiers)
        edge_adj = g.build_edge_adjacency()
        edge_feats = self.edge_attn(g, edge_adj=edge_adj)
        return self.classifier(edge_feats)


class PostAttentionModel(nn.Module):
    """New approach: run full attention, observe weights, then prune for classification."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.dual_attn = DualParallelAttention(d_node, d_edge, num_heads)
        self.pruner = PostAttentionPruner(d_node, d_edge)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, k_ratio=0.5):
        edge_adj = graph.build_edge_adjacency()
        result, node_attn_w, edge_attn_w = self.dual_attn(
            graph, edge_adj=edge_adj, return_weights=True
        )
        # Score importance from observed attention
        node_scores, edge_scores = self.pruner.compute_importance(
            result, node_attn_w, edge_attn_w
        )
        # Prune: zero out unimportant edges
        _, edge_mask = self.pruner.prune(
            result, node_scores, edge_scores,
            node_k_ratio=k_ratio, edge_k_ratio=k_ratio,
        )
        pruned_edge_feats = result.edge_features * edge_mask.float().unsqueeze(-1)
        return self.classifier(pruned_edge_feats)


class FullAttentionModel(nn.Module):
    """Baseline: full attention, no pruning."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, **kwargs):
        edge_adj = graph.build_edge_adjacency()
        edge_feats = self.edge_attn(graph, edge_adj=edge_adj)
        return self.classifier(edge_feats)


def train_eval(model, graph, labels, train_idx, test_idx,
               epochs=200, lr=1e-3, **fwd_kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    num_rels = labels.max().item() + 1
    per_rel_best = None

    for epoch in range(epochs):
        model.train()
        logits = model(graph, **fwd_kwargs)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(graph, **fwd_kwargs)
                preds = logits.argmax(-1)
                acc = (preds[test_idx] == labels[test_idx]).float().mean().item()

                per_rel = {}
                for r in range(num_rels):
                    mask = labels[test_idx] == r
                    if mask.any():
                        per_rel[r] = (preds[test_idx][mask] == r).float().mean().item()

                if acc > best_acc:
                    best_acc = acc
                    per_rel_best = per_rel

                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}  Acc={acc:.3f}")

    return best_acc, per_rel_best


def main():
    print("=" * 70)
    print("PHASE 16: Post-Attention Pruning vs Pre-Attention Routing")
    print("=" * 70)
    print()
    print("Fix 1+6 validation: Does pruning based on OBSERVED attention")
    print("outperform the old pre-attention importance scoring?")
    print()

    d_node, d_edge = 64, 32
    torch.manual_seed(42)

    graph, labels, metadata = create_synthetic_kg_benchmark(
        num_entities=100, num_relations=10, num_triples=500,
        d_node=d_node, d_edge=d_edge,
    )
    num_classes = metadata['num_relations']
    train_idx = metadata['train_idx']
    test_idx = metadata['test_idx']

    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"Relations: {num_classes}, Train: {len(train_idx)}, Test: {len(test_idx)}")
    print()

    results = {}

    print("--- Full Attention (no pruning, upper bound) ---")
    torch.manual_seed(42)
    m = FullAttentionModel(d_node, d_edge, num_classes)
    acc, pr = train_eval(m, graph, labels, train_idx, test_idx)
    results['Full (no prune)'] = acc

    print("\n--- Old Pre-Attention Router @ 50% ---")
    torch.manual_seed(42)
    m = OldRouterModel(d_node, d_edge, num_classes)
    acc, pr = train_eval(m, graph, labels, train_idx, test_idx, k_ratio=0.5)
    results['Old Router 50%'] = acc

    print("\n--- Post-Attention Pruner @ 50% ---")
    torch.manual_seed(42)
    m = PostAttentionModel(d_node, d_edge, num_classes)
    acc, pr = train_eval(m, graph, labels, train_idx, test_idx, k_ratio=0.5)
    results['PostAttn 50%'] = acc

    print("\n--- Post-Attention Pruner @ 30% (aggressive) ---")
    torch.manual_seed(42)
    m = PostAttentionModel(d_node, d_edge, num_classes)
    acc, pr = train_eval(m, graph, labels, train_idx, test_idx, k_ratio=0.3)
    results['PostAttn 30%'] = acc

    # Summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<25s} {'Test Acc':>10s}")
    print(f"  {'-'*25} {'-'*10}")
    for name, acc in results.items():
        bar = '#' * int(acc * 40)
        print(f"  {name:<25s} {acc:>10.3f}  {bar}")

    old_50 = results['Old Router 50%']
    new_50 = results['PostAttn 50%']
    delta = new_50 - old_50
    print(f"\n  Post-attn 50% vs Old router 50%: {delta:+.3f}")
    if delta > 0:
        print("  >> Post-attention pruning outperforms pre-attention routing!")
    elif delta == 0:
        print("  >> Same accuracy — post-attention pruning matches pre-attention.")
    else:
        print("  >> Pre-attention routing still better at this scale.")


if __name__ == '__main__':
    main()
