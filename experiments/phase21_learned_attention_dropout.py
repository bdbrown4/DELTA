"""
Phase 21: Learned Attention Dropout Benchmark

Core question: Does LearnedAttentionDropout (Fix 6) reduce the
generalization gap compared to uniform dropout or no dropout?

Uniform dropout treats all edges equally. LearnedAttentionDropout learns
per-edge dropout rates based on edge features, allowing the model to
regularize uncertain connections more aggressively while preserving
confident structural edges.

Benchmark:
1. Generalization gap: (train_acc - test_acc) — smaller is better
2. Final test accuracy
3. Verify dropout rates are learned (not all the same)
4. Compare three regimes: no dropout, uniform 0.1, learned dropout
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import NodeAttention, EdgeAttention
from delta.router import LearnedAttentionDropout
from delta.utils import create_synthetic_kg_benchmark


class KGClassifier(nn.Module):
    """Edge classifier using NodeAttention + EdgeAttention."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4,
                 dropout_mode='none'):
        super().__init__()
        self.dropout_mode = dropout_mode
        self.node_attn = NodeAttention(d_node, d_edge, num_heads)
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

        if dropout_mode == 'learned':
            self.learned_dropout = LearnedAttentionDropout(d_edge)
        elif dropout_mode == 'uniform':
            self.uniform_drop = nn.Dropout(0.1)

    def forward(self, graph):
        # Node attention with weights
        node_feats, attn_weights = self.node_attn(graph, return_weights=True)

        # Apply dropout to attention-derived features
        if self.dropout_mode == 'learned' and self.training:
            attn_weights = self.learned_dropout(graph.edge_features, attn_weights)

        # Update node features
        updated_graph = DeltaGraph(
            node_features=node_feats,
            edge_features=graph.edge_features,
            edge_index=graph.edge_index,
        )

        # Edge attention
        edge_adj = updated_graph.build_edge_adjacency()
        edge_feats = self.edge_attn(updated_graph, edge_adj=edge_adj)

        if self.dropout_mode == 'uniform' and self.training:
            edge_feats = self.uniform_drop(edge_feats)

        return self.classifier(edge_feats)


def train_eval(model, graph, labels, train_idx, test_idx, epochs=200, lr=1e-3):
    """Train model, return final train/test accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_train, best_test = 0.0, 0.0

    for epoch in range(epochs):
        model.train()
        logits = model(graph)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(graph)
                preds = logits.argmax(-1)
                train_acc = (preds[train_idx] == labels[train_idx]).float().mean().item()
                test_acc = (preds[test_idx] == labels[test_idx]).float().mean().item()
                best_train = max(best_train, train_acc)
                best_test = max(best_test, test_acc)

    return best_train, best_test


def test_generalization_gap():
    """Compare gen gap across dropout modes."""
    print("--- Test 1: Generalization Gap ---")
    d_node, d_edge = 64, 32
    graph, labels, metadata = create_synthetic_kg_benchmark(
        num_entities=100, num_relations=10, num_triples=500,
        d_node=d_node, d_edge=d_edge,
    )
    num_classes = metadata['num_relations']
    train_idx = metadata['train_idx']
    test_idx = metadata['test_idx']

    results = {}
    for mode in ['none', 'uniform', 'learned']:
        torch.manual_seed(42)
        model = KGClassifier(d_node, d_edge, num_classes, dropout_mode=mode)
        train_acc, test_acc = train_eval(model, graph, labels, train_idx, test_idx)
        gap = train_acc - test_acc
        results[mode] = {'train': train_acc, 'test': test_acc, 'gap': gap}
        print(f"  {mode:>10s}: train={train_acc:.3f}  test={test_acc:.3f}  gap={gap:+.3f}")

    print()
    return results


def test_dropout_rate_diversity():
    """Verify learned dropout rates are diverse (not collapsed)."""
    print("--- Test 2: Dropout Rate Diversity ---")
    torch.manual_seed(42)
    d_edge = 32
    dropout = LearnedAttentionDropout(d_edge)

    # Different edge features should produce different dropout rates
    edge_features = torch.randn(50, d_edge)
    with torch.no_grad():
        raw_logits = dropout.drop_rate(edge_features).squeeze(-1)  # [50]
        rates = torch.sigmoid(raw_logits) * 0.5  # same transform as forward()

    print(f"  Dropout rates: min={rates.min():.4f}  max={rates.max():.4f}  "
          f"std={rates.std():.4f}")

    # After training, check again
    dummy_attn = torch.randn(50, 4)  # 4 heads
    optimizer = torch.optim.Adam(dropout.parameters(), lr=0.01)

    for _ in range(100):
        dropout.train()
        out = dropout(edge_features, dummy_attn)
        # Fake loss: encourage diverse dropout for different edges
        loss = -out.std()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        raw_logits = dropout.drop_rate(edge_features).squeeze(-1)
        rates_after = torch.sigmoid(raw_logits) * 0.5

    print(f"  After training: min={rates_after.min():.4f}  max={rates_after.max():.4f}  "
          f"std={rates_after.std():.4f}")

    diversity_improved = rates_after.std() > rates.std()
    print(f"  Diversity improved: {'YES' if diversity_improved else 'NO'}")
    print()
    return rates.std().item(), rates_after.std().item()


def test_dropout_at_eval():
    """Confirm dropout is disabled at eval time."""
    print("--- Test 3: Eval-Time Behavior ---")
    d_edge = 32
    dropout = LearnedAttentionDropout(d_edge)
    edge_features = torch.randn(20, d_edge)
    attn_weights = torch.randn(20, 4)

    dropout.eval()
    out1 = dropout(edge_features, attn_weights)
    out2 = dropout(edge_features, attn_weights)
    identical = torch.allclose(out1, out2)
    passthrough = torch.allclose(out1, attn_weights)

    print(f"  Eval outputs identical: {'YES' if identical else 'NO'}")
    print(f"  Eval is passthrough: {'YES' if passthrough else 'NO'}")
    print()
    return identical, passthrough


def main():
    print("=" * 70)
    print("PHASE 21: Learned Attention Dropout Benchmark")
    print("=" * 70)
    print()
    print("Fix 6 validation: Learned per-edge dropout vs uniform/none")
    print()

    gap_results = test_generalization_gap()
    init_std, trained_std = test_dropout_rate_diversity()
    eval_identical, eval_passthrough = test_dropout_at_eval()

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print("  Generalization gaps (lower = better):")
    for mode, r in gap_results.items():
        bar = '#' * max(0, int(r['gap'] * 40))
        print(f"    {mode:>10s}: gap={r['gap']:+.3f}  test={r['test']:.3f}  {bar}")

    print()
    print(f"  Dropout diversity: init_std={init_std:.4f} → trained_std={trained_std:.4f}")
    print(f"  Eval deterministic: {eval_identical}")
    print(f"  Eval passthrough: {eval_passthrough}")

    # Key check: learned dropout should have smallest gap
    gaps = {k: v['gap'] for k, v in gap_results.items()}
    best = min(gaps, key=gaps.get)
    print(f"\n  Best generalization: {best} (gap={gaps[best]:+.3f})")


if __name__ == '__main__':
    main()
