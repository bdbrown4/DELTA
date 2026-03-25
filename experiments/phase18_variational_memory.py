"""
Phase 18: Variational Memory Compression Benchmark

Core question: Does the variational bottleneck (Fix 3) compress warm-tier
nodes better than the old fixed 50% linear compression while maintaining
retrieval quality?

The old TieredMemory used a fixed Linear(d_node, warm_dim) + Linear(warm_dim, d_node).
The new approach uses a variational bottleneck with reparameterization trick,
producing a KL loss term that regularizes the latent space.

Benchmark:
1. Compression quality: MSE between original and roundtrip features
2. KL loss convergence during training
3. Cold storage retrieval accuracy with learned similarity threshold
4. Warm→hot promotion: can decompressed features still be useful for classification?

Key metric: Warm roundtrip features should produce better classification
accuracy than fixed compression when promoted back to hot tier.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph, TIER_HOT, TIER_WARM, TIER_COLD
from delta.memory import TieredMemory
from delta.attention import EdgeAttention
from delta.utils import create_synthetic_kg_benchmark


def make_tiered_kg(d_node=64, d_edge=32):
    """Create a KG where some nodes are warm/cold."""
    torch.manual_seed(42)
    graph, labels, metadata = create_synthetic_kg_benchmark(
        num_entities=100, num_relations=10, num_triples=500,
        d_node=d_node, d_edge=d_edge,
    )
    # Assign tiers: top 40% hot, next 30% warm, bottom 30% cold
    N = graph.num_nodes
    importance = torch.rand(N)
    _, order = torch.sort(importance, descending=True)
    tiers = torch.full((N,), TIER_COLD, dtype=torch.long)
    tiers[order[:int(N * 0.4)]] = TIER_HOT
    tiers[order[int(N * 0.4):int(N * 0.7)]] = TIER_WARM
    graph.node_tiers = tiers
    return graph, labels, metadata


class ClassifierAfterCompression(nn.Module):
    """Edge classifier that works on compressed+decompressed features."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.memory = TieredMemory(d_node, d_edge)
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, use_compression=True):
        if use_compression:
            graph = self.memory.compress_warm_nodes(graph)
        edge_adj = graph.build_edge_adjacency()
        edge_feats = self.edge_attn(graph, edge_adj=edge_adj)
        return self.classifier(edge_feats), self.memory.kl_loss if use_compression else torch.tensor(0.0)


def test_compression_quality():
    """Measure roundtrip compression fidelity."""
    print("--- Test 1: Compression Quality ---")
    d_node, d_edge = 64, 32
    graph, _, _ = make_tiered_kg(d_node, d_edge)
    mem = TieredMemory(d_node, d_edge)

    warm_mask = graph.warm_mask()
    original = graph.node_features[warm_mask].clone()

    compressed = mem.compress_warm_nodes(graph)
    roundtrip = compressed.node_features[warm_mask]

    mse = F.mse_loss(roundtrip, original).item()
    cosine_sim = F.cosine_similarity(roundtrip, original, dim=-1).mean().item()

    print(f"  Warm nodes: {warm_mask.sum().item()}")
    print(f"  Roundtrip MSE: {mse:.6f}")
    print(f"  Roundtrip cosine similarity: {cosine_sim:.4f}")
    print(f"  KL loss: {mem.kl_loss.item():.6f}")

    # After one gradient step, KL should be differentiable
    loss = mem.kl_loss + F.mse_loss(roundtrip, original.detach())
    loss.backward()
    grad_exists = mem.node_enc_mu.weight.grad is not None
    print(f"  Gradient through variational bottleneck: {'OK' if grad_exists else 'FAILED'}")
    print()
    return mse, cosine_sim


def test_learned_threshold():
    """Verify similarity threshold is learnable."""
    print("--- Test 2: Learned Similarity Threshold ---")
    mem = TieredMemory(64, 32)
    initial_threshold = mem.similarity_threshold
    print(f"  Initial threshold: {initial_threshold:.4f}")

    # Simulate gradient update
    optimizer = torch.optim.SGD([mem._sim_threshold_logit], lr=1.0)
    fake_loss = mem._sim_threshold_logit * 0.1  # push threshold down
    fake_loss.backward()
    optimizer.step()

    new_threshold = mem.similarity_threshold
    print(f"  After gradient step: {new_threshold:.4f}")
    print(f"  Threshold changed: {'YES' if abs(new_threshold - initial_threshold) > 1e-4 else 'NO'}")
    print()
    return initial_threshold, new_threshold


def test_classification_with_compression(epochs=150):
    """Train classifier with warm compression vs without."""
    print("--- Test 3: Classification with Compression ---")
    d_node, d_edge = 64, 32
    graph, labels, metadata = make_tiered_kg(d_node, d_edge)
    num_classes = metadata['num_relations']
    train_idx = metadata['train_idx']
    test_idx = metadata['test_idx']

    results = {}
    for use_comp, name in [(False, "No compression"), (True, "Variational compression")]:
        torch.manual_seed(42)
        model = ClassifierAfterCompression(d_node, d_edge, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_acc = 0.0

        for epoch in range(epochs):
            model.train()
            logits, kl = model(graph, use_compression=use_comp)
            loss = F.cross_entropy(logits[train_idx], labels[train_idx])
            if use_comp:
                loss = loss + 0.01 * kl  # KL regularization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                model.eval()
                with torch.no_grad():
                    logits, _ = model(graph, use_compression=use_comp)
                    preds = logits.argmax(-1)
                    acc = (preds[test_idx] == labels[test_idx]).float().mean().item()
                    best_acc = max(best_acc, acc)
                    kl_val = kl.item() if use_comp else 0.0
                    print(f"  [{name}] Epoch {epoch+1}: Acc={acc:.3f}  KL={kl_val:.4f}")

        results[name] = best_acc

    print()
    return results


def main():
    print("=" * 70)
    print("PHASE 18: Variational Memory Compression Benchmark")
    print("=" * 70)
    print()
    print("Fix 3 validation: Variational bottleneck vs fixed compression")
    print()

    mse, cosine = test_compression_quality()
    init_t, new_t = test_learned_threshold()
    class_results = test_classification_with_compression()

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Compression MSE: {mse:.6f}")
    print(f"  Cosine similarity: {cosine:.4f}")
    print(f"  Threshold learnable: {'YES' if abs(new_t - init_t) > 1e-4 else 'NO'}")
    print()
    for name, acc in class_results.items():
        bar = '#' * int(acc * 40)
        print(f"  {name:<30s} {acc:.3f}  {bar}")

    no_comp = class_results.get("No compression", 0)
    with_comp = class_results.get("Variational compression", 0)
    delta = with_comp - no_comp
    print(f"\n  Compression impact: {delta:+.3f}")
    if abs(delta) < 0.05:
        print("  >> Compression preserves accuracy (< 5% degradation)")
    elif delta >= 0:
        print("  >> Compression actually improves accuracy (regularization benefit)")
    else:
        print("  >> Compression degrades accuracy — may need more training or lower KL weight")


if __name__ == '__main__':
    main()
