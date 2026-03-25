"""Unit tests for tiered memory."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from delta.graph import DeltaGraph, TIER_HOT, TIER_WARM, TIER_COLD
from delta.memory import TieredMemory


def make_tiered_graph():
    N, d_node, d_edge = 12, 32, 16
    E = 15
    edge_index = torch.stack([torch.randint(0, N, (E,)), torch.randint(0, N, (E,))])
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    E = edge_index.shape[1]

    tiers = torch.zeros(N, dtype=torch.long)
    tiers[4:8] = TIER_WARM
    tiers[8:] = TIER_COLD

    return DeltaGraph(
        node_features=torch.randn(N, d_node),
        edge_features=torch.randn(E, d_edge),
        edge_index=edge_index,
        node_tiers=tiers,
    )


def test_compress_warm():
    g = make_tiered_graph()
    mem = TieredMemory(32, 16)
    compressed = mem.compress_warm_nodes(g)
    assert compressed.num_nodes == g.num_nodes
    # Should have KL loss from variational bottleneck
    assert hasattr(mem, 'kl_loss')
    assert mem.kl_loss.item() >= 0
    print("[PASS] test_compress_warm")


def test_active_subgraph():
    g = make_tiered_graph()
    mem = TieredMemory(32, 16)
    active = mem.get_active_subgraph(g)
    # Should have 8 nodes (4 hot + 4 warm), cold excluded
    assert active.num_nodes == 8
    print("[PASS] test_active_subgraph")


def test_cold_retrieval():
    g = make_tiered_graph()
    mem = TieredMemory(32, 16)
    query = torch.randn(32)
    feats, indices = mem.retrieve_from_cold(g, query, top_k=2)
    assert feats.shape[0] == 2
    assert indices.shape[0] == 2
    # Indices should be in the cold range (8-11)
    assert (indices >= 8).all()
    print("[PASS] test_cold_retrieval")


def test_empty_cold_retrieval():
    g = DeltaGraph(
        node_features=torch.randn(5, 32),
        edge_features=torch.randn(3, 16),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        node_tiers=torch.zeros(5, dtype=torch.long),  # all hot
    )
    mem = TieredMemory(32, 16)
    feats, indices = mem.retrieve_from_cold(g, torch.randn(32))
    assert feats.shape[0] == 0
    print("[PASS] test_empty_cold_retrieval")


def test_learned_threshold():
    mem = TieredMemory(32, 16)
    # Default should be approximately 0.85 (sigmoid(1.7))
    threshold = mem.similarity_threshold
    assert 0.8 < threshold < 0.9, f"Expected ~0.85, got {threshold}"
    # It should be a learnable parameter
    assert mem._sim_threshold_logit.requires_grad
    print("[PASS] test_learned_threshold")


def test_kl_loss_no_warm():
    """When no warm nodes, KL loss should be 0."""
    g = DeltaGraph(
        node_features=torch.randn(5, 32),
        edge_features=torch.randn(3, 16),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        node_tiers=torch.zeros(5, dtype=torch.long),  # all hot
    )
    mem = TieredMemory(32, 16)
    mem.compress_warm_nodes(g)
    assert mem.kl_loss.item() == 0.0
    print("[PASS] test_kl_loss_no_warm")


if __name__ == '__main__':
    test_compress_warm()
    test_active_subgraph()
    test_cold_retrieval()
    test_empty_cold_retrieval()
    test_learned_threshold()
    test_kl_loss_no_warm()
    print("\nAll memory tests passed.")
