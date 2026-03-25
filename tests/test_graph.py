"""Unit tests for DeltaGraph."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from delta.graph import DeltaGraph, TIER_HOT, TIER_WARM, TIER_COLD


def test_basic_properties():
    g = DeltaGraph(
        node_features=torch.randn(5, 16),
        edge_features=torch.randn(7, 8),
        edge_index=torch.randint(0, 5, (2, 7)),
    )
    assert g.num_nodes == 5
    assert g.num_edges == 7
    assert g.d_node == 16
    assert g.d_edge == 8
    assert g.node_tiers is not None
    assert (g.node_tiers == TIER_HOT).all()
    print("[PASS] test_basic_properties")


def test_tier_masks():
    g = DeltaGraph(
        node_features=torch.randn(6, 8),
        edge_features=torch.randn(3, 4),
        edge_index=torch.tensor([[0, 1, 2], [3, 4, 5]]),
        node_tiers=torch.tensor([0, 0, 1, 1, 2, 2]),
    )
    assert g.hot_mask().sum() == 2
    assert g.warm_mask().sum() == 2
    assert g.cold_mask().sum() == 2
    print("[PASS] test_tier_masks")


def test_subgraph():
    g = DeltaGraph(
        node_features=torch.randn(5, 8),
        edge_features=torch.randn(4, 4),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
    )
    mask = torch.tensor([True, True, True, False, False])
    sub = g.subgraph(mask)
    assert sub.num_nodes == 3
    assert sub.num_edges == 2  # edges 0->1 and 1->2
    print("[PASS] test_subgraph")


def test_edge_adjacency():
    # Triangle: 0→1, 1→2, 2→0
    g = DeltaGraph(
        node_features=torch.randn(3, 8),
        edge_features=torch.randn(3, 4),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
    )
    adj = g.build_edge_adjacency()
    # All edges share nodes, so all pairs should be adjacent
    assert adj.shape[1] == 6  # 3 edges × 2 neighbors each
    print("[PASS] test_edge_adjacency")


def test_neighbor_edges():
    g = DeltaGraph(
        node_features=torch.randn(4, 8),
        edge_features=torch.randn(3, 4),
        edge_index=torch.tensor([[0, 0, 2], [1, 2, 3]]),
    )
    # Edge 0 (0→1) and Edge 1 (0→2) share node 0
    neighbors = g.neighbor_edges(0)
    assert 1 in neighbors
    print("[PASS] test_neighbor_edges")


def test_to_device():
    g = DeltaGraph(
        node_features=torch.randn(3, 8),
        edge_features=torch.randn(2, 4),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
    )
    g2 = g.to(torch.device('cpu'))
    assert g2.device == torch.device('cpu')
    print("[PASS] test_to_device")


if __name__ == '__main__':
    test_basic_properties()
    test_tier_masks()
    test_subgraph()
    test_edge_adjacency()
    test_neighbor_edges()
    test_to_device()
    print("\nAll graph tests passed.")
