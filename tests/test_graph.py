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


def _bruteforce_2hop(edge_index):
    """Reference 2-hop edge adjacency: pattern of (A + A@A) off-diagonal,
    where A[i,j]=1 iff edges i,j share an endpoint."""
    E = edge_index.shape[1]
    A = torch.zeros(E, E)
    for i in range(E):
        for j in range(E):
            if i == j:
                continue
            si = set(edge_index[:, i].tolist())
            sj = set(edge_index[:, j].tolist())
            if si & sj:
                A[i, j] = 1.0
    comb = ((A + A @ A) > 0).float()
    comb.fill_diagonal_(0)
    return set(map(tuple, comb.nonzero(as_tuple=False).tolist()))


def test_edge_adjacency_2hop_matches_bruteforce():
    # Path graph: edges (0,1),(1,2),(2,3),(3,4). 2-hop must add (0,1)~(2,3)
    # and (1,2)~(3,4), which 1-hop does not contain.
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    g = DeltaGraph(node_features=torch.randn(5, 8),
                   edge_features=torch.randn(4, 4), edge_index=ei)
    adj1 = g.build_edge_adjacency(hops=1)
    g._edge_adj_cache = None
    adj2 = g.build_edge_adjacency(hops=2)
    s1 = set(map(tuple, adj1.t().tolist()))
    s2 = set(map(tuple, adj2.t().tolist()))
    # hops=2 must be a strict superset of hops=1 and match the brute-force set
    assert s1.issubset(s2)
    assert s2 == _bruteforce_2hop(ei)
    assert (0, 2) in s2 and (1, 3) in s2  # the genuinely-2-hop pairs
    # no self loops
    assert all(i != j for i, j in s2)
    print("[PASS] test_edge_adjacency_2hop_matches_bruteforce")


def test_edge_adjacency_2hop_random_graphs():
    # Randomized fuzz: new sparse-matmul 2-hop path must match brute force.
    torch.manual_seed(0)
    for _ in range(4):
        N = int(torch.randint(5, 11, (1,)))
        E = int(torch.randint(8, 25, (1,)))
        ei = torch.randint(0, N, (2, E))
        ei = ei[:, ei[0] != ei[1]]  # drop self-loop edges
        g = DeltaGraph(node_features=torch.randn(N, 4),
                       edge_features=torch.randn(ei.shape[1], 4), edge_index=ei)
        got = set(map(tuple, g.build_edge_adjacency(hops=2).t().tolist()))
        assert got == _bruteforce_2hop(ei)
    print("[PASS] test_edge_adjacency_2hop_random_graphs")


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
    test_edge_adjacency_2hop_matches_bruteforce()
    test_edge_adjacency_2hop_random_graphs()
    test_neighbor_edges()
    test_to_device()
    print("\nAll graph tests passed.")
