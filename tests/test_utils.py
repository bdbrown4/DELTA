"""Unit tests for utility functions."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from delta.graph import DeltaGraph
from delta.utils import calculate_graph_statistics


def test_calculate_graph_statistics_basic():
    """Test graph statistics calculation on a simple graph."""
    # Create a simple graph with 4 nodes and 3 edges
    # 0→1, 1→2, 2→3
    g = DeltaGraph(
        node_features=torch.randn(4, 16),
        edge_features=torch.randn(3, 8),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
    )
    
    stats = calculate_graph_statistics(g)
    
    # Verify basic counts
    assert stats['num_nodes'] == 4, f"Expected 4 nodes, got {stats['num_nodes']}"
    assert stats['num_edges'] == 3, f"Expected 3 edges, got {stats['num_edges']}"
    assert stats['node_feature_dim'] == 16, f"Expected node dim 16, got {stats['node_feature_dim']}"
    assert stats['edge_feature_dim'] == 8, f"Expected edge dim 8, got {stats['edge_feature_dim']}"
    
    # Verify degree statistics
    # Node 0: out-degree 1, in-degree 0 → total degree 1
    # Node 1: out-degree 1, in-degree 1 → total degree 2
    # Node 2: out-degree 1, in-degree 1 → total degree 2
    # Node 3: out-degree 0, in-degree 1 → total degree 1
    # Average: (1 + 2 + 2 + 1) / 4 = 1.5
    assert abs(stats['avg_degree'] - 1.5) < 1e-5, f"Expected avg degree 1.5, got {stats['avg_degree']}"
    assert stats['max_degree'] == 2, f"Expected max degree 2, got {stats['max_degree']}"
    assert stats['min_degree'] == 1, f"Expected min degree 1, got {stats['min_degree']}"
    
    # Verify density
    # For 4 nodes: possible edges = 4 * 3 = 12
    # Actual edges: 3
    # Density: 3/12 = 0.25
    expected_density = 3.0 / 12.0
    assert abs(stats['density'] - expected_density) < 1e-5, f"Expected density {expected_density}, got {stats['density']}"
    
    print("[PASS] test_calculate_graph_statistics_basic")


def test_calculate_graph_statistics_complete_graph():
    """Test statistics on a densely connected graph."""
    # Create a graph with 3 nodes, fully connected (excluding self-loops)
    # 0→1, 0→2, 1→0, 1→2, 2→0, 2→1
    g = DeltaGraph(
        node_features=torch.randn(3, 8),
        edge_features=torch.randn(6, 4),
        edge_index=torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]),
    )
    
    stats = calculate_graph_statistics(g)
    
    assert stats['num_nodes'] == 3
    assert stats['num_edges'] == 6
    
    # Each node has in-degree 2 and out-degree 2, total degree 4
    assert abs(stats['avg_degree'] - 4.0) < 1e-5, f"Expected avg degree 4.0, got {stats['avg_degree']}"
    assert stats['max_degree'] == 4
    assert stats['min_degree'] == 4
    
    # Density: 6 / (3 * 2) = 6 / 6 = 1.0 (fully connected)
    assert abs(stats['density'] - 1.0) < 1e-5, f"Expected density 1.0, got {stats['density']}"
    
    print("[PASS] test_calculate_graph_statistics_complete_graph")


def test_calculate_graph_statistics_isolated_node():
    """Test statistics when some nodes are isolated."""
    # Create a graph with 4 nodes, but node 3 is isolated
    # 0→1, 1→2
    g = DeltaGraph(
        node_features=torch.randn(4, 8),
        edge_features=torch.randn(2, 4),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
    )
    
    stats = calculate_graph_statistics(g)
    
    assert stats['num_nodes'] == 4
    assert stats['num_edges'] == 2
    
    # Node degrees: 0→1, 1→1+1=2, 2→1, 3→0
    # Average: (1 + 2 + 1 + 0) / 4 = 1.0
    assert abs(stats['avg_degree'] - 1.0) < 1e-5, f"Expected avg degree 1.0, got {stats['avg_degree']}"
    assert stats['max_degree'] == 2
    assert stats['min_degree'] == 0  # Isolated node
    
    print("[PASS] test_calculate_graph_statistics_isolated_node")


def test_calculate_graph_statistics_single_node():
    """Test statistics on a graph with a single node and no edges."""
    g = DeltaGraph(
        node_features=torch.randn(1, 8),
        edge_features=torch.randn(0, 4),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    
    stats = calculate_graph_statistics(g)
    
    assert stats['num_nodes'] == 1
    assert stats['num_edges'] == 0
    assert stats['avg_degree'] == 0.0
    assert stats['max_degree'] == 0
    assert stats['min_degree'] == 0
    assert stats['density'] == 0.0
    
    print("[PASS] test_calculate_graph_statistics_single_node")


if __name__ == '__main__':
    test_calculate_graph_statistics_basic()
    test_calculate_graph_statistics_complete_graph()
    test_calculate_graph_statistics_isolated_node()
    test_calculate_graph_statistics_single_node()
    print("\nAll utils tests passed.")
