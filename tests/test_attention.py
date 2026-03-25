"""Unit tests for attention modules."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from delta.graph import DeltaGraph
from delta.attention import NodeAttention, EdgeAttention, DualParallelAttention


def make_test_graph(N=10, E=15, d_node=32, d_edge=16):
    edge_index = torch.stack([
        torch.randint(0, N, (E,)),
        torch.randint(0, N, (E,)),
    ])
    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    E = edge_index.shape[1]

    return DeltaGraph(
        node_features=torch.randn(N, d_node),
        edge_features=torch.randn(E, d_edge),
        edge_index=edge_index,
    )


def test_node_attention():
    g = make_test_graph()
    attn = NodeAttention(32, 16, num_heads=4)
    out = attn(g)
    assert out.shape == g.node_features.shape
    print("[PASS] test_node_attention")


def test_edge_attention():
    g = make_test_graph()
    attn = EdgeAttention(16, 32, num_heads=4)
    out = attn(g)
    assert out.shape == g.edge_features.shape
    print("[PASS] test_edge_attention")


def test_dual_parallel():
    g = make_test_graph()
    dual = DualParallelAttention(32, 16, num_heads=4)
    out = dual(g)
    assert out.node_features.shape == g.node_features.shape
    assert out.edge_features.shape == g.edge_features.shape
    print("[PASS] test_dual_parallel")


def test_gradient_through_attention():
    g = make_test_graph()
    g.node_features.requires_grad_(True)
    g.edge_features.requires_grad_(True)

    dual = DualParallelAttention(32, 16, num_heads=4)
    out = dual(g)
    loss = out.node_features.sum() + out.edge_features.sum()
    loss.backward()

    assert g.node_features.grad is not None
    assert g.edge_features.grad is not None
    print("[PASS] test_gradient_through_attention")


def test_node_attention_with_mask():
    g = make_test_graph()
    attn = NodeAttention(32, 16, num_heads=4)
    mask = torch.zeros(g.num_nodes, dtype=torch.bool)
    mask[:5] = True
    out = attn(g, mask=mask)
    assert out.shape == g.node_features.shape
    # Unmasked nodes should keep original features
    assert torch.allclose(out[5:], g.node_features[5:])
    print("[PASS] test_node_attention_with_mask")


if __name__ == '__main__':
    test_node_attention()
    test_edge_attention()
    test_dual_parallel()
    test_gradient_through_attention()
    test_node_attention_with_mask()
    print("\nAll attention tests passed.")
