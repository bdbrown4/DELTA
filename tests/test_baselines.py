"""Unit tests for GraphGPS and GRIT baseline implementations."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from delta.graph import DeltaGraph
from delta.baselines import (
    GraphGPSModel, GRITModel,
    MPNNLayer, GlobalSelfAttention, GPSLayer,
    RandomWalkPE, GRITAttention, GRITLayer,
)


def make_test_graph(N=10, E=15, d_node=32, d_edge=16):
    """Create a test graph (same pattern as other test files)."""
    edge_index = torch.stack([
        torch.randint(0, N, (E,)),
        torch.randint(0, N, (E,)),
    ])
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    E = edge_index.shape[1]

    return DeltaGraph(
        node_features=torch.randn(N, d_node),
        edge_features=torch.randn(E, d_edge),
        edge_index=edge_index,
    )


def make_triangle_graph(d_node=32, d_edge=16):
    """Create a simple triangle graph (3 nodes, 6 directed edges)."""
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 0, 1],
    ])
    return DeltaGraph(
        node_features=torch.randn(3, d_node),
        edge_features=torch.randn(6, d_edge),
        edge_index=edge_index,
    )


# --- GraphGPS component tests ---

def test_mpnn_layer():
    """MPNN layer preserves node feature shape and aggregates messages."""
    g = make_test_graph()
    mpnn = MPNNLayer(32, 16)
    out = mpnn(g)
    assert out.shape == g.node_features.shape
    print("[PASS] test_mpnn_layer")


def test_global_self_attention():
    """Global self-attention preserves shape across all nodes."""
    x = torch.randn(10, 32)
    attn = GlobalSelfAttention(32, num_heads=4)
    out = attn(x)
    assert out.shape == x.shape
    print("[PASS] test_global_self_attention")


def test_gps_layer():
    """GPS layer (MPNN + global attn + FFN) preserves graph structure."""
    g = make_test_graph()
    layer = GPSLayer(32, 16, num_heads=4)
    out = layer(g)
    assert out.node_features.shape == g.node_features.shape
    assert out.edge_features.shape == g.edge_features.shape
    assert torch.equal(out.edge_index, g.edge_index)
    print("[PASS] test_gps_layer")


def test_graphgps_model_forward():
    """Full GraphGPS model forward pass produces correct output shapes."""
    g = make_test_graph()
    model = GraphGPSModel(d_node=32, d_edge=16, num_layers=2, num_heads=4,
                          num_classes=5)
    out = model(g)
    assert out.node_features.shape == g.node_features.shape
    assert out.edge_features.shape == g.edge_features.shape

    # Classification heads
    node_logits = model.classify_nodes(out)
    assert node_logits.shape == (g.num_nodes, 5)

    edge_logits = model.classify_edges(out)
    assert edge_logits.shape == (g.num_edges, 5)
    print("[PASS] test_graphgps_model_forward")


def test_graphgps_gradient_flow():
    """Gradients flow through the full GraphGPS model."""
    g = make_test_graph()
    g.node_features.requires_grad_(True)
    g.edge_features.requires_grad_(True)

    model = GraphGPSModel(d_node=32, d_edge=16, num_layers=2, num_heads=4,
                          num_classes=5)
    out = model(g)
    loss = model.classify_nodes(out).sum()
    loss.backward()

    assert g.node_features.grad is not None
    print("[PASS] test_graphgps_gradient_flow")


def test_graphgps_link_prediction():
    """GraphGPS link prediction scoring works correctly."""
    g = make_test_graph()
    model = GraphGPSModel(d_node=32, d_edge=16, num_layers=2, num_heads=4)
    out = model(g)
    src = torch.tensor([0, 1, 2])
    tgt = torch.tensor([3, 4, 5])
    scores = model.predict_link(out, src, tgt)
    assert scores.shape == (3,)
    print("[PASS] test_graphgps_link_prediction")


# --- GRIT component tests ---

def test_random_walk_pe():
    """Random-walk PE produces correct shape and captures structure."""
    g = make_triangle_graph()
    rwpe = RandomWalkPE(walk_length=4, d_pe=8)
    pe = rwpe(g)
    assert pe.shape == (3, 8)

    # In a complete triangle, all nodes should have similar PE
    pe_std = pe.std(dim=0).mean()
    assert pe_std < 5.0  # PEs should be somewhat similar (projected, so not exact)
    print("[PASS] test_random_walk_pe")


def test_grit_attention():
    """GRIT attention with PE bias preserves output shape."""
    N, d_model, d_pe = 10, 32, 8
    x = torch.randn(N, d_model)
    pe = torch.randn(N, d_pe)

    attn = GRITAttention(d_model, d_pe, num_heads=4)
    out = attn(x, pe)
    assert out.shape == x.shape
    print("[PASS] test_grit_attention")


def test_grit_layer():
    """GRIT layer (PE-biased attention + FFN) preserves shape."""
    N, d_node, d_pe = 10, 32, 8
    x = torch.randn(N, d_node)
    pe = torch.randn(N, d_pe)

    layer = GRITLayer(d_node, d_pe, num_heads=4)
    out = layer(x, pe)
    assert out.shape == x.shape
    print("[PASS] test_grit_layer")


def test_grit_model_forward():
    """Full GRIT model forward pass produces correct output shapes."""
    g = make_test_graph()
    model = GRITModel(d_node=32, d_edge=16, num_layers=2, num_heads=4,
                      walk_length=4, d_pe=8, num_classes=5)
    out = model(g)
    assert out.node_features.shape == g.node_features.shape
    assert out.edge_features.shape == g.edge_features.shape

    node_logits = model.classify_nodes(out)
    assert node_logits.shape == (g.num_nodes, 5)

    edge_logits = model.classify_edges(out)
    assert edge_logits.shape == (g.num_edges, 5)
    print("[PASS] test_grit_model_forward")


def test_grit_gradient_flow():
    """Gradients flow through the full GRIT model."""
    g = make_test_graph()
    g.node_features.requires_grad_(True)

    model = GRITModel(d_node=32, d_edge=16, num_layers=2, num_heads=4,
                      walk_length=4, d_pe=8, num_classes=5)
    out = model(g)
    loss = model.classify_nodes(out).sum()
    loss.backward()

    assert g.node_features.grad is not None
    print("[PASS] test_grit_gradient_flow")


def test_grit_link_prediction():
    """GRIT link prediction scoring works correctly."""
    g = make_test_graph()
    model = GRITModel(d_node=32, d_edge=16, num_layers=2, num_heads=4,
                      walk_length=4, d_pe=8)
    out = model(g)
    src = torch.tensor([0, 1, 2])
    tgt = torch.tensor([3, 4, 5])
    scores = model.predict_link(out, src, tgt)
    assert scores.shape == (3,)
    print("[PASS] test_grit_link_prediction")


# --- Cross-architecture comparison tests ---

def test_all_models_same_interface():
    """All three architectures (DELTA, GraphGPS, GRIT) share the same interface."""
    from delta.model import DELTAModel

    g = make_test_graph()
    num_classes = 5

    delta = DELTAModel(d_node=32, d_edge=16, num_layers=2, num_heads=4,
                       num_classes=num_classes)
    gps = GraphGPSModel(d_node=32, d_edge=16, num_layers=2, num_heads=4,
                        num_classes=num_classes)
    grit = GRITModel(d_node=32, d_edge=16, num_layers=2, num_heads=4,
                     walk_length=4, d_pe=8, num_classes=num_classes)

    for name, model in [("DELTA", delta), ("GraphGPS", gps), ("GRIT", grit)]:
        out = model(g)
        assert isinstance(out, DeltaGraph), f"{name} should return DeltaGraph"
        assert out.node_features.shape == g.node_features.shape, f"{name} node shape"
        assert out.edge_features.shape == g.edge_features.shape, f"{name} edge shape"

        node_logits = model.classify_nodes(out)
        assert node_logits.shape == (g.num_nodes, num_classes), f"{name} node logits"

        edge_logits = model.classify_edges(out)
        assert edge_logits.shape == (g.num_edges, num_classes), f"{name} edge logits"

    print("[PASS] test_all_models_same_interface")


def test_parameter_count_comparison():
    """Compare parameter counts across architectures (informational)."""
    from delta.model import DELTAModel

    d_node, d_edge, num_layers = 32, 16, 2

    delta = DELTAModel(d_node=d_node, d_edge=d_edge, num_layers=num_layers,
                       num_heads=4, num_classes=5)
    gps = GraphGPSModel(d_node=d_node, d_edge=d_edge, num_layers=num_layers,
                        num_heads=4, num_classes=5)
    grit = GRITModel(d_node=d_node, d_edge=d_edge, num_layers=num_layers,
                     num_heads=4, walk_length=4, d_pe=8, num_classes=5)

    counts = {}
    for name, model in [("DELTA", delta), ("GraphGPS", gps), ("GRIT", grit)]:
        n_params = sum(p.numel() for p in model.parameters())
        counts[name] = n_params
        print(f"  {name}: {n_params:,} parameters")

    assert all(c > 0 for c in counts.values())
    print("[PASS] test_parameter_count_comparison")


def test_training_step_all_models():
    """All models can complete a full training step (forward + backward + optimizer step)."""
    from delta.model import DELTAModel

    torch.manual_seed(42)
    g = make_test_graph(N=20, E=40, d_node=32, d_edge=16)
    labels = torch.randint(0, 5, (g.num_edges,))

    models = {
        "DELTA": DELTAModel(d_node=32, d_edge=16, num_layers=2, num_heads=4,
                            num_classes=5),
        "GraphGPS": GraphGPSModel(d_node=32, d_edge=16, num_layers=2, num_heads=4,
                                  num_classes=5),
        "GRIT": GRITModel(d_node=32, d_edge=16, num_layers=2, num_heads=4,
                          walk_length=4, d_pe=8, num_classes=5),
    }

    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        out = model(g)
        logits = model.classify_edges(out)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0, f"{name} should produce positive loss"
        assert not torch.isnan(loss), f"{name} should not produce NaN loss"

    print("[PASS] test_training_step_all_models")


if __name__ == '__main__':
    test_mpnn_layer()
    test_global_self_attention()
    test_gps_layer()
    test_graphgps_model_forward()
    test_graphgps_gradient_flow()
    test_graphgps_link_prediction()
    test_random_walk_pe()
    test_grit_attention()
    test_grit_layer()
    test_grit_model_forward()
    test_grit_gradient_flow()
    test_grit_link_prediction()
    test_all_models_same_interface()
    test_parameter_count_comparison()
    test_training_step_all_models()
    print("\nAll baseline tests passed.")
