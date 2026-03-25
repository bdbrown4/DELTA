"""Unit tests for post-attention pruning router."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from delta.graph import DeltaGraph
from delta.router import PostAttentionPruner, LearnedAttentionDropout, ImportanceRouter


def make_test_graph():
    N, E, d_node, d_edge = 10, 15, 32, 16
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


def test_post_attention_pruner():
    g = make_test_graph()
    pruner = PostAttentionPruner(32, 16)
    # Simulate attention weights
    E = g.num_edges
    node_attn_w = torch.rand(E, 4)  # [E, H=4]
    edge_attn_w = torch.rand(20, 4)  # [E_adj, H=4]
    node_scores, edge_scores = pruner.compute_importance(g, node_attn_w, edge_attn_w)
    assert node_scores.shape == (g.num_nodes,)
    assert edge_scores.shape == (g.num_edges,)
    assert (node_scores >= 0).all() and (node_scores <= 1).all()
    assert (edge_scores >= 0).all() and (edge_scores <= 1).all()
    print("[PASS] test_post_attention_pruner")


def test_prune():
    g = make_test_graph()
    pruner = PostAttentionPruner(32, 16)
    E = g.num_edges
    node_attn_w = torch.rand(E, 4)
    edge_attn_w = torch.rand(20, 4)
    node_scores, edge_scores = pruner.compute_importance(g, node_attn_w, edge_attn_w)
    node_mask, edge_mask = pruner.prune(g, node_scores, edge_scores, 0.5, 0.5)
    assert node_mask.sum() == max(1, int(g.num_nodes * 0.5))
    print("[PASS] test_prune")


def test_tier_update():
    g = make_test_graph()
    pruner = PostAttentionPruner(32, 16)
    E = g.num_edges
    node_attn_w = torch.rand(E, 4)
    edge_attn_w = torch.rand(20, 4)
    node_scores, _ = pruner.compute_importance(g, node_attn_w, edge_attn_w)
    tiers = pruner.update_tiers(g, node_scores)
    assert tiers.shape == (g.num_nodes,)
    assert set(tiers.unique().tolist()).issubset({0, 1, 2})
    print("[PASS] test_tier_update")


def test_learned_attention_dropout():
    edge_features = torch.randn(15, 16)
    attn_weights = torch.rand(15, 4)
    drop = LearnedAttentionDropout(16)
    drop.train()
    out = drop(edge_features, attn_weights)
    assert out.shape == attn_weights.shape
    drop.eval()
    out_eval = drop(edge_features, attn_weights)
    assert torch.allclose(out_eval, attn_weights)
    print("[PASS] test_learned_attention_dropout")


def test_legacy_importance_router():
    """Ensure backward-compatible ImportanceRouter still works."""
    g = make_test_graph()
    router = ImportanceRouter(32, 16)
    node_scores, edge_scores = router(g)
    assert node_scores.shape == (g.num_nodes,)
    assert edge_scores.shape == (g.num_edges,)
    tiers = router.update_tiers(g, node_scores)
    assert tiers.shape == (g.num_nodes,)
    print("[PASS] test_legacy_importance_router")


if __name__ == '__main__':
    test_post_attention_pruner()
    test_prune()
    test_tier_update()
    test_learned_attention_dropout()
    test_legacy_importance_router()
    print("\nAll router tests passed.")
