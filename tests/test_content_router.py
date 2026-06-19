"""Tests for content-dependent edge adjacency (Phase 75): ContentRouter + builder + the
attention route_prob gradient hook."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from delta.graph import DeltaGraph, build_content_edge_adjacency
from delta.router import ContentRouter
from delta.attention import EdgeAttention


def _toy(E=40, R=6, d_edge=8, seed=0):
    torch.manual_seed(seed)
    rel = torch.randint(0, R, (E,))
    rel_emb = torch.randn(R, d_edge)
    edge_feats = rel_emb[rel].clone()            # per-relation features
    edge_index = torch.randint(0, 20, (2, E))
    return rel, edge_feats, edge_index, R


def test_content_adjacency_linear_and_capped():
    rel, ef, _, R = _toy()
    r = ContentRouter(ef.shape[1], d_r=4)
    K = 5
    adj, rp = r.build(ef, rel, K, num_relations=R)
    E = ef.shape[0]
    assert adj.shape[1] <= E * K                          # linear bound
    assert rp.shape[0] == adj.shape[1]                    # route_prob aligned
    assert (adj[0] != adj[1]).all()                       # no self pairs
    assert torch.bincount(adj[1], minlength=E).max() <= K  # per-target cap
    print("[PASS] test_content_adjacency_linear_and_capped")


def test_content_router_gradient_through_attention():
    rel, ef, ei, R = _toy()
    r = ContentRouter(ef.shape[1], d_r=4)
    adj, rp = r.build(ef, rel, K=5, num_relations=R)
    # d_edge must be divisible by num_heads for EdgeAttention
    att = EdgeAttention(d_edge=ef.shape[1], d_node=4, num_heads=2)
    g = DeltaGraph(node_features=torch.randn(20, 4),
                   edge_features=ef, edge_index=ei)
    out = att(g, edge_adj=adj, route_prob=rp)
    loss = out.sum()
    loss.backward()
    assert r.Wt.weight.grad is not None and r.Wt.weight.grad.abs().sum() > 0, \
        "no gradient reached ContentRouter keys via route_prob"
    print("[PASS] test_content_router_gradient_through_attention")


def test_route_prob_via_graph_attr():
    """EdgeAttention reads graph._route_prob when route_prob arg is None (harness path)."""
    rel, ef, ei, R = _toy()
    r = ContentRouter(ef.shape[1], d_r=4)
    adj, rp = r.build(ef, rel, K=5, num_relations=R)
    att = EdgeAttention(d_edge=ef.shape[1], d_node=4, num_heads=2)
    g = DeltaGraph(node_features=torch.randn(20, 4), edge_features=ef, edge_index=ei)
    g._route_prob = rp
    out = att(g, edge_adj=adj)            # no route_prob arg -> reads g._route_prob
    out.sum().backward()
    assert r.Wt.weight.grad is not None and r.Wt.weight.grad.abs().sum() > 0
    print("[PASS] test_route_prob_via_graph_attr")


def test_content_sim_tied_keys():
    rel, ef, _, R = _toy()
    r = ContentRouter(ef.shape[1], d_r=4, tie=True)
    assert r.Wh is r.Wt                   # the symmetric (Ah==At) control, by construction
    adj, rp = r.build(ef, rel, K=5, num_relations=R)
    assert adj.shape[1] > 0
    print("[PASS] test_content_sim_tied_keys")


def test_aux_loss_gradient():
    rel, ef, ei, R = _toy()
    ef = ef.clone().requires_grad_(True)
    r = ContentRouter(ef.shape[1], d_r=4)
    comp_Y = (torch.rand(R, R) < 0.1).float()           # toy composability matrix
    aux = r.aux_losses(ef, rel, comp_Y, R)
    assert torch.isfinite(aux)                          # bounded (no NaN/explosion)
    aux.backward()
    assert r.Wt.weight.grad is not None and r.Wt.weight.grad.abs().sum() > 0
    print("[PASS] test_aux_loss_gradient")


def test_frozen_router_no_grad():
    rel, ef, _, R = _toy()
    r = ContentRouter(ef.shape[1], d_r=4, freeze=True)   # content_random control
    assert all(not p.requires_grad for p in r.parameters())
    print("[PASS] test_frozen_router_no_grad")


if __name__ == '__main__':
    test_content_adjacency_linear_and_capped()
    test_content_router_gradient_through_attention()
    test_route_prob_via_graph_attr()
    test_content_sim_tied_keys()
    test_aux_loss_gradient()
    test_frozen_router_no_grad()
    print("\nAll content-router tests passed.")
