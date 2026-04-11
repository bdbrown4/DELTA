"""
Attention mechanisms for DELTA.

Three levels:
1. NodeAttention — GAT-style: nodes attend to neighbor nodes via edges
2. EdgeAttention — DELTA's core novelty: edges attend to structurally adjacent edges
3. DualParallelAttention — runs both simultaneously and reconciles

Edge-to-edge attention is what makes DELTA different from existing graph networks.
Standard GAT treats edges as scalar gates. DELTA treats edges as computational
units that attend to each other — enabling reasoning about relationships between
relationships (structural analogy).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from delta.graph import DeltaGraph


class NodeAttention(nn.Module):
    """Multi-head attention over graph nodes (GAT-style).

    Each node attends to its neighbors, with edge features incorporated
    as bias/gating on the attention scores.
    """

    def __init__(self, d_node: int, d_edge: int, num_heads: int = 4, dropout: float = 0.1,
                 init_temp: float = 1.0):
        super().__init__()
        self.d_node = d_node
        self.num_heads = num_heads
        self.d_head = d_node // num_heads
        assert d_node % num_heads == 0

        self.W_q = nn.Linear(d_node, d_node)
        self.W_k = nn.Linear(d_node, d_node)
        self.W_v = nn.Linear(d_node, d_node)
        self.W_edge_bias = nn.Linear(d_edge, num_heads)  # edge features -> attention bias per head
        self.W_out = nn.Linear(d_node, d_node)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_node)

        # Learnable per-head temperature (multiplier on attn scores before softmax).
        # Higher → sharper attention.  Default 1.0 = standard scaled dot-product.
        self._log_temp = nn.Parameter(torch.full((num_heads,), math.log(init_temp)))

    def forward(self, graph: DeltaGraph, mask: Optional[torch.Tensor] = None,
                return_weights: bool = False):
        """
        Args:
            graph: DeltaGraph with node_features, edge_features, edge_index
            mask: optional [N] boolean mask — only update these nodes
            return_weights: if True, also return per-edge attention weights [E, H]

        Returns:
            Updated node features [N, d_node], or (features, attn_weights) if return_weights
        """
        N = graph.num_nodes
        H = self.num_heads
        d_h = self.d_head

        x = graph.node_features  # [N, d_node]
        Q = self.W_q(x).view(N, H, d_h)  # [N, H, d_h]
        K = self.W_k(x).view(N, H, d_h)
        V = self.W_v(x).view(N, H, d_h)

        src, tgt = graph.edge_index  # [E] each

        # Compute attention scores for each edge: query from target, key from source
        q_tgt = Q[tgt]  # [E, H, d_h]
        k_src = K[src]  # [E, H, d_h]

        # Scaled dot product per edge
        attn_scores = (q_tgt * k_src).sum(dim=-1) / math.sqrt(d_h)  # [E, H]

        # Add edge feature bias
        edge_bias = self.W_edge_bias(graph.edge_features)  # [E, H]
        attn_scores = attn_scores + edge_bias

        # Apply learnable per-head temperature (sharpen/soften attention)
        temp = self._log_temp.exp()  # [H], always positive
        attn_scores = attn_scores * temp  # [E, H]

        # Softmax over incoming edges per target node
        attn_weights = self._scatter_softmax(attn_scores, tgt, N)  # [E, H]
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of source values per target node
        v_src = V[src]  # [E, H, d_h]
        weighted = v_src * attn_weights.unsqueeze(-1)  # [E, H, d_h]

        # Scatter-add to target nodes
        out = torch.zeros(N, H, d_h, device=x.device)
        tgt_expanded = tgt.unsqueeze(-1).unsqueeze(-1).expand_as(weighted)
        out.scatter_add_(0, tgt_expanded, weighted)

        out = out.reshape(N, self.d_node)
        out = self.W_out(out)

        # Residual + norm
        if mask is not None:
            result = graph.node_features.clone()
            result[mask] = self.norm(graph.node_features[mask] + out[mask])
            if return_weights:
                return result, attn_weights
            return result
        result = self.norm(x + out)
        if return_weights:
            return result, attn_weights
        return result

    def _scatter_softmax(self, scores: torch.Tensor, index: torch.Tensor, N: int) -> torch.Tensor:
        """Softmax of scores grouped by index (per-node normalization)."""
        # Stability: subtract max per group
        max_vals = torch.full((N, scores.shape[1]), -1e9, device=scores.device)
        idx_expanded = index.unsqueeze(-1).expand_as(scores)
        max_vals.scatter_reduce_(0, idx_expanded, scores, reduce='amax', include_self=False)
        scores = scores - max_vals.gather(0, idx_expanded)

        exp_scores = torch.exp(scores)
        sum_exp = torch.zeros(N, scores.shape[1], device=scores.device)
        sum_exp.scatter_add_(0, idx_expanded, exp_scores)
        denom = sum_exp.gather(0, idx_expanded)
        return exp_scores / (denom + 1e-10)


class EdgeAttention(nn.Module):
    """Multi-head attention over graph edges — DELTA's core novelty.

    Structurally adjacent edges (sharing at least one endpoint) attend to
    each other. This enables the model to reason about relationships between
    relationships — e.g., recognizing that "is capital of" edges share a
    pattern regardless of the specific nodes involved.
    """

    def __init__(self, d_edge: int, d_node: int, num_heads: int = 4, dropout: float = 0.1,
                 init_temp: float = 1.0):
        super().__init__()
        self.d_edge = d_edge
        self.num_heads = num_heads
        self.d_head = d_edge // num_heads
        assert d_edge % num_heads == 0

        self.W_q = nn.Linear(d_edge, d_edge)
        self.W_k = nn.Linear(d_edge, d_edge)
        self.W_v = nn.Linear(d_edge, d_edge)
        # Node context: incorporate endpoint features into edge attention
        self.W_ctx = nn.Linear(2 * d_node, num_heads)
        self.W_out = nn.Linear(d_edge, d_edge)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_edge)

        # Learnable per-head temperature (same semantics as NodeAttention)
        self._log_temp = nn.Parameter(torch.full((num_heads,), math.log(init_temp)))

    def forward(self, graph: DeltaGraph, edge_adj: Optional[torch.Tensor] = None,
                return_weights: bool = False):
        """
        Args:
            graph: DeltaGraph
            edge_adj: [2, E_adj] edge-to-edge adjacency (precomputed).
                      If None, computed from graph.
            return_weights: if True, also return per-edge-adj attention weights

        Returns:
            Updated edge features [E, d_edge], or (features, attn_weights) if return_weights
        """
        if edge_adj is None:
            edge_adj = graph.build_edge_adjacency()

        E = graph.num_edges
        H = self.num_heads
        d_h = self.d_head

        e = graph.edge_features  # [E, d_edge]
        Q = self.W_q(e).view(E, H, d_h)
        K = self.W_k(e).view(E, H, d_h)
        V = self.W_v(e).view(E, H, d_h)

        if edge_adj.shape[1] == 0:
            if return_weights:
                return self.norm(e), torch.zeros(0, H, device=e.device)
            return self.norm(e)

        src_edges, tgt_edges = edge_adj  # [E_adj] each

        # Attention scores between adjacent edges
        q_tgt = Q[tgt_edges]  # [E_adj, H, d_h]
        k_src = K[src_edges]  # [E_adj, H, d_h]
        attn_scores = (q_tgt * k_src).sum(dim=-1) / math.sqrt(d_h)  # [E_adj, H]

        # Node context bias: use endpoint features of both edges
        src_edge_ctx = torch.cat([
            graph.node_features[graph.edge_index[0, src_edges]],
            graph.node_features[graph.edge_index[1, src_edges]]
        ], dim=-1)  # [E_adj, 2*d_node]
        ctx_bias = self.W_ctx(src_edge_ctx)  # [E_adj, H]
        attn_scores = attn_scores + ctx_bias

        # Apply learnable per-head temperature
        temp = self._log_temp.exp()  # [H], always positive
        attn_scores = attn_scores * temp  # [E_adj, H]

        # Softmax over neighbor edges per target edge
        attn_weights = self._scatter_softmax(attn_scores, tgt_edges, E)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        v_src = V[src_edges]  # [E_adj, H, d_h]
        weighted = v_src * attn_weights.unsqueeze(-1)

        out = torch.zeros(E, H, d_h, device=e.device)
        tgt_expanded = tgt_edges.unsqueeze(-1).unsqueeze(-1).expand_as(weighted)
        out.scatter_add_(0, tgt_expanded, weighted)

        out = out.reshape(E, self.d_edge)
        out = self.W_out(out)

        result = self.norm(e + out)
        if return_weights:
            return result, attn_weights
        return result

    def _scatter_softmax(self, scores: torch.Tensor, index: torch.Tensor, N: int) -> torch.Tensor:
        max_vals = torch.full((N, scores.shape[1]), -1e9, device=scores.device)
        idx_expanded = index.unsqueeze(-1).expand_as(scores)
        max_vals.scatter_reduce_(0, idx_expanded, scores, reduce='amax', include_self=False)
        scores = scores - max_vals.gather(0, idx_expanded)
        exp_scores = torch.exp(scores)
        sum_exp = torch.zeros(N, scores.shape[1], device=scores.device)
        sum_exp.scatter_add_(0, idx_expanded, exp_scores)
        denom = sum_exp.gather(0, idx_expanded)
        return exp_scores / (denom + 1e-10)


class DualParallelAttention(nn.Module):
    """Runs node and edge attention in parallel, then reconciles.

    This is the core DELTA mechanism: both node-to-node and edge-to-edge
    attention fire simultaneously. After both complete, a reconciliation
    step ensures the updated node features inform edges and vice versa.
    """

    def __init__(self, d_node: int, d_edge: int, num_heads: int = 4, dropout: float = 0.1,
                 init_temp: float = 1.0):
        super().__init__()
        self.node_attn = NodeAttention(d_node, d_edge, num_heads, dropout, init_temp=init_temp)
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads, dropout, init_temp=init_temp)
        self.reconciliation = ReconciliationBridge(d_node, d_edge)

    def forward(self, graph: DeltaGraph,
                edge_adj: Optional[torch.Tensor] = None,
                node_mask: Optional[torch.Tensor] = None,
                return_weights: bool = False):
        """
        Args:
            graph: input DeltaGraph
            edge_adj: precomputed edge adjacency
            node_mask: optional mask for selective node updates
            return_weights: if True, return (graph, node_attn_weights, edge_attn_weights)

        Returns:
            DeltaGraph, or (DeltaGraph, node_attn_weights, edge_attn_weights)
        """
        # Parallel: both use the SAME input graph state
        node_result = self.node_attn(graph, mask=node_mask, return_weights=return_weights)
        edge_result = self.edge_attn(graph, edge_adj=edge_adj, return_weights=return_weights)

        if return_weights:
            new_node_features, node_attn_w = node_result
            new_edge_features, edge_attn_w = edge_result
        else:
            new_node_features = node_result
            new_edge_features = edge_result

        # Reconcile: let updated nodes inform edges and updated edges inform nodes
        final_nodes, final_edges = self.reconciliation(
            new_node_features, new_edge_features, graph.edge_index
        )

        result_graph = DeltaGraph(
            node_features=final_nodes,
            edge_features=final_edges,
            edge_index=graph.edge_index,
            node_tiers=graph.node_tiers,
            node_importance=graph.node_importance,
            edge_importance=graph.edge_importance,
        )
        # Propagate edge adjacency cache (structure unchanged across layers)
        result_graph._edge_adj_cache = graph._edge_adj_cache

        if return_weights:
            return result_graph, node_attn_w, edge_attn_w
        return result_graph


class ReconciliationBridge(nn.Module):
    """Reconciles node and edge features after parallel attention.

    After node attention updates nodes and edge attention updates edges
    independently, this bridge ensures bidirectional information flow:
    - Edges receive context from their updated endpoint nodes
    - Nodes receive context from their updated incident edges
    """

    def __init__(self, d_node: int, d_edge: int):
        super().__init__()
        # Edge absorbs updated node context
        self.edge_from_nodes = nn.Linear(d_edge + 2 * d_node, d_edge)
        self.edge_norm = nn.LayerNorm(d_edge)
        # Node absorbs updated edge context
        self.node_from_edges = nn.Linear(d_node + d_edge, d_node)
        self.node_norm = nn.LayerNorm(d_node)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor,
                edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src, tgt = edge_index

        # --- Edges absorb node context ---
        src_feats = node_features[src]  # [E, d_node]
        tgt_feats = node_features[tgt]  # [E, d_node]
        edge_ctx = torch.cat([edge_features, src_feats, tgt_feats], dim=-1)
        new_edges = self.edge_norm(edge_features + self.edge_from_nodes(edge_ctx))

        # --- Nodes absorb edge context ---
        # Mean-pool incident edge features per node
        N = node_features.shape[0]
        edge_sum = torch.zeros(N, edge_features.shape[1], device=node_features.device)
        edge_count = torch.zeros(N, 1, device=node_features.device)

        # Both endpoints receive each edge's features
        all_nodes = torch.cat([src, tgt])
        all_edges = torch.cat([new_edges, new_edges])
        node_idx = all_nodes.unsqueeze(-1).expand_as(all_edges)
        edge_sum.scatter_add_(0, node_idx, all_edges)
        edge_count.scatter_add_(0, all_nodes.unsqueeze(-1),
                                torch.ones(all_nodes.shape[0], 1, device=node_features.device))
        edge_mean = edge_sum / (edge_count + 1e-10)

        node_ctx = torch.cat([node_features, edge_mean], dim=-1)
        new_nodes = self.node_norm(node_features + self.node_from_edges(node_ctx))

        return new_nodes, new_edges
