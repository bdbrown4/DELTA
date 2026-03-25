"""
Post-Attention Pruning Router for DELTA.

Replaces the old pre-attention ImportanceRouter with a post-attention paradigm:
1. Run full attention first → observe actual attention weights
2. Prune based on OBSERVED importance (not predicted importance)
3. LearnedAttentionDropout: per-edge dropout conditioned on edge features

This eliminates the chicken-and-egg problem of the old approach (scoring
elements before seeing how they participate in attention).

The router still assigns memory tiers (hot/warm/cold) for the memory system.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from delta.graph import DeltaGraph, TIER_HOT, TIER_WARM, TIER_COLD


class PostAttentionPruner(nn.Module):
    """Prunes graph elements based on observed attention weights.

    After dual parallel attention runs and produces attention weights,
    this module computes importance from those observed weights and
    prunes low-importance nodes/edges for the next layer.
    """

    def __init__(self, d_node: int, d_edge: int):
        super().__init__()
        # Small projections to combine attention-derived importance with features
        self.node_gate = nn.Linear(d_node + 1, 1)  # +1 for attention importance
        self.edge_gate = nn.Linear(d_edge + 1, 1)  # +1 for attention importance

    def compute_importance(self, graph: DeltaGraph,
                           node_attn_weights: torch.Tensor,
                           edge_attn_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute importance scores from observed attention weights.

        Args:
            graph: current DeltaGraph
            node_attn_weights: [E, H] per-edge attention weights from NodeAttention
                               (how much each edge was attended to during node attention)
            edge_attn_weights: [E_adj, H] per-edge-adj attention weights from EdgeAttention

        Returns:
            node_scores: [N] importance in [0, 1]
            edge_scores: [E] importance in [0, 1]
        """
        N = graph.num_nodes
        E = graph.num_edges
        device = graph.device

        # --- Node importance from attention received ---
        # Sum attention weights across heads, then aggregate per target node
        if node_attn_weights.numel() > 0:
            edge_attn_sum = node_attn_weights.mean(dim=-1)  # [E] mean across heads
            node_attn_agg = torch.zeros(N, device=device)
            node_attn_agg.scatter_add_(0, graph.edge_index[1], edge_attn_sum)
            # Normalize to [0, 1]
            node_attn_agg = node_attn_agg / (node_attn_agg.max() + 1e-10)
        else:
            node_attn_agg = torch.ones(N, device=device)

        node_input = torch.cat([graph.node_features, node_attn_agg.unsqueeze(-1)], dim=-1)
        node_scores = torch.sigmoid(self.node_gate(node_input).squeeze(-1))

        # --- Edge importance from both node-attn and edge-attn ---
        if node_attn_weights.numel() > 0:
            edge_importance_from_node_attn = node_attn_weights.mean(dim=-1)  # [E]
        else:
            edge_importance_from_node_attn = torch.ones(E, device=device)

        # Aggregate edge-to-edge attention importance per edge
        if edge_attn_weights.numel() > 0:
            edge_attn_received = torch.zeros(E, device=device)
            # edge_attn_weights is [E_adj, H] — we need edge_adj to know targets
            # For simplicity, use node_attn-derived importance for edges
            edge_imp = edge_importance_from_node_attn
        else:
            edge_imp = edge_importance_from_node_attn

        edge_imp_normalized = edge_imp / (edge_imp.max() + 1e-10)
        edge_input = torch.cat([graph.edge_features, edge_imp_normalized.unsqueeze(-1)], dim=-1)
        edge_scores = torch.sigmoid(self.edge_gate(edge_input).squeeze(-1))

        return node_scores, edge_scores

    def prune(self, graph: DeltaGraph, node_scores: torch.Tensor,
              edge_scores: torch.Tensor,
              node_k_ratio: float = 0.7,
              edge_k_ratio: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return boolean masks for top-k nodes and edges to keep.

        Args:
            node_k_ratio: fraction of nodes to keep active (0-1)
            edge_k_ratio: fraction of edges to keep active (0-1)

        Returns:
            node_mask: [N] boolean
            edge_mask: [E] boolean
        """
        node_k = max(1, int(graph.num_nodes * node_k_ratio))
        edge_k = max(1, int(graph.num_edges * edge_k_ratio))

        _, node_top = torch.topk(node_scores, min(node_k, graph.num_nodes))
        node_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.device)
        node_mask[node_top] = True

        _, edge_top = torch.topk(edge_scores, min(edge_k, graph.num_edges))
        edge_mask = torch.zeros(graph.num_edges, dtype=torch.bool, device=graph.device)
        edge_mask[edge_top] = True

        return node_mask, edge_mask

    def update_tiers(self, graph: DeltaGraph, node_scores: torch.Tensor,
                     hot_threshold: float = 0.6,
                     cold_threshold: float = 0.2) -> torch.Tensor:
        """Assign memory tiers based on observed importance scores.

        Returns:
            new_tiers: [N] tensor with TIER_HOT / TIER_WARM / TIER_COLD
        """
        tiers = torch.full_like(graph.node_tiers, TIER_WARM)
        tiers[node_scores > hot_threshold] = TIER_HOT
        tiers[node_scores < cold_threshold] = TIER_COLD
        return tiers


class LearnedAttentionDropout(nn.Module):
    """Per-edge learned dropout conditioned on edge features.

    Instead of uniform dropout, each edge gets a learned dropout probability
    based on its features. This allows the model to learn which types of
    connections benefit from regularization vs. which should be preserved.
    """

    def __init__(self, d_edge: int):
        super().__init__()
        self.drop_rate = nn.Sequential(
            nn.Linear(d_edge, d_edge // 2),
            nn.GELU(),
            nn.Linear(d_edge // 2, 1),
        )

    def forward(self, edge_features: torch.Tensor,
                attn_weights: torch.Tensor) -> torch.Tensor:
        """Apply learned per-edge dropout to attention weights.

        Args:
            edge_features: [E, d_edge] edge feature vectors
            attn_weights: [E, H] attention weights from NodeAttention

        Returns:
            Dropped attention weights [E, H]
        """
        if not self.training:
            return attn_weights

        # Compute per-edge dropout probability in [0, 0.5] range
        p = torch.sigmoid(self.drop_rate(edge_features).squeeze(-1)) * 0.5  # [E]

        # Sample Bernoulli mask per edge (same across heads for stability)
        keep_mask = torch.bernoulli(1.0 - p).unsqueeze(-1)  # [E, 1]

        # Inverted dropout: scale by 1/(1-p) to preserve expected value
        scale = 1.0 / (1.0 - p + 1e-10)
        return attn_weights * keep_mask * scale.unsqueeze(-1)


# Legacy alias for backward compatibility with existing experiments
class ImportanceRouter(nn.Module):
    """Backward-compatible wrapper around PostAttentionPruner.

    Existing experiments that import ImportanceRouter will still work.
    The forward() method produces scores via a simple feature-based MLP
    (same as before), but the preferred path is PostAttentionPruner.
    """

    def __init__(self, d_node: int, d_edge: int, hidden_dim: int = 64):
        super().__init__()
        self._pruner = PostAttentionPruner(d_node, d_edge)
        # Keep the old MLP-based scorer for backward compat
        self.node_scorer = nn.Sequential(
            nn.Linear(d_node + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_scorer = nn.Sequential(
            nn.Linear(d_edge + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph: DeltaGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score all nodes and edges (legacy pre-attention path)."""
        N = graph.num_nodes
        device = graph.device

        degree = torch.zeros(N, device=device)
        degree.scatter_add_(0, graph.edge_index[0],
                           torch.ones(graph.num_edges, device=device))
        degree.scatter_add_(0, graph.edge_index[1],
                           torch.ones(graph.num_edges, device=device))
        degree = degree / (degree.max() + 1e-10)

        tier_normalized = graph.node_tiers.float() / 2.0

        node_input = torch.cat([
            graph.node_features,
            degree.unsqueeze(-1),
            tier_normalized.unsqueeze(-1),
        ], dim=-1)
        node_scores = torch.sigmoid(self.node_scorer(node_input).squeeze(-1))

        src_importance = node_scores[graph.edge_index[0]]
        tgt_importance = node_scores[graph.edge_index[1]]
        edge_input = torch.cat([
            graph.edge_features,
            src_importance.unsqueeze(-1),
            tgt_importance.unsqueeze(-1),
        ], dim=-1)
        edge_scores = torch.sigmoid(self.edge_scorer(edge_input).squeeze(-1))

        return node_scores, edge_scores

    def apply_top_k(self, graph: DeltaGraph, node_scores: torch.Tensor,
                    edge_scores: torch.Tensor,
                    node_k_ratio: float = 0.5,
                    edge_k_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._pruner.prune(graph, node_scores, edge_scores,
                                  node_k_ratio, edge_k_ratio)

    def apply_top_k_gumbel(self, graph: DeltaGraph, node_scores: torch.Tensor,
                           edge_scores: torch.Tensor,
                           node_k_ratio: float = 0.5,
                           edge_k_ratio: float = 0.5,
                           temperature: float = 1.0,
                           hard: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy Gumbel top-k — now just delegates to prune."""
        return self._pruner.prune(graph, node_scores, edge_scores,
                                  node_k_ratio, edge_k_ratio)

    def update_tiers(self, graph: DeltaGraph, node_scores: torch.Tensor,
                     hot_threshold: float = 0.6,
                     cold_threshold: float = 0.2) -> torch.Tensor:
        return self._pruner.update_tiers(graph, node_scores,
                                         hot_threshold, cold_threshold)
