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
    """Soft differentiable gating based on observed attention weights.

    The key insight: hard pruning after attention is destructive because:
    1. It zeroes features that other features were computed from (inconsistent)
    2. topk is non-differentiable — the pruner can never learn
    3. A single scalar importance signal is too thin

    Instead, this module produces SOFT continuous gates [0, 1] per edge,
    fully differentiable via sigmoid. Sparsity is encouraged through:
    - L1 regularization on gate values (target_sparsity loss)
    - Temperature annealing: low temp = soft gates, high temp = sharp gates
    - At convergence, gates approach binary without hard cutoffs

    This is the post-attention analogue of curriculum routing (Phase 12).
    """

    def __init__(self, d_node: int, d_edge: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads

        # Rich importance signal: per-head attention statistics → gate
        # Edge gate sees: edge features + per-head attention weights + endpoint importance
        self.edge_gate = nn.Sequential(
            nn.Linear(d_edge + num_heads + 2, d_edge),
            nn.GELU(),
            nn.Linear(d_edge, 1),
        )
        # Node gate sees: node features + aggregated per-head attention received
        self.node_gate = nn.Sequential(
            nn.Linear(d_node + num_heads, d_node // 2),
            nn.GELU(),
            nn.Linear(d_node // 2, 1),
        )
        # Cached sparsity loss for external access
        self.sparsity_loss = torch.tensor(0.0)

    def compute_importance(self, graph: DeltaGraph,
                           node_attn_weights: torch.Tensor,
                           edge_attn_weights: torch.Tensor,
                           temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute soft importance gates from observed attention weights.

        Args:
            graph: current DeltaGraph (post-attention features)
            node_attn_weights: [E, H] per-edge attention weights from NodeAttention
            edge_attn_weights: [E_adj, H] per-edge-adj attention weights from EdgeAttention
            temperature: gate sharpness — low=soft (start), high=sharp (end)

        Returns:
            node_gates: [N] soft importance in [0, 1]
            edge_gates: [E] soft importance in [0, 1]
        """
        N = graph.num_nodes
        E = graph.num_edges
        H = self.num_heads
        device = graph.device

        # --- Per-head attention statistics for edges ---
        if node_attn_weights.numel() > 0:
            # Clamp to num_heads columns (attention may have different H)
            attn_h = node_attn_weights[:, :H] if node_attn_weights.shape[1] >= H \
                else F.pad(node_attn_weights, (0, H - node_attn_weights.shape[1]))
        else:
            attn_h = torch.zeros(E, H, device=device)

        # Per-head node importance: scatter attention per head per target node
        node_attn_per_head = torch.zeros(N, H, device=device)
        for h in range(H):
            node_attn_per_head[:, h].scatter_add_(
                0, graph.edge_index[1], attn_h[:, h]
            )
        # Normalize per head
        head_max = node_attn_per_head.max(dim=0, keepdim=True).values + 1e-10
        node_attn_per_head = node_attn_per_head / head_max

        # --- Node gates ---
        node_input = torch.cat([graph.node_features, node_attn_per_head], dim=-1)
        node_logits = self.node_gate(node_input).squeeze(-1)  # [N]
        node_gates = torch.sigmoid(node_logits * temperature)

        # --- Edge gates: features + per-head attn + endpoint gate values ---
        src_gates = node_gates[graph.edge_index[0]].unsqueeze(-1)  # [E, 1]
        tgt_gates = node_gates[graph.edge_index[1]].unsqueeze(-1)  # [E, 1]
        edge_input = torch.cat([
            graph.edge_features,  # [E, d_edge]
            attn_h,               # [E, H] — per-head attention weights
            src_gates,            # [E, 1]
            tgt_gates,            # [E, 1]
        ], dim=-1)
        edge_logits = self.edge_gate(edge_input).squeeze(-1)  # [E]
        edge_gates = torch.sigmoid(edge_logits * temperature)

        return node_gates, edge_gates

    def soft_prune(self, graph: DeltaGraph, edge_gates: torch.Tensor,
                   target_sparsity: float = 0.5) -> Tuple[DeltaGraph, torch.Tensor]:
        """Apply soft gates to edge features and compute sparsity loss.

        Unlike hard pruning, this is fully differentiable. The sparsity loss
        encourages gates to approach the target sparsity level.

        Args:
            graph: DeltaGraph with post-attention features
            edge_gates: [E] soft gates in [0, 1]
            target_sparsity: fraction of edges to suppress (0.5 = 50% pruned)

        Returns:
            gated_graph: DeltaGraph with soft-gated edge features
            sparsity_loss: scalar loss encouraging target sparsity
        """
        # Soft gating: multiply edge features by gates
        gated_edges = graph.edge_features * edge_gates.unsqueeze(-1)

        gated_graph = DeltaGraph(
            node_features=graph.node_features,
            edge_features=gated_edges,
            edge_index=graph.edge_index,
            node_tiers=graph.node_tiers,
        )

        # Sparsity loss: encourage mean gate value to match (1 - target_sparsity)
        # e.g., target_sparsity=0.5 → want mean gate ≈ 0.5
        target_active = 1.0 - target_sparsity
        mean_gate = edge_gates.mean()
        self.sparsity_loss = (mean_gate - target_active) ** 2

        return gated_graph, self.sparsity_loss

    def prune(self, graph: DeltaGraph, node_scores: torch.Tensor,
              edge_scores: torch.Tensor,
              node_k_ratio: float = 0.7,
              edge_k_ratio: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hard top-k pruning (backward compat for legacy ImportanceRouter).

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
        self._pruner = PostAttentionPruner(d_node, d_edge, num_heads=4)
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
