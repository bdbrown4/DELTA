"""
Lightweight baseline implementations of GraphGPS and GRIT for comparison testing.

These are faithful simplified implementations capturing the core architectural ideas:
- GraphGPS (Rampášek et al., 2022): GPS layer = MPNN + Global Attention + FFN
- GRIT (Ma et al., 2023): Graph Inductive Bias Transformer with relative random-walk
  positional encodings injected into attention

These implementations operate on DeltaGraph structures so we can compare
DELTA, GraphGPS, and GRIT on identical data with identical evaluation protocols.

NOTE: These are research baselines for controlled comparison, not full replicas
of the original papers. For publication, run against official implementations.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from delta.graph import DeltaGraph


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Two-layer FFN with GELU, shared by both baselines."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# GraphGPS baseline
# ---------------------------------------------------------------------------

class MPNNLayer(nn.Module):
    """Message-Passing Neural Network layer (GIN-style aggregation).

    Each node aggregates neighbor features through edges:
        h_i' = MLP( (1 + eps) * h_i + sum_{j in N(i)} msg(h_j, e_ij) )
    """

    def __init__(self, d_node: int, d_edge: int, dropout: float = 0.1):
        super().__init__()
        self.msg_fn = nn.Sequential(
            nn.Linear(d_node + d_edge, d_node),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.update_fn = nn.Sequential(
            nn.Linear(d_node, d_node),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, graph: DeltaGraph) -> torch.Tensor:
        src, tgt = graph.edge_index  # [E]
        src_feats = graph.node_features[src]  # [E, d_node]
        messages = self.msg_fn(torch.cat([src_feats, graph.edge_features], dim=-1))

        # Scatter-add messages to target nodes
        agg = torch.zeros_like(graph.node_features)
        agg.index_add_(0, tgt, messages)

        out = self.update_fn((1 + self.eps) * graph.node_features + agg)
        return out


class GlobalSelfAttention(nn.Module):
    """Standard multi-head self-attention over all nodes (no graph structure).

    This is the global attention component of GraphGPS — treating nodes
    as a set (like a Transformer).
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, d_model] — all node features as a flat set."""
        N, D = x.shape
        qkv = self.qkv(x).reshape(N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3)  # [3, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(0, 1).reshape(N, D)  # [N, D]
        return self.proj_drop(self.out_proj(out))


class GPSLayer(nn.Module):
    """GraphGPS layer: local MPNN + global attention + FFN with residuals.

    GPS(h) = Norm(h + MPNN(h)) → Norm(h + GlobalAttn(h)) → Norm(h + FFN(h))

    Reference: Rampášek et al., "Recipe for a General, Powerful, Scalable
    Graph Transformer" (NeurIPS 2022).
    """

    def __init__(self, d_node: int, d_edge: int, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.mpnn = MPNNLayer(d_node, d_edge, dropout)
        self.global_attn = GlobalSelfAttention(d_node, num_heads, dropout)
        self.ffn = FeedForward(d_node, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_node)
        self.norm2 = nn.LayerNorm(d_node)
        self.norm3 = nn.LayerNorm(d_node)

    def forward(self, graph: DeltaGraph) -> DeltaGraph:
        h = graph.node_features

        # Local MPNN
        h = self.norm1(h + self.mpnn(graph))

        # Global self-attention
        h = self.norm2(h + self.global_attn(h))

        # FFN
        h = self.norm3(h + self.ffn(h))

        return DeltaGraph(
            node_features=h,
            edge_features=graph.edge_features,
            edge_index=graph.edge_index,
            node_tiers=graph.node_tiers,
            node_importance=graph.node_importance,
            edge_importance=graph.edge_importance,
        )


class GraphGPSModel(nn.Module):
    """GraphGPS baseline model.

    Stacks GPS layers (MPNN + global attention + FFN) with optional
    classification heads for node and edge tasks.

    Args:
        d_node: node feature dimension
        d_edge: edge feature dimension
        num_layers: number of GPS layers
        num_heads: attention heads for global attention
        num_classes: output classes (None = no classifier)
        dropout: dropout rate
    """

    def __init__(self, d_node: int = 64, d_edge: int = 32,
                 num_layers: int = 3, num_heads: int = 4,
                 num_classes: int = None, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GPSLayer(d_node, d_edge, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.node_classifier = None
        self.edge_classifier = None
        if num_classes is not None:
            self.node_classifier = nn.Sequential(
                nn.Linear(d_node, d_node),
                nn.GELU(),
                nn.Linear(d_node, num_classes),
            )
            self.edge_classifier = nn.Sequential(
                nn.Linear(d_edge + 2 * d_node, d_edge),
                nn.GELU(),
                nn.Linear(d_edge, num_classes),
            )

    def forward(self, graph: DeltaGraph) -> DeltaGraph:
        for layer in self.layers:
            graph = layer(graph)
        return graph

    def classify_nodes(self, graph: DeltaGraph) -> torch.Tensor:
        assert self.node_classifier is not None
        return self.node_classifier(graph.node_features)

    def classify_edges(self, graph: DeltaGraph) -> torch.Tensor:
        assert self.edge_classifier is not None
        src, tgt = graph.edge_index
        edge_repr = torch.cat([
            graph.node_features[src],
            graph.node_features[tgt],
            graph.edge_features,
        ], dim=-1)
        return self.edge_classifier(edge_repr)

    def predict_link(self, graph: DeltaGraph, src: torch.Tensor,
                     tgt: torch.Tensor) -> torch.Tensor:
        src_feats = graph.node_features[src]
        tgt_feats = graph.node_features[tgt]
        return (src_feats * tgt_feats).sum(dim=-1)


# ---------------------------------------------------------------------------
# GRIT baseline
# ---------------------------------------------------------------------------

class RandomWalkPE(nn.Module):
    """Random-walk positional encoding for GRIT.

    Computes k-step random-walk landing probabilities as positional features,
    then projects them into the model dimension.

    For a pair (i, j), the PE is based on the probability of a random walk
    from i reaching j in k steps. This gives GRIT its structural awareness.
    """

    def __init__(self, walk_length: int = 8, d_pe: int = 16):
        super().__init__()
        self.walk_length = walk_length
        self.d_pe = d_pe
        self.pe_proj = nn.Linear(walk_length, d_pe)

    def compute_rw_probs(self, graph: DeltaGraph) -> torch.Tensor:
        """Compute random-walk landing probabilities for each node.

        Returns: [N, walk_length] tensor of k-step return probabilities.
        """
        N = graph.num_nodes
        device = graph.device

        # Build adjacency matrix
        src, tgt = graph.edge_index
        adj = torch.zeros(N, N, device=device)
        adj[src, tgt] = 1.0
        adj[tgt, src] = 1.0  # Symmetrize

        # Row-normalize to get transition matrix
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        T = adj / deg  # [N, N]

        # Compute k-step probabilities
        rw_probs = torch.zeros(N, self.walk_length, device=device)
        Tk = T.clone()
        for k in range(self.walk_length):
            rw_probs[:, k] = Tk.diagonal()  # k-step return probability
            if k < self.walk_length - 1:
                Tk = Tk @ T

        return rw_probs

    def forward(self, graph: DeltaGraph) -> torch.Tensor:
        """Returns [N, d_pe] positional encodings."""
        rw_probs = self.compute_rw_probs(graph)
        return self.pe_proj(rw_probs)


class GRITAttention(nn.Module):
    """GRIT-style attention with relative positional encoding bias.

    Key difference from standard attention: the positional encoding is
    injected as a bias into the attention scores, giving the model
    structural awareness without explicit message passing.

    Reference: Ma et al., "Graph Inductive Biases in Transformers without
    Message Passing" (ICML 2023).
    """

    def __init__(self, d_model: int, d_pe: int = 16, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.pe_bias = nn.Linear(d_pe, num_heads)  # PE → per-head bias
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, d_model] node features
            pe: [N, d_pe] positional encodings
        Returns:
            [N, d_model] updated features
        """
        N, D = x.shape
        qkv = self.qkv(x).reshape(N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3)  # [3, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Standard attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [heads, N, N]

        # Relative PE bias: project per-node PE, then form pairwise differences
        pe_proj = self.pe_bias(pe)  # [N, num_heads]
        pe_bias = (pe_proj.unsqueeze(1) - pe_proj.unsqueeze(0)).permute(2, 0, 1)  # [heads, N, N]
        attn = attn + pe_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(0, 1).reshape(N, D)
        return self.proj_drop(self.out_proj(out))


class GRITLayer(nn.Module):
    """GRIT layer: PE-biased attention + FFN with residuals.

    GRIT(h) = Norm(h + GRITAttn(h, PE)) → Norm(h + FFN(h))
    """

    def __init__(self, d_node: int, d_pe: int = 16, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = GRITAttention(d_node, d_pe, num_heads, dropout)
        self.ffn = FeedForward(d_node, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_node)
        self.norm2 = nn.LayerNorm(d_node)

    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, pe))
        x = self.norm2(x + self.ffn(x))
        return x


class GRITModel(nn.Module):
    """GRIT baseline model.

    Graph Inductive Bias Transformer: uses random-walk PE to inject
    structural information into a pure Transformer (no message passing).

    Args:
        d_node: node feature dimension
        d_edge: edge feature dimension (used for edge classification head)
        num_layers: number of GRIT layers
        num_heads: attention heads
        walk_length: random walk steps for PE
        d_pe: PE embedding dimension
        num_classes: output classes (None = no classifier)
        dropout: dropout rate
    """

    def __init__(self, d_node: int = 64, d_edge: int = 32,
                 num_layers: int = 3, num_heads: int = 4,
                 walk_length: int = 8, d_pe: int = 16,
                 num_classes: int = None, dropout: float = 0.1):
        super().__init__()
        self.d_edge = d_edge
        self.rwpe = RandomWalkPE(walk_length, d_pe)
        self.layers = nn.ModuleList([
            GRITLayer(d_node, d_pe, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.node_classifier = None
        self.edge_classifier = None
        if num_classes is not None:
            self.node_classifier = nn.Sequential(
                nn.Linear(d_node, d_node),
                nn.GELU(),
                nn.Linear(d_node, num_classes),
            )
            self.edge_classifier = nn.Sequential(
                nn.Linear(d_edge + 2 * d_node, d_edge),
                nn.GELU(),
                nn.Linear(d_edge, num_classes),
            )

    def forward(self, graph: DeltaGraph) -> DeltaGraph:
        pe = self.rwpe(graph)
        h = graph.node_features

        for layer in self.layers:
            h = layer(h, pe)

        return DeltaGraph(
            node_features=h,
            edge_features=graph.edge_features,
            edge_index=graph.edge_index,
            node_tiers=graph.node_tiers,
            node_importance=graph.node_importance,
            edge_importance=graph.edge_importance,
        )

    def classify_nodes(self, graph: DeltaGraph) -> torch.Tensor:
        assert self.node_classifier is not None
        return self.node_classifier(graph.node_features)

    def classify_edges(self, graph: DeltaGraph) -> torch.Tensor:
        assert self.edge_classifier is not None
        src, tgt = graph.edge_index
        edge_repr = torch.cat([
            graph.node_features[src],
            graph.node_features[tgt],
            graph.edge_features,
        ], dim=-1)
        return self.edge_classifier(edge_repr)

    def predict_link(self, graph: DeltaGraph, src: torch.Tensor,
                     tgt: torch.Tensor) -> torch.Tensor:
        src_feats = graph.node_features[src]
        tgt_feats = graph.node_features[tgt]
        return (src_feats * tgt_feats).sum(dim=-1)
