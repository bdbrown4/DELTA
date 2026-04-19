"""
Full DELTA Model — assembles all components into the complete architecture.

Flow:
    Raw Input → Graph Constructor → Router → Partitioner →
    Dual Parallel Attention (per partition) → Reconciliation →
    Hierarchical Global Attention → Memory Tier Update →
    Output + Updated Graph State
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from delta.graph import DeltaGraph
from delta.attention import DualParallelAttention, NodeAttention
from delta.router import PostAttentionPruner, LearnedAttentionDropout, ImportanceRouter
from delta.memory import TieredMemory
from delta.partition import GraphPartitioner
from delta.constructor import GraphConstructor


class DELTALayer(nn.Module):
    """A single DELTA processing layer.

    Post-attention paradigm:
    1. Run dual parallel attention → get features + attention weights
    2. Compute importance from observed attention weights
    3. Update memory tiers based on observed importance
    4. Prune for next layer (optional)
    """

    def __init__(self, d_node: int, d_edge: int, num_heads: int = 4,
                 max_partition_size: int = 32, dropout: float = 0.1,
                 sparse_ratio: float = 0.7, init_temp: float = 1.0,
                 topk_edges: Optional[int] = None):
        super().__init__()
        self.pruner = PostAttentionPruner(d_node, d_edge)
        self.attn_dropout = LearnedAttentionDropout(d_edge)
        self.partitioner = GraphPartitioner(max_partition_size)
        self.dual_attn = DualParallelAttention(d_node, d_edge, num_heads, dropout,
                                               init_temp=init_temp, topk_edges=topk_edges)
        self.memory = TieredMemory(d_node, d_edge)
        self.sparse_ratio = sparse_ratio

        # Legacy router for backward compatibility
        self.router = ImportanceRouter(d_node, d_edge)

        # Hierarchical attention for cross-partition communication
        self.global_node_attn = NodeAttention(d_node, d_edge, num_heads, dropout, init_temp=init_temp)

    def forward(self, graph: DeltaGraph, use_router: bool = True,
                use_partitioning: bool = True,
                use_memory: bool = True,
                gumbel_temperature: float = 0.0) -> DeltaGraph:
        """
        Args:
            graph: input graph
            use_router: enable post-attention importance scoring
            use_partitioning: enable graph partitioning (useful at scale)
            use_memory: enable tiered memory management
            gumbel_temperature: ignored (kept for backward compat)
        """
        # --- Step 1: Memory tier management (before attention) ---
        if use_memory:
            graph = self.memory.compress_warm_nodes(graph)
            active_graph = self.memory.get_active_subgraph(graph)
        else:
            active_graph = graph

        # --- Step 2: Partition ---
        if use_partitioning and active_graph.num_nodes > self.partitioner.max_partition_size:
            partitions = self.partitioner.partition(
                active_graph,
                importance=active_graph.node_importance
            )
        else:
            partitions = [torch.arange(active_graph.num_nodes, device=active_graph.device)]

        # --- Step 3: Full dual parallel attention (attend FIRST, then prune) ---
        edge_adj = active_graph.build_edge_adjacency()

        if len(partitions) == 1:
            result = self.dual_attn(active_graph, edge_adj=edge_adj, return_weights=use_router)
        else:
            # Run full attention, then global boundary communication
            result = self.dual_attn(active_graph, edge_adj=edge_adj, return_weights=use_router)

        if use_router and isinstance(result, tuple):
            active_graph, node_attn_w, edge_attn_w = result

            # Post-attention importance scoring from observed weights
            node_scores, edge_scores = self.pruner.compute_importance(
                active_graph, node_attn_w, edge_attn_w
            )
            active_graph.node_importance = node_scores
            active_graph.edge_importance = edge_scores

            # Update memory tiers based on observed importance
            new_tiers = self.pruner.update_tiers(active_graph, node_scores)
            active_graph.node_tiers = new_tiers
        else:
            active_graph = result if not isinstance(result, tuple) else result[0]

        # --- Step 4: Cross-partition global attention ---
        if len(partitions) > 1:
            boundary_lists = self.partitioner.get_boundary_nodes(active_graph, partitions)
            if boundary_lists:
                all_boundary = torch.unique(torch.cat(boundary_lists))
                if len(all_boundary) > 0:
                    boundary_mask = torch.zeros(active_graph.num_nodes, dtype=torch.bool,
                                                device=active_graph.device)
                    boundary_mask[all_boundary] = True
                    active_graph = DeltaGraph(
                        node_features=self.global_node_attn(active_graph, mask=boundary_mask),
                        edge_features=active_graph.edge_features,
                        edge_index=active_graph.edge_index,
                        node_tiers=active_graph.node_tiers,
                        node_importance=active_graph.node_importance,
                        edge_importance=active_graph.edge_importance,
                    )

        # --- Step 5: Write updated features back if we used memory subsetting ---
        if use_memory and graph.num_nodes != active_graph.num_nodes:
            active_mask = graph.hot_mask() | graph.warm_mask()
            active_indices = torch.where(active_mask)[0]
            new_node_feats = graph.node_features.clone()
            new_node_feats[active_indices] = active_graph.node_features

            graph = DeltaGraph(
                node_features=new_node_feats,
                edge_features=graph.edge_features,
                edge_index=graph.edge_index,
                node_tiers=graph.node_tiers,
                node_importance=graph.node_importance,
                edge_importance=graph.edge_importance,
            )
            # Propagate edge adjacency cache
            graph._edge_adj_cache = active_graph._edge_adj_cache
        else:
            graph = active_graph

        return graph


class DELTAModel(nn.Module):
    """Complete DELTA model with configurable depth and optional graph construction.

    For experiments that start with pre-built graphs (Phases 1-4),
    set use_constructor=False.
    For end-to-end experiments (Phase 5+), use_constructor=True
    to bootstrap graphs from raw token sequences.
    """

    def __init__(self, d_node: int = 64, d_edge: int = 32,
                 num_layers: int = 3, num_heads: int = 4,
                 max_partition_size: int = 32, dropout: float = 0.1,
                 sparse_ratio: float = 0.7,
                 init_temp: float = 1.0,
                 topk_edges: Optional[int] = None,
                 # Phase 60: residual gating for depth scaling
                 residual_gate: bool = False,
                 residual_gate_init: float = 0.1,
                 # Constructor params (Phase 5+)
                 use_constructor: bool = False,
                 vocab_size: int = 1000, d_model: int = 128,
                 constructor_layers: int = 2,
                 # Output head
                 num_classes: int = None):
        super().__init__()
        self.d_node = d_node
        self.d_edge = d_edge
        self.use_constructor = use_constructor
        self.residual_gate = residual_gate

        if use_constructor:
            self.constructor = GraphConstructor(
                vocab_size=vocab_size, d_model=d_model,
                d_node=d_node, d_edge=d_edge,
                num_layers=constructor_layers, num_heads=num_heads,
            )

        self.layers = nn.ModuleList([
            DELTALayer(d_node, d_edge, num_heads, max_partition_size,
                       dropout, sparse_ratio, init_temp=init_temp,
                       topk_edges=topk_edges)
            for _ in range(num_layers)
        ])

        # Phase 60: learnable per-layer residual gates
        # gate_alpha in [0,1] via sigmoid; initialized so sigmoid(x) ≈ residual_gate_init
        if residual_gate and num_layers > 1:
            import math
            init_logit = math.log(residual_gate_init / (1 - residual_gate_init))
            self.node_gate_logits = nn.ParameterList([
                nn.Parameter(torch.tensor(init_logit))
                for _ in range(num_layers)
            ])
            self.edge_gate_logits = nn.ParameterList([
                nn.Parameter(torch.tensor(init_logit))
                for _ in range(num_layers)
            ])
        else:
            self.node_gate_logits = None
            self.edge_gate_logits = None
        self.classifier = None
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(d_node, d_node),
                nn.GELU(),
                nn.Linear(d_node, num_classes),
            )

        # Edge classification head (for link prediction / relation typing)
        self.edge_classifier = None
        if num_classes is not None:
            self.edge_classifier = nn.Sequential(
                nn.Linear(d_edge, d_edge),
                nn.GELU(),
                nn.Linear(d_edge, num_classes),
            )

    def forward(self, input_data, use_router: bool = True,
                use_partitioning: bool = False,
                use_memory: bool = False,
                gumbel_temperature: float = 0.0) -> DeltaGraph:
        """
        Args:
            input_data: either a DeltaGraph (Phases 1-4) or token_ids tensor (Phase 5+)
            use_router: enable importance routing
            use_partitioning: enable graph partitioning
            use_memory: enable tiered memory
            gumbel_temperature: if > 0, use differentiable Gumbel-softmax routing

        Returns:
            Processed DeltaGraph with updated features
        """
        if self.use_constructor and not isinstance(input_data, DeltaGraph):
            graph = self.constructor(input_data)
        else:
            graph = input_data

        for i, layer in enumerate(self.layers):
            if self.node_gate_logits is not None:
                # Save pre-layer features for residual gating
                prev_node = graph.node_features
                prev_edge = graph.edge_features

            graph = layer(graph, use_router=use_router,
                         use_partitioning=use_partitioning,
                         use_memory=use_memory,
                         gumbel_temperature=gumbel_temperature)

            if self.node_gate_logits is not None:
                # alpha near 0 at init → output ≈ prev (residual dominates)
                node_alpha = torch.sigmoid(self.node_gate_logits[i])
                edge_alpha = torch.sigmoid(self.edge_gate_logits[i])
                gated_node = node_alpha * graph.node_features + (1 - node_alpha) * prev_node
                gated_edge = edge_alpha * graph.edge_features + (1 - edge_alpha) * prev_edge
                cached = getattr(graph, '_edge_adj_cache', None)
                graph = DeltaGraph(
                    node_features=gated_node,
                    edge_features=gated_edge,
                    edge_index=graph.edge_index,
                    node_tiers=getattr(graph, 'node_tiers', None),
                    node_importance=getattr(graph, 'node_importance', None),
                    edge_importance=getattr(graph, 'edge_importance', None),
                )
                graph._edge_adj_cache = cached

        return graph

    def classify_nodes(self, graph: DeltaGraph) -> torch.Tensor:
        """Node-level classification. Returns [N, num_classes] logits."""
        assert self.classifier is not None
        return self.classifier(graph.node_features)

    def classify_edges(self, graph: DeltaGraph) -> torch.Tensor:
        """Edge-level classification. Returns [E, num_classes] logits."""
        assert self.edge_classifier is not None
        return self.edge_classifier(graph.edge_features)

    def predict_link(self, graph: DeltaGraph, src: torch.Tensor,
                     tgt: torch.Tensor) -> torch.Tensor:
        """Score potential edges between src and tgt node pairs.

        Returns [num_pairs] scores.
        """
        src_feats = graph.node_features[src]
        tgt_feats = graph.node_features[tgt]
        return (src_feats * tgt_feats).sum(dim=-1)
