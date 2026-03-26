"""
Core graph data structure for DELTA.

A DeltaGraph holds:
- Node features: [N, d_node] tensor — each node is a concept/memory unit
- Edge features: [E, d_edge] tensor — each edge is a typed relational representation
- Edge index: [2, E] tensor — source/target node indices per edge
- Node tiers: [N] tensor — memory tier (0=hot, 1=warm, 2=cold)
- Node/edge importance scores: cached from the router

Unlike standard graph representations where edges are passive scalar weights,
DELTA treats edges as first-class computational units with rich feature vectors.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Optional


TIER_HOT = 0
TIER_WARM = 1
TIER_COLD = 2


@dataclass
class DeltaGraph:
    """Core graph structure where nodes and edges are co-equal citizens."""

    node_features: torch.Tensor          # [N, d_node]
    edge_features: torch.Tensor          # [E, d_edge]
    edge_index: torch.Tensor             # [2, E] — row 0 = source, row 1 = target
    node_tiers: Optional[torch.Tensor] = None   # [N] — 0/1/2
    node_importance: Optional[torch.Tensor] = None  # [N]
    edge_importance: Optional[torch.Tensor] = None  # [E]

    def __post_init__(self):
        N = self.node_features.shape[0]
        if self.node_tiers is None:
            self.node_tiers = torch.zeros(N, dtype=torch.long,
                                          device=self.node_features.device)
        self._edge_adj_cache = None  # (hops, result) tuple

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_features.shape[0]

    @property
    def d_node(self) -> int:
        return self.node_features.shape[1]

    @property
    def d_edge(self) -> int:
        return self.edge_features.shape[1]

    @property
    def device(self) -> torch.device:
        return self.node_features.device

    def hot_mask(self) -> torch.Tensor:
        """Boolean mask for nodes in the hot (active) tier."""
        return self.node_tiers == TIER_HOT

    def warm_mask(self) -> torch.Tensor:
        """Boolean mask for nodes in the warm (compressed) tier."""
        return self.node_tiers == TIER_WARM

    def cold_mask(self) -> torch.Tensor:
        """Boolean mask for nodes in the cold (archived) tier."""
        return self.node_tiers == TIER_COLD

    def edges_for_node(self, node_idx: int) -> torch.Tensor:
        """Return edge indices where node_idx is source or target."""
        src_mask = self.edge_index[0] == node_idx
        tgt_mask = self.edge_index[1] == node_idx
        return torch.where(src_mask | tgt_mask)[0]

    def neighbor_edges(self, edge_idx: int) -> torch.Tensor:
        """Return indices of edges that share at least one endpoint with edge_idx.

        This is the structural adjacency used for edge-to-edge attention:
        two edges are neighbors if they share a node.
        """
        src = self.edge_index[0, edge_idx]
        tgt = self.edge_index[1, edge_idx]
        src_match = (self.edge_index[0] == src) | (self.edge_index[1] == src)
        tgt_match = (self.edge_index[0] == tgt) | (self.edge_index[1] == tgt)
        neighbors = src_match | tgt_match
        # Exclude self
        neighbors[edge_idx] = False
        return torch.where(neighbors)[0]

    def build_edge_adjacency(self, hops: int = 1) -> torch.Tensor:
        """Build a sparse edge-to-edge adjacency based on shared endpoints.

        Args:
            hops: 1 = edges sharing an endpoint (default, original behavior)
                  2 = also include edges connected through an intermediate edge
                  Higher hops enable multi-hop relational reasoning (e.g.,
                  composing worksAt + locatedIn → livesIn).

        Returns [2, E_adj] tensor of (edge_i, edge_j) pairs where edges
        are reachable within `hops` edge-hops. Used for edge-to-edge attention.
        """
        # Return cached result if available
        if self._edge_adj_cache is not None:
            cached_hops, cached_result = self._edge_adj_cache
            if cached_hops >= hops:
                return cached_result

        E = self.num_edges
        if E == 0:
            result = torch.zeros(2, 0, dtype=torch.long, device=self.device)
            self._edge_adj_cache = (hops, result)
            return result

        src_nodes = self.edge_index[0]  # [E]
        tgt_nodes = self.edge_index[1]  # [E]

        if E <= 500:
            # Fast path: incidence matrix multiply (avoids Python for-loop)
            all_nodes = torch.cat([src_nodes, tgt_nodes])
            all_edge_ids = torch.arange(E, device=self.device).repeat(2)
            N = all_nodes.max().item() + 1
            # Build incidence matrix I[node, edge] = 1 if edge touches node
            inc = torch.zeros(N, E, device=self.device)
            inc[all_nodes, all_edge_ids] = 1.0
            # I^T @ I gives co-incidence: adj[i,j] > 0 iff edges i,j share a node
            co = inc.T @ inc  # [E, E]
            co.fill_diagonal_(0)
            adj_1hop = co.nonzero(as_tuple=False).T.long()  # [2, num_pairs]
        else:
            # Original approach for large graphs (avoids O(E^2) dense matrix)
            edge_pairs_src = []
            edge_pairs_tgt = []

            all_nodes = torch.cat([src_nodes, tgt_nodes])
            all_edge_ids = torch.arange(E, device=self.device).repeat(2)

            for node_id in torch.unique(all_nodes):
                incident = all_edge_ids[all_nodes == node_id]
                if len(incident) < 2:
                    continue
                grid = torch.meshgrid(incident, incident, indexing='ij')
                mask = grid[0] != grid[1]
                edge_pairs_src.append(grid[0][mask])
                edge_pairs_tgt.append(grid[1][mask])

            if edge_pairs_src:
                adj_1hop = torch.stack([
                    torch.cat(edge_pairs_src),
                    torch.cat(edge_pairs_tgt)
                ])
                adj_1hop = self._deduplicate_edge_adj(adj_1hop)
            else:
                adj_1hop = torch.zeros(2, 0, dtype=torch.long, device=self.device)

        if hops <= 1 or adj_1hop.shape[1] == 0:
            self._edge_adj_cache = (hops, adj_1hop)
            return adj_1hop

        # Multi-hop: use SPARSE matrix powers to avoid O(E²) memory
        E = self.num_edges
        indices = adj_1hop  # [2, nnz]
        values = torch.ones(adj_1hop.shape[1], device=self.device)
        adj_sparse = torch.sparse_coo_tensor(indices, values, (E, E), device=self.device).coalesce()

        # Compose by sparse matrix multiplication
        combined = adj_sparse
        power = adj_sparse
        for _ in range(hops - 1):
            power = torch.sparse.mm(power, adj_sparse.to_dense()).to_sparse().coalesce()
            combined = (combined + power).coalesce()

        # Extract non-zero entries (excluding self-loops)
        combined = combined.coalesce()
        rows = combined.indices()[0]
        cols = combined.indices()[1]
        non_self = rows != cols
        rows, cols = rows[non_self], cols[non_self]
        if len(rows) == 0:
            self._edge_adj_cache = (hops, adj_1hop)
            return adj_1hop
        result = torch.stack([rows, cols])
        self._edge_adj_cache = (hops, result)
        return result

    def _deduplicate_edge_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Remove duplicate edge pairs from adjacency."""
        if adj.shape[1] == 0:
            return adj
        adj_flat = adj[0] * self.num_edges + adj[1]
        _, inverse, counts = torch.unique(adj_flat, return_inverse=True, return_counts=True)
        # Keep first occurrence of each unique pair
        first_occurrence = torch.zeros(len(counts), dtype=torch.long, device=self.device)
        first_occurrence.scatter_(0, inverse, torch.arange(adj.shape[1] - 1, -1, -1, device=self.device))
        return adj[:, first_occurrence.sort().values]

    def subgraph(self, node_mask: torch.Tensor) -> DeltaGraph:
        """Extract a subgraph containing only the masked nodes and their edges."""
        node_indices = torch.where(node_mask)[0]
        node_map = torch.full((self.num_nodes,), -1, dtype=torch.long,
                              device=self.device)
        node_map[node_indices] = torch.arange(len(node_indices), device=self.device)

        # Keep edges where both endpoints are in the subgraph
        src_in = node_mask[self.edge_index[0]]
        tgt_in = node_mask[self.edge_index[1]]
        edge_mask = src_in & tgt_in

        new_edge_index = node_map[self.edge_index[:, edge_mask]]

        return DeltaGraph(
            node_features=self.node_features[node_indices],
            edge_features=self.edge_features[edge_mask],
            edge_index=new_edge_index,
            node_tiers=self.node_tiers[node_indices] if self.node_tiers is not None else None,
            node_importance=self.node_importance[node_indices] if self.node_importance is not None else None,
            edge_importance=self.edge_importance[edge_mask] if self.edge_importance is not None else None,
        )

    def to(self, device: torch.device) -> DeltaGraph:
        """Move all tensors to the specified device."""
        return DeltaGraph(
            node_features=self.node_features.to(device),
            edge_features=self.edge_features.to(device),
            edge_index=self.edge_index.to(device),
            node_tiers=self.node_tiers.to(device) if self.node_tiers is not None else None,
            node_importance=self.node_importance.to(device) if self.node_importance is not None else None,
            edge_importance=self.edge_importance.to(device) if self.edge_importance is not None else None,
        )
