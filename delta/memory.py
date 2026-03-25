"""
Tiered Memory for DELTA.

Three tiers:
- Hot:  Active nodes/edges with full attention. The working set.
- Warm: Compressed representations with sparse attention. Recently relevant.
- Cold: Archived nodes, retrieval-only. Not in attention computation.

The graph at rest IS the memory. Forgetting = edge pruning + node absorption.
This is not a separate memory system bolted on — it's the graph structure itself
managed by importance scores from the router.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple

from delta.graph import DeltaGraph, TIER_HOT, TIER_WARM, TIER_COLD


class TieredMemory(nn.Module):
    """Manages memory tiers within the graph.

    Handles:
    - Compression of warm-tier nodes to reduced dimensionality
    - Decompression when warm nodes get promoted back to hot
    - Cold storage (detached from computation graph)
    - Node absorption: merging similar cold nodes to free capacity
    """

    def __init__(self, d_node: int, d_edge: int, warm_dim: int = None):
        super().__init__()
        self.d_node = d_node
        self.d_edge = d_edge
        self.warm_dim = warm_dim or d_node // 2

        # Variational bottleneck compression: learned per-node compression
        # Encoder produces mean and log-variance for warm representation
        self.node_enc_mu = nn.Linear(d_node, self.warm_dim)
        self.node_enc_logvar = nn.Linear(d_node, self.warm_dim)
        self.node_decompress = nn.Linear(self.warm_dim, d_node)

        self.edge_enc_mu = nn.Linear(d_edge, self.warm_dim)
        self.edge_enc_logvar = nn.Linear(d_edge, self.warm_dim)
        self.edge_decompress = nn.Linear(self.warm_dim, d_edge)

        # Learned similarity threshold for cold absorption
        self._sim_threshold_logit = nn.Parameter(torch.tensor(1.7))  # sigmoid(1.7) ≈ 0.85

    @property
    def similarity_threshold(self) -> float:
        """Learned similarity threshold for cold node absorption."""
        return torch.sigmoid(self._sim_threshold_logit).item()

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample from N(mu, sigma^2)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def compress_warm_nodes(self, graph: DeltaGraph) -> DeltaGraph:
        """Compress warm-tier nodes via variational bottleneck.

        Uses reparameterization trick during training for gradient flow.
        The KL divergence term (self.kl_loss) should be added to the
        training loss to regularize the latent space.
        """
        warm = graph.warm_mask()
        if not warm.any():
            self.kl_loss = torch.tensor(0.0, device=graph.device)
            return graph

        new_features = graph.node_features.clone()
        warm_feats = graph.node_features[warm]

        mu = self.node_enc_mu(warm_feats)
        logvar = self.node_enc_logvar(warm_feats)
        z = self._reparameterize(mu, logvar)
        new_features[warm] = self.node_decompress(z)

        # Store KL loss for training: KL(q(z|x) || N(0,1))
        self.kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return DeltaGraph(
            node_features=new_features,
            edge_features=graph.edge_features,
            edge_index=graph.edge_index,
            node_tiers=graph.node_tiers,
            node_importance=graph.node_importance,
            edge_importance=graph.edge_importance,
        )

    def get_active_subgraph(self, graph: DeltaGraph) -> DeltaGraph:
        """Extract the hot + warm subgraph for attention computation.

        Cold nodes are excluded from attention entirely — they're archived
        and only accessible via explicit retrieval.
        """
        active_mask = graph.hot_mask() | graph.warm_mask()
        return graph.subgraph(active_mask)

    def retrieve_from_cold(self, graph: DeltaGraph, query: torch.Tensor,
                           top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cold nodes most similar to a query vector.

        This is the retrieval-only access path for archived nodes.
        Cold nodes can be promoted back to warm/hot if retrieved.

        Args:
            query: [d_node] or [B, d_node] query vector(s)
            top_k: number of cold nodes to retrieve

        Returns:
            features: [top_k, d_node] retrieved node features
            indices: [top_k] original indices in the graph
        """
        cold = graph.cold_mask()
        if not cold.any():
            empty_feats = torch.zeros(0, graph.d_node, device=graph.device)
            empty_idx = torch.zeros(0, dtype=torch.long, device=graph.device)
            return empty_feats, empty_idx

        cold_indices = torch.where(cold)[0]
        cold_features = graph.node_features[cold_indices]

        if query.dim() == 1:
            query = query.unsqueeze(0)

        # Cosine similarity
        query_norm = query / (query.norm(dim=-1, keepdim=True) + 1e-10)
        cold_norm = cold_features / (cold_features.norm(dim=-1, keepdim=True) + 1e-10)
        sim = (query_norm @ cold_norm.T).squeeze(0)  # [num_cold]

        k = min(top_k, len(cold_indices))
        _, top_idx = torch.topk(sim, k)

        return cold_features[top_idx], cold_indices[top_idx]

    def absorb_similar_cold(self, graph: DeltaGraph) -> DeltaGraph:
        """Merge highly similar cold nodes to reduce graph size.

        This is how DELTA "forgets" — not by deleting, but by absorbing
        redundant information into fewer nodes. Edges of absorbed nodes
        are redirected to the surviving node.
        """
        cold = graph.cold_mask()
        if cold.sum() < 2:
            return graph

        cold_indices = torch.where(cold)[0]
        cold_feats = graph.node_features[cold_indices].detach()  # stop gradient on cold

        # Pairwise cosine similarity among cold nodes
        norms = cold_feats / (cold_feats.norm(dim=-1, keepdim=True) + 1e-10)
        sim_matrix = norms @ norms.T

        # Use learned threshold
        threshold = torch.sigmoid(self._sim_threshold_logit).item()

        # Find pairs above threshold (upper triangle only)
        mask = torch.triu(sim_matrix > threshold, diagonal=1)

        if not mask.any():
            return graph

        # Greedy merge: for each similar pair, absorb the second into the first
        absorbed = set()
        redirect_map = {}  # old_idx -> surviving_idx

        for i in range(mask.shape[0]):
            if i in absorbed:
                continue
            for j in range(i + 1, mask.shape[1]):
                if j in absorbed:
                    continue
                if mask[i, j]:
                    absorbed.add(j)
                    redirect_map[cold_indices[j].item()] = cold_indices[i].item()
                    # Average features into the surviving node
                    graph.node_features[cold_indices[i]] = (
                        graph.node_features[cold_indices[i]] +
                        graph.node_features[cold_indices[j]]
                    ) / 2.0

        if not redirect_map:
            return graph

        # Redirect edges from absorbed nodes
        new_edge_index = graph.edge_index.clone()
        for old_idx, new_idx in redirect_map.items():
            new_edge_index[0][new_edge_index[0] == old_idx] = new_idx
            new_edge_index[1][new_edge_index[1] == old_idx] = new_idx

        # Remove self-loops created by redirection
        valid_edges = new_edge_index[0] != new_edge_index[1]

        # Remove absorbed nodes
        keep_mask = torch.ones(graph.num_nodes, dtype=torch.bool, device=graph.device)
        for old_idx in redirect_map:
            keep_mask[old_idx] = False

        # Remap edge indices to new node numbering
        node_map = torch.full((graph.num_nodes,), -1, dtype=torch.long, device=graph.device)
        node_map[keep_mask] = torch.arange(keep_mask.sum(), device=graph.device)

        final_edge_index = node_map[new_edge_index[:, valid_edges]]

        return DeltaGraph(
            node_features=graph.node_features[keep_mask],
            edge_features=graph.edge_features[valid_edges],
            edge_index=final_edge_index,
            node_tiers=graph.node_tiers[keep_mask],
            node_importance=graph.node_importance[keep_mask] if graph.node_importance is not None else None,
            edge_importance=graph.edge_importance[valid_edges] if graph.edge_importance is not None else None,
        )
