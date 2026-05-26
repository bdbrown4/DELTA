"""Brain Architecture: Differentiable Graph Construction + Self-Bootstrap.

Ports the Phase 46b SelfBootstrapped DELTA concept into the core package,
replacing hard-threshold edge selection (GraphConstructor) with Gumbel-sigmoid
differentiable construction.

BrainConstructor: Learns which edges to add from enriched node features
BrainEncoder: 3-stage pipeline (bootstrap → construct → reason)

This is Horizon 2→3 of The Brain roadmap:
  - Horizon 2: Adaptive Architecture (constructor fixes)
  - Horizon 3: Sequence Domain Generalization (structure discovery)

Key improvements over delta/constructor.py's GraphConstructor:
  1. Differentiable: Gumbel-sigmoid replaces hard attention threshold
  2. Active edge types: Edge features are learned outputs, not dead weights
  3. Feature-based: MLP scores edges from enriched features, not raw attention

Reference:
  Phase 46b: SelfBootstrappedDELTA achieved +57% over FixedChain on path composition
  Phase 39: Self-bootstrap at 157% of FixedChain (Brain page)
  Phase 40: SelfBootstrapHybrid MRR 0.5089 on FB15k-237 (Brain page)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.model import DELTALayer


def gumbel_sigmoid(logits, tau=1.0, hard=False):
    """Differentiable binary selection via Gumbel-sigmoid.

    Args:
        logits: raw scores [...]
        tau: temperature (lower = sharper decisions)
        hard: if True, use straight-through estimator for hard 0/1

    Returns:
        Soft or hard binary selections in [0, 1], same shape as logits.
    """
    if not logits.requires_grad:
        return torch.sigmoid(logits)

    # Sample Gumbel noise
    u = torch.rand_like(logits).clamp(1e-10, 1 - 1e-10)
    gumbel = -torch.log(-torch.log(u))
    y_soft = torch.sigmoid((logits + gumbel) / tau)

    if hard:
        y_hard = (y_soft > 0.5).float()
        return y_hard - y_soft.detach() + y_soft  # STE
    return y_soft


class BrainConstructor(nn.Module):
    """Differentiable graph constructor using Gumbel-sigmoid edge selection.

    Given enriched node features [N, d_node], scores all potential edges via
    MLP and selects a sparse subset using Gumbel-sigmoid for differentiable
    training. Returns new edges with learned features.

    Architecture (from Phase 46b FeatureBasedConstructor):
        Concatenate source + target features → MLP → edge score
        Gumbel-sigmoid → soft binary selection
        Project concatenated features → edge features (weighted by prob)
    """

    def __init__(self, d_node, d_edge, target_density=0.005, chunk_size=64):
        super().__init__()
        self.d_edge = d_edge
        self.target_density = target_density
        self._chunk_size = chunk_size  # rows processed per iteration in Phase 1

        # Edge score MLP: (src || tgt) → score
        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * d_node, d_node),
            nn.GELU(),
            nn.Linear(d_node, 1),
        )

        # Edge feature projector: (src || tgt) → d_edge features
        self.edge_projector = nn.Sequential(
            nn.Linear(2 * d_node, d_edge),
            nn.GELU(),
        )

    def forward(self, node_features, tau=1.0, hard=False):
        """Score and select edges from node features.

        Two-phase approach to avoid O(N²) backward:
          Phase 1 (no grad): Score N² pairs in row-chunks, select top-k by logit.
            Memory: O(chunk_size × N) instead of O(N²).
            Collects 2k candidates across all chunks then takes global top-k.
          Phase 2 (with grad): Re-score selected k pairs for gradient flow.
            Memory: O(k). Backward only through k pairs.

        Args:
            node_features: [N, d_node] enriched node features
            tau: Gumbel temperature (lower = more discrete)
            hard: Use STE for hard binary decisions

        Returns:
            edge_index: [2, E'] indices of selected edges
            edge_features: [E', d_edge] features for new edges
            confidence_loss: scalar encouraging high probs for selected edges
        """
        N, d = node_features.shape
        device = node_features.device

        k = int(self.target_density * N * (N - 1))
        k = max(k, N)  # at least N edges

        # --- Phase 1: chunked row-wise scoring (no gradient) ---
        # Process chunk_size rows at a time → peak alloc O(chunk_size × N × 2d).
        # Collect a 2× safety buffer of candidates per chunk, then global top-k.
        with torch.no_grad():
            chunk_size   = self._chunk_size
            # Candidates to keep per chunk: (2k / total_chunks) + margin.
            # 2× safety factor ensures the true global top-k survives the cull.
            cand_per_row = max(2 * k // N + 1, 1)

            cand_scores: list = []
            cand_flat:   list = []

            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                C   = end - start

                # [C, N, 2d] — peak chunk allocation
                pairs = torch.cat([
                    node_features[start:end].unsqueeze(1).expand(-1, N, -1),
                    node_features.unsqueeze(0).expand(C, -1, -1),
                ], dim=-1)
                logits = self.edge_scorer(pairs).squeeze(-1)  # [C, N]

                # Mask self-loops (row i ↔ global node start+i)
                logits[
                    torch.arange(C, device=device),
                    torch.arange(start, start + C, device=device),
                ] = float('-inf')

                # Keep top-(cand_per_row * C) from this chunk
                buf  = min(cand_per_row * C, C * N)
                flat = logits.reshape(-1)
                top_scores, top_local = flat.topk(min(buf, flat.numel()))

                row_global = top_local // N + start
                col_idx    = top_local %  N
                cand_scores.append(top_scores)
                cand_flat.append(row_global * N + col_idx)

            # Global top-k from all chunk candidates
            all_scores = torch.cat(cand_scores)
            all_flat   = torch.cat(cand_flat)
            _, final_idx = all_scores.topk(min(k, all_scores.numel()))
            topk_flat  = all_flat[final_idx]
            src_idx    = topk_flat // N
            tgt_idx    = topk_flat %  N

        # --- Phase 2: Re-score selected pairs WITH gradient (O(k)) ---
        sel_src = node_features[src_idx]  # [k, d]
        sel_tgt = node_features[tgt_idx]  # [k, d]
        sel_pairs = torch.cat([sel_src, sel_tgt], dim=-1)  # [k, 2d]

        sel_logits = self.edge_scorer(sel_pairs).squeeze(-1)  # [k]
        if self.training:
            probs = gumbel_sigmoid(sel_logits, tau=tau, hard=hard)
        else:
            probs = torch.sigmoid(sel_logits)

        edge_index = torch.stack([src_idx, tgt_idx])

        # Edge features weighted by selection probability
        edge_features = self.edge_projector(sel_pairs)  # [k, d_edge]
        edge_features = edge_features * probs.unsqueeze(-1)

        # Confidence loss: encourage high probs for selected edges
        confidence_loss = (1.0 - probs).mean()

        return edge_index, edge_features, confidence_loss


class BrainEncoder(nn.Module):
    """Multi-round self-constructing graph encoder.

    Stage 0: Bootstrap DELTALayers (once) → initial structural enrichment
    Round r (for r in 0..rounds-1):
        Bridge → BrainConstructor → augment → DELTALayers

    Each round sees node/edge features enriched by all prior rounds.
    Edge-replace strategy: each round augments from ORIGINAL edges only
    (not cumulative), so constructed edges never accumulate and VRAM stays bounded.

    rounds=1 is backward-compatible with the original 3-stage pipeline.

    When hybrid=True (default), each round's DELTA operates on original + constructed edges.
    When hybrid=False, each round's DELTA uses only constructed edges.
    """

    def __init__(self, d_node, d_edge, bootstrap_layers=1, delta_layers=2,
                 num_heads=4, dropout=0.1, target_density=0.005,
                 hybrid=True, init_temp=1.0, topk_edges=None,
                 use_router_in_delta=False, rounds=1):
        super().__init__()

        self.hybrid = hybrid
        self.use_router_in_delta = use_router_in_delta
        self.rounds = rounds
        self.last_sparsity_loss = torch.tensor(0.0)
        self.last_num_constructed_edges = 0

        # Stage 0: Bootstrap DELTA layers (router always OFF — enrichment only)
        self.bootstrap_layers = nn.ModuleList([
            DELTALayer(d_node, d_edge, num_heads, dropout=dropout,
                       init_temp=init_temp, topk_edges=topk_edges)
            for _ in range(bootstrap_layers)
        ])

        # Per-round modules (rounds=1 gives same capacity as original single-round code)
        # Each round independently: bridges features, constructs edges, reasons over them.
        self.round_node_bridges = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_node), nn.Linear(d_node, d_node), nn.GELU())
            for _ in range(rounds)
        ])
        self.round_edge_bridges = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_edge), nn.Linear(d_edge, d_edge), nn.GELU())
            for _ in range(rounds)
        ])
        self.constructors = nn.ModuleList([
            BrainConstructor(d_node, d_edge, target_density)
            for _ in range(rounds)
        ])
        # rounds × delta_layers_per_round independent DELTA layers
        self.round_delta_layers = nn.ModuleList([
            nn.ModuleList([
                DELTALayer(d_node, d_edge, num_heads, dropout=dropout,
                           init_temp=init_temp, topk_edges=topk_edges)
                for _ in range(delta_layers)
            ])
            for _ in range(rounds)
        ])

    def forward(self, graph):
        """Process graph through multi-round Brain pipeline.

        Stage 0: Bootstrap pass on original graph (once).
        Round r (for r in 0..rounds-1):
            1. Bridge: LayerNorm + Linear projects current node/edge features.
            2. BrainConstructor: scores all node pairs (chunked O(chunk×N)),
               selects top-k edges, builds differentiable edge features.
            3. Augment: original edges + round-r constructed edges.
            4. DELTALayers: reason over augmented graph.

        Edge-replace strategy ensures VRAM stays bounded:
            Each round augments from ORIGINAL edges only. Round r's constructed
            edges are discarded before round r+1 constructs fresh ones.
            VRAM ≈ O(bootstrap_E_adj + augmented_E_adj_per_round) regardless of rounds.

        Memory note (98 GB server GPU, bootstrap_edge_budget path):
            Same budget logic as original 3-stage code applies to each round.
        """
        original_edge_index = graph.edge_index
        n_orig = original_edge_index.shape[1]
        bootstrap_budget = getattr(self, 'bootstrap_edge_budget', None)

        # Stage 0: Bootstrap pass on original graph (optionally subsampled E_adj)
        if bootstrap_budget is not None:
            if graph._edge_adj_cache is None:
                orig_adj = graph.build_edge_adjacency()
            else:
                _, orig_adj = graph._edge_adj_cache
            if orig_adj.shape[1] > bootstrap_budget:
                perm = torch.randperm(orig_adj.shape[1], device=orig_adj.device)
                sub_adj = orig_adj[:, perm[:bootstrap_budget]]
                graph._edge_adj_cache = (1, sub_adj)

        for layer in self.bootstrap_layers:
            graph = layer(graph, use_router=False,
                         use_partitioning=False, use_memory=False)

        # Multi-round construction loop
        total_sparsity_loss = graph.node_features.new_zeros(1).squeeze()
        total_new_edges = 0
        tau = getattr(self, '_constructor_tau', 1.0)
        stage3_budget = getattr(self, 'bootstrap_edge_budget', None)

        for r in range(self.rounds):
            # Bridge: project current features (prevents gradient shortcut across rounds)
            node_feats = self.round_node_bridges[r](graph.node_features)
            # Extract original-edge features: first n_orig rows of edge_features.
            # After round r-1, graph.edge_features = [orig_ef_updated | constructed_ef_{r-1}].
            # We take the original slice and re-bridge it for this round.
            orig_ef_slice = graph.edge_features[:n_orig]
            edge_feats = self.round_edge_bridges[r](orig_ef_slice)

            # Construct new edges from current (bridged) node features
            new_ei, new_ef, sparsity_loss = self.constructors[r](node_feats, tau=tau)
            total_sparsity_loss = total_sparsity_loss + sparsity_loss
            total_new_edges += new_ei.shape[1]

            # Build augmented graph (edge-replace: original + this round's edges)
            if self.hybrid and n_orig > 0:
                aug_ei = torch.cat([original_edge_index, new_ei], dim=1)
                aug_ef = torch.cat([edge_feats, new_ef], dim=0)
            else:
                aug_ei = new_ei
                aug_ef = new_ef

            augmented_graph = DeltaGraph(
                node_features=node_feats,
                edge_features=aug_ef,
                edge_index=aug_ei,
            )

            # Reason over augmented graph
            if not self.use_router_in_delta:
                # Router OFF: edge_index fixed across layers → cache E_adj once per round
                aug_edge_adj = augmented_graph.build_edge_adjacency()
                if stage3_budget is not None and aug_edge_adj.shape[1] > stage3_budget:
                    perm3 = torch.randperm(aug_edge_adj.shape[1], device=aug_edge_adj.device)
                    aug_edge_adj = aug_edge_adj[:, perm3[:stage3_budget]]
                    augmented_graph._edge_adj_cache = (1, aug_edge_adj)
                for layer in self.round_delta_layers[r]:
                    augmented_graph = layer(augmented_graph, use_router=False,
                                           use_partitioning=False, use_memory=False)
                    # Restore cache on new graph object returned by layer
                    augmented_graph._edge_adj_cache = (1, aug_edge_adj)
            else:
                # Router ON: pruning changes edge_index each layer → rebuild E_adj per layer
                for layer in self.round_delta_layers[r]:
                    aug_edge_adj = augmented_graph.build_edge_adjacency()
                    if stage3_budget is not None and aug_edge_adj.shape[1] > stage3_budget:
                        perm3 = torch.randperm(aug_edge_adj.shape[1], device=aug_edge_adj.device)
                        augmented_graph._edge_adj_cache = (1, aug_edge_adj[:, perm3[:stage3_budget]])
                    augmented_graph = layer(augmented_graph, use_router=True,
                                           use_partitioning=False, use_memory=False)

            # Pass enriched graph to next round
            # augmented_graph.edge_features[:n_orig] = orig-edge feats updated by this round
            graph = augmented_graph

        self.last_sparsity_loss = total_sparsity_loss / self.rounds
        self.last_num_constructed_edges = total_new_edges  # total across all rounds

        return graph
