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

    def __init__(self, d_node, d_edge, target_density=0.005):
        super().__init__()
        self.d_edge = d_edge
        self.target_density = target_density

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
          Phase 1 (no grad): Score all N² pairs, select top-k by logit value
          Phase 2 (with grad): Re-score selected k pairs for gradient flow

        This gives O(N²) forward but only O(k) backward.

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

        # --- Phase 1: Select top-k edges (no gradient, O(N²)) ---
        k = int(self.target_density * N * (N - 1))
        k = max(k, N)  # at least N edges

        with torch.no_grad():
            src_exp = node_features.unsqueeze(1).expand(-1, N, -1)
            tgt_exp = node_features.unsqueeze(0).expand(N, -1, -1)
            all_pairs = torch.cat([src_exp, tgt_exp], dim=-1)     # [N, N, 2d]
            logits_all = self.edge_scorer(all_pairs).squeeze(-1)  # [N, N]
            logits_all.fill_diagonal_(float('-inf'))
            _, topk_idx = logits_all.view(-1).topk(min(k, logits_all.numel()))
            src_idx = topk_idx // N
            tgt_idx = topk_idx % N

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
    """3-stage self-bootstrap encoder for the Brain architecture.

    Stage 1: Bootstrap DELTALayers on input graph → enriched features
    Stage 2: BrainConstructor → learn new edges from enriched features
    Stage 3: Full DELTALayers on augmented graph → final features

    When hybrid=True (default), Stage 3 operates on original edges + new edges.
    When hybrid=False, Stage 3 uses only constructed edges.

    This is the core of The Brain: DELTA constructs its own graph and reasons
    over it, reducing dependency on pre-defined topology.
    """

    def __init__(self, d_node, d_edge, bootstrap_layers=1, delta_layers=2,
                 num_heads=4, dropout=0.1, target_density=0.005,
                 hybrid=True, init_temp=1.0, topk_edges=None,
                 use_router_in_delta=False):
        super().__init__()

        self.hybrid = hybrid
        self.use_router_in_delta = use_router_in_delta
        self.last_sparsity_loss = torch.tensor(0.0)
        self.last_num_constructed_edges = 0

        # Stage 1: Bootstrap DELTA layers (router always OFF — enrichment only)
        self.bootstrap_layers = nn.ModuleList([
            DELTALayer(d_node, d_edge, num_heads, dropout=dropout,
                       init_temp=init_temp, topk_edges=topk_edges)
            for _ in range(bootstrap_layers)
        ])

        # Bridge between Stage 1 and Stage 2/3
        # Prevents gradient shortcut where Stage 3 ignores Stage 1
        self.node_bridge = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, d_node),
            nn.GELU(),
        )
        self.edge_bridge = nn.Sequential(
            nn.LayerNorm(d_edge),
            nn.Linear(d_edge, d_edge),
            nn.GELU(),
        )

        # Stage 2: Differentiable graph constructor
        self.constructor = BrainConstructor(d_node, d_edge, target_density)

        # Stage 3: Full DELTA layers on augmented graph
        # topk_edges applies to Stage 3 attention over augmented E_adj
        self.delta_layers = nn.ModuleList([
            DELTALayer(d_node, d_edge, num_heads, dropout=dropout,
                       init_temp=init_temp, topk_edges=topk_edges)
            for _ in range(delta_layers)
        ])

    def forward(self, graph):
        """Process graph through 3-stage Brain pipeline.

        Args:
            graph: DeltaGraph with node_features, edge_features, edge_index

        Returns:
            DeltaGraph with enriched features (on augmented graph)

        Memory note (N=5000):
            Without checkpointing, Stage 1 saves ~75GB of activations for
            backward. Stage 3 forward then needs another ~75GB → OOM on 98GB.
            Gradient checkpointing frees Stage 1 activations after forward
            and recomputes them during backward. Peak = max(75, 75) = 75GB.
        """
        from torch.utils.checkpoint import checkpoint as _ckpt
        original_edge_index = graph.edge_index
        orig_adj = graph._edge_adj_cache[1] if graph._edge_adj_cache is not None else None

        # Stage 1: Bootstrap with gradient checkpointing.
        # Activations are freed after this forward, then recomputed during backward.
        def _run_bootstrap(nf, ef, ei):
            g = DeltaGraph(node_features=nf, edge_features=ef, edge_index=ei)
            if orig_adj is not None:
                g._edge_adj_cache = (1, orig_adj)  # inject cached E_adj (captured)
            for layer in self.bootstrap_layers:
                g = layer(g, use_router=False, use_partitioning=False, use_memory=False)
            return g.node_features, g.edge_features

        boot_nf, boot_ef = _ckpt(
            _run_bootstrap,
            graph.node_features, graph.edge_features, graph.edge_index,
            use_reentrant=False,
        )

        # Bridge: transform features between stages
        bridged_nf = self.node_bridge(boot_nf)
        bridged_ef = self.edge_bridge(boot_ef)

        # Stage 2: Construct new edges from enriched features
        tau = getattr(self, '_constructor_tau', 1.0)
        new_ei, new_ef, sparsity_loss = self.constructor(bridged_nf, tau=tau)

        self.last_sparsity_loss = sparsity_loss
        self.last_num_constructed_edges = new_ei.shape[1]

        # Build augmented graph
        if self.hybrid and original_edge_index.shape[1] > 0:
            # Keep original edges + add new edges
            aug_ei = torch.cat([original_edge_index, new_ei], dim=1)
            aug_ef = torch.cat([bridged_ef, new_ef], dim=0)
        else:
            # Only constructed edges
            aug_ei = new_ei
            aug_ef = new_ef

        augmented_graph = DeltaGraph(
            node_features=bridged_nf,
            edge_features=aug_ef,
            edge_index=aug_ei,
        )

        # Stage 3: Full DELTA on augmented graph
        if not self.use_router_in_delta:
            # Router OFF: edge_index unchanged across layers → cache E_adj once
            aug_edge_adj = augmented_graph.build_edge_adjacency()
            # Free E_adj build temps from allocator pool before Stage 3 forward
            torch.cuda.empty_cache()
            for layer in self.delta_layers:
                augmented_graph = layer(augmented_graph, use_router=False,
                                       use_partitioning=False, use_memory=False)
                # Restore cache on new graph object (same edge_index)
                augmented_graph._edge_adj_cache = (1, aug_edge_adj)
        else:
            # Router ON: pruning changes edge_index each layer → rebuild E_adj per layer
            for layer in self.delta_layers:
                augmented_graph.build_edge_adjacency()
                torch.cuda.empty_cache()
                augmented_graph = layer(augmented_graph, use_router=True,
                                       use_partitioning=False, use_memory=False)

        return augmented_graph
