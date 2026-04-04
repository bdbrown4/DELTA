"""
Phase 46b: Self-Bootstrapped DELTA -- DELTA bootstraps DELTA

Phase 46 showed:
  - Hybrid (TF -> chain + learned edges):  0.452 +/-0.006, 98% of FixedChain
  - Pure Differentiable (TF -> all edges):  0.393 +/-0.017, 85% of FixedChain
  - FixedChain target:                      0.461 +/-0.034

The pure differentiable constructor failed because the MLP edge scorer can't
reliably discover sequential structure from transformer embeddings alone.

Key insight: replace the transformer bootstrap with FixedChain DELTA. The
sequential chain is trivially constructible from any ordered input. Running
DELTA's own relational machinery (dual attention + ReconciliationBridge) on
it produces relationally-enriched features where adjacent nodes have already
absorbed each other's context -- making the edge scorer's job much easier.

Pipeline (no transformer at all):
  Stage 1: Embedding -> trivial sequential chain -> DELTALayers -> enriched features
  Stage 2: FeatureBasedConstructor scores edges on enriched features
  Stage 3: Full DELTA on dynamically-constructed graph -> classification

Success criterion: SelfBootstrapped >= FixedChain (0.461)
Target: close the pure-differentiable gap (0.393 -> 0.461+)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.model import DELTALayer
from delta.constructor import PositionalEncoding
from delta.graph import DeltaGraph

# Reuse shared components from Phase 46
from phase46_differentiable_constructor import (
    generate_path_composition_task,
    TransformerBaseline,
    FixedChainDELTA,
    DifferentiableDELTA,
    HybridDifferentiableConstructor,
    train_and_evaluate,
    linear_anneal,
)


# ----------------------------------------------------------------------
#  Feature-based constructors (operate on DELTA-enriched features)
# ----------------------------------------------------------------------

def gumbel_sigmoid(logits, temperature, hard=False):
    """Differentiable approximation to Bernoulli sampling."""
    u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
    gumbel_noise = -torch.log(-torch.log(u))
    y_soft = torch.sigmoid((logits + gumbel_noise) / temperature)
    if hard:
        y_hard = (y_soft > 0.5).float()
        return y_hard - y_soft.detach() + y_soft
    return y_soft


class FeatureBasedConstructor(nn.Module):
    """Differentiable edge selection from pre-computed node features.

    Unlike DifferentiableConstructor (which embeds tokens via its own
    transformer), this operates on features already enriched by DELTA layers.
    The edge scorer sees relational context, not raw embeddings.
    """

    def __init__(self, d_node, d_edge, target_density=0.3, temperature_init=1.0):
        super().__init__()
        self.target_density = target_density
        self.temperature = temperature_init

        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * d_node + 1, d_node),
            nn.GELU(),
            nn.Linear(d_node, 1),
        )
        self.to_edge = nn.Linear(2 * d_node + 1, d_edge)
        self.sparsity_loss = 0.0

    def forward(self, node_features, hard=False):
        """
        Args:
            node_features: [S, d_node] from bootstrap DELTA stage
        Returns:
            DeltaGraph with differentiable edge gates, edge_gates tensor
        """
        S = node_features.shape[0]
        device = node_features.device

        src_idx = torch.arange(S, device=device).unsqueeze(1).expand(S, S).reshape(-1)
        tgt_idx = torch.arange(S, device=device).unsqueeze(0).expand(S, S).reshape(-1)
        mask = src_idx != tgt_idx
        src_idx = src_idx[mask]
        tgt_idx = tgt_idx[mask]

        pos_diff = (src_idx.float() - tgt_idx.float()).unsqueeze(-1) / S
        src_feat = node_features[src_idx]
        tgt_feat = node_features[tgt_idx]
        scorer_input = torch.cat([src_feat, tgt_feat, pos_diff], dim=-1)

        edge_logits = self.edge_scorer(scorer_input).squeeze(-1)
        edge_gates = gumbel_sigmoid(edge_logits, self.temperature, hard=hard)

        self.sparsity_loss = (edge_gates.mean() - self.target_density) ** 2

        edge_features = self.to_edge(scorer_input) * edge_gates.unsqueeze(-1)
        edge_index = torch.stack([src_idx, tgt_idx])

        return DeltaGraph(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
        ), edge_gates


class FeatureBasedHybridConstructor(FeatureBasedConstructor):
    """Preserves sequential chain edges (gate=1) + learns additional edges.

    Same hybrid principle as Phase 46's HybridDifferentiableConstructor,
    but operating on DELTA-enriched features instead of transformer embeddings.
    """

    def forward(self, node_features, hard=False):
        S = node_features.shape[0]
        device = node_features.device

        # Base chain (always kept)
        fwd_src = torch.arange(S - 1, device=device)
        fwd_tgt = torch.arange(1, S, device=device)
        base_src = torch.cat([fwd_src, fwd_tgt])
        base_tgt = torch.cat([fwd_tgt, fwd_src])
        num_base = base_src.shape[0]

        # Candidate edges (all non-self, non-base)
        all_src = torch.arange(S, device=device).unsqueeze(1).expand(S, S).reshape(-1)
        all_tgt = torch.arange(S, device=device).unsqueeze(0).expand(S, S).reshape(-1)
        not_self = all_src != all_tgt
        all_src = all_src[not_self]
        all_tgt = all_tgt[not_self]

        base_set = base_src * S + base_tgt
        cand_set = all_src * S + all_tgt
        is_base = torch.isin(cand_set, base_set)
        cand_src = all_src[~is_base]
        cand_tgt = all_tgt[~is_base]

        # Score candidates on enriched features
        pos_diff = (cand_src.float() - cand_tgt.float()).unsqueeze(-1) / S
        src_feat = node_features[cand_src]
        tgt_feat = node_features[cand_tgt]
        scorer_input = torch.cat([src_feat, tgt_feat, pos_diff], dim=-1)
        edge_logits = self.edge_scorer(scorer_input).squeeze(-1)
        cand_gates = gumbel_sigmoid(edge_logits, self.temperature, hard=hard)

        self.sparsity_loss = (cand_gates.mean() - self.target_density) ** 2

        # Combine base (gate=1) + candidates (learned)
        combined_src = torch.cat([base_src, cand_src])
        combined_tgt = torch.cat([base_tgt, cand_tgt])
        combined_gates = torch.cat([torch.ones(num_base, device=device), cand_gates])

        all_pos_diff = (combined_src.float() - combined_tgt.float()).unsqueeze(-1) / S
        all_src_feat = node_features[combined_src]
        all_tgt_feat = node_features[combined_tgt]
        edge_input = torch.cat([all_src_feat, all_tgt_feat, all_pos_diff], dim=-1)
        edge_features = self.to_edge(edge_input) * combined_gates.unsqueeze(-1)

        edge_index = torch.stack([combined_src, combined_tgt])

        return DeltaGraph(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
        ), combined_gates


# ----------------------------------------------------------------------
#  Self-Bootstrapped DELTA models
# ----------------------------------------------------------------------

class SelfBootstrappedDELTA(nn.Module):
    """DELTA bootstraps DELTA -- no transformer in the pipeline.

    Stage 1: Embedding -> trivial chain graph -> DELTALayers -> enriched features
    Stage 2: FeatureBasedConstructor -> dynamic topology
    Stage 3: DELTALayers on dynamic graph -> classification

    The transformer is completely removed. Sequential adjacency provides the
    trivial initial structure; DELTA's own relational machinery (dual attention +
    ReconciliationBridge) enriches the features before dynamic construction.
    """

    def __init__(self, vocab_size, d_model, d_node, d_edge, num_classes,
                 num_heads=4, bootstrap_layers=1, delta_layers=2,
                 constructor_cls=FeatureBasedConstructor,
                 target_density=0.3):
        super().__init__()
        # Embedding only -- NO transformer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.to_node = nn.Linear(d_model, d_node)
        self.to_edge_bootstrap = nn.Linear(2 * d_node, d_edge)

        # Stage 1: DELTA on fixed chain
        self.bootstrap_layers = nn.ModuleList([
            DELTALayer(d_node, d_edge, num_heads) for _ in range(bootstrap_layers)
        ])

        # Stage 2: Differentiable constructor on enriched features
        self.constructor = constructor_cls(
            d_node=d_node, d_edge=d_edge,
            target_density=target_density,
        )

        # Stage 3: DELTA on dynamic graph
        self.delta_layers = nn.ModuleList([
            DELTALayer(d_node, d_edge, num_heads) for _ in range(delta_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(d_node, d_node), nn.GELU(), nn.Linear(d_node, num_classes),
        )

    def _build_chain_graph(self, node_features):
        """Build trivial bidirectional sequential chain."""
        S = node_features.shape[0]
        device = node_features.device
        fwd_src = torch.arange(S - 1, device=device)
        fwd_tgt = torch.arange(1, S, device=device)
        edge_index = torch.stack([
            torch.cat([fwd_src, fwd_tgt]),
            torch.cat([fwd_tgt, fwd_src]),
        ])
        src_feats = node_features[edge_index[0]]
        tgt_feats = node_features[edge_index[1]]
        edge_features = self.to_edge_bootstrap(
            torch.cat([src_feats, tgt_feats], dim=-1)
        )
        return DeltaGraph(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
        )

    def forward(self, token_ids, hard=False):
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        batch_logits = []
        total_sparsity = 0.0

        for i in range(token_ids.shape[0]):
            # Stage 0: Embed (NO transformer)
            x = self.embedding(token_ids[i:i+1])
            x = self.pos_enc(x).squeeze(0)  # [S, d_model]
            node_features = self.to_node(x)  # [S, d_node]

            # Stage 1: DELTA on fixed chain -> enriched features
            chain_graph = self._build_chain_graph(node_features)
            for layer in self.bootstrap_layers:
                chain_graph = layer(
                    chain_graph, use_router=False,
                    use_partitioning=False, use_memory=False,
                )
            enriched = chain_graph.node_features

            # Stage 2: Constructor scores edges on enriched features
            dynamic_graph, gates = self.constructor(enriched, hard=hard)
            total_sparsity += self.constructor.sparsity_loss

            # Stage 3: Full DELTA on dynamic graph
            for layer in self.delta_layers:
                dynamic_graph = layer(
                    dynamic_graph, use_router=False,
                    use_partitioning=False, use_memory=False,
                )

            pooled = dynamic_graph.node_features.mean(dim=0, keepdim=True)
            batch_logits.append(self.classifier(pooled))

        self._sparsity_loss = total_sparsity / token_ids.shape[0]
        return torch.cat(batch_logits, dim=0)


# ----------------------------------------------------------------------
#  Experiment
# ----------------------------------------------------------------------

def run_experiment(num_samples=1000, epochs=100, num_seeds=3, device_str='cpu'):
    device = torch.device(device_str)

    print("=" * 72)
    print("Phase 46b: Self-Bootstrapped DELTA")
    print("=" * 72)
    print(f"Config: {num_samples} samples, {epochs} epochs, {num_seeds} seeds")
    print(f"Device: {device}")
    print()

    d_model = 64
    d_node = 64
    d_edge = 32
    num_heads = 4
    accum_steps = 32

    results = {}

    for seed in range(num_seeds):
        print(f"\n{'-' * 72}")
        print(f"Seed {seed + 1}/{num_seeds}")
        print(f"{'-' * 72}")

        data, labels, vocab_size, num_classes = generate_path_composition_task(
            num_samples=num_samples, seed=42 + seed,
        )

        n_train = int(0.7 * len(data))
        perm = torch.randperm(
            len(data), generator=torch.Generator().manual_seed(seed),
        )
        train_idx, test_idx = perm[:n_train], perm[n_train:]
        train_data, train_labels = data[train_idx], labels[train_idx]
        test_data, test_labels = data[test_idx], labels[test_idx]

        models = {
            "Transformer": (
                TransformerBaseline(vocab_size, d_model, num_classes).to(device),
                True, None,
            ),
            "FixedChain": (
                FixedChainDELTA(
                    vocab_size, d_model, d_node, d_edge, num_classes,
                ).to(device),
                False, None,
            ),
            "P46_Hybrid": (
                DifferentiableDELTA(
                    vocab_size, d_model, d_node, d_edge, num_classes,
                    constructor_cls=HybridDifferentiableConstructor,
                    target_density=0.2,
                ).to(device),
                False, lambda e, t: linear_anneal(e, t, 0.5, 5.0),
            ),
            "SelfBootstrap": (
                SelfBootstrappedDELTA(
                    vocab_size, d_model, d_node, d_edge, num_classes,
                    constructor_cls=FeatureBasedConstructor,
                    bootstrap_layers=1, delta_layers=2,
                    target_density=0.3,
                ).to(device),
                False, lambda e, t: linear_anneal(e, t, 0.5, 5.0),
            ),
            "SelfBootstrapHybrid": (
                SelfBootstrappedDELTA(
                    vocab_size, d_model, d_node, d_edge, num_classes,
                    constructor_cls=FeatureBasedHybridConstructor,
                    bootstrap_layers=1, delta_layers=2,
                    target_density=0.2,
                ).to(device),
                False, lambda e, t: linear_anneal(e, t, 0.5, 5.0),
            ),
        }

        # Print parameter counts once
        if seed == 0:
            print("\n  Parameter counts:")
            for name, (model, _, _) in models.items():
                n_params = sum(p.numel() for p in model.parameters())
                print(f"    {name:<22} {n_params:>8,d}")
            print()

        for name, (model, is_tf, temp_sched) in models.items():
            print(f"\n  Training {name}...")
            best_acc, final_acc, elapsed = train_and_evaluate(
                name, model, train_data, train_labels,
                test_data, test_labels,
                epochs=epochs, accum_steps=accum_steps,
                is_transformer=is_tf,
                temperature_schedule=temp_sched,
                sparsity_weight=0.1,
            )
            if name not in results:
                results[name] = []
            results[name].append({
                'best_acc': best_acc, 'final_acc': final_acc, 'elapsed': elapsed,
            })
            print(f"  {name}: best={best_acc:.3f}  final={final_acc:.3f}"
                  f"  time={elapsed:.1f}s")

    # -- Summary --
    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    print(f"{'Model':<22} {'Best Acc':>10} {'Std':>8} {'Final Acc':>10}"
          f" {'Time':>8}")
    print("-" * 62)

    model_order = [
        "Transformer", "FixedChain", "P46_Hybrid",
        "SelfBootstrap", "SelfBootstrapHybrid",
    ]
    for name in model_order:
        bests = [r['best_acc'] for r in results[name]]
        finals = [r['final_acc'] for r in results[name]]
        times = [r['elapsed'] for r in results[name]]
        mean_best = np.mean(bests)
        std_best = np.std(bests)
        mean_final = np.mean(finals)
        mean_time = np.mean(times)
        print(f"{name:<22} {mean_best:>9.3f}  +/-{std_best:.3f}"
              f"  {mean_final:>9.3f}  {mean_time:>7.1f}s")

    print()
    print("Phase 46 reference (3-seed, 1000 samples, 100 epochs):")
    print("  Transformer:    0.419 +/-0.027")
    print("  FixedChain:     0.461 +/-0.034")
    print("  Differentiable: 0.393 +/-0.017 (85% of FixedChain)")
    print("  Hybrid (TF):    0.452 +/-0.006 (98% of FixedChain)")
    print()

    fc_mean = np.mean([r['best_acc'] for r in results["FixedChain"]])
    sb_mean = np.mean([r['best_acc'] for r in results["SelfBootstrap"]])
    sbh_mean = np.mean([r['best_acc'] for r in results["SelfBootstrapHybrid"]])
    p46h_mean = np.mean([r['best_acc'] for r in results["P46_Hybrid"]])

    print("KEY QUESTIONS:")
    print()
    print("  1. Does DELTA bootstrap beat TF bootstrap for hybrid construction?")
    print(f"     SelfBootstrapHybrid: {sbh_mean:.3f}  vs  P46_Hybrid: {p46h_mean:.3f}")
    if sbh_mean > p46h_mean + 0.01:
        print(f"     --> YES, DELTA bootstrap is superior"
              f" (+{sbh_mean - p46h_mean:.3f})")
    elif sbh_mean >= p46h_mean - 0.01:
        print(f"     --> Comparable (within +/-0.01)")
    else:
        print(f"     --> No, TF bootstrap still helps"
              f" ({p46h_mean - sbh_mean:+.3f})")

    print()
    print("  2. Does DELTA enrichment close the pure-differentiable gap?")
    print(f"     Phase 46 Differentiable (TF):  0.393 (85% of FixedChain)")
    print(f"     SelfBootstrap (DELTA):         {sb_mean:.3f}"
          f" ({sb_mean/fc_mean*100:.0f}% of FixedChain)")
    if sb_mean >= fc_mean * 0.95:
        print("     --> YES! DELTA-enriched features make edge discovery"
              " reliable.")
        print("     --> The transformer bootstrap is FULLY replaceable.")
    elif sb_mean > 0.393 + 0.01:
        print(f"     --> DELTA enrichment helps (+{sb_mean - 0.393:.3f}"
              f" over TF-based) but gap remains.")
    else:
        print("     --> No improvement over TF-based differentiable.")

    print()
    print("  3. Does self-bootstrap match FixedChain?")
    print(f"     SelfBootstrapHybrid: {sbh_mean:.3f}  vs  FixedChain:"
          f" {fc_mean:.3f} ({sbh_mean/fc_mean*100:.0f}%)")
    print()

    if sbh_mean >= fc_mean * 0.95:
        print("CONCLUSION: Self-bootstrapped DELTA validates full"
              " transformer independence.")
        print("            DELTA constructs its own graph from trivial"
              " sequential input.")
        print("            The transformer scaffold comes down.")
    elif sb_mean >= 0.393 * 1.05:
        print("CONCLUSION: DELTA enrichment improves over raw TF embeddings"
              " for edge scoring.")
        print("            Self-bootstrap is a viable direction but needs"
              " further refinement.")
        print("            Consider: deeper bootstrap, curriculum schedule,"
              " or contrastive")
        print("            construction loss.")
    else:
        print("CONCLUSION: Self-bootstrap does not improve over TF-based"
              " construction.")
        print("            The relational enrichment from 1 DELTA layer on"
              " a chain may be")
        print("            insufficient. Consider: more bootstrap layers,"
              " residual connections,")
        print("            or the chain topology itself as the final answer.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Phase 46b: Self-Bootstrapped DELTA",
    )
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_experiment(
        num_samples=args.num_samples,
        epochs=args.epochs,
        num_seeds=args.seeds,
        device_str=args.device,
    )
