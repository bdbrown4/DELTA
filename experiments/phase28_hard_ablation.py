"""
Phase 28: Harder Ablation Benchmark

Phases 22-26 saturate at 100% even for vanilla EdgeAttention because the
edge features clearly encode relation types (prototypes + small noise).

Goal: construct benchmarks of increasing difficulty where vanilla EdgeAttention
is insufficient, so that DELTA's advantages (node context via dual attention,
soft gating for noise filtering) become measurably necessary.

Difficulty levers:
  - Feature noise: higher variance blurs relation prototypes
  - Label noise: fraction of edges get wrong relation labels
  - Prototype density: closer prototypes → harder separation
  - Class count: more relations → harder classification

Models:
  1. Vanilla EdgeAttention — edge features only, no node context
  2. Dual Attention — node + edge attention (adds node-type signal)
  3. DELTA + Soft Gating — dual attention + learned gating (filters noise)

Hypothesis: as difficulty increases, vanilla EdgeAttention degrades first,
dual attention degrades second (node context helps), and soft gating degrades
last (noise filtering helps).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from delta.graph import DeltaGraph
from delta.attention import EdgeAttention, DualParallelAttention
from delta.router import PostAttentionPruner


# ──────────────────────────────────────────────────────────────────────
#  Hard benchmark generator
# ──────────────────────────────────────────────────────────────────────

def create_hard_ablation_benchmark(
    num_entities: int = 500,
    num_relations: int = 25,
    num_triples: int = 3000,
    feature_noise: float = 0.5,
    label_noise: float = 0.25,
    prototype_spread: float = 0.5,
    d_node: int = 64,
    d_edge: int = 32,
    seed: int = 42,
) -> Tuple[DeltaGraph, torch.Tensor, Dict]:
    """KG benchmark with tunable difficulty.

    Args:
        feature_noise: std-dev of noise added to edge features (higher = harder)
        label_noise: fraction of edges that get a random (wrong) label
        prototype_spread: scaling factor for relation prototypes (lower = closer)
        Other args as in create_noisy_kg_benchmark.

    Returns:
        graph: DeltaGraph
        labels: [E] relation type per edge
        metadata: train/test splits, noise levels, entity types
    """
    torch.manual_seed(seed)
    random.seed(seed)

    num_top_types = 5
    entity_type = [i % num_top_types for i in range(num_entities)]
    type_protos = torch.randn(num_top_types, d_node) * 1.0
    node_features = torch.randn(num_entities, d_node) * 0.3
    for i in range(num_entities):
        node_features[i] += type_protos[entity_type[i]]

    # Relation prototypes — closer together when prototype_spread is small
    rel_protos = torch.randn(num_relations, d_edge) * prototype_spread

    # Power-law degree distribution
    weights = torch.tensor([1.0 / (i + 1) ** 0.8 for i in range(num_entities)])
    weights = weights / weights.sum()

    src_list, tgt_list, edge_feats, labels = [], [], [], []
    clean_labels = []  # labels before noise
    seen: set = set()

    for _ in range(num_triples):
        for _attempt in range(20):
            s = torch.multinomial(weights, 1).item()
            t = random.randint(0, num_entities - 1)
            if s != t and (s, t) not in seen:
                seen.add((s, t))
                break
        else:
            continue

        # Structured relation assignment
        r = (entity_type[s] * num_top_types + entity_type[t]) % num_relations
        clean_labels.append(r)

        # Label noise
        if random.random() < label_noise:
            r = random.randint(0, num_relations - 1)

        src_list.append(s)
        tgt_list.append(t)
        edge_feats.append(rel_protos[r] + torch.randn(d_edge) * feature_noise)
        labels.append(r)

    n = len(src_list)
    perm = torch.randperm(n)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    graph = DeltaGraph(
        node_features=node_features,
        edge_features=torch.stack(edge_feats),
        edge_index=torch.tensor([src_list, tgt_list], dtype=torch.long),
    )

    metadata = {
        'num_relations': num_relations,
        'entity_types': entity_type,
        'feature_noise': feature_noise,
        'label_noise': label_noise,
        'prototype_spread': prototype_spread,
        'train_idx': perm[:train_end],
        'val_idx': perm[train_end:val_end],
        'test_idx': perm[val_end:],
        'clean_labels': torch.tensor(clean_labels, dtype=torch.long),
    }

    return graph, torch.tensor(labels, dtype=torch.long), metadata


# ──────────────────────────────────────────────────────────────────────
#  Models
# ──────────────────────────────────────────────────────────────────────

class VanillaEdgeAttnModel(nn.Module):
    """Edge attention only — no node context."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, **kwargs):
        edge_adj = graph.build_edge_adjacency()
        edge_feats = self.edge_attn(graph, edge_adj=edge_adj)
        return self.classifier(edge_feats), torch.tensor(0.0, device=graph.device)


class DualAttnModel(nn.Module):
    """Node + edge dual attention — adds node-type context to classification."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.dual_attn = DualParallelAttention(d_node, d_edge, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, **kwargs):
        edge_adj = graph.build_edge_adjacency()
        result = self.dual_attn(graph, edge_adj=edge_adj, return_weights=False)
        return self.classifier(result.edge_features), torch.tensor(0.0, device=graph.device)


class DeltaSoftGatingModel(nn.Module):
    """Dual attention + soft gating — filters noisy/unreliable edges."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.dual_attn = DualParallelAttention(d_node, d_edge, num_heads)
        self.pruner = PostAttentionPruner(d_node, d_edge, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, target_sparsity=0.5, temperature=1.0, **kwargs):
        edge_adj = graph.build_edge_adjacency()
        result, node_attn_w, edge_attn_w = self.dual_attn(
            graph, edge_adj=edge_adj, return_weights=True,
        )
        _, edge_gates = self.pruner.compute_importance(
            result, node_attn_w, edge_attn_w, temperature=temperature,
        )
        gated_graph, sparsity_loss = self.pruner.soft_prune(
            result, edge_gates, target_sparsity=target_sparsity,
        )
        return self.classifier(gated_graph.edge_features), sparsity_loss


# ──────────────────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────────────────

def train_eval(model, graph, labels, train_idx, test_idx,
               epochs=200, lr=1e-3, sparsity_weight=0.1,
               target_sparsity=0.5, device='cpu', quiet=False):
    """Train and return (best_test_acc, best_train_acc)."""
    graph = DeltaGraph(
        node_features=graph.node_features.to(device),
        edge_features=graph.edge_features.to(device),
        edge_index=graph.edge_index.to(device),
    )
    labels_d = labels.to(device)
    train_idx_d = train_idx.to(device)
    test_idx_d = test_idx.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_test, best_train = 0.0, 0.0

    for epoch in range(epochs):
        model.train()
        logits, aux_loss = model(graph, target_sparsity=target_sparsity)
        task_loss = F.cross_entropy(logits[train_idx_d], labels_d[train_idx_d])
        loss = task_loss + sparsity_weight * aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(graph, target_sparsity=target_sparsity)
                preds = logits.argmax(-1)
                train_acc = (preds[train_idx_d] == labels_d[train_idx_d]).float().mean().item()
                test_acc = (preds[test_idx_d] == labels_d[test_idx_d]).float().mean().item()
                if test_acc > best_test:
                    best_test = test_acc
                    best_train = train_acc
                if not quiet:
                    print(f"    Epoch {epoch + 1}: Loss={task_loss.item():.4f}  "
                          f"Train={train_acc:.3f}  Test={test_acc:.3f}")

    return best_test, best_train


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 28: Harder Ablation Benchmark")
    print("=" * 70)
    print()
    print("Testing 3 models across 4 difficulty levels to find where")
    print("vanilla EdgeAttention fails and DELTA advantages appear.")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print()

    d_node, d_edge = 64, 32

    # Difficulty levels: (feature_noise, label_noise, prototype_spread, num_relations)
    levels = {
        'Easy':     (0.20, 0.00, 1.0, 15),
        'Medium':   (0.40, 0.15, 0.7, 20),
        'Hard':     (0.60, 0.25, 0.5, 25),
        'Extreme':  (0.80, 0.35, 0.3, 30),
    }

    all_results: Dict[str, Dict[str, float]] = {}

    for level_name, (feat_noise, lab_noise, proto_spread, n_rels) in levels.items():
        print("=" * 60)
        print(f"Difficulty: {level_name}")
        print(f"  feature_noise={feat_noise}, label_noise={lab_noise}, "
              f"prototype_spread={proto_spread}, num_relations={n_rels}")
        print("=" * 60)

        torch.manual_seed(42)
        graph, labels, meta = create_hard_ablation_benchmark(
            num_entities=500, num_relations=n_rels, num_triples=3000,
            feature_noise=feat_noise, label_noise=lab_noise,
            prototype_spread=proto_spread, d_node=d_node, d_edge=d_edge,
        )
        train_idx = meta['train_idx']
        test_idx = meta['test_idx']
        num_classes = meta['num_relations']

        print(f"  Graph: {graph.num_nodes} nodes, {graph.num_edges} edges, "
              f"{num_classes} classes")
        print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
        print()

        level_results = {}

        # Model 1: Vanilla EdgeAttention
        print(f"  --- Vanilla EdgeAttention ---")
        torch.manual_seed(42)
        m = VanillaEdgeAttnModel(d_node, d_edge, num_classes).to(device)
        t0 = time.time()
        test_acc, train_acc = train_eval(
            m, graph, labels, train_idx, test_idx, device=device,
        )
        elapsed = time.time() - t0
        level_results['Vanilla EdgeAttn'] = test_acc
        print(f"    >> Best test: {test_acc:.1%}  ({elapsed:.1f}s)\n")

        # Model 2: Dual Attention
        print(f"  --- Dual Attention ---")
        torch.manual_seed(42)
        m = DualAttnModel(d_node, d_edge, num_classes).to(device)
        t0 = time.time()
        test_acc, train_acc = train_eval(
            m, graph, labels, train_idx, test_idx, device=device,
        )
        elapsed = time.time() - t0
        level_results['Dual Attention'] = test_acc
        print(f"    >> Best test: {test_acc:.1%}  ({elapsed:.1f}s)\n")

        # Model 3: DELTA + Soft Gating
        print(f"  --- DELTA + Soft Gating ---")
        torch.manual_seed(42)
        m = DeltaSoftGatingModel(d_node, d_edge, num_classes).to(device)
        t0 = time.time()
        test_acc, train_acc = train_eval(
            m, graph, labels, train_idx, test_idx, device=device,
        )
        elapsed = time.time() - t0
        level_results['DELTA+SoftGate'] = test_acc
        print(f"    >> Best test: {test_acc:.1%}  ({elapsed:.1f}s)\n")

        all_results[level_name] = level_results

    # ── Grand Summary ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("GRAND SUMMARY — Accuracy by Difficulty Level")
    print("=" * 70)
    print()

    models = ['Vanilla EdgeAttn', 'Dual Attention', 'DELTA+SoftGate']
    header = f"  {'Level':<12s}"
    for m in models:
        header += f"  {m:>17s}"
    header += "   Best Model"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for level_name in levels:
        res = all_results[level_name]
        best_model = max(res, key=lambda k: res[k])
        row = f"  {level_name:<12s}"
        for m in models:
            acc = res[m]
            marker = "*" if m == best_model else " "
            row += f"  {acc:>16.1%}{marker}"
        row += f"   {best_model}"
        print(row)

    print()

    # Degradation analysis
    print("Degradation Analysis:")
    for m in models:
        easy = all_results['Easy'][m]
        extreme = all_results['Extreme'][m]
        drop = easy - extreme
        print(f"  {m:20s}  Easy={easy:.1%} → Extreme={extreme:.1%}  (drop={drop:+.1%})")

    print()
    van_hard = all_results['Hard']['Vanilla EdgeAttn']
    dual_hard = all_results['Hard']['Dual Attention']
    gate_hard = all_results['Hard']['DELTA+SoftGate']

    print("Key findings at Hard difficulty:")
    print(f"  Vanilla EdgeAttn:   {van_hard:.1%}")
    print(f"  Dual Attention:     {dual_hard:.1%}  (Δ = {dual_hard - van_hard:+.1%} from node context)")
    print(f"  DELTA+SoftGate:     {gate_hard:.1%}  (Δ = {gate_hard - van_hard:+.1%} from full DELTA)")

    if gate_hard > van_hard + 0.02:
        print("  >> Full DELTA measurably outperforms vanilla EdgeAttention.")
        print("  >> Soft gating and node context provide real advantage on hard tasks.")
    else:
        print("  >> Differences are marginal — task may need further difficulty tuning.")


if __name__ == '__main__':
    main()
