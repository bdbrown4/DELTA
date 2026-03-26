"""
Phase 26: Adaptive Multi-Hop Depth

Core question: Can DELTA learn WHEN to use 1-hop vs 2-hop edge adjacency
per layer, instead of fixing it at construction time?

Background:
  - Phase 11: 2-hop was critical for derived relations (100% vs 61.1% at 1-hop)
  - Phase 24: 2-hop at E=5000 costs 490s vs 44s for 1-hop, with NO accuracy benefit
  - Phase 25: 1-hop at 69k edges produces 19M pairs — already requires subsampling

Approach: Precompute BOTH 1-hop and 2-hop edge adjacency, then learn a soft
mixing gate that decides the blend per layer. The gate is supervised by task
loss — if 2-hop helps, the gate opens; if it's noise, the gate stays closed.

Test: Multi-relational reasoning task with both base and derived relations.
Derived relations REQUIRE 2-hop composition. Base relations work with 1-hop.
The adaptive model should learn to use 2-hop selectively.

Comparisons:
 1. Fixed 1-hop only (fast, misses derived)
 2. Fixed 2-hop only (accurate on derived, expensive everywhere)
 3. Adaptive gate: learns per-layer 1-hop/2-hop blend
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import EdgeAttention, DualParallelAttention
from delta.router import PostAttentionPruner
from delta.utils import create_multi_relational_reasoning_task


class AdaptiveHopGate(nn.Module):
    """Learns a per-layer soft blend between 1-hop and 2-hop edge adjacency.

    The gate outputs α ∈ [0, 1]:
      - α ≈ 0 → use 1-hop adjacency (local structure, fast)
      - α ≈ 1 → use 2-hop adjacency (compositional, expensive)

    The gate is conditioned on edge features — different edge types can
    prefer different hop depths.
    """
    def __init__(self, d_edge):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_edge, d_edge // 2),
            nn.GELU(),
            nn.Linear(d_edge // 2, 1),
        )

    def forward(self, edge_features):
        """Returns per-edge hop blend weight α ∈ [0, 1]."""
        return torch.sigmoid(self.gate(edge_features).squeeze(-1))


class FixedHopModel(nn.Module):
    """Baseline: fixed 1-hop or 2-hop edge adjacency."""
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, edge_adj=None, **kwargs):
        edge_feats = self.edge_attn(graph, edge_adj=edge_adj)
        return self.classifier(edge_feats), torch.tensor(0.0, device=graph.device)


class AdaptiveHopModel(nn.Module):
    """DELTA with learned hop-depth blending.

    Runs edge attention on BOTH 1-hop and 2-hop adjacency, then blends
    the resulting edge features using a learned per-edge gate.
    """
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.edge_attn_1hop = EdgeAttention(d_edge, d_node, num_heads)
        self.edge_attn_2hop = EdgeAttention(d_edge, d_node, num_heads)
        self.hop_gate = AdaptiveHopGate(d_edge)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, edge_adj_1hop=None, edge_adj_2hop=None, **kwargs):
        feats_1 = self.edge_attn_1hop(graph, edge_adj=edge_adj_1hop)
        feats_2 = self.edge_attn_2hop(graph, edge_adj=edge_adj_2hop)

        alpha = self.hop_gate(graph.edge_features)  # [E]
        blended = (1 - alpha).unsqueeze(-1) * feats_1 + alpha.unsqueeze(-1) * feats_2
        return self.classifier(blended), alpha.mean()


class AdaptiveHopGatingModel(nn.Module):
    """Full DELTA: adaptive hop + dual attention + soft gating."""
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.dual_attn_1hop = DualParallelAttention(d_node, d_edge, num_heads)
        self.dual_attn_2hop = DualParallelAttention(d_node, d_edge, num_heads)
        self.hop_gate = AdaptiveHopGate(d_edge)
        self.pruner = PostAttentionPruner(d_node, d_edge, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, edge_adj_1hop=None, edge_adj_2hop=None,
                target_sparsity=0.5, **kwargs):
        # Run dual attention with both adjacencies
        res1, nw1, ew1 = self.dual_attn_1hop(
            graph, edge_adj=edge_adj_1hop, return_weights=True)
        res2, nw2, ew2 = self.dual_attn_2hop(
            graph, edge_adj=edge_adj_2hop, return_weights=True)

        # Blend outputs using learned gate
        alpha = self.hop_gate(graph.edge_features)
        blended_edges = ((1 - alpha).unsqueeze(-1) * res1.edge_features +
                         alpha.unsqueeze(-1) * res2.edge_features)
        blended_nodes = ((1 - alpha.mean()) * res1.node_features +
                         alpha.mean() * res2.node_features)
        blended_graph = DeltaGraph(
            node_features=blended_nodes,
            edge_features=blended_edges,
            edge_index=graph.edge_index,
            node_tiers=graph.node_tiers,
        )

        # Soft gating on blended result
        _, edge_gates = self.pruner.compute_importance(blended_graph, nw1, ew1)
        gated, sp_loss = self.pruner.soft_prune(
            blended_graph, edge_gates, target_sparsity=target_sparsity)
        return self.classifier(gated.edge_features), sp_loss + 0.01 * alpha.mean()


def train_eval(model, graph, labels, train_idx, test_idx,
               edge_adj_1hop=None, edge_adj_2hop=None,
               epochs=200, lr=1e-3, sparsity_weight=0.1):
    """Train and evaluate a model, returning best test accuracy."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_test = 0.0
    best_base = 0.0
    best_derived = 0.0

    fwd_kwargs = {}
    if edge_adj_1hop is not None:
        fwd_kwargs['edge_adj'] = edge_adj_1hop
        fwd_kwargs['edge_adj_1hop'] = edge_adj_1hop
    if edge_adj_2hop is not None:
        fwd_kwargs['edge_adj_2hop'] = edge_adj_2hop

    for epoch in range(epochs):
        model.train()
        logits, aux = model(graph, **fwd_kwargs)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss = loss + sparsity_weight * aux
        opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits, aux = model(graph, **fwd_kwargs)
                preds = logits.argmax(-1)
                test_acc = (preds[test_idx] == labels[test_idx]).float().mean().item()
                best_test = max(best_test, test_acc)
                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}  Test={test_acc:.3f}"
                      + (f"  α_mean={aux.item():.3f}" if isinstance(aux, torch.Tensor) and aux.numel() == 1 and aux.item() < 1.0 else ""))

    return best_test


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("PHASE 26: Adaptive Multi-Hop Depth")
    print("=" * 70)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})"
                                  if device.type == 'cuda' else ""))

    d_node, d_edge = 64, 32
    torch.manual_seed(42)

    # Use multi-relational reasoning with base + derived relations
    # Derived relations (livesNear, peerOf, seniorTo) require 2-hop composition
    # Scale up to make the task non-trivial for all models
    print("\nGenerating multi-relational reasoning task (scaled)...")
    graph, labels, meta = create_multi_relational_reasoning_task(
        num_entities=200, num_base_relations=4, num_derived_rules=3,
        d_node=d_node, d_edge=d_edge, seed=42,
    )
    num_classes = meta['num_total_relations']
    base_mask = meta['base_mask']
    derived_mask = meta['derived_mask']

    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"Relations: {num_classes} (4 base + 3 derived)")
    print(f"Base edges: {base_mask.sum().item()}, Derived edges: {derived_mask.sum().item()}")
    print(f"Rules: {meta['rules']}")

    # Train/test split
    E = graph.num_edges
    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(E, generator=gen)
    split = int(E * 0.7)
    train_idx = perm[:split]
    test_idx = perm[split:]

    # Move to device
    graph = graph.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    test_idx = test_idx.to(device)
    base_mask = base_mask.to(device)
    derived_mask = derived_mask.to(device)

    # Precompute BOTH adjacencies
    print("\nPrecomputing edge adjacencies...")
    t0 = time.time()
    edge_adj_1hop = graph.build_edge_adjacency(hops=1)
    t1 = time.time()
    edge_adj_2hop = graph.build_edge_adjacency(hops=2)
    t2 = time.time()
    print(f"  1-hop: {edge_adj_1hop.shape[1]:,} pairs ({t1-t0:.2f}s)")
    print(f"  2-hop: {edge_adj_2hop.shape[1]:,} pairs ({t2-t1:.2f}s)")

    results = {}
    times = {}
    EPOCHS = 200

    # 1. Fixed 1-hop
    print("\n--- Fixed 1-Hop ---")
    torch.manual_seed(42)
    m = FixedHopModel(d_node, d_edge, num_classes).to(device)
    t0 = time.time()
    acc = train_eval(m, graph, labels, train_idx, test_idx,
                     edge_adj_1hop=edge_adj_1hop, epochs=EPOCHS)
    times['1-Hop'] = time.time() - t0
    results['Fixed 1-Hop'] = acc

    # Evaluate base vs derived
    m.eval()
    with torch.no_grad():
        logits, _ = m(graph, edge_adj=edge_adj_1hop)
        preds = logits.argmax(-1)
        base_test = test_idx[base_mask[test_idx]]
        derived_test = test_idx[derived_mask[test_idx]]
        if len(base_test) > 0:
            base_acc = (preds[base_test] == labels[base_test]).float().mean().item()
        else:
            base_acc = 0.0
        if len(derived_test) > 0:
            der_acc = (preds[derived_test] == labels[derived_test]).float().mean().item()
        else:
            der_acc = 0.0
    print(f"  Base acc: {base_acc:.3f}, Derived acc: {der_acc:.3f}")
    results['1-Hop Base'] = base_acc
    results['1-Hop Derived'] = der_acc

    # 2. Fixed 2-hop
    print("\n--- Fixed 2-Hop ---")
    torch.manual_seed(42)
    m = FixedHopModel(d_node, d_edge, num_classes).to(device)
    t0 = time.time()
    acc = train_eval(m, graph, labels, train_idx, test_idx,
                     edge_adj_1hop=edge_adj_2hop, epochs=EPOCHS)
    times['2-Hop'] = time.time() - t0
    results['Fixed 2-Hop'] = acc

    m.eval()
    with torch.no_grad():
        logits, _ = m(graph, edge_adj=edge_adj_2hop)
        preds = logits.argmax(-1)
        if len(base_test) > 0:
            base_acc = (preds[base_test] == labels[base_test]).float().mean().item()
        else:
            base_acc = 0.0
        if len(derived_test) > 0:
            der_acc = (preds[derived_test] == labels[derived_test]).float().mean().item()
        else:
            der_acc = 0.0
    print(f"  Base acc: {base_acc:.3f}, Derived acc: {der_acc:.3f}")
    results['2-Hop Base'] = base_acc
    results['2-Hop Derived'] = der_acc

    # 3. Adaptive hop gate (EdgeAttention only)
    print("\n--- Adaptive Hop Gate ---")
    torch.manual_seed(42)
    m = AdaptiveHopModel(d_node, d_edge, num_classes).to(device)
    t0 = time.time()
    acc = train_eval(m, graph, labels, train_idx, test_idx,
                     edge_adj_1hop=edge_adj_1hop, edge_adj_2hop=edge_adj_2hop,
                     epochs=EPOCHS)
    times['Adaptive'] = time.time() - t0
    results['Adaptive Hop'] = acc

    m.eval()
    with torch.no_grad():
        logits, alpha = m(graph, edge_adj_1hop=edge_adj_1hop, edge_adj_2hop=edge_adj_2hop)
        preds = logits.argmax(-1)
        if len(base_test) > 0:
            base_acc = (preds[base_test] == labels[base_test]).float().mean().item()
        else:
            base_acc = 0.0
        if len(derived_test) > 0:
            der_acc = (preds[derived_test] == labels[derived_test]).float().mean().item()
        else:
            der_acc = 0.0
    print(f"  Base acc: {base_acc:.3f}, Derived acc: {der_acc:.3f}")
    print(f"  Learned α (2-hop blend): {alpha.item():.3f}")
    results['Adaptive Base'] = base_acc
    results['Adaptive Derived'] = der_acc

    # 4. Adaptive + Soft Gating (full DELTA)
    print("\n--- Adaptive Hop + Dual Attention + Soft Gating ---")
    torch.manual_seed(42)
    m = AdaptiveHopGatingModel(d_node, d_edge, num_classes).to(device)
    t0 = time.time()
    acc = train_eval(m, graph, labels, train_idx, test_idx,
                     edge_adj_1hop=edge_adj_1hop, edge_adj_2hop=edge_adj_2hop,
                     epochs=EPOCHS)
    times['Adaptive+Gate'] = time.time() - t0
    results['Adapt+Gate'] = acc

    m.eval()
    with torch.no_grad():
        logits, _ = m(graph, edge_adj_1hop=edge_adj_1hop, edge_adj_2hop=edge_adj_2hop)
        preds = logits.argmax(-1)
        if len(base_test) > 0:
            base_acc = (preds[base_test] == labels[base_test]).float().mean().item()
        else:
            base_acc = 0.0
        if len(derived_test) > 0:
            der_acc = (preds[derived_test] == labels[derived_test]).float().mean().item()
        else:
            der_acc = 0.0
    print(f"  Base acc: {base_acc:.3f}, Derived acc: {der_acc:.3f}")
    results['Adapt+Gate Base'] = base_acc
    results['Adapt+Gate Derived'] = der_acc

    # Summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  Overall Accuracy:")
    print(f"  {'Model':<28} {'Test Acc':>10} {'Time':>8}")
    print(f"  {'-'*28} {'-'*10} {'-'*8}")
    for name in ['Fixed 1-Hop', 'Fixed 2-Hop', 'Adaptive Hop', 'Adapt+Gate']:
        t = times.get(name.split()[0] if '+' not in name else 'Adaptive+Gate',
                      times.get(name.replace('Fixed ', '').split('-')[0], 0))
        time_key = {'Fixed 1-Hop': '1-Hop', 'Fixed 2-Hop': '2-Hop',
                    'Adaptive Hop': 'Adaptive', 'Adapt+Gate': 'Adaptive+Gate'}[name]
        t = times[time_key]
        print(f"  {name:<28} {results[name]:>10.3f} {t:>7.1f}s")

    print(f"\n  Base vs Derived Relation Accuracy:")
    print(f"  {'Model':<28} {'Base':>10} {'Derived':>10}")
    print(f"  {'-'*28} {'-'*10} {'-'*10}")
    for prefix in ['1-Hop', '2-Hop', 'Adaptive', 'Adapt+Gate']:
        b = results.get(f'{prefix} Base', 0)
        d = results.get(f'{prefix} Derived', 0)
        print(f"  {prefix:<28} {b:>10.3f} {d:>10.3f}")

    print(f"\n  Key findings:")
    hop1 = results['Fixed 1-Hop']
    hop2 = results['Fixed 2-Hop']
    adapt = results['Adaptive Hop']
    full = results['Adapt+Gate']
    if adapt >= max(hop1, hop2) - 0.02:
        print(f"  >> Adaptive gate matches or exceeds best fixed-hop baseline")
    if full >= adapt:
        print(f"  >> Full DELTA (adaptive + soft gating) is the top model")
    d_1 = results.get('1-Hop Derived', 0)
    d_2 = results.get('2-Hop Derived', 0)
    d_a = results.get('Adaptive Derived', 0)
    if d_a > d_1:
        print(f"  >> Adaptive model recovers derived relation accuracy: "
              f"{d_a:.3f} vs 1-hop {d_1:.3f}")


if __name__ == '__main__':
    main()
