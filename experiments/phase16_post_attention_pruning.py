"""
Phase 16: Post-Attention Soft Gating with Curriculum

Addresses the 29% accuracy gap from the original Phase 16. Root causes:
1. Hard top-k pruning after attention is non-differentiable — pruner can't learn
2. Zeroing features after attention creates inconsistent representations
3. Single-scalar importance signal is too thin for the gating network

Solution: Soft differentiable gating + curriculum annealing
- Continuous sigmoid gates instead of hard top-k (full gradient flow)
- Sparsity regularization instead of hard cutoffs (learned compression)
- Temperature curriculum: soft → sharp gates over training (dense → sparse)
- Rich per-head attention features feed the gating network

Compares:
1. Full attention (upper bound, no pruning)
2. Old pre-attention router at 50% (hard top-k baseline)
3. Old post-attention hard pruning at 50% (the broken approach)
4. New soft gating at 50% target sparsity (differentiable)
5. Soft gating + curriculum: start dense, anneal to 50% (best of both)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import EdgeAttention, DualParallelAttention
from delta.router import PostAttentionPruner, ImportanceRouter
from delta.utils import create_synthetic_kg_benchmark


class OldRouterModel(nn.Module):
    """Phase 15-style model: pre-attention scoring, then masked attention."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.router = ImportanceRouter(d_node, d_edge)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, k_ratio=0.5, **kwargs):
        node_scores, edge_scores = self.router(graph)
        _, edge_mask = self.router.apply_top_k(
            graph, node_scores, edge_scores,
            node_k_ratio=k_ratio, edge_k_ratio=k_ratio,
        )
        ef = graph.edge_features * edge_mask.float().unsqueeze(-1)
        g = DeltaGraph(node_features=graph.node_features, edge_features=ef,
                       edge_index=graph.edge_index, node_tiers=graph.node_tiers)
        edge_adj = g.build_edge_adjacency()
        edge_feats = self.edge_attn(g, edge_adj=edge_adj)
        return self.classifier(edge_feats), torch.tensor(0.0)


class HardPostAttnModel(nn.Module):
    """The original (broken) approach: hard top-k after attention."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.dual_attn = DualParallelAttention(d_node, d_edge, num_heads)
        self.pruner = PostAttentionPruner(d_node, d_edge, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, k_ratio=0.5, **kwargs):
        edge_adj = graph.build_edge_adjacency()
        result, node_attn_w, edge_attn_w = self.dual_attn(
            graph, edge_adj=edge_adj, return_weights=True
        )
        node_scores, edge_scores = self.pruner.compute_importance(
            result, node_attn_w, edge_attn_w
        )
        _, edge_mask = self.pruner.prune(
            result, node_scores, edge_scores,
            node_k_ratio=k_ratio, edge_k_ratio=k_ratio,
        )
        pruned = result.edge_features * edge_mask.float().unsqueeze(-1)
        return self.classifier(pruned), torch.tensor(0.0)


class SoftGatingModel(nn.Module):
    """New approach: soft differentiable gating after attention."""

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
            graph, edge_adj=edge_adj, return_weights=True
        )
        _, edge_gates = self.pruner.compute_importance(
            result, node_attn_w, edge_attn_w, temperature=temperature,
        )
        gated_graph, sparsity_loss = self.pruner.soft_prune(
            result, edge_gates, target_sparsity=target_sparsity,
        )
        return self.classifier(gated_graph.edge_features), sparsity_loss


class FullAttentionModel(nn.Module):
    """Baseline: full attention, no pruning."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, **kwargs):
        edge_adj = graph.build_edge_adjacency()
        edge_feats = self.edge_attn(graph, edge_adj=edge_adj)
        return self.classifier(edge_feats), torch.tensor(0.0)


def train_eval(model, graph, labels, train_idx, test_idx,
               epochs=300, lr=1e-3, sparsity_weight=0.1,
               curriculum=False, target_sparsity=0.5,
               temp_start=0.5, temp_end=5.0):
    """Train with optional curriculum annealing."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        progress = epoch / max(1, epochs - 1)

        # Curriculum: anneal temperature (soft → sharp) and sparsity (0 → target)
        if curriculum:
            temperature = temp_start + (temp_end - temp_start) * progress
            current_sparsity = target_sparsity * min(1.0, progress * 2)  # reach target at 50% training
            fwd_kwargs = dict(target_sparsity=current_sparsity, temperature=temperature)
        else:
            fwd_kwargs = dict(target_sparsity=target_sparsity, temperature=1.0)

        logits, aux_loss = model(graph, **fwd_kwargs)
        task_loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss = task_loss + sparsity_weight * aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                eval_kwargs = dict(target_sparsity=target_sparsity, temperature=temp_end if curriculum else 1.0)
                logits, _ = model(graph, **eval_kwargs)
                preds = logits.argmax(-1)
                acc = (preds[test_idx] == labels[test_idx]).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch + 1

                extra = ""
                if curriculum:
                    extra = f"  τ={temperature:.2f} sparsity={current_sparsity:.2f}"
                print(f"  Epoch {epoch+1}: Loss={task_loss.item():.4f}  Acc={acc:.3f}{extra}")

    return best_acc, best_epoch


def main():
    print("=" * 70)
    print("PHASE 16: Post-Attention Soft Gating with Curriculum")
    print("=" * 70)
    print()
    print("Root cause of original 29% gap:")
    print("  1. Hard top-k after attention = non-differentiable (pruner can't learn)")
    print("  2. Zeroing post-attention features = inconsistent representations")
    print("  3. Single scalar importance = too weak a signal")
    print()
    print("Fix: Soft sigmoid gates + sparsity regularization + temperature curriculum")
    print()

    d_node, d_edge = 64, 32
    torch.manual_seed(42)

    graph, labels, metadata = create_synthetic_kg_benchmark(
        num_entities=100, num_relations=10, num_triples=500,
        d_node=d_node, d_edge=d_edge,
    )
    num_classes = metadata['num_relations']
    train_idx = metadata['train_idx']
    test_idx = metadata['test_idx']

    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"Relations: {num_classes}, Train: {len(train_idx)}, Test: {len(test_idx)}")
    print()

    results = {}

    # 1. Full attention baseline
    print("--- Full Attention (no pruning, upper bound) ---")
    torch.manual_seed(42)
    m = FullAttentionModel(d_node, d_edge, num_classes)
    acc, ep = train_eval(m, graph, labels, train_idx, test_idx, epochs=300)
    results['Full (no prune)'] = acc

    # 2. Old pre-attention router
    print("\n--- Old Pre-Attention Router @ 50% (hard top-k) ---")
    torch.manual_seed(42)
    m = OldRouterModel(d_node, d_edge, num_classes)
    acc, ep = train_eval(m, graph, labels, train_idx, test_idx, epochs=300)
    results['Old Router 50%'] = acc

    # 3. Hard post-attention pruning (the broken original)
    print("\n--- Hard Post-Attention Pruning @ 50% (original broken approach) ---")
    torch.manual_seed(42)
    m = HardPostAttnModel(d_node, d_edge, num_classes)
    acc, ep = train_eval(m, graph, labels, train_idx, test_idx, epochs=300)
    results['Hard PostAttn 50%'] = acc

    # 4. Soft gating (no curriculum)
    print("\n--- Soft Gating @ 50% target sparsity (differentiable) ---")
    torch.manual_seed(42)
    m = SoftGatingModel(d_node, d_edge, num_classes)
    acc, ep = train_eval(m, graph, labels, train_idx, test_idx, epochs=300,
                         target_sparsity=0.5, sparsity_weight=0.1)
    results['Soft Gate 50%'] = acc

    # 5. Soft gating + curriculum
    print("\n--- Soft Gating + Curriculum (dense→50% sparsity, τ: 0.5→5.0) ---")
    torch.manual_seed(42)
    m = SoftGatingModel(d_node, d_edge, num_classes)
    acc, ep = train_eval(m, graph, labels, train_idx, test_idx, epochs=300,
                         curriculum=True, target_sparsity=0.5, sparsity_weight=0.1,
                         temp_start=0.5, temp_end=5.0)
    results['Soft+Curriculum 50%'] = acc

    # Summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<30s} {'Test Acc':>10s}")
    print(f"  {'-'*30} {'-'*10}")
    for name, acc in results.items():
        bar = '#' * int(acc * 40)
        print(f"  {name:<30s} {acc:>10.3f}  {bar}")

    full = results['Full (no prune)']
    old = results['Old Router 50%']
    hard = results['Hard PostAttn 50%']
    soft = results['Soft Gate 50%']
    curr = results['Soft+Curriculum 50%']

    print(f"\n  Hard post-attn vs Old router:   {hard - old:+.3f}")
    print(f"  Soft gating vs Old router:      {soft - old:+.3f}")
    print(f"  Soft+Curriculum vs Old router:  {curr - old:+.3f}")
    print(f"  Soft+Curriculum vs Full:        {curr - full:+.3f}")

    if curr > old:
        print("\n  >> Soft gating + curriculum BEATS pre-attention routing!")
    if curr > hard:
        print(f"  >> Curriculum closes the gap: hard={hard:.3f} → curriculum={curr:.3f}")
    if soft > hard:
        print(f"  >> Soft gating alone helps: hard={hard:.3f} → soft={soft:.3f}")


if __name__ == '__main__':
    main()
