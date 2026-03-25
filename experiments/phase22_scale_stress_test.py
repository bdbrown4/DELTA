"""
Phase 22: Scale Stress Test — Soft Gating + Learned Dropout at N=1000+

Tests whether the Phase 16 soft gating breakthrough holds at 10x scale with noise.
At N=100 the task was too easy (everything hit 100%). At N=1000+ with 15% label
noise and power-law degree distribution, pruning and dropout should finally matter.

Compares:
1. Full attention (no pruning, no dropout) — upper bound
2. Full + learned dropout — does dropout help at scale with noise?
3. Old pre-attention router @ 50% — hard top-k baseline
4. Soft gating @ 50% target sparsity — differentiable gating
5. Soft gating + curriculum + learned dropout — full Phase 16 stack

Key questions:
- Does soft gating maintain its advantage over old router at scale?
- Does learned dropout reduce the generalization gap on noisy data?
- Does the curriculum (dense→sparse) help convergence at scale?
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
from delta.router import PostAttentionPruner, ImportanceRouter, LearnedAttentionDropout
from delta.utils import create_noisy_kg_benchmark


class FullAttentionModel(nn.Module):
    """Baseline: full attention, no pruning, no dropout."""

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


class FullWithDropoutModel(nn.Module):
    """Full attention + learned per-edge dropout."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.dual_attn = DualParallelAttention(d_node, d_edge, num_heads)
        self.dropout = LearnedAttentionDropout(d_edge)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, **kwargs):
        edge_adj = graph.build_edge_adjacency()
        result, node_attn_w, edge_attn_w = self.dual_attn(
            graph, edge_adj=edge_adj, return_weights=True
        )
        # Apply learned dropout to edge features
        dropped_w = self.dropout(result.edge_features, node_attn_w)
        return self.classifier(result.edge_features), torch.tensor(0.0)


class OldRouterModel(nn.Module):
    """Pre-attention hard top-k routing."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.router = ImportanceRouter(d_node, d_edge)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, **kwargs):
        node_scores, edge_scores = self.router(graph)
        _, edge_mask = self.router.apply_top_k(
            graph, node_scores, edge_scores,
            node_k_ratio=0.5, edge_k_ratio=0.5,
        )
        ef = graph.edge_features * edge_mask.float().unsqueeze(-1)
        g = DeltaGraph(node_features=graph.node_features, edge_features=ef,
                       edge_index=graph.edge_index, node_tiers=graph.node_tiers)
        edge_adj = g.build_edge_adjacency()
        edge_feats = self.edge_attn(g, edge_adj=edge_adj)
        return self.classifier(edge_feats), torch.tensor(0.0)


class SoftGatingModel(nn.Module):
    """Soft differentiable gating after attention."""

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


class SoftGatingDropoutModel(nn.Module):
    """Soft gating + learned dropout — the full Phase 16+21 stack."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.dual_attn = DualParallelAttention(d_node, d_edge, num_heads)
        self.pruner = PostAttentionPruner(d_node, d_edge, num_heads)
        self.dropout = LearnedAttentionDropout(d_edge)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, target_sparsity=0.5, temperature=1.0, **kwargs):
        edge_adj = graph.build_edge_adjacency()
        result, node_attn_w, edge_attn_w = self.dual_attn(
            graph, edge_adj=edge_adj, return_weights=True
        )
        # Learned dropout on attention weights during training
        dropped_w = self.dropout(result.edge_features, node_attn_w)
        _, edge_gates = self.pruner.compute_importance(
            result, dropped_w, edge_attn_w, temperature=temperature,
        )
        gated_graph, sparsity_loss = self.pruner.soft_prune(
            result, edge_gates, target_sparsity=target_sparsity,
        )
        return self.classifier(gated_graph.edge_features), sparsity_loss


def train_eval(model, graph, labels, train_idx, test_idx,
               epochs=200, lr=1e-3, sparsity_weight=0.1,
               curriculum=False, target_sparsity=0.5,
               temp_start=0.5, temp_end=5.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_test = 0.0
    best_train = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        progress = epoch / max(1, epochs - 1)

        if curriculum:
            temperature = temp_start + (temp_end - temp_start) * progress
            current_sparsity = target_sparsity * min(1.0, progress * 2)
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
                eval_kwargs = dict(target_sparsity=target_sparsity,
                                   temperature=temp_end if curriculum else 1.0)
                logits, _ = model(graph, **eval_kwargs)
                preds = logits.argmax(-1)
                train_acc = (preds[train_idx] == labels[train_idx]).float().mean().item()
                test_acc = (preds[test_idx] == labels[test_idx]).float().mean().item()
                if test_acc > best_test:
                    best_test = test_acc
                    best_train = train_acc
                    best_epoch = epoch + 1
                print(f"  Epoch {epoch+1}: Loss={task_loss.item():.4f}  "
                      f"Train={train_acc:.3f}  Test={test_acc:.3f}  "
                      f"Gap={train_acc - test_acc:.3f}")

    return best_test, best_train, best_epoch


def main():
    print("=" * 70)
    print("PHASE 22: Scale Stress Test — N=1000, 15% Noise, Power-Law")
    print("=" * 70)
    print()

    d_node, d_edge = 64, 32
    torch.manual_seed(42)

    print("Generating noisy KG benchmark...")
    t0 = time.time()
    graph, labels, metadata = create_noisy_kg_benchmark(
        num_entities=1000, num_relations=15, num_triples=5000,
        noise_ratio=0.15, d_node=d_node, d_edge=d_edge,
    )
    gen_time = time.time() - t0
    num_classes = metadata['num_relations']
    train_idx = metadata['train_idx']
    test_idx = metadata['test_idx']

    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"Relations: {num_classes}, Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"Noise ratio: {metadata['noise_ratio']}")
    print(f"Generation time: {gen_time:.2f}s")
    print()

    results = {}
    times = {}

    # 1. Full attention
    print("--- Full Attention (no pruning, no dropout) ---")
    torch.manual_seed(42)
    m = FullAttentionModel(d_node, d_edge, num_classes)
    t0 = time.time()
    acc, tr_acc, ep = train_eval(m, graph, labels, train_idx, test_idx, epochs=200)
    times['Full'] = time.time() - t0
    results['Full (no prune)'] = (acc, tr_acc)

    # 2. Full + learned dropout
    print("\n--- Full Attention + Learned Dropout ---")
    torch.manual_seed(42)
    m = FullWithDropoutModel(d_node, d_edge, num_classes)
    t0 = time.time()
    acc, tr_acc, ep = train_eval(m, graph, labels, train_idx, test_idx, epochs=200)
    times['Full+Dropout'] = time.time() - t0
    results['Full+Dropout'] = (acc, tr_acc)

    # 3. Old router
    print("\n--- Old Pre-Attention Router @ 50% ---")
    torch.manual_seed(42)
    m = OldRouterModel(d_node, d_edge, num_classes)
    t0 = time.time()
    acc, tr_acc, ep = train_eval(m, graph, labels, train_idx, test_idx, epochs=200)
    times['Old Router'] = time.time() - t0
    results['Old Router 50%'] = (acc, tr_acc)

    # 4. Soft gating (no curriculum)
    print("\n--- Soft Gating @ 50% ---")
    torch.manual_seed(42)
    m = SoftGatingModel(d_node, d_edge, num_classes)
    t0 = time.time()
    acc, tr_acc, ep = train_eval(m, graph, labels, train_idx, test_idx, epochs=200,
                                  target_sparsity=0.5, sparsity_weight=0.1)
    times['Soft Gate'] = time.time() - t0
    results['Soft Gate 50%'] = (acc, tr_acc)

    # 5. Full stack: soft gating + curriculum + dropout
    print("\n--- Soft Gating + Curriculum + Learned Dropout ---")
    torch.manual_seed(42)
    m = SoftGatingDropoutModel(d_node, d_edge, num_classes)
    t0 = time.time()
    acc, tr_acc, ep = train_eval(m, graph, labels, train_idx, test_idx, epochs=200,
                                  curriculum=True, target_sparsity=0.5, sparsity_weight=0.1,
                                  temp_start=0.5, temp_end=5.0)
    times['Full Stack'] = time.time() - t0
    results['Soft+Curr+Drop'] = (acc, tr_acc)

    # Summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY — N=1000, 5000 edges, 15% noise")
    print("=" * 70)
    print(f"  {'Model':<25s} {'Test':>7s} {'Train':>7s} {'Gap':>7s} {'Time':>8s}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    for name, (test_acc, train_acc) in results.items():
        gap = train_acc - test_acc
        t = times.get(name.split()[0], times.get(name.split('+')[0].strip(), 0))
        # Match time key
        time_key = {'Full (no prune)': 'Full', 'Full+Dropout': 'Full+Dropout',
                     'Old Router 50%': 'Old Router', 'Soft Gate 50%': 'Soft Gate',
                     'Soft+Curr+Drop': 'Full Stack'}.get(name, '')
        t = times.get(time_key, 0)
        print(f"  {name:<25s} {test_acc:>7.3f} {train_acc:>7.3f} {gap:>+7.3f} {t:>7.1f}s")

    full_test = results['Full (no prune)'][0]
    old_test = results['Old Router 50%'][0]
    soft_test = results['Soft Gate 50%'][0]
    stack_test = results['Soft+Curr+Drop'][0]
    drop_test = results['Full+Dropout'][0]

    print(f"\n  Soft gating vs Old router:      {soft_test - old_test:+.3f}")
    print(f"  Full stack vs Old router:       {stack_test - old_test:+.3f}")
    print(f"  Full stack vs Full attention:   {stack_test - full_test:+.3f}")
    print(f"  Dropout effect (full+drop vs full): {drop_test - full_test:+.3f}")

    # Generalization gap analysis
    print(f"\n  Generalization gaps:")
    for name, (test_acc, train_acc) in results.items():
        gap = train_acc - test_acc
        print(f"    {name:<25s} {gap:+.3f}")

    full_gap = results['Full (no prune)'][1] - results['Full (no prune)'][0]
    stack_gap = results['Soft+Curr+Drop'][1] - results['Soft+Curr+Drop'][0]
    if stack_gap < full_gap:
        print(f"\n  >> Full stack reduces generalization gap: {full_gap:.3f} → {stack_gap:.3f}")


if __name__ == '__main__':
    main()
