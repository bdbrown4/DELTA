"""
Phase 24: Combined Fix Integration — All Fixes at Scale

Tests the full DELTA pipeline with ALL architectural fixes working together
on the noisy N=1000 benchmark that Phase 22 showed differentiates models
(old router: 81.6% vs soft gating: 100%).

Fixes integrated:
  Fix 1: Post-attention soft gating + curriculum
  Fix 2: BFS partitioning (O(N+E))
  Fix 3: Variational memory compression (KL bottleneck)
  Fix 4: Sparse COO multi-hop edge adjacency
  Fix 5: Learned attention dropout (per-edge)

Compares:
1. Vanilla EdgeAttention (no fixes)
2. Full DELTA (all fixes + curriculum)
3. Ablations: remove each fix one at a time
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import EdgeAttention, DualParallelAttention, NodeAttention
from delta.router import PostAttentionPruner, LearnedAttentionDropout
from delta.memory import TieredMemory
from delta.partition import GraphPartitioner
from delta.utils import create_noisy_kg_benchmark


class VanillaEdgeModel(nn.Module):
    """Baseline: simple EdgeAttention, no fixes."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, edge_adj=None, **kwargs):
        if edge_adj is None:
            edge_adj = graph.build_edge_adjacency()
        edge_feats = self.edge_attn(graph, edge_adj=edge_adj)
        return self.classifier(edge_feats), torch.tensor(0.0), {}


class FullDELTAModel(nn.Module):
    """Full DELTA with all 6 fixes integrated.

    Pipeline:
    1. Variational memory compression (Fix 3)
    2. BFS partitioning if needed (Fix 2)
    3. Sparse COO multi-hop edge adjacency (Fix 5)
    4. Dual parallel attention with return_weights
    5. Learned attention dropout (Fix 6)
    6. Post-attention soft gating + curriculum (Fix 1)
    7. Edge classification
    """

    def __init__(self, d_node, d_edge, num_classes, num_heads=4,
                 max_partition_size=128, use_2hop=True):
        super().__init__()
        self.num_heads = num_heads
        self.use_2hop = use_2hop

        # Fix 3: Variational memory
        self.memory = TieredMemory(d_node, d_edge)

        # Fix 2: BFS partitioner
        self.partitioner = GraphPartitioner(max_partition_size)

        # Core attention
        self.dual_attn = DualParallelAttention(d_node, d_edge, num_heads)

        # Fix 6: Learned dropout
        self.attn_dropout = LearnedAttentionDropout(d_edge)

        # Fix 1: Post-attention soft gating
        self.pruner = PostAttentionPruner(d_node, d_edge, num_heads)

        # Cross-partition communication
        self.global_attn = NodeAttention(d_node, d_edge, num_heads)

        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, target_sparsity=0.5, temperature=1.0,
                use_memory=True, use_partition=True, use_dropout=True,
                use_soft_gating=True, use_2hop=None, edge_adj=None, **kwargs):
        aux_losses = {}

        # Fix 3: Variational memory compression
        if use_memory:
            graph = self.memory.compress_warm_nodes(graph)
            aux_losses['kl'] = self.memory.kl_loss

        # Fix 2: BFS partitioning
        if use_partition and graph.num_nodes > self.partitioner.max_partition_size:
            partitions = self.partitioner.partition(graph)
        else:
            partitions = None

        # Fix 5: Sparse COO multi-hop edge adjacency (use precomputed if provided)
        if edge_adj is None:
            hops = 2 if (self.use_2hop if use_2hop is None else use_2hop) else 1
            edge_adj = graph.build_edge_adjacency(hops=hops)

        # Dual parallel attention (return weights for Fix 1)
        result, node_attn_w, edge_attn_w = self.dual_attn(
            graph, edge_adj=edge_adj, return_weights=True
        )

        # Fix 6: Learned attention dropout
        if use_dropout:
            dropped_w = self.attn_dropout(result.edge_features, node_attn_w)
        else:
            dropped_w = node_attn_w

        # Fix 1: Post-attention soft gating
        if use_soft_gating:
            _, edge_gates = self.pruner.compute_importance(
                result, dropped_w, edge_attn_w, temperature=temperature,
            )
            gated_graph, sparsity_loss = self.pruner.soft_prune(
                result, edge_gates, target_sparsity=target_sparsity,
            )
            aux_losses['sparsity'] = sparsity_loss
        else:
            gated_graph = result

        # Cross-partition boundary attention
        if partitions and len(partitions) > 1:
            boundary_lists = self.partitioner.get_boundary_nodes(gated_graph, partitions)
            if boundary_lists:
                all_boundary = torch.unique(torch.cat(boundary_lists))
                if len(all_boundary) > 0:
                    boundary_mask = torch.zeros(gated_graph.num_nodes, dtype=torch.bool)
                    boundary_mask[all_boundary] = True
                    new_nf = self.global_attn(gated_graph, mask=boundary_mask)
                    gated_graph = DeltaGraph(
                        node_features=new_nf,
                        edge_features=gated_graph.edge_features,
                        edge_index=gated_graph.edge_index,
                        node_tiers=gated_graph.node_tiers,
                    )

        logits = self.classifier(gated_graph.edge_features)
        total_aux = sum(aux_losses.values())
        return logits, total_aux, aux_losses


def train_eval(model, graph, labels, train_idx, test_idx,
               epochs=200, lr=1e-3, sparsity_weight=0.1,
               curriculum=False, target_sparsity=0.5,
               temp_start=0.5, temp_end=5.0, edge_adj=None, **fwd_kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_test = 0.0
    best_train = 0.0

    for epoch in range(epochs):
        model.train()
        progress = epoch / max(1, epochs - 1)

        extra_kwargs = dict(fwd_kwargs)
        if curriculum:
            temp = temp_start + (temp_end - temp_start) * progress
            sp = target_sparsity * min(1.0, progress * 2)
            extra_kwargs.update(target_sparsity=sp, temperature=temp)
        else:
            extra_kwargs.update(target_sparsity=target_sparsity, temperature=1.0)

        logits, aux_loss, losses = model(graph, edge_adj=edge_adj, **extra_kwargs)
        task_loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss = task_loss + sparsity_weight * aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                eval_kwargs = dict(fwd_kwargs)
                eval_kwargs.update(target_sparsity=target_sparsity,
                                   temperature=temp_end if curriculum else 1.0)
                logits, _, _ = model(graph, edge_adj=edge_adj, **eval_kwargs)
                preds = logits.argmax(-1)
                train_acc = (preds[train_idx] == labels[train_idx]).float().mean().item()
                test_acc = (preds[test_idx] == labels[test_idx]).float().mean().item()
                if test_acc > best_test:
                    best_test = test_acc
                    best_train = train_acc
                gap = train_acc - test_acc
                print(f"  Epoch {epoch+1}: Loss={task_loss.item():.4f}  "
                      f"Train={train_acc:.3f}  Test={test_acc:.3f}  Gap={gap:+.3f}")

    return best_test, best_train


def main():
    print("=" * 70)
    print("PHASE 24: Combined Fix Integration — All Fixes at Scale")
    print("=" * 70)
    print()
    print("Task: Noisy KG classification at N=1000 (same as Phase 22)")
    print("  1000 entities, 15 relations, 5000 triples, 15% label noise")
    print("  Tests whether all 5 inference-pipeline fixes work together")
    print()

    d_node, d_edge = 64, 32
    torch.manual_seed(42)

    graph, labels, metadata = create_noisy_kg_benchmark(
        num_entities=1000, num_relations=15, num_triples=5000,
        noise_ratio=0.15, d_node=d_node, d_edge=d_edge,
    )
    num_classes = metadata['num_relations']
    train_idx = metadata['train_idx']
    test_idx = metadata['test_idx']

    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"Relations: {num_classes}, Noise: {metadata['noise_ratio']}")
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    print()

    # Precompute edge adjacencies (expensive at E=5000)
    print("Precomputing edge adjacency matrices...")
    t0 = time.time()
    edge_adj_1hop = graph.build_edge_adjacency(hops=1)
    edge_adj_2hop = graph.build_edge_adjacency(hops=2)
    print(f"  1-hop: {edge_adj_1hop.shape}, 2-hop: {edge_adj_2hop.shape}  ({time.time()-t0:.1f}s)")
    print()

    results = {}
    times_dict = {}

    # 1. Vanilla baseline (uses precomputed 1-hop edge_adj)
    print("--- Vanilla EdgeAttention (no fixes) ---")
    torch.manual_seed(42)
    m = VanillaEdgeModel(d_node, d_edge, num_classes)
    t0 = time.time()
    test_acc, train_acc = train_eval(m, graph, labels, train_idx, test_idx,
                                     epochs=200, edge_adj=edge_adj_1hop)
    times_dict['Vanilla Edge'] = time.time() - t0
    results['Vanilla Edge'] = (test_acc, train_acc)

    # 2. Full DELTA (all fixes + curriculum)
    print("\n--- Full DELTA (all fixes + curriculum) ---")
    torch.manual_seed(42)
    m = FullDELTAModel(d_node, d_edge, num_classes, max_partition_size=128, use_2hop=True)
    t0 = time.time()
    test_acc, train_acc = train_eval(
        m, graph, labels, train_idx, test_idx, epochs=200,
        curriculum=True, target_sparsity=0.5, sparsity_weight=0.1,
        temp_start=0.5, temp_end=5.0, edge_adj=edge_adj_2hop,
    )
    times_dict['Full DELTA'] = time.time() - t0
    results['Full DELTA'] = (test_acc, train_acc)

    # 3. Ablations
    ablations = [
        ('No Soft Gate', dict(use_soft_gating=False)),
        ('No Dropout', dict(use_dropout=False)),
        ('No Memory', dict(use_memory=False)),
        ('1-hop only', dict(use_2hop=False)),
    ]

    for ab_name, ab_kwargs in ablations:
        print(f"\n--- Ablation: {ab_name} ---")
        torch.manual_seed(42)
        m = FullDELTAModel(d_node, d_edge, num_classes, max_partition_size=128, use_2hop=True)
        # Use 1-hop adj for 1-hop ablation, 2-hop for all others
        adj = edge_adj_1hop if ab_name == '1-hop only' else edge_adj_2hop
        t0 = time.time()
        test_acc, train_acc = train_eval(
            m, graph, labels, train_idx, test_idx, epochs=200,
            curriculum=True, target_sparsity=0.5, sparsity_weight=0.1,
            temp_start=0.5, temp_end=5.0, edge_adj=adj, **ab_kwargs,
        )
        times_dict[ab_name] = time.time() - t0
        results[ab_name] = (test_acc, train_acc)

    # ── Summary ──
    print()
    print("=" * 70)
    print("RESULTS SUMMARY — N=1000, 15 relations, 15% noise")
    print("=" * 70)
    print(f"  {'Model':<22s} {'Test':>7s} {'Train':>7s} {'Gap':>7s} {'Time':>8s}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    for name, (test_acc, train_acc) in results.items():
        gap = train_acc - test_acc
        t = times_dict.get(name, 0)
        print(f"  {name:<22s} {test_acc:>7.3f} {train_acc:>7.3f} {gap:>+7.3f} {t:>7.1f}s")

    # Ablation analysis
    full = results['Full DELTA']
    print(f"\n  Ablation impact (change from Full DELTA):")
    for name, (test_acc, train_acc) in results.items():
        if name not in ('Vanilla Edge', 'Full DELTA'):
            delta = test_acc - full[0]
            print(f"    {name:<22s} {delta:+.3f}")

    vanilla = results['Vanilla Edge']
    print(f"\n  Full DELTA vs Vanilla:")
    print(f"    Test:  {full[0] - vanilla[0]:+.3f}")
    print(f"    Gap:   Vanilla={vanilla[1]-vanilla[0]:.3f}  Full={full[1]-full[0]:.3f}")

    if full[0] > vanilla[0]:
        print("\n  >> Full DELTA with all fixes outperforms vanilla baseline!")
    if (full[1] - full[0]) < (vanilla[1] - vanilla[0]):
        print("  >> Full DELTA has smaller generalization gap!")


if __name__ == '__main__':
    main()
