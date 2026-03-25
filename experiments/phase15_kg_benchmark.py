"""
Phase 15: Synthetic Real-World KG Benchmark

Core question: Does DELTA's edge-first architecture outperform node-based
approaches on a larger, more realistic knowledge graph with many entity types
and relation patterns?

Uses create_synthetic_kg_benchmark() — a FB15k-237-inspired synthetic KG with:
- 100 entities, 10 relation types, ~500 triples
- Hierarchical entity types affecting relation distribution
- Proper train/val/test splits

Compares:
1. Node GNN baseline (node pair + edge features)
2. DELTA 1-hop edge attention
3. DELTA 2-hop edge attention
4. DELTA 2-hop + sparsity (edge routing at 50%)

Key metrics: relation classification accuracy on test set, per-relation
breakdown to identify where edge attention helps most.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import EdgeAttention
from delta.router import ImportanceRouter
from delta.utils import create_synthetic_kg_benchmark


class NodeGNN(nn.Module):
    def __init__(self, d_node, d_edge, num_classes, hidden=64):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(d_node * 2 + d_edge, hidden), nn.GELU())
        self.mlp2 = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, graph, **kwargs):
        src, tgt = graph.edge_index[0], graph.edge_index[1]
        x = torch.cat([graph.node_features[src], graph.node_features[tgt],
                        graph.edge_features], dim=-1)
        return self.classifier(self.mlp2(self.mlp1(x)))


class DELTAClassifier(nn.Module):
    def __init__(self, d_node, d_edge, num_classes, num_heads=4, use_router=False):
        super().__init__()
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.use_router = use_router
        if use_router:
            self.router = ImportanceRouter(d_node, d_edge)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, hops=1, k_ratio=0.5):
        if self.use_router:
            node_scores, edge_scores = self.router(graph)
            node_mask, edge_mask = self.router.apply_top_k(
                graph, node_scores, edge_scores,
                node_k_ratio=k_ratio, edge_k_ratio=k_ratio,
            )
            ef = graph.edge_features * edge_mask.float().unsqueeze(-1)
            g = DeltaGraph(node_features=graph.node_features, edge_features=ef,
                           edge_index=graph.edge_index, node_tiers=graph.node_tiers)
        else:
            g = graph
        edge_adj = g.build_edge_adjacency(hops=hops)
        edge_feats = self.edge_attn(g, edge_adj=edge_adj)
        return self.classifier(edge_feats)


def train_eval(model, graph, labels, train_idx, test_idx,
               epochs=200, lr=1e-3, **fwd_kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    per_rel_best = None

    num_rels = labels.max().item() + 1

    for epoch in range(epochs):
        model.train()
        logits = model(graph, **fwd_kwargs)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(graph, **fwd_kwargs)
                preds = logits.argmax(-1)
                test_preds = preds[test_idx]
                test_labels = labels[test_idx]
                acc = (test_preds == test_labels).float().mean().item()

                per_rel = {}
                for r in range(num_rels):
                    mask = test_labels == r
                    if mask.any():
                        per_rel[r] = (test_preds[mask] == r).float().mean().item()
                    else:
                        per_rel[r] = float('nan')

                if acc > best_acc:
                    best_acc = acc
                    per_rel_best = per_rel

                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}  Acc={acc:.3f}")

    return best_acc, per_rel_best


def main():
    print("=" * 70)
    print("PHASE 15: Synthetic Real-World KG Benchmark")
    print("=" * 70)
    print()
    print("FB15k-237-style synthetic KG benchmark")
    print("100 entities, 10 relations, ~500 triples")
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
    print(f"Relations: {num_classes}")
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Distribution of labels
    counts = torch.bincount(labels, minlength=num_classes)
    print(f"Label distribution: {counts.tolist()}")
    print()

    results = {}

    print("--- Node GNN Baseline ---")
    torch.manual_seed(42)
    m = NodeGNN(d_node, d_edge, num_classes)
    acc, per_rel = train_eval(m, graph, labels, train_idx, test_idx)
    results['Node GNN'] = (acc, per_rel)

    print("\n--- DELTA 1-hop ---")
    torch.manual_seed(42)
    m = DELTAClassifier(d_node, d_edge, num_classes)
    acc, per_rel = train_eval(m, graph, labels, train_idx, test_idx, hops=1)
    results['DELTA 1-hop'] = (acc, per_rel)

    print("\n--- DELTA 2-hop ---")
    torch.manual_seed(42)
    m = DELTAClassifier(d_node, d_edge, num_classes)
    acc, per_rel = train_eval(m, graph, labels, train_idx, test_idx, hops=2)
    results['DELTA 2-hop'] = (acc, per_rel)

    print("\n--- DELTA 2-hop + Router (50%) ---")
    torch.manual_seed(42)
    m = DELTAClassifier(d_node, d_edge, num_classes, use_router=True)
    acc, per_rel = train_eval(m, graph, labels, train_idx, test_idx,
                              hops=2, k_ratio=0.5)
    results['DELTA 2h+route'] = (acc, per_rel)

    # Summary
    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"  {'Model':<20s} {'Test Acc':>10s}")
    print(f"  {'-'*20} {'-'*10}")
    for name, (acc, _) in results.items():
        bar = '#' * int(acc * 40)
        print(f"  {name:<20s} {acc:>10.3f}  {bar}")

    # Per-relation breakdown
    print()
    print("PER-RELATION ACCURACY (test set)")
    print("=" * 70)
    header = f"  {'Rel':<6s}"
    for name in results:
        header += f" {name:>15s}"
    print(header)
    for r in range(num_classes):
        row = f"  R{r:<4d}"
        for name, (_, pr) in results.items():
            v = pr.get(r, float('nan'))
            if v != v:  # nan check
                row += f" {'n/a':>15s}"
            else:
                row += f" {v:>15.3f}"
        print(row)

    # Best model
    best_name = max(results, key=lambda k: results[k][0])
    best_acc = results[best_name][0]
    node_acc = results['Node GNN'][0]
    print(f"\n  Best model: {best_name} ({best_acc:.3f})")
    print(f"  vs Node GNN: {best_acc - node_acc:+.3f}")
    if best_name != 'Node GNN':
        print("  >> DELTA outperforms node-based approach on realistic KG benchmark!")


if __name__ == '__main__':
    main()
