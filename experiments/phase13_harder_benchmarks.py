"""
Phase 13: Harder Compositional Benchmarks

Core question: Can DELTA's edge-to-edge attention discover derived relations
that require composing multiple base relations through logical rules?

Uses create_multi_relational_reasoning_task() which creates:
  Rule 1: worksAt + locatedIn => livesNear
  Rule 2: friendOf + friendOf => peerOf
  Rule 3: manages + manages => seniorTo

Compares:
1. Node GNN baseline (standard message passing)
2. DELTA 1-hop (original edge adjacency)
3. DELTA 2-hop (multi-hop edges from Phase 11)

Key metric: accuracy on DERIVED relations specifically — these require
compositional reasoning that node-level message passing may miss but
edge-to-edge attention with multi-hop adjacency should capture.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import EdgeAttention
from delta.utils import create_multi_relational_reasoning_task


class NodeGNN(nn.Module):
    def __init__(self, d_node, d_edge, num_classes, hidden=64):
        super().__init__()
        self.node_mlp1 = nn.Sequential(nn.Linear(d_node * 2, hidden), nn.GELU())
        self.node_mlp2 = nn.Sequential(nn.Linear(hidden + d_edge, hidden), nn.GELU())
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, graph):
        src, tgt = graph.edge_index[0], graph.edge_index[1]
        pair_feat = torch.cat([graph.node_features[src], graph.node_features[tgt]], dim=-1)
        h = self.node_mlp1(pair_feat)
        h = self.node_mlp2(torch.cat([h, graph.edge_features], dim=-1))
        return self.classifier(h)


class DELTAEdgeClassifier(nn.Module):
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, hops=1):
        edge_adj = graph.build_edge_adjacency(hops=hops)
        edge_feats = self.edge_attn(graph, edge_adj=edge_adj)
        return self.classifier(edge_feats)


def train_eval(model, graph, labels, train_mask, test_mask, derived_mask,
               epochs=200, lr=1e-3, hops=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    best_base_acc = 0.0
    best_derived_acc = 0.0

    for epoch in range(epochs):
        model.train()
        logits = model(graph, hops=hops) if hops else model(graph)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(graph, hops=hops) if hops else model(graph)
                preds = logits.argmax(-1)

                # Overall
                test_correct = (preds[test_mask] == labels[test_mask]).float()
                acc = test_correct.mean().item()

                # Base relations only
                base_test = test_mask & ~derived_mask
                if base_test.any():
                    base_acc = (preds[base_test] == labels[base_test]).float().mean().item()
                else:
                    base_acc = 0.0

                # Derived relations only
                derived_test = test_mask & derived_mask
                if derived_test.any():
                    derived_acc = (preds[derived_test] == labels[derived_test]).float().mean().item()
                else:
                    derived_acc = 0.0

                best_acc = max(best_acc, acc)
                best_base_acc = max(best_base_acc, base_acc)
                best_derived_acc = max(best_derived_acc, derived_acc)

                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}  "
                      f"All={acc:.3f}  Base={base_acc:.3f}  Derived={derived_acc:.3f}")

    return best_acc, best_base_acc, best_derived_acc


def main():
    print("=" * 70)
    print("PHASE 13: Harder Compositional Benchmarks")
    print("=" * 70)
    print()
    print("Task: Classify edges including DERIVED relations from logical rules:")
    print("  Rule 1: worksAt + locatedIn => livesNear")
    print("  Rule 2: friendOf + friendOf => peerOf")
    print("  Rule 3: manages + manages => seniorTo")
    print()

    d_node, d_edge = 64, 32
    torch.manual_seed(42)

    graph, labels, metadata = create_multi_relational_reasoning_task(
        num_entities=40, num_base_relations=4, num_derived_rules=3,
        d_node=d_node, d_edge=d_edge,
    )
    num_classes = metadata['num_total_relations']
    derived_mask = metadata['derived_mask']

    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"  Base edges: {metadata['n_base']}, Derived edges: {metadata['n_derived']}")
    print(f"  Relation types: {num_classes} (4 base + 3 derived)")
    print()

    # Train/test split
    n = graph.num_edges
    perm = torch.randperm(n)
    train_end = int(n * 0.7)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:train_end]] = True
    test_mask = ~train_mask

    results = {}

    print("--- Node GNN Baseline ---")
    torch.manual_seed(42)
    m = NodeGNN(d_node, d_edge, num_classes)
    all_acc, base_acc, der_acc = train_eval(m, graph, labels, train_mask, test_mask, derived_mask)
    results['Node GNN'] = (all_acc, base_acc, der_acc)

    print("\n--- DELTA 1-hop ---")
    torch.manual_seed(42)
    m = DELTAEdgeClassifier(d_node, d_edge, num_classes)
    all_acc, base_acc, der_acc = train_eval(m, graph, labels, train_mask, test_mask, derived_mask, hops=1)
    results['DELTA 1-hop'] = (all_acc, base_acc, der_acc)

    print("\n--- DELTA 2-hop ---")
    torch.manual_seed(42)
    m = DELTAEdgeClassifier(d_node, d_edge, num_classes)
    all_acc, base_acc, der_acc = train_eval(m, graph, labels, train_mask, test_mask, derived_mask, hops=2)
    results['DELTA 2-hop'] = (all_acc, base_acc, der_acc)

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<20s} {'Overall':>8s} {'Base':>8s} {'Derived':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for name, (a, b, d) in results.items():
        print(f"  {name:<20s} {a:>8.3f} {b:>8.3f} {d:>8.3f}")

    d2 = results['DELTA 2-hop']
    ng = results['Node GNN']
    print(f"\n  DELTA 2-hop vs Node GNN on derived: {d2[2] - ng[2]:+.3f}")
    if d2[2] > ng[2]:
        print("  >> DELTA 2-hop outperforms Node GNN on compositional relations!")
    if d2[2] > results['DELTA 1-hop'][2]:
        print("  >> Multi-hop adjacency improves compositional reasoning over 1-hop.")


if __name__ == '__main__':
    main()
