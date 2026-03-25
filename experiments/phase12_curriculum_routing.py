"""
Phase 12: Gumbel Curriculum Routing

Core question: Does starting with dense attention and gradually increasing
sparsity let the Gumbel router learn better selection policies?

Phase 7 showed Gumbel enables gradient flow (12/12 params) but trails
hard top-k on accuracy because the router hasn't learned what's important
yet when sparsity kicks in. A curriculum approach:
  1. Start with k_ratio=1.0 (full attention) so the model learns the task
  2. Gradually reduce k_ratio → force the router to identify critical elements
  3. High temperature → low temperature for sharper selection

Compared:
1. Hard top-k at fixed 40% (Phase 7 baseline)
2. Gumbel fixed 40%, no curriculum
3. Gumbel curriculum: 100% → 40% sparsity over training
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
from delta.utils import create_knowledge_graph


class RoutedEdgeClassifier(nn.Module):
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.router = ImportanceRouter(d_node, d_edge)
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, mode='hard', k_ratio=0.4, temperature=1.0):
        node_scores, edge_scores = self.router(graph)

        if mode == 'hard':
            node_mask, edge_mask = self.router.apply_top_k(
                graph, node_scores, edge_scores,
                node_k_ratio=k_ratio, edge_k_ratio=k_ratio,
            )
            weighted = graph.edge_features * edge_mask.float().unsqueeze(-1)
        elif mode == 'gumbel':
            _, edge_weights = self.router.apply_top_k_gumbel(
                graph, node_scores, edge_scores,
                node_k_ratio=k_ratio, edge_k_ratio=k_ratio,
                temperature=temperature, hard=True,
            )
            weighted = graph.edge_features * edge_weights.unsqueeze(-1)
        else:
            weighted = graph.edge_features

        wg = DeltaGraph(node_features=graph.node_features, edge_features=weighted,
                        edge_index=graph.edge_index, node_tiers=graph.node_tiers)
        edge_adj = wg.build_edge_adjacency()
        edge_feats = self.edge_attn(wg, edge_adj=edge_adj)
        return self.classifier(edge_feats)


def count_router_grads(model):
    g, t = 0, 0
    for name, p in model.named_parameters():
        if 'router' in name:
            t += 1
            if p.grad is not None and p.grad.abs().sum() > 0:
                g += 1
    return g, t


def train_and_evaluate(model, graph, labels, train_mask, test_mask,
                       mode='hard', epochs=300, lr=1e-3,
                       curriculum=False, k_start=1.0, k_end=0.4,
                       temp_start=2.0, temp_end=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    final_grads = 0

    for epoch in range(epochs):
        model.train()
        progress = epoch / max(1, epochs - 1)

        if curriculum:
            k_ratio = k_start + (k_end - k_start) * progress
            temperature = temp_start * (temp_end / temp_start) ** progress
        else:
            k_ratio = k_end
            temperature = 1.0

        logits = model(graph, mode=mode, k_ratio=k_ratio, temperature=temperature)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        g, t = count_router_grads(model)
        final_grads = g
        optimizer.step()

        if (epoch + 1) % 75 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(graph, mode=mode, k_ratio=k_end, temperature=0.1)
                acc = (logits[test_mask].argmax(-1) == labels[test_mask]).float().mean().item()
                best_acc = max(best_acc, acc)
                extra = ""
                if curriculum:
                    extra = f"  k={k_ratio:.2f} τ={temperature:.3f}"
                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}  Acc={acc:.3f}  "
                      f"Grads={g}/{t}{extra}")

    return best_acc, final_grads


def main():
    print("=" * 70)
    print("PHASE 12: Gumbel Curriculum Routing")
    print("=" * 70)
    print()
    print("Question: Does a dense→sparse curriculum help Gumbel routing?")
    print("Curriculum: k_ratio 1.0→0.4, temperature 2.0→0.1 over 300 epochs")
    print()

    d_node, d_edge = 64, 32
    num_classes = 6
    torch.manual_seed(42)

    graph, metadata = create_knowledge_graph(
        num_entities=40, num_relation_types=num_classes,
        edges_per_entity=4, d_node=d_node, d_edge=d_edge,
    )
    labels = torch.tensor(metadata['edge_labels'], dtype=torch.long)

    n_edges = graph.num_edges
    perm = torch.randperm(n_edges)
    train_end = int(n_edges * 0.7)
    train_mask = torch.zeros(n_edges, dtype=torch.bool)
    train_mask[perm[:train_end]] = True
    test_mask = ~train_mask

    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges\n")

    results = {}

    print("--- Full Attention (no sparsity) ---")
    torch.manual_seed(42)
    m = RoutedEdgeClassifier(d_node, d_edge, num_classes)
    acc, g = train_and_evaluate(m, graph, labels, train_mask, test_mask, mode='full')
    results['Full (baseline)'] = (acc, g)

    print("\n--- Hard Top-K (40%) ---")
    torch.manual_seed(42)
    m = RoutedEdgeClassifier(d_node, d_edge, num_classes)
    acc, g = train_and_evaluate(m, graph, labels, train_mask, test_mask, mode='hard')
    results['Hard 40%'] = (acc, g)

    print("\n--- Gumbel Fixed (40%, no curriculum) ---")
    torch.manual_seed(42)
    m = RoutedEdgeClassifier(d_node, d_edge, num_classes)
    acc, g = train_and_evaluate(m, graph, labels, train_mask, test_mask,
                                mode='gumbel', curriculum=False)
    results['Gumbel fixed'] = (acc, g)

    print("\n--- Gumbel Curriculum (100%→40%) ---")
    torch.manual_seed(42)
    m = RoutedEdgeClassifier(d_node, d_edge, num_classes)
    acc, g = train_and_evaluate(m, graph, labels, train_mask, test_mask,
                                mode='gumbel', curriculum=True)
    results['Gumbel curriculum'] = (acc, g)

    # Summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for name, (acc, grads) in results.items():
        bar = '#' * int(acc * 40)
        print(f"  {name:25s} Acc: {acc:.3f}  {bar}  Grads: {grads}")

    gc = results['Gumbel curriculum'][0]
    gf = results['Gumbel fixed'][0]
    hk = results['Hard 40%'][0]
    print(f"\n  Curriculum vs Fixed Gumbel: {gc - gf:+.3f}")
    print(f"  Curriculum vs Hard Top-K:   {gc - hk:+.3f}")
    if gc > gf:
        print("  >> Curriculum improves Gumbel routing — the router learns what to keep.")
    if gc > hk:
        print("  >> Gumbel curriculum beats hard top-k — differentiable routing pays off.")


if __name__ == '__main__':
    main()
