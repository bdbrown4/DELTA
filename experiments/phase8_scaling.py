"""
Phase 8: Scaling Experiment

Core question: How does DELTA's compute cost scale vs. accuracy as graph
size increases? Does sparse routing maintain quality at larger scales?

Tests:
1. Accuracy vs. graph size (50 → 500 nodes)
2. Wall-clock time vs. graph size
3. Effective sparsity (how many elements the router selects)
4. Memory tier distribution at different scales
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn.functional as F

from delta.model import DELTAModel
from delta.utils import create_knowledge_graph


def benchmark_at_scale(num_entities, d_node=32, d_edge=16, num_classes=5,
                       epochs=100, sparse_ratio=0.5):
    """Train DELTA at a given graph scale, return metrics."""
    graph, metadata = create_knowledge_graph(
        num_entities=num_entities,
        num_relation_types=num_classes,
        edges_per_entity=3,
        d_node=d_node, d_edge=d_edge,
    )
    labels = torch.tensor(metadata['edge_labels'], dtype=torch.long)

    # Train/test split
    n_edges = graph.num_edges
    perm = torch.randperm(n_edges)
    train_end = int(n_edges * 0.7)
    train_mask = torch.zeros(n_edges, dtype=torch.bool)
    train_mask[perm[:train_end]] = True
    test_mask = ~train_mask

    model = DELTAModel(
        d_node=d_node, d_edge=d_edge,
        num_layers=2, num_heads=4,
        num_classes=num_classes,
        sparse_ratio=sparse_ratio,
    )
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_time = time.time()
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        output = model(graph, use_router=True, use_memory=True)
        logits = model.classify_edges(output)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                output = model(graph, use_router=True, use_memory=True)
                logits = model.classify_edges(output)
                acc = (logits[test_mask].argmax(-1) == labels[test_mask]).float().mean().item()
                best_acc = max(best_acc, acc)

    elapsed = time.time() - start_time

    # Get tier distribution from final pass
    model.eval()
    with torch.no_grad():
        output = model(graph, use_router=True, use_memory=True)
        hot = (output.node_tiers == 0).sum().item()
        warm = (output.node_tiers == 1).sum().item()
        cold = (output.node_tiers == 2).sum().item()

    return {
        'nodes': graph.num_nodes,
        'edges': graph.num_edges,
        'params': param_count,
        'best_acc': best_acc,
        'time_sec': elapsed,
        'time_per_epoch': elapsed / epochs,
        'hot': hot, 'warm': warm, 'cold': cold,
    }


def main():
    print("=" * 70)
    print("PHASE 8: Scaling Experiment")
    print("=" * 70)
    print()
    print("Question: How does DELTA scale with graph size?")
    print("Measuring: accuracy, wall-clock time, memory tier distribution")
    print()

    torch.manual_seed(42)

    scales = [20, 50, 100, 200, 400]
    results = []

    for n_entities in scales:
        print(f"--- Scale: {n_entities} entities ---")
        r = benchmark_at_scale(n_entities, epochs=80)
        results.append(r)
        print(f"  Graph: {r['nodes']} nodes, {r['edges']} edges")
        print(f"  Best Acc: {r['best_acc']:.3f}")
        print(f"  Time: {r['time_sec']:.1f}s ({r['time_per_epoch']*1000:.0f}ms/epoch)")
        print(f"  Tiers — Hot: {r['hot']}, Warm: {r['warm']}, Cold: {r['cold']}")
        print()

    # --- Summary ---
    print("=" * 70)
    print("SCALING SUMMARY")
    print("=" * 70)
    print(f"  {'Nodes':>6}  {'Edges':>6}  {'Acc':>6}  {'ms/epoch':>10}  {'Hot':>4}  {'Warm':>5}  {'Cold':>5}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*4}  {'-'*5}  {'-'*5}")
    for r in results:
        print(f"  {r['nodes']:>6}  {r['edges']:>6}  {r['best_acc']:>6.3f}  "
              f"{r['time_per_epoch']*1000:>10.0f}  "
              f"{r['hot']:>4}  {r['warm']:>5}  {r['cold']:>5}")

    # Scaling analysis
    if len(results) >= 2:
        t0 = results[0]['time_per_epoch']
        t_last = results[-1]['time_per_epoch']
        n0 = results[0]['nodes']
        n_last = results[-1]['nodes']
        if t0 > 0 and n0 > 0:
            import math
            ratio_t = t_last / t0
            ratio_n = n_last / n0
            exponent = math.log(ratio_t) / math.log(ratio_n) if ratio_n > 1 else float('inf')
            print(f"\n  Scaling exponent: O(n^{exponent:.2f})")
            if exponent < 1.5:
                print("  >> Sub-quadratic scaling — router sparsity is effective.")
            elif exponent < 2.0:
                print("  >> Near-quadratic — attention dominates at this scale.")
            else:
                print("  >> Super-quadratic — partitioning needed for larger graphs.")


if __name__ == '__main__':
    main()
