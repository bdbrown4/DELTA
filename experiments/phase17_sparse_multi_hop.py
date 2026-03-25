"""
Phase 17: Sparse Multi-Hop Scaling Benchmark

Core question: Does the sparse COO multi-hop implementation (Fix 5) scale
to larger graphs where the old dense O(E²) approach timed out?

The old graph.py used dense torch.zeros(E, E) matrix multiplication for
multi-hop edge adjacency. This timed out at ~500 edges. The new sparse
implementation uses torch.sparse_coo_tensor + torch.sparse.mm.

Benchmark:
1. Time 1-hop and 2-hop edge adjacency at increasing graph sizes
2. Verify correctness: 2-hop should find more edge pairs than 1-hop
3. Memory usage: sparse should use O(nnz) not O(E²)

Key metric: 2-hop adjacency completes at 500+ edges (was timeout before).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
from delta.graph import DeltaGraph


def make_graph(num_nodes, num_edges, d_node=32, d_edge=16):
    """Create a random graph with given node/edge counts."""
    edge_index = torch.stack([
        torch.randint(0, num_nodes, (num_edges,)),
        torch.randint(0, num_nodes, (num_edges,)),
    ])
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    E = edge_index.shape[1]
    return DeltaGraph(
        node_features=torch.randn(E if E < num_nodes else num_nodes, d_node)
        if False else torch.randn(num_nodes, d_node),
        edge_features=torch.randn(E, d_edge),
        edge_index=edge_index,
    )


def benchmark_adjacency(graph, hops, label=""):
    """Time edge adjacency construction and report stats."""
    start = time.perf_counter()
    adj = graph.build_edge_adjacency(hops=hops)
    elapsed = time.perf_counter() - start
    num_pairs = adj.shape[1]
    print(f"  {label}: {num_pairs:>8d} edge-adj pairs in {elapsed:.4f}s")
    return elapsed, num_pairs


def main():
    print("=" * 70)
    print("PHASE 17: Sparse Multi-Hop Scaling Benchmark")
    print("=" * 70)
    print()
    print("Fix 5 validation: Sparse COO multi-hop vs old dense approach")
    print("Old approach: dense O(E²) matrix — timed out at ~500 edges")
    print("New approach: sparse COO tensor ops — should handle 1000+ edges")
    print()

    torch.manual_seed(42)

    # Test at increasing scales
    scales = [
        (20,  50,   "Small"),
        (50,  200,  "Medium"),
        (100, 500,  "Large (old timeout)"),
        (200, 1000, "XL"),
        (500, 2500, "XXL"),
    ]

    results_1h = []
    results_2h = []

    print(f"  {'Scale':<22s} {'Nodes':>6s} {'Edges':>6s} {'1-hop pairs':>12s} {'1-hop time':>10s} {'2-hop pairs':>12s} {'2-hop time':>10s}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

    for num_nodes, num_edges, label in scales:
        graph = make_graph(num_nodes, num_edges)
        actual_edges = graph.num_edges

        # 1-hop
        start = time.perf_counter()
        adj_1 = graph.build_edge_adjacency(hops=1)
        t1 = time.perf_counter() - start
        p1 = adj_1.shape[1]

        # 2-hop
        start = time.perf_counter()
        adj_2 = graph.build_edge_adjacency(hops=2)
        t2 = time.perf_counter() - start
        p2 = adj_2.shape[1]

        results_1h.append((label, actual_edges, p1, t1))
        results_2h.append((label, actual_edges, p2, t2))

        print(f"  {label:<22s} {num_nodes:>6d} {actual_edges:>6d} {p1:>12d} {t1:>10.4f}s {p2:>12d} {t2:>10.4f}s")

        # Correctness check: 2-hop should find >= 1-hop pairs
        assert p2 >= p1, f"2-hop ({p2}) < 1-hop ({p1}) — incorrect!"

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    # Scaling analysis
    if len(results_2h) >= 2:
        base_e = results_2h[0][1]
        base_t = results_2h[0][2]
        last_e = results_2h[-1][1]
        last_t = results_2h[-1][2]

        if base_t > 0 and last_t > 0 and base_e > 0 and last_e > 0:
            import math
            edge_ratio = last_e / base_e
            time_ratio = results_2h[-1][3] / max(results_2h[0][3], 1e-6)
            if time_ratio > 0 and edge_ratio > 1:
                exponent = math.log(time_ratio) / math.log(edge_ratio)
                print(f"  2-hop scaling exponent: O(E^{exponent:.2f})")
                if exponent < 2.0:
                    print("  >> Sub-quadratic scaling confirmed (sparse ops working!)")
                else:
                    print("  >> Still quadratic — sparse approach may need optimization")

    all_passed = all(r2[2] >= r1[2] for r1, r2 in zip(results_1h, results_2h))
    timeout_scale = [r for r in results_2h if r[1] >= 400]
    if timeout_scale and all(r[3] < 30.0 for r in timeout_scale):
        print(f"  >> 500+ edge graphs complete in <30s (was timeout before)!")
    print(f"  >> Correctness: {'ALL PASSED' if all_passed else 'FAILURES DETECTED'}")


if __name__ == '__main__':
    main()
