"""
Phase 20: BFS Partition Scaling Benchmark

Core question: Does the BFS seed-expansion partitioner (Fix 2) scale
better than spectral clustering?

The old GraphPartitioner used O(N^3) spectral clustering (eigendecomposition
of Laplacian). The new BFS approach runs in O(N + E) time.

Benchmark:
1. Wall-clock time at increasing graph sizes (50, 200, 500, 1000, 2500 nodes)
2. Partition balance quality (std of partition sizes)
3. Boundary edge ratio (edges crossing partition boundaries)
4. Scaling exponent: fit log(time) vs log(N)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import math
import torch

from delta.graph import DeltaGraph
from delta.partition import GraphPartitioner


def make_random_graph(num_nodes, avg_degree=6, d_node=32, d_edge=16):
    """Create a random graph with given average degree."""
    num_edges = num_nodes * avg_degree
    src = torch.randint(0, num_nodes, (num_edges,))
    tgt = torch.randint(0, num_nodes, (num_edges,))
    # Remove self-loops
    mask = src != tgt
    src, tgt = src[mask], tgt[mask]
    edge_index = torch.stack([src, tgt])

    return DeltaGraph(
        node_features=torch.randn(num_nodes, d_node),
        edge_features=torch.randn(edge_index.shape[1], d_edge),
        edge_index=edge_index,
    )


def test_scaling():
    """Time BFS partition at increasing graph sizes."""
    print("--- Test 1: BFS Partition Scaling ---")
    scales = [50, 200, 500, 1000, 2500]
    partitioner = GraphPartitioner(max_partition_size=32)
    times = []

    for N in scales:
        graph = make_random_graph(N)
        num_parts = max(2, N // 32)

        # Warm up
        partitioner.partition(graph, num_partitions=num_parts)

        # Timed run (average 3)
        elapsed = []
        for _ in range(3):
            t0 = time.perf_counter()
            partitions = partitioner.partition(graph, num_partitions=num_parts)
            t1 = time.perf_counter()
            elapsed.append(t1 - t0)

        avg_time = sum(elapsed) / len(elapsed)
        times.append(avg_time)
        num_actual = len(partitions)
        sizes = [len(p) for p in partitions]

        print(f"  N={N:5d}: {avg_time*1000:8.2f} ms  |  {num_actual} partitions  "
              f"|  sizes: min={min(sizes)} max={max(sizes)} mean={sum(sizes)/len(sizes):.1f}")

    # Compute scaling exponent
    if len(times) >= 2:
        log_n = [math.log(n) for n in scales]
        log_t = [math.log(max(t, 1e-9)) for t in times]
        # Linear regression on log-log
        n = len(log_n)
        sum_x = sum(log_n)
        sum_y = sum(log_t)
        sum_xy = sum(x * y for x, y in zip(log_n, log_t))
        sum_x2 = sum(x * x for x in log_n)
        exponent = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        print(f"\n  Scaling exponent: {exponent:.2f}  (O(N^{exponent:.2f}))")
        print(f"  Expected for BFS: ~1.0-1.5  |  Spectral would be: ~2.5-3.0")
    print()
    return times, scales


def test_partition_quality():
    """Measure balance and boundary edge quality."""
    print("--- Test 2: Partition Quality ---")
    torch.manual_seed(42)
    graph = make_random_graph(500, avg_degree=8)
    partitioner = GraphPartitioner(max_partition_size=32)
    partitions = partitioner.partition(graph)

    sizes = [len(p) for p in partitions]
    mean_size = sum(sizes) / len(sizes)
    std_size = (sum((s - mean_size) ** 2 for s in sizes) / len(sizes)) ** 0.5
    balance_ratio = min(sizes) / max(sizes) if max(sizes) > 0 else 1.0

    # Count boundary edges
    node_to_part = torch.zeros(graph.num_nodes, dtype=torch.long)
    for pid, members in enumerate(partitions):
        node_to_part[members] = pid

    src, tgt = graph.edge_index
    cross_boundary = (node_to_part[src] != node_to_part[tgt]).sum().item()
    total_edges = graph.num_edges
    boundary_ratio = cross_boundary / total_edges if total_edges > 0 else 0

    print(f"  Partitions: {len(partitions)}")
    print(f"  Size stats: mean={mean_size:.1f}  std={std_size:.1f}  "
          f"min={min(sizes)}  max={max(sizes)}")
    print(f"  Balance ratio (min/max): {balance_ratio:.3f}")
    print(f"  Boundary edges: {cross_boundary}/{total_edges} ({boundary_ratio:.1%})")
    print()
    return balance_ratio, boundary_ratio


def test_importance_seeds():
    """Verify importance-based seed selection works."""
    print("--- Test 3: Importance-Based Seeding ---")
    torch.manual_seed(42)
    graph = make_random_graph(200, avg_degree=6)
    partitioner = GraphPartitioner(max_partition_size=32)

    # Create importance: first 20 nodes are "important"
    importance = torch.zeros(200)
    importance[:20] = 10.0

    partitions = partitioner.partition(graph, importance=importance)

    # Check: important nodes should be spread across partitions (as seeds)
    important_nodes = set(range(20))
    parts_with_important = 0
    for pid, members in enumerate(partitions):
        members_set = set(members.tolist())
        if members_set & important_nodes:
            parts_with_important += 1

    spread = parts_with_important / len(partitions)
    print(f"  Partitions with important nodes: {parts_with_important}/{len(partitions)}")
    print(f"  Spread ratio: {spread:.2f}  (higher = better coverage)")
    print()
    return spread


def main():
    print("=" * 70)
    print("PHASE 20: BFS Partition Scaling Benchmark")
    print("=" * 70)
    print()
    print("Fix 2 validation: BFS seed-expansion vs spectral complexity")
    print()

    times, scales = test_scaling()
    balance, boundary = test_partition_quality()
    spread = test_importance_seeds()

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Scaling: {times[0]*1000:.1f}ms (N=50) → {times[-1]*1000:.1f}ms (N=2500)")
    print(f"  Balance ratio: {balance:.3f}  (1.0 = perfect)")
    print(f"  Boundary ratio: {boundary:.1%}  (lower = better locality)")
    print(f"  Importance spread: {spread:.2f}")
    print()

    # Key check: partition at N=2500 should complete in <1s
    if times[-1] < 1.0:
        print("  >> BFS scaling: PASS — 2500-node partition in <1s")
    else:
        print("  >> BFS scaling: WARN — 2500-node partition took >1s")


if __name__ == '__main__':
    main()
