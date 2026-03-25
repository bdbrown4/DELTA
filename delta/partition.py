"""
Graph Partitioner for DELTA.

Lightweight partitioning for scalable attention. Instead of O(N³) spectral
clustering with eigendecomposition, uses a greedy seed-expansion strategy
that runs in O(N + E) time: pick seed nodes, grow partitions by BFS along
high-importance edges.

Only activates when graphs exceed max_partition_size — at small scale,
single-partition full attention is used (no overhead).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional

from delta.graph import DeltaGraph


class GraphPartitioner(nn.Module):
    """Lightweight graph partitioner using seed-expansion BFS.

    O(N + E) time, O(N) memory — scalable to large graphs.
    Replaces the previous O(N³) spectral clustering approach.
    """

    def __init__(self, max_partition_size: int = 32):
        super().__init__()
        self.max_partition_size = max_partition_size

    def partition(self, graph: DeltaGraph, num_partitions: Optional[int] = None,
                  importance: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Partition graph nodes into balanced clusters via seed-expansion.

        Args:
            graph: the input graph
            num_partitions: explicit partition count (auto-computed if None)
            importance: [N] node importance scores — seeds are highest-importance nodes

        Returns:
            List of [P_i] tensors, each containing node indices for that partition
        """
        N = graph.num_nodes
        if num_partitions is None:
            num_partitions = max(1, N // self.max_partition_size)

        if num_partitions <= 1:
            return [torch.arange(N, device=graph.device)]

        # Build adjacency list (O(E))
        adj_list: list = [[] for _ in range(N)]
        src, tgt = graph.edge_index[0].tolist(), graph.edge_index[1].tolist()
        for s, t in zip(src, tgt):
            adj_list[s].append(t)
            adj_list[t].append(s)

        # Pick seeds: highest-importance nodes, spread apart
        if importance is not None:
            seed_order = torch.argsort(importance, descending=True)
        else:
            # Degree-based fallback
            degree = torch.zeros(N, device=graph.device)
            degree.scatter_add_(0, graph.edge_index[0],
                                torch.ones(graph.num_edges, device=graph.device))
            degree.scatter_add_(0, graph.edge_index[1],
                                torch.ones(graph.num_edges, device=graph.device))
            seed_order = torch.argsort(degree, descending=True)

        # Greedy seed selection: pick k seeds that are spread apart
        seeds = []
        assigned = set()
        for idx in seed_order.tolist():
            if idx not in assigned:
                seeds.append(idx)
                assigned.add(idx)
                # Mark 1-hop neighbors as unavailable for seeding
                for neighbor in adj_list[idx]:
                    assigned.add(neighbor)
            if len(seeds) >= num_partitions:
                break

        # If not enough seeds, fill from remaining nodes
        if len(seeds) < num_partitions:
            for idx in seed_order.tolist():
                if idx not in set(seeds):
                    seeds.append(idx)
                if len(seeds) >= num_partitions:
                    break

        # BFS expansion from seeds — assign each node to nearest seed
        assignment = torch.full((N,), -1, dtype=torch.long, device=graph.device)
        queue = []
        for pid, seed in enumerate(seeds):
            assignment[seed] = pid
            queue.append(seed)

        # Round-robin BFS to keep partitions balanced
        partition_queues: list = [[seed] for seed in seeds]
        max_size = (N + num_partitions - 1) // num_partitions  # ceil division
        partition_sizes = [1] * len(seeds)

        changed = True
        while changed:
            changed = False
            for pid in range(len(seeds)):
                next_queue = []
                for node in partition_queues[pid]:
                    for neighbor in adj_list[node]:
                        if assignment[neighbor].item() == -1 and partition_sizes[pid] < max_size:
                            assignment[neighbor] = pid
                            partition_sizes[pid] += 1
                            next_queue.append(neighbor)
                            changed = True
                partition_queues[pid] = next_queue

        # Assign any remaining unassigned nodes to smallest partition
        unassigned = (assignment == -1).nonzero(as_tuple=True)[0]
        for node in unassigned:
            min_pid = min(range(len(seeds)), key=lambda p: partition_sizes[p])
            assignment[node] = min_pid
            partition_sizes[min_pid] += 1

        # Build partition lists
        partitions = []
        for pid in range(len(seeds)):
            members = (assignment == pid).nonzero(as_tuple=True)[0]
            if len(members) > 0:
                partitions.append(members)

        return partitions if partitions else [torch.arange(N, device=graph.device)]

    def get_boundary_nodes(self, graph: DeltaGraph,
                           partitions: List[torch.Tensor]) -> List[torch.Tensor]:
        """Find nodes at partition boundaries (connected to other partitions)."""
        node_to_partition = torch.full((graph.num_nodes,), -1, dtype=torch.long,
                                       device=graph.device)
        for i, part in enumerate(partitions):
            node_to_partition[part] = i

        src, tgt = graph.edge_index
        src_part = node_to_partition[src]
        tgt_part = node_to_partition[tgt]
        cross_mask = src_part != tgt_part

        boundary_nodes_per_partition = []
        for i in range(len(partitions)):
            part_src_boundary = (src_part == i) & cross_mask
            part_tgt_boundary = (tgt_part == i) & cross_mask
            boundary_idx = torch.unique(torch.cat([
                src[part_src_boundary],
                tgt[part_tgt_boundary]
            ]))
            boundary_nodes_per_partition.append(boundary_idx)

        return boundary_nodes_per_partition
