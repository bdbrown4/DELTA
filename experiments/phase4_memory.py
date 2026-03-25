"""
Phase 4: Validate Tiered Memory

Core question: Can DELTA's tiered memory (hot/warm/cold) maintain accuracy
on tasks requiring long-range recall while reducing active computation?

The graph at rest IS the memory. Important nodes stay hot (full attention),
less important compress to warm (sparse attention), irrelevant archive to
cold (retrieval only). The router decides what belongs where.

Test: Sequential task where early "facts" must be recalled later.
Compare: all-hot (everything in memory) vs tiered (router manages tiers).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import DualParallelAttention
from delta.router import ImportanceRouter
from delta.memory import TieredMemory
from delta.utils import create_sequential_memory_task


class MemoryModel(nn.Module):
    """DELTA with tiered memory management."""
    def __init__(self, d_node, d_edge, num_heads=4):
        super().__init__()
        self.router = ImportanceRouter(d_node, d_edge)
        self.memory = TieredMemory(d_node, d_edge)
        self.dual_attn = DualParallelAttention(d_node, d_edge, num_heads)
        # Recall head: given a query node, predict which fact node it should retrieve
        self.recall_head = nn.Sequential(
            nn.Linear(d_node * 2, d_node),
            nn.GELU(),
            nn.Linear(d_node, 1),
        )

    def forward(self, graph, use_tiers=True):
        # Route
        node_scores, edge_scores = self.router(graph)
        graph.node_importance = node_scores
        graph.edge_importance = edge_scores

        if use_tiers:
            new_tiers = self.router.update_tiers(graph, node_scores)
            graph.node_tiers = new_tiers
            graph = self.memory.compress_warm_nodes(graph)

        # Attention
        edge_adj = graph.build_edge_adjacency()
        graph = self.dual_attn(graph, edge_adj=edge_adj)
        return graph

    def score_recall(self, graph, query_idx, candidate_indices):
        """Score how well query nodes match candidate fact nodes."""
        query_feats = graph.node_features[query_idx]  # [Q, d_node]
        scores = []
        for cand in candidate_indices:
            cand_feat = graph.node_features[cand].unsqueeze(0).expand_as(query_feats)
            pair = torch.cat([query_feats, cand_feat], dim=-1)
            s = self.recall_head(pair).squeeze(-1)
            scores.append(s)
        return torch.stack(scores, dim=-1)  # [Q, num_candidates]


def main():
    print("=" * 70)
    print("PHASE 4: Tiered Memory Validation")
    print("=" * 70)
    print()
    print("Task: Recall facts from earlier in a sequence.")
    print("Question: Does tiered memory maintain recall while reducing active nodes?")
    print()

    d_node, d_edge = 64, 32
    seq_length = 40
    num_facts = 6
    num_recall = 5

    graph, recall_tasks = create_sequential_memory_task(
        seq_length=seq_length, d_node=d_node, d_edge=d_edge,
        num_facts=num_facts, recall_positions=num_recall, seed=42,
    )
    print(f"Sequence length: {seq_length}")
    print(f"Facts to remember: {num_facts}")
    print(f"Recall queries: {len(recall_tasks)}")
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges\n")

    query_nodes = torch.tensor([q for q, _ in recall_tasks])
    answer_nodes = [a for _, a in recall_tasks]
    # All fact positions as candidates
    fact_positions = list(set(answer_nodes))

    # Create labels: for each query, which fact index is correct
    labels = torch.tensor([fact_positions.index(a) for _, a in recall_tasks])

    # --- All-hot baseline (no memory management) ---
    print("--- All-Hot Baseline (no tiers) ---")
    model_full = MemoryModel(d_node, d_edge)
    optimizer = torch.optim.Adam(model_full.parameters(), lr=1e-3)

    for epoch in range(300):
        model_full.train()
        updated = model_full(graph, use_tiers=False)
        logits = model_full.score_recall(updated, query_nodes, fact_positions)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            model_full.eval()
            with torch.no_grad():
                updated = model_full(graph, use_tiers=False)
                logits = model_full.score_recall(updated, query_nodes, fact_positions)
                acc = (logits.argmax(-1) == labels).float().mean()
                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}  Recall Acc={acc.item():.3f}")

    model_full.eval()
    with torch.no_grad():
        updated = model_full(graph, use_tiers=False)
        logits = model_full.score_recall(updated, query_nodes, fact_positions)
        full_acc = (logits.argmax(-1) == labels).float().mean().item()
        hot_count = graph.num_nodes  # all nodes are hot
    print(f"  Final: Acc={full_acc:.3f}, Active nodes={hot_count}\n")

    # --- Tiered memory ---
    print("--- Tiered Memory (router manages hot/warm/cold) ---")
    model_tiered = MemoryModel(d_node, d_edge)
    optimizer = torch.optim.Adam(model_tiered.parameters(), lr=1e-3)

    for epoch in range(300):
        model_tiered.train()
        updated = model_tiered(graph, use_tiers=True)
        logits = model_tiered.score_recall(updated, query_nodes, fact_positions)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            model_tiered.eval()
            with torch.no_grad():
                updated = model_tiered(graph, use_tiers=True)
                logits = model_tiered.score_recall(updated, query_nodes, fact_positions)
                acc = (logits.argmax(-1) == labels).float().mean()
                node_scores, _ = model_tiered.router(graph)
                tiers = model_tiered.router.update_tiers(graph, node_scores)
                hot = (tiers == 0).sum().item()
                warm = (tiers == 1).sum().item()
                cold = (tiers == 2).sum().item()
                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}  Recall Acc={acc.item():.3f}  "
                      f"Hot={hot} Warm={warm} Cold={cold}")

    model_tiered.eval()
    with torch.no_grad():
        updated = model_tiered(graph, use_tiers=True)
        logits = model_tiered.score_recall(updated, query_nodes, fact_positions)
        tiered_acc = (logits.argmax(-1) == labels).float().mean().item()
        node_scores, _ = model_tiered.router(graph)
        tiers = model_tiered.router.update_tiers(graph, node_scores)
        active = ((tiers == 0) | (tiers == 1)).sum().item()

    print(f"  Final: Acc={tiered_acc:.3f}, Active nodes={active}/{graph.num_nodes}\n")

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  All-hot:  Acc={full_acc:.3f}  Active={graph.num_nodes}/{graph.num_nodes}")
    print(f"  Tiered:   Acc={tiered_acc:.3f}  Active={active}/{graph.num_nodes}")
    compute_saved = (1 - active / graph.num_nodes) * 100
    print(f"  Compute saved: {compute_saved:.0f}%")
    print(f"  Accuracy delta: {tiered_acc - full_acc:+.3f}")


if __name__ == '__main__':
    main()
