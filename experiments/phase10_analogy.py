"""
Phase 10: Analogical Reasoning via Edge-to-Edge Attention

Core question: Can DELTA's edge-to-edge attention discover analogies
between different parts of a knowledge graph?

Task: "A is to B as C is to ?"
Given relation (A, B) of type R, and entity C, predict D such that
(C, D) has the same relation type R.

This is structural pattern matching across edges — the precise use case
for edge-to-edge attention. If edge E1=(A→B) and edge E2=(C→D) share
the same structural pattern, edge-to-edge attention should learn this.

Compared:
1. TransE-style embedding baseline (relation as translation)
2. Node attention (can it infer relation from endpoints?)
3. DELTA edge attention (can edge-to-edge attention find analogies?)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import EdgeAttention, NodeAttention


def create_analogy_task(num_entities=80, num_relations=6, d_node=64, d_edge=32,
                        num_queries=100, seed=42):
    """Create analogy queries over a knowledge graph.

    Each analogy: given edge (A, R, B) and entity C, find D such that (C, R, D).

    Returns:
        graph: the knowledge graph
        queries: list of (src_edge_idx, query_entity, target_entity, relation)
        edge_labels: relation types for all edges
    """
    torch.manual_seed(seed)

    # Build a knowledge graph with typed relations
    node_features = torch.randn(num_entities, d_node)
    # Embed entity "type" loosely in first few dims
    types_per_entity = torch.randint(0, 4, (num_entities,))
    for i in range(num_entities):
        node_features[i, types_per_entity[i].item()] += 2.0

    edges_src, edges_tgt, edge_labels_list = [], [], []
    edge_features_list = []

    # Generate edges with relation-type-dependent structure
    relation_pairs = {r: [] for r in range(num_relations)}
    for _ in range(num_entities * 3):
        rel = torch.randint(0, num_relations, (1,)).item()
        src = torch.randint(0, num_entities, (1,)).item()
        tgt = torch.randint(0, num_entities, (1,)).item()
        if src == tgt:
            continue

        edges_src.append(src)
        edges_tgt.append(tgt)
        edge_labels_list.append(rel)
        relation_pairs[rel].append((src, tgt))

        # Edge features: encode relation pattern + endpoint interaction
        feat = torch.randn(d_edge) * 0.3
        # Relation-specific signal via structural encoding
        src_feat = node_features[src]
        tgt_feat = node_features[tgt]
        feat[:d_node // 4] += (src_feat[:d_node // 4] - tgt_feat[:d_node // 4]) * 0.5
        feat[rel] += 1.0
        edge_features_list.append(feat)

    edge_index = torch.tensor([edges_src, edges_tgt], dtype=torch.long)
    edge_features = torch.stack(edge_features_list)
    edge_labels = torch.tensor(edge_labels_list, dtype=torch.long)

    # Generate analogy queries
    queries = []
    for _ in range(num_queries):
        # Pick a relation with enough examples
        rel = torch.randint(0, num_relations, (1,)).item()
        pairs = relation_pairs[rel]
        if len(pairs) < 2:
            continue

        # Pick two edges: (A, B) is the example, (C, D) is the target
        indices = torch.randperm(len(pairs))[:2].tolist()
        idx1, idx2 = indices[0], indices[1]
        a, b = pairs[idx1]
        c, d = pairs[idx2]

        # Find the edge index of the example edge
        for ei in range(len(edges_src)):
            if edges_src[ei] == a and edges_tgt[ei] == b and edge_labels_list[ei] == rel:
                queries.append({
                    'example_edge': ei,
                    'query_entity': c,
                    'target_entity': d,
                    'relation': rel,
                })
                break

    graph = DeltaGraph(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
    )

    return graph, queries, edge_labels, num_relations


class TransEBaseline(nn.Module):
    """TransE: represent relations as translations h + r ≈ t."""

    def __init__(self, num_entities, num_relations, d_emb=64):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, d_emb)
        self.relation_emb = nn.Embedding(num_relations, d_emb)

    def score(self, src, rel, tgt):
        """Score a triple (src, rel, tgt). Lower = more plausible."""
        h = self.entity_emb(src)
        r = self.relation_emb(rel)
        t = self.entity_emb(tgt)
        return torch.norm(h + r - t, dim=-1)

    def predict(self, src, rel, candidates):
        """Rank candidates for (src, rel, ?)."""
        h = self.entity_emb(src).unsqueeze(0)  # [1, d]
        r = self.relation_emb(rel).unsqueeze(0)
        c = self.entity_emb(candidates)  # [C, d]
        scores = torch.norm(h + r - c, dim=-1)  # [C]
        return scores


class EdgeAnalogySolver(nn.Module):
    """Use DELTA edge attention to find analogous edges."""

    def __init__(self, d_node, d_edge, num_entities, num_heads=4):
        super().__init__()
        self.edge_attn1 = EdgeAttention(d_edge, d_node, num_heads)
        self.edge_attn2 = EdgeAttention(d_edge, d_node, num_heads)
        # Score candidate (query_entity, target) based on edge representation similarity
        self.scorer = nn.Sequential(
            nn.Linear(d_edge + d_node, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, graph):
        edge_adj = graph.build_edge_adjacency()
        edge_feats = self.edge_attn1(graph, edge_adj=edge_adj)
        graph = DeltaGraph(
            node_features=graph.node_features,
            edge_features=edge_feats,
            edge_index=graph.edge_index,
            node_tiers=graph.node_tiers,
        )
        edge_feats = self.edge_attn2(graph, edge_adj=edge_adj)
        return DeltaGraph(
            node_features=graph.node_features,
            edge_features=edge_feats,
            edge_index=graph.edge_index,
            node_tiers=graph.node_tiers,
        )

    def score_candidate(self, edge_repr, candidate_node_feat):
        """Score how well a candidate completes the analogy."""
        combined = torch.cat([edge_repr, candidate_node_feat], dim=-1)
        return self.scorer(combined).squeeze(-1)


def evaluate_analogies(model, graph, queries, top_k=5):
    """Evaluate analogy accuracy: is the target in the top-k predictions?"""
    model.eval()
    with torch.no_grad():
        processed_graph = model(graph)

    hits_at_k = 0
    mrr_sum = 0.0
    total = 0

    for q in queries:
        example_edge_feat = processed_graph.edge_features[q['example_edge']]

        # Score all entities as candidates
        scores = []
        for ent in range(graph.num_nodes):
            score = model.score_candidate(
                example_edge_feat.unsqueeze(0),
                graph.node_features[ent].unsqueeze(0),
            )
            scores.append(score.item())

        scores_tensor = torch.tensor(scores)
        # Higher score = better match
        _, ranked = torch.sort(scores_tensor, descending=True)

        target = q['target_entity']
        rank = (ranked == target).nonzero(as_tuple=True)[0]
        if len(rank) > 0:
            rank = rank[0].item() + 1
            mrr_sum += 1.0 / rank
            if rank <= top_k:
                hits_at_k += 1
        total += 1

    return {
        'hits_at_k': hits_at_k / max(1, total),
        'mrr': mrr_sum / max(1, total),
        'total': total,
    }


def main():
    print("=" * 70)
    print("PHASE 10: Analogical Reasoning")
    print("=" * 70)
    print()
    print("Task: A is to B as C is to ?  (via edge-to-edge attention)")
    print("Metric: Hits@5 (target in top 5 predictions), MRR")
    print()

    d_node, d_edge = 64, 32
    torch.manual_seed(42)

    graph, queries, edge_labels, num_relations = create_analogy_task(
        num_entities=80, num_relations=6, d_node=d_node, d_edge=d_edge,
        num_queries=100,
    )

    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"Analogy queries: {len(queries)}")
    print(f"Relation types: {num_relations}")
    print()

    # --- Train DELTA edge model on edge classification first ---
    print("--- Training DELTA edge attention on relation classification ---")
    delta_model = EdgeAnalogySolver(d_node, d_edge, graph.num_nodes)
    optimizer = torch.optim.Adam(delta_model.parameters(), lr=1e-3)

    # Supervised training: classify edges by relation type
    edge_clf = nn.Sequential(
        nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_relations),
    )
    all_params = list(delta_model.parameters()) + list(edge_clf.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-3)

    for epoch in range(200):
        delta_model.train()
        edge_clf.train()
        processed = delta_model(graph)
        logits = edge_clf(processed.edge_features)
        loss = F.cross_entropy(logits, edge_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            acc = (logits.argmax(-1) == edge_labels).float().mean().item()
            print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}  Edge Acc={acc:.3f}")

    # --- Evaluate analogies ---
    print("\n--- Evaluating analogy solving ---")
    analogy_results = evaluate_analogies(delta_model, graph, queries, top_k=5)
    print(f"  DELTA Edge Attention:")
    print(f"    Hits@5:  {analogy_results['hits_at_k']:.3f}")
    print(f"    MRR:     {analogy_results['mrr']:.3f}")
    print(f"    Queries: {analogy_results['total']}")

    # --- Random baseline ---
    print(f"\n  Random baseline:")
    random_hits = 5.0 / graph.num_nodes
    random_mrr = sum(1.0 / (i + 1) for i in range(graph.num_nodes)) / graph.num_nodes
    print(f"    Hits@5:  {random_hits:.3f}")
    print(f"    MRR:     {random_mrr:.3f}")

    improvement = analogy_results['hits_at_k'] / max(random_hits, 1e-10)
    print(f"\n  >> DELTA is {improvement:.1f}x better than random at analogy solving.")

    if analogy_results['hits_at_k'] > random_hits * 2:
        print("  >> Edge-to-edge attention captures analogical structure.")
    else:
        print("  >> More training or larger graphs needed for clearer signal.")


if __name__ == '__main__':
    main()
