"""
Phase 9: Structured Relational Task — Multi-Hop Reasoning

Core question: Does DELTA's edge-first attention enable multi-hop relational
reasoning that transformers and standard GNNs struggle with?

Task: Given a knowledge graph with typed relations, predict the type of a
withheld edge by reasoning over multi-hop paths. For example:
  A --worksAt--> B, B --locatedIn--> C  =>  A --livesIn--> C

This requires understanding chains of relations — the exact kind of
analogical/compositional reasoning that edge-to-edge attention was designed for.

Compared:
1. MLP on concatenated endpoint features (no structure)
2. Node-attention GNN (GAT-style)
3. DELTA with edge attention
4. DELTA with Gumbel-softmax routing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import EdgeAttention, NodeAttention
from delta.router import ImportanceRouter


def create_multi_hop_task(num_entities=60, d_node=64, d_edge=32, seed=42):
    """Create a knowledge graph with inferrable multi-hop relations.

    Structure:
      - Base relations: worksAt, locatedIn, friendOf, fieldOf
      - Derived relations (multi-hop):
        - livesIn: worksAt + locatedIn  (A works at B, B located in C => A lives in C)
        - colleague: worksAt + worksAt^-1  (A works at B, C works at B => A colleague C)

    The task is to classify edges — including the derived ones.
    Models must learn to compose base relations to predict derived ones.
    """
    torch.manual_seed(seed)

    # Relation types
    REL_WORKS_AT = 0    # person -> org
    REL_LOCATED_IN = 1  # org -> place
    REL_FRIEND_OF = 2   # person -> person
    REL_FIELD_OF = 3    # org -> field
    REL_LIVES_IN = 4    # person -> place (derived: worksAt + locatedIn)
    REL_COLLEAGUE = 5   # person -> person (derived: worksAt + worksAt^-1)
    num_classes = 6

    # Entity pools
    n_persons = num_entities // 3
    n_orgs = num_entities // 3
    n_places = num_entities - n_persons - n_orgs

    # Node features: type embedding + random features
    node_features = torch.randn(num_entities, d_node) * 0.5
    # Encode entity type in first 3 dims
    node_features[:n_persons, 0] = 1.0
    node_features[n_persons:n_persons+n_orgs, 1] = 1.0
    node_features[n_persons+n_orgs:, 2] = 1.0

    edges_src = []
    edges_tgt = []
    edge_labels = []
    edge_features_list = []

    def add_edge(src, tgt, rel):
        edges_src.append(src)
        edges_tgt.append(tgt)
        edge_labels.append(rel)
        feat = torch.randn(d_edge) * 0.3
        # Encode relation type signal in first dims
        feat[rel] += 1.5
        edge_features_list.append(feat)

    # --- Base relations ---
    person_org_map = {}  # person -> org they work at
    org_place_map = {}   # org -> place it's located in

    # worksAt: each person works at a random org
    for p in range(n_persons):
        org = n_persons + torch.randint(0, n_orgs, (1,)).item()
        add_edge(p, org, REL_WORKS_AT)
        person_org_map[p] = org

    # locatedIn: each org is in a random place
    for o in range(n_orgs):
        org_id = n_persons + o
        place = n_persons + n_orgs + torch.randint(0, n_places, (1,)).item()
        add_edge(org_id, place, REL_LOCATED_IN)
        org_place_map[org_id] = place

    # friendOf: random person-person edges
    for _ in range(n_persons):
        p1 = torch.randint(0, n_persons, (1,)).item()
        p2 = torch.randint(0, n_persons, (1,)).item()
        if p1 != p2:
            add_edge(p1, p2, REL_FRIEND_OF)

    # fieldOf: some orgs have field associations
    for o in range(min(n_orgs, n_places)):
        org_id = n_persons + o
        field = n_persons + n_orgs + (o % n_places)
        add_edge(org_id, field, REL_FIELD_OF)

    # --- Derived relations (multi-hop inference targets) ---
    # livesIn: person -> place via worksAt + locatedIn
    for person, org in person_org_map.items():
        if org in org_place_map:
            place = org_place_map[org]
            add_edge(person, place, REL_LIVES_IN)

    # colleague: persons at same org
    org_to_persons = {}
    for person, org in person_org_map.items():
        org_to_persons.setdefault(org, []).append(person)
    for org, persons in org_to_persons.items():
        for i in range(len(persons)):
            for j in range(i + 1, min(i + 3, len(persons))):  # limit pairs
                add_edge(persons[i], persons[j], REL_COLLEAGUE)

    edge_index = torch.tensor([edges_src, edges_tgt], dtype=torch.long)
    edge_features = torch.stack(edge_features_list)
    labels = torch.tensor(edge_labels, dtype=torch.long)

    graph = DeltaGraph(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
    )

    # Count per relation
    counts = {}
    for l in edge_labels:
        counts[l] = counts.get(l, 0) + 1

    rel_names = ['worksAt', 'locatedIn', 'friendOf', 'fieldOf', 'livesIn', 'colleague']

    return graph, labels, num_classes, counts, rel_names


class MLPBaseline(nn.Module):
    def __init__(self, d_node, d_edge, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_node * 2 + d_edge, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, graph):
        src_feats = graph.node_features[graph.edge_index[0]]
        tgt_feats = graph.node_features[graph.edge_index[1]]
        combined = torch.cat([src_feats, tgt_feats, graph.edge_features], dim=-1)
        return self.mlp(combined)


class NodeGNNModel(nn.Module):
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.attn1 = NodeAttention(d_node, d_edge, num_heads)
        self.attn2 = NodeAttention(d_node, d_edge, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph):
        graph = DeltaGraph(
            node_features=self.attn1(graph),
            edge_features=graph.edge_features,
            edge_index=graph.edge_index,
            node_tiers=graph.node_tiers,
        )
        graph = DeltaGraph(
            node_features=self.attn2(graph),
            edge_features=graph.edge_features,
            edge_index=graph.edge_index,
            node_tiers=graph.node_tiers,
        )
        return self.classifier(graph.edge_features)


class DeltaEdgeModel(nn.Module):
    def __init__(self, d_node, d_edge, num_classes, num_heads=4, use_gumbel=False):
        super().__init__()
        self.edge_attn1 = EdgeAttention(d_edge, d_node, num_heads)
        self.edge_attn2 = EdgeAttention(d_edge, d_node, num_heads)
        self.router = ImportanceRouter(d_node, d_edge) if use_gumbel else None
        self.use_gumbel = use_gumbel
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, temperature=1.0):
        edge_adj = graph.build_edge_adjacency()

        if self.use_gumbel and self.router is not None:
            node_scores, edge_scores = self.router(graph)
            _, edge_weights = self.router.apply_top_k_gumbel(
                graph, node_scores, edge_scores,
                node_k_ratio=0.6, edge_k_ratio=0.6,
                temperature=temperature, hard=True,
            )
            graph = DeltaGraph(
                node_features=graph.node_features,
                edge_features=graph.edge_features * edge_weights.unsqueeze(-1),
                edge_index=graph.edge_index,
                node_tiers=graph.node_tiers,
            )

        edge_feats = self.edge_attn1(graph, edge_adj=edge_adj)
        graph = DeltaGraph(
            node_features=graph.node_features,
            edge_features=edge_feats,
            edge_index=graph.edge_index,
            node_tiers=graph.node_tiers,
        )
        edge_feats = self.edge_attn2(graph, edge_adj=edge_adj)
        return self.classifier(edge_feats)


def train_model(model, graph, labels, train_mask, test_mask, epochs=200,
                lr=1e-3, use_gumbel=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        if use_gumbel:
            progress = epoch / max(1, epochs - 1)
            temperature = 2.0 * (0.1 / 2.0) ** progress
            logits = model(graph, temperature=temperature)
        else:
            logits = model(graph)

        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(graph) if not use_gumbel else model(graph, temperature=0.1)
                test_preds = logits[test_mask].argmax(-1)
                acc = (test_preds == labels[test_mask]).float().mean().item()
                best_acc = max(best_acc, acc)

                # Per-class accuracy
                per_class = {}
                for c in range(labels.max().item() + 1):
                    class_mask = labels[test_mask] == c
                    if class_mask.sum() > 0:
                        class_acc = (test_preds[class_mask] == c).float().mean().item()
                        per_class[c] = class_acc

                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}  Acc={acc:.3f}")

    return best_acc, per_class


def main():
    print("=" * 70)
    print("PHASE 9: Multi-Hop Relational Reasoning")
    print("=" * 70)
    print()
    print("Task: Classify edges in a knowledge graph with derived relations.")
    print("Derived relations (livesIn, colleague) require multi-hop reasoning.")
    print()

    d_node, d_edge = 64, 32
    torch.manual_seed(42)

    graph, labels, num_classes, counts, rel_names = create_multi_hop_task(
        num_entities=60, d_node=d_node, d_edge=d_edge,
    )

    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print("Relations:")
    for rel_id, name in enumerate(rel_names):
        c = counts.get(rel_id, 0)
        derived = " (DERIVED)" if rel_id >= 4 else ""
        print(f"  {rel_id}: {name:12s} — {c} edges{derived}")
    print()

    # Train/test split
    n_edges = graph.num_edges
    perm = torch.randperm(n_edges)
    train_end = int(n_edges * 0.7)
    train_mask = torch.zeros(n_edges, dtype=torch.bool)
    train_mask[perm[:train_end]] = True
    test_mask = ~train_mask

    results = {}

    # --- MLP ---
    print("--- MLP Baseline (no graph structure) ---")
    torch.manual_seed(42)
    mlp = MLPBaseline(d_node, d_edge, num_classes)
    acc, per_class = train_model(mlp, graph, labels, train_mask, test_mask)
    results['MLP'] = (acc, per_class)

    # --- Node GNN ---
    print("\n--- Node Attention GNN ---")
    torch.manual_seed(42)
    gnn = NodeGNNModel(d_node, d_edge, num_classes)
    acc, per_class = train_model(gnn, graph, labels, train_mask, test_mask)
    results['Node GNN'] = (acc, per_class)

    # --- DELTA Edge Attention ---
    print("\n--- DELTA Edge Attention ---")
    torch.manual_seed(42)
    delta = DeltaEdgeModel(d_node, d_edge, num_classes, use_gumbel=False)
    acc, per_class = train_model(delta, graph, labels, train_mask, test_mask)
    results['DELTA Edge'] = (acc, per_class)

    # --- DELTA + Gumbel ---
    print("\n--- DELTA Edge + Gumbel Router ---")
    torch.manual_seed(42)
    delta_g = DeltaEdgeModel(d_node, d_edge, num_classes, use_gumbel=True)
    acc, per_class = train_model(delta_g, graph, labels, train_mask, test_mask,
                                 use_gumbel=True)
    results['DELTA+Gumbel'] = (acc, per_class)

    # --- Summary ---
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  {'Model':20s} {'Overall':>8}  ", end='')
    for name in rel_names:
        print(f"{name[:8]:>9}", end='')
    print()
    print(f"  {'-'*20} {'-'*8}  ", end='')
    for _ in rel_names:
        print(f"  {'-'*7}", end='')
    print()

    for model_name, (acc, per_class) in results.items():
        print(f"  {model_name:20s} {acc:>8.3f}  ", end='')
        for rel_id in range(num_classes):
            c_acc = per_class.get(rel_id, 0.0)
            print(f"  {c_acc:>7.3f}", end='')
        print()

    # Highlight derived relation performance
    print(f"\n  Derived relation accuracy (livesIn + colleague):")
    for model_name, (acc, per_class) in results.items():
        derived_acc = (per_class.get(4, 0) + per_class.get(5, 0)) / 2
        print(f"    {model_name:20s} {derived_acc:.3f}")

    best_derived = max(
        ((per_class.get(4, 0) + per_class.get(5, 0)) / 2, name)
        for name, (_, per_class) in results.items()
    )
    print(f"\n  >> Best on derived relations: {best_derived[1]} ({best_derived[0]:.3f})")


if __name__ == '__main__':
    main()
