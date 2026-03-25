"""
Phase 11: Multi-Hop Edge Adjacency

Core question: Does 2-hop edge adjacency let DELTA reason about
transitive/composed relations that 1-hop misses?

Phase 9 showed Node GNN beat DELTA on derived relations (livesIn, colleague)
because those require composing paths (worksAt + locatedIn → livesIn).
1-hop edge adjacency can't see across the intermediate node.
2-hop edge adjacency connects edges that share an intermediate edge,
enabling compositional relational reasoning.

Compared:
1. DELTA with 1-hop edge adjacency (Phase 9 baseline)
2. DELTA with 2-hop edge adjacency
3. Node GNN baseline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import EdgeAttention, NodeAttention


def create_multi_hop_task(num_entities=60, d_node=64, d_edge=32, seed=42):
    """Knowledge graph with base + derived (transitive) relations."""
    torch.manual_seed(seed)

    REL_WORKS_AT = 0
    REL_LOCATED_IN = 1
    REL_FRIEND_OF = 2
    REL_FIELD_OF = 3
    REL_LIVES_IN = 4     # derived: worksAt + locatedIn
    REL_COLLEAGUE = 5    # derived: worksAt + worksAt^-1
    num_classes = 6

    n_persons = num_entities // 3
    n_orgs = num_entities // 3
    n_places = num_entities - n_persons - n_orgs

    node_features = torch.randn(num_entities, d_node) * 0.5
    node_features[:n_persons, 0] = 1.0
    node_features[n_persons:n_persons+n_orgs, 1] = 1.0
    node_features[n_persons+n_orgs:, 2] = 1.0

    edges_src, edges_tgt, edge_labels = [], [], []
    edge_features_list = []

    def add_edge(src, tgt, rel):
        edges_src.append(src)
        edges_tgt.append(tgt)
        edge_labels.append(rel)
        feat = torch.randn(d_edge) * 0.3
        feat[rel] += 1.5
        edge_features_list.append(feat)

    person_org_map = {}
    org_place_map = {}

    for p in range(n_persons):
        org = n_persons + torch.randint(0, n_orgs, (1,)).item()
        add_edge(p, org, REL_WORKS_AT)
        person_org_map[p] = org

    for o in range(n_orgs):
        org_id = n_persons + o
        place = n_persons + n_orgs + torch.randint(0, n_places, (1,)).item()
        add_edge(org_id, place, REL_LOCATED_IN)
        org_place_map[org_id] = place

    for _ in range(n_persons):
        p1 = torch.randint(0, n_persons, (1,)).item()
        p2 = torch.randint(0, n_persons, (1,)).item()
        if p1 != p2:
            add_edge(p1, p2, REL_FRIEND_OF)

    for o in range(min(n_orgs, n_places)):
        org_id = n_persons + o
        field = n_persons + n_orgs + (o % n_places)
        add_edge(org_id, field, REL_FIELD_OF)

    # Derived
    for person, org in person_org_map.items():
        if org in org_place_map:
            add_edge(person, org_place_map[org], REL_LIVES_IN)

    org_to_persons = {}
    for person, org in person_org_map.items():
        org_to_persons.setdefault(org, []).append(person)
    for org, persons in org_to_persons.items():
        for i in range(len(persons)):
            for j in range(i + 1, min(i + 3, len(persons))):
                add_edge(persons[i], persons[j], REL_COLLEAGUE)

    edge_index = torch.tensor([edges_src, edges_tgt], dtype=torch.long)
    edge_features = torch.stack(edge_features_list)
    labels = torch.tensor(edge_labels, dtype=torch.long)

    graph = DeltaGraph(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
    )

    rel_names = ['worksAt', 'locatedIn', 'friendOf', 'fieldOf', 'livesIn', 'colleague']
    return graph, labels, num_classes, rel_names


class DeltaEdgeModel(nn.Module):
    def __init__(self, d_node, d_edge, num_classes, num_heads=4, hops=1):
        super().__init__()
        self.hops = hops
        self.edge_attn1 = EdgeAttention(d_edge, d_node, num_heads)
        self.edge_attn2 = EdgeAttention(d_edge, d_node, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph):
        edge_adj = graph.build_edge_adjacency(hops=self.hops)
        edge_feats = self.edge_attn1(graph, edge_adj=edge_adj)
        graph2 = DeltaGraph(
            node_features=graph.node_features,
            edge_features=edge_feats,
            edge_index=graph.edge_index,
            node_tiers=graph.node_tiers,
        )
        edge_feats = self.edge_attn2(graph2, edge_adj=edge_adj)
        return self.classifier(edge_feats)


class NodeGNNModel(nn.Module):
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.attn1 = NodeAttention(d_node, d_edge, num_heads)
        self.attn2 = NodeAttention(d_node, d_edge, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph):
        node_feats = self.attn1(graph)
        g2 = DeltaGraph(node_features=node_feats, edge_features=graph.edge_features,
                        edge_index=graph.edge_index, node_tiers=graph.node_tiers)
        node_feats = self.attn2(g2)
        return self.classifier(graph.edge_features)


def train_eval(model, graph, labels, train_mask, test_mask, epochs=250, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    best_per_class = {}

    for epoch in range(epochs):
        model.train()
        logits = model(graph)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(graph)
                preds = logits[test_mask].argmax(-1)
                tgt = labels[test_mask]
                acc = (preds == tgt).float().mean().item()
                if acc >= best_acc:
                    best_acc = acc
                    for c in range(labels.max().item() + 1):
                        cm = tgt == c
                        if cm.sum() > 0:
                            best_per_class[c] = (preds[cm] == c).float().mean().item()
                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}  Acc={acc:.3f}")

    return best_acc, best_per_class


def main():
    print("=" * 70)
    print("PHASE 11: Multi-Hop Edge Adjacency")
    print("=" * 70)
    print()
    print("Question: Does 2-hop edge adjacency help compose transitive relations?")
    print()

    d_node, d_edge = 64, 32
    torch.manual_seed(42)

    graph, labels, num_classes, rel_names = create_multi_hop_task(
        num_entities=60, d_node=d_node, d_edge=d_edge,
    )

    n_edges = graph.num_edges
    perm = torch.randperm(n_edges)
    train_end = int(n_edges * 0.7)
    train_mask = torch.zeros(n_edges, dtype=torch.bool)
    train_mask[perm[:train_end]] = True
    test_mask = ~train_mask

    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    adj1 = graph.build_edge_adjacency(hops=1)
    adj2 = graph.build_edge_adjacency(hops=2)
    print(f"1-hop edge pairs: {adj1.shape[1]}, 2-hop edge pairs: {adj2.shape[1]}")
    print()

    results = {}

    print("--- Node GNN Baseline ---")
    torch.manual_seed(42)
    m = NodeGNNModel(d_node, d_edge, num_classes)
    acc, pc = train_eval(m, graph, labels, train_mask, test_mask)
    results['Node GNN'] = (acc, pc)

    print("\n--- DELTA Edge (1-hop) ---")
    torch.manual_seed(42)
    m = DeltaEdgeModel(d_node, d_edge, num_classes, hops=1)
    acc, pc = train_eval(m, graph, labels, train_mask, test_mask)
    results['DELTA 1-hop'] = (acc, pc)

    print("\n--- DELTA Edge (2-hop) ---")
    torch.manual_seed(42)
    m = DeltaEdgeModel(d_node, d_edge, num_classes, hops=2)
    acc, pc = train_eval(m, graph, labels, train_mask, test_mask)
    results['DELTA 2-hop'] = (acc, pc)

    # Summary
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

    # Derived relation comparison
    print(f"\n  Derived relations (livesIn + colleague):")
    for model_name, (acc, pc) in results.items():
        derived = (pc.get(4, 0) + pc.get(5, 0)) / 2
        print(f"    {model_name:20s} {derived:.3f}")

    d1 = (results['DELTA 1-hop'][1].get(4, 0) + results['DELTA 1-hop'][1].get(5, 0)) / 2
    d2 = (results['DELTA 2-hop'][1].get(4, 0) + results['DELTA 2-hop'][1].get(5, 0)) / 2
    print(f"\n  2-hop vs 1-hop on derived relations: {d2 - d1:+.3f}")
    if d2 > d1:
        print("  >> Multi-hop edge adjacency improves transitive reasoning.")
    else:
        print("  >> Multi-hop didn't help — may need deeper attention or path encoding.")


if __name__ == '__main__':
    main()
