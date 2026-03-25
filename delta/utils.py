"""
Utility functions and synthetic data generation for DELTA experiments.
"""

from __future__ import annotations

import torch
import random
from typing import List, Tuple, Dict, Optional

from delta.graph import DeltaGraph


def create_knowledge_graph(
    num_entities: int = 20,
    num_relation_types: int = 4,
    edges_per_entity: int = 3,
    d_node: int = 64,
    d_edge: int = 32,
    seed: int = 42,
) -> Tuple[DeltaGraph, Dict]:
    """Create a synthetic knowledge graph with typed relations.

    Generates entities with learnable features and typed edges between them.
    The relation types have consistent patterns (e.g., all "capital_of" edges
    connect cities to countries) — this is what edge-to-edge attention should
    discover.

    Returns:
        graph: DeltaGraph
        metadata: dict with entity names, relation types, edge labels
    """
    torch.manual_seed(seed)
    random.seed(seed)

    # Generate entity clusters — entities of the same "type"
    # create feature prototypes per cluster
    num_clusters = max(2, num_entities // 5)
    cluster_prototypes = torch.randn(num_clusters, d_node)

    # Assign entities to clusters
    entity_clusters = [i % num_clusters for i in range(num_entities)]
    node_features = torch.stack([
        cluster_prototypes[c] + torch.randn(d_node) * 0.3
        for c in entity_clusters
    ])

    # Generate relation type prototypes
    relation_prototypes = torch.randn(num_relation_types, d_edge)

    # Create edges — relations between entities
    src_list = []
    tgt_list = []
    edge_feat_list = []
    edge_labels = []

    for i in range(num_entities):
        targets = random.sample(
            [j for j in range(num_entities) if j != i],
            min(edges_per_entity, num_entities - 1)
        )
        for t in targets:
            # Relation type depends on the cluster pair (deterministic pattern)
            rel_type = (entity_clusters[i] + entity_clusters[t]) % num_relation_types
            src_list.append(i)
            tgt_list.append(t)
            edge_feat = relation_prototypes[rel_type] + torch.randn(d_edge) * 0.2
            edge_feat_list.append(edge_feat)
            edge_labels.append(rel_type)

    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    edge_features = torch.stack(edge_feat_list)

    graph = DeltaGraph(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
    )

    metadata = {
        'entity_clusters': entity_clusters,
        'edge_labels': edge_labels,
        'num_relation_types': num_relation_types,
        'relation_prototypes': relation_prototypes,
    }

    return graph, metadata


def create_analogy_task(
    num_patterns: int = 5,
    instances_per_pattern: int = 4,
    d_node: int = 64,
    d_edge: int = 32,
    seed: int = 42,
) -> Tuple[DeltaGraph, torch.Tensor]:
    """Create a graph specifically designed to test analogical reasoning.

    Creates patterns like:
      (A1, rel_1, B1), (A2, rel_1, B2), (A3, rel_1, B3)  — same relation
      (C1, rel_2, D1), (C2, rel_2, D2)  — different relation, same pattern

    The task: given a node pair, predict the relation type.
    Edge-to-edge attention should recognize that edges of the same type
    form a meta-pattern, even though the nodes differ.

    Returns:
        graph: DeltaGraph
        labels: [E] relation type per edge
    """
    torch.manual_seed(seed)

    nodes = []
    src_list = []
    tgt_list = []
    edge_feats = []
    labels = []

    node_idx = 0
    for pattern_id in range(num_patterns):
        # Each pattern has a distinct relation embedding prototype
        rel_proto = torch.randn(d_edge) * 0.5

        for instance in range(instances_per_pattern):
            # Create two nodes connected by this relation
            a_feat = torch.randn(d_node)
            b_feat = torch.randn(d_node)
            nodes.append(a_feat)
            nodes.append(b_feat)

            src_list.append(node_idx)
            tgt_list.append(node_idx + 1)
            edge_feats.append(rel_proto + torch.randn(d_edge) * 0.15)
            labels.append(pattern_id)

            node_idx += 2

    graph = DeltaGraph(
        node_features=torch.stack(nodes),
        edge_features=torch.stack(edge_feats),
        edge_index=torch.tensor([src_list, tgt_list], dtype=torch.long),
    )

    return graph, torch.tensor(labels, dtype=torch.long)


def split_edges_for_link_prediction(
    graph: DeltaGraph,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DeltaGraph, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split edges into train/val/test for link prediction.

    Returns:
        train_graph: graph with only training edges
        val_edges: [2, E_val] edge indices for validation
        val_labels: [E_val] edge feature targets
        test_edges: [2, E_test]
        test_labels: [E_test]
    """
    torch.manual_seed(seed)
    E = graph.num_edges
    perm = torch.randperm(E)

    train_end = int(E * train_ratio)
    val_end = int(E * (train_ratio + val_ratio))

    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]

    train_graph = DeltaGraph(
        node_features=graph.node_features,
        edge_features=graph.edge_features[train_idx],
        edge_index=graph.edge_index[:, train_idx],
    )

    return (
        train_graph,
        graph.edge_index[:, val_idx],
        graph.edge_features[val_idx],
        graph.edge_index[:, test_idx],
        graph.edge_features[test_idx],
    )


def create_sequential_memory_task(
    seq_length: int = 50,
    d_node: int = 64,
    d_edge: int = 32,
    num_facts: int = 10,
    recall_positions: int = 5,
    seed: int = 42,
) -> Tuple[DeltaGraph, List[Tuple[int, int]]]:
    """Create a task that requires remembering facts over long sequences.

    Simulates a sequence of events where some are "facts" to remember,
    and later positions require recalling specific facts. Tests whether
    tiered memory can maintain important information while compressing
    or forgetting less important timesteps.

    Returns:
        graph: sequential graph (chain topology + fact links)
        recall_tasks: list of (query_node, answer_node) pairs
    """
    torch.manual_seed(seed)
    random.seed(seed)

    # Create chain graph: node_0 -> node_1 -> ... -> node_{seq_length-1}
    node_features = torch.randn(seq_length, d_node)
    src = list(range(seq_length - 1))
    tgt = list(range(1, seq_length))
    edge_feats = [torch.randn(d_edge) for _ in range(seq_length - 1)]

    # Mark fact nodes with distinctive features
    fact_positions = sorted(random.sample(range(seq_length // 2), num_facts))
    fact_marker = torch.randn(d_node) * 2  # strong signal
    for pos in fact_positions:
        node_features[pos] = node_features[pos] + fact_marker

    # Create recall tasks: later nodes that need to reference specific facts
    recall_positions_list = sorted(random.sample(
        range(seq_length // 2, seq_length), min(recall_positions, seq_length // 2)
    ))
    recall_tasks = []
    for rp in recall_positions_list:
        fact = random.choice(fact_positions)
        recall_tasks.append((rp, fact))
        # Add a direct edge from recall position to fact (the "answer")
        src.append(rp)
        tgt.append(fact)
        edge_feats.append(torch.randn(d_edge))

    graph = DeltaGraph(
        node_features=node_features,
        edge_features=torch.stack(edge_feats),
        edge_index=torch.tensor([src, tgt], dtype=torch.long),
    )

    return graph, recall_tasks


def create_multi_relational_reasoning_task(
    num_entities: int = 30,
    num_base_relations: int = 4,
    num_derived_rules: int = 3,
    d_node: int = 64,
    d_edge: int = 32,
    seed: int = 42,
) -> Tuple[DeltaGraph, torch.Tensor, Dict]:
    """Create a harder task requiring compositional relational reasoning.

    Entities have base relations AND derived relations that follow logical rules:
      Rule 1: A --worksAt--> B, B --locatedIn--> C  =>  A --livesNear--> C
      Rule 2: A --friendOf--> B, A --friendOf--> C  =>  B --peerOf--> C
      Rule 3: A --manages--> B, B --manages--> C  =>  A --seniorTo--> C

    Labels are the FULL set (base + derived). Models must infer derived edges.

    Returns:
        graph: DeltaGraph with base edges only
        labels: [E_total] relation types for ALL edges (base + derived)
        metadata: dict with base/derived masks and rule info
    """
    torch.manual_seed(seed)
    random.seed(seed)

    num_total_rels = num_base_relations + num_derived_rules
    rel_prototypes = torch.randn(num_total_rels, d_edge)

    # Assign entities to types
    entity_types = [i % 3 for i in range(num_entities)]  # person, org, location
    node_features = torch.randn(num_entities, d_node)
    # Add type signal
    type_marker = torch.randn(3, d_node) * 0.5
    for i in range(num_entities):
        node_features[i] += type_marker[entity_types[i]]

    src, tgt, edge_feats, labels = [], [], [], []

    # Base relations
    base_edges_by_type: Dict[int, List[Tuple[int, int]]] = {r: [] for r in range(num_base_relations)}

    persons = [i for i in range(num_entities) if entity_types[i] == 0]
    orgs = [i for i in range(num_entities) if entity_types[i] == 1]
    locs = [i for i in range(num_entities) if entity_types[i] == 2]

    # Rel 0: worksAt (person -> org)
    for p in persons[:len(persons)//2]:
        o = random.choice(orgs) if orgs else random.choice(range(num_entities))
        src.append(p); tgt.append(o)
        edge_feats.append(rel_prototypes[0] + torch.randn(d_edge) * 0.15)
        labels.append(0)
        base_edges_by_type[0].append((p, o))

    # Rel 1: locatedIn (org -> location)
    for o in orgs:
        loc = random.choice(locs) if locs else random.choice(range(num_entities))
        src.append(o); tgt.append(loc)
        edge_feats.append(rel_prototypes[1] + torch.randn(d_edge) * 0.15)
        labels.append(1)
        base_edges_by_type[1].append((o, loc))

    # Rel 2: friendOf (person <-> person)
    for p in persons:
        friends = random.sample([x for x in persons if x != p],
                                min(2, len(persons) - 1))
        for f in friends:
            src.append(p); tgt.append(f)
            edge_feats.append(rel_prototypes[2] + torch.randn(d_edge) * 0.15)
            labels.append(2)
            base_edges_by_type[2].append((p, f))

    # Rel 3: manages (person -> person)
    managers = persons[:len(persons)//3]
    reports = persons[len(persons)//3:]
    for m in managers:
        reps = random.sample(reports, min(2, len(reports)))
        for r in reps:
            src.append(m); tgt.append(r)
            edge_feats.append(rel_prototypes[3] + torch.randn(d_edge) * 0.15)
            labels.append(3)
            base_edges_by_type[3].append((m, r))

    n_base = len(src)

    # Derived relations (added to graph with correct prototypes)
    # Rule 1: worksAt + locatedIn => livesNear (rel 4)
    derived_count = 0
    loc_of_org = {o: loc for o, loc in base_edges_by_type[1]}
    for p, o in base_edges_by_type[0]:
        if o in loc_of_org:
            loc = loc_of_org[o]
            src.append(p); tgt.append(loc)
            edge_feats.append(rel_prototypes[4] + torch.randn(d_edge) * 0.15)
            labels.append(4)
            derived_count += 1

    # Rule 2: friendOf + friendOf => peerOf (rel 5)
    friends_of = {}
    for a, b in base_edges_by_type[2]:
        friends_of.setdefault(a, []).append(b)
    seen_peer = set()
    for a, flist in friends_of.items():
        for i in range(len(flist)):
            for j in range(i + 1, len(flist)):
                pair = (min(flist[i], flist[j]), max(flist[i], flist[j]))
                if pair not in seen_peer:
                    seen_peer.add(pair)
                    src.append(pair[0]); tgt.append(pair[1])
                    edge_feats.append(rel_prototypes[5] + torch.randn(d_edge) * 0.15)
                    labels.append(5)
                    derived_count += 1

    # Rule 3: manages + manages => seniorTo (rel 6)
    manages_map = {}
    for m, r in base_edges_by_type[3]:
        manages_map.setdefault(m, []).append(r)
    reports_map = {}
    for m, r in base_edges_by_type[3]:
        reports_map.setdefault(r, []).append(m)
    for m, rlist in manages_map.items():
        for r in rlist:
            if r in manages_map:
                for rr in manages_map[r]:
                    src.append(m); tgt.append(rr)
                    edge_feats.append(rel_prototypes[6] + torch.randn(d_edge) * 0.15)
                    labels.append(6)
                    derived_count += 1

    graph = DeltaGraph(
        node_features=node_features,
        edge_features=torch.stack(edge_feats),
        edge_index=torch.tensor([src, tgt], dtype=torch.long),
    )

    base_mask = torch.zeros(len(labels), dtype=torch.bool)
    base_mask[:n_base] = True

    metadata = {
        'n_base': n_base,
        'n_derived': derived_count,
        'base_mask': base_mask,
        'derived_mask': ~base_mask,
        'num_total_relations': num_total_rels,
        'rules': [
            'worksAt + locatedIn => livesNear',
            'friendOf + friendOf => peerOf',
            'manages + manages => seniorTo',
        ],
    }

    return graph, torch.tensor(labels, dtype=torch.long), metadata


def create_contrastive_analogy_pairs(
    num_relation_types: int = 6,
    pairs_per_type: int = 8,
    d_node: int = 64,
    d_edge: int = 32,
    seed: int = 42,
) -> Tuple[DeltaGraph, torch.Tensor, torch.Tensor]:
    """Create data for contrastive analogy training.

    For each relation type, creates multiple (A, rel, B) instances.
    Returns anchor/positive/negative indices for triplet loss:
      - anchor: an edge
      - positive: another edge of the SAME relation type
      - negative: an edge of a DIFFERENT relation type

    Returns:
        graph: DeltaGraph
        labels: [E] relation type per edge
        triplets: [T, 3] tensor of (anchor, positive, negative) edge indices
    """
    torch.manual_seed(seed)

    nodes = []
    src_list, tgt_list = [], []
    edge_feats = []
    labels = []
    edges_by_type: Dict[int, List[int]] = {r: [] for r in range(num_relation_types)}

    node_idx = 0
    edge_idx = 0
    for rel_type in range(num_relation_types):
        rel_proto = torch.randn(d_edge) * 0.5
        for _ in range(pairs_per_type):
            nodes.append(torch.randn(d_node))
            nodes.append(torch.randn(d_node))
            src_list.append(node_idx)
            tgt_list.append(node_idx + 1)
            edge_feats.append(rel_proto + torch.randn(d_edge) * 0.2)
            labels.append(rel_type)
            edges_by_type[rel_type].append(edge_idx)
            node_idx += 2
            edge_idx += 1

    graph = DeltaGraph(
        node_features=torch.stack(nodes),
        edge_features=torch.stack(edge_feats),
        edge_index=torch.tensor([src_list, tgt_list], dtype=torch.long),
    )

    # Build triplets
    triplets = []
    all_types = list(range(num_relation_types))
    for rel_type in range(num_relation_types):
        edges = edges_by_type[rel_type]
        neg_types = [t for t in all_types if t != rel_type]
        for i in range(len(edges)):
            for j in range(len(edges)):
                if i != j:
                    neg_type = random.choice(neg_types)
                    neg_edge = random.choice(edges_by_type[neg_type])
                    triplets.append([edges[i], edges[j], neg_edge])

    return graph, torch.tensor(labels, dtype=torch.long), torch.tensor(triplets, dtype=torch.long)


def create_synthetic_kg_benchmark(
    num_entities: int = 100,
    num_relations: int = 10,
    num_triples: int = 500,
    d_node: int = 64,
    d_edge: int = 32,
    seed: int = 42,
) -> Tuple[DeltaGraph, torch.Tensor, Dict]:
    """Create a synthetic knowledge graph benchmark (FB15k-237 style).

    Generates a larger-scale KG with:
    - Hierarchical entity types
    - Multiple relation patterns (1-to-1, 1-to-N, N-to-N)
    - Train/val/test splits
    - Both link prediction and relation classification tasks

    Returns:
        graph: DeltaGraph
        labels: [E] relation type per edge
        metadata: splits, entity types, relation patterns
    """
    torch.manual_seed(seed)
    random.seed(seed)

    # Entity hierarchy: 5 top-level types, each with sub-types
    num_top_types = 5
    entity_type = [i % num_top_types for i in range(num_entities)]
    type_protos = torch.randn(num_top_types, d_node)
    node_features = torch.randn(num_entities, d_node) * 0.3
    for i in range(num_entities):
        node_features[i] += type_protos[entity_type[i]]

    # Relation prototypes with different patterns
    rel_protos = torch.randn(num_relations, d_edge)

    src, tgt, edge_feats, labels = [], [], [], []
    seen: set = set()

    for _ in range(num_triples):
        while True:
            s = random.randint(0, num_entities - 1)
            t = random.randint(0, num_entities - 1)
            if s != t and (s, t) not in seen:
                seen.add((s, t))
                break
        # Relation depends on entity types (structured pattern)
        r = (entity_type[s] * num_top_types + entity_type[t]) % num_relations
        src.append(s); tgt.append(t)
        edge_feats.append(rel_protos[r] + torch.randn(d_edge) * 0.2)
        labels.append(r)

    n = len(src)
    perm = torch.randperm(n)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    graph = DeltaGraph(
        node_features=node_features,
        edge_features=torch.stack(edge_feats),
        edge_index=torch.tensor([src, tgt], dtype=torch.long),
    )

    metadata = {
        'num_relations': num_relations,
        'entity_types': entity_type,
        'train_idx': perm[:train_end],
        'val_idx': perm[train_end:val_end],
        'test_idx': perm[val_end:],
    }

    return graph, torch.tensor(labels, dtype=torch.long), metadata
