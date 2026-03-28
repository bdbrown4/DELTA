"""
Real knowledge graph dataset loading for DELTA experiments.

Supports:
  - FB15k-237 (Toutanova & Chen, 2015): 14,541 entities, 237 relations, 310,116 triples
  - WN18RR (Dettmers et al., 2018): 40,943 entities, 11 relations, 93,003 triples

Downloads from standard academic mirrors and caches locally in data/.
"""

import os
import urllib.request
import torch
from typing import Tuple, Dict

from delta.graph import DeltaGraph

# Standard public KG benchmark dataset URLs
_BASE_URL = (
    'https://raw.githubusercontent.com/'
    'villmow/datasets_knowledge_embedding/master'
)

DATASET_URLS = {
    'fb15k-237': {
        'train': f'{_BASE_URL}/FB15k-237/train.txt',
        'valid': f'{_BASE_URL}/FB15k-237/valid.txt',
        'test':  f'{_BASE_URL}/FB15k-237/test.txt',
    },
    'wn18rr': {
        'train': f'{_BASE_URL}/WN18RR/train.txt',
        'valid': f'{_BASE_URL}/WN18RR/valid.txt',
        'test':  f'{_BASE_URL}/WN18RR/test.txt',
    },
}


def download_dataset(name: str, data_dir: str = 'data') -> str:
    """Download a KG benchmark dataset if not already cached.

    Args:
        name: 'fb15k-237' or 'wn18rr'
        data_dir: base directory for cached data

    Returns:
        Path to the dataset directory
    """
    if name not in DATASET_URLS:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(DATASET_URLS.keys())}")

    dataset_dir = os.path.join(data_dir, name)
    os.makedirs(dataset_dir, exist_ok=True)

    urls = DATASET_URLS[name]
    for split, url in urls.items():
        filepath = os.path.join(dataset_dir, f'{split}.txt')
        if not os.path.exists(filepath):
            print(f"  Downloading {name}/{split}.txt ...")
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {url}: {e}\n"
                    f"Please manually download and place files in {dataset_dir}/\n"
                    f"Expected files: train.txt, valid.txt, test.txt "
                    f"(TSV: head\\trelation\\ttail)"
                ) from e

    return dataset_dir


def _load_triples(filepath: str) -> list:
    """Load triples from a TSV file."""
    triples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def load_real_kg(
    name: str,
    d_node: int = 64,
    d_edge: int = 32,
    data_dir: str = 'data',
    seed: int = 42,
) -> Tuple[DeltaGraph, torch.Tensor, Dict]:
    """Load a real KG benchmark as a DeltaGraph.

    Node features: Xavier-initialized random vectors, one per entity.
    Edge features: relation prototype + small noise per triple instance.

    Args:
        name: 'fb15k-237' or 'wn18rr'
        d_node: node feature dimension
        d_edge: edge feature dimension
        data_dir: cache directory for downloaded data
        seed: random seed for deterministic feature initialization

    Returns:
        (graph, labels, metadata) where:
          graph: DeltaGraph with all triples (train+valid+test)
          labels: [E] relation ID per edge
          metadata: dict with splits, vocabularies, dataset stats
    """
    dataset_dir = download_dataset(name, data_dir)

    train_triples = _load_triples(os.path.join(dataset_dir, 'train.txt'))
    valid_triples = _load_triples(os.path.join(dataset_dir, 'valid.txt'))
    test_triples = _load_triples(os.path.join(dataset_dir, 'test.txt'))

    all_triples = train_triples + valid_triples + test_triples

    # Build vocabularies (sorted for determinism)
    entities = set()
    relations = set()
    for h, r, t in all_triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}

    num_entities = len(entity2id)
    num_relations = len(relation2id)

    print(f"  Loaded {name}: {num_entities} entities, {num_relations} relations, "
          f"{len(all_triples)} triples")
    print(f"    Splits: {len(train_triples)} train / "
          f"{len(valid_triples)} valid / {len(test_triples)} test")

    # Deterministic node features (Xavier init, one vector per entity)
    torch.manual_seed(seed)
    node_features = torch.nn.init.xavier_normal_(
        torch.empty(num_entities, d_node))

    # Relation prototypes (one per relation type)
    torch.manual_seed(seed + 1000)
    relation_prototypes = torch.nn.init.xavier_normal_(
        torch.empty(num_relations, d_edge))

    # Build edge index, features, and labels
    src_list, tgt_list, edge_feats, labels_list = [], [], [], []
    torch.manual_seed(seed + 2000)
    for h, r, t in all_triples:
        src_list.append(entity2id[h])
        tgt_list.append(entity2id[t])
        r_id = relation2id[r]
        noise = torch.randn(d_edge) * 0.1
        edge_feats.append(relation_prototypes[r_id] + noise)
        labels_list.append(r_id)

    graph = DeltaGraph(
        node_features=node_features,
        edge_features=torch.stack(edge_feats),
        edge_index=torch.tensor([src_list, tgt_list], dtype=torch.long),
    )

    labels = torch.tensor(labels_list, dtype=torch.long)

    # Official split indices (ordered: train, then valid, then test)
    n_train = len(train_triples)
    n_valid = len(valid_triples)

    metadata = {
        'name': name,
        'num_entities': num_entities,
        'num_relations': num_relations,
        'entity2id': entity2id,
        'relation2id': relation2id,
        'relation_prototypes': relation_prototypes,
        'train_idx': torch.arange(0, n_train),
        'val_idx': torch.arange(n_train, n_train + n_valid),
        'test_idx': torch.arange(n_train + n_valid, len(all_triples)),
        'num_train': n_train,
        'num_valid': n_valid,
        'num_test': len(test_triples),
    }

    return graph, labels, metadata
