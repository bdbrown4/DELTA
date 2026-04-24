"""Phase 68: BetaE 9-Query-Type Benchmark (Standard Comparison)

Motivation: A NeurIPS reviewer flagged that DELTA's multi-hop results use a
"homemade benchmark" not directly comparable to published BetaE/GQE results.
Phases 42-44 used a custom chain query generator on a subgraph. This phase
evaluates DELTA on the standard BetaE benchmark to enable head-to-head
comparison with published BetaE, GQE, and NBFNet results.

BetaE benchmark details:
  - Dataset: FB15k-237 (full, 14,505 entities, 237 relations)
  - Query types: 1p, 2p, 3p, 2i, 3i, ip, pi, 2u, up (9 types)
  - 67,421 test queries total across types
  - Standard train/valid/test splits with easy/hard answer sets
  - Download: https://snap.stanford.edu/betae/KG_data.zip

Published baselines (HITS@10 on hard answers, from BetaE paper, NEURIPS 2020):
  GQE:          1p=54.6  2p=15.3  3p=10.8  2i=39.7  3i=51.4  ip=19.8  pi=22.4  2u=11.3  up=11.8
  Query2box:    1p=67.2  2p=20.9  3p=14.2  2i=46.0  3i=61.6  ip=27.7  pi=35.8  2u=20.7  up=19.7
  BetaE:        1p=65.1  2p=25.7  3p=24.7  2i=55.8  3i=66.5  ip=43.9  pi=49.5  2u=40.0  up=37.0

DELTA's competitive position: designed for chain queries (1p/2p/3p). May not
excel at intersection queries (2i/3i) or union queries (2u/up) without explicit
conjunction/disjunction modules. The honest paper framing: "DELTA addresses
chain reasoning; intersection/union queries require dedicated modules (future work)."

Implementation:
  - Chain queries (1p/2p/3p): direct soft entity traversal (same as Phase 42)
  - Intersection queries (2i/3i): score = product of individual path scores
  - IP (intersection then project): chain then intersect
  - PI (project then intersect): intersect then chain
  - Union queries (2u/up): score = max of individual path scores (de Morgan's)
  - Hard answer evaluation: filter out 'easy' answers from ranking

This gives DELTA's maximum performance on the standard benchmark using its
current architecture, without requiring new modules.

Usage:
  # Download BetaE data first:
  #   mkdir -p data/betae && cd data/betae
  #   wget https://snap.stanford.edu/betae/KG_data.zip
  #   unzip KG_data.zip
  # Then:
  python experiments/phase68_betae_queries.py

  # Quick test (1p/2p/3p only, subset of queries):
  python experiments/phase68_betae_queries.py --query_types 1p,2p,3p --max_queries 1000

  # Full benchmark with DELTA trained on full graph:
  python experiments/phase68_betae_queries.py --checkpoint phase67_best.pt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path

from delta.graph import DeltaGraph
from delta.model import DELTAModel
from experiments.phase46c_link_prediction import (
    load_lp_data,
    build_train_graph_tensors,
    create_lp_model,
    train_epoch,
    evaluate_lp,
    LinkPredictionModel,
)
from experiments.phase67_full_fb15k237 import (
    build_topk_adj,
    _encode_with_adj,
    _evaluate_with_adj,
    _train_epoch_with_adj,
)


# ════════════════════════════════════════════════════════════════════════════
# BetaE data loading
# ════════════════════════════════════════════════════════════════════════════

BETAE_QUERY_TYPES = ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2u', 'up']

# BetaE query structure names → our shorthand
BETAE_NAME_MAP = {
    '1p':  ('e', ('r',)),
    '2p':  ('e', ('r', 'r')),
    '3p':  ('e', ('r', 'r', 'r')),
    '2i':  (('e', ('r',)), ('e', ('r',))),
    '3i':  (('e', ('r',)), ('e', ('r',)), ('e', ('r',))),
    'ip':  ((('e', ('r',)), ('e', ('r',))), ('r',)),
    'pi':  (('e', ('r', 'r')), ('e', ('r',))),
    '2u':  (('e', ('r',)), ('e', ('r',)), ('u',)),
    'up':  ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)),
}


def load_betae_data(betae_dir):
    """Load BetaE benchmark queries from the standard directory structure.

    BetaE distributes data as:
      FB15k-237-betae/
        train_queries.pkl  — queries over train graph
        valid_queries.pkl  — valid queries (answer keys in valid_answers.pkl)
        test_queries.pkl   — test queries (answer keys in test_answers.pkl)
        valid_hard_answers.pkl
        test_hard_answers.pkl
        valid_easy_answers.pkl
        test_easy_answers.pkl
        id2ent.pkl, id2rel.pkl, ent2id.pkl, rel2id.pkl
        train.txt, valid.txt, test.txt

    Returns:
        dict with keys: test_queries, test_hard_answers, test_easy_answers,
                        id2ent, id2rel, ent2id, rel2id,
                        train_triples, valid_triples, test_triples
    """
    betae_dir = Path(betae_dir)
    fb_dir = betae_dir / 'FB15k-237-betae'

    if not fb_dir.exists():
        raise FileNotFoundError(
            f"BetaE data not found at {fb_dir}\n"
            "Download from https://snap.stanford.edu/betae/KG_data.zip\n"
            "Then: unzip KG_data.zip -d data/betae/")

    def load_pkl(name):
        with open(fb_dir / name, 'rb') as f:
            return pickle.load(f)

    print(f"  Loading BetaE data from {fb_dir}...")
    test_queries = load_pkl('test_queries.pkl')
    test_hard_answers = load_pkl('test_hard_answers.pkl')
    test_easy_answers = load_pkl('test_easy_answers.pkl')
    id2ent = load_pkl('id2ent.pkl')
    id2rel = load_pkl('id2rel.pkl')
    ent2id = load_pkl('ent2id.pkl')
    rel2id = load_pkl('rel2id.pkl')

    N = len(id2ent)
    R = len(id2rel)
    print(f"  Entities: {N:,}  Relations: {R:,}")

    # Load triples as numpy arrays
    def load_triples(fname):
        triples = []
        with open(fb_dir / fname) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    h, r, t = parts
                    if h in ent2id and r in rel2id and t in ent2id:
                        triples.append([ent2id[h], rel2id[r], ent2id[t]])
        return torch.tensor(triples, dtype=torch.long).t()  # [3, E]

    train_triples = load_triples('train.txt')
    print(f"  Train: {train_triples.shape[1]:,} triples")

    return {
        'test_queries': test_queries,
        'test_hard_answers': test_hard_answers,
        'test_easy_answers': test_easy_answers,
        'id2ent': id2ent,
        'id2rel': id2rel,
        'ent2id': ent2id,
        'rel2id': rel2id,
        'train_triples': train_triples,
        'num_entities': N,
        'num_relations': R,
    }


# ════════════════════════════════════════════════════════════════════════════
# Query evaluation by type
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_chain_query(model, qtype, queries, hard_answers, easy_answers,
                          node_feats, device, batch_size=64):
    """Evaluate chain queries (1p/2p/3p) on BetaE hard answer sets.

    Args:
        model: trained LP model with decoder_rel_emb
        qtype: '1p', '2p', or '3p'
        queries: list of (anchor, (r1, r2, ...)) tuples from BetaE
        hard_answers: dict mapping query → set of hard answer entity ids
        easy_answers: dict mapping query → set of easy answer entity ids
        node_feats: [N, d] entity representations
        device: torch device
        batch_size: queries per batch

    Returns:
        dict with MRR, Hits@1, Hits@3, Hits@10 on hard answers
    """
    all_ranks = []
    query_list = [(q, hard_answers.get(q, set()), easy_answers.get(q, set()))
                  for q in queries if q in hard_answers and hard_answers[q]]

    for start in range(0, len(query_list), batch_size):
        batch = query_list[start:start + batch_size]
        B = len(batch)

        for qdata, hard_ans, easy_ans in batch:
            anchor = qdata[0] if not isinstance(qdata[0], tuple) else qdata[0][0]
            rels = qdata[1]  # tuple of relation ids

            # Soft entity traversal: chain multiply
            curr_emb = node_feats[anchor].unsqueeze(0)  # [1, d]
            for r in rels[:-1]:
                rel_emb = model.decoder_rel_emb(
                    torch.tensor([r], device=device))
                scores = (curr_emb * rel_emb) @ node_feats.t()  # [1, N]
                weights = torch.softmax(scores, dim=-1)
                curr_emb = weights @ node_feats  # [1, d]

            # Final hop: get ranking scores
            r_last = rels[-1]
            rel_emb = model.decoder_rel_emb(
                torch.tensor([r_last], device=device))
            scores = (curr_emb * rel_emb) @ node_feats.t()  # [1, N]
            scores = scores.squeeze(0)  # [N]

            # Filter easy answers (BetaE protocol: only rank hard answers)
            for ea in easy_ans:
                scores[ea] = float('-inf')

            # Rank each hard answer
            for ha in hard_ans:
                if ha >= len(scores):
                    continue
                # Mask out other hard answers for this specific ranking
                scores_copy = scores.clone()
                for other_ha in hard_ans:
                    if other_ha != ha:
                        scores_copy[other_ha] = float('-inf')
                rank = int((scores_copy >= scores_copy[ha]).sum().item())
                all_ranks.append(max(rank, 1))

    if not all_ranks:
        return {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0,
                'count': 0}

    ranks = np.array(all_ranks, dtype=np.float64)
    return {
        'MRR':     float(np.mean(1.0 / ranks)),
        'Hits@1':  float(np.mean(ranks <= 1)),
        'Hits@3':  float(np.mean(ranks <= 3)),
        'Hits@10': float(np.mean(ranks <= 10)),
        'count':   len(query_list),
    }


@torch.no_grad()
def evaluate_intersection_query(model, qtype, queries, hard_answers, easy_answers,
                                  node_feats, device, batch_size=32):
    """Evaluate intersection queries (2i/3i) using product-of-scores.

    2i: (anchor1, r1) AND (anchor2, r2) → target
    3i: (anchor1, r1) AND (anchor2, r2) AND (anchor3, r3) → target

    Score = product of individual 1-hop scores for each (anchor, rel) pair.
    """
    all_ranks = []
    query_list = [(q, hard_answers.get(q, set()), easy_answers.get(q, set()))
                  for q in queries if q in hard_answers and hard_answers[q]]

    for qdata, hard_ans, easy_ans in query_list:
        # Each query is a tuple of (anchor_i, (rel_i,)) pairs
        combined_scores = None

        for sub_query in qdata:
            anchor = sub_query[0]
            rel = sub_query[1][0]  # single-hop
            rel_emb = model.decoder_rel_emb(
                torch.tensor([rel], device=device))
            anchor_emb = node_feats[anchor].unsqueeze(0)
            scores = (anchor_emb * rel_emb) @ node_feats.t()  # [1, N]
            scores = torch.sigmoid(scores).squeeze(0)  # [N], interpret as prob

            if combined_scores is None:
                combined_scores = scores
            else:
                combined_scores = combined_scores * scores  # product = AND

        # Filter easy answers
        for ea in easy_ans:
            combined_scores[ea] = float('-inf')

        # Rank each hard answer (max score = most likely intersection member)
        combined_log = torch.log(combined_scores + 1e-10)
        for ha in hard_ans:
            if ha >= len(combined_log):
                continue
            scores_copy = combined_log.clone()
            for other_ha in hard_ans:
                if other_ha != ha:
                    scores_copy[other_ha] = float('-inf')
            rank = int((scores_copy >= scores_copy[ha]).sum().item())
            all_ranks.append(max(rank, 1))

    if not all_ranks:
        return {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0,
                'count': 0}

    ranks = np.array(all_ranks, dtype=np.float64)
    return {
        'MRR':     float(np.mean(1.0 / ranks)),
        'Hits@1':  float(np.mean(ranks <= 1)),
        'Hits@3':  float(np.mean(ranks <= 3)),
        'Hits@10': float(np.mean(ranks <= 10)),
        'count':   len(query_list),
    }


@torch.no_grad()
def evaluate_union_query(model, qtype, queries, hard_answers, easy_answers,
                          node_feats, device):
    """Evaluate union queries (2u) using max-of-scores (de Morgan's).

    2u: (anchor1, r1) OR (anchor2, r2) → target
    Score = max of individual 1-hop scores.
    """
    all_ranks = []
    query_list = [(q, hard_answers.get(q, set()), easy_answers.get(q, set()))
                  for q in queries if q in hard_answers and hard_answers[q]]

    for qdata, hard_ans, easy_ans in query_list:
        combined_scores = None

        # 2u: ((e, (r,)), (e, (r,)), (u,)) — skip the 'u' marker
        sub_queries = [sq for sq in qdata if isinstance(sq, tuple) and sq != ('u',)]

        for sub_query in sub_queries:
            anchor = sub_query[0]
            rel = sub_query[1][0]
            rel_emb = model.decoder_rel_emb(
                torch.tensor([rel], device=device))
            anchor_emb = node_feats[anchor].unsqueeze(0)
            scores = ((anchor_emb * rel_emb) @ node_feats.t()).squeeze(0)

            if combined_scores is None:
                combined_scores = scores
            else:
                combined_scores = torch.maximum(combined_scores, scores)

        for ea in easy_ans:
            combined_scores[ea] = float('-inf')

        for ha in hard_ans:
            if ha >= len(combined_scores):
                continue
            scores_copy = combined_scores.clone()
            for other_ha in hard_ans:
                if other_ha != ha:
                    scores_copy[other_ha] = float('-inf')
            rank = int((scores_copy >= scores_copy[ha]).sum().item())
            all_ranks.append(max(rank, 1))

    if not all_ranks:
        return {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0,
                'count': 0}

    ranks = np.array(all_ranks, dtype=np.float64)
    return {
        'MRR':     float(np.mean(1.0 / ranks)),
        'Hits@1':  float(np.mean(ranks <= 1)),
        'Hits@3':  float(np.mean(ranks <= 3)),
        'Hits@10': float(np.mean(ranks <= 10)),
        'count':   len(query_list),
    }


@torch.no_grad()
def evaluate_ip_query(model, queries, hard_answers, easy_answers,
                       node_feats, device):
    """Evaluate ip (intersect-then-project) queries.

    ip: ((e, (r,)), (e, (r,))) then (r,)
    → intersect two 1-hop endpoints, then project through one more relation.
    """
    all_ranks = []
    query_list = [(q, hard_answers.get(q, set()), easy_answers.get(q, set()))
                  for q in queries if q in hard_answers and hard_answers[q]]

    for qdata, hard_ans, easy_ans in query_list:
        # qdata = (((e, (r,)), (e, (r,))), (r,))
        intersection_part, final_rel_tuple = qdata[0], qdata[1]
        final_rel = final_rel_tuple[0]

        # Intersect
        combined = None
        for sub_q in intersection_part:
            anchor = sub_q[0]
            rel = sub_q[1][0]
            rel_emb = model.decoder_rel_emb(torch.tensor([rel], device=device))
            anchor_emb = node_feats[anchor].unsqueeze(0)
            scores = torch.sigmoid(
                (anchor_emb * rel_emb) @ node_feats.t()).squeeze(0)
            combined = scores if combined is None else combined * scores

        # Project: use intermediate distribution as soft anchor
        weights = torch.softmax(torch.log(combined + 1e-10), dim=-1)
        inter_emb = (weights.unsqueeze(0) @ node_feats)  # [1, d]

        final_rel_emb = model.decoder_rel_emb(torch.tensor([final_rel], device=device))
        final_scores = (inter_emb * final_rel_emb) @ node_feats.t()
        final_scores = final_scores.squeeze(0)

        for ea in easy_ans:
            final_scores[ea] = float('-inf')

        for ha in hard_ans:
            if ha >= len(final_scores):
                continue
            scores_copy = final_scores.clone()
            for other_ha in hard_ans:
                if other_ha != ha:
                    scores_copy[other_ha] = float('-inf')
            rank = int((scores_copy >= scores_copy[ha]).sum().item())
            all_ranks.append(max(rank, 1))

    if not all_ranks:
        return {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0,
                'count': 0}

    ranks = np.array(all_ranks, dtype=np.float64)
    return {
        'MRR':     float(np.mean(1.0 / ranks)),
        'Hits@1':  float(np.mean(ranks <= 1)),
        'Hits@3':  float(np.mean(ranks <= 3)),
        'Hits@10': float(np.mean(ranks <= 10)),
        'count':   len(query_list),
    }


@torch.no_grad()
def evaluate_pi_query(model, queries, hard_answers, easy_answers,
                       node_feats, device):
    """Evaluate pi (project-then-intersect) queries.

    pi: (e, (r, r)) AND (e, (r,)) — first sub-query is a 2-hop chain.
    """
    all_ranks = []
    query_list = [(q, hard_answers.get(q, set()), easy_answers.get(q, set()))
                  for q in queries if q in hard_answers and hard_answers[q]]

    for qdata, hard_ans, easy_ans in query_list:
        # qdata = ((e, (r, r)), (e, (r,)))
        combined = None
        for sub_q in qdata:
            anchor = sub_q[0]
            rels = sub_q[1]
            curr_emb = node_feats[anchor].unsqueeze(0)
            for r in rels[:-1]:
                rel_emb = model.decoder_rel_emb(torch.tensor([r], device=device))
                scores = (curr_emb * rel_emb) @ node_feats.t()
                weights = torch.softmax(scores, dim=-1)
                curr_emb = weights @ node_feats
            r_last = rels[-1]
            rel_emb = model.decoder_rel_emb(torch.tensor([r_last], device=device))
            scores = torch.sigmoid(
                (curr_emb * rel_emb) @ node_feats.t()).squeeze(0)
            combined = scores if combined is None else combined * scores

        final_scores = torch.log(combined + 1e-10)
        for ea in easy_ans:
            final_scores[ea] = float('-inf')

        for ha in hard_ans:
            if ha >= len(final_scores):
                continue
            scores_copy = final_scores.clone()
            for other_ha in hard_ans:
                if other_ha != ha:
                    scores_copy[other_ha] = float('-inf')
            rank = int((scores_copy >= scores_copy[ha]).sum().item())
            all_ranks.append(max(rank, 1))

    if not all_ranks:
        return {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0,
                'count': 0}

    ranks = np.array(all_ranks, dtype=np.float64)
    return {
        'MRR':     float(np.mean(1.0 / ranks)),
        'Hits@1':  float(np.mean(ranks <= 1)),
        'Hits@3':  float(np.mean(ranks <= 3)),
        'Hits@10': float(np.mean(ranks <= 10)),
        'count':   len(query_list),
    }


@torch.no_grad()
def evaluate_up_query(model, queries, hard_answers, easy_answers,
                       node_feats, device):
    """Evaluate up (union-then-project) queries.

    up: ((e, (r,)) OR (e, (r,))) then (r,)
    → union two 1-hop results, then project through final relation.
    """
    all_ranks = []
    query_list = [(q, hard_answers.get(q, set()), easy_answers.get(q, set()))
                  for q in queries if q in hard_answers and hard_answers[q]]

    for qdata, hard_ans, easy_ans in query_list:
        # qdata = (((e, (r,)), (e, (r,)), (u,)), (r,))
        union_part, final_rel_tuple = qdata[0], qdata[1]
        final_rel = final_rel_tuple[0]

        union_scores = None
        for sub_q in union_part:
            if sub_q == ('u',):
                continue
            anchor = sub_q[0]
            rel = sub_q[1][0]
            rel_emb = model.decoder_rel_emb(torch.tensor([rel], device=device))
            anchor_emb = node_feats[anchor].unsqueeze(0)
            scores = ((anchor_emb * rel_emb) @ node_feats.t()).squeeze(0)
            union_scores = scores if union_scores is None else torch.maximum(union_scores, scores)

        # Project union distribution
        weights = torch.softmax(union_scores, dim=-1)
        inter_emb = (weights.unsqueeze(0) @ node_feats)
        final_rel_emb = model.decoder_rel_emb(torch.tensor([final_rel], device=device))
        final_scores = (inter_emb * final_rel_emb) @ node_feats.t()
        final_scores = final_scores.squeeze(0)

        for ea in easy_ans:
            final_scores[ea] = float('-inf')

        for ha in hard_ans:
            if ha >= len(final_scores):
                continue
            scores_copy = final_scores.clone()
            for other_ha in hard_ans:
                if other_ha != ha:
                    scores_copy[other_ha] = float('-inf')
            rank = int((scores_copy >= scores_copy[ha]).sum().item())
            all_ranks.append(max(rank, 1))

    if not all_ranks:
        return {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0,
                'count': 0}

    ranks = np.array(all_ranks, dtype=np.float64)
    return {
        'MRR':     float(np.mean(1.0 / ranks)),
        'Hits@1':  float(np.mean(ranks <= 1)),
        'Hits@3':  float(np.mean(ranks <= 3)),
        'Hits@10': float(np.mean(ranks <= 10)),
        'count':   len(query_list),
    }


def evaluate_all_query_types(model, betae_data, adj, cache_hops, device,
                              edge_index, edge_types, query_types=None,
                              max_queries=None):
    """Evaluate DELTA on all requested BetaE query types.

    Returns dict: query_type → metrics dict
    """
    if query_types is None:
        query_types = BETAE_QUERY_TYPES

    # Encode graph once
    print("  Encoding graph...")
    node_feats = _encode_with_adj(
        model, edge_index, edge_types, adj, cache_hops, device)
    print(f"  node_feats: {node_feats.shape}")

    test_queries = betae_data['test_queries']
    hard_answers = betae_data['test_hard_answers']
    easy_answers = betae_data['test_easy_answers']

    results = {}

    for qtype in query_types:
        t0 = time.time()
        # Find the matching query structure key in BetaE data
        query_struct = BETAE_NAME_MAP.get(qtype)
        if query_struct not in test_queries:
            print(f"  {qtype}: no queries in test set")
            results[qtype] = {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0,
                               'Hits@10': 0.0, 'count': 0}
            continue

        raw_queries = list(test_queries[query_struct].keys())
        if max_queries:
            raw_queries = raw_queries[:max_queries]
        n = len(raw_queries)

        if qtype in ('1p', '2p', '3p'):
            res = evaluate_chain_query(
                model, qtype, raw_queries, hard_answers, easy_answers,
                node_feats, device)
        elif qtype in ('2i', '3i'):
            res = evaluate_intersection_query(
                model, qtype, raw_queries, hard_answers, easy_answers,
                node_feats, device)
        elif qtype == 'ip':
            res = evaluate_ip_query(
                model, raw_queries, hard_answers, easy_answers,
                node_feats, device)
        elif qtype == 'pi':
            res = evaluate_pi_query(
                model, raw_queries, hard_answers, easy_answers,
                node_feats, device)
        elif qtype == '2u':
            res = evaluate_union_query(
                model, qtype, raw_queries, hard_answers, easy_answers,
                node_feats, device)
        elif qtype == 'up':
            res = evaluate_up_query(
                model, raw_queries, hard_answers, easy_answers,
                node_feats, device)
        else:
            print(f"  {qtype}: unsupported")
            continue

        elapsed = time.time() - t0
        print(f"  {qtype:4s}: MRR={res['MRR']:.4f}  H@1={res['Hits@1']:.4f}  "
              f"H@3={res['Hits@3']:.4f}  H@10={res['Hits@10']:.4f}  "
              f"(n={res['count']}, {elapsed:.1f}s)")
        results[qtype] = res

    return results


# ════════════════════════════════════════════════════════════════════════════
# Summary table with published baselines
# ════════════════════════════════════════════════════════════════════════════

PUBLISHED_BASELINES = {
    'GQE (Hamilton+2018)': {
        '1p': 0.546, '2p': 0.153, '3p': 0.108, '2i': 0.397, '3i': 0.514,
        'ip': 0.198, 'pi': 0.224, '2u': 0.113, 'up': 0.118,
    },
    'Query2box (Ren+2020)': {
        '1p': 0.672, '2p': 0.209, '3p': 0.142, '2i': 0.460, '3i': 0.616,
        'ip': 0.277, 'pi': 0.358, '2u': 0.207, 'up': 0.197,
    },
    'BetaE (Ren+2020)': {
        '1p': 0.651, '2p': 0.257, '3p': 0.247, '2i': 0.558, '3i': 0.665,
        'ip': 0.439, 'pi': 0.495, '2u': 0.400, 'up': 0.370,
    },
}


def print_summary_table(delta_results, query_types=None):
    """Print comparison table vs published BetaE baselines."""
    if query_types is None:
        query_types = [qt for qt in BETAE_QUERY_TYPES if qt in delta_results]

    print("\n" + "=" * 100)
    print("PHASE 68: BetaE BENCHMARK — HITS@10 ON HARD ANSWERS")
    print("  Comparison to published BetaE/GQE baselines (H@10, hard answer evaluation)")
    print("=" * 100)

    col_width = 8
    header = f"{'Model':<28}" + "".join(f"{qt:>{col_width}}" for qt in query_types)
    print(f"\n{header}")
    print("-" * 100)

    # Published baselines
    for model_name, baseline in PUBLISHED_BASELINES.items():
        row = f"{model_name:<28}"
        for qt in query_types:
            val = baseline.get(qt, 0.0)
            row += f"{val*100:>{col_width}.1f}"
        print(row)

    print("-" * 100)

    # DELTA result
    row = f"{'DELTA (Phase 68)':<28}"
    for qt in query_types:
        r = delta_results.get(qt, {})
        val = r.get('Hits@10', 0.0)
        row += f"{val*100:>{col_width}.1f}"
    print(row)
    print("-" * 100)

    # MRR section
    print(f"\n{'Model':<28}" + "".join(f"{qt:>{col_width}}" for qt in query_types))
    print("  (MRR — for reference)")
    print("-" * 100)
    row = f"{'DELTA (Phase 68)':<28}"
    for qt in query_types:
        r = delta_results.get(qt, {})
        val = r.get('MRR', 0.0)
        row += f"{val*100:>{col_width}.1f}"
    print(row)
    print("-" * 100)

    # Chain query focus
    chain_types = [qt for qt in ['1p', '2p', '3p'] if qt in delta_results]
    if chain_types and len(chain_types) >= 2:
        delta_chain = np.mean([delta_results[qt]['Hits@10'] for qt in chain_types])
        betae_chain = np.mean([PUBLISHED_BASELINES['BetaE (Ren+2020)'].get(qt, 0)
                                for qt in chain_types])
        print(f"\n  Chain query avg H@10 (1p/2p/3p):")
        print(f"    BetaE:  {betae_chain*100:.1f}%")
        print(f"    DELTA:  {delta_chain*100:.1f}%  "
              f"({'ABOVE' if delta_chain > betae_chain else 'BELOW'} BetaE)")
    print()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Phase 68: BetaE 9-query benchmark')
    p.add_argument('--betae_dir', type=str, default='data/betae',
                   help='Directory containing BetaE data (default: data/betae)')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Path to trained model checkpoint (default: train fresh)')
    p.add_argument('--epochs', type=int, default=200,
                   help='Training epochs if no checkpoint (default: 200)')
    p.add_argument('--eval_every', type=int, default=25)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--batch_size', type=int, default=4096)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--topk', type=int, default=128,
                   help='Top-k sparse adj (default: 128)')
    p.add_argument('--hops', type=int, default=1,
                   help='Edge adj hops: 1 or 2 (default: 1)')
    p.add_argument('--query_types', type=str,
                   default='1p,2p,3p,2i,3i,ip,pi,2u,up',
                   help='Comma-separated query types to evaluate')
    p.add_argument('--max_queries', type=int, default=None,
                   help='Cap queries per type (default: None = all)')
    p.add_argument('--output', type=str, default='phase68_output.json',
                   help='Output JSON (default: phase68_output.json)')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    query_types = [q.strip() for q in args.query_types.split(',')]

    print("Phase 68: BetaE 9-Query-Type Benchmark")
    print(f"  device={device}  hops={args.hops}  topk={args.topk}")
    print(f"  query_types={query_types}")

    # ── 1. Load BetaE data ────────────────────────────────────────────────
    print("\n[1/4] Loading BetaE benchmark data...")
    betae = load_betae_data(args.betae_dir)
    N = betae['num_entities']
    R = betae['num_relations']

    # ── 2. Build or load model ────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    edge_index = betae['train_triples'][:2].clone()  # [2, E] head and tail
    edge_types = betae['train_triples'][1].clone()   # [E] relation ids
    # Note: train_triples is [3, E]: [head, rel, tail]
    ei = torch.stack([betae['train_triples'][0], betae['train_triples'][2]])  # [2, E]
    et = betae['train_triples'][1]  # [E]

    # ── 3. Build edge adjacency ───────────────────────────────────────────
    print(f"\n[2/4] Building hops={args.hops} edge adj with topk={args.topk}...")
    adj = build_topk_adj(ei, device, hops=args.hops, topk=args.topk)
    cache_hops = args.hops

    # ── 4. Train model (or load checkpoint) ──────────────────────────────
    print("\n[3/4] Training DELTA-Matched on full FB15k-237...")
    model = create_lp_model('delta_matched', N, R).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"  Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        if args.checkpoint:
            print(f"  WARNING: checkpoint {args.checkpoint} not found. Training fresh.")

        # Build LP data for train/val/test filtering
        lp_data = load_lp_data('fb15k-237', 'data')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        adj_dev = adj.to(device)

        best_val_mrr = 0.0
        best_state = None
        evals_no_improve = 0
        t0 = time.time()

        for epoch in range(1, args.epochs + 1):
            loss = _train_epoch_with_adj(
                model, lp_data['train'], ei, et,
                optimizer, device, args.batch_size, adj_dev, cache_hops)

            if epoch % args.eval_every == 0 or epoch == args.epochs:
                val = _evaluate_with_adj(
                    model, lp_data['val'], ei, et,
                    lp_data['hr_to_tails'], lp_data['rt_to_heads'],
                    device, adj_dev, cache_hops)
                print(f"  Ep {epoch:4d}  loss={loss:.4f}  "
                      f"val_MRR={val['MRR']:.4f}  H@10={val['Hits@10']:.4f}"
                      f"  [{time.time()-t0:.0f}s]")

                if val['MRR'] > best_val_mrr:
                    best_val_mrr = val['MRR']
                    best_state = {k: v.clone() for k, v in
                                  model.state_dict().items()}
                    evals_no_improve = 0
                else:
                    evals_no_improve += 1
                    if args.patience > 0 and evals_no_improve >= args.patience:
                        print(f"  Early stopping at epoch {epoch}")
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

    # ── 5. Evaluate on BetaE ─────────────────────────────────────────────
    print("\n[4/4] Evaluating on BetaE queries...")
    adj_dev = adj.to(device)
    results = evaluate_all_query_types(
        model, betae, adj_dev, cache_hops, device,
        ei, et, query_types=query_types, max_queries=args.max_queries)

    print_summary_table(results, query_types=query_types)
    print(f"Note: DELTA chain queries (1p/2p/3p) use soft entity traversal.")
    print(f"Intersection/union queries use product/max-of-scores approximation.")
    print(f"Dedicated logic modules (BetaE-style) would improve 2i/3i/ip/pi/2u/up.")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), '..', args.output)
    with open(out_path, 'w') as f:
        json.dump({'results': results,
                   'args': vars(args),
                   'num_entities': N,
                   'num_relations': R}, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
