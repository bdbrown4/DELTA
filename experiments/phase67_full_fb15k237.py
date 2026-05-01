"""Phase 67: Full FB15k-237 Hop-Depth Ablation + NBFNet Baselines

Phase 66 rejected the hypothesis that hops=2 outperforms hops=1 on the dense
N=500 FB15k-237 subgraph (mean degree ~19.7). All three conditions (node_only,
hops=1, hops=2) were statistically indistinguishable, with node_only actually
leading on all metrics (2p=0.728, 3p=0.742).

The key question that Phase 66 could NOT answer: does the 2-hop mechanism help
on the SPARSE full graph? At N=14,541 with mean degree ~4.1, a 1-hop neighbor
covers only ~4 nodes on average. Multi-hop chain queries traverse genuinely
limited local neighborhoods. This is the regime where 2-hop edge adjacency
should theoretically add signal by bridging structural gaps.

This phase answers the Phase 66 open question at full scale.

Design:
  Condition A — hops=1: DELTALayer default (current paper config)
  Condition B — hops=2: 2-hop edge adjacency (paper's claimed mechanism)
  Condition C — DistMult: non-GNN baseline (reproduces Phase 62 reference)
  Condition D — GraphGPS: GNN without edge adjacency (paper comparison baseline)

  NOTE: node_only omitted — Phase 66 showed it ties/beats edge attention on
  dense subgraph, but it's not architecturally interesting for the paper's
  claim. The paper compares DELTA vs GraphGPS; DistMult is the embedding baseline.

Memory strategy (Phase 64 validated):
  - Full FB15k-237 edge adjacency: ~210M pairs estimated (Phase 64 projected)
  - Use topk=128 sparse attention — validated in Phase 64 as lossless vs full softmax
  - hops=1 adj already large; hops=2 adj is ~18x larger → topk is MANDATORY at full N
  - Expected: topk=128 constrains attention to 128 most-relevant structural neighbors

Hypothesis:
  On sparse full FB15k-237 (mean degree ~4.1), hops=2 outperforms hops=1 on
  multi-hop queries by > 0.010 MRR on both 2p and 3p. The dense-subgraph null
  result from Phase 66 was a density artifact, not a fundamental architectural flaw.

Expected:
  - hops=2 2p: > hops=1 2p by +0.010
  - hops=2 3p: > hops=1 3p by +0.010
  - DELTA hops=1 LP MRR > 0.50 (Phase 62 showed ~0.24 at N=5000; full graph differs)

Hardware requirements:
  - RunPod H100 (80GB) or A100 (80GB) — local RTX 3080 Ti (12.9GB) insufficient
  - Estimated cost: ~4hr/seed × 3 seeds × 4 conditions = 48 GPU-hours ≈ $96 at $2/hr
  - topk=128 caps memory to tractable level (Phase 64 confirmed within 80GB at N=5000)

Usage:
  # RunPod H100 (full run — 3 seeds):
  python experiments/phase67_full_fb15k237.py --epochs 200 --eval_every 25 \\
      --patience 5 --seeds 42,123,456 --conditions hops1,hops2,distmult,graphgps

  # Single seed smoke test:
  python experiments/phase67_full_fb15k237.py --epochs 5 --seeds 42 \\
      --conditions hops1,hops2

  # hops ablation only (skip baselines):
  python experiments/phase67_full_fb15k237.py --epochs 200 --seeds 42,123,456 \\
      --conditions hops1,hops2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import warnings
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from delta.graph import DeltaGraph
from delta.model import DELTAModel, DELTALayer
from delta.datasets import download_dataset, _load_triples

from experiments.phase46c_link_prediction import (
    load_lp_data,
    build_train_graph_tensors,
    create_lp_model,
    train_epoch,
    evaluate_lp,
    LinkPredictionModel,
    ALL_MODELS,
)
from experiments.phase42_multihop import (
    generate_multihop_queries,
    audit_queries,
    build_full_adjacency,
    evaluate_multihop,
    compute_valid_answers,
)


# ════════════════════════════════════════════════════════════════════════════
# Sparse top-k edge adjacency
# ════════════════════════════════════════════════════════════════════════════

def build_topk_adj(edge_index, device, hops=1, topk=128, seed=0):
    """Build a hop-depth adj, then apply topk subsampling per target edge.

    For full FB15k-237 (N=14,541), the natural edge adjacency is enormous:
      - hops=1: estimated ~8M pairs  (each of 310K edges × avg 26 neighbors)
      - hops=2: estimated ~140M pairs

    topk=128 keeps the 128 highest-indexed neighbors per target edge
    (an index-based proxy for score-based top-k before training, since we don't
    have attention scores yet). During training, the EdgeAttention module uses
    the provided adj structure as the attention graph — topk at adj-build time
    gives a fixed sparsity budget of 128 per target edge.

    This matches Phase 64's validated approach: topk=128 preserves full-softmax
    quality (test MRR=0.2472 vs 0.2457) on N=5000, 63M pairs.

    Args:
        edge_index: [2, E] training edges
        device: torch device
        hops: 1 or 2
        topk: max neighbors per target edge (128 recommended)
        seed: for reproducible random tie-breaking

    Returns:
        adj [2, K] where K <= E * topk
    """
    if hops == 2 and topk is not None:
        return _build_two_hop_topk_adj(edge_index, device, topk, seed=seed)

    import warnings

    N = edge_index.max().item() + 1
    E = edge_index.shape[1]

    tmp = DeltaGraph(
        node_features=torch.zeros(N, 1, device=device),
        edge_features=torch.zeros(E, 1, device=device),
        edge_index=edge_index.to(device),
    )

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Sparse invariant checks')
        adj = tmp.build_edge_adjacency(hops=hops)
    build_time = time.time() - t0
    raw_pairs = adj.shape[1]
    print(f"    hops={hops}: {raw_pairs:,} pairs built in {build_time:.1f}s")

    if topk is not None and raw_pairs > 0:
        t1 = time.time()
        adj = _apply_topk_per_target(adj, E, topk, seed=seed)
        topk_time = time.time() - t1
        print(f"    hops={hops} after topk={topk}: {adj.shape[1]:,} pairs "
              f"[{topk_time:.1f}s]")

    del tmp
    return adj


def _build_two_hop_topk_adj(edge_index, device, topk, seed=0):
    """Build hops<=2 adjacency directly, without materializing the full 2-hop line graph.

    For an edge (u, v), edges within two hops in the line graph are exactly the
    edges incident to any node in {u, v} ∪ N(u) ∪ N(v). We build those candidate
    sets directly and cap them to `topk` per target edge before moving to GPU.
    """
    edge_index_cpu = edge_index.cpu().numpy()
    src = edge_index_cpu[0]
    dst = edge_index_cpu[1]
    num_edges = edge_index_cpu.shape[1]
    num_nodes = int(edge_index_cpu.max()) + 1

    incident_edges = [[] for _ in range(num_nodes)]
    neighbor_nodes = [set() for _ in range(num_nodes)]

    for edge_id, (head, tail) in enumerate(zip(src, dst)):
        head = int(head)
        tail = int(tail)
        incident_edges[head].append(edge_id)
        if tail != head:
            incident_edges[tail].append(edge_id)
            neighbor_nodes[head].add(tail)
            neighbor_nodes[tail].add(head)

    max_pairs = num_edges * topk
    sources = np.empty(max_pairs, dtype=np.int32)
    targets = np.empty(max_pairs, dtype=np.int32)
    rng = np.random.default_rng(seed)

    write_ptr = 0
    t0 = time.time()
    for edge_id, (head, tail) in enumerate(zip(src, dst)):
        candidate_nodes = {int(head), int(tail)}
        candidate_nodes.update(neighbor_nodes[int(head)])
        candidate_nodes.update(neighbor_nodes[int(tail)])

        candidate_edges = set()
        for node_id in candidate_nodes:
            candidate_edges.update(incident_edges[node_id])

        candidate_edges.discard(edge_id)
        if not candidate_edges:
            continue

        candidate_array = np.fromiter(candidate_edges, dtype=np.int32)
        if candidate_array.shape[0] > topk:
            candidate_array = rng.choice(candidate_array, size=topk, replace=False)

        next_ptr = write_ptr + candidate_array.shape[0]
        sources[write_ptr:next_ptr] = candidate_array
        targets[write_ptr:next_ptr] = edge_id
        write_ptr = next_ptr

        if (edge_id + 1) % 25000 == 0:
            elapsed = time.time() - t0
            print(f"    hops=2 progress: {edge_id + 1:,}/{num_edges:,} edges "
                  f"[{elapsed:.1f}s]")

    elapsed = time.time() - t0
    adj = torch.from_numpy(
        np.stack([sources[:write_ptr], targets[:write_ptr]])
    ).to(device=device, dtype=torch.long)
    print(f"    hops=2 direct topk={topk}: {adj.shape[1]:,} pairs built in {elapsed:.1f}s")
    return adj


def _apply_topk_per_target(adj, num_edges, topk, seed=0):
    """Keep at most `topk` neighbors per target edge in adj [2, K].

    adj[0] = source edges, adj[1] = target edges.
    Groups by target and randomly keeps topk per group.

    Args:
        adj: [2, K] edge adjacency tensor (on GPU or CPU)
        num_edges: total number of edges (for bounds checking)
        topk: max neighbors per target
        seed: random seed for reproducible subsampling

    Returns:
        [2, K'] with K' <= num_edges * topk
    """
    if adj.shape[1] == 0:
        return adj

    device = adj.device
    targets = adj[1]  # [K]

    # Sort by target edge for efficient grouping
    sort_idx = torch.argsort(targets, stable=True)
    adj_sorted = adj[:, sort_idx]
    targets_sorted = adj_sorted[1]

    # Count neighbors per target
    _, counts = torch.unique_consecutive(targets_sorted, return_counts=True)

    # Build keep mask: for each group, keep at most topk
    keep_mask = torch.zeros(adj_sorted.shape[1], dtype=torch.bool, device=device)
    torch.manual_seed(seed)

    offset = 0
    for c in counts.tolist():
        if c <= topk:
            keep_mask[offset:offset + c] = True
        else:
            # Random subsample within this group
            perm = torch.randperm(c, device=device)[:topk]
            keep_mask[offset + perm] = True
        offset += c

    return adj_sorted[:, keep_mask]


# ════════════════════════════════════════════════════════════════════════════
# Encode with injected adjacency
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _encode_with_adj(model, edge_index, edge_types, adj, cache_hops, device):
    """Encode graph with a pre-built edge adjacency injected into cache."""
    ei = edge_index.to(device)
    et = edge_types.to(device)

    nf = model.entity_emb.weight
    ef = model.edge_rel_emb(et)

    if model.needs_proj:
        nf = model.node_proj_in(nf)
        ef = model.edge_proj_in(ef)

    graph = DeltaGraph(
        node_features=nf,
        edge_features=ef,
        edge_index=ei,
    )
    graph._edge_adj_cache = (cache_hops, adj.to(device))

    if model.encoder is None:
        nf_out = nf
    else:
        encoded = model.encoder(graph)
        nf_out = encoded.node_features

    if model.needs_proj:
        nf_out = model.node_proj_out(nf_out)
    return nf_out


def _encode_with_adj_grad(model, edge_index, edge_types, adj, cache_hops, device):
    """Same as _encode_with_adj but with gradients (for training)."""
    ei = edge_index.to(device)
    et = edge_types.to(device)

    nf = model.entity_emb.weight
    ef = model.edge_rel_emb(et)

    if model.needs_proj:
        nf = model.node_proj_in(nf)
        ef = model.edge_proj_in(ef)

    graph = DeltaGraph(
        node_features=nf,
        edge_features=ef,
        edge_index=ei,
    )
    graph._edge_adj_cache = (cache_hops, adj.to(device))

    if model.encoder is None:
        nf_out = nf
    else:
        encoded = model.encoder(graph)
        nf_out = encoded.node_features

    if model.needs_proj:
        nf_out = model.node_proj_out(nf_out)
    return nf_out


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════

def _train_epoch_with_adj(model, train_triples, edge_index, edge_types,
                           optimizer, device, batch_size, adj, cache_hops,
                           label_smoothing=0.1):
    """One training epoch with controlled adj injection."""
    model.train()
    n = train_triples.shape[1]
    perm = torch.randperm(n)
    total_loss = 0.0
    num_batches = 0

    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        h = train_triples[0, idx].to(device)
        r = train_triples[1, idx].to(device)
        t = train_triples[2, idx].to(device)
        B = h.shape[0]
        N = model.num_entities

        node_feats = _encode_with_adj_grad(model, edge_index, edge_types,
                                            adj, cache_hops, device)

        # Tail prediction
        scores_t = model.score_all_tails(node_feats, h, r)
        targets_t = torch.zeros(B, N, device=device)
        targets_t[torch.arange(B, device=device), t] = 1.0
        if label_smoothing > 0:
            targets_t = targets_t * (1 - label_smoothing) + label_smoothing / N
        loss_t = F.binary_cross_entropy_with_logits(scores_t, targets_t)

        # Head prediction
        scores_h = model.score_all_heads(node_feats, r, t)
        targets_h = torch.zeros(B, N, device=device)
        targets_h[torch.arange(B, device=device), h] = 1.0
        if label_smoothing > 0:
            targets_h = targets_h * (1 - label_smoothing) + label_smoothing / N
        loss_h = F.binary_cross_entropy_with_logits(scores_h, targets_h)

        loss = (loss_t + loss_h) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def _evaluate_with_adj(model, triples, edge_index, edge_types,
                        hr_to_tails, rt_to_heads, device, adj, cache_hops,
                        batch_size=256):
    """Filtered LP evaluation with controlled adj."""
    model.eval()
    node_feats = _encode_with_adj(model, edge_index, edge_types,
                                   adj, cache_hops, device)
    all_ranks = []

    for start in range(0, triples.shape[1], batch_size):
        batch = triples[:, start:start + batch_size].to(device)
        h, r, t = batch[0], batch[1], batch[2]
        B = h.shape[0]

        scores_t = model.score_all_tails(node_feats, h, r)
        for i in range(B):
            hi, ri, ti = h[i].item(), r[i].item(), t[i].item()
            for tt in hr_to_tails.get((hi, ri), set()):
                if tt != ti:
                    scores_t[i, tt] = float('-inf')
            rank = int((scores_t[i] >= scores_t[i, ti]).sum().item())
            all_ranks.append(max(rank, 1))

        scores_h = model.score_all_heads(node_feats, r, t)
        for i in range(B):
            hi, ri, ti = h[i].item(), r[i].item(), t[i].item()
            for th_id in rt_to_heads.get((ri, ti), set()):
                if th_id != hi:
                    scores_h[i, th_id] = float('-inf')
            rank = int((scores_h[i] >= scores_h[i, hi]).sum().item())
            all_ranks.append(max(rank, 1))

    ranks = np.array(all_ranks, dtype=np.float64)
    return {
        'MRR':     float(np.mean(1.0 / ranks)),
        'Hits@1':  float(np.mean(ranks <= 1)),
        'Hits@3':  float(np.mean(ranks <= 3)),
        'Hits@10': float(np.mean(ranks <= 10)),
    }


@torch.no_grad()
def _evaluate_multihop_with_adj(model, queries_by_type, edge_index, edge_types,
                                  full_hr2t, device, adj, cache_hops,
                                  temperature=1.0, batch_size=64):
    """Multi-hop MRR eval with controlled adj."""
    model.eval()
    node_feats = _encode_with_adj(model, edge_index, edge_types,
                                   adj, cache_hops, device)
    results = {}

    for qtype in ['1p', '2p', '3p']:
        queries = queries_by_type.get(qtype, [])
        if not queries:
            results[qtype] = {'MRR': 0.0, 'Hits@1': 0.0,
                               'Hits@3': 0.0, 'Hits@10': 0.0, 'count': 0}
            continue

        num_hops_q = len(queries[0][1])
        all_ranks = []

        for start in range(0, len(queries), batch_size):
            batch = queries[start:start + batch_size]
            B = len(batch)
            anchors = torch.tensor([q[0] for q in batch], device=device)
            current_emb = node_feats[anchors]

            for hop in range(num_hops_q):
                rels = torch.tensor([q[1][hop] for q in batch], device=device)
                hr = current_emb * model.decoder_rel_emb(rels)
                scores = hr @ node_feats.t()

                if hop < num_hops_q - 1:
                    weights = torch.softmax(scores / temperature, dim=-1)
                    current_emb = weights @ node_feats

            for i in range(B):
                anchor = batch[i][0]
                rel_chain = batch[i][1]
                answer = batch[i][2]
                valid = compute_valid_answers(anchor, rel_chain, full_hr2t)
                for va in valid:
                    if va != answer:
                        scores[i, va] = float('-inf')
                rank = int((scores[i] >= scores[i, answer]).sum().item())
                all_ranks.append(max(rank, 1))

        ranks = np.array(all_ranks, dtype=np.float64)
        results[qtype] = {
            'MRR':    float(np.mean(1.0 / ranks)),
            'Hits@1': float(np.mean(ranks <= 1)),
            'Hits@3': float(np.mean(ranks <= 3)),
            'Hits@10':float(np.mean(ranks <= 10)),
            'count':  len(queries),
        }
    return results


# ════════════════════════════════════════════════════════════════════════════
# Single condition run
# ════════════════════════════════════════════════════════════════════════════

def run_condition(condition_name, adj_tensor, cache_hops,
                  data, queries, full_hr2t,
                  epochs, lr, device, batch_size, seed,
                  eval_every, patience):
    """Train + evaluate one (condition, seed). Returns metrics dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Select model type based on condition name
    if condition_name == 'distmult':
        model_key = 'distmult'
        adj_tensor_run = torch.zeros(2, 0, dtype=torch.long, device=device)
        cache_hops_run = 1
    elif condition_name == 'graphgps':
        model_key = 'graphgps'
        adj_tensor_run = torch.zeros(2, 0, dtype=torch.long, device=device)
        cache_hops_run = 1
    else:
        model_key = 'delta_matched'
        adj_tensor_run = adj_tensor
        cache_hops_run = cache_hops

    base_model = create_lp_model(
        model_key, data['num_entities'], data['num_relations'])
    base_model = base_model.to(device)

    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"\n  [{model_key} / {condition_name}] seed={seed}, {n_params:,} params")

    edge_index, edge_types = build_train_graph_tensors(data['train'])
    adj_dev = adj_tensor_run.to(device)
    optimizer = torch.optim.Adam(base_model.parameters(), lr=lr)

    best_val_mrr = 0.0
    best_state = None
    evals_no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # For DistMult/GraphGPS: use standard train_epoch (no adj injection needed)
        if condition_name in ('distmult', 'graphgps'):
            loss = train_epoch(
                base_model, data['train'], edge_index, edge_types,
                optimizer, device, batch_size)
        else:
            loss = _train_epoch_with_adj(
                base_model, data['train'], edge_index, edge_types,
                optimizer, device, batch_size, adj_dev, cache_hops_run)

        if epoch % eval_every == 0 or epoch == epochs:
            if condition_name in ('distmult', 'graphgps'):
                val = evaluate_lp(
                    base_model, data['val'], edge_index, edge_types,
                    data['hr_to_tails'], data['rt_to_heads'], device)
            else:
                val = _evaluate_with_adj(
                    base_model, data['val'], edge_index, edge_types,
                    data['hr_to_tails'], data['rt_to_heads'], device,
                    adj_dev, cache_hops_run)

            elapsed = time.time() - t0
            print(f"    Ep {epoch:4d}  loss={loss:.4f}  "
                  f"val_MRR={val['MRR']:.4f}  H@10={val['Hits@10']:.4f}"
                  f"  [{elapsed:.0f}s]")

            if val['MRR'] > best_val_mrr:
                best_val_mrr = val['MRR']
                best_state = {k: v.clone() for k, v in
                              base_model.state_dict().items()}
                evals_no_improve = 0
            else:
                evals_no_improve += 1
                if patience > 0 and evals_no_improve >= patience:
                    print(f"    Early stopping at epoch {epoch}")
                    break

    elapsed = time.time() - t0
    if best_state is not None:
        base_model.load_state_dict(best_state)
    print(f"    Training done: best_val_MRR={best_val_mrr:.4f} [{elapsed:.0f}s]")

    # Test LP evaluation
    if condition_name in ('distmult', 'graphgps'):
        lp_test = evaluate_lp(
            base_model, data['test'], edge_index, edge_types,
            data['hr_to_tails'], data['rt_to_heads'], device)
        # Multi-hop with no adj injection for non-DELTA models
        mh = _evaluate_multihop_with_adj(
            base_model, queries, edge_index, edge_types,
            full_hr2t, device,
            torch.zeros(2, 0, dtype=torch.long, device=device), 1)
    else:
        lp_test = _evaluate_with_adj(
            base_model, data['test'], edge_index, edge_types,
            data['hr_to_tails'], data['rt_to_heads'], device,
            adj_dev, cache_hops_run)
        mh = _evaluate_multihop_with_adj(
            base_model, queries, edge_index, edge_types,
            full_hr2t, device, adj_dev, cache_hops_run)

    print(f"    LP test: MRR={lp_test['MRR']:.4f}  H@10={lp_test['Hits@10']:.4f}")
    for qt in ['1p', '2p', '3p']:
        r = mh[qt]
        if r['count'] > 0:
            print(f"    {qt}: MRR={r['MRR']:.4f}  H@10={r['Hits@10']:.4f}"
                  f"  (n={r['count']})")

    return {
        'condition': condition_name,
        'seed': seed,
        'params': n_params,
        'train_time_s': elapsed,
        'best_val_MRR': best_val_mrr,
        'lp_MRR': lp_test['MRR'],
        'lp_H1': lp_test['Hits@1'],
        'lp_H3': lp_test['Hits@3'],
        'lp_H10': lp_test['Hits@10'],
        **{f'{qt}_MRR':   mh[qt]['MRR']   for qt in ['1p', '2p', '3p']},
        **{f'{qt}_H10':   mh[qt]['Hits@10'] for qt in ['1p', '2p', '3p']},
        **{f'{qt}_count': mh[qt]['count']  for qt in ['1p', '2p', '3p']},
    }


# ════════════════════════════════════════════════════════════════════════════
# Summary printing
# ════════════════════════════════════════════════════════════════════════════

def print_summary(all_results):
    """Print Phase 67 ablation summary table."""
    if not all_results:
        return

    by_cond = defaultdict(list)
    for r in all_results:
        by_cond[r['condition']].append(r)

    order = ['distmult', 'graphgps', 'hops1', 'hops2']

    print("\n" + "=" * 100)
    print("PHASE 67: FULL FB15k-237 HOP-DEPTH ABLATION")
    print("  Hypothesis: hops=2 > hops=1 by >0.010 MRR on 2p and 3p (sparse full graph)")
    print("=" * 100)
    header = (f"{'Condition':<14} {'Seeds':>5} {'LP MRR':>9} {'1p MRR':>9} "
              f"{'2p MRR':>9} {'3p MRR':>9} {'3p H@10':>9} {'2p→3p':>8}")
    print(f"\n{header}")
    print("-" * 100)

    for cond in order:
        results = by_cond.get(cond, [])
        if not results:
            continue
        n = len(results)

        def fmt(key):
            vals = [r[key] for r in results]
            m = np.mean(vals)
            s = np.std(vals) if n > 1 else 0.0
            return f"{m:.3f}±{s:.3f}" if n > 1 else f"{m:.4f}"

        p2_m = np.mean([r['2p_MRR'] for r in results])
        p3_m = np.mean([r['3p_MRR'] for r in results])
        delta_23 = p3_m - p2_m

        print(f"{cond:<14} {n:>5} {fmt('lp_MRR'):>9} {fmt('1p_MRR'):>9} "
              f"{fmt('2p_MRR'):>9} {fmt('3p_MRR'):>9} {fmt('3p_H10'):>9} "
              f"{delta_23:>+8.4f}")

    print("-" * 100)

    h1 = by_cond.get('hops1', [])
    h2 = by_cond.get('hops2', [])

    if h1 and h2:
        g_lp = np.mean([r['lp_MRR'] for r in h2]) - np.mean([r['lp_MRR'] for r in h1])
        g_2p = np.mean([r['2p_MRR'] for r in h2]) - np.mean([r['2p_MRR'] for r in h1])
        g_3p = np.mean([r['3p_MRR'] for r in h2]) - np.mean([r['3p_MRR'] for r in h1])
        sig_2p = "CONFIRMED" if g_2p > 0.010 else "REJECTED"
        sig_3p = "CONFIRMED" if g_3p > 0.010 else "REJECTED"
        print(f"\n  hops=2 vs hops=1:  LP gap={g_lp:+.4f}  "
              f"2p gap={g_2p:+.4f} [{sig_2p}]  3p gap={g_3p:+.4f} [{sig_3p}]")
        print(f"  (Threshold: >+0.010 required for CONFIRMED)")

    dm = by_cond.get('distmult', [])
    gps = by_cond.get('graphgps', [])
    if h1 and dm:
        g_dm = (np.mean([r['lp_MRR'] for r in h1])
                - np.mean([r['lp_MRR'] for r in dm]))
        print(f"  DELTA(hops=1) vs DistMult: LP gap={g_dm:+.4f}")
    if h1 and gps:
        g_3p_gps = (np.mean([r['3p_MRR'] for r in h1])
                    - np.mean([r['3p_MRR'] for r in gps]))
        print(f"  DELTA(hops=1) vs GraphGPS: 3p gap={g_3p_gps:+.4f}")

    print()


# ════════════════════════════════════════════════════════════════════════════
# Adjacency pre-build
# ════════════════════════════════════════════════════════════════════════════

def build_all_adjs(edge_index, device, topk=128):
    """Build hops=1 and hops=2 adjacencies with topk sparsification.

    For full FB15k-237, both adjs need topk to fit in GPU memory.
    topk=128 validated in Phase 64 as lossless vs full softmax.
    """
    print("  Building hops=1 adj with topk={topk}...".format(topk=topk))
    adj_1hop = build_topk_adj(edge_index, device, hops=1, topk=topk)

    print(f"  Building hops=2 adj with topk={topk}...")
    adj_2hop = build_topk_adj(edge_index, device, hops=2, topk=topk)

    return {
        'hops1': (adj_1hop, 1),
        'hops2': (adj_2hop, 2),
    }


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Phase 67: Full FB15k-237 hop-depth ablation')
    p.add_argument('--epochs', type=int, default=200,
                   help='Max epochs per condition/seed (default: 200)')
    p.add_argument('--eval_every', type=int, default=25,
                   help='Evaluate every N epochs (default: 25)')
    p.add_argument('--patience', type=int, default=5,
                   help='Early stopping patience in eval intervals (default: 5)')
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--batch_size', type=int, default=4096)
    p.add_argument('--seeds', type=str, default='42',
                   help='Comma-separated seeds, e.g. 42,123,456')
    p.add_argument('--conditions', type=str,
                   default='hops1,hops2,distmult,graphgps',
                   help='Comma-separated conditions: hops1,hops2,distmult,graphgps')
    p.add_argument('--topk', type=int, default=128,
                   help='Top-k neighbors for edge adj sparsification (default: 128)')
    p.add_argument('--max_entities', type=int, default=None,
                   help='Limit graph to top N entities (default: None = full graph)')
    p.add_argument('--max_queries', type=int, default=10000,
                   help='Max multi-hop queries per type (default: 10000)')
    p.add_argument('--output', type=str, default='phase67_output.json',
                   help='Output JSON file (default: phase67_output.json)')
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    conditions = [c.strip() for c in args.conditions.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Phase 67: Full FB15k-237 Hop-Depth Ablation + Baselines")
    print(f"  device={device}  epochs={args.epochs}  seeds={seeds}")
    print(f"  conditions={conditions}  topk={args.topk}")
    if args.max_entities:
        print(f"  NOTE: limiting to top {args.max_entities} entities (smoke test)")
    else:
        print("  Running on FULL FB15k-237 graph (N≈14,541)")

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("\n[1/4] Loading FB15k-237...")
    data = load_lp_data('fb15k-237', 'data', max_entities=args.max_entities)
    N = data['num_entities']
    R = data['num_relations']
    n_train = data['train'].shape[1]
    mean_degree = n_train / N if N > 0 else 0
    print(f"  Entities: {N:,}  Relations: {R:,}  Train triples: {n_train:,}")
    print(f"  Mean entity degree: {mean_degree:.1f} "
          f"({'SPARSE — topk adj meaningful' if mean_degree < 10 else 'dense'})")

    # ── 2. Multi-hop queries ──────────────────────────────────────────────
    print("\n[2/4] Generating multi-hop queries...")
    queries = generate_multihop_queries(
        data, max_queries_per_type=args.max_queries, seed=42)
    for qt in ['1p', '2p', '3p']:
        print(f"  {qt}: {len(queries.get(qt, []))} queries")

    issues = audit_queries(queries, data)
    if issues:
        print(f"  WARNING: {len(issues)} leakage issues found!")
        for iss in issues[:5]:
            print(f"    {iss}")
    else:
        print("  Leakage audit: PASSED")

    full_hr2t = build_full_adjacency(data)

    # ── 3. Pre-build adjacencies ──────────────────────────────────────────
    print("\n[3/4] Pre-building edge adjacencies (topk={})...".format(args.topk))
    edge_index, edge_types = build_train_graph_tensors(data['train'])

    delta_conditions = [c for c in conditions if c in ('hops1', 'hops2')]
    all_adjs = {}
    if delta_conditions:
        all_adjs = build_all_adjs(edge_index, device, topk=args.topk)
    else:
        print("  No DELTA conditions requested — skipping adj build")

    # ── 4. Run conditions × seeds ─────────────────────────────────────────
    print("\n[4/4] Running conditions...")
    all_results = []
    t_total = time.time()

    for cond_name in conditions:
        print(f"\n{'='*65}")
        if cond_name in ('distmult', 'graphgps'):
            adj_tensor = torch.zeros(2, 0, dtype=torch.long, device=device)
            cache_hops = 1
            print(f"Condition: {cond_name} (baseline — no edge adjacency)")
        elif cond_name in all_adjs:
            adj_tensor, cache_hops = all_adjs[cond_name]
            print(f"Condition: {cond_name}  adj_pairs={adj_tensor.shape[1]:,}  "
                  f"cache_hops={cache_hops}")
        else:
            print(f"  Unknown condition: {cond_name}. Skipping.")
            continue
        print('='*65)

        for seed in seeds:
            try:
                result = run_condition(
                    cond_name, adj_tensor, cache_hops,
                    data, queries, full_hr2t,
                    args.epochs, args.lr, device, args.batch_size,
                    seed, args.eval_every, args.patience)
                all_results.append(result)
            except Exception as e:
                import traceback
                print(f"  FAILED ({cond_name}, seed={seed}): {e}")
                traceback.print_exc()

    elapsed_total = time.time() - t_total
    print_summary(all_results)
    print(f"Total experiment time: {elapsed_total:.0f}s ({elapsed_total/3600:.1f}h)")

    # ── Save results ──────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), '..', args.output)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
