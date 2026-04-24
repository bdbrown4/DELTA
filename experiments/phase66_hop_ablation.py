"""Phase 66: 1-Hop vs 2-Hop Edge Adjacency Ablation (NeurIPS Reviewer Response)

This is the missing ablation flagged by the NeurIPS reviewer:
  "No ablations of the core mechanism. Where's the 1-hop vs 2-hop edge
   adjacency ablation on the main table?"

The paper's central architectural claim is that 2-hop edge adjacency enables
direct relational composition. This ablation tests that claim directly on
FB15k-237 (top-500 subgraph) with the standard LP + multi-hop evaluation.

Three conditions:
  A) node_only  — EdgeAttention stream disabled (empty E_adj)
                  Reduces DELTA to GAT-style node attention only.
  B) hops=1     — Edges sharing an endpoint (standard 1-hop adj, CURRENT DEFAULT)
                  DELTALayer.forward() calls build_edge_adjacency() with hops=1.
  C) hops=2     — Edges two steps away (paper's claimed mechanism)
                  A_E^{(2)} = (B^T B)^2 — enables attending to edges 2 steps away.

Key question: does hops=2 actually outperform hops=1 on multi-hop queries?
This will either (a) vindicate the paper's 2-hop claim, or (b) reveal that the
reported results were produced with hops=1 and the paper's architectural
description needs reconciliation.

Also serves as a practical guide: should DELTALayer default to hops=2?

Usage:
  # Quick smoke test (3 epochs, 1 seed)
  python experiments/phase66_hop_ablation.py --epochs 5

  # Standard run (matches Phase 42/44 protocol)
  python experiments/phase66_hop_ablation.py --epochs 500 --eval_every 25 --patience 10 --seeds 3

  # Single condition
  python experiments/phase66_hop_ablation.py --conditions hops2 --epochs 500 --patience 10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import copy
import numpy as np
import torch
import torch.nn as nn
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


# ═══════════════════════════════════════════════════════════════════════════
# Hops-Controlled Model Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class HopsControlledEncoder(nn.Module):
    """Wraps a DELTAModel encoder, injecting a pre-built edge adjacency.

    This lets us test different hop depths without modifying DELTALayer or
    build_edge_adjacency(). The key mechanism:

    DeltaGraph._edge_adj_cache = (hop_label, adj_tensor)

    When DELTALayer calls build_edge_adjacency() (default hops=1), it checks:
        if cached_hops >= requested_hops:
            return cached_result

    So caching with hop_label >= 1 causes the cached adj to be returned.
    """

    def __init__(self, delta_encoder: nn.Module, edge_adj: torch.Tensor,
                 cache_hops: int = 2):
        """
        Args:
            delta_encoder: the underlying DELTA encoder (DELTAModel, etc.)
            edge_adj: pre-built [2, E_adj] adjacency tensor (or [2, 0] for empty)
            cache_hops: the hop label to store in the cache (use 2 for hops=2,
                        1 for hops=1; any value >= 1 will be accepted for hops=1
                        default in DELTALayer)
        """
        super().__init__()
        self.delta_encoder = delta_encoder
        self.register_buffer('_prebuilt_adj', edge_adj)
        self.cache_hops = cache_hops

    def forward(self, graph: DeltaGraph) -> DeltaGraph:
        # Inject the pre-built adjacency into the graph's cache.
        # DELTALayer.forward() calls graph.build_edge_adjacency() (hops=1);
        # the cache check (cached_hops >= 1) will return our prebuilt adj.
        graph._edge_adj_cache = (self.cache_hops, self._prebuilt_adj)
        return self.delta_encoder(graph)


class HopsControlledLPModel(LinkPredictionModel):
    """LinkPredictionModel that injects a pre-built edge adj at encode time."""

    def __init__(self, base_model: LinkPredictionModel,
                 prebuilt_adj: torch.Tensor, cache_hops: int):
        # Don't call super().__init__() — share all state with base model
        # instead by copying the __dict__ reference after build.
        # We use composition: wrap the encoder only.
        super().__init__(
            encoder=None,
            num_entities=base_model.num_entities,
            num_relations=base_model.entity_emb.weight.shape[0],  # won't match
            d_node=base_model.d_node,
            d_edge=0,  # placeholder
        )
        # Discard the placeholder state and share base model's state directly
        self.__dict__ = base_model.__dict__.copy()
        self._adj = prebuilt_adj
        self._cache_hops = cache_hops
        # Wrap encoder with hops injection
        if base_model.encoder is not None:
            self._wrapped_encoder = HopsControlledEncoder(
                base_model.encoder, prebuilt_adj, cache_hops)
        else:
            self._wrapped_encoder = None

    def encode(self, edge_index, edge_types, cached_edge_adj=None):
        """Override encode to inject the pre-built adjacency."""
        N = self.entity_emb.num_embeddings
        E = edge_index.shape[1]
        device = self.entity_emb.weight.device

        nf = self.entity_emb.weight  # [N, d_node]
        et_emb = self.edge_rel_emb(edge_types.to(device))  # [E, d_edge]

        if self.needs_proj:
            nf = self.node_proj_in(nf)
            et_emb = self.edge_proj_in(et_emb)

        graph = DeltaGraph(
            node_features=nf,
            edge_features=et_emb,
            edge_index=edge_index.to(device),
        )
        # Inject pre-built adjacency — this is the key ablation control
        graph._edge_adj_cache = (self._cache_hops, self._adj.to(device))

        if self._wrapped_encoder is None:
            return nf  # DistMult baseline — no GNN
        encoded = self._wrapped_encoder.delta_encoder(graph)
        nf_out = encoded.node_features

        if self.needs_proj:
            nf_out = self.node_proj_out(nf_out)
        return nf_out


# ═══════════════════════════════════════════════════════════════════════════
# Pre-build edge adjacencies for all conditions
# ═══════════════════════════════════════════════════════════════════════════

def _cap_adj(adj, max_pairs, seed=0):
    """Randomly subsample adj pairs to at most max_pairs (for CPU testing)."""
    if max_pairs is None or adj.shape[1] <= max_pairs:
        return adj
    rng = torch.Generator()
    rng.manual_seed(seed)
    idx = torch.randperm(adj.shape[1], generator=rng)[:max_pairs]
    return adj[:, idx.sort().values]


def build_condition_adjs(edge_index, device, max_adj_pairs=None):
    """Build all three adjacency conditions for the ablation.

    Args:
        edge_index: [2, E] training edge index
        device: torch device
        max_adj_pairs: if set, randomly cap hops=2 adj to this many pairs.
            Use for CPU smoke tests only — full run should use None (GPU).
            Note: hops=1 on FB15k-237 N=500 gives ~1.5M pairs;
                  hops=2 gives ~28M pairs (3.5GB tensors, needs GPU).

    Returns dict: condition_name → (adj_tensor, cache_hops)
    """
    import warnings
    N = edge_index.max().item() + 1
    E = edge_index.shape[1]

    # Use a temporary graph to call build_edge_adjacency
    tmp = DeltaGraph(
        node_features=torch.zeros(N, 1, device=device),
        edge_features=torch.zeros(E, 1, device=device),
        edge_index=edge_index.to(device),
    )

    print("  Building edge adjacencies...")
    t0 = time.time()

    adj_1hop = tmp.build_edge_adjacency(hops=1)
    t1 = time.time()
    print(f"    hops=1: {adj_1hop.shape[1]:,} pairs [{t1-t0:.1f}s]")

    tmp._edge_adj_cache = None  # clear cache before building hops=2
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Sparse invariant checks')
        adj_2hop = tmp.build_edge_adjacency(hops=2)
    t2 = time.time()
    raw_count = adj_2hop.shape[1]
    adj_2hop = _cap_adj(adj_2hop, max_adj_pairs)
    cap_note = f" [capped from {raw_count:,}]" if max_adj_pairs and adj_2hop.shape[1] < raw_count else ""
    print(f"    hops=2: {adj_2hop.shape[1]:,} pairs [{t2-t1:.1f}s]{cap_note}")

    adj_empty = torch.zeros(2, 0, dtype=torch.long, device=device)
    print(f"    node_only: 0 pairs (edge attention disabled)")

    return {
        'node_only': (adj_empty, 1),   # cache_hops=1 so check passes; empty adj
        'hops1':     (adj_1hop, 1),
        'hops2':     (adj_2hop, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Training — wraps phase42 train_model but injects controlled adj
# ═══════════════════════════════════════════════════════════════════════════

def train_with_controlled_adj(condition_name, adj_tensor, cache_hops,
                               data, epochs, lr, device, batch_size, seed,
                               eval_every=25, patience=10):
    """Train DELTA-Matched with a fixed edge adjacency condition.

    Returns (trained_model, best_val_mrr, edge_index, edge_types, elapsed)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build base model
    base_model = create_lp_model(
        'delta_matched', data['num_entities'], data['num_relations'])
    base_model = base_model.to(device)

    # Count params
    n_params = sum(p.numel() for p in base_model.parameters())
    n_enc = sum(p.numel() for p in base_model.encoder.parameters()) \
        if base_model.encoder is not None else 0
    print(f"\n  [DELTA-Matched / {condition_name}] seed={seed}, "
          f"{n_params:,} params ({n_enc:,} encoder)")

    edge_index, edge_types = build_train_graph_tensors(data['train'])

    # Pre-move adj to device
    adj_dev = adj_tensor.to(device)

    optimizer = torch.optim.Adam(base_model.parameters(), lr=lr)

    best_val_mrr = 0.0
    best_state = None
    evals_no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Train with the controlled adjacency
        loss = _train_epoch_with_adj(
            base_model, data['train'], edge_index, edge_types,
            optimizer, device, batch_size, adj_dev, cache_hops)

        if epoch % eval_every == 0 or epoch == epochs:
            val = _evaluate_with_adj(
                base_model, data['val'], edge_index, edge_types,
                data['hr_to_tails'], data['rt_to_heads'], device,
                adj_dev, cache_hops)
            elapsed = time.time() - t0
            print(f"    Ep {epoch:4d}  loss={loss:.4f}  "
                  f"val_MRR={val['MRR']:.4f}  val_H@10={val['Hits@10']:.4f}"
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
    return base_model, best_val_mrr, edge_index, edge_types, elapsed, n_params


def _encode_with_adj(model, edge_index, edge_types, adj, cache_hops, device):
    """Encode graph with a specific pre-built adjacency injected."""
    N = model.num_entities
    E = edge_index.shape[1]
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
    graph._edge_adj_cache = (cache_hops, adj)

    if model.encoder is None:
        nf_out = nf
    else:
        encoded = model.encoder(graph)
        nf_out = encoded.node_features

    if model.needs_proj:
        nf_out = model.node_proj_out(nf_out)
    return nf_out


def _train_epoch_with_adj(model, train_triples, edge_index, edge_types,
                           optimizer, device, batch_size, adj, cache_hops,
                           label_smoothing=0.1):
    """One training epoch with controlled edge adjacency."""
    import torch.nn.functional as F

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

        node_feats = _encode_with_adj(model, edge_index, edge_types,
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
                        batch_size=128):
    """Filtered link prediction evaluation with controlled adjacency."""
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
            true_tails = hr_to_tails.get((hi, ri), set())
            for tt in true_tails:
                if tt != ti:
                    scores_t[i, tt] = float('-inf')
            rank = int((scores_t[i] >= scores_t[i, ti]).sum().item())
            all_ranks.append(max(rank, 1))

        scores_h = model.score_all_heads(node_feats, r, t)
        for i in range(B):
            hi, ri, ti = h[i].item(), r[i].item(), t[i].item()
            true_heads = rt_to_heads.get((ri, ti), set())
            for th_id in true_heads:
                if th_id != hi:
                    scores_h[i, th_id] = float('-inf')
            rank = int((scores_h[i] >= scores_h[i, hi]).sum().item())
            all_ranks.append(max(rank, 1))

    ranks = np.array(all_ranks, dtype=np.float64)
    return {
        'MRR': float(np.mean(1.0 / ranks)),
        'Hits@1': float(np.mean(ranks <= 1)),
        'Hits@3': float(np.mean(ranks <= 3)),
        'Hits@10': float(np.mean(ranks <= 10)),
    }


@torch.no_grad()
def _evaluate_multihop_with_adj(model, queries_by_type, edge_index, edge_types,
                                  full_hr2t, device, adj, cache_hops,
                                  temperature=1.0, batch_size=64):
    """Multi-hop evaluation with controlled adjacency."""
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
            'MRR': float(np.mean(1.0 / ranks)),
            'Hits@1': float(np.mean(ranks <= 1)),
            'Hits@3': float(np.mean(ranks <= 3)),
            'Hits@10': float(np.mean(ranks <= 10)),
            'count': len(queries),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Single condition run
# ═══════════════════════════════════════════════════════════════════════════

def run_condition(condition_name, adj_tensor, cache_hops,
                  data, queries, full_hr2t,
                  epochs, lr, device, batch_size, seed,
                  eval_every, patience):
    """Run one (condition, seed) pair. Returns metrics dict."""
    model, best_val, edge_index, edge_types, elapsed, n_params = \
        train_with_controlled_adj(
            condition_name, adj_tensor, cache_hops,
            data, epochs, lr, device, batch_size, seed,
            eval_every, patience)

    adj_dev = adj_tensor.to(device)

    # Standard LP test
    lp_test = _evaluate_with_adj(
        model, data['test'], edge_index, edge_types,
        data['hr_to_tails'], data['rt_to_heads'], device,
        adj_dev, cache_hops)
    print(f"    LP test: MRR={lp_test['MRR']:.4f}  "
          f"H@10={lp_test['Hits@10']:.4f}")

    # Multi-hop evaluation
    mh = _evaluate_multihop_with_adj(
        model, queries, edge_index, edge_types,
        full_hr2t, device, adj_dev, cache_hops)

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
        'best_val_MRR': best_val,
        'lp_MRR': lp_test['MRR'],
        'lp_H1': lp_test['Hits@1'],
        'lp_H3': lp_test['Hits@3'],
        'lp_H10': lp_test['Hits@10'],
        **{f'{qt}_MRR': mh[qt]['MRR'] for qt in ['1p', '2p', '3p']},
        **{f'{qt}_H10': mh[qt]['Hits@10'] for qt in ['1p', '2p', '3p']},
        **{f'{qt}_count': mh[qt]['count'] for qt in ['1p', '2p', '3p']},
    }


# ═══════════════════════════════════════════════════════════════════════════
# Summary printing
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(all_results):
    """Print ablation summary table."""
    if not all_results:
        return

    # Aggregate by condition
    from itertools import groupby
    conditions_order = ['node_only', 'hops1', 'hops2']
    by_condition = defaultdict(list)
    for r in all_results:
        by_condition[r['condition']].append(r)

    print("\n" + "=" * 95)
    print("PHASE 66: 1-HOP vs 2-HOP EDGE ADJACENCY ABLATION")
    print("  Hypothesis: hops=2 provides measurably better multi-hop MRR than hops=1")
    print("  DELTA-Matched on FB15k-237 top-500 subgraph")
    print("=" * 95)

    header = (f"{'Condition':<12} {'Seeds':>5} {'LP MRR':>8} {'1p MRR':>8} "
              f"{'2p MRR':>8} {'3p MRR':>8} {'3p H@10':>8} {'2p→3p':>8}")
    print(f"\n{header}")
    print("-" * 95)

    for cond in conditions_order:
        results = by_condition.get(cond, [])
        if not results:
            continue
        n = len(results)

        def mean_std(key):
            vals = [r[key] for r in results]
            m = np.mean(vals)
            s = np.std(vals) if n > 1 else 0.0
            return m, s

        def fmt(key):
            m, s = mean_std(key)
            return f"{m:.4f}" if s == 0 else f"{m:.3f}±{s:.3f}"

        lp_m, _ = mean_std('lp_MRR')
        p1_m, _ = mean_std('1p_MRR')
        p2_m, _ = mean_std('2p_MRR')
        p3_m, _ = mean_std('3p_MRR')
        p3h10_m, _ = mean_std('3p_H10')
        delta_23 = p3_m - p2_m

        print(f"{cond:<12} {n:>5} {fmt('lp_MRR'):>8} {fmt('1p_MRR'):>8} "
              f"{fmt('2p_MRR'):>8} {fmt('3p_MRR'):>8} "
              f"{fmt('3p_H10'):>8} {delta_23:>+8.4f}")

    print("-" * 95)

    # Gap analysis
    h1 = by_condition.get('hops1', [])
    h2 = by_condition.get('hops2', [])
    no = by_condition.get('node_only', [])

    if h1 and h2:
        g_2p = np.mean([r['2p_MRR'] for r in h2]) - np.mean([r['2p_MRR'] for r in h1])
        g_3p = np.mean([r['3p_MRR'] for r in h2]) - np.mean([r['3p_MRR'] for r in h1])
        print(f"\n  hops=2 vs hops=1:  2p gap={g_2p:+.4f}  3p gap={g_3p:+.4f}")
    if h1 and no:
        g_2p = np.mean([r['2p_MRR'] for r in h1]) - np.mean([r['2p_MRR'] for r in no])
        g_3p = np.mean([r['3p_MRR'] for r in h1]) - np.mean([r['3p_MRR'] for r in no])
        print(f"  hops=1 vs node_only: 2p gap={g_2p:+.4f}  3p gap={g_3p:+.4f}")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--eval_every', type=int, default=25)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--batch_size', type=int, default=4096)
    p.add_argument('--seeds', type=str, default='42',
                   help='Comma-separated seeds, e.g. 42,123,456')
    p.add_argument('--conditions', type=str, default='node_only,hops1,hops2',
                   help='Comma-separated conditions to run')
    p.add_argument('--max_entities', type=int, default=500)
    p.add_argument('--max_queries', type=int, default=10000)
    p.add_argument('--max_adj_pairs', type=int, default=None,
                   help='Cap hops=2 adj pairs for CPU smoke tests '
                        '(e.g. 2000000). Use None for full GPU run.')
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    conditions = [c.strip() for c in args.conditions.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Phase 66: 1-hop vs 2-hop Edge Adjacency Ablation")
    print(f"  device={device}  epochs={args.epochs}  seeds={seeds}")
    print(f"  conditions={conditions}")

    # Load data
    print("\n[1/4] Loading FB15k-237 data...")
    data = load_lp_data('fb15k-237', 'data', max_entities=args.max_entities)

    # Generate multi-hop queries (once, shared across all conditions)
    print("\n[2/4] Generating multi-hop queries...")
    queries = generate_multihop_queries(
        data, max_queries_per_type=args.max_queries, seed=42)
    print(f"  1p: {len(queries['1p'])}, 2p: {len(queries['2p'])}, "
          f"3p: {len(queries['3p'])}")

    issues = audit_queries(queries, data)
    if issues:
        print(f"  WARNING: {len(issues)} leakage issues found!")
        for issue in issues[:5]:
            print(f"    {issue}")
    else:
        print("  Leakage audit: PASSED")

    full_hr2t = build_full_adjacency(data)

    # Pre-build edge adjacencies
    print("\n[3/4] Pre-building edge adjacencies...")
    edge_index, edge_types = build_train_graph_tensors(data['train'])
    if args.max_adj_pairs:
        print(f"  NOTE: hops=2 adj capped at {args.max_adj_pairs:,} pairs "
              f"(CPU smoke test mode; use None for full GPU run)")
    all_adjs = build_condition_adjs(edge_index, device,
                                    max_adj_pairs=args.max_adj_pairs)

    # Run all conditions × seeds
    print("\n[4/4] Running ablation conditions...")
    all_results = []
    t_total = time.time()

    for cond_name in conditions:
        if cond_name not in all_adjs:
            print(f"  Unknown condition: {cond_name}. Skipping.")
            continue
        adj_tensor, cache_hops = all_adjs[cond_name]
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} (cache_hops={cache_hops}, "
              f"adj_pairs={adj_tensor.shape[1]:,})")
        print('='*60)

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

    # Save raw results
    import json
    out_path = os.path.join(os.path.dirname(__file__), '..', 'phase66_output.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to phase66_output.json")


if __name__ == '__main__':
    main()
