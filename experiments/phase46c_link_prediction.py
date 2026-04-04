"""
Phase 46c: Correct Link Prediction Evaluation on FB15k-237

Phase 37 achieved 0.993 test accuracy on FB15k-237 — but this was due to
5 critical evaluation flaws:

  1. Edge features encoded the label (relation_prototype + noise, SNR ~15:1)
  2. Task was relation classification, not link prediction
  3. Test/val edges present in the training graph
  4. Target edges not masked during message passing
  5. No negative sampling — every sample was positive

This phase implements correct link prediction evaluation:
  - Edge features are LEARNED (nn.Embedding), not label-encoding prototypes
  - Task is link prediction: given (h, r, ?), rank all entities
  - Train graph contains ONLY training triples
  - Evaluation uses filtered MRR / Hits@K (standard KGE protocol)
  - DistMult scoring function for fair comparison across all models

Models (same GNN architectures as Phase 37, plus a no-GNN baseline):
  1. DELTA-Full     (d_node=64, d_edge=32, 3 layers)
  2. DELTA-Matched  (d_node=48, d_edge=24, 2 layers)
  3. GraphGPS       (d_node=64, d_edge=32, 3 layers)
  4. GRIT           (d_node=64, d_edge=32, 3 layers)
  5. DistMult       (no GNN — raw entity embeddings only)

  All share the same DistMult decoder + entity/relation embeddings.
  The no-GNN DistMult baseline isolates each GNN's marginal value.

FB15k-237 reference scores (NOT targets to beat):
  DistMult (Dettmers 2018):  MRR ~0.241  H@1 ~0.155  H@10 ~0.419
  CompGCN  (Vashishth 2020): MRR ~0.355  H@1 ~0.264  H@10 ~0.535
  RotatE   (Sun 2019):       MRR ~0.338  H@1 ~0.241  H@10 ~0.533

Usage:
  # Smoke test (2000 entities, 5 epochs, 1 seed) — ~2 min CPU
  python experiments/phase46c_link_prediction.py

  # Full FB15k-237, 3 seeds — GPU recommended
  python experiments/phase46c_link_prediction.py --full --seeds 3

  # Single model
  python experiments/phase46c_link_prediction.py --models delta_full --epochs 20

Requirements:
  pip install torch numpy
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from delta.graph import DeltaGraph
from delta.model import DELTAModel, DELTALayer
from delta.baselines import GraphGPSModel, GRITModel
from delta.datasets import download_dataset, _load_triples


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading — proper train/val/test separation
# ═══════════════════════════════════════════════════════════════════════════

def load_lp_data(name='fb15k-237', data_dir='data', max_entities=None):
    """Load KG for link prediction with STRICT split separation.

    Unlike load_real_kg() (Phase 37), this function:
      ✗ Does NOT merge all splits into one graph
      ✗ Does NOT create edge features from relation prototypes
      ✓ Returns separate train/val/test triple tensors
      ✓ Builds filter dicts from ALL triples (for filtered evaluation)

    Returns dict with keys:
        train, val, test: [3, N] tensors (row 0=head, 1=relation, 2=tail)
        num_entities, num_relations: int
        hr_to_tails: dict (h,r) → set of true tails (all splits)
        rt_to_heads: dict (r,t) → set of true heads (all splits)
    """
    dataset_dir = download_dataset(name, data_dir)

    train_raw = _load_triples(os.path.join(dataset_dir, 'train.txt'))
    val_raw = _load_triples(os.path.join(dataset_dir, 'valid.txt'))
    test_raw = _load_triples(os.path.join(dataset_dir, 'test.txt'))

    all_raw = train_raw + val_raw + test_raw

    # Full vocabularies (sorted for determinism)
    all_entities = sorted({e for h, r, t in all_raw for e in (h, t)})
    all_relations = sorted({r for _, r, _ in all_raw})

    num_entities = len(all_entities)
    num_relations = len(all_relations)

    # Subsample by entity degree (densest subset) for smoke testing
    if max_entities and max_entities < num_entities:
        degree = defaultdict(int)
        for h, r, t in train_raw:
            degree[h] += 1
            degree[t] += 1
        top = sorted(degree.keys(), key=lambda e: degree[e], reverse=True)
        keep = set(top[:max_entities])

        train_raw = [(h, r, t) for h, r, t in train_raw if h in keep and t in keep]
        val_raw = [(h, r, t) for h, r, t in val_raw if h in keep and t in keep]
        test_raw = [(h, r, t) for h, r, t in test_raw if h in keep and t in keep]

        # Rebuild vocabularies from the surviving triples
        alive_ents = sorted({e for h, r, t in (train_raw + val_raw + test_raw)
                             for e in (h, t)})
        alive_rels = sorted({r for _, r, _ in (train_raw + val_raw + test_raw)})
        all_entities = alive_ents
        all_relations = alive_rels
        num_entities = len(all_entities)
        num_relations = len(all_relations)

    entity2id = {e: i for i, e in enumerate(all_entities)}
    relation2id = {r: i for i, r in enumerate(all_relations)}

    def to_ids(triples):
        if not triples:
            return torch.zeros(3, 0, dtype=torch.long)
        h = [entity2id[x[0]] for x in triples]
        r = [relation2id[x[1]] for x in triples]
        t = [entity2id[x[2]] for x in triples]
        return torch.tensor([h, r, t], dtype=torch.long)

    train = to_ids(train_raw)
    val = to_ids(val_raw)
    test = to_ids(test_raw)

    # Filter dicts: ALL triples (for filtered evaluation)
    all_id_triples = torch.cat([train, val, test], dim=1)
    hr_to_tails = defaultdict(set)
    rt_to_heads = defaultdict(set)
    for i in range(all_id_triples.shape[1]):
        h = all_id_triples[0, i].item()
        r = all_id_triples[1, i].item()
        t = all_id_triples[2, i].item()
        hr_to_tails[(h, r)].add(t)
        rt_to_heads[(r, t)].add(h)

    print(f"  Loaded {name} for link prediction:")
    print(f"    {num_entities} entities, {num_relations} relations")
    print(f"    {train.shape[1]} train / {val.shape[1]} val / {test.shape[1]} test")

    return {
        'train': train, 'val': val, 'test': test,
        'num_entities': num_entities,
        'num_relations': num_relations,
        'hr_to_tails': dict(hr_to_tails),
        'rt_to_heads': dict(rt_to_heads),
    }


def build_train_graph_tensors(train_triples):
    """Extract edge_index [2, E] and edge_types [E] from training triples."""
    edge_index = train_triples[[0, 2]]   # head, tail
    edge_types = train_triples[1]         # relation
    return edge_index, edge_types


# ═══════════════════════════════════════════════════════════════════════════
# Link Prediction Model
# ═══════════════════════════════════════════════════════════════════════════

class LinkPredictionModel(nn.Module):
    """GNN encoder + DistMult decoder for link prediction.

    Architecture:
      Entity embs ─┐
                    ├─→ DeltaGraph ─→ GNN encoder ─→ enriched node feats ─┐
      Relation embs┘                                                       │
                                                                           ▼
      Decoder: score(h, r, t) = (h_enc · r_dec · t_enc).sum()  [DistMult]

    Key difference from Phase 37:
      - Edge features are LEARNED nn.Embedding (no label leakage)
      - Scoring is DistMult on node features (not edge classification)
      - encoder=None → pure DistMult baseline (no GNN)
    """

    def __init__(self, encoder, num_entities, num_relations, d_node, d_edge,
                 encoder_d_node=None, encoder_d_edge=None):
        super().__init__()
        self.d_node = d_node
        self.num_entities = num_entities

        # Learnable embeddings
        self.entity_emb = nn.Embedding(num_entities, d_node)
        self.edge_rel_emb = nn.Embedding(num_relations, d_edge)
        self.decoder_rel_emb = nn.Embedding(num_relations, d_node)

        # Input/output projections for dimension-mismatched encoders
        enc_dn = encoder_d_node or d_node
        enc_de = encoder_d_edge or d_edge
        self.needs_proj = (enc_dn != d_node or enc_de != d_edge)
        if self.needs_proj:
            self.node_proj_in = nn.Linear(d_node, enc_dn)
            self.edge_proj_in = nn.Linear(d_edge, enc_de)
            self.node_proj_out = nn.Linear(enc_dn, d_node)

        self.encoder = encoder  # None = pure DistMult (no GNN)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.edge_rel_emb.weight)
        nn.init.xavier_uniform_(self.decoder_rel_emb.weight)

    def encode(self, edge_index, edge_types):
        """Run GNN on train graph → enriched entity features [N, d_node].

        If encoder is None (DistMult baseline), returns raw entity embeddings.
        """
        nf = self.entity_emb.weight

        if self.encoder is None:
            return nf

        ef = self.edge_rel_emb(edge_types)

        if self.needs_proj:
            nf = self.node_proj_in(nf)
            ef = self.edge_proj_in(ef)

        graph = DeltaGraph(
            node_features=nf,
            edge_features=ef,
            edge_index=edge_index,
        )
        encoded = self.encoder(graph)
        nf_out = encoded.node_features

        if self.needs_proj:
            nf_out = self.node_proj_out(nf_out)
        return nf_out

    def score_all_tails(self, node_feats, h, r):
        """Score (h, r, ?) for all entities. Returns [B, N]."""
        hr = node_feats[h] * self.decoder_rel_emb(r)   # [B, d]
        return hr @ node_feats.t()                       # [B, N]

    def score_all_heads(self, node_feats, r, t):
        """Score (?, r, t) for all entities. Returns [B, N]."""
        rt = self.decoder_rel_emb(r) * node_feats[t]   # [B, d]
        return rt @ node_feats.t()                       # [B, N]


# ═══════════════════════════════════════════════════════════════════════════
# Self-Bootstrap Encoder (Phase 46b insight)
# ═══════════════════════════════════════════════════════════════════════════

class SelfBootstrapDELTAEncoder(nn.Module):
    """Two-stage DELTA encoder inspired by Phase 46b.

    Stage 1: Lightweight DELTA (1 layer) on the raw KG -> enriched features.
    Stage 2: Full DELTA (N layers) on the SAME graph with enriched input.

    This gives DELTA a 'warm start': by the time Stage 2's edge-to-edge
    attention runs, node/edge features already carry relational context from
    Stage 1. Solves the cold-start problem that makes DELTA converge slowly
    versus MPNN-based models.
    """

    def __init__(self, d_node, d_edge, bootstrap_layers=1, delta_layers=2,
                 num_heads=4, dropout=0.1):
        super().__init__()
        self.bootstrap_layers = nn.ModuleList([
            DELTALayer(d_node, d_edge, num_heads, dropout=dropout)
            for _ in range(bootstrap_layers)
        ])
        # Feature bridge: project stage-1 outputs into fresh stage-2 inputs
        # (prevents gradient shortcut where stage 2 ignores stage 1)
        self.node_bridge = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, d_node),
            nn.GELU(),
        )
        self.edge_bridge = nn.Sequential(
            nn.LayerNorm(d_edge),
            nn.Linear(d_edge, d_edge),
            nn.GELU(),
        )
        self.delta_layers = nn.ModuleList([
            DELTALayer(d_node, d_edge, num_heads, dropout=dropout)
            for _ in range(delta_layers)
        ])

    def forward(self, graph):
        # Stage 1: bootstrap pass
        for layer in self.bootstrap_layers:
            graph = layer(graph, use_router=False,
                         use_partitioning=False, use_memory=False)

        # Bridge: transform features between stages
        graph = DeltaGraph(
            node_features=self.node_bridge(graph.node_features),
            edge_features=self.edge_bridge(graph.edge_features),
            edge_index=graph.edge_index,
        )

        # Stage 2: full DELTA pass
        for layer in self.delta_layers:
            graph = layer(graph, use_router=False,
                         use_partitioning=False, use_memory=False)

        return graph


# ═══════════════════════════════════════════════════════════════════════════
# Model Factories
# ═══════════════════════════════════════════════════════════════════════════

ALL_MODELS = ['delta_full', 'delta_matched', 'graphgps', 'grit', 'distmult',
              'self_bootstrap', 'self_bootstrap_hybrid']


def create_lp_model(model_type, num_entities, num_relations,
                     d_node=64, d_edge=32):
    """Create a LinkPredictionModel for the specified GNN encoder type."""

    if model_type == 'delta_full':
        enc = DELTAModel(d_node=d_node, d_edge=d_edge,
                         num_layers=3, num_heads=4)
        return LinkPredictionModel(enc, num_entities, num_relations,
                                   d_node, d_edge)

    elif model_type == 'delta_matched':
        md, me = 48, 24
        enc = DELTAModel(d_node=md, d_edge=me, num_layers=2, num_heads=4)
        return LinkPredictionModel(enc, num_entities, num_relations,
                                   d_node, d_edge,
                                   encoder_d_node=md, encoder_d_edge=me)

    elif model_type == 'graphgps':
        enc = GraphGPSModel(d_node=d_node, d_edge=d_edge,
                            num_layers=3, num_heads=4)
        return LinkPredictionModel(enc, num_entities, num_relations,
                                   d_node, d_edge)

    elif model_type == 'grit':
        enc = GRITModel(d_node=d_node, d_edge=d_edge,
                        num_layers=3, num_heads=4)
        return LinkPredictionModel(enc, num_entities, num_relations,
                                   d_node, d_edge)

    elif model_type == 'distmult':
        # No GNN — pure DistMult baseline (entity embeddings only)
        return LinkPredictionModel(None, num_entities, num_relations,
                                   d_node, d_edge)

    elif model_type == 'self_bootstrap':
        # Phase 46b: 2-stage DELTA (1 bootstrap layer + 2 full layers)
        enc = SelfBootstrapDELTAEncoder(
            d_node=d_node, d_edge=d_edge,
            bootstrap_layers=1, delta_layers=2, num_heads=4)
        return LinkPredictionModel(enc, num_entities, num_relations,
                                   d_node, d_edge)

    elif model_type == 'self_bootstrap_hybrid':
        # Phase 46b hybrid: 1 bootstrap + 3 full layers (same depth as delta_full)
        enc = SelfBootstrapDELTAEncoder(
            d_node=d_node, d_edge=d_edge,
            bootstrap_layers=1, delta_layers=3, num_heads=4)
        return LinkPredictionModel(enc, num_entities, num_relations,
                                   d_node, d_edge)

    else:
        raise ValueError(f"Unknown model: {model_type}. Choose from {ALL_MODELS}")


# ═══════════════════════════════════════════════════════════════════════════
# Training — 1-vs-all BCE loss
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(model, train_triples, edge_index, edge_types,
                optimizer, device, batch_size=512, label_smoothing=0.1):
    """One training epoch with 1-vs-all DistMult scoring + BCE loss.

    For each batch of (h, r, t):
      - Tail: score(h, r, e) for all e, BCE against one-hot at t
      - Head: score(e, r, t) for all e, BCE against one-hot at h
      - Label smoothing prevents overconfident embeddings
    """
    model.train()
    n = train_triples.shape[1]
    perm = torch.randperm(n)
    total_loss = 0.0
    num_batches = 0

    ei = edge_index.to(device)
    et = edge_types.to(device)

    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        h = train_triples[0, idx].to(device)
        r = train_triples[1, idx].to(device)
        t = train_triples[2, idx].to(device)
        B = h.shape[0]
        N = model.num_entities

        # Encode (gradients flow through GNN)
        node_feats = model.encode(ei, et)

        # --- Tail prediction ---
        scores_t = model.score_all_tails(node_feats, h, r)   # [B, N]
        targets_t = torch.zeros(B, N, device=device)
        targets_t[torch.arange(B, device=device), t] = 1.0
        if label_smoothing > 0:
            targets_t = targets_t * (1 - label_smoothing) + label_smoothing / N
        loss_t = F.binary_cross_entropy_with_logits(scores_t, targets_t)

        # --- Head prediction ---
        scores_h = model.score_all_heads(node_feats, r, t)   # [B, N]
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


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation — filtered ranking
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_lp(model, triples, edge_index, edge_types,
                hr_to_tails, rt_to_heads, device, batch_size=128):
    """Filtered MRR / Hits@K evaluation (standard KGE protocol).

    For each triple (h, r, t):
      Tail: rank t among score(h, r, e) for all e (filter known true tails)
      Head: rank h among score(e, r, t) for all e (filter known true heads)
    """
    model.eval()
    n = triples.shape[1]
    if n == 0:
        return {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0}

    ei = edge_index.to(device)
    et = edge_types.to(device)

    # Encode once for evaluation (no gradient needed)
    node_feats = model.encode(ei, et)

    all_ranks = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        h = triples[0, start:end].to(device)
        r = triples[1, start:end].to(device)
        t = triples[2, start:end].to(device)
        B = h.shape[0]

        # --- Tail prediction ---
        scores_t = model.score_all_tails(node_feats, h, r)   # [B, N]
        for i in range(B):
            hi, ri, ti = h[i].item(), r[i].item(), t[i].item()
            true_tails = hr_to_tails.get((hi, ri), set())
            for tt in true_tails:
                if tt != ti:
                    scores_t[i, tt] = float('-inf')
            rank = int((scores_t[i] >= scores_t[i, ti]).sum().item())
            all_ranks.append(max(rank, 1))

        # --- Head prediction ---
        scores_h = model.score_all_heads(node_feats, r, t)   # [B, N]
        for i in range(B):
            hi, ri, ti = h[i].item(), r[i].item(), t[i].item()
            true_heads = rt_to_heads.get((ri, ti), set())
            for th in true_heads:
                if th != hi:
                    scores_h[i, th] = float('-inf')
            rank = int((scores_h[i] >= scores_h[i, hi]).sum().item())
            all_ranks.append(max(rank, 1))

    ranks = np.array(all_ranks, dtype=np.float64)
    return {
        'MRR': float(np.mean(1.0 / ranks)),
        'Hits@1': float(np.mean(ranks <= 1)),
        'Hits@3': float(np.mean(ranks <= 3)),
        'Hits@10': float(np.mean(ranks <= 10)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Single model run
# ═══════════════════════════════════════════════════════════════════════════

def run_single(model_type, data, epochs, lr, device, batch_size, seed,
               eval_every=10, patience=5, weight_decay=0.0):
    """Train + evaluate one model on one seed. Returns metrics dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = create_lp_model(model_type,
                            data['num_entities'], data['num_relations'])
    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_encoder = (sum(p.numel() for p in model.encoder.parameters())
                 if model.encoder is not None else 0)
    print(f"\n  [{model_type}] seed={seed}, "
          f"{n_total:,} total params ({n_encoder:,} encoder), device={device}"
          + (f", wd={weight_decay}" if weight_decay > 0 else ""))

    edge_index, edge_types = build_train_graph_tensors(data['train'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    best_val_mrr = 0.0
    best_test = None
    evals_no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data['train'], edge_index, edge_types,
                           optimizer, device, batch_size)

        if epoch % eval_every == 0 or epoch == epochs:
            val = evaluate_lp(model, data['val'], edge_index, edge_types,
                              data['hr_to_tails'], data['rt_to_heads'], device)
            elapsed = time.time() - t0
            print(f"    Ep {epoch:4d}  loss={loss:.4f}  "
                  f"val_MRR={val['MRR']:.4f}  val_H@10={val['Hits@10']:.4f}  "
                  f"[{elapsed:.0f}s]")

            if val['MRR'] > best_val_mrr:
                best_val_mrr = val['MRR']
                evals_no_improve = 0
                best_test = evaluate_lp(
                    model, data['test'], edge_index, edge_types,
                    data['hr_to_tails'], data['rt_to_heads'], device)
            else:
                evals_no_improve += 1
                if patience > 0 and evals_no_improve >= patience:
                    print(f"    Early stopping at epoch {epoch}")
                    break

    elapsed = time.time() - t0
    if best_test is None:
        best_test = {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0}

    print(f"    Done: val_MRR={best_val_mrr:.4f}  "
          f"test_MRR={best_test['MRR']:.4f}  "
          f"test_H@10={best_test['Hits@10']:.4f}  [{elapsed:.0f}s]")

    return {
        'model': model_type,
        'seed': seed,
        'params_total': n_total,
        'params_encoder': n_encoder,
        'best_val_MRR': best_val_mrr,
        **{f'test_{k}': v for k, v in best_test.items()},
        'time_s': elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Multi-seed runner + summary
# ═══════════════════════════════════════════════════════════════════════════

def run_multi_seed(model_type, data, args, device, seeds):
    """Run one model across multiple seeds, return aggregated metrics."""
    results = []
    wd = getattr(args, 'weight_decay', 0.0)
    for seed in seeds:
        try:
            r = run_single(model_type, data, args.epochs, args.lr, device,
                           args.batch_size, seed, args.eval_every, args.patience,
                           weight_decay=wd)
            results.append(r)
        except Exception as e:
            print(f"    FAILED seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        return None

    metrics = ['test_MRR', 'test_Hits@1', 'test_Hits@3', 'test_Hits@10']
    agg = {
        'model': model_type,
        'params_total': results[0]['params_total'],
        'params_encoder': results[0]['params_encoder'],
        'num_seeds': len(results),
    }
    for m in metrics:
        vals = [r[m] for r in results]
        agg[f'{m}_mean'] = float(np.mean(vals))
        agg[f'{m}_std'] = float(np.std(vals)) if len(vals) > 1 else 0.0
    agg['time_mean'] = float(np.mean([r['time_s'] for r in results]))
    return agg


def print_summary(all_results):
    """Print final comparison table."""
    valid = [r for r in all_results if r is not None]
    if not valid:
        print("\nNo results to summarize.")
        return

    print("\n" + "=" * 90)
    print("PHASE 46c: LINK PREDICTION RESULTS — FILTERED RANKING")
    print("=" * 90)

    header = (f"{'Model':<18} {'Total':>8} {'Enc':>7} "
              f"{'MRR':>14} {'H@1':>14} {'H@3':>14} {'H@10':>14}")
    print(f"\n{header}")
    print("-" * 90)

    for r in valid:
        def fmt(key):
            m = r[f'{key}_mean']
            s = r[f'{key}_std']
            return f"{m:.4f}±{s:.3f}" if s > 0 else f"{m:.4f}      "

        print(f"{r['model']:<18} {r['params_total']:>8,} {r['params_encoder']:>7,} "
              f"{fmt('test_MRR'):>14} {fmt('test_Hits@1'):>14} "
              f"{fmt('test_Hits@3'):>14} {fmt('test_Hits@10'):>14}")

    print("-" * 90)
    print("\nReference (full FB15k-237, published results):")
    print("  DistMult: MRR 0.241  H@1 0.155  H@3 -----  H@10 0.419")
    print("  CompGCN:  MRR 0.355  H@1 0.264  H@3 0.390  H@10 0.535")
    print("  RotatE:   MRR 0.338  H@1 0.241  H@3 0.375  H@10 0.533")

    # Comparative analysis
    if len(valid) >= 2:
        by_mrr = sorted(valid, key=lambda x: x['test_MRR_mean'], reverse=True)
        print(f"\n  Best model: {by_mrr[0]['model']} "
              f"(MRR {by_mrr[0]['test_MRR_mean']:.4f})")

        dm = next((r for r in valid if r['model'] == 'delta_matched'), None)
        baselines = [r for r in valid if r['model'] in ('graphgps', 'grit')]
        if dm and baselines:
            best_bl = max(baselines, key=lambda x: x['test_MRR_mean'])
            diff = dm['test_MRR_mean'] - best_bl['test_MRR_mean']
            print(f"  DELTA-Matched vs {best_bl['model']}: "
                  f"{'+'if diff > 0 else ''}{diff:.4f} MRR")

        df = next((r for r in valid if r['model'] == 'delta_full'), None)
        if df and dm:
            diff = df['test_MRR_mean'] - dm['test_MRR_mean']
            print(f"  DELTA-Full vs DELTA-Matched: "
                  f"{'+'if diff > 0 else ''}{diff:.4f} MRR")

        no_gnn = next((r for r in valid if r['model'] == 'distmult'), None)
        if no_gnn:
            gnn_models = [r for r in valid if r['model'] != 'distmult']
            if gnn_models:
                best_gnn = max(gnn_models, key=lambda x: x['test_MRR_mean'])
                lift = best_gnn['test_MRR_mean'] - no_gnn['test_MRR_mean']
                print(f"  GNN lift over raw DistMult: "
                      f"{'+'if lift > 0 else ''}{lift:.4f} MRR "
                      f"(best GNN: {best_gnn['model']})")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Phase 46c: Correct link prediction evaluation')
    parser.add_argument('--full', action='store_true',
                        help='Full FB15k-237 (default: top-2000 entities)')
    parser.add_argument('--max_entities', type=int, default=2000,
                        help='Entity limit for subset mode (ignored with --full)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Training epochs (default: 5 subset / 200 full)')
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of random seeds')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_every', type=int, default=None,
                        help='Evaluate every N epochs (default: auto)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (eval intervals)')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model names (default: all)')
    parser.add_argument('--dataset', type=str, default='fb15k-237',
                        choices=['fb15k-237', 'wn18rr'])
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='L2 weight decay for Adam optimizer (default: 0.0)')
    args = parser.parse_args()

    # Defaults based on mode
    if args.epochs is None:
        args.epochs = 200 if args.full else 5
    if args.eval_every is None:
        args.eval_every = 25 if args.full else 1

    max_ent = None if args.full else args.max_entities
    models = args.models.split(',') if args.models else ALL_MODELS
    seeds = list(range(1, args.seeds + 1))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Phase 46c: Correct Link Prediction Evaluation")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Mode:     {'FULL' if args.full else f'subset (top {args.max_entities} entities by degree)'}")
    print(f"  Models:   {models}")
    print(f"  Seeds:    {seeds}")
    print(f"  Epochs:   {args.epochs} (eval every {args.eval_every}, patience {args.patience})")
    print(f"  Device:   {device}")
    print()

    data = load_lp_data(args.dataset, max_entities=max_ent)

    all_results = []
    for model_type in models:
        print(f"\n{'─' * 60}")
        print(f"  Model: {model_type}")
        print(f"{'─' * 60}")
        agg = run_multi_seed(model_type, data, args, device, seeds)
        all_results.append(agg)

    print_summary(all_results)


if __name__ == '__main__':
    main()
