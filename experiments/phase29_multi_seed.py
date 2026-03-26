"""
Phase 29: Multi-Seed Evaluation

All 25 previous phases used a single seed (42).  This phase re-runs
the three key result phases (22, 23, 25) with 5 seeds each and
reports mean ± std to establish statistical credibility.

Phase 22: Scale stress test (N=1000, 15% noise)
  Key comparison: Soft Gating vs Old Router vs Full Attention

Phase 23: Realistic KG benchmark (N=2000, 20 relation types)
  Key comparison: DELTA+Gate vs CompGCN vs TransE vs RotatE

Phase 25: Real FB15k-237 on GPU (2000-entity subgraph)
  Key comparison: DELTA+Gate vs CompGCN vs DELTA Edge
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from delta.graph import DeltaGraph
from delta.attention import EdgeAttention, DualParallelAttention
from delta.router import PostAttentionPruner, ImportanceRouter
from delta.utils import create_noisy_kg_benchmark, create_realistic_kg_benchmark


SEEDS = [42, 123, 456, 789, 1024]


# ──────────────────────────────────────────────────────────────────────
#  Phase 22 models
# ──────────────────────────────────────────────────────────────────────

class FullAttentionModel(nn.Module):
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes))

    def forward(self, graph, edge_adj=None, **kw):
        if edge_adj is None:
            edge_adj = graph.build_edge_adjacency()
        return self.classifier(self.edge_attn(graph, edge_adj=edge_adj)), torch.tensor(0.0, device=graph.device)


class OldRouterModel(nn.Module):
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.router = ImportanceRouter(d_node, d_edge)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes))

    def forward(self, graph, edge_adj=None, **kw):
        ns, es = self.router(graph)
        _, edge_mask = self.router.apply_top_k(graph, ns, es,
                                               node_k_ratio=0.5, edge_k_ratio=0.5)
        ef = graph.edge_features * edge_mask.float().unsqueeze(-1)
        g = DeltaGraph(node_features=graph.node_features, edge_features=ef,
                       edge_index=graph.edge_index, node_tiers=graph.node_tiers)
        if edge_adj is None:
            edge_adj = g.build_edge_adjacency()
        return self.classifier(self.edge_attn(g, edge_adj=edge_adj)), torch.tensor(0.0, device=graph.device)


class SoftGatingModel(nn.Module):
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.dual_attn = DualParallelAttention(d_node, d_edge, num_heads)
        self.pruner = PostAttentionPruner(d_node, d_edge, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes))

    def forward(self, graph, edge_adj=None, target_sparsity=0.5, **kw):
        if edge_adj is None:
            edge_adj = graph.build_edge_adjacency()
        result, nw, ew = self.dual_attn(graph, edge_adj=edge_adj, return_weights=True)
        _, eg = self.pruner.compute_importance(result, nw, ew)
        gated, sp = self.pruner.soft_prune(result, eg, target_sparsity=target_sparsity)
        return self.classifier(gated.edge_features), sp


# ──────────────────────────────────────────────────────────────────────
#  Phase 23/25 models
# ──────────────────────────────────────────────────────────────────────

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim=64):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, num_relations))
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, src, tgt, **kw):
        h = self.entity_emb(src)
        t = self.entity_emb(tgt)
        r = self.relation_emb.weight.mean(0, keepdim=True).expand(h.shape[0], -1)
        return self.classifier(torch.cat([h, r, t], dim=-1))


class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, dim=64):
        super().__init__()
        self.half = dim // 2
        self.entity_re = nn.Embedding(num_entities, self.half)
        self.entity_im = nn.Embedding(num_entities, self.half)
        self.relation_phase = nn.Embedding(num_relations, self.half)
        self.classifier = nn.Sequential(
            nn.Linear(self.half * 4, dim), nn.GELU(), nn.Linear(dim, num_relations))
        nn.init.xavier_uniform_(self.entity_re.weight)
        nn.init.xavier_uniform_(self.entity_im.weight)
        nn.init.uniform_(self.relation_phase.weight, -3.14159, 3.14159)

    def forward(self, src, tgt, **kw):
        return self.classifier(torch.cat([
            self.entity_re(src), self.entity_im(src),
            self.entity_re(tgt), self.entity_im(tgt),
        ], dim=-1))


class CompGCN(nn.Module):
    def __init__(self, d_node, d_edge, num_classes, num_relations):
        super().__init__()
        self.rel_embeddings = nn.Embedding(num_relations, d_edge)
        self.msg_fn = nn.Sequential(nn.Linear(d_node * 2 + d_edge, d_node), nn.GELU())
        self.update_fn = nn.GRUCell(d_node, d_node)
        self.classifier = nn.Sequential(
            nn.Linear(d_node * 2 + d_edge, d_edge), nn.GELU(),
            nn.Linear(d_edge, num_classes))

    def forward(self, graph, **kw):
        nf = graph.node_features
        src, tgt = graph.edge_index[0], graph.edge_index[1]
        msg = self.msg_fn(torch.cat([nf[src], nf[tgt], graph.edge_features], dim=-1))
        agg = torch.zeros_like(nf)
        agg.scatter_add_(0, tgt.unsqueeze(-1).expand_as(msg), msg)
        nf_up = self.update_fn(agg, nf)
        return self.classifier(torch.cat([nf_up[src], nf_up[tgt], graph.edge_features], dim=-1))


class DELTAEdge(nn.Module):
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes))

    def forward(self, graph, edge_adj=None, **kw):
        if edge_adj is None:
            edge_adj = graph.build_edge_adjacency()
        return self.classifier(self.edge_attn(graph, edge_adj=edge_adj))


class DELTAGate(nn.Module):
    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.dual_attn = DualParallelAttention(d_node, d_edge, num_heads)
        self.pruner = PostAttentionPruner(d_node, d_edge, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes))

    def forward(self, graph, edge_adj=None, target_sparsity=0.5, **kw):
        if edge_adj is None:
            edge_adj = graph.build_edge_adjacency()
        r, nw, ew = self.dual_attn(graph, edge_adj=edge_adj, return_weights=True)
        _, eg = self.pruner.compute_importance(r, nw, ew)
        g, sp = self.pruner.soft_prune(r, eg, target_sparsity=target_sparsity)
        return self.classifier(g.edge_features), sp


# ──────────────────────────────────────────────────────────────────────
#  Training functions
# ──────────────────────────────────────────────────────────────────────

def train_graph_model(model, graph, labels, train_idx, test_idx,
                      edge_adj=None, epochs=200, lr=1e-3, sparsity_w=0.1):
    """Train a graph-based model (EdgeAttn, DualAttn, CompGCN, etc.)."""
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 0.0
    for epoch in range(epochs):
        model.train()
        result = model(graph, edge_adj=edge_adj)
        if isinstance(result, tuple):
            logits, aux = result
        else:
            logits, aux = result, torch.tensor(0.0, device=device)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx]) + sparsity_w * aux
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                out = model(graph, edge_adj=edge_adj)
                lg = out[0] if isinstance(out, tuple) else out
                acc = (lg.argmax(-1)[test_idx] == labels[test_idx]).float().mean().item()
                best = max(best, acc)
    return best


def train_embedding_model(model, heads, tails, labels, train_idx, test_idx,
                          epochs=200, lr=1e-3):
    """Train TransE/RotatE for relation classification."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 0.0
    for epoch in range(epochs):
        model.train()
        logits = model(heads[train_idx], tails[train_idx])
        loss = F.cross_entropy(logits, labels[train_idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(heads[test_idx], tails[test_idx]).argmax(-1) == labels[test_idx]).float().mean().item()
                best = max(best, acc)
    return best


def fmt(values):
    """Format mean ± std."""
    arr = np.array(values) * 100
    return f"{arr.mean():.1f}% ± {arr.std():.1f}%"


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 29: Multi-Seed Evaluation (5 seeds)")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print()

    d_node, d_edge = 64, 32

    # ══════════════════════════════════════════════════════════════════
    #  Phase 22: Scale Stress Test (N=1000, 15% noise)
    # ══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("PHASE 22 — Scale Stress Test (N=1000, 15% noise)")
    print("=" * 70)

    p22 = defaultdict(list)
    t0 = time.time()

    for si, seed in enumerate(SEEDS):
        print(f"\n  Seed {seed} ({si + 1}/5)")
        graph, labels, meta = create_noisy_kg_benchmark(
            num_entities=1000, num_relations=15, num_triples=5000,
            noise_ratio=0.15, d_node=d_node, d_edge=d_edge, seed=seed)
        num_cls = meta['num_relations']
        tr_idx = meta['train_idx'].to(device)
        te_idx = meta['test_idx'].to(device)
        labels_d = labels.to(device)
        g = DeltaGraph(node_features=graph.node_features.to(device),
                       edge_features=graph.edge_features.to(device),
                       edge_index=graph.edge_index.to(device))
        ea = g.build_edge_adjacency()

        for name, model_fn in [
            ('Full Attn', lambda: FullAttentionModel(d_node, d_edge, num_cls).to(device)),
            ('Old Router', lambda: OldRouterModel(d_node, d_edge, num_cls).to(device)),
            ('Soft Gate', lambda: SoftGatingModel(d_node, d_edge, num_cls).to(device)),
        ]:
            torch.manual_seed(seed)
            model = model_fn()
            acc = train_graph_model(model, g, labels_d, tr_idx, te_idx,
                                    edge_adj=ea, epochs=200)
            p22[name].append(acc)
            print(f"    {name}: {acc:.3f}")

    p22_time = time.time() - t0
    print(f"\n  Phase 22 total time: {p22_time:.0f}s")

    # ══════════════════════════════════════════════════════════════════
    #  Phase 23: Realistic KG Benchmark (N=2000, 20 rel types)
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("PHASE 23 — Realistic KG Benchmark (N=2000, 20 relations)")
    print("=" * 70)

    p23 = defaultdict(list)
    t0 = time.time()

    for si, seed in enumerate(SEEDS):
        print(f"\n  Seed {seed} ({si + 1}/5)")
        graph, labels, meta = create_realistic_kg_benchmark(
            num_entities=2000, num_relations=20, num_triples=8000,
            d_node=d_node, d_edge=d_edge, seed=seed)
        num_cls = meta['num_relations']
        tr_idx = meta['train_idx'].to(device)
        te_idx = meta['test_idx'].to(device)
        labels_d = labels.to(device)
        g = DeltaGraph(node_features=graph.node_features.to(device),
                       edge_features=graph.edge_features.to(device),
                       edge_index=graph.edge_index.to(device))
        heads = g.edge_index[0]
        tails = g.edge_index[1]
        ea = g.build_edge_adjacency()

        for name, model_fn in [
            ('TransE', lambda: TransE(2000, num_cls, dim=d_node).to(device)),
            ('RotatE', lambda: RotatE(2000, num_cls, dim=d_node).to(device)),
            ('CompGCN', lambda: CompGCN(d_node, d_edge, num_cls, num_cls).to(device)),
            ('DELTA Edge', lambda: DELTAEdge(d_node, d_edge, num_cls).to(device)),
            ('DELTA+Gate', lambda: DELTAGate(d_node, d_edge, num_cls).to(device)),
        ]:
            torch.manual_seed(seed)
            model = model_fn()
            if name in ('TransE', 'RotatE'):
                acc = train_embedding_model(model, heads, tails, labels_d,
                                            tr_idx, te_idx, epochs=200)
            else:
                acc = train_graph_model(model, g, labels_d, tr_idx, te_idx,
                                        edge_adj=ea, epochs=200)
            p23[name].append(acc)
            print(f"    {name}: {acc:.3f}")

    p23_time = time.time() - t0
    print(f"\n  Phase 23 total time: {p23_time:.0f}s")

    # ══════════════════════════════════════════════════════════════════
    #  Phase 25: Real FB15k-237 on GPU
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("PHASE 25 — Real FB15k-237 (2000-entity subgraph)")
    print("=" * 70)

    p25 = defaultdict(list)
    t0 = time.time()

    # Load FB15k-237 data once
    try:
        from experiments.phase25_fb15k237_gpu import (
            load_fb15k237_subgraph, CompGCNClassifier,
        )
        # Reuse TransE/RotatE from local scope; CompGCN from Phase 25
        p25_graph, p25_meta = load_fb15k237_subgraph(
            top_entities=2000, d_node=d_node, d_edge=d_edge, seed=42)
        N25 = p25_meta['num_entities']
        num_cls25 = p25_meta['num_relations']
        M25 = p25_meta['num_triples']
        heads25 = p25_meta['heads'].to(device)
        tails25 = p25_meta['tails'].to(device)
        labels25 = p25_meta['labels'].to(device)
        g25 = p25_graph.to(device)

        print(f"  Subgraph: {N25} entities, {num_cls25} relations, {M25} triples")

        # Precompute edge adjacency once (expensive)
        print("  Precomputing edge adjacency...")
        ea25_full = g25.build_edge_adjacency(hops=1)
        MAX_ADJ = 5_000_000
        if ea25_full.shape[1] > MAX_ADJ:
            perm = torch.randperm(ea25_full.shape[1], device=device)[:MAX_ADJ]
            ea25 = ea25_full[:, perm]
            print(f"  Sampled {MAX_ADJ:,} / {ea25_full.shape[1]:,} pairs")
        else:
            ea25 = ea25_full

        for si, seed in enumerate(SEEDS):
            print(f"\n  Seed {seed} ({si + 1}/5)")
            # Vary train/test split per seed
            gen = torch.Generator().manual_seed(seed)
            perm = torch.randperm(M25, generator=gen)
            split_pt = int(M25 * 0.8)
            tr_idx = perm[:split_pt].to(device)
            te_idx = perm[split_pt:].to(device)

            for name, model_fn in [
                ('CompGCN', lambda: CompGCN(d_node, d_edge, num_cls25, num_cls25).to(device)),
                ('DELTA Edge', lambda: DELTAEdge(d_node, d_edge, num_cls25).to(device)),
                ('DELTA+Gate', lambda: DELTAGate(d_node, d_edge, num_cls25).to(device)),
            ]:
                torch.manual_seed(seed)
                model = model_fn()
                acc = train_graph_model(model, g25, labels25, tr_idx, te_idx,
                                        edge_adj=ea25, epochs=200)
                p25[name].append(acc)
                print(f"    {name}: {acc:.3f}")

        p25_time = time.time() - t0
        print(f"\n  Phase 25 total time: {p25_time:.0f}s")

    except Exception as e:
        print(f"  Phase 25 skipped: {e}")
        p25_time = 0

    # ══════════════════════════════════════════════════════════════════
    #  Grand Summary
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("GRAND SUMMARY — Mean ± Std over 5 Seeds")
    print("=" * 70)

    print("\n  Phase 22: Scale Stress Test (N=1000, 15% noise)")
    print(f"  {'Model':<20s}  {'Mean ± Std':>18s}  {'Range':>16s}")
    print(f"  {'-'*20}  {'-'*18}  {'-'*16}")
    for name in ['Full Attn', 'Old Router', 'Soft Gate']:
        vals = p22[name]
        if vals:
            arr = np.array(vals) * 100
            print(f"  {name:<20s}  {fmt(vals):>18s}  "
                  f"[{arr.min():.1f}-{arr.max():.1f}%]")

    print(f"\n  Phase 23: Realistic KG Benchmark (N=2000)")
    print(f"  {'Model':<20s}  {'Mean ± Std':>18s}  {'Range':>16s}")
    print(f"  {'-'*20}  {'-'*18}  {'-'*16}")
    for name in ['TransE', 'RotatE', 'CompGCN', 'DELTA Edge', 'DELTA+Gate']:
        vals = p23[name]
        if vals:
            arr = np.array(vals) * 100
            print(f"  {name:<20s}  {fmt(vals):>18s}  "
                  f"[{arr.min():.1f}-{arr.max():.1f}%]")

    if p25:
        print(f"\n  Phase 25: Real FB15k-237 (2000-entity subgraph)")
        print(f"  {'Model':<20s}  {'Mean ± Std':>18s}  {'Range':>16s}")
        print(f"  {'-'*20}  {'-'*18}  {'-'*16}")
        for name in ['CompGCN', 'DELTA Edge', 'DELTA+Gate']:
            vals = p25[name]
            if vals:
                arr = np.array(vals) * 100
                print(f"  {name:<20s}  {fmt(vals):>18s}  "
                      f"[{arr.min():.1f}-{arr.max():.1f}%]")

    total_time = p22_time + p23_time + p25_time
    print(f"\n  Total runtime: {total_time:.0f}s ({total_time / 60:.1f} min)")

    # Key statistical findings
    print("\n  Key findings:")
    if p22.get('Soft Gate') and p22.get('Old Router'):
        sg = np.array(p22['Soft Gate']) * 100
        or_ = np.array(p22['Old Router']) * 100
        delta = sg.mean() - or_.mean()
        print(f"    Phase 22: Soft Gate - Old Router = {delta:+.1f}%  "
              f"(Soft Gate {sg.mean():.1f}% ± {sg.std():.1f}% vs "
              f"Old Router {or_.mean():.1f}% ± {or_.std():.1f}%)")

    if p23.get('DELTA+Gate') and p23.get('CompGCN'):
        dg = np.array(p23['DELTA+Gate']) * 100
        cg = np.array(p23['CompGCN']) * 100
        delta = dg.mean() - cg.mean()
        print(f"    Phase 23: DELTA+Gate - CompGCN = {delta:+.1f}%  "
              f"(DELTA+Gate {dg.mean():.1f}% ± {dg.std():.1f}% vs "
              f"CompGCN {cg.mean():.1f}% ± {cg.std():.1f}%)")

    if p25.get('DELTA+Gate') and p25.get('CompGCN'):
        dg = np.array(p25['DELTA+Gate']) * 100
        cg = np.array(p25['CompGCN']) * 100
        delta = dg.mean() - cg.mean()
        print(f"    Phase 25: DELTA+Gate - CompGCN = {delta:+.1f}%  "
              f"(DELTA+Gate {dg.mean():.1f}% ± {dg.std():.1f}% vs "
              f"CompGCN {cg.mean():.1f}% ± {cg.std():.1f}%)")


if __name__ == '__main__':
    main()
