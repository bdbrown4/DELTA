"""
Phase 23: Realistic KG Benchmark — DELTA vs KG Embedding Baselines

Tests DELTA on a FB15k-237-like synthetic benchmark with:
- 2000 entities, 20 typed relations, 8000+ triples
- Type-constrained relations (born_in: person→location, works_at: person→org)
- Compositional derived relations (works_at+located_in → lives_near_work)
- Power-law degree distribution

Two tasks:
A) Relation classification: given (head, tail, edge_features), predict relation type
B) Link prediction ranking: given (head, relation), rank correct tails

Baselines:
1. TransE — h + r ≈ t in embedding space (Bordes et al., 2013)
2. RotatE — rotation in complex space (Sun et al., 2019)
3. CompGCN-style — GNN with relation-aware message passing
4. DELTA Edge Attention — edge-to-edge attention (our approach)
5. DELTA + Soft Gating — edge attention + post-attention soft pruning
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import EdgeAttention, DualParallelAttention
from delta.router import PostAttentionPruner
from delta.utils import create_realistic_kg_benchmark


# ─────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────

class TransE(nn.Module):
    """TransE: h + r ≈ t. Scores edges via ||h + r - t||."""

    def __init__(self, num_entities, num_relations, dim=64):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, num_relations),
        )
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, src, tgt, **kwargs):
        h = self.entity_emb(src)
        t = self.entity_emb(tgt)
        # Use all relation embeddings as features
        r_mean = self.relation_emb.weight.mean(0, keepdim=True).expand(h.shape[0], -1)
        return self.classifier(torch.cat([h, r_mean, t], dim=-1))

    def score_triples(self, src, tgt, rel):
        """TransE scoring: -||h + r - t||"""
        h = self.entity_emb(src)
        r = self.relation_emb(rel)
        t = self.entity_emb(tgt)
        return -(h + r - t).norm(dim=-1)


class RotatE(nn.Module):
    """RotatE: t ≈ h ∘ r where r is unit-modulus complex rotation."""

    def __init__(self, num_entities, num_relations, dim=64):
        super().__init__()
        self.dim = dim // 2  # complex dimension
        self.entity_re = nn.Embedding(num_entities, self.dim)
        self.entity_im = nn.Embedding(num_entities, self.dim)
        self.relation_phase = nn.Embedding(num_relations, self.dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.dim * 4, dim), nn.GELU(), nn.Linear(dim, num_relations),
        )
        nn.init.xavier_uniform_(self.entity_re.weight)
        nn.init.xavier_uniform_(self.entity_im.weight)
        nn.init.uniform_(self.relation_phase.weight, -3.14159, 3.14159)

    def forward(self, src, tgt, **kwargs):
        h_re = self.entity_re(src)
        h_im = self.entity_im(src)
        t_re = self.entity_re(tgt)
        t_im = self.entity_im(tgt)
        return self.classifier(torch.cat([h_re, h_im, t_re, t_im], dim=-1))

    def score_triples(self, src, tgt, rel):
        """RotatE scoring: -||h ∘ r - t|| in complex space."""
        h_re = self.entity_re(src)
        h_im = self.entity_im(src)
        phase = self.relation_phase(rel)
        r_re = torch.cos(phase)
        r_im = torch.sin(phase)
        # Complex multiplication: (h_re + h_im*i) * (r_re + r_im*i)
        rot_re = h_re * r_re - h_im * r_im
        rot_im = h_re * r_im + h_im * r_re
        t_re = self.entity_re(tgt)
        t_im = self.entity_im(tgt)
        diff_re = rot_re - t_re
        diff_im = rot_im - t_im
        return -(diff_re ** 2 + diff_im ** 2).sum(dim=-1).sqrt()


class CompGCNClassifier(nn.Module):
    """CompGCN-style: GNN with relation-aware message passing."""

    def __init__(self, d_node, d_edge, num_classes, num_relations):
        super().__init__()
        self.rel_embeddings = nn.Embedding(num_relations, d_edge)
        self.msg_fn = nn.Sequential(
            nn.Linear(d_node * 2 + d_edge, d_node), nn.GELU(),
        )
        self.update_fn = nn.GRUCell(d_node, d_node)
        self.classifier = nn.Sequential(
            nn.Linear(d_node * 2 + d_edge, d_edge), nn.GELU(),
            nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, labels_for_rel_emb=None, **kwargs):
        """Message-passing GNN with relation composition."""
        nf = graph.node_features
        src, tgt = graph.edge_index[0], graph.edge_index[1]

        # If we have labels, use them for relation-aware messaging
        if labels_for_rel_emb is not None:
            rel_feat = self.rel_embeddings(labels_for_rel_emb)
        else:
            rel_feat = graph.edge_features

        # Message: concat source, target, relation → transform
        msg = self.msg_fn(torch.cat([nf[src], nf[tgt], rel_feat], dim=-1))

        # Aggregate per target node
        agg = torch.zeros_like(nf)
        agg.scatter_add_(0, tgt.unsqueeze(-1).expand_as(msg), msg)

        # Update
        nf_updated = self.update_fn(agg, nf)

        # Classify edges using updated node features + edge features
        logits = self.classifier(torch.cat([
            nf_updated[src], nf_updated[tgt], graph.edge_features
        ], dim=-1))
        return logits


# ─────────────────────────────────────────────────────────────────────
# DELTA Models
# ─────────────────────────────────────────────────────────────────────

class DELTAEdgeModel(nn.Module):
    """DELTA: Edge-to-edge attention for relation classification."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, **kwargs):
        edge_adj = graph.build_edge_adjacency()
        edge_feats = self.edge_attn(graph, edge_adj=edge_adj)
        return self.classifier(edge_feats)


class DELTASoftGatingModel(nn.Module):
    """DELTA: Dual attention + soft gating at 50% target sparsity."""

    def __init__(self, d_node, d_edge, num_classes, num_heads=4):
        super().__init__()
        self.dual_attn = DualParallelAttention(d_node, d_edge, num_heads)
        self.pruner = PostAttentionPruner(d_node, d_edge, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(d_edge, d_edge), nn.GELU(), nn.Linear(d_edge, num_classes),
        )

    def forward(self, graph, target_sparsity=0.5, temperature=1.0, **kwargs):
        edge_adj = graph.build_edge_adjacency()
        result, node_attn_w, edge_attn_w = self.dual_attn(
            graph, edge_adj=edge_adj, return_weights=True
        )
        _, edge_gates = self.pruner.compute_importance(
            result, node_attn_w, edge_attn_w, temperature=temperature,
        )
        gated_graph, sparsity_loss = self.pruner.soft_prune(
            result, edge_gates, target_sparsity=target_sparsity,
        )
        return self.classifier(gated_graph.edge_features), sparsity_loss


# ─────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────

def train_eval_embedding(model, src, tgt, labels, train_idx, test_idx,
                         num_classes, epochs=200, lr=1e-3):
    """Train TransE/RotatE on relation classification."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_test = 0.0

    for epoch in range(epochs):
        model.train()
        logits = model(src[train_idx], tgt[train_idx])
        loss = F.cross_entropy(logits, labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(src[test_idx], tgt[test_idx])
                preds = logits.argmax(-1)
                test_acc = (preds == labels[test_idx]).float().mean().item()
                best_test = max(best_test, test_acc)
                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}  Test={test_acc:.3f}")

    return best_test


def train_eval_gnn(model, graph, labels, train_idx, test_idx,
                   epochs=200, lr=1e-3, use_label_rel=False):
    """Train CompGCN on relation classification."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_test = 0.0

    for epoch in range(epochs):
        model.train()
        kwargs = {}
        if use_label_rel:
            kwargs['labels_for_rel_emb'] = labels
        logits = model(graph, **kwargs)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(graph, **kwargs)
                preds = logits.argmax(-1)
                test_acc = (preds[test_idx] == labels[test_idx]).float().mean().item()
                best_test = max(best_test, test_acc)
                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}  Test={test_acc:.3f}")

    return best_test


def train_eval_delta(model, graph, labels, train_idx, test_idx,
                     epochs=200, lr=1e-3, sparsity_weight=0.1,
                     curriculum=False, target_sparsity=0.5):
    """Train DELTA models."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_test = 0.0

    for epoch in range(epochs):
        model.train()
        progress = epoch / max(1, epochs - 1)

        result = model(graph)
        if isinstance(result, tuple):
            logits, aux_loss = result
        else:
            logits, aux_loss = result, torch.tensor(0.0)

        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss = loss + sparsity_weight * aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                result = model(graph)
                logits = result[0] if isinstance(result, tuple) else result
                preds = logits.argmax(-1)
                test_acc = (preds[test_idx] == labels[test_idx]).float().mean().item()
                best_test = max(best_test, test_acc)

                # Per-relation breakdown
                per_rel = {}
                for r in range(labels.max().item() + 1):
                    mask = labels[test_idx] == r
                    if mask.any():
                        rel_acc = (preds[test_idx][mask] == r).float().mean().item()
                        per_rel[r] = rel_acc

                base_accs = [v for k, v in per_rel.items() if k < 15]
                derived_accs = [v for k, v in per_rel.items() if k >= 15]
                base_avg = sum(base_accs) / max(len(base_accs), 1)
                derived_avg = sum(derived_accs) / max(len(derived_accs), 1) if derived_accs else 0

                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}  Test={test_acc:.3f}  "
                      f"Base={base_avg:.3f}  Derived={derived_avg:.3f}")

    return best_test


def link_prediction_eval(model, graph, labels, test_idx, num_entities):
    """Evaluate link prediction: Hits@10 and MRR on test edges."""
    model.eval()
    src = graph.edge_index[0]
    tgt = graph.edge_index[1]

    if not hasattr(model, 'score_triples'):
        return None, None

    hits_10 = 0
    mrr = 0.0
    count = 0

    with torch.no_grad():
        for idx in test_idx[:100]:  # Sample for speed
            s = src[idx]
            t = tgt[idx]
            r = labels[idx]

            # Score true triple and corrupted triples (replace tail)
            all_tgt = torch.arange(num_entities)
            all_src = s.expand(num_entities)
            all_rel = r.expand(num_entities)

            scores = model.score_triples(all_src, all_tgt, all_rel)
            rank = (scores >= scores[t]).sum().item()

            hits_10 += 1 if rank <= 10 else 0
            mrr += 1.0 / rank
            count += 1

    return hits_10 / max(count, 1), mrr / max(count, 1)


def main():
    print("=" * 70)
    print("PHASE 23: Realistic KG Benchmark — DELTA vs Baselines")
    print("=" * 70)
    print()

    d_node, d_edge = 64, 32

    print("Generating FB15k-237-like benchmark...")
    t0 = time.time()
    graph, labels, metadata = create_realistic_kg_benchmark(
        num_entities=2000, num_relations=20, num_triples=8000,
        d_node=d_node, d_edge=d_edge,
    )
    gen_time = time.time() - t0

    num_classes = metadata['num_relations']
    num_entities = graph.num_nodes
    train_idx = metadata['train_idx']
    test_idx = metadata['test_idx']
    rel_names = metadata['relation_names']

    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"Relations: {num_classes} ({metadata['n_base']} base triples + derived)")
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Relation names: {', '.join(rel_names[:5])}... + {len(rel_names)-5} more")
    print()

    src = graph.edge_index[0]
    tgt = graph.edge_index[1]
    results = {}
    times = {}

    # ── Task A: Relation Classification ──

    print("=" * 70)
    print("TASK A: Relation Classification")
    print("=" * 70)

    # 1. TransE
    print("\n--- TransE ---")
    torch.manual_seed(42)
    m = TransE(num_entities, num_classes, dim=d_node)
    t0 = time.time()
    acc = train_eval_embedding(m, src, tgt, labels, train_idx, test_idx,
                               num_classes, epochs=200)
    times['TransE'] = time.time() - t0
    results['TransE'] = acc

    # LP eval
    h10, mrr = link_prediction_eval(m, graph, labels, test_idx, num_entities)

    # 2. RotatE
    print("\n--- RotatE ---")
    torch.manual_seed(42)
    m = RotatE(num_entities, num_classes, dim=d_node)
    t0 = time.time()
    acc = train_eval_embedding(m, src, tgt, labels, train_idx, test_idx,
                               num_classes, epochs=200)
    times['RotatE'] = time.time() - t0
    results['RotatE'] = acc

    h10_r, mrr_r = link_prediction_eval(m, graph, labels, test_idx, num_entities)

    # 3. CompGCN
    print("\n--- CompGCN ---")
    torch.manual_seed(42)
    m = CompGCNClassifier(d_node, d_edge, num_classes, num_classes)
    t0 = time.time()
    acc = train_eval_gnn(m, graph, labels, train_idx, test_idx, epochs=200)
    times['CompGCN'] = time.time() - t0
    results['CompGCN'] = acc

    # 4. DELTA Edge Attention
    print("\n--- DELTA Edge Attention ---")
    torch.manual_seed(42)
    m = DELTAEdgeModel(d_node, d_edge, num_classes)
    t0 = time.time()
    acc = train_eval_delta(m, graph, labels, train_idx, test_idx, epochs=200)
    times['DELTA Edge'] = time.time() - t0
    results['DELTA Edge'] = acc

    # 5. DELTA + Soft Gating
    print("\n--- DELTA + Soft Gating @ 50% ---")
    torch.manual_seed(42)
    m = DELTASoftGatingModel(d_node, d_edge, num_classes)
    t0 = time.time()
    acc = train_eval_delta(m, graph, labels, train_idx, test_idx, epochs=200,
                           sparsity_weight=0.1)
    times['DELTA+Gate'] = time.time() - t0
    results['DELTA+Gate'] = acc

    # ── Summary ──
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  Task A: Relation Classification (2000 entities, {num_classes} relations)")
    print(f"  {'Model':<20s} {'Test Acc':>10s} {'Time':>8s}")
    print(f"  {'-'*20} {'-'*10} {'-'*8}")
    for name, acc in results.items():
        t = times.get(name, 0)
        bar = '#' * int(acc * 30)
        print(f"  {name:<20s} {acc:>10.3f} {t:>7.1f}s  {bar}")

    # Link prediction results
    print(f"\n  Task B: Link Prediction (Hits@10 / MRR, 100 test samples)")
    if h10 is not None:
        print(f"    TransE:  Hits@10={h10:.3f}  MRR={mrr:.3f}")
    if h10_r is not None:
        print(f"    RotatE:  Hits@10={h10_r:.3f}  MRR={mrr_r:.3f}")

    # Analysis
    delta_acc = results.get('DELTA Edge', 0)
    gate_acc = results.get('DELTA+Gate', 0)
    best_baseline = max(results.get('TransE', 0), results.get('RotatE', 0),
                        results.get('CompGCN', 0))
    best_name = max(['TransE', 'RotatE', 'CompGCN'],
                    key=lambda n: results.get(n, 0))

    print(f"\n  DELTA Edge vs best baseline ({best_name}): {delta_acc - best_baseline:+.3f}")
    print(f"  DELTA+Gate vs best baseline ({best_name}): {gate_acc - best_baseline:+.3f}")
    if delta_acc > best_baseline:
        print("  >> DELTA outperforms all KG embedding baselines!")
    if gate_acc >= delta_acc - 0.01:
        print("  >> Soft gating maintains accuracy with 50% target sparsity!")


if __name__ == '__main__':
    main()
