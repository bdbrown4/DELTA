"""
Phase 14: Contrastive Analogy Training

Core question: Can contrastive learning on edge representations enable
true analogical retrieval — finding edges with the same relational pattern
rather than just classifying them?

Phase 10 showed edge classification hits 100% but analogy retrieval is random.
The problem: classification doesn't shape the embedding SPACE — it only needs
a decision boundary, not cluster structure.

Solution: Triplet/contrastive loss that explicitly pulls same-type edges
together and pushes different-type edges apart in embedding space.

Compared:
1. Classification-only training (cross-entropy on edge type)
2. Contrastive-only training (triplet loss)
3. Joint training (classification + contrastive)

Metric: Retrieval accuracy — given an anchor edge, is the nearest neighbor
(by cosine similarity) of the same relation type?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import EdgeAttention
from delta.utils import create_contrastive_analogy_pairs


class EdgeEncoder(nn.Module):
    """Shared encoder that produces edge embeddings via edge attention."""
    def __init__(self, d_edge, d_node, embed_dim=32, num_heads=4):
        super().__init__()
        self.edge_attn = EdgeAttention(d_edge, d_node, num_heads)
        self.project = nn.Sequential(
            nn.Linear(d_edge, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, graph):
        edge_adj = graph.build_edge_adjacency()
        edge_feats = self.edge_attn(graph, edge_adj=edge_adj)
        return self.project(edge_feats)


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, embeddings):
        return self.fc(embeddings)


def triplet_loss(embeddings, triplets, margin=0.3):
    anchor = embeddings[triplets[:, 0]]
    positive = embeddings[triplets[:, 1]]
    negative = embeddings[triplets[:, 2]]

    # Cosine distance (1 - cosine_similarity)
    pos_dist = 1 - F.cosine_similarity(anchor, positive)
    neg_dist = 1 - F.cosine_similarity(anchor, negative)

    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def retrieval_accuracy(embeddings, labels):
    """For each edge, check if its nearest neighbor has the same label."""
    embeddings = F.normalize(embeddings, dim=-1)
    sim = embeddings @ embeddings.T
    # Zero out self-similarity
    sim.fill_diagonal_(-1)
    nearest = sim.argmax(dim=1)
    correct = (labels[nearest] == labels).float().mean().item()
    return correct


def train_and_evaluate(encoder, head, graph, labels, triplets,
                       mode='classification', epochs=300, lr=1e-3,
                       contrastive_weight=1.0):
    params = list(encoder.parameters())
    if head is not None:
        params += list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    best_cls_acc = 0.0
    best_ret_acc = 0.0

    for epoch in range(epochs):
        encoder.train()
        embeddings = encoder(graph)

        loss = torch.tensor(0.0)
        if mode in ('classification', 'joint'):
            logits = head(embeddings)
            loss = loss + F.cross_entropy(logits, labels)
        if mode in ('contrastive', 'joint'):
            loss = loss + contrastive_weight * triplet_loss(embeddings, triplets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 75 == 0:
            encoder.eval()
            with torch.no_grad():
                emb = encoder(graph)
                ret_acc = retrieval_accuracy(emb, labels)
                best_ret_acc = max(best_ret_acc, ret_acc)

                cls_acc = 0.0
                if head is not None:
                    logits = head(emb)
                    cls_acc = (logits.argmax(-1) == labels).float().mean().item()
                    best_cls_acc = max(best_cls_acc, cls_acc)

                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}  "
                      f"Cls={cls_acc:.3f}  Retrieval={ret_acc:.3f}")

    return best_cls_acc, best_ret_acc


def main():
    print("=" * 70)
    print("PHASE 14: Contrastive Analogy Training")
    print("=" * 70)
    print()
    print("Question: Does contrastive loss enable true analogical retrieval?")
    print("Metric: Nearest-neighbor retrieval accuracy (same relation type)")
    print()

    d_node, d_edge, embed_dim = 64, 32, 32
    num_rels = 6
    torch.manual_seed(42)

    graph, labels, triplets = create_contrastive_analogy_pairs(
        num_relation_types=num_rels, pairs_per_type=8,
        d_node=d_node, d_edge=d_edge,
    )
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print(f"Relation types: {num_rels}, Triplets: {len(triplets)}")
    print()

    results = {}

    print("--- Classification Only ---")
    torch.manual_seed(42)
    enc = EdgeEncoder(d_edge, d_node, embed_dim)
    head = ClassificationHead(embed_dim, num_rels)
    cls_acc, ret_acc = train_and_evaluate(enc, head, graph, labels, triplets,
                                          mode='classification')
    results['Classification'] = (cls_acc, ret_acc)

    print("\n--- Contrastive Only ---")
    torch.manual_seed(42)
    enc = EdgeEncoder(d_edge, d_node, embed_dim)
    cls_acc, ret_acc = train_and_evaluate(enc, None, graph, labels, triplets,
                                          mode='contrastive')
    results['Contrastive'] = (0.0, ret_acc)  # no classifier head

    print("\n--- Joint (Classification + Contrastive) ---")
    torch.manual_seed(42)
    enc = EdgeEncoder(d_edge, d_node, embed_dim)
    head = ClassificationHead(embed_dim, num_rels)
    cls_acc, ret_acc = train_and_evaluate(enc, head, graph, labels, triplets,
                                          mode='joint', contrastive_weight=0.5)
    results['Joint'] = (cls_acc, ret_acc)

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<20s} {'Classify':>10s} {'Retrieval':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")
    for name, (c, r) in results.items():
        cls_bar = '#' * int(c * 15) if c > 0 else '(n/a)'
        ret_bar = '#' * int(r * 15)
        print(f"  {name:<20s} {c:>10.3f} {r:>10.3f}  {ret_bar}")

    cr = results['Contrastive'][1]
    cl = results['Classification'][1]
    jr = results['Joint'][1]
    print(f"\n  Contrastive retrieval vs Classification retrieval: {cr - cl:+.3f}")
    print(f"  Joint retrieval vs Classification retrieval:       {jr - cl:+.3f}")
    if cr > cl or jr > cl:
        print("  >> Contrastive loss enables meaningful analogy retrieval!")
    if jr > cr:
        print("  >> Joint training gets best of both: classification + retrieval.")


if __name__ == '__main__':
    main()
