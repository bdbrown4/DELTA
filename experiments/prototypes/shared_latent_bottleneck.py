"""
Prototype: Shared Latent Bottleneck for DELTA

Standalone experiment comparing three cross-stream interaction designs:
  1. ReconciliationBridge (current DELTA — baseline)
  2. Cross-Attention Gates (Option 1 — recommended)
  3. Shared Latent Bottleneck (Option 3 — experimental)

Uses a synthetic multi-hop reasoning task designed to stress-test
iterative refinement: nodes must propagate information across 3+ hops
through specific edge types to classify target edges correctly.

This is ISOLATED from the main DELTA codebase — no imports from delta/
except DeltaGraph. Safe to experiment freely.

Usage:
    python experiments/prototypes/shared_latent_bottleneck.py
    python experiments/prototypes/shared_latent_bottleneck.py --d_latent 16
    python experiments/prototypes/shared_latent_bottleneck.py --hops 4
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import math
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from delta.graph import DeltaGraph


# ============================================================================
# Synthetic multi-hop reasoning task
# ============================================================================

def create_multihop_task(num_nodes=200, num_edges=800, num_classes=5,
                         hops=3, d_node=32, d_edge=16, seed=42):
    """Synthetic KG task modeled on FB15k-237 / WN18RR properties.

    Realistic properties:
    1. Power-law degree distribution (hub entities, long tail)
    2. Zipf-distributed entity types (~Person, Location, Org...)
    3. Relation types assigned by entity-type pairs (Person→Location = born_in)
    4. Labels = compositional multi-hop: base_rel + neighbor_rels + hop chain
    5. Partial observability → natural accuracy ceiling ~70-85%
    6. Weak feature signal (SNR ~0.5) — must learn from structure

    Difficulty knobs:
    - temperature: higher = more stochastic relation assignment = harder
    - noise_scale: feature noise level
    - signal_scale: feature signal level
    """
    from collections import Counter
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    num_rel_types = num_classes
    num_entity_types = max(num_classes * 2, 10)

    # --- 1. Zipf-distributed entity types ---
    type_probs = np.array([1.0 / (i + 1) ** 0.8 for i in range(num_entity_types)])
    type_probs /= type_probs.sum()
    entity_types = torch.from_numpy(
        rng.choice(num_entity_types, size=num_nodes, p=type_probs))

    # --- 2. Power-law graph (Barabási-Albert) ---
    m = max(3, num_edges // num_nodes)  # edges per new node
    edge_set = set()
    src_list, tgt_list = [], []

    # Seed clique
    core = min(m + 1, num_nodes)
    for i in range(core):
        for j in range(i + 1, core):
            edge_set.add((i, j))
            src_list.extend([i, j])
            tgt_list.extend([j, i])

    degree = np.zeros(num_nodes, dtype=np.float64)
    for s in src_list:
        degree[s] += 1

    # Preferential attachment
    for new_node in range(core, num_nodes):
        existing = np.arange(new_node)
        probs = degree[:new_node] + 1.0
        probs /= probs.sum()
        targets = rng.choice(existing, size=min(m, new_node), replace=False, p=probs)
        for t in targets:
            if (new_node, t) not in edge_set:
                edge_set.add((new_node, t))
                src_list.extend([new_node, t])
                tgt_list.extend([t, new_node])
                degree[new_node] += 1
                degree[t] += 1

    # Fill to target edge count
    attempts = 0
    while len(src_list) < num_edges and attempts < num_edges * 5:
        s = rng.randint(0, num_nodes)
        probs = degree.copy() + 1.0
        probs[s] = 0
        probs /= probs.sum()
        t = rng.choice(num_nodes, p=probs)
        if (s, t) not in edge_set:
            edge_set.add((s, t))
            src_list.append(s)
            tgt_list.append(t)
            degree[s] += 1
            degree[t] += 1
        attempts += 1

    E = min(len(src_list), num_edges)
    edge_index = torch.tensor([src_list[:E], tgt_list[:E]], dtype=torch.long)

    # --- 3. Relation types from entity-type pairs ---
    # Each (type_h, type_t) pair has a preferred relation distribution
    pair_to_rel_logits = torch.randn(num_entity_types, num_entity_types, num_rel_types) * 2.0
    for et_h in range(num_entity_types):
        for et_t in range(num_entity_types):
            dominant = (et_h * 7 + et_t * 13) % num_rel_types
            pair_to_rel_logits[et_h, et_t, dominant] += 3.0

    # Adjacency list
    adj = [[] for _ in range(num_nodes)]
    for e_idx in range(E):
        s, t = edge_index[0, e_idx].item(), edge_index[1, e_idx].item()
        adj[s].append((t, e_idx))

    # --- 4. Labels: compositional multi-hop ---
    edge_rel_types = torch.zeros(E, dtype=torch.long)

    # Phase A: base relation from entity type pair (stochastic → ambiguity)
    for e_idx in range(E):
        s, t = edge_index[0, e_idx].item(), edge_index[1, e_idx].item()
        et_s, et_t = entity_types[s].item(), entity_types[t].item()
        logits = pair_to_rel_logits[et_s, et_t]
        probs = F.softmax(logits / 1.5, dim=0).numpy()
        edge_rel_types[e_idx] = rng.choice(num_rel_types, p=probs)

    # Phase B: final label from multi-hop composition
    labels = torch.zeros(E, dtype=torch.long)
    for e_idx in range(E):
        s = edge_index[0, e_idx].item()
        t = edge_index[1, e_idx].item()
        base_rel = edge_rel_types[e_idx].item()

        # 2-hop: neighbor relation context from target's outgoing edges
        neighbor_rels = []
        for (nbr, nbr_e_idx) in adj[t]:
            if nbr_e_idx != e_idx:
                neighbor_rels.append(edge_rel_types[nbr_e_idx].item())

        if neighbor_rels:
            rel_counts = Counter(neighbor_rels)
            top_rels = [r for r, _ in rel_counts.most_common(min(3, len(rel_counts)))]
        else:
            top_rels = []

        # Compose: base_rel + neighbor context + entity types
        h = base_rel * 1000
        for i, r in enumerate(top_rels):
            h = (h * 31 + r * (i + 7)) % 100000
        h = (h + entity_types[s].item() * 137 + entity_types[t].item() * 53) % 100000

        # Deeper hops: walk from t along high-degree neighbors
        current = t
        for hop in range(max(0, hops - 2)):
            if adj[current]:
                best_idx = max(range(len(adj[current])),
                               key=lambda i: degree[adj[current][i][0]])
                next_node, next_e = adj[current][best_idx]
                h = (h * 31 + edge_rel_types[next_e].item() * 19 +
                     entity_types[next_node].item() * 11) % 100000
                current = next_node

        labels[e_idx] = h % num_rel_types

    # --- 5. Node features: weak entity-type signal + noise (SNR ~0.5) ---
    type_embeddings = torch.randn(num_entity_types, d_node)
    type_embeddings = F.normalize(type_embeddings, dim=1) * 0.5
    node_features = torch.randn(num_nodes, d_node) * 0.7
    for i in range(num_nodes):
        node_features[i] += type_embeddings[entity_types[i]]

    # --- 6. Edge features: weak relation signal + TransE-style context ---
    rel_embeddings = torch.randn(num_rel_types, d_edge)
    rel_embeddings = F.normalize(rel_embeddings, dim=1) * 0.4
    edge_features = torch.randn(E, d_edge) * 0.7
    for e_idx in range(E):
        s, t = edge_index[0, e_idx].item(), edge_index[1, e_idx].item()
        edge_features[e_idx] += rel_embeddings[edge_rel_types[e_idx]]
        # TransE-style: tail - head
        min_d = min(d_node, d_edge)
        edge_features[e_idx, :min_d] += (node_features[t, :min_d] -
                                          node_features[s, :min_d]) * 0.2

    graph = DeltaGraph(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=edge_index,
    )

    # Report task stats
    label_dist = torch.bincount(labels, minlength=num_classes).tolist()
    print(f"  Task stats: {num_entity_types} entity types, {num_rel_types} relation types")
    print(f"  Degree: mean={degree.mean():.1f}, max={degree.max():.0f}, "
          f"median={np.median(degree):.0f}")
    print(f"  Labels: {label_dist}")
    majority = max(label_dist) / sum(label_dist)
    print(f"  Majority baseline: {majority:.3f}")

    return graph, labels, num_classes


# ============================================================================
# Component: Node Self-Attention (simplified)
# ============================================================================

class SimpleNodeAttn(nn.Module):
    def __init__(self, d_node, d_edge, num_heads=4, dropout=0.0):
        super().__init__()
        self.d_node = d_node
        self.num_heads = num_heads
        self.d_head = d_node // num_heads
        self.W_q = nn.Linear(d_node, d_node)
        self.W_k = nn.Linear(d_node, d_node)
        self.W_v = nn.Linear(d_node, d_node)
        self.W_out = nn.Linear(d_node, d_node)
        self.norm = nn.LayerNorm(d_node)
        self.drop = nn.Dropout(dropout)

    def forward(self, node_features, edge_index):
        N = node_features.shape[0]
        H, d_h = self.num_heads, self.d_head
        Q = self.W_q(node_features).view(N, H, d_h)
        K = self.W_k(node_features).view(N, H, d_h)
        V = self.W_v(node_features).view(N, H, d_h)
        src, tgt = edge_index
        q_t = Q[tgt]
        k_s = K[src]
        scores = (q_t * k_s).sum(-1) / math.sqrt(d_h)
        # Scatter softmax
        max_vals = torch.full((N, H), -1e9, device=scores.device)
        idx = tgt.unsqueeze(-1).expand_as(scores)
        max_vals.scatter_reduce_(0, idx, scores, reduce='amax', include_self=False)
        scores = scores - max_vals.gather(0, idx)
        exp_s = torch.exp(scores)
        sum_exp = torch.zeros(N, H, device=scores.device)
        sum_exp.scatter_add_(0, idx, exp_s)
        attn = exp_s / (sum_exp.gather(0, idx) + 1e-10)
        weighted = V[src] * attn.unsqueeze(-1)
        out = torch.zeros(N, H, d_h, device=node_features.device)
        out.scatter_add_(0, tgt.unsqueeze(-1).unsqueeze(-1).expand_as(weighted), weighted)
        out = self.drop(self.W_out(out.reshape(N, self.d_node)))
        return self.norm(node_features + out)


class SimpleEdgeAttn(nn.Module):
    def __init__(self, d_edge, d_node, num_heads=4, dropout=0.0):
        super().__init__()
        self.d_edge = d_edge
        self.num_heads = num_heads
        self.d_head = d_edge // num_heads
        self.W_q = nn.Linear(d_edge, d_edge)
        self.W_k = nn.Linear(d_edge, d_edge)
        self.W_v = nn.Linear(d_edge, d_edge)
        self.W_out = nn.Linear(d_edge, d_edge)
        self.norm = nn.LayerNorm(d_edge)
        self.drop = nn.Dropout(dropout)

    def forward(self, edge_features, edge_adj):
        E = edge_features.shape[0]
        H, d_h = self.num_heads, self.d_head
        Q = self.W_q(edge_features).view(E, H, d_h)
        K = self.W_k(edge_features).view(E, H, d_h)
        V = self.W_v(edge_features).view(E, H, d_h)
        if edge_adj.shape[1] == 0:
            return self.norm(edge_features)
        src_e, tgt_e = edge_adj
        q_t = Q[tgt_e]
        k_s = K[src_e]
        scores = (q_t * k_s).sum(-1) / math.sqrt(d_h)
        max_vals = torch.full((E, H), -1e9, device=scores.device)
        idx = tgt_e.unsqueeze(-1).expand_as(scores)
        max_vals.scatter_reduce_(0, idx, scores, reduce='amax', include_self=False)
        scores = scores - max_vals.gather(0, idx)
        exp_s = torch.exp(scores)
        sum_exp = torch.zeros(E, H, device=scores.device)
        sum_exp.scatter_add_(0, idx, exp_s)
        attn = exp_s / (sum_exp.gather(0, idx) + 1e-10)
        weighted = V[src_e] * attn.unsqueeze(-1)
        out = torch.zeros(E, H, d_h, device=edge_features.device)
        out.scatter_add_(0, tgt_e.unsqueeze(-1).unsqueeze(-1).expand_as(weighted), weighted)
        out = self.drop(self.W_out(out.reshape(E, self.d_edge)))
        return self.norm(edge_features + out)


# ============================================================================
# Design 1: ReconciliationBridge (current DELTA baseline)
# ============================================================================

class ReconciliationBridge(nn.Module):
    """Exact replica of DELTA's current ReconciliationBridge."""

    def __init__(self, d_node, d_edge):
        super().__init__()
        self.edge_from_nodes = nn.Linear(d_edge + 2 * d_node, d_edge)
        self.edge_norm = nn.LayerNorm(d_edge)
        self.node_from_edges = nn.Linear(d_node + d_edge, d_node)
        self.node_norm = nn.LayerNorm(d_node)

    def forward(self, node_features, edge_features, edge_index):
        src, tgt = edge_index
        # Edges absorb nodes
        edge_ctx = torch.cat([edge_features,
                              node_features[src], node_features[tgt]], dim=-1)
        new_edges = self.edge_norm(edge_features + self.edge_from_nodes(edge_ctx))
        # Nodes absorb edges
        N = node_features.shape[0]
        edge_sum = torch.zeros(N, edge_features.shape[1], device=node_features.device)
        edge_count = torch.zeros(N, 1, device=node_features.device)
        all_nodes = torch.cat([src, tgt])
        all_edges = torch.cat([new_edges, new_edges])
        node_idx = all_nodes.unsqueeze(-1).expand_as(all_edges)
        edge_sum.scatter_add_(0, node_idx, all_edges)
        edge_count.scatter_add_(0, all_nodes.unsqueeze(-1),
                                torch.ones(all_nodes.shape[0], 1, device=node_features.device))
        edge_mean = edge_sum / (edge_count + 1e-10)
        node_ctx = torch.cat([node_features, edge_mean], dim=-1)
        new_nodes = self.node_norm(node_features + self.node_from_edges(node_ctx))
        return new_nodes, new_edges


# ============================================================================
# Design 2: Cross-Attention Gates (Option 1)
# ============================================================================

class CrossAttentionGates(nn.Module):
    """Gated cross-attention between node and edge streams.

    During attention, nodes peek at edge features and vice versa.
    Learned gates control interaction strength per head.
    """

    def __init__(self, d_node, d_edge, num_heads=4):
        super().__init__()
        # Node ← Edge cross-attention
        self.node_cross_q = nn.Linear(d_node, d_node)
        self.node_cross_k = nn.Linear(d_edge, d_node)  # project edge to node dim
        self.node_cross_v = nn.Linear(d_edge, d_node)
        self.node_gate = nn.Parameter(torch.zeros(num_heads))  # init closed

        # Edge ← Node cross-attention
        self.edge_cross_q = nn.Linear(d_edge, d_edge)
        self.edge_cross_k = nn.Linear(d_node, d_edge)  # project node to edge dim
        self.edge_cross_v = nn.Linear(d_node, d_edge)
        self.edge_gate = nn.Parameter(torch.zeros(num_heads))  # init closed

        self.node_norm = nn.LayerNorm(d_node)
        self.edge_norm = nn.LayerNorm(d_edge)
        self.num_heads = num_heads
        self.d_node = d_node
        self.d_edge = d_edge

    def forward(self, node_features, edge_features, edge_index):
        src, tgt = edge_index
        N, E = node_features.shape[0], edge_features.shape[0]
        H = self.num_heads
        d_nh = self.d_node // H
        d_eh = self.d_edge // H

        # --- Node ← Edge: each node attends to its incident edges ---
        nq = self.node_cross_q(node_features).view(N, H, d_nh)
        ek = self.node_cross_k(edge_features).view(E, H, d_nh)
        ev = self.node_cross_v(edge_features).view(E, H, d_nh)

        # Each edge contributes to both endpoints
        all_nodes = torch.cat([src, tgt])  # [2E]
        all_ek = torch.cat([ek, ek])  # [2E, H, d_nh]
        all_ev = torch.cat([ev, ev])

        scores = (nq[all_nodes] * all_ek).sum(-1) / math.sqrt(d_nh)  # [2E, H]
        # Scatter softmax per node
        max_vals = torch.full((N, H), -1e9, device=scores.device)
        idx = all_nodes.unsqueeze(-1).expand_as(scores)
        max_vals.scatter_reduce_(0, idx, scores, reduce='amax', include_self=False)
        scores = scores - max_vals.gather(0, idx)
        exp_s = torch.exp(scores)
        sum_exp = torch.zeros(N, H, device=scores.device)
        sum_exp.scatter_add_(0, idx, exp_s)
        attn = exp_s / (sum_exp.gather(0, idx) + 1e-10)

        weighted = all_ev * attn.unsqueeze(-1)
        node_cross_out = torch.zeros(N, H, d_nh, device=node_features.device)
        node_cross_out.scatter_add_(
            0, all_nodes.unsqueeze(-1).unsqueeze(-1).expand_as(weighted), weighted)

        # Apply learned gate (sigmoid, per head)
        g_ne = torch.sigmoid(self.node_gate).view(1, H, 1)
        node_cross_out = (g_ne * node_cross_out).reshape(N, self.d_node)
        new_nodes = self.node_norm(node_features + node_cross_out)

        # --- Edge ← Node: each edge attends to its endpoint nodes ---
        eq = self.edge_cross_q(edge_features).view(E, H, d_eh)
        nk = self.edge_cross_k(node_features).view(N, H, d_eh)
        nv = self.edge_cross_v(node_features).view(N, H, d_eh)

        # Each edge has exactly 2 endpoint nodes
        src_k = nk[src]  # [E, H, d_eh]
        tgt_k = nk[tgt]
        src_v = nv[src]
        tgt_v = nv[tgt]

        # Simple 2-element softmax
        s_src = (eq * src_k).sum(-1) / math.sqrt(d_eh)
        s_tgt = (eq * tgt_k).sum(-1) / math.sqrt(d_eh)
        max_s = torch.maximum(s_src, s_tgt)
        a_src = torch.exp(s_src - max_s)
        a_tgt = torch.exp(s_tgt - max_s)
        denom = a_src + a_tgt + 1e-10
        edge_cross_out = (a_src.unsqueeze(-1) * src_v +
                          a_tgt.unsqueeze(-1) * tgt_v) / denom.unsqueeze(-1)

        g_en = torch.sigmoid(self.edge_gate).view(1, H, 1)
        edge_cross_out = (g_en * edge_cross_out).reshape(E, self.d_edge)
        new_edges = self.edge_norm(edge_features + edge_cross_out)

        return new_nodes, new_edges


# ============================================================================
# Design 3: Shared Latent Bottleneck (Option 3)
# ============================================================================

class SharedLatentBottleneck(nn.Module):
    """Both streams project into a shared latent space, interact, project back.

    Node stream → project down → shared space ← project down ← Edge stream
                     ↓  cross-attend in shared space  ↓
    Node stream ← project up ←  shared space  → project up → Edge stream

    This forces both streams to develop a common intermediate representation.
    The bottleneck dimension (d_latent) controls compression.
    """

    def __init__(self, d_node, d_edge, d_latent=None, num_heads=4):
        super().__init__()
        if d_latent is None:
            d_latent = min(d_node, d_edge)  # compress to smaller dim
        self.d_latent = d_latent
        self.num_heads = num_heads
        d_head = d_latent // num_heads

        # Project down to shared space
        self.node_down = nn.Linear(d_node, d_latent)
        self.edge_down = nn.Linear(d_edge, d_latent)

        # Self-attention in shared space (nodes and edges as one sequence)
        self.shared_q = nn.Linear(d_latent, d_latent)
        self.shared_k = nn.Linear(d_latent, d_latent)
        self.shared_v = nn.Linear(d_latent, d_latent)
        self.shared_out = nn.Linear(d_latent, d_latent)
        self.shared_norm = nn.LayerNorm(d_latent)

        # Project back up
        self.node_up = nn.Linear(d_latent, d_node)
        self.edge_up = nn.Linear(d_latent, d_edge)

        # Residual gates (learn how much bottleneck output to mix in)
        self.node_gate = nn.Parameter(torch.zeros(1))
        self.edge_gate = nn.Parameter(torch.zeros(1))

        self.node_norm = nn.LayerNorm(d_node)
        self.edge_norm = nn.LayerNorm(d_edge)

    def forward(self, node_features, edge_features, edge_index):
        N, E = node_features.shape[0], edge_features.shape[0]
        H = self.num_heads
        d_h = self.d_latent // H

        # Project down to shared latent space
        node_latent = self.node_down(node_features)  # [N, d_latent]
        edge_latent = self.edge_down(edge_features)  # [E, d_latent]

        # Concatenate into one sequence: [N+E, d_latent]
        # Nodes get indices 0..N-1, edges get N..N+E-1
        combined = torch.cat([node_latent, edge_latent], dim=0)
        S = N + E

        # Self-attention in shared space
        # For efficiency, use sparse attention: each element only attends to
        # structurally adjacent elements (nodes ↔ their edges)
        Q = self.shared_q(combined).view(S, H, d_h)
        K = self.shared_k(combined).view(S, H, d_h)
        V = self.shared_v(combined).view(S, H, d_h)

        # Build cross-adjacency: node i ↔ edge j if edge j has endpoint i
        src, tgt = edge_index
        edge_indices = torch.arange(E, device=edge_index.device) + N  # offset by N

        # Node → Edge connections (both endpoints)
        cross_src = torch.cat([src, tgt, edge_indices, edge_indices])  # from
        cross_tgt = torch.cat([edge_indices, edge_indices, src, tgt])  # to

        # Attention scores
        q_t = Q[cross_tgt]
        k_s = K[cross_src]
        scores = (q_t * k_s).sum(-1) / math.sqrt(d_h)

        # Scatter softmax per target
        max_vals = torch.full((S, H), -1e9, device=scores.device)
        idx = cross_tgt.unsqueeze(-1).expand_as(scores)
        max_vals.scatter_reduce_(0, idx, scores, reduce='amax', include_self=False)
        scores = scores - max_vals.gather(0, idx)
        exp_s = torch.exp(scores)
        sum_exp = torch.zeros(S, H, device=scores.device)
        sum_exp.scatter_add_(0, idx, exp_s)
        attn = exp_s / (sum_exp.gather(0, idx) + 1e-10)

        # Weighted values
        weighted = V[cross_src] * attn.unsqueeze(-1)
        out = torch.zeros(S, H, d_h, device=combined.device)
        out.scatter_add_(0, cross_tgt.unsqueeze(-1).unsqueeze(-1).expand_as(weighted), weighted)
        out = self.shared_out(out.reshape(S, self.d_latent))
        combined = self.shared_norm(combined + out)

        # Split back and project up
        node_latent_out = combined[:N]
        edge_latent_out = combined[N:]

        node_update = self.node_up(node_latent_out)
        edge_update = self.edge_up(edge_latent_out)

        # Gated residual (gates init at 0 = starts as identity)
        g_n = torch.sigmoid(self.node_gate)
        g_e = torch.sigmoid(self.edge_gate)
        new_nodes = self.node_norm(node_features + g_n * node_update)
        new_edges = self.edge_norm(edge_features + g_e * edge_update)

        return new_nodes, new_edges


# ============================================================================
# Full model wrapper (parameterized by interaction design)
# ============================================================================

class DualAttentionModel(nn.Module):
    """DELTA-like model with pluggable cross-stream interaction."""

    def __init__(self, d_node, d_edge, num_layers, num_heads, num_classes,
                 interaction='reconciliation', d_latent=None, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(DualAttentionLayer(
                d_node, d_edge, num_heads, interaction, d_latent, dropout))
        self.classifier = nn.Linear(d_edge, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, graph):
        node_f = graph.node_features
        edge_f = graph.edge_features
        edge_adj = graph.build_edge_adjacency()
        for layer in self.layers:
            node_f, edge_f = layer(node_f, edge_f, graph.edge_index, edge_adj)
        return edge_f

    def classify_edges(self, edge_features):
        return self.classifier(self.drop(edge_features))


class DualAttentionLayer(nn.Module):
    def __init__(self, d_node, d_edge, num_heads, interaction, d_latent, dropout=0.0):
        super().__init__()
        self.node_attn = SimpleNodeAttn(d_node, d_edge, num_heads, dropout)
        self.edge_attn = SimpleEdgeAttn(d_edge, d_node, num_heads, dropout)

        if interaction == 'reconciliation':
            self.interact = ReconciliationBridge(d_node, d_edge)
        elif interaction == 'cross_attention':
            self.interact = CrossAttentionGates(d_node, d_edge, num_heads)
        elif interaction == 'shared_latent':
            self.interact = SharedLatentBottleneck(d_node, d_edge, d_latent, num_heads)
        else:
            raise ValueError(f"Unknown interaction: {interaction}")

    def forward(self, node_features, edge_features, edge_index, edge_adj):
        # Parallel attention (read same input)
        new_nodes = self.node_attn(node_features, edge_index)
        new_edges = self.edge_attn(edge_features, edge_adj)
        # Cross-stream interaction
        final_nodes, final_edges = self.interact(new_nodes, new_edges, edge_index)
        return final_nodes, final_edges


# ============================================================================
# Training + evaluation
# ============================================================================

def train_and_eval(model, graph, labels, epochs=100, lr=1e-3, device='cpu',
                   weight_decay=0.0, batch_size=512):
    model = model.to(device)
    graph = graph.to(device)
    labels = labels.to(device)

    E = labels.shape[0]
    perm = torch.randperm(E)
    train_idx = perm[:int(E * 0.7)].to(device)
    val_idx = perm[int(E * 0.7):int(E * 0.85)].to(device)
    test_idx = perm[int(E * 0.85):].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine annealing: decay LR from `lr` to `lr/50` over training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 50)

    best_val, best_test = 0.0, 0.0
    best_val_loss = float('inf')
    # Track stabilized metrics: val/test where val_loss is lowest
    stab_val, stab_test, stab_loss, stab_epoch = 0.0, 0.0, float('inf'), 0
    # Track last-N-epochs average for stability report
    recent_vals, recent_tests, recent_losses = [], [], []

    num_train = train_idx.shape[0]
    for epoch in range(epochs):
        model.train()
        # Mini-batch: shuffle train edges, iterate in chunks
        shuffle = train_idx[torch.randperm(num_train, device=device)]
        epoch_loss = 0.0
        n_batches = 0
        for b_start in range(0, num_train, batch_size):
            b_idx = shuffle[b_start:b_start + batch_size]
            # Forward full graph (needed for attention), but loss on batch only
            edge_out = model(graph)
            logits = model.classify_edges(edge_out)
            loss = F.cross_entropy(logits[b_idx], labels[b_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = epoch_loss / n_batches

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                edge_out = model(graph)
                logits = model.classify_edges(edge_out)
                train_acc = (logits[train_idx].argmax(-1) == labels[train_idx]).float().mean().item()
                val_acc = (logits[val_idx].argmax(-1) == labels[val_idx]).float().mean().item()
                test_acc = (logits[test_idx].argmax(-1) == labels[test_idx]).float().mean().item()
                val_loss = F.cross_entropy(logits[val_idx], labels[val_idx]).item()
                cur_lr = scheduler.get_last_lr()[0]

                # Best peak val
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc

                # Stabilized: best val_loss (low loss + best generalization)
                if val_loss < stab_loss:
                    stab_loss = val_loss
                    stab_val = val_acc
                    stab_test = test_acc
                    stab_epoch = epoch + 1

                # Track recent for end-of-training average
                recent_vals.append(val_acc)
                recent_tests.append(test_acc)
                recent_losses.append(val_loss)
                # Keep last 5 checkpoints
                if len(recent_vals) > 5:
                    recent_vals.pop(0)
                    recent_tests.pop(0)
                    recent_losses.pop(0)

                if (epoch + 1) % 25 == 0 or epoch == epochs - 1:
                    print(f"    Epoch {epoch+1:3d}  Loss: {avg_loss:.4f}  ValLoss: {val_loss:.4f}  "
                          f"Train: {train_acc:.3f}  Val: {val_acc:.3f}  Test: {test_acc:.3f}  "
                          f"LR: {cur_lr:.6f}")

    params = sum(p.numel() for p in model.parameters())
    # Stabilized end-of-training: average of last 5 checkpoints
    avg_val = np.mean(recent_vals)
    avg_test = np.mean(recent_tests)
    avg_vloss = np.mean(recent_losses)
    return {
        'best_val': best_val, 'best_test': best_test,
        'stab_val': stab_val, 'stab_test': stab_test,
        'stab_loss': stab_loss, 'stab_epoch': stab_epoch,
        'final_val': avg_val, 'final_test': avg_test, 'final_vloss': avg_vloss,
        'params': params,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Shared Latent Bottleneck Prototype")
    parser.add_argument('--d_node', type=int, default=32)
    parser.add_argument('--d_edge', type=int, default=16)
    parser.add_argument('--d_latent', type=int, default=None,
                        help='Latent bottleneck dim (default: min(d_node, d_edge))')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_nodes', type=int, default=200)
    parser.add_argument('--num_edges', type=int, default=800)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--hops', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--dropout', type=float, default=0.15,
                        help='Dropout rate (default: 0.15)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Adam weight decay (default: 5e-4)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Mini-batch size for edge classification (default: 512)')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("PROTOTYPE: Cross-Stream Interaction Comparison")
    print("=" * 70)
    print(f"  Graph: {args.num_nodes} nodes, {args.num_edges} edges, "
          f"{args.num_classes} classes, {args.hops}-hop reasoning")
    print(f"  Model: {args.num_layers} layers, d_node={args.d_node}, d_edge={args.d_edge}")
    print(f"  Device: {device}, Seeds: {args.seeds}, Epochs: {args.epochs}")
    print(f"  Regularization: dropout={args.dropout}, weight_decay={args.weight_decay}")
    d_latent = args.d_latent or min(args.d_node, args.d_edge)
    print(f"  Shared latent dim: {d_latent}")
    print()

    # Create task
    graph, labels, num_classes = create_multihop_task(
        num_nodes=args.num_nodes, num_edges=args.num_edges,
        num_classes=args.num_classes, hops=args.hops,
        d_node=args.d_node, d_edge=args.d_edge)
    print(f"  Graph created: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print()

    designs = [
        ('Reconciliation (baseline)', 'reconciliation'),
        ('Cross-Attention Gates', 'cross_attention'),
        ('Shared Latent Bottleneck', 'shared_latent'),
    ]

    all_results = {}
    for name, interaction in designs:
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

        seed_results = []
        for s in range(args.seeds):
            seed = 42 + s * 100
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"\n  --- Seed {s+1}/{args.seeds} (seed={seed}) ---")

            model = DualAttentionModel(
                d_node=args.d_node, d_edge=args.d_edge,
                num_layers=args.num_layers, num_heads=args.num_heads,
                num_classes=num_classes, interaction=interaction,
                d_latent=d_latent, dropout=args.dropout)

            start = time.time()
            res = train_and_eval(
                model, graph, labels, epochs=args.epochs, lr=1e-3, device=device,
                weight_decay=args.weight_decay, batch_size=args.batch_size)
            elapsed = time.time() - start
            res['time'] = elapsed
            seed_results.append(res)
            print(f"    Peak:  val={res['best_val']:.3f}, test={res['best_test']:.3f}")
            print(f"    Stab:  val={res['stab_val']:.3f}, test={res['stab_test']:.3f}, "
                  f"vloss={res['stab_loss']:.4f} @ep{res['stab_epoch']}")
            print(f"    Final: val={res['final_val']:.3f}, test={res['final_test']:.3f}, "
                  f"vloss={res['final_vloss']:.4f}")
            print(f"    Params: {res['params']:,d}, Time: {elapsed:.1f}s")

        all_results[name] = {
            # Peak val
            'peak_val': np.mean([r['best_val'] for r in seed_results]),
            'peak_val_std': np.std([r['best_val'] for r in seed_results]),
            'peak_test': np.mean([r['best_test'] for r in seed_results]),
            'peak_test_std': np.std([r['best_test'] for r in seed_results]),
            # Stabilized (lowest val loss)
            'stab_val': np.mean([r['stab_val'] for r in seed_results]),
            'stab_val_std': np.std([r['stab_val'] for r in seed_results]),
            'stab_test': np.mean([r['stab_test'] for r in seed_results]),
            'stab_test_std': np.std([r['stab_test'] for r in seed_results]),
            'stab_loss': np.mean([r['stab_loss'] for r in seed_results]),
            'stab_epoch': np.mean([r['stab_epoch'] for r in seed_results]),
            # Final (last 5 checkpoints avg)
            'final_val': np.mean([r['final_val'] for r in seed_results]),
            'final_val_std': np.std([r['final_val'] for r in seed_results]),
            'final_test': np.mean([r['final_test'] for r in seed_results]),
            'final_test_std': np.std([r['final_test'] for r in seed_results]),
            'final_vloss': np.mean([r['final_vloss'] for r in seed_results]),
            'params': seed_results[0]['params'],
            'time_mean': np.mean([r['time'] for r in seed_results]),
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — GOLDEN SPOT REPORT (low domain loss + high val)")
    print("=" * 70)

    print(f"\n  Target: val ~0.69, domain loss minimal")
    print(f"  Task: {args.num_nodes} nodes, {args.num_edges} edges, {args.hops}-hop, "
          f"{args.num_classes} classes")
    print(f"  Training: {args.epochs} epochs, cosine LR, batch_size={args.batch_size}, "
          f"dropout={args.dropout}, wd={args.weight_decay}")

    print(f"\n  {'─'*100}")
    print(f"  {'Design':35s}  {'Params':>7s}  │ {'Stab Val':>8s}  {'Stab Test':>9s}  "
          f"{'ValLoss':>7s}  {'@Epoch':>6s}  │ {'Final Val':>9s}  {'Final Test':>10s}  "
          f"{'VLoss':>6s}")
    print(f"  {'─'*100}")

    for name, res in all_results.items():
        print(f"  {name:35s}  {res['params']:>7,d}  │ "
              f"{res['stab_val']:.3f}±{res['stab_val_std']:.3f}  "
              f"{res['stab_test']:.3f}±{res['stab_test_std']:.3f}  "
              f"{res['stab_loss']:.4f}  "
              f"{res['stab_epoch']:>5.0f}   │ "
              f"{res['final_val']:.3f}±{res['final_val_std']:.3f}  "
              f"{res['final_test']:.3f}±{res['final_test_std']:.3f}  "
              f"{res['final_vloss']:.4f}")

    print(f"  {'─'*100}")

    print(f"\n  Peak Val (for reference):")
    for name, res in all_results.items():
        print(f"    {name:35s}  val={res['peak_val']:.3f}±{res['peak_val_std']:.3f}  "
              f"test={res['peak_test']:.3f}±{res['peak_test_std']:.3f}  "
              f"time={res['time_mean']:.1f}s")

    # Distance from golden spot
    print(f"\n  Distance from golden spot (val=0.69):")
    for name, res in all_results.items():
        gap = 0.69 - res['stab_val']
        status = "✓ HIT" if gap <= 0.02 else f"gap={gap:.3f}"
        print(f"    {name:35s}  stab_val={res['stab_val']:.3f}  → {status}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
