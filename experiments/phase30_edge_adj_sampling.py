"""
Phase 30: GPU Edge Adjacency Sampling Strategy

Phase 25 randomly samples 5M of 19M 1-hop edge-adjacency pairs to fit GPU
VRAM, meaning DELTA operates on ~26% of structural context.

Goal: implement degree-weighted and importance-guided sampling so hub entities
(highest structural influence) are preferentially included, and measure the
accuracy delta vs uniform (random) sampling.

Sampling strategies:
  1. Uniform — random subset (Phase 25 baseline)
  2. Degree-weighted — edges incident to high-degree nodes sampled more
  3. Importance-weighted — use node/edge importance scores from a pre-pass
  4. Stratified — equal representation per node

All tested on FB15k-237 with DELTA+Gate at 50% target sparsity.
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


# ──────────────────────────────────────────────────────────────────────
#  Sampling strategies
# ──────────────────────────────────────────────────────────────────────

def sample_uniform(edge_adj, budget, device):
    """Random uniform sampling (Phase 25 baseline)."""
    n = edge_adj.shape[1]
    if n <= budget:
        return edge_adj
    perm = torch.randperm(n, device=device)[:budget]
    return edge_adj[:, perm]


def sample_degree_weighted(edge_adj, budget, graph, device):
    """Prefer edge pairs involving high-degree nodes.

    Computes degree for each edge pair as the sum of degrees of all
    nodes incident to the two edges. Higher degree → more likely sampled.
    """
    n = edge_adj.shape[1]
    if n <= budget:
        return edge_adj

    src, tgt = graph.edge_index[0], graph.edge_index[1]
    # Compute node degree
    N = graph.num_nodes
    degree = torch.zeros(N, device=device)
    degree.scatter_add_(0, src, torch.ones(src.shape[0], device=device))
    degree.scatter_add_(0, tgt, torch.ones(tgt.shape[0], device=device))

    # For each edge pair (e_i, e_j), weight = degree(nodes of e_i) + degree(nodes of e_j)
    edge_i = edge_adj[0]  # [n]
    edge_j = edge_adj[1]  # [n]
    weight_i = degree[src[edge_i]] + degree[tgt[edge_i]]
    weight_j = degree[src[edge_j]] + degree[tgt[edge_j]]
    weights = weight_i + weight_j
    weights = weights.clamp(min=1e-8)

    # Gumbel-topk trick (works for >2^24 categories unlike multinomial)
    gumbel = -torch.empty_like(weights).exponential_().log()
    idx = (weights.log() + gumbel).topk(budget).indices
    return edge_adj[:, idx]


def sample_stratified(edge_adj, budget, graph, device):
    """Equal representation per node: for each node, sample a fixed quota
    of its incident edge pairs.

    Ensures every node's local structure is represented, preventing hub
    dominance while still capturing sparse boundary nodes.
    """
    n = edge_adj.shape[1]
    if n <= budget:
        return edge_adj

    src, tgt = graph.edge_index[0], graph.edge_index[1]
    N = graph.num_nodes
    E = graph.num_edges

    # Assign each pair to a representative node (source of first edge)
    edge_i = edge_adj[0]  # [n]
    pair_node = src[edge_i]

    # Count pairs per node
    unique_nodes, inverse, counts = pair_node.unique(
        return_inverse=True, return_counts=True)
    num_unique = len(unique_nodes)
    quota = max(1, budget // num_unique)

    # For each pair, compute a priority: random within its node's group,
    # capped at quota per node. Use scatter to count per-node selections.
    rand_scores = torch.rand(n, device=device)
    # Within each node group, rank pairs by random score
    # Pairs with rank <= quota get selected
    # Efficient: add large penalty for pairs beyond quota
    node_rank = torch.zeros(n, dtype=torch.long, device=device)
    sort_idx = torch.argsort(inverse * n + torch.arange(n, device=device))
    running = torch.zeros(num_unique, dtype=torch.long, device=device)
    # Vectorized approach: just sample proportionally from each node
    weight = torch.ones(n, device=device)
    node_total = counts[inverse].float()
    weight = (quota / node_total).clamp(max=1.0)
    # Gumbel-topk with node-equalized weights
    gumbel = -torch.empty(n, device=device).exponential_().log()
    scores = weight.log() + gumbel
    idx = scores.topk(min(budget, n)).indices
    return edge_adj[:, idx]


def sample_importance_weighted(edge_adj, budget, graph, model, device):
    """Use a lightweight pre-pass to estimate edge importance, then sample.

    Runs a single forward pass with a small subset, computes edge importance
    scores from attention weights, and uses them to weight sampling.
    """
    n = edge_adj.shape[1]
    if n <= budget:
        return edge_adj

    # Quick pre-pass with uniform subsample
    pre_budget = min(budget, n)
    pre_perm = torch.randperm(n, device=device)[:pre_budget]
    pre_adj = edge_adj[:, pre_perm]

    model.eval()
    with torch.no_grad():
        result, nw, ew = model.dual_attn(graph, edge_adj=pre_adj, return_weights=True)

    # Compute edge importance from attention weights
    edge_importance = ew.abs().mean(dim=-1) if ew.dim() > 1 else ew.abs()
    # edge_importance is [E] — importance of each edge in the graph

    # Weight edge-adj pairs by importance of their constituent edges
    edge_i = edge_adj[0]
    edge_j = edge_adj[1]
    ei_imp = edge_importance[edge_i]
    ej_imp = edge_importance[edge_j]
    weights = ei_imp + ej_imp
    weights = weights.clamp(min=1e-8)

    # Gumbel-topk trick (works for >2^24 categories unlike multinomial)
    gumbel = -torch.empty_like(weights).exponential_().log()
    idx = (weights.log() + gumbel).topk(budget).indices
    return edge_adj[:, idx]


# ──────────────────────────────────────────────────────────────────────
#  Model
# ──────────────────────────────────────────────────────────────────────

class DELTAGateModel(nn.Module):
    """DELTA+Gate for relation classification (accepts precomputed edge_adj)."""

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
#  Training
# ──────────────────────────────────────────────────────────────────────

def train_eval(model, graph, labels, train_idx, test_idx,
               edge_adj, epochs=200, lr=1e-3, sparsity_w=0.1):
    """Train DELTA+Gate and return best test accuracy."""
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 0.0
    for epoch in range(epochs):
        model.train()
        logits, aux = model(graph, edge_adj=edge_adj)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx]) + sparsity_w * aux
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                lg, _ = model(graph, edge_adj=edge_adj)
                acc = (lg.argmax(-1)[test_idx] == labels[test_idx]).float().mean().item()
                best = max(best, acc)
                print(f"    Epoch {epoch + 1}: Loss={loss.item():.4f}  Test={acc:.3f}")
    return best


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 30: GPU Edge Adjacency Sampling Strategy")
    print("=" * 70)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print()

    d_node, d_edge = 64, 32
    BUDGET = 5_000_000  # same as Phase 25 VRAM budget

    # ── Load FB15k-237 ────────────────────────────────────────────────
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from phase25_fb15k237_gpu import load_fb15k237_subgraph
    except ImportError:
        raise RuntimeError("Phase 25 experiment file required for FB15k-237 loading")

    t0 = time.time()
    graph, meta = load_fb15k237_subgraph(
        top_entities=2000, d_node=d_node, d_edge=d_edge, seed=42)
    N = meta['num_entities']
    num_classes = meta['num_relations']
    M = meta['num_triples']
    labels = meta['labels'].to(device)
    train_idx = meta['train_idx'].to(device)
    test_idx = meta['test_idx'].to(device)
    graph = graph.to(device)
    print(f"Subgraph: {N} entities, {num_classes} relations, {M} triples")
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"Data load: {time.time() - t0:.1f}s")

    # ── Build full edge adjacency ─────────────────────────────────────
    print("\nBuilding full 1-hop edge adjacency...")
    t0 = time.time()
    ea_full = graph.build_edge_adjacency(hops=1)
    print(f"Full edge adjacency: {ea_full.shape[1]:,} pairs ({time.time() - t0:.1f}s)")
    print(f"Budget: {BUDGET:,} pairs ({BUDGET / ea_full.shape[1]:.0%} of full)")
    print()

    # ── Test each sampling strategy ───────────────────────────────────
    strategies = {
        'Uniform (random)': lambda ea: sample_uniform(ea, BUDGET, device),
        'Degree-weighted':  lambda ea: sample_degree_weighted(ea, BUDGET, graph, device),
        'Stratified':       lambda ea: sample_stratified(ea, BUDGET, graph, device),
    }

    results = {}

    for strat_name, sampler in strategies.items():
        print("-" * 60)
        print(f"Strategy: {strat_name}")
        print("-" * 60)

        torch.manual_seed(42)
        ea_sampled = sampler(ea_full)
        print(f"  Sampled: {ea_sampled.shape[1]:,} pairs")

        torch.manual_seed(42)
        model = DELTAGateModel(d_node, d_edge, num_classes).to(device)
        t0 = time.time()
        acc = train_eval(model, graph, labels, train_idx, test_idx,
                         edge_adj=ea_sampled, epochs=200)
        elapsed = time.time() - t0
        results[strat_name] = (acc, elapsed)
        print(f"  >> Best: {acc:.1%}  ({elapsed:.1f}s)\n")

    # Importance-weighted needs a pre-trained model
    print("-" * 60)
    print("Strategy: Importance-weighted")
    print("-" * 60)
    torch.manual_seed(42)
    # Use the uniform-trained model for importance estimation
    pre_model = DELTAGateModel(d_node, d_edge, num_classes).to(device)
    # Quick pre-train with uniform sample
    pre_ea = sample_uniform(ea_full, BUDGET, device)
    print("  Pre-training model for importance estimation (50 epochs)...")
    pre_opt = torch.optim.Adam(pre_model.parameters(), lr=1e-3)
    for ep in range(50):
        pre_model.train()
        lg, aux = pre_model(graph, edge_adj=pre_ea)
        loss = F.cross_entropy(lg[train_idx], labels[train_idx]) + 0.1 * aux
        pre_opt.zero_grad()
        loss.backward()
        pre_opt.step()

    ea_imp = sample_importance_weighted(ea_full, BUDGET, graph, pre_model, device)
    print(f"  Sampled: {ea_imp.shape[1]:,} pairs")

    torch.manual_seed(42)
    model = DELTAGateModel(d_node, d_edge, num_classes).to(device)
    t0 = time.time()
    acc = train_eval(model, graph, labels, train_idx, test_idx,
                     edge_adj=ea_imp, epochs=200)
    elapsed = time.time() - t0
    results['Importance-weighted'] = (acc, elapsed)
    print(f"  >> Best: {acc:.1%}  ({elapsed:.1f}s)\n")

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 70)
    print("RESULTS SUMMARY — Sampling Strategy Comparison")
    print("=" * 70)
    print(f"  Full edge adjacency: {ea_full.shape[1]:,} pairs")
    print(f"  Budget: {BUDGET:,} pairs ({BUDGET / ea_full.shape[1]:.0%})")
    print()

    best_name = max(results, key=lambda k: results[k][0])
    baseline_acc = results.get('Uniform (random)', (0, 0))[0]

    print(f"  {'Strategy':<25s}  {'Accuracy':>10s}  {'Time':>8s}  {'vs Uniform':>12s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*8}  {'-'*12}")
    for name in ['Uniform (random)', 'Degree-weighted', 'Stratified', 'Importance-weighted']:
        if name in results:
            acc, t = results[name]
            delta = acc - baseline_acc
            marker = " << BEST" if name == best_name else ""
            print(f"  {name:<25s}  {acc:>9.1%}  {t:>7.1f}s  {delta:>+11.1%}{marker}")

    print()
    print("Analysis:")
    if best_name != 'Uniform (random)':
        best_acc = results[best_name][0]
        delta = best_acc - baseline_acc
        print(f"  {best_name} outperforms Uniform by {delta:+.1%}")
        print(f"  Smart sampling extracts more signal from the same VRAM budget.")
    else:
        print(f"  Uniform sampling is competitive — random coverage is sufficient")
        print(f"  for this graph density and budget proportion.")


if __name__ == '__main__':
    main()
