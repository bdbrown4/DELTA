"""
Phase 33: Task-Aware Graph Construction

Phase 27b confirmed the problem: GraphConstructor's attention-thresholding
discards sequential adjacency edges that Fixed Chain DELTA preserves.
Result: Fixed Chain (40.7%) > Bootstrap (34.3%) on path composition.

Goal: design a constructor that preserves task-relevant structure (positional
ordering for paths, adjacency for sequences) while still learning which
non-local connections to add.

Approach: Hybrid construction with structure-preserving constraints:
  1. Start with a base topology (chain, kNN, or task-defined)
  2. Use the learned constructor to ADD edges (not remove base edges)
  3. Score new edges with attention, but never prune base structure

This separates the roles:
  - Base topology → guaranteed structural inductive bias
  - Learned edges → discovered long-range / cross-cutting connections

Requirements:
    - pip install torch numpy
    - GPU recommended for larger experiments

Usage:
    python experiments/phase33_task_aware_construction.py [--epochs 200]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.model import DELTAModel
from delta.constructor import GraphConstructor


# ---------------------------------------------------------------------------
# Task-Aware Graph Constructor
# ---------------------------------------------------------------------------

class TaskAwareConstructor(nn.Module):
    """Hybrid graph constructor: preserves base topology + learns new edges.

    Key insight from Phase 27b: attention-thresholding in the standard
    GraphConstructor can discard edges that carry essential structural signal
    (like adjacency in sequential tasks). This constructor:

    1. Preserves ALL base edges (never prunes them)
    2. Uses a learned attention mechanism to propose NEW edges
    3. Merges base + learned edges into the final graph

    This ensures the model always has the structural inductive bias from the
    task (e.g., sequential adjacency for path tasks) while being free to
    discover additional long-range connections.
    """

    def __init__(self, d_node, d_edge, num_heads=4, edge_threshold=0.3):
        super().__init__()
        self.d_node = d_node
        self.d_edge = d_edge
        self.edge_threshold = edge_threshold

        # Edge scorer: given node pair, predict edge probability
        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * d_node, d_node),
            nn.GELU(),
            nn.Linear(d_node, 1),
            nn.Sigmoid(),
        )

        # Edge feature generator: given node pair, generate edge features
        self.edge_generator = nn.Sequential(
            nn.Linear(2 * d_node, d_edge),
            nn.GELU(),
            nn.Linear(d_edge, d_edge),
        )

    def propose_new_edges(self, node_features, existing_edge_index):
        """Score all possible non-existing edges and return high-scoring ones."""
        N = node_features.shape[0]
        device = node_features.device

        # Build set of existing edges for fast lookup
        existing = set()
        src, tgt = existing_edge_index
        for s, t in zip(src.tolist(), tgt.tolist()):
            existing.add((s, t))

        # For efficiency, sample candidate pairs (up to 5× existing edges)
        max_candidates = min(N * (N - 1), existing_edge_index.shape[1] * 5)
        candidates_src = []
        candidates_tgt = []

        attempts = 0
        while len(candidates_src) < max_candidates and attempts < max_candidates * 3:
            s = random.randint(0, N - 1)
            t = random.randint(0, N - 1)
            if s != t and (s, t) not in existing:
                candidates_src.append(s)
                candidates_tgt.append(t)
                existing.add((s, t))
            attempts += 1

        if not candidates_src:
            return torch.zeros(2, 0, dtype=torch.long, device=device), \
                   torch.zeros(0, self.d_edge, device=device)

        cand_src = torch.tensor(candidates_src, device=device)
        cand_tgt = torch.tensor(candidates_tgt, device=device)

        # Score candidate pairs
        pair_feats = torch.cat([
            node_features[cand_src],
            node_features[cand_tgt],
        ], dim=-1)
        scores = self.edge_scorer(pair_feats).squeeze(-1)

        # Keep edges above threshold
        keep = scores > self.edge_threshold
        new_src = cand_src[keep]
        new_tgt = cand_tgt[keep]

        if len(new_src) == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=device), \
                   torch.zeros(0, self.d_edge, device=device)

        # Generate features for new edges
        new_pair_feats = torch.cat([
            node_features[new_src],
            node_features[new_tgt],
        ], dim=-1)
        new_edge_feats = self.edge_generator(new_pair_feats)

        new_edge_index = torch.stack([new_src, new_tgt])
        return new_edge_index, new_edge_feats

    def forward(self, graph):
        """Augment graph with learned edges while preserving base structure."""
        new_edge_index, new_edge_feats = self.propose_new_edges(
            graph.node_features, graph.edge_index
        )

        if new_edge_index.shape[1] == 0:
            return graph

        # Merge base + new edges
        merged_edge_index = torch.cat([graph.edge_index, new_edge_index], dim=1)
        merged_edge_feats = torch.cat([graph.edge_features, new_edge_feats], dim=0)

        return DeltaGraph(
            node_features=graph.node_features,
            edge_features=merged_edge_feats,
            edge_index=merged_edge_index,
            node_tiers=graph.node_tiers,
            node_importance=graph.node_importance,
            edge_importance=graph.edge_importance,
        )


# ---------------------------------------------------------------------------
# Path composition task (same structure as Phase 27b)
# ---------------------------------------------------------------------------

def create_path_task(num_entities=30, num_relations=4, num_paths=100,
                     path_length=3, d_node=64, d_edge=32, seed=42):
    """Create a path composition task where the label depends on the path.

    Each path is a sequence of edges: e1 → e2 → e3.
    The label for the final edge depends on the composition of relation types
    along the path.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    rel_protos = torch.randn(num_relations, d_edge)
    node_features = torch.randn(num_entities, d_node)

    src_list, tgt_list, edge_feats, labels = [], [], [], []
    num_classes = num_relations  # Composed label is modular sum

    for _ in range(num_paths):
        nodes = random.sample(range(num_entities), path_length + 1)
        rels = [random.randint(0, num_relations - 1) for _ in range(path_length)]

        for i in range(path_length):
            src_list.append(nodes[i])
            tgt_list.append(nodes[i + 1])
            edge_feats.append(rel_protos[rels[i]] + torch.randn(d_edge) * 0.1)

            # Label: modular sum of relations up to this point
            composed = sum(rels[:i + 1]) % num_classes
            labels.append(composed)

    graph = DeltaGraph(
        node_features=node_features,
        edge_features=torch.stack(edge_feats),
        edge_index=torch.tensor([src_list, tgt_list], dtype=torch.long),
    )

    return graph, torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Models for comparison
# ---------------------------------------------------------------------------

class FixedTopologyDELTA(nn.Module):
    """DELTA with fixed (given) graph topology — baseline from Phase 27b."""

    def __init__(self, d_node, d_edge, num_classes, num_layers=3, num_heads=4):
        super().__init__()
        self.model = DELTAModel(
            d_node=d_node, d_edge=d_edge, num_layers=num_layers,
            num_heads=num_heads, num_classes=num_classes,
        )

    def forward(self, graph):
        return self.model(graph)

    def classify_edges(self, graph):
        return self.model.classify_edges(graph)


class AugmentedDELTA(nn.Module):
    """DELTA with task-aware constructor that adds edges to base topology."""

    def __init__(self, d_node, d_edge, num_classes, num_layers=3, num_heads=4,
                 edge_threshold=0.3):
        super().__init__()
        self.constructor = TaskAwareConstructor(d_node, d_edge, num_heads,
                                                edge_threshold)
        self.model = DELTAModel(
            d_node=d_node, d_edge=d_edge, num_layers=num_layers,
            num_heads=num_heads, num_classes=num_classes,
        )

    def forward(self, graph):
        # Augment graph with learned edges (preserves base topology)
        augmented = self.constructor(graph)
        return self.model(augmented)

    def classify_edges(self, graph):
        return self.model.classify_edges(graph)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(model, graph, labels, epochs=200, lr=1e-3,
                       seed=42, device='cpu'):
    """Train and evaluate, returns best test accuracy."""
    torch.manual_seed(seed)
    model = model.to(device)
    graph = graph.to(device)
    labels = labels.to(device)

    E = labels.shape[0]
    perm = torch.randperm(E, device=device)
    train_idx = perm[:int(E * 0.7)]
    test_idx = perm[int(E * 0.7):]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_test_acc = 0.0

    for epoch in range(epochs):
        model.train()
        out = model(graph)
        logits = model.classify_edges(out)
        # Only compute loss on original edges (not augmented ones)
        loss = F.cross_entropy(logits[:E][train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                out = model(graph)
                logits = model.classify_edges(out)
                test_acc = (logits[:E][test_idx].argmax(-1) == labels[test_idx]).float().mean().item()
                best_test_acc = max(best_test_acc, test_acc)

    return best_test_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 33: Task-Aware Graph Construction")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    d_node, d_edge = 64, 32

    print("=" * 70)
    print("PHASE 33: Task-Aware Graph Construction")
    print("=" * 70)
    print(f"  Device: {device}, Epochs: {args.epochs}, Seeds: {args.seeds}")
    print()
    print("  Hypothesis: Preserving base topology + learning new edges")
    print("  should outperform both fixed topology and attention-thresholded")
    print("  construction on path composition tasks.")
    print()

    fixed_accs = []
    augmented_accs = []

    for seed in range(args.seeds):
        print(f"--- Seed {seed} ---")

        graph, labels = create_path_task(
            num_entities=60, num_relations=4, num_paths=150,
            path_length=3, d_node=d_node, d_edge=d_edge, seed=seed,
        )
        num_classes = labels.max().item() + 1
        print(f"  Graph: {graph.num_nodes} nodes, {graph.num_edges} edges, "
              f"{num_classes} classes")

        # Fixed topology (Phase 27b baseline)
        fixed_model = FixedTopologyDELTA(d_node, d_edge, num_classes)
        fixed_acc = train_and_evaluate(fixed_model, graph, labels,
                                       args.epochs, 1e-3, seed, device)
        fixed_accs.append(fixed_acc)
        print(f"  Fixed Topology:  {fixed_acc:.3f}")

        # Task-aware augmented
        aug_model = AugmentedDELTA(d_node, d_edge, num_classes,
                                    edge_threshold=0.3)
        aug_acc = train_and_evaluate(aug_model, graph, labels,
                                     args.epochs, 1e-3, seed, device)
        augmented_accs.append(aug_acc)
        print(f"  Augmented:       {aug_acc:.3f}")
        print()

    # Summary
    print("=" * 70)
    print("PHASE 33 RESULTS")
    print("=" * 70)

    fixed_mean = sum(fixed_accs) / len(fixed_accs)
    aug_mean = sum(augmented_accs) / len(augmented_accs)

    print(f"  Fixed Topology:  {fixed_mean:.3f} (mean over {args.seeds} seeds)")
    print(f"  Augmented:       {aug_mean:.3f} (mean over {args.seeds} seeds)")
    print(f"  Difference:      {aug_mean - fixed_mean:+.3f}")
    print()

    if aug_mean > fixed_mean:
        print("  Task-aware augmentation improves over fixed topology. ✓")
        print("  The constructor learns useful long-range connections.")
    else:
        print("  Fixed topology still wins — base structure is sufficient here.")
        print("  Consider: harder tasks with missing edges, or lower threshold.")

    print()
    print("  Next steps:")
    print("    - Test on larger graphs (Phase 31 compute)")
    print("    - Compare with standard GraphConstructor (Phase 27b)")
    print("    - Integrate into DELTAModel as optional constructor mode")


if __name__ == '__main__':
    main()
