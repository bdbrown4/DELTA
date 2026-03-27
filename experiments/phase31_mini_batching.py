"""
Phase 31: Mini-Batching for Full-Scale FB15k-237

Phase 25/30 maxed at 2000 entities (69K edges) on a 12 GB GPU. Real KGs have
millions of entities. This phase implements subgraph sampling + gradient
accumulation to scale DELTA beyond single-GPU VRAM.

Key technique: Neighbor sampling.
  - For each training edge, sample a k-hop neighborhood around src/tgt nodes
  - Build a mini-graph from the sampled subgraph
  - Run DELTA on the mini-graph, accumulate gradients
  - Step optimizer every `accum_steps` mini-graphs

This combines the gradient accumulation proven in Phase 27b with subgraph
sampling to enable full FB15k-237 (14,505 entities, 310K edges) training.

Requirements:
    - GPU recommended (H100 80GB ideal, A100 40GB good, any GPU for validation)
    - pip install torch numpy
    - For real FB15k-237: download dataset first (see Phase 25)

Usage:
    python experiments/phase31_mini_batching.py [--num_entities 500] [--epochs 50]
    python experiments/phase31_mini_batching.py --full  # Full FB15k-237 (GPU required)
    python experiments/phase31_mini_batching.py --full --max_neighbors 500  # H100 80GB
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
from collections import defaultdict

from delta.graph import DeltaGraph
from delta.model import DELTAModel
from delta.utils import create_realistic_kg_benchmark


# ---------------------------------------------------------------------------
# Subgraph sampler
# ---------------------------------------------------------------------------

class NeighborSampler:
    """Sample k-hop neighborhoods around target edges for mini-batch training.

    For each edge (s, t), sample up to `max_neighbors` nodes within `k_hops`
    of both s and t. Build a mini-graph from the sampled subgraph.
    """

    def __init__(self, edge_index, num_nodes, k_hops=2, max_neighbors=50):
        self.num_nodes = num_nodes
        self.k_hops = k_hops
        self.max_neighbors = max_neighbors

        # Build adjacency list for fast neighbor lookup
        self.adj = defaultdict(set)
        src, tgt = edge_index[0].tolist(), edge_index[1].tolist()
        for s, t in zip(src, tgt):
            self.adj[s].add(t)
            self.adj[t].add(s)

        # Store edge index for subgraph extraction
        self.edge_index = edge_index

    def sample_neighborhood(self, seed_nodes):
        """BFS expansion from seed nodes up to k_hops, capped at max_neighbors."""
        visited = set(seed_nodes)
        frontier = set(seed_nodes)

        for _ in range(self.k_hops):
            next_frontier = set()
            for node in frontier:
                neighbors = self.adj.get(node, set())
                for n in neighbors:
                    if n not in visited:
                        next_frontier.add(n)
            visited.update(next_frontier)
            frontier = next_frontier

            if len(visited) >= self.max_neighbors:
                limited = sorted(visited)[:self.max_neighbors]
                visited = set(limited)
                break

        return sorted(visited)

    def sample_subgraph(self, target_edges, node_features, edge_features,
                        edge_labels):
        """Build a mini-graph from neighborhoods around target edges.

        Args:
            target_edges: list of edge indices to include
            node_features: [N, d_node] full node features
            edge_features: [E, d_edge] full edge features
            edge_labels: [E] edge labels

        Returns:
            (mini_graph, mini_labels, local_target_idx) tuple
        """
        # Collect seed nodes from target edges
        src_nodes = self.edge_index[0, target_edges].tolist()
        tgt_nodes = self.edge_index[1, target_edges].tolist()
        seed_nodes = list(set(src_nodes + tgt_nodes))

        # BFS expand
        sampled_nodes = self.sample_neighborhood(seed_nodes)
        node_set = set(sampled_nodes)

        # Build node mapping: global → local
        node_map = {g: l for l, g in enumerate(sampled_nodes)}
        N_local = len(sampled_nodes)

        # Find all edges within the sampled subgraph using a vectorized mask
        device = node_features.device
        edge_index = self.edge_index.to(device)
        full_src = edge_index[0]
        full_tgt = edge_index[1]

        # Tensor of sampled node ids for membership checks
        sampled_idx = torch.tensor(sampled_nodes, dtype=torch.long, device=device)

        # Boolean mask: edges where both endpoints are in the sampled node set
        src_in = torch.isin(full_src, sampled_idx)
        tgt_in = torch.isin(full_tgt, sampled_idx)
        edge_mask = src_in & tgt_in
        selected_edges = edge_mask.nonzero(as_tuple=False).view(-1)

        if selected_edges.numel() == 0:
            return None, None, None

        # Global node ids of endpoints of selected edges
        sel_src_global = full_src[selected_edges]
        sel_tgt_global = full_tgt[selected_edges]

        # Map global node ids to local node ids
        local_edges_src = [node_map[int(s)] for s in sel_src_global.tolist()]
        local_edges_tgt = [node_map[int(t)] for t in sel_tgt_global.tolist()]

        # Slice edge features and labels for selected edges
        local_edge_feats = edge_features[selected_edges]
        local_edge_labels = edge_labels[selected_edges]

        # Map global edge index to local edge index
        global_to_local_edge = {
            int(global_e_idx): int(local_idx)
            for local_idx, global_e_idx in enumerate(selected_edges.tolist())
        }

        sampled_idx = sampled_idx.to(device)

        mini_graph = DeltaGraph(
            node_features=node_features[sampled_idx],
            edge_features=local_edge_feats,
            edge_index=torch.stack(
                [
                    torch.tensor(local_edges_src, dtype=torch.long, device=device),
                    torch.tensor(local_edges_tgt, dtype=torch.long, device=device),
                ],
                dim=0,
            ),
        )

        mini_labels = torch.tensor(local_edge_labels, dtype=torch.long,
                                   device=device)

        # Map target edges to local indices
        local_targets = []
        for e_idx in target_edges:
            e_int = e_idx.item() if isinstance(e_idx, torch.Tensor) else e_idx
            if e_int in global_to_local_edge:
                local_targets.append(global_to_local_edge[e_int])

        if not local_targets:
            return None, None, None

        local_target_idx = torch.tensor(local_targets, dtype=torch.long,
                                        device=device)

        return mini_graph, mini_labels, local_target_idx


# ---------------------------------------------------------------------------
# Training with mini-batching + gradient accumulation
# ---------------------------------------------------------------------------

def train_with_mini_batching(model, full_graph, labels, sampler,
                             epochs=50, lr=1e-3, batch_size=32,
                             accum_steps=4, device='cpu'):
    """Train DELTA with mini-batch subgraph sampling.

    Args:
        model: DELTAModel with classification head
        full_graph: full DeltaGraph (stays on CPU if needed, subgraphs go to device)
        labels: [E] edge labels
        sampler: NeighborSampler
        epochs: number of epochs
        lr: learning rate
        batch_size: edges per mini-batch
        accum_steps: gradient accumulation steps
        device: 'cpu' or 'cuda'

    Returns:
        dict with training metrics
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    E = labels.shape[0]
    all_edges = list(range(E))
    split = int(E * 0.7)

    best_test_acc = 0.0
    train_losses = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        random.shuffle(all_edges)
        train_edges = all_edges[:split]
        test_edges = all_edges[split:]

        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for batch_start in range(0, len(train_edges), batch_size):
            batch_edges = train_edges[batch_start:batch_start + batch_size]

            mini_graph, mini_labels, target_idx = sampler.sample_subgraph(
                batch_edges,
                full_graph.node_features,
                full_graph.edge_features,
                labels,
            )

            if mini_graph is None:
                continue

            # Move mini-graph to device
            mini_graph = mini_graph.to(device)
            mini_labels = mini_labels.to(device)
            target_idx = target_idx.to(device)

            out = model(mini_graph)
            logits = model.classify_edges(out)
            loss = F.cross_entropy(logits[target_idx], mini_labels[target_idx])
            loss = loss / accum_steps
            loss.backward()

            epoch_loss += loss.item() * accum_steps
            num_batches += 1

            if num_batches % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Final accumulation step
        if num_batches % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        # Evaluate on test edges (sample a few batches)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_start in range(0, min(len(test_edges), batch_size * 5),
                                         batch_size):
                    batch = test_edges[batch_start:batch_start + batch_size]
                    mini_graph, mini_labels, target_idx = sampler.sample_subgraph(
                        batch,
                        full_graph.node_features,
                        full_graph.edge_features,
                        labels,
                    )
                    if mini_graph is None:
                        continue
                    mini_graph = mini_graph.to(device)
                    mini_labels = mini_labels.to(device)
                    target_idx = target_idx.to(device)

                    out = model(mini_graph)
                    logits = model.classify_edges(out)
                    preds = logits[target_idx].argmax(-1)
                    correct += (preds == mini_labels[target_idx]).sum().item()
                    total += len(target_idx)

            test_acc = correct / max(total, 1)
            best_test_acc = max(best_test_acc, test_acc)
            print(f"  Epoch {epoch+1:3d}  Loss: {avg_loss:.4f}  "
                  f"Test Acc: {test_acc:.3f}  Best: {best_test_acc:.3f}")

    return {
        'best_test_acc': best_test_acc,
        'final_loss': train_losses[-1] if train_losses else 0.0,
        'training_time_s': time.time() - start_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 31: Mini-Batching")
    parser.add_argument('--num_entities', type=int, default=500,
                        help='Number of entities (500 for quick test, 14505 for full)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Edges per batch')
    parser.add_argument('--accum_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--k_hops', type=int, default=2,
                        help='Neighborhood hops for sampling')
    parser.add_argument('--max_neighbors', type=int, default=100,
                        help='Max nodes per subgraph (100=A100, 300-500=H100 80GB)')
    parser.add_argument('--full', action='store_true',
                        help='Run on full FB15k-237 (requires GPU)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu/cuda)')
    args = parser.parse_args()

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    if args.full:
        args.num_entities = 14505
        if device == 'cpu':
            print("WARNING: --full requires GPU. Use --device cuda or run on Colab.")
        # Auto-scale subgraph size based on available VRAM
        if device == 'cuda' and args.max_neighbors == 100:
            vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            if vram_gb >= 70:  # H100 80GB
                args.max_neighbors = 500
                args.batch_size = 64
                print(f"  H100 detected ({vram_gb:.0f}GB) — scaling up: "
                      f"max_neighbors={args.max_neighbors}, batch_size={args.batch_size}")
            elif vram_gb >= 35:  # A100 40GB
                args.max_neighbors = 200
                args.batch_size = 32
                print(f"  A100 detected ({vram_gb:.0f}GB) — scaling up: "
                      f"max_neighbors={args.max_neighbors}, batch_size={args.batch_size}")

    print("=" * 70)
    print("PHASE 31: Mini-Batching for Full-Scale KG Training")
    print("=" * 70)
    print(f"  Entities: {args.num_entities}, Epochs: {args.epochs}, "
          f"Batch: {args.batch_size}, Accum: {args.accum_steps}")
    print(f"  Neighborhood: {args.k_hops} hops, max {args.max_neighbors} nodes")
    print(f"  Device: {device}")
    print()

    # Create data
    print("Creating benchmark data...")
    graph, labels, metadata = create_realistic_kg_benchmark(
        num_entities=args.num_entities,
        d_node=64, d_edge=32,
        seed=42,
    )
    num_classes = metadata['num_relations']
    print(f"  Nodes: {graph.num_nodes}, Edges: {graph.num_edges}, "
          f"Relations: {num_classes}")

    # Create sampler
    sampler = NeighborSampler(
        graph.edge_index, graph.num_nodes,
        k_hops=args.k_hops, max_neighbors=args.max_neighbors,
    )

    # Create model
    model = DELTAModel(
        d_node=64, d_edge=32, num_layers=3, num_heads=4,
        num_classes=num_classes,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    print()

    # Train
    print("Training with mini-batch subgraph sampling...")
    results = train_with_mini_batching(
        model, graph, labels, sampler,
        epochs=args.epochs, lr=1e-3,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        device=device,
    )

    print()
    print("=" * 70)
    print("PHASE 31 RESULTS")
    print("=" * 70)
    print(f"  Best test accuracy: {results['best_test_acc']:.3f}")
    print(f"  Final loss: {results['final_loss']:.4f}")
    print(f"  Training time: {results['training_time_s']:.1f}s")
    print()

    if results['best_test_acc'] > 0.5:
        print("  Mini-batching works — DELTA trains with subgraph sampling. ✓")
    else:
        print("  Low accuracy — may need more epochs, larger neighborhoods, "
              "or tuning.")

    print()
    print("  Next steps:")
    print("    - Run with --full on H100/A100 GPU for full FB15k-237")
    print("    - Compare with Phase 25 full-graph results (97.6%)")
    print("    - Enable Phase 32 (cross-domain) and Phase 34b (full-scale comparison)")


if __name__ == '__main__':
    main()
