"""
Phase 32: Cross-Graph Transfer — Train FB15k-237, Eval WN18RR

Measures whether DELTA's edge-attention representations generalize across
knowledge graph domains without retraining. This is a zero-shot transfer test:
train on Freebase (entities = companies, people, movies), evaluate on WordNet
(entities = word senses, hypernyms, synonyms).

Protocol:
  1. Train DELTA on FB15k-237-like data (relation classification)
  2. Freeze encoder, evaluate on WN18RR-like data
  3. Compare with fine-tuned baseline and random baseline

If zero-shot transfer works → DELTA learns generalizable relational patterns.
If it doesn't → DELTA is domain-specific (still useful, but limits claims).

Requirements:
    - Phase 31 mini-batching (for full-scale; synthetic mode works without)
    - GPU recommended for full-scale
    - pip install torch numpy

Usage:
    python experiments/phase32_cross_graph_transfer.py [--source_entities 500]
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
from delta.utils import create_realistic_kg_benchmark

# Import mini-batch sampler from Phase 31
from experiments.phase31_mini_batching import NeighborSampler


# ---------------------------------------------------------------------------
# Domain-specific data generation
# ---------------------------------------------------------------------------

def create_domain_data(num_entities, num_relations, d_node, d_edge,
                       domain_seed, domain_offset=0.0, num_triples=None):
    """Create a domain-specific KG benchmark.

    Different domains are simulated by shifting the feature prototypes
    (domain_offset) while preserving the relational structure.

    Args:
        num_entities: number of entities
        num_relations: number of relation types
        d_node: node feature dimension
        d_edge: edge feature dimension
        domain_seed: random seed for this domain
        domain_offset: offset applied to feature prototypes (simulates domain shift)
        num_triples: number of triples (defaults to ~21x num_entities if None)

    Returns:
        (graph, labels, num_relations)
    """
    if num_triples is None:
        num_triples = num_entities * 21  # ~21 edges/entity (FB15k-237 density)
    graph, labels, metadata = create_realistic_kg_benchmark(
        num_entities=num_entities,
        num_triples=num_triples,
        d_node=d_node, d_edge=d_edge,
        seed=domain_seed,
    )

    # Apply domain shift to features
    if domain_offset != 0.0:
        torch.manual_seed(domain_seed + 1000)
        shift = torch.randn_like(graph.node_features) * domain_offset
        graph = DeltaGraph(
            node_features=graph.node_features + shift,
            edge_features=graph.edge_features,
            edge_index=graph.edge_index,
            node_tiers=graph.node_tiers,
            node_importance=graph.node_importance,
            edge_importance=graph.edge_importance,
        )

    return graph, labels, metadata['num_relations']


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_model(model, graph, labels, epochs, lr, device, log_every=20,
                sampler=None, batch_size=64, accum_steps=4, patience=0):
    """Train model on a source domain. Uses mini-batching if sampler is provided.

    Args:
        patience: Early stopping patience (0 = disabled). Stops after this many
                  consecutive evaluation rounds with no improvement.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if sampler is None:
        # --- Full-graph training (synthetic scale) ---
        graph = graph.to(device)
        labels = labels.to(device)
        E = labels.shape[0]
        perm = torch.randperm(E, device=device)
        train_idx = perm[:int(E * 0.8)]
        val_idx = perm[int(E * 0.8):]
        best_val_acc = 0.0
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            out = model(graph)
            logits = model.classify_edges(out)
            loss = F.cross_entropy(logits[train_idx], labels[train_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
                model.eval()
                with torch.no_grad():
                    out = model(graph)
                    logits = model.classify_edges(out)
                    val_acc = (logits[val_idx].argmax(-1) == labels[val_idx]).float().mean().item()
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        no_improve = 0
                    else:
                        no_improve += 1
                    print(f"    Epoch {epoch+1:3d}  Val Acc: {val_acc:.3f}  Best: {best_val_acc:.3f}")
                    if patience > 0 and no_improve >= patience:
                        print(f"    Early stopping at epoch {epoch+1} (patience={patience})")
                        break

        return best_val_acc

    # --- Mini-batch training (full scale) ---
    E = labels.shape[0]
    all_edges = list(range(E))
    split = int(E * 0.8)
    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        random.shuffle(all_edges)
        train_edges = all_edges[:split]
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for batch_start in range(0, len(train_edges), batch_size):
            batch_edges = train_edges[batch_start:batch_start + batch_size]
            mini_graph, mini_labels, target_idx = sampler.sample_subgraph(
                batch_edges, graph.node_features, graph.edge_features, labels)
            if mini_graph is None:
                continue
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

        if num_batches % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # Evaluate on validation batches
        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            model.eval()
            val_edges = all_edges[split:]
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_start in range(0, min(len(val_edges), batch_size * 5), batch_size):
                    batch = val_edges[batch_start:batch_start + batch_size]
                    mini_graph, mini_labels, target_idx = sampler.sample_subgraph(
                        batch, graph.node_features, graph.edge_features, labels)
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

            val_acc = correct / max(total, 1)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
            else:
                no_improve += 1
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"    Epoch {epoch+1:3d}  Loss: {avg_loss:.4f}  "
                  f"Val Acc: {val_acc:.3f}  Best: {best_val_acc:.3f}")
            if patience > 0 and no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1} (patience={patience})")
                break

    return best_val_acc


def evaluate_zero_shot(model, graph, labels, device, sampler=None, batch_size=64):
    """Evaluate a frozen model on a target domain (zero-shot transfer)."""
    model = model.to(device)
    model.eval()

    if sampler is None:
        # Full-graph evaluation
        graph = graph.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            out = model(graph)
            logits = model.classify_edges(out)
            acc = (logits.argmax(-1) == labels).float().mean().item()
        return acc

    # Mini-batch evaluation
    all_edges = list(range(labels.shape[0]))
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_start in range(0, len(all_edges), batch_size):
            batch = all_edges[batch_start:batch_start + batch_size]
            mini_graph, mini_labels, target_idx = sampler.sample_subgraph(
                batch, graph.node_features, graph.edge_features, labels)
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

    return correct / max(total, 1)


def evaluate_fine_tuned(model, graph, labels, epochs, lr, device,
                        sampler=None, batch_size=64, accum_steps=4, patience=0):
    """Fine-tune on target domain (few-shot) and evaluate.

    Args:
        patience: Early stopping patience (0 = disabled). Stops after this many
                  consecutive evaluation rounds with no improvement in fine-tuning loss.
    """
    model = model.to(device)

    if sampler is None:
        # Full-graph fine-tuning
        graph = graph.to(device)
        labels = labels.to(device)
        E = labels.shape[0]
        perm = torch.randperm(E, device=device)
        finetune_idx = perm[:int(E * 0.2)]
        test_idx = perm[int(E * 0.2):]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.1)
        best_loss = float('inf')
        no_improve = 0
        for epoch in range(epochs // 2):
            model.train()
            out = model(graph)
            logits = model.classify_edges(out)
            loss = F.cross_entropy(logits[finetune_idx], labels[finetune_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss - 1e-4:
                best_loss = loss_val
                no_improve = 0
            else:
                no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"    Fine-tune early stop at epoch {epoch+1} (patience={patience})")
                break

        model.eval()
        with torch.no_grad():
            out = model(graph)
            logits = model.classify_edges(out)
            acc = (logits[test_idx].argmax(-1) == labels[test_idx]).float().mean().item()
        return acc

    # Mini-batch fine-tuning
    E = labels.shape[0]
    all_edges = list(range(E))
    random.shuffle(all_edges)
    finetune_edges = all_edges[:int(E * 0.2)]
    test_edges = all_edges[int(E * 0.2):]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.1)
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs // 2):
        model.train()
        random.shuffle(finetune_edges)
        optimizer.zero_grad()
        num_batches = 0
        epoch_loss = 0.0

        for batch_start in range(0, len(finetune_edges), batch_size):
            batch = finetune_edges[batch_start:batch_start + batch_size]
            mini_graph, mini_labels, target_idx = sampler.sample_subgraph(
                batch, graph.node_features, graph.edge_features, labels)
            if mini_graph is None:
                continue
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

        if num_batches % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(num_batches, 1)
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
        if patience > 0 and no_improve >= patience:
            print(f"    Fine-tune early stop at epoch {epoch+1} (patience={patience})")
            break

    # Evaluate on test edges
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_start in range(0, len(test_edges), batch_size):
            batch = test_edges[batch_start:batch_start + batch_size]
            mini_graph, mini_labels, target_idx = sampler.sample_subgraph(
                batch, graph.node_features, graph.edge_features, labels)
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

    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 32: Cross-Graph Transfer")
    parser.add_argument('--source_entities', type=int, default=500,
                        help='Entities in source domain')
    parser.add_argument('--target_entities', type=int, default=300,
                        help='Entities in target domain')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs for source domain')
    parser.add_argument('--log_every', type=int, default=None,
                        help='Log every N epochs (default: 1 when --full, 20 otherwise)')
    parser.add_argument('--full', action='store_true',
                        help='Run full-scale: FB15k-237 (14505) -> WN18RR (40943)')
    parser.add_argument('--patience', type=int, default=0,
                        help='Early stopping patience (0 = disabled). Stops after N '
                             'consecutive eval rounds with no improvement.')
    parser.add_argument('--finetune_epochs', type=int, default=None,
                        help='Fine-tuning epochs (default: epochs // 2)')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    d_node, d_edge = 64, 32

    # Resolve log_every default based on --full
    if args.log_every is None:
        args.log_every = 1 if args.full else 20

    # Default patience: 10 eval rounds when --full (saves hours), disabled otherwise
    if args.patience == 0 and args.full:
        args.patience = 10

    # Default finetune epochs
    if args.finetune_epochs is None:
        args.finetune_epochs = args.epochs // 2

    if args.full:
        args.source_entities = 14505
        args.target_entities = 40943
        if device == 'cuda':
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU detected: {torch.cuda.get_device_name(0)} ({vram_gb:.0f}GB)")
            # Auto-scale mini-batch params
            if vram_gb >= 70:  # H100 80GB
                args.max_neighbors = 500
                args.batch_size = 64
                print(f"  H100 scaling: max_neighbors={args.max_neighbors}, "
                      f"batch_size={args.batch_size}")
            elif vram_gb >= 35:  # A100 40GB
                args.max_neighbors = 200
                args.batch_size = 32
                print(f"  A100 scaling: max_neighbors={args.max_neighbors}, "
                      f"batch_size={args.batch_size}")
            else:
                args.max_neighbors = 100
                args.batch_size = 16
        if device == 'cpu':
            print("WARNING: --full requires GPU for realistic scale.")

    print("=" * 70)
    print("PHASE 32: Cross-Graph Transfer")
    print("=" * 70)
    print(f"  Source: {args.source_entities} entities, "
          f"Target: {args.target_entities} entities")
    print(f"  Device: {device}, Epochs: {args.epochs}, "
          f"Log every: {args.log_every} epoch(s)")
    patience_str = f", Patience: {args.patience}" if args.patience > 0 else ""
    print(f"  Fine-tune epochs: {args.finetune_epochs}{patience_str}")
    print()

    # --- Create source domain (FB15k-237-like) ---
    print("Creating source domain data (FB15k-237-like)...")
    src_graph, src_labels, num_relations = create_domain_data(
        args.source_entities, 20, d_node, d_edge,
        domain_seed=42, domain_offset=0.0,
        num_triples=args.source_entities * 21,  # ~FB15k-237 density
    )
    print(f"  Source: {src_graph.num_nodes} nodes, {src_graph.num_edges} edges, "
          f"{num_relations} relations")

    # --- Create target domain (WN18RR-like, with domain shift) ---
    print("Creating target domain data (WN18RR-like)...")
    tgt_graph, tgt_labels, tgt_num_relations = create_domain_data(
        args.target_entities, 20, d_node, d_edge,
        domain_seed=123, domain_offset=0.3,
        num_triples=args.target_entities * 2,  # ~WN18RR density (~93K / 40943 ≈ 2.3x)
    )
    # Remap labels to match source relation count if they differ
    tgt_labels = tgt_labels % num_relations
    print(f"  Target: {tgt_graph.num_nodes} nodes, {tgt_graph.num_edges} edges")
    print()

    # --- Create samplers for full-scale mini-batching ---
    src_sampler = None
    tgt_sampler = None
    batch_size = getattr(args, 'batch_size', 64)
    max_neighbors = getattr(args, 'max_neighbors', 100)

    if args.full:
        print("Creating source domain sampler (mini-batch training)...")
        src_sampler = NeighborSampler(
            src_graph.edge_index, src_graph.num_nodes,
            k_hops=2, max_neighbors=max_neighbors)
        print("Creating target domain sampler (mini-batch evaluation)...")
        tgt_sampler = NeighborSampler(
            tgt_graph.edge_index, tgt_graph.num_nodes,
            k_hops=2, max_neighbors=max_neighbors)
        print()

    # --- Train on source domain ---
    print("Training DELTA on source domain...")
    model = DELTAModel(
        d_node=d_node, d_edge=d_edge, num_layers=3, num_heads=4,
        num_classes=num_relations,
    )
    source_acc = train_model(model, src_graph, src_labels,
                             args.epochs, 1e-3, device,
                             log_every=args.log_every,
                             sampler=src_sampler,
                             batch_size=batch_size,
                             patience=args.patience)
    print(f"  Source domain accuracy: {source_acc:.3f}")
    print()

    # --- Zero-shot transfer to target domain ---
    print("Zero-shot transfer to target domain (frozen encoder)...")
    zero_shot_acc = evaluate_zero_shot(model, tgt_graph, tgt_labels, device,
                                       sampler=tgt_sampler, batch_size=batch_size)
    print(f"  Zero-shot accuracy: {zero_shot_acc:.3f}")

    # --- Fine-tuned transfer ---
    print(f"Fine-tuned transfer (20% target data, {args.finetune_epochs} epochs)...")
    import copy
    model_ft = copy.deepcopy(model)
    fine_tuned_acc = evaluate_fine_tuned(model_ft, tgt_graph, tgt_labels,
                                         args.finetune_epochs * 2, 1e-3, device,
                                         sampler=tgt_sampler,
                                         batch_size=batch_size,
                                         patience=args.patience)
    print(f"  Fine-tuned accuracy: {fine_tuned_acc:.3f}")

    # --- Random baseline ---
    random_acc = 1.0 / num_relations
    print(f"  Random baseline: {random_acc:.3f}")
    print()

    # --- Summary ---
    print("=" * 70)
    print("PHASE 32 RESULTS")
    print("=" * 70)
    print(f"  Source domain accuracy:  {source_acc:.3f}")
    print(f"  Zero-shot transfer:      {zero_shot_acc:.3f}")
    print(f"  Fine-tuned transfer:     {fine_tuned_acc:.3f}")
    print(f"  Random baseline:         {random_acc:.3f}")
    print()

    if zero_shot_acc > random_acc * 2:
        print("  Zero-shot transfer shows meaningful generalization. ✓")
    else:
        print("  Zero-shot transfer is weak — features may be domain-specific.")

    if fine_tuned_acc > zero_shot_acc + 0.05:
        print("  Fine-tuning improves significantly → pre-training helps.")
    elif fine_tuned_acc > zero_shot_acc:
        print("  Fine-tuning provides modest improvement.")
    else:
        print("  Fine-tuning doesn't help much → consider more target data.")

    print()
    print("  Next steps:")
    print("    - Run on full FB15k-237 → WN18RR with Phase 31 mini-batching")
    print("    - Compare DELTA transfer vs GraphGPS/GRIT transfer")


if __name__ == '__main__':
    main()
