"""
Phase 27: Bootstrap on a Relational Task

Phase 5 confirmed the Transformer→DELTA pipeline preserves accuracy at 98.3%
on a sequence classification task, but that task has no relational structure.

Goal: design a task where multi-hop graph structure is necessary and measure
whether the transformer bootstrap produces a better starting graph than a
fixed construction strategy.

Task: 2-hop path composition with distractors.
  Input: 12-token sequence with a 5-token relational path [ent, rel1, ent, rel2, ent]
  placed at a random position, surrounded by noise tokens (including false
  relation-range tokens).
  Label: (rel1, rel2) ordered-pair composition class.

Models:
  1. TransformerBaseline — sequence → mean-pool → classify
  2. BootstrappedDELTA  — transformer attention → GraphConstructor → DELTA → classify
  3. FixedChainDELTA    — transformer embeddings + fixed chain graph → DELTA → classify

Hypothesis: attention-based graph construction learns to connect path tokens,
giving DELTA structural advantage over fixed topology and pure sequence processing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.model import DELTAModel, DELTALayer
from delta.constructor import MiniTransformerBlock, PositionalEncoding
from delta.graph import DeltaGraph


# ──────────────────────────────────────────────────────────────────────
#  Task generation
# ──────────────────────────────────────────────────────────────────────

def generate_path_composition_task(
    num_samples=500,
    num_entity_types=8,
    num_relations=4,
    path_length=5,
    total_length=12,
    seed=42,
):
    """2-hop path composition with distractors.

    Path tokens: [ent, rel1, ent, rel2, ent] placed at a random start
    position within a 12-token sequence. Remaining positions filled with
    noise from the full vocab range (including false relation tokens).

    Label = index of (rel1, rel2) ordered pair.

    Returns:
        data: [N, total_length] token tensor
        labels: [N] class indices
        vocab_size: total vocabulary size
        num_classes: number of composition classes
    """
    torch.manual_seed(seed)
    random.seed(seed)

    E_BASE = 1                        # entity tokens 1 .. num_entity_types
    R_BASE = E_BASE + num_entity_types  # relation tokens R_BASE .. R_BASE+num_relations-1
    VOCAB = R_BASE + num_relations

    compositions = [
        (r1, r2) for r1 in range(num_relations) for r2 in range(num_relations)
    ]
    num_classes = len(compositions)

    data, labels = [], []
    for _ in range(num_samples):
        comp_idx = random.randint(0, num_classes - 1)
        r1, r2 = compositions[comp_idx]

        e1 = E_BASE + random.randint(0, num_entity_types - 1)
        e2 = E_BASE + random.randint(0, num_entity_types - 1)
        e3 = E_BASE + random.randint(0, num_entity_types - 1)

        path = [e1, R_BASE + r1, e2, R_BASE + r2, e3]

        max_start = total_length - path_length
        start = random.randint(0, max_start)

        tokens = []
        for i in range(total_length):
            if start <= i < start + path_length:
                tokens.append(path[i - start])
            else:
                tokens.append(random.randint(1, VOCAB - 1))

        data.append(torch.tensor(tokens, dtype=torch.long))
        labels.append(comp_idx)

    return torch.stack(data), torch.tensor(labels), VOCAB, num_classes


# ──────────────────────────────────────────────────────────────────────
#  Model 1: Transformer Baseline
# ──────────────────────────────────────────────────────────────────────

class TransformerBaseline(nn.Module):
    """Pure transformer sequence classifier — no graph structure."""

    def __init__(self, vocab_size, d_model, num_classes, num_layers=2, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            MiniTransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, token_ids):
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        x = self.embedding(token_ids)
        x = self.pos_enc(x)
        for layer in self.layers:
            x, _ = layer(x)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


# ──────────────────────────────────────────────────────────────────────
#  Model 2: Bootstrapped DELTA (attention → GraphConstructor → DELTA)
# ──────────────────────────────────────────────────────────────────────

class BootstrappedDELTA(nn.Module):
    """Transformer attention → GraphConstructor → DELTA → classify.

    Uses DELTAModel with use_constructor=True.  The GraphConstructor
    thresholds transformer attention to select edges, creating a
    task-adapted graph topology.
    """

    def __init__(self, vocab_size, d_model, d_node, d_edge, num_classes,
                 num_heads=4, delta_layers=2):
        super().__init__()
        self.delta = DELTAModel(
            d_node=d_node, d_edge=d_edge,
            num_layers=delta_layers, num_heads=num_heads,
            use_constructor=True,
            vocab_size=vocab_size, d_model=d_model,
            constructor_layers=2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_node, d_node),
            nn.GELU(),
            nn.Linear(d_node, num_classes),
        )

    def forward(self, token_ids):
        graph = self.delta(token_ids, use_router=False, use_memory=False)
        pooled = graph.node_features.mean(dim=0, keepdim=True)
        return self.classifier(pooled)


# ──────────────────────────────────────────────────────────────────────
#  Model 3: Fixed Chain DELTA (same transformer, fixed topology)
# ──────────────────────────────────────────────────────────────────────

class FixedChainDELTA(nn.Module):
    """Transformer embeddings → fixed bidirectional chain graph → DELTA → classify.

    Uses the same transformer for embeddings but builds a fixed chain
    topology (adjacent-token edges).  Represents a naive graph construction
    baseline — no attention-based edge selection.
    """

    def __init__(self, vocab_size, d_model, d_node, d_edge, num_classes,
                 num_heads=4, delta_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.tf_layers = nn.ModuleList([
            MiniTransformerBlock(d_model, num_heads) for _ in range(2)
        ])
        self.to_node = nn.Linear(d_model, d_node)
        self.to_edge = nn.Linear(2 * d_node, d_edge)
        self.delta_layers = nn.ModuleList([
            DELTALayer(d_node, d_edge, num_heads) for _ in range(delta_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_node, d_node),
            nn.GELU(),
            nn.Linear(d_node, num_classes),
        )

    def forward(self, token_ids):
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        x = self.embedding(token_ids)
        x = self.pos_enc(x)
        for layer in self.tf_layers:
            x, _ = layer(x)
        x = x.squeeze(0)  # [S, d_model]

        node_features = self.to_node(x)  # [S, d_node]
        S = node_features.shape[0]
        device = x.device

        # Bidirectional chain: 0↔1↔2↔…↔(S-1)
        fwd_src = torch.arange(S - 1, device=device)
        fwd_tgt = torch.arange(1, S, device=device)
        edge_index = torch.stack([
            torch.cat([fwd_src, fwd_tgt]),
            torch.cat([fwd_tgt, fwd_src]),
        ])

        src_feats = node_features[edge_index[0]]
        tgt_feats = node_features[edge_index[1]]
        edge_features = self.to_edge(torch.cat([src_feats, tgt_feats], dim=-1))

        graph = DeltaGraph(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
        )
        for delta_layer in self.delta_layers:
            graph = delta_layer(
                graph, use_router=False,
                use_partitioning=False, use_memory=False,
            )

        pooled = graph.node_features.mean(dim=0, keepdim=True)
        return self.classifier(pooled)


# ──────────────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────────────

def train_and_evaluate(name, model, train_data, train_labels,
                       test_data, test_labels, epochs=200, lr=1e-3):
    """Train model sample-by-sample and return (final_acc, elapsed_seconds)."""
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i in range(len(train_data)):
            logits = model(train_data[i].to(device))
            loss = F.cross_entropy(logits, train_labels[i:i + 1].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for i in range(len(test_data)):
                    logits = model(test_data[i].to(device))
                    if logits.argmax(-1).item() == test_labels[i].item():
                        correct += 1
            acc = correct / len(test_data)
            elapsed = time.time() - t0
            print(f"  [{name}] Epoch {epoch + 1}: "
                  f"Loss={total_loss / len(train_data):.4f}  "
                  f"Test Acc={acc:.1%}  ({elapsed:.1f}s)")

    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(len(test_data)):
            logits = model(test_data[i].to(device))
            if logits.argmax(-1).item() == test_labels[i].item():
                correct += 1
    final_acc = correct / len(test_data)
    total_time = time.time() - t0
    return final_acc, total_time


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 27: Bootstrap on a Relational Task")
    print("=" * 70)
    print()
    print("Task: 2-hop path composition with distractors.")
    print("  Input : 12-token sequence with [ent, rel1, ent, rel2, ent]")
    print("          at a random position; rest = noise (incl. false rels)")
    print("  Label : (rel1, rel2) ordered-pair composition class")
    print()
    print("Question: Does attention-based graph construction outperform")
    print("  fixed topology on a genuinely relational task?")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print()

    # ── Data ──────────────────────────────────────────────────────────
    data, labels, vocab_size, num_classes = generate_path_composition_task(
        num_samples=500, num_entity_types=8, num_relations=4,
        path_length=5, total_length=12, seed=42,
    )
    print(f"Vocab size: {vocab_size}  (8 entity types + 4 relations + pad)")
    print(f"Composition classes: {num_classes}")
    print(f"Samples: {len(data)}   Sequence length: {data.shape[1]}")

    perm = torch.randperm(len(data))
    train_end = int(len(data) * 0.7)
    train_data = data[perm[:train_end]]
    train_labels = labels[perm[:train_end]]
    test_data = data[perm[train_end:]]
    test_labels = labels[perm[train_end:]]
    print(f"Train: {len(train_data)}   Test: {len(test_data)}")
    print(f"Class balance: ~{len(train_data) // num_classes} train / class")
    print(f"Random baseline: {1.0 / num_classes:.1%}")
    print()

    d_model = 64
    d_node = 64
    d_edge = 32

    results = {}

    # ── Model 1: Transformer Baseline ─────────────────────────────────
    print("-" * 60)
    print("Model 1: Transformer Baseline")
    print("-" * 60)
    tf_model = TransformerBaseline(vocab_size, d_model, num_classes).to(device)
    n_params = sum(p.numel() for p in tf_model.parameters())
    print(f"  Parameters: {n_params:,}")
    acc, t = train_and_evaluate(
        "Transformer", tf_model,
        train_data, train_labels, test_data, test_labels, epochs=200,
    )
    results['Transformer'] = (acc, t)
    print(f"  >> Final: {acc:.1%}  ({t:.1f}s)\n")

    # ── Model 2: Bootstrapped DELTA ───────────────────────────────────
    print("-" * 60)
    print("Model 2: Bootstrapped DELTA (attention → graph → DELTA)")
    print("-" * 60)
    boot_model = BootstrappedDELTA(
        vocab_size, d_model, d_node, d_edge, num_classes,
    ).to(device)
    n_params = sum(p.numel() for p in boot_model.parameters())
    print(f"  Parameters: {n_params:,}")
    acc, t = train_and_evaluate(
        "Bootstrap", boot_model,
        train_data, train_labels, test_data, test_labels, epochs=200,
    )
    results['Bootstrapped DELTA'] = (acc, t)
    print(f"  >> Final: {acc:.1%}  ({t:.1f}s)\n")

    # ── Model 3: Fixed Chain DELTA ────────────────────────────────────
    print("-" * 60)
    print("Model 3: Fixed Chain DELTA (embeddings → chain graph → DELTA)")
    print("-" * 60)
    chain_model = FixedChainDELTA(
        vocab_size, d_model, d_node, d_edge, num_classes,
    ).to(device)
    n_params = sum(p.numel() for p in chain_model.parameters())
    print(f"  Parameters: {n_params:,}")
    acc, t = train_and_evaluate(
        "Chain", chain_model,
        train_data, train_labels, test_data, test_labels, epochs=200,
    )
    results['Fixed Chain DELTA'] = (acc, t)
    print(f"  >> Final: {acc:.1%}  ({t:.1f}s)\n")

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Random baseline: {1.0 / num_classes:.1%}  ({num_classes} classes)")
    print()
    best_name = max(results, key=lambda k: results[k][0])
    for name, (acc, t) in results.items():
        marker = " << BEST" if name == best_name else ""
        print(f"  {name:25s}  {acc:.1%}  ({t:.1f}s){marker}")
    print()

    # ── Analysis ──────────────────────────────────────────────────────
    tf_acc = results['Transformer'][0]
    boot_acc = results['Bootstrapped DELTA'][0]
    chain_acc = results['Fixed Chain DELTA'][0]

    print("Analysis:")
    if boot_acc > tf_acc:
        print(f"  Bootstrap DELTA > Transformer: {boot_acc:.1%} vs {tf_acc:.1%}")
        print("  Attention-based graph construction adds value on relational tasks.")
    else:
        print(f"  Transformer >= Bootstrap DELTA: {tf_acc:.1%} vs {boot_acc:.1%}")
        print("  Sequence processing sufficient for this task complexity.")

    if boot_acc > chain_acc:
        print(f"  Bootstrap DELTA > Fixed Chain:  {boot_acc:.1%} vs {chain_acc:.1%}")
        print("  Learned graph topology > fixed topology for relational reasoning.")
    else:
        print(f"  Fixed Chain >= Bootstrap DELTA: {chain_acc:.1%} vs {boot_acc:.1%}")
        print("  Fixed chain topology sufficient for sequential path structure.")

    if chain_acc > tf_acc:
        print(f"  Fixed Chain > Transformer:      {chain_acc:.1%} vs {tf_acc:.1%}")
        print("  Graph processing (even fixed) helps relational tasks.")
    else:
        print(f"  Transformer >= Fixed Chain:     {tf_acc:.1%} vs {chain_acc:.1%}")

    print()
    delta_bvc = boot_acc - chain_acc
    print(f"Key metric — Bootstrap vs Chain: {delta_bvc:+.1%}")
    if delta_bvc > 0.02:
        print("  YES — attention-based construction learns meaningful structure.")
    elif delta_bvc > -0.02:
        print("  MARGINAL — both approaches achieve similar accuracy.")
    else:
        print("  NO — fixed topology works as well; task may not need adaptive structure.")


if __name__ == '__main__':
    main()
