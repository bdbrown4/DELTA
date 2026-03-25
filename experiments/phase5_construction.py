"""
Phase 5: Validate Transformer-Bootstrapped Graph Construction

Core question: Can a transformer produce an initial graph that DELTA
processes more effectively than the transformer alone?

This validates the bootstrap strategy:
1. Transformer encodes text → embeddings + attention matrix
2. Graph constructor converts these into a DeltaGraph
3. DELTA processes the graph
4. Compare: transformer alone vs transformer → DELTA pipeline

Task: Simple sequence classification. Small enough to run on CPU.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.model import DELTAModel
from delta.constructor import GraphConstructor, MiniTransformerBlock, PositionalEncoding


class TransformerBaseline(nn.Module):
    """Pure transformer classifier — no graph construction."""
    def __init__(self, vocab_size, d_model, num_classes, num_layers=2, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            MiniTransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
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
        # Mean pool
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


class DeltaPipeline(nn.Module):
    """Transformer → Graph Constructor → DELTA → classify."""
    def __init__(self, vocab_size, d_model, d_node, d_edge, num_classes,
                 num_heads=4, delta_layers=2):
        super().__init__()
        self.delta = DELTAModel(
            d_node=d_node, d_edge=d_edge,
            num_layers=delta_layers, num_heads=num_heads,
            use_constructor=True,
            vocab_size=vocab_size, d_model=d_model,
            constructor_layers=2,
            num_classes=num_classes,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_node, d_node),
            nn.GELU(),
            nn.Linear(d_node, num_classes),
        )

    def forward(self, token_ids):
        graph = self.delta(token_ids, use_router=False, use_memory=False)
        # Mean pool node features
        pooled = graph.node_features.mean(dim=0, keepdim=True)
        return self.classifier(pooled)


def generate_synthetic_task(
    num_samples=200, seq_length=16, vocab_size=50, num_classes=3, seed=42
):
    """Generate a synthetic classification task.

    Each class has a distinctive token distribution pattern.
    The task is to classify sequences by pattern.
    """
    torch.manual_seed(seed)

    data = []
    labels = []

    for _ in range(num_samples):
        label = torch.randint(0, num_classes, (1,)).item()
        # Each class emphasizes different token ranges
        low = (vocab_size // num_classes) * label
        high = low + vocab_size // num_classes

        # Mix: 70% from class range, 30% random
        class_tokens = torch.randint(low, high, (int(seq_length * 0.7),))
        random_tokens = torch.randint(0, vocab_size, (seq_length - len(class_tokens),))
        tokens = torch.cat([class_tokens, random_tokens])
        # Shuffle
        tokens = tokens[torch.randperm(len(tokens))]

        data.append(tokens)
        labels.append(label)

    return torch.stack(data), torch.tensor(labels)


def main():
    print("=" * 70)
    print("PHASE 5: Graph Construction Validation")
    print("=" * 70)
    print()
    print("Task: Sequence classification from synthetic token patterns.")
    print("Question: Does transformer → DELTA pipeline add value over transformer alone?")
    print()

    vocab_size = 50
    d_model = 64
    d_node = 64
    d_edge = 32
    num_classes = 3
    seq_length = 16

    data, labels = generate_synthetic_task(
        num_samples=200, seq_length=seq_length,
        vocab_size=vocab_size, num_classes=num_classes,
    )

    # Train/test split
    perm = torch.randperm(len(data))
    train_end = int(len(data) * 0.7)
    train_data, train_labels = data[perm[:train_end]], labels[perm[:train_end]]
    test_data, test_labels = data[perm[train_end:]], labels[perm[train_end:]]

    print(f"Train: {len(train_data)}, Test: {len(test_data)}\n")

    # --- Transformer baseline ---
    print("--- Transformer Only ---")
    tf_model = TransformerBaseline(vocab_size, d_model, num_classes)
    optimizer = torch.optim.Adam(tf_model.parameters(), lr=1e-3)

    for epoch in range(150):
        tf_model.train()
        total_loss = 0
        for i in range(len(train_data)):
            logits = tf_model(train_data[i])
            loss = F.cross_entropy(logits, train_labels[i:i+1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            tf_model.eval()
            correct = 0
            with torch.no_grad():
                for i in range(len(test_data)):
                    logits = tf_model(test_data[i])
                    if logits.argmax(-1).item() == test_labels[i].item():
                        correct += 1
            acc = correct / len(test_data)
            print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_data):.4f}  Test Acc={acc:.3f}")

    tf_model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(len(test_data)):
            if tf_model(test_data[i]).argmax(-1).item() == test_labels[i].item():
                correct += 1
    tf_acc = correct / len(test_data)
    print(f"  Final: {tf_acc:.3f}\n")

    # --- DELTA pipeline ---
    print("--- Transformer → DELTA Pipeline ---")
    delta_model = DeltaPipeline(vocab_size, d_model, d_node, d_edge, num_classes)
    optimizer = torch.optim.Adam(delta_model.parameters(), lr=1e-3)

    for epoch in range(75):
        delta_model.train()
        total_loss = 0
        for i in range(len(train_data)):
            logits = delta_model(train_data[i])
            loss = F.cross_entropy(logits, train_labels[i:i+1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 25 == 0:
            delta_model.eval()
            correct = 0
            with torch.no_grad():
                for i in range(len(test_data)):
                    logits = delta_model(test_data[i])
                    if logits.argmax(-1).item() == test_labels[i].item():
                        correct += 1
            acc = correct / len(test_data)
            print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_data):.4f}  Test Acc={acc:.3f}")

    delta_model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(len(test_data)):
            if delta_model(test_data[i]).argmax(-1).item() == test_labels[i].item():
                correct += 1
    delta_acc = correct / len(test_data)
    print(f"  Final: {delta_acc:.3f}\n")

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Transformer only:     {tf_acc:.3f}")
    print(f"  Transformer → DELTA:  {delta_acc:.3f}")
    print(f"  Delta:                {delta_acc - tf_acc:+.3f}")
    print()
    if delta_acc > tf_acc:
        print(">> DELTA pipeline outperforms transformer alone.")
        print(">> Graph construction adds value — relational processing helps.")
    else:
        print(">> Transformer alone performed as well or better on this task.")
        print(">> This is expected for simple tasks — DELTA's advantage is relational.")
        print(">> Try a task with richer relational structure for a better test.")


if __name__ == '__main__':
    main()
