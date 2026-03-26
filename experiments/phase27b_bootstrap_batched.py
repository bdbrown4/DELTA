"""
Phase 27b: Bootstrap on a Relational Task — Improved Training

Re-runs Phase 27 with fixes suggested by external review:
  1. Gradient accumulation (effective batch=32) instead of sample-by-sample updates
  2. More data (1000 samples instead of 500 → doubles samples per class)
  3. ReduceLROnPlateau scheduler

Key insight: DELTA models can't batch in the traditional sense — each sample
builds a unique graph topology via the GraphConstructor, so forward passes are
inherently per-sample. We use gradient accumulation to get clean gradients
without requiring parallel graph processing.

Goal: determine whether Phase 27's result (Transformer >> DELTA) was caused
by suboptimal training (noisy batch-1 gradients, too few samples) or by a
genuine structural limitation (graph construction loses positional ordering).

If the ranking reverses with proper training → Phase 27 was training-confounded.
If DELTA still underperforms → the structural critique is confirmed.
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
#  Task generation (same as Phase 27, but more samples by default)
# ──────────────────────────────────────────────────────────────────────

def generate_path_composition_task(
    num_samples=5000,
    num_entity_types=8,
    num_relations=4,
    path_length=5,
    total_length=12,
    seed=42,
):
    """2-hop path composition with distractors.

    Path tokens: [ent, rel1, ent, rel2, ent] placed at a random start
    position within a total_length-token sequence. Remaining positions
    filled with noise from the full vocab range.

    Label = index of (rel1, r2) ordered pair.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    E_BASE = 1
    R_BASE = E_BASE + num_entity_types
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
#  Models (identical architectures to Phase 27)
# ──────────────────────────────────────────────────────────────────────

class TransformerBaseline(nn.Module):
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


class BootstrappedDELTA(nn.Module):
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
        # Process one sample at a time through DELTA (graph construction
        # is per-sample since each input builds a different graph topology)
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        batch_logits = []
        for i in range(token_ids.shape[0]):
            graph = self.delta(token_ids[i:i+1], use_router=False, use_memory=False)
            pooled = graph.node_features.mean(dim=0, keepdim=True)
            batch_logits.append(self.classifier(pooled))
        return torch.cat(batch_logits, dim=0)


class FixedChainDELTA(nn.Module):
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
        batch_logits = []
        for i in range(token_ids.shape[0]):
            x = self.embedding(token_ids[i:i+1])
            x = self.pos_enc(x)
            for layer in self.tf_layers:
                x, _ = layer(x)
            x = x.squeeze(0)

            node_features = self.to_node(x)
            S = node_features.shape[0]
            device = x.device

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
            batch_logits.append(self.classifier(pooled))
        return torch.cat(batch_logits, dim=0)


# ──────────────────────────────────────────────────────────────────────
#  Training loop — gradient accumulation for DELTA, real batching for Transformer
# ──────────────────────────────────────────────────────────────────────

def train_and_evaluate(name, model, train_data, train_labels,
                       test_data, test_labels, epochs=100, lr=1e-3,
                       accum_steps=32, is_transformer=False):
    """Train model and return (final_acc, elapsed_seconds).

    For transformer: real mini-batch training (fast, fully parallelizable).
    For DELTA models: gradient accumulation over accum_steps samples
    (each sample builds a unique graph, can't be batched).
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=False,
    )

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_samples = 0

        if is_transformer:
            # Real batching for transformer (processes full batch in one forward)
            perm = torch.randperm(len(train_data))
            for start in range(0, len(train_data), accum_steps):
                idx = perm[start:start + accum_steps]
                batch_x = train_data[idx].to(device)
                batch_y = train_labels[idx].to(device)
                logits = model(batch_x)
                loss = F.cross_entropy(logits, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)
                n_samples += len(idx)
        else:
            # Gradient accumulation for DELTA (per-sample forward, accumulated updates)
            perm = torch.randperm(len(train_data))
            optimizer.zero_grad()
            accum_loss = 0.0
            for i, idx in enumerate(perm):
                logits = model(train_data[idx].to(device))
                loss = F.cross_entropy(logits, train_labels[idx:idx + 1].to(device))
                (loss / accum_steps).backward()  # Scale loss for accumulation
                accum_loss += loss.item()
                n_samples += 1

                if (i + 1) % accum_steps == 0 or (i + 1) == len(train_data):
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += accum_loss
                    accum_loss = 0.0

        avg_loss = total_loss / n_samples
        scheduler.step(avg_loss)

        if (epoch + 1) % 25 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i in range(len(test_data)):
                    logits = model(test_data[i].to(device))
                    if logits.argmax(-1).item() == test_labels[i].item():
                        correct += 1
                    total += 1
            acc = correct / total
            cur_lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - t0
            print(f"  [{name}] Epoch {epoch + 1}: "
                  f"Loss={avg_loss:.4f}  "
                  f"Test Acc={acc:.1%}  LR={cur_lr:.1e}  ({elapsed:.1f}s)")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(test_data)):
            logits = model(test_data[i].to(device))
            if logits.argmax(-1).item() == test_labels[i].item():
                correct += 1
            total += 1
    final_acc = correct / total
    total_time = time.time() - t0
    return final_acc, total_time


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 27b: Bootstrap Relational Task — Improved Training")
    print("=" * 70)
    print()
    print("Re-running Phase 27 with training fixes from external review:")
    print("  1. Gradient accumulation (effective batch=32) vs batch=1")
    print("  2. 2x more data (1000 vs 500 samples)")
    print("  3. ReduceLROnPlateau scheduler")
    print()
    print("Note: DELTA models process each sample individually (each builds")
    print("  a unique graph via GraphConstructor). We use gradient accumulation")
    print("  to get clean gradients without requiring parallel graph processing.")
    print()
    print("Question: Was Phase 27's result (Transformer >> DELTA) caused by")
    print("  suboptimal training or by a genuine structural limitation?")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print()

    # ── Data ──────────────────────────────────────────────────────────
    data, labels, vocab_size, num_classes = generate_path_composition_task(
        num_samples=1000, num_entity_types=8, num_relations=4,
        path_length=5, total_length=12, seed=42,
    )
    print(f"Vocab size: {vocab_size}  (8 entity types + 4 relations + pad)")
    print(f"Composition classes: {num_classes}")
    print(f"Samples: {len(data)}   Sequence length: {data.shape[1]}")

    perm = torch.randperm(len(data), generator=torch.Generator().manual_seed(42))
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
    accum_steps = 32
    epochs = 100

    # Phase 27 original results (N=500, batch=1, no scheduler):
    #   Transformer: 32.7%, Bootstrap DELTA: 8.0%, Fixed Chain: 5.3%
    results_orig = {
        'Transformer': (0.327, 0),
        'Bootstrapped DELTA': (0.080, 0),
        'Fixed Chain DELTA': (0.053, 0),
    }
    print("Phase 27 original results (N=500, batch=1, no scheduler):")
    for name, (acc, _) in results_orig.items():
        print(f"  {name:25s}  {acc:.1%}")
    print()

    # ── Improved settings ─────────────────────────────────────────────
    print("=" * 70)
    print(f"Improved settings (N=1000, accum={accum_steps}, epochs={epochs}, LR scheduler)")
    print("=" * 70)
    print()

    results_fixed = {}

    # ── Model 1: Transformer ──────────────────────────────────────────
    print("-" * 60)
    print("Transformer (improved settings)")
    print("-" * 60)
    tf_fixed = TransformerBaseline(vocab_size, d_model, num_classes).to(device)
    print(f"  Parameters: {sum(p.numel() for p in tf_fixed.parameters()):,}")
    acc, t = train_and_evaluate(
        "TF", tf_fixed, train_data, train_labels,
        test_data, test_labels, epochs=epochs, accum_steps=accum_steps,
        is_transformer=True,
    )
    results_fixed['Transformer'] = (acc, t)
    print(f"  >> Final: {acc:.1%}  ({t:.1f}s)\n")

    # ── Model 2: Bootstrapped DELTA ───────────────────────────────────
    print("-" * 60)
    print("Bootstrapped DELTA (improved settings)")
    print("-" * 60)
    boot_fixed = BootstrappedDELTA(
        vocab_size, d_model, d_node, d_edge, num_classes,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in boot_fixed.parameters()):,}")
    acc, t = train_and_evaluate(
        "Boot", boot_fixed, train_data, train_labels,
        test_data, test_labels, epochs=epochs, accum_steps=accum_steps,
        is_transformer=False,
    )
    results_fixed['Bootstrapped DELTA'] = (acc, t)
    print(f"  >> Final: {acc:.1%}  ({t:.1f}s)\n")

    # ── Model 3: Fixed Chain DELTA ────────────────────────────────────
    print("-" * 60)
    print("Fixed Chain DELTA (improved settings)")
    print("-" * 60)
    chain_fixed = FixedChainDELTA(
        vocab_size, d_model, d_node, d_edge, num_classes,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in chain_fixed.parameters()):,}")
    acc, t = train_and_evaluate(
        "Chain", chain_fixed, train_data, train_labels,
        test_data, test_labels, epochs=epochs, accum_steps=accum_steps,
        is_transformer=False,
    )
    results_fixed['Fixed Chain DELTA'] = (acc, t)
    print(f"  >> Final: {acc:.1%}  ({t:.1f}s)\n")

    # ── Comparative summary ───────────────────────────────────────────
    print("=" * 70)
    print("COMPARATIVE SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Model':25s}  {'Ph27 (N=500,B=1)':>20s}  {'27b (N=1000,acc=32)':>22s}  {'Delta':>8s}")
    print("-" * 85)
    for name in ['Transformer', 'Bootstrapped DELTA', 'Fixed Chain DELTA']:
        a_orig = results_orig[name][0]
        a_fixed = results_fixed[name][0]
        delta = a_fixed - a_orig
        print(f"  {name:23s}  {a_orig:>18.1%}  {a_fixed:>20.1%}  {delta:>+7.1%}")
    print()
    print(f"  Random baseline: {1.0 / num_classes:.1%}")
    print()

    # ── Verdict ───────────────────────────────────────────────────────
    tf_orig_acc = results_orig['Transformer'][0]
    boot_orig_acc = results_orig['Bootstrapped DELTA'][0]
    tf_fixed_acc = results_fixed['Transformer'][0]
    boot_fixed_acc = results_fixed['Bootstrapped DELTA'][0]
    chain_fixed_acc = results_fixed['Fixed Chain DELTA'][0]

    print("Analysis:")
    print()

    orig_gap = tf_orig_acc - boot_orig_acc
    fixed_gap = tf_fixed_acc - boot_fixed_acc
    print(f"  Transformer vs Bootstrap gap:")
    print(f"    Original:  {orig_gap:+.1%}  (TF {tf_orig_acc:.1%} vs Boot {boot_orig_acc:.1%})")
    print(f"    Improved:  {fixed_gap:+.1%}  (TF {tf_fixed_acc:.1%} vs Boot {boot_fixed_acc:.1%})")
    print()

    if fixed_gap < orig_gap * 0.5:
        print("  VERDICT: Training fixes substantially narrowed the gap.")
        print("  Phase 27's result was partially training-confounded.")
    elif boot_fixed_acc > tf_fixed_acc:
        print("  VERDICT: Bootstrap DELTA now outperforms Transformer!")
        print("  Phase 27's result was entirely training-confounded.")
    else:
        print("  VERDICT: Gap persists even with proper training.")
        print("  Phase 27's structural critique (positional info loss) is CONFIRMED.")

    print()
    boot_vs_chain = boot_fixed_acc - chain_fixed_acc
    print(f"  Bootstrap vs Fixed Chain (improved settings): {boot_vs_chain:+.1%}")
    if boot_vs_chain > 0.05:
        print("  Attention-based construction adds genuine structural value.")
    elif boot_vs_chain > 0.0:
        print("  Marginal benefit from attention-based construction.")
    else:
        print("  Fixed chain topology sufficient — adaptive construction not helping.")


if __name__ == '__main__':
    main()
