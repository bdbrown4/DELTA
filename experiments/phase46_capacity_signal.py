"""Phase 46: Capacity Signal Measurement
========================================

Hypothesis (falsifiable):
  "DELTA-Full's attention heads show significantly higher redundancy than
  DELTA-Matched's — at least 25% of per-head attention distributions will be
  near-uniform (entropy > 90% of maximum), indicating excess capacity. DELTA-
  Matched will have <10% near-uniform heads. Additionally, attention routing
  patterns will measurably differ between 1p and 5p queries (cosine similarity
  of per-layer edge attention profiles < 0.85), confirming depth-dependent
  computation paths."

Measurements:
  1. Per-layer, per-head attention entropy (averaged over target nodes)
  2. Per-layer, per-head attention sparsity (Gini coefficient)
  3. Head utilization: fraction of "dead" heads (near-uniform attention)
  4. Cross-depth attention consistency (1p vs 3p vs 5p cosine similarity)
  5. Layer similarity (attention pattern correlation across layers)

Protocol:
  - Train delta_matched and delta_full normally (same LP pipeline as Phase 45)
  - Post-training, run instrumentation passes to collect per-layer attention
  - Evaluate on 1p/2p/3p/4p/5p with attention collection per query depth
  - Compare distributions between models and across depths

Regression safety:
  - Must reproduce Phase 45 LP/multi-hop baselines within 1 std

Usage:
  # Smoke test
  python experiments/phase46_capacity_signal.py --epochs 5

  # Full run
  python experiments/phase46_capacity_signal.py --epochs 500 --eval_every 25 --patience 10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from delta.graph import DeltaGraph
from delta.model import DELTAModel, DELTALayer
from delta.attention import DualParallelAttention

from experiments.phase46c_link_prediction import (
    load_lp_data,
    build_train_graph_tensors,
    create_lp_model,
    train_epoch,
    evaluate_lp,
    LinkPredictionModel,
)

from experiments.phase42_multihop import (
    generate_multihop_queries,
    audit_queries,
    build_full_adjacency,
    evaluate_multihop,
)

from experiments.phase44_depth import generate_extended_queries


# ═══════════════════════════════════════════════════════════════════════════
# Attention Instrumentation (forward hooks)
# ═══════════════════════════════════════════════════════════════════════════

class AttentionCollector:
    """Captures per-layer, per-head attention weights via forward hooks.

    Hooks on DualParallelAttention modules to intercept attention weights
    when return_weights=True (which is the default when use_router=True
    in DELTAModel).
    """

    def __init__(self):
        self.layer_data = {}   # layer_idx → dict of attention stats
        self.hooks = []
        self._active = False

    def instrument(self, model):
        """Register hooks on all DELTALayers in a LinkPredictionModel."""
        encoder = model.encoder
        if encoder is None:
            return

        if isinstance(encoder, DELTAModel):
            layers = list(encoder.layers)
        else:
            # SelfBootstrap or other — find DELTALayers recursively
            layers = [m for m in encoder.modules() if isinstance(m, DELTALayer)]

        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._make_hook(i))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input_args, output):
            if not self._active:
                return
            # output is a DeltaGraph; attention weights are computed inside
            # We need a different approach — hook on the dual_attn instead
            pass
        return hook_fn

    def instrument_attention(self, model):
        """Register hooks on DualParallelAttention to capture raw weights.

        This is the primary instrumentation — captures [E, H] attention
        weights from node attention and [E_adj, H] from edge attention.
        """
        encoder = model.encoder
        if encoder is None:
            return

        if isinstance(encoder, DELTAModel):
            layers = list(encoder.layers)
        else:
            layers = [m for m in encoder.modules() if isinstance(m, DELTALayer)]

        for i, layer in enumerate(layers):
            # Wrap the dual_attn forward to always capture weights
            self._wrap_dual_attn(layer, i)

    def _wrap_dual_attn(self, delta_layer, layer_idx):
        """Monkey-patch DualParallelAttention.forward to capture weights."""
        original_forward = delta_layer.dual_attn.forward

        def instrumented_forward(*args, **kwargs):
            # Force return_weights=True when collecting
            if self._active:
                kwargs['return_weights'] = True
            result = original_forward(*args, **kwargs)
            if self._active and isinstance(result, tuple) and len(result) == 3:
                graph, node_w, edge_w = result
                self.layer_data[layer_idx] = {
                    'node_attn': node_w.detach(),  # [E, H]
                    'edge_attn': edge_w.detach(),  # [E_adj, H]
                    'edge_index': graph.edge_index.detach(),  # [2, E]
                    'num_nodes': graph.num_nodes,
                }
            return result

        delta_layer.dual_attn.forward = instrumented_forward

    def start(self):
        self._active = True
        self.layer_data = {}

    def stop(self):
        self._active = False

    def get_stats(self):
        """Compute statistics from captured attention weights."""
        stats = {}
        for layer_idx, data in sorted(self.layer_data.items()):
            node_w = data['node_attn']  # [E, H]
            edge_w = data['edge_attn']  # [E_adj, H]
            edge_index = data.get('edge_index')  # [2, E]
            num_nodes = data.get('num_nodes', 0)

            stats[layer_idx] = {
                'node_attn': _compute_attn_stats(node_w, edge_index, num_nodes),
                'edge_attn': _compute_attn_stats(edge_w),
            }
        return stats

    def get_flat_attention_profile(self):
        """Return concatenated per-head mean attention as a flat vector.

        Used for cross-depth cosine similarity analysis.
        """
        profile = []
        for layer_idx in sorted(self.layer_data.keys()):
            data = self.layer_data[layer_idx]
            for key in ['node_attn', 'edge_attn']:
                w = data[key]  # [E, H]
                per_head_mean = w.mean(dim=0)  # [H]
                profile.append(per_head_mean.cpu())
        if profile:
            return torch.cat(profile)
        return torch.zeros(1)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def _compute_attn_stats(weights, edge_index=None, num_nodes=None):
    """Compute per-head statistics from attention weight tensor [E, H].

    If edge_index is provided, computes per-TARGET-NODE entropy (correct for
    GAT attention where softmax is per-target). Otherwise falls back to global.
    """
    E, H = weights.shape
    if E == 0:
        return {
            'per_head_entropy': [0.0] * H,
            'per_head_norm_entropy': [0.0] * H,
            'per_head_gini': [0.0] * H,
            'per_head_mean': [0.0] * H,
            'per_head_std': [0.0] * H,
            'dead_head_frac': 0.0,
        }

    stats = {
        'per_head_entropy': [],
        'per_head_norm_entropy': [],
        'per_head_gini': [],
        'per_head_mean': [],
        'per_head_std': [],
    }

    for h in range(H):
        w = weights[:, h]  # [E]

        if edge_index is not None and num_nodes is not None and num_nodes > 0:
            # Per-target-node entropy (correct for GAT softmax)
            targets = edge_index[1]
            entropies = []
            for node in range(num_nodes):
                mask = targets == node
                deg = mask.sum().item()
                if deg < 2:
                    continue
                w_node = w[mask]
                w_pos = w_node.clamp(min=1e-10)
                w_norm = w_pos / w_pos.sum()
                ent = float(-(w_norm * torch.log(w_norm + 1e-10)).sum())
                max_ent = float(np.log(deg))
                norm_ent = ent / max(max_ent, 1e-10)
                entropies.append(norm_ent)
            avg_norm_entropy = float(np.mean(entropies)) if entropies else 1.0
            stats['per_head_norm_entropy'].append(avg_norm_entropy)
            stats['per_head_entropy'].append(avg_norm_entropy)
        else:
            # Global fallback (for edge attention or when no topology)
            w_pos = w.clamp(min=1e-10)
            w_norm = w_pos / w_pos.sum()
            entropy = float(-(w_norm * torch.log(w_norm + 1e-10)).sum())
            max_entropy = float(np.log(max(E, 2)))
            norm_entropy = entropy / max(max_entropy, 1e-10)
            stats['per_head_entropy'].append(entropy)
            stats['per_head_norm_entropy'].append(norm_entropy)

        # Gini coefficient (sparsity measure: 0=equal, 1=maximally sparse)
        sorted_w = torch.sort(w)[0]
        n = len(sorted_w)
        index = torch.arange(1, n + 1, device=sorted_w.device, dtype=torch.float)
        gini = float((2 * (index * sorted_w).sum() / (n * sorted_w.sum() + 1e-10)
                       - (n + 1) / n))
        stats['per_head_gini'].append(max(0.0, gini))

        # Basic stats
        stats['per_head_mean'].append(float(w.mean()))
        stats['per_head_std'].append(float(w.std()))

    # Dead head: normalized entropy > 0.90 means near-uniform (unfocused)
    dead_count = sum(1 for e in stats['per_head_norm_entropy'] if e > 0.90)
    stats['dead_head_frac'] = dead_count / H if H > 0 else 0.0

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Gate Statistics (from PostAttentionPruner)
# ═══════════════════════════════════════════════════════════════════════════

class GateCollector:
    """Captures per-layer importance gates from DELTALayer output graphs."""

    def __init__(self):
        self.layer_gates = {}
        self.hooks = []
        self._active = False

    def instrument(self, model):
        encoder = model.encoder
        if encoder is None:
            return
        if isinstance(encoder, DELTAModel):
            layers = list(encoder.layers)
        else:
            layers = [m for m in encoder.modules() if isinstance(m, DELTALayer)]

        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._make_hook(i))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input_args, output):
            if not self._active:
                return
            if (hasattr(output, 'edge_importance') and
                    output.edge_importance is not None):
                self.layer_gates[layer_idx] = {
                    'node': output.node_importance.detach(),
                    'edge': output.edge_importance.detach(),
                }
        return hook_fn

    def start(self):
        self._active = True
        self.layer_gates = {}

    def stop(self):
        self._active = False

    def get_stats(self):
        """Compute gate statistics."""
        stats = {}
        for layer_idx, data in sorted(self.layer_gates.items()):
            for gate_type in ['node', 'edge']:
                gates = data[gate_type]
                key = f'L{layer_idx}_{gate_type}'
                stats[key] = {
                    'mean': float(gates.mean()),
                    'std': float(gates.std()),
                    'frac_below_0.1': float((gates < 0.1).float().mean()),
                    'frac_below_0.01': float((gates < 0.01).float().mean()),
                    'entropy': float(_gate_entropy(gates)),
                    'gini': float(_gate_gini(gates)),
                }
        return stats

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def _gate_entropy(gates):
    """Entropy of gate distribution (binned into 20 bins)."""
    hist = torch.histc(gates.float(), bins=20, min=0.0, max=1.0)
    hist = hist / (hist.sum() + 1e-10)
    return float(-(hist * torch.log(hist + 1e-10)).sum())


def _gate_gini(gates):
    """Gini coefficient of gate values."""
    sorted_g = torch.sort(gates.flatten())[0]
    n = len(sorted_g)
    if n == 0:
        return 0.0
    index = torch.arange(1, n + 1, device=sorted_g.device, dtype=torch.float)
    return float((2 * (index * sorted_g).sum() / (n * sorted_g.sum() + 1e-10)
                   - (n + 1) / n).clamp(min=0))


# ═══════════════════════════════════════════════════════════════════════════
# Training with capacity instrumentation
# ═══════════════════════════════════════════════════════════════════════════

def train_instrumented(model_type, data, epochs, lr, device, batch_size, seed,
                       eval_every=25, patience=10):
    """Train a model and collect attention stats at each eval checkpoint.

    Returns (model, edge_index, edge_types, checkpoint_stats, best_val_mrr).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = create_lp_model(model_type,
                            data['num_entities'], data['num_relations'])
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = (sum(p.numel() for p in model.encoder.parameters())
             if model.encoder is not None else 0)
    print(f"\n  [{model_type}] seed={seed}, {n_params:,} params "
          f"({n_enc:,} encoder), device={device}")

    # Set up instrumentation
    attn_collector = AttentionCollector()
    gate_collector = GateCollector()
    is_delta = model.encoder is not None and isinstance(model.encoder, DELTAModel)
    if is_delta:
        attn_collector.instrument_attention(model)
        gate_collector.instrument(model)

    edge_index, edge_types = build_train_graph_tensors(data['train'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_mrr = 0.0
    best_state = None
    evals_no_improve = 0
    checkpoint_stats = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data['train'], edge_index, edge_types,
                           optimizer, device, batch_size)

        if epoch % eval_every == 0 or epoch == epochs:
            val = evaluate_lp(model, data['val'], edge_index, edge_types,
                              data['hr_to_tails'], data['rt_to_heads'], device)
            elapsed = time.time() - t0
            print(f"    Ep {epoch:4d}  loss={loss:.4f}  "
                  f"val_MRR={val['MRR']:.4f}  val_H@10={val['Hits@10']:.4f}  "
                  f"[{elapsed:.0f}s]")

            # Collect attention/gate stats at this checkpoint
            if is_delta:
                model.eval()
                attn_collector.start()
                gate_collector.start()
                with torch.no_grad():
                    ei = edge_index.to(device)
                    et = edge_types.to(device)
                    model.encode(ei, et)
                attn_collector.stop()
                gate_collector.stop()

                cp = {
                    'epoch': epoch,
                    'val_MRR': val['MRR'],
                    'attention': attn_collector.get_stats(),
                    'gates': gate_collector.get_stats(),
                }
                checkpoint_stats.append(cp)

            if val['MRR'] > best_val_mrr:
                best_val_mrr = val['MRR']
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}
                evals_no_improve = 0
            else:
                evals_no_improve += 1
                if patience > 0 and evals_no_improve >= patience:
                    print(f"    Early stopping at epoch {epoch}")
                    break

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"    Training done: best_val_MRR={best_val_mrr:.4f} [{elapsed:.0f}s]")

    # Final instrumentation pass with best model
    if is_delta:
        model.eval()
        attn_collector.start()
        gate_collector.start()
        with torch.no_grad():
            ei = edge_index.to(device)
            et = edge_types.to(device)
            model.encode(ei, et)
        attn_collector.stop()
        gate_collector.stop()

    return (model, edge_index, edge_types, checkpoint_stats,
            best_val_mrr, attn_collector, gate_collector)


# ═══════════════════════════════════════════════════════════════════════════
# Cross-depth attention analysis
# ═══════════════════════════════════════════════════════════════════════════

def cross_depth_analysis(model, data, attn_collector, device,
                         edge_index, edge_types):
    """Evaluate on 1p-5p and collect attention profiles per depth.

    Returns dict: depth → {metrics, attention_profile}
    """
    print("\n  Cross-depth attention analysis...")
    is_delta = (model.encoder is not None and
                isinstance(model.encoder, DELTAModel))

    # Generate extended queries
    queries = generate_extended_queries(data, max_queries_per_type=500, seed=42)

    ei = edge_index.to(device)
    et = edge_types.to(device)

    # Build full adjacency for multi-hop filtering
    full_hr2t = data['hr_to_tails']

    # For each depth, encode with instrumentation and evaluate
    depth_results = {}
    depth_profiles = {}

    for depth_label in ['1p', '2p', '3p', '4p', '5p']:
        qs = queries.get(depth_label, [])
        if not qs:
            print(f"    {depth_label}: no queries available")
            continue

        # Cap queries for efficiency
        qs = qs[:500]

        # Collect attention profile during encoding
        model.eval()
        if is_delta:
            attn_collector.start()

        with torch.no_grad():
            node_feats = model.encode(ei, et)

        if is_delta:
            attn_collector.stop()
            depth_profiles[depth_label] = attn_collector.get_flat_attention_profile()

        # Evaluate multi-hop at this depth
        num_hops = int(depth_label[0])
        all_ranks = []

        for q in qs:
            anchor, rel_chain, answer = q[0], q[1], q[2]
            current_emb = node_feats[anchor].unsqueeze(0)  # [1, d]

            for hop in range(num_hops):
                rel = torch.tensor([rel_chain[hop]], device=device)
                hr = current_emb * model.decoder_rel_emb(rel)
                scores = hr @ node_feats.t()  # [1, N]

                if hop < num_hops - 1:
                    weights = torch.softmax(scores, dim=-1)
                    current_emb = weights @ node_feats
                else:
                    # Final hop — rank
                    rank = int((scores[0] >= scores[0, answer]).sum().item())
                    all_ranks.append(max(rank, 1))

        if all_ranks:
            ranks = np.array(all_ranks, dtype=np.float64)
            depth_results[depth_label] = {
                'MRR': float(np.mean(1.0 / ranks)),
                'Hits@10': float(np.mean(ranks <= 10)),
                'count': len(all_ranks),
            }
            print(f"    {depth_label}: MRR={depth_results[depth_label]['MRR']:.4f}  "
                  f"H@10={depth_results[depth_label]['Hits@10']:.4f}  "
                  f"(n={depth_results[depth_label]['count']})")

    # Compute cross-depth cosine similarities
    cos_sims = {}
    depth_keys = sorted(depth_profiles.keys())
    for i, d1 in enumerate(depth_keys):
        for d2 in depth_keys[i + 1:]:
            p1 = depth_profiles[d1].float()
            p2 = depth_profiles[d2].float()
            cos = float(F.cosine_similarity(p1.unsqueeze(0),
                                            p2.unsqueeze(0)).item())
            cos_sims[f'{d1}_vs_{d2}'] = cos

    return {
        'depth_metrics': depth_results,
        'depth_profiles': {k: v.tolist() for k, v in depth_profiles.items()},
        'cross_depth_cosine': cos_sims,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Analysis and reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_attention_report(model_type, attn_stats, gate_stats):
    """Print formatted attention analysis for one model."""
    print(f"\n{'='*70}")
    print(f"  CAPACITY SIGNAL: {model_type}")
    print(f"{'='*70}")

    if not attn_stats:
        print("  (No attention data — non-DELTA model)")
        return

    # Per-layer attention analysis
    for layer_idx in sorted(attn_stats.keys()):
        layer = attn_stats[layer_idx]
        print(f"\n  Layer {layer_idx}:")

        for attn_type in ['node_attn', 'edge_attn']:
            s = layer[attn_type]
            H = len(s['per_head_norm_entropy'])
            norm_ent = s['per_head_norm_entropy']

            print(f"    {attn_type} ({H} heads):")
            print(f"      Entropy (norm):  {' '.join(f'{e:.3f}' for e in norm_ent)}")
            print(f"      Gini:            {' '.join(f'{g:.3f}' for g in s['per_head_gini'])}")
            print(f"      Mean weight:     {' '.join(f'{m:.4f}' for m in s['per_head_mean'])}")
            print(f"      Dead heads (>90% norm entropy): "
                  f"{s['dead_head_frac']*100:.0f}%")

    # Gate analysis
    if gate_stats:
        print(f"\n  Gate statistics:")
        for key in sorted(gate_stats.keys()):
            gs = gate_stats[key]
            print(f"    {key}: mean={gs['mean']:.3f}  std={gs['std']:.3f}  "
                  f"frac<0.1={gs['frac_below_0.1']:.3f}  "
                  f"frac<0.01={gs['frac_below_0.01']:.3f}  "
                  f"entropy={gs['entropy']:.3f}  gini={gs['gini']:.3f}")


def print_cross_depth_report(cross_depth_data):
    """Print cross-depth attention analysis."""
    print(f"\n{'='*70}")
    print(f"  CROSS-DEPTH ROUTING ANALYSIS")
    print(f"{'='*70}")

    metrics = cross_depth_data.get('depth_metrics', {})
    if metrics:
        print("\n  Multi-hop MRR by depth:")
        for d in ['1p', '2p', '3p', '4p', '5p']:
            if d in metrics:
                m = metrics[d]
                print(f"    {d}: MRR={m['MRR']:.4f}  H@10={m['Hits@10']:.4f}  (n={m['count']})")

    cos = cross_depth_data.get('cross_depth_cosine', {})
    if cos:
        print("\n  Attention profile cosine similarity across depths:")
        for pair, sim in sorted(cos.items()):
            print(f"    {pair}: {sim:.4f}")


def print_training_dynamics(checkpoint_stats, model_type):
    """Show how attention statistics evolve during training."""
    if not checkpoint_stats:
        return

    print(f"\n{'='*70}")
    print(f"  TRAINING DYNAMICS: {model_type}")
    print(f"{'='*70}")
    print(f"\n  {'Epoch':>6}  {'val_MRR':>8}  ", end='')

    # Header: per-layer dead head fraction
    first = checkpoint_stats[0]
    layers = sorted(first.get('attention', {}).keys())
    for li in layers:
        print(f"  L{li}_node_dead  L{li}_edge_dead", end='')
    print()
    print(f"  {'-'*60}")

    for cp in checkpoint_stats:
        ep = cp['epoch']
        vmrr = cp['val_MRR']
        print(f"  {ep:6d}  {vmrr:8.4f}  ", end='')
        for li in layers:
            attn = cp.get('attention', {}).get(li, {})
            nd = attn.get('node_attn', {}).get('dead_head_frac', 0)
            ed = attn.get('edge_attn', {}).get('dead_head_frac', 0)
            print(f"  {nd*100:11.0f}%  {ed*100:11.0f}%", end='')
        print()

    # Gate sparsity evolution
    has_gates = any(cp.get('gates') for cp in checkpoint_stats)
    if has_gates:
        print(f"\n  Gate sparsity (frac < 0.1) over training:")
        gate_keys = sorted(checkpoint_stats[0].get('gates', {}).keys())
        print(f"  {'Epoch':>6}  " + "  ".join(f"{k:>16}" for k in gate_keys))
        for cp in checkpoint_stats:
            ep = cp['epoch']
            print(f"  {ep:6d}", end='')
            for gk in gate_keys:
                frac = cp.get('gates', {}).get(gk, {}).get('frac_below_0.1', 0)
                print(f"  {frac:16.3f}", end='')
            print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Phase 46: Capacity Signal Measurement')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs (default: 5 smoke test)')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_every', type=int, default=None,
                        help='Evaluate every N epochs (default: auto)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_entities', type=int, default=500,
                        help='Entity limit (use 0 for full dataset, default=500 matches Phase 45)')
    args = parser.parse_args()

    if args.eval_every is None:
        args.eval_every = max(1, args.epochs // 10)

    max_ent = None if args.max_entities == 0 else args.max_entities
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("PHASE 46: CAPACITY SIGNAL MEASUREMENT")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, eval_every: {args.eval_every}")
    print(f"  Seed: {args.seed}")
    print(f"  Max entities: {max_ent or 'full dataset'}")

    # Load data
    print("\n  Loading data...")
    data = load_lp_data('fb15k-237', max_entities=max_ent)

    # Models to compare
    models_to_run = ['delta_matched', 'delta_full']
    all_results = {}

    for model_type in models_to_run:
        print(f"\n{'='*70}")
        print(f"  TRAINING: {model_type}")
        print(f"{'='*70}")

        (model, edge_index, edge_types, checkpoint_stats,
         best_val_mrr, attn_collector, gate_collector) = train_instrumented(
            model_type, data, args.epochs, args.lr, device,
            args.batch_size, args.seed, args.eval_every, args.patience)

        # Regression check: standard LP
        lp_test = evaluate_lp(model, data['test'], edge_index, edge_types,
                              data['hr_to_tails'], data['rt_to_heads'], device)
        print(f"\n  Standard LP: test_MRR={lp_test['MRR']:.4f}  "
              f"test_H@10={lp_test['Hits@10']:.4f}")

        # Final attention stats
        attn_stats = attn_collector.get_stats()
        gate_stats = gate_collector.get_stats()

        print_attention_report(model_type, attn_stats, gate_stats)
        print_training_dynamics(checkpoint_stats, model_type)

        # Cross-depth analysis
        is_delta = (model.encoder is not None and
                    isinstance(model.encoder, DELTAModel))
        cross_depth = None
        if is_delta:
            cross_depth = cross_depth_analysis(
                model, data, attn_collector, device, edge_index, edge_types)
            print_cross_depth_report(cross_depth)

        all_results[model_type] = {
            'best_val_mrr': best_val_mrr,
            'lp_test': lp_test,
            'attention_stats': _serialize_stats(attn_stats),
            'gate_stats': gate_stats,
            'checkpoint_stats': _serialize_checkpoints(checkpoint_stats),
            'cross_depth': cross_depth,
        }

    # ═══════════════════════════════════════════════════════════════════
    # Comparative analysis
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  COMPARATIVE ANALYSIS")
    print(f"{'='*70}")

    dm = all_results.get('delta_matched', {})
    df = all_results.get('delta_full', {})

    if dm and df:
        dm_attn = dm.get('attention_stats', {})
        df_attn = df.get('attention_stats', {})

        # Count dead heads per model
        dm_dead = 0
        dm_total = 0
        for li in dm_attn:
            for at in ['node_attn', 'edge_attn']:
                heads = dm_attn[li].get(at, {})
                H = len(heads.get('per_head_norm_entropy', []))
                dead = int(heads.get('dead_head_frac', 0) * H)
                dm_dead += dead
                dm_total += H

        df_dead = 0
        df_total = 0
        for li in df_attn:
            for at in ['node_attn', 'edge_attn']:
                heads = df_attn[li].get(at, {})
                H = len(heads.get('per_head_norm_entropy', []))
                dead = int(heads.get('dead_head_frac', 0) * H)
                df_dead += dead
                df_total += H

        print(f"\n  Head utilization:")
        print(f"    DELTA-Matched: {dm_dead}/{dm_total} dead heads "
              f"({dm_dead/max(dm_total,1)*100:.0f}%)")
        print(f"    DELTA-Full:    {df_dead}/{df_total} dead heads "
              f"({df_dead/max(df_total,1)*100:.0f}%)")

        # Gate sparsity comparison
        dm_gates = dm.get('gate_stats', {})
        df_gates = df.get('gate_stats', {})

        if dm_gates and df_gates:
            dm_sparse = np.mean([v['frac_below_0.1'] for v in dm_gates.values()])
            df_sparse = np.mean([v['frac_below_0.1'] for v in df_gates.values()])
            print(f"\n  Edge gate sparsity (mean frac < 0.1):")
            print(f"    DELTA-Matched: {dm_sparse:.3f}")
            print(f"    DELTA-Full:    {df_sparse:.3f}")

        # Cross-depth consistency
        dm_cos = dm.get('cross_depth', {}).get('cross_depth_cosine', {})
        df_cos = df.get('cross_depth', {}).get('cross_depth_cosine', {})

        if dm_cos or df_cos:
            print(f"\n  Cross-depth attention consistency (cosine sim):")
            all_pairs = sorted(set(list(dm_cos.keys()) + list(df_cos.keys())))
            for pair in all_pairs:
                dm_v = dm_cos.get(pair, None)
                df_v = df_cos.get(pair, None)
                dm_str = f"{dm_v:.4f}" if dm_v is not None else "  N/A "
                df_str = f"{df_v:.4f}" if df_v is not None else "  N/A "
                print(f"    {pair}: Matched={dm_str}  Full={df_str}")

    # ═══════════════════════════════════════════════════════════════════
    # Hypothesis evaluation
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  HYPOTHESIS EVALUATION")
    print(f"{'='*70}")

    if dm and df:
        # H1: DELTA-Full has >25% dead heads, DELTA-Matched has <10%
        df_dead_pct = df_dead / max(df_total, 1) * 100
        dm_dead_pct = dm_dead / max(dm_total, 1) * 100
        h1_full = df_dead_pct > 25
        h1_matched = dm_dead_pct < 10
        h1 = h1_full and h1_matched
        print(f"\n  H1 (excess capacity → dead heads):")
        print(f"    DELTA-Full dead heads: {df_dead_pct:.1f}% (need >25%): "
              f"{'CONFIRMED' if h1_full else 'REJECTED'}")
        print(f"    DELTA-Matched dead heads: {dm_dead_pct:.1f}% (need <10%): "
              f"{'CONFIRMED' if h1_matched else 'REJECTED'}")
        print(f"    → H1 overall: {'CONFIRMED' if h1 else 'REJECTED'}")

        # H2: Cross-depth cosine < 0.85 for 1p vs 5p
        cos_1p_5p = dm_cos.get('1p_vs_5p', df_cos.get('1p_vs_5p', None))
        if cos_1p_5p is not None:
            h2 = cos_1p_5p < 0.85
            print(f"\n  H2 (depth-dependent routing):")
            print(f"    1p vs 5p cosine similarity: {cos_1p_5p:.4f} (need <0.85): "
                  f"{'CONFIRMED' if h2 else 'REJECTED'}")
            if cos_1p_5p > 0.99:
                print(f"    NOTE: Cosine ≈ 1.0 is expected — graph encoding is")
                print(f"    query-independent. The attention pattern is a property")
                print(f"    of the graph structure, not the query depth.")
                print(f"    → Compositional advantage lives in the REPRESENTATION,")
                print(f"      not in depth-dependent routing.")
        else:
            print(f"\n  H2 (depth-dependent routing): INSUFFICIENT DATA")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), '..', 'phase46_output.json')
    output_path = os.path.abspath(output_path)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    print(f"\n{'='*70}")
    print(f"  PHASE 46 COMPLETE")
    print(f"{'='*70}")


def _serialize_stats(attn_stats):
    """Convert attention stats to JSON-serializable format."""
    result = {}
    for layer_idx, data in attn_stats.items():
        result[layer_idx] = {}
        for attn_type, stats in data.items():
            result[layer_idx][attn_type] = {
                k: (v if isinstance(v, (int, float, str, bool))
                     else [round(x, 6) for x in v] if isinstance(v, list) else v)
                for k, v in stats.items()
            }
    return result


def _serialize_checkpoints(checkpoint_stats):
    """Convert checkpoint stats to JSON-serializable format."""
    result = []
    for cp in checkpoint_stats:
        ser = {'epoch': cp['epoch'], 'val_MRR': cp['val_MRR']}
        if 'attention' in cp:
            ser['attention'] = _serialize_stats(cp['attention'])
        if 'gates' in cp:
            ser['gates'] = cp['gates']
        result.append(ser)
    return result


if __name__ == '__main__':
    main()
