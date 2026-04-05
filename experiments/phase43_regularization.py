"""Phase 43: GNN Regularization for Multi-hop Compositional Reasoning
=====================================================================

Phase 42 showed DELTA-Matched is the only model that improves 2p→3p.
But all GNN models peak ~epoch 200 then degrade (standard overfitting
on 494-node graph — not graph drift; the topology is fixed).

This phase tests whether **DropEdge** regularization can:
  1. Push peak val_MRR higher
  2. Reduce the val-test gap (tighter generalization)
  3. Differentially benefit multi-hop (2p/3p) vs standard LP (1p)

DropEdge: randomly mask a fraction of edges during each training forward
pass. The GNN encoder sees a random subgraph of the training graph per
batch, forcing it to learn patterns that generalize across substructures
rather than memorizing specific neighborhoods.

Key design decisions:
  - DropEdge applies ONLY during training — evaluation uses full graph
  - Edge types are dropped in sync with edge_index
  - Training triples (positive examples) are independent of edge masking:
    DropEdge affects what the GNN sees, not what it trains on
  - Standard LP loss throughout — only the GNN's input changes
  - Multi-hop evaluation identical to Phase 42 (no leakage)

Models: delta_matched (3p champion) + graphgps (1p champion)
  - These are the two most interesting: does DropEdge differentially help
    edge-to-edge (DELTA) vs MPNN+global attention (GraphGPS)?

Drop rates: 0.0, 0.1, 0.2, 0.3, 0.4
  - 0.0 = baseline (should reproduce Phase 42)
  - Higher rates = more aggressive regularization

Usage:
  # Smoke test
  python experiments/phase43_regularization.py --epochs 5 --models delta_matched --drop_rates 0.0,0.1

  # Full run
  python experiments/phase43_regularization.py --epochs 500 --eval_every 25 --patience 10

  # With multi-hop evaluation
  python experiments/phase43_regularization.py --epochs 500 --eval_every 25 --patience 10 --multihop
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F

from experiments.phase46c_link_prediction import (
    load_lp_data,
    build_train_graph_tensors,
    create_lp_model,
    evaluate_lp,
)

from experiments.phase42_multihop import (
    generate_multihop_queries,
    audit_queries,
    build_full_adjacency,
    evaluate_multihop,
)


# ═══════════════════════════════════════════════════════════════════════════
# DropEdge — training-only edge masking
# ═══════════════════════════════════════════════════════════════════════════

def drop_edges(edge_index, edge_types, drop_rate):
    """Randomly drop a fraction of edges. Returns masked (edge_index, edge_types).

    Args:
        edge_index: [2, E] tensor — head/tail indices
        edge_types: [E] tensor — relation IDs
        drop_rate: fraction of edges to drop (0.0 = keep all)

    Returns:
        (masked_edge_index, masked_edge_types) — both reduced in size
    """
    if drop_rate <= 0.0:
        return edge_index, edge_types

    E = edge_index.shape[1]
    # Boolean mask: True = keep
    keep_mask = torch.rand(E, device=edge_index.device) >= drop_rate
    # Guarantee at least 1 edge survives
    if not keep_mask.any():
        keep_mask[torch.randint(E, (1,))] = True

    return edge_index[:, keep_mask], edge_types[keep_mask]


# ═══════════════════════════════════════════════════════════════════════════
# Training — DropEdge-augmented LP training
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch_dropedge(model, train_triples, edge_index, edge_types,
                         optimizer, device, batch_size=512,
                         label_smoothing=0.1, drop_rate=0.0):
    """One training epoch with DropEdge.

    Identical to phase46c train_epoch except:
      - Before each batch, randomly drop edges from the message-passing graph
      - The training triples (positives) are NOT dropped — only GNN input changes
    """
    model.train()
    n = train_triples.shape[1]
    perm = torch.randperm(n)
    total_loss = 0.0
    num_batches = 0

    ei = edge_index.to(device)
    et = edge_types.to(device)

    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        h = train_triples[0, idx].to(device)
        r = train_triples[1, idx].to(device)
        t = train_triples[2, idx].to(device)
        B = h.shape[0]
        N = model.num_entities

        # DropEdge: mask the GNN's message-passing graph (not training labels)
        ei_drop, et_drop = drop_edges(ei, et, drop_rate)

        # Encode with masked graph
        node_feats = model.encode(ei_drop, et_drop)

        # --- Tail prediction ---
        scores_t = model.score_all_tails(node_feats, h, r)
        targets_t = torch.zeros(B, N, device=device)
        targets_t[torch.arange(B, device=device), t] = 1.0
        if label_smoothing > 0:
            targets_t = targets_t * (1 - label_smoothing) + label_smoothing / N
        loss_t = F.binary_cross_entropy_with_logits(scores_t, targets_t)

        # --- Head prediction ---
        scores_h = model.score_all_heads(node_feats, r, t)
        targets_h = torch.zeros(B, N, device=device)
        targets_h[torch.arange(B, device=device), h] = 1.0
        if label_smoothing > 0:
            targets_h = targets_h * (1 - label_smoothing) + label_smoothing / N
        loss_h = F.binary_cross_entropy_with_logits(scores_h, targets_h)

        loss = (loss_t + loss_h) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Training loop with checkpointing
# ═══════════════════════════════════════════════════════════════════════════

def train_with_dropedge(model_type, data, epochs, lr, device, batch_size,
                        seed, eval_every, patience, drop_rate):
    """Train a model with DropEdge and return best-val checkpoint.

    Returns (model, best_val_mrr, edge_index, edge_types, train_time,
             peak_epoch, history)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = create_lp_model(model_type,
                            data['num_entities'], data['num_relations'])
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = (sum(p.numel() for p in model.encoder.parameters())
             if model.encoder is not None else 0)
    print(f"\n  [{model_type}] drop={drop_rate}, seed={seed}, "
          f"{n_params:,} params ({n_enc:,} encoder)")

    edge_index, edge_types = build_train_graph_tensors(data['train'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_mrr = 0.0
    best_state = None
    peak_epoch = 0
    evals_no_improve = 0
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        loss = train_epoch_dropedge(
            model, data['train'], edge_index, edge_types,
            optimizer, device, batch_size,
            drop_rate=drop_rate)

        if epoch % eval_every == 0 or epoch == epochs:
            # Evaluate on FULL graph (no dropping)
            val = evaluate_lp(model, data['val'], edge_index, edge_types,
                              data['hr_to_tails'], data['rt_to_heads'], device)
            elapsed = time.time() - t0
            print(f"    Ep {epoch:4d}  loss={loss:.4f}  "
                  f"val_MRR={val['MRR']:.4f}  val_H@10={val['Hits@10']:.4f}  "
                  f"[{elapsed:.0f}s]")

            history.append({
                'epoch': epoch, 'loss': loss,
                'val_MRR': val['MRR'], 'val_H@10': val['Hits@10'],
            })

            if val['MRR'] > best_val_mrr:
                best_val_mrr = val['MRR']
                peak_epoch = epoch
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

    print(f"    Done: best_val_MRR={best_val_mrr:.4f} "
          f"(peak ep {peak_epoch}) [{elapsed:.0f}s]")

    return model, best_val_mrr, edge_index, edge_types, elapsed, peak_epoch, history


# ═══════════════════════════════════════════════════════════════════════════
# Single configuration run
# ═══════════════════════════════════════════════════════════════════════════

def run_single(model_type, data, drop_rate, args, device,
               queries=None, full_hr2t=None):
    """Train with DropEdge, evaluate LP + optional multi-hop.

    Returns a dict with all metrics.
    """
    model, best_val, edge_index, edge_types, train_time, peak_epoch, history = \
        train_with_dropedge(
            model_type, data, args.epochs, args.lr, device, args.batch_size,
            args.seed, args.eval_every, args.patience, drop_rate)

    # Standard LP on test set
    lp_test = evaluate_lp(model, data['test'], edge_index, edge_types,
                          data['hr_to_tails'], data['rt_to_heads'], device)
    print(f"    LP test: MRR={lp_test['MRR']:.4f}  "
          f"H@1={lp_test['Hits@1']:.4f}  "
          f"H@3={lp_test['Hits@3']:.4f}  "
          f"H@10={lp_test['Hits@10']:.4f}")

    result = {
        'model': model_type,
        'drop_rate': drop_rate,
        'seed': args.seed,
        'params': sum(p.numel() for p in model.parameters()),
        'best_val_MRR': best_val,
        'peak_epoch': peak_epoch,
        'train_time': train_time,
        'lp_test_MRR': lp_test['MRR'],
        'lp_test_H@1': lp_test['Hits@1'],
        'lp_test_H@3': lp_test['Hits@3'],
        'lp_test_H@10': lp_test['Hits@10'],
    }

    # Multi-hop evaluation (optional — slow for DELTA models)
    if queries is not None and full_hr2t is not None:
        print(f"    Evaluating multi-hop queries...")
        t0 = time.time()
        mh = evaluate_multihop(model, queries, edge_index, edge_types,
                               full_hr2t, device, temperature=args.temperature)
        eval_time = time.time() - t0

        for qt in ['1p', '2p', '3p']:
            r = mh[qt]
            if r['count'] > 0:
                print(f"    {qt}: MRR={r['MRR']:.4f}  H@1={r['Hits@1']:.4f}  "
                      f"H@3={r['Hits@3']:.4f}  H@10={r['Hits@10']:.4f}  "
                      f"(n={r['count']})")
        print(f"    Multi-hop eval: {eval_time:.1f}s")

        result['eval_time'] = eval_time
        for qt in ['1p', '2p', '3p']:
            for k, v in mh[qt].items():
                result[f'{qt}_{k}'] = v

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Summary output
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(results, has_multihop):
    """Print results table grouped by model, sorted by drop_rate."""
    print("\n" + "=" * 100)
    print("PHASE 43: DROPEDGE REGULARIZATION RESULTS")
    print("=" * 100)

    models = []
    seen = set()
    for r in results:
        if r['model'] not in seen:
            models.append(r['model'])
            seen.add(r['model'])

    # Standard LP table
    print(f"\n{'─' * 100}")
    print(f"  Standard LP (test set)")
    print(f"{'─' * 100}")
    print(f"  {'Model':<22s} {'Drop':>6s} {'Peak Ep':>8s} "
          f"{'val_MRR':>10s} {'test_MRR':>10s} {'test_H@1':>10s} "
          f"{'test_H@3':>10s} {'test_H@10':>10s} {'Time':>8s}")
    print(f"  {'-' * 94}")

    for model in models:
        model_results = sorted(
            [r for r in results if r['model'] == model],
            key=lambda r: r['drop_rate'])
        for r in model_results:
            print(f"  {r['model']:<22s} {r['drop_rate']:>6.1%} "
                  f"{r['peak_epoch']:>8d} "
                  f"{r['best_val_MRR']:>10.4f} "
                  f"{r['lp_test_MRR']:>10.4f} "
                  f"{r['lp_test_H@1']:>10.4f} "
                  f"{r['lp_test_H@3']:>10.4f} "
                  f"{r['lp_test_H@10']:>10.4f} "
                  f"{r['train_time']:>7.0f}s")
        print()

    if not has_multihop:
        return

    # Multi-hop tables
    for qt in ['1p', '2p', '3p']:
        mrr_key = f'{qt}_MRR'
        if mrr_key not in results[0]:
            continue
        label = {'1p': '1-hop', '2p': '2-hop', '3p': '3-hop'}[qt]
        print(f"{'─' * 100}")
        print(f"  Multi-hop: {label} queries")
        print(f"{'─' * 100}")
        print(f"  {'Model':<22s} {'Drop':>6s} "
              f"{'MRR':>10s} {'H@1':>10s} {'H@3':>10s} {'H@10':>10s} "
              f"{'n':>8s}")
        print(f"  {'-' * 76}")

        for model in models:
            model_results = sorted(
                [r for r in results if r['model'] == model],
                key=lambda r: r['drop_rate'])
            for r in model_results:
                count_key = f'{qt}_count'
                print(f"  {r['model']:<22s} {r['drop_rate']:>6.1%} "
                      f"{r[mrr_key]:>10.4f} "
                      f"{r[f'{qt}_Hits@1']:>10.4f} "
                      f"{r[f'{qt}_Hits@3']:>10.4f} "
                      f"{r[f'{qt}_Hits@10']:>10.4f} "
                      f"{r[count_key]:>8d}")
            print()

    # Differential analysis: best drop_rate per model per query type
    print(f"{'─' * 100}")
    print(f"  Optimal DropEdge rate per model (by MRR)")
    print(f"{'─' * 100}")

    metrics = ['lp_test_MRR']
    metric_labels = ['LP test']
    if has_multihop:
        metrics += ['1p_MRR', '2p_MRR', '3p_MRR']
        metric_labels += ['1p', '2p', '3p']

    print(f"  {'Model':<22s}", end="")
    for label in metric_labels:
        print(f" {'best_' + label:>14s}", end="")
    print()
    print(f"  {'-' * (22 + 15 * len(metric_labels))}")

    for model in models:
        model_results = [r for r in results if r['model'] == model]
        print(f"  {model:<22s}", end="")
        for metric in metrics:
            if metric not in model_results[0]:
                print(f" {'n/a':>14s}", end="")
                continue
            best = max(model_results, key=lambda r: r[metric])
            drop_str = f"{best['drop_rate']:.0%}"
            val_str = f"{best[metric]:.4f}"
            print(f" {val_str}@{drop_str:>4s}", end="  ")
        print()

    print()
    print("=" * 100)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Phase 43: DropEdge Regularization for Multi-hop')
    parser.add_argument('--max_entities', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval_every', type=int, default=25)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_queries', type=int, default=10000,
                        help='Max multi-hop queries per type')
    parser.add_argument('--models', type=str, default='delta_matched,graphgps',
                        help='Comma-separated model names')
    parser.add_argument('--drop_rates', type=str, default='0.0,0.1,0.2,0.3,0.4',
                        help='Comma-separated DropEdge rates')
    parser.add_argument('--multihop', action='store_true',
                        help='Run multi-hop evaluation (slower)')
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',')]
    drop_rates = [float(d.strip()) for d in args.drop_rates.split(',')]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Phase 43: DropEdge Regularization for Multi-hop")
    print(f"  Dataset:     fb15k-237")
    print(f"  Entities:    top {args.max_entities} by degree")
    print(f"  Models:      {models}")
    print(f"  Drop rates:  {drop_rates}")
    print(f"  Seed:        {args.seed}")
    print(f"  Epochs:      {args.epochs} (eval every {args.eval_every}, "
          f"patience {args.patience})")
    print(f"  Multi-hop:   {'yes' if args.multihop else 'no (use --multihop)'}")
    print(f"  Device:      {device}")

    # Load data
    data = load_lp_data('fb15k-237', max_entities=args.max_entities)

    # Multi-hop queries (if requested)
    queries = None
    full_hr2t = None
    if args.multihop:
        print("\nGenerating multi-hop queries...")
        queries = generate_multihop_queries(
            data, max_queries_per_type=args.max_queries, seed=args.seed)
        for qt in ['1p', '2p', '3p']:
            label = {'1p': 'standard LP baseline',
                     '2p': '2-hop compositional',
                     '3p': '3-hop compositional'}[qt]
            print(f"  {qt}: {len(queries[qt])} queries ({label})")

        print("\nRunning leakage audit...")
        issues = audit_queries(queries, data)
        if issues:
            print(f"  FAILED: {len(issues)} leakage issues found!")
            for issue in issues[:5]:
                print(f"    - {issue}")
            sys.exit(1)
        total_q = sum(len(v) for v in queries.values())
        print(f"  PASSED: {total_q} multi-hop queries verified leak-free")

        full_hr2t = build_full_adjacency(data)

    # Run all model × drop_rate combinations
    results = []

    for model_type in models:
        for drop_rate in drop_rates:
            print(f"\n{'═' * 60}")
            print(f"  Model: {model_type}  |  DropEdge: {drop_rate:.0%}")
            print(f"{'═' * 60}")

            result = run_single(
                model_type, data, drop_rate, args, device,
                queries=queries, full_hr2t=full_hr2t)
            results.append(result)

    # Print summary
    print_summary(results, has_multihop=args.multihop)


if __name__ == '__main__':
    main()
