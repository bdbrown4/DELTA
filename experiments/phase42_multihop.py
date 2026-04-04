"""Phase 42: Multi-hop Path Queries (1p / 2p / 3p)
=================================================

Tests compositional reasoning: can models follow chains of relations?

Query types:
  1p: (anchor, [r1]) → answer                [standard LP — baseline]
  2p: (anchor, [r1, r2]) → answer            [2-hop compositional]
  3p: (anchor, [r1, r2, r3]) → answer        [3-hop compositional]

Methodology:
  1. Train each model with standard LP loss (identical to Phase 40).
  2. Generate multi-hop queries from the knowledge graph:
     - 2p: (h, r1, m) ∈ TRAIN, (m, r2, t) ∈ TEST  → query (h, [r1,r2], t)
     - 3p: (h,r1,m1)∈TRAIN, (m1,r2,m2)∈TRAIN, (m2,r3,t)∈TEST
           → query (h, [r1,r2,r3], t)
  3. Score via soft entity traversal:
     - For each hop: score all entities, softmax-weight, and average embeddings.
     - Final hop: rank all entities by score.
  4. Filtered ranking: filter all valid multi-hop answers from all graph splits.

Why this tests DELTA's edge-to-edge attention:
  Multi-hop queries require composing relational patterns. DELTA propagates
  information along edges (not just nodes), potentially capturing relational
  chains more naturally than MPNN-based models like GraphGPS.

Leakage prevention (lessons from Phase 37):
  ✓ GNN encoder sees ONLY training graph edges
  ✓ Intermediate hops use TRAINING edges; final hop uses TEST edges
  ✓ Edge features are learned nn.Embedding (no label encoding)
  ✓ Filtered ranking removes ALL valid multi-hop answers from all splits
  ✓ 1-hop shortcuts excluded: 2p/3p queries require that the answer is
    NOT directly reachable from the anchor via a single training edge
  ✓ Built-in leakage audit verifies every query before evaluation

Usage:
  # Smoke test (500 entities, 5 epochs)
  python experiments/phase42_multihop.py

  # Full Phase 40-style training + multi-hop eval
  python experiments/phase42_multihop.py --epochs 500 --eval_every 25 --patience 10 --seeds 3

  # Single model quick test
  python experiments/phase42_multihop.py --models graphgps --epochs 20
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import torch
from collections import defaultdict

from experiments.phase46c_link_prediction import (
    load_lp_data,
    build_train_graph_tensors,
    create_lp_model,
    train_epoch,
    evaluate_lp,
    ALL_MODELS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-hop Query Generation (LEAK-FREE)
# ═══════════════════════════════════════════════════════════════════════════

def build_incoming_index(triples):
    """Build reverse adjacency: entity → list of (head, relation) from training.

    For entity m, returns all (h, r) such that (h, r, m) exists in triples.
    """
    incoming = defaultdict(list)
    for i in range(triples.shape[1]):
        h = triples[0, i].item()
        r = triples[1, i].item()
        t = triples[2, i].item()
        incoming[t].append((h, r))
    return dict(incoming)


def build_direct_pairs(triples):
    """Build set of all (head, tail) pairs reachable in 1 hop."""
    pairs = set()
    for i in range(triples.shape[1]):
        h = triples[0, i].item()
        t = triples[2, i].item()
        pairs.add((h, t))
    return pairs


def generate_multihop_queries(data, max_queries_per_type=10000, seed=42):
    """Generate 1p, 2p, 3p queries with strict leakage prevention.

    Construction:
      1p: Each test triple (h, r, t) → query (h, [r], t)
      2p: For test triple (m, r2, t), find training edges (h, r1) → m
          → query (h, [r1, r2], t) with intermediate m
      3p: For test triple (m2, r3, t), find training chains
          (h, r1) → m1, (m1, r2) → m2
          → query (h, [r1, r2, r3], t) with intermediates [m1, m2]

    Leakage filters:
      - Exclude queries where anchor == answer (degenerate)
      - Exclude 2p/3p where (anchor, answer) is reachable in 1 training hop
      - Exclude 3p where anchor == m2 (trivial cycle)
      - Deduplicate: same (anchor, rel_chain, answer) kept only once

    Returns dict '1p'/'2p'/'3p' → list of (anchor, rel_chain, answer, intermediates)
    """
    train = data['train']
    test = data['test']

    train_incoming = build_incoming_index(train)
    train_direct = build_direct_pairs(train)

    # ── 1p: test triples ──
    queries_1p = []
    for i in range(test.shape[1]):
        h, r, t = test[0, i].item(), test[1, i].item(), test[2, i].item()
        queries_1p.append((h, [r], t, []))

    # ── 2p: TRAIN hop → TEST hop ──
    queries_2p = []
    for i in range(test.shape[1]):
        m, r2, t = test[0, i].item(), test[1, i].item(), test[2, i].item()
        for h, r1 in train_incoming.get(m, []):
            if h == t:                       # anchor == answer: degenerate
                continue
            if (h, t) in train_direct:       # 1-hop shortcut in training
                continue
            queries_2p.append((h, [r1, r2], t, [m]))

    # ── 3p: TRAIN → TRAIN → TEST ──
    queries_3p = []
    for i in range(test.shape[1]):
        m2, r3, t = test[0, i].item(), test[1, i].item(), test[2, i].item()
        for m1, r2 in train_incoming.get(m2, []):
            for h, r1 in train_incoming.get(m1, []):
                if h == t:                   # anchor == answer
                    continue
                if h == m2:                  # trivial cycle h→m1→h→...
                    continue
                if (h, t) in train_direct:   # 1-hop shortcut
                    continue
                queries_3p.append((h, [r1, r2, r3], t, [m1, m2]))

    # Deduplicate: same (anchor, rel_chain, answer) via different intermediates
    def dedup(queries):
        seen = set()
        unique = []
        for q in queries:
            key = (q[0], tuple(q[1]), q[2])
            if key not in seen:
                seen.add(key)
                unique.append(q)
        return unique

    queries_2p = dedup(queries_2p)
    queries_3p = dedup(queries_3p)

    # Subsample if too many
    rng = np.random.RandomState(seed)
    if len(queries_2p) > max_queries_per_type:
        idx = rng.choice(len(queries_2p), max_queries_per_type, replace=False)
        queries_2p = [queries_2p[i] for i in sorted(idx)]
    if len(queries_3p) > max_queries_per_type:
        idx = rng.choice(len(queries_3p), max_queries_per_type, replace=False)
        queries_3p = [queries_3p[i] for i in sorted(idx)]

    return {'1p': queries_1p, '2p': queries_2p, '3p': queries_3p}


# ═══════════════════════════════════════════════════════════════════════════
# Leakage Audit
# ═══════════════════════════════════════════════════════════════════════════

def audit_queries(queries, data):
    """Verify every multi-hop query is leak-free. Returns list of issues.

    Checks (for 2p / 3p):
      1. All intermediate hops exist in TRAINING triples
      2. Final hop exists in TEST triples
      3. No 1-hop shortcut: (anchor, answer) NOT directly connected in training
      4. Anchor ≠ answer
    """
    train_set = set()
    for i in range(data['train'].shape[1]):
        h = data['train'][0, i].item()
        r = data['train'][1, i].item()
        t = data['train'][2, i].item()
        train_set.add((h, r, t))

    test_set = set()
    for i in range(data['test'].shape[1]):
        h = data['test'][0, i].item()
        r = data['test'][1, i].item()
        t = data['test'][2, i].item()
        test_set.add((h, r, t))

    train_direct = build_direct_pairs(data['train'])

    issues = []
    for qt in ['2p', '3p']:
        for idx, (anchor, rels, answer, intermediates) in enumerate(queries[qt]):
            chain = [anchor] + intermediates + [answer]

            # Check intermediate hops are in training
            for hop in range(len(rels) - 1):
                triple = (chain[hop], rels[hop], chain[hop + 1])
                if triple not in train_set:
                    issues.append(
                        f"{qt}[{idx}]: intermediate hop {triple} not in train")

            # Check final hop is in test
            final = (chain[-2], rels[-1], chain[-1])
            if final not in test_set:
                issues.append(f"{qt}[{idx}]: final hop {final} not in test")

            # Check no 1-hop shortcut
            if (anchor, answer) in train_direct:
                issues.append(
                    f"{qt}[{idx}]: 1-hop shortcut ({anchor} → {answer}) in train")

            # Check anchor ≠ answer
            if anchor == answer:
                issues.append(f"{qt}[{idx}]: anchor == answer ({anchor})")

    return issues


# ═══════════════════════════════════════════════════════════════════════════
# Multi-hop Filtering & Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def build_full_adjacency(data):
    """Build (h, r) → {tails} from ALL splits for filtered evaluation."""
    all_triples = torch.cat([data['train'], data['val'], data['test']], dim=1)
    hr2t = defaultdict(set)
    for i in range(all_triples.shape[1]):
        h = all_triples[0, i].item()
        r = all_triples[1, i].item()
        t = all_triples[2, i].item()
        hr2t[(h, r)].add(t)
    return dict(hr2t)


def compute_valid_answers(anchor, rel_chain, full_hr2t):
    """Compute ALL entities reachable via the relation chain from anchor.

    Walks the full graph (train + val + test):
      reachable_0 = {anchor}
      reachable_i = ∪_{e ∈ reachable_{i-1}} full_hr2t[(e, r_i)]
    Returns the final reachable set.
    """
    current = {anchor}
    for rel in rel_chain:
        nxt = set()
        for e in current:
            nxt.update(full_hr2t.get((e, rel), set()))
        current = nxt
        if not current:
            break
    return current


@torch.no_grad()
def evaluate_multihop(model, queries_by_type, edge_index, edge_types,
                      full_hr2t, device, batch_size=256, temperature=1.0):
    """Evaluate multi-hop queries via soft entity traversal with filtered ranking.

    Scoring (for a K-hop query):
      current_emb = node_feats[anchor]
      For each hop i = 1..K:
        scores_i = (current_emb ⊙ decoder_rel_emb(r_i)) @ node_feats.T
        if i < K:  current_emb = softmax(scores_i / τ) @ node_feats  [soft intermediate]
        if i == K:  rank entities by scores_i                         [final answer]

    Filtering:
      For each query, compute all valid answers reachable through the FULL graph.
      Filter those (except the target) from the ranking — same principle as
      filtered LP but extended to multi-hop paths.

    Returns dict: query_type → {MRR, Hits@1, Hits@3, Hits@10, count}
    """
    model.eval()
    ei = edge_index.to(device)
    et = edge_types.to(device)
    node_feats = model.encode(ei, et)

    results = {}
    for qtype in ['1p', '2p', '3p']:
        queries = queries_by_type.get(qtype, [])
        if not queries:
            results[qtype] = {
                'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0,
                'count': 0,
            }
            continue

        num_hops = len(queries[0][1])
        all_ranks = []

        for start in range(0, len(queries), batch_size):
            batch = queries[start:start + batch_size]
            B = len(batch)

            anchors = torch.tensor([q[0] for q in batch], device=device)
            current_emb = node_feats[anchors]                   # [B, d]

            for hop in range(num_hops):
                rels = torch.tensor([q[1][hop] for q in batch], device=device)
                hr = current_emb * model.decoder_rel_emb(rels)  # [B, d]
                scores = hr @ node_feats.t()                     # [B, N]

                if hop < num_hops - 1:
                    weights = torch.softmax(scores / temperature, dim=-1)
                    current_emb = weights @ node_feats           # [B, d]

            # Filtered ranking on final scores
            for i in range(B):
                anchor = batch[i][0]
                rel_chain = batch[i][1]
                answer = batch[i][2]

                valid = compute_valid_answers(anchor, rel_chain, full_hr2t)
                for va in valid:
                    if va != answer:
                        scores[i, va] = float('-inf')

                rank = int((scores[i] >= scores[i, answer]).sum().item())
                all_ranks.append(max(rank, 1))

        ranks = np.array(all_ranks, dtype=np.float64)
        results[qtype] = {
            'MRR': float(np.mean(1.0 / ranks)),
            'Hits@1': float(np.mean(ranks <= 1)),
            'Hits@3': float(np.mean(ranks <= 3)),
            'Hits@10': float(np.mean(ranks <= 10)),
            'count': len(queries),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Training — wraps Phase 40 components, returns the trained model
# ═══════════════════════════════════════════════════════════════════════════

def train_model(model_type, data, epochs, lr, device, batch_size, seed,
                eval_every=10, patience=5):
    """Train with standard LP loss and return best-val checkpoint.

    Same training procedure as Phase 40 (phase46c_link_prediction.py), but
    returns the model object for downstream multi-hop evaluation.

    Returns (model, best_val_mrr, edge_index, edge_types, train_time_s)
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

    edge_index, edge_types = build_train_graph_tensors(data['train'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_mrr = 0.0
    best_state = None
    evals_no_improve = 0
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
    return model, best_val_mrr, edge_index, edge_types, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Single model run (train + multi-hop eval)
# ═══════════════════════════════════════════════════════════════════════════

def run_single_multihop(model_type, data, queries, full_hr2t, epochs, lr,
                        device, batch_size, seed, eval_every, patience,
                        temperature=1.0):
    """Train one model, evaluate on 1p/2p/3p queries.

    Returns dict with per-query-type metrics + training metadata.
    """
    model, best_val, edge_index, edge_types, train_time = train_model(
        model_type, data, epochs, lr, device, batch_size, seed,
        eval_every, patience)

    # Standard LP on test set (sanity check against Phase 40)
    lp_test = evaluate_lp(model, data['test'], edge_index, edge_types,
                          data['hr_to_tails'], data['rt_to_heads'], device)
    print(f"    Standard LP: test_MRR={lp_test['MRR']:.4f}  "
          f"test_H@10={lp_test['Hits@10']:.4f}")

    # Multi-hop evaluation
    print(f"    Evaluating multi-hop queries...")
    t0 = time.time()
    mh = evaluate_multihop(model, queries, edge_index, edge_types,
                           full_hr2t, device, temperature=temperature)
    eval_time = time.time() - t0

    for qt in ['1p', '2p', '3p']:
        r = mh[qt]
        if r['count'] > 0:
            print(f"    {qt}: MRR={r['MRR']:.4f}  H@1={r['Hits@1']:.4f}  "
                  f"H@3={r['Hits@3']:.4f}  H@10={r['Hits@10']:.4f}  "
                  f"(n={r['count']})")
        else:
            print(f"    {qt}: no queries generated")
    print(f"    Multi-hop eval: {eval_time:.1f}s")

    result = {
        'model': model_type,
        'seed': seed,
        'params': sum(p.numel() for p in model.parameters()),
        'best_val_MRR': best_val,
        'lp_test_MRR': lp_test['MRR'],
        'lp_test_H@10': lp_test['Hits@10'],
        'train_time': train_time,
        'eval_time': eval_time,
    }
    for qt in ['1p', '2p', '3p']:
        for k, v in mh[qt].items():
            result[f'{qt}_{k}'] = v

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Multi-seed aggregation
# ═══════════════════════════════════════════════════════════════════════════

def run_multi_seed(model_type, data, queries, full_hr2t, args, device, seeds):
    """Run one model across seeds, aggregate metrics."""
    results = []
    for seed in seeds:
        try:
            r = run_single_multihop(
                model_type, data, queries, full_hr2t,
                args.epochs, args.lr, device, args.batch_size, seed,
                args.eval_every, args.patience, args.temperature)
            results.append(r)
        except Exception as e:
            print(f"    FAILED seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        return None

    agg = {
        'model': model_type,
        'params': results[0]['params'],
        'num_seeds': len(results),
    }

    for metric in ['lp_test_MRR', 'lp_test_H@10']:
        vals = [r[metric] for r in results]
        agg[f'{metric}_mean'] = float(np.mean(vals))

    for qt in ['1p', '2p', '3p']:
        agg[f'{qt}_count'] = results[0].get(f'{qt}_count', 0)
        for metric in ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']:
            key = f'{qt}_{metric}'
            vals = [r[key] for r in results]
            agg[f'{key}_mean'] = float(np.mean(vals))
            agg[f'{key}_std'] = float(np.std(vals)) if len(vals) > 1 else 0.0

    agg['time_mean'] = float(np.mean([r['train_time'] for r in results]))
    return agg


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(all_results, queries):
    """Print comprehensive results table."""
    valid = [r for r in all_results if r is not None]
    if not valid:
        print("\nNo results to summarize.")
        return

    print("\n" + "=" * 100)
    print("PHASE 42: MULTI-HOP PATH QUERY RESULTS")
    print("=" * 100)

    print(f"\nQuery counts: 1p={len(queries['1p'])}, "
          f"2p={len(queries['2p'])}, 3p={len(queries['3p'])}")

    for qt in ['1p', '2p', '3p']:
        n = valid[0].get(f'{qt}_count', 0)
        if n == 0:
            print(f"\n  {qt.upper()}: no queries — skipped")
            continue

        hops = qt[0]
        print(f"\n{'─' * 100}")
        print(f"  {qt.upper()} queries ({hops}-hop, n={n})")
        print(f"{'─' * 100}")

        header = (f"  {'Model':<22} {'Params':>8} "
                  f"{'MRR':>14} {'H@1':>14} {'H@3':>14} {'H@10':>14}")
        print(header)
        print(f"  {'-' * 94}")

        for r in valid:
            def fmt(metric):
                m = r[f'{qt}_{metric}_mean']
                s = r[f'{qt}_{metric}_std']
                return f"{m:.4f}±{s:.3f}" if s > 0 else f"{m:.4f}      "

            print(f"  {r['model']:<22} {r['params']:>8,} "
                  f"{fmt('MRR'):>14} {fmt('Hits@1'):>14} "
                  f"{fmt('Hits@3'):>14} {fmt('Hits@10'):>14}")

    # Multi-hop degradation analysis
    has_multihop = any(valid[0].get(f'{qt}_count', 0) > 0
                       for qt in ['2p', '3p'])
    if has_multihop:
        print(f"\n{'─' * 100}")
        print("  Multi-hop degradation (MRR): 1p → 2p → 3p")
        print(f"{'─' * 100}")

        for r in valid:
            mrr_1p = r.get('1p_MRR_mean', 0)
            parts = [f"  {r['model']:<22} 1p={mrr_1p:.4f}"]

            for qt in ['2p', '3p']:
                if r.get(f'{qt}_count', 0) > 0:
                    mrr = r[f'{qt}_MRR_mean']
                    drop = mrr - mrr_1p
                    parts.append(f"{qt}={mrr:.4f} (Δ={drop:+.4f})")

            print("  ".join(parts))

    # LP sanity check
    print(f"\n{'─' * 100}")
    print("  Standard LP test MRR (sanity check vs Phase 40)")
    print(f"{'─' * 100}")
    for r in valid:
        print(f"  {r['model']:<22} MRR={r['lp_test_MRR_mean']:.4f}  "
              f"H@10={r['lp_test_H@10_mean']:.4f}")

    print(f"\n{'=' * 100}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Phase 42: Multi-hop path query evaluation')
    parser.add_argument('--max_entities', type=int, default=500,
                        help='Entity subset size (default: 500 = Phase 40 config)')
    parser.add_argument('--full', action='store_true',
                        help='Full FB15k-237 (no entity limit)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs (default: 5 for smoke test)')
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate every N epochs')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model names (default: all)')
    parser.add_argument('--dataset', type=str, default='fb15k-237')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Softmax temperature for soft traversal')
    parser.add_argument('--max_queries', type=int, default=10000,
                        help='Max queries per type (default: 10000)')
    args = parser.parse_args()

    max_ent = None if args.full else args.max_entities
    models = args.models.split(',') if args.models else ALL_MODELS
    seeds = list(range(1, args.seeds + 1))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Phase 42: Multi-hop Path Query Evaluation")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Entities:    {'FULL' if args.full else f'top {args.max_entities} by degree'}")
    print(f"  Models:      {models}")
    print(f"  Seeds:       {seeds}")
    print(f"  Epochs:      {args.epochs} (eval every {args.eval_every}, "
          f"patience {args.patience})")
    print(f"  Temperature: {args.temperature}")
    print(f"  Device:      {device}")
    print()

    # ── Load data ──
    data = load_lp_data(args.dataset, max_entities=max_ent)

    # ── Generate queries ──
    print("\nGenerating multi-hop queries...")
    queries = generate_multihop_queries(
        data, max_queries_per_type=args.max_queries)
    print(f"  1p: {len(queries['1p'])} queries (standard LP baseline)")
    print(f"  2p: {len(queries['2p'])} queries (2-hop compositional)")
    print(f"  3p: {len(queries['3p'])} queries (3-hop compositional)")

    if len(queries['2p']) == 0:
        print("\n  WARNING: No 2p queries generated. The entity subset may be")
        print("  too sparse for multi-hop paths. Try --full or larger "
              "--max_entities.")

    # ── Leakage audit ──
    print("\nRunning leakage audit...")
    issues = audit_queries(queries, data)
    if issues:
        print(f"  LEAKAGE DETECTED — {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"    {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")
        print("  ABORTING. Fix query generation before proceeding.")
        sys.exit(1)
    else:
        total_mh = len(queries['2p']) + len(queries['3p'])
        print(f"  PASSED: {total_mh} multi-hop queries verified leak-free")

    # ── Build full-graph adjacency for multi-hop filtered ranking ──
    full_hr2t = build_full_adjacency(data)

    # ── Run all models ──
    all_results = []
    for model_type in models:
        print(f"\n{'═' * 60}")
        print(f"  Model: {model_type}")
        print(f"{'═' * 60}")
        agg = run_multi_seed(model_type, data, queries, full_hr2t,
                             args, device, seeds)
        all_results.append(agg)

    print_summary(all_results, queries)


if __name__ == '__main__':
    main()
