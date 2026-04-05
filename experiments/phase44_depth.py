"""Phase 44: Extended Multi-hop Depth (4p / 5p Compositional Queries)
===================================================================

Phase 42 showed DELTA-Matched is the only model that IMPROVES from 2p→3p.
Phase 44 asks: does this pattern continue at 4-hop and 5-hop depth?

If DELTA-Matched's advantage scales with hop count, it validates the
thesis that edge-to-edge attention with 2-hop adjacency provides
fundamentally better compositional reasoning than node-based GNNs.

Query construction (same leak-free methodology as Phase 42):
  4p: (h,r1,m1)∈TRAIN → (m1,r2,m2)∈TRAIN → (m2,r3,m3)∈TRAIN → (m3,r4,t)∈TEST
  5p: (h,r1,m1)∈TRAIN → ... → (m3,r4,m4)∈TRAIN → (m4,r5,t)∈TEST

This script reuses Phase 42's evaluation engine (soft entity traversal)
and Phase 46c's training infrastructure. It does NOT retrain models —
it loads checkpoints from pre-trained runs and evaluates on extended queries.

Key design:
  - Train once per model, evaluate on ALL hop counts (1p–5p)
  - Same training procedure as Phase 42 (standard LP loss, best-val checkpoint)
  - 4p/5p queries may be sparse on 494-node graph — report counts honestly
  - Leakage audit extended to 4p/5p: intermediate hops in TRAIN, final in TEST,
    no 1-hop shortcuts, no trivial cycles

Usage:
  # Smoke test
  python experiments/phase44_depth.py --epochs 5 --models delta_matched --max_queries 50

  # Full run (key comparison models)
  python experiments/phase44_depth.py --epochs 500 --eval_every 25 --patience 10

  # All models
  python experiments/phase44_depth.py --epochs 500 --eval_every 25 --patience 10 --models delta_matched,graphgps,distmult,grit
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
)

from experiments.phase42_multihop import (
    build_incoming_index,
    build_direct_pairs,
    build_full_adjacency,
    compute_valid_answers,
)


# ═══════════════════════════════════════════════════════════════════════════
# Extended Multi-hop Query Generation (1p through 5p)
# ═══════════════════════════════════════════════════════════════════════════

def generate_extended_queries(data, max_queries_per_type=10000, seed=42):
    """Generate 1p, 2p, 3p, 4p, 5p queries with strict leakage prevention.

    Construction pattern:
      Kp: (K-1) TRAIN hops → 1 TEST hop
      Last hop is always a TEST triple — ensures no info from test leaks into
      the intermediate reasoning chain.

    Returns dict '1p'/'2p'/'3p'/'4p'/'5p' → list of (anchor, rel_chain, answer, intermediates)
    """
    train = data['train']
    test = data['test']

    train_incoming = build_incoming_index(train)
    train_direct = build_direct_pairs(train)
    rng = np.random.RandomState(seed)

    # ── 1p: test triples ──
    queries_1p = []
    for i in range(test.shape[1]):
        h, r, t = test[0, i].item(), test[1, i].item(), test[2, i].item()
        queries_1p.append((h, [r], t, []))

    # ── Kp for K >= 2: (K-1) TRAIN hops → 1 TEST hop ──
    def generate_kp(k, cap=None):
        """Generate k-hop queries via recursive chain building.

        For each test triple (m_last, r_last, t), we recursively find
        (k-1) training hops leading to m_last.
        """
        queries = []
        effective_cap = cap or (max_queries_per_type * 20)  # oversample then dedup

        for i in range(test.shape[1]):
            m_last, r_last, t = (test[0, i].item(), test[1, i].item(),
                                 test[2, i].item())

            # Build chains of length (k-1) ending at m_last, all in TRAIN
            # Each chain is ([h, m1, ..., m_{k-2}], [r1, ..., r_{k-1}])
            chains = _build_train_chains(m_last, k - 1, train_incoming)

            for nodes, rels in chains:
                anchor = nodes[0]
                intermediates = nodes[1:]  # chain already ends at m_last
                rel_chain = rels + [r_last]
                answer = t

                # Leakage filters
                if anchor == answer:
                    continue
                if (anchor, answer) in train_direct:
                    continue
                # No cycles: anchor must not appear as any intermediate
                if anchor in intermediates:
                    continue
                # No repeated intermediates (avoid degenerate loops)
                if len(set(intermediates)) < len(intermediates):
                    continue

                queries.append((anchor, rel_chain, answer, intermediates))

                if len(queries) >= effective_cap:
                    break
            if len(queries) >= effective_cap:
                break

        return queries

    def _build_train_chains(target, depth, incoming, _max_fan=50):
        """Recursively build chains of TRAIN edges ending at target.

        Returns list of (nodes, rels):
          nodes = [start, m1, m2, ...] (length = depth)
          rels  = [r1, r2, ...]        (length = depth)

        _max_fan limits branching to avoid combinatorial explosion.
        """
        if depth == 0:
            return [([target], [])]
        if depth == 1:
            parents = incoming.get(target, [])
            if len(parents) > _max_fan:
                parents = parents[:_max_fan]
            return [([h, target], [r]) for h, r in parents]

        results = []
        parents = incoming.get(target, [])
        if len(parents) > _max_fan:
            parents = parents[:_max_fan]
        for parent, rel in parents:
            sub_chains = _build_train_chains(parent, depth - 1, incoming,
                                             _max_fan=_max_fan)
            for nodes, rels in sub_chains:
                results.append((nodes + [target], rels + [rel]))
                if len(results) > _max_fan * _max_fan:
                    return results  # cap early
        return results

    # Generate all query types
    queries = {'1p': queries_1p}
    for k in [2, 3, 4, 5]:
        label = f'{k}p'
        raw = generate_kp(k)
        deduped = _dedup_queries(raw)

        # Subsample
        if len(deduped) > max_queries_per_type:
            idx = rng.choice(len(deduped), max_queries_per_type, replace=False)
            deduped = [deduped[i] for i in sorted(idx)]

        queries[label] = deduped

    return queries


def _dedup_queries(queries):
    """Deduplicate queries by (anchor, rel_chain, answer)."""
    seen = set()
    unique = []
    for q in queries:
        key = (q[0], tuple(q[1]), q[2])
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


# ═══════════════════════════════════════════════════════════════════════════
# Extended Leakage Audit
# ═══════════════════════════════════════════════════════════════════════════

def audit_extended_queries(queries, data):
    """Verify every query (1p–5p) is leak-free.

    Checks:
      1. All intermediate hops exist in TRAINING triples
      2. Final hop exists in TEST triples
      3. No 1-hop shortcut: (anchor, answer) NOT directly connected in training
      4. Anchor ≠ answer
      5. No cycles: anchor does not appear as any intermediate
      6. No repeated intermediates
    """
    train_set = set()
    for i in range(data['train'].shape[1]):
        h, r, t = (data['train'][0, i].item(), data['train'][1, i].item(),
                    data['train'][2, i].item())
        train_set.add((h, r, t))

    test_set = set()
    for i in range(data['test'].shape[1]):
        h, r, t = (data['test'][0, i].item(), data['test'][1, i].item(),
                    data['test'][2, i].item())
        test_set.add((h, r, t))

    train_direct = build_direct_pairs(data['train'])

    issues = []
    for qt in sorted(queries.keys()):
        if qt == '1p':
            continue  # 1p is just test triples, always valid
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

            # Anchor ≠ answer
            if anchor == answer:
                issues.append(f"{qt}[{idx}]: anchor == answer ({anchor})")

            # No cycles
            if anchor in intermediates:
                issues.append(f"{qt}[{idx}]: anchor {anchor} in intermediates")

            # No repeated intermediates
            if len(set(intermediates)) < len(intermediates):
                issues.append(f"{qt}[{idx}]: repeated intermediates {intermediates}")

    return issues


# ═══════════════════════════════════════════════════════════════════════════
# Extended Multi-hop Evaluation (1p–5p)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_extended_multihop(model, queries_by_type, edge_index, edge_types,
                               full_hr2t, device, batch_size=256,
                               temperature=1.0):
    """Evaluate multi-hop queries (1p–5p) via soft entity traversal.

    Same scoring as Phase 42, extended to arbitrary hop counts.
    """
    model.eval()
    ei = edge_index.to(device)
    et = edge_types.to(device)
    node_feats = model.encode(ei, et)

    results = {}
    for qtype in sorted(queries_by_type.keys()):
        queries = queries_by_type[qtype]
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
            current_emb = node_feats[anchors]

            for hop in range(num_hops):
                rels = torch.tensor([q[1][hop] for q in batch], device=device)
                hr = current_emb * model.decoder_rel_emb(rels)
                scores = hr @ node_feats.t()

                if hop < num_hops - 1:
                    weights = torch.softmax(scores / temperature, dim=-1)
                    current_emb = weights @ node_feats

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
# Training (reuses Phase 42 infrastructure)
# ═══════════════════════════════════════════════════════════════════════════

def train_model(model_type, data, epochs, lr, device, batch_size, seed,
                eval_every=10, patience=5):
    """Train with standard LP loss and return best-val checkpoint."""
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
    peak_epoch = 0
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
    return model, best_val_mrr, edge_index, edge_types, elapsed, peak_epoch


# ═══════════════════════════════════════════════════════════════════════════
# Run & Report
# ═══════════════════════════════════════════════════════════════════════════

def run_single(model_type, data, queries, full_hr2t, args, device):
    """Train one model and evaluate on all hop depths."""
    model, best_val, edge_index, edge_types, train_time, peak_ep = train_model(
        model_type, data, args.epochs, args.lr, device, args.batch_size,
        args.seed, args.eval_every, args.patience)

    # Standard LP sanity check
    lp_test = evaluate_lp(model, data['test'], edge_index, edge_types,
                          data['hr_to_tails'], data['rt_to_heads'], device)
    print(f"    LP test: MRR={lp_test['MRR']:.4f}  H@10={lp_test['Hits@10']:.4f}")

    # Extended multi-hop evaluation
    print(f"    Evaluating multi-hop queries (1p–5p)...")
    t0 = time.time()
    mh = evaluate_extended_multihop(model, queries, edge_index, edge_types,
                                    full_hr2t, device,
                                    temperature=args.temperature)
    eval_time = time.time() - t0

    for qt in sorted(mh.keys()):
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
        'seed': args.seed,
        'params': sum(p.numel() for p in model.parameters()),
        'best_val_MRR': best_val,
        'peak_epoch': peak_ep,
        'lp_test_MRR': lp_test['MRR'],
        'lp_test_H@10': lp_test['Hits@10'],
        'train_time': train_time,
        'eval_time': eval_time,
    }
    for qt in sorted(mh.keys()):
        for k, v in mh[qt].items():
            result[f'{qt}_{k}'] = v

    return result


def print_summary(results, queries):
    """Print comprehensive results across all hop depths."""
    print("\n" + "=" * 110)
    print("PHASE 44: EXTENDED MULTI-HOP DEPTH RESULTS (1p–5p)")
    print("=" * 110)

    qtypes = sorted(queries.keys())
    print(f"\nQuery counts: " +
          ", ".join(f"{qt}={len(queries[qt])}" for qt in qtypes))

    # Per-hop tables
    for qt in qtypes:
        n = results[0].get(f'{qt}_count', 0)
        if n == 0:
            print(f"\n  {qt.upper()}: no queries generated — skipped")
            continue

        hops = qt[:-1]
        print(f"\n{'─' * 110}")
        print(f"  {qt.upper()} queries ({hops}-hop, n={n})")
        print(f"{'─' * 110}")
        print(f"  {'Model':<22s} {'Params':>8s} "
              f"{'MRR':>10s} {'H@1':>10s} {'H@3':>10s} {'H@10':>10s}")
        print(f"  {'-' * 72}")

        for r in results:
            print(f"  {r['model']:<22s} {r['params']:>8,} "
                  f"{r[f'{qt}_MRR']:>10.4f} "
                  f"{r[f'{qt}_Hits@1']:>10.4f} "
                  f"{r[f'{qt}_Hits@3']:>10.4f} "
                  f"{r[f'{qt}_Hits@10']:>10.4f}")

    # MRR trajectory across hops
    print(f"\n{'─' * 110}")
    print(f"  MRR trajectory across hop depths")
    print(f"{'─' * 110}")

    active_qts = [qt for qt in qtypes
                  if results[0].get(f'{qt}_count', 0) > 0]

    header = f"  {'Model':<22s}"
    for qt in active_qts:
        header += f" {qt:>10s}"
    header += "  2p→3p   3p→4p   4p→5p"
    print(header)
    print(f"  {'-' * (22 + 11 * len(active_qts) + 24)}")

    for r in results:
        line = f"  {r['model']:<22s}"
        mrrs = {}
        for qt in active_qts:
            mrr = r[f'{qt}_MRR']
            mrrs[qt] = mrr
            line += f" {mrr:>10.4f}"

        # Delta columns
        for pair in [('2p', '3p'), ('3p', '4p'), ('4p', '5p')]:
            if pair[0] in mrrs and pair[1] in mrrs:
                delta = mrrs[pair[1]] - mrrs[pair[0]]
                line += f" {delta:>+7.4f}"
            else:
                line += f" {'n/a':>7s}"
        print(line)

    # Standard LP sanity check
    print(f"\n{'─' * 110}")
    print(f"  Standard LP test (sanity check)")
    print(f"{'─' * 110}")
    for r in results:
        print(f"  {r['model']:<22s} MRR={r['lp_test_MRR']:.4f}  "
              f"H@10={r['lp_test_H@10']:.4f}  "
              f"(peak ep {r['peak_epoch']})")

    print(f"\n{'=' * 110}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Phase 44: Extended Multi-hop Depth (4p/5p)')
    parser.add_argument('--max_entities', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval_every', type=int, default=25)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_queries', type=int, default=10000,
                        help='Max queries per hop type')
    parser.add_argument('--models', type=str,
                        default='delta_matched,graphgps,distmult',
                        help='Comma-separated model names')
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',')]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Phase 44: Extended Multi-hop Depth (1p–5p)")
    print(f"  Dataset:     fb15k-237")
    print(f"  Entities:    top {args.max_entities} by degree")
    print(f"  Models:      {models}")
    print(f"  Seed:        {args.seed}")
    print(f"  Epochs:      {args.epochs} (eval every {args.eval_every}, "
          f"patience {args.patience})")
    print(f"  Device:      {device}")

    # Load data
    data = load_lp_data('fb15k-237', max_entities=args.max_entities)

    # Generate extended queries
    print("\nGenerating multi-hop queries (1p–5p)...")
    queries = generate_extended_queries(
        data, max_queries_per_type=args.max_queries, seed=args.seed)

    for qt in sorted(queries.keys()):
        n = len(queries[qt])
        hops = qt[:-1]
        label = {
            '1': 'standard LP baseline',
            '2': '2-hop compositional',
            '3': '3-hop compositional',
            '4': '4-hop compositional',
            '5': '5-hop compositional',
        }.get(hops, f'{hops}-hop')
        print(f"  {qt}: {n} queries ({label})")

    # Leakage audit
    print("\nRunning leakage audit (all hop depths)...")
    issues = audit_extended_queries(queries, data)
    if issues:
        print(f"  FAILED: {len(issues)} leakage issues found!")
        for issue in issues[:10]:
            print(f"    - {issue}")
        sys.exit(1)
    total_q = sum(len(v) for v in queries.values())
    print(f"  PASSED: {total_q} queries verified leak-free")

    # Build full adjacency for filtered ranking
    full_hr2t = build_full_adjacency(data)

    # Run all models
    results = []
    for model_type in models:
        print(f"\n{'═' * 60}")
        print(f"  Model: {model_type}")
        print(f"{'═' * 60}")

        result = run_single(model_type, data, queries, full_hr2t, args, device)
        results.append(result)

    # Print summary
    print_summary(results, queries)


if __name__ == '__main__':
    main()
