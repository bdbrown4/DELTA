"""Phase 45: Inference Timing + Multi-seed Headline Validation
=============================================================

Two concrete pre-submission needs:

1. **Inference timing.** Separate training cost from deployment cost.
   Run timed evaluation passes on 1p/2p/3p for DELTA-Matched and GraphGPS.
   If DELTA's inference is within 2-3x of GraphGPS, the 35x training cost
   is a manageable limitation. If inference is also 35x slower, harder problem.

2. **Multi-seed on headline configuration.** Phase 42-43 are single-seed.
   Run 3 seeds on:
     - DELTA-Matched @10% DropEdge (recommended headline from Phase 43)
     - GraphGPS @0% DropEdge (baseline competitor)
   Report mean ± std on 1p/2p/3p to confirm the advantage holds.

Usage:
  # Smoke test
  python experiments/phase45_inference_timing.py --epochs 5 --seeds 1

  # Full run (3 seeds)
  python experiments/phase45_inference_timing.py --epochs 500 --eval_every 25 --patience 10 --seeds 1,2,3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F

from experiments.phase46c_link_prediction import (
    load_lp_data,
    build_train_graph_tensors,
    create_lp_model,
    train_epoch,
    evaluate_lp,
)

from experiments.phase42_multihop import (
    generate_multihop_queries,
    audit_queries,
    build_full_adjacency,
    evaluate_multihop,
)

from experiments.phase43_regularization import (
    train_epoch_dropedge,
)


# ═══════════════════════════════════════════════════════════════════════════
# Training (supports optional DropEdge)
# ═══════════════════════════════════════════════════════════════════════════

def train_model(model_type, data, epochs, lr, device, batch_size, seed,
                eval_every, patience, drop_rate=0.0):
    """Train with optional DropEdge, return best-val checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = create_lp_model(model_type,
                            data['num_entities'], data['num_relations'])
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = (sum(p.numel() for p in model.encoder.parameters())
             if model.encoder is not None else 0)
    drop_str = f", drop={drop_rate}" if drop_rate > 0 else ""
    print(f"\n  [{model_type}] seed={seed}{drop_str}, "
          f"{n_params:,} params ({n_enc:,} encoder)")

    edge_index, edge_types = build_train_graph_tensors(data['train'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_mrr = 0.0
    best_state = None
    peak_epoch = 0
    evals_no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        if drop_rate > 0:
            loss = train_epoch_dropedge(
                model, data['train'], edge_index, edge_types,
                optimizer, device, batch_size, drop_rate=drop_rate)
        else:
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
# Inference Timing
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def timed_inference(model, queries_by_type, edge_index, edge_types,
                    full_hr2t, device, batch_size=256, temperature=1.0,
                    warmup_runs=3, timed_runs=10):
    """Run multi-hop evaluation with precise timing.

    Separates:
      - Encoding time (GNN forward pass — run once)
      - Per-query-type scoring time (soft entity traversal)
      - Filtered ranking time (compute_valid_answers + rank)

    Returns dict with timing breakdown and metrics.
    """
    from experiments.phase42_multihop import compute_valid_answers

    model.eval()
    ei = edge_index.to(device)
    et = edge_types.to(device)

    # ── Encoding time (GNN forward pass) ──
    # Warmup
    for _ in range(warmup_runs):
        _ = model.encode(ei, et)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    encode_times = []
    for _ in range(timed_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        node_feats = model.encode(ei, et)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        encode_times.append(time.perf_counter() - t0)

    # ── Per-query-type scoring + ranking ──
    timing = {
        'encode_mean_ms': np.mean(encode_times) * 1000,
        'encode_std_ms': np.std(encode_times) * 1000,
    }

    node_feats = model.encode(ei, et)  # final encoding for scoring

    for qtype in sorted(queries_by_type.keys()):
        queries = queries_by_type.get(qtype, [])
        if not queries:
            continue

        num_hops = len(queries[0][1])

        # Warmup
        for _ in range(min(warmup_runs, 1)):
            _run_scoring(model, queries[:min(batch_size, len(queries))],
                         node_feats, device, num_hops, temperature,
                         full_hr2t, batch_size)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        score_times = []
        for _ in range(timed_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _run_scoring(model, queries, node_feats, device, num_hops,
                         temperature, full_hr2t, batch_size)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            score_times.append(time.perf_counter() - t0)

        n_queries = len(queries)
        total_ms = np.mean(score_times) * 1000
        per_query_us = total_ms / n_queries * 1000  # microseconds

        timing[f'{qtype}_total_ms'] = total_ms
        timing[f'{qtype}_total_std_ms'] = np.std(score_times) * 1000
        timing[f'{qtype}_per_query_us'] = per_query_us
        timing[f'{qtype}_count'] = n_queries

    return timing


def _run_scoring(model, queries, node_feats, device, num_hops, temperature,
                 full_hr2t, batch_size):
    """Run scoring + filtered ranking for a set of queries (for timing)."""
    from experiments.phase42_multihop import compute_valid_answers

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

        # Filtered ranking
        for i in range(B):
            anchor = batch[i][0]
            rel_chain = batch[i][1]
            answer = batch[i][2]
            valid = compute_valid_answers(anchor, rel_chain, full_hr2t)
            for va in valid:
                if va != answer:
                    scores[i, va] = float('-inf')
            _ = int((scores[i] >= scores[i, answer]).sum().item())


# ═══════════════════════════════════════════════════════════════════════════
# Run & Report
# ═══════════════════════════════════════════════════════════════════════════

def run_config(model_type, drop_rate, data, queries, full_hr2t, args,
               device, seeds):
    """Train + evaluate + time inference across seeds for one config."""
    drop_str = f"@{int(drop_rate*100)}%drop" if drop_rate > 0 else "@0%drop"
    config_name = f"{model_type}{drop_str}"

    all_results = []
    all_timing = []

    for seed in seeds:
        print(f"\n{'─' * 60}")
        print(f"  {config_name} — seed {seed}")
        print(f"{'─' * 60}")

        model, best_val, edge_index, edge_types, train_time, peak_ep = \
            train_model(model_type, data, args.epochs, args.lr, device,
                        args.batch_size, seed, args.eval_every,
                        args.patience, drop_rate)

        # Standard LP
        lp_test = evaluate_lp(model, data['test'], edge_index, edge_types,
                              data['hr_to_tails'], data['rt_to_heads'], device)
        print(f"    LP test: MRR={lp_test['MRR']:.4f}  "
              f"H@10={lp_test['Hits@10']:.4f}")

        # Multi-hop eval
        print(f"    Evaluating multi-hop...")
        mh = evaluate_multihop(model, queries, edge_index, edge_types,
                               full_hr2t, device)
        for qt in ['1p', '2p', '3p']:
            r = mh[qt]
            if r['count'] > 0:
                print(f"    {qt}: MRR={r['MRR']:.4f}  H@10={r['Hits@10']:.4f}"
                      f"  (n={r['count']})")

        # Inference timing
        print(f"    Timing inference (10 runs)...")
        timing = timed_inference(model, queries, edge_index, edge_types,
                                 full_hr2t, device)
        print(f"    Encode: {timing['encode_mean_ms']:.2f} ± "
              f"{timing['encode_std_ms']:.2f} ms")
        for qt in ['1p', '2p', '3p']:
            key = f'{qt}_total_ms'
            if key in timing:
                print(f"    {qt}: {timing[key]:.1f}ms total, "
                      f"{timing[f'{qt}_per_query_us']:.1f}μs/query "
                      f"(n={timing[f'{qt}_count']})")

        result = {
            'seed': seed,
            'config': config_name,
            'model': model_type,
            'drop_rate': drop_rate,
            'params': sum(p.numel() for p in model.parameters()),
            'best_val_MRR': best_val,
            'peak_epoch': peak_ep,
            'train_time': train_time,
            'lp_test_MRR': lp_test['MRR'],
            'lp_test_H@10': lp_test['Hits@10'],
        }
        for qt in ['1p', '2p', '3p']:
            for k, v in mh[qt].items():
                result[f'{qt}_{k}'] = v

        all_results.append(result)
        all_timing.append(timing)

    return all_results, all_timing


def print_summary(configs_results, configs_timing):
    """Print comprehensive multi-seed summary with timing."""
    print("\n" + "=" * 110)
    print("PHASE 45: INFERENCE TIMING + MULTI-SEED HEADLINE VALIDATION")
    print("=" * 110)

    # Group by config
    configs = {}
    for results, timing in zip(configs_results, configs_timing):
        name = results[0]['config']
        configs[name] = (results, timing)

    # ── Multi-seed MRR summary ──
    print(f"\n{'─' * 110}")
    print(f"  Multi-hop MRR (mean ± std across seeds)")
    print(f"{'─' * 110}")

    header = (f"  {'Config':<30s} {'Seeds':>5s} {'Params':>8s} "
              f"{'1p MRR':>12s} {'2p MRR':>12s} {'3p MRR':>12s} "
              f"{'2p→3p':>8s}")
    print(header)
    print(f"  {'-' * 100}")

    for name, (results, timing) in configs.items():
        n = len(results)
        params = results[0]['params']

        mrrs = {}
        for qt in ['1p', '2p', '3p']:
            vals = [r[f'{qt}_MRR'] for r in results]
            mean = np.mean(vals)
            std = np.std(vals)
            mrrs[qt] = (mean, std)

        delta_23 = mrrs['3p'][0] - mrrs['2p'][0]

        def fmt(m, s):
            if n > 1:
                return f"{m:.4f}±{s:.4f}"
            return f"{m:.4f}"

        print(f"  {name:<30s} {n:>5d} {params:>8,} "
              f"{fmt(*mrrs['1p']):>12s} {fmt(*mrrs['2p']):>12s} "
              f"{fmt(*mrrs['3p']):>12s} {delta_23:>+8.4f}")

    # ── Standard LP summary ──
    print(f"\n{'─' * 110}")
    print(f"  Standard LP test (mean ± std)")
    print(f"{'─' * 110}")

    for name, (results, timing) in configs.items():
        mrrs = [r['lp_test_MRR'] for r in results]
        h10s = [r['lp_test_H@10'] for r in results]
        trains = [r['train_time'] for r in results]
        peaks = [r['peak_epoch'] for r in results]
        n = len(results)

        if n > 1:
            print(f"  {name:<30s} MRR={np.mean(mrrs):.4f}±{np.std(mrrs):.4f}  "
                  f"H@10={np.mean(h10s):.4f}±{np.std(h10s):.4f}  "
                  f"train={np.mean(trains):.0f}±{np.std(trains):.0f}s  "
                  f"peak_ep={np.mean(peaks):.0f}")
        else:
            print(f"  {name:<30s} MRR={mrrs[0]:.4f}  "
                  f"H@10={h10s[0]:.4f}  "
                  f"train={trains[0]:.0f}s  peak_ep={peaks[0]}")

    # ── Inference timing ──
    print(f"\n{'─' * 110}")
    print(f"  Inference Timing (mean across seeds, 10 timed runs each)")
    print(f"{'─' * 110}")

    header = (f"  {'Config':<30s} {'Encode (ms)':>14s} "
              f"{'1p total':>10s} {'1p/query':>10s} "
              f"{'2p total':>10s} {'2p/query':>10s} "
              f"{'3p total':>10s} {'3p/query':>10s}")
    print(header)
    print(f"  {'-' * 108}")

    for name, (results, timings) in configs.items():
        enc = np.mean([t['encode_mean_ms'] for t in timings])

        parts = [f"  {name:<30s} {enc:>11.2f}ms"]
        for qt in ['1p', '2p', '3p']:
            key_t = f'{qt}_total_ms'
            key_q = f'{qt}_per_query_us'
            if key_t in timings[0]:
                total = np.mean([t[key_t] for t in timings])
                per_q = np.mean([t[key_q] for t in timings])
                parts.append(f"{total:>8.1f}ms {per_q:>8.1f}μs")
            else:
                parts.append(f"{'n/a':>8s} {'n/a':>8s}")
        print(" ".join(parts))

    # ── Timing ratio ──
    config_list = list(configs.keys())
    if len(config_list) >= 2:
        t1 = configs[config_list[0]][1]
        t2 = configs[config_list[1]][1]

        enc1 = np.mean([t['encode_mean_ms'] for t in t1])
        enc2 = np.mean([t['encode_mean_ms'] for t in t2])

        print(f"\n  Encoding ratio: {config_list[0]} / {config_list[1]} = "
              f"{enc1/enc2:.1f}×")

        for qt in ['1p', '2p', '3p']:
            key = f'{qt}_per_query_us'
            if key in t1[0] and key in t2[0]:
                pq1 = np.mean([t[key] for t in t1])
                pq2 = np.mean([t[key] for t in t2])
                print(f"  {qt} per-query ratio: {pq1/pq2:.1f}×")

        train1 = np.mean([r['train_time'] for r in configs[config_list[0]][0]])
        train2 = np.mean([r['train_time'] for r in configs[config_list[1]][0]])
        print(f"  Training ratio: {train1/train2:.1f}× "
              f"({train1:.0f}s vs {train2:.0f}s)")

    print(f"\n{'=' * 110}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Phase 45: Inference Timing + Multi-seed Headline')
    parser.add_argument('--max_entities', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval_every', type=int, default=25)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seeds', type=str, default='1,2,3',
                        help='Comma-separated seeds')
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Phase 45: Inference Timing + Multi-seed Headline Validation")
    print(f"  Seeds:     {seeds}")
    print(f"  Epochs:    {args.epochs} (eval every {args.eval_every}, "
          f"patience {args.patience})")
    print(f"  Device:    {device}")
    print(f"  Configs:   DELTA-Matched @10% drop | GraphGPS @0% drop")

    data = load_lp_data('fb15k-237', max_entities=args.max_entities)

    print("\nGenerating multi-hop queries...")
    queries = generate_multihop_queries(data)
    for qt in ['1p', '2p', '3p']:
        print(f"  {qt}: {len(queries[qt])} queries")

    print("\nRunning leakage audit...")
    issues = audit_queries(queries, data)
    if issues:
        print(f"  FAILED: {len(issues)} issues!")
        sys.exit(1)
    total = sum(len(v) for v in queries.values())
    print(f"  PASSED: {total} queries verified leak-free")

    full_hr2t = build_full_adjacency(data)

    # Config 1: DELTA-Matched @10% DropEdge (headline from Phase 43)
    print(f"\n{'═' * 60}")
    print(f"  Config: DELTA-Matched @10% DropEdge")
    print(f"{'═' * 60}")
    delta_results, delta_timing = run_config(
        'delta_matched', 0.1, data, queries, full_hr2t, args, device, seeds)

    # Config 2: GraphGPS @0% DropEdge (baseline competitor)
    print(f"\n{'═' * 60}")
    print(f"  Config: GraphGPS @0% DropEdge")
    print(f"{'═' * 60}")
    gps_results, gps_timing = run_config(
        'graphgps', 0.0, data, queries, full_hr2t, args, device, seeds)

    print_summary(
        [delta_results, gps_results],
        [delta_timing, gps_timing])


if __name__ == '__main__':
    main()
