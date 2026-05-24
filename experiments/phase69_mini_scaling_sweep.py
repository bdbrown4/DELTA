"""Phase 69: Mini-Architecture Scaling Sweep — Empirical Scaling Laws for DELTA and Brain

Motivation
----------
Phase 64 validated topk=128 sparse attention at N=5000 (DELTA 1L).
Phase 65 deferred Brain at N=5000 (816s/epoch × 75ep ≈ 34hr, cost-prohibitive).
The open question: does sparse attention actually deliver LINEAR scaling with N?
And where does the BrainConstructor's O(N²) bottleneck become the dominant cost?

Approach
--------
Instead of running one large, expensive experiment, we shrink the architecture
(d_node=32, d_edge=16 — half Phase 64/65 dimensions) and sweep N linearly:
  N ∈ {500, 1000, 2000, 3000, 5000}

This gives 5 data points per architecture from which we fit a power-law curve:

    time(N) = a · N^b

If b ≈ 1.0 for DELTA (with topk) → sparse attention is genuinely linear.
If b ≈ 2.0 for Brain constructor → constructor is the bottleneck at large N.
The fitted curve then extrapolates to N=10K, 20K, 50K with quantified uncertainty.

Two architecture modes:
  mini_delta  — DELTAModel, 1 layer, 4 heads, topk=128
  mini_brain  — BrainEncoder, 1 bootstrap + 1 delta layer, 4 heads, topk=128,
                density=0.001 (fixed across N to expose the N² constructor cost)

Training: 30 epochs per point, eval every 10 (3 evals). No early stopping —
we want consistent timing across all N values, not peak MRR.

Measurements per (N, arch) point:
  adj_build_s     — E_adj construction time
  ep1_s           — epoch 1 wall time (cold start overhead)
  mean_ep_s       — mean of epochs 2–10 (steady-state)
  peak_vram_mb    — peak GPU memory allocated (MB)
  e_adj_pairs     — number of edge-adjacency pairs
  val_mrr         — val MRR at epoch 30 (approximate quality signal)
  brain_edges     — constructed edges (brain only; 0 for DELTA)

Scaling law output:
  Prints log-log regression coefficients and an extrapolation table to N=50K.

Cost estimate (mini arch reduces compute ~4× vs Phase 64/65):
  DELTA sweep (5 points × 30ep): ~2–3hr at $1.89/hr ≈ $5
  Brain sweep (5 points × 30ep): ~3–6hr ≈ $11
  Total: ~$16 worst case

Usage:
  # Full sweep (DELTA + Brain, all N)
  python experiments/phase69_mini_scaling_sweep.py

  # DELTA-only sweep (quick — no Brain overhead)
  python experiments/phase69_mini_scaling_sweep.py --delta-only

  # Custom N subset for smoke test
  python experiments/phase69_mini_scaling_sweep.py --n-values 500 1000 --delta-only

  # Brain-only at one N for spot check
  python experiments/phase69_mini_scaling_sweep.py --brain-only --n-values 2000
"""

import sys, os, gc, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from experiments.phase46c_link_prediction import (
    load_lp_data, build_train_graph_tensors,
    train_epoch, evaluate_lp, LinkPredictionModel,
)
from delta.model import DELTAModel
from delta.brain import BrainEncoder
from delta.graph import DeltaGraph

# ── Mini architecture — half Phase 64/65 dimensions ──────────────────────────
D_NODE = 32
D_EDGE = 16
TOPK   = 128          # Phase 64's validated sparse-attention budget

# ── Brain constructor config ──────────────────────────────────────────────────
# Fixed density across all N so the N² scaling cost is fully visible.
# At N=5000 this yields ~25K constructed edges, consistent with Phase 65.
BRAIN_DENSITY = 0.001

# ── Training config ───────────────────────────────────────────────────────────
MAX_EPOCHS  = 30
EVAL_EVERY  = 10
BS          = 4096
LR          = 0.003
SEED        = 42

# ── N sweep ───────────────────────────────────────────────────────────────────
DEFAULT_N_VALUES = [500, 1000, 2000, 3000, 5000]

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ═════════════════════════════════════════════════════════════════════════════
# Infrastructure helpers
# ═════════════════════════════════════════════════════════════════════════════

def reset_vram():
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> float:
    if device == 'cuda':
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def build_edge_adj(N: int, E_train: int, ei: torch.Tensor) -> tuple:
    """Build full edge adjacency. Returns (edge_adj, n_pairs, build_time_s)."""
    t0 = time.time()
    with torch.no_grad():
        tmp = DeltaGraph(
            node_features=torch.zeros(N, D_NODE, device=device),
            edge_features=torch.zeros(E_train, D_EDGE, device=device),
            edge_index=ei.to(device),
        )
        tmp.build_edge_adjacency()
        full_adj = tmp._edge_adj_cache[1]
        del tmp
        torch.cuda.empty_cache() if device == 'cuda' else None
    return full_adj, full_adj.shape[1], time.time() - t0


# ═════════════════════════════════════════════════════════════════════════════
# Per-point training loop
# ═════════════════════════════════════════════════════════════════════════════

def run_sweep_point(model, data, ei, et, cached_edge_adj, label: str) -> dict:
    """Train for MAX_EPOCHS and collect timing + quality measurements.

    Returns a dict with keys:
      ep1_s, mean_ep_s, peak_vram_mb, val_mrr, ep_times
    """
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ep_times = []
    best_val_mrr = 0.0

    for ep in range(1, MAX_EPOCHS + 1):
        reset_vram() if ep == 1 else None
        t_ep = time.time()
        train_epoch(model, data['train'], ei, et, opt, device,
                    batch_size=BS, cached_edge_adj=cached_edge_adj)
        ep_times.append(time.time() - t_ep)

        if ep == 1:
            vram = peak_vram_mb()
            print(f'    [{label}] ep1 done in {ep_times[0]:.1f}s, '
                  f'peak VRAM={vram:.0f} MB', flush=True)

        if ep % EVAL_EVERY == 0 or ep == MAX_EPOCHS:
            val = evaluate_lp(model, data['val'], ei, et,
                              data['hr_to_tails'], data['rt_to_heads'],
                              device, cached_edge_adj=cached_edge_adj)
            best_val_mrr = max(best_val_mrr, val['MRR'])
            elapsed = sum(ep_times)
            print(f'    [{label}] Ep{ep:3d}  MRR={val["MRR"]:.4f}  '
                  f'H@10={val["Hits@10"]:.4f}  [{elapsed:.0f}s total]',
                  flush=True)

    mean_ep = float(np.mean(ep_times[1:10])) if len(ep_times) > 1 else ep_times[0]
    return {
        'ep1_s':       ep_times[0],
        'mean_ep_s':   mean_ep,
        'peak_vram_mb': vram,
        'val_mrr':     best_val_mrr,
        'ep_times':    ep_times,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Architecture factories
# ═════════════════════════════════════════════════════════════════════════════

def make_delta_model(num_entities: int, num_relations: int) -> torch.nn.Module:
    enc = DELTAModel(d_node=D_NODE, d_edge=D_EDGE,
                     num_layers=1, num_heads=4,
                     init_temp=1.0, topk_edges=TOPK)
    return LinkPredictionModel(enc, num_entities, num_relations,
                               D_NODE, D_EDGE).to(device)


def make_brain_model(num_entities: int, num_relations: int) -> torch.nn.Module:
    enc = BrainEncoder(
        d_node=D_NODE, d_edge=D_EDGE,
        bootstrap_layers=1, delta_layers=1,
        num_heads=4,
        target_density=BRAIN_DENSITY,
        hybrid=True,
        init_temp=1.0,
        topk_edges=TOPK,
    )
    return LinkPredictionModel(enc, num_entities, num_relations,
                               D_NODE, D_EDGE).to(device)


# ═════════════════════════════════════════════════════════════════════════════
# Log-log scaling law fit + extrapolation
# ═════════════════════════════════════════════════════════════════════════════

def fit_scaling_law(ns: list, times: list) -> dict:
    """Fit time ∝ N^b via log-log linear regression.

    Returns dict with keys: a, b, r2, and extrapolation dict.
    """
    log_n = np.log(ns)
    log_t = np.log(times)
    b, log_a = np.polyfit(log_n, log_t, 1)
    a = np.exp(log_a)

    # R² in log-log space
    predicted = log_a + b * log_n
    ss_res = np.sum((log_t - predicted) ** 2)
    ss_tot = np.sum((log_t - np.mean(log_t)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    extrap = {n: a * (n ** b) for n in [10_000, 20_000, 50_000, 100_000]}
    return {'a': a, 'b': b, 'r2': r2, 'extrapolation': extrap}


def print_scaling_report(arch: str, ns: list, results: list, law: dict):
    print(f'\n{"─" * 60}')
    print(f'  Scaling law — {arch}')
    print(f'  time(N) ≈ {law["a"]:.4f} × N^{law["b"]:.3f}  (R²={law["r2"]:.4f})')
    print(f'  {"N":>8}  {"ep_time(s)":>12}  {"VRAM(MB)":>10}  {"val_MRR":>9}')
    for n, r in zip(ns, results):
        print(f'  {n:>8,}  {r["mean_ep_s"]:>12.1f}  {r["peak_vram_mb"]:>10.0f}  '
              f'{r["val_mrr"]:>9.4f}')
    print(f'\n  Extrapolated epoch time:')
    for n, t in law['extrapolation'].items():
        hrs = t / 3600
        print(f'    N={n:>7,}  →  {t:>8.0f}s/ep  ({hrs:.1f}hr per epoch)')
    print(f'{"─" * 60}\n', flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Main sweep
# ═════════════════════════════════════════════════════════════════════════════

def run_arch_sweep(arch: str, n_values: list) -> list:
    """Run one architecture across all N values. Returns list of result dicts."""
    all_results = []

    for N in n_values:
        print(f'\n{"═" * 60}')
        print(f'  {arch.upper()}  N={N:,}  (d_node={D_NODE}, topk={TOPK})')
        print(f'{"═" * 60}', flush=True)

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # Load data subset
        data = load_lp_data('fb15k-237', max_entities=N)
        actual_N  = data['num_entities']
        ei, et    = build_train_graph_tensors(data['train'])
        E_train   = ei.shape[1]

        print(f'  Entities={actual_N:,}  Edges={E_train:,}', flush=True)

        # Build E_adj
        print(f'  Building E_adj...', flush=True)
        cached_adj, n_pairs, adj_t = build_edge_adj(actual_N, E_train, ei)
        print(f'  E_adj: {n_pairs:,} pairs in {adj_t:.1f}s', flush=True)

        # Create model
        if arch == 'delta':
            model = make_delta_model(data['num_entities'], data['num_relations'])
        else:
            model = make_brain_model(data['num_entities'], data['num_relations'])

        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Model params: {n_params:,}', flush=True)

        # Train + measure
        label = f'{arch}@N={actual_N}'
        timing = run_sweep_point(model, data, ei.to(device), et.to(device),
                                 cached_adj, label)

        # Count brain edges if applicable
        brain_edges = 0
        if arch == 'brain' and hasattr(model.encoder, 'last_n_constructed'):
            brain_edges = model.encoder.last_n_constructed

        result = {
            'arch':          arch,
            'N_requested':   N,
            'N_actual':      actual_N,
            'E_train':       E_train,
            'adj_build_s':   adj_t,
            'e_adj_pairs':   n_pairs,
            'ep1_s':         timing['ep1_s'],
            'mean_ep_s':     timing['mean_ep_s'],
            'peak_vram_mb':  timing['peak_vram_mb'],
            'val_mrr':       timing['val_mrr'],
            'brain_edges':   brain_edges,
            'ep_times':      timing['ep_times'],
        }
        all_results.append(result)
        print(f'\n  ✓ {label}  mean_ep={result["mean_ep_s"]:.1f}s  '
              f'VRAM={result["peak_vram_mb"]:.0f}MB  MRR={result["val_mrr"]:.4f}',
              flush=True)

        # Free before next N
        del model, cached_adj
        reset_vram()

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Phase 69: Mini-arch scaling sweep')
    parser.add_argument('--delta-only', action='store_true',
                        help='Run DELTA sweep only (skip Brain)')
    parser.add_argument('--brain-only', action='store_true',
                        help='Run Brain sweep only (skip DELTA)')
    parser.add_argument('--n-values', nargs='+', type=int,
                        default=DEFAULT_N_VALUES,
                        help=f'N values to sweep (default: {DEFAULT_N_VALUES})')
    args = parser.parse_args()

    archs = []
    if not args.brain_only:
        archs.append('delta')
    if not args.delta_only:
        archs.append('brain')

    print(f'\nPhase 69: Mini-Architecture Scaling Sweep')
    print(f'  Device:        {device}')
    print(f'  Architecture:  d_node={D_NODE}, d_edge={D_EDGE}, topk={TOPK}')
    print(f'  N sweep:       {args.n_values}')
    print(f'  Architectures: {archs}')
    print(f'  Epochs/point:  {MAX_EPOCHS}\n', flush=True)

    all_results = {}
    for arch in archs:
        results = run_arch_sweep(arch, args.n_values)
        all_results[arch] = results

        # Fit scaling law (need ≥2 points)
        ns     = [r['N_actual']  for r in results]
        times  = [r['mean_ep_s'] for r in results]
        if len(ns) >= 2 and all(t > 0 for t in times):
            law = fit_scaling_law(ns, times)
            print_scaling_report(arch, ns, results, law)
            all_results[f'{arch}_scaling_law'] = law

    # Comparison table if both archs ran
    if 'delta' in all_results and 'brain' in all_results:
        print('\n  Overhead ratio  brain / delta  (mean epoch time):')
        d_map = {r['N_actual']: r for r in all_results['delta']}
        b_map = {r['N_actual']: r for r in all_results['brain']}
        print(f'  {"N":>8}  {"delta(s)":>10}  {"brain(s)":>10}  {"ratio":>8}')
        for n in sorted(set(d_map) & set(b_map)):
            dt = d_map[n]['mean_ep_s']
            bt = b_map[n]['mean_ep_s']
            ratio = bt / dt if dt > 0 else float('inf')
            print(f'  {n:>8,}  {dt:>10.1f}  {bt:>10.1f}  {ratio:>8.2f}×')

    # Save JSON results
    out_path = os.path.join(os.path.dirname(__file__),
                            '..', 'delta', 'phase69_output.json')
    with open(out_path, 'w') as f:
        # ep_times lists make JSON bulky — include for analysis but cap at 30 entries
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n  Results saved → {os.path.abspath(out_path)}', flush=True)


if __name__ == '__main__':
    main()
