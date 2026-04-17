"""Phase 63: Edge Adjacency Subsampling Ablation at N=5000

Phase 62 found DELTA's test MRR gap over DistMult collapsed from +0.076 (N=2000)
to +0.016 (N=5000). Two competing explanations:
  A) DELTA's inductive bias genuinely plateaus at larger scale
  B) Aggressive E_adj subsampling (23.8% retention) crippled DELTA's mechanism

Key evidence: At N=2000, subsampling to 98.6% retention IMPROVED test MRR by
+0.028 (0.3371 vs 0.3088). This regularization effect suggests there's an
optimal retention level — Phase 63 finds it at N=5000.

Design: Run DELTA 1L at N=5000 with 4 E_adj retention levels:
  A) 15M pairs (23.8%) — Phase 62 baseline, skip (reuse result)
  B) 30M pairs (47.6%)
  C) 45M pairs (71.4%)
  D) 63M pairs (100% — full, no subsampling)

DistMult is NOT rerun — it doesn't use E_adj, so Phase 62 result (0.2244) is
the fixed baseline.

Hypothesis: DELTA test MRR at ≥47.6% E_adj retention exceeds the 23.8%
baseline (0.2404) by ≥ 0.02, demonstrating that subsampling — not a genuine
plateau — suppressed DELTA's advantage at N=5000.

Cost analysis (from Phase 62 timing):
  - DELTA 1L at N=5000 with 15M E_adj: 107s/epoch × 200 epochs = 21,400s (5.9hr)
  - Higher E_adj budgets will increase epoch time roughly proportionally
  - Condition B (30M): ~214s/epoch × 200ep = ~42,800s (~11.9hr)
  - Condition C (45M): ~321s/epoch × 200ep = ~64,200s (~17.8hr)
  - Condition D (63M): ~450s/epoch × 200ep = ~90,000s (~25hr)
  - Running B+C+D sequentially: ~55hr total
  - Running just B+C: ~30hr total
  
  COST-SAVING STRATEGY: Use early stopping + reduced epochs.
  Phase 62 showed DELTA peaks at ep125 then declines. We only need to find
  the peak, not train to completion. Strategy:
  - Run 150 epochs max (not 200) — saves 25% compute
  - Eval every 25 epochs
  - If val MRR declines for 2 consecutive evals (50 epochs), stop early
  
  With early stopping (likely ~125ep):
  - Condition B (30M, 125ep): ~214s × 125 = ~26,750s (~7.4hr)
  - Condition C (45M, 125ep): ~321s × 125 = ~40,125s (~11.1hr)
  - Condition D (63M, 125ep): ~450s × 125 = ~56,250s (~15.6hr) — IF it fits in VRAM
  
  Most likely plan: Run B and C (~18.5hr), decide on D based on trend.
  At $1.89/hr: B+C ≈ $35, B+C+D ≈ $65
  At $0.89/hr (cheaper GPU): B+C ≈ $16, B+C+D ≈ $30

VRAM note: Phase 62 used 30.7GB at 15M pairs. 30M may use ~45GB, 45M ~60GB,
63M ~75-80GB. The RTX PRO 6000 Blackwell has 98GB — all should fit, but
condition D is tight. The script includes OOM handling.
"""
import sys, os, gc, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.phase46c_link_prediction import (
    load_lp_data, build_train_graph_tensors,
    train_epoch, evaluate_lp, LinkPredictionModel,
)
from delta.model import DELTAModel
from delta.graph import DeltaGraph
import torch, numpy as np, copy

device = 'cuda'
d_node, d_edge = 64, 32

# ── Configuration ──
# Conditions to run. Each is an E_adj budget (number of pairs).
# Condition A (15M) is the Phase 62 baseline — we already have its result.
CONDITIONS = {
    'B': 30_000_000,   # 47.6%
    'C': 45_000_000,   # 71.4%
    'D': 63_100_000,   # 100% (full — slightly over actual count to ensure no subsample)
}

# Phase 62 baseline (condition A) — no need to rerun
PHASE62_BASELINE = {
    'budget': 15_000_000,
    'retention': 0.238,
    'test_MRR': 0.2404,
    'test_H1': 0.1397,
    'test_H10': 0.4566,
    'peak_val': 0.2420,
    'best_ep': 125,
    'time': 21688,
}

# Training config — same as Phase 62
MAX_EPOCHS = 150       # Reduced from 200 — Phase 62 peaked at ep125
EVAL_EVERY = 25
PATIENCE = 2           # Stop if val MRR declines for 2 consecutive evals (50ep)
BS = 4096
LR = 0.003

# DistMult baseline from Phase 62 (doesn't use E_adj, no rerun needed)
DM_TEST_MRR = 0.2244


def subsample_edge_adj(edge_adj, budget):
    """Uniformly subsample edge adjacency pairs to a budget."""
    n_pairs = edge_adj.shape[1]
    if n_pairs <= budget:
        return edge_adj, n_pairs, 1.0
    perm = torch.randperm(n_pairs, device=edge_adj.device)[:budget]
    return edge_adj[:, perm], budget, budget / n_pairs


def build_edge_adj(N, E_train, ei):
    """Build full edge adjacency and return (full_adj, n_pairs, build_time)."""
    t0 = time.time()
    with torch.no_grad():
        tmp_graph = DeltaGraph(
            node_features=torch.zeros(N, d_node, device=device),
            edge_features=torch.zeros(E_train, d_edge, device=device),
            edge_index=ei.to(device),
        )
        tmp_graph.build_edge_adjacency()
        full_edge_adj = tmp_graph._edge_adj_cache[1]
        del tmp_graph
        torch.cuda.empty_cache()
    return full_edge_adj, full_edge_adj.shape[1], time.time() - t0


def run_delta_1layer(data, ei, et, cached_edge_adj, label=''):
    """Run 1-layer DELTA with early stopping. Returns best-val test metrics."""
    enc = DELTAModel(d_node=d_node, d_edge=d_edge,
                     num_layers=1, num_heads=4, init_temp=1.0)
    model = LinkPredictionModel(enc, data['num_entities'],
                                data['num_relations'], d_node, d_edge).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()
    best_val = {'MRR': 0}
    best_state = None
    best_ep = 0
    decline_count = 0
    last_val_mrr = 0.0

    for ep in range(1, MAX_EPOCHS + 1):
        loss = train_epoch(model, data['train'], ei, et, opt, device,
                           batch_size=BS, cached_edge_adj=cached_edge_adj)
        if ep == 1:
            t1 = time.time() - t0
            est_total = t1 * MAX_EPOCHS
            est_hours = est_total / 3600
            print(f'  [{label}] First epoch: {t1:.1f}s. '
                  f'Estimated total ({MAX_EPOCHS}ep): {est_total:.0f}s ({est_hours:.1f}hr)')
            sys.stdout.flush()

        if ep % EVAL_EVERY == 0 or ep == MAX_EPOCHS:
            val = evaluate_lp(model, data['val'], ei, et,
                              data['hr_to_tails'], data['rt_to_heads'], device,
                              cached_edge_adj=cached_edge_adj)
            elapsed = time.time() - t0
            print(f'  [{label}] Ep {ep:4d}  loss={loss:.4f}  '
                  f'MRR={val["MRR"]:.4f}  H@1={val["Hits@1"]:.4f}  '
                  f'H@10={val["Hits@10"]:.4f}  [{elapsed:.0f}s]')
            sys.stdout.flush()

            if val['MRR'] > best_val['MRR']:
                best_val = val.copy()
                best_val['best_epoch'] = ep
                best_ep = ep
                best_state = copy.deepcopy(model.state_dict())
                decline_count = 0
            else:
                if ep > EVAL_EVERY and val['MRR'] < last_val_mrr:
                    decline_count += 1
                    if decline_count >= PATIENCE:
                        print(f'  [{label}] Early stopping at ep {ep} '
                              f'(val MRR declined {decline_count} consecutive evals)')
                        sys.stdout.flush()
                        break
            last_val_mrr = val['MRR']

    # Test eval on BEST VALIDATION checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate_lp(model, data['test'], ei, et,
                       data['hr_to_tails'], data['rt_to_heads'], device,
                       cached_edge_adj=cached_edge_adj)
    elapsed = time.time() - t0
    del model, opt, best_state
    gc.collect(); torch.cuda.empty_cache()
    return {'val_best': best_val, 'test_at_best_val': test,
            'time': elapsed, 'best_ep': best_ep}


# ── Main ──
print()
print('=' * 70)
print('  PHASE 63: Edge Adjacency Subsampling Ablation at N=5000')
print('=' * 70)
sys.stdout.flush()

total_t0 = time.time()
results = {}

# ── Load data (once) ──
torch.manual_seed(42); np.random.seed(42)
data = load_lp_data('fb15k-237', max_entities=5000)
N = data['num_entities']
R = data['num_relations']
E_train = data['train'].shape[1]
triples_per_ent = E_train / N
print(f'  {N} entities, {R} relations')
print(f'  {E_train} train / {data["val"].shape[1]} val / {data["test"].shape[1]} test')
print(f'  Triples/entity: {triples_per_ent:.1f}')
sys.stdout.flush()

ei, et = build_train_graph_tensors(data['train'])

# ── Build full edge adjacency (once) ──
print(f'\n  Building full edge adjacency for N=5000...')
sys.stdout.flush()
full_edge_adj, n_full, t_build = build_edge_adj(N, E_train, ei)
print(f'  Full edge adjacency: {n_full:,} pairs ({t_build:.1f}s)')
sys.stdout.flush()

# ── Reference: Phase 62 baseline ──
print(f'\n  Phase 62 Baseline (Condition A):')
print(f'    Budget: 15M pairs ({PHASE62_BASELINE["retention"]:.1%} retention)')
print(f'    test_MRR={PHASE62_BASELINE["test_MRR"]:.4f}  '
      f'test_H@1={PHASE62_BASELINE["test_H1"]:.4f}  '
      f'test_H@10={PHASE62_BASELINE["test_H10"]:.4f}  '
      f'best_ep={PHASE62_BASELINE["best_ep"]}  {PHASE62_BASELINE["time"]:.0f}s')
print(f'  DistMult baseline: test_MRR={DM_TEST_MRR:.4f}')
sys.stdout.flush()

# ── Run each condition ──
for cond_name, budget in CONDITIONS.items():
    print(f'\n{"="*70}')
    retention = min(budget / n_full, 1.0)
    print(f'  Condition {cond_name}: E_adj budget = {budget:,} '
          f'({retention:.1%} of {n_full:,})')
    print(f'{"="*70}')
    sys.stdout.flush()

    # Subsample
    sub_adj, n_sub, frac = subsample_edge_adj(full_edge_adj, budget)
    actual_retention = n_sub / n_full
    if frac < 1.0:
        print(f'  Subsampled to {n_sub:,} pairs ({actual_retention:.1%})')
    else:
        print(f'  Using full adjacency ({n_sub:,} pairs, no subsampling)')
    sys.stdout.flush()

    # Run DELTA
    label = f'1L_{cond_name}_{n_sub//1_000_000}M'
    torch.manual_seed(42); np.random.seed(42)
    try:
        result = run_delta_1layer(data, ei, et, sub_adj, label=label)
        results[cond_name] = {
            'budget': budget,
            'actual_pairs': n_sub,
            'retention': actual_retention,
            'val_best_MRR': result['val_best']['MRR'],
            'best_ep': result['best_ep'],
            'test_MRR': result['test_at_best_val']['MRR'],
            'test_H1': result['test_at_best_val']['Hits@1'],
            'test_H10': result['test_at_best_val']['Hits@10'],
            'time': result['time'],
        }
        r = results[cond_name]
        gap = r['test_MRR'] - DM_TEST_MRR
        print(f'\n  Condition {cond_name} done: test_MRR={r["test_MRR"]:.4f} '
              f'(gap vs DM: {gap:+.4f}), best_ep={r["best_ep"]}, '
              f'{r["time"]:.0f}s')
        sys.stdout.flush()
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f'\n  Condition {cond_name} FAILED (OOM): {e}')
        results[cond_name] = {'budget': budget, 'error': str(e)}
        sys.stdout.flush()
    finally:
        del sub_adj
        gc.collect(); torch.cuda.empty_cache()

# Clean up full adjacency
del full_edge_adj
gc.collect(); torch.cuda.empty_cache()

total_time = time.time() - total_t0

# ── Summary ──
print(f'\n{"="*70}')
print(f'  PHASE 63 SUMMARY')
print(f'{"="*70}')

print(f'\n--- Subsampling Ablation Results ---')
print(f'{"Condition":<12} {"Budget":<12} {"Retention":<12} {"peak_val":<10} '
      f'{"best_ep":<10} {"test_MRR":<10} {"test_H@1":<10} {"test_H@10":<10} '
      f'{"gap_vs_DM":<12} {"Time":<10}')
print('-' * 108)

# Print baseline A first
A = PHASE62_BASELINE
gap_a = A['test_MRR'] - DM_TEST_MRR
print(f'{"A (P62)":<12} {"15M":<12} {"23.8%":<12} {A["peak_val"]:<10.4f} '
      f'{A["best_ep"]:<10} {A["test_MRR"]:<10.4f} {A["test_H1"]:<10.4f} '
      f'{A["test_H10"]:<10.4f} {gap_a:<+12.4f} {A["time"]:<10.0f}')

# Print each condition
for cond_name in ['B', 'C', 'D']:
    if cond_name not in results:
        continue
    r = results[cond_name]
    if 'error' in r:
        print(f'{cond_name:<12} {r["budget"]//1_000_000}M{"":>8} OOM: {r["error"][:60]}')
        continue
    gap = r['test_MRR'] - DM_TEST_MRR
    budget_str = f'{r["actual_pairs"]//1_000_000}M'
    ret_str = f'{r["retention"]:.1%}'
    print(f'{cond_name:<12} {budget_str:<12} {ret_str:<12} '
          f'{r["val_best_MRR"]:<10.4f} {r["best_ep"]:<10} '
          f'{r["test_MRR"]:<10.4f} {r["test_H1"]:<10.4f} '
          f'{r["test_H10"]:<10.4f} {gap:<+12.4f} {r["time"]:<10.0f}')

print(f'\n  DistMult baseline (Phase 62): test_MRR={DM_TEST_MRR:.4f}')

# Hypothesis evaluation
if results:
    best_cond = max(
        [(k, v) for k, v in results.items() if 'test_MRR' in v],
        key=lambda x: x[1]['test_MRR'],
        default=(None, None)
    )
    if best_cond[0] is not None:
        best_mrr = best_cond[1]['test_MRR']
        delta_vs_baseline = best_mrr - A['test_MRR']
        print(f'\n  Best condition: {best_cond[0]} '
              f'(test_MRR={best_mrr:.4f}, Δ vs A: {delta_vs_baseline:+.4f})')
        if delta_vs_baseline >= 0.02:
            print(f'  Hypothesis: CONFIRMED — higher retention improves test MRR by ≥0.02')
        elif delta_vs_baseline > 0:
            print(f'  Hypothesis: PARTIAL — improvement but below 0.02 threshold')
        else:
            print(f'  Hypothesis: REJECTED — higher retention does not improve test MRR')
        gap_best = best_mrr - DM_TEST_MRR
        print(f'  Best gap vs DistMult: {gap_best:+.4f}')

print(f'\nTotal wall time: {total_time:.0f}s ({total_time/3600:.1f}hr)')
print('Done.')
sys.stdout.flush()
