"""Phase 64: Top-k Sparse Edge-to-Edge Attention at N=5000

Phase 63 proved attention dilution — not E_adj subsampling — is the real
scaling bottleneck. At N=5000, each edge attends to ~3000+ neighbors on average
(63M pairs / ~20K edges). The signal drowns in noise.

Solution: top-k sparse attention. Instead of softmax over ALL neighbor edges,
keep only the k highest-scoring neighbors per target edge. This:
  1. Focuses attention on the most informative structural neighbors
  2. Reduces effective E_adj from millions to O(E * k) ≈ E*128 ≈ 2.5M
  3. Sharpens gradient signal (no attention waste on irrelevant neighbors)

Implementation: EdgeAttention now accepts topk_edges parameter. After computing
raw QK+context scores for all E_adj pairs, a vectorized filter keeps only
the top-k per target edge, then runs softmax on the filtered set.

Design: 3 conditions (+ 1 baseline from Phase 63):
  A) 30M uniform subsample, no topk — Phase 63 Condition B (REUSE)
  B) Full 63M E_adj, topk=128 — sparse attention on complete adjacency
  C) Full 63M E_adj, topk=64  — tighter sparsity
  D) Full 63M E_adj, topk=256 — looser sparsity

Hypothesis: DELTA 1L with topk=128 on full E_adj (Condition B) achieves
test MRR ≥ 0.2471 (matching Phase 63's best, Condition B with 30M random),
demonstrating that score-based sparse attention matches or beats uniform
random subsampling while directly addressing the dilution mechanism.

If B matches/beats A at lower wall time, this validates sparse attention
as the solution to the scaling bottleneck identified in Phases 59-63.

Cost analysis:
  - Full E_adj QK computation: ~78,000s (Phase 63 Condition D), BUT
  - After topk filtering, softmax/scatter on ~2.5M pairs (vs 63M) → much faster
  - The QK + context computation is O(E_adj) regardless of topk
  - Net: similar to Condition D on QK, but faster on aggregation
  - Expected: ~80-90% of Condition D's wall time per epoch
  
  With 150 epochs + early stopping (~125ep):
  - Condition B (63M, topk=128): ~500s/epoch × 125ep ≈ 62,500s (~17hr)
  - Condition C (63M, topk=64):  ~500s/epoch × 125ep ≈ 62,500s (~17hr)  
  - Condition D (63M, topk=256): ~500s/epoch × 125ep ≈ 62,500s (~17hr)
  - All three: ~51hr total. At $1.89/hr: ~$96
  
VRAM note: Full E_adj (63M) used ~75-80GB in Phase 63 Condition D.
topk filtering reduces post-score memory but peak is still at QK computation.
RTX PRO 6000 Blackwell (98GB) should handle all conditions.
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
CONDITIONS = {
    'B': {'budget': None, 'topk': 128, 'label': 'full_E_adj+topk128'},
    'C': {'budget': None, 'topk': 64,  'label': 'full_E_adj+topk64'},
    'D': {'budget': None, 'topk': 256, 'label': 'full_E_adj+topk256'},
}

# Phase 63 Condition B baseline (best result from subsampling ablation)
PHASE63_BASELINE = {
    'budget': 30_000_000,
    'retention': 0.476,
    'topk': None,
    'test_MRR': 0.2471,
    'test_H1': 0.1481,
    'test_H10': 0.4562,
    'peak_val': 0.2499,
    'best_ep': 125,
    'time': 37653,
}

# DistMult baseline from Phase 62
DM_TEST_MRR = 0.2244

# Training config — same as Phase 62/63
MAX_EPOCHS = 150
EVAL_EVERY = 25
PATIENCE = 2
BS = 4096
LR = 0.003


def subsample_edge_adj(edge_adj, budget):
    """Uniformly subsample edge adjacency pairs to a budget."""
    n_pairs = edge_adj.shape[1]
    if budget is None or n_pairs <= budget:
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


def run_delta_1layer(data, ei, et, cached_edge_adj, topk_edges=None, label=''):
    """Run 1-layer DELTA with optional top-k sparse attention and early stopping."""
    enc = DELTAModel(d_node=d_node, d_edge=d_edge,
                     num_layers=1, num_heads=4, init_temp=1.0,
                     topk_edges=topk_edges)
    model = LinkPredictionModel(enc, data['num_entities'],
                                data['num_relations'], d_node, d_edge).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()
    best_val = {'MRR': 0}
    best_state = None
    best_ep = 0
    decline_count = 0
    last_val_mrr = 0.0

    # Track topk stats from first eval
    topk_str = f'topk={topk_edges}' if topk_edges else 'no topk'

    for ep in range(1, MAX_EPOCHS + 1):
        loss = train_epoch(model, data['train'], ei, et, opt, device,
                           batch_size=BS, cached_edge_adj=cached_edge_adj)
        if ep == 1:
            t1 = time.time() - t0
            est_total = t1 * MAX_EPOCHS
            est_hours = est_total / 3600
            print(f'  [{label}] First epoch: {t1:.1f}s ({topk_str}). '
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
print('  PHASE 64: Top-k Sparse Edge-to-Edge Attention at N=5000')
print('=' * 70)
sys.stdout.flush()

total_t0 = time.time()
results = {}

# ── Load data (once) ──
torch.manual_seed(42); np.random.seed(42)
data = load_lp_data('fb15k-237', max_entities=5000)
N = data['num_entities']
E_train = data['train'].shape[1]
ei, et = build_train_graph_tensors(data['train'])
print(f'\n  Data: {N} entities, {data["num_relations"]} relations, '
      f'{E_train} train edges')
sys.stdout.flush()

# ── Build full edge adjacency (once) ──
print('\n  Building full edge adjacency...')
sys.stdout.flush()
full_edge_adj, n_full_pairs, build_time = build_edge_adj(N, E_train, ei)
print(f'  Full E_adj: {n_full_pairs:,} pairs, built in {build_time:.1f}s')
sys.stdout.flush()

# ── Run conditions ──
for cond_name, cfg in CONDITIONS.items():
    print(f'\n{"─" * 60}')
    print(f'  Condition {cond_name}: {cfg["label"]}')
    print(f'{"─" * 60}')
    sys.stdout.flush()

    # Subsample if budget specified
    if cfg['budget'] is not None:
        edge_adj, n_pairs, retention = subsample_edge_adj(full_edge_adj, cfg['budget'])
        print(f'  E_adj: {n_pairs:,} of {n_full_pairs:,} ({retention:.1%} retention)')
    else:
        edge_adj = full_edge_adj
        n_pairs = n_full_pairs
        retention = 1.0
        print(f'  E_adj: {n_pairs:,} pairs (full, no subsampling)')
    sys.stdout.flush()

    try:
        r = run_delta_1layer(data, ei, et, edge_adj,
                             topk_edges=cfg['topk'],
                             label=f'Cond_{cond_name}')
        results[cond_name] = {
            'budget': cfg['budget'],
            'topk': cfg['topk'],
            'retention': retention,
            'n_pairs': n_pairs,
            'test_MRR': r['test_at_best_val']['MRR'],
            'test_H1': r['test_at_best_val']['Hits@1'],
            'test_H10': r['test_at_best_val']['Hits@10'],
            'peak_val': r['val_best']['MRR'],
            'best_ep': r['best_ep'],
            'time': r['time'],
        }
        print(f'\n  Cond {cond_name} DONE: test_MRR={r["test_at_best_val"]["MRR"]:.4f}, '
              f'H@1={r["test_at_best_val"]["Hits@1"]:.4f}, '
              f'H@10={r["test_at_best_val"]["Hits@10"]:.4f}, '
              f'peak_val={r["val_best"]["MRR"]:.4f}, best_ep={r["best_ep"]}, '
              f'time={r["time"]:.0f}s')
        sys.stdout.flush()

    except torch.cuda.OutOfMemoryError:
        print(f'\n  !! Condition {cond_name} OOM — skipping')
        results[cond_name] = {'status': 'OOM', 'topk': cfg['topk']}
        gc.collect(); torch.cuda.empty_cache()
        sys.stdout.flush()

    except Exception as e:
        print(f'\n  !! Condition {cond_name} FAILED: {e}')
        results[cond_name] = {'status': f'ERROR: {e}', 'topk': cfg['topk']}
        gc.collect(); torch.cuda.empty_cache()
        sys.stdout.flush()

total_time = time.time() - total_t0

# ── Summary ──
print(f'\n\n{"=" * 70}')
print(f'  PHASE 64 RESULTS — Top-k Sparse Edge Attention at N=5000')
print(f'{"=" * 70}')
print(f'\n  Total time: {total_time:.0f}s ({total_time/3600:.1f}hr)')
print(f'\n  Phase 63 Baseline (Condition A): 30M subsample, no topk')
print(f'    test_MRR={PHASE63_BASELINE["test_MRR"]:.4f}  '
      f'H@1={PHASE63_BASELINE["test_H1"]:.4f}  '
      f'H@10={PHASE63_BASELINE["test_H10"]:.4f}  '
      f'peak_val={PHASE63_BASELINE["peak_val"]:.4f}  '
      f'best_ep={PHASE63_BASELINE["best_ep"]}  '
      f'time={PHASE63_BASELINE["time"]}s')
print(f'\n  DistMult baseline: test_MRR={DM_TEST_MRR:.4f}')

print(f'\n  {"Cond":<6} {"Budget":<10} {"TopK":<6} {"peak_val":<10} '
      f'{"best_ep":<8} {"test_MRR":<10} {"test_H@1":<10} {"test_H@10":<10} '
      f'{"gap_vs_DM":<10} {"Time":<10}')
print(f'  {"-"*6} {"-"*10} {"-"*6} {"-"*10} {"-"*8} {"-"*10} {"-"*10} {"-"*10} '
      f'{"-"*10} {"-"*10}')

# Print baseline
print(f'  {"A(P63)":<6} {"30M":<10} {"—":<6} '
      f'{PHASE63_BASELINE["peak_val"]:<10.4f} '
      f'{PHASE63_BASELINE["best_ep"]:<8} '
      f'{PHASE63_BASELINE["test_MRR"]:<10.4f} '
      f'{PHASE63_BASELINE["test_H1"]:<10.4f} '
      f'{PHASE63_BASELINE["test_H10"]:<10.4f} '
      f'{PHASE63_BASELINE["test_MRR"] - DM_TEST_MRR:<+10.4f} '
      f'{PHASE63_BASELINE["time"]:<10.0f}')

for cond_name in CONDITIONS:
    r = results.get(cond_name, {})
    if 'test_MRR' in r:
        budget_str = f'{r["n_pairs"]//1_000_000}M' if r['budget'] is None else f'{r["budget"]//1_000_000}M'
        print(f'  {cond_name:<6} {budget_str:<10} {r["topk"]:<6} '
              f'{r["peak_val"]:<10.4f} '
              f'{r["best_ep"]:<8} '
              f'{r["test_MRR"]:<10.4f} '
              f'{r["test_H1"]:<10.4f} '
              f'{r["test_H10"]:<10.4f} '
              f'{r["test_MRR"] - DM_TEST_MRR:<+10.4f} '
              f'{r["time"]:<10.0f}')
    else:
        print(f'  {cond_name:<6} {"—":<10} {CONDITIONS[cond_name]["topk"]:<6} '
              f'{"—":<10} {"—":<8} {r.get("status", "SKIP"):<10}')

# vs baseline deltas
print(f'\n  vs. Phase 63 Baseline (A, 30M uniform subsample):')
for cond_name in CONDITIONS:
    r = results.get(cond_name, {})
    if 'test_MRR' in r:
        delta_mrr = r['test_MRR'] - PHASE63_BASELINE['test_MRR']
        speedup = PHASE63_BASELINE['time'] / r['time'] if r['time'] > 0 else 0
        print(f'    {cond_name} (topk={r["topk"]}): '
              f'Δ test_MRR={delta_mrr:+.4f}, '
              f'speedup={speedup:.2f}x')

print(f'\n  Phase 64 complete.')
sys.stdout.flush()
