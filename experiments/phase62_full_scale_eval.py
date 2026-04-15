"""Phase 62: Scaling DELTA to N=5000 — Test MRR Generalization Advantage

Phase 61b CONFIRMED DELTA's generalization advantage at N=2000:
  - DistMult: val 0.3185 but test 0.2329 (catastrophic overfitting)
  - DELTA 1L: val 0.3357, test 0.3088 (+0.076 test MRR gap)

The lesson: val MRR is misleading — DistMult overfits. DELTA's real advantage
shows up in *test* MRR. Phase 62 tests whether this generalisation advantage
persists at N=5000, giving a 3-point scaling curve (N=500, N=2000, N=5000).

Two-step design:
  Step 1 — SANITY CHECK: Run DELTA 1L at N=2000 with subsampled edge adjacency
           (capped at 15M pairs). Compare to existing test MRR 0.3088 (full
           edge adjacency). Pass if subsampled MRR >= ref - 0.01 (asymmetric:
           only fail if subsampling hurts, improvements are fine).
  Step 2 — N=5000: DistMult (2000 ep) + DELTA 1L (200 ep) with subsampled
           edge adjacency. Both evaluated at best-validated checkpoint.

Hypothesis: At N=5000, 1-layer DELTA achieves test MRR exceeding DistMult by
≥ 0.04, evaluated at each model's best validation checkpoint. Both models are
given sufficient compute to converge (DistMult: 2000 epochs at ~$0.01/epoch;
DELTA: 200 epochs at ~$0.50/epoch — epoch counts reflect convergence profiles,
not compute budgets).

Epoch asymmetry justification: DistMult converges by ep100-200 but benefits
from extended training for peak val checkpoint selection. DELTA converges by
ep150-175. Both are evaluated at their respective best validation epoch,
making the comparison fair regardless of total epoch count.

N=14,541 (full FB15k-237) is deferred to Phase 63, contingent on N=5000
confirming the scaling trend.
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

# Edge adjacency budget: cap to keep attention within GPU memory.
# At N=2000, full E_adj is 15.2M pairs. Budget 15M → comparable or subsampled.
# Phase 30 showed subsampling strategies within ±0.2% at N=500.
# Step 1 validates this assumption at N=2000 before relying on it at N=5000.
MAX_EDGE_ADJ_PAIRS = 15_000_000

# Hyperparameters: same as Phase 61 N=2000 for comparability
DM_EPOCHS = 2000   # DistMult: fast (~seconds/epoch), give full convergence window
DELTA_EPOCHS = 200  # DELTA: converges by ep150-175, cost ~30s/epoch at N=2000
BS = 4096
LR = 0.003


def subsample_edge_adj(edge_adj, budget):
    """Uniformly subsample edge adjacency pairs to a budget."""
    n_pairs = edge_adj.shape[1]
    if n_pairs <= budget:
        return edge_adj, n_pairs, 1.0
    perm = torch.randperm(n_pairs, device=edge_adj.device)[:budget]
    return edge_adj[:, perm], budget, budget / n_pairs


def build_edge_adj(N, E_train, ei):
    """Build edge adjacency and return (full_adj, build_time)."""
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
    return full_edge_adj, time.time() - t0


def run_distmult(data, ei, et, bs, lr, num_epochs, label=''):
    """Run DistMult (no GNN) baseline. Returns best-val checkpoint test metrics."""
    model = LinkPredictionModel(None, data['num_entities'],
                                data['num_relations'], d_node, d_edge).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    best_val = {'MRR': 0}
    best_state = None
    eval_every = 100

    for ep in range(1, num_epochs + 1):
        loss = train_epoch(model, data['train'], ei, et,
                           opt, device, batch_size=bs)
        if ep % eval_every == 0 or ep == num_epochs:
            val = evaluate_lp(model, data['val'], ei, et,
                              data['hr_to_tails'], data['rt_to_heads'], device)
            elapsed = time.time() - t0
            print(f'  [{label}] Ep {ep:4d}  loss={loss:.4f}  '
                  f'MRR={val["MRR"]:.4f}  H@1={val["Hits@1"]:.4f}  '
                  f'H@10={val["Hits@10"]:.4f}  [{elapsed:.0f}s]')
            sys.stdout.flush()
            if val['MRR'] > best_val['MRR']:
                best_val = val.copy()
                best_val['best_epoch'] = ep
                best_state = copy.deepcopy(model.state_dict())

    # Test eval on BEST VALIDATION checkpoint (not final epoch)
    # Save final-epoch state first for comparison
    final_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate_lp(model, data['test'], ei, et,
                       data['hr_to_tails'], data['rt_to_heads'], device)
    # Also eval at final epoch (shows overfitting extent)
    model.load_state_dict(final_state)
    final_test = evaluate_lp(model, data['test'], ei, et,
                             data['hr_to_tails'], data['rt_to_heads'], device)
    elapsed = time.time() - t0
    del model, opt, best_state, final_state
    gc.collect(); torch.cuda.empty_cache()
    return {'val_best': best_val, 'test_at_best_val': test,
            'test_final': final_test, 'time': elapsed, 'params': params}


def run_delta_1layer(data, ei, et, cached_edge_adj, bs, lr, num_epochs, label=''):
    """Run 1-layer DELTA. Returns best-val checkpoint test metrics."""
    enc = DELTAModel(d_node=d_node, d_edge=d_edge,
                     num_layers=1, num_heads=4, init_temp=1.0)
    model = LinkPredictionModel(enc, data['num_entities'],
                                data['num_relations'], d_node, d_edge).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    best_val = {'MRR': 0}
    best_state = None
    eval_every = 25

    for ep in range(1, num_epochs + 1):
        loss = train_epoch(model, data['train'], ei, et, opt, device,
                           batch_size=bs, cached_edge_adj=cached_edge_adj)
        if ep == 1:
            t1 = time.time() - t0
            est_total = t1 * num_epochs
            est_hours = est_total / 3600
            print(f'  [{label}] First epoch: {t1:.1f}s. '
                  f'Estimated total: {est_total:.0f}s ({est_hours:.1f}hr)')
            sys.stdout.flush()
            if est_hours > 6.0:
                new_epochs = max(50, int(5 * 3600 / t1))
                print(f'  [{label}] Reducing from {num_epochs} to {new_epochs} '
                      f'epochs (budget: ~5hr)')
                sys.stdout.flush()
                num_epochs = new_epochs
                eval_every = max(10, num_epochs // 10)

        if ep % eval_every == 0 or ep == num_epochs:
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
                best_state = copy.deepcopy(model.state_dict())

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
            'time': elapsed, 'params': params}


# ── Main ──
print()
print('=' * 70)
print('  PHASE 62: Scaling DELTA to N=5000 — Test MRR Generalization')
print('=' * 70)
sys.stdout.flush()

all_results = {}
total_t0 = time.time()


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: SANITY CHECK — Subsampled E_adj at N=2000
# ═══════════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f'  STEP 1: SANITY CHECK — Subsampled edge adjacency at N=2000')
print(f'  Reference: DELTA 1L N=2000 full E_adj test MRR = 0.3088')
print(f'  Pass criterion: subsampled test MRR >= 0.3088 - 0.01 (asymmetric)')
print(f'{"="*70}')
sys.stdout.flush()

torch.manual_seed(42); np.random.seed(42)
data_2k = load_lp_data('fb15k-237', max_entities=2000)
N_2k = data_2k['num_entities']
E_train_2k = data_2k['train'].shape[1]
triples_per_ent_2k = E_train_2k / N_2k
print(f'  {N_2k} entities, {E_train_2k} train triples')
print(f'  Triples/entity: {triples_per_ent_2k:.1f}')
sys.stdout.flush()

ei_2k, et_2k = build_train_graph_tensors(data_2k['train'])

# Build full edge adjacency, then subsample
full_adj_2k, t_build = build_edge_adj(N_2k, E_train_2k, ei_2k)
n_full_2k = full_adj_2k.shape[1]
print(f'  Full edge adjacency: {n_full_2k:,} pairs ({t_build:.1f}s)')
sub_adj_2k, n_sub_2k, frac_2k = subsample_edge_adj(full_adj_2k, MAX_EDGE_ADJ_PAIRS)
if frac_2k < 1.0:
    print(f'  Subsampled to {n_sub_2k:,} pairs ({frac_2k:.1%} of full)')
else:
    print(f'  No subsampling needed ({n_full_2k:,} <= {MAX_EDGE_ADJ_PAIRS:,})')
del full_adj_2k; torch.cuda.empty_cache()
sys.stdout.flush()

# Run DELTA 1L with subsampled E_adj
print(f'\n  --- 1-layer DELTA (subsampled E_adj), {DELTA_EPOCHS} epochs ---')
sys.stdout.flush()
torch.manual_seed(42); np.random.seed(42)
sanity = run_delta_1layer(data_2k, ei_2k, et_2k, sub_adj_2k,
                          BS, LR, DELTA_EPOCHS, label='Sanity_N=2000')
all_results['Sanity_1L_N=2000'] = sanity
sanity_mrr = sanity['test_at_best_val']['MRR']
ref_mrr = 0.3088
delta_from_ref = sanity_mrr - ref_mrr
# Asymmetric gate: only fail if subsampling HURTS (drops MRR below reference - 0.01).
# Improvement is fine — subsampling may act as regularization.
sanity_pass = delta_from_ref > -0.01  # pass if subsampled >= ref - 0.01
print(f'\n  SANITY CHECK RESULT:')
print(f'    Subsampled test MRR: {sanity_mrr:.4f}')
print(f'    Reference (full E_adj): {ref_mrr:.4f}')
print(f'    Delta: {delta_from_ref:+.4f}')
print(f'    Pass (sub >= ref - 0.01): {"YES" if sanity_pass else "NO — ABORTING N=5000"}')
sys.stdout.flush()

del sub_adj_2k, data_2k, ei_2k, et_2k
gc.collect(); torch.cuda.empty_cache()

if not sanity_pass:
    print(f'\n  SANITY CHECK FAILED. Subsampling degrades test MRR by {abs(delta_from_ref):.4f}.')
    print(f'  Cannot trust subsampled results at N=5000. Aborting.')
    total_time = time.time() - total_t0
    print(f'\nTotal wall time: {total_time:.0f}s ({total_time/3600:.1f}hr)')
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: N=5000 — DistMult + DELTA 1L
# ═══════════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f'  STEP 2: N=5000 — DistMult vs 1-Layer DELTA')
print(f'{"="*70}')
sys.stdout.flush()

torch.manual_seed(42); np.random.seed(42)
data_5k = load_lp_data('fb15k-237', max_entities=5000)
N_5k = data_5k['num_entities']
R_5k = data_5k['num_relations']
E_train_5k = data_5k['train'].shape[1]
E_val_5k = data_5k['val'].shape[1]
E_test_5k = data_5k['test'].shape[1]
triples_per_ent_5k = E_train_5k / N_5k
print(f'  {N_5k} entities, {R_5k} relations')
print(f'  {E_train_5k} train / {E_val_5k} val / {E_test_5k} test triples')
print(f'  Triples/entity: {triples_per_ent_5k:.1f}')
steps_per_epoch = (E_train_5k + BS - 1) // BS
print(f'  {steps_per_epoch} steps/epoch (bs={BS})')
sys.stdout.flush()

ei_5k, et_5k = build_train_graph_tensors(data_5k['train'])

# ── DistMult at N=5000 ──
print(f'\n  --- DistMult (no GNN), {DM_EPOCHS} epochs ---')
sys.stdout.flush()
torch.manual_seed(42); np.random.seed(42)
dm_5k = run_distmult(data_5k, ei_5k, et_5k, BS, LR, DM_EPOCHS, label='DM_N=5000')
all_results['DM_N=5000'] = dm_5k
print(f'  DM N=5000: peak val={dm_5k["val_best"]["MRR"]:.4f} '
      f'(ep{dm_5k["val_best"].get("best_epoch","?")}), '
      f'test@best_val={dm_5k["test_at_best_val"]["MRR"]:.4f}, '
      f'{dm_5k["time"]:.0f}s')
sys.stdout.flush()

# ── Build edge adjacency for N=5000 ──
print(f'\n  Building edge adjacency for N=5000...')
sys.stdout.flush()
try:
    full_adj_5k, t_build = build_edge_adj(N_5k, E_train_5k, ei_5k)
    n_full_5k = full_adj_5k.shape[1]
    print(f'  Full edge adjacency: {n_full_5k:,} pairs ({t_build:.1f}s)')
    sub_adj_5k, n_sub_5k, frac_5k = subsample_edge_adj(full_adj_5k, MAX_EDGE_ADJ_PAIRS)
    if frac_5k < 1.0:
        print(f'  Subsampled to {n_sub_5k:,} pairs ({frac_5k:.1%} of full)')
    else:
        print(f'  No subsampling needed ({n_full_5k:,} <= {MAX_EDGE_ADJ_PAIRS:,})')
    del full_adj_5k; torch.cuda.empty_cache()
    sys.stdout.flush()
    edge_adj_ok = True
except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
    print(f'  Edge adjacency build FAILED: {e}')
    print(f'  Skipping DELTA at N=5000')
    sys.stdout.flush()
    edge_adj_ok = False
    sub_adj_5k = None

# ── 1-layer DELTA at N=5000 ──
if edge_adj_ok:
    print(f'\n  --- 1-layer DELTA, up to {DELTA_EPOCHS} epochs ---')
    sys.stdout.flush()
    torch.manual_seed(42); np.random.seed(42)
    try:
        d1_5k = run_delta_1layer(data_5k, ei_5k, et_5k, sub_adj_5k,
                                 BS, LR, DELTA_EPOCHS, label='1L_N=5000')
        all_results['1L_N=5000'] = d1_5k
        print(f'  1L N=5000: peak val={d1_5k["val_best"]["MRR"]:.4f} '
              f'(ep{d1_5k["val_best"].get("best_epoch","?")}), '
              f'test@best_val={d1_5k["test_at_best_val"]["MRR"]:.4f}, '
              f'{d1_5k["time"]:.0f}s')
        sys.stdout.flush()
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f'  DELTA training FAILED (OOM): {e}')
        sys.stdout.flush()
    finally:
        del sub_adj_5k
        gc.collect(); torch.cuda.empty_cache()
else:
    if sub_adj_5k is not None:
        del sub_adj_5k
    gc.collect(); torch.cuda.empty_cache()

del data_5k, ei_5k, et_5k
gc.collect(); torch.cuda.empty_cache()

total_time = time.time() - total_t0


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f'\n{"="*70}')
print(f'PHASE 62 SUMMARY')
print(f'{"="*70}')

# Scaling curve with triples/entity
print(f'\n--- Scaling Curve: Triples/Entity ---')
print(f'  N=500:  triples/ent ≈ 37.4  (Phase 61)')
print(f'  N=2000: triples/ent ≈ {triples_per_ent_2k:.1f}')
print(f'  N=5000: triples/ent ≈ {triples_per_ent_5k:.1f}')

print(f'\n--- Results Table ---')
print(f'{"Model":<25} {"N":>6} {"trip/ent":>9} {"peak_val":>9} {"best_ep":>8} '
      f'{"test_MRR":>10} {"test_H@1":>9} {"test_H@10":>10} {"Time":>7}')
print(f'{"-"*98}')

# Phase 61 references (N=500)
print(f'{"DM@500 (P61)":<25} {"500":>6} {"37.4":>9} {"0.4779":>9} {"—":>8} '
      f'{"0.4778":>10} {"0.3419":>9} {"0.7567":>10} {"—":>7}')
print(f'{"1L@500 (P61)":<25} {"500":>6} {"37.4":>9} {"0.4818":>9} {"—":>8} '
      f'{"0.4818":>10} {"0.3540":>9} {"0.7359":>10} {"—":>7}')
print(f'{"-"*98}')

# Phase 61/61b references (N=2000)
print(f'{"DM@2000 (P61b)":<25} {"2000":>6} {"—":>9} {"0.3185":>9} {"100":>8} '
      f'{"0.2329":>10} {"0.1159":>9} {"0.5031":>10} {"261s":>7}')
print(f'{"1L@2000 (P61)":<25} {"2000":>6} {"—":>9} {"0.3357":>9} {"175":>8} '
      f'{"0.3088":>10} {"0.1787":>9} {"0.5963":>10} {"5896s":>7}')

# Sanity check
if 'Sanity_1L_N=2000' in all_results:
    s = all_results['Sanity_1L_N=2000']
    sv = s['val_best']
    st = s['test_at_best_val']
    print(f'{"1L@2000 (sub E_adj)":<25} {"2000":>6} '
          f'{triples_per_ent_2k:>9.1f} {sv["MRR"]:>9.4f} '
          f'{sv.get("best_epoch","?"):>8} {st["MRR"]:>10.4f} '
          f'{st["Hits@1"]:>9.4f} {st["Hits@10"]:>10.4f} {s["time"]:>6.0f}s')
print(f'{"-"*98}')

# N=5000 results
for key in ['DM_N=5000', '1L_N=5000']:
    if key in all_results:
        r = all_results[key]
        v = r['val_best']
        t = r['test_at_best_val']
        print(f'{key:<25} {"5000":>6} {triples_per_ent_5k:>9.1f} '
              f'{v["MRR"]:>9.4f} {v.get("best_epoch","?"):>8} '
              f'{t["MRR"]:>10.4f} {t["Hits@1"]:>9.4f} '
              f'{t["Hits@10"]:>10.4f} {r["time"]:>6.0f}s')

# ── DELTA vs DistMult gaps (TEST MRR at best val checkpoint) ──
print(f'\n--- DELTA vs DistMult Gaps (TEST MRR at best val checkpoint) ---')
print(f'  N=500  (P61):  DM test=0.4778, 1L test=0.4818, gap=+0.0040')
print(f'  N=2000 (P61b): DM test=0.2329, 1L test=0.3088, gap=+0.0759')

if 'DM_N=5000' in all_results and '1L_N=5000' in all_results:
    dm_t = all_results['DM_N=5000']['test_at_best_val']['MRR']
    d1_t = all_results['1L_N=5000']['test_at_best_val']['MRR']
    gap = d1_t - dm_t
    print(f'  N=5000 (P62):  DM test={dm_t:.4f}, 1L test={d1_t:.4f}, gap={gap:+.4f}')
    print(f'\n--- Scaling Trend ---')
    print(f'  N=500:  gap=+0.004  (dense data, both models strong)')
    print(f'  N=2000: gap=+0.076  (sparser data, DM overfits)')
    print(f'  N=5000: gap={gap:+.3f}  {"(trend continues)" if gap > 0.04 else "(trend weakens)" if gap > 0 else "(advantage lost)"}')
elif 'DM_N=5000' in all_results:
    dm_t = all_results['DM_N=5000']['test_at_best_val']['MRR']
    print(f'  N=5000: DM test={dm_t:.4f}, 1L=SKIPPED')

print(f'\nTotal wall time: {total_time:.0f}s ({total_time/3600:.1f}hr)')
print(f'Done.')
