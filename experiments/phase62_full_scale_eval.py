"""Phase 62: Full-Scale FB15k-237 Evaluation — DELTA vs DistMult Beyond N=2000

Phase 61/61b confirmed DELTA's advantage at N=2000: peak val MRR +0.017 over
DistMult, plus dramatically better overfitting resistance. But N=2000 is only
13.7% of full FB15k-237 (14,541 entities).

This phase answers: does DELTA's advantage persist, grow, or vanish at larger
scales? We test at N=5000 and full FB15k-237 (N=14,541).

Critical constraint: edge adjacency at full scale is estimated at ~211M pairs,
requiring ~80-100GB for attention tensors alone. Solution: subsample edge
adjacency to a budget (Phase 30 showed ±0.2% impact at N=500).

- If DELTA's advantage grows with scale → strong scaling story for publication
- If DELTA's advantage holds at ~+0.02 → confirms genuine inductive bias
- If DELTA's advantage vanishes → scale ceiling; DELTA only helps at small N
"""
import sys, os, gc, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.phase46c_link_prediction import (
    load_lp_data, build_train_graph_tensors,
    train_epoch, evaluate_lp, LinkPredictionModel,
)
from delta.model import DELTAModel
from delta.graph import DeltaGraph
import torch, numpy as np

device = 'cuda'
d_node, d_edge = 64, 32

# Edge adjacency budget: cap to keep attention within GPU memory.
# At N=2000, 15.2M pairs used ~2.4GB for attention. Budget 15M → ~2.4GB.
# Phase 30 showed subsampling to 26% VRAM had ±0.2% impact on MRR.
MAX_EDGE_ADJ_PAIRS = 15_000_000

# Scale configurations
# N=5000: intermediate scale, should be feasible
# N=full: max scale, DELTA may be very slow but let's try
SCALES = [
    {'max_entities': 5000,  'label': 'N=5000'},
    {'max_entities': None,  'label': 'N=full'},
]

# Hyperparameters: same as Phase 61 N=2000 for comparability
DM_EPOCHS = 2000   # DistMult is fast; give it plenty of epochs (Phase 61b pattern)
DELTA_EPOCHS = 200  # DELTA converges by ep150-175 historically
BS = 4096
LR = 0.003


def subsample_edge_adj(edge_adj, budget):
    """Uniformly subsample edge adjacency pairs to a budget."""
    n_pairs = edge_adj.shape[1]
    if n_pairs <= budget:
        return edge_adj, n_pairs, 1.0
    perm = torch.randperm(n_pairs, device=edge_adj.device)[:budget]
    return edge_adj[:, perm], budget, budget / n_pairs


def run_distmult(data, ei, et, bs, lr, num_epochs, label=''):
    """Run DistMult (no GNN) baseline."""
    model = LinkPredictionModel(None, data['num_entities'],
                                data['num_relations'], d_node, d_edge).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    best_val = {'MRR': 0}
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

    # Test eval on final model
    test = evaluate_lp(model, data['test'], ei, et,
                       data['hr_to_tails'], data['rt_to_heads'], device)
    elapsed = time.time() - t0
    del model, opt
    gc.collect(); torch.cuda.empty_cache()
    return {'val_best': best_val, 'test': test, 'time': elapsed, 'params': params}


def run_delta_1layer(data, ei, et, cached_edge_adj, bs, lr, num_epochs, label=''):
    """Run 1-layer DELTA with pre-computed (possibly subsampled) edge adjacency."""
    enc = DELTAModel(d_node=d_node, d_edge=d_edge,
                     num_layers=1, num_heads=4, init_temp=1.0)
    model = LinkPredictionModel(enc, data['num_entities'],
                                data['num_relations'], d_node, d_edge).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    best_val = {'MRR': 0}
    eval_every = 25

    # Time first epoch to estimate total
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
                # Reduce epochs to fit within ~5 hours
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

    # Test eval on final model
    test = evaluate_lp(model, data['test'], ei, et,
                       data['hr_to_tails'], data['rt_to_heads'], device,
                       cached_edge_adj=cached_edge_adj)
    elapsed = time.time() - t0
    del model, opt
    gc.collect(); torch.cuda.empty_cache()
    return {'val_best': best_val, 'test': test, 'time': elapsed, 'params': params}


# ── Main ──
print()
print('=' * 70)
print('  PHASE 62: Full-Scale FB15k-237 — DELTA vs DistMult Beyond N=2000')
print('=' * 70)
sys.stdout.flush()

all_results = {}
total_t0 = time.time()

for scale_cfg in SCALES:
    max_ent = scale_cfg['max_entities']
    scale_label = scale_cfg['label']

    print(f'\n{"="*70}')
    print(f'  SCALE: {scale_label}  (max_entities={max_ent})')
    print(f'{"="*70}')
    sys.stdout.flush()

    torch.manual_seed(42); np.random.seed(42)
    data = load_lp_data('fb15k-237', max_entities=max_ent)
    N = data['num_entities']
    R = data['num_relations']
    E_train = data['train'].shape[1]
    E_val = data['val'].shape[1]
    E_test = data['test'].shape[1]
    print(f'  {N} entities, {R} relations')
    print(f'  {E_train} train / {E_val} val / {E_test} test triples')
    steps_per_epoch = (E_train + BS - 1) // BS
    print(f'  {steps_per_epoch} steps/epoch (bs={BS})')
    sys.stdout.flush()

    ei, et = build_train_graph_tensors(data['train'])

    # ── DistMult ──
    print(f'\n  --- DistMult (no GNN), {DM_EPOCHS} epochs ---')
    sys.stdout.flush()
    torch.manual_seed(42); np.random.seed(42)
    dm = run_distmult(data, ei, et, BS, LR, DM_EPOCHS, label=f'DM_{scale_label}')
    all_results[f'DM_{scale_label}'] = dm
    print(f'  DM {scale_label}: peak val={dm["val_best"]["MRR"]:.4f} '
          f'(ep{dm["val_best"].get("best_epoch","?")}), '
          f'test={dm["test"]["MRR"]:.4f}, {dm["time"]:.0f}s')
    sys.stdout.flush()

    # ── Build edge adjacency ──
    print(f'\n  Building edge adjacency...')
    sys.stdout.flush()
    t_adj = time.time()
    try:
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
        n_full = full_edge_adj.shape[1]
        t_build = time.time() - t_adj
        print(f'  Edge adjacency: {n_full:,} pairs in {t_build:.1f}s')
        sys.stdout.flush()

        # Subsample if needed
        cached_edge_adj, n_used, keep_frac = subsample_edge_adj(
            full_edge_adj, MAX_EDGE_ADJ_PAIRS)
        if keep_frac < 1.0:
            print(f'  Subsampled to {n_used:,} pairs ({keep_frac:.1%} of full)')
            sys.stdout.flush()
        del full_edge_adj
        torch.cuda.empty_cache()
        edge_adj_ok = True
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f'  Edge adjacency build FAILED: {e}')
        print(f'  Skipping DELTA at {scale_label}')
        sys.stdout.flush()
        edge_adj_ok = False
        cached_edge_adj = None

    # ── 1-layer DELTA ──
    if edge_adj_ok:
        print(f'\n  --- 1-layer DELTA, up to {DELTA_EPOCHS} epochs ---')
        sys.stdout.flush()
        torch.manual_seed(42); np.random.seed(42)
        try:
            d1 = run_delta_1layer(data, ei, et, cached_edge_adj,
                                  BS, LR, DELTA_EPOCHS,
                                  label=f'1L_{scale_label}')
            all_results[f'1L_{scale_label}'] = d1
            print(f'  1L {scale_label}: peak val={d1["val_best"]["MRR"]:.4f} '
                  f'(ep{d1["val_best"].get("best_epoch","?")}), '
                  f'test={d1["test"]["MRR"]:.4f}, {d1["time"]:.0f}s')
            sys.stdout.flush()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f'  DELTA training FAILED (OOM): {e}')
            sys.stdout.flush()
        finally:
            del cached_edge_adj
            gc.collect(); torch.cuda.empty_cache()
    else:
        del cached_edge_adj if cached_edge_adj is not None else None
        gc.collect(); torch.cuda.empty_cache()

    # Free data for next scale
    del data, ei, et
    gc.collect(); torch.cuda.empty_cache()

total_time = time.time() - total_t0

# ── Summary ──
print(f'\n{"="*70}')
print(f'PHASE 62 SUMMARY')
print(f'{"="*70}')
print(f'{"Model":<20} {"N":>7} {"peak_val":>9} {"best_ep":>8} '
      f'{"test_MRR":>10} {"test_H@1":>9} {"test_H@10":>10} {"Time":>7}')
print(f'{"-"*85}')

# Phase 61b references
print(f'{"DM@2000 (P61b)":<20} {"2000":>7} {"0.3185":>9} {"100":>8} '
      f'{"0.2329":>10} {"0.1159":>9} {"0.5031":>10} {"261s":>7}')
print(f'{"1L@2000 (P61)":<20} {"2000":>7} {"0.3357":>9} {"175":>8} '
      f'{"0.3088":>10} {"0.1787":>9} {"0.5963":>10} {"5896s":>7}')
print(f'{"-"*85}')

for key, r in all_results.items():
    v = r['val_best']
    t = r['test']
    # Extract N from label
    n_str = key.split('_')[1]  # 'N=5000' or 'N=full'
    print(f'{key:<20} {n_str:>7} {v["MRR"]:>9.4f} '
          f'{v.get("best_epoch","?"):>8} {t["MRR"]:>10.4f} '
          f'{t["Hits@1"]:>9.4f} {t["Hits@10"]:>10.4f} {r["time"]:>6.0f}s')

# ── DELTA vs DistMult gaps ──
print(f'\n--- DELTA vs DistMult Gaps (peak val MRR) ---')
print(f'  N=2000 (Phase 61b): DM=0.3185, 1L=0.3357, gap=+0.0172')

for scale_cfg in SCALES:
    sl = scale_cfg['label']
    dm_key = f'DM_{sl}'
    d1_key = f'1L_{sl}'
    if dm_key in all_results and d1_key in all_results:
        dm_v = all_results[dm_key]['val_best']['MRR']
        d1_v = all_results[d1_key]['val_best']['MRR']
        gap = d1_v - dm_v
        dm_t = all_results[dm_key]['test']['MRR']
        d1_t = all_results[d1_key]['test']['MRR']
        test_gap = d1_t - dm_t
        print(f'  {sl}: DM={dm_v:.4f}, 1L={d1_v:.4f}, '
              f'val gap={gap:+.4f}, test gap={test_gap:+.4f}')
    elif dm_key in all_results:
        dm_v = all_results[dm_key]['val_best']['MRR']
        print(f'  {sl}: DM={dm_v:.4f}, 1L=SKIPPED (OOM or timeout)')

print(f'\nTotal wall time: {total_time:.0f}s ({total_time/3600:.1f}hr)')
print(f'Done.')
