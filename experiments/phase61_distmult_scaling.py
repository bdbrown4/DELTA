"""Phase 61: DistMult-vs-DELTA Across Scales — The Existence Question

Does DELTA's edge-to-edge attention provide *any* measurable lift over a
trivial no-GNN baseline (DistMult) at *any* scale?

Phase 60 showed all models converge to ~0.31 MRR at N=2000: DistMult (0.3185),
1-layer DELTA (0.3094), 2L+gate (0.3065), 3L+gate (0.3138). DELTA's mechanism
contributes zero measurable value at N=2000.

Phase 40 showed DistMult=0.4841 and DELTA-Full=0.4938 at N=500 (gap: +0.010) —
but DistMult hadn't converged (500 epochs, still climbing).

This phase answers the existence question with controlled comparisons:
  - DistMult vs 1-layer DELTA at N=500, N=1000, N=2000
  - Same hyperparameters, same eval schedule
  - 200 epochs each (fast, cheap)

Three possible outcomes:
  1. DistMult ≈ DELTA at N=500 → mechanism never helped at any scale
  2. DELTA > DistMult at N=500, advantage dissipates by N=1000-2000 → scale ceiling
  3. DELTA > DistMult at all scales → mechanism works, just needs optimization

This is a ~2 hour phase total.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.phase46c_link_prediction import (
    load_lp_data, build_train_graph_tensors,
    train_epoch, evaluate_lp, LinkPredictionModel,
)
from delta.model import DELTAModel
from delta.graph import DeltaGraph
import torch, numpy as np, time, gc

device = 'cuda'
d_node, d_edge = 64, 32

# Per-scale hyperparameters: batch_size scales with dataset size to ensure
# enough gradient steps. lr scales linearly with bs (linear scaling rule).
# N=500 gets more epochs since DistMult was still climbing at 500 in Phase 40.
SCALE_CONFIG = {
    500:  {'bs': 512,  'lr': 0.001, 'epochs': 500},
    1000: {'bs': 2048, 'lr': 0.002, 'epochs': 300},
    2000: {'bs': 4096, 'lr': 0.003, 'epochs': 200},
}


def run_distmult(data, ei, et, bs, lr, num_epochs, label=''):
    """Run DistMult (no GNN) baseline."""
    model = LinkPredictionModel(None, data['num_entities'],
                                 data['num_relations'], d_node, d_edge).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    best_val = {'MRR': 0}
    eval_every = max(25, num_epochs // 8)

    for ep in range(1, num_epochs + 1):
        loss = train_epoch(model, data['train'], ei, et, opt, device, batch_size=bs)
        if ep % eval_every == 0 or ep == num_epochs:
            val = evaluate_lp(model, data['val'], ei, et,
                              data['hr_to_tails'], data['rt_to_heads'], device)
            elapsed = time.time() - t0
            print(f'  [{label}] Ep {ep:3d}  loss={loss:.4f}  '
                  f'MRR={val["MRR"]:.4f}  H@1={val["Hits@1"]:.4f}  '
                  f'H@10={val["Hits@10"]:.4f}  [{elapsed:.0f}s]')
            if val['MRR'] > best_val['MRR']:
                best_val = val.copy()
                best_val['best_epoch'] = ep

    test = evaluate_lp(model, data['test'], ei, et,
                       data['hr_to_tails'], data['rt_to_heads'], device)
    elapsed = time.time() - t0
    return {'val_best': best_val, 'test': test, 'time': elapsed, 'params': params}


def run_delta_1layer(data, ei, et, cached_edge_adj, bs, lr, num_epochs, label=''):
    """Run 1-layer DELTA."""
    enc = DELTAModel(d_node=d_node, d_edge=d_edge,
                      num_layers=1, num_heads=4, init_temp=1.0)
    model = LinkPredictionModel(enc, data['num_entities'],
                                 data['num_relations'], d_node, d_edge).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    best_val = {'MRR': 0}
    eval_every = max(25, num_epochs // 8)

    for ep in range(1, num_epochs + 1):
        loss = train_epoch(model, data['train'], ei, et, opt, device,
                           batch_size=bs, cached_edge_adj=cached_edge_adj)
        if ep % eval_every == 0 or ep == num_epochs:
            val = evaluate_lp(model, data['val'], ei, et,
                              data['hr_to_tails'], data['rt_to_heads'], device,
                              cached_edge_adj=cached_edge_adj)
            elapsed = time.time() - t0
            print(f'  [{label}] Ep {ep:3d}  loss={loss:.4f}  '
                  f'MRR={val["MRR"]:.4f}  H@1={val["Hits@1"]:.4f}  '
                  f'H@10={val["Hits@10"]:.4f}  [{elapsed:.0f}s]')
            if val['MRR'] > best_val['MRR']:
                best_val = val.copy()
                best_val['best_epoch'] = ep

    test = evaluate_lp(model, data['test'], ei, et,
                       data['hr_to_tails'], data['rt_to_heads'], device,
                       cached_edge_adj=cached_edge_adj)
    elapsed = time.time() - t0
    return {'val_best': best_val, 'test': test, 'time': elapsed, 'params': params}


def run_delta_3layer(data, ei, et, cached_edge_adj, bs, lr, num_epochs, label=''):
    """Run 3-layer DELTA (standard config from Phases 40-58)."""
    enc = DELTAModel(d_node=d_node, d_edge=d_edge,
                      num_layers=3, num_heads=4, init_temp=1.0)
    model = LinkPredictionModel(enc, data['num_entities'],
                                 data['num_relations'], d_node, d_edge).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    best_val = {'MRR': 0}
    eval_every = max(25, num_epochs // 8)

    for ep in range(1, num_epochs + 1):
        loss = train_epoch(model, data['train'], ei, et, opt, device,
                           batch_size=bs, cached_edge_adj=cached_edge_adj)
        if ep % eval_every == 0 or ep == num_epochs:
            val = evaluate_lp(model, data['val'], ei, et,
                              data['hr_to_tails'], data['rt_to_heads'], device,
                              cached_edge_adj=cached_edge_adj)
            elapsed = time.time() - t0
            print(f'  [{label}] Ep {ep:3d}  loss={loss:.4f}  '
                  f'MRR={val["MRR"]:.4f}  H@1={val["Hits@1"]:.4f}  '
                  f'H@10={val["Hits@10"]:.4f}  [{elapsed:.0f}s]')
            if val['MRR'] > best_val['MRR']:
                best_val = val.copy()
                best_val['best_epoch'] = ep

    test = evaluate_lp(model, data['test'], ei, et,
                       data['hr_to_tails'], data['rt_to_heads'], device,
                       cached_edge_adj=cached_edge_adj)
    elapsed = time.time() - t0
    return {'val_best': best_val, 'test': test, 'time': elapsed, 'params': params}


results = {}

for N in [500, 1000, 2000]:
    cfg = SCALE_CONFIG[N]
    bs, lr, num_epochs = cfg['bs'], cfg['lr'], cfg['epochs']
    print(f'\n{"="*70}')
    print(f'  SCALE N={N}  (bs={bs}, lr={lr}, epochs={num_epochs})')
    print(f'{"="*70}')

    torch.manual_seed(42); np.random.seed(42)
    data = load_lp_data('fb15k-237', max_entities=N)
    print(f'  {data["num_entities"]} ent, {data["num_relations"]} rel, '
          f'{data["train"].shape[1]} train triples')
    steps_per_epoch = (data['train'].shape[1] + bs - 1) // bs
    print(f'  {steps_per_epoch} steps/epoch, {steps_per_epoch * num_epochs} total steps')

    ei, et = build_train_graph_tensors(data['train'])

    # Pre-compute edge adjacency for DELTA
    print(f'  Building edge adjacency...')
    t_adj = time.time()
    with torch.no_grad():
        tmp_graph = DeltaGraph(
            node_features=torch.zeros(data['num_entities'], d_node, device=device),
            edge_features=torch.zeros(data['train'].shape[1], d_edge, device=device),
            edge_index=ei.to(device),
        )
        tmp_graph.build_edge_adjacency()
        cached_edge_adj = tmp_graph._edge_adj_cache[1]
        del tmp_graph
        torch.cuda.empty_cache()
    print(f'  Edge adjacency: {cached_edge_adj.shape[1]:,} pairs in {time.time()-t_adj:.1f}s')

    # DistMult
    print(f'\n  --- DistMult (no GNN) ---')
    torch.manual_seed(42); np.random.seed(42)
    dm = run_distmult(data, ei, et, bs, lr, num_epochs, label=f'DM@{N}')
    results[f'distmult@{N}'] = dm

    # 1-layer DELTA
    print(f'\n  --- 1-layer DELTA ---')
    torch.manual_seed(42); np.random.seed(42)
    d1 = run_delta_1layer(data, ei, et, cached_edge_adj, bs, lr, num_epochs, label=f'1L@{N}')
    results[f'delta_1L@{N}'] = d1

    # 3-layer DELTA (only at N=500 where it historically works)
    if N == 500:
        print(f'\n  --- 3-layer DELTA ---')
        torch.manual_seed(42); np.random.seed(42)
        d3 = run_delta_3layer(data, ei, et, cached_edge_adj, bs, lr, num_epochs, label=f'3L@{N}')
        results[f'delta_3L@{N}'] = d3

    gc.collect()
    torch.cuda.empty_cache()


# ── Summary ──
print(f'\n{"="*70}')
print(f'PHASE 61 SUMMARY — DistMult vs DELTA Across Scales')
print(f'{"="*70}')
print(f'{"Model":<22} {"N":>5} {"val_MRR":>9} {"test_MRR":>10} {"test_H@1":>9} '
      f'{"test_H@10":>10} {"Δ vs DM":>8} {"Time":>7}')
print(f'{"-"*80}')

for N in [500, 1000, 2000]:
    dm_key = f'distmult@{N}'
    dm_mrr = results[dm_key]['test']['MRR']

    for model_key in [dm_key, f'delta_1L@{N}', f'delta_3L@{N}']:
        if model_key not in results:
            continue
        r = results[model_key]
        t = r['test']
        v = r['val_best']
        name = model_key.replace('@', ' N=')
        delta = t['MRR'] - dm_mrr
        delta_str = f'{delta:+.4f}' if 'distmult' not in model_key else '—'
        print(f'{name:<22} {N:>5} {v["MRR"]:>9.4f} {t["MRR"]:>10.4f} '
              f'{t["Hits@1"]:>9.4f} {t["Hits@10"]:>10.4f} {delta_str:>8} {r["time"]:>6.0f}s')
    print()

# ── Verdict ──
print(f'VERDICT:')
dm500 = results['distmult@500']['test']['MRR']
d1_500 = results['delta_1L@500']['test']['MRR']
dm1000 = results['distmult@1000']['test']['MRR']
d1_1000 = results['delta_1L@1000']['test']['MRR']
dm2000 = results['distmult@2000']['test']['MRR']
d1_2000 = results['delta_1L@2000']['test']['MRR']

print(f'  N=500:  DELTA-DistMult gap = {d1_500 - dm500:+.4f}')
print(f'  N=1000: DELTA-DistMult gap = {d1_1000 - dm1000:+.4f}')
print(f'  N=2000: DELTA-DistMult gap = {d1_2000 - dm2000:+.4f}')

if d1_500 - dm500 < 0.01:
    print(f'\n  SCENARIO 1: DELTA ≈ DistMult at N=500. Mechanism never helped at any scale.')
elif d1_1000 - dm1000 < 0.01:
    print(f'\n  SCENARIO 2: DELTA beats DistMult at N=500 (+{d1_500-dm500:.4f}) '
          f'but gap closes by N=1000. Scale ceiling identified.')
else:
    print(f'\n  SCENARIO 3: DELTA beats DistMult at all tested scales. '
          f'Mechanism works, needs optimization.')
