"""Phase 61b: DistMult Convergence Control at N=2000

Phase 61 compared DistMult vs DELTA at equal epoch count (200 epochs).
But DistMult finished in 60s while DELTA took 5,896s — a 98× wall-clock gap.
The "DELTA wins at N=2000" result (+0.079 test MRR) could simply be because
DELTA got 98× more compute, not because it has a better inductive bias.

This script answers the critical question: does DistMult's MRR keep climbing
with more epochs at N=2000, or does it plateau?

- If DM reaches ~0.30+ with 1000-2000 epochs → Phase 61's "advantage" collapses.
  DELTA is just slower, not better.
- If DM plateaus well below 0.30 → the GNN's parameter sharing provides
  inductive bias that DistMult can't replicate regardless of training budget.
  That would be a real, publishable result.

This experiment costs ~5 minutes of wall-clock time.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.phase46c_link_prediction import (
    load_lp_data, build_train_graph_tensors,
    train_epoch, evaluate_lp, LinkPredictionModel,
)
import torch, numpy as np, time

device = 'cuda'
d_node, d_edge = 64, 32

# ── Config ──
# Use same hyperparameters as Phase 61 for direct comparison,
# but train for much longer.
BS = 4096
LR = 0.003
NUM_EPOCHS = 2000  # 10× longer than Phase 61's 200

# Also test with the Phase 59 hyperparameters (bs=512, lr=0.001)
# that gave DistMult its best-known result (val MRR=0.3185).
CONFIGS = [
    {'label': 'DM_bs4096_lr003', 'bs': 4096, 'lr': 0.003, 'epochs': 2000},
    {'label': 'DM_bs512_lr001',  'bs': 512,  'lr': 0.001, 'epochs': 2000},
]


def run_distmult(data, ei, et, bs, lr, num_epochs, label=''):
    """Run DistMult (no GNN) baseline with detailed logging."""
    model = LinkPredictionModel(None, data['num_entities'],
                                 data['num_relations'], d_node, d_edge).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    best_val = {'MRR': 0}

    # Eval schedule: every 100 epochs to keep output manageable
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
            if val['MRR'] > best_val['MRR']:
                best_val = val.copy()
                best_val['best_epoch'] = ep

    # Test eval on final model
    test = evaluate_lp(model, data['test'], ei, et,
                       data['hr_to_tails'], data['rt_to_heads'], device)
    elapsed = time.time() - t0
    return {'val_best': best_val, 'test': test, 'time': elapsed, 'params': params}


# ── Main ──
print()
print('=' * 70)
print('  PHASE 61b: DistMult Convergence Control at N=2000')
print('=' * 70)

torch.manual_seed(42)
np.random.seed(42)
data = load_lp_data('fb15k-237', max_entities=2000)
print(f'  {data["num_entities"]} ent, {data["num_relations"]} rel, '
      f'{data["train"].shape[1]} train triples')

ei, et = build_train_graph_tensors(data['train'])

results = {}
for cfg in CONFIGS:
    label = cfg['label']
    bs, lr, epochs = cfg['bs'], cfg['lr'], cfg['epochs']
    steps_per_epoch = (data['train'].shape[1] + bs - 1) // bs
    total_steps = steps_per_epoch * epochs

    print(f'\n  --- {label} (bs={bs}, lr={lr}, {epochs} epochs, '
          f'{steps_per_epoch} steps/ep, {total_steps} total steps) ---')

    torch.manual_seed(42)
    np.random.seed(42)
    r = run_distmult(data, ei, et, bs, lr, epochs, label=label[:12])
    results[label] = r

# ── Summary ──
print(f'\n{"=" * 70}')
print(f'SUMMARY')
print(f'{"=" * 70}')
print(f'{"Config":<25} {"best_val":>9} {"best_ep":>8} {"test_MRR":>10} '
      f'{"test_H@1":>9} {"test_H@10":>10} {"Time":>7}')
print(f'{"-" * 80}')

for label, r in results.items():
    v = r['val_best']
    t = r['test']
    print(f'{label:<25} {v["MRR"]:>9.4f} {v.get("best_epoch","?"):>8} '
          f'{t["MRR"]:>10.4f} {t["Hits@1"]:>9.4f} {t["Hits@10"]:>10.4f} '
          f'{r["time"]:>6.0f}s')

# ── Comparison with Phase 61 DELTA ──
print(f'\n  Phase 61 reference:')
print(f'    1L-DELTA@2000: val_best=0.3357, test=0.3088 (5,896s)')
print(f'    DM@2000 (200ep): val_best=0.2271, test=0.2297 (60s)')

for label, r in results.items():
    v = r['val_best']
    t = r['test']
    delta_test = 0.3088
    gap = t['MRR'] - delta_test
    print(f'\n    {label}: test={t["MRR"]:.4f} (gap vs DELTA: {gap:+.4f})')
    if t['MRR'] >= 0.29:
        print(f'    >>> DM reaches ~0.30 range. Phase 61 "advantage" is compute artifact.')
    else:
        print(f'    >>> DM stuck below 0.29. DELTA advantage may reflect real inductive bias.')

print(f'\nDone.')
