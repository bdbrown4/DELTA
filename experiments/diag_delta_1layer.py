"""Diagnostic: 1-layer DELTA at N=2000 — test if depth causes over-smoothing.

If 1-layer gets MRR ~0.05+, depth is the problem → Phase 60 reduces layers.
If 1-layer is still ~0.001, the edge-to-edge attention itself is broken at
this edge count → Phase 60 must attack attention sparsity directly.

Compare against:
  - 3-layer DELTA (Phase 59): MRR = 0.0018 (near-random)
  - DistMult (no GNN):        MRR = 0.3185 (functional)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.phase46c_link_prediction import (
    load_lp_data, build_train_graph_tensors, create_lp_model,
    train_epoch, evaluate_lp, LinkPredictionModel,
)
from delta.model import DELTAModel
from delta.graph import DeltaGraph
import torch, numpy as np, time, gc

device = 'cuda'
data = load_lp_data('fb15k-237', max_entities=2000)
print(f'Loaded: {data["num_entities"]} ent, {data["num_relations"]} rel, '
      f'{data["train"].shape[1]} train triples')

torch.manual_seed(42); np.random.seed(42)

# --- 1-layer DELTA ---
d_node, d_edge = 64, 32
enc_1L = DELTAModel(d_node=d_node, d_edge=d_edge,
                     num_layers=1, num_heads=4, init_temp=1.0)
model_1L = LinkPredictionModel(enc_1L, data['num_entities'],
                                data['num_relations'], d_node, d_edge).to(device)
params_1L = sum(p.numel() for p in model_1L.parameters())
print(f'1-layer DELTA params: {params_1L:,}')

ei, et = build_train_graph_tensors(data['train'])

# Pre-compute edge adjacency
print('Pre-computing edge adjacency...')
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
print(f'Edge adjacency: {cached_edge_adj.shape[1]} pairs in {time.time()-t_adj:.1f}s')

opt = torch.optim.Adam(model_1L.parameters(), lr=0.003)
t0 = time.time()

for ep in range(1, 201):
    loss = train_epoch(model_1L, data['train'], ei, et, opt, device,
                       batch_size=4096, cached_edge_adj=cached_edge_adj)
    if ep % 10 == 0:
        elapsed = time.time() - t0
        print(f'Ep {ep:3d}  loss={loss:.4f}  [{elapsed:.0f}s]')
    if ep in (25, 50, 75, 100, 150, 200):
        val = evaluate_lp(model_1L, data['val'], ei, et,
                          data['hr_to_tails'], data['rt_to_heads'], device,
                          cached_edge_adj=cached_edge_adj)
        elapsed = time.time() - t0
        print(f'  EVAL Ep {ep:3d}  MRR={val["MRR"]:.4f}  H@1={val["Hits@1"]:.4f}  '
              f'H@3={val["Hits@3"]:.4f}  H@10={val["Hits@10"]:.4f}  [{elapsed:.0f}s]')

# Final test eval
test = evaluate_lp(model_1L, data['test'], ei, et,
                   data['hr_to_tails'], data['rt_to_heads'], device,
                   cached_edge_adj=cached_edge_adj)
elapsed = time.time() - t0
print(f'\n=== 1-LAYER DELTA FINAL ===')
print(f'  test_MRR={test["MRR"]:.4f}  H@1={test["Hits@1"]:.4f}  '
      f'H@3={test["Hits@3"]:.4f}  H@10={test["Hits@10"]:.4f}')
print(f'  Total time: {elapsed:.0f}s')

# Verdict
if test['MRR'] > 0.05:
    print('\nVERDICT: Depth is part of the problem — 1-layer recovers signal.')
    print('Phase 60 should start with depth reduction + residual tuning.')
elif test['MRR'] > 0.01:
    print('\nVERDICT: Marginal improvement — depth contributes but mechanism also weak.')
    print('Phase 60 needs both depth reduction AND attention sparsification.')
else:
    print('\nVERDICT: Edge-to-edge attention is broken at this scale regardless of depth.')
    print('Phase 60 must attack the attention mechanism directly (sparse/local E2E).')
