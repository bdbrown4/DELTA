"""Phase 64 local validation — quick test at N=50 on CPU.

Uses N=50 to stay under 500 edges (dense matmul path, no torch_sparse).
This is a functional validation, not a performance benchmark.
"""
import sys, os, gc, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.phase46c_link_prediction import (
    load_lp_data, build_train_graph_tensors,
    train_epoch, evaluate_lp, LinkPredictionModel,
)
from delta.model import DELTAModel
from delta.graph import DeltaGraph
import torch, numpy as np, copy

device = 'cpu'
d_node, d_edge = 64, 32

# Load very small dataset (N=50 → ~200 edges, fits dense matmul path)
torch.manual_seed(42); np.random.seed(42)
data = load_lp_data('fb15k-237', max_entities=50)
N = data['num_entities']
E_train = data['train'].shape[1]
ei, et = build_train_graph_tensors(data['train'])
print(f'Data: {N} entities, {data["num_relations"]} relations, {E_train} train edges')

# Build edge adjacency
tmp = DeltaGraph(
    node_features=torch.zeros(N, d_node, device=device),
    edge_features=torch.zeros(E_train, d_edge, device=device),
    edge_index=ei.to(device),
)
tmp.build_edge_adjacency()
full_ea = tmp._edge_adj_cache[1]
n_pairs = full_ea.shape[1]
avg_deg = n_pairs / E_train
print(f'E_adj: {n_pairs:,} pairs, {avg_deg:.0f} avg per edge')

# Quick 20-epoch test for 3 configs: no topk, topk=32, topk=8
configs = [
    ('baseline (no topk)', None),
    ('topk=32', 32),
    ('topk=8', 8),
]

for name, topk in configs:
    torch.manual_seed(42); np.random.seed(42)
    enc = DELTAModel(d_node=d_node, d_edge=d_edge,
                     num_layers=1, num_heads=4, init_temp=1.0,
                     topk_edges=topk)
    model = LinkPredictionModel(enc, data['num_entities'],
                                data['num_relations'], d_node, d_edge).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    t0 = time.time()

    for ep in range(1, 21):
        loss = train_epoch(model, data['train'], ei, et, opt, device,
                           batch_size=4096, cached_edge_adj=full_ea)

    val = evaluate_lp(model, data['val'], ei, et,
                      data['hr_to_tails'], data['rt_to_heads'], device,
                      cached_edge_adj=full_ea)
    elapsed = time.time() - t0
    print(f'  {name:25s}: val_MRR={val["MRR"]:.4f} H@10={val["Hits@10"]:.4f} ({elapsed:.1f}s)')
    del model, enc, opt
    gc.collect()

print('\nLocal test complete!')
