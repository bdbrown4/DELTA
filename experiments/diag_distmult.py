"""Quick diagnostic: DistMult (no GNN) at N=2000."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.phase46c_link_prediction import (
    load_lp_data, build_train_graph_tensors, create_lp_model,
    train_epoch, evaluate_lp,
)
import torch, numpy as np, time

device = 'cuda'
data = load_lp_data('fb15k-237', max_entities=2000)
print(f'Loaded: {data["num_entities"]} entities, {data["train"].shape[1]} train')

torch.manual_seed(42); np.random.seed(42)
model = create_lp_model('distmult', data['num_entities'], data['num_relations']).to(device)
print(f'DistMult params: {sum(p.numel() for p in model.parameters()):,}')

ei, et = build_train_graph_tensors(data['train'])
opt = torch.optim.Adam(model.parameters(), lr=0.001)
t0 = time.time()
for ep in range(1, 101):
    loss = train_epoch(model, data['train'], ei, et, opt, device, 512)
    if ep <= 3 or ep % 10 == 0:
        val = evaluate_lp(model, data['val'], ei, et,
                          data['hr_to_tails'], data['rt_to_heads'], device)
        print(f'Ep {ep:3d}  loss={loss:.4f}  MRR={val["MRR"]:.4f}  H@10={val["Hits@10"]:.4f}  [{time.time()-t0:.0f}s]')
