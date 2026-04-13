"""Phase 60: Residual Gating for Depth Scaling at N=2000

Hypothesis: 2-layer DELTA with residual gating achieves LP MRR >= 0.30
at N=2000, matching or exceeding 1-layer baseline (0.3338 val).

Conditions:
  A: 2-layer DELTA + residual gate  (gate_init=0.1 → favours residual)
  B: 3-layer DELTA + residual gate  (gate_init=0.1)
  C: 1-layer DELTA (control, no gate — reproducing Phase 59 diagnostic)

All: d_node=64, d_edge=32, num_heads=4, lr=0.003, bs=4096,
     200 epochs, seed=42, cached_edge_adj, eval at 25/50/75/100/150/200

Success Criteria:
  - A (2L+gate) MRR >= 0.30  → gating enables multi-layer
  - B (3L+gate) MRR > Phase 59 3L (0.0018) → gating mitigates collapse
  - Gate logits shift during training → model learns gating
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.phase46c_link_prediction import (
    load_lp_data, build_train_graph_tensors,
    train_epoch, evaluate_lp, LinkPredictionModel,
)
from delta.model import DELTAModel
from delta.graph import DeltaGraph
import torch, numpy as np, time

device = 'cuda'
data = load_lp_data('fb15k-237', max_entities=2000)
print(f'Loaded: {data["num_entities"]} ent, {data["num_relations"]} rel, '
      f'{data["train"].shape[1]} train triples')

torch.manual_seed(42); np.random.seed(42)

d_node, d_edge = 64, 32
ei, et = build_train_graph_tensors(data['train'])

# Pre-compute edge adjacency (shared across conditions)
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


def run_condition(name, encoder, num_epochs=200):
    """Run a single condition and return results dict."""
    model = LinkPredictionModel(encoder, data['num_entities'],
                                 data['num_relations'], d_node, d_edge).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'\n{"="*60}')
    print(f'Condition {name}  |  params={params:,}')
    print(f'{"="*60}')

    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    t0 = time.time()
    results = {'name': name, 'params': params, 'trajectory': []}

    for ep in range(1, num_epochs + 1):
        loss = train_epoch(model, data['train'], ei, et, opt, device,
                           batch_size=4096, cached_edge_adj=cached_edge_adj)
        if ep % 10 == 0:
            elapsed = time.time() - t0
            # Print gate values if available
            gate_str = ''
            if hasattr(encoder, 'node_gate_logits') and encoder.node_gate_logits is not None:
                alphas = [torch.sigmoid(g).item() for g in encoder.node_gate_logits]
                gate_str = f'  gates=[{", ".join(f"{a:.3f}" for a in alphas)}]'
            print(f'Ep {ep:3d}  loss={loss:.4f}{gate_str}  [{elapsed:.0f}s]')

        if ep in (25, 50, 75, 100, 150, 200):
            val = evaluate_lp(model, data['val'], ei, et,
                              data['hr_to_tails'], data['rt_to_heads'], device,
                              cached_edge_adj=cached_edge_adj)
            elapsed = time.time() - t0
            results['trajectory'].append({'epoch': ep, **val})
            gate_str = ''
            if hasattr(encoder, 'node_gate_logits') and encoder.node_gate_logits is not None:
                node_alphas = [torch.sigmoid(g).item() for g in encoder.node_gate_logits]
                edge_alphas = [torch.sigmoid(g).item() for g in encoder.edge_gate_logits]
                gate_str = (f'\n    node_gates=[{", ".join(f"{a:.4f}" for a in node_alphas)}]'
                           f'\n    edge_gates=[{", ".join(f"{a:.4f}" for a in edge_alphas)}]')
            print(f'  EVAL Ep {ep:3d}  MRR={val["MRR"]:.4f}  H@1={val["Hits@1"]:.4f}  '
                  f'H@3={val["Hits@3"]:.4f}  H@10={val["Hits@10"]:.4f}  [{elapsed:.0f}s]{gate_str}')

    # Final test
    test = evaluate_lp(model, data['test'], ei, et,
                       data['hr_to_tails'], data['rt_to_heads'], device,
                       cached_edge_adj=cached_edge_adj)
    results['test'] = test
    results['time'] = time.time() - t0

    # Final gate values
    if hasattr(encoder, 'node_gate_logits') and encoder.node_gate_logits is not None:
        results['final_node_gates'] = [torch.sigmoid(g).item() for g in encoder.node_gate_logits]
        results['final_edge_gates'] = [torch.sigmoid(g).item() for g in encoder.edge_gate_logits]

    print(f'\n  TEST  MRR={test["MRR"]:.4f}  H@1={test["Hits@1"]:.4f}  '
          f'H@3={test["Hits@3"]:.4f}  H@10={test["Hits@10"]:.4f}')
    return results


# ── Condition A: 2-layer + residual gate ──
enc_A = DELTAModel(d_node=d_node, d_edge=d_edge,
                    num_layers=2, num_heads=4, init_temp=1.0,
                    residual_gate=True, residual_gate_init=0.1)
res_A = run_condition('A: 2L+gate', enc_A)

# ── Condition B: 3-layer + residual gate ──
torch.manual_seed(42); np.random.seed(42)
enc_B = DELTAModel(d_node=d_node, d_edge=d_edge,
                    num_layers=3, num_heads=4, init_temp=1.0,
                    residual_gate=True, residual_gate_init=0.1)
res_B = run_condition('B: 3L+gate', enc_B)

# ── Condition C: 1-layer control (no gate) ──
torch.manual_seed(42); np.random.seed(42)
enc_C = DELTAModel(d_node=d_node, d_edge=d_edge,
                    num_layers=1, num_heads=4, init_temp=1.0)
res_C = run_condition('C: 1L ctrl', enc_C)


# ── Summary table ──
print(f'\n{"="*70}')
print(f'PHASE 60 SUMMARY — Residual Gating for Depth Scaling (N=2000)')
print(f'{"="*70}')
print(f'{"Condition":<20} {"MRR":>7} {"H@1":>7} {"H@3":>7} {"H@10":>7} {"Time":>7}')
print(f'{"-"*55}')
for r in [res_A, res_B, res_C]:
    t = r['test']
    print(f'{r["name"]:<20} {t["MRR"]:>7.4f} {t["Hits@1"]:>7.4f} {t["Hits@3"]:>7.4f} '
          f'{t["Hits@10"]:>7.4f} {r["time"]:>6.0f}s')

print(f'\nPhase 59 references:')
print(f'  3L no-gate:  MRR=0.0018  (catastrophic)')
print(f'  1L no-gate:  MRR=0.3338  (val), 0.3094 (test)')
print(f'  DistMult:    MRR=0.3185  (val)')

# Gate evolution
for r in [res_A, res_B]:
    if 'final_node_gates' in r:
        print(f'\n{r["name"]} final node gates: {r["final_node_gates"]}')
        print(f'{r["name"]} final edge gates: {r["final_edge_gates"]}')
