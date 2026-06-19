"""Phase 76 — DIAGNOSE the 5p edge-attention benefit (the sole surviving positive).

Phase 73 found hops1/hops2 beat node_only by ~+0.02 at 5p (4/5 seeds, borderline CI). Is it
a genuine STRUCTURAL-LOCAL effect, a soft-traversal τ=1.0 artifact, or generic capacity?
This re-trains the structural conditions (Phase 73 didn't checkpoint) + a RANDOM-adjacency
capacity control, saves checkpoints, and runs two decisive eval sweeps.

Conditions (frac=0.10 edge-sampled sparse testbed, 5 seeds, 600ep, paired):
  node_only      — floor (no edge attention)
  hops1          — structural 1-hop (the 5p winner)
  hops2          — structural 2-hop (independent 5p signal, different seed)
  random_struct  — RANDOM edge adjacency, same pair count as hops1 (CAPACITY control:
                   if it also gets +0.02 at 5p, the benefit is "having edges", not locality)

Sweeps (eval-only on the trained checkpoints):
  TAU SWEEP: multi-hop eval at temperature τ ∈ {0.1, 0.5, 1.0, 2.0, 5.0}.
    Genuine-structural: hops1/hops2 − node_only 5p gap is stable or GROWS as τ→0 (hard traversal).
    Artifact: the gap exists only at τ=1.0 and vanishes at τ→0.

Pre-registered (symmetric): GENUINE if 5p gap survives τ→0 AND is killed by random_struct;
ARTIFACT/CAPACITY if τ=1.0-specific OR reproduced by random_struct. Either is decisive.

Usage:
  python experiments/phase76_diagnose_5p.py --smoke
  python experiments/phase76_diagnose_5p.py --sample_frac 0.10 --epochs 600 --seeds 42,123,456,7,99
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, time, json
import torch
from experiments.phase46c_link_prediction import load_lp_data, build_train_graph_tensors
from experiments.phase66_hop_ablation import build_condition_adjs, train_with_controlled_adj, _evaluate_with_adj
from experiments.phase71_sparse_hop_ablation import evaluate_multihop_extended_with_adj, QTYPES
from experiments.phase42_multihop import build_full_adjacency
from experiments.phase44_depth import generate_extended_queries, audit_extended_queries

CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def random_adjacency(n_pairs, E, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, E, (n_pairs,), generator=g)
    tgt = torch.randint(0, E, (n_pairs,), generator=g)
    m = src != tgt
    return torch.stack([src[m], tgt[m]]).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--sample_frac', type=float, default=0.10)
    ap.add_argument('--sample_seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--eval_every', type=int, default=25)
    ap.add_argument('--patience', type=int, default=0)
    ap.add_argument('--lr', type=float, default=0.003)
    ap.add_argument('--batch_size', type=int, default=4096)
    ap.add_argument('--seeds', type=str, default='42,123,456,7,99')
    ap.add_argument('--conditions', type=str, default='node_only,hops1,hops2,random_struct')
    ap.add_argument('--taus', type=str, default='0.1,0.5,1.0,2.0,5.0')
    ap.add_argument('--max_queries', type=int, default=10000)
    ap.add_argument('--out', type=str, default='phase76_output.json')
    args = ap.parse_args()
    if args.smoke:
        args.sample_frac = 0.03; args.epochs = 8; args.eval_every = 4
        args.seeds = '42'; args.conditions = 'node_only,hops1,random_struct'
        args.taus = '0.5,1.0'; args.max_queries = 800
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [int(s) for s in args.seeds.split(',')]
    conditions = [c.strip() for c in args.conditions.split(',')]
    taus = [float(t) for t in args.taus.split(',')]
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"Phase 76 DIAGNOSE_5p | device={device} conditions={conditions} seeds={seeds} taus={taus}")

    data = load_lp_data('fb15k-237', 'data', sample_mode='edge',
                        sample_frac=args.sample_frac, sample_seed=args.sample_seed, use_cache=True)
    ei, et = build_train_graph_tensors(data['train'])
    E = ei.shape[1]
    print(f"  ents={data['num_entities']} E={E} test={data['test'].shape[1]}")

    structural = build_condition_adjs(ei, device, max_adj_pairs=None)
    n1 = structural['hops1'][0].shape[1]
    adjs = dict(structural)
    adjs['random_struct'] = (random_adjacency(n1, E, device, seed=0), 1)
    print(f"  hops1 pairs={n1:,}  random_struct pairs={adjs['random_struct'][0].shape[1]:,}")

    print("\n[queries]"); queries = generate_extended_queries(data, max_queries_per_type=args.max_queries, seed=42)
    print("  " + ", ".join(f"{q}={len(queries.get(q,[]))}" for q in QTYPES))
    print("  audit:", "PASS" if not audit_extended_queries(queries, data) else "ISSUES")
    full_hr2t = build_full_adjacency(data)

    results = []
    for cond in conditions:
        adj, ch = adjs[cond]
        for seed in seeds:
            print(f"\n=== {cond} seed={seed} ===")
            ckpt = os.path.join(CKPT_DIR, f'p76_{cond}_s{seed}_f{args.sample_frac}.pt')
            model, best_val, eix, etx, elapsed, npar = train_with_controlled_adj(
                cond, adj, ch, data, args.epochs, args.lr, device, args.batch_size, seed,
                args.eval_every, args.patience)
            torch.save({'state': model.state_dict(), 'cond': cond, 'seed': seed}, ckpt)
            adj_dev = adj.to(device)
            lp = _evaluate_with_adj(model, data['test'], eix, etx, data['hr_to_tails'],
                                    data['rt_to_heads'], device, adj_dev, ch)
            row = {'condition': cond, 'seed': seed, 'lp_MRR': lp['MRR'], 'params': int(npar)}
            for tau in taus:
                mh = evaluate_multihop_extended_with_adj(model, queries, eix, etx, full_hr2t,
                                                         device, adj_dev, ch, temperature=tau)
                for q in QTYPES:
                    row[f'{q}_MRR_t{tau}'] = mh[q]['MRR']
            print("  LP=%.4f | 5p@tau: %s" % (lp['MRR'],
                  " ".join("%.2f=%.4f" % (t, row[f'5p_MRR_t{t}']) for t in taus)))
            results.append(row)
            with open(os.path.join(os.path.dirname(__file__), '..', args.out), 'w') as f:
                json.dump(results, f, indent=2)
    print("\nDONE. saved", args.out)


if __name__ == '__main__':
    main()
