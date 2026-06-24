"""Phase 79 — Is the 'JIT ~2x AOT' gain the trained readout or the path mechanism?

Trains, per seed, on the SAME frozen 48-d nf JIT uses (p76_hops1), three paired arms + an untrained floor,
all judged on the SAME queries / valid_cache / penult-strat (eval_arm contract from Phase 77):

  aot_readout : node-only z (seed_norm(W_seed(nf[e]))) + trained pooled readout, NO edges. PRIMARY control.
  pec         : full PEC-pf (edge-composed z + same readout).        gap_path = pec - aot_readout.
  static      : edge-as-INPUT, composition OFF, params matched.      conjunctive guard for the +ve branch.
  aotsoft48   : the SAME aot_readout module left UNTRAINED (random init, frozen, eval-only) = the shared-
                space floor.  gap_head = aot_readout - aotsoft48 (the value of training the head).

K=1 identity (verified in tests): at 1p aot_readout == pec by construction (no traversal) — the 1p plumbing
gate. Seed = replication unit; n=15; reuses the frozen p76_hops1 encoders (NO new encoder training).
Resumable per (arm, seed). See docs/phase_79.md.

Usage:
  python experiments/phase79_readout_control.py --smoke
  python experiments/phase79_readout_control.py --seeds 42,123,456,7,99,11,23,5,17,31,2,8,13,29,37 --epochs 300
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, time, json
import numpy as np
import torch

from experiments.phase46c_link_prediction import load_lp_data, build_train_graph_tensors, create_lp_model
from experiments.phase42_multihop import build_full_adjacency, compute_valid_answers
from experiments.phase44_depth import generate_extended_queries
from experiments.phase71_sparse_hop_ablation import QTYPES
from experiments.phase77_jit_path_score import generate_train_chains, train_arm, eval_arm, CKPT_DIR
from delta.path_compose import PathComposerPF, encode_with_edges, build_csr, build_hr2t
from delta.aot_readout import AOTReadout

TRAINED_ARMS = ['aot_readout', 'pec', 'static']      # paired, trained in-session
FLOOR_ARM = 'aotsoft48'                               # untrained AOTReadout, eval-only
ALL_ARMS = TRAINED_ARMS + [FLOOR_ARM]


def _store_arm(rr_store, results, out_path, mode, seed, res, train_s):
    row = {'arm': mode, 'seed': seed, 'train_s': train_s}
    for qt in QTYPES:
        row[f'{qt}_MRR'] = res[qt]['MRR']
        row[f'{qt}_MRR_strat'] = res[qt]['MRR_strat']
        row[f'{qt}_MRR_maskonly'] = res[qt]['MRR_maskonly']
        row[f'{qt}_n_strat'] = res[qt]['count_strat']
        rr_store[f'{mode}_{qt}_rr'] = res[qt]['_rr']
        rr_store[f'{mode}_{qt}_maskrr'] = res[qt]['_rr_mask']
    results.append(row)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--sample_frac', type=float, default=0.10)
    ap.add_argument('--sample_seed', type=int, default=42)
    ap.add_argument('--seeds', type=str, default='42,123,456,7,99,11,23,5,17,31,2,8,13,29,37')
    ap.add_argument('--arms', type=str, default=','.join(ALL_ARMS))
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--n_2p', type=int, default=8000)
    ap.add_argument('--n_3p', type=int, default=8000)
    ap.add_argument('--max_queries', type=int, default=10000)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--out', type=str, default='phase79_output.json')
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 12; args.seeds = '42'; args.arms = ','.join(ALL_ARMS)
        args.n_2p = 3000; args.n_3p = 3000; args.max_queries = 400; args.out = 'phase79_smoke.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [int(s) for s in args.seeds.split(',')]
    arms = [a.strip() for a in args.arms.split(',')]
    assert all(a in ALL_ARMS for a in arms), arms
    k_counts = {1: None, 2: args.n_2p, 3: args.n_3p}      # K=1 = ALL train triples (LP-scale readout)
    root = os.path.join(os.path.dirname(__file__), '..')
    print(f"Phase 79 readout-control | device={device} seeds={seeds} arms={arms} epochs={args.epochs}")

    data = load_lp_data('fb15k-237', 'data', sample_mode='edge',
                        sample_frac=args.sample_frac, sample_seed=args.sample_seed, use_cache=True)
    ei, et = build_train_graph_tensors(data['train'])
    ei = ei.to(device); et = et.to(device)
    N, Rr = data['num_entities'], data['num_relations']
    print(f"  ents={N} rels={Rr} E={ei.shape[1]} test={data['test'].shape[1]}")
    train_hr2t = build_hr2t(data['train'])
    full_hr2t = build_full_adjacency(data)
    csr = build_csr(ei, et, Rr, device)

    print("[queries] generating + valid_cache + penult masks (shared, fixed seed=42) ...")
    t0 = time.time()
    queries = generate_extended_queries(data, max_queries_per_type=args.max_queries, seed=42)
    valid_cache = {qt: [set(compute_valid_answers(q[0], q[1], full_hr2t)) for q in queries.get(qt, [])]
                   for qt in QTYPES}
    probe = PathComposerPF(Rr).to(device)
    penult_masks = {qt: [probe.penult_reachable_mask(q[0], q[1], train_hr2t, N, device)
                         for q in queries.get(qt, [])] for qt in QTYPES}
    print("  " + ", ".join(f"{q}={len(queries.get(q,[]))}" for q in QTYPES) + f"  [{time.time()-t0:.0f}s]")

    out_path = os.path.join(root, args.out)
    results = json.load(open(out_path)) if os.path.exists(out_path) else []

    for seed in seeds:
        enc_ckpt = os.path.join(CKPT_DIR, f'p76_hops1_s{seed}_f{args.sample_frac}.pt')
        if not os.path.exists(enc_ckpt):
            print(f"  !! seed {seed}: missing {os.path.basename(enc_ckpt)}; skipping"); continue
        print(f"\n=== seed {seed} | frozen encoder {os.path.basename(enc_ckpt)} ===")
        enc = create_lp_model('delta_matched', N, Rr).to(device)
        enc.load_state_dict(torch.load(enc_ckpt, map_location=device)['state'])
        nf, ef0 = encode_with_edges(enc, ei, et, device)
        chains = generate_train_chains(data, k_counts, seed=seed)
        print(f"  frozen nf={tuple(nf.shape)} ef0={tuple(ef0.shape)} | train_chains={len(chains)}")

        rr_path = os.path.join(root, f'phase79_rr_s{seed}.npz')
        rr_store = dict(np.load(rr_path)) if os.path.exists(rr_path) else {}
        for qt in QTYPES:
            k = f'strat_{qt}'
            if k not in rr_store:
                rr_store[k] = np.asarray([bool(pm.any()) for pm in penult_masks[qt]], dtype=bool)

        for mode in arms:
            if f'{mode}_3p_rr' in rr_store and not args.force:
                print(f"  skip {mode} (already in rr store)"); continue
            t1 = time.time()
            if mode == FLOOR_ARM:                          # untrained AOTReadout = shared-space floor
                torch.manual_seed(seed)
                model = AOTReadout(Rr, d_node=nf.shape[1], d_edge=ef0.shape[1]).to(device)
                model.eval()
            else:
                model = train_arm(mode, nf, ef0, ei, csr, chains, train_hr2t, Rr, N, device,
                                  args.epochs, args.lr, args.batch, seed, et=et)
            res = eval_arm(model, queries, valid_cache, penult_masks, nf, ef0, ei, csr, N, device)
            npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
            _store_arm(rr_store, results, out_path, mode, seed, res, time.time() - t1)
            np.savez(rr_path, **rr_store)
            print(f"  {mode} (train_params={npar}): " + " ".join(f"{qt}={res[qt]['MRR']:.4f}" for qt in QTYPES)
                  + f"  [{time.time()-t1:.0f}s]")

    print("\nDONE. saved", args.out)


if __name__ == '__main__':
    main()
