"""Phase 78 — Coherence-clean edge-content controls for the JIT query-time composition thesis.

REDESIGNED after a 4-lens adversarial red-team killed the first cut (docs/phase_78.md). The lesion
ladder (all are 'pec' byte-for-byte except WHICH edge content the operator sees, via a fixed
within-relation-type permutation — routing always uses the REAL graph):

  pec          — the learned edge-to-edge operator (reference).
  shuffle_full — PRIMARY control: permute the WHOLE edge identity (ej AND the W_ctx endpoints) within
                 type. Coherent full-identity swap -> pec~shuffle_full is a CLEAN refutation of
                 edge-instance use (no endpoint-channel loophole, no incoherent-triple artifact).
  shuffle      — diagnostic: permute only ej, keep W_ctx endpoints real (isolates the ej channel).
  ef0_shuffle  — sanity: permute only the type-dominated ef0 (GATE C: ~no within-relation signal) ->
                 near-noop; a large pec-ef0_shuffle gap would expose a coherence artifact.
  rrprior      — the relation-TYPE prior (for the instance_fraction = (pec-shuffle_full)/(pec-rrprior)).

Plus, from the trained pec model (free): pec_lesion (train-real / eval-shuffle_full, the distribution-
shift discriminator) and a positive-control DISPLACEMENT (how much the perm moves pec's readout scores
on the penult set — a near-zero displacement means the lesion has no lever -> a 'pec~lesion' result is
testbed-capacity, NOT a refutation) + scramble strength.

Seed is the unit of replication. n=15 (5 existing-encoder seeds + 10 fresh). All arms trained
IN-SESSION for every seed (no cross-run inheritance -> no batch effect). Resumable per (arm,seed).

Usage:
  python experiments/phase78_shuffle_control.py --smoke
  python experiments/phase78_shuffle_control.py \
      --seeds 42,123,456,7,99,11,23,5,17,31,2,8,13,29,37 --train_encoders --epochs 300
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
from experiments.phase66_hop_ablation import build_condition_adjs, train_with_controlled_adj
from experiments.phase77_jit_path_score import (generate_train_chains, train_arm, eval_arm, CKPT_DIR)
from delta.path_compose import PathComposerPF, encode_with_edges, build_csr, build_hr2t

CORE_ARMS = ['pec', 'rrprior', 'shuffle', 'shuffle_full', 'ef0_shuffle']
ALL_ARMS = CORE_ARMS + ['capacity', 'static']


def ensure_encoder(seed, data, ei, device, frac, epochs, lr, batch):
    """Path to the frozen p76_hops1 encoder for `seed`, training+saving it (hops1, no early-stop) if
    absent. New seeds use the SAME recipe as the existing checkpoints, so they are exchangeable."""
    ckpt = os.path.join(CKPT_DIR, f'p76_hops1_s{seed}_f{frac}.pt')
    if os.path.exists(ckpt):
        return ckpt
    print(f"  [encoder] fresh p76_hops1 seed={seed} ({epochs} ep, no early-stop) ...")
    # hops1 ONLY — build_condition_adjs also builds a 17.7M-pair hops=2 adj (big transient + reserved
    # cache) that the encoder never uses; that waste thrashes / OOM-crashes the 12GB GPU. Same recipe
    # as the existing p76_hops1 checkpoints (build_edge_adjacency(hops=1)), so new seeds stay exchangeable.
    from delta.graph import DeltaGraph
    _Nn, _Ee = int(ei.max().item()) + 1, ei.shape[1]
    _tmp = DeltaGraph(node_features=torch.zeros(_Nn, 1, device=device),
                      edge_features=torch.zeros(_Ee, 1, device=device), edge_index=ei.to(device))
    adj, ch = _tmp.build_edge_adjacency(hops=1), 1
    del _tmp
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model = train_with_controlled_adj('hops1', adj, ch, data, epochs, lr, device, batch, seed, 25, 0)[0]
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({'state': model.state_dict(), 'cond': 'hops1', 'seed': seed}, ckpt)
    print(f"  [encoder] saved {os.path.basename(ckpt)}")
    return ckpt


def _store_arm(rr_store, results, out_path, mode, seed, res, train_s, extra=None):
    row = {'arm': mode, 'seed': seed, 'train_s': train_s}
    for qt in QTYPES:
        row[f'{qt}_MRR'] = res[qt]['MRR']
        row[f'{qt}_MRR_strat'] = res[qt]['MRR_strat']
        row[f'{qt}_MRR_maskonly'] = res[qt]['MRR_maskonly']
        row[f'{qt}_n_strat'] = res[qt]['count_strat']
        rr_store[f'{mode}_{qt}_rr'] = res[qt]['_rr']
        rr_store[f'{mode}_{qt}_maskrr'] = res[qt]['_rr_mask']
    if extra:
        row.update(extra)
    results.append(row)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)


@torch.no_grad()
def pec_lesion_and_displacement(model, queries, valid_cache, penult_masks, nf, ef0, ei, et, csr, N, device, seed):
    """From the trained pec model: scramble strength, the pec_lesion eval (train-real / eval under the
    shuffle_full perm), and per-depth readout DISPLACEMENT (||s_pec - s_lesion||/||s_pec|| on the
    penult set). Restores the model to clean pec state. The displacement is the positive control: if it
    is ~0 the lesion cannot move ranks, so a 'pec~lesion' MRR result is testbed-capacity, not refutation."""
    model.set_edge_perm(et, seed, edge_index=ei)
    frac, rel = model.scramble_strength(nf, ef0, ei)
    model.mode = 'shuffle_full'
    res_l = eval_arm(model, queries, valid_cache, penult_masks, nf, ef0, ei, csr, N, device)
    disp = {}
    for qt in QTYPES:
        qs = queries.get(qt, [])
        idxs = [i for i in range(len(qs)) if bool(penult_masks[qt][i].any())][:400]
        if not idxs:
            disp[qt] = 0.0; continue
        batch = [(qs[i][0], qs[i][1]) for i in idxs]
        model.mode = 'pec';          s_pec = model.score_batch_vec(batch, nf, ef0, ei, csr)
        model.mode = 'shuffle_full'; s_les = model.score_batch_vec(batch, nf, ef0, ei, csr)
        rels = []
        for bi, i in enumerate(idxs):
            m = penult_masks[qt][i]
            a = s_pec[bi][m]; b = s_les[bi][m]
            rels.append(float((a - b).norm() / (a.norm() + 1e-9)))
        disp[qt] = float(np.mean(rels))
    model.mode = 'pec'; model.edge_perm = None; model.ctx_src = None; model.ctx_tgt = None
    return frac, rel, disp, res_l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--sample_frac', type=float, default=0.10)
    ap.add_argument('--sample_seed', type=int, default=42)
    ap.add_argument('--seeds', type=str, default='42,123,456,7,99')
    ap.add_argument('--arms', type=str, default=','.join(CORE_ARMS))
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--enc_epochs', type=int, default=600)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--n_2p', type=int, default=8000)
    ap.add_argument('--n_3p', type=int, default=8000)
    ap.add_argument('--max_queries', type=int, default=10000)
    ap.add_argument('--train_encoders', action='store_true', help='train a fresh p76_hops1 encoder for seeds lacking one')
    ap.add_argument('--force', action='store_true', help='retrain arms even if already in the rr store')
    ap.add_argument('--out', type=str, default='phase78_output.json')
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 12; args.seeds = '42'; args.arms = 'pec,shuffle,shuffle_full,ef0_shuffle,rrprior'
        args.n_2p = 3000; args.n_3p = 3000; args.max_queries = 400; args.out = 'phase78_smoke.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [int(s) for s in args.seeds.split(',')]
    arms = [a.strip() for a in args.arms.split(',')]
    assert all(a in ALL_ARMS for a in arms), arms
    k_counts = {1: None, 2: args.n_2p, 3: args.n_3p}      # K=1 = ALL train triples (LP-scale readout)
    root = os.path.join(os.path.dirname(__file__), '..')
    print(f"Phase 78 (redesigned) | device={device} seeds={seeds} arms={arms} epochs={args.epochs}")

    data = load_lp_data('fb15k-237', 'data', sample_mode='edge',
                        sample_frac=args.sample_frac, sample_seed=args.sample_seed, use_cache=True)
    ei, et = build_train_graph_tensors(data['train'])
    ei = ei.to(device); et = et.to(device)
    N, Rr = data['num_entities'], data['num_relations']
    print(f"  ents={N} rels={Rr} E={ei.shape[1]} test={data['test'].shape[1]}")
    train_hr2t = build_hr2t(data['train'])
    full_hr2t = build_full_adjacency(data)
    csr = build_csr(ei, et, Rr, device)

    print("[queries] generating + precomputing valid_cache + penult masks (shared, fixed seed=42) ...")
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
        enc_ckpt = ensure_encoder(seed, data, ei, device, args.sample_frac,
                                  args.enc_epochs, args.lr, 4096) if args.train_encoders \
            else os.path.join(CKPT_DIR, f'p76_hops1_s{seed}_f{args.sample_frac}.pt')
        if not os.path.exists(enc_ckpt):
            print(f"  !! seed {seed}: missing encoder {os.path.basename(enc_ckpt)} (use --train_encoders); skipping")
            continue
        print(f"\n=== seed {seed} | frozen encoder {os.path.basename(enc_ckpt)} ===")
        enc = create_lp_model('delta_matched', N, Rr).to(device)
        enc.load_state_dict(torch.load(enc_ckpt, map_location=device)['state'])
        nf, ef0 = encode_with_edges(enc, ei, et, device)
        chains = generate_train_chains(data, k_counts, seed=seed)
        print(f"  frozen nf={tuple(nf.shape)} ef0={tuple(ef0.shape)} | train_chains={len(chains)}")

        # fresh per seed (resume from phase78's own rr only — NO cross-run inheritance)
        rr_path = os.path.join(root, f'phase78_rr_s{seed}.npz')
        rr_store = dict(np.load(rr_path)) if os.path.exists(rr_path) else {}
        for qt in QTYPES:
            k = f'strat_{qt}'
            if k not in rr_store:
                rr_store[k] = np.asarray([bool(pm.any()) for pm in penult_masks[qt]], dtype=bool)

        for mode in arms:
            if f'{mode}_3p_rr' in rr_store and not args.force:
                print(f"  skip {mode} (already in rr store)"); continue
            t1 = time.time()
            model = train_arm(mode, nf, ef0, ei, csr, chains, train_hr2t, Rr, N, device,
                              args.epochs, args.lr, args.batch, seed, et=et)
            res = eval_arm(model, queries, valid_cache, penult_masks, nf, ef0, ei, csr, N, device)
            extra = None
            if mode == 'pec':
                frac, rel, disp, res_l = pec_lesion_and_displacement(
                    model, queries, valid_cache, penult_masks, nf, ef0, ei, et, csr, N, device, seed)
                extra = {'scramble_frac': frac, 'scramble_rel': rel, **{f'disp_{qt}': disp[qt] for qt in QTYPES}}
            _store_arm(rr_store, results, out_path, mode, seed, res, time.time() - t1, extra)
            print(f"  {mode}: " + " ".join(f"{qt}={res[qt]['MRR']:.4f}" for qt in QTYPES) + f"  [{time.time()-t1:.0f}s]")
            if mode == 'pec':
                _store_arm(rr_store, results, out_path, 'pec_lesion', seed, res_l, 0.0)
                print(f"  pec_lesion(full): " + " ".join(f"{qt}={res_l[qt]['MRR']:.4f}" for qt in QTYPES)
                      + f" | scramble frac={frac:.2f} rel={rel:.3f} | disp " + " ".join(f"{qt}={disp[qt]:.3f}" for qt in QTYPES))
            np.savez(rr_path, **rr_store)

    print("\nDONE. saved", args.out)


if __name__ == '__main__':
    main()
