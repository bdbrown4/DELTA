"""Phase 77 — JIT (PEC-pf) driver: train the query-time edge-composition arms + run the A/B vs AOT.

Frozen encoder (per seed: p76_hops1) -> nf[N,48]/ef0[E,24]. Train each JIT arm (pec/capacity/rrprior/
static) with all-N multi-label BCE on TRAIN-only chains. Eval 1p-5p with filtered ranking (full-graph
valid answers to -inf except gold) three ways: full all-N MRR, MRR on the FIXED penult-reachable
stratum, and the MASK-ONLY floor. AOT arms (node_only/hops1/random_struct p76 checkpoints) via
fast_mh_eval at matched tau. Incremental JSON save + resume. Per-query bootstrap CIs computed
post-hoc in phase77_analyze.py.

Usage:
  python experiments/phase77_jit_path_score.py --smoke
  python experiments/phase77_jit_path_score.py --seeds 42 --epochs 200          # seed-42 decision gate
  python experiments/phase77_jit_path_score.py --seeds 42,123,456,7,99 --epochs 300
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, time, json
import numpy as np
import torch
import torch.nn.functional as F

from experiments.phase46c_link_prediction import load_lp_data, build_train_graph_tensors, create_lp_model
from experiments.phase42_multihop import (build_incoming_index, build_direct_pairs,
                                          build_full_adjacency, compute_valid_answers)
from experiments.phase44_depth import generate_extended_queries
from experiments.phase71_sparse_hop_ablation import QTYPES
from experiments.phase76_diagnose_5p import fast_mh_eval, random_adjacency
from experiments.phase66_hop_ablation import build_condition_adjs
from delta.path_compose import PathComposerPF, encode_with_edges, build_R2E, build_csr, build_hr2t, MODES

CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
JIT_ARMS = ['pec', 'capacity', 'rrprior', 'static']


# ───────────────────────── train-only chain generation ─────────────────────────
def _train_chains_for_k(train, incoming, train_direct, K, n_target, rng):
    """Distinct (anchor, rel_chain, intermediates) chains with ALL K hops in TRAIN."""
    out, seen = [], set()
    Etr = train.shape[1]
    for j in rng.permutation(Etr):
        m_last, r_last, t = train[0, j].item(), train[1, j].item(), train[2, j].item()
        # (K-1) train hops ending at m_last
        stack = [([m_last], [])]
        chains = []
        for _ in range(K - 1):
            nxt = []
            for nodes, rels in stack:
                head = nodes[0]
                for (h, r) in incoming.get(head, [])[:50]:
                    nxt.append(([h] + nodes, [r] + rels))
            stack = nxt
        chains = stack
        for nodes, rels in chains:
            anchor = nodes[0]
            rel_chain = rels + [r_last]
            inter = nodes[1:]
            if anchor == t or (anchor, t) in train_direct or anchor in inter:
                continue
            if len(set(inter)) < len(inter):
                continue
            key = (anchor, tuple(rel_chain), t)
            if key in seen:
                continue
            seen.add(key)
            out.append((anchor, rel_chain, t))
            if len(out) >= n_target:
                return out
    return out


def generate_train_chains(data, k_counts, seed=42):
    """k_counts: {K: n}. K=1 uses ALL train triples (LP-scale readout calibration — without this the
    1p/readout is starved and JIT can't match AOT even where there is no composition)."""
    train = data['train']
    incoming = build_incoming_index(train)
    train_direct = build_direct_pairs(train)
    rng = np.random.RandomState(seed)
    chains = []
    for K, n_k in k_counts.items():
        if K == 1:
            chains += [(train[0, j].item(), [train[1, j].item()], train[2, j].item())
                       for j in range(train.shape[1])]
        else:
            chains += _train_chains_for_k(train, incoming, train_direct, K, n_k, rng)
    rng.shuffle(chains)
    return chains


# ───────────────────────── train / eval ─────────────────────────
def train_arm(mode, nf, ef0, ei, csr, chains, train_hr2t, num_relations, N, device,
              epochs, lr, batch, seed, label_smooth=0.1, log_every=25):
    torch.manual_seed(seed)
    model = PathComposerPF(num_relations, d_node=nf.shape[1], d_edge=ef0.shape[1], mode=mode).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    tgt_sets = [sorted(compute_valid_answers(a, rc, train_hr2t)) for (a, rc, _t) in chains]
    idx = np.arange(len(chains))
    for ep in range(epochs):
        np.random.RandomState(seed + ep).shuffle(idx)
        tot = 0.0; nb = 0
        model.train()
        for s in range(0, len(idx), batch):
            bidx = idx[s:s + batch]
            qs = [(chains[i][0], chains[i][1]) for i in bidx]
            scores = model.score_batch_vec(qs, nf, ef0, ei, csr)       # [B, N] (vectorized)
            y = torch.zeros(len(bidx), N, device=device)
            for bi, i in enumerate(bidx):
                a = tgt_sets[i]
                if a:
                    y[bi, torch.tensor(a, device=device)] = 1.0
            y = y * (1.0 - label_smooth)
            loss = F.binary_cross_entropy_with_logits(scores, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"    [{mode}] ep{ep:3d} loss={tot/max(nb,1):.4f}")
    return model


@torch.no_grad()
def eval_arm(model, queries, valid_cache, penult_masks, nf, ef0, ei, csr, N, device, bs=512):
    """Per-qtype MRR: full all-N, stratified within nonempty penult mask, and MASK-ONLY floor."""
    model.eval()
    out = {}
    for qt in QTYPES:
        qs = queries.get(qt, [])
        if not qs:
            out[qt] = {'MRR': 0.0, 'MRR_strat': 0.0, 'MRR_maskonly': 0.0, 'count': 0, 'count_strat': 0}
            continue
        rr, rr_strat, rr_mask = [], [], []
        for s0 in range(0, len(qs), bs):
            batch = qs[s0:s0 + bs]
            scores = model.score_batch_vec([(q[0], q[1]) for q in batch], nf, ef0, ei, csr)   # [B,N]
            for bi, (anchor, rel_chain, answer, _inter) in enumerate(batch):
                qi = s0 + bi
                va = valid_cache[qt][qi]
                filt = scores[bi].clone()
                for v in va:
                    if v != answer:
                        filt[v] = float('-inf')
                rank = int((filt >= filt[answer]).sum().item())
                rr.append(1.0 / max(rank, 1))
                pmask = penult_masks[qt][qi]
                mscore = pmask.float().clone()
                for v in va:
                    if v != answer:
                        mscore[v] = float('-inf')
                rr_mask.append(1.0 / max(int((mscore >= mscore[answer]).sum().item()), 1))
                if bool(pmask.any()):
                    rr_strat.append(1.0 / max(rank, 1))
        out[qt] = {'MRR': float(np.mean(rr)), 'MRR_strat': float(np.mean(rr_strat)) if rr_strat else 0.0,
                   'MRR_maskonly': float(np.mean(rr_mask)), 'count': len(qs), 'count_strat': len(rr_strat),
                   '_rr': np.asarray(rr, dtype=np.float32), '_rr_mask': np.asarray(rr_mask, dtype=np.float32)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--sample_frac', type=float, default=0.10)
    ap.add_argument('--sample_seed', type=int, default=42)
    ap.add_argument('--seeds', type=str, default='42')
    ap.add_argument('--arms', type=str, default=','.join(JIT_ARMS))
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--n_2p', type=int, default=8000)
    ap.add_argument('--n_3p', type=int, default=8000)
    ap.add_argument('--max_queries', type=int, default=10000)
    ap.add_argument('--aot_tau', type=float, default=1.0)
    ap.add_argument('--out', type=str, default='phase77_output.json')
    args = ap.parse_args()
    if args.smoke:
        args.sample_frac = 0.10; args.epochs = 20; args.seeds = '42'
        args.n_2p = 3000; args.n_3p = 3000; args.max_queries = 400; args.out = 'phase77_smoke.json'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [int(s) for s in args.seeds.split(',')]
    arms = [a.strip() for a in args.arms.split(',')]
    k_counts = {1: None, 2: args.n_2p, 3: args.n_3p}   # K=1 = ALL train triples (LP-scale)
    print(f"Phase 77 JIT/PEC-pf | device={device} seeds={seeds} arms={arms} epochs={args.epochs}")

    data = load_lp_data('fb15k-237', 'data', sample_mode='edge',
                        sample_frac=args.sample_frac, sample_seed=args.sample_seed, use_cache=True)
    ei, et = build_train_graph_tensors(data['train'])
    ei = ei.to(device); et = et.to(device)
    N, Rr = data['num_entities'], data['num_relations']
    print(f"  ents={N} rels={Rr} E={ei.shape[1]} test={data['test'].shape[1]}")
    train_hr2t = build_hr2t(data['train'])
    full_hr2t = build_full_adjacency(data)
    csr = build_csr(ei, et, Rr, device)

    print("[queries] generating + precomputing valid_cache + penult masks (shared) ...")
    t0 = time.time()
    queries = generate_extended_queries(data, max_queries_per_type=args.max_queries, seed=42)
    valid_cache = {qt: [set(compute_valid_answers(q[0], q[1], full_hr2t)) for q in queries.get(qt, [])]
                   for qt in QTYPES}
    # penult-reachable mask (train-only) per query — shared across all JIT arms
    probe = PathComposerPF(Rr).to(device)
    penult_masks = {qt: [probe.penult_reachable_mask(q[0], q[1], train_hr2t, N, device)
                         for q in queries.get(qt, [])] for qt in QTYPES}
    print("  " + ", ".join(f"{q}={len(queries.get(q,[]))}" for q in QTYPES) + f"  [{time.time()-t0:.0f}s]")

    out_path = os.path.join(os.path.dirname(__file__), '..', args.out)
    results = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {(r['arm'], r['seed']) for r in results}

    for seed in seeds:
        enc_ckpt = os.path.join(CKPT_DIR, f'p76_hops1_s{seed}_f{args.sample_frac}.pt')
        print(f"\n=== seed {seed} | frozen encoder {os.path.basename(enc_ckpt)} ===")
        enc = create_lp_model('delta_matched', N, Rr).to(device)
        enc.load_state_dict(torch.load(enc_ckpt, map_location=device)['state'])
        nf, ef0 = encode_with_edges(enc, ei, et, device)            # frozen internal nf[N,48]/ef0[E,24]
        chains = generate_train_chains(data, k_counts, seed=seed)
        print(f"  frozen nf={tuple(nf.shape)} ef0={tuple(ef0.shape)} | train_chains={len(chains)}")

        # per-query RR store (for the pre-registered per-query bootstrap) + shared strat masks
        rr_path = os.path.join(os.path.dirname(__file__), '..', f'phase77_rr_s{seed}.npz')
        rr_store = dict(np.load(rr_path)) if os.path.exists(rr_path) else {}
        for qt in QTYPES:
            k = f'strat_{qt}'
            if k not in rr_store:
                rr_store[k] = np.asarray([bool(pm.any()) for pm in penult_masks[qt]], dtype=bool)

        # AOT arms (eval-only on existing p76 checkpoints), shared valid_cache, matched tau
        if ('AOT', seed) not in done:
            nf_aot_cache = {}
            for cond in ['node_only', 'hops1', 'random_struct']:
                ck = os.path.join(CKPT_DIR, f'p76_{cond}_s{seed}_f{args.sample_frac}.pt')
                if not os.path.exists(ck):
                    continue
                am = create_lp_model('delta_matched', N, Rr).to(device)
                am.load_state_dict(torch.load(ck, map_location=device)['state'])
                adjs = dict(build_condition_adjs(ei, device, max_adj_pairs=None))
                adjs['random_struct'] = (random_adjacency(adjs['hops1'][0].shape[1], ei.shape[1], device, seed=0), 1)
                adj, ch = adjs[cond]
                mh = fast_mh_eval(am, queries, valid_cache, ei, et, adj.to(device), ch, device, args.aot_tau)
                row = {'arm': f'AOT_{cond}', 'seed': seed, 'tau': args.aot_tau}
                for qt in QTYPES:
                    row[f'{qt}_MRR'] = mh[qt]['MRR']
                results.append(row)
                print(f"  AOT_{cond}: " + " ".join(f"{qt}={mh[qt]['MRR']:.4f}" for qt in QTYPES))
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)

        for mode in arms:
            if (mode, seed) in done:
                print(f"  skip {mode} seed={seed}"); continue
            t1 = time.time()
            model = train_arm(mode, nf, ef0, ei, csr, chains, train_hr2t, Rr, N, device,
                              args.epochs, args.lr, args.batch, seed)
            res = eval_arm(model, queries, valid_cache, penult_masks, nf, ef0, ei, csr, N, device)
            row = {'arm': mode, 'seed': seed, 'train_s': time.time() - t1}
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
            np.savez(rr_path, **rr_store)
            print(f"  {mode}: " + " ".join(f"{qt}={res[qt]['MRR']:.4f}" for qt in QTYPES)
                  + f"  [{time.time()-t1:.0f}s]")
    print("\nDONE. saved", args.out)


if __name__ == '__main__':
    main()
