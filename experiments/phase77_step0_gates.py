"""Phase 77 — Step 0 FREE GATES for the JIT (query-time edge composition) pivot.

NO TRAINING. Three pre-registered abort gates on the real edge-sampled FB15k-237 testbed.
If any gate fails we learn the testbed cannot support a fair instance-level composition test
*before* writing/training the PEC-pf model — a clean finding for ~minutes of compute.

GATES (see docs/phase_77.md for the full pre-registration):
  A. Penult-frontier non-empty rate. PEC-pf traverses hops 1..K-1 over TRAIN edges then applies
     r_K as an all-N readout. For the readout to have anything to compose, the penultimate frontier
     (entities reached after K-1 train hops from the anchor) must be non-empty. PASS if ~100% at
     every depth. (Also reports the OLD full-chain gold-reachability rate that KILLED plain PEC, as
     a sanity cross-check against the design's verified 0/3.3/9.9/17.7/21.3%.)
  B. Usable TRAIN chains per K. The matching objective trains on TRAIN-only chains (all K hops in
     train). PASS if >=2000 distinct usable chains for K in {2,3} (penult non-empty AND |train-only
     answer set| >= 1). Fewer -> drop the depth-increasing claim for that K.
  C. ej instance-vs-type variance ratio. The edge state the DP composes is
     ej = LayerNorm(W_ein([ef0 ; nf_src ; nf_tgt])). ef0 = edge_proj_in(edge_rel_emb(r)) is a pure
     RELATION-TYPE function (identical for all edges of a relation -> zero within-relation variance).
     The endpoint entity features nf_src/nf_tgt are the ONLY instance signal. PASS if the
     within-relation variance fraction of ej is non-negligible; if ej is type-dominated, PEC-pf is
     identical to the JIT-RRprior control BY CONSTRUCTION and the test is unidentifiable -> ABORT.

Usage:
  python experiments/phase77_step0_gates.py
  python experiments/phase77_step0_gates.py --sample_frac 0.10 --ckpt checkpoints/p76_hops1_s42_f0.1.pt
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, time, json
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

from experiments.phase46c_link_prediction import load_lp_data, build_train_graph_tensors, create_lp_model
from experiments.phase42_multihop import build_incoming_index, build_direct_pairs, compute_valid_answers
from experiments.phase44_depth import generate_extended_queries
from experiments.phase71_sparse_hop_ablation import QTYPES


def build_hr2t(triples):
    """(h, r) -> set(tails) over the given triples ONLY (used train-only here)."""
    hr2t = defaultdict(set)
    for i in range(triples.shape[1]):
        hr2t[(triples[0, i].item(), triples[1, i].item())].add(triples[2, i].item())
    return dict(hr2t)


def traverse(anchor, rels, hr2t):
    """Apply the relation list `rels` from {anchor} over hr2t; return the reached set (or empty)."""
    cur = {anchor}
    for r in rels:
        nxt = set()
        for e in cur:
            nxt |= hr2t.get((e, r), set())
        cur = nxt
        if not cur:
            return cur
    return cur


# ───────────────────────── Gate A ─────────────────────────
def gate_a(queries, train_hr2t):
    print("\n=== GATE A: penult-frontier non-empty rate (PEC-pf) + old full-chain gold reach (killed plain PEC) ===")
    rows = {}
    ok = True
    for qt in QTYPES:
        qs = queries.get(qt, [])
        if not qs:
            continue
        K = len(qs[0][1])
        penult_nonempty = 0
        gold_reach_old = 0  # answer reachable by ALL-K train-only typed traversal (the broken PEC scheme)
        for (anchor, rels, answer, _inter) in qs:
            # penult frontier = first K-1 train hops (for 1p K=1 -> 0 hops -> {anchor})
            penult = {anchor} if K == 1 else traverse(anchor, rels[:K - 1], train_hr2t)
            if penult:
                penult_nonempty += 1
            full = traverse(anchor, rels, train_hr2t)  # all K hops over train (old PEC)
            if answer in full:
                gold_reach_old += 1
        n = len(qs)
        pr = penult_nonempty / n
        gr = gold_reach_old / n
        rows[qt] = {'n': n, 'penult_nonempty_rate': pr, 'old_full_chain_gold_reach': gr}
        flag = "OK" if pr > 0.99 else ("low" if pr > 0.90 else "FAIL")
        if pr <= 0.99:
            ok = ok and (pr > 0.999)  # only truly-100% passes hard; near-100 noted
        print(f"  {qt}: n={n:5d}  penult_nonempty={pr:6.3%}  [{flag}]   (old full-chain gold-reach={gr:6.3%})")
    print(f"  -> GATE A {'PASS' if all(r['penult_nonempty_rate'] > 0.99 for r in rows.values()) else 'CHECK'}"
          f" (need ~100% penult non-empty so the readout always has a frontier to compose)")
    return rows


# ───────────────────────── Gate B ─────────────────────────
def _build_train_chains(target, depth, incoming, max_fan=50):
    """Chains of TRAIN edges of length `depth` ending at target. Returns list of (nodes, rels)."""
    if depth == 0:
        return [([target], [])]
    parents = incoming.get(target, [])
    if len(parents) > max_fan:
        parents = parents[:max_fan]
    if depth == 1:
        return [([h, target], [r]) for h, r in parents]
    out = []
    for parent, rel in parents:
        for nodes, rels in _build_train_chains(parent, depth - 1, incoming, max_fan):
            out.append((nodes + [target], rels + [rel]))
            if len(out) > max_fan * max_fan:
                return out
    return out


def gate_b(data, train_hr2t, target_per_k=2000, seed=42):
    print(f"\n=== GATE B: usable TRAIN-only chains per K (target >= {target_per_k} for K in {{2,3}}) ===")
    train = data['train']
    incoming = build_incoming_index(train)          # entity -> [(h, r)] over TRAIN
    train_direct = build_direct_pairs(train)
    rng = np.random.RandomState(seed)
    # final TRAIN edges to anchor each chain (sample to bound cost)
    Etr = train.shape[1]
    idx = rng.permutation(Etr)
    rows = {}
    ok = True
    for K in [2, 3, 4]:
        usable = set()        # distinct (anchor, tuple(rels), answer)
        penult_nonempty = 0
        considered = 0
        for j in idx:
            m_last, r_last, t = train[0, j].item(), train[1, j].item(), train[2, j].item()
            chains = _build_train_chains(m_last, K - 1, incoming)  # (K-1) train hops ending at m_last
            for nodes, rels in chains:
                anchor = nodes[0]
                rel_chain = rels + [r_last]
                intermediates = nodes[1:]
                # same leak/degeneracy filters as generate_extended_queries
                if anchor == t or (anchor, t) in train_direct or anchor in intermediates:
                    continue
                if len(set(intermediates)) < len(intermediates):
                    continue
                considered += 1
                A = compute_valid_answers(anchor, rel_chain, train_hr2t)  # train-only answer set
                if not A:
                    continue
                penult = {anchor} if K == 1 else traverse(anchor, rel_chain[:K - 1], train_hr2t)
                if not penult:
                    continue
                penult_nonempty += 1
                usable.add((anchor, tuple(rel_chain), t))
                if len(usable) >= target_per_k * 3:   # plenty; stop early
                    break
            if len(usable) >= target_per_k * 3:
                break
        rows[K] = {'usable_distinct': len(usable), 'considered': considered,
                   'penult_nonempty_among_considered': penult_nonempty}
        passed = len(usable) >= target_per_k
        if K in (2, 3):
            ok = ok and passed
        print(f"  K={K}: usable_distinct_chains={len(usable):6d} (considered {considered})  "
              f"[{'PASS' if passed else 'FAIL'}{'' if K in (2,3) else ' (informational)'}]")
    print(f"  -> GATE B {'PASS' if ok else 'FAIL'}")
    return rows, ok


# ───────────────────────── Gate C ─────────────────────────
@torch.no_grad()
def gate_c(data, ei, et, ckpt_path, device, seed=42):
    print(f"\n=== GATE C: ej instance-vs-type variance ratio (ckpt={os.path.basename(ckpt_path)}) ===")
    model = create_lp_model('delta_matched', data['num_entities'], data['num_relations']).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['state'])
    model.eval()
    nf = model.encode(ei.to(device), et.to(device)).detach()           # [N, 64] AOT node table
    # ef0 = encoder edge input = edge_proj_in(edge_rel_emb(r)) -> pure relation-type [E, 24]
    er = model.edge_rel_emb(et.to(device))                             # [E, 32]
    ef0 = model.edge_proj_in(er) if hasattr(model, 'edge_proj_in') and model.edge_proj_in is not None else er
    src = ei[0].to(device); tgt = ei[1].to(device)
    nf_src, nf_tgt = nf[src], nf[tgt]                                  # [E, 64] each
    din = ef0.shape[1] + nf.shape[1] * 2
    g = torch.Generator(device='cpu').manual_seed(seed)
    W = nn.Linear(din, 24).to(device)
    with torch.no_grad():
        W.weight.copy_(torch.randn(24, din, generator=g).to(device) / (din ** 0.5))
        W.bias.zero_()
    ln = nn.LayerNorm(24).to(device)

    def ej_of(use_endpoints):
        if use_endpoints:
            x = torch.cat([ef0, nf_src, nf_tgt], -1)
        else:
            x = torch.cat([ef0, torch.zeros_like(nf_src), torch.zeros_like(nf_tgt)], -1)
        return ln(W(x))                                               # [E, 24]

    def within_between(ej, rels):
        """within = mean over relations of per-relation feature variance (avg over dims), weighted by count;
           between = variance across per-relation MEANS (avg over dims). Returns (within, between, frac)."""
        rels = rels.cpu().numpy()
        ej = ej.cpu().numpy()
        R = int(rels.max()) + 1
        order = np.argsort(rels, kind='stable')
        rs = rels[order]; es = ej[order]
        bounds = np.searchsorted(rs, np.arange(R + 1))
        means = []; counts = []; within_acc = 0.0; tot = 0
        for r in range(R):
            a, b = bounds[r], bounds[r + 1]
            if b - a <= 0:
                continue
            block = es[a:b]
            m = block.mean(0)
            means.append(m); counts.append(b - a)
            if b - a >= 2:
                within_acc += block.var(0).mean() * (b - a)
                tot += (b - a)
        means = np.array(means); counts = np.array(counts)
        within = within_acc / max(tot, 1)
        # between = weighted variance of per-relation means across dims
        gm = (means * counts[:, None]).sum(0) / counts.sum()
        between = ((means - gm) ** 2 * counts[:, None]).sum(0).sum() / counts.sum() / means.shape[1]
        frac = within / (within + between + 1e-12)
        return within, between, frac

    ej_full = ej_of(True)
    ej_type = ej_of(False)
    w_full, b_full, frac_full = within_between(ej_full, et)
    w_type, b_type, frac_type = within_between(ej_type, et)
    print(f"  type-only ej (ef0 only):     within={w_type:.5f} between={b_type:.5f} within_frac={frac_type:.4f}"
          f"   (expect within~0 by construction)")
    print(f"  full ej (+endpoint inject):  within={w_full:.5f} between={b_full:.5f} within_frac={frac_full:.4f}")
    THRESH = 0.15
    passed = frac_full >= THRESH
    print(f"  -> instance within-relation variance fraction = {frac_full:.4f} "
          f"(threshold {THRESH}). GATE C {'PASS' if passed else 'FAIL (ej type-dominated -> PEC-pf == RRprior; ABORT)'}")
    return {'within_full': float(w_full), 'between_full': float(b_full), 'within_frac_full': float(frac_full),
            'within_type': float(w_type), 'within_frac_type': float(frac_type), 'threshold': THRESH,
            'passed': bool(passed)}, passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample_frac', type=float, default=0.10)
    ap.add_argument('--sample_seed', type=int, default=42)
    ap.add_argument('--max_queries', type=int, default=10000)
    ap.add_argument('--ckpt', type=str, default='checkpoints/p76_hops1_s42_f0.1.pt')
    ap.add_argument('--target_per_k', type=int, default=2000)
    ap.add_argument('--out', type=str, default='phase77_step0_gates.json')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    print(f"Phase 77 Step-0 gates | device={device} sample_frac={args.sample_frac}")

    data = load_lp_data('fb15k-237', 'data', sample_mode='edge',
                        sample_frac=args.sample_frac, sample_seed=args.sample_seed, use_cache=True)
    ei, et = build_train_graph_tensors(data['train'])
    print(f"  ents={data['num_entities']} rels={data['num_relations']} E={ei.shape[1]} test={data['test'].shape[1]}")
    train_hr2t = build_hr2t(data['train'])

    print("\n[queries] generating 1p-5p (leak-checked) ...")
    queries = generate_extended_queries(data, max_queries_per_type=args.max_queries, seed=42)
    print("  " + ", ".join(f"{q}={len(queries.get(q, []))}" for q in QTYPES))

    a_rows = gate_a(queries, train_hr2t)
    b_rows, b_ok = gate_b(data, train_hr2t, target_per_k=args.target_per_k)
    ckpt = os.path.join(os.path.dirname(__file__), '..', args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt
    c_row, c_ok = gate_c(data, ei, et, ckpt, device)

    a_ok = all(r['penult_nonempty_rate'] > 0.99 for r in a_rows.values())
    verdict = a_ok and b_ok and c_ok
    print("\n" + "=" * 70)
    print(f"GATE A (penult non-empty)      : {'PASS' if a_ok else 'CHECK'}")
    print(f"GATE B (>= {args.target_per_k} train chains K2,K3): {'PASS' if b_ok else 'FAIL'}")
    print(f"GATE C (ej instance signal)    : {'PASS' if c_ok else 'FAIL'}")
    print(f"OVERALL: {'PROCEED to build PEC-pf' if verdict else 'ABORT / revisit design'}")
    print(f"[{time.time() - t0:.0f}s]")

    out = {'gate_a': a_rows, 'gate_b': {str(k): v for k, v in b_rows.items()}, 'gate_c': c_row,
           'a_ok': bool(a_ok), 'b_ok': bool(b_ok), 'c_ok': bool(c_ok), 'verdict_proceed': bool(verdict)}
    out_path = os.path.join(os.path.dirname(__file__), '..', args.out)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print("saved", args.out)


if __name__ == '__main__':
    main()
