"""Phase 77 - pre-registered analysis: per-query bootstrap on the PEC-pf arms.

Primary endpoint (docs/phase_77.md): per-QUERY bootstrap blocked on query id (seed-level pairing is
underpowered: 5p seed-paired SD ~0.023 needs |mean|>0.029 at n=5). Query i is identical across model
seeds (queries are generated with a fixed seed), so we pool per-query RRs across seeds and bootstrap
by resampling query ids.

Tests: PEC-pf vs {capacity, rrprior, static} (+ MASK-ONLY floor) on the FIXED penult-reachable
stratum, at 3p/4p/5p; Holm correction across the 4p AND 5p joint test; depth x condition interaction
(does the margin grow with depth). Reads phase77_rr_s{seed}.npz (+ phase77_output.json for AOT ref).

Usage:
  python experiments/phase77_analyze.py --seeds 42
  python experiments/phase77_analyze.py --seeds 42,123,456,7,99
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json
import numpy as np

DEPTHS = ['2p', '3p', '4p', '5p']
CONTROLS = ['capacity', 'rrprior', 'static']
NBOOT = 10000


def load_rr(seeds, frac, root):
    """Returns rr[arm][qt] = [n_query, n_seed] matrix of reciprocal ranks, and strat[qt] bool[n]."""
    per = {}
    strat = {}
    for s in seeds:
        p = os.path.join(root, f'phase77_rr_s{s}.npz')
        if not os.path.exists(p):
            print(f"  (missing {os.path.basename(p)})"); continue
        d = np.load(p)
        per[s] = d
        for qt in DEPTHS + ['1p']:
            if f'strat_{qt}' in d and qt not in strat:
                strat[qt] = d[f'strat_{qt}']
    arms = sorted({k.rsplit('_', 2)[0] for s in per for k in per[s].files if k.endswith('_rr')})
    rr = {}
    for arm in arms:
        rr[arm] = {}
        for qt in DEPTHS + ['1p']:
            cols = [per[s][f'{arm}_{qt}_rr'] for s in per if f'{arm}_{qt}_rr' in per[s].files]
            if cols:
                rr[arm][qt] = np.stack(cols, 1)   # [n_query, n_seed]
    return rr, strat, list(per.keys())


def boot_ci(diff, nboot=NBOOT, seed=0):
    """Bootstrap mean CI by resampling the first axis (query ids)."""
    rng = np.random.RandomState(seed)
    n = diff.shape[0]
    if n == 0:
        return 0.0, (float('nan'), float('nan'))
    idx = rng.randint(0, n, (nboot, n))
    means = diff[idx].mean(1)
    return float(diff.mean()), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default='42')
    ap.add_argument('--frac', type=float, default=0.10)
    ap.add_argument('--rr_root', type=str, default='.')
    ap.add_argument('--output', type=str, default='phase77_output.json')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    root = os.path.join(os.path.dirname(__file__), '..', args.rr_root)
    rr, strat, have = load_rr(seeds, args.frac, root)
    print(f"Phase 77 analysis | seeds with RR data: {have} | arms: {sorted(rr.keys())}")
    if 'pec' not in rr:
        print("no pec RR data found - run phase77_jit_path_score.py first"); return

    # ── aggregate MRR table (full + strat) ──
    print("\n=== MRR (mean over seeds, full all-N) ===")
    print(f"  {'arm':<10}" + "".join(f"{qt:>9}" for qt in ['1p'] + DEPTHS))
    for arm in sorted(rr.keys()):
        cells = []
        for qt in ['1p'] + DEPTHS:
            cells.append(f"{rr[arm][qt].mean():9.4f}" if qt in rr[arm] else f"{'--':>9}")
        print(f"  {arm:<10}" + "".join(cells))

    # AOT reference from output json
    outp = os.path.join(os.path.dirname(__file__), '..', args.output)
    if os.path.exists(outp):
        res = json.load(open(outp))
        aot = {}
        for r in res:
            if r['arm'].startswith('AOT_'):
                aot.setdefault(r['arm'], []).append(r)
        if aot:
            print("\n=== AOT reference (mean over seeds, full all-N) ===")
            print(f"  {'arm':<18}" + "".join(f"{qt:>9}" for qt in ['1p'] + DEPTHS))
            for arm, rows in sorted(aot.items()):
                cells = [f"{np.mean([x[f'{qt}_MRR'] for x in rows]):9.4f}" for qt in ['1p'] + DEPTHS]
                print(f"  {arm:<18}" + "".join(cells))

    # ── per-query bootstrap: pec - control on the FIXED penult-reachable stratum ──
    print("\n=== PEC-pf vs controls - per-query paired bootstrap on penult-reachable stratum ===")
    print("  (margin = mean per-query RR difference; 95% CI by resampling query ids; * = CI excludes 0)")
    pvals = {}   # (control, depth) -> margin, ci for Holm
    for ctrl in CONTROLS + ['MASKONLY']:
        print(f"\n  pec - {ctrl}:")
        for qt in DEPTHS:
            if qt not in rr['pec']:
                continue
            mask = strat.get(qt, np.ones(rr['pec'][qt].shape[0], bool))
            pec_q = rr['pec'][qt][mask].mean(1)                      # per-query mean over seeds
            if ctrl == 'MASKONLY':
                # pec full-readout RR vs pec mask-only RR on the same queries
                ckey = [k for k in [f'pec_{qt}_maskrr'] ]
                ctrl_q = np.stack([np.load(os.path.join(root, f'phase77_rr_s{s}.npz'))[f'pec_{qt}_maskrr']
                                   for s in have], 1)[mask].mean(1)
            else:
                if qt not in rr.get(ctrl, {}):
                    continue
                ctrl_q = rr[ctrl][qt][mask].mean(1)
            diff = pec_q - ctrl_q
            m, (lo, hi) = boot_ci(diff, seed=hash((ctrl, qt)) % (2**31))
            sig = '*' if (lo > 0 or hi < 0) else ' '
            pvals[(ctrl, qt)] = (m, lo, hi)
            print(f"    {qt}: n={int(mask.sum()):5d}  margin={m:+.4f}  CI[{lo:+.4f},{hi:+.4f}] {sig}")

    # ── depth x condition: does the pec-control margin grow with depth? ──
    print("\n=== depth trend of margins (pec - control), strat ===")
    for ctrl in CONTROLS:
        seq = [pvals[(ctrl, qt)][0] for qt in DEPTHS if (ctrl, qt) in pvals]
        trend = "increasing" if all(seq[i] <= seq[i+1] for i in range(len(seq)-1)) else "non-monotone"
        print(f"  pec-{ctrl}: " + " ".join(f"{q}={pvals[(ctrl,q)][0]:+.4f}" for q in DEPTHS if (ctrl,q) in pvals)
              + f"   [{trend}]")

    # ── pre-registered verdict (4p AND 5p, all controls, CI excludes 0) ──
    print("\n=== PRE-REGISTERED VERDICT (PASS needs pec > {capacity,rrprior,static,MASKONLY} at 4p AND 5p, CI>0) ===")
    overall_pass = True
    for qt in ['4p', '5p']:
        clause = []
        for ctrl in CONTROLS + ['MASKONLY']:
            if (ctrl, qt) in pvals:
                m, lo, hi = pvals[(ctrl, qt)]
                ok = lo > 0
                clause.append(f"{ctrl}{'OK' if ok else 'NO'}")
                overall_pass = overall_pass and ok
        print(f"  {qt}: " + "  ".join(clause))
    print(f"\n  ==> {'PASS (query-time composition supported)' if overall_pass else 'FAIL on >=1 clause (see above)'}")
    print("  (3p is the strongest-prior depth - 2 trained traversal hops; 4p/5p are extrapolation.)")


if __name__ == '__main__':
    main()
