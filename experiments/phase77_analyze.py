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

    # helper: per-seed strat margin matrix (pec - control), shape [n_seed]
    def seed_margins(ctrl, qt):
        mask = strat.get(qt)
        ms = []
        for j, s in enumerate(have):
            d = np.load(os.path.join(root, f'phase77_rr_s{s}.npz'))
            pec = d[f'pec_{qt}_rr'][mask]
            ctl = d[f'pec_{qt}_maskrr'][mask] if ctrl == 'MASKONLY' else d[f'{ctrl}_{qt}_rr'][mask]
            ms.append(float((pec - ctl).mean()))
        return np.array(ms)

    # ── PRIMARY (CORRECTED): seed is the unit of replication, NOT the query. The per-query bootstrap
    #    below treats ~8.5k cross-seed-correlated queries as iid and understates variance ~10-15x; the
    #    honest test resamples/contrasts the 5 independent seed trainings. ──
    from scipy import stats as _st
    print("\n=== PRIMARY: SEED-LEVEL paired test (n_seed={}, the correct replication unit) ===".format(len(have)))
    print("  margin = per-seed mean strat RR diff; t-test across seeds (sig p<0.05); per-seed shows fragility")
    seedp = {}
    for ctrl in CONTROLS + ['MASKONLY']:
        print(f"\n  pec - {ctrl}:")
        for qt in DEPTHS:
            if qt not in rr['pec']:
                continue
            m = seed_margins(ctrl, qt)
            t, p = _st.ttest_1samp(m, 0.0) if m.std() > 0 else (float('inf'), 0.0)
            # seed-cluster bootstrap CI (resample seeds with replacement)
            rng = np.random.RandomState(7)
            bs = np.array([m[rng.randint(0, len(m), len(m))].mean() for _ in range(20000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            sig = '*' if (p < 0.05 and lo > 0) else ' '
            seedp[(ctrl, qt)] = (float(m.mean()), p, lo, hi)
            ps = " ".join(f"{x:+.3f}" for x in m)
            print(f"    {qt}: mean={m.mean():+.4f} t-p={p:.3f} clusterCI[{lo:+.4f},{hi:+.4f}] {sig}  per-seed[{ps}]")

    print("\n=== CORRECTED VERDICT (PASS needs pec > {capacity,rrprior,static,MASKONLY} at 4p AND 5p, seed-level) ===")
    overall_pass = True
    for qt in ['4p', '5p']:
        clause = []
        for ctrl in CONTROLS + ['MASKONLY']:
            if (ctrl, qt) in seedp:
                _, p, lo, hi = seedp[(ctrl, qt)]
                ok = (p < 0.05 and lo > 0)
                clause.append(f"{ctrl}{'OK' if ok else 'NO'}")
                overall_pass = overall_pass and ok
        print(f"  {qt}: " + "  ".join(clause))
    print(f"\n  ==> {'PASS' if overall_pass else 'FAIL (query-time composition NOT robustly supported at seed level)'}")

    # ── SECONDARY (INVALID as inference — shown only to expose the inflation): per-query bootstrap ──
    print("\n=== SECONDARY: per-query bootstrap (INVALID PRIMARY — understates variance ~10-15x; for comparison) ===")
    for ctrl in CONTROLS + ['MASKONLY']:
        cells = []
        for qt in ['4p', '5p']:
            if qt not in rr['pec']:
                continue
            mask = strat.get(qt)
            pec_q = rr['pec'][qt][mask].mean(1)
            ctrl_q = (np.stack([np.load(os.path.join(root, f'phase77_rr_s{s}.npz'))[f'pec_{qt}_maskrr']
                                for s in have], 1)[mask].mean(1) if ctrl == 'MASKONLY' else rr[ctrl][qt][mask].mean(1))
            m, (lo, hi) = boot_ci(pec_q - ctrl_q, seed=hash((ctrl, qt)) % (2**31))
            cells.append(f"{qt} {m:+.4f}[{lo:+.4f},{hi:+.4f}]{'*' if lo>0 else ' '}")
        print(f"  pec-{ctrl}: " + "   ".join(cells))


if __name__ == '__main__':
    main()
