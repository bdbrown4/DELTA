"""Phase 78 analysis — seed-level de-confounding of the query-time composition thesis (REDESIGNED).

Unit of replication = the SEED (Phase-77's per-query bootstrap understated variance ~10-15x). PRIMARY
confirmatory test (pre-registered, docs/phase_78.md): pec vs shuffle_full at the only TRAINED depth 3p,
paired-t. shuffle_full is the coherence-clean control (permutes the whole edge identity — ej AND the
W_ctx endpoints — within relation type), so it closes the endpoint loophole that plain 'shuffle' leaves.

Decision rules (no bare point-estimate gate — that was the killed v1):
  CONFIRM   : pec > shuffle_full at 3p (paired-t p<0.05 AND 95% CI excludes 0). Instance content used.
  REFUTE    : TOST — paired-t (1-2alpha) CI lies entirely within +-delta AND the lesion has a LEVER
              (median readout displacement >= disp_min). Equivalence, honestly earned.
  NO-LEVER  : equivalent BUT displacement < disp_min -> testbed-capacity INCONCLUSIVE, NOT a refutation.
  INCONCLUSIVE: CI wider than +-delta -> underpowered; never 'refuted'.
Plus instance_fraction = (pec-shuffle_full)/(pec-rrprior): >0.5 instance load-bearing, <0.2 type-prior.
Diagnostics (exploratory): shuffle (ej channel only), ef0_shuffle (sanity), pec_lesion (distribution-
shift discriminator), 4p/5p (extrapolation).

Usage:
  python experiments/phase78_analyze.py --seeds 42,123,456,7,99,11,23,5,17,31,2,8,13,29,37
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json
import numpy as np
from scipy import stats as st

DEPTHS = ['2p', '3p', '4p', '5p']
PRIMARY = 'shuffle_full'        # the coherence-clean control
LADDER = ['shuffle_full', 'shuffle', 'ef0_shuffle', 'rrprior', 'pec_lesion']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default='42,123,456,7,99')
    ap.add_argument('--rr_root', type=str, default='.')
    ap.add_argument('--out', type=str, default='phase78_output.json')
    ap.add_argument('--delta', type=float, default=0.015,
                    help='pre-registered TOST equivalence margin (NOT a point-estimate threshold)')
    ap.add_argument('--disp_min', type=float, default=0.02,
                    help='min median readout displacement for the lesion to count as having a lever')
    ap.add_argument('--final_n', type=int, default=15,
                    help='pre-registered seed count; below this the verdict is labeled INTERIM (anti-peeking)')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    root = os.path.join(os.path.dirname(__file__), '..', args.rr_root)

    data, strat = {}, {}
    for s in seeds:
        p = os.path.join(root, f'phase78_rr_s{s}.npz')
        if not os.path.exists(p):
            print(f"  (missing {os.path.basename(p)})"); continue
        d = np.load(p)
        if f'{PRIMARY}_3p_rr' not in d.files:
            print(f"  (seed {s}: no {PRIMARY} arm yet)"); continue
        data[s] = d
        for qt in DEPTHS + ['1p']:
            if f'strat_{qt}' in d.files and qt not in strat:
                strat[qt] = d[f'strat_{qt}']
    have = list(data.keys())
    if not have:
        print(f"no seeds with a {PRIMARY} arm — run phase78_shuffle_control.py first"); return
    arms = sorted({k.rsplit('_', 2)[0] for s in have for k in data[s].files if k.endswith('_rr')})
    print(f"Phase 78 analysis | seeds={have} (n={len(have)}) | arms={arms}")

    # displacement / scramble (positive control) from the pec rows of the output json
    disp = {qt: [] for qt in DEPTHS}; scr = []
    op = os.path.join(root, args.out)
    if os.path.exists(op):
        pec_rows = {}
        for r in json.load(open(op)):
            if r.get('arm') == 'pec' and r.get('seed') in have:
                pec_rows[r['seed']] = r                 # last wins (resume-safe)
        for r in pec_rows.values():
            for qt in DEPTHS:
                if f'disp_{qt}' in r:
                    disp[qt].append(r[f'disp_{qt}'])
            if 'scramble_rel' in r:
                scr.append(r['scramble_rel'])

    # ── MRR table ──
    print("\n=== MRR (mean over seeds, full all-N) ===")
    print(f"  {'arm':<12}" + "".join(f"{qt:>9}" for qt in ['1p'] + DEPTHS))
    for arm in arms:
        cells = []
        for qt in ['1p'] + DEPTHS:
            cols = [data[s][f'{arm}_{qt}_rr'].mean() for s in have if f'{arm}_{qt}_rr' in data[s].files]
            cells.append(f"{np.mean(cols):9.4f}" if cols else f"{'--':>9}")
        print(f"  {arm:<12}" + "".join(cells))
    if any(disp.values()):
        print("  positive control: scramble rel-disp(ej)=%.3f | readout displacement " % (np.median(scr) if scr else 0.0)
              + " ".join(f"{qt}={np.median(disp[qt]):.3f}" for qt in DEPTHS if disp[qt]))

    def col(d, arm, qt, mask):
        arr = d[f'{arm}_{qt}_maskrr'] if arm == 'MASKONLY' else d[f'{arm}_{qt}_rr']
        return arr[mask]

    def margins(armA, armB, qt):
        mask = strat[qt]
        seeds_ok = [s for s in have
                    if (armA == 'MASKONLY' or f'{armA}_{qt}_rr' in data[s].files)
                    and (armB == 'MASKONLY' or f'{armB}_{qt}_rr' in data[s].files)]
        return np.array([float((col(data[s], armA, qt, mask) - col(data[s], armB, qt, mask)).mean())
                         for s in seeds_ok])

    def paired(m):
        """paired-t mean, p, 95% CI, 90% CI (the (1-2a) interval used for TOST)."""
        n = len(m)
        if n < 2:
            return float(m.mean()) if n else 0.0, 1.0, (float('nan'),) * 2, (float('nan'),) * 2, n
        mean = float(m.mean()); se = float(m.std(ddof=1)) / np.sqrt(n)
        _, p = st.ttest_1samp(m, 0.0)
        t95, t90 = st.t.ppf(0.975, n - 1), st.t.ppf(0.95, n - 1)
        return mean, float(p), (mean - t95 * se, mean + t95 * se), (mean - t90 * se, mean + t90 * se), n

    def report(armB):
        print(f"\n  pec - {armB}:")
        res = {}
        for qt in DEPTHS:
            m = margins('pec', armB, qt)
            if len(m) == 0:
                continue
            mean, p, ci95, _, n = paired(m)
            sig = '*' if (p < 0.05 and ci95[0] > 0) else ' '
            res[qt] = (mean, p, ci95)
            ps = " ".join(f"{x:+.3f}" for x in m)
            print(f"    {qt}: mean={mean:+.4f} (n={n}) t-p={p:.3f} 95%CI[{ci95[0]:+.4f},{ci95[1]:+.4f}] {sig}  per-seed[{ps}]")
        return res

    print(f"\n=== SEED-LEVEL paired tests (paired-t; * = p<0.05 AND 95% CI excludes 0) ===")
    rep = {arm: report(arm) for arm in LADDER if any(f'{arm}_3p_rr' in data[s].files for s in have)}

    # ── PRIMARY VERDICT: pec vs shuffle_full at 3p ──
    print(f"\n=== PRIMARY VERDICT: pec vs {PRIMARY} at 3p (the only TRAINED depth) ===")
    print(f"  delta={args.delta:.3f} (TOST margin), disp_min={args.disp_min:.3f} (lesion-lever floor)")
    m3 = margins('pec', PRIMARY, '3p')
    if len(m3) == 0:
        print("  no 3p contrast available"); return
    mean3, p3, ci95, ci90, n3 = paired(m3)
    d3 = np.median(disp['3p']) if disp['3p'] else float('nan')
    has_lever = (not np.isnan(d3)) and d3 >= args.disp_min
    print(f"  pec-{PRIMARY} 3p: mean={mean3:+.4f} (n={n3}) paired-t p={p3:.3f} 95%CI[{ci95[0]:+.4f},{ci95[1]:+.4f}]")
    print(f"  lesion lever (median 3p readout displacement) = {d3:.3f}  -> {'OK' if has_lever else 'NO LEVER'}")
    # Four quadrants of (statistically distinguishable from 0?) x (within the meaningful-effect band?).
    # delta=0.015 is the PRE-REGISTERED smallest MEANINGFUL effect, so a significant effect that is still
    # inside +-delta is real-but-trivial -> it substantively REFUTES the thesis (instance identity is not
    # MEANINGFULLY used), and must NOT be mislabeled "CONFIRM". TOST (equivalence) is checked alongside, not
    # after, significance.
    sig = (p3 < 0.05 and ci95[0] > 0)                                   # detectably > 0
    equivalent = (not np.isnan(ci90[0])) and ci90[0] > -args.delta and ci90[1] < args.delta  # within meaningful band
    interim = n3 < args.final_n
    if interim:
        print(f"\n  [INTERIM PEEK: n={n3} < pre-registered {args.final_n}. DIRECTIONAL ONLY, NOT the verdict —")
        print( "   small-n TOST/CI are unstable and early seeds can flip (cf. Phase-77 seed-456). Do not act on this.]")
    tag = 'interim, leaning ' if interim else ''
    if sig and not equivalent:
        print(f"\n  ==> {tag}CONFIRM: pec > shuffle_full at 3p by +{mean3:.4f}, reaching the meaningful threshold")
        print(f"      (delta={args.delta:.3f}). Edge-INSTANCE identity IS load-bearing — the thread is REAL.")
    elif sig and equivalent:
        print(f"\n  ==> {tag}SIGNIFICANT-BUT-NEGLIGIBLE: pec > shuffle_full is statistically detectable (+{mean3:.4f}, p={p3:.3f})")
        print(f"      but lies INSIDE the pre-registered equivalence band +-{args.delta:.3f} (smallest MEANINGFUL effect).")
        print("      => substantively REFUTES the thesis: edge-instance identity is used, but not MEANINGFULLY.")
    elif equivalent and has_lever:
        print(f"\n  ==> {tag}REFUTE (TOST): pec ~ shuffle_full within +-{args.delta:.3f}, and the lesion HAS a lever.")
        if not interim:
            print("      pec does NOT use edge-instance identity -> query-time edge composition refuted; retire it.")
    elif equivalent and not has_lever:
        print(f"\n  ==> {tag}INCONCLUSIVE (NO LEVER): pec~shuffle_full but the perm barely moves scores on this")
        print("      degree-1.48 / type-dominated testbed -> testbed-CAPACITY, NOT a refutation of the mechanism.")
    else:
        print(f"\n  ==> INCONCLUSIVE: 90%CI[{ci90[0]:+.4f},{ci90[1]:+.4f}] exceeds +-delta at n={n3}. Underpowered;")
        print("      add seeds. NEVER report 'refuted' from a non-significant point estimate.")

    # ── instance fraction: how much of pec's edge advantage is INSTANCE vs relation-TYPE ──
    mr = margins('pec', 'rrprior', '3p')
    if len(mr) and float(mr.mean()) > 1e-6:
        frac = float(m3.mean()) / float(mr.mean())     # (pec-shuffle_full)/(pec-rrprior)
        tag = 'instance load-bearing' if frac > 0.5 else ('type-prior dominates' if frac < 0.2 else 'mixed')
        print(f"\n  instance_fraction (pec-{PRIMARY})/(pec-rrprior) @3p = {frac:+.2f}  -> {tag}")
    else:
        print(f"\n  instance_fraction: pec-rrprior@3p ~ 0 (mean={float(mr.mean()):+.4f}) -> no edge advantage to apportion")

    # secondary/exploratory: extrapolation depths
    if PRIMARY in rep:
        sec = " ".join(f"{qt}:{rep[PRIMARY][qt][0]:+.4f}(p={rep[PRIMARY][qt][1]:.2f})"
                       for qt in ['4p', '5p'] if qt in rep[PRIMARY])
        if sec:
            print(f"  secondary/exploratory (extrapolation depths, pec-{PRIMARY}): {sec}")


if __name__ == '__main__':
    main()
