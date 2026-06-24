"""Phase 79 analysis — is the 'JIT ~2x AOT' gain the trained readout or the path mechanism?

Seed-level paired-t (n=15), primary depth 3p, delta=0.015, reusing the Phase-78 four-quadrant rule.
Arms in phase79_rr_s{seed}.npz: aot_readout (node-only + trained head, NO edges), pec (edge-composed),
static (edges-as-input, composition off), aotsoft48 (untrained head = shared-space floor).

Contrasts:
  gap_path = pec - aot_readout   (PRIMARY @3p: the edge/path axis, one-axis isolation)
  gap_head = aot_readout - aotsoft48     (value of training the head)
  gap_total = pec - aotsoft48
  path_fraction = gap_path / gap_total
  guard = pec - static           (conjunctive guard on the genuine-positive branch)
  1p check = gap_path @1p ~ 0     (zero-traversal sanity: the two arms must agree at 1p)

Usage:
  python experiments/phase79_analyze.py --seeds 42,123,456,7,99,11,23,5,17,31,2,8,13,29,37
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
from scipy import stats as st

DEPTHS = ['2p', '3p', '4p', '5p']
ARMS = ['aot_readout', 'pec', 'static', 'aotsoft48']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default='42,123,456,7,99,11,23,5,17,31,2,8,13,29,37')
    ap.add_argument('--rr_root', type=str, default='.')
    ap.add_argument('--delta', type=float, default=0.015, help='TOST / meaningful-effect margin')
    ap.add_argument('--final_n', type=int, default=15, help='below this the verdict is labeled INTERIM')
    ap.add_argument('--gate_tol', type=float, default=0.010, help='1p zero-traversal agreement tolerance')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    root = os.path.join(os.path.dirname(__file__), '..', args.rr_root)

    data, strat = {}, {}
    for s in seeds:
        p = os.path.join(root, f'phase79_rr_s{s}.npz')
        if not os.path.exists(p):
            print(f"  (missing {os.path.basename(p)})"); continue
        d = np.load(p)
        if 'pec_3p_rr' not in d.files or 'aot_readout_3p_rr' not in d.files:
            print(f"  (seed {s}: incomplete)"); continue
        data[s] = d
        for qt in DEPTHS + ['1p']:
            if f'strat_{qt}' in d.files and qt not in strat:
                strat[qt] = d[f'strat_{qt}']
    have = list(data.keys())
    if not have:
        print("no complete seeds yet"); return
    present = sorted({k.rsplit('_', 2)[0] for s in have for k in data[s].files if k.endswith('_rr')})
    print(f"Phase 79 analysis | seeds={have} (n={len(have)}) | arms={present}")

    print("\n=== MRR (mean over seeds, full all-N) ===")
    print(f"  {'arm':<12}" + "".join(f"{qt:>9}" for qt in ['1p'] + DEPTHS))
    for arm in present:
        cells = []
        for qt in ['1p'] + DEPTHS:
            cols = [data[s][f'{arm}_{qt}_rr'].mean() for s in have if f'{arm}_{qt}_rr' in data[s].files]
            cells.append(f"{np.mean(cols):9.4f}" if cols else f"{'--':>9}")
        print(f"  {arm:<12}" + "".join(cells))
    print("  (context: published 64-d AOT-soft from Phase 77 ~ 1p .106 / 2p .093 / 3p .107 / 4p .135 / 5p .165)")

    def col(d, arm, qt, mask):
        return d[f'{arm}_{qt}_rr'][mask]

    def margins(armA, armB, qt):
        mask = strat[qt]
        ok = [s for s in have if f'{armA}_{qt}_rr' in data[s].files and f'{armB}_{qt}_rr' in data[s].files]
        return np.array([float((col(data[s], armA, qt, mask) - col(data[s], armB, qt, mask)).mean()) for s in ok])

    def paired(m):
        n = len(m)
        if n < 2:
            return (float(m.mean()) if n else 0.0), 1.0, (float('nan'),) * 2, (float('nan'),) * 2, n
        mean = float(m.mean()); se = float(m.std(ddof=1)) / np.sqrt(n)
        _, p = st.ttest_1samp(m, 0.0)
        return mean, float(p), (mean - st.t.ppf(0.975, n - 1) * se, mean + st.t.ppf(0.975, n - 1) * se), \
            (mean - st.t.ppf(0.95, n - 1) * se, mean + st.t.ppf(0.95, n - 1) * se), n

    def report(armA, armB):
        print(f"\n  {armA} - {armB}:")
        res = {}
        for qt in ['1p'] + DEPTHS:
            m = margins(armA, armB, qt)
            if len(m) == 0:
                continue
            mean, p, ci95, _, n = paired(m)
            sig = '*' if (p < 0.05 and (ci95[0] > 0 or ci95[1] < 0)) else ' '
            res[qt] = (mean, p, ci95)
            print(f"    {qt}: mean={mean:+.4f} (n={n}) t-p={p:.3f} 95%CI[{ci95[0]:+.4f},{ci95[1]:+.4f}] {sig}")
        return res

    print("\n=== SEED-LEVEL paired tests (paired-t; * = p<0.05 AND CI excludes 0) ===")
    gp = report('pec', 'aot_readout')       # gap_path (PRIMARY)
    report('aot_readout', 'aotsoft48')      # gap_head
    guard = report('pec', 'static')         # conjunctive guard

    # ── 1p zero-traversal sanity gate ──
    print("\n=== 1p ZERO-TRAVERSAL GATE (pec and aot_readout must agree at 1p; neither traverses) ===")
    if '1p' in gp:
        m1, p1, ci1 = gp['1p']
        ok = abs(m1) < args.gate_tol and ci1[0] > -args.delta and ci1[1] < args.delta
        print(f"  gap_path @1p = {m1:+.4f} (|.|<{args.gate_tol}? {'OK' if abs(m1) < args.gate_tol else 'FAIL'}) "
              f"-> {'PASS' if ok else 'CHECK'} (large 1p gap = structural divergence, not an edge finding)")

    # ── PRIMARY VERDICT: gap_path @3p, four-quadrant + conjunctive guard ──
    print("\n=== PRIMARY VERDICT: gap_path = pec - aot_readout @3p (is there edge/path signal beyond the head?) ===")
    print(f"  delta={args.delta:.3f}; genuine-positive also requires the JIT-static guard (pec > static @3p)")
    if '3p' not in gp:
        print("  no 3p contrast yet"); return
    m3 = margins('pec', 'aot_readout', '3p')
    mean3, p3, ci95, ci90, n3 = paired(m3)
    interim = n3 < args.final_n
    sig = (p3 < 0.05 and ci95[0] > 0)
    equivalent = (not np.isnan(ci90[0])) and ci90[0] > -args.delta and ci90[1] < args.delta
    guard_ok = False
    if '3p' in guard:
        gm, gp_, gci = guard['3p']
        guard_ok = (gp_ < 0.05 and gci[0] > 0)
    print(f"  gap_path @3p: mean={mean3:+.4f} (n={n3}) p={p3:.3f} 95%CI[{ci95[0]:+.4f},{ci95[1]:+.4f}]"
          f" | guard pec>static @3p: {'OK' if guard_ok else 'no'}")
    if interim:
        print(f"  [INTERIM PEEK n={n3} < {args.final_n} — directional only, not the verdict]")
    tag = 'interim, leaning ' if interim else ''
    if sig and not equivalent and guard_ok:
        print(f"\n  ==> {tag}GENUINE POSITIVE: edge/path adds meaningful value beyond a trained head (and pec>static).")
        print("      The query-time path mechanism is real -> motivates Phase-80 co-training.")
    elif sig and not equivalent and not guard_ok:
        print(f"\n  ==> {tag}AMBIGUOUS: gap_path meaningful BUT the JIT-static guard fails -> the gain is params/")
        print("      ef0-as-input, NOT edge composition. Not a clean positive.")
    elif sig and equivalent:
        print(f"\n  ==> {tag}SIGNIFICANT-BUT-NEGLIGIBLE: gap_path detectable (+{mean3:.4f}) but 5x inside +-{args.delta:.3f}")
        print("      => the 2x is the trained READOUT/objective, not the path. (Expected; tightens the negative.)")
    elif equivalent:
        print(f"\n  ==> {tag}READOUT, FULL STOP: pec ~ aot_readout within +-{args.delta:.3f}. The 2x is the trained head;")
        print("      query-time edge participation adds nothing beyond it.")
    else:
        print(f"\n  ==> INCONCLUSIVE: 90%CI[{ci90[0]:+.4f},{ci90[1]:+.4f}] exceeds +-delta at n={n3}. Add seeds.")

    # ── decomposition ──
    mt = margins('pec', 'aotsoft48', '3p')
    if len(mt) and float(mt.mean()) > 1e-6:
        print(f"\n  path_fraction = gap_path/gap_total @3p = {float(m3.mean())/float(mt.mean()):+.2f}"
              f"  (gap_total pec-aotsoft48 = {float(mt.mean()):+.4f}; head explains the rest)")


if __name__ == '__main__':
    main()
