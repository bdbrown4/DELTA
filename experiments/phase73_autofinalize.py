"""Auto-finalize Phase 73: turn phase73_output.json into a phone-readable results doc.

Run by the detached watcher when the Phase 73 run completes. Deterministically
computes the per-seed analysis (means, hops gaps, verdict) and writes
docs/phase_73_results.md. No model/torch needed — pure JSON -> markdown.

This is an AUTO-GENERATED snapshot for remote viewing; a polished docs/phase_73.md
(+ research_state.json update) is written by hand when back in an interactive session.
"""
import sys, os, json
from collections import defaultdict
from statistics import mean, pstdev

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, 'phase73_output.json')
OUT = os.path.join(ROOT, 'docs', 'phase_73_results.md')
QT = ['1p', '2p', '3p', '4p', '5p']


def ms(rows, k):
    v = [r[k] for r in rows]
    return (mean(v), (pstdev(v) if len(v) > 1 else 0.0)) if v else (float('nan'), 0.0)


def fmt(rows, k):
    m, s = ms(rows, k)
    return f"{m:.3f}±{s:.3f}" if len(rows) > 1 else f"{m:.4f}"


def main():
    if not os.path.exists(IN):
        print("no phase73_output.json"); return 1
    d = json.load(open(IN))
    meta, R = d.get('meta', {}), d.get('results', [])
    by = defaultdict(list)
    for r in R:
        by[r['condition']].append(r)
    seeds = sorted({r['seed'] for r in R})
    conds = [c for c in ['node_only', 'hops1', 'hops2'] if by.get(c)]
    complete = len(R) >= 15

    L = []
    L.append("# Phase 73 — High-Power Edge-Sampled Hop Ablation (AUTO-GENERATED snapshot)")
    L.append("")
    L.append(f"> Auto-written from `phase73_output.json` on run completion. "
             f"{'COMPLETE' if complete else 'PARTIAL ('+str(len(R))+'/15 runs)'}. "
             f"A polished `docs/phase_73.md` + `research_state.json` update follow in an interactive session.")
    L.append("")
    L.append("## Config")
    L.append(f"- Edge-sampled FB15k-237: **{meta.get('num_entities','?')} entities**, "
             f"{meta.get('E_train','?')} train edges, **{meta.get('test','?')} test triples**, "
             f"mean degree **{meta.get('mean_deg',0):.2f}** (sample_frac={meta.get('sample_frac','?')})")
    L.append(f"- hops=1 pairs={meta.get('n1','?'):,}; hops=2 fair-capped (K={meta.get('cap_k','?')}) "
             f"pairs={meta.get('n2','?'):,}")
    L.append(f"- seeds={seeds}, conditions={conds}")
    if R:
        qn = {q: by[conds[0]][0].get(f'{q}_count', 0) for q in QT}
        L.append(f"- multi-hop query counts: " + ", ".join(f"{q}={qn[q]}" for q in QT))
    L.append("")

    # Means table
    L.append("## Results (mean±std across seeds)")
    L.append("")
    L.append("| condition | n | LP | 1p | 2p | 3p | 4p | 5p |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in conds:
        rows = by[c]
        cells = " | ".join(fmt(rows, f'{q}_MRR') for q in ['1p', '2p', '3p', '4p', '5p'])
        L.append(f"| {c} | {len(rows)} | {fmt(rows,'lp_MRR')} | {cells} |")
    L.append("")

    # Gap analyses (per-seed)
    def gaps(a, b, label):
        if not (by.get(a) and by.get(b)):
            return
        ba = {r['seed']: r for r in by[a]}
        bb = {r['seed']: r for r in by[b]}
        common = [s for s in seeds if s in ba and s in bb]
        L.append(f"### {label}")
        L.append("")
        L.append("| metric | mean gap | per-seed | >0 |")
        L.append("|---|---|---|---|")
        for q in ['2p', '3p', '4p', '5p']:
            per = [ba[s][f'{q}_MRR'] - bb[s][f'{q}_MRR'] for s in common]
            mark = " **>0.010**" if abs(mean(per)) > 0.010 else ""
            ps = ", ".join(f"{x:+.3f}" for x in per)
            L.append(f"| {q} | {mean(per):+.4f}{mark} | {ps} | {sum(1 for x in per if x>0)}/{len(per)} |")
        L.append("")

    L.append("## Gap analysis (per-seed is what matters — see Phases 71/72)")
    L.append("")
    gaps('hops2', 'hops1', "hops=2 − hops=1  (the 2-hop claim)")
    gaps('hops1', 'node_only', "hops=1 − node_only  (does edge attention help at all?)")
    gaps('hops2', 'node_only', "hops=2 − node_only")

    # Heuristic verdict
    L.append("## Heuristic verdict (auto)")
    L.append("")
    def mean_gap(a, b, q):
        ba = {r['seed']: r for r in by.get(a, [])}; bb = {r['seed']: r for r in by.get(b, [])}
        common = [s for s in seeds if s in ba and s in bb]
        return mean([ba[s][f'{q}_MRR'] - bb[s][f'{q}_MRR'] for s in common]) if common else float('nan')
    if by.get('hops1') and by.get('hops2') and by.get('node_only'):
        g_h2h1 = mean_gap('hops2', 'hops1', '3p')
        g_h1no = mean_gap('hops1', 'node_only', '3p')
        ba = {r['seed']: r for r in by['hops2']}; bb = {r['seed']: r for r in by['hops1']}
        common = [s for s in seeds if s in ba and s in bb]
        h2_pos = sum(1 for s in common if ba[s]['3p_MRR'] - bb[s]['3p_MRR'] > 0)
        L.append(f"- **2-hop (hops2 vs hops1), 3p:** mean {g_h2h1:+.4f}, positive in {h2_pos}/{len(common)} seeds "
                 f"→ {'SUPPORTED' if g_h2h1>0.010 and h2_pos>len(common)/2 else 'NOT robust'}.")
        L.append(f"- **edge attention (hops1 vs node_only), 3p:** mean {g_h1no:+.4f} "
                 f"→ {'helps' if g_h1no>0.010 else 'inconclusive/none'}.")
    L.append("")
    L.append("_Raw per-(condition,seed) data is in the committed `phase73_output.json`._")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w', encoding='utf-8').write("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(R)} rows, complete={complete})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
