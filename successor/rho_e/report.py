"""Step-2/3 report: pair computed rho_E (spectrum.json) against PUBLISHED edge-aware gains, and run the
pre-registered kill-gate comparison (SUCCESSOR_SCOPE §2 / §4).

The kill gate (verbatim): does rho_E, on NON-EMPTY-z datasets, order/predict the published
content-isolating gains BETTER than the binary schema check "does the dataset have edge attributes"?
Alignment uses content-isolating feature-ablation deltas only (same architecture, z on/off), NOT
architecture-vs-architecture deltas. This script assembles the table; the Step-3 verdict is a human call
on the assembled evidence (kept honest by the pre-registered fail condition).

Published-gain evidence is transcribed from the literature (SUCCESSOR_SCOPE §3) with source tags; these
are effect *directions/magnitudes*, normalized qualitatively (LOW/MID/HIGH gain) because raw effect sizes
are incommensurable across MRR/AUC/MAE (a named Step-3 threat).

Usage: python successor/rho_e/report.py
"""
import json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPECTRUM = os.path.join(ROOT, "successor", "rho_e", "spectrum.json")

# Published content-isolating edge-aware gain, per dataset (SUCCESSOR_SCOPE §3). gain_band in
# {none, low, mid, high}; has_edge_attr is the binary schema check.
PUBLISHED = {
    "fb15k237":  dict(has_edge_attr=False, gain_band="none",
                      evidence="DELTA 79-phase null; Li et al. ACL2023 (MLP~MPNN)"),
    "wn18rr":    dict(has_edge_attr=False, gain_band="none",
                      evidence="symbolic KG; no edge attributes"),
    "icews14":   dict(has_edge_attr=True, gain_band="low",
                      evidence="Radstok et al. 2021: timestamp-blind baselines competitive; temporal gains marginal"),
    "wd50k_0":   dict(has_edge_attr=True, gain_band="low",
                      evidence="StarE EMNLP2020: base ~14% qualifiers, small qualifier lift"),
    "wd50k_33":  dict(has_edge_attr=True, gain_band="mid",
                      evidence="StarE: qualifier lift grows with qualifier fraction (monotone)"),
    "wd50k_66":  dict(has_edge_attr=True, gain_band="mid",
                      evidence="StarE: qualifier lift grows with qualifier fraction (monotone)"),
    "wd50k_100": dict(has_edge_attr=True, gain_band="high",
                      evidence="StarE: up to +25 MRR at full qualifier fraction (monotone)"),
    "ogbl_collab": dict(has_edge_attr=True, gain_band="unknown",
                        evidence="no clean isolating ablation exists — our own ablation is the falsifiable test"),
    "alignn_jarvis": dict(has_edge_attr=True, gain_band="high",
                          evidence="ALIGNN npj2021: up to ~85% rel. error reduction on angle-sensitive props"),
}
BAND_ORDER = {"none": 0, "low": 1, "mid": 2, "high": 3}


def main():
    spec = json.load(open(SPECTRUM)) if os.path.exists(SPECTRUM) else {}
    rows = []
    for name, pub in PUBLISHED.items():
        s = spec.get(name)
        rho = s["rho_e"] if s else None
        status = s["status"] if s else "not_computed"
        rows.append(dict(name=name, rho=rho, status=status, **pub))

    print("\n=== rho_E SPECTRUM vs PUBLISHED edge-aware gain ===")
    print(f"  {'dataset':<13}{'rho_E':>8}{'status':>16}{'has_z':>7}{'gain':>7}  evidence")
    for r in sorted(rows, key=lambda r: (r["rho"] is None, r["rho"] or 0)):
        rho = f"{r['rho']:.3f}" if r["rho"] is not None else "  --"
        print(f"  {r['name']:<13}{rho:>8}{r['status']:>16}{str(r['has_edge_attr']):>7}"
              f"{r['gain_band']:>7}  {r['evidence'][:52]}")

    # ── kill-gate view: non-empty-z rows only (schema-degenerate rows excluded, per SCOPE) ──
    live = [r for r in rows if r["rho"] is not None and r["status"] == "ok"
            and r["gain_band"] not in ("none", "unknown")]
    print("\n=== KILL-GATE VIEW (non-empty-z, known-gain rows only) ===")
    if len(live) < 3:
        print(f"  only {len(live)} live rows so far — need the WD50K dial + a molecular/materials point.")
        print("  (schema-degenerate anchors agree with the binary check by construction and are excluded)")
    for r in sorted(live, key=lambda r: r["rho"]):
        print(f"  {r['name']:<13} rho_E={r['rho']:.3f}  published_gain={r['gain_band']}")
    if len(live) >= 3:
        rhos = [r["rho"] for r in sorted(live, key=lambda r: BAND_ORDER[r["gain_band"]])]
        mono = all(rhos[i] <= rhos[i + 1] + 0.05 for i in range(len(rhos) - 1))
        print(f"\n  rho_E monotone-nondecreasing along published gain bands? {'YES' if mono else 'NO'}")
        print("  (a monotone rise beats the binary schema check => diagnostic PASSES the mid regime;")
        print("   flat/non-monotone => drop #2 to a companion observation, ship #1 alone — the pre-registered kill)")
    print()


if __name__ == "__main__":
    main()
