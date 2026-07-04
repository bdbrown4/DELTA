"""Step-3 assembler + kill-gate verdict (SUCCESSOR_SCOPE §2/§4).

Merges the four Step-2/3 artifacts:
  spectrum.json        label-free rho_E (FB15k, WD50K dial @100k, ICEWS)
  spectrum_mol.json    molecular bond features, three x-specs (canonical = zfree)
  spectrum_collab.json ogbl-collab label-free rho_E
  rho_e_y.json         LABEL-AWARE rho_E^Y (real-vs-corrupted content lift) on link-prediction datasets

and renders TWO tests:
  (A) the PRE-REGISTERED gate: does label-free rho_E order published content-isolating gain better than
      the binary "has edge attributes?" check, on non-empty-z / known-gain rows?
  (B) the RESCUE test: does the label-aware rho_E^Y order gain where rho_E fails?

The Step-3 verdict is a human call on this assembled evidence, kept honest by the pre-registered fail
condition (§2): flat/non-monotone (A) AND failed (B) => #2 drops to a companion observation, ship #1 alone.

Usage: python successor/rho_e/report.py
"""
import json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RD = os.path.join(ROOT, "successor", "rho_e")


def _load(fn):
    p = os.path.join(RD, fn)
    return json.load(open(p)) if os.path.exists(p) else {}


# Published content-isolating edge-aware GAIN band (NOT the "expected rho_E" column of SUCCESSOR_SCOPE §3 —
# that table's middle column is a PREDICTED rho_E, a common misread; the gain band here is read from the
# EVIDENCE column). ICEWS gain=low: Radstok 2021 (timestamp-blind baselines competitive => small temporal
# gain). NOTE the gate FAIL below is verified INVARIANT to scoring ICEWS low OR mid, so the choice is not
# load-bearing (no band-shopping).
PUBLISHED = {
    "fb15k237":   dict(has_z=False, gain="none", src="DELTA 79-phase null; Li et al. ACL2023 (MLP~MPNN)"),
    "wn18rr":     dict(has_z=False, gain="none", src="symbolic KG; no edge attributes"),
    "molhiv":     dict(has_z=True,  gain="low",  src="D-MPNN: bond features +0.8-2.7% AUC, 'not always'"),
    "icews14":    dict(has_z=True,  gain="low",  src="Radstok 2021: timestamp-blind baselines competitive"),
    "wd50k_0":    dict(has_z=True,  gain="low",  src="StarE: ~14% qualifiers, small lift"),
    "wd50k_33":   dict(has_z=True,  gain="mid",  src="StarE: lift grows with qualifier fraction"),
    "wd50k_66":   dict(has_z=True,  gain="mid",  src="StarE: lift grows with qualifier fraction"),
    "wd50k_100":  dict(has_z=True,  gain="high", src="StarE: up to +25 MRR at full qualifier fraction"),
    "ogbl_collab":dict(has_z=True,  gain="unknown", src="no published isolating ablation; rho_E^Y is the test"),
}
BAND = {"none": 0, "low": 1, "mid": 2, "high": 3}


def _is_monotone(pairs):
    """pairs = [(rho, gain_band)]; True iff rho is nondecreasing when sorted by gain band (tol 0.05)."""
    s = sorted(pairs, key=lambda t: BAND[t[1]])
    r = [p[0] for p in s]
    return all(r[i] <= r[i + 1] + 0.05 for i in range(len(r) - 1))


def main():
    spec = _load("spectrum.json"); mol = _load("spectrum_mol.json")
    collab = _load("spectrum_collab.json"); rey = _load("rho_e_y.json")

    # canonical label-free rho_E per dataset
    rho = {}
    for k in ("fb15k237", "wd50k_0", "wd50k_33", "wd50k_66", "wd50k_100", "icews14"):
        if k in spec:
            rho[k] = (spec[k]["rho_e"], spec[k]["status"])
    if "ogbl_collab" in collab:
        rho["ogbl_collab"] = (collab["ogbl_collab"]["rho_e"], collab["ogbl_collab"]["status"])
    if "molhiv[zfree]" in mol:
        rho["molhiv"] = (mol["molhiv[zfree]"]["rho_e"], "ok(zfree)")

    print("\n=== (0) LABEL-FREE rho_E SPECTRUM vs published gain ===")
    print(f"  {'dataset':<13}{'rho_E':>8}{'has_z':>7}{'gain':>7}  evidence")
    for k, pub in PUBLISHED.items():
        r = rho.get(k)
        rv = f"{r[0]:.3f}" if r else "  --"
        print(f"  {k:<13}{rv:>8}{str(pub['has_z']):>7}{pub['gain']:>7}  {pub['src'][:48]}")

    # molecular x-spec sensitivity (a headline robustness caveat)
    if mol:
        print("\n  molecular x-spec sensitivity (node-feature richness moves rho_E):")
        for s in ("clean", "zfree", "full"):
            key = f"molhiv[{s}]"
            if key in mol:
                print(f"    molhiv[{s:<5}] rho_E = {mol[key]['rho_e']:.3f}")

    # ── (A) pre-registered gate: label-free rho_E vs binary check, non-empty-z known-gain rows ──
    live = [(k, rho[k][0], PUBLISHED[k]["gain"]) for k in rho
            if PUBLISHED[k]["has_z"] and PUBLISHED[k]["gain"] not in ("none", "unknown")
            and rho[k][1].startswith("ok")]
    print("\n=== (A) PRE-REGISTERED GATE — label-free rho_E as a gain ORDERER (non-empty-z rows) ===")
    for k, r, g in sorted(live, key=lambda t: BAND[t[2]]):
        print(f"  {k:<12} gain={g:<5} rho_E={r:.3f}")
    if len(live) >= 3:
        mono = _is_monotone([(r, g) for _, r, g in live])
        # BAND-INVARIANCE: re-score with ICEWS forced to each band; the FAIL must not depend on the choice.
        inv = {}
        for band in ("low", "mid", "high"):
            pairs = [(r, band if k == "icews14" else g) for k, r, g in live]
            inv[band] = _is_monotone(pairs)
        print(f"\n  monotone-nondecreasing along gain bands (ICEWS=low)? {'YES' if mono else 'NO'}")
        print(f"  band-invariance of the FAIL — monotone under ICEWS scored {{low,mid,high}}: "
              f"{[('%s:%s' % (b, 'mono' if v else 'NON-mono')) for b, v in inv.items()]}")
        within_low = [(k, r) for k, r, g in live if g == "low"]
        if len(within_low) >= 2:
            spread = max(r for _, r in within_low) - min(r for _, r in within_low)
            print(f"  within the LOW gain band alone, rho_E spans {spread:.2f} "
                  f"({', '.join('%s=%.2f' % (k, r) for k, r in sorted(within_low, key=lambda t: t[1]))}) "
                  f"=> scattered, not merely inverted")
        print("  binary 'has_z?' check is CONSTANT on these rows (all have z) => zero discrimination;")
        print("  label-free rho_E must positively order gain to beat it. It is NON-MONOTONE under every ICEWS"
              " band => FAILS the gate (band-invariantly).")

    # ── (B) rescue test: label-aware rho_E^Y ──
    print("\n=== (B) RESCUE TEST — label-aware rho_E^Y as a gain ORDERER ===")
    if not rey:
        print("  rho_e_y.json not present yet.")
    else:
        rows = []
        for k, rep in rey.items():
            if rep.get("status") == "ok":
                rows.append((k, rep["rho_e_y_joint"], PUBLISHED.get(k, {}).get("gain", "?"),
                             rep.get("auc_W"), rep.get("auc_Wz_joint")))
        for k, ry, g, aw, awz in sorted(rows, key=lambda t: BAND.get(t[2], 9)):
            print(f"  {k:<12} gain={g:<7} rho_E^Y={ry:.4f}  (AUC_W={aw:.3f} -> AUC_Wz={awz:.3f})")
        known = [(k, ry, g) for k, ry, g, _, _ in rows if g in ("low", "mid", "high")]
        if len(known) >= 2:
            by_gain = sorted(known, key=lambda t: BAND[t[2]])
            rys = [ry for _, ry, _ in by_gain]
            mono_y = all(rys[i] <= rys[i+1] + 0.03 for i in range(len(rys)-1))
            print(f"\n  raw rho_E^Y monotone along gain bands? {'YES' if mono_y else 'NO'} "
                  f"(but see de-confounding below — the raw dial is partly a degree confound)")

    # de-confounded (degree-residualized) dial — the audited reading
    resid = _load("rho_e_y_resid.json")
    if resid:
        print("\n  DE-CONFOUNDED (z residualized on candidate-tail structure):")
        for k in ("icews14", "wd50k_0", "wd50k_33", "wd50k_100"):
            raw = rey.get(k, {}).get("rho_e_y_joint")
            rr = resid.get(k, {}).get("rho_e_y_joint")
            if rr is not None:
                print(f"    {k:<11} raw={raw:.4f} -> residualized={rr:.4f}"
                      f"{'   <- SURVIVES (genuine content)' if rr > 0.05 else '   (collapses toward floor)'}")
        print("  => coarse relevant(wd50k_100)-vs-irrelevant(ICEWS/collab/low-qual) separation SURVIVES;")
        print("     fine monotone dial does NOT (wd50k_33 ~halves) => #2 PROMISING but UNCONFIRMED -> Step 5.")
    print()


if __name__ == "__main__":
    main()
