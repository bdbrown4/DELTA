# Step 3 — the pre-registered kill gate + the label-aware tier, adversarially audited (sprint artifact)

**Status:** Step 3 COMPLETE (2026-07-03). Spectrum complete (8 label-free points incl. two new OGB
anchors, molecular in four x-specs); Step-2 findings hardened; the label-aware tier ρ_E^Y built, run, and
**adversarially de-confounded** after a three-referee red-team (`step3-adversarial-verify`, Opus panel)
caught a structural confound in the first reading. This document renders the pre-registered decision on the
corrected evidence. DELTA itself remains CLOSED.

> **One-line verdict.** The **label-free** ρ_E is **killed as a gain predictor** — it is non-monotone in
> published gain (band-invariantly) and cannot beat the constant "has-z?" check. It **survives as a
> one-directional necessary condition** (low ρ_E ⇒ no gain), now supported by a real molecular point.
> The **label-aware ρ_E^Y**, *after adversarial de-confounding*, cleanly separates clearly-relevant from
> clearly-irrelevant edge content (and zeroes the two irreducible-but-irrelevant cases) — but its
> fine-grained gain *calibration* is substantially a degree confound and is **not established at the desk
> stage**. So #2 is a **promising, partially-validated relevance direction**, promoted to Step 5 under a
> pre-registered de-confounding protocol; the **theory note #1 ships** as the primary contribution.

---

## The assembled spectrum (label-free ρ_E)

| dataset | ρ_E (label-free) | published gain | has z | note |
|---|---|---|---|---|
| FB15k-237 | 0.000 | none | no | schema anchor (DELTA null; Li et al. ACL'23) |
| WN18RR | 0.000 | none | no | schema anchor |
| **molhiv** (bond features) | **0.06–0.09** *(principled)* | low (D-MPNN +0.8–2.7% AUC) | yes | **new**; four x-specs — see below |
| ICEWS14 | **0.940** | low | yes | robust to bucketing, multigraph frame, estimator de-bias |
| WD50K (14% qual) | 0.588 ± 0.007 *(fixed-n)* | low | yes | @100k was 0.611 (n-confounded) |
| WD50K_33 | 0.615 ± 0.051 | mid | yes | @100k 0.567 (order reversed under fixed-n → n-confound confirmed) |
| WD50K_66 | 0.654 ± 0.030 | mid | yes | |
| WD50K_100 | 0.637 ± 0.025 | high | yes | |
| **ogbl-collab** | **0.745** | unknown → our ρ_E^Y test | yes | **new**; year+weight, true multigraph |

## Step-2 audit — the three findings, hardened (one corrected)

**Finding 2 (anti-ordering) — CONFIRMED ROBUST.** Worry: ICEWS's 0.94 is an equal-key-weighting artifact
on a multigraph (dup 0.585; one triple recurs 188×). Test (`icews_audit.py`): on the **size-1-key subset**
(within-key term = 0 by construction) ρ_E = **0.936**; cap-at-p95 → 0.931; and de-biasing the max-selection
estimator (mean of the kNN/GBT pair rather than max) *raises* it to ~0.96. The high value is genuine
content-irreducibility (timestamps ~unpredictable from structure, V≈0.06), not an artifact of the frame or
the estimator.

**Finding 1 (one-directional low end) — NOW SUPPORTED (not merely asserted).** Previously rested only on
FB15k-237, a `schema_empty` anchor the gate excludes. The molecular point is a real (simple-graph:
key_size=1, so it exercises only the across-key endpoint-predictability term) dataset that *has* edge
attributes: bond features are largely endpoint-inferable (ρ_E ≈ 0.06–0.09 under the principled z-free atom
spec) **and** the published gain is low. No counterexample (low ρ_E with a real gain) exists anywhere on the
spectrum. The `full`-spec residual (ρ_E ≈ 0.024 with the small real D-MPNN bump) is the exact
**information-vs-accessibility gap** STEP1 Prop 1 pre-named — predicted signature, not refutation.

*Molecular x-spec sensitivity (an honest robustness caveat, quantified).* ρ_E depends on how rich the
z-free node features are — a legitimate researcher choice:

| molecular x-spec | ρ_E | what it includes |
|---|---|---|
| `full` (all 9 atom feats, **leaky**) | 0.024 | + hybridization/is_aromatic/numH — RDKit-derived from bond orders ⇒ x-leak the z |
| `zfree` (STEP1 principled) | 0.063 | atomic num, chirality, charge, radical, **is_in_ring** + structural |
| `zfree_noring` (bounds the ring bias) | 0.085 | drops is_in_ring |
| `clean` (atomic number only) | 0.462 | element + structural degree/PR — an impoverished node model no molecular GNN uses |

Two caveats, both now stated in-code and here: (i) `is_in_ring` is z-free by *provenance* (SSSR topology,
not a bond-attribute field) but z-*correlated* (aromatic/conjugated bonds imply both endpoints in a ring),
so it biases ρ_E slightly *downward* — hence the principled value is a small **range [0.063, 0.085]**, not a
single canonical number, and both ends are LOW. (ii) The 7× swing to `clean`=0.46 comes from stripping the
node model to atomic-number-only; that is a strawman, not a real molecular GNN input, so molhiv reads **LOW
under every reasonable node-feature set** and MID only under the impoverished extreme. The one-directional
"low ρ_E ⇒ low gain" reading holds under all four specs (all low-to-mid, gain low, no counterexample).
Modeling note: putting bond_type in z with τ=const is the *generous* choice; τ=bond_type would collapse ρ_E
toward ~0.02, so the LOW reading survives that alternative too.

**Finding 3 (WD50K dial) — CORRECTED.** Step 2 asserted "flat within noise" but never computed the band and
mixed n across variants. The fixed-n control (`wd50k_fixed_n.py`, all four variants at 31k, 3 seeds) shows a
**weak, noisy, non-monotone upward drift**: 0.588 → 0.615 → 0.654 → 0.637 (dips at 100%), across-dial range
0.066 vs within-seed 2σ ≈ 0.056. So it is *not* perfectly flat — removing the n-confound reveals a slight
real coverage sensitivity (and the @100k ordering *reversed*, confirming the @100k dial was n-confounded).
But the drift is ~0.07 across the full qualifier dial while the gain spans low→high, and the whole dial sits
far below ICEWS's 0.94 — so label-free ρ_E's coverage sensitivity is far too weak and noisy to serve as the
gain predictor. Finding 3's substance (the dial is about coverage, which a per-edge dispersion ratio barely
sees) holds; the "flat" wording is corrected to "weak/noisy/non-monotone, within ≈2σ."

---

## (A) The pre-registered gate — label-free ρ_E as a gain ORDERER

Scored on non-empty-z, known-gain rows (schema anchors excluded by construction). The binary "has-z?" check
is **constant** on these rows → zero discrimination; ρ_E must positively order gain to beat it.

- **NON-MONOTONE, band-invariantly.** Sorted by gain, ρ_E does not increase: within the **low** band alone it
  spans **0.88** (molhiv 0.06, wd50k_0 0.59, ICEWS 0.94) — scattered, not merely inverted. The FAIL is
  invariant to how ICEWS is scored: non-monotone under ICEWS = low, mid, **and** high (`report.py`
  band-invariance check). *(The SCOPE §3 table's "mid" for temporal KGs is a predicted **ρ_E**, not a gain
  band — a documented misread; the gain is low per Radstok. The gate outcome does not depend on the choice.)*
- **Verdict (A): label-free ρ_E FAILS the pre-registered gate.** It loses to the constant binary check as a
  gain orderer, because irreducibility ≠ task-relevance (ICEWS: irreducible, irrelevant) and ≈ coverage only
  weakly (WD50K fixed-n drift ~0.07). **This is the pre-registered kill of the diagnostic law as a gain
  predictor.**

What survives (A): the **one-directional necessary condition** — low ρ_E ⇒ no gain (Finding 1; molhiv +
FB15k, no counterexample). That is the clause Proposition 1 actually proves.

---

## (B) The label-aware tier ρ_E^Y — promising, but de-confounded before any claim

ρ_E^Y = the chance-corrected lift in **real-vs-corrupted link discrimination** that z adds beyond
W = (x_u, τ, x_v) — the model-free image of SCOPE's content-isolating z-on/off ablation (`rho_e_y.py`).

**First reading (seeds 0/1), before de-confounding** — a clean monotone dial:

| dataset | gain | ρ_E^Y seed0 | seed1 |
|---|---|---|---|
| icews14 | low | 0.0015 | 0.0042 |
| ogbl_collab | unknown | 0.0084 | — |
| wd50k_0 | low | 0.0100 | 0.0052 |
| wd50k_33 | mid | 0.0301 | 0.0262 |
| wd50k_100 | high | 0.0682 | 0.0873 |

It ordered gain (low ≤0.01 < mid ~0.028 < high ~0.078, seed-robust) where label-free ρ_E could not, and
collapsed both irreducible-but-irrelevant cases (ICEWS 0.94→~0.003, collab 0.75→~0.008). **But the
adversarial panel found this is partly a structural confound**, and I confirmed it two independent ways:

- **Referee within-degree-decile permutation** (destroys z content, preserves the n_qual↔tail-degree
  coupling): **49–86%** of the WD50K lift survives content-scrambling — i.e. is confound, not content —
  because n_qual proxies tail in/PageRank (corr rising 0.01→0.26 along the dial) and relation-constrained
  corruption makes true tails ~1.5 log-degree higher than negatives. The **positive control** (0.774) used
  `z_leak = x[dst,0]` = tail log-out-degree, so it validated *structural-leak* detection — the confound
  channel itself — not clean content.
- **Degree-residualized z** (`--residualize`: strip from z the part predictable from the candidate tail's
  structural block, OOF): the dial partly collapses — but **not uniformly**:

| dataset | ρ_E^Y raw | ρ_E^Y **residualized** |
|---|---|---|
| icews14 | 0.0015 | **0.0000** |
| wd50k_0 | 0.0100 | 0.0139 |
| wd50k_33 | 0.0301 | **0.0164** (≈ halves → floor) |
| wd50k_100 | 0.0682 | **0.0770** (survives, ~unchanged) |

**What this means (the honest, de-confounded read).** The two de-confounding methods converge:
1. **The coarse relevant-vs-irrelevant separation is REAL and de-confound-robust.** WD50K_100 (highest gain)
   retains a genuine content signal after stripping candidate-tail structure (0.077, and ~half survives the
   referee's permutation), clearly above the floor; the two irreducible-but-irrelevant cases (ICEWS, collab)
   sit at ~0. ρ_E^Y *does* tell "content the model needs" from "content it cannot derive but does not need."
2. **The fine-grained monotone dial is NOT established.** WD50K_33's apparent mid-signal ≈ halves to the
   floor under de-confounding (mostly the degree confound), so the smooth low<mid<high ordering is partly an
   artifact. And there is **no clean content-orthogonal positive control yet** (the synthetic one leaked
   degree), so the *magnitude/calibration* of ρ_E^Y is not trustworthy at the desk stage.

- **Verdict (B): ρ_E^Y is a PROMISING but UNCONFIRMED relevance filter.** Maximally defensible claim: *after
  adversarial de-confounding, ρ_E^Y separates clearly-relevant (WD50K_100) from clearly-irrelevant
  (ICEWS/collab/low-qualifier) edge content and correctly flags the two irreducible-but-irrelevant cases; its
  fine-grained gain calibration is confounded by a qualifier-count↔tail-degree correlation and must be re-run
  in Step 5 with (a) a degree-residualized z as default, (b) a content-orthogonal positive control, and (c)
  the full qualifier-**entity** encoding (excluded here for label-free estimator tractability, which is why
  the measured lifts are small) before any survival/ordering law is asserted.* The word "rescue" from the
  first reading is withdrawn.

---

## Step-3 decision (pre-registered logic, on the audited evidence)

| clause | status |
|---|---|
| Label-free ρ_E as a gain **predictor/orderer** | **KILLED** — non-monotone band-invariantly; loses to the constant has-z? check (the pre-registered outcome) |
| Label-free ρ_E as a **one-directional necessary condition** (low ρ_E ⇒ no gain) | **SURVIVES** — supported by a real molecular point (0.06–0.09, low gain); no counterexample |
| Label-aware **ρ_E^Y** | **PROMISING, PARTIALLY VALIDATED** — coarse relevant/irrelevant separation survives de-confounding; fine calibration confounded → **Step 5** under a pre-registered de-confounding protocol |
| Theory note #1 (feature-determinism line-graph boundary) | **SHIPS** — never depended on the diagnostic; the primary contribution |
| Diagnostic robustness | caveated: node-feature-richness sensitivity (molecular range), and the ρ_E^Y degree confound, both recorded |

**The diligence pipeline did its job — twice.** It killed the diagnostic's ambitious label-free form at the
desk stage, and when the label-aware tier looked like a clean rescue, an adversarial red-team caught that the
rescue was substantially a degree confound and a leaky positive control — before any of it shipped as a
claim. What remains is honest and still worth a paper: the carved theory note (#1) as the primary
contribution, plus a diagnostic *direction* with a proven necessary-condition (ρ_E) and a
partially-validated, de-confounded relevance filter (ρ_E^Y) whose confirmation is a pre-registered Step-5
experiment. That is a sharper and more defensible outcome than either an unexamined "it works" or a flat kill.
