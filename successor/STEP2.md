# Step 2 — Spectrum computation + the kill-gate signal (sprint artifact)

**Status:** Step 2 IN PROGRESS (2026-06-25). Six of the seven spectrum points computed at a common
100k-edge budget with ρ_E v0.2.1 (16/16 gating tests). **The diagnostic law (#2) is trending strongly
toward its pre-registered kill; the one-directional claim and the theory note (#1) survive.** This is the
diligence pipeline refuting our own diagnostic's ambitious form — the intended behavior of the gate.

## The spectrum (common 100k-edge budget)

| dataset | ρ_E | status | n_edges | z-encoding | published edge-aware gain |
|---|---|---|---|---|---|
| FB15k-237 | **0.000** | schema_empty | 310k | none | none (DELTA null; Li et al. ACL'23) |
| WD50K (14% qual) | 0.611 | ok | 100k | n_qual + primary_qual_rel | low (StarE base) |
| WD50K_33 | 0.567 | ok | 100k | ″ | mid (StarE, ↑ with fraction) |
| WD50K_66 | 0.613 | ok | 49k | ″ | mid |
| WD50K_100 | 0.649 | ok | 31k | ″ | high (StarE +25 MRR) |
| ICEWS14 | **0.940** | ok | 91k | timestamp (day) | low (Radstok: static competitive) |

ICEWS ρ_E is **robust to time-bucket coarsening** (0.940 day → 0.951 year; `icews_buckets.py`) — the
irreducibility is real, not a granularity artifact.

## Finding 1 — the one-directional claim SURVIVES (so far)

FB15k-237: ρ_E = 0 (schema), published gain = none. Consistent with `low ρ_E ⇒ no content gain`. **No
counterexample found** — we have no dataset with ρ_E ≈ 0 yet a real edge-aware gain. The necessary-
condition (expressivity-level) claim of Proposition 1 stands on the evidence so far. (Molecular
bond-features — the predicted *low* point — is the key missing test of this direction; not yet computed.)

## Finding 2 — the PREDICTIVE claim is FAILING: irreducibility ≠ task-relevance (demonstrated)

ρ_E does **not** order datasets by published gain — it *anti*-orders them across the two non-empty-z
families:

- **ICEWS14: ρ_E = 0.94 (highest) but gain = low.** Event timestamps are highly irreducible from
  (h,r,t)+structure, yet timestamp-blind baselines are competitive (Radstok et al. 2021). This is the
  textbook **irreducible-but-not-task-relevant** case — content the model *cannot derive* but *does not
  need*. It is exactly the gap Proposition 1's honest scope names, now shown empirically.
- **WD50K_100: ρ_E = 0.65 but gain = high.** Qualifiers are less irreducible by our measure yet help a
  lot.

So the highest-ρ_E dataset has among the lowest gain: the label-free ρ_E is **worse than uninformative**
as a *predictor/orderer* of gain in the mid regime. It does not beat — it loses to — the binary schema
check. **This is the pre-registered kill condition for the diagnostic law as a gain predictor.**

## Finding 3 — the WD50K dial measures coverage, not per-edge irreducibility

Across the qualifier-fraction dial (14→33→66→100%), ρ_E is **flat within noise (0.567–0.649)**, not the
monotone rise a naive reading predicted. Two reasons, both structural:

1. **The dial varies coverage, not per-edge irreducibility.** The variants *subsample statements* to a
   target qualifier fraction; the per-qualifier content patterns are ~constant, so ρ_E (a per-edge
   dispersion ratio) is ~constant. StarE's MRR gain rises with fraction because it scales with *how many*
   edges carry helpful content — a population/coverage quantity ρ_E does not encode. The correct
   gain-predictor here would be ρ_E **× qualifier-fraction** (irreducibility × coverage), not ρ_E alone.
2. **The mild high-end tilt is an n-confound, not signal.** WD50K_100 (ρ = 0.649) has only 31k edges vs
   100k for the low-fraction variants; smaller n ⇒ weaker estimator ⇒ upward ρ_E bias (SCOPE axiom 4).
   The tilt is within the size-artifact band, so we do **not** read it as a real dial. (A fixed-n
   re-run — subsample all variants to 31k — is the clean control; queued.)

Encoding note: we excluded qualifier *entities* (pure per-instance, high-cardinality → would pin ρ ≈ 1
for any qualifier-bearing variant and choke the estimator). The coverage-not-irreducibility conclusion is
robust to this: *either* encoding fails to track a coverage dial, because fraction is not a per-edge
property.

## Kill-gate status (pre-registered)

- **Diagnostic law #2, as a label-free PREDICTOR of gain: trending strongly toward KILL.** ρ_E
  anti-orders gain across families (Finding 2) and cannot track the coverage dial (Finding 3). Per SCOPE
  §2, this drops #2 to a **companion observation** and ships the theory note #1 alone — *pending* the two
  honest completions below so the kill is final, not premature.
- **The one-directional necessary-condition (low ρ_E ⇒ no gain): intact** (Finding 1). This is the part
  Proposition 1 actually proves; it is not touched by the predictive failure.
- **The label-aware tier ρ_E^Y is now MOTIVATED, not refuted.** Findings 2–3 are precisely why the
  label-free metric can't predict gain; whether the task-relevance variant ρ_E^Y (or ρ_E × coverage)
  *can* is the open question that decides whether #2 survives as more than a companion note.
- **Theory note #1: unaffected.** It never depended on the diagnostic.

### To finalize (or overturn) the kill — Step 2 remainder + Step 3
1. **Molecular bond-features** (predicted low ρ_E, low gain) — the missing test of the one-directional
   claim's low end, and the point that would make Finding 1 load-bearing. Needs a bond-feature loader
   (rdkit/ogb not installed — raw-parse an OGB mol edge table or a public SMILES set).
2. **ogbl-collab** (our own z-on/off ablation) — the falsifiable mid-point.
3. **Fixed-n WD50K re-run** (all variants at 31k) — removes the Finding-3 n-confound.
4. **ρ_E^Y and ρ_E × coverage** on the assembled spectrum — the decisive test of whether *any* variant of
   the diagnostic predicts gain, or whether #2 is a companion observation and #1 ships alone.

## Honest read

We set out to build a diagnostic that predicts when edge-aware architectures help. On the evidence so far,
the **label-free** version does not — it measures irreducibility, and irreducibility is neither coverage
(WD50K) nor task-relevance (ICEWS). That is a real, if deflating, finding, and it is *exactly* the
pre-registered outcome the kill gate exists to catch cheaply. What survives — the theory note and the
one-directional necessary condition — is smaller than hoped but honest. Whether the label-aware tier
rescues the predictive claim is the next question; the gate says decide it on the assembled spectrum, not
on hope.
