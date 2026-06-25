# Phase 78 — Coherence-clean edge-content controls (de-confounding the Phase-77 thread)

## Status: COMPLETE (n=15, 2026-06-24) — SUBSTANTIVE REFUTE.

## RESULT

Primary test (`pec` vs `shuffle_full` @3p, the only trained depth, seed-level paired-t, **n=15**):

```
pec - shuffle_full @3p: mean = +0.0029   p = 0.034   95% CI [+0.0003, +0.0054]
lesion lever (median 3p readout displacement) = 0.65   (>> disp_min 0.02 -> NOT a no-lever artifact)
=> SIGNIFICANT-BUT-NEGLIGIBLE: statistically detectable, but 5x INSIDE the +-0.015 meaningful band
   => substantively REFUTES the thesis — edge-instance identity is used, but NOT meaningfully.
```

3p MRR ladder (mean over 15 seeds):
`pec 0.2022 > shuffle 0.1998 ~ shuffle_full 0.1993 > rrprior 0.1966 > pec_lesion 0.1941`

- **pec − rrprior @3p = +0.0045 (p=0.001):** the operator does beat a pure relation-type prior — but only
  by a hair. `instance_fraction = (pec−shuffle_full)/(pec−rrprior) = 0.63` → ~63% of that tiny edge is the
  instance axis, ~37% operator-structure-beyond-type. The instance slice is the *larger part of a
  negligible whole*, NOT support for the thesis.
- **ef0_shuffle ≈ pec** (pec−ef0_shuffle @3p = +0.0012) → the type-dominated `ef0` carries ~no instance
  signal (GATE C confirmed operationally; the sanity control behaved).
- **pec − pec_lesion = +0.008 (p<0.001)** but `shuffle_full` (trained under the perm) ≈ pec → the operator
  *adapts* to wrong-bound content; the lesion gap is distribution-shift, not instance use.
- The result was **stable across every interim read from n=5 to n=15** (the consistent +0.003 margin only
  crossed into statistical significance ~n=13 as the SD-tight estimate sharpened — a significance story,
  not a substance story).

**Conclusion.** The pinned thesis ([[query-time-edge-composition]]) resolves OPEN → **CLOSED-NEGATIVE**.
Across Phases 66–78, edge-to-edge composition (structural *and* query-time/instance) shows no *meaningful*
task advantage under either AOT or JIT — even with the eval redesigned to reward it and the lesion verified
to have a lever. Edge attention's value is generic **capacity**, not composition. Scope: refuted on this
testbed (edge-sampled FB15k-237 frac=0.10, degree ~4.7, type-dominated) under a FROZEN encoder; end-to-end
co-training and the "JIT ≈ 2× AOT" readout question remain the only unexplored doors (Phase 79 candidate).

### Operational notes (for reproducibility)
- A 4-lens adversarial red-team killed the v1 design *pre-run* (false-REFUTE point-estimate gate +
  `W_ctx`-loophole `shuffle`); fixes below.
- The run crashed once overnight (silent CUDA OOM, no traceback). Root cause: `ensure_encoder` called
  `build_condition_adjs`, which also builds a useless **17.7M-pair hops=2 adjacency** → reserved-cache
  bloat → 12GB GPU maxed → 10× thrash + OOM. Fixed to build **hops1 only** + `empty_cache`. A self-healing
  2-hour cron auto-restarted on crash (resumable per arm/seed) and finalized at 15/15.

---

## (Pre-registration, below — design as committed before the run)

## Status: PRE-REGISTERED + redesigned after adversarial red-team. Running the n=15 sweep.

Phase 77 ended **FAIL** ("query-time edge composition NOT robustly demonstrated"): the corrected
seed-level test left exactly one thin, seed-fragile, extrapolation-only thread alive — a *hint* that
`pec` beats the relation-type prior `rrprior` at 5p. That hint is uninterpretable because **`rrprior`
differs from `pec` on TWO axes** (attention on/off AND message type), so `pec > rrprior` cannot be
attributed to edge-**instance** content. Phase 78 builds the de-confounding control that changes ONE
axis (edge-instance identity), pre-registers the decision, and powers it to n=15.

## What the v1 design got wrong (caught by a 4-lens red-team BEFORE the ~30h run)

A 4-adversary red-team (mechanism / statistics / confounds / plumbing) returned **2 `kills_design`**:

1. **Invalid accept-the-null.** v1 declared "REFUTED" on a bare `|point estimate| < 0.005` gate. That is
   not an equivalence test, and 0.005 is *unreachable* by any valid equivalence test at n≤15 (paired CI
   half-width ≈ 0.011–0.022). So "refuted" could only fire by sampling chance — a **false-REFUTE
   artifact**, symmetric to the Phase-77 false-PASS.
2. **`shuffle` was not a clean one-axis control.** Per Phase-77 GATE C the per-instance signal lives in
   the *endpoint* features `nf_src/nf_tgt`. v1's `shuffle` permuted the endpoint-built `ej` but left the
   `W_ctx(nf_src,nf_tgt)` endpoint channel (real endpoints, both arms) intact. So `pec≈shuffle` was
   biased toward false-refute (working endpoint channel retained) and `pec≫shuffle` toward false-confirm
   (an incoherent-`(ej, endpoint, z_src)`-triple distribution-shift penalty mimics composition). On a
   degree ~4.7, type-dominated graph the within-type permutation moves so little that `pec≈shuffle` was
   partly pre-determined.

The mechanism *implementation* was verified sound (within-type/deterministic/device-safe permutation;
real routing → frontier-invariant; `compose_scores` bit-identical; seed = replication unit;
inherited-vs-fresh eval query/mask/order-identical).

## The redesigned lesion ladder (`delta/path_compose.py`, `LESION_MODES`)

All arms are `pec` byte-for-byte except **which edge content the operator sees**, via a fixed
within-relation-type permutation (`set_edge_perm`). Routing always uses the REAL graph (frontier
invariant). Permutation is fixed per seed → trained AND evaluated under it (the operator's best shot at
wrong-bound content).

| arm | what is permuted | role |
|---|---|---|
| `pec` | nothing | reference (the learned operator) |
| **`shuffle_full`** | the whole edge identity: `ej` **and** the `W_ctx` endpoints | **PRIMARY control** — coherent full-identity swap, closes the endpoint loophole |
| `shuffle` | `ej` only (W_ctx endpoints stay real) | diagnostic: the `ej` channel alone |
| `ef0_shuffle` | only the frozen `ef0` (pre-injection) | sanity: `ef0` is type-dominated (GATE C var≈0) → near-noop |
| `rrprior` | (uniform attention + type-only message) | the relation-TYPE prior, for `instance_fraction` |

Free, from the trained `pec` model:
- **`pec_lesion`** — `pec` weights evaluated under the `shuffle_full` perm (train-real / eval-shuffled):
  the distribution-shift discriminator. If `pec≫shuffle_full` but `pec_lesion≈shuffle_full`, the gap is
  instance-binding sensitivity; if `pec_lesion≪shuffle_full`, training adapted to the shuffled
  distribution and the gap is contaminated by distribution-shift.
- **displacement positive-control** — median relative L2 change in `pec`'s readout scores on the penult
  set when the perm is applied. If ~0, the lesion has **no lever** on this testbed → a `pec≈lesion`
  result is testbed-capacity, **NOT** a refutation.
- **scramble strength** — fraction of edges moved + mean `||ej[perm]-ej||/||ej||`.

## Pre-registered decision (`experiments/phase78_analyze.py`)

PRIMARY test = `pec` vs `shuffle_full` at **3p** (the only TRAINED depth; 4p/5p are extrapolation,
reported as secondary/exploratory only). Seed-level paired-t. δ = 0.015 (TOST equivalence margin, the
smallest concludable at n=15 given the Phase-77 SESOI ≈ 0.017 and optimistic SD ≈ 0.013). disp_min = 0.02.

- **CONFIRM** — paired-t p<0.05 AND 95% CI excludes 0 → edge-instance identity is load-bearing; the
  thread is REAL; query-time composition uses the specific edge, not just its type.
- **REFUTE** — TOST: paired-t 90% CI ⊂ [−δ,+δ] **AND** the lesion has a lever (displacement ≥ disp_min)
  → `pec ≈ shuffle_full`; edge-instance identity not used; retire the mechanism.
- **NO-LEVER** — equivalent but displacement < disp_min → testbed-capacity INCONCLUSIVE, not refutation.
- **INCONCLUSIVE** — CI wider than ±δ → underpowered; add seeds. Never "refuted" from a non-sig point est.

`instance_fraction = (pec − shuffle_full)/(pec − rrprior)` @3p: >0.5 instance load-bearing; <0.2
type-prior dominates; between = mixed. The confirmatory test is the single pre-registered contrast (no
multiplicity penalty); everything else (other arms, 4p/5p, percentile bootstrap) is exploratory.

## Run

`n=15` seeds (5 existing-encoder + 10 fresh): `42,123,456,7,99,11,23,5,17,31,2,8,13,29,37`. Every arm
trained IN-SESSION per seed (no cross-run inheritance → no batch effect). Resumable per (arm,seed).
Edge-sampled FB15k-237 frac=0.10 (11,563 ents / 237 rels / 27,211 train edges / 18,159 test), fixed
sample_seed=42; only the MODEL seed varies. ~5 arms × 15 seeds + 10 encoders.

## Honest priors

A clean negative is still the most likely outcome (Phase-74 [R,R]-prior; degree ~4.7 type-dominated
testbed; 11 prior phases of caught artifacts). With the displacement positive-control, a structurally
pre-determined `pec≈shuffle_full` is correctly labeled testbed-capacity rather than a refutation — so
the experiment can no longer manufacture either a false REFUTE or a false CONFIRM.
