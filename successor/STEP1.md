# Step 1 — Metric formalization (sprint artifact)

**Project:** "When Does Edge Content Matter?" (successor; see [../docs/SUCCESSOR_SCOPE.md](../docs/SUCCESSOR_SCOPE.md)).
**Status:** Step 1 IN PROGRESS (started 2026-06-25). DELTA itself remains CLOSED.

This document discharges the four **blocking exit criteria** from SUCCESSOR_SCOPE §2 and records the
label-aware decision. The reference implementation lives in [`rho_e/metric.py`](rho_e/metric.py) with
gating tests in [`rho_e/test_metric.py`](rho_e/test_metric.py) that mirror the red-team flaws one-to-one.

---

## (iii) Formal statement of the informational conservativeness claim

Setup. Fix a probability space carrying an edge-valued random element with components
`U, V` (endpoint node ids), `T` (relation type), `Z` (edge-attribute vector), and an arbitrary task
target `Y`. Let `W := (x_U, T, x_V)` collect the **z-free observables** of the edge, where `x` is the
node-observable map from the x-specification below (§(ii)); by that specification, `x` is computed
without reference to any edge attributes, so `W` is well-defined independently of `Z`.

**Proposition 1 (informational conservativeness).**

**(a) Exact determinism.** If `Z = φ(W)` a.s. for some fixed measurable `φ`, then `σ(W, Z) = σ(W)`
**up to P-null sets** (literal equality under the exact identity `Z = φ(W)`; under a.s. equality it
holds modulo completion — and the information/risk consequences below are null-set-insensitive either
way). Consequently `I(Z; Y | W) = 0` and the Bayes risk satisfies `R*(Y | W, Z) = R*(Y | W)` for every
loss. *No function of the inputs-with-Z exists that is not (a.s.) a function of the inputs-without-Z.*
*Proof sketch:* conditional degeneracy — `P(Z ∈ · | W = w) = δ_{φ(w)}` a.s., so the conditional joint
factorizes and the conditional mutual information vanishes; the feasible predictor sets for `σ(W,Z)`
and `σ(W)` coincide a.s., so the risk infima agree for every loss. ∎

**(b) Approximate determinism.** Suppose `E[ d²(Z, φ(W)) ] ≤ ε` for some measurable `φ`, where `d` is
the (standardized) metric underlying the dispersion `D` used by ρ_E. Fix a predictor class `F` whose
members are `L`-Lipschitz in the `z`-argument (w.r.t. `d`) and closed under the substitution
`g(w, z) ↦ g(w, φ(w))`, and a loss `L_ℓ`-Lipschitz in the prediction (boundedness is not needed for the
rate — only for existence/finiteness of minimizers). Then the **class-relative** best risks satisfy
`|R_F(Y | W, Z) − R_F(Y | W)| ≤ L·L_ℓ·√ε`: plug `φ(W)` into the z-slot of the `(W,Z)`-optimal predictor
to build a `W`-only predictor, and bound the loss gap by Lipschitz composition + Jensen
(`E[d] ≤ √(E[d²])`). The constant and the `√ε` rate are exact.
*Honest qualifier:* the Lipschitz restriction is necessary — the fully nonparametric version is FALSE,
and small `ε` need **not** bound `I(Z; Y | W)`. Counterexample shape: `Z = φ(W) + √ε·Y` — the residual
has `L²`-mass `ε` yet determines `Y` exactly; exploiting it needs a decoder with Lipschitz constant
`1/√ε`, which is exactly how the bound survives. The claim is therefore stated for smooth-in-`z`
predictor classes, which covers the architectures at issue (attention/MLP message functions are
Lipschitz on compacts). ∎ (sketch)

**(c) Estimation link.** ρ_E (as implemented: key-level evaluation on `k = (U, T, V)`, z-free `x`,
best-of-pair predictability `Ṽ`, both terms of `rho_dim` in the **same within-type frame**,
pooled-then-clipped) is an **asymptotically conservative (upper) bound** on the normalized beyond-type
residual dispersion of (b) — *not* a consistent estimator of it: the fixed-capacity estimator pair
(capped-k kNN, default-capacity GBT) is not Bayes-consistent, so ρ_E converges to a value **≥** the
truth. The finite-n error decomposes into two channels with different signs:
- *approximation error* (population): predictor suboptimality can only understate `Ṽ`, hence
  **overstate** ρ_E — one-directionally conservative;
- *finite-sample fluctuation*: **not sign-controlled** — the max over the estimator pair
  (clip-then-max) carries a maximal-selection ("winner's-curse") upward bias in `Ṽ` of order the CV
  standard error, i.e. a small **anti-conservative** bias in ρ_E. This third, *statistical* channel is
  not closed by construction; it is controlled by the fold-CI and fixed-budget ρ_E(n) reporting (SCOPE
  axiom 4), which is load-bearing *here*, not merely for cross-dataset comparability.

Three *mechanical* anti-conservative channels are closed **by definition**:
- *twin leakage* — key-level evaluation guarantees no test row shares its `(U, T, V)` key with a train
  row, so `Ṽ` cannot be inflated by lookup of a recurring triple's twin;
- *x-leakage* — the x-specification forbids any feature computed from edge attributes, so `Z` cannot be
  "predicted" from a disguised copy of itself;
- *frame mixing* — the within-key share and the chance-corrected `Ṽ` are normalized in the same
  within-type frame, so type-determined mass cannot dilute the irreducible term. (This channel existed
  in the first v0.2 implementation and was caught by numeric probe in the adversarial audit — closed in
  v0.2.1.)

**Corollary (the one-directional claim, final form).**
> **Exact case** (determinism / degenerate dims, ρ_E = 0): the edge attributes carry **no information**
> beyond the z-free observables (`I(Z;Y|W) = 0`), and no advantage of any kind is available.
> **Approximate case** (ρ_E ≈ 0): **no expressivity-level (best-achievable-risk) advantage** is
> available to *smooth-in-z* edge-aware architectures, up to the `O(L·L_ℓ·√ε)` term of (b). The
> information clause is deliberately **not** asserted here — per (b)'s qualifier, small residual mass
> need not bound `I(Z;Y|W)`; only the smooth-class risk conclusion is licensed.

**Named gap (out of scope, threat to Step 3).** (a)–(c) bound the *informational/expressivity*
advantage only. They do **not** bound differences in optimization dynamics or sample efficiency between
an architecture that *materializes* `z` on edges and one that must re-derive `φ(W)` implicitly inside
its message function (finite width/data; representations overwritten across layers). Benchmark gains
bundle both. The Step-3 meta-analysis validates against benchmark gains and therefore tests the
informational claim only approximately; residual positive deltas at ρ_E ≈ 0 (e.g. D-MPNN's +0.8–2.7%
AUC with largely endpoint-inferable bond features) are the *predicted signature* of this gap, not a
refutation of the claim.

## (ii) The z-free x-specification (per dataset class)

`x` must be **z-free** (computable with all edge attributes deleted) and **model-free** (no trained
embeddings — that would resurrect the Gate-C circularity). Fixed specification:

| Dataset class | x_v definition |
|---|---|
| Featureless KGs (FB15k-237, WN18RR, ICEWS entities, WD50K) | structural set: log(1+in-degree), log(1+out-degree), per-relation-type incidence histogram (top-R types by frequency, in/out separately, row-normalized; R=50 default), PageRank (α=0.85, power iteration), local clustering coefficient on the undirected simple graph |
| Temporal KGs | the structural set computed on the **timestamp-stripped** multigraph collapsed to a simple typed graph (timestamps are `z` — they must not shape `x`) |
| Hyper-relational KGs (WD50K) | structural set on the **qualifier-stripped** triple graph, **collapsed to simple** (qualifiers and duplicate main-triples are `z`) |
| Attributed graphs (molecular, materials) | the given intrinsic node features (atom type, etc.) — verified to contain no bond/edge-derived fields — plus the structural set |
| ogbl-collab | the provided 128-d node features + structural set on the year/weight-stripped simple graph |

Rule of construction: **delete all `z`, collapse parallel edges to a simple typed graph, then compute
`x`.** Edge *multiplicity* across a stripped z-coordinate is itself `z` (a lossless count of z-stamped
events — repeated timestamps, co-authorship weight, qualifier-bearing duplicates), so it must be
collapsed away, not retained as structure. Relation *types* τ are not `z` and may shape `x` (the
incidence histogram); anything else on edges may not.

## (i), (iv) — discharged in the implementation

- **(i) Key-level (twin-leakage-proof) evaluation on `(u, τ, v)`** is hard-coded in `rho_e/metric.py` —
  implemented as KFold over *unique keys*, which is leakage-equivalent to GroupKFold on edges (no key
  can straddle folds); there is no ungrouped code path. Estimand caveat (documented; v0.3 item): `Ṽ`
  weights keys equally while dispersion shares are edge-weighted; heavy-tailed key sizes are flagged
  via the reported `key_size_max` / `key_size_p95`. Gating test `test_antileakage_recurring_triples`
  constructs the exact red-team counterexample and asserts key-level evaluation reports high ρ_E while
  simulated naive edge-level CV demonstrably falls into the trap.
- **(iv) Degenerate-case conventions** are hard-coded: zero-within-type-dispersion dims are excluded
  (`Ṽ_j := 1, w_j := 0`); all-excluded ⇒ ρ_E := 0 with status `schema/constant-degenerate` (extended
  empty-z convention); the 0/0 chance correction cannot arise. Gating tests cover type-determined,
  constant, and schema-empty z.

## The label-aware decision (forced by the noise-dim case; decided, not deferred)

**Decision: two-tier.** The **primary** metric is the label-free ρ_E with the one-directional
informational claim (Proposition 1) — it requires no task labels and its conservative direction is
exact. A **secondary, label-aware relevance filter** ρ_E^Y is defined for datasets where a task target
exists: per dim, the out-of-fold predictive value of `z_j` for `Y` *beyond* `W` (same estimator pair,
same grouped CV; skill of `f(W, z_j)` minus skill of `f(W)`). Its role is **confined to Step 3**: when
the kill-gate alignment scatters, ρ_E^Y separates "irreducible but task-irrelevant" (noise) dims from
task-relevant ones, addressing the noise-dim threat without weakening the primary claim. ρ_E^Y makes
**no** appearance in the headline law; the law remains one-directional and label-free.

## Step-1 exit status

| Criterion | Status |
|---|---|
| (i) key-level (twin-leakage-proof) evaluation definitional | DONE (implementation + gating test) |
| (ii) z-free x-spec per dataset class | DONE (this doc, incl. multiplicity-collapse rule; loaders must comply) |
| (iii) formal conservativeness statement | DONE (Proposition 1, adversarially refereed: (b) independently re-derived as correct; (c) rescoped to "asymptotically conservative bound" with the maximal-selection statistical channel named) |
| (iv) degenerate-case conventions | DONE (relative two-criterion exclusion + summary-level collapse; implementation + gating tests) |
| adversarial audit of v0.2 implementation | DONE — 2 metric-breaking bugs found by numeric probe (frame mixing; per-call entropy base) and fixed in v0.2.1; 15/15 gating tests |
| label-aware decision | DONE (two-tier; ρ_E^Y confined to Step 3) |
| non-blocking: time-bucket sensitivity (ICEWS) | OPEN — Step 2, report ρ_E across bucketings |
| non-blocking: mixed-scale weighting details | addressed by global standardization; verify in Step 2 |
