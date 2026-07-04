# Successor project scope — "When Does Edge Content Matter?" (post-DELTA)

**Status:** SCOPED + KILL-GATED (2026-06-25). **Steps 1–3 now COMPLETE — see
[../successor/STEP3.md](../successor/STEP3.md) for the audited verdict** (label-free ρ_E killed as a gain
predictor, survives one-directional; label-aware ρ_E^Y promising-but-unconfirmed after de-confounding;
theory note #1 ships). This document is the **pre-registration** and is left intact below; the Step-3 file
records what actually happened against it. This is **not** a DELTA phase — DELTA is CLOSED
([DELTA_FINAL.md](DELTA_FINAL.md)). This document scopes the successor bet that DELTA's negative uniquely
equips: a theory note + a pre-training diagnostic for *when edge-aware architectures can help*, with
DELTA's 79-phase null as the anchor at the low end of the curve.

Feasibility was established by a 4-lens literature sweep (32 papers) + adversarial synthesis
(workflow `delta-successor-feasibility`, 2026-06-25). Verdicts below are its output, stated harshly.

---

## 1. The two contributions and their honest novelty status

### #1 — Theory note: the feature-determinism boundary for line-graph attention

**Verdict: partially claimed. The strong version is FALSE as originally conceived.**

- **Preempted (named):** Huang, Romero Orth, Ceylan & Barceló (NeurIPS 2023, arXiv:2302.02209, Thm 5.1)
  already prove the KG-level instance (the path/edge view adds no structural expressivity over
  relation-conditioned node message passing; query-conditioning is the lever). Chen–Li–Bruna (ICLR 2019)
  and NBA-GNN (arXiv:2310.07430) prove line-graph message passing **does** gain structural power via
  non-backtracking operators with zero edge content — so *"only capacity, never structure"* is false
  unconditionally. Yang & Huang (arXiv:2410.16138) already map part of the structural "when L(G) helps"
  boundary (strongly-regular/CFI families).
- **Free slot:** the **feature-determinism axis** — no paper factors L(G)'s two channels (structural walk
  sensitivity vs irreducible edge-instance content) or states the determinism condition as an
  information-redundancy result for *featured* graphs.

**The carved statement to prove** (from the synthesis, verbatim in substance):

> Setting: directed labeled graph G, node features x, edge types τ(e), edge features z(e).
> **H1 (determinism):** z(e) = φ(x_u, τ(e), x_v) for fixed measurable φ. **H2 (simplicity):** at most one
> edge per (u, τ, v). *Claim:* any function computed by an L-layer attention-MPNN on the (standard,
> backtracking-inclusive) line graph L(G) is computed by an (L+O(1))-layer attention-MPNN on G with
> polynomial width increase — zero content-composition gain; capacity-only.
>
> **Non-negotiable carve-outs:** (1) the non-backtracking channel is explicitly quotiented out (cite
> Chen–Li–Bruna, NBA-GNN as the channel the theorem does NOT cover — without this the theorem is false);
> (2) stated for general featured graphs with Huang et al. 2023 as the KG instantiation (else it overlaps
> their Thm 5.1); (3) "capacity" formally = width/parameter increase, "structure" = refinement of
> distinguishing power; (4) multigraphs are *excluded by hypothesis* — that exclusion IS the positive
> prediction. The converse (irreducible AND task-relevant content ⇒ strict gain) stays **empirical**
> (StarE, ALIGNN), not claimed as a theorem.

Must also disarm: Edge Transformers (NeurIPS 2021 — dynamic pair-state, not static edge identity) and
KGFM's relation-TYPE line graph (ICML 2025, arXiv:2502.13339 — a line graph that *does* add power, over
types not instances). Standalone this is a workshop-note; it earns its keep paired with #2.

### #2 — Diagnostic law: the Edge-Content Irreducibility ratio predicts edge-aware gain

**Verdict: novel-in-object, crowded-in-method, with a pre-published failure mode.**

- **Free slot:** nobody has a within-relation edge-content statistic as a pre-training predictor of
  edge-aware-vs-(node-only + type-prior) gain.
- **Occupied method template (must be named):** Platonov et al. (NeurIPS 2023 D&B, arXiv:2209.06177) own
  "dataset statistic predicts graph-aware gain" (label informativeness + the axiom desiderata); Luan et
  al. (NeurIPS 2023, arXiv:2304.14274) own the hypothesis-test framing; Dir-GNN (LoG 2023) ties an
  edge-structural property to payoff. The genre's failure mode is documented: homophily-vs-gain ρ≈0.46
  (Platonov ICLR 2023); design-rule reversal under benchmark composition (arXiv:2606.10249).
- **The kill criterion (pre-registered):** the law lives or dies in the **MID regime**. On pure triple
  stores the ratio is 0 *by schema*; at the high end (qualifiers, geometry) gains are already published.
  If the ratio does not discriminate partly-redundant attributes (type-predictable timestamps,
  endpoint-inferable bond features) **better than the binary check "does this dataset have per-edge
  attributes?"**, #2 is dead — drop it to a companion observation and ship #1 alone as a short note.
- **One-directional claim only:** low ratio ⇒ no content gain available (necessary-condition test).
  High ratio does NOT guarantee gain — irreducibility ≠ task-relevance. The converse stays empirical.

---

## 2. STEP 1 ARTIFACT — the model-free metric, v0.2 (post-adversarial-red-team)

**Why redefined:** DELTA's Gate-C (`experiments/phase77_step0_gates.py`) is computed on a *trained
model's* edge state `ej` — circular as a "pre-training" diagnostic. The successor metric must be a
**data-level** quantity.

**Red-team provenance:** v0.1 was adversarially reviewed (2026-06-25) before being committed. The review
found **two breaks-the-metric flaws** (twin-edge CV leakage on multigraphs; `x` undefined on featureless
KGs — both anti-conservative, i.e. producing false "no gain" on exactly the ogbl-collab/ICEWS rows the
project lives on) and found the v0.1 conservativeness claim **false as stated** (it ignored the
information-vs-accessibility gap). v0.2 below incorporates all fixes; the four blocking items are Step-1
exit criteria.

### Definition — Edge-Content Irreducibility, ρ_E

Setting: edges e = (u, τ(e), v) with edge-attribute vector z(e) (possibly empty), node observables x.

> **ρ_E = 1 − Ṽ**, where **Ṽ** is the cross-validated, chance-corrected predictability of z(e) from
> (x_u, τ(e), x_v), using a fixed, disclosed, simple estimator class — no task model, no trained GNN.

- **Estimator class (fixed, report both):** k-NN and gradient-boosted trees, out-of-fold, with
  **mandatory GroupKFold on key k = (u, τ(e), v)**. Random edge-CV leaks recurring-triple "twins" across
  folds (a k-NN lookup then "predicts" the test twin exactly), falsely driving ρ_E → 0 on multigraph and
  temporal datasets — an *anti-conservative* failure. Grouped CV is definitional, not an option.
- **Node observables x (part of the definition):** restricted to **z-free, model-free** quantities —
  structural features (degree, relation-type incidence histograms, PageRank) plus intrinsic node
  attributes where they exist. **Never** pretrained embeddings (reintroduces the Gate-C circularity) and
  **never** anything computed from edge attributes (x-leakage → false ρ_E ≈ 0). For featureless KGs
  (FB15k-237, ICEWS entities), x = the structural set; the exact per-dataset-class specification ships
  with the tool.
- **Normalization:** categorical dims: (acc − acc₀)/(1 − acc₀), acc₀ = *out-of-fold* type-only majority;
  continuous dims: skill score 1 − MSE_oof(model)/MSE_oof(type-conditional mean) with type means also
  estimated out-of-fold; **pool folds, then clip** to [0,1] (per-fold clipping biases Ṽ upward). The
  type-only baseline makes type-determined content contribute 0 to ρ_E by construction.
- **Per-dim aggregation:** weights w_j = each dim's share of within-type dispersion computed on
  **globally standardized** dims (otherwise unit choices rewrite the weights, and pure-noise dims get
  maximal weight and drag ρ_E → 1, gutting mid-regime discrimination). **The per-dim vector (Ṽ_j, w_j)
  is the primary report; the scalar ρ_E is a summary.**
- **Degenerate-dim convention:** any dim with zero within-type dispersion is excluded (Ṽ_j := 1,
  w_j := 0) — this also resolves the 0/0 in the chance correction when acc₀ = 1. If ALL dims are
  excluded, ρ_E := 0 under the **extended empty-z convention** ("no within-type dispersion", covering
  constant-z as well as schema-empty) — reported as *schema/constant-degenerate*, never as a measured
  value.
- **Grouped decomposition (multigraphs / recurring triples):** by the law of total variance, irreducible
  content = the **exact within-key term** (ρ_grp = normalized E[Var(z | k)] — irreducible for *any*
  predictor, since identical inputs carry differing targets) **plus** the unpredicted share of the
  across-key term (predict the *group summaries* E[z | k] from (x_u, τ, x_v)). ρ_grp is a **component
  (lower bound) of ρ_E, not independent corroboration**; report the two terms separately. For simple
  graphs the within-key term vanishes and the predictive term is the whole story.

### Properties + axiom mapping (Platonov arXiv:2412.09663-style checklist)

1. **Conservative direction — RESCOPED (the v0.1 wording was false; red-team flaw 3).** The safe
   one-directional claim is **informational**: *low ρ_E ⇒ no expressivity-level content gain is
   available* — the content is derivable from inputs the node-only model already has. Within that scope,
   estimator weakness only *inflates* ρ_E (errs toward "maybe gain"), **provided** the two mechanical
   anti-conservative channels are closed by definition: GroupKFold (twin leakage) and z-free x
   (x-leakage). What the claim does **NOT** rule out: finite-sample / optimization gains from
   *materializing* derivable content on edges (an inductive-bias effect — plausibly the D-MPNN
   +0.8–2.7% bond-feature bumps). This **information-vs-accessibility gap is a named threat to the
   Step-3 meta-analysis**, which validates against *benchmark* gains: the informational claim is exact;
   the benchmark-level claim is explicitly approximate.
2. **Constant baseline:** under H1-determinism (z = φ(x_u, τ, x_v), learnable φ), ρ_E → 0;
   type-determined z gives ρ_E = 0 exactly (via the degenerate-dim convention).
3. **Monotonicity:** mixing in endpoint-predictable content lowers ρ_E by construction of Ṽ.
4. **Consistency:** standard cross-validation arguments; report fold-CIs **and ρ_E(n) curves at a fixed
   common edge budget** — estimator skill grows with n, so large datasets get lower ρ_E from sample size
   alone; comparisons are made at matched budgets.
5. **Cross-dataset comparability: a GOAL, not yet an earned property.** Acc-based and R²-based per-dim
   scores are not strictly commensurable in one mean; mitigations are the fixed edge budget, the chance
   corrections, and the per-dim vector as the primary cross-dataset object.

**Step-1 exit criteria (BLOCKING — the sprint does not proceed to Step 2 without these):**
(i) GroupKFold in the definition; (ii) the z-free x specification per dataset class; (iii) a **formal**
statement of the rescoped informational conservativeness claim (the informal v0.1 version was wrong —
this is now load-bearing, not an open item); (iv) the degenerate-case conventions. Non-blocking Step-1
items: time-bucket sensitivity for temporal ρ_E (ICEWS); the label-aware variant decision (**decide in
Step 1** — the pure-noise-dim case forces the question, it cannot be deferred); mixed-scale weighting
details.

### The kill-gate, operationalized

Compute ρ_E on the 7-dataset spectrum (§3). **Pass** iff the mid-regime ordering is
`bond features (low-mid) < timestamps (mid) < qualifiers/geometry (high)` *and* the ρ_E-vs-published-gain
alignment beats the binary schema check under normalized effect sizes — **scored on non-empty-z datasets
only** (the schema-degenerate rows agree with the binary check by construction and would dilute the very
contrast the gate measures). Alignment uses **content-isolating feature-ablation deltas only** (same
architecture, z on/off — ALIGNN-style), never architecture-vs-architecture deltas, which bundle the
structural line-graph channel the companion theorem explicitly quotients out; temporal-split vs random-CV
mismatches are flagged when citing published temporal-KG gains. **Fail** ⇒ #2 drops to a companion
observation; ship #1 alone.

---

## 3. Validation spectrum (what exists vs what we must run)

> **Column clarification (added post-hoc, no content changed):** the middle column is the *predicted*
> ρ_E, **not** the published gain band. (A Step-3 referee misread "mid" for temporal KGs as a *gain*
> band; the gain is read from the Evidence column — e.g. temporal = *low* per Radstok. The Step-3 gate
> outcome is invariant to that choice; see STEP3.md §A.) Measured ρ_E often *differed* from the prediction
> — notably temporal KGs came in at 0.94, not "mid" — which is itself part of the Step-3 finding.

| Dataset | expected ρ_E | Existing evidence | We must run |
|---|---|---|---|
| FB15k-237 / WN18RR (public splits) | ~0 (schema) | DELTA's 79-phase null + Li et al. ACL 2023 | ρ_E (trivial) + **one** edge-attn-vs-node-only replication on public splits (kills "custom subgraph, n=1") |
| Molecular, bond-type features (Chemprop/OGB) | low | D-MPNN: bond features ≈ +0.8–2.7% AUC, "not always" | ρ_E only (are bond features endpoint-inferable?) |
| Temporal KGs (ICEWS14/05-15, GDELT) | mid | Radstok et al. 2021: timestamp-blind baselines competitive | ρ_E + likely ONE controlled static-vs-temporal pair — **the key discriminative test** |
| ogbl-collab (year+weight, true multigraph) | mid | no clean isolating ablation exists | our own ablation — the falsifiable mid-point |
| KG numeric literals (LiteralE-style) | mid | numeric-aware > numeric-blind, esp. extreme values | ρ_E only; secondary |
| **WD50K qualifier dial (0/33/66/100)** | high | StarE: up to +25 MRR, monotone in qualifier fraction | ρ_E across the dial; optionally one architecture across it (**the crown-jewel figure**) |
| Materials w/ bond angles (ALIGNN) | high | ALIGNN: up to ~85% rel. error reduction | ρ_E only; published ablation is the evidence |

## 4. Plan (kill-gated), effort, risk, venues

1. **Step 1 (~1 wk, desk):** finalize the metric v0.1 → v1 against the axiom checklist. *Abort if no
   definition survives.*
2. **Step 2 (~1–2 wks, desk):** ρ_E on all 7 datasets → the first ratio table.
3. **Step 3 (KILL GATE, ~1 wk, desk):** meta-analysis vs published deltas, normalized effect sizes.
   *Pre-registered abort per §2.*
4. **Step 4 (parallel, ~2–3 wks):** draft the carved theorem; **arXiv early** (scoop insurance — the
   likely scoop vector is a one-paragraph corollary from the Huang/Ceylan/Barceló orbit).
5. **Step 5 (~4–6 wks, single GPU):** the three evidentiary holes: public-split FB15k-237/WN18RR
   replication; ogbl-collab ablation; optionally the WD50K dial. DELTA guardrails throughout
   (seed-level tests, TOST, tuned type-prior baselines, leakage-clean splits).
6. **Step 6 (~1–2 wks):** held-out dataset-FAMILY test (fit threshold on KGs, predict
   molecules/materials) — the anti-(arXiv:2606.10249) robustness move.
7. **Step 7:** one package (theorem + diagnostic + spectrum + DELTA null as the ρ≈0 anchor) → **TMLR**
   (primary); LoG, NeurIPS D&B, MLRC as alternates. Release the diagnostic tool.

**Effort:** kill-or-continue = Steps 1–3, ~3–4 weeks, zero GPU. Full package ~3–4 months part-time,
modest single-GPU. **Scoop risk:** moderate-low; mitigated by early arXiv of #1.

## 5. The skeptic case (pre-registered, verbatim in substance)

Most likely fizzle: the mid regime fails to discriminate — low end 0 by schema, high end is other
people's published results with our axis labels, mid points scatter (replicating homophily's ρ≈0.46
weakness one level down) — and the law collapses into a schema check while the carved theorem reads as
"deterministic functions add no information, obvious once stated." End state: a technically-correct
paper cited as a footnote. Secondary fizzle: cross-paper effect-size normalization (MRR vs AUC vs MAE)
proves unconvincing. **The response to either is the kill gate, not persistence.**
