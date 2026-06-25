# DELTA — Final Findings Declaration (PROJECT CLOSED)

**Status:** CLOSED, terminal at Phase 79 (2026-06-25). This document is the authoritative, adversarially-
disclosed record of what DELTA found. It supersedes any earlier framing in the phase docs, `research_state.json`,
the MkDocs site, and the NeurIPS draft where they conflict.

> **One line:** DELTA set out to show that edge-to-edge attention gives a knowledge-graph reasoner a real
> multi-hop *composition* advantage. After 79 phases it found — **on its own testbed** — that edge
> composition adds **no meaningful advantage** (structural *or* query-time); the early "compositional" wins
> were depth-robust *behavior* mis-attributed to a mechanism that later ablations trace to **generic
> capacity**; and the one durable positive is **methodological**, not architectural.

---

## 1. The central negative (scoped honestly)

On a **custom edge-sampled FB15k-237 subgraph** (frac=0.10; 11,563 entities, 27,211 train / 18,159 test
triples; **undirected incident-edge degree ≈ 4.7**; relation-type-dominated) under a **FROZEN edge-attention
encoder**, edge-to-edge composition shows **no meaningful task advantage** under either the encode-once (AOT)
paradigm or the query-time (JIT) paradigm.

This is a refutation **on this testbed**, not a universal refutation of edge attention. End-to-end
co-training and rich-edge-content domains were *declined, not tested* (§7).

> **Note on "degree":** the figure "1.48" used in earlier notes is **not a node degree** — it is
> train-edges/test-triples (27,211/18,159 ≈ 1.50). The testbed's degree is **≈ 4.7** (undirected incident-edge,
> = 2·27,211/11,563), as Phase 73 reports. Use ≈ 4.7.

## 2. Negative findings (with numbers)

1. **Structural 2-hop edge composition is rejected at high power** across every density tested (degree
   ≈ 4.7 / 5.1 / 7.3 / 20; Phases 66/71/72/73). Phase 73 (n=5, same density as the paper's Phase 67, fair
   per-target K=128 cap): `hops2−hops1 @3p = −0.0013`, positive in only 2/5 seeds. **Phase 67's headline
   +0.012/+0.017 "confirmation" does not reproduce.**
2. **The sole surviving edge positive is generic capacity, not structure.** `hops1`/`hops2` over `node_only`
   at 5p is ~+0.02 — but `random_struct` over `node_only` @5p = +0.025 ≈ `hops1` +0.026, and
   `hops1−random_struct` = +0.002 (3/5 seeds, n.s.). **Random connectivity reproduces ~95% of it** (Phase 76).
   It survives τ→0, so it's real-but-generic-connectivity, not a soft-averaging artifact.
3. **Content/composability-routed edges do not help** (Phase 75): `content_comp` ties `content_sim` and
   ≈ `content_random`; the +0.03 LP gain is reproduced by **frozen random-key** routing (capacity); content
   routing **inverts** the depth signature (+0.004 @1p → −0.023 @5p).
4. **Query-time edge-INSTANCE composition is SIGNIFICANT-BUT-NEGLIGIBLE** (Phase 78, n=15):
   `pec − shuffle_full @3p = +0.0029` (p=0.034, 95% CI [+0.0003, +0.0054]), **5× inside** the pre-registered
   ±0.015 meaningful band, with a **verified lever** (displacement 0.65 ≫ 0.02). `instance_fraction` 0.63
   means edge-instance identity is *the larger part of a negligible whole*, not support for the thesis.
5. **The "JIT ≈ 2× AOT" gain is the trained query-time READOUT, not edges** (Phase 79, n=15): a node-only
   `AOTReadout` with **no edge operator** recovers ~all of the 2×. `gap_path = pec − aot_readout @3p =
   +0.0050` (p=0.007, 95% CI [+0.0016, +0.0085]), 5× inside ±0.015; `path_fraction = 0.03` (~97% of the gain
   over the untrained floor is the trained head); and even that +0.005 residual **fails the conjunctive guard**
   `pec > static @3p` (p=0.054), so it is not even clean edge-content.
6. **The encode-once soft-traversal eval cannot distinguish structural composition from random connectivity**
   at any hop depth (Phases 75/76) — a property of the *eval protocol*, not of DELTA.

## 3. Reconciling the early "compositional" positives (Phases 42–65)

They were **real effects mis-attributed to an edge-composition mechanism**, measured single-/3-seed on a
small **dense** top-500 degree-biased subgraph (mean degree ≈ 19.7). The high-power negative arc re-examined them:

- **Phase 44** ("advantage grows with depth," +0.004→+0.100 @2p→5p, *landmark*): a **single-seed behavioral**
  observation — DELTA-Matched was the only one of 7 models whose chain MRR didn't degrade with depth — but it
  does **not isolate composition**: the same-subgraph mechanism ablation (Phase 66) found `node_only ≥ hops1 ≥
  hops2` on every multi-hop metric, Phase 53 showed 500-query single-seed multi-hop is too noisy for
  conclusions, and Phases 75–76 trace any surviving edge benefit to capacity. **Honest restatement:
  "depth-robust multi-hop *behavior* (mechanism unattributed / shown to be generic capacity)." The word
  "compositional" is unearned.**
- **Phase 42** (2p→3p non-degradation): the project's most durable property — but it reproduces *within*
  `node_only` and even random conditions (Phases 68/71/72/73), so it's a property of query-difficulty
  structure, not edge composition.
- **Phase 45** (3-seed 3p win over GraphGPS, 0.742±0.009 vs 0.713±0.007): a genuine *subgraph-scoped
  architectural* comparison, not mechanism evidence.
- **Phases 59–61** ("edge attention works"/"surpasses DistMult"): a modest, scale-dependent inductive bias
  (+0.016–0.023 at N=5000); the headline N=2000 +0.076 was **DistMult overfitting** (admitted Phase 61/62)
  and depth adds nothing (Phase 60).
- **Phases 46–52** temperature "landmarks" (K's "3p ceiling" 0.4148, "three operating modes", N's 4p/5p
  records): **revoked** by Phase 53/54 as 500-query single-seed artifacts. Only the LP-MRR temperature gains
  and the edge-up/node-down drift direction survive multi-seed.

## 4. Durable positives (at the altitude they earned)

1. **Methodology — the strongest defensible positive (a *process* contribution, not architectural).** A
   pre-registration + adversarial-red-team + correct-replication-unit pipeline that repeatedly caught its
   **own** false positives: the per-query-bootstrap false PASS in Phase 77 (seed, not query, is the
   replication unit; ~10–15× variance understatement), two false-verdict design flaws in Phase 78 (a
   false-REFUTE point-estimate gate and a `W_ctx`-loophole shuffle), and a 64-d/48-d space confound in Phase
   79. The n=15 verdicts use the corrected machinery (seed-level paired-t, TOST band δ=0.015, displacement
   positive-control, four-quadrant rule, conjunctive guard). Transferable to ML experimental hygiene.
2. **Eval-methodology note (scoped, DELTA-only).** On DELTA's own frozen edge-attention encoder and this
   custom subgraph, a trained query-conditioned readout roughly **doubles** the parameter-free soft-traversal
   MRR of the *same* frozen encoder (**0.107 → 0.196 @3p**; node-only head, zero edge participation).
   Mechanically: "a trained query-conditioned readout beats a parameter-free soft-traversal scorer" — a
   legitimate caveat that the community-standard soft-traversal protocol is a *weak readout* that understates a
   frozen encoder's multi-hop signal. The non-trivial part is the *controlled attribution* (node-only recovers
   it → readout, not the path), not the existence of the 2×. **Not validated on any other architecture or
   standard benchmark.**
3. **Engineering facts, orthogonal to the thesis.** DELTA's one-time encoding is 51.8× slower, training 34×
   slower than GraphGPS, but per-query scoring is 0.84–0.94× (a timing measurement). BrainEncoder's
   differentiable edge construction improves H@10 +4.7% on the dense top-500 subgraph (d=0.01, 3 seeds) while
   matching MRR — a subgraph-only, LP-only *preliminary* that doesn't scale past N≈1000 and was never
   connected to multi-hop reasoning.

## 5. Scope & limitations

- **Testbed:** custom edge-sampled FB15k-237 (frac=0.10), **not** full FB15k-237 and **not** standard
  complex-query benchmarks (BetaE/GQE). Multi-hop queries are homemade chains, not comparable to published
  baselines.
- **Frozen encoder** throughout the negative arc; end-to-end co-training is untested.
- **Sub-SOTA absolute performance:** negative-arc 3p MRR ≈ 0.20 and full-graph 1p MRR ≈ 0.205, well below
  NBFNet ≈ 0.415 / CompGCN ≈ 0.355 / RotatE ≈ 0.338 on full FB15k-237. **DELTA is not a competitive
  link-prediction result.** (The ≈ 0.50 LP MRR in older `best_results` is the *top-500 dense subgraph only* —
  never conflate it with the full/edge-sampled graph.)
- **The "2×"** is shown on exactly one frozen encoder family and one custom subgraph; it is *not* validated on
  any independent architecture or standard benchmark.
- **Synthetic benchmarks** (Phase 11 transitive 100% vs 61.1%; Phase 34 +0.573/+0.290/+0.095) were never
  re-validated in the 66–79 arc, are single-seed, and Phase 66 found they do not generalize to FB15k-237; the
  +0.573 edge-classification gap is partly a probe-architecture artifact (only the +0.095 shared-protocol
  path-composition number is a fair cross-architecture comparison).

## 6. What is NOT claimed (explicit non-claims)

- **Not claimed:** that edge-to-edge attention provides a multi-hop advantage via *composition* (refuted as a
  meaningful driver here, structurally and at query time).
- **Not claimed:** that 2-hop edge adjacency beats 1-hop on real KG multi-hop queries (refuted at high power;
  the paper's Phase-67 +0.012 does not reproduce).
- **Not claimed:** that the early "+0.100 @5p compositional advantage" is a robust compositional result
  (single-seed, dense subgraph; attributed to capacity/eval-noise by the project's own later work).
- **Not claimed:** that the "2×" is a competitive or impressive number. It is ~2× over DELTA's *own* weak
  parameter-free floor (0.107 → 0.196 @3p), landing at ≈ 0.20 absolute MRR — sub-SOTA; the multiplier is
  sensitive to the baseline (vs the untrained floor 0.0011 it would be ~180×).
- **Not claimed:** that "encode-once soft-traversal understates multi-hop ability ~2×" is a general statement
  about evaluation paradigms — shown only on DELTA; needs cross-architecture + standard-benchmark replication.
- **Not claimed:** that the negative is universal — edge attention is not refuted in general; co-training and
  rich-edge domains are *declined doors*, not negative results.
- **Not claimed:** that NBFNet / conditional-message-passing would do better — the "strategic compass" is a
  *hypothesis/recommendation*, not a finding (never tested here).
- **Not claimed:** that BrainEncoder, the temperature "three operating modes", or any small-n transfer result
  are validated.

## 7. Declined doors (explicitly not pursued)

1. **End-to-end co-training** (un-freeze the encoder) — weakly motivated by 11 phases of caught artifacts; no
   positive signal; declined.
2. **Rich-edge-content domains** (temporal / featured graphs) where edge identity carries real per-instance
   information — a new testbed; the one path that would keep edge attention central; speculative; declined.
3. **Standard benchmarks** (BetaE/GQE, full FB15k-237 at competitive MRR) — needed for any general claim; not run.
4. **Cross-architecture replication** of the soft-traversal-understatement note — the highest-value follow-up
   if that note is to become a contribution; declined.

## 8. Publishability (honest)

DELTA is **not** publishable as a positive architecture paper, and the committed NeurIPS draft
(`paper/delta_neurips2026.tex`) **cannot ship as written** — its abstract, Contribution 3, Tables
`tab:phase67`/`tab:cross_density`, and conclusion still assert the *refuted* thesis (Phase-67 +0.012
"validating the core mechanism"; "advantage grows with compositional depth"; +0.100 @5p), and its temperature
"three operating modes" was revoked in Phase 53/54. **A truthful paper is a different paper:** (1) a
rigorously-controlled **negative** result (edge composition adds no meaningful advantage under AOT or JIT;
capacity-not-composition attribution); (2) an **eval-methodology note** (soft-traversal understates a frozen
encoder's multi-hop signal ~2× here; needs cross-arch validation); (3) the **self-refuting-rigor methodology**
as a transferable experimental-hygiene contribution. It leads with the negative + methodology, scopes
everything to the custom sub-SOTA testbed, and presents the early subgraph positives only as exploratory,
heavily hedged.

---

*Generated by a 5-lens adversarial disclosure (each lens hunting overclaims in both directions) + synthesis;
issues found and reconciled: 6 overclaims, 2 inconsistencies, 7 stale statements, 8 missing caveats. The
phase docs, `research_state.json`, README/MkDocs site, and the NeurIPS draft are corrected or flagged
accordingly.*
