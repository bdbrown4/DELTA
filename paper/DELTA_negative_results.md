# When Edge Attention Is Capacity, Not Composition: A Rigorously-Controlled Negative Result on Knowledge-Graph Multi-Hop Reasoning

## Abstract

Edge-to-edge attention — letting a knowledge-graph encoder attend over relations rather than only over entities — is an intuitively appealing inductive bias for *compositional* multi-hop reasoning: a model that can route an edge to its composable neighbors ought to chain relations more faithfully across hops. We set out to demonstrate this. After 79 experimental phases on a custom edge-sampled FB15k-237 subgraph under a frozen edge-attention encoder, we report the opposite: on this testbed, edge-to-edge *composition* adds **no meaningful task advantage**, under either the encode-once (AOT) paradigm or the query-time (JIT) paradigm. The early "compositional" wins we observed (e.g. a +0.100 5p MRR gap that grew with depth) were real depth-robust *behavior* mis-attributed to a composition mechanism that later, higher-powered ablations trace to **generic capacity** — random connectivity reproduces ~95% of the only surviving edge benefit. Every density we tested (undirected degree ≈ 4.7 / 5.1 / 7.3 / 20) rejects a 2-hop-over-1-hop composition advantage; content/composability-routed edges do not help; query-time edge-instance composition is statistically detectable but five-fold inside our pre-registered meaningfulness band; and the headline "JIT ≈ 2× AOT" gain is recovered in full by a node-only trained readout with **zero edge participation**. The two durable positives are not architectural. First, a **self-refuting rigor methodology** — pre-registration, adversarial red-teaming, seed-as-replication-unit inference, equivalence testing (TOST), and a displacement positive-control — that repeatedly caught its *own* false positives, including a ~10–15× variance-understating false-PASS. Second, a narrow, **DELTA-only eval-methodology note**: a trained query-conditioned readout roughly doubles (0.107 → 0.196 @3p) the parameter-free soft-traversal MRR of the *same* frozen encoder, suggesting the standard soft-traversal protocol understates a frozen encoder's multi-hop signal. Absolute performance is sub-SOTA (≈ 0.20 MRR vs NBFNet ≈ 0.42). We scope every claim to this custom testbed and frozen encoder, present the early positives only as hedged exploratory characterization, and position conditional message-passing (NBFNet-family) as the lever — stated explicitly as a hypothesis, not a finding.

---

## 1. Introduction

**The thesis we tried to prove.** A knowledge graph encodes relational structure as labeled edges. Most graph encoders attend over *entities*; a smaller line attends over *edges*, in the hope that an edge-to-edge attention operator can learn to *compose* relations — to recognize that an edge of type `r₁` followed by an edge of type `r₂` supports a query about a composite relation, and to propagate that signal across multiple hops. DELTA was built to test a sharp version of this claim: **edge-to-edge attention gives a knowledge-graph reasoner a genuine multi-hop *composition* advantage, and that advantage grows with reasoning depth.**

Early evidence looked encouraging. On a small dense subgraph, a DELTA-Matched model was the only one of seven architectures whose chain-query MRR did not degrade with hop depth; the gap over a strong GraphGPS baseline appeared to *accelerate* with depth (+0.004 at 2p → +0.100 at 5p). This is exactly the signature a composition mechanism should produce.

**The honest finding.** It did not hold. When we subjected the claim to high-powered, pre-registered, adversarially-controlled tests, the composition story dissolved at every layer:

- **Structurally**, 2-hop edge adjacency does not beat 1-hop at any density tested, and the single surviving edge benefit (a small deep-hop gain) is reproduced almost entirely by *random* connectivity — it is capacity, not structure.
- **By content**, edges routed by learned composability do not beat edges routed by frozen-random keys.
- **At query time**, where the thesis arguably deserved its fairest test, edge-instance composition is statistically detectable but negligibly small, and the much-hyped "JIT ≈ 2× AOT" multiplier turns out to be the trained query-conditioned readout — a node-only model with no edge operator recovers it.

This paper documents that arc honestly. We lead with the negative result and the methodology that produced it, scope **every** claim to our custom, sub-SOTA testbed and frozen encoder, and reconcile the early positives as exploratory, heavily-hedged observations that we now attribute to capacity and evaluation noise. We make no claim that edge attention is refuted in general; co-training and rich-edge-content domains are *declined doors*, not negative results.

---

## 2. Setup

### 2.1 Testbed

All negative-arc experiments run on a **custom edge-sampled FB15k-237 subgraph** (sampling fraction `frac = 0.10`): 11,563 entities, 27,211 train and 18,159 test triples, relation-type-dominated. The **undirected incident-edge degree is ≈ 4.7** (= 2 · 27,211 / 11,563). We flag a measurement correction we ourselves made: a "1.48" figure in early notes is *not* a node degree — it is the train-edges-to-test-triples ratio (27,211/18,159 ≈ 1.50). The testbed degree is ≈ 4.7.

This testbed is deliberately **not** full FB15k-237 and **not** a standard complex-query benchmark (BetaE / GQE). The multi-hop queries are homemade relation chains of length 1–5 ("1p"…"5p"), generated from the graph; they are not comparable to published complex-query baselines. We use this testbed throughout because it lets us run the *number* of seeds and controls that the central question demands, but it is a limitation we return to in §6.

### 2.2 Frozen encoder

Across the entire negative arc (Phases 66–79) the edge-attention encoder is **frozen**: we train readouts, routing, and query-time operators on top of fixed entity/edge representations, but we never co-train the encoder end-to-end. This is a deliberate scoping choice — it isolates the *encoded structure* from the optimization that might exploit it — and it is also the single most important caveat on our negative result (§6, §7). End-to-end co-training is untested.

### 2.3 Queries and evaluation paradigms

We evaluate under two paradigms:

- **Encode-once (AOT):** the encoder produces static node/edge embeddings; multi-hop queries are answered by a parameter-free **soft-traversal** scorer that propagates over those embeddings. This is the community-standard protocol we inherited.
- **Query-time (JIT):** the compositional operator runs *along the query chain at inference*, with a matching training objective (within-reached discriminative multi-label BCE over co-reached decoys, trained on train-only chains, with explicit leakage assertions). This is the fair test the AOT eval could not give to a *composition* claim.

### 2.4 Methodology: the inference machinery that caught our own errors

The most consequential design decisions are statistical, and they were not all correct on the first try. The corrected machinery — used for the n = 15 verdicts in §3.3–3.4 — is:

- **Seed is the unit of replication.** A deep-learning contrast that sign-flips across the handful of independent training seeds must be tested at the *seed* level. A per-query bootstrap that collapses the seed axis and resamples fixed, cross-seed-correlated query ids as i.i.d. understates variance by roughly **10–15×** and manufactures significance. We use seed-level paired-t tests (n = 15).
- **Accepting the null requires equivalence testing (TOST), not a small point estimate.** A bare `|point estimate| < ε` gate is not an equivalence test, and at n ≤ 15 the paired CI half-width (~0.011–0.022) makes small thresholds unreachable — a small estimate would manufacture a false-REFUTE symmetric to the false-PASS above. We pre-register a meaningfulness band δ = 0.015 and test whether the effect lies inside it.
- **Displacement positive-control.** A one-axis lesion is only interpretable if it actually moves the model. We verify a median readout-displacement *lever* (e.g. 0.65 ≫ a 0.02 floor) so that a "no-lever" testbed reads *inconclusive*, not *refuted*.
- **Conjunctive guards and a four-quadrant verdict.** We require a contrast to clear multiple controls jointly and classify each result by (significant?) × (inside-band?), so that a "significant-but-negligible" outcome is named as such rather than spun in either direction.

We emphasize that these are not after-the-fact rationalizations: the pipeline **caught its own false positives** (a Phase-77 false-PASS, two Phase-78 design flaws, a Phase-79 representation-space confound) *before* the verdicts in §3 were finalized.

---

## 3. Negative results

### 3.1 Structural composition ≈ capacity across densities

We ran high-powered hop-depth ablations (node-only vs 1-hop vs 2-hop edge adjacency) across **every density we could construct**: undirected degree ≈ 4.7, 5.1, 7.3, and 20 (Phases 66/71/72/73).

**2-hop does not beat 1-hop.** The definitive test (Phase 73; n = 5 seeds, same density as the earlier "confirming" Phase 67, with a fair per-target 2-hop cap K = 128) gives `hops2 − hops1 @3p = −0.0013`, positive in only 2 of 5 seeds. The earlier Phase-67 headline (+0.012/+0.017 @3p, "confirming 2-hop adjacency is beneficial") was single-source and **does not reproduce**. Lower-power runs that appeared to clear the bar were single-seed artifacts: Phase 71's mean +0.019 @3p came from per-seed values of +0.000 / −0.002 / +0.060 with σ = 0.030; Phase 72 actively *dilutes* (`hops2 − hops1 @3p = −0.009`, worsening with depth).

**The one surviving edge benefit is capacity, not structure.** At the deepest hop (5p), 1-hop and 2-hop edge adjacency beat node-only by ≈ +0.02, and this survives hard traversal (τ → 0), so it is not a soft-averaging artifact. But it is *generic connectivity*, not the learned structure: a `random_struct` control (random adjacency) over node-only at 5p = +0.025 ≈ `hops1` +0.026, and the direct `hops1 − random_struct = +0.002` (3/5 seeds, not significant). **Random connectivity reproduces ~95% of the benefit** (Phase 76). LP MRR is tied across all conditions.

The one robust *within-condition* property — 2p→3p depth non-degradation — reproduces inside node-only and even random conditions (Phases 68/71/72/73), so it is a property of query-difficulty structure, not edge composition.

### 3.2 Content-routing null

If the issue were that *structural* adjacency is the wrong wiring, then routing edges by *learned composability* should help. It does not (Phase 75; pre-registered factorial, 5 seeds × 600 epochs). A bilinear composability-routed condition (`content_comp`) **fails every pre-registered clause**: it ties similarity-routed (`content_comp − content_sim @3p = −0.0063`) and barely separates from random-routed (`content_comp − content_random @3p = +0.0037`, below the 0.010 threshold). Three diagnostics drive the verdict:

1. **`composable_frac` is unpredictive of MRR** — `content_sim` (0.528) ≈ `content_comp` (0.185) ≈ `content_random` (0.150) all land at the same MRR despite a 3.5× spread in measured composability.
2. **The LP gain is capacity.** The +0.03–0.04 LP improvement over structural is reproduced by **frozen-random keys** — global reach, not learned content.
3. **Content routing inverts the depth signature.** `content_comp − hops1` runs +0.004 (1p) → −0.023 (5p): it helps shallow link prediction and *hurts* the deepest hop, the exact inverse of the composition prediction (lost locality where deep hops need it).

The mechanism provably *learns* composability (the routing gate rises during training); learning it simply does not help the task.

### 3.3 Query-time instance composition is negligible (Phase 78, n = 15)

The AOT soft-traversal eval scores over static node embeddings, so it can only reward composition through embeddings — arguably an unfair test for a *query-time* mechanism. So we built the fair test: a query-time edge operator that composes along the query chain, with a matching objective, and a within-relation-type edge-content permutation ladder that holds routing fixed (frontier-invariant) and perturbs only edge-instance identity.

A four-lens adversarial red-team killed our v1 design *before* the ~30-hour run: the "REFUTE" gate was a bare point-estimate (a false-REFUTE artifact), and the v1 shuffle left the real-endpoint channel intact (where the per-instance signal lives), biasing the verdict in both directions. The redesigned primary control, `shuffle_full`, permutes the *whole* edge identity coherently.

**Result (n = 15, seed-level paired-t):** `pec − shuffle_full @3p = +0.0029` (p = 0.034, 95% CI [+0.0003, +0.0054]). The effect is **statistically detectable but lies five-fold inside the pre-registered ±0.015 band**, with a verified lever (median readout displacement 0.65 ≫ 0.02 floor). Verdict: **SIGNIFICANT-BUT-NEGLIGIBLE**. The 3p MRR ladder is `pec 0.2022 > shuffle ≈ shuffle_full 0.1993 > rrprior 0.1966 > pec_lesion 0.1941`. The decomposition `instance_fraction = (pec − shuffle_full)/(pec − rrprior) = 0.63` means edge-instance identity is the *larger slice of an already-negligible whole* — not support for the thesis. Edge attention here uses instance identity, but not meaningfully.

### 3.4 The JIT "2×" is the readout, not the edges (Phase 79, n = 15)

A striking observation in the JIT arc was that query-time scoring roughly *doubles* the encode-once soft-traversal MRR. Read naively, that "JIT ≈ 2× AOT" looks like evidence that query-time edge composition unlocks a paradigm-level advantage. It is not.

We isolated the source with a fair control (red-teamed pre-run to remove a 64-d/48-d representation-space confound): an `aot_readout` that is **node-only** — a trained, query-conditioned pooled-DistMult readout with **no edge operator** — pooling the *same* discrete train-reachable penultimate frontier as the query-time model, on the *same* frozen 48-d encoder.

**Result (n = 15):** the node-only readout recovers essentially all of the 2×. The 3p MRR ladder is `pec 0.2020 ≈ static 0.1991 ≈ aot_readout 0.1963 ≫ aot-soft (untrained floor) 0.0011` — all roughly 2× the published 64-d AOT-soft (≈ 0.107 @3p). The residual `gap_path = pec − aot_readout @3p = +0.0050` (p = 0.007, 95% CI [+0.0016, +0.0085]) is again **five-fold inside ±0.015**; `path_fraction = 0.03` (≈ 97% of the gain over the untrained floor is the trained *head*, ≈ 3% the path); and even that residual **fails the conjunctive guard** `pec > static @3p` (p = 0.054), so it is not even clean edge-content. A 1p plumbing gate confirms the arms are byte-identical where they should be.

**Conclusion of the arc.** A node-only model with no edges recovers the JIT 2×. The gain is the trained query-time readout / objective, *independent of edge composition*. This closes the loop: edge-to-edge composition adds no meaningful value under AOT (structural, §3.1–3.2), under JIT-instance (§3.3), or as the source of the JIT 2× (§3.4). A corollary is methodological: the encode-once soft-traversal eval cannot distinguish structural composition from random connectivity at any hop depth (§3.1), and it understates a frozen encoder's multi-hop signal (§5).

---

## 4. Reconciling the early subgraph positives (exploratory only)

The early "compositional" wins (Phases 42–65) were **real effects mis-attributed to an edge-composition mechanism**, measured single- or 3-seed on a small *dense* top-500 degree-biased subgraph (mean degree ≈ 19.7). We present them only as exploratory characterization; the high-power arc re-examined each, and none survives as composition evidence.

- **Phase 44** (the landmark: "advantage grows with depth," +0.004 → +0.100 @2p→5p): a **single-seed behavioral** observation — DELTA-Matched was the only one of seven models whose chain MRR did not degrade with depth. It does **not isolate composition**. The same-subgraph mechanism ablation (Phase 66) found `node_only ≥ hops1 ≥ hops2` on every multi-hop metric; Phase 53 showed 500-query single-seed multi-hop is too noisy for conclusions; Phases 75–76 trace any surviving edge benefit to capacity. Honest restatement: *depth-robust multi-hop behavior, mechanism unattributed and later shown to be generic capacity.* The word "compositional" is unearned.
- **Phase 42** (2p→3p non-degradation): the most durable property — but it reproduces within `node_only` and even random conditions (Phases 68/71/72/73), so it is a property of query-difficulty structure, not edge composition.
- **Phase 45** (3-seed 3p win over GraphGPS, 0.742 ± 0.009 vs 0.713 ± 0.007): a genuine *subgraph-scoped architectural* comparison, not mechanism evidence.
- **Phases 59–61** ("edge attention works" / "surpasses DistMult"): a modest, scale-dependent inductive bias (+0.016–0.023 at N = 5000); the headline N = 2000 +0.076 was DistMult overfitting (admitted Phase 61/62), and depth adds nothing (Phase 60).
- **Phases 46–52** temperature "landmarks" (a 3p "ceiling break" to 0.4148, "three operating modes," 4p/5p records): **revoked** by Phase 53/54 as 500-query single-seed artifacts (multi-seed K mean 3p = 0.3699 ± 0.0200, *below* the baseline; even seed-42 reruns are non-reproducible under CUDA non-determinism). Only the LP-MRR temperature gains and the edge-up/node-down temperature-drift direction survive multi-seed.

We make no robustness claim for any of these.

---

## 5. An eval-methodology note (scoped DELTA-only)

One positive observation survives the controls, at a carefully-bounded altitude. On DELTA's **own** frozen edge-attention encoder and this custom subgraph, a **trained query-conditioned readout roughly doubles the parameter-free soft-traversal MRR of the *same* frozen encoder: 0.107 → 0.196 @3p** — using a **node-only head with zero edge participation**.

Mechanically the statement is mundane — "a trained query-conditioned readout beats a parameter-free soft-traversal scorer" — but it carries a legitimate caveat for the field: the community-standard soft-traversal protocol is a *weak readout* that can **understate a frozen encoder's multi-hop signal**. The non-trivial part is the *controlled attribution* (§3.4): a node-only model recovers the 2×, so the gain is the readout, not the path. The multiplier is baseline-sensitive (against the untrained floor of 0.0011 it would be ~180×) and we do not claim it is impressive in absolute terms.

This note is **not validated on any other architecture or standard benchmark.** It is a hypothesis about evaluation paradigms scoped to DELTA, and its highest-value follow-up is cross-architecture + standard-benchmark replication (§6, §7).

---

## 6. Limitations

We state these plainly because the central result is a negative, and a negative is only as strong as its scope.

- **Testbed.** Custom edge-sampled FB15k-237 (`frac = 0.10`), **not** full FB15k-237 and **not** standard complex-query benchmarks (BetaE / GQE). The multi-hop chains are homemade and not comparable to published baselines.
- **Frozen encoder.** The entire negative arc freezes the encoder; end-to-end co-training is untested. This is the single most important caveat: it is conceivable (though, after 11 phases of caught artifacts, weakly motivated) that co-training would let an edge operator exploit structure the frozen encoder leaves on the table.
- **Sub-SOTA absolute performance.** Negative-arc 3p MRR ≈ 0.20 and full-graph 1p MRR ≈ 0.205 — well below NBFNet ≈ 0.415, CompGCN ≈ 0.355, RotatE ≈ 0.338 on full FB15k-237. **DELTA is not a competitive link-prediction result.** (The ≈ 0.50 LP MRR in earlier records is the *top-500 dense subgraph only* and must never be conflated with the full/edge-sampled graph.)
- **The "2×"** is shown on exactly one frozen encoder family and one custom subgraph; it is *not* validated on any independent architecture or standard benchmark.
- **Synthetic benchmarks** (an early transitive-closure 100% vs 61.1%; +0.573/+0.290/+0.095 probe gaps) were never re-validated in the 66–79 arc, are single-seed, and Phase 66 found they do not generalize to FB15k-237; the +0.573 edge-classification gap is partly a probe-architecture artifact. Only the +0.095 shared-protocol path-composition number is a fair cross-architecture comparison.
- **Engineering facts** (orthogonal to the thesis): DELTA's one-time encoding is 51.8× slower and training 34× slower than GraphGPS, but per-query scoring is 0.8–0.9× (a timing measurement). A differentiable edge-construction variant improves H@10 +4.7% on the dense top-500 subgraph (3 seeds) while matching MRR — but this is a subgraph-only, LP-only preliminary that does not scale past N ≈ 1000 and was never connected to multi-hop reasoning.
- **Declined doors** (explicitly not pursued, and therefore *not* claimed as negative): end-to-end co-training; rich-edge-content (temporal / featured) domains where edge identity carries real per-instance information; standard BetaE/GQE benchmarks at competitive MRR; cross-architecture replication of the eval-methodology note.

### Explicit non-claims

To prevent over-reading the negative: we do **not** claim that edge-to-edge attention provides a multi-hop advantage via composition (refuted as a meaningful driver *here*, structurally and at query time); that 2-hop adjacency beats 1-hop on real KG multi-hop queries; that the early "+0.100 @5p" is a robust compositional result; that the "2×" is competitive or impressive; that "soft-traversal understates multi-hop ability ~2×" is a general statement about evaluation paradigms (shown only on DELTA); that the negative is *universal* (edge attention is not refuted in general; co-training and rich-edge domains are declined doors); that NBFNet / conditional message-passing would do better (a hypothesis, never tested here); or that the differentiable-construction variant, the temperature "operating modes," or any small-n transfer result are validated.

---

## 7. Related work and positioning

Our negative is best read against the conditional message-passing line — NBFNet and its relatives — which conditions the entire propagation on the query (a learned query-specific labeling of the source, propagated by a relational operator) rather than encoding the graph once and reading out. Two threads of our results point, consistently and independently, at *query-conditioning* as the lever that mattered on our testbed:

1. The only thing that moved the needle in the JIT arc was the **trained query-conditioned readout** — present even at 1p with zero traversal — not edge participation (§3.4).
2. The eval-methodology note is precisely a statement that a *query-conditioned* readout extracts ~2× more multi-hop signal from the same frozen encoder than a parameter-free traversal (§5).

We therefore state, **as a hypothesis and a strategic recommendation rather than a finding**, that a competitive pivot from this work points toward the NBFNet / conditional-message-passing family — query-conditioned propagation — and *away* from edge-to-edge attention as the central mechanism. We did not run NBFNet here; this is a compass, not a result.

The most edge-attention-favorable open direction we did *not* take is a **rich-edge-content domain** (temporal or featured graphs) where edge identity carries genuine per-instance information — the one setting in which the edge-composition premise might hold. It remains speculative and was declined.

---

## 8. Conclusion

We tried to show that edge-to-edge attention buys a knowledge-graph reasoner a compositional multi-hop advantage. On our custom edge-sampled FB15k-237 subgraph, under a frozen encoder, it does not — not structurally (2-hop never beats 1-hop across degree ≈ 4.7/5.1/7.3/20; the lone surviving deep-hop edge gain is ~95% reproduced by random connectivity), not by content (composability routing ties random-key routing), not at query time (instance composition is significant-but-negligible, five-fold inside our meaningfulness band), and not as the source of the "JIT 2×" (a node-only trained readout recovers it). Edge attention's value on this testbed is **generic capacity, not composition**.

What endures is not the architecture but the *process*: a pre-registration + adversarial-red-team + correct-replication-unit + equivalence-test + displacement-control pipeline that repeatedly caught its **own** false positives — a transferable contribution to ML experimental hygiene. Alongside it sits a single carefully-scoped empirical note — that the standard soft-traversal protocol understates this frozen encoder's multi-hop signal ~2× (0.107 → 0.196 @3p), pending cross-architecture validation — and a hypothesis, not a finding, that query-conditioning (NBFNet-family) rather than edge attention is the lever worth pursuing next. The result is sub-SOTA (≈ 0.20 MRR vs NBFNet ≈ 0.42) and bounded to this testbed; we present the early "compositional" wins only as exploratory, heavily-hedged behavior we later traced to capacity and evaluation noise. A clean, rigorously-controlled negative — disclosed with its own scope — is the honest paper this project earned.