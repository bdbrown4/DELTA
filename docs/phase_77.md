# Phase 77 — JIT pivot: query-time edge composition (PEC-pf). PRE-REGISTRATION + Step-0 gates

## RESULTS (5 seeds [42,123,456,7,99], edge-sampled FB15k-237 frac=0.10, 10k queries/depth) — **FAIL** (PASS was a statistical artifact, caught by adversarial verification)

**CORRECTED VERDICT: FAIL.** The initially-reported PASS did NOT survive correct statistics. A 3-adversary
red-team (jit-verify-positive workflow) found the pre-registered primary endpoint counted the WRONG unit
of replication: phase77_analyze.py collapsed the 5-seed axis (`.mean(1)`) then bootstrapped by resampling
~8.5k FIXED, cross-seed-correlated QUERY ids — treating them as iid replicates and understating variance
~10-15x. The true unit is the SEED (one independent training). The pre-registration itself recorded that
seed-level was "underpowered" and switched to per-query — i.e. the analysis that manufactures stars was
chosen over the honest one (a methodological error, NOT a code bug; the mechanism is cleanly isolated and
all 10 gating tests pass). Independently reproduced.

SEED-LEVEL paired test (n=5, the correct unit; * = p<0.05 AND seed-cluster CI excludes 0):
```
pec - control   3p              4p               5p          per-seed @5p (fragility)
capacity      -0.001 p.69    +0.014 p.17     +0.029 p.12    [+.017 +.002 +.088 +.018 +.022]
rrprior       +0.005 p.004*  +0.005 p.44     +0.017 p.041*  [+.016 +.010 +.034 +.024 +.000]
static        +0.003 p.39    +0.010 p.18     +0.013 p.15    [+.017 -.004 +.033 +.025 -.005]
MASKONLY      +0.045 p.000*  +0.021 p.000*   +0.039 p.005*  (the weak reachability floor)
```
CORRECTED conjunctive gate (pec > capacity AND rrprior AND static AND MASKONLY at 4p AND 5p): FAILS — 5 of
the 6 within-JIT {capacity,static,rrprior}x{4p,5p} clauses are non-significant; only pec-rrprior 5p
(p=0.041, not Holm-robust) and pec-MASKONLY survive. Result is also load-bearing on ONE seed (456
contributes 40-97% of contested 4p/5p margins; drop-456 flips even the inflated per-query PASS).

### What honestly survives (thin)
- pec robustly beats only the MASK-ONLY reachability floor (a weak bar the non-compositional `static` arm
  also clears).
- A SUGGESTIVE but seed-fragile, extrapolation-only, AXIS-CONFOUNDED instance-over-type signal
  (pec-rrprior 3p/5p; rrprior differs from pec on TWO axes — attention on/off AND message type), NOT shown
  to use edge-instance content (no shuffled-edge control run), appearing at UNTRAINED depths (trained K<=3;
  4p/5p are extrapolation) — the wrong signature for learned routing.
- "JIT ~2x AOT" is real but is the trained all-N BCE READOUT/objective, present at 1p where zero traversal
  occurs (forward_query short-circuit) — NOT query-time edge participation. Not a composition result.

### Conclusion
Query-time edge composition is NOT robustly demonstrated by this experiment. The thesis pin
([[query-time-edge-composition]]) remains OPEN but UNPROVEN. This is another rigorously-caught false
positive (cf. early-stop, small-sample, capacity confounds in Phases 66-76) — the diligence pipeline
working as intended. Optional cheap next steps (only if chasing the thin residual): shuffled-edge control
(does pec use edge instance content?), one-axis instance-vs-type message control (de-confound pec-rrprior),
train-at-depth (in-distribution vs extrapolation), and more seeds for a powered seed-level test.

---

### (Superseded) original per-query analysis — kept for the record; this is the INVALID inference
The numbers below used the per-query bootstrap later shown to be invalid (wrong replication unit). They are
NOT the verdict; see the corrected seed-level analysis above.

Pre-registered PASS (per-query bootstrap — INVALID): PEC-pf beats ALL four controls (capacity, rrprior,
static, MASK-ONLY) at BOTH 4p and 5p on the penult-reachable stratum, per-query bootstrap CI excluding 0.

MRR (mean over 5 seeds, full all-N):
```
arm          1p      2p      3p      4p      5p
pec        0.2201  0.1614  0.2006  0.2508  0.3128
capacity   0.2226  0.1617  0.2019  0.2386  0.2859
rrprior    0.2195  0.1579  0.1948  0.2458  0.2958
static     0.2159  0.1573  0.1962  0.2414  0.2989
AOT-best  ~0.106  ~0.093  ~0.107  ~0.135  ~0.165   (existing p76 checkpoints, soft-traversal)
```

Per-query bootstrap (pec - control, penult stratum, * = 95% CI excludes 0):
```
              2p        3p        4p        5p
capacity   +0.0004   -0.0011   +0.0137*  +0.0294*    (vs RANDOM operator)
rrprior    +0.0022*  +0.0051*  +0.0048*  +0.0169*    (vs [R,R] relation-type prior)
static     +0.0024*  +0.0034*  +0.0097*  +0.0133*    (vs NO path composition)
MASKONLY   +0.0513*  +0.0446*  +0.0211*  +0.0391*    (vs reachability floor)
```

### Headline findings
1. **Query-time ≈ 2× ahead-of-time.** PEC-pf roughly doubles AOT soft-traversal MRR at every depth
   (5p 0.31 vs ~0.16). Invoking edges at query time with a matching objective recovers the signal the
   AOT encode-once paradigm discards — the converse of the Phase-76 diagnosis, and the payoff of the pivot.
2. **Composition beats no-composition / random / type-prior / mask at 4p AND 5p (pre-registered PASS).**
   The seed-42 4p/rrprior tie (−0.0008) was seed noise: pooled over 5 seeds it is +0.0048*.
3. **The signal concentrates at DEEP hops (4p/5p), which are EXTRAPOLATION depths** (trained on K<=3,
   i.e. <=2 penult traversal hops; 4p/5p penult = 3/4 hops). The learned operator's advantage over a
   RANDOM operator compounds with depth (capacity contrast: 3p ~0 -> 5p +0.029) and generalizes beyond
   its training depth. pec beats static and rrprior at ALL depths; it beats capacity specifically at 4p/5p.

### Honest caveats (carry into verification)
- Effect sizes are modest in absolute MRR (~0.01-0.03 at 4p/5p) though statistically robust (n~8.5k/depth,
  5 seeds, tight CIs).
- Depth trend is "deep >> shallow" but NOT strictly monotone (2p/3p margins ~0 for capacity); could be
  "shallow hops saturate / are too easy" rather than "composition specifically helps deep." Both readings
  are interesting; do not over-claim strict monotonic interaction.
- SCOPE: the encoder is FROZEN (hops1 per seed) and shared by all JIT arms — this controls the embedding
  confound by construction, but the result is "given fixed edge-attention embeddings, query-time
  composition of them helps." End-to-end co-training is untested (a follow-up if this survives verification).
- The JIT≈2×AOT gap conflates query-time edge participation WITH a trained multi-hop readout; the clean
  composition claim rests on the WITHIN-JIT controls (pec vs capacity/static/rrprior), which share training.
- This is a POSITIVE in a project whose positives have repeatedly turned out to be artifacts (early-stop,
  small-sample, capacity). It must survive the same adversarial scrutiny before being treated as final.

Artifacts: phase77_output.json, phase77_rr_s{seed}.npz, experiments/phase77_jit_path_score.py,
experiments/phase77_analyze.py, delta/path_compose.py, tests/test_path_compose.py (10 gating tests).

## Status

PRE-REGISTERED + Step-0 abort gates PASSED (no training). Building the PEC-pf model next.
This is the query-time ("JIT") fair test the thesis never got: across Phases 66–76 DELTA's
edge-to-edge structural composition showed no robust advantage over random connectivity, but the
diagnosed root cause is methodological — the eval reasons by soft-traversal over a FROZEN [N,d]
node table, so edges never participate at query time and the eval *structurally cannot reward
composition*. Phase 77 makes edges participate along the query's relation chain at inference, with
a matching training objective.

## The design: PEC-pf (Penultimate-Frontier path-composition Edge-readout)

Chosen via a 4-design tournament + judge panel + 3-adversary red-team. The judge-winning design
(PEC, sum-over-paths DP whose per-hop step IS DELTA's EdgeAttention operator, decoder_rel_emb
deleted) had a KILLS-DESIGN flaw all three adversaries found and the synthesizer verified on the
real testbed: every eval query's final hop is a TEST edge and the DP traverses TRAIN-only edges, so
"traverse the whole chain then score reached entities" makes the gold answer rankable for only
0/3.3/9.9/17.7/21.3% of 1p–5p queries — and that reachability RISES with depth, so "margin grows
with depth" would be mimicked by pure selection bias. **The fix (PEC-pf):** traverse hops 1..K-1
over TRAIN edges to build a penultimate per-entity path-state z_{K-1}, then apply the final relation
r_K as an **all-N readout** (every entity gets a finite score; gold rankable 100% at every depth,
verified). The learned edge-to-edge compose operator runs in the K-1 traversal hops AND the readout.

### Mechanism (delta/path_compose.py — PathComposerPF)
- Shared one-time encode (amortized, NBFNet-style): one `_encode_with_adj` per (checkpoint,
  source-graph) → `nf` [N,48] and recovered edge features `ef0` [E,24].
- **Instance-signal injection** (anti-[R,R]-prior): per-edge DP state
  `ej = LayerNorm(W_ein([ef0 ; nf_src ; nf_tgt]))` so the edge state varies per instance even though
  ef0 is relation-type-dominated.
- `compose_scores` STATICMETHOD refactored out of EdgeAttention (attention.py:190-216) with **no
  `graph._route_prob` read** and explicit `W_ctx` arg order (closes the layer-0-only route_prob-drop
  trap). DP calls it directly per hop (never recurses through DualParallelAttention) and builds the
  per-hop typed edge set fresh (never `build_edge_adjacency` → stale-cache trap structurally avoided).
- Per hop i=1..K-1 (relation r_i): gather typed edges from the current frontier, compose-score,
  scatter-softmax over targets, sum-over-paths (Bellman-Ford additive semiring) → z'. Frontier cap
  MAX_FRONTIER=2048 (p95 frontier=91 → cap ~never binds).
- Final-relation all-N readout over the penult frontier → finite score for every entity (+ learned
  global fallback bias so unreached entities still rank). K=1 (1p/LP) = zero-traversal readout.
- decoder_rel_emb is GONE; a fresh `rel_emb(r_K)` [R,24] feeds ONLY the readout compose step.

### Matching training objective (TRAIN-only, leak-safe)
- Within-reached discriminative multi-label BCE on the penult-frontier all-N readout. Positives =
  train-only answer set ∩ reached; decoys = reached \ answers; trivially-unreached excluded from the
  loss so every gradient teaches the operator to rank gold above CO-REACHED entities (the only thing
  composition can help with). Answer-set down-weighting (pos weight 1/|A|), label smoothing 0.1.
- TRAIN chains via `generate_train_chains` (final hop from TRAIN, not TEST) and a distinctly-named
  `build_train_only_adjacency` (NOT build_full_adjacency, which concatenates train+val+test);
  assertion that no target leaks from val/test.
- Hop mix K∈{1,2,3} weights {0.1,0.4,0.6}; eval 1p–5p (4p/5p untrained extrapolation, reported as
  such; operator hop-shared). Optimizer Adam lr 3e-3, batch 256 chains, ≤400 ep, early-stop on a
  TRAIN-chain val split, grad-clip 1.0, phase66 best_state checkpointing. Headline must hold with
  `lambda_aux=0` (no encoder-warmup aux LP loss) so "better embeddings" can't explain a win.

## Conditions (all on the same queries / valid_cache / reachability mask / denominator)
- AOT-node_only / AOT-hops1 / AOT-random_struct — existing 15 p76_* checkpoints, fast_mh_eval,
  tau-sweep (the encode-once baseline arm; AOT-hops1 is the strongest to beat).
- **JIT-PEC-pf** — the learned edge-to-edge operator at query time.
- JIT-capacity — PEC-pf with ALL of {W_q,W_k,W_v,W_out} frozen at random init (ContentRouter(freeze)
  analog): "just DP machinery + params" floor.
- JIT-RRprior — instance compatibility replaced by a pure [R,R] table: the relation-type prior that
  reproduced the gains in Phases 74–76.
- JIT-static-readout — compose operator ablated to identity, only W_ctx endpoint reads kept:
  path-conditioned static-embedding lookup (the 66–76 ceiling).
- MASK-ONLY floor — uniform score over the penult-reachable set: what pure reachability buys.

## Pre-registered PASS (query-time instance-level composition is REAL)
Primary endpoint = **per-query bootstrap blocked on query id** (not seed-level: 5p seed-paired
SD=0.0234 needs |mean|>0.029 at n=5 — verified underpowered), on the FIXED penult-reachable stratum,
Holm-corrected across the **4p AND 5p joint** test:
1. JIT-PEC-pf beats JIT-capacity AND JIT-RRprior AND JIT-static-readout AND AOT-best, CI excluding 0
   at BOTH 4p and 5p.
2. Holds with lambda_aux=0 (path objective, not embeddings, drives it).
3. Beats the MASK-ONLY floor by a CI-excluding-0 margin (the operator, not the mask, ranks).
4. Significant depth×condition interaction (margin grows with depth) — replaces fragile seed-level
   monotonicity.
5. (Gate, pre-sweep) ej instance-vs-type variance non-negligible — else PEC-pf ≡ RRprior by
   construction.

## Pre-registered FAIL (thesis refuted, cleanly)
- PEC-pf ≤ RRprior at 4p/5p → the signal is a relation-TYPE prior even under query-time invocation.
- PEC-pf ≤ capacity → gain is DP machinery / params, not composition.
- PEC-pf ≤ static-readout → gain is path-conditioned static-embedding lookup (the 66–76 ceiling).
- PEC-pf ~ MASK-ONLY floor → reachability does the work.
- Win on full all-N but NOT on the fixed stratum → selection bias from depth-rising reachability.
- Win requires lambda_aux>0 → embedding-quality effect.
- ej type-dominated even after endpoint injection → testbed cannot express instance composition
  (clean testbed-capacity finding; do not spend the full sweep).

Honest priors: the deepest TRAINED composition is 2 hops (K-1 for 3p), so the claim is strongest at
**3p**; 4p/5p are extrapolation. A clean negative (PEC-pf ≤ RRprior) is the more likely outcome given
Phase 74's [R,R]-prior finding and this sparse (degree ~4.7) testbed — and that is an accepted,
decisive result, not a disappointment.

## Step-0 gate results (experiments/phase77_step0_gates.py → phase77_step0_gates.json) — ALL PASS
Edge-sampled FB15k-237 frac=0.10: N=11,563 ents / 237 rels / 27,211 train edges / 18,159 test.
- **GATE A (penult-frontier non-empty):** 100.000% at every depth 1p–5p → the readout always has a
  frontier to compose. Cross-check: the OLD full-chain gold-reachability is 0/3.28/9.91/17.69/21.28%
  — reproducing the design's verified numbers *exactly* and confirming the killed-PEC flaw was real.
- **GATE B (usable TRAIN-only chains):** ≥6,000 distinct usable chains for K=2,3 (and 4) ≫ 2,000
  target → the matching objective has ample training signal.
- **GATE C (ej instance-vs-type variance):** type-only within-relation variance = 0.0000 (zero by
  construction, as predicted); after endpoint injection within-relation variance fraction = **0.3233**
  (threshold 0.15) → instance signal is substantial; PEC-pf is identifiable apart from the [R,R]
  prior.

VERDICT: PROCEED to build PEC-pf.

## Build plan (gated)
0. [DONE] Step-0 free gates.  1. Refactor compose_scores staticmethod + unit test.  2.
build_train_only_adjacency + generate_train_chains + ef_out from encode.  3. delta/path_compose.py
PathComposerPF.  4. tests/test_path_compose.py (7 gating tests incl. per-hop anchor-sensitivity,
RRprior entity-permutation-invariance, train/full hr2t no-leak, four-arms-identical-frontier).  5.
phase77_jit_path_score.py driver (incremental JSON, resume).  6. Benchmark one real epoch + eval
(ms/step, peak VRAM) BEFORE the sweep.  7. Smoke → seed-42 decision gate → full sweep → per-query
bootstrap + interaction + lambda_aux=0 + MASK-ONLY robustness.
