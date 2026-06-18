# Phase 72 — Sparse Hop Ablation at Low Degree, 5 Seeds (1p–5p)

## Result

```
Phase: 72 — Low-Degree Sparse Hop Ablation (FB15k-237 random N=1500, DELTA-Matched, 5 seeds × 1000 epochs)
Hypothesis: At lower degree than Phase 71 (degree ~5 vs 7.3), uncapped hops=2 beats hops=1
  on 3p by > 0.010, robust across 5 seeds (Phase 67 local reproduction at the right density).
Expected: hops=2 − hops=1 3p gap > 0.010, positive in a majority of seeds.
Seeds: [42, 123, 456, 7, 99]
Result: REJECTED (hops=2 ≤ hops=1) — but 1-hop edge attention shows a sparse-regime benefit over node-only.

Metrics (multi-hop MRR — FB15k-237 random subgraph, 1213 entities, mean degree 5.14,
         3120 train edges, 5 seeds × 1000 epochs, hops=2 FULLY UNCAPPED at 567,774 pairs):
Condition    adj_pairs    LP MRR        3p           4p           5p
node_only        0        0.194±0.013  0.248±0.035  0.241±0.034  0.251±0.036
hops=1       133,379      0.201±0.010  0.274±0.032  0.271±0.023  0.285±0.025
hops=2       567,774      0.203±0.012  0.265±0.045  0.261±0.035  0.271±0.034

hops=2 − hops=1 (the 2-hop test):
  3p mean=−0.0091  per-seed=[−0.023, −0.035, +0.019, −0.017, +0.011]  (>0 in 2/5)
  4p mean=−0.0103  per-seed=[−0.026, −0.024, +0.029, −0.026, −0.005]  (>0 in 1/5)
  5p mean=−0.0140  per-seed=[−0.029, −0.034, +0.028, −0.024, −0.010]  (>0 in 1/5)

hops=1 − node_only (does edge attention help at all?):
  3p mean=+0.0257  per-seed=[+0.050, −0.056, +0.037, −0.009, +0.107]  (>0 in 3/5)
  4p mean=+0.0306  per-seed=[+0.016, −0.048, +0.045, +0.021, +0.119]  (>0 in 4/5)
  5p mean=+0.0332  per-seed=[+0.034, −0.052, +0.044, +0.016, +0.124]  (>0 in 4/5)

Reference: Phase 71 (degree 7.3): hops=2−hops=1 3p=+0.019 (single-seed artifact); hops1≈node_only.
           Phase 66 (degree 20): hops=2−hops=1 3p=+0.002 (REJECTED). Phase 67 (degree 4.1, H200): +0.017.

Key insight: The 2-hop > 1-hop claim FAILS again at degree 5.1 — hops=2 is slightly WORSE than
  hops=1 (3p −0.009, negative in ~4/5 seeds). The 2nd hop adds dilution, not signal. Across
  degree 5/7/20 the 2-hop benefit does not reproduce locally; Phase 67's +0.017 is the only
  positive and is single-source/unreproduced. BUT 1-hop edge attention now trends above node_only
  (+0.026/+0.031/+0.033 on 3p/4p/5p, positive in 3–4/5 seeds) — an effect absent at degree 7.3.
  DELTA's value looks like "edge-to-edge attention helps when node neighborhoods are starved
  (sparse regime), and 1 hop suffices", not "2-hop composition".
Next question: Does the 1-hop>node_only effect hold with real statistical power (edge-sampled
  full-entity graph, ~18K test triples, degree ~4.7) — and is 2-hop still ≤ 1-hop there?
Status: LOGGED as REJECTED (2-hop) / PARTIAL-POSITIVE (1-hop sparse benefit). Motivates Phase 73
  (high-power edge-sampled confirmation with a fair per-target-edge 2-hop cap).
```

## Details

### Hypothesis

Phase 71 (degree 7.3) found the hops=2 advantage was a single-seed artifact and noted degree 7.3
may be too dense (Phase 67 needs ~4.1). This phase goes sparser (random N=1500 → mean degree 5.1)
with 5 seeds (Phase 53/54 bar) to test whether the 2-hop benefit becomes robust at low degree.

### Experimental Design

Reuses `experiments/phase71_sparse_hop_ablation.py` with `--max_entities 1500 --sample_mode random
--seeds 42,123,456,7,99 --epochs 1000 --patience 0`. Conditions node_only / hops=1 / hops=2;
hops=2 (567,774 pairs) **fully uncapped**. delta_matched, lr=0.003, batch=4096, no early stop.
Multi-hop queries (leak-free, Phase 44 generator): 1p=211, 2p=583, 3p=1924, 4p=4752, 5p=10000.

### Key Observations

1. **hops=2 ≤ hops=1.** 3p gap −0.009 (negative in ~4/5 seeds), worsening with depth (5p −0.014).
   Only seed 456 favors hops=2. The second hop dilutes rather than composes — consistent with the
   "fixed-pattern (structural) neighbors add noise" intuition (cf. SubQ's content-vs-fixed argument).
2. **hops=1 > node_only (suggestive).** +0.026/+0.031/+0.033 on 3p/4p/5p, positive in 3–4/5 seeds.
   Edge attention helps when neighborhoods are starved — ABSENT at Phase 71's degree 7.3 (tied there).
   High variance (seed 123 reversed −0.05; seed 99 +0.12), so suggestive not yet conclusive.
3. **2p→3p monotonicity robust again** (+0.070 node_only, +0.082 hops=1, +0.076 hops=2) — Phase 68
   reproduced for the third time; the depth-monotonic effect is the most reliable DELTA property.
4. **Small-sample caveat.** N=1500 yields only 211 1p / 583 2p test queries — a contributor to the
   high variance. Phase 73's edge-sampled config (≈18K test) addresses this.

### Classification: REJECTED (2-hop) / PARTIAL-POSITIVE (1-hop sparse benefit)

- hops=2 fails the >0.010 threshold (negative mean, negative in majority of seeds). 2-hop REJECTED
  at degree 5.1, completing the local picture: no 2-hop benefit at degree 5/7/20.
- hops=1 > node_only clears +0.010 on mean but is noisy (3–4/5 seeds) — promising, needs power.

### Impact

- **Paper:** the central "2-hop relational composition" claim does not reproduce locally at any
  reachable density; it rests entirely on the single-source, uncommitted Phase 67. Calibrate the
  framing toward what is robust: edge-to-edge attention helping in sparse regimes (1-hop) + 2p→3p
  monotonicity. The 2-hop framing should be softened or moved to "scale-dependent, full-graph only".
- **Mechanism:** 1 hop suffices; the 2nd hop dilutes. This is the graph analogue of SubQ's
  fixed-pattern-vs-content critique — and motivates content-dependent (not deeper-structural) edges.

### Next Steps (Phase 73)

1. High-power confirmation: edge-sampled subgraph (frac≈0.10 → ~11.5K entities, ~18K test triples,
   degree ~4.7) with a **fair per-target-edge 2-hop cap** (each edge keeps ≤K 2-hop neighbors —
   the structural cap that fixes Phase 66's *random* 5.3% cap) so 17.5M pairs fit on 12GB. 5 seeds.
2. Settle both questions with real power: (a) does 1-hop edge attention beat node_only? (b) does
   2-hop still ≤ 1-hop at scale? Then recalibrate the paper accordingly.
```
