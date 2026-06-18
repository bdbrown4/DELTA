# Phase 71 — Clean Uncapped Hop-Depth Ablation in the Sparse Regime (1p–5p)

## Result

```
Phase: 71 — Sparse-Regime Uncapped Hop Ablation (FB15k-237 random N=2500, DELTA-Matched, 3 seeds × 1000 epochs)
Hypothesis: On a SPARSE subgraph (mean degree ~7), edge-to-edge attention provides a
  multi-hop advantage absent on the dense Phase 66 subgraph — specifically hops=2 beats
  hops=1 on 3p by > 0.010 (a local, uncapped reproduction of Phase 67's +0.017).
Expected: hops=2 − hops=1 3p gap > 0.010, robust across 3 seeds; gap non-decreasing with depth.
Seeds: [42, 123, 456]
Result: REJECTED (not robust — apparent gap is a single-seed artifact)

Metrics (multi-hop MRR — FB15k-237 random subgraph, 2206 entities, mean degree 7.3,
         8069 train edges, 3 seeds × 1000 epochs, hops=2 FULLY UNCAPPED at 3.5M pairs):
Condition    adj_pairs    LP MRR        1p           2p           3p           4p           5p
node_only        0        0.238±0.011  0.291±0.014  0.256±0.002  0.297±0.011  0.278±0.004  0.315±0.011
hops=1       529,480      0.232±0.003  0.287±0.007  0.251±0.005  0.288±0.007  0.286±0.010  0.340±0.012
hops=2     3,497,782      0.234±0.010  0.283±0.023  0.261±0.010  0.307±0.030  0.309±0.024  0.346±0.024

hops=2 − hops=1 gap:  2p=+0.0099  3p=+0.0194  4p=+0.0229  5p=+0.0053
BUT per-seed 3p gap:  seed42=+0.000  seed123=−0.002  seed456=+0.060  (driven entirely by ONE seed)
hops=2 3p across seeds: 0.2789 / 0.2934 / 0.3489 (σ=0.030 — gap sits well inside 1σ)

Reference (Phase 66, DENSE top-500 subgraph, mean degree ~20, hops=2 RANDOM-capped to 5.3%):
  hops=2 − hops=1: 2p=+0.0047, 3p=+0.0024 (REJECTED, within 1σ)
Reference (Phase 67, full sparse graph, mean degree 4.1, H200): hops=2 − hops=1 3p=+0.017 (CONFIRMED)
Reference (Phase 68, random subgraph, mean degree 7.6): hops=2 − hops=1 3p=+0.010 (borderline)

Key insight: At mean degree 7.3, uncapped hops=2 does NOT robustly beat hops=1 — the +0.019
  3p mean gap is driven entirely by seed 456 (+0.060); seeds 42/123 show ~0. This reproduces
  Phase 53's single-seed-outlier warning AND the density story: degree 7.3 is too dense to
  trigger Phase 67's effect (which needs ~4.1). Edge attention is competitive but provides no
  robust multi-hop advantage at this density.
Next question: Does a genuinely sparse (degree ~4, k-core/BFS) local subgraph with 5+ seeds
  reproduce Phase 67's +0.017, now that hops=2 can run fully uncapped?
Status: LOGGED as REJECTED — motivates Phase 72 (sparser testbed + more seeds). Infrastructure
  (uncapped 2-hop construction, sparse sampler) validated and reusable.
```

## Details

### Hypothesis

Phase 66 REJECTED the 2-hop claim on the dense top-500 subgraph, but with two confounds the
doc itself flagged: (1) hops=2 (28.2M pairs) was **randomly subsampled to 1.5M (5.3%)** for VRAM,
so it tested a random fraction of the 2-hop neighborhood vs the full 1-hop neighborhood; (2) the
dense subgraph (mean degree ~20) saturates 1-hop adjacency, leaving no room for 2-hop. Phase 67
CONFIRMED a +0.017 3p benefit on the full **sparse** graph (degree 4.1) but is H200-only with no
committed data in the repo.

This phase tests whether the 2-hop advantage reproduces **locally, cheaply, and fully uncapped**
in the sparse regime, extended to 4p/5p.

### Experimental Design

Three conditions (identical to Phase 66), held constant: model (DELTA-Matched, 271K params with
projections), training (1000 epochs, lr=0.003, batch=4096, label_smoothing=0.1, **no early stop**),
eval (filtered LP + 1p–5p chain queries, 10k per type for 3p/4p/5p).

- **node_only** — edge attention disabled (empty adjacency).
- **hops=1** — edges sharing an endpoint (current default), 529,480 pairs.
- **hops=2** — edges two steps away, **3,497,782 pairs, FULLY UNCAPPED** (no random subsample).
  Enabled by the `delta/graph.py` sparse-matrix-power fix (the 2-hop construction no longer
  densifies a `[E,E]` matrix, which was the OOM driver behind Phase 66's cap).

Subgraph: **random** N=2500 entities (`sample_mode='random'`) → 2206 entities, mean degree 7.3 —
the sparse regime. The default top-degree sampler cannot reach this (top-N is degree ~20–40 at any N).

### Configuration

- Dataset: FB15k-237 random subgraph (2206 entities, 8069 train / val / test)
- Model: DELTA-Matched (d_node=48, d_edge=24, 2 layers, 4 heads)
- Optimizer: Adam, lr=0.003; Epochs: 1000, eval_every=25, **patience=0 (no early stopping)**
- Batch size: 4096; Label smoothing: 0.1; Seeds: [42, 123, 456]
- Multi-hop queries: 1p=541, 2p=2392, 3p/4p/5p=10000 (leakage audit: PASSED)
- Hardware: RTX 3080 Ti (12.9GB). hops=2 uncapped peaked ~11.1GB VRAM — fits.
- Runtime: node_only ~58s/seed, hops=1 ~305s/seed, hops=2 ~980s/seed. Total ~2110s × ~2 (run+rerun).

### Two runs — under-training confound found and removed

A first run (500 epochs, patience=10) produced an apparent hops=2 +0.015 3p win. It was an
**artifact**: (1) the edge stream has an intrinsic ~350-epoch "attention-dilution" warmup, so
500 epochs left it under-trained while node_only converged; (2) `patience=10` fired *during* the
flat warmup and killed seed 42's edge runs (LP 0.02). An LR probe (0.01) confirmed the warmup is
not LR-fixable (it was *slower*). The corrected run (1000 epochs, no early stop) converged all
conditions to LP ~0.23–0.25 (vs ~0.11 under-trained) and dissolved the fake gap.

### Key Observations

1. **hops=2 vs hops=1 is a single-seed effect.** Per-seed 3p gap: seed42 +0.000, seed123 −0.002,
   seed456 +0.060. The +0.019 mean is entirely seed 456. hops=2 3p σ=0.030 >> the gap. Not robust.
2. **Edge attention is competitive, not harmful.** node_only 3p (0.297) sits between hops=1 (0.288)
   and hops=2 (0.307); LP is statistically tied (0.232–0.238). Unlike the under-trained first run
   where node_only spuriously "won".
3. **Within-condition 2p→3p depth-monotonicity is robust** for all conditions (+0.042 node_only,
   +0.037 hops=1, +0.046 hops=2) — reproduces Phase 68's density-independent depth effect.
4. **Density story holds.** No robust 2-hop benefit at degree 7.3, consistent with Phase 68's
   borderline +0.010 at 7.6 and Phase 66's null at ~20. Phase 67's +0.017 needs degree ~4.1.
5. **Uncapping works.** Full 28× larger 2-hop adjacency (3.5M vs 1-hop 529K) trained on a 12GB card
   — the construction fix removes the constraint that forced Phase 66's random subsample.

### Classification: REJECTED

- Mean 3p/4p gaps clear the 0.010 threshold, but are single-seed-driven and within 1σ.
- Two of three seeds show zero hops=2 advantage — fails the "3+ seeds robust" bar (Phase 53/54).
- The honest read: no robust 2-hop multi-hop advantage at mean degree 7.3.

### Impact

- **Methodology:** the cheap "uncap + converge + 3 seeds + per-seed reporting" pipeline killed
  both a under-training artifact (run-1) and a single-seed-outlier mirage (run-2 mean). This is the
  calibration discipline the project should hold its claims to.
- **Infrastructure (reusable):** `delta/graph.py` sparse 2-hop (no dense `[E,E]`); `load_lp_data`
  `sample_mode='random'` + opt-in cache; regression + harness smoke tests; brain edge-count bug fix.
- **Paper:** Phase 67's central +0.017 claim remains the only robust 2-hop evidence and still lacks
  committed local reproduction — Phase 72 should close that at degree ~4 with 5+ seeds.

### Next Steps (Phase 72)

1. Build a genuinely sparse (mean degree ~4) local testbed — k-core or BFS-grown subgraph, or a
   degree-targeted sampler — to match Phase 67's regime while staying on the 12GB card (now feasible
   uncapped).
2. Run the same node_only / hops=1 / hops=2 ablation with **5+ seeds** to beat the multi-hop noise
   floor (Phase 53/54: 3 seeds at this scale is insufficient).
3. If +0.010 3p holds across 5 seeds at degree ~4 → first robust LOCAL reproduction of Phase 67.
   If not → the 2-hop benefit may be specific to the full-graph scale, and the paper's framing
   should be calibrated accordingly.
```
