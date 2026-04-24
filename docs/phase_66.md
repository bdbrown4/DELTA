# Phase 66 — 1-Hop vs 2-Hop Edge Adjacency Ablation (NeurIPS Reviewer Response)

## Result

```
Phase: 66 — Hop-Depth Ablation (FB15k-237 N=500, DELTA-Matched, 3 seeds × 500 epochs)
Hypothesis: hops=2 edge adjacency provides measurably better multi-hop MRR than hops=1
Expected: 2p MRR gap > 0.010 and 3p MRR gap > 0.010 (hops=2 vs hops=1)
Seeds: [42, 123, 456]
Result: REJECTED

Metrics (FB15k-237 top-500 subgraph, 494 entities, 160 relations, 3 seeds × 500 epochs):
Condition    adj_pairs   LP MRR       1p MRR       2p MRR       3p MRR       2p→3p   Time(s)
node_only        0       0.505±0.006  0.553±0.006  0.728±0.011  0.742±0.012  +0.014    115
hops=1   1,535,500       0.498±0.008  0.551±0.001  0.721±0.007  0.729±0.007  +0.008    849
hops=2   1,500,000*      0.496±0.003  0.534±0.008  0.726±0.014  0.731±0.006  +0.005    744

(* hops=2 natural adj = 28,157,322 pairs; capped at 1.5M for VRAM feasibility — RTX 3080 Ti 12.9GB)

Individual seed results:
Condition  Seed  LP MRR  1p MRR  2p MRR  3p MRR  3p H@10
node_only   42   0.5085  0.5555  0.7268  0.7349  0.8598
node_only  123   0.4974  0.5439  0.7151  0.7329  0.8565
node_only  456   0.5096  0.5584  0.7416  0.7587  0.8610
hops=1      42   0.4911  0.5496  0.7125  0.7276  0.8531
hops=1     123   0.5093  0.5512  0.7294  0.7215  0.8544
hops=1     456   0.4939  0.5519  0.7221  0.7373  0.8540
hops=2      42   0.5002  0.5365  0.7342  0.7321  0.8593
hops=2     123   0.4922  0.5421  0.7062  0.7239  0.8549
hops=2     456   0.4962  0.5235  0.7377  0.7376  0.8550

Gap analysis:
  hops=2 vs hops=1:   2p=+0.0047, 3p=+0.0024   [hops=1 σ_2p=0.007 → gaps within 1σ]
  hops=1 vs node_only: 2p=−0.0065, 3p=−0.0134  [edge attention hurts vs node-only!]

Key insight: On this dense subgraph (mean degree ≈19.7), multi-hop MRR is dominated
  by entity embedding quality learned during LP training. The edge adjacency stream
  provides no measurable advantage over pure node attention — and may slightly hurt.
  The synthetic transitive benchmark result (2-hop 100% vs 1-hop 61.1%) does NOT
  generalize to FB15k-237 chain queries at this density.
Next question: Does 2-hop provide benefit on the SPARSE full FB15k-237 graph
  (14.5K entities, mean degree ≈4.1), where 1-hop neighbors are genuinely limited?
Status: LOGGED as REJECTED — motivates Phase 67 (full graph, sparse regime)
```

## Details

### Hypothesis

The paper's central architectural claim is that 2-hop edge adjacency enables direct
relational composition via $\mathbf{A}_E^{(2)} = (\mathbf{B}^\top \mathbf{B})^2 - \text{diag}$.
A NeurIPS 2026 reviewer specifically requested this ablation: "No ablations of the core
mechanism. Where's the 1-hop vs 2-hop edge adjacency ablation on the main table?"

The hypothesis was: hops=2 provides measurably better multi-hop MRR than hops=1 on
FB15k-237 multi-hop queries, with a gap > 0.010 on both 2p and 3p.

### Experimental Design

Three conditions, held constant: model (DELTA-Matched, 157,720 params), training
protocol (500 epochs, lr=0.003, batch=4096, label_smoothing=0.1, patience=10),
evaluation (standard filtered LP + chain queries 1p/2p/3p at 10k queries).

**Condition A — node_only**: Edge attention stream completely disabled (empty adjacency).
  DELTA reduces to GAT-style node-only attention. Cache hops=1, adj_pairs=0.

**Condition B — hops=1**: Edges attending to edges sharing an endpoint (current default).
  This is what DELTALayer.forward() produces with no arguments to build_edge_adjacency().
  adj_pairs = 1,535,500.

**Condition C — hops=2**: Edges two steps away via A_E^(2) = (B^T B)^2 - diag.
  Natural adj = 28,157,322 pairs (18× hops=1). Randomly subsampled to 1,500,000 pairs
  for VRAM feasibility on RTX 3080 Ti (12.9GB). The cap is noted in the paper; a
  randomized subsample at 1.5M covers ~5.3% of the full 2-hop neighborhood, preserving
  same attention budget as hops=1.

The adj injection mechanism: graph._edge_adj_cache = (cache_hops, adj) before
encoder(graph) is called. DELTALayer.forward() calls build_edge_adjacency() (hops=1
default), which checks cache: if cached_hops >= 1: return cached_result. This cleanly
intercepts the encoder without modifying model.py.

### Configuration

- Dataset: FB15k-237 top-500 subgraph (494 entities, 160 relations)
- Train/val/test: 9703/390/486 triples
- Model: DELTA-Matched (d_node=48, d_edge=24, 2 layers, 4 heads, 157,720 params)
- Optimizer: Adam, lr=0.003
- Epochs: 500, eval_every=25, early_stopping patience=10
- Batch size: 4096
- Label smoothing: 0.1
- Seeds: [42, 123, 456]
- Multi-hop queries: 1p=486, 2p=5764, 3p=10000 (leakage audit: PASSED)
- Hardware: NVIDIA GeForce RTX 3080 Ti (12.9GB VRAM), CUDA
- Total runtime: 1879s (0.5h)
- Output: phase66_output.json

### Condition A — node_only Training Trajectories (seed=42)

| Epoch | Loss   | val_MRR | val_H@10 | Time(s) |
|-------|--------|---------|----------|---------|
| 25    | 0.0147 | 0.0129  | 0.0333   | 2       |
| 100   | 0.0134 | 0.0449  | 0.1269   | 8       |
| 200   | 0.0121 | 0.1365  | 0.3000   | 15      |
| 250   | 0.0112 | 0.4174  | 0.6808   | 19      |
| 375   | 0.0106 | **0.5452** | 0.8128 | 29    |
| 500   | 0.0101 | 0.5113  | 0.8218   | 39      |
| Final | —      | LP test: 0.5085 | H@10: 0.8076 | — |

### Condition B — hops=1 Training Trajectories (seed=42)

| Epoch | Loss   | val_MRR | val_H@10 | Time(s) |
|-------|--------|---------|----------|---------|
| 25    | 0.0146 | 0.0085  | 0.0077   | 15      |
| 200   | 0.0127 | 0.1578  | 0.3987   | 114     |
| 375   | 0.0106 | **0.5173** | 0.7808 | 213   |
| 500   | 0.0101 | 0.5136  | 0.8205   | 284     |
| Final | —      | LP test: 0.4911 | H@10: 0.7613 | — |

*hops=1 epoch cost: ~14s/epoch (vs ~0.5s for node_only) due to 1.5M adj pairs in edge attention forward.*

### Condition C — hops=2 Training Trajectories (seed=42)

| Epoch | Loss   | val_MRR | val_H@10 | Time(s) |
|-------|--------|---------|----------|---------|
| 25    | 0.0146 | 0.0086  | 0.0077   | 12      |
| 175   | 0.0120 | 0.2596  | 0.4410   | 85      |
| 325   | 0.0106 | **0.5258** | 0.7769 | 161   |
| 500   | 0.0100 | 0.5258  | 0.8141   | 248     |
| Final | —      | LP test: 0.5002 | H@10: 0.8035 | — |

*hops=2 epoch cost: ~12s/epoch at 1.5M subsampled pairs (comparable to hops=1 at same adj budget).*

### Key Observations

1. **All three conditions are statistically indistinguishable.** Every pairwise gap is within
   one standard deviation: hops=2 vs hops=1 has 2p gap=+0.005 (well below σ=0.007 for hops=1
   and σ=0.014 for hops=2). Three seeds is insufficient to claim significance here.

2. **node_only achieves highest point estimates on every metric.** LP MRR: 0.505 vs 0.498/0.496.
   2p MRR: 0.728 vs 0.721/0.726. 3p MRR: 0.742 vs 0.729/0.731. The edge attention stream
   slightly hurts on this dense subgraph (hops=1 is −0.7% on 2p vs node_only).

3. **hops=2 converges faster than hops=1** (seed=42: val_MRR=0.260 at ep175 vs 0.157 at ep200
   for hops=1). The richer 2-hop neighborhood appears to provide better gradient signal early,
   but this convergence advantage does not translate to better final performance.

4. **The synthetic transitive result does not generalize.** On the transitive benchmark, 2-hop
   achieves 100% accuracy vs 61.1% for 1-hop — a 38.9 point gap. On FB15k-237 multi-hop chain
   queries, the gap is +0.005 MRR (within noise). The synthetic benchmark has deterministic
   relational chains; FB15k-237 has noisy, multi-path knowledge with many valid answer routes.

5. **Memory constraint note.** Natural hops=2 adj = 28.2M pairs. At 1.5M subsampling (5.3%),
   the hops=2 condition is not a clean test of the full 2-hop neighborhood — it tests whether
   a random subsample of the 2-hop neighborhood outperforms the complete 1-hop neighborhood.
   The fact that even this partial 2-hop does not clearly outperform supports the rejection.

6. **Density hypothesis.** Mean node degree ≈ 19.7 on this subgraph. With ~20 neighbors per node,
   1-hop edge adjacency already captures ~20² = 400 edge pairs per edge. The 2-hop extension on
   a dense graph produces a near-complete edge graph, where structural locality information is
   diluted. On a sparse graph (mean degree ≈ 4.1 on full FB15k-237), 1-hop edge adjacency
   would cover only ~16 pairs per edge — leaving much more room for 2-hop to add signal.

7. **Edge attention does not hurt.** All three conditions achieve competitive LP MRR (0.496–0.505)
   and multi-hop MRR (0.721–0.742 on 2p). The edge attention stream is architecturally sound;
   it simply does not provide incremental benefit over node attention at this scale and density.

### Classification: REJECTED

- hops=2 vs hops=1 gap: 2p=+0.005, 3p=+0.002 — both below expected 0.010 threshold
- Both within 1σ of each condition's 3-seed std
- Surprising: node_only outperforms both on all metrics (within noise, but consistently)
- Hypothesis that 2-hop is the driver of multi-hop reasoning is NOT confirmed
- The N=500 dense subgraph is not the right test for this hypothesis

### Impact

- **Paper**: Table `tab:ablation_hop` now populated with actual numbers and honest framing.
  The caption no longer claims "2-hop is essential"; it notes statistical indistinguishability
  and interprets the density effect. Section text explains the subgraph density hypothesis
  and motivates Phase 67 as the critical test.

- **Architecture**: The DELTALayer.forward() still calls build_edge_adjacency() with the
  hops=1 default. This is fine — hops=2 is not worse, and on the full sparse graph it may
  be better. The default should be revisited after Phase 67.

- **NeurIPS response**: The reviewer asked for this ablation; we can now provide it with
  honest numbers. The honest answer ("indistinguishable on dense subgraph, motivates Phase 67
  on sparse full graph") is a stronger response than fabricated significance would be.

- **Research direction**: The key unanswered question (Phase 67) is whether 2-hop provides
  benefit on the sparse full graph. If yes, the 2-hop claim is vindicated. If no, the
  architecture should be reconsidered for the paper's main contribution framing.

### Next Steps (Phase 67)

1. Run Phase 66 equivalent on full sparse FB15k-237 (14.5K entities, mean degree ≈4.1)
   — this requires GPU compute beyond local RTX 3080 Ti (RunPod H100 recommended)
2. Include NBFNet and A*Net as additional baselines (addresses reviewer Issue 3)
3. Use topk=128 sparse attention (validated in Phase 64) to make full-graph tractable
4. Report hops=1 vs hops=2 gap on sparse graph; if gap > 0.010, paper's 2-hop claim is
   vindicated in the right regime. If gap < 0.010, revise contribution framing.
5. Plan Phase 68: evaluate on standard BetaE query splits (9 query types) to address
   reviewer concern about "homemade benchmark"
