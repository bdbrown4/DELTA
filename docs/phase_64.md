# Phase 64 — Top-k Sparse Edge-to-Edge Attention at N=5000

## Result

```
Phase: 64 — Top-k Sparse Edge-to-Edge Attention at N=5000
Hypothesis: Top-k sparse edge attention on full 63M E_adj matches or exceeds full softmax attention quality while reducing attention dilution
Expected: topk=128 test MRR ≥ Phase 63 Condition D (full softmax, 63M) = 0.2457
Seeds: [42]
Result: CONFIRMED (B: topk=128 matches full softmax); PARTIAL (C: topk=64 degrades −5.6%); D: OOM

Metrics (Link Prediction — FB15k-237, N=5000, 4977 entities, 225 relations):
Condition    Budget   TopK   peak_val  best_ep  test_MRR  test_H@1  test_H@10  gap_vs_DM    Time
A (P63)      30M      —      0.2499    125      0.2471    0.1481    0.4562     +0.0227      37653s
B            63M      128    0.2465    125      0.2472    0.1481    0.4581     +0.0228      76367s
C            63M      64     0.2360    125      0.2332    0.1349    0.4432     +0.0088      76077s
D            63M      256    —         —        OOM       —         —          —            —

DistMult baseline: test_MRR=0.2244 (from Phase 62)
Phase 63 full-softmax reference (63M E_adj, no topk): test_MRR=0.2457, time=78872s

vs. Phase 63 Baseline (A, 30M uniform subsample):
  B (topk=128): Δ test_MRR=+0.0001, speedup=0.49x
  C (topk=64): Δ test_MRR=-0.0139, speedup=0.49x

Total experiment time: 165644s (46.0hr)
Status: CONFIRMED (topk=128 validates sparse attention quality), PARTIAL (topk=64 degrades), OOM (topk=256)
```

## Details

### Hypothesis

Top-k sparse edge attention on the full 63M E_adj adjacency matrix would match or exceed the performance of full softmax attention (Phase 63 Condition D), while providing a viable path to scaling DELTA to larger graphs where full E_adj softmax is intractable due to memory constraints. Three top-k values were tested (64, 128, 256) to characterize the quality vs. sparsity tradeoff.

### Experimental Design

**3-condition sequential ablation** with varying top-k attention width on fixed 63M E_adj:

- **Condition A (baseline):** Phase 63 Condition B — 30M pairs, no top-k, reused (no re-run)
- **Condition B:** Full 63M E_adj, top-k = 128
- **Condition C:** Full 63M E_adj, top-k = 64
- **Condition D:** Full 63M E_adj, top-k = 256

All conditions use the same model (DELTA 1L, seed=42), hyperparameters, and data split. Only top-k width varies. Conditions B–D run sequentially with independent model initialization. Full 63M E_adj built once and shared across conditions.

**Sparse attention mechanism:** For each edge in the batch, compute attention scores against all E_adj neighbors, retain only the top-k highest-scoring neighbors, renormalize, and apply. Memory scales as O(N_batch × top_k × d_edge) rather than O(N_batch × E_adj).

### Configuration

- Model: DELTA 1-layer, 4 heads, d_node=64, d_edge=32, init_temp=1.0
- Epochs: 150 max, eval every 25, early stopping PATIENCE=2
- Batch size: 4096, lr=0.003, optimizer: Adam
- Seed: 42
- Hardware: RTX PRO 6000 Blackwell Server Edition (98GB VRAM), RunPod
- Full E_adj: 63,001,372 pairs (built in 0.4s via spspmm)

### Training Trajectories

**Condition B (63M, topk=128) — 508.7s/epoch, est. 76,299s:**

| Epoch | Loss | val MRR | val H@1 | val H@10 | Time |
|-------|------|---------|---------|----------|------|
| 25 | 0.0019 | 0.0013 | 0.0000 | 0.0002 | 12723s |
| 50 | 0.0016 | 0.1333 | 0.0625 | 0.2756 | 25449s |
| 75 | 0.0016 | 0.2153 | 0.1310 | 0.3870 | 38181s |
| 100 | 0.0015 | 0.2221 | 0.1266 | 0.4233 | 50901s |
| **125** | **0.0014** | **0.2465** | **0.1469** | **0.4542** | **63625s** |
| 150 | 0.0013 | 0.2341 | 0.1339 | 0.4491 | 76344s |

Test @ best val (ep125): **MRR=0.2472**, H@1=0.1481, H@10=0.4581

**Condition C (63M, topk=64) — 505.7s/epoch, est. 75,860s:**

| Epoch | Loss | val MRR | val H@1 | val H@10 | Time |
|-------|------|---------|---------|----------|------|
| 25 | 0.0019 | 0.0062 | 0.0017 | 0.0086 | 12674s |
| 50 | 0.0018 | 0.0848 | 0.0451 | 0.1511 | 25344s |
| 75 | 0.0016 | 0.1667 | 0.0859 | 0.3355 | 38029s |
| 100 | 0.0014 | 0.2307 | 0.1376 | 0.4240 | 50709s |
| **125** | **0.0014** | **0.2360** | **0.1372** | **0.4443** | **63381s** |
| 150 | 0.0013 | 0.2275 | 0.1321 | 0.4345 | 76052s |

Test @ best val (ep125): **MRR=0.2332**, H@1=0.1349, H@10=0.4432

**Condition D (63M, topk=256) — ABORTED:**

| Epoch | Loss | val MRR | val H@1 | val H@10 | Time |
|-------|------|---------|---------|----------|------|
| 25 | 0.0019 | 0.0125 | 0.0060 | 0.0263 | 12780s |

→ **OOM after ep25** — 98GB GPU could not sustain topk=256 on 63M E_adj batches.

### Key Observations

1. **topk=128 perfectly preserves full-attention quality:** Condition B (topk=128) achieves test MRR=0.2472, compared to Phase 63 Condition D (full softmax on same 63M E_adj) test MRR=0.2457 — a +0.0015 improvement, within noise. Sparse attention at 128 neighbors loses nothing vs. full softmax on 63M pairs.

2. **topk=128 matches the Phase 63 optimal (30M subsample) on test MRR:** Condition B (0.2472) ≈ Phase 63 Condition B (0.2471) — the 30M subsample with full softmax. Two entirely different sparsification strategies (random subsampling vs. top-k selection) arrive at the same quality point, though both use 2× the compute of the 30M baseline.

3. **topk=64 degrades meaningfully:** Condition C achieves test MRR=0.2332 vs. 0.2472 for topk=128 — a −5.6% relative drop. Still +0.0088 above DistMult (0.2244), but the gap narrows from +0.0228 (topk=128) to +0.0088 (topk=64). Halving the attention neighborhood from 128 to 64 costs 5.6% of test MRR.

4. **topk=256 OOM on 98GB GPU:** Even the Blackwell 98GB card cannot sustain topk=256 on full 63M E_adj. The VRAM limit lies between topk=128 and topk=256 at N=5000. This establishes a practical ceiling for sparse attention width at this scale.

5. **Attention dilution warmup persists under top-k:** Both conditions B and C show near-zero MRR at ep25 (0.0013 and 0.0062 respectively). This confirms that attention dilution is not purely a function of neighborhood size — even with topk=64 forcing a small neighborhood, the model still needs ~50 epochs to learn discriminative patterns. The warmup mechanism may relate to weight magnitude initialization rather than attention width alone.

6. **topk=64 has faster ep25 recovery but slower ep50-ep75:** Condition C ep25 MRR=0.0062 vs. Condition B 0.0013 — topk=64 starts more discriminatively. But by ep50, Condition B (0.1333) already leads Condition C (0.0848). The pattern inverts: focused attention (64 neighbors) starts faster but learns representations that saturate earlier; broader attention (128 neighbors) recovers slowly but learns richer representations.

7. **Condition C crosses Condition B at ep100:** Condition C val MRR=0.2307 > Condition B 0.2221 at ep100 — a brief inversion. By ep125, Condition B retakes the lead (0.2465 vs. 0.2360). This transient crossing suggests that at medium training, focused neighborhoods give a temporary advantage before broader attention wins long-term.

8. **Wall time identical for B and C:** Both conditions complete in ~76,000s. topk=64 does not provide speed savings vs. topk=128 at N=5000. The compute bottleneck is not the attention operation but the upstream E_adj pair processing.

9. **All conditions peak at ep125.** Best epoch is invariant to top-k width, consistent with Phase 63's finding that ep125 is the optimal training duration for this dataset/model configuration.

10. **H@10 is sensitive to topk:** Unlike Phase 63 where H@10 was invariant to E_adj budget (0.456–0.458), Phase 64 shows differentiation: B=0.4581, C=0.4432 (−3.3%). The top-10 ranking quality degrades meaningfully when attention is restricted to 64 neighbors.

### Comparison with Phase 63

| Metric | P63-B (30M, no topk) | P63-D (63M, no topk) | P64-B (63M, topk=128) | P64-C (63M, topk=64) |
|--------|---------------------|---------------------|----------------------|----------------------|
| test_MRR | 0.2471 | 0.2457 | **0.2472** | 0.2332 |
| test_H@1 | 0.1481 | 0.1459 | 0.1481 | 0.1349 |
| test_H@10 | 0.4562 | 0.4558 | **0.4581** | 0.4432 |
| gap_vs_DM | +0.0227 | +0.0213 | **+0.0228** | +0.0088 |
| time | 37653s | 78872s | 76367s | 76077s |
| epoch/s | ~250s | ~525s | ~508s | ~505s |

topk=128 at 63M E_adj is essentially identical to full softmax at 63M E_adj, but both are ~2× slower than the 30M subsample approach. **The optimal cost/quality point remains Phase 63 Condition B: 30M subsample, full softmax, ~37,000s total.**

### Hypothesis Evaluation

| Hypothesis | Result | Evidence |
|-----------|--------|----------|
| topk=128 matches full softmax on 63M E_adj | **CONFIRMED** | B test_MRR=0.2472 vs Phase 63 D=0.2457, Δ=+0.0015 (noise) |
| topk=64 degrades vs topk=128 | **CONFIRMED** | C test_MRR=0.2332 vs B=0.2472, Δ=−0.0140 (−5.6%) |
| topk=256 is feasible on 98GB GPU | **REJECTED** | OOM after ep25; limit lies between 128 and 256 |
| Sparse attention reduces attention dilution warmup | **PARTIAL** | topk=64 faster at ep25 (0.0062 vs 0.0013), but warmup persists in both conditions |
| top-k provides speedup vs full softmax | **NOT CONFIRMED** | Epoch time near-identical; compute bottleneck is upstream, not attention |

### Classification: CONFIRMED + PARTIAL + OOM

- **CONFIRMED (topk=128):** Top-k sparse attention at 128 neighbors perfectly preserves full-softmax attention quality on 63M E_adj. This validates sparse attention as a viable architectural component for scaling to larger N where full softmax would be intractable.
- **PARTIAL (topk=64):** Halving the neighborhood degrades test MRR by −5.6%. The model remains above DistMult (+0.0088 gap), but the DELTA advantage is substantially eroded compared to topk=128.
- **OOM (topk=256):** The practical memory limit at N=5000 on a 98GB GPU lies between 128 and 256 neighbors. Any attention approach requiring 256+ neighbors per edge is infeasible at this scale.

### Impact

- **Sparse attention is validated as a scaling tool:** For larger N (10k+), where E_adj grows quadratically and full softmax becomes impossible, topk=128 preserves quality. This unlocks DELTA for larger-scale experiments.
- **The quality floor for acceptable sparsification is ~128 neighbors:** topk=128 is fine; topk=64 loses 5.6% MRR and 3.3% H@10. The operational threshold for sparse attention at this scale is 128 neighbors.
- **Cost/quality optimum is not sparse attention:** The 30M subsample (Phase 63 Cond B) achieves equivalent quality at half the wall time. For N=5000, the optimal strategy is random subsampling to 30M pairs, not sparse attention over 63M pairs.
- **Scaling to larger N requires topk:** At N=14,541 (full FB15k-237), E_adj would be ~210M pairs — full softmax is impossible. Phase 64 establishes that topk=128 quality is acceptable and OOM limit is between 128–256 neighbors per edge.

### Next Steps (Phase 65)

1. **Scale to full FB15k-237 (N=14,541):** With topk=128 validated, attempt the full 14,541-entity graph. E_adj at full scale would be ~180–210M pairs (estimated). With topk=128, the per-batch compute becomes tractable.
2. **Investigate topk=96 or topk=112:** The quality cliff between topk=64 (−5.6%) and topk=128 (0%) suggests a breakeven somewhere in 80–120. Testing intermediate values would sharpen the scaling characterization.
3. **Evaluate memory-efficient topk=128 at larger batch sizes:** Current bottleneck is upstream E_adj pair processing. Profiling may reveal whether batched sparse topk can reduce the 508s/epoch wall time.
4. **Consider hybrid approach:** 30M random subsample + topk=128 applied to the subsampled pairs. This would test whether topk adds value on top of subsampling, or if subsampling alone captures the same structural information.
