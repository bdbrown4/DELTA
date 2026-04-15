# Phase 61: DistMult vs DELTA Across Scales

**Date:** 2026-04-15
**Hardware:** RTX PRO 6000 Blackwell (98 GB), RunPod pod smooth_amaranth_wildcat
**Commit:** (this commit)

## Goal

Determine whether DELTA's edge-to-edge attention provides any measurable lift over a DistMult no-GNN baseline at N=500, 1000, and 2000. Phase 60 concluded that all depths + DistMult converge to ~0.31 MRR at N=2000, raising the question: does DELTA help at *any* scale?

## Hypothesis

**At least one scale N shows DELTA 1-layer test MRR ≥ DistMult + 0.01**, demonstrating the edge-to-edge attention mechanism provides genuine value.

## Design

| Scale | Entities | Relations | Train triples | Batch size | LR | Epochs | Steps/epoch | Total steps |
|-------|----------|-----------|---------------|------------|------|--------|-------------|-------------|
| N=500 | 494 | 160 | 9,703 | 512 | 0.001 | 500 | 19 | 9,500 |
| N=1000 | 998 | 186 | 26,761 | 2,048 | 0.002 | 300 | 14 | 4,200 |
| N=2000 | 1,991 | 207 | 62,733 | 4,096 | 0.003 | 200 | 16 | 3,200 |

Models tested:
- **DistMult** (no GNN): Learned entity/relation embeddings + DistMult scoring. No message passing.
- **1-layer DELTA**: Edge-to-edge attention (1 layer) + DistMult scoring.
- **3-layer DELTA** (N=500 only): Same but 3 layers to test depth contribution.

All: d_node=64, d_edge=32, num_heads=4, seed=42, Adam optimizer.

### Design Note: Uneven Gradient Step Budgets

The per-scale hyperparameters result in **uneven total gradient steps**: N=500 gets 9,500 steps while N=2000 gets only 3,200. This creates a systematic confound — larger scales get proportionally fewer gradient steps relative to their complexity. This turns out to be the dominant factor in the results.

## Results

### Summary Table (Test Set)

| Model | N | val_MRR | test_MRR | test_H@1 | test_H@10 | Δ vs DM | Time |
|-------|---:|--------:|---------:|---------:|----------:|--------:|-----:|
| DistMult | 500 | 0.4736 | 0.3747 | 0.2407 | 0.6698 | — | 10s |
| 1L-DELTA | 500 | 0.4892 | 0.3583 | 0.2150 | 0.6636 | **-0.0163** | 1054s |
| 3L-DELTA | 500 | 0.4882 | 0.3816 | 0.2418 | 0.6862 | +0.0069 | 3124s |
| DistMult | 1000 | 0.3288 | 0.3342 | 0.2257 | 0.5503 | — | 12s |
| 1L-DELTA | 1000 | 0.4024 | 0.3604 | 0.2159 | 0.6613 | **+0.0262** | 1626s |
| DistMult | 2000 | 0.2271 | 0.2297 | 0.1475 | 0.3845 | — | 60s |
| 1L-DELTA | 2000 | 0.3357 | 0.3088 | 0.1787 | 0.5963 | **+0.0791** | 5896s |

### Val-Best MRR Summary (Peak Validation)

| Scale | DM peak | DM peak epoch | 1L peak | 1L peak epoch | Gap |
|------:|--------:|--------------:|--------:|--------------:|----:|
| 500 | 0.4736 | ep372 | 0.4892 | ep124 | +0.0156 |
| 1000 | 0.3288 | ep300 (still rising) | 0.4024 | ep148 | +0.0736 |
| 2000 | 0.2271 | ep200 (still rising) | 0.3357 | ep175 | +0.1086 |

### VERDICT (from script)

```
N=500:  DELTA-DistMult gap = -0.0163
N=1000: DELTA-DistMult gap = +0.0262
N=2000: DELTA-DistMult gap = +0.0791
```

Script classified as "SCENARIO 1: DELTA ≈ DistMult at N=500" — **this is misleading** because it only checks N=500 first and ignores the large positive gaps at N=1000 and N=2000.

## Key Findings

### 1. DELTA Converges 3× Faster Per Gradient Step

The most significant finding is about **convergence speed**, not final accuracy:

| Scale | DM peak at step | 1L peak at step | Speedup |
|------:|----------------:|----------------:|--------:|
| 500 | 7,068 (ep372) | 2,356 (ep124) | **3.0×** |
| 1000 | 4,200+ (ep300+, still rising) | 2,072 (ep148) | **>2.0×** |
| 2000 | 3,200+ (ep200+, still rising) | 2,800 (ep175) | **>1.1×** |

DELTA's GNN layers provide an inductive bias that accelerates learning. DistMult starts from random embeddings and must discover structure purely from gradient updates; DELTA's message passing provides structural priors via edge adjacency.

### 2. With Sufficient Training, DistMult Catches Up (N=500)

At N=500, both models received 9,500 gradient steps — enough for both to converge and overfit:
- **DM**: Peaked at ep372 (7,068 steps), then declined
- **1L**: Peaked at ep124 (2,356 steps), then declined steeply

1L reached its peak 3× faster but overfit harder. The final-epoch 1L model (MRR=0.3976) was worse than the final-epoch DM model (MRR=0.4077). Test evaluation on the final model gives DM the edge: test MRR 0.3747 vs 0.3583.

**Conclusion at N=500**: DELTA has no final-accuracy advantage when both models have enough training. The advantage is purely in convergence speed.

### 3. Under Limited Training Budgets, DELTA's Advantage Grows (N=1000, N=2000)

At N=1000 (4,200 steps) and N=2000 (3,200 steps), DistMult was **still improving** at the end of training — it hadn't converged. DELTA, converging faster, reached better performance within the same budget:

| Scale | DM still rising? | 1L peaked? | Test gap |
|------:|:-----------------:|:----------:|----------|
| 500 | No (peaked ep372) | Yes (peaked ep124) | DM wins -0.016 |
| 1000 | **Yes** (still rising ep300) | Yes (peaked ep148) | 1L wins +0.026 |
| 2000 | **Yes** (still rising ep200) | Yes (peaked ep175) | 1L wins +0.079 |

The growing DELTA advantage at larger N reflects the shrinking step budget relative to graph complexity, **not** an intrinsic scale-dependent superiority.

### 4. Cross-Reference with Phase 59/60

Phase 59 used bs=512, lr=0.001 at N=2000, giving DistMult **24,600 gradient steps**:
- DistMult: val MRR=0.3185 (Phase 59 reference)
- 1L-DELTA: val MRR=0.3338 (Phase 59), test MRR=0.3094 (Phase 60)

Phase 61 used bs=4096, lr=0.003 at N=2000, giving only **3,200 gradient steps**:
- DistMult: val MRR=0.2271 (stuck at ep200)
- 1L-DELTA: val MRR=0.3357, test MRR=0.3088

With 24,600 steps, DM reaches 0.3185 and nearly matches DELTA's 0.31. With 3,200 steps, DM reaches only 0.2271 while DELTA still reaches 0.31. **DELTA's performance is step-count–invariant; DistMult's is not.**

### 5. 3-Layer DELTA ≈ 1-Layer DELTA at N=500

| Model | Peak val MRR | Test MRR | Time |
|-------|------------:|----------:|-----:|
| 1L | 0.4892 | 0.3583 | 1054s |
| 3L | 0.4882 | 0.3816 | 3124s |

3L achieves marginally better test MRR (+0.023) due to slower overfitting, but peak val is identical (0.489). Additional depth does not improve the learned representations — consistent with Phase 60's finding that gates freeze at ~10% contribution.

### 6. Design Limitation: No Early Stopping

Test evaluation uses the final-epoch model, not the best-validation checkpoint. This disadvantages models that overfit (especially 1L-DELTA at N=500, which peaked at ep124 but was evaluated at ep500). A proper implementation with early stopping would likely show:
- N=500: 1L test ≈ DM test (both peak high, both overfit)
- N=1000/2000: 1L test > DM test (DELTA peaks earlier, DM hasn't peaked yet)

## Convergence Trajectories (1L-DELTA@2000)

| Epoch | Loss | val_MRR | H@1 | H@10 | Time |
|------:|-----:|--------:|----:|-----:|-----:|
| 25 | 0.0044 | 0.0024 | 0.0000 | 0.0000 | 736s |
| 50 | 0.0040 | 0.0716 | 0.0424 | 0.1255 | 1471s |
| 75 | 0.0038 | 0.1608 | 0.0814 | 0.3161 | 2208s |
| 100 | 0.0035 | 0.2646 | 0.1584 | 0.4741 | 2945s |
| 125 | 0.0033 | 0.3148 | 0.2011 | 0.5478 | 3681s |
| 150 | 0.0032 | 0.3340 | 0.2118 | 0.5887 | 4418s |
| 175 | 0.0030 | **0.3357** | 0.2093 | 0.6036 | 5155s |
| 200 | 0.0029 | 0.3174 | 0.1905 | 0.5969 | 5891s |

At ep125 (2,000 steps), 1L@2000 already reached MRR=0.3148 — matching Phase 59's DistMult at 24,600 steps. By ep175 (2,800 steps), it exceeded DistMult's best (0.3357 vs 0.3185). Then declined due to overfitting at ep200.

## Hypothesis Evaluation

| Hypothesis | Result | Evidence |
|-----------|--------|----------|
| DELTA test MRR ≥ DM + 0.01 at some N | **CONFIRMED at N=1000 and N=2000** | N=1000: +0.026, N=2000: +0.079 |
| DELTA test MRR ≥ DM at N=500 | **REJECTED** | -0.016 due to harder overfitting |
| DELTA peak val > DM at all N | **CONFIRMED** | +0.016, +0.074, +0.109 |
| 3L improves over 1L at N=500 | **NOT CONFIRMED** | Test +0.023 but peak val identical |

## Classification

**UNRESOLVED** — Phase 61 demonstrates DELTA converges faster per gradient step, but this experiment has a critical uncontrolled confound that prevents any publishable conclusion.

### The Wall-Clock Confound

At N=2000, DistMult trained for 60 seconds. DELTA trained for 5,896 seconds — **98× more wall-clock compute**. The +0.079 test MRR gap could simply reflect that DELTA got 98× more compute time, not that it has a better inductive bias.

"Sample efficiency" (faster convergence per gradient step) sounds like a DELTA advantage, but flip it: **DistMult reaches MRR=0.23 in 60 seconds; DELTA reaches MRR=0.31 in 5,896 seconds.** Per second of compute, DistMult is vastly more efficient. Per FLOP, DistMult wins by orders of magnitude.

The "convergence speed per step" framing is only meaningful if steps are the bottleneck. In practice, nobody training on FB15k-237 is compute-constrained enough for step-efficiency to matter when each DELTA step takes ~100× longer than a DistMult step.

### What's Actually Known

- DELTA converges in fewer gradient steps than DistMult
- DELTA takes ~100× more wall-clock time per step
- At equal epoch count with sufficient epochs (N=500, 500 epochs), DistMult slightly beats DELTA on test
- At equal epoch count with insufficient epochs (N=2000, 200 epochs), DELTA wins — but DistMult was still climbing

### What's NOT Known (Critical)

**Does DistMult's MRR keep climbing with more epochs at N=2000?**

Phase 59 reference: DM reached val MRR=0.3185 with bs=512, lr=0.001 over 200 epochs (24,600 steps). Phase 61 gave DM only 3,200 steps with bs=4096, lr=0.003 — of course it underperformed.

The decisive experiment: DistMult at N=2000 for 1000–2000 epochs with the Phase 61 hyperparameters. This costs ~5 minutes of wall-clock time.

- **If DM reaches ~0.30+ MRR:** Phase 61's "advantage" is a compute artifact. DELTA is slower, not better.
- **If DM plateaus well below 0.30:** The GNN provides genuine inductive bias that DistMult cannot replicate regardless of training budget. That would be a real result.

**This experiment is Phase 61b. No conclusions from Phase 61 should be cited until 61b is complete.**

### Revision of Phase 60 Conclusion

Phase 60 concluded "DELTA contributes zero measurable value at N=2000." Phase 61 does NOT refute this — it merely shows DELTA converges faster per step while taking 98× longer per step. Whether this represents genuine value depends on Phase 61b.

## Runtime

| Component | Time |
|-----------|------|
| N=500 (DM + 1L + 3L) | 4,187s (~70 min) |
| N=1000 (DM + 1L) | 1,637s (~27 min) |
| N=2000 (DM + 1L) | 5,956s (~99 min) |
| **Total** | **~196 min (~3.3 hr)** |
