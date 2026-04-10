# Phase 57 — Brain Temperature Annealing

## Result

```
Phase: 57 — Brain Temperature Annealing (Closing the 0.480 MRR Gap)
Hypothesis: Applying K-style node annealing (4→2) to brain_hybrid @ d=0.01 achieves LP MRR >= 0.480
Expected: Temperature annealing pushes brain_hybrid past the 0.480 MRR threshold missed in Phase 56 (0.4794)
Seeds: [42]
Result: PARTIAL

Metrics (LP — FB15k-237, 494 entities, 160 relations):
Model                    node_t  edge_t  Anneal  MRR     H@1     H@3     H@10    Time(s)
A: brain_baseline        1.0     1.0     no      0.4808  0.3251  0.5504  0.8076  2336
B: brain_K_anneal        4.0     6.0     yes     0.4818  0.3508  0.5370  0.7613  2636
C: brain_Q_anneal        4.0     7.0     yes     0.4769  0.3405  0.5370  0.7603  2445

Reference (Phase 56, same model @ d=0.01, seed=42):
brain_hybrid d=0.01      1.0     1.0     no      0.4794  0.3230  0.5658  0.8076  3671

vs. Phase 56:
  A (baseline, 200ep): MRR +0.0014 over Phase 56 C (300ep) — shorter training is better!
  B (K-style anneal):  MRR +0.0024 over Phase 56 C — best MRR of all brain_hybrid runs
  C (Q-style anneal):  MRR -0.0025 over Phase 56 C — higher edge temp hurts

Key insight: brain_hybrid @ d=0.01 EXCEEDS 0.480 MRR with both A (0.4808) and B (0.4818)! The improvement
comes primarily from reducing epochs from 300→200 (less overfitting), not from annealing. B's K-style annealing
adds marginal MRR (+0.001 over A) but TRADES H@10 (−0.046). Temperature annealing on brain_hybrid increases
H@1 precision at the cost of H@10 recall — opposite of the brain_hybrid value proposition (recall boost).
Status: PARTIAL — 0.480 MRR target reached (A: 0.4808, B: 0.4818) but annealing provides no benefit over baseline.
```

## Details

### Hypothesis

Applying K-style node temperature annealing (4→2 over first 50% of training) to brain_hybrid @ density=0.01 achieves LP MRR >= 0.480 on FB15k-237 (500-entity dense subset).

### Experimental Design

Phase 57 tests three temperature configurations on brain_hybrid @ d=0.01:
- **Condition A**: Baseline — temp=1.0, no annealing (Phase 56 C reference at 200 epochs)
- **Condition B**: K-style annealing — node temp 4→2 (linear over first 50%), edge temp=6.0
- **Condition C**: Q-style annealing — node temp 4→2 (linear over first 50%), edge temp=7.0

Temperature annealing applies only to Stage 3 (delta) layers of BrainEncoder. Node temperatures are annealed linearly from init to target over the first 50% of epochs, then made fully learnable. Edge temperatures are fixed at initialization then made learnable after annealing period.

### Configuration

- Epochs: 200, eval_every: 30, patience: 10
- batch_size: 512, lr: 0.001
- sparsity_weight: 0.01, target_density: 0.01
- Seeds: [42]
- Params: 311,361 total (264,385 encoder), 2,435 constructed edges
- Hardware: NVIDIA RTX PRO 6000 Blackwell (98GB VRAM) via Colab SSH + tmux

### Condition A (baseline, temp=1.0) — Training Trajectory

| Epoch | Loss   | val_MRR   | val_H@10 | Edges |
|-------|--------|-----------|----------|-------|
| 30    | 0.0135 | 0.0302    | 0.0513   | 2435  |
| 60    | 0.0121 | 0.2645    | 0.4308   | 2435  |
| 90    | 0.0110 | 0.4435    | 0.7064   | 2435  |
| 120   | 0.0106 | 0.4815    | 0.7615   | 2435  |
| 150   | 0.0103 | **0.5056**| 0.7910   | 2435  |
| 180   | 0.0100 | 0.5025    | 0.8051   | 2435  |
| 200   | 0.0097 | 0.4958    | 0.7987   | 2435  |

**Peak**: val_MRR=0.5056 at epoch 150. Slight decline after.
**Test** (best checkpoint): MRR=**0.4808**, H@1=0.3251, H@3=0.5504, H@10=**0.8076**.

### Condition B (K-style: node anneal 4→2, edge=6.0) — Training Trajectory

| Epoch | Loss   | val_MRR   | val_H@10 | node_t   |
|-------|--------|-----------|----------|----------|
| 30    | 0.0135 | 0.0178    | 0.0321   | 3.40     |
| 60    | 0.0119 | 0.2974    | 0.4603   | 2.80     |
| 90    | 0.0110 | 0.4712    | 0.7000   | 2.20     |
| 120   | 0.0106 | **0.5135**| 0.7474   | learnable|
| 150   | 0.0102 | 0.4972    | 0.7949   | learnable|
| 180   | 0.0099 | 0.5075    | 0.7962   | learnable|
| 200   | 0.0097 | 0.4985    | 0.8013   | learnable|

**Peak**: val_MRR=**0.5135** at epoch 120 (highest of all conditions! — at annealing→learnable transition).
**Test** (best checkpoint): MRR=**0.4818**, H@1=**0.3508**, H@3=0.5370, H@10=0.7613.

Annealing completes at epoch 100 (50% of 200). The val_MRR peak at epoch 120 occurs just 20 epochs after transition to learnable temperatures — the model finds its optimal temperature configuration quickly.

### Condition C (Q-style: node anneal 4→2, edge=7.0) — Training Trajectory

| Epoch | Loss   | val_MRR   | val_H@10 | node_t   |
|-------|--------|-----------|----------|----------|
| 30    | 0.0135 | 0.0172    | 0.0295   | 3.40     |
| 60    | 0.0120 | 0.2854    | 0.4538   | 2.80     |
| 90    | 0.0111 | 0.4662    | 0.7205   | 2.20     |
| 120   | 0.0106 | **0.4994**| 0.7577   | learnable|
| 150   | 0.0103 | 0.4959    | **0.8103**| learnable|
| 180   | 0.0099 | 0.4971    | 0.8103   | learnable|
| 200   | 0.0097 | 0.4820    | 0.7885   | learnable|

**Peak**: val_MRR=0.4994 at epoch 120. H@10 peaks at 0.8103 (ep 150-180).
**Test** (best checkpoint): MRR=0.4769, H@1=0.3405, H@3=0.5370, H@10=0.7603.

### Key Observations

1. **0.480 MRR target REACHED**: Both A (0.4808) and B (0.4818) exceed 0.480. This is the first time brain_hybrid crosses this threshold. The improvement over Phase 56 C (0.4794) comes primarily from shorter training (200 vs 300 epochs), reducing overfitting.

2. **Annealing provides marginal MRR benefit but hurts H@10**: B's K-style annealing adds only +0.001 MRR over baseline A (0.4818 vs 0.4808) while dropping H@10 by −0.046 (0.7613 vs 0.8076). This is the opposite trade-off brain_hybrid was designed for — constructed edges are supposed to boost recall.

3. **H@1 improves with annealing, H@10 degrades**: B achieves best H@1 (0.3508 vs A's 0.3251, +0.026) while having worst H@10 (0.7613). Temperature sharpening on brain_hybrid concentrates probability mass at rank-1, trading top-10 coverage — the standard precision/recall trade-off from temperature tuning on delta_full (Phases 46-52).

4. **Edge temp 7.0 (C) is worse than 6.0 (B) on brain_hybrid**: Unlike delta_full where edge=7.0 consistently boosted LP MRR (Phase 52: Q 0.4905 vs K 0.4819), on brain_hybrid edge=7.0 HURTS (C 0.4769 vs B 0.4818). The constructed edges may already provide sufficient edge-level information, making further edge sharpening redundant or harmful.

5. **B achieves highest val_MRR of all brain_hybrid conditions** (0.5135), peaking exactly at the annealing→learnable transition (ep 120). This suggests the annealing trajectory helps intermediate training even if final test metrics don't improve much.

6. **200 epochs is optimal for brain_hybrid @ d=0.01**: A at 200ep (MRR=0.4808) beats Phase 56 C at 300ep (MRR=0.4794, same seed, same config). The model peaks around ep 150 in all conditions.

### Comparison with delta_full Temperature Annealing (Phases 50-52)

| Property | delta_full | brain_hybrid |
|----------|-----------|-------------|
| Annealing MRR benefit | +0.002 (K vs baseline) | +0.001 (B vs A) |
| Annealing H@10 impact | Marginal | **−0.046** (large degradation) |
| Edge=7.0 effect | +0.009 LP boost | **−0.005 LP loss** |
| Optimal config | Annealing for 3p depth | **No annealing (baseline)** |

Temperature annealing works differently on brain_hybrid than on delta_full. The constructed edges already provide structural information that overlaps with what temperature sharpening achieves, making annealing redundant for MRR and harmful for recall.

### Conclusion

Phase 57 confirms brain_hybrid @ density=0.01 is competitive with delta_full (MRR ~0.481) with superior recall (H@10=0.8076). Temperature annealing is NOT recommended for brain_hybrid — the baseline configuration at 200 epochs achieves the best balance of precision and recall. Future optimization should focus on constructor quality (density tuning, architecture improvements) rather than attention temperature.
