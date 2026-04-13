# Phase 60: Residual Gating for Depth Scaling

**Date:** 2026-04-13
**Hardware:** RTX PRO 6000 Blackwell (98 GB), RunPod pod coloured_teal_wolf
**Commit:** (this commit)

## Goal

Test whether learnable residual gates enable multi-layer DELTA at N=2000, recovering from Phase 59's catastrophic 3-layer over-smoothing (MRR=0.0018).

## Hypothesis

**2-layer DELTA with residual gating achieves LP MRR ≥ 0.30 at N=2000**, matching or exceeding the 1-layer baseline (val MRR=0.3338, test MRR=0.3094).

## Mechanism

Per-layer learnable gate logits (one per layer for both node and edge features):
```
output = sigmoid(α) * layer_output + (1 - sigmoid(α)) * input
```
- `α` initialized so sigmoid(α) ≈ 0.1 → residual dominates at start
- Separate gate parameters for node and edge features
- Gates are learned end-to-end via backpropagation

## Conditions

| Condition | Layers | Gate | Init α | Description |
|-----------|--------|------|--------|-------------|
| A | 2 | Yes | 0.1 | 2-layer + residual gate |
| B | 3 | Yes | 0.1 | 3-layer + residual gate |
| C | 1 | No | — | 1-layer control (no gate) |

All: d_node=64, d_edge=32, num_heads=4, lr=0.003, bs=4096, 200 epochs, seed=42, cached_edge_adj, N=2000.

## Results

### Summary Table (Test Set)

| Condition | Test MRR | H@1 | H@3 | H@10 | Params | Time |
|-----------|----------|------|------|------|--------|------|
| A: 2L+gate | 0.3065 | 0.1851 | 0.3490 | 0.5689 | 311,652 | 3.2hr |
| B: 3L+gate | **0.3138** | **0.1962** | **0.3511** | 0.5591 | 393,830 | 4.8hr |
| C: 1L ctrl | 0.3093 | 0.1796 | 0.3495 | **0.5939** | 229,472 | 1.6hr |

### Phase 59 References

| Model | MRR | Note |
|-------|-----|------|
| 3-layer ungated | 0.0018 | Catastrophic over-smoothing |
| 1-layer | 0.3094 | Phase 59 test |
| DistMult | 0.3185 | No GNN baseline (val) |

### Condition A: 2-Layer + Gate — Training Trajectory

| Epoch | Loss | val_MRR | H@1 | H@3 | H@10 | Node Gates | Edge Gates |
|-------|------|---------|------|------|------|------------|------------|
| 25 | 0.0043 | 0.0100 | 0.0000 | 0.0097 | 0.0145 | [0.132, 0.133] | [0.104, 0.100] |
| 50 | 0.0042 | 0.0303 | 0.0114 | 0.0140 | 0.0909 | [0.133, 0.133] | [0.103, 0.100] |
| 75 | 0.0041 | 0.0337 | 0.0050 | 0.0168 | 0.1074 | [0.133, 0.134] | [0.102, 0.100] |
| 100 | 0.0038 | 0.1303 | 0.0666 | 0.1451 | 0.2528 | [0.133, 0.134] | [0.096, 0.100] |
| 150 | 0.0034 | 0.2951 | 0.1886 | 0.3286 | 0.5098 | [0.134, 0.135] | [0.095, 0.100] |
| 200 | 0.0031 | **0.3177** | 0.2007 | 0.3519 | 0.5768 | [0.135, 0.137] | [0.097, 0.100] |

### Condition B: 3-Layer + Gate — Training Trajectory

| Epoch | Loss | val_MRR | H@1 | H@3 | H@10 | Node Gates | Edge Gates |
|-------|------|---------|------|------|------|------------|------------|
| 25 | 0.0043 | 0.0168 | 0.0084 | 0.0094 | 0.0251 | [0.127, 0.128, 0.127] | [0.106, 0.104, 0.100] |
| 50 | 0.0043 | 0.0211 | 0.0062 | 0.0226 | 0.0318 | [0.127, 0.128, 0.127] | [0.107, 0.104, 0.100] |
| 75 | 0.0040 | 0.0642 | 0.0276 | 0.0564 | 0.1425 | [0.127, 0.128, 0.127] | [0.106, 0.104, 0.100] |
| 100 | 0.0037 | 0.1851 | 0.1075 | 0.1962 | 0.3452 | [0.128, 0.129, 0.128] | [0.106, 0.103, 0.100] |
| 150 | 0.0034 | 0.2742 | 0.1651 | 0.3047 | 0.5020 | [0.129, 0.130, 0.129] | [0.106, 0.103, 0.100] |
| 200 | 0.0031 | **0.3141** | 0.2001 | 0.3485 | 0.5647 | [0.129, 0.130, 0.130] | [0.105, 0.102, 0.100] |

### Condition C: 1-Layer Control — Training Trajectory

| Epoch | val_MRR | H@1 | H@3 | H@10 |
|-------|---------|------|------|------|
| 25 | 0.0024 | 0.0000 | 0.0000 | 0.0000 |
| 50 | 0.0716 | 0.0424 | 0.0792 | 0.1255 |
| 75 | 0.1608 | 0.0814 | 0.1781 | 0.3161 |
| 100 | 0.2652 | 0.1599 | 0.3041 | 0.4696 |
| 150 | **0.3335** | 0.2127 | 0.3812 | 0.5839 |
| 200 | 0.3176 | 0.1892 | 0.3566 | 0.5988 |

## Key Observations

1. **Residual gating completely eliminates catastrophic over-smoothing.** 3-layer+gate (MRR=0.3138) vs 3-layer ungated (MRR=0.0018) — a **174× improvement**.

2. **All three conditions produce statistically indistinguishable test MRR**: A=0.3065, B=0.3138, C=0.3093. The gating mechanism successfully preserves signal through multiple layers.

3. **3-layer+gate slightly outperforms 1-layer** on test MRR (0.3138 vs 0.3093) and H@1 (0.1962 vs 0.1796), suggesting depth has marginal value with proper residual gating.

4. **Gates are essentially frozen at initialization.** Node gates moved from 0.100 to 0.130–0.137; edge gates barely moved (0.095–0.105). The model does NOT learn to open the gates — it achieves performance by operating at ~10% layer contribution, 90% residual.

5. **Edge gates for layer 1 decrease** in condition A (0.104→0.097), suggesting the model prefers LESS first-layer edge transformation. Last-layer edge gates in all conditions remain at exactly the initialization value.

6. **Warmup plateau extends with depth.** 1-layer: breakthrough at ep40-50. 2-layer: breakthrough at ep75-100. 3-layer: breakthrough at ep75-100. More layers delay the learning onset.

7. **1-layer still achieves the highest peak val MRR (0.3338)** and highest H@10 (0.5939). Gating enables multi-layer but doesn't produce a clear accuracy advantage.

8. **Cost scales linearly with depth.** 2-layer: 3.2hr, 3-layer: 4.8hr, 1-layer: 1.6hr. Each layer adds ~1.6hr at N=2000. Given near-identical accuracy, 1-layer remains most efficient.

## Hypothesis Evaluation

| Hypothesis | Result | Evidence |
|-----------|--------|----------|
| 2L+gate MRR ≥ 0.30 at N=2000 | **CONFIRMED** | Test MRR=0.3065, meets threshold. |
| 3L+gate recovers from catastrophic over-smoothing | **CONFIRMED** | MRR 0.0018→0.3138 (174× improvement). |
| Gate logits shift during training | **PARTIAL** | Node gates shift slightly (0.100→0.130); edge gates essentially frozen. Gates are NOT actively learned. |
| Multi-layer outperforms 1-layer with gating | **NOT CONFIRMED** | 3L=0.3138, 2L=0.3065, 1L=0.3093 — within noise. No clear multi-layer advantage. |

## Classification

**CONFIRMED (narrowly)** — Residual gating eliminates depth over-smoothing. But the harder finding is that the fix revealed the layers are not contributing useful signal. Gates frozen at ~10% mean the model learned that layer outputs are mostly noise and should be mostly ignored. When ignored, all depths and DistMult itself converge to the same ~0.31 MRR — meaning **DELTA's edge-to-edge attention contributes zero measurable value at N=2000**.

The 174× improvement over ungated 3-layer is real but is an improvement over a broken baseline. The meaningful comparison is against DistMult, and DELTA isn't beating it:
- DistMult (no GNN): 0.3185
- 1-layer DELTA: 0.3093
- 2-layer+gate: 0.3065
- 3-layer+gate: 0.3138

All within noise. The depth problem is solved but the value-of-attention problem is exposed.

## Implications for Phases 61–62

- Phase 60 shifted the question from "can we make DELTA work at scale?" to **"is there any scale beyond N=500 where DELTA's edge-to-edge attention provides measurable lift over a trivial baseline?"**
- The critical missing data point: **DistMult at N=500**. If DistMult also hits ~0.48 at N=500, then DELTA never beat DistMult at any scale — every prior "DELTA works" result is "this dataset is easy and any model works."
- Phase 61 must answer the existence question before any further mechanism work (sparsification, full-scale eval) is justified.
