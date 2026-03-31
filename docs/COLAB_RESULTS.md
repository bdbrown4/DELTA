# DELTA Colab Results

Results from Google Colab Pro+ experiments (Phases 35–37).

**Status:** Ready for Phase 35–37 execution on Colab Pro+ H100 / RTX PRO 6000 Blackwell.

---

## GPU Info

```
[Will be populated on next Colab run]

GPU: NVIDIA H100 80GB HBM3 or RTX PRO 6000 Blackwell 102GB
CUDA: 13.0+
Driver: 580.82.07+
```

---

## Tests (44/44)

All 44 unit tests passing locally. Will re-verify on Colab before running experiments.

---

## Phase 35: Domain-Agnostic Relational Transfer

**Objective:** Solve Phase 32's zero-shot transfer failure (0.048 ≈ random).

**Data:** `--full` uses **real FB15k-237** (14,541 entities, 237 relations) → **real WN18RR** (40,943 entities, 11 relations). Auto-downloads on first run.

**Design:** 3-step diagnostic pipeline
1. **Linear Probe** — Freeze encoder, train fresh classifier head on target domain
2. **Domain-Adversarial Training (GRL)** — Gradient reversal layer with DANN λ warmup
3. **Constructor Entanglement Ablation** — Compare baseline vs GRL via probe (cross-domain: 237→11 classes, direct zero-shot N/A)

**Success Criteria:**
- GRL probe > baseline probe (measures domain-invariant representation quality)
- GRL + probe > 0.7
- Linear probe distinguishes encoder vs head entanglement

**Estimated Runtime:** 1-2 hours on H100

**Status:** IN PROGRESS (Steps 0–1 complete, Step 2 interrupted)

### Step 0: Baseline (Phase 32 Reproduction)

| Metric | Value |
|--------|-------|
| Source accuracy (FB15k-237 val) | **0.992** |
| Zero-shot on WN18RR | N/A (237 → 11 classes) |
| Convergence | 85 epochs (plateau at 0.992) |

Training curve:
```
Epoch   5  Val: 0.977    Epoch  50  Val: 0.990
Epoch  10  Val: 0.983    Epoch  65  Val: 0.991
Epoch  20  Val: 0.987    Epoch  85  Val: 0.992 (best)
Epoch  30  Val: 0.988    Epoch 100  Val: 0.990
```

### Step 1: Linear Probe (Diagnostic)

| Metric | Value |
|--------|-------|
| Probe accuracy (100 samples) | **0.961** |
| Random baseline | 0.091 |
| Verdict | **Encoder transfers! Head was the bottleneck.** |

Key finding: Frozen encoder + fresh 11-class head on just 100 WN18RR samples → 0.961.
DELTA's attention patterns capture domain-agnostic structural features.

### Step 2: Domain-Adversarial Training (GRL)

**Status:** [PENDING — restarting with `--skip-to-step 2`]

```
[Will be populated on next run]
```

### Step 3: Constructor Entanglement Ablation

**Status:** [PENDING]

```
[Will be populated on next run]
```

---

## Phase 36: Task-Aware Construction at Scale

**Objective:** Scale Phase 33's flat results to graphs where missing edges matter.

**Design:** 3 experiments
- **Experiment A:** Sparse path graphs (500–2000 nodes, 33% edges removed)
- **Experiment B:** Cross-cluster reasoning (5–10 clusters, sparse inter-cluster bridges)
- **Experiment C:** Edge threshold sweep (0.3, 0.1, 0.05)

**Success Criteria:**
- Augmented DELTA > Fixed topology by > 3% on ≥ 50% of configs
- Lower thresholds enable more edge proposals without hurting accuracy

**Estimated Runtime:** 2–4 hours on H100

**Status:** [PENDING]

```
[Results will be populated after Colab execution]
```

---

## Phase 37: Real FB15k-237 Parameter-Matched Comparison

**Objective:** Isolate DELTA's architectural advantage from parameter count (60K vs ~30K).

**Data:** `--full` uses **real FB15k-237** (14,541 entities, 237 relations, 310,116 triples) with official train/valid/test splits. Auto-downloads on first run.

**Design:** 4 models, 5 seeds, full FB15k-237
- DELTA-Full (d_node=64, d_edge=32, layers=3) → ~60K params
- DELTA-Matched (d_node=48, d_edge=24, layers=2) → ~30K params ← **key test**
- GraphGPS (d_node=64, d_edge=32, layers=3) → ~33K params
- GRIT (d_node=64, d_edge=32, layers=3) → ~28K params

**Success Criteria:**
- DELTA-Matched (30K) > GraphGPS (33K) by ≥ 2% (5-seed mean)
- DELTA-Full ≥ DELTA-Matched (more params should help)

**Estimated Runtime:** 6–10 hours on H100 (all 4 models × 5 seeds)

**Status:** [PENDING]

```
[Results will be populated after Colab execution]
```

---

## Execution Guide

### Phase 35
```bash
# Quick test — synthetic data (500 source, 300 target)
python experiments/phase35_relational_transfer.py

# Full scale — REAL FB15k-237 → WN18RR (auto-downloads)
python experiments/phase35_relational_transfer.py --full
```

### Phase 36
```bash
# Quick test (500-1000 nodes, synthetic)
python experiments/phase36_task_aware_at_scale.py --min_nodes 500

# Full scale (500-5000 nodes, all 3 experiments, synthetic)
python experiments/phase36_task_aware_at_scale.py --full
```

### Phase 37
```bash
# Quick test — synthetic (500 entities, 3 seeds)
python experiments/phase37_real_comparison.py --entities 500 --num_seeds 3

# Full publication run — REAL FB15k-237, official splits, 5 seeds (auto-downloads)
python experiments/phase37_real_comparison.py --full --num_seeds 5
```

---

## Quick Reference

| Phase | Task | Models | Data | Estimated Time |
|-------|------|--------|------|-----------------|
| 35 | Domain transfer diagnosis | DELTA | **Real** FB15k-237 → WN18RR | 1–2h |
| 36 | Constructor at scale | DELTA + Fixed baseline | Synthetic 500–5000 node sparse graphs | 2–4h |
| 37 | Architecture comparison | 4 models × 5 seeds | **Real** FB15k-237 (14.5K entities, official splits) | 6–10h |

**Total estimated time:** 9–16 hours on H100 for all three phases, 5 seeds.

---

*Last updated: March 28, 2026. Phases 35/37 use real KG benchmarks (auto-download). Phase 36 uses synthetic. Awaiting Colab execution.*
