# DELTA: Publication Roadmap

**Target:** NeurIPS / ICLR (top-tier ML venue)
**Title placeholder:** *"DELTA: Edge-Centric Dual Attention for Relational Reasoning on Knowledge Graphs"*
**Status:** Phases 38–39 complete (differentiable construction + self-bootstrap). Phase 40 (correct LP evaluation) in progress.

**Current evidence base:** 40 experiment phases, 44 unit tests, competitive link prediction on FB15k-237 (MRR 0.497, still converging at 200 epochs), synthetic task dominance over GraphGPS/GRIT. **Self-bootstrapped DELTA at 157% of FixedChain** (Phase 39). Cross-domain transfer: 0.961 on WN18RR with 100 samples (frozen encoder).

---

## Phase Completion Overview

| Phase | Experiment | Status | Notes |
|-------|-----------|--------|-------|
| 34 | GraphGPS/GRIT vs DELTA — synthetic baseline | ✅ Complete | DELTA 0.880 vs GraphGPS 0.293 (H100) |
| 35 | Domain-agnostic relational transfer (GRL + linear probe) | ✅ Complete | Probe 0.961; GRL unnecessary |
| 36 | Task-aware construction at scale (500–5000 nodes) | ✅ Complete | Constructor adds ≤1.3%; hard thresholding blocked gradients |
| 37 | Real FB15k-237 parameter-matched comparison | ⚠️ Invalidated | 5 critical leakage issues found. Scale infra valid. Replaced by Phase 40 |
| **38** | **Differentiable task-aware constructor (3 variants, 3 seeds)** | ✅ Complete | **Hybrid 0.452 ± 0.006 (98% of FixedChain)** |
| **39** | **Self-bootstrapped DELTA (no transformer)** | ✅ Complete | **0.757 ± 0.041 (157% of FixedChain)** |
| **40** | **Correct LP evaluation (7 models, filtered MRR)** | ⏳ Running | 200-epoch: DELTA MRR 0.497 vs GraphGPS 0.513. 500-epoch in progress |
| 41 | Component ablation on real FB15k-237 | 🔲 Planned | |
| 42 | Multi-hop path queries (1p/2p/3p) | 🔲 Planned | |
| 43 | YAGO3-10 benchmark (123K entities) | 🔲 Planned | |
| 44 | Scaling analysis (500→123K entities) | 🔲 Planned | |
| 45 | Interpretability (EdgeAttention top-k + t-SNE) | 🔲 Planned | |

---

## Recently Completed: Graph Construction Breakthrough

### Phase 38 — Differentiable Task-Aware Constructor

*(Experiment file: `experiments/phase46_differentiable_constructor.py`)*

Gumbel-sigmoid edge selection with straight-through estimators. 3 variants × 3 seeds.

**Result:** Hybrid constructor reaches **98% of FixedChain** (0.452 ± 0.006 vs 0.461 ± 0.034). Pure differentiable plateaus at ~85%.

### Phase 39 — Self-Bootstrapped DELTA

*(Experiment file: `experiments/phase46b_self_bootstrapped.py`)*

Replace transformer bootstrap with FixedChain DELTA layer. DELTA all the way down.

**Result:** **0.757 ± 0.041 (157% of FixedChain)**. Transformer scaffold fully removable.

### Phase 40 — Correct Link Prediction (In Progress)

*(Experiment file: `experiments/phase46c_link_prediction.py`)*

Rebuilt evaluation fixing all 5 Phase 37 leakage issues. 7 models, filtered MRR/Hits@K.

**200-epoch results:** DELTA-Matched MRR 0.497 vs GraphGPS 0.513 (DELTA still converging). 500-epoch convergence study running — GraphGPS shows overfitting (peaked at 0.530, declining).

---

## Phase II — Close Critical Proof Gaps

**Goal:** Prove that every architectural component contributes (pre-empt ablation reviewers), and establish multi-hop reasoning advantage.

### Phase 41 — Component Ablation on Real FB15k-237

**Script:** `experiments/phase41_component_ablation.py` *(to be written)*

**Ablation matrix** (5 seeds each, real FB15k-237):
| Config | Component removed | Expected drop |
|--------|------------------|--------------|
| Full DELTA-Matched | — (baseline) | — |
| No NodeAttention | Remove node parallel attention | > 2% drop |
| No EdgeAttention | Remove edge parallel attention | > 5% drop (edge-first thesis) |
| No ReconciliationBridge | Remove co-update layer | > 2% drop |
| No PostAttentionPruner | Remove pruning | ≈ 0% (accuracy) + inference time delta |
| 1-hop only | Remove 2-hop edge adjacency | > 3% drop (Phase 11 validated on synthetic) |

**Template:** `experiments/phase28_hard_ablation.py` (component ablation design) + `experiments/phase46c_link_prediction.py` (correct evaluation loop).

**Estimated runtime:** ~4–6h H100.

**Verification gate:** Every ablation hurts MRR on real FB15k-237.

---

## Phase III — Strengthen Novelty Claims

**Goal:** Multi-hop path reasoning + additional benchmark dataset. This answers: "does DELTA generalize beyond Freebase?"

### Phase 42 — Multi-hop Path Queries on Real FB15k-237

**Script:** `experiments/phase42_multihop_path_queries.py` *(to be written)*

**Protocol:** Standard BetaE/ConE 1p/2p/3p path query splits on FB15k-237.

**Required code change:**
- Add `load_path_queries(dataset_name)` to `delta/datasets.py`
- Path query files available alongside the main FB15k-237 split

**Baselines:** DELTA vs GraphGPS vs GRIT vs TransE (the classic multi-hop baseline).

**Connection to prior work:** Phase 11 (100% on derived 2-hop edge adjacency, synthetic) predicts DELTA's 2-hop architecture should excel on 2p/3p queries.

**Estimated runtime:** ~3–4h H100.

**Verification gate:** DELTA MRR ≥ GraphGPS on 2p and 3p query types.

---

### Phase 43 — YAGO3-10 Benchmark

**Script:** `experiments/phase43_yago3_benchmark.py` *(to be written)*

**Dataset:** YAGO3-10 — 123,182 entities, 37 relations, 1,079,040 training triples. Add to `delta/datasets.py`.

**Protocol:** Same evaluation protocol as Phase 40 (filtered MRR/Hits@K, train-only graph).

**Estimated runtime:** ~8–12h H100.

**Verification gate:** DELTA-Matched beats GraphGPS on YAGO3-10 (cross-family generalization — YAGO is Wikidata-sourced, not Freebase).

---

## Phase IV — Completeness

**Goal:** Scaling story + interpretability figure.

### Phase 44 — Scaling Analysis

**Script:** `experiments/phase44_scaling_analysis.py` *(to be written)*

**Protocol:** Train DELTA on graph size subsets and measure:
- Accuracy vs. entity count: 500 / 2K / 5K / 14.5K / 123K (via YAGO3-10 + NeighborSampler)
- Training time vs. entity count
- Memory usage vs. entity count

**Expected result:** O(E^x) with x < 2 (mini-batching breaks the quadratic edge adjacency wall). Reference: Phase 8 found O(n^0.81) on synthetic.

**Estimated runtime:** ~3–4h H100.

**Verification gate:** Plot shows sub-quadratic scaling; 123K-entity YAGO3-10 finishes without OOM.

---

### Phase 45 — Edge Attention Interpretability

**Script:** `experiments/phase45_interpretability.py` *(to be written)*

**Content:**
1. **Top-k attention heatmap** — for a known 2-hop reasoning chain on FB15k-237, show which edge pairs receive highest `EdgeAttention` weights
2. **t-SNE of edge embeddings** — FB15k-237 vs WN18RR embeddings from *same* frozen encoder (Phase 35 story: structural clustering independent of domain)
3. **Relation pair attention matrix** — 237×237 heatmap of average edge-to-edge attention, showing which relation pairs DELTA learns to compose

**Verification gate:** At least one interpretable attention pattern that matches known relational composition rules.

---

## Beyond Publication: The Brain

Phases 46+ move from "prove DELTA works" to "build toward The Brain." See [The Brain](the-brain.md) for the full vision.

- **Horizon 2 (Phases 46–55):** Iterative graph refinement, temporal reasoning, multi-scale construction
- **Horizon 3 (Phases 56+):** Multi-modal graph construction, associative memory, compositional generalization

---

## Paper Structure (Draft)

```
Title: DELTA: Edge-Centric Dual Attention for Relational Reasoning
       with Self-Bootstrapped Graph Construction

Abstract: 3 sentences — gap, method, result

1. Introduction
   - The three-paradigm gap (Transformers → GNNs → DELTA)
   - Edges as first-class computational citizens
   - Self-bootstrapped graph construction (Phase 39)
   - Summary of contributions

2. Related Work
   - Message-passing GNNs (CompGCN, RGCN, RotatE)
   - Transformers on graphs (GraphGPS, GRIT)
   - Knowledge graph completion (TransE, BetaE)

3. Architecture
   - DualParallelAttention (node + edge in parallel)
   - 2-hop edge adjacency construction
   - ReconciliationBridge (co-update)
   - Self-bootstrapped graph construction (DELTA bootstraps DELTA)

4. Experiments
   4.1 Setup (FB15k-237, WN18RR, YAGO3-10)
   4.2 Link prediction — Phase 40 flagship table
   4.3 Component ablation — Phase 41 table
   4.4 Multi-hop path queries — Phase 42 results
   4.5 Self-bootstrap vs transformer bootstrap — Phase 39 results

5. Domain Transfer
   - Phase 35: frozen encoder → 0.961 on WN18RR

6. Scaling + Interpretability
   - Phases 44–45

7. Conclusion + The Brain vision
```

### Results Tables to Compile

| Table | Phases | Models |
|-------|--------|--------|
| Main LP comparison | 40, 43 | DELTA / SelfBootstrap / GraphGPS / GRIT / DistMult |
| Ablation | 41 | Full / -NodeAttn / -EdgeAttn / -Reconcile / -Pruner / 1-hop |
| Multi-hop queries | 42 | DELTA / GraphGPS / GRIT / TransE on 1p / 2p / 3p |
| Bootstrap comparison | 38, 39 | Transformer / FixedChain / Hybrid / SelfBootstrap |
| Transfer | 35 | Probe accuracy vs sample count |

---

## Relevant Files

| File | Role |
|------|------|
| `experiments/phase46_differentiable_constructor.py` | Phase 38: Differentiable constructor (3 variants) |
| `experiments/phase46b_self_bootstrapped.py` | Phase 39: Self-bootstrapped DELTA |
| `experiments/phase46c_link_prediction.py` | Phase 40: Correct LP evaluation (7 models) |
| `experiments/phase35_relational_transfer.py` | GRL training template; NeighborSampler + real data |
| `experiments/phase37_real_comparison.py` | Phase 37 (invalidated): mini-batch training infrastructure |
| `experiments/phase28_hard_ablation.py` | Ablation template (synthetic) |
| `delta/datasets.py` | Dataset loading; `load_lp_data()` for Phase 40+ |
| `delta/attention.py` | `EdgeAttention`, `NodeAttention`, `ReconciliationBridge` |
| `delta/baselines.py` | `GraphGPSModel`, `GRITModel` |
| `mkdocs-src/COLAB_RESULTS.md` | Running results log |

---

## Verification Summary

| Gate | Target | Measured |
|------|--------|---------|
| Phase 35: frozen probe | > 0.5 on WN18RR | ✅ 0.961 |
| Phase 37: DELTA-Matched vs GraphGPS | DELTA > GraphGPS (real FB15k-237) | ⚠️ Invalidated (leakage) |
| Phase 38: differentiable ≥ 95% FixedChain | Hybrid ≥ 0.438 | ✅ 0.452 (98%) |
| Phase 39: self-bootstrap ≥ FixedChain | SelfBootstrap ≥ 0.481 | ✅ 0.757 (157%) |
| Phase 40: DELTA competitive on LP | MRR within 10% of GraphGPS | ✅ 0.497 vs 0.513 (97%) |
| Phase 41: each ablation hurts | All MRR drops > 0% | TBD |
| Phase 42: multi-hop | DELTA ≥ GraphGPS on 2p/3p | TBD |
| Phase 43: YAGO3-10 | DELTA > GraphGPS | TBD |
| Phase 44: scaling | O(E^x), x < 2 | TBD |
| Phase 45: interpretability | ≥1 interpretable attention pattern | TBD |

All publication-grade results: **5 seeds, mean ± std reported.** Phases 38–40 use 3 seeds for rapid iteration.

---

*Last updated: April 2026. Phase 39 (self-bootstrapped DELTA) validated: 0.757 ± 0.041, 157% of FixedChain — transformer scaffold removed. Phase 40 (correct LP evaluation) in progress: DELTA MRR 0.497 at 200 epochs, competitive with GraphGPS (0.513). See [The Brain](the-brain.md) for the long-term vision.*
