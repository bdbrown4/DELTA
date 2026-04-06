# DELTA: Publication Roadmap

**Target:** NeurIPS / ICLR (top-tier ML venue)
**Title placeholder:** *"DELTA: Edge-Centric Dual Attention for Relational Reasoning on Knowledge Graphs"*
**Status:** 48 experiment phases complete. Active temperature optimization research ongoing.

**Current evidence base:** 48 experiment phases, competitive link prediction on FB15k-237 subset (LP MRR 0.4856), multi-hop compositional reasoning dominance (3p MRR 0.742±0.009, only model improving with depth), per-query inference 0.8–0.9× GraphGPS, learnable temperature revealing edge/node asymmetry.

---

## Phase Completion Overview

| Phase | Experiment | Status | Key Result |
|-------|-----------|--------|------------|
| 34 | GraphGPS/GRIT vs DELTA — synthetic | ✅ | DELTA 0.880 vs GraphGPS 0.293 |
| 35 | Domain-agnostic relational transfer | ✅ | Probe 0.961 on WN18RR (frozen encoder) |
| 36 | Task-aware construction at scale | ✅ | Constructor adds ≤1.3% |
| 37 | FB15k-237 parameter-matched comparison | ⚠️ Invalidated | 5 leakage issues → replaced by Phase 40 |
| 38 | Differentiable task-aware constructor | ✅ | Hybrid 0.452 ± 0.006 (98% of FixedChain) |
| 39 | Self-bootstrapped DELTA | ✅ | 0.757 ± 0.041 (157% of FixedChain) |
| 40 | Correct LP evaluation (7 models) | ✅ | SBHybrid MRR 0.5089, H@10 0.8158 |
| 41 | Generalization gap investigation | ✅ | Negative result — gap caused by val-set noise |
| 42 | Multi-hop path queries (1p/2p/3p) | ✅ | DELTA-Matched 3p MRR 0.738, only model with 2p→3p improvement |
| 43 | DropEdge robustness (5 drop rates) | ✅ | Advantage holds at all drop rates |
| 44 | Extended depth (4p/5p) | ✅ | 5p MRR 0.790; advantage accelerates with depth |
| 45 | Inference timing + multi-seed | ✅ | Per-query 0.8–0.9× GraphGPS; 3-seed robust |
| 46 | Learnable attention temperature | ✅ | Dead heads 83%→38%; edge/node asymmetry discovered |
| 47 | Layer-specific temperature | ✅ | B (L0=1, L1+L2=4) best LP MRR 0.4783 |
| 48 | Asymmetric node/edge temperature | ✅ | E (node=2, edge=6) LP MRR **0.4856** (new record) |

---

## Verified Publication Claims

### Claim 1: Edge-centric dual attention excels at relational reasoning
- Phase 1/9/11/13: foundational synthetic evidence
- Phase 28: +24% over vanilla at extreme noise
- Phase 34: DELTA 0.880 vs GraphGPS 0.293 on edge classification

### Claim 2: DELTA is competitive on real link prediction
- Phase 40: SBHybrid MRR 0.5089 (within 0.004 of GraphGPS, beats on H@10)
- Phase 48: DELTA-Full LP MRR 0.4856 with asymmetric temperature

### Claim 3: DELTA dominates multi-hop compositional reasoning
- Phase 42: Only model with positive 2p→3p trajectory
- Phase 44: Advantage accelerates — 5p MRR 0.790 (+0.100 over GraphGPS)
- Phase 43/45: 3-seed robust (3p MRR 0.742±0.009), advantage holds at all DropEdge rates

### Claim 4: Inference cost is deployment-friendly
- Phase 45: Per-query scoring 0.8–0.9× GraphGPS (faster)
- Training 34× slower but one-time cost; encoding 52× slower but per-graph (amortized)

### Claim 5: Learnable temperature reveals attention asymmetry
- Phase 46: Edge temps drift UP, node temps drift DOWN — model discovers the distinction
- Phase 47: Selective sharpening (L0 soft + L1+L2 sharp) outperforms uniform
- Phase 48: Asymmetric init (node=2, edge=6) yields new LP record; node temps "set and forget"

---

## Remaining Gaps Before Submission

| Gap | Priority | Notes |
|-----|----------|-------|
| Full-scale FB15k-237 (14.5K entities) | High | Currently on 494-entity subset |
| YAGO3-10 / WN18RR benchmark | High | Cross-family generalization |
| LP/3p trade-off resolution | Medium | Phase 49 targets this |
| Scaling analysis (500→14.5K) | Medium | Sub-quadratic claim needs verification at scale |
| Interpretability figure | Medium | Edge attention heatmap for known reasoning chains |

---

## Paper Structure (Draft)

```
Title: DELTA: Edge-Centric Dual Attention for Relational Reasoning
       with Self-Bootstrapped Graph Construction

Abstract: 3 sentences — gap, method, result

1. Introduction
   - The three-paradigm gap (Transformers → GNNs → DELTA)
   - Edges as first-class computational citizens
   - Self-bootstrapped graph construction

2. Related Work
   - Message-passing GNNs (CompGCN, RGCN)
   - Transformers on graphs (GraphGPS, GRIT)
   - KG completion (TransE, RotatE, BetaE)

3. Architecture
   - DualParallelAttention (node + edge in parallel)
   - 2-hop edge adjacency construction
   - ReconciliationBridge (co-update)
   - Self-bootstrapped graph construction
   - Learnable per-head temperature with edge/node asymmetry

4. Experiments
   4.1 Setup (FB15k-237 subset, evaluation protocol)
   4.2 Link prediction — Phase 40 table + Phase 48 temperature table
   4.3 Multi-hop path queries — Phase 42/44 (1p–5p results)
   4.4 Robustness — Phase 43 (DropEdge), Phase 45 (multi-seed)
   4.5 Temperature analysis — Phases 46–48 (asymmetry discovery)

5. Inference Efficiency
   - Phase 45: per-query faster than GraphGPS

6. Self-Bootstrap Results
   - Phase 39: 157% of FixedChain

7. Conclusion + The Brain vision
```

---

*See [The Brain](the-brain.md) for the long-term vision beyond publication. See [Key Findings](key-findings.md) for all 29 findings.*

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
| Phase 40: DELTA competitive on LP | MRR within 10% of GraphGPS | ✅ SelfBootstrapHybrid 0.5089 vs GraphGPS 0.5126 (99.6%). Beats GraphGPS on H@10 |
| Phase 41: each ablation hurts | All MRR drops > 0% | TBD |
| Phase 42: multi-hop | DELTA ≥ GraphGPS on 2p/3p | TBD |
| Phase 43: YAGO3-10 | DELTA > GraphGPS | TBD |
| Phase 44: scaling | O(E^x), x < 2 | TBD |
| Phase 45: interpretability | ≥1 interpretable attention pattern | TBD |

All publication-grade results: **5 seeds, mean ± std reported.** Phases 38–40 use 3 seeds for rapid iteration.

---

*Last updated: April 2026. Phase 39 (self-bootstrapped DELTA) validated: 0.757 ± 0.041, 157% of FixedChain — transformer scaffold removed. Phase 40 (correct LP evaluation) complete: SelfBootstrapHybrid MRR 0.5089 on FB15k-237, within 0.004 of GraphGPS and beating it on Hits@10. Self-bootstrap advantage confirmed on real data. See [The Brain](the-brain.md) for the long-term vision.*
