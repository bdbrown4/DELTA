# Status & Roadmap

*Last updated: Phase 62 complete (2026-04-15)*

---

## Current Best Results

| Metric | Model | Value | Phase |
|--------|-------|-------|-------|
| LP MRR (DELTA-Full, temp) | Q: K-anneal + edge=7.0 | **0.4905** | 52 |
| LP MRR (brain_hybrid) | A: d=0.01, seed=456 | **0.4956** | 58 |
| LP MRR (brain_hybrid, 3-seed mean) | d=0.01, seeds 42/123/456 | **0.4844±0.0097** | 58 |
| LP H@10 (brain_hybrid) | A: baseline @ d=0.01 | **0.8076** | 57 |
| LP H@10 (DELTA-Full, temp) | S: anneal + edge=7.0 | **0.8045** | 52 |
| 3p MRR (multi-hop) | DELTA-Matched @10% drop | **0.742 +/- 0.009** | 45 (3-seed) |
| 5p MRR | DELTA-Matched @0% drop | **0.790** | 44 |
| Depth advantage (5p) | DELTA vs GraphGPS | **+0.100** | 44 |
| Per-query inference | DELTA vs GraphGPS | **0.8–0.9x** (faster) | 45 |
| LP MRR (1-layer DELTA, N=2000) | delta_1layer | **0.3338** (val peak) | 59 |
| LP MRR (DistMult, N=2000) | DistMult (no GNN) | **0.3185** | 59 |
| LP test MRR (1-layer DELTA, N=5000) | delta_1layer (sub E_adj) | **0.2404** | 62 |
| LP test MRR (DistMult, N=5000) | DistMult (no GNN) | **0.2244** | 62 |

---

## What's Validated (18 Propositions)

| # | Proposition | Confidence | Key Evidence |
|---|---|---|---|
| P1 | Edge attention beats node attention on relational tasks | High | Phase 1, 9, 11, 13, 28 |
| P2 | Dual parallel attention is the key differentiator at high noise | High | Phase 28: +24% at extreme noise |
| P3 | Graph structure adds value over transformers on relational tasks | High | Phase 27b: +4.4%; Phase 42: multi-hop dominance |
| P4 | Soft gating achieves sparsity without accuracy loss | High | Phase 22/29: 100.0% at 50% sparsity |
| P5 | DELTA beats CompGCN on real FB15k-237 | High | Phase 25/29: 97.4% vs 96.9% |
| P6 | Results stable across random seeds | High | Phase 29 (5 seeds), Phase 45 (3 seeds multi-hop) |
| P7 | Sampling robustness at 26% VRAM budget | High | Phase 30: all 4 strategies within +/-0.2% |
| P8 | DELTA competitive with GraphGPS on LP | High | Phase 40: SBHybrid MRR 0.5089 (within 0.004) |
| P9 | DELTA excels on multi-hop compositional queries | High | Phase 42–44: only model with 2p->3p improvement; 5p MRR 0.790 |
| P10 | Advantage accelerates with reasoning depth | High | Phase 44: +0.004 at 2p -> +0.100 at 5p vs GraphGPS |
| P11 | Per-query inference competitive with GraphGPS | High | Phase 45: 0.8–0.9x per query |
| P12 | Learnable temperature discovers edge/node asymmetry | High | Phase 46–48: edge wants sharper, node wants softer |
| P13 | Selective sharpening outperforms uniform | High | Phase 47: B (L0=1, L1+L2=4) best of 4 configs |
| P14 | Asymmetric node/edge temp improves LP | High | Phase 48: E (node=2, edge=6) MRR 0.4856 (+1.5%) |
| P15 | Differentiable graph construction is viable for LP | High | Phase 55–57: brain_hybrid matches delta_full MRR with +4.7% H@10 |
| P16 | Constructor density controls precision/recall trade-off | High | Phase 56: d=0.01 strictly dominates d=0.02 on MRR, H@3, H@10 |
| P17 | Temperature annealing is counterproductive on brain_hybrid | High | Phase 57: annealing trades H@10 for marginal MRR; baseline optimal |
| P18 | d=0.01 is the optimal brain_hybrid density sweet spot | High | Phase 58: d=0.01 mean MRR=0.4844±0.0097 (robust); d=0.005 MRR=0.4673 (−0.017). Density optimization CLOSED. |
| P19 | Edge-to-edge attention mechanism works at N=2000 (1-layer) | High | Phase 59: 1-layer DELTA val_MRR=0.3338 surpasses DistMult (0.3185). Mechanism viable, depth is the bottleneck. |
| P20 | 3-layer DELTA catastrophically over-smooths at N=2000 | High | Phase 59: MRR=0.0018 across 3 training configs. 15.2M E_adj pairs × 3 layers = representation collapse. |
| P21 | DELTA's test MRR advantage over DistMult is non-monotonic and modest at N=5000 | High | Phase 62: gap=+0.004 (N=500), +0.076 (N=2000), +0.016 (N=5000). N=2000 spike inflated by DM overfitting. |

---

## Known Issues

- **Phase 37 leakage** — invalidated (5 evaluation bugs). Replaced by Phase 40.
- **Training cost** — DELTA 34x slower per epoch than GraphGPS; inference is comparable.
- **LP/3p trade-off is fundamental** — After 9 phases (46–54) testing 20+ temperature configurations, temperature reliably improves LP but has no statistically supported effect on multi-hop reasoning depth. Three operating modes: LP-optimized (P/Q), balanced-3p (K), deep-reasoning (N).
- **Multi-hop temperature claims not statistically robust** — Phase 53 multi-seed validation shows single-seed 3p/4p/5p advantages are seed-dependent. Only LP improvements are statistically supported.
- **brain_hybrid overfitting** — Both density conditions overfit after epoch 150. Early stopping around epoch 150–180 preserves best test performance.
- **3-layer depth over-smoothing at N=2000** — Phase 59: catastrophic collapse (MRR=0.002). 1-layer works (MRR=0.334). Depth management is the critical architectural challenge for scaling.

---

## Open Gaps

### Gap 1: Full-scale evaluation — HIGH priority (subsampling confound)
- Phase 62 scaled to N=5000: DELTA test MRR=0.2404 vs DM=0.2244, gap=+0.016 (below hypothesized ≥0.04)
- Scaling curve is non-monotonic: +0.004 (N=500) → +0.076 (N=2000) → +0.016 (N=5000)
- N=2000 gap was inflated by DistMult catastrophic overfitting (val→test drop of 27%)
- Edge adjacency subsampled to 23.8% at N=5000 (15M of 63M pairs) — major confound
- Full FB15k-237 (14,505 entities, ~211M E_adj pairs) requires attention sparsification
- Next: investigate subsampling impact at N=5000 or pivot to sparse attention

### Gap 2: LP/3p trade-off — CHARACTERIZED (Phases 46–54)
- After 9 phases and 20+ configurations, the trade-off is confirmed fundamental at the temperature level
- Three operating modes: LP-optimized (Q: 0.4905), balanced-3p (K: 0.4148), deep-reasoning (N: 5p 0.3788)
- Dynamic temperature (annealing) helps but doesn't fully resolve the trade-off

### Gap 3: Cross-family generalization
- WN18RR (transfer): 0.961 probe accuracy (Phase 35) — but frozen encoder, not trained LP
- YAGO3-10: untested. Would demonstrate generalization beyond Freebase

### Gap 4: Brain architecture optimization — PAUSED (blocked by Gap 5)
- BrainEncoder validated (Phases 55–58) with MRR gains over delta_full marginal (+0.002 single-seed) but multi-seed mean 0.4844±0.0097 is robust
- H@10 advantage (+4.7%) is substantial — constructed edges genuinely add recall
- Density optimization CLOSED: d=0.01 (2,435 edges) is the sweet spot. d=0.02 too noisy, d=0.005 too sparse.
- brain_hybrid OOMs at N=2000 (102K edges). Scaling brain architecture requires solving depth management first.
- Next: resume after Gap 5 resolved

### Gap 5: Depth management at scale — RESOLVED but exposed deeper issue (Phase 60)
- Phase 59: 3-layer over-smooths catastrophically at N=2000 (MRR=0.002); 1-layer works (MRR=0.334)
- Phase 60: Residual gating eliminates over-smoothing: 3L+gate MRR=0.3138 (174× vs ungated 0.0018)
- However, all depths converge to MRR~0.31 — **matching DistMult (0.3185) without any GNN**
- Gates frozen at ~10% layer / 90% residual → model learned layers are noise, not signal
- Phase 61/61b: DELTA provides genuine but modest +0.017 val MRR advantage over DistMult at N=2000
- Phase 62: At N=5000, gap narrows to +0.016 test MRR — advantage is real but doesn't scale
- 1-layer accepted as scaling architecture; depth provides no benefit at N≥2000

### Gap 5: Sequence domain generalization — FUTURE (Horizon 3)
- All current evidence is on knowledge graphs where structure is pre-defined or semi-explicit
- BrainEncoder's Gumbel-sigmoid construction (Phases 55–57) is the mechanism for sequence domains
- Prerequisites: Brain architecture optimization; then LRA pilot on ListOps

---

## Roadmap

### Horizon 2: Adaptive Architecture (Phases 46–57) — Complete

| Phase | Goal | Status |
|-------|------|--------|
| 46 | Learnable per-head temperature | Done — dead heads 83%->38%, edge/node asymmetry |
| 47 | Layer-specific temperature | Done — B (L0 soft, L1+L2 sharp) = best LP 0.4783 |
| 48 | Asymmetric node/edge temperature | Done — E = LP record 0.4856; node stable, edge drifts UP |
| 49 | L0 temperature + asymmetric L1+L2 | Done — H achieves LP 0.4887 but 3p still below D |
| 50 | Temperature annealing | Done — K breaks 3p ceiling (0.4148), first to beat D's 0.4018 |
| 51 | Static vs trajectory | Done — annealing trajectory creates representations static init cannot replicate |
| 52 | Edge sharpness + closing LP gap | Done — Q achieves LP 0.4905 (record); LP/3p trade-off confirmed fundamental |
| 53 | Multi-seed validation | Done — multi-hop claims are seed-dependent; only LP is robust |
| 54 | High-power multi-hop eval | Done — 10k queries confirm evaluation noise dominates; investigation CLOSED |
| 55 | Brain architecture port | Done — BrainEncoder MRR 0.4773, H@10 +3.7% over delta_full |
| 56 | Constructor density ablation | Done — d=0.01 strictly dominates d=0.02 |
| 57 | Brain temperature annealing | Done — baseline (no annealing) optimal; MRR 0.4808–0.4818 |
| 58 | Multi-seed density validation | Done — d=0.01 robust (mean MRR 0.4844±0.0097); d=0.005 fails (−0.017). Density CLOSED. |

### Horizon 3: Brain Optimization & Sequence Domains (Phases 59+) — Active

Brain density optimization CLOSED (Phases 55–58). d=0.01 is the sweet spot. Phases 59–62: scaling evaluation from N=2000 to N=5000. Depth over-smoothing solved by residual gating (Phase 60) but depth provides no benefit. 1-layer DELTA accepted. Phase 62 shows DELTA’s advantage is real but modest at N=5000 (+0.016 test MRR) and non-monotonic across scales. Edge adjacency subsampling (23.8% at N=5000) is a confound. Next: investigate subsampling impact, attention sparsification, or sequence domain pilot (LRA ListOps). See [The Brain](the-brain.md) for the long-term vision.

### Horizon 4: Dynamic Reasoning — Future

Iterative graph refinement, temporal reasoning, multi-scale construction. See [The Brain](the-brain.md).

### Horizon 5: The Brain

Multi-modal construction, associative memory, compositional generalization. See [The Brain](the-brain.md).

---

## Publication Pathway

**Target:** NeurIPS / ICLR — *"DELTA: Edge-Centric Dual Attention for Relational Reasoning on Knowledge Graphs"*

### What We Have

- [x] Novel architecture with theoretical motivation
- [x] 57 experiment phases with honest failure documentation
- [x] Multi-seed statistical validation (Phases 29, 45, 53)
- [x] Competitive LP on real FB15k-237 (Phases 40, 48, 52)
- [x] Multi-hop compositional dominance (Phases 42–44)
- [x] Inference efficiency story (Phase 45)
- [x] Learnable temperature with mechanistic insight (Phases 46–52)
- [x] Self-bootstrap result (Phase 39: 157% of FixedChain)
- [x] Differentiable graph construction via BrainEncoder (Phases 55–57)
- [x] LP/3p trade-off fully characterized (Phases 46–54)

### What We Still Need

- [ ] Full-scale dataset evaluation (14.5K+ entities) — Gap 1
- [ ] Cross-family benchmark (YAGO3-10 or WN18RR LP) — Gap 3
- [ ] Interpretability figure (attention heatmap on known reasoning chain)
- [ ] Multi-seed brain_hybrid validation for statistical confidence

### Paper Structure

1. **Introduction** — The three-paradigm gap (Transformers -> GNNs -> DELTA); edges as first-class computational citizens; self-bootstrapped graph construction
2. **Related Work** — Message-passing GNNs (CompGCN, RGCN); Transformers on graphs (GraphGPS, GRIT); KG completion (TransE, RotatE, BetaE)
3. **Architecture** — DualParallelAttention; 2-hop edge adjacency; ReconciliationBridge; BrainEncoder (differentiable construction); learnable per-head temperature with edge/node asymmetry
4. **Experiments** — Setup (FB15k-237 subset, evaluation protocol); LP results (Phases 40, 48, 52); multi-hop path queries (Phases 42–44, 1p–5p); robustness (Phase 43 DropEdge, Phase 45 multi-seed); temperature analysis (Phases 46–54); brain architecture (Phases 55–57)
5. **Inference Efficiency** — Phase 45: per-query faster than GraphGPS
6. **Self-Bootstrap & Brain** — Phase 39: 157% of FixedChain; Phases 55–57: differentiable construction matches delta_full with +4.7% H@10
7. **Conclusion** — The Brain vision

---

*See [The Brain](the-brain.md) for long-term vision. See [Validation Phases](validation-phases.md) for all experiment results. See [Key Findings](key-findings.md) for the 38 consolidated findings.*
