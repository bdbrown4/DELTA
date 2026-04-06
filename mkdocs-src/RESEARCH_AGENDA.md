# DELTA Research Agenda

Research propositions, validated claims, and remaining gaps — updated through Phase 48.

---

## Validated Propositions

| # | Proposition | Status | Confidence | Key Evidence |
|---|---|---|---|---|
| P1 | Edge attention beats node attention on relational tasks | ✅ | High | Phase 1, 9, 11, 13, 28 |
| P2 | Dual parallel attention is the key differentiator at high noise | ✅ | High | Phase 28: +24% at extreme noise |
| P3 | Graph structure adds value over transformers on relational tasks | ✅ | High | Phase 27b: +4.4%; Phase 42: multi-hop dominance |
| P4 | Soft gating achieves sparsity without accuracy loss | ✅ | High | Phase 22/29: 100.0% ± 0.0% at 50% sparsity |
| P5 | DELTA beats CompGCN on real FB15k-237 | ✅ | High | Phase 25/29: 97.4% ± 0.1% vs 96.9% ± 0.3% |
| P6 | Results stable across random seeds | ✅ | High | Phase 29 (5 seeds), Phase 45 (3 seeds multi-hop) |
| P7 | Sampling robustness at 26% VRAM budget | ✅ | High | Phase 30: all 4 strategies within ±0.2% |
| P8 | DELTA competitive with GraphGPS on LP | ✅ | High | Phase 40: SBHybrid MRR 0.5089 (within 0.004) |
| P9 | DELTA excels on multi-hop compositional queries | ✅ | High | Phase 42–44: only model with 2p→3p improvement; 5p MRR 0.790 |
| P10 | Advantage accelerates with reasoning depth | ✅ | High | Phase 44: +0.004 at 2p → +0.100 at 5p vs GraphGPS |
| P11 | Per-query inference competitive with GraphGPS | ✅ | High | Phase 45: 0.8–0.9× per query |
| P12 | Learnable temperature discovers edge/node asymmetry | ✅ | High | Phase 46–48: edge wants sharper, node wants softer |
| P13 | Selective sharpening outperforms uniform | ✅ | High | Phase 47: B (L0=1, L1+L2=4) best of 4 configs |
| P14 | Asymmetric node/edge temp improves LP | ✅ | High | Phase 48: E (node=2, edge=6) MRR 0.4856 (+1.5%) |

---

## Open Gaps

### Gap 1: Full-scale evaluation — HIGH priority
- Current: 494-entity FB15k-237 subset (3.4% of full dataset)
- Needed: Full FB15k-237 (14,505 entities) or YAGO3-10 (123K entities)
- Requires mini-batching infrastructure for 2-hop edge adjacency at scale

### Gap 2: LP/3p trade-off — ACTIVE (Phase 49)
- Best LP config (E: node=2, edge=6, L0=1.0) and best 3p config (D: all=4.0) diverge
- Hypothesis: D's L0=4.0 may be key to 3p advantage. Phase 49 will test L0 temperature combined with E's asymmetric L1+L2

### Gap 3: Cross-family generalization
- WN18RR (transfer): 0.961 probe accuracy (Phase 35) — but frozen encoder, not trained LP
- YAGO3-10: untested. Would demonstrate generalization beyond Freebase

---

## Publication Pathway

### Already Have ✅
- [x] Novel architecture with theoretical motivation
- [x] 48 experiment phases with honest failure documentation
- [x] Multi-seed statistical validation (Phases 29, 45)
- [x] Competitive LP on real FB15k-237 (Phase 40, 48)
- [x] Multi-hop compositional dominance (Phases 42–44)
- [x] Inference efficiency story (Phase 45)
- [x] Learnable temperature with mechanistic insight (Phases 46–48)
- [x] Self-bootstrap result (Phase 39: 157% of FixedChain)

### Still Needs ❌
- [ ] Full-scale dataset evaluation (14.5K+ entities)
- [ ] Cross-family benchmark (YAGO3-10 or WN18RR LP)
- [ ] LP/3p trade-off resolution or clear characterization
- [ ] Interpretability figure (attention heatmap on known reasoning chain)

---

## Next Phases

| Phase | Goal | Priority |
|-------|------|----------|
| **49** | L0 temperature + asymmetric L1+L2 (close LP/3p gap) | 🔴 Active |
| 50 | Multi-scale adaptive architecture | 🟡 Next |
| 51+ | Full-scale FB15k-237 or YAGO3-10 | 🔴 High (publication-blocking) |

---

*See [Publication Roadmap](PUBLICATION_ROADMAP.md) for paper structure. See [The Brain](the-brain.md) for long-term vision.*

> "DELTA's clearest differentiation isn't raw accuracy — it's compositional relational reasoning with efficiency via sparsity."

Full review text available on request.

---

*Last updated: March 26, 2026 — Phases 31–34 experiment scripts ready (mini-batching, cross-graph transfer, task-aware construction, GraphGPS/GRIT baselines). Colab Pro+ setup guide available.*
