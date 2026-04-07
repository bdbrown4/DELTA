# Status & Roadmap

*Last updated: Phase 49 active (April 6, 2026)*

---

## Current Best Results

| Metric | Model | Value | Phase |
|--------|-------|-------|-------|
| LP MRR (DELTA-Full, temp) | E: node=2, edge=6 | **0.4856** | 48 |
| LP H@10 (DELTA-Full, temp) | F: node=3, edge=5 | **0.8014** | 48 |
| 3p MRR (multi-hop) | DELTA-Matched @10% drop | **0.742 +/- 0.009** | 45 (3-seed) |
| 5p MRR | DELTA-Matched @0% drop | **0.790** | 44 |
| Depth advantage (5p) | DELTA vs GraphGPS | **+0.100** | 44 |
| Per-query inference | DELTA vs GraphGPS | **0.8-0.9x** (faster) | 45 |

---

## What's Validated (14 Propositions)

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
| P9 | DELTA excels on multi-hop compositional queries | High | Phase 42-44: only model with 2p->3p improvement; 5p MRR 0.790 |
| P10 | Advantage accelerates with reasoning depth | High | Phase 44: +0.004 at 2p -> +0.100 at 5p vs GraphGPS |
| P11 | Per-query inference competitive with GraphGPS | High | Phase 45: 0.8-0.9x per query |
| P12 | Learnable temperature discovers edge/node asymmetry | High | Phase 46-48: edge wants sharper, node wants softer |
| P13 | Selective sharpening outperforms uniform | High | Phase 47: B (L0=1, L1+L2=4) best of 4 configs |
| P14 | Asymmetric node/edge temp improves LP | High | Phase 48: E (node=2, edge=6) MRR 0.4856 (+1.5%) |

---

## Known Issues

- **Phase 37 leakage** -- invalidated (5 evaluation bugs). Replaced by Phase 40.
- **Training cost** -- DELTA 34x slower per epoch than GraphGPS; inference is comparable.
- **3p gap** -- D (all temp=4.0) achieves best 3p MRR but L0 temperature contribution unresolved.
- **Constructor gradient wall** -- `constructor.py` line 127 uses a hard binary threshold (`attn > 0.1`) to create edges. No gradient flows through this decision -- the constructor cannot learn which edges help the downstream task. The fix (Gumbel-sigmoid, same as Phase 38) exists in the codebase and needs porting to the constructor.
- **Dead `edge_type_weights`** -- `constructor.py` lines 173-175 compute `edge_type_weights` via softmax but never fold them into the returned `DeltaGraph`. The edge type classification head contributes zero to representation quality despite consuming parameters. One-line fix: concatenate `edge_type_weights` into `edge_features` before returning.

---

## Open Gaps

### Gap 1: Full-scale evaluation -- HIGH priority
- Current: 494-entity FB15k-237 subset (3.4% of full dataset)
- Needed: Full FB15k-237 (14,505 entities) or YAGO3-10 (123K entities)
- Requires mini-batching infrastructure for 2-hop edge adjacency at scale

### Gap 2: LP/3p trade-off -- ACTIVE (Phase 49)
- Best LP config (E: node=2, edge=6, L0=1.0) and best 3p config (D: all=4.0) diverge
- Hypothesis: D's L0=4.0 may be key to 3p advantage. Phase 49 tests H (L0=4,4 + E's L1+L2), I (L0=4,4 + F's), J (L0=2,4 + E's)

### Gap 3: Cross-family generalization
- WN18RR (transfer): 0.961 probe accuracy (Phase 35) -- but frozen encoder, not trained LP
- YAGO3-10: untested. Would demonstrate generalization beyond Freebase

### Gap 4: Constructor gradient wall -- BLOCKING for sequence domains
- `constructor.py:127` uses a hard threshold (`attn > 0.1`) to select edges -- no gradient flows through this decision
- The constructor cannot learn to build task-optimal graphs because the task loss can't reach the edge-selection step
- Fix: Gumbel-sigmoid differentiable edge selection (mechanism already validated in Phase 38)
- Priority: apply to KG LP first to confirm no regression, then enable sequence domain work

### Gap 5: Dead `edge_type_weights` in constructor -- LOW effort fix
- `constructor.py:173-175` computes edge type weights via softmax but discards the result before returning the `DeltaGraph`
- The edge type classification head contributes zero to representation quality
- Fix: fold `edge_type_weights` into `edge_features` (one-line change, backward-compatible)

### Gap 6: Sequence domain generalization -- FUTURE (Horizon 3)
- All current evidence is on knowledge graphs where structure is pre-defined or semi-explicit
- LRA (Long Range Arena) requires discovering structure from raw sequences -- this is what Phase 33/36 failed on at small scale
- The SelfBootstrap result (157% of FixedChain, Phase 39) is the closest precedent: DELTA learning its own structure
- Prerequisites: Gaps 4 and 5 fixed; then LRA pilot on ListOps (hierarchical numerical expressions map naturally to DELTA's edge-first reasoning)

---

## Roadmap

### Horizon 2: Adaptive Architecture (Phases 46-50) -- Active

| Phase | Goal | Status |
|-------|------|--------|
| 46 | Learnable per-head temperature | Done -- Dead heads 83%->38%, edge/node asymmetry |
| 47 | Layer-specific temperature | Done -- B (L0 soft, L1+L2 sharp) = best LP 0.4783 |
| 48 | Asymmetric node/edge temperature | Done -- E = new LP record 0.4856; node stable, edge drifts UP |
| 49 | L0 temperature + asymmetric L1+L2 | Active -- H (L0=4,4 + E's L1+L2), I (L0=4,4 + F's L1+L2), J (L0=2,4 + E's L1+L2); target: LP >= 0.4856 and 3p >= 0.4018 |
| 50 | Constructor fixes + differentiable edge selection | Planned -- fix gradient wall (Gap 4), dead edge_type_weights (Gap 5), then port Gumbel-sigmoid selector |

### Horizon 3: Sequence Domain Generalization (Phases 51-60) -- Planned

Differentiable edge construction for sequence inputs (LRA pilot: ListOps first), position-preserving edges, task-conditioned construction. The jump from KG reasoning to sequence domains requires the constructor fixes in Phase 50 -- see [Architecture Overview](architecture.md#graph-constructor) for details.

### Horizon 4: Dynamic Reasoning (Phases 61-70) -- Future

Iterative graph refinement, temporal reasoning, multi-scale construction. See [The Brain](the-brain.md).

### Horizon 5: The Brain (Phases 71+)

Multi-modal construction, associative memory, compositional generalization. See [The Brain](the-brain.md).

---

## Publication Pathway

**Target:** NeurIPS / ICLR -- *"DELTA: Edge-Centric Dual Attention for Relational Reasoning on Knowledge Graphs"*

### What We Have

- [x] Novel architecture with theoretical motivation
- [x] 48 experiment phases with honest failure documentation
- [x] Multi-seed statistical validation (Phases 29, 45)
- [x] Competitive LP on real FB15k-237 (Phase 40, 48)
- [x] Multi-hop compositional dominance (Phases 42-44)
- [x] Inference efficiency story (Phase 45)
- [x] Learnable temperature with mechanistic insight (Phases 46-48)
- [x] Self-bootstrap result (Phase 39: 157% of FixedChain)

### What We Still Need

- [ ] Full-scale dataset evaluation (14.5K+ entities) -- Gap 1
- [ ] Cross-family benchmark (YAGO3-10 or WN18RR LP) -- Gap 3
- [ ] LP/3p trade-off resolution or clear characterization -- Gap 2, Phase 49 active
- [ ] Interpretability figure (attention heatmap on known reasoning chain)
- [ ] Constructor gradient wall fix -- Gap 4, needed before sequence domain work
- [ ] Dead `edge_type_weights` fix -- Gap 5, low effort, high correctness value

### Paper Structure

1. **Introduction** -- The three-paradigm gap (Transformers -> GNNs -> DELTA); edges as first-class computational citizens; self-bootstrapped graph construction
2. **Related Work** -- Message-passing GNNs (CompGCN, RGCN); Transformers on graphs (GraphGPS, GRIT); KG completion (TransE, RotatE, BetaE)
3. **Architecture** -- DualParallelAttention; 2-hop edge adjacency; ReconciliationBridge; self-bootstrapped construction; learnable per-head temperature with edge/node asymmetry
4. **Experiments** -- Setup (FB15k-237 subset, evaluation protocol); LP results (Phase 40 + Phase 48 temperature); multi-hop path queries (Phase 42/44, 1p-5p); robustness (Phase 43 DropEdge, Phase 45 multi-seed); temperature analysis (Phases 46-48)
5. **Inference Efficiency** -- Phase 45: per-query faster than GraphGPS
6. **Self-Bootstrap Results** -- Phase 39: 157% of FixedChain
7. **Conclusion** -- The Brain vision

---

*See [The Brain](the-brain.md) for long-term vision. See [Validation Phases](validation-phases.md) for all experiment results. See [Key Findings](key-findings.md) for the 29 consolidated findings.*
