# Current Status & Roadmap

---

## What's Working (Validated)

### Core Architecture (Phases 1–30)

- **Edge-first dual attention** — consistently outperforms node-only approaches on relational tasks
- **Multi-hop edge adjacency** — 100% on derived relations, sparse and scalable (O(E^0.97))
- **Post-attention soft gating** — 100% accuracy at 50% sparsity, +14.7% over pre-attention routing
- **BFS partitioning** — O(N^0.99), balanced with importance-aware seeding
- **Variational memory** — preserves accuracy, KL converges, threshold learnable
- **Graph structure adds value** — Phase 27b: FixedChain DELTA 40.7% > Transformer 36.3% on 2-hop paths

### Scale & Baselines (Phases 31–37)

- **Mini-batching scales to full FB15k-237** — 14,505 entities, 304K edges, subgraph sampling on H100
- **DELTA dominates synthetic relational tasks** — Phase 34: +57% over GraphGPS on edge classification, +29% on noise robustness
- **Domain-agnostic transfer** — Phase 35: frozen encoder → 0.961 on WN18RR with 100 samples
- **44/44 unit tests passing**, backward compatibility confirmed

### Graph Construction Breakthrough (Phases 38–39)

- **Differentiable construction works** — Phase 38: Hybrid Gumbel-sigmoid reaches 98% of FixedChain (0.452 ± 0.006)
- **Self-bootstrapped DELTA** — Phase 39: **0.757 ± 0.041, 157% of FixedChain**. No transformer needed. DELTA bootstraps DELTA
- **Correct link prediction** — Phase 40 complete: SelfBootstrapHybrid MRR 0.5089 (H@10 0.8158), within 0.004 of GraphGPS (0.5126) and beating it on Hits@10. DELTA-Matched 0.4950 MRR with 69% of GraphGPS's parameters

### Compositional Reasoning (Phase 42 — preliminary)

- **DELTA-Matched dominates multi-hop** — Phase 42: 3p MRR **0.738**, beating GraphGPS (0.697) by +0.041. The only model out of 7 that *improves* from 2p→3p. Edge-to-edge attention with 2-hop adjacency composes without information loss.
- **Capacity sweet spot** — DELTA-Matched (158K params) beats DELTA-Full (293K) at 3p by +0.046. Constrained capacity forces generalizable relational representations.

---

## Known Issues

### Phase 37 Leakage (Invalidated)

Phase 37's reported accuracy (0.991–0.994) was invalidated after a systematic audit found 5 critical evaluation issues: edge features encoding the answer, wrong metric, test edges in training graph, no target masking, no negatives. **Scale validation remains valid.** Phase 40 replaces Phase 37 with correct evaluation. See [Validation Phases](validation-phases.md#phase-37-leakage-audit).

### Open Architectural Questions

- **Self-bootstrap advantage on real data** — Confirmed. SelfBootstrapHybrid is the best DELTA variant on FB15k-237 LP (MRR 0.5089), beating all vanilla DELTA architectures
- **Speed gap** — DELTA 43-100× slower per epoch than GraphGPS. Competitive at equal epochs but needs optimization for wall-clock parity
- **Parameter count** — DELTA uses ~2× more parameters than GraphGPS/GRIT. Phase 40 includes parameter-matched variants

---

## Roadmap

### Recently Completed

| Phase | Goal | Result |
|-------|------|--------|
| **38** | Differentiable task-aware constructor (3 variants) | Hybrid 98% of FixedChain (0.452 ± 0.006) |
| **39** | Self-bootstrapped DELTA | **0.757 ± 0.041** — 157% of FixedChain |
| **40** | Correct LP evaluation on FB15k-237 (7 models, filtered MRR) | SelfBootstrapHybrid MRR 0.5089, beats GraphGPS on H@10 |
| **41** | Generalization gap investigation — weight decay sweep | Negative result — gap is val-set noise, not overfitting |
| **42** | Multi-hop path queries (1p/2p/3p) — all 7 models | DELTA-Matched 3p MRR **0.738** — only model to improve 2p→3p. Beats GraphGPS by +0.041 |

### In Progress

| Phase | Goal | Status |
|-------|------|--------|
| — | *No active experiments* | See Horizon 1 below |

### Horizon 1: Prove the Core (Phases 43–45)

| Phase | Goal | Priority |
|-------|------|----------|
| **43** | DropEdge regularization — can GNN overfitting reduction push peak MRR higher? | **High** |
| **44** | YAGO3-10 benchmark (123K entities) | Medium |
| **45** | Interpretability (attention visualization, edge type analysis) | Medium |

### Horizon 2: Dynamic Reasoning (Phases 46–55)

Iterative graph refinement, temporal reasoning, multi-scale construction, online learning. See [The Brain](the-brain.md) for details.

### Horizon 3: The Brain (Phases 56+)

Multi-modal graph construction, associative memory, compositional generalization, autonomous structure discovery. See [The Brain](the-brain.md).

---

*See [The Brain](the-brain.md) for the long-term vision. See [Publication Roadmap](PUBLICATION_ROADMAP.md) for the path toward publication.*
