# Architecture Evolution

DELTA has gone through six development stages, each building on validated results from the previous stage.

---

## Stage 1: Core Validation (Phases 1–15)

Proved the thesis: edge-first dual attention outperforms node-only GNNs on compositional reasoning. Multi-hop edge adjacency (Phase 11) and compositional logic rules (Phase 13) were the headline results.

**Key milestones:**

- Edge-to-edge attention: 100% vs Node attention 26.7% (Phase 1)
- 2-hop edge adjacency: 100% on derived relations vs 61.1% 1-hop (Phase 11)
- DELTA 100% on all 7 relation types including derived (Phase 13)
- O(n^0.81) sub-linear scaling (Phase 8)

---

## Stage 2: Pitfall Analysis & Fixes (6 Fixes)

Identified and fixed 6 architectural weaknesses:

- **Routing paradigm shift**: Pre-attention prediction → post-attention pruning based on observed weights
- **Scaling fixes**: Dense O(E²) → sparse COO, spectral O(N³) → BFS O(N+E)
- **Memory upgrade**: Fixed compression → variational bottleneck with KL regularization
- **Constructor upgrade**: Averaged attention → per-layer edge projections
- **Regularization**: Uniform dropout → learned per-edge dropout

---

## Stage 3: Soft Gating Breakthrough (Phase 16 Redesign)

All fixes implemented and backward-compatible. Scaling bottlenecks (sparse multi-hop, BFS partition) are definitively solved. **Post-attention pruning is now fully validated** — the original 29% accuracy gap was caused by non-differentiable hard top-k gates, not a fundamental paradigm problem.

The redesigned soft sigmoid gating with per-head attention features achieves 100% accuracy at 50% target sparsity, and curriculum annealing (dense→sparse) also reaches 100%. Remaining awaiting-scale items (learned dropout, variational compression advantage) are architecturally sound but need harder benchmarks to demonstrate their value.

---

## Stage 4: Scale & Integration Validation (Phases 22–24)

Validated the full architecture at 10× scale. **Soft gating holds at N=1000 with 15% noise** (+18.4% over old router). DELTA matches CompGCN and crushes TransE/RotatE on a realistic KG benchmark with faithful baseline implementations:

- TransE: Bordes 2013 translation scoring
- RotatE: Sun 2019 complex rotation
- CompGCN: GRU message passing with relation composition

Link prediction evaluated with proper margin-based ranking training (separate from classification) — low absolute numbers (Hits@10 ~0.02) reflect data sparsity (~4 triples/entity) not broken methodology. All 5 inference-pipeline fixes integrate cleanly with zero degradation.

The remaining challenge: designing tasks hard enough to differentiate individual fix contributions — current synthetic benchmarks are solvable by vanilla EdgeAttention at this scale.

---

## Stage 5: Real-World Data on GPU (Phase 25)

**First benchmark on actual real-world data**, running on GPU. Phase 25 downloaded the real FB15k-237 dataset (14,505 entities, 237 relations, 310k Freebase triples) and evaluated on the 2000-entity dense subgraph (69,626 real triples, 210 relation types).

| Model | Accuracy |
|-------|----------|
| **DELTA+Gate** | **97.6%** |
| CompGCN | 97.2% |
| TransE | 78.8% |
| RotatE | 77.8% |

TransE link prediction Hits@10=0.480 (96× above random) confirms representations generalize on real data. GPU enablement required a CUDA-build PyTorch install and two library fixes (sparse tensor device propagation in `graph.py`, CPU tensor attribute in `router.py`).

---

## Stage 6: Near-Term Roadmap Validation + External Review (Phases 26–30, 27b)

Validated 5 roadmap items and stress-tested a core assumption with help from an external review.

**Phase 26** confirmed the adaptive hop-depth architecture: `AdaptiveHopGate` learns α→0.019 (suppressing the more expensive 2-hop when 1-hop suffices), validating cost-efficiency selection on a task that saturates at N=200.

**Phase 27b** (corrected from initial Phase 27) is the most significant finding of this stage. An external review identified that Phase 27's batch-1 training was severely handicapping DELTA models. With proper gradient accumulation and 2× data: Fixed Chain DELTA (40.7%) beat the pure Transformer (36.3%) — proving graph structure adds value on relational tasks. Bootstrap DELTA (34.3%) underperformed Fixed Chain due to the GraphConstructor discarding path-ordering edges. This also prompted a performance fix in `graph.py` (edge adjacency caching + vectorized incidence matrix for small graphs).

**Phase 28** found that at Extreme difficulty (noise=0.8), Dual Attention (+24% over Vanilla) is the key differentiator — node context matters when edge features are noisy. Soft gating adds only ±0.6% beyond dual attention here, confirming gating's value is efficiency rather than peak accuracy.

**Phase 29** confirmed all headline results are statistically stable across 5 seeds (DELTA+Gate 97.4% ± 0.1%, Soft Gate 100.0% ± 0.0%).

**Phase 30** confirmed random edge-adjacency sampling at 26% GPU budget is sufficient — all 4 sampling strategies (uniform, degree-weighted, stratified, importance-weighted) achieve within ±0.2% of each other on FB15k-237.

---

## Beyond Stage 6: H100 + Colab (Phases 31–37)

- **Phase 31:** Mini-batching scales to full FB15k-237 (14,505 entities). 100% test accuracy on H100.
- **Phase 32:** Cross-domain transfer — zero-shot fails (head mismatch), but fine-tuning recovers to 1.000.
- **Phase 33:** Task-aware construction — no improvement on 60-node tasks.
- **Phase 34:** DELTA dominates GraphGPS and GRIT on all three synthetic tasks (+57% edge classification).
- **Phase 35:** Frozen encoder achieves 0.961 on WN18RR with 100 samples. Encoder is already domain-invariant.
- **Phase 36:** Constructor adds ≤1.3% at scale. De-emphasized.
- **Phase 37:** Parameter-matched comparison on real FB15k-237 — in progress.

---

*See [Validation Phases](validation-phases.md) for complete tables, [Publication Roadmap](PUBLICATION_ROADMAP.md) for Phases 38–45.*
