# Current Status & Roadmap

---

## What's Working (Validated)

- **Edge-first dual attention** — consistently outperforms node-only approaches on relational tasks
- **Multi-hop edge adjacency** — 100% on derived relations, now sparse and scalable
- **BFS partitioning** — O(N^0.99) confirmed, balanced partitions with importance-aware seeding
- **Sparse COO operations** — O(E^0.97) confirmed, handles 2500+ edges where dense timed out
- **Variational memory** — preserves accuracy, KL converges, threshold is learnable
- **Per-layer edge projections** — higher type diversity, matching accuracy
- **Post-attention soft gating** — 100% accuracy at 50% target sparsity, beats pre-attention router (85.3%) by +14.7%. Soft differentiable gates with per-head attention features fully close the original 29% gap
- **Curriculum dense→sparse annealing** — temperature annealing (τ: 0.5→5.0) + sparsity ramp (0→50%) integrated with post-attention pruning, achieves 100% accuracy matching full attention
- **Graph structure adds value on relational tasks** — Phase 27b confirmed Fixed Chain DELTA (40.7%) beats pure Transformer (36.3%) on 2-hop path composition with proper training
- **Edge adjacency caching + vectorized incidence matrix** — `graph.py` fast path for E≤500 replaces Python for-loop, enabling efficient per-sample training for graph-based models
- **Mini-batching scales to full FB15k-237** — Phase 31 confirmed 14,505 entities / 304K edges trains to 100% with subgraph sampling on H100. Scale ceiling lifted from 2K→14.5K entities
- **DELTA dominates GraphGPS and GRIT on synthetic relational tasks** — Phase 34 showed +57% on edge classification, +29% on noise robustness, perfect multi-hop composition. 5 seeds × 500 epochs, zero variance at noise=0.8
- **Fine-tuned cross-domain transfer works** — Phase 32 fine-tuned transfer reaches 1.000, confirming pre-training helps with adaptation
- **44/44 unit tests passing**, backward compatibility confirmed

---

## Architecturally Sound, Awaiting Scale Proof

- **Learned attention dropout** — mechanisms confirmed (eval passthrough, rate diversity), but all benchmarks including N=1000 are too easy to show gap reduction
- **Variational compression advantage** — preserves accuracy but hasn't shown improvement over fixed compression yet
- **Ablation differentiation** — at N=1000, vanilla EdgeAttention also reaches 100%, so fix ablations show zero impact. The fixes provide *efficiency and robustness*, not accuracy gains on tasks a baseline can already solve

---

## Open Gaps

- **Zero-shot transfer fails (0.048 ≈ random)** — Phase 32 showed DELTA's edge-attention features are domain-specific at the head level: perfect source accuracy but chance-level zero-shot transfer. Phase 35 proved this was purely a head mismatch (237→11 classes), not encoder entanglement — frozen encoder + 100-sample probe → 0.961 on WN18RR.

- **Task-aware constructor doesn't improve over fixed topology** — Phase 33 showed augmented ≈ fixed on 60-node path composition. Phase 36 confirmed at scale (500–5000 nodes): max +1.3%. Constructor adds no measurable value.

- **All Phase 34 comparisons are synthetic** — DELTA dominates GraphGPS/GRIT on synthetic data but the comparison hasn't run on real FB15k-237. Phase 37 is the critical remaining validation.

- **DELTA uses 2× more parameters than baselines** — 60,594 params vs GraphGPS 33,388 / GRIT 28,130. Phase 37 includes a parameter-matched comparison (DELTA-Matched at ~30K params).

- **Soft gating marginal on extreme noise** — Phase 28 showed dual attention is the key differentiator at extreme difficulty (+24%); soft gating adds only ±0.6% beyond that. Gating's value remains efficiency, not peak accuracy on current benchmarks.

---

## Roadmap

### Active (Colab Pro+)

| Phase | Goal | Status |
|-------|------|--------|
| **37** | Real FB15k-237 parameter-matched: DELTA vs GraphGPS vs GRIT (4 models × 5 seeds) | ⏳ In progress |

### Planned (Phases 38–43)

| Phase | Goal | Priority |
|-------|------|----------|
| **38** | Component ablation on real FB15k-237 (5 components × 5 seeds) | 🔴 High |
| **39** | Multi-hop path queries (1p/2p/3p on FB15k-237) | 🔴 High |
| **40** | YAGO3-10 benchmark (123K entities) | 🟡 Medium |
| **41** | Codex-M benchmark (17K entities, 51 relations) | 🟡 Medium |
| **42** | Scaling analysis (500→123K entities, O(E^x) characterization) | 🟠 Medium |
| **43** | Edge attention interpretability (top-k + t-SNE) | 🟠 Medium |

### Conditional

| Phase | Goal | Trigger |
|-------|------|---------|
| **44** | ReasoningMesh (gated cross-attention between streams) | Only if Phase 39 shows >15% drop on 3p vs 1p queries. **Prototype evidence suggests this won't help** — cross-attention gates scored at majority baseline (0.218) vs ReconciliationBridge (0.889). |
### Paper Assembly

| Phase | Goal | Depends On |
|-------|------|------------|
| **45** | Paper assembly and NeurIPS/ICLR submission | Phases 38–43 (44 optional) |
### Long-Term

- **Replace transformer bootstrap** — Requires domain-agnostic transfer + useful constructor. Phase 35 solved the encoder side; constructor remains unhelpful.
- **Multi-modal input** — Extend graph constructor beyond token sequences (images, tables, structured data)
- **Real-world application** — Knowledge graph completion, drug-target interaction, or recommendation systems at production scale

---

*See [Publication Roadmap](PUBLICATION_ROADMAP.md) for the full path to NeurIPS/ICLR submission including per-phase verification gates.*
