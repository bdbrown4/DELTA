# Validation Phases

All experiment phases with results. Phases 1–30 validated core architecture and fixes. Phases 31–37 scaled to real-world data and baselines.

---

## Phase 1–15: Core Architecture Validation

| Phase | Validates | Task | Result |
|-------|-----------|------|--------|
| 1 | Edge-to-edge attention discovers relational patterns | Edge attention vs node attention vs MLP | Edge 100%, Node 26.7%, MLP 100% |
| 2 | Dual parallel attention outperforms sequential | Sequential vs dual (1/2 layers) | All 100%; dual converges 2.7x faster |
| 3 | Importance router enables sparse attention | Accuracy vs sparsity (100% → 20%) | 100% accuracy at 80% sparsity |
| 4 | Tiered memory maintains recall | Sequential recall task | Perfect recall maintained |
| 5 | Transformer-bootstrapped graph construction | Transformer alone vs transformer→DELTA (150 epochs each) | TF: 98.3%, DELTA: 98.3% — pipeline preserves accuracy (non-relational task) |
| 6 | Full DELTA integration | All components end-to-end | All 4 sub-tests PASS |
| 7 | Gumbel-softmax differentiable routing | Hard top-k vs Gumbel-softmax | 12/12 router params get gradients (vs 0/12) |
| 8 | Scaling behavior | 20 → 400 nodes, time vs accuracy | O(n^0.81) sub-linear scaling |
| 9 | Multi-hop relational reasoning | Knowledge graph with derived relations | DELTA: 90.6%, Node GNN: 87.5%, MLP: 37.5% |
| 10 | Analogical reasoning | "A is to B as C is to ?" | Edge classification 100%; analogy retrieval needs contrastive training |
| 11 | Multi-hop edge adjacency | 1-hop vs 2-hop on transitive relations | **2-hop: 100% derived** (vs 1-hop 61.1%, Node GNN 83.3%) |
| 12 | Gumbel curriculum routing | Dense→sparse curriculum vs fixed | Gradient flow confirmed (12/12); curriculum needs larger scale |
| 13 | Compositional benchmarks | Logical rule-derived relations (7 types) | **DELTA 100%** all relations (vs Node GNN 87.5% derived) |
| 14 | Contrastive analogy training | Classification vs contrastive vs joint | All methods 100% retrieval — edge attention is inherently discriminative |
| 15 | Synthetic KG benchmark | 100 entities, 10 relations, 500 triples | Node GNN, DELTA 1-hop, DELTA 2-hop all reach 100%; router at 50% sparsity degrades to 65.3% |

---

## Phase 16–21: Architectural Fix Benchmarks

After Phase 15, a pitfall analysis identified 6 architectural weaknesses. Each was fixed and validated:

### The Six Fixes

| Fix | Problem | Solution | Phase |
|-----|---------|----------|-------|
| 1 | Router scores elements before seeing attention (chicken-and-egg) | PostAttentionPruner: prune based on *observed* attention weights | 16 |
| 2 | Spectral partitioning is O(N³) — won't scale | BFS seed-expansion partitioner in O(N+E) | 20 |
| 3 | Fixed linear memory compression loses information | Variational bottleneck with KL regularization | 18 |
| 4 | Single averaged edge projection ignores per-layer structure | Per-layer edge projections + edge combiner + active edge_type_head | 19 |
| 5 | Dense O(E²) multi-hop adjacency times out at ~500 edges | Sparse COO tensor operations | 17 |
| 6 | Uniform dropout doesn't distinguish structural vs noisy edges | LearnedAttentionDropout: per-edge dropout conditioned on features | 21 |

### Fix Validation Results

| Phase | Validates | Benchmark | Result |
|-------|-----------|-----------|--------|
| 16 | Post-attention soft gating + curriculum | KG classification at 50% sparsity | **Soft gating 100%, curriculum 100%**, old router 85.3%, hard post-attn 65.3%. Soft differentiable gating eliminates the original 29% gap and beats pre-attention routing by +14.7% |
| 17 | Sparse COO multi-hop scaling | 1-hop/2-hop timing at 50→2500 edges | **O(E^0.97) scaling confirmed.** 2500-edge 2-hop in 0.18s (was timeout with dense). All correctness checks pass |
| 18 | Variational memory compression | Compression quality + downstream accuracy | Accuracy preserved at 100% with compression. KL converges 0.126→0.026. Gradient flows through bottleneck. Similarity threshold is learnable |
| 19 | Per-layer edge projections | Edge type diversity + relational classification | Both reach 100% accuracy. Per-layer produces higher edge-type entropy (1.632 vs 1.562) — richer type diversity |
| 20 | BFS partition scaling | Wall-clock time at 50→2500 nodes | **O(N^0.99) scaling.** 2.0ms→90.8ms. Balance ratio 0.79. Importance-based seed spread: 100% |
| 21 | Learned attention dropout | Generalization gap: no dropout vs uniform vs learned | All modes reach 0 gap on current dataset (saturated). Eval-time passthrough confirmed. Needs harder benchmark |

---

## Phase 22–25: Scale & Integration Validation

| Phase | Validates | Benchmark | Result |
|-------|-----------|-----------|--------|
| 22 | Soft gating holds at 10× scale with noise | N=1000, 15 relations, 5000 triples, 15% label noise | **Soft gating 100%, old router 81.6%** (+18.4% gap). Zero generalization gap. |
| 23 | DELTA vs KG embedding baselines (TransE, RotatE, CompGCN) | FB15k-237-like: 2000 entities, 20 typed relations, 8000 triples | **DELTA 100%, CompGCN 100%**, TransE 67.6%, RotatE 70.7%. LP: TransE Hits@10=0.020, RotatE 0.010 (4×/2× random; sparse synthetic data). Soft gating maintains accuracy at 50% sparsity |
| 24 | All fixes integrated at scale | N=1000, 15% noise, full pipeline + ablations | All fixes integrate cleanly — zero degradation. 1-hop ablation runs 10× faster (44s vs 490s) |
| 25 | DELTA on **real** FB15k-237 (GPU) | Actual Freebase triples: 2000-entity dense subgraph, 69,626 edges, 210 relation types, RTX 3080 Ti | **DELTA+Gate 97.6%, CompGCN 97.2%**, TransE 78.8%, RotatE 77.8%. LP: TransE Hits@10=0.480, RotatE 0.335 (vs random 0.005). First real-data benchmark on GPU. |

*Phases 22, 23, and 25 were replicated with 5 seeds each in Phase 29 — see Phase 26–30 table for multi-seed statistics.*

---

## Phase 26–30 + 27b: Near-Term Roadmap Validation

| Phase | Validates | Benchmark | Result |
|-------|-----------|-----------|--------|
| 26 | Adaptive multi-hop: learn when to use 1-hop vs 2-hop | N=200, 3 models (FixedHop, AdaptiveHop, AdaptiveHopGating) | All models 100% at N=200. AdaptiveHopGate learns α→0.019 (suppresses 2-hop). Task saturates — validates cost-efficiency selection architecture. |
| 27 | *(initial — broken training)* Bootstrap on 2-hop path composition | N=500, batch=1, 200 epochs, no scheduler | TF 32.7%, Bootstrap DELTA 8.0%, Fixed Chain 5.3%. **Results confounded by batch-1 training** — see Phase 27b. |
| **27b** | **Corrected** bootstrap evaluation (gradient accum + 2× data) | N=1000, accum=32, 100 epochs, LR scheduler | **Fixed Chain DELTA 40.7% > Transformer 36.3% > Bootstrap 34.3%.** Graph structure helps (+4.4% over Transformer). GraphConstructor attention-thresholding discards path ordering — the constructor is the bottleneck, not graph processing. |
| 28 | Hard ablation: find difficulty threshold where DELTA advantages emerge | 4 levels (Easy/Medium/Hard/Extreme) × 3 models (Vanilla EdgeAttn, DualAttn, DELTA+Gate) | Extreme: **Dual Attention 64.2% >> Vanilla 40.2%** (+24%). Node context is the key DELTA advantage at high noise. Soft gating adds ±0.6% beyond dual attention at extreme difficulty. |
| 29 | Multi-seed statistical credibility | Phases 22, 23, 25 re-run with 5 seeds each | DELTA+Gate **97.4% ± 0.1%** on FB15k-237. Soft Gate 100.0% ± 0.0% vs Old Router 79.7% ± 1.1%. All results statistically stable. Total: 3.6 minutes. |
| 30 | GPU edge adj sampling strategy vs random | Uniform, degree-weighted, stratified, importance-weighted sampling at 26% budget on FB15k-237 | All 4 strategies within ±0.2% (~97.5%). Random sampling is sufficient at this graph density. |

---

## Phase 31–34: H100 / RTX PRO 6000 Experiments

| Phase | Validates | Benchmark | Result |
|-------|-----------|-----------|--------|
| 31 | Mini-batching scales to full FB15k-237 | Full FB15k-237: 14,505 entities, 304K edges, 20 relations | 100% test accuracy, converges by epoch 10. Mini-batch subgraph sampling + gradient accumulation confirmed on H100 80GB. |
| 32 | Cross-graph transfer | Train FB15k-237, eval WN18RR | Source 1.000. **Zero-shot: 0.048 (≈ random 0.050)** — features are domain-specific. Fine-tuned: 1.000 — pre-training helps with adaptation. Early stopping patience=10 reduces runtime from 24h → 3-4h. |
| 33 | Task-aware construction | Fixed topology vs augmented with learned edges | Fixed 0.347 ≈ Augmented 0.344 (5 seeds × 500 epochs). No improvement — 60-node tasks too small for learned edges to add value. |
| 34 | DELTA vs GraphGPS vs GRIT — synthetic | Edge classification, noise robustness, path composition (H100) | **DELTA dominates all three tasks.** Edge: DELTA 0.880 vs GraphGPS 0.293 vs GRIT 0.307 (+0.573). Noise@0.8: DELTA 1.000 vs GraphGPS 0.697 vs GRIT 0.710. Path: DELTA 1.000 vs GraphGPS 0.905 vs GRIT 0.893. All 5 seeds × 500 epochs. |

---

## Phase 35–37: Colab Pro+ Experiments

For detailed results, see [Colab Results](COLAB_RESULTS.md).

| Phase | Validates | Status | Headline Result |
|-------|-----------|--------|-----------------|
| 35 | Domain-agnostic relational transfer (GRL + linear probe) | ✅ Complete | Frozen encoder → **0.961 on WN18RR** with 100 samples. GRL unnecessary — encoder already domain-invariant. |
| 36 | Task-aware construction at scale (500–5000 nodes) | ✅ Complete | Constructor adds ≤1.3%. De-emphasize in paper. |
| 37 | Real FB15k-237 parameter-matched comparison (4 models × 5 seeds) | ⏳ Running | DELTA-Full seed 1: test 0.991. DELTA-Matched seeds 1-2: test 0.986/0.987. Remaining seeds in progress. |

---

## Next Steps (Phases 38–45)

See [Publication Roadmap](PUBLICATION_ROADMAP.md) for full details.

| Phase | Experiment | Status |
|-------|-----------|--------|
| 38 | Component ablation on real FB15k-237 | 🔲 Planned |
| 39 | Multi-hop path queries (1p/2p/3p) | 🔲 Planned |
| 40 | YAGO3-10 benchmark (123K entities) | 🔲 Planned |
| 41 | Codex-M benchmark (17K entities, 51 relations) | 🔲 Planned |
| 42 | Scaling analysis (500→123K entities) | 🔲 Planned |
| 43 | Interpretability (EdgeAttention top-k + t-SNE) | 🔲 Planned |
| 44 | ReasoningMesh (gated cross-attention between streams) | 🔲 Conditional — only if Phase 39 shows >15% 3p accuracy drop. Prototype evidence suggests this won't help. |
| 45 | Paper assembly and NeurIPS/ICLR submission | 🔲 Planned — depends on 38–43 (44 optional) |

---

*All publication-grade results use 5 seeds, mean ± std reported.*
