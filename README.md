# DELTA: Dynamic Edge-and-Node Architecture with Layered, Tiered Attention

A research implementation of the DELTA architecture — a novel AI framework that operates on dynamic graphs with dual parallel attention across nodes and edges, tiered memory, and a learned importance router.

## Core Thesis

Reality is a graph. Language is a lossy compression of reality into sequences. Transformers reconstruct relational structure from flat sequences. DELTA operates on relational structure directly.

**The three-paradigm gap — visual explainer:** [ARCHITECTURE_VISUAL.md](./ARCHITECTURE_VISUAL.md) walks through Transformer → GNN → DELTA with an [interactive diagram](https://raw.githack.com/bdbrown4/DELTA/main/assets/transformer_vs_graph_vs_delta_standalone.html). The key insight: GNN edges are passive scalar wires; DELTA edges are first-class computational citizens that attend to each other. That edge-to-edge attention is what produces the Phase 28 +24% noise robustness gap.

## Architecture Overview

```
Raw Input (any modality)
    → Graph Constructor (transformer-bootstrapped: per-layer edge projections + typed edges)
    → BFS Graph Partitioner (O(N+E) seed-expansion clustering)
    → PARALLEL DUAL ATTENTION
        [Node Attention + Edge Attention across all partitions simultaneously]
    → Post-Attention Pruner (prune based on OBSERVED attention weights)
    → Learned Attention Dropout (per-edge regularization)
    → Reconciliation Layer (nodes and edges co-update)
    → Hierarchical Global Attention (cluster representatives)
    → Variational Memory Compression (warm nodes → bottleneck → KL regularization)
    → Memory Tier Update (importance-based hot/warm/cold promotion)
    → Output + Updated Graph State
```

## Research Agenda

Proposition tracking, open gaps, compute options, and the publication pathway are maintained in **[RESEARCH_AGENDA.md](./RESEARCH_AGENDA.md)** — including the external Phase 30 assessment and next steps.

## Project Structure

```
DELTA/
├── delta/                  # Core library
│   ├── graph.py            # Graph data structures + sparse COO multi-hop edge adjacency
│   ├── attention.py        # Node, edge, and dual parallel attention (return_weights support)
│   ├── router.py           # PostAttentionPruner + LearnedAttentionDropout + legacy ImportanceRouter
│   ├── memory.py           # Variational bottleneck tiered memory (hot/warm/cold)
│   ├── partition.py        # BFS seed-expansion graph partitioning (O(N+E))
│   ├── reconciliation.py   # Node-edge co-update
│   ├── constructor.py      # Transformer-based graph bootstrap (per-layer edge projections)
│   ├── model.py            # Full DELTA model (post-attention paradigm)
│   ├── baselines.py        # GraphGPS (2022) and GRIT (2023) baseline implementations
│   └── utils.py            # Helpers, synthetic data, and 9 benchmark generators
├── experiments/            # Phase-by-phase validation
│   ├── phase1_edge_attention.py       # Edge vs node attention
│   ├── phase2_dual_attention.py       # Sequential vs dual parallel
│   ├── phase3_router.py               # Sparsity vs accuracy
│   ├── phase4_memory.py               # Tiered memory recall
│   ├── phase5_construction.py         # Graph construction validation
│   ├── phase6_full_model.py           # End-to-end integration
│   ├── phase7_gumbel_routing.py       # Differentiable routing
│   ├── phase8_scaling.py              # Scaling analysis
│   ├── phase9_multi_hop.py            # Multi-hop reasoning
│   ├── phase10_analogy.py             # Analogical reasoning
│   ├── phase11_multi_hop_edges.py     # Multi-hop edge adjacency
│   ├── phase12_curriculum_routing.py  # Gumbel curriculum
│   ├── phase13_harder_benchmarks.py   # Compositional reasoning
│   ├── phase14_contrastive_analogy.py # Contrastive training
│   ├── phase15_kg_benchmark.py        # Synthetic KG benchmark
│   ├── phase16_post_attention_pruning.py  # [Fix 1+6] Post-attention pruning
│   ├── phase17_sparse_multi_hop.py        # [Fix 5] Sparse scaling benchmark
│   ├── phase18_variational_memory.py      # [Fix 3] Variational compression
│   ├── phase19_per_layer_constructor.py   # [Fix 4] Per-layer edge projections
│   ├── phase20_bfs_partition.py           # [Fix 2] BFS partition scaling
│   ├── phase21_learned_attention_dropout.py # [Fix 6] Learned dropout
│   ├── phase22_scale_stress_test.py         # Scale stress at N=1000 with noise
│   ├── phase23_realistic_kg_benchmark.py    # DELTA vs TransE/RotatE/CompGCN
│   ├── phase24_combined_integration.py      # All fixes integrated at scale
│   ├── phase25_fb15k237_gpu.py              # Real FB15k-237 on GPU
│   ├── phase26_adaptive_hop_depth.py        # Adaptive multi-hop depth learning
│   ├── phase27_bootstrap_relational.py      # Bootstrap relational task (initial, broken training)
│   ├── phase27b_bootstrap_batched.py        # Bootstrap relational task (gradient accum, corrected)
│   ├── phase28_hard_ablation.py             # Hard ablation: difficulty levels vs models
│   ├── phase29_multi_seed.py                # Multi-seed statistical evaluation
│   ├── phase30_edge_adj_sampling.py         # GPU edge adjacency sampling strategies
│   ├── phase31_mini_batching.py             # Subgraph sampling + gradient accumulation
│   ├── phase32_cross_graph_transfer.py      # Train FB15k-237, eval WN18RR (zero-shot)
│   ├── phase33_task_aware_construction.py   # Hybrid constructor: base topology + learned edges
│   └── phase34_graphgps_grit_comparison.py  # DELTA vs GraphGPS vs GRIT (Gap 1)
├── notebooks/              # Colab-ready infrastructure
│   └── delta_colab_ready.py  # Automated Colab setup + Phase 34 runner
├── tests/                  # Unit tests (44/44 passing)
│   ├── test_graph.py
│   ├── test_attention.py
│   ├── test_router.py
│   ├── test_memory.py
│   ├── test_utils.py
│   └── test_baselines.py     # GraphGPS + GRIT baseline tests
├── COLAB_SETUP.md          # Google Colab Pro+ setup instructions
├── RESEARCH_AGENDA.md      # Research gaps, compute options, publication pathway
├── requirements.txt
└── README.md
```

## Validation Phases

### Phase 1–15: Core Architecture Validation

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

### Phase 16–21: Architectural Fix Benchmarks

After Phase 15, a pitfall analysis identified 6 architectural weaknesses. Each was fixed and validated:

| Fix | Problem | Solution | Phase |
|-----|---------|----------|-------|
| 1 | Router scores elements before seeing attention (chicken-and-egg) | PostAttentionPruner: prune based on *observed* attention weights | 16 |
| 2 | Spectral partitioning is O(N³) — won't scale | BFS seed-expansion partitioner in O(N+E) | 20 |
| 3 | Fixed linear memory compression loses information | Variational bottleneck with KL regularization | 18 |
| 4 | Single averaged edge projection ignores per-layer structure | Per-layer edge projections + edge combiner + active edge_type_head | 19 |
| 5 | Dense O(E²) multi-hop adjacency times out at ~500 edges | Sparse COO tensor operations | 17 |
| 6 | Uniform dropout doesn't distinguish structural vs noisy edges | LearnedAttentionDropout: per-edge dropout conditioned on features | 21 |

| Phase | Validates | Benchmark | Result |
|-------|-----------|-----------|--------|
| 16 | Post-attention soft gating + curriculum | KG classification at 50% sparsity | **Soft gating 100%, curriculum 100%**, old router 85.3%, hard post-attn 65.3%. Soft differentiable gating eliminates the original 29% gap and beats pre-attention routing by +14.7% |
| 17 | Sparse COO multi-hop scaling | 1-hop/2-hop timing at 50→2500 edges | **O(E^0.97) scaling confirmed.** 2500-edge 2-hop in 0.18s (was timeout with dense). All correctness checks pass |
| 18 | Variational memory compression | Compression quality + downstream accuracy | Accuracy preserved at 100% with compression. KL converges 0.126→0.026. Gradient flows through bottleneck. Similarity threshold is learnable |
| 19 | Per-layer edge projections | Edge type diversity + relational classification | Both reach 100% accuracy. Per-layer produces higher edge-type entropy (1.632 vs 1.562) — richer type diversity |
| 20 | BFS partition scaling | Wall-clock time at 50→2500 nodes | **O(N^0.99) scaling.** 2.0ms→90.8ms. Balance ratio 0.79. Importance-based seed spread: 100% |
| 21 | Learned attention dropout | Generalization gap: no dropout vs uniform vs learned | All modes reach 0 gap on current dataset (saturated). Eval-time passthrough confirmed. Needs harder benchmark |

### Phase 22–25: Scale & Integration Validation

| Phase | Validates | Benchmark | Result |
|-------|-----------|-----------|--------|
| 22 | Soft gating holds at 10× scale with noise | N=1000, 15 relations, 5000 triples, 15% label noise | **Soft gating 100%, old router 81.6%** (+18.4% gap). Zero generalization gap. |
| 23 | DELTA vs KG embedding baselines (TransE, RotatE, CompGCN) | FB15k-237-like: 2000 entities, 20 typed relations, 8000 triples | **DELTA 100%, CompGCN 100%**, TransE 67.6%, RotatE 70.7%. LP: TransE Hits@10=0.020, RotatE 0.010 (4×/2× random; sparse synthetic data). Soft gating maintains accuracy at 50% sparsity |
| 24 | All fixes integrated at scale | N=1000, 15% noise, full pipeline + ablations | All fixes integrate cleanly — zero degradation. 1-hop ablation runs 10× faster (44s vs 490s) |
| 25 | DELTA on **real** FB15k-237 (GPU) | Actual Freebase triples: 2000-entity dense subgraph, 69,626 edges, 210 relation types, RTX 3080 Ti | **DELTA+Gate 97.6%, CompGCN 97.2%**, TransE 78.8%, RotatE 77.8%. LP: TransE Hits@10=0.480, RotatE 0.335 (vs random 0.005). First real-data benchmark on GPU. |

*Phases 22, 23, and 25 were replicated with 5 seeds each in Phase 29 — see the Phase 26–30 table for multi-seed statistics.*

### Phase 26–30 + 27b: Near-Term Roadmap Validation

| Phase | Validates | Benchmark | Result |
|-------|-----------|-----------|--------|
| 26 | Adaptive multi-hop: learn when to use 1-hop vs 2-hop | N=200, 3 models (FixedHop, AdaptiveHop, AdaptiveHopGating) | All models 100% at N=200. AdaptiveHopGate learns α→0.019 (suppresses 2-hop). Task saturates — validates cost-efficiency selection architecture. |
| 27 | *(initial — broken training)* Bootstrap on 2-hop path composition | N=500, batch=1, 200 epochs, no scheduler | TF 32.7%, Bootstrap DELTA 8.0%, Fixed Chain 5.3%. **Results confounded by batch-1 training** — see Phase 27b. |
| **27b** | **Corrected** bootstrap evaluation (gradient accum + 2× data) | N=1000, accum=32, 100 epochs, LR scheduler | **Fixed Chain DELTA 40.7% > Transformer 36.3% > Bootstrap 34.3%.** Graph structure helps (+4.4% over Transformer). GraphConstructor attention-thresholding discards path ordering — the constructor is the bottleneck, not graph processing. |
| 28 | Hard ablation: find difficulty threshold where DELTA advantages emerge | 4 levels (Easy/Medium/Hard/Extreme) × 3 models (Vanilla EdgeAttn, DualAttn, DELTA+Gate) | Extreme: **Dual Attention 64.2% >> Vanilla 40.2%** (+24%). Node context is the key DELTA advantage at high noise. Soft gating adds ±0.6% beyond dual attention at extreme difficulty. |
| 29 | Multi-seed statistical credibility | Phases 22, 23, 25 re-run with 5 seeds each | DELTA+Gate **97.4% ± 0.1%** on FB15k-237. Soft Gate 100.0% ± 0.0% vs Old Router 79.7% ± 1.1%. All results statistically stable. Total: 3.6 minutes. |
| 30 | GPU edge adj sampling strategy vs random | Uniform, degree-weighted, stratified, importance-weighted sampling at 26% budget on FB15k-237 | All 4 strategies within ±0.2% (~97.5%). Random sampling is sufficient at this graph density. |

## The Bootstrap Strategy

The "chicken and egg" problem of graph construction (you need to understand the input to build the graph, but the graph is how you understand input) is solved pragmatically:

1. Use a lightweight transformer to bootstrap an initial graph from raw input
2. DELTA processes and refines the graph
3. Over time, trained DELTA models can replace the transformer bootstrap — using their own graph representations to construct graphs for new input
4. The transformer is scaffolding, not a permanent dependency
Phase 5 confirms the pipeline preserves accuracy: with equal training (150 epochs each), the transformer→DELTA pipeline matches the transformer alone (98.3%) on a non-relational task.

**Phase 27b clarified the bootstrap's true role.** On a 2-hop path composition task (16 relational classes, N=1000), with properly trained models (gradient accumulation, LR scheduler):

| Model | Accuracy | vs. Random (6.2%) |
|---|---|---|
| Fixed Chain DELTA | **40.7%** | 6.6× |
| Transformer | 36.3% | 5.9× |
| Bootstrap DELTA (GraphConstructor) | 34.3% | 5.5× |

Key conclusions:
- **Graph structure genuinely helps relational tasks**: Fixed Chain DELTA beats the pure Transformer by +4.4% using the same transformer embeddings but with explicit adjacency structure
- **The GraphConstructor is the bottleneck, not graph processing**: Attention-thresholded construction (Bootstrap) *underperforms* the fixed chain — the constructor discards sequential connections essential for path composition
- **Phase 27's original result (DELTA << Transformer) was entirely training-confounded**: batch-1 gradient updates with Adam produced chaotic, non-converging updates specifically for the deeper DELTA models. Transformer was less affected because it processes the full batch in one shot regardless.
- **The fix is task-aware construction (Phase 33 roadmap)**: A constructor that preserves positional/path ordering for sequential tasks would give Bootstrap DELTA the benefits of both worlds.
## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
# Phase 1-6: Core validation
python -m experiments.phase1_edge_attention
python -m experiments.phase2_dual_attention
python -m experiments.phase3_router
python -m experiments.phase4_memory
python -m experiments.phase5_construction
python -m experiments.phase6_full_model

# Phase 7-10: Extended experimentation
python -m experiments.phase7_gumbel_routing    # Differentiable routing
python -m experiments.phase8_scaling           # Scaling analysis
python -m experiments.phase9_multi_hop         # Multi-hop reasoning
python -m experiments.phase10_analogy          # Analogical reasoning

# Phase 11-15: Advanced validation
python -m experiments.phase11_multi_hop_edges  # Multi-hop edge adjacency
python -m experiments.phase12_curriculum_routing  # Gumbel curriculum
python -m experiments.phase13_harder_benchmarks   # Compositional reasoning
python -m experiments.phase14_contrastive_analogy # Contrastive training
python -m experiments.phase15_kg_benchmark        # Synthetic KG benchmark

# Phase 16-21: Architectural fix benchmarks
python -m experiments.phase16_post_attention_pruning  # Post-attn vs pre-attn routing
python -m experiments.phase17_sparse_multi_hop        # Sparse COO scaling
python -m experiments.phase18_variational_memory      # Variational compression
python -m experiments.phase19_per_layer_constructor   # Per-layer edge projections
python -m experiments.phase20_bfs_partition           # BFS partition scaling
python -m experiments.phase21_learned_attention_dropout # Learned dropout

# Phase 22-24: Scale & integration validation
python -m experiments.phase22_scale_stress_test        # N=1000, 15% noise
python -m experiments.phase23_realistic_kg_benchmark   # vs TransE/RotatE/CompGCN
python -m experiments.phase24_combined_integration     # All fixes at scale
python experiments/phase25_fb15k237_gpu.py             # Real FB15k-237 on GPU

# Phase 26-30: Near-term roadmap validation
python experiments/phase26_adaptive_hop_depth.py       # Adaptive multi-hop depth
python experiments/phase27_bootstrap_relational.py     # Bootstrap relational (initial)
python experiments/phase27b_bootstrap_batched.py       # Bootstrap relational (corrected)
python experiments/phase28_hard_ablation.py            # Hard ablation benchmark
python experiments/phase29_multi_seed.py               # Multi-seed evaluation (5 seeds)
python experiments/phase30_edge_adj_sampling.py        # GPU edge adj sampling strategies

# Phase 31-34: Next-step experiments (infrastructure ready)
python experiments/phase31_mini_batching.py             # Subgraph sampling + gradient accumulation
python experiments/phase32_cross_graph_transfer.py      # Train FB15k-237, eval WN18RR (zero-shot)
python experiments/phase33_task_aware_construction.py   # Hybrid constructor: base topology + learned edges
python experiments/phase34_graphgps_grit_comparison.py  # DELTA vs GraphGPS vs GRIT (Gap 1)

# Run all tests
python -m pytest tests/ -q  # 44/44 should pass
```

## Key Findings

### Core Architecture (Phases 1–15)

1. **Edge attention is DELTA's strongest signal**: Edge-to-edge attention perfectly solves relational classification (100%) where node attention collapses to 26.7%. This validates the thesis that edges deserve first-class attention.

2. **Sub-linear scaling**: DELTA scales at O(n^0.81) from 20→400 nodes thanks to the importance router's sparse attention, maintaining 100% accuracy across all tested scales.

3. **Gumbel-softmax routing enables differentiable selection**: The straight-through estimator lets all 12 router parameters receive gradients (vs 0 with hard top-k), enabling the router to learn from task loss. Temperature annealing provides a curriculum from exploration to exploitation.

4. **Multi-hop edge adjacency solves compositional reasoning**: Phase 11 showed 2-hop edge adjacency achieves **100% accuracy on derived/transitive relations** — a +38.9% jump from 1-hop (61.1%) and beating Node GNN (83.3%). This was the biggest architectural improvement in the project: edges that can "see" 2 hops away compose transitive inferences naturally.

5. **Compositional logic rules are DELTA's sweet spot**: Phase 13 tested 7 relation types (4 base + 3 derived from logical rules). DELTA hit 100% on all including derived relations, while Node GNN achieved only 87.5% on derived. Edge-to-edge attention discovers compositional patterns that node message passing misses.

6. **Edge embeddings are inherently discriminative**: Phase 14 showed all training methods (classification, contrastive, joint) achieve 100% nearest-neighbor retrieval. Edge attention creates well-clustered embeddings by default — the Phase 10 analogy failure was a task formulation issue, not an embedding quality issue.

7. **Router sparsity needs careful calibration**: Phases 12 and 15 consistently show aggressive sparsity (40-50%) hurts on relation classification. The router works well for maintaining accuracy at moderate sparsity (80% in Phase 3) but needs gentler schedules or task-specific tuning at higher compression.

8. **DELTA matches Node GNN at scale, excels on derived relations**: On the 500-triple KG benchmark (Phase 15), both architectures reach 100% — but DELTA's advantage emerges specifically on compositional/transitive reasoning tasks where edges must attend to other edges' relational context.

### Architectural Fix Validation (Phases 16–21)

9. **Sparse COO multi-hop is the clearest win**: Phase 17 confirmed O(E^0.97) sub-quadratic scaling — 2500-edge 2-hop completes in 0.18s where the old dense approach timed out at ~500 edges. This removes the main scaling bottleneck for multi-hop reasoning.

10. **BFS partitioning scales linearly**: Phase 20 confirmed O(N^0.99) scaling from 50→2500 nodes (2ms→91ms), replacing O(N³) spectral clustering. Balance ratio of 0.79 with 100% importance-node coverage across partitions.

11. **Variational memory compression preserves accuracy**: Phase 18 showed the variational bottleneck achieves identical downstream accuracy (100%) to uncompressed features, while KL loss converges during training (0.126→0.026), confirming the latent space is being regularized. The learned similarity threshold adapts under gradient.

12. **Per-layer edge projections increase type diversity**: Phase 19 showed per-layer constructor produces higher edge-type entropy (1.632 vs 1.562) — richer diversity of inferred edge types — while matching classification accuracy.

13. **Soft gating solves post-attention pruning**: The original Phase 16 showed a 29% accuracy gap (hard post-attn 61.3% vs old router 90.7%). Root cause: hard top-k is non-differentiable — the pruner gates received zero gradient and never learned. The redesigned PostAttentionPruner uses soft sigmoid gates with per-head attention features, achieving **100% accuracy at 50% target sparsity** — matching full attention and beating pre-attention routing by +14.7%. Curriculum annealing (temperature 0.5→5.0, sparsity 0→50%) also reaches 100%.

14. **Learned dropout needs harder benchmarks**: Phase 21 showed all dropout modes reach 0 generalization gap on the current 100-entity KG. The dataset is too easy to differentiate — dropout benefits will emerge at larger scale with noise and distribution shift.

### Scale & Integration Findings (Phases 22–24)

15. **Soft gating holds at 10× scale**: Phase 22 scaled from N=100 to N=1000 with 15% label noise and power-law degree distribution. The old pre-attention router dropped to 81.6% while soft gating held at 100% — an +18.4% advantage. Zero generalization gap (vs +0.019 for the old router). This is the definitive scale validation.

16. **DELTA matches CompGCN, crushes embedding baselines**: Phase 23 compared against faithful implementations of TransE (Bordes 2013; 67.6%), RotatE (Sun 2019 complex rotation; 70.7%), and CompGCN (Vashishth 2020 GRU message passing; 100%) on an FB15k-237-like benchmark with 2000 entities, 20 typed relations, and compositional derived edges. DELTA matches the best GNN baseline while providing sparsity-efficient inference via soft gating. Link prediction trained separately with margin-based ranking loss (standard LP protocol); low Hits@10 (~0.02) reflects sparse synthetic data (~4 triples/entity), not broken evaluation.

17. **All fixes integrate cleanly at scale**: Phase 24 ran the full pipeline (variational memory + BFS partition + sparse 2-hop adjacency + dual attention + learned dropout + soft gating) on the N=1000 noisy benchmark. No fix caused degradation — all ablations matched full DELTA at 100%. The 2-hop edge adjacency (4.3M entries at E=5000) is the dominant compute cost, with 1-hop ablation running ~10× faster.

18. **DELTA+Gate outperforms all baselines on REAL FB15k-237 data**: Phase 25 ran on actual Freebase triples (2000-entity dense subgraph, 69,626 edges, 210 real relation types) on a GPU (RTX 3080 Ti). DELTA+Gate reached **97.6%** relation classification accuracy, narrowly beating CompGCN (97.2%), TransE (78.8%), and RotatE (77.8%). Embedding baselines gain +10-11% over synthetic Phase 23 results — reflecting richer real-world structural patterns. TransE LP Hits@10=0.480 (96× random) on real triples confirms learned representations generalize. Edge adjacency capped at 5M of 19M pairs (~26%) to fit GPU VRAM — DELTA still wins despite seeing a fraction of all structural context.

19. **Graph structure genuinely helps on relational tasks — GraphConstructor is the bottleneck**: Phase 27b tested the bootstrap pipeline with proper training (gradient accumulation, 2× data) on 2-hop path composition. Fixed Chain DELTA (40.7%) beat the pure Transformer (36.3%) — confirming graph processing adds value on relational tasks. However, Bootstrap DELTA (34.3%) underperformed Fixed Chain by −6.3% because attention-thresholded construction discards the sequential adjacency edges needed for path composition. Phase 27's original conclusion (Transformer >> DELTA) was entirely a training artifact: batch-1 Adam updates cause chaotic gradients in deeper DELTA models but don't affect transformers that process full batches in one forward pass.

20. **Soft gating's advantage is efficiency at scale, not accuracy at extreme noise**: Phase 28 designed 4 difficulty levels to find where individual DELTA components differentiate. At Extreme difficulty (noise=0.8, proto_spread=0.3), Dual Attention (64.2%) beats Vanilla EdgeAttention (40.2%) by +24% — confirming node context is the key architectural advantage. Soft gating adds only ±0.6% beyond dual attention here. Gating's value is sparsity and inference cost, not peak accuracy.

21. **All key results are statistically stable across 5 seeds**: Phase 29 confirmed DELTA+Gate 97.4% ± 0.1% on FB15k-237, Soft Gate 100.0% ± 0.0% vs Old Router 79.7% ± 1.1% at N=1000. Very low variance across seeds rules out lucky initialization as an explanation for any headline result.

## Backward Compatibility

After implementing all 6 fixes, backward compatibility was verified against 5 critical original phases:

| Phase | Metric | Original | After Fixes | Status |
|-------|--------|----------|-------------|--------|
| 1 | Edge Attention accuracy | 100% | 100% | ✅ Match |
| 7 | Gumbel routing at 60% sparsity | 62.5% | 62.5% | ✅ Match |
| 9 | DELTA Edge multi-hop | 84.4% | 84.4% | ✅ Match |
| 13 | DELTA 2-hop on derived | 100% | 100% | ✅ Match |
| 15 | Full / Router@50% | 100% / 65.3% | 100% / 74.7% | ✅ Improved |

Phase 15's Router@50% improved from 65.3%→74.7% because the legacy `ImportanceRouter` wrapper now delegates to `PostAttentionPruner.prune()` with `min()` safety bounds.

## Architecture Evolution

DELTA has gone through six development stages:

### Stage 1: Core Validation (Phases 1–15)
Proved the thesis: edge-first dual attention outperforms node-only GNNs on compositional reasoning. Multi-hop edge adjacency (Phase 11) and compositional logic rules (Phase 13) were the headline results.

### Stage 2: Pitfall Analysis & Fixes (6 Fixes)
Identified and fixed 6 architectural weaknesses:
- **Routing paradigm shift**: Pre-attention prediction → post-attention pruning based on observed weights
- **Scaling fixes**: Dense O(E²) → sparse COO, spectral O(N³) → BFS O(N+E)
- **Memory upgrade**: Fixed compression → variational bottleneck with KL regularization
- **Constructor upgrade**: Averaged attention → per-layer edge projections
- **Regularization**: Uniform dropout → learned per-edge dropout

### Stage 3: Soft Gating Breakthrough (Phase 16 Redesign)
All fixes implemented and backward-compatible. Scaling bottlenecks (sparse multi-hop, BFS partition) are definitively solved. **Post-attention pruning is now fully validated** — the original 29% accuracy gap was caused by non-differentiable hard top-k gates, not a fundamental paradigm problem. The redesigned soft sigmoid gating with per-head attention features achieves 100% accuracy at 50% target sparsity, and curriculum annealing (dense→sparse) also reaches 100%. Remaining awaiting-scale items (learned dropout, variational compression advantage) are architecturally sound but need harder benchmarks to demonstrate their value.

### Stage 4: Scale & Integration Validation (Phases 22–24)
Validated the full architecture at 10× scale. **Soft gating holds at N=1000 with 15% noise** (+18.4% over old router). DELTA matches CompGCN and crushes TransE/RotatE on a realistic KG benchmark with faithful baseline implementations (TransE: Bordes 2013 translation scoring, RotatE: Sun 2019 complex rotation, CompGCN: GRU message passing with relation composition). Link prediction evaluated with proper margin-based ranking training (separate from classification) — low absolute numbers (Hits@10 ~0.02) reflect data sparsity (~4 triples/entity) not broken methodology. All 5 inference-pipeline fixes integrate cleanly with zero degradation. The remaining challenge: designing tasks hard enough to differentiate individual fix contributions — current synthetic benchmarks are solvable by vanilla EdgeAttention at this scale.

### Stage 5: Real-World Data on GPU (Phase 25)
**First benchmark on actual real-world data**, running on GPU. Phase 25 downloaded the real FB15k-237 dataset (14,505 entities, 237 relations, 310k Freebase triples) and evaluated on the 2000-entity dense subgraph (69,626 real triples, 210 relation types). **DELTA+Gate 97.6%** outperforms all baselines (CompGCN 97.2%, TransE 78.8%, RotatE 77.8%). TransE link prediction Hits@10=0.480 (96× above random) confirms representations generalize on real data. GPU enablement required a CUDA-build PyTorch install and two library fixes (sparse tensor device propagation in `graph.py`, CPU tensor attribute in `router.py`).

### Stage 6: Near-Term Roadmap Validation + External Review (Phases 26–30, 27b)
Validated 5 roadmap items and stress-tested a core assumption with help from an external review.

**Phase 26** confirmed the adaptive hop-depth architecture: `AdaptiveHopGate` learns α→0.019 (suppressing the more expensive 2-hop when 1-hop suffices), validating cost-efficiency selection on a task that saturates at N=200.

**Phase 27b** (corrected from initial Phase 27) is the most significant finding of this stage. An external review identified that Phase 27's batch-1 training was severely handicapping DELTA models. With proper gradient accumulation and 2× data: Fixed Chain DELTA (40.7%) beat the pure Transformer (36.3%) — proving graph structure adds value on relational tasks. Bootstrap DELTA (34.3%) underperformed Fixed Chain due to the GraphConstructor discarding path-ordering edges. Conclusion: the pipeline is sound; the constructor needs to be task-aware. This also prompted a performance fix in `graph.py` (edge adjacency caching + vectorized incidence matrix for small graphs, replacing a Python for-loop that made sequential training prohibitively slow).

**Phase 28** found that at Extreme difficulty (noise=0.8), Dual Attention (+24% over Vanilla) is the key differentiator — node context matters when edge features are noisy. Soft gating adds only ±0.6% beyond dual attention here, confirming gating's value is efficiency rather than peak accuracy.

**Phase 29** confirmed all headline results are statistically stable across 5 seeds (DELTA+Gate 97.4% ± 0.1%, Soft Gate 100.0% ± 0.0%).

**Phase 30** confirmed random edge-adjacency sampling at 26% GPU budget is sufficient — all 4 sampling strategies (uniform, degree-weighted, stratified, importance-weighted) achieve within ±0.2% of each other on FB15k-237.

## Current Status: What's Working vs What Needs Proof

### ✅ Validated & Working
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
- **44/44 unit tests passing**, backward compatibility confirmed

### ⚠️ Architecturally Sound, Awaiting Scale Proof
- **Learned attention dropout** — mechanisms confirmed (eval passthrough, rate diversity), but all benchmarks including N=1000 are too easy to show gap reduction
- **Variational compression advantage** — preserves accuracy but hasn't shown improvement over fixed compression yet
- **Ablation differentiation** — at N=1000, vanilla EdgeAttention also reaches 100%, so fix ablations show zero impact. The fixes provide *efficiency and robustness*, not accuracy gains on tasks a baseline can already solve

### ❌ Open Gaps
- **Scale ceiling unknown** — Phase 25/30 reached 2000 entities / 69,626 edges on GPU; real KGs have millions. Full-scale training requires mini-batching and subgraph sampling (Phase 31).
- **GraphConstructor needs task-aware construction** — Phase 27b confirmed attention-thresholding discards sequential/path structure. A constructor that preserves positional ordering would let Bootstrap DELTA match or exceed Fixed Chain (Phase 33).
- **Soft gating marginal on extreme noise** — Phase 28 showed dual attention is the key differentiator at extreme difficulty (+24%); soft gating adds only ±0.6% beyond that. Gating's value remains efficiency, not peak accuracy on current benchmarks.
- **Cross-domain generalization untested** — All results are within FB15k-237. Whether DELTA's edge-attention representations transfer to WN18RR or other KG domains without retraining is unknown (Phase 32).

## Roadmap

### Completed
| Phase / Item | Result |
|---|---|
| **Phase 22: Scale stress test** | Soft gating 100% ± 0.0% vs old router 79.7% ± 1.1% at N=1000 with 15% noise. +20.3% advantage (5 seeds). |
| **Phase 23: Synthetic KG baseline comparison** | DELTA matches CompGCN (100% ± 0.0%), beats TransE (68.1% ± 0.4%) and RotatE (68.4% ± 1.4%). 5-seed statistics. |
| **Phase 24: Combined fix integration** | All 6 fixes integrate cleanly at N=1000. Zero degradation across ablations. 1-hop runs 10× faster than 2-hop with no accuracy cost. |
| **Phase 25: Real FB15k-237 on GPU** | DELTA+Gate 97.4% ± 0.1% on actual Freebase triples — outperforms CompGCN 96.9% ± 0.3%. 5-seed statistics on 2000-entity subgraph. |
| **Phase 26: Adaptive multi-hop depth** | AdaptiveHopGate learns α→0.019 (suppresses 2-hop when unnecessary). All models 100% at N=200; validates cost-efficiency selection. |
| **Phase 27: Bootstrap relational task (initial — broken training)** | Transformer 32.7% >> Bootstrap DELTA 8.0% > Fixed Chain 5.3%. Results confounded by batch-1 training. See Phase 27b for corrected results. |
| **Phase 27b: Bootstrap relational (gradient accum, corrected)** | **Fixed Chain DELTA 40.7% > Transformer 36.3% > Bootstrap 34.3%.** Graph structure helps (+4.4% over Transformer). GraphConstructor attention-thresholding is the bottleneck — discards path ordering. Phase 27's gap was entirely training-confounded. Also: edge adjacency caching + vectorized fast path added to `graph.py`. |
| **Phase 28: Harder ablation benchmark** | At Extreme difficulty (noise=0.8, label_noise=0.35): Dual Attention 64.2% >> Vanilla EdgeAttn 40.2% (+24%). Node context is the key DELTA advantage when edge features are noisy. |
| **Phase 29: Multi-seed evaluation** | All key results confirmed stable across 5 seeds. DELTA+Gate 97.4% ± 0.1% on FB15k-237. Soft Gate 100% ± 0.0% vs Old Router 79.7% ± 1.1%. |
| **Phase 30: GPU edge adj sampling** | Uniform, degree-weighted, stratified, and importance-weighted sampling all achieve 97.3–97.5% on FB15k-237 at 26% budget. Random sampling is sufficient at this density. |

### Next Steps

**Phase 31: Mini-batching for large graphs** *(experiment ready)*
Phase 25/30 maxed at 2000 entities (69K edges) on a 12 GB GPU. Real KGs have millions of entities. Implements neighbor-sampling subgraph extraction + gradient accumulation to scale beyond single-GPU VRAM. See `experiments/phase31_mini_batching.py`. Phase 27b demonstrated gradient accumulation works well for DELTA — same technique applies here.

**Phase 32: Cross-graph transfer** *(experiment ready)*
Train on FB15k-237, evaluate zero-shot on WN18RR. Measures whether DELTA edge-attention representations generalize across KG domains without retraining. See `experiments/phase32_cross_graph_transfer.py`.

**Phase 33: Task-aware graph construction** *(experiment ready)*
Phase 27b confirmed the problem: GraphConstructor's attention-thresholding discards sequential adjacency edges that Fixed Chain DELTA preserves. Result: Fixed Chain (40.7%) > Bootstrap (34.3%) on path composition. Implements hybrid construction: preserve base topology + learn new edges. See `experiments/phase33_task_aware_construction.py`.

**Phase 34: DELTA vs GraphGPS vs GRIT comparison** *(infrastructure ready)*
Critical baseline currency gap (Gap 1 in [RESEARCH_AGENDA.md](./RESEARCH_AGENDA.md)). CompGCN (2020) is the current strongest baseline — the community will ask about GraphGPS (2022) and GRIT (2023). Lightweight implementations in `delta/baselines.py`, 16 tests in `tests/test_baselines.py`, experiment script at `experiments/phase34_graphgps_grit_comparison.py`. Synthetic comparison runs immediately on CPU; full FB15k-237 comparison requires Phase 31 compute. See `COLAB_SETUP.md` for Google Colab Pro+ ($49.99/mo) setup instructions and `notebooks/delta_colab_ready.py` for automated infrastructure.

### Long-Term
- **Replace transformer bootstrap** — Use trained DELTA models to construct graphs for new inputs (self-bootstrapping), removing the scaffolding dependency
- **Multi-modal input** — Extend graph constructor beyond token sequences (images, tables, structured data)
- **Real-world application** — Knowledge graph completion, drug-target interaction, or recommendation systems at production scale

## Requirements

- Python 3.10+
- PyTorch 2.0+ (GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu124`)
- CPU sufficient for Phases 1–24; GPU (6 GB+ VRAM) recommended for Phase 25+

---

*DELTA architecture — conceived March 25, 2026. 35 experiment phases (Phases 1–30 + Phase 27b correction + Phases 31–34), 6 architectural fixes, 44 unit tests. Phases 31–34 experiments ready for GPU execution.*
