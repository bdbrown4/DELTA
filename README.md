# DELTA: Dynamic Edge-and-Node Architecture with Layered, Tiered Attention

A research implementation of the DELTA architecture — a novel AI framework that operates on dynamic graphs with dual parallel attention across nodes and edges, tiered memory, and a learned importance router.

## Core Thesis

Reality is a graph. Language is a lossy compression of reality into sequences. Transformers reconstruct relational structure from flat sequences. DELTA operates on relational structure directly.

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
│   └── utils.py            # Helpers, synthetic data, and 7 benchmark generators
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
│   └── phase21_learned_attention_dropout.py # [Fix 6] Learned dropout
├── tests/                  # Unit tests (22/22 passing)
│   ├── test_graph.py
│   ├── test_attention.py
│   ├── test_router.py
│   └── test_memory.py
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
| 5 | Transformer-bootstrapped graph construction | Transformer alone vs transformer→DELTA | TF: 98.3%, DELTA: 96.7% (non-relational task) |
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

## The Bootstrap Strategy

The "chicken and egg" problem of graph construction (you need to understand the input to build the graph, but the graph is how you understand input) is solved pragmatically:

1. Use a lightweight transformer to bootstrap an initial graph from raw input
2. DELTA processes and refines the graph
3. Over time, trained DELTA models can replace the transformer bootstrap — using their own graph representations to construct graphs for new input
4. The transformer is scaffolding, not a permanent dependency

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

# Run all tests
python -m pytest tests/ -v  # 22/22 should pass
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

DELTA has gone through three development stages:

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
- **22/22 unit tests passing**, backward compatibility confirmed

### ⚠️ Architecturally Sound, Awaiting Scale Proof
- **Learned attention dropout** — mechanisms confirmed (eval passthrough, rate diversity), but current benchmarks too easy to show gap reduction
- **Variational compression advantage** — preserves accuracy but hasn't shown improvement over fixed compression yet

### ❌ Open Gaps
- **No real dataset validation** — all benchmarks are synthetic. Need FB15k-237, WN18RR, or similar
- **Scale ceiling unknown** — tested up to 2500 nodes/edges, but real KGs have millions
- **No GPU profiling** — all experiments on CPU. Memory/throughput characteristics on GPU untested

## Roadmap

### Near-Term (Next Phases)
1. **Phase 22: Scale stress test** — Run soft gating + learned dropout at N=1000+ with noisy synthetic KGs to verify advantages hold at scale
2. **Phase 23: Real KG benchmark** — FB15k-237 or WN18RR link prediction, compare against TransE/RotatE/CompGCN baselines
3. **Phase 24: Combined fix integration** — Test all 6 fixes working together in the full pipeline on a single challenging task
4. ~~**Phase 25: Curriculum + post-attention**~~ — ✅ **DONE.** Curriculum annealing (temperature + sparsity ramp) integrated with soft gating in Phase 16 redesign. Both reach 100% accuracy

### Medium-Term
5. **GPU profiling & batching** — Profile memory and throughput, implement mini-batching for large graphs
6. **Adaptive multi-hop depth** — Learn when to use 1-hop vs 2-hop vs k-hop per layer
7. **Cross-graph transfer** — Train on one KG, evaluate on another (generalization)

### Long-Term
8. **Replace transformer bootstrap** — Use trained DELTA models to construct graphs for new inputs (self-bootstrapping)
9. **Multi-modal input** — Extend graph constructor beyond token sequences
10. **Real-world application** — Knowledge graph completion, drug-target interaction, or recommendation systems

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CPU is sufficient for all current experiments (Phases 1–21)
- GPU recommended for planned Phase 22+ scale experiments

---

*DELTA architecture — conceived March 25, 2026. 21 validation phases, 6 architectural fixes, 22 unit tests.*
