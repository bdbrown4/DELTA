# Project Structure

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
│   ├── phase34_graphgps_grit_comparison.py  # DELTA vs GraphGPS vs GRIT (Gap 1)
│   ├── phase35_relational_transfer.py       # Domain-agnostic transfer: linear probe + GRL + ablation
│   ├── phase36_task_aware_at_scale.py       # Task-aware construction at 500-2000 nodes
│   ├── phase37_real_comparison.py           # Real FB15k-237 parameter-matched comparison
│   ├── phase38_component_ablation.py        # [planned] Real FB15k-237 ablation (5 components × 5 seeds)
│   ├── phase39_multihop_path_queries.py     # [planned] 1p/2p/3p path query evaluation
│   ├── phase40_yago3_benchmark.py           # [planned] YAGO3-10 (123K entities, 4-model × 5 seeds)
│   ├── phase41_codexm_benchmark.py          # [planned] Codex-M (17K entities, 51 relations)
│   ├── phase42_scaling_analysis.py          # [planned] Sub-quadratic scaling: 500→123K entities
│   └── phase43_interpretability.py          # [planned] EdgeAttention top-k + t-SNE visualizations
├── notebooks/              # Colab-ready infrastructure
│   └── delta_colab_ready.py  # Automated Colab setup + Phase 34 runner
├── tests/                  # Unit tests (44/44 passing)
│   ├── test_graph.py
│   ├── test_attention.py
│   ├── test_router.py
│   ├── test_memory.py
│   ├── test_utils.py
│   └── test_baselines.py     # GraphGPS + GRIT baseline tests
├── docs/                   # Documentation (this site)
│   ├── index.md               # Home page
│   ├── architecture.md        # Architecture overview
│   ├── ARCHITECTURE_VISUAL.md # Three-paradigm visual explainer
│   ├── bootstrap-strategy.md  # Bootstrap strategy
│   ├── architecture-evolution.md # Evolution stages
│   ├── validation-phases.md   # Phase result tables
│   ├── key-findings.md        # 21 key findings
│   ├── COLAB_RESULTS.md       # Colab experiment results
│   ├── backward-compatibility.md # Fix compatibility
│   ├── RESEARCH_AGENDA.md     # Research agenda
│   ├── PUBLICATION_ROADMAP.md # Publication roadmap
│   ├── research-methodology.md # AI disclosure
│   ├── status-and-roadmap.md  # Current status
│   ├── setup-and-running.md   # Setup guide
│   ├── COLAB_SETUP.md         # Colab setup
│   └── project-structure.md   # This page
├── requirements.txt
├── mkdocs.yml              # Documentation site config
└── README.md
```

---

## Core Library (`delta/`)

| File | Purpose |
|------|---------|
| `graph.py` | `DeltaGraph` data structure, sparse COO multi-hop edge adjacency, edge adjacency caching, vectorized incidence matrix |
| `attention.py` | `NodeAttention`, `EdgeAttention`, `DualParallelAttention`, `ReconciliationBridge` — all with `return_weights` support |
| `router.py` | `PostAttentionPruner` (soft sigmoid gating), `LearnedAttentionDropout`, legacy `ImportanceRouter` |
| `memory.py` | Tiered memory (hot/warm/cold) with variational bottleneck and KL regularization |
| `partition.py` | BFS seed-expansion partitioning in O(N+E) with importance-aware seeding |
| `reconciliation.py` | Node-edge co-update (ReconciliationBridge) |
| `constructor.py` | Transformer-based graph bootstrap with per-layer edge projections |
| `model.py` | `DELTAModel` — full pipeline: constructor → partition → dual attention → pruner → reconciliation → memory |
| `baselines.py` | `GraphGPSModel` (2022) and `GRITModel` (2023) implementations for comparison |
| `utils.py` | Synthetic data generators, benchmark tasks, helper functions |

## Baselines (`delta/baselines.py`)

| Model | Reference | Params (typical) |
|-------|-----------|-------------------|
| GraphGPS | Rampášek et al. (2022) | ~33K |
| GRIT | Ma et al. (2023) | ~28K |
| CompGCN | Vashishth et al. (2020) | Via `utils.py` |
| TransE | Bordes et al. (2013) | Via `utils.py` |
| RotatE | Sun et al. (2019) | Via `utils.py` |
