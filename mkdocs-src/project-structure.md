# Project Structure

```
DELTA/
|-- delta/                    Core library
|   |-- graph.py              DeltaGraph + sparse COO multi-hop edge adjacency
|   |-- attention.py          Node, edge, dual parallel attention (return_weights)
|   |-- router.py             PostAttentionPruner + LearnedAttentionDropout
|   |-- memory.py             Variational bottleneck tiered memory (hot/warm/cold)
|   |-- partition.py          BFS seed-expansion partitioning O(N+E)
|   |-- reconciliation.py     Node-edge co-update (ReconciliationBridge)
|   |-- constructor.py        Transformer-based graph bootstrap
|   |-- model.py              Full DELTA model
|   |-- baselines.py          GraphGPS + GRIT implementations
|   |-- datasets.py           Dataset loading (load_lp_data for FB15k-237)
|   +-- utils.py              Helpers, synthetic data, benchmark generators
|
|-- experiments/              Phase-by-phase validation (49 phases)
|   |-- phase1-15             Core architecture validation
|   |-- phase16-21            Architectural fix benchmarks
|   |-- phase22-25            Scale & integration on GPU
|   |-- phase26-30            Roadmap validation + multi-seed
|   |-- phase31-37            H100 / Colab experiments
|   |-- phase38-40            Differentiable constructor, self-bootstrap, correct LP
|   |-- phase41-45            Multi-hop compositional reasoning + inference timing
|   +-- phase46-49            Attention temperature optimization
|
|-- notebooks/                Colab-ready infrastructure
|   +-- delta_colab_ready.py
|
|-- tests/                    Unit tests (44/44 passing)
|   |-- test_graph.py
|   |-- test_attention.py
|   |-- test_router.py
|   |-- test_memory.py
|   |-- test_utils.py
|   +-- test_baselines.py
|
|-- mkdocs-src/               Documentation source (this site)
|   |-- index.md              Home page
|   |-- architecture.md       Architecture + bootstrap + timeline + compat
|   |-- ARCHITECTURE_VISUAL.md  Interactive three-paradigm visual
|   |-- the-brain.md          Vision + capacity paradox + roadmap
|   |-- key-findings.md       29 key findings by stage
|   |-- validation-phases.md  Complete phase result tables
|   |-- status-and-roadmap.md Status + gaps + publication pathway
|   |-- research-methodology.md  AI assistance disclosure
|   |-- setup-and-running.md  Setup + commands + cloud GPU
|   +-- project-structure.md  This page
|
|-- requirements.txt
|-- mkdocs.yml
+-- README.md
```

---

## Core Library (`delta/`)

| File | Purpose |
|------|---------|
| `graph.py` | `DeltaGraph` data structure, sparse COO multi-hop edge adjacency, edge adjacency caching, vectorized incidence matrix |
| `attention.py` | `NodeAttention`, `EdgeAttention`, `DualParallelAttention`, `ReconciliationBridge` -- all with `return_weights` support |
| `router.py` | `PostAttentionPruner` (soft sigmoid gating), `LearnedAttentionDropout`, legacy `ImportanceRouter` |
| `memory.py` | Tiered memory (hot/warm/cold) with variational bottleneck and KL regularization |
| `partition.py` | BFS seed-expansion partitioning in O(N+E) with importance-aware seeding |
| `reconciliation.py` | Node-edge co-update (ReconciliationBridge) |
| `constructor.py` | Transformer-based graph bootstrap with per-layer edge projections |
| `model.py` | `DELTAModel` -- full pipeline: constructor -> partition -> dual attention -> pruner -> reconciliation -> memory |
| `baselines.py` | `GraphGPSModel` (2022) and `GRITModel` (2023) implementations for comparison |
| `datasets.py` | Dataset loading utilities; `load_lp_data()` for FB15k-237 experiments |
| `utils.py` | Synthetic data generators, benchmark tasks, helper functions |

## Baselines (`delta/baselines.py`)

| Model | Reference | Params (typical) |
|-------|-----------|-------------------|
| GraphGPS | Rampasek et al. (2022) | ~228K |
| GRIT | Ma et al. (2023) | ~197K |
| CompGCN | Vashishth et al. (2020) | Via `utils.py` |
| TransE | Bordes et al. (2013) | Via `utils.py` |
| RotatE | Sun et al. (2019) | Via `utils.py` |
| DistMult | Yang et al. (2015) | ~47K |
