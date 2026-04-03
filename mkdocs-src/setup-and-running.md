# Getting Started

## Requirements

- Python 3.10+
- PyTorch 2.0+ (GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu124`)
- CPU sufficient for Phases 1–24; GPU (6 GB+ VRAM) recommended for Phase 25+

## Setup

```bash
pip install -r requirements.txt
```

---

## Running Experiments

### Phase 1–6: Core Validation

```bash
python -m experiments.phase1_edge_attention
python -m experiments.phase2_dual_attention
python -m experiments.phase3_router
python -m experiments.phase4_memory
python -m experiments.phase5_construction
python -m experiments.phase6_full_model
```

### Phase 7–10: Extended Experimentation

```bash
python -m experiments.phase7_gumbel_routing    # Differentiable routing
python -m experiments.phase8_scaling           # Scaling analysis
python -m experiments.phase9_multi_hop         # Multi-hop reasoning
python -m experiments.phase10_analogy          # Analogical reasoning
```

### Phase 11–15: Advanced Validation

```bash
python -m experiments.phase11_multi_hop_edges  # Multi-hop edge adjacency
python -m experiments.phase12_curriculum_routing  # Gumbel curriculum
python -m experiments.phase13_harder_benchmarks   # Compositional reasoning
python -m experiments.phase14_contrastive_analogy # Contrastive training
python -m experiments.phase15_kg_benchmark        # Synthetic KG benchmark
```

### Phase 16–21: Architectural Fix Benchmarks

```bash
python -m experiments.phase16_post_attention_pruning  # Post-attn vs pre-attn routing
python -m experiments.phase17_sparse_multi_hop        # Sparse COO scaling
python -m experiments.phase18_variational_memory      # Variational compression
python -m experiments.phase19_per_layer_constructor   # Per-layer edge projections
python -m experiments.phase20_bfs_partition           # BFS partition scaling
python -m experiments.phase21_learned_attention_dropout # Learned dropout
```

### Phase 22–24: Scale & Integration Validation

```bash
python -m experiments.phase22_scale_stress_test        # N=1000, 15% noise
python -m experiments.phase23_realistic_kg_benchmark   # vs TransE/RotatE/CompGCN
python -m experiments.phase24_combined_integration     # All fixes at scale
python experiments/phase25_fb15k237_gpu.py             # Real FB15k-237 on GPU
```

### Phase 26–30: Near-Term Roadmap Validation

```bash
python experiments/phase26_adaptive_hop_depth.py       # Adaptive multi-hop depth
python experiments/phase27_bootstrap_relational.py     # Bootstrap relational (initial)
python experiments/phase27b_bootstrap_batched.py       # Bootstrap relational (corrected)
python experiments/phase28_hard_ablation.py            # Hard ablation benchmark
python experiments/phase29_multi_seed.py               # Multi-seed evaluation (5 seeds)
python experiments/phase30_edge_adj_sampling.py        # GPU edge adj sampling strategies
```

### Phase 31–34: H100 / Colab Experiments

```bash
python experiments/phase31_mini_batching.py             # Subgraph sampling + gradient accumulation
python experiments/phase32_cross_graph_transfer.py      # Train FB15k-237, eval WN18RR (zero-shot)
python experiments/phase33_task_aware_construction.py   # Hybrid constructor: base topology + learned edges
python experiments/phase34_graphgps_grit_comparison.py  # DELTA vs GraphGPS vs GRIT (Gap 1)
```

### Phase 35–37: Transfer, Scale, and Parameter-Matched

```bash
python experiments/phase35_relational_transfer.py       # Domain-agnostic transfer (GRL + linear probe)
python experiments/phase36_task_aware_at_scale.py       # Task-aware construction at scale (500-2000 nodes)
python experiments/phase37_real_comparison.py           # Real FB15k-237 parameter-matched (4 models × 5 seeds)
```

---

## Running Tests

```bash
# Run all tests (44/44 should pass)
python -m pytest tests/ -q
```

---

## GPU Setup

For Phases 25+, install PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

For cloud GPU setup (H100/A100), see the [Colab Setup Guide](COLAB_SETUP.md).
