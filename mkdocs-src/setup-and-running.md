# Getting Started

## Requirements

- Python 3.10+
- PyTorch 2.0+ (GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu124`)
- CPU sufficient for Phases 1-24; GPU (6 GB+ VRAM) recommended for Phase 25+

## Local Setup

```bash
pip install -r requirements.txt
```

---

## Running Experiments

### Phase 1-15: Core Validation

```bash
python -m experiments.phase1_edge_attention       # Edge vs node attention
python -m experiments.phase2_dual_attention        # Sequential vs dual parallel
python -m experiments.phase3_router                # Sparsity vs accuracy
python -m experiments.phase4_memory                # Tiered memory recall
python -m experiments.phase5_construction          # Graph construction
python -m experiments.phase6_full_model            # End-to-end integration
python -m experiments.phase7_gumbel_routing        # Differentiable routing
python -m experiments.phase8_scaling               # Scaling analysis
python -m experiments.phase9_multi_hop             # Multi-hop reasoning
python -m experiments.phase10_analogy              # Analogical reasoning
python -m experiments.phase11_multi_hop_edges      # Multi-hop edge adjacency
python -m experiments.phase12_curriculum_routing    # Gumbel curriculum
python -m experiments.phase13_harder_benchmarks     # Compositional reasoning
python -m experiments.phase14_contrastive_analogy   # Contrastive training
python -m experiments.phase15_kg_benchmark          # Synthetic KG benchmark
```

### Phase 16-24: Fixes & Scale Validation

```bash
python -m experiments.phase16_post_attention_pruning   # Post-attn vs pre-attn routing
python -m experiments.phase17_sparse_multi_hop         # Sparse COO scaling
python -m experiments.phase18_variational_memory       # Variational compression
python -m experiments.phase19_per_layer_constructor    # Per-layer edge projections
python -m experiments.phase20_bfs_partition            # BFS partition scaling
python -m experiments.phase21_learned_attention_dropout # Learned dropout
python -m experiments.phase22_scale_stress_test        # N=1000, 15% noise
python -m experiments.phase23_realistic_kg_benchmark   # vs TransE/RotatE/CompGCN
python -m experiments.phase24_combined_integration     # All fixes at scale
```

### Phase 25-37: Real Data & GPU

```bash
python experiments/phase25_fb15k237_gpu.py             # Real FB15k-237 on GPU
python experiments/phase26_adaptive_hop_depth.py       # Adaptive multi-hop depth
python experiments/phase27b_bootstrap_batched.py       # Bootstrap relational (corrected)
python experiments/phase28_hard_ablation.py            # Hard ablation benchmark
python experiments/phase29_multi_seed.py               # Multi-seed evaluation (5 seeds)
python experiments/phase30_edge_adj_sampling.py        # GPU edge adj sampling
python experiments/phase31_mini_batching.py            # Full FB15k-237 mini-batching
python experiments/phase32_cross_graph_transfer.py     # FB15k-237 -> WN18RR
python experiments/phase33_task_aware_construction.py  # Hybrid constructor
python experiments/phase34_graphgps_grit_comparison.py # DELTA vs GraphGPS vs GRIT
python experiments/phase35_relational_transfer.py      # Domain-agnostic transfer
python experiments/phase36_task_aware_at_scale.py      # Constructor at scale
python experiments/phase37_real_comparison.py           # Parameter-matched comparison
```

### Phase 38-49: Construction & Temperature

```bash
python experiments/phase46_differentiable_constructor.py  # Phase 38: Differentiable constructor
python experiments/phase46b_self_bootstrapped.py          # Phase 39: Self-bootstrapped DELTA
python experiments/phase46c_link_prediction.py            # Phase 40: Correct LP evaluation
python experiments/phase41_generalization_gap.py          # Phase 41: Weight decay investigation
python experiments/phase42_multihop.py                    # Phase 42: Multi-hop 1p/2p/3p
python experiments/phase43_regularization.py              # Phase 43: DropEdge robustness
python experiments/phase44_depth.py                       # Phase 44: Extended depth 4p/5p
python experiments/phase45_inference_timing.py            # Phase 45: Inference + multi-seed
python experiments/phase46_capacity_signal.py             # Phase 46: Learnable temperature
python experiments/phase47_layer_specific_temp.py         # Phase 47: Layer-specific temp
python experiments/phase48_asymmetric_temp.py             # Phase 48: Asymmetric node/edge temp
python experiments/phase49_l0_temp.py --epochs 500        # Phase 49: L0 temp + asymmetric L1+L2
```

---

## Running Tests

```bash
# Run all tests (44/44 should pass)
python -m pytest tests/ -q
```

---

## Cloud GPU Setup (Colab / RunPod)

### Google Colab Pro+

For Phases 25+ and full-scale experiments. **Colab Pro+** ($49.99/month) gives access to H100 (80GB) and A100 (40GB) GPUs.

1. Subscribe at [colab.research.google.com](https://colab.research.google.com)
2. Runtime -> Change runtime type -> GPU -> **H100** (first choice) or **A100**
3. Verify: `!nvidia-smi`

```python
# Clone and install
!git clone https://github.com/bdbrown4/DELTA.git
%cd DELTA
!pip install torch>=2.0.0 numpy>=1.24.0

# Verify
!python -c "from delta import DELTAModel; print('DELTA ready')"

# Run tests
!python -m pytest tests/ -q

# Run experiments
!python experiments/phase31_mini_batching.py --full --epochs 50
```

### RunPod / vast.ai

For longer runtimes or 80GB A100/H100:

- **RunPod** ([runpod.io](https://runpod.io)): ~$1.50/hr for A100 80GB. Deploy GPU Pod -> Select PyTorch template -> SSH in.
- **vast.ai** ([vast.ai](https://vast.ai)): ~$0.80-2/hr for A100. Cheapest option for burst compute.

### Estimated GPU Times

| Experiment | Est. Time (H100) |
|-----------|-------------------|
| Phase 34 (synthetic, 5 seeds) | 5-8 min |
| Phase 31 (full FB15k-237, 50 epochs) | ~3.7 hours |
| Phase 32 (cross-domain, 100 epochs) | ~2-4 hours |
| Phase 42-45 (multi-hop + timing) | ~2-3 hours |
