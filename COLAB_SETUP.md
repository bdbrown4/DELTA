# Google Colab Pro Setup for DELTA

Instructions for setting up Google Colab Pro to run DELTA's compute-intensive experiments (Phase 31+, Phase 34 full-scale comparison).

---

## Why Colab Pro?

| Feature | Free Colab | Colab Pro ($9.99/mo) | **Colab Pro+ ($49.99/mo)** |
|---------|-----------|---------------------|--------------------------|
| GPU | T4 (15GB) | T4 / V100 (16GB) | **A100 (40GB)** |
| RAM | 12GB | 25GB | **51GB** |
| Runtime | 12h max | 24h max | **24h, background execution** |
| Priority | Low | Medium | **Highest** |

**Recommendation:** Colab Pro+ ($49.99/month) gives access to **H100 (80GB)** and **A100 (40GB)** GPUs:
- **H100 80GB (best choice):** ~2-3x faster than A100 with double the VRAM. Full FB15k-237 (14,505 entities) may fit without mini-batching. Select this if available.
- **A100 40GB (great fallback):** All three architectures (DELTA, GraphGPS, GRIT) use `F.scaled_dot_product_attention` for automatic FlashAttention-2 / memory-efficient dispatch, enabling full-scale runs.
- **WN18RR (Phase 32):** 40,943 entities — H100 80GB may handle directly; A100 40GB requires mini-batching from Phase 31.

### Available GPUs (Pro+ tier, ranked)
| GPU | VRAM | Speed vs A100 | Recommendation |
|-----|------|---------------|----------------|
| **H100** | 80GB | ~2-3x faster | **Best choice** — select this first |
| **A100** | 40GB | 1x (baseline) | Great fallback if H100 unavailable |
| L4 | 24GB | ~0.5x | Viable for synthetic experiments, tight for full-scale |
| T4 | 16GB | ~0.3x | Too small for Phase 31+; fine for Phases 1-24 |
| G4 | ~16GB | ~0.3x | Similar to T4 |

---

## Step 1: Subscribe to Google Colab Pro+

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Click the **gear icon** (⚙️) in the top-right → **Colab Pro**
3. Select **"Colab Pro+"** plan ($49.99/month)
4. Complete payment with Google account
5. You can cancel anytime — charges are monthly with no commitment

---

## Step 2: Set Up Runtime (Required Before Any GPU Code)

1. Open a new Colab notebook
2. Go to **Runtime → Change runtime type**
3. Select:
   - **Hardware accelerator:** GPU
   - **GPU type:** **H100** (first choice) or **A100** (if H100 unavailable)
   - **High-RAM:** Enabled
4. Click **Save** — this restarts the runtime

### Verify GPU access
After setting the runtime type, create a **code cell** and run:
```python
!nvidia-smi
```
You should see an **NVIDIA H100 80GB HBM3** or **A100-SXM4-40GB** (depending on which you selected).

> **If you see `nvidia-smi: command not found`:** The runtime has no GPU attached. Go back to Runtime → Change runtime type and select GPU.
>
> **If you see `!nvidia: event not found`:** You're in a terminal, not a code cell. The `!` prefix only works in notebook code cells — in a terminal, run `nvidia-smi` without `!`.

---

## Step 3: Clone and Install DELTA

```python
# Clone the repository
!git clone https://github.com/bdbrown4/DELTA.git
%cd DELTA

# Install dependencies
!pip install torch>=2.0.0 numpy>=1.24.0

# Verify installation
!python -c "from delta import DELTAModel; print('DELTA ready')"
!python -c "from delta.baselines import GraphGPSModel, GRITModel; print('Baselines ready')"
```

---

## Step 4: Run Existing Tests (Validation)

```python
# Run all tests to verify environment
!python tests/test_graph.py
!python tests/test_attention.py
!python tests/test_router.py
!python tests/test_memory.py
!python tests/test_utils.py
!python tests/test_baselines.py
```

All 44 tests should pass.

---

## Step 5: Run Phase 34 (GraphGPS / GRIT Comparison)

### Quick run (synthetic data, ~5 minutes)
```python
!python experiments/phase34_graphgps_grit_comparison.py --seeds 3 --epochs 200 --log_every 50
```

### Full run (5 seeds, more epochs, ~15 minutes)
```python
!python experiments/phase34_graphgps_grit_comparison.py --seeds 5 --epochs 500 --log_every 100
```

---

## Step 6: Run Phase 33 (Task-Aware Construction)

Phase 33 validates the hybrid graph constructor that preserves base topology while learning long-range edges (fixes the Phase 27b attention-thresholding issue). Synthetic data, runs fast.

```python
# Quick run (~1-2 min on H100)
!python experiments/phase33_task_aware_construction.py --seeds 3 --epochs 200

# Full run (~3-5 min on H100)
!python experiments/phase33_task_aware_construction.py --seeds 5 --epochs 500
```

---

## Step 7: Run Full-Scale Experiments (Phase 31+)

Phase 31 auto-detects your GPU and scales subgraph sizes accordingly:
- **H100 80GB:** 500 nodes/subgraph, batch_size=64 (auto-set)
- **A100 40GB:** 200 nodes/subgraph, batch_size=32 (auto-set)

All full-scale experiments log every epoch by default (`--log_every 1` when `--full`).

```python
# Phase 31: Full FB15k-237 mini-batching (20 epochs safe for 24h Colab limit)
# Logs every epoch so you can track progress during multi-hour runs
!python experiments/phase31_mini_batching.py --full --epochs 20

# Or manually set epochs and log interval:
!python experiments/phase31_mini_batching.py --full --epochs 50 --log_every 5
```

> **Colab Pro+ 24-hour session limit:** 50 epochs at full scale takes ~33 hours and WILL be killed. Use `--epochs 20` (~13 hours) to stay safely under the limit. Results at 20 epochs are sufficient to demonstrate the mechanism.

### Phase 34b: Full-scale GraphGPS / GRIT comparison (same FB15k-237 dataset)

Run this after Phase 31 while still on FB15k-237. It compares all three architectures (DELTA, GraphGPS, GRIT) at full scale.

```python
# Full synthetic benchmark, 3 models × 5 seeds (~15-20 min on H100)
# Phase 34 is a controlled synthetic comparison — no full-scale mode
!python experiments/phase34_graphgps_grit_comparison.py --seeds 5 --epochs 500 --log_every 100
```

### Phase 32: Cross-domain WN18RR (different dataset)

After completing both FB15k-237 runs above, switch to WN18RR:

```python
# Full-scale cross-domain transfer: FB15k-237 → WN18RR
# Logs every epoch by default when --full
!python experiments/phase32_cross_graph_transfer.py --full --epochs 100

# Or with custom epoch count and logging:
!python experiments/phase32_cross_graph_transfer.py --full --epochs 50 --log_every 5
```

The Colab-ready infrastructure script automates GPU setup and experiment execution:

```python
# Run the full Colab infrastructure script
!python notebooks/delta_colab_ready.py
```

Or use individual sections — see `notebooks/delta_colab_ready.py` for details.

---

## Cost Management Tips

1. **Disconnect when idle:** Runtime → Disconnect and delete runtime
2. **Use background execution** (Pro+ feature): Runtime → Run all, then close the tab — execution continues in the cloud
3. **Monitor usage:** Check your Colab usage at [https://colab.research.google.com/signup](https://colab.research.google.com/signup)
4. **Budget expectation:** $49.99/month gets you ~80 compute units. Phase 34 full-scale + Phase 31 should use about 30-40 units.
5. **Cancel after completing milestones:** Subscribe for 1-2 months, run Phase 31 + 34, cancel if done

---

## Expected GPU Time Estimates

| Experiment | Data Size | Est. Time (H100) | Est. Time (A100) |
|-----------|-----------|-------------------|-------------------|
| Phase 34 (synthetic, 3 seeds) | ~100 nodes | 2-3 min | 5 min |
| Phase 34 (synthetic, 5 seeds) | ~100 nodes | 5-8 min | 15 min |
| Phase 33 (synthetic, 3 seeds) | 60 nodes | 1-2 min | 3-5 min |
| Phase 33 (synthetic, 5 seeds) | 60 nodes | 3-5 min | 8-12 min |
| Phase 31 (full FB15k-237, 20 epochs) | 14,505 entities, 305K edges | **~13 hours** | ~20+ hours |
| Phase 31 (full FB15k-237, 50 epochs) | 14,505 entities, 305K edges | ~33 hours (**exceeds 24h limit**) | N/A |
| Phase 34b (full synthetic, 3 models × 5 seeds) | ~50-100 nodes | 15-20 min | 30-45 min |
| Phase 32 (cross-domain, 100 epochs) | 14,505 → 40,943 entities | **~2-4 hours** | 4-8 hours |

---

## Troubleshooting

### "GPU not available"
- Verify Pro+ subscription is active
- Try: Runtime → Change runtime type → GPU → A100
- If A100 unavailable, try again later (high demand periods)

### "Out of memory" on full FB15k-237
- Reduce batch size in experiment config
- Use gradient accumulation (built into Phase 31+ scripts)
- Try `torch.cuda.empty_cache()` between runs

### "Runtime disconnected"
- Pro+ supports background execution — your run continues even if disconnected
- Use `!nohup python experiment.py > output.log 2>&1 &` for long runs
- Check output with `!cat output.log`

### Saving results
```python
# Mount Google Drive to save results persistently
from google.colab import drive
drive.mount('/content/drive')

# Copy results to Drive
!cp -r results/ /content/drive/MyDrive/DELTA_results/
```

---

## Alternative: RunPod / vast.ai (On-Demand Burst Compute)

If you need more VRAM (80GB A100 or H100) or longer runtimes:

1. **RunPod** ([runpod.io](https://runpod.io)): ~$1.50/hr for A100 80GB
   - Create account → Deploy GPU Pod → Select PyTorch template
   - SSH in, clone repo, run experiments
   - Cost: ~$5-15 for Phase 34b full run

2. **vast.ai** ([vast.ai](https://vast.ai)): ~$0.80-2/hr for A100
   - Cheapest option for burst compute
   - Less stable but significantly cheaper

3. **TPU Research Cloud** ([sites.research.google/trc](https://sites.research.google/trc)): Free
   - Apply with research proposal + GitHub repo link
   - Best for extended compute needs after Phase 31 results exist

---

*Last updated: March 2026 — Phase 34 infrastructure ready*
