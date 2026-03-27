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

**Recommendation:** Colab Pro+ ($49.99/month) is the right choice for DELTA. The A100 40GB provides:
- **Synthetic benchmarks (Phase 34):** Run immediately — all three architectures scale to any synthetic graph size.
- **Full FB15k-237 (Phase 31+):** 14,505 entities. All three architectures (DELTA, GraphGPS, GRIT) use `F.scaled_dot_product_attention` for automatic FlashAttention-2 / memory-efficient dispatch on A100, enabling full-scale runs without subsampling.
- **WN18RR (Phase 32):** 40,943 entities — requires mini-batching from Phase 31. All models scale with subgraph sampling + gradient accumulation.

---

## Step 1: Subscribe to Google Colab Pro+

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Click the **gear icon** (⚙️) in the top-right → **Colab Pro**
3. Select **"Colab Pro+"** plan ($49.99/month)
4. Complete payment with Google account
5. You can cancel anytime — charges are monthly with no commitment

### Verify GPU access
After subscribing, in any notebook run:
```python
!nvidia-smi
```
You should see an **A100-SXM4-40GB** (or similar A100 variant).

---

## Step 2: Set Up Runtime

1. Open a new Colab notebook
2. Go to **Runtime → Change runtime type**
3. Select:
   - **Hardware accelerator:** GPU
   - **GPU type:** A100 (available with Pro+)
   - **High-RAM:** Enabled
4. Click **Save**

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

All 40 tests should pass (24 original + 16 baseline).

---

## Step 5: Run Phase 34 (GraphGPS / GRIT Comparison)

### Quick run (synthetic data, ~5 minutes)
```python
!python experiments/phase34_graphgps_grit_comparison.py --seeds 3 --epochs 200
```

### Full run (5 seeds, more epochs, ~15 minutes)
```python
!python experiments/phase34_graphgps_grit_comparison.py --seeds 5 --epochs 500
```

---

## Step 6: Run Full-Scale Experiments (Phase 31+)

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

| Experiment | Data Size | Est. Time (A100) | Compute Units |
|-----------|-----------|-------------------|---------------|
| Phase 34 (synthetic, 3 seeds) | ~100 nodes | 5 min | < 1 |
| Phase 34 (synthetic, 5 seeds) | ~100 nodes | 15 min | < 1 |
| Phase 31 (full FB15k-237) | 14,505 entities | 2-4 hours | 5-10 |
| Phase 34b (full FB15k-237, 3 models × 5 seeds) | 14,505 entities | 6-12 hours | 15-25 |
| Phase 32 (cross-domain WN18RR) | 40,943 entities | 4-8 hours | 10-20 |

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
