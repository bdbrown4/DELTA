# Phase 55: Brain Architecture Port — Google Colab Setup Guide

## Why Colab for Phase 55?

Local 12 GB GPU hits memory fragmentation and timing bottlenecks with the BrainConstructor's Gumbel-sigmoid edge selection creating 12K-edge augmented graphs. Colab's T4 GPU (15 GB) provides:

- **Fresh CUDA state** per run (no fragmentation)
- **Sufficient memory** for 2% target density (2,435 edges safely)
- **Fast for validation**: ~30 min for 1 seed × 150 epochs
- **Easy reproducibility**: Upload results to Drive

## Step 1: Open Colab and Clone DELTA

```python
# Cell 1: Install and clone
!git clone https://github.com/bdbrown4/DELTA.git /content/DELTA
%cd /content/DELTA
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify:
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## Step 2: Quick Validation (Recommended First Run)

```bash
# Cell 2: Run Phase 55 with minimal overhead
# 1 seed, 150 epochs, ~30-40 minutes
!python phase55_colab_launcher.py \
    --seeds 42 \
    --epochs 150 \
    --eval_every 30 \
    --target_density 0.02
```

Expected output:
```
✅ DELTA PHASE 55: Brain Architecture Port — COLAB EXECUTION
✅ GPU: Tesla T4
✅ GPU Memory: 15.0 GB

... training progress ...

PHASE 55 SUMMARY
================
brain_hybrid:      Best Val MRR: 0.4750+ | Test MRR: 0.4750+
delta_full:        Best Val MRR: 0.4720+ | Test MRR: 0.4720+
```

**Success criteria:**
- `brain_hybrid` ≥ 0.475 MRR → **Hypothesis CONFIRMED**
- No OOM errors
- Completes in <1 hour

## Step 3: Save Results to Google Drive (Optional)

```python
# Cell 3: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 4: Copy results
import shutil
import os

os.makedirs('/content/drive/MyDrive/DELTA', exist_ok=True)
shutil.copy(
    '/content/DELTA/phase55_output.json',
    '/content/drive/MyDrive/DELTA/phase55_output.json'
)
print("✅ Results saved to Google Drive/DELTA/")
```

## Step 4: Full Validation (Optional, 3-4 hours)

For publication-grade results, run 3 seeds:

```bash
!python phase55_colab_launcher.py \
    --seeds 42,123,456 \
    --epochs 300 \
    --eval_every 50 \
    --target_density 0.02
```

This gives:
- Robust mean ± std across 3 seeds
- Proper statistical validation
- Total runtime: 3-4 hours

## Step 5: Inspect Results

```python
# Cell 5: Load and display results
import json

with open('/content/DELTA/phase55_output.json') as f:
    results = json.load(f)

# Summary statistics
for model, data in results['summary'].items():
    if isinstance(data, dict):
        print(f"\n{model}:")
        print(f"  Params: {data.get('params', 'N/A')}")
        print(f"  Best Val MRR: {data['best_val_mrr']:.4f}")
        if 'test_mrr' in data:
            print(f"  Test MRR: {data['test_mrr']:.4f}")
        if 'std' in data:
            print(f"  Std (across seeds): {data['std']:.4f}")

# Hypothesis verdict
print("\n" + "="*60)
baseline_a = 0.4744
brain_mrr = results['summary']['brain_hybrid'].get('test_mrr', 0)

if brain_mrr >= 0.475:
    print(f"✅ HYPOTHESIS CONFIRMED: brain_hybrid ({brain_mrr:.4f}) ≥ 0.475")
else:
    print(f"❌ HYPOTHESIS REJECTED: brain_hybrid ({brain_mrr:.4f}) < 0.475")

if brain_mrr > baseline_a:
    delta = brain_mrr - baseline_a
    pct = (delta / baseline_a) * 100
    print(f"✅ Improvement over baseline A: +{delta:.4f} ({pct:.1f}%)")
else:
    delta = baseline_a - brain_mrr
    pct = (delta / baseline_a) * 100
    print(f"❌ Degradation vs baseline A: -{delta:.4f} ({pct:.1f}%)")
```

## Troubleshooting

### OOM: "CUDA out of memory"
- Reduce `--target_density` to 0.01 (1% instead of 2%)
- Reduce `--epochs` to 100
- Use one model: `--models brain_hybrid`

### Timeout: Runs >1 hour with 150 epochs
- Check if you're on CPU instead of GPU: `torch.cuda.is_available()`
- Reduce `--epochs` to 100
- Reduce `--batch_size` to 256 (slower, uses less memory per step)

### Results don't save to Drive
- Make sure you mounted Drive before running Phase 55
- Check `/content/DELTA/phase55_output.json` exists locally first

### "ImportError: cannot import name 'FB15k237'"
- Make sure git clone got the entire repo: `!ls -la /content/DELTA/data/`
- If data/ is missing, pull it: `!git lfs pull` (requires LFS)

## Key Files

- `phase55_colab_launcher.py` — Colab entry point (this file runs the actual experiment)
- `experiments/phase55_brain_port.py` — Full Phase 55 experiment script
- `delta/brain.py` — BrainEncoder and BrainConstructor classes
- `phase55_output.json` — Results after run (locally or in Drive)

## Next Steps (After Phase 55)

If Phase 55 succeeds (brain_hybrid ≥ 0.475):

1. **Phase 56: Iterative Refinement** — Multi-pass reason→reconstruct→reason loops
2. **Phase 57: Sequence Domain** — Self-bootstrap on LRA ListOps (non-relational inputs)
3. **Phase 58: Adaptive Pruning** — Use capacity signals from Phase 46-52 to learn pruning

If Phase 55 fails:

1. Debug the BrainConstructor (check edge selection gradients)
2. Try `brain_pure` mode (no hybrid) to isolate the issue
3. Consider reducing model capacity (d_node=32, num_heads=2) to speed up convergence
