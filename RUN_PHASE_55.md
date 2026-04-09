# Phase 55: How to Run

You have **two options** to run Phase 55. Choose based on your local setup.

## Option 1: Local CLI (Recommended if you have local GPU)

**Best for:** You have a GPU on your machine and want full control.

```bash
cd c:\dev\DELTA
python phase55_colab_launcher.py --seeds 42 --epochs 150 --eval_every 30
```

Results stream to your terminal and save to `phase55_output.json`.

📖 **Full guide:** [LOCAL_EXECUTION.md](LOCAL_EXECUTION.md)

**Pros:**
- ✅ No latency, results available immediately
- ✅ Full control over parameters
- ✅ Can monitor GPU/CPU in real-time
- ✅ No Colab account needed

**Cons:**
- ❌ Requires local GPU (or CPU, but slow)
- ❌ Ties up your machine during execution

---

## Option 2: Google Colab (Free GPU if no local hardware)

**Best for:** You don't have a GPU or want to free up your machine.

1. Open https://colab.research.google.com
2. Create a new notebook
3. In the first cell, paste:
```python
!git clone https://github.com/bdbrown4/DELTA.git /content/DELTA
%cd /content/DELTA
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. In the second cell, paste:
```python
!python phase55_colab_launcher.py --seeds 42 --epochs 150 --eval_every 30 --target_density 0.02
```

📖 **Full guide:** [COLAB_SETUP.md](COLAB_SETUP.md)

**Pros:**
- ✅ Free A100 GPU access (if eligible)
- ✅ No local hardware needed
- ✅ Frees up your machine for other tasks

**Cons:**
- ❌ Depends on Google's quotas
- ❌ Slightly slower (network latency)
- ❌ Requires Google account

---

## Quick Decision Matrix

| Scenario | Recommendation |
|----------|---|
| I have a local GPU | **Local CLI** — fastest & simplest |
| I only have CPU | **Colab GPU** — much faster |
| I don't want to tie up my machine | **Colab** — run in background |
| I want full control over everything | **Local CLI** — direct execution |

---

## Running Both (Recommended for validation)

You can run Phase 55 on both local and Colab to compare results:

```bash
# Local (faster, immediate results)
python phase55_colab_launcher.py --seeds 42 --epochs 150 --eval_every 30

# Then later, run on Colab with additional seeds for more robust validation
# python phase55_colab_launcher.py --seeds 42,123,456 --epochs 300 --eval_every 50
```

---

## Parameters Explained

- `--seeds 42` — Use seed 42 (reproducible). Use `42,123,456` for 3 runs.
- `--epochs 150` — Train for 150 epochs. Increase for better convergence.
- `--eval_every 30` — Evaluate validation set every 30 epochs.
- `--batch_size 512` — Batch size (reduce if out of memory).
- `--target_density 0.02` — Graph sparsity (2% density). Lower = faster but less learning.
- `--models brain_hybrid,delta_full` — Which models to train.

---

## What Gets Produced

After execution:

- **phase55_output.json** — Full results with MRR, Hits@1/3/10 for each model
- **phase55_run.log** — (if you used `tee`) Console output captured to file

Example results structure:
```json
{
  "summary": {
    "delta_full": {
      "best_val_mrr": 0.485,
      "test_mrr": 0.492,
      "hits_at_1": 0.345
    },
    "brain_hybrid": {
      "best_val_mrr": 0.478,
      "test_mrr": 0.485,
      "hits_at_1": 0.340
    }
  }
}
```

---

## Troubleshooting

**Local: "Out of memory"**
```bash
python phase55_colab_launcher.py --batch_size 256 --target_density 0.01
```

**Local: "Module not found"**
```bash
pip install -r requirements.txt
```

**Colab: "Runtime disconnected"**
- Colab times out after 12 hours. Reduce `--epochs` or run with fewer seeds.

**Colab: "CUDA out of memory"**
- Colab's A100 has 40GB. Use `--target_density 0.01` or `--batch_size 256`.

---

## Monitoring Execution

### Local with real-time log

```bash
# Terminal 1: Run the experiment
python phase55_colab_launcher.py --seeds 42 --epochs 150 | tee phase55_run.log

# Terminal 2: Monitor in real-time
Get-Content phase55_run.log -Wait  # PowerShell
# or
tail -f phase55_run.log  # Git Bash
```

### Colab
- Notebook cell output updates live
- GPU memory visible in system monitor

---

## Next Steps

After Phase 55 completes:

1. Review `phase55_output.json` results
2. If MRR >= 0.475 on `brain_hybrid`, the architecture is validated
3. If >= 0.490, it's competitive with temperature-tuned DELTA-Full
4. Proceed to Phase 56: Scale to full FB15k-237 (if needed)

---

Questions? See:
- [LOCAL_EXECUTION.md](LOCAL_EXECUTION.md) for detailed local setup
- [COLAB_SETUP.md](COLAB_SETUP.md) for detailed Colab setup
- [PHASE_55_README.md](PHASE_55_README.md) for technical details
