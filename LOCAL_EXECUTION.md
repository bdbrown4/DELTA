# Local Execution Guide

Run DELTA experiments directly from your machine's CLI without needing Colab.

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ (CPU or GPU)
- The DELTA repository cloned

## Quick Start

### 1. Install dependencies (one-time)

```bash
cd DELTA
pip install -r requirements.txt
```

For GPU support with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. Run an experiment

```bash
# Core validation (CPU OK)
python -m experiments.phase1_edge_attention

# Brain architecture (GPU recommended)
python experiments/phase55_brain_port.py --seeds 42 --epochs 150 --eval_every 30

# Run all tests
python -m pytest tests/ -q  # 44/44 should pass
```

Results stream to your terminal and save to output JSON files.

## Optional: Save output to file while monitoring

```bash
python phase55_colab_launcher.py --seeds 42 --epochs 150 --eval_every 30 --target_density 0.02 | tee phase55_run.log
```

Then in another terminal, monitor progress:
```bash
# PowerShell
Get-Content phase55_run.log -Wait

# Or just tail the file
tail -f phase55_run.log
```

## Available Arguments

```
--seeds SEEDS              Seeds to run (comma-separated, default: 42)
--epochs EPOCHS            Epochs to train (default: 150)
--eval_every EVAL_EVERY    Evaluate every N epochs (default: 30)
--batch_size BATCH_SIZE    Batch size (default: 512)
--lr LR                    Learning rate (default: 0.001)
--target_density TARGET_DENSITY  Constructor density (default: 0.02)
--models MODELS            Models to train (comma-separated)
```

## Examples

**Quick validation (1-2 hours with GPU)**
```bash
python phase55_colab_launcher.py --seeds 42 --epochs 150 --eval_every 30
```

**Full validation with multiple seeds (3-4 hours with GPU)**
```bash
python phase55_colab_launcher.py --seeds 42,123,456 --epochs 300 --eval_every 50
```

**Brain models only (fast)**
```bash
python phase55_colab_launcher.py --seeds 42 --epochs 150 --models brain_hybrid,brain_pure
```

**CPU-only (slower, still works)**
```bash
python phase55_colab_launcher.py --seeds 42 --epochs 100 --eval_every 25
```

## Expected Output

- **Console**: Real-time training progress with epoch metrics
- **File**: `phase55_output.json` containing full results summary
- **Timing**: Depends on hardware, typically 1-4 hours with GPU

## GPU Status Check

The script will automatically detect and report GPU availability:
```
✅ GPU: NVIDIA GeForce RTX <model>
✅ GPU Memory: <X> GB
```

If you see:
```
⚠️ WARNING: No GPU detected. Will run on CPU (slow).
```

The code will still run on CPU, just slower.

## Results

After execution, open `phase55_output.json` to see:
```json
{
  "summary": {
    "delta_full": {"best_val_mrr": 0.XXX, "test_mrr": 0.XXX},
    "brain_hybrid": {"best_val_mrr": 0.XXX, "test_mrr": 0.XXX},
    "brain_pure": {"best_val_mrr": 0.XXX, "test_mrr": 0.XXX}
  }
}
```

## Troubleshooting

**Out of memory?**
- Reduce `--batch_size` (e.g., `--batch_size 256`)
- Reduce `--epochs` 
- Reduce graph density with `--target_density 0.01`

**Script not found?**
- Make sure you're in the DELTA directory: `cd c:\dev\DELTA`
- Verify `phase55_colab_launcher.py` exists in current directory

**Import errors?**
- Run `pip install -r requirements.txt` first
- Make sure DELTA package is in Python path (should be automatic if running from DELTA directory)

## Why Local Execution?

✅ No Colab latency  
✅ Full control over parameters  
✅ Results available immediately in your filesystem  
✅ Can monitor GPU/CPU in real-time  
✅ Free GPU hours still available if you prefer Colab later
