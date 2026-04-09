# Phase 55: Brain Architecture Port

## Quick Start (Colab Recommended)

Local 12GB GPU hits memory fragmentation with BrainConstructor's augmented graphs. **Use Google Colab instead.**

### 3-Step Colab Launch

1. **Open [Google Colab](https://colab.research.google.com)**
2. **Run this in a cell:**
```python
!git clone https://github.com/bdbrown4/DELTA.git /content/DELTA
%cd /content/DELTA
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!python phase55_colab_launcher.py --seeds 42 --epochs 150 --eval_every 30
```
3. **Check results** in `/content/DELTA/phase55_output.json`

**Expected runtime:** 30-40 minutes on T4 GPU

---

## What Phase 55 Tests

**Hypothesis:** BrainEncoder (differentiable self-bootstrap graph construction) achieves LP MRR ≥ 0.475 on FB15k-237

**Models:**
- `brain_hybrid` — DELTA bootstrap → learned edges → DELTA reasoning (NEW)
- `delta_full` — baseline (fixed KG, no learning)

**Success Criteria:**
- ✅ `brain_hybrid` MRR ≥ 0.475 (baseline A = 0.4744)
- ✅ No OOM, completes in <1 hour on Colab T4

---

## Detailed Setup

See [COLAB_SETUP.md](COLAB_SETUP.md) for:
- Full step-by-step instructions
- Save results to Google Drive
- Troubleshooting (OOM, timeout, etc.)
- Full validation (3 seeds, 3-4 hours)

---

## Local Execution (Not Recommended)

If you have >16GB GPU:
```bash
python experiments/phase55_brain_port.py \
    --seeds 42 \
    --epochs 150 \
    --eval_every 30 \
    --batch_size 512 \
    --target_density 0.02 \
    --models brain_hybrid,delta_full
```

**Warning:** 12GB GPUs will likely OOM. Colab is the supported path.

---

## Key Files

- `COLAB_SETUP.md` — Complete setup guide (READ THIS FIRST)
- `phase55_colab_launcher.py` — Colab entry point
- `experiments/phase55_brain_port.py` — Full experiment harness
- `delta/brain.py` — BrainEncoder + BrainConstructor implementation

---

## After Phase 55 Succeeds

**Next phases:**
- **Phase 56:** Iterative Refinement (multi-pass reason→reconstruct→reason)
- **Phase 57:** Sequence Domain (non-relational inputs like LRA)
- **Phase 58:** Adaptive Pruning (capacity-aware architecture)

See [research_state.json](research_state.json) for full roadmap.
