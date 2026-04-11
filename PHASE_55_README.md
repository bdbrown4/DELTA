# Phase 55: Brain Architecture Port

> **Status: COMPLETE (PARTIAL)** — See [docs/phase_55.md](docs/phase_55.md) for full results.

## Result

BrainEncoder achieves LP MRR **0.4773** on FB15k-237 (misses 0.475 threshold by 0.002) but delivers **+3.7% H@10** over delta_full (0.7973 vs 0.7603). First successful integration of differentiable graph construction with DELTA.

## Hypothesis

BrainEncoder (differentiable self-bootstrap graph construction) achieves LP MRR >= 0.475 on FB15k-237.

## How to Run

```bash
python experiments/phase55_brain_port.py \
    --seeds 42 \
    --epochs 150 \
    --eval_every 30 \
    --batch_size 512 \
    --target_density 0.02
```

## Key Files

- [experiments/phase55_brain_port.py](experiments/phase55_brain_port.py) — Full experiment harness
- [delta/brain.py](delta/brain.py) — BrainEncoder + BrainConstructor implementation
- [docs/phase_55.md](docs/phase_55.md) — Detailed results and analysis

## Follow-Up

- **Phase 56:** Constructor density ablation — d=0.01 strictly dominates d=0.02
- **Phase 57:** Brain temperature annealing — baseline (no annealing) is optimal
