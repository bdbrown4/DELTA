# DELTA Research: Phase 54–57 Index

> **Status:** Phases 54–57 all complete. Brain architecture validated.

## Quick Navigation

### Phase results
→ [docs/phase_55.md](docs/phase_55.md) — Brain architecture port (PARTIAL: MRR 0.4773, H@10 +3.7%)
→ [docs/phase_56.md](docs/phase_56.md) — Constructor density ablation (d=0.01 dominates d=0.02)
→ [docs/phase_57.md](docs/phase_57.md) — Brain temperature annealing (baseline optimal, MRR 0.4808–0.4818)

### Setup and execution
→ [COLAB_SETUP.md](COLAB_SETUP.md) — Google Colab setup guide
→ [LOCAL_EXECUTION.md](LOCAL_EXECUTION.md) — Local CLI execution guide

### Architecture and vision
→ [mkdocs-src/the-brain.md](mkdocs-src/the-brain.md) — Five horizons roadmap
→ [mkdocs-src/architecture.md](mkdocs-src/architecture.md) — Architecture overview (includes BrainEncoder)

### Full history
→ [research_state.json](research_state.json) — Complete experiment log

---

## Phase Summary

| Phase | Title | Status | Result |
|-------|-------|--------|--------|
| 54 | Multi-hop high-power eval | ✅ Complete | Evaluation noise dominant; multi-hop investigation CLOSED |
| 55 | Brain architecture port | ✅ Complete (PARTIAL) | MRR 0.4773, H@10 +3.7% over delta_full |
| 56 | Constructor density ablation | ✅ Complete (PARTIAL) | d=0.01 dominates d=0.02; MRR 0.4794 |
| 57 | Brain temperature annealing | ✅ Complete (PARTIAL) | MRR 0.4808–0.4818; annealing counterproductive |

## Key Files

| File | Purpose |
|------|---------|
| `delta/brain.py` | BrainEncoder + BrainConstructor implementation |
| `experiments/phase55_brain_port.py` | Brain architecture LP validation |
| `experiments/phase56_density_ablation.py` | Constructor density ablation |
| `experiments/phase57_brain_temp_anneal.py` | Brain temperature annealing |
| `docs/phase_55.md` | Phase 55 detailed results |
| `docs/phase_56.md` | Phase 56 detailed results |
| `docs/phase_57.md` | Phase 57 detailed results |

## Decision Rationale

### Why BrainEncoder (Option B) Instead of Constructor Fixes Alone (Option A)?
- Fixing constructor in isolation is hygiene that doesn't advance the Brain
- Porting self-bootstrap to core forces constructor deficiencies to be fixed during integration
- Result: Gumbel-sigmoid + density control validated on real architecture

### Why This Memory Profile?
- **Once-per-epoch encoding:** Avoids 19x per-batch rebuilds (gradient graph explosion)
- **Top-k selection:** Guarantees controlled edge count (scales linearly with N)
- **Gradient-efficient:** Selection in no-grad, gradients flow only through selected edges
