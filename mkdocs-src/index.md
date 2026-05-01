# DELTA: Dual Edge-Linked Transformer Architecture

A research implementation of the DELTA architecture — building toward **[The Brain](the-brain.md)**: a system that dynamically constructs its own relational graphs and reasons over them, without pre-defined topology or transformer scaffolding.

---

## Core Thesis

> Reality is a graph. Language is a lossy compression of reality into sequences. Transformers pay a quadratic tax to rediscover structure. DELTA operates on relational structure directly — and now constructs that structure from scratch.

**The three-paradigm gap:** GNN edges are passive scalar wires; DELTA edges are first-class computational citizens that attend to each other. That edge-to-edge attention produces the Phase 28 **+24% noise robustness gap** — and Phase 39's self-bootstrapped DELTA proves the system can build its own graph without any transformer scaffold.

**Current evidence base:** 63 completed experiment phases, 44 unit tests, competitive link prediction on FB15k-237, multi-hop compositional reasoning dominance, validated differentiable graph construction via BrainEncoder (Phases 55–58), and scaling evaluation with subsampling ablation at N=5000 (Phases 59–63).

---

## Quick Stats

| Metric | Value | Phase |
|--------|-------|-------|
| Experiment phases | 63 complete | 1–63 |
| Unit tests | 44/44 passing | — |
| Best LP MRR (DELTA-Full, temp-tuned) | **0.4905** | 52 |
| Brain LP MRR (self-constructed graph) | **0.4818** with H@10 **0.8076** | 57 |
| Multi-hop champion | DELTA-Matched 3p MRR **0.742 +/- 0.009** | 45 (3-seed) |
| Depth scaling | 5p MRR **0.790** vs GraphGPS 0.690 | 44 |
| Self-bootstrap breakthrough | **0.757 +/- 0.041** — 157% of FixedChain | 39 |
| Inference speed | Per-query **0.8–0.9x** GraphGPS (faster) | 45 |
| Noise robustness | **+24%** over vanilla GNN at 80% corruption | 28 |

---

## Documentation

### Start Here

- **[DELTA, Explained for a Software Engineer](explainer.md)** — The 30-second version, the core argument, layered evidence, and why this matters beyond the immediate result

### Architecture

- **[Architecture Overview](architecture.md)** — Components, self-bootstrap, development timeline, backward compatibility
- **[Visual Explainer](ARCHITECTURE_VISUAL.md)** — Interactive three-paradigm comparison (Transformer -> GNN -> DELTA)
- **[The Brain: End Goal](the-brain.md)** — Long-term vision, capacity paradox, roadmap horizons

### Results

- **[Key Findings](key-findings.md)** — 44 findings organized by research stage
- **[Validation Phases](validation-phases.md)** — All phase result tables (Phases 1–63)

### Planning

- **[Status & Roadmap](status-and-roadmap.md)** — What's validated, open gaps, roadmap, publication pathway
- **[Research Methodology](research-methodology.md)** — AI assistance disclosure

### Setup

- **[Getting Started](setup-and-running.md)** — Installation, experiment commands, cloud GPU setup
- **[Project Structure](project-structure.md)** — Repository directory layout

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ (GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu124`)
- CPU sufficient for Phases 1-24; GPU (6 GB+ VRAM) recommended for Phase 25+

---

*DELTA architecture — conceived March 25, 2026. 63 experiment phases, 6 architectural fixes, 44 unit tests. KG scaling evaluation complete (Phases 59–63). Pivoting to sparse attention and sequence domains. See [The Brain](the-brain.md) for the long-term vision.*
