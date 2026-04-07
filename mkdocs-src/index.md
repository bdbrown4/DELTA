# DELTA: Dynamic Edge-and-Node Architecture with Layered, Tiered Attention

A research implementation of the DELTA architecture -- building toward **[The Brain](the-brain.md)**: a system that dynamically constructs its own relational graphs and reasons over them, without pre-defined topology or transformer scaffolding.

---

## Core Thesis

> Reality is a graph. Language is a lossy compression of reality into sequences. Transformers pay a quadratic tax to rediscover structure. DELTA operates on relational structure directly -- and now constructs that structure from scratch.

**The three-paradigm gap:** GNN edges are passive scalar wires; DELTA edges are first-class computational citizens that attend to each other. That edge-to-edge attention produces the Phase 28 **+24% noise robustness gap** -- and Phase 39's self-bootstrapped DELTA proves the system can build its own graph without any transformer scaffold.

**Current evidence base:** 48 completed experiment phases (Phase 49 active), 44 unit tests, competitive link prediction on FB15k-237, multi-hop compositional reasoning dominance.

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Experiment phases | 49 active (Phases 1-48 complete) |
| Unit tests | 44/44 passing |
| Best LP MRR (DELTA-Full, temp-tuned) | **0.4856** -- asymmetric node/edge temperature (Phase 48) |
| Multi-hop champion | DELTA-Matched 3p MRR **0.742 +/- 0.009** -- only model improving with depth (Phase 45, 3-seed) |
| Depth scaling | 5p MRR **0.790** vs GraphGPS 0.690 -- advantage doubles per hop (Phase 44) |
| Self-bootstrap breakthrough | **0.757 +/- 0.041** -- 157% of FixedChain (Phase 39) |
| Inference speed | Per-query **0.8-0.9x** GraphGPS (faster) despite 34x training cost (Phase 45) |
| Noise robustness | **+24%** over vanilla GNN at 80% feature corruption (Phase 28) |

---

## Documentation

### Architecture

- **[Architecture Overview](architecture.md)** -- Components, self-bootstrap, development timeline, backward compatibility
- **[Visual Explainer](ARCHITECTURE_VISUAL.md)** -- Interactive three-paradigm comparison (Transformer -> GNN -> DELTA)
- **[The Brain: End Goal](the-brain.md)** -- Long-term vision, capacity paradox, roadmap horizons

### Results

- **[Key Findings](key-findings.md)** -- 29 findings organized by research stage
- **[Validation Phases](validation-phases.md)** -- All phase result tables (Phases 1-48)

### Planning

- **[Status & Roadmap](status-and-roadmap.md)** -- What's validated, open gaps, roadmap, publication pathway
- **[Research Methodology](research-methodology.md)** -- AI assistance disclosure

### Setup

- **[Getting Started](setup-and-running.md)** -- Installation, experiment commands, cloud GPU setup
- **[Project Structure](project-structure.md)** -- Repository directory layout

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ (GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu124`)
- CPU sufficient for Phases 1-24; GPU (6 GB+ VRAM) recommended for Phase 25+

---

*DELTA architecture -- conceived March 25, 2026. Phase 49 active, 6 architectural fixes, 44 unit tests. Three constructor deficiencies identified (gradient wall, dead edge_type_weights, no token clustering) -- see [Architecture Overview](architecture.md#graph-constructor). See [The Brain](the-brain.md) for the long-term vision.*
