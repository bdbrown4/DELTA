# DELTA: Dynamic Edge-and-Node Architecture with Layered, Tiered Attention

A research implementation of the DELTA architecture — building toward **[The Brain](the-brain.md)**: a system that dynamically constructs its own relational graphs and reasons over them, without pre-defined topology or transformer scaffolding.

---

## Core Thesis

> Reality is a graph. Language is a lossy compression of reality into sequences. Transformers pay a quadratic tax to rediscover structure. DELTA operates on relational structure directly — and now constructs that structure from scratch.

**The three-paradigm gap:** GNN edges are passive scalar wires; DELTA edges are first-class computational citizens that attend to each other. That edge-to-edge attention produces the Phase 28 **+24% noise robustness gap** — and Phase 39's self-bootstrapped DELTA proves the system can build its own graph without any transformer scaffold.

**Current evidence base:** 48 experiment phases, 44 unit tests, competitive link prediction on FB15k-237, multi-hop compositional reasoning dominance. DELTA-Matched is the **only model that improves with reasoning depth** (Phase 44: 5p MRR 0.790 vs GraphGPS 0.690). Asymmetric temperature tuning achieves LP MRR **0.4856** (Phase 48). Self-bootstrapped DELTA at **157% of FixedChain** (Phase 39).

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Experiment phases | 48 (Phases 1–48) |
| Unit tests | 44/44 passing |
| Best LP MRR (DELTA-Full, temp-tuned) | **0.4856** — asymmetric node/edge temperature (Phase 48) |
| Multi-hop champion | DELTA-Matched 3p MRR **0.742 ± 0.009** — only model improving with depth (Phase 45, 3-seed) |
| Depth scaling | 5p MRR **0.790** vs GraphGPS 0.690 — advantage doubles per hop (Phase 44) |
| Self-bootstrap breakthrough | **0.757 ± 0.041** — 157% of FixedChain (Phase 39) |
| Inference speed | Per-query **0.8–0.9×** GraphGPS (faster) despite 34× training cost (Phase 45) |
| Noise robustness | **+24%** over vanilla GNN at 80% feature corruption (Phase 28) |

---

## Documentation Guide

### Vision & Architecture

- **[The Brain: End Goal](the-brain.md)** — The long-term vision: dynamic graph construction → relational reasoning → brain-like computation
- **[Architecture Overview](architecture.md)** — Core thesis, architecture diagram, and how DELTA differs from Transformers and GNNs
- **[Visual Explainer](ARCHITECTURE_VISUAL.md)** — Interactive three-paradigm comparison (Transformer → GNN → DELTA)
- **[Bootstrap Strategy](bootstrap-strategy.md)** — From transformer scaffolding to full self-bootstrap
- **[Architecture Evolution](architecture-evolution.md)** — Six development stages from core validation to real-world data

### Experiment Results

- **[Validation Phases](validation-phases.md)** — All phase result tables (Phases 1–48)
- **[Key Findings](key-findings.md)** — 29 key findings organized by research stage
- **[Colab Results](COLAB_RESULTS.md)** — Phase 35/36/37 results from Colab Pro+ runs
- **[Backward Compatibility](backward-compatibility.md)** — Verification that architectural fixes don't break prior results

### Research & Publication

- **[Research Agenda](RESEARCH_AGENDA.md)** — Validated propositions, open gaps, compute options, publication pathway
- **[Publication Roadmap](PUBLICATION_ROADMAP.md)** — Phase structure and path toward publication
- **[Research Methodology](research-methodology.md)** — AI assistance disclosure and methodology

### Status & Setup

- **[Current Status & Roadmap](status-and-roadmap.md)** — What's working, what needs proof, and next steps
- **[Getting Started](setup-and-running.md)** — Installation and experiment commands
- **[Colab Setup Guide](COLAB_SETUP.md)** — Google Colab Pro+ setup for GPU experiments
- **[Project Structure](project-structure.md)** — Repository directory layout

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ (GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu124`)
- CPU sufficient for Phases 1–24; GPU (6 GB+ VRAM) recommended for Phase 25+

---

*DELTA architecture — conceived March 25, 2026. 48 experiment phases, 6 architectural fixes, 44 unit tests. Multi-hop compositional reasoning validated (Phases 42–45). Attention temperature optimization in progress (Phases 46–49). See [The Brain](the-brain.md) for the long-term vision.*
