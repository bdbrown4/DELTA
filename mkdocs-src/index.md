# DELTA: Dynamic Edge-and-Node Architecture with Layered, Tiered Attention

A research implementation of the DELTA architecture — building toward **[The Brain](the-brain.md)**: a system that dynamically constructs its own relational graphs and reasons over them, without pre-defined topology or transformer scaffolding.

---

## Core Thesis

> Reality is a graph. Language is a lossy compression of reality into sequences. Transformers pay a quadratic tax to rediscover structure. DELTA operates on relational structure directly — and now constructs that structure from scratch.

**The three-paradigm gap:** GNN edges are passive scalar wires; DELTA edges are first-class computational citizens that attend to each other. That edge-to-edge attention produces the Phase 28 **+24% noise robustness gap** — and Phase 39's self-bootstrapped DELTA proves the system can build its own graph without any transformer scaffold.

**Current evidence base:** 40 experiment phases, 44 unit tests, competitive link prediction on FB15k-237 (MRR 0.497, still converging), synthetic task dominance over GraphGPS/GRIT. Self-bootstrapped DELTA at **157% of FixedChain** (Phase 39). Cross-domain transfer: 0.961 on WN18RR with 100 samples (frozen encoder).

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Experiment phases | 40 (Phases 1–40) |
| Unit tests | 44/44 passing |
| Self-bootstrap breakthrough | **0.757 ± 0.041** — 157% of FixedChain (Phase 39) |
| Link prediction (FB15k-237) | MRR **0.497** at 200 epochs, still converging (Phase 40) |
| Cross-domain transfer | **0.961** on WN18RR (100 samples, frozen encoder) |
| Synthetic dominance | DELTA **0.880** vs GraphGPS 0.293 vs GRIT 0.307 (edge classification) |
| Noise robustness | **+24%** over vanilla GNN at 80% feature corruption |

---

## Documentation Guide

### Vision & Architecture

- **[The Brain: End Goal](the-brain.md)** — The long-term vision: dynamic graph construction → relational reasoning → brain-like computation
- **[Architecture Overview](architecture.md)** — Core thesis, architecture diagram, and how DELTA differs from Transformers and GNNs
- **[Visual Explainer](ARCHITECTURE_VISUAL.md)** — Interactive three-paradigm comparison (Transformer → GNN → DELTA)
- **[Bootstrap Strategy](bootstrap-strategy.md)** — From transformer scaffolding to full self-bootstrap
- **[Architecture Evolution](architecture-evolution.md)** — Six development stages from core validation to real-world data

### Experiment Results

- **[Validation Phases](validation-phases.md)** — All phase result tables (Phases 1–40)
- **[Key Findings](key-findings.md)** — Key findings organized by research stage
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

*DELTA architecture — conceived March 25, 2026. 40 experiment phases, 6 architectural fixes, 44 unit tests. Self-bootstrapped DELTA validated (Phase 39). Correct link prediction evaluation in progress (Phase 40). See [The Brain](the-brain.md) for the long-term vision.*
