# DELTA: Dynamic Edge-and-Node Architecture with Layered, Tiered Attention

A research implementation of the DELTA architecture — a novel AI framework that operates on dynamic graphs with dual parallel attention across nodes and edges, tiered memory, and a learned importance router.

---

## Core Thesis

> Reality is a graph. Language is a lossy compression of reality into sequences. Transformers reconstruct relational structure from flat sequences. DELTA operates on relational structure directly.

**The three-paradigm gap:** GNN edges are passive scalar wires; DELTA edges are first-class computational citizens that attend to each other. That edge-to-edge attention is what produces the Phase 28 **+24% noise robustness gap**.

**Current evidence base:** 37 experiment phases, 44 unit tests, real FB15k-237 results (97.4% ± 0.1% over 5 seeds), synthetic task superiority over GraphGPS/GRIT. Cross-domain transfer: 0.961 on WN18RR with 100 samples (frozen encoder).

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Experiment phases | 37 (Phases 1–30, 27b, 31–37) |
| Unit tests | 44/44 passing |
| Best real-data result | DELTA+Gate **97.4% ± 0.1%** on FB15k-237 (5 seeds) |
| Cross-domain transfer | **0.961** on WN18RR (100 samples, frozen encoder) |
| Synthetic dominance | DELTA **0.880** vs GraphGPS 0.293 vs GRIT 0.307 (edge classification) |
| Noise robustness | **+24%** over vanilla GNN at 80% feature corruption |

---

## Documentation Guide

### Architecture & Design

- **[Architecture Overview](architecture.md)** — Core thesis, architecture diagram, and how DELTA differs from Transformers and GNNs
- **[Visual Explainer](ARCHITECTURE_VISUAL.md)** — Interactive three-paradigm comparison (Transformer → GNN → DELTA)
- **[Bootstrap Strategy](bootstrap-strategy.md)** — How DELTA solves the graph construction chicken-and-egg problem
- **[Architecture Evolution](architecture-evolution.md)** — Six development stages from core validation to real-world data

### Experiment Results

- **[Validation Phases](validation-phases.md)** — All phase result tables (Phases 1–37)
- **[Key Findings](key-findings.md)** — 21 key findings organized by research stage
- **[Colab Results](COLAB_RESULTS.md)** — Phase 35/36/37 results from Colab Pro+ runs
- **[Backward Compatibility](backward-compatibility.md)** — Verification that architectural fixes don't break prior results

### Research & Publication

- **[Research Agenda](RESEARCH_AGENDA.md)** — Validated propositions, open gaps, compute options, publication pathway
- **[Publication Roadmap](PUBLICATION_ROADMAP.md)** — Full path to NeurIPS/ICLR (Phases 38–45)
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

*DELTA architecture — conceived March 25, 2026. 37 experiment phases, 6 architectural fixes, 44 unit tests. Phases 31–34 complete (H100/RTX PRO 6000). Phases 35–37 running on Colab Pro+. Phases 38–43 planned — see [Publication Roadmap](PUBLICATION_ROADMAP.md).*
