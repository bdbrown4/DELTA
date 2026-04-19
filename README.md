# DELTA: Dynamic Edge-and-Node Architecture with Layered, Tiered Attention

> **[Full Documentation](https://bdbrown4.github.io/DELTA/)** — Architecture, experiment results, research agenda, and setup guides.

A research implementation of the DELTA architecture — a novel AI framework that operates on dynamic graphs with dual parallel attention across nodes and edges, tiered memory, and a learned importance router. Building toward **[The Brain](https://bdbrown4.github.io/DELTA/the-brain/)**: a system that dynamically constructs its own relational graphs and reasons over them.

## Core Thesis

Reality is a graph. Language is a lossy compression of reality into sequences. Transformers reconstruct relational structure from flat sequences. DELTA operates on relational structure directly.

GNN edges are passive scalar wires. DELTA edges are **first-class computational citizens** that attend to each other. That edge-to-edge attention is what produces the Phase 28 +24% noise robustness gap — and Phase 39's self-bootstrapped DELTA proves the system can build its own graph without any transformer scaffold.

## Headline Results

| Metric | Value | Phase |
|--------|-------|-------|
| Real FB15k-237 (5 seeds) | DELTA+Gate **97.4% ± 0.1%** | 29 |
| Best LP MRR (DELTA-Full, temp-tuned) | **0.4905** | 52 |
| Brain LP MRR (self-constructed graph) | **0.4818** with H@10 **0.8076** | 55–57 |
| Multi-hop 5p MRR | **0.790** vs GraphGPS 0.690 | 44 |
| Cross-domain transfer (frozen encoder, 100 samples) | **0.961** on WN18RR | 35 |
| Noise robustness at 80% corruption | **+24%** over vanilla GNN | 28 |
| Experiment phases | 63 complete (Phases 1–63) | — |
| Unit tests | 44/44 passing | — |

## Architecture

```
Raw Input (any modality)
    → Graph Constructor (transformer-bootstrapped or BrainConstructor)
    → BFS Graph Partitioner (O(N+E))
    → PARALLEL DUAL ATTENTION
        [Node Attention + Edge Attention simultaneously]
    → Post-Attention Pruner (observed attention weights)
    → Learned Attention Dropout
    → ReconciliationBridge (nodes and edges co-update)
    → Variational Memory Compression
    → Output + Updated Graph State
```

## Quick Start

```bash
pip install -r requirements.txt

# Run core validation (Phases 1–6)
python -m experiments.phase1_edge_attention
python -m experiments.phase2_dual_attention

# Run all tests
python -m pytest tests/ -q  # 44/44 should pass
```

## Documentation

Full documentation is hosted at **[bdbrown4.github.io/DELTA](https://bdbrown4.github.io/DELTA/)**:

- **[Architecture Overview](https://bdbrown4.github.io/DELTA/architecture/)** — Core thesis, components, and how DELTA differs from Transformers and GNNs
- **[The Brain: End Goal](https://bdbrown4.github.io/DELTA/the-brain/)** — Long-term vision, capacity paradox, roadmap horizons
- **[Validation Phases](https://bdbrown4.github.io/DELTA/validation-phases/)** — All 63 phase result tables
- **[Key Findings](https://bdbrown4.github.io/DELTA/key-findings/)** — 44 key findings from experiments
- **[Status & Roadmap](https://bdbrown4.github.io/DELTA/status-and-roadmap/)** — What's validated, open gaps, next steps
- **[Getting Started](https://bdbrown4.github.io/DELTA/setup-and-running/)** — Installation, experiment commands, cloud GPU setup
- **[Research Methodology](https://bdbrown4.github.io/DELTA/research-methodology/)** — AI assistance disclosure

## Requirements

- Python 3.10+
- PyTorch 2.0+ (GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu124`)
- CPU sufficient for Phases 1–24; GPU (6 GB+ VRAM) recommended for Phase 25+

## Research Methodology

This project was developed by a solo software engineer using LLMs and AI agents as force-multipliers. All architectural decisions, experimental design, and scientific direction were driven by the human researcher. AI tools (Claude, Gemini, GitHub Copilot) were used for code generation, debugging, and documentation. See [full disclosure](https://bdbrown4.github.io/DELTA/research-methodology/).

---

*DELTA architecture — conceived March 25, 2026. 63 experiment phases, 6 architectural fixes, 44 unit tests. KG scaling complete (Phases 59–63). Pivoting to sparse attention and sequence domains.*
