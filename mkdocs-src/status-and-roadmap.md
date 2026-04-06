# Current Status & Roadmap

*Last updated: Phase 48 (April 6, 2026)*

---

## Current Best Results

| Metric | Model | Value | Phase |
|--------|-------|-------|-------|
| LP MRR (DELTA-Full, temp) | E: node=2, edge=6 | **0.4856** | 48 |
| LP H@10 (DELTA-Full, temp) | F: node=3, edge=5 | **0.8014** | 48 |
| 3p MRR (multi-hop) | DELTA-Matched @10% drop | **0.742 ± 0.009** | 45 (3-seed) |
| 5p MRR | DELTA-Matched @0% drop | **0.790** | 44 |
| Depth advantage (5p) | DELTA vs GraphGPS | **+0.100** | 44 |
| Per-query inference | DELTA vs GraphGPS | **0.8–0.9×** (faster) | 45 |

---

## What's Validated

### Core Architecture (Phases 1–30)

- **Edge-first dual attention** — outperforms node-only on relational tasks
- **Multi-hop edge adjacency** — 100% on derived relations, O(E^0.97) scaling
- **Post-attention soft gating** — 100% accuracy at 50% sparsity
- **Graph structure adds value** — FixedChain DELTA 40.7% > Transformer 36.3% (Phase 27b)

### Scale & Real Data (Phases 31–40)

- **Mini-batching** scales to full FB15k-237 (14,505 entities, 304K edges)
- **Correct LP evaluation** — 7 models benchmarked with filtered MRR/Hits@K (Phase 40)
- **Self-bootstrapped DELTA** — 157% of FixedChain, no transformer needed (Phase 39)
- **Domain transfer** — frozen encoder → 0.961 on WN18RR with 100 samples (Phase 35)

### Multi-Hop Compositional Reasoning (Phases 42–45)

- **DELTA-Matched is the only model that improves with reasoning depth** (2p→3p→4p→5p)
- **Advantage doubles per hop**: +0.004 (2p) → +0.026 (3p) → +0.066 (4p) → +0.100 (5p) vs GraphGPS
- **Robust across seeds** — 3-seed: 0.742 ± 0.009 (3p), std bars don't overlap with GraphGPS
- **Robust across regularization** — DELTA leads at all 5 DropEdge rates (0–40%)
- **Inference competitive** — per-query scoring 0.8–0.9× GraphGPS despite 34× training cost

### Attention Temperature (Phases 46–48)

- **Dead heads reduced** 83% → 33% via learnable temperature (Phases 46–48)
- **Edge/node asymmetry discovered** — edge temps drift UP, node temps stable (Phase 46–48)
- **Selective sharpening** — L0 soft, L1+L2 sharp outperforms uniform (Phase 47)
- **Asymmetric temperature** — node=2, edge=6 achieves LP MRR 0.4856, new record (Phase 48)
- **LP/3p trade-off identified** — best 3p (D: 0.4018) uses L0=4.0; best LP uses L0=1.0

---

## Known Issues

- **Phase 37 leakage** — invalidated (5 evaluation bugs). Replaced by Phase 40
- **Training cost** — DELTA 34× slower per epoch than GraphGPS; inference is comparable
- **3p gap** — D (all temp=4.0) achieves best 3p MRR but L0 temperature contribution unresolved

---

## Roadmap

### Active: Horizon 2 — Adaptive Architecture (Phases 46–50)

| Phase | Goal | Status |
|-------|------|--------|
| 46 | Learnable per-head temperature | ✅ Dead heads 83%→38%, edge/node asymmetry |
| 47 | Layer-specific temperature | ✅ B (L0 soft, L1+L2 sharp) = best LP 0.4783 |
| 48 | Asymmetric node/edge temperature | ✅ E = new LP record 0.4856; node stable, edge drifts UP |
| 49 | L0 temperature + asymmetric L1+L2 | **Next** |
| 50 | Multi-scale adaptive routing | Planned |

### Upcoming: Horizon 3 — Dynamic Reasoning (Phases 51–60)

Iterative graph refinement, temporal reasoning, multi-scale construction. See [The Brain](the-brain.md).

### Long-term: Horizon 4 — The Brain (Phases 61+)

Multi-modal construction, associative memory, compositional generalization. See [The Brain](the-brain.md).

---

*See [The Brain](the-brain.md) for the long-term vision. See [Validation Phases](validation-phases.md) for all experiment results.*
