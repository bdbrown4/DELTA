# DELTA Research Agenda & External Assessment

This document tracks research propositions, open claims, compute options, and the publication pathway based on external cross-analysis after Phase 30 (March 26, 2026).

---

## Validated Propositions (Scorecard)

Track which architectural claims have been empirically confirmed and at what confidence.

| # | Proposition | Status | Confidence | Phase Evidence |
|---|---|---|---|---|
| P1 | Edge attention categorically beats node attention on relational tasks | ✅ Confirmed | High | Phase 1, 9, 11, 13 |
| P2 | Dual parallel attention is the key differentiator at high noise | ✅ Confirmed | High | Phase 28: Dual 64.2% vs Vanilla 40.2% (+24%) at extreme noise |
| P3 | Graph structure adds value over transformers on relational tasks | ✅ Confirmed | Medium-High | Phase 27b: Fixed Chain 40.7% > Transformer 36.3% (+4.4%) |
| P4 | Soft gating achieves sparsity without accuracy loss | ✅ Confirmed | High | Phase 22/29: 100.0% ± 0.0% at 50% sparsity |
| P5 | DELTA beats CompGCN on real FB15k-237 | ✅ Confirmed | High | Phase 25/29: 97.4% ± 0.1% vs 96.9% ± 0.3% (5 seeds) |
| P6 | Results stable across random seeds | ✅ Confirmed | High | Phase 29: all key results ≤ 0.4% std |
| P7 | Sampling robustness at 26% VRAM budget | ✅ Confirmed | High | Phase 30: all 4 strategies within ±0.2% |
| P8 | GraphConstructor adds value vs fixed topology | ❌ Not yet | Medium | Phase 27b: Bootstrap 34.3% < Fixed Chain 40.7% (-6.3%) |
| P9 | DELTA scales beyond 2000 entities | ❌ Untested | Unknown | Phase 31 target |
| P10 | Cross-domain generalization (WN18RR) | ❌ Untested | Unknown | Phase 32 target |
| P11 | DELTA advantage vs post-2021 graph transformers (GraphGPS, GRIT) | ❌ Untested | Unknown | High priority before publication |

---

## Open Gaps (Priority Ordered)

### Gap 1: Baseline currency — CRITICAL before any writeup
- CompGCN is a 2020 baseline. The community will ask about GraphGPS (2022) and GRIT (2023).
- **Action:** Run DELTA against at least GraphGPS on full FB15k-237 once Phase 31 mini-batching works.
- If DELTA wins → strong paper. If it doesn't → reframe contribution around efficiency + compositionality.
- Status: [ ] Not started

### Gap 2: Full-scale evaluation — CRITICAL
- Current status: 2000-entity subgraph (14% of FB15k-237's 14,505 entities; 22% of triples).
- WN18RR has 40,943 entities. Real-world KGs have millions.
- **Action:** Phase 31 mini-batching to enable full FB15k-237 training+eval without subsampling.
- Status: [ ] Not started (roadmap Phase 31)

### Gap 3: Phase 27b's +4.4% needs a harder, larger replication
- 40.7% vs 36.3% on a 16-class problem is a real but modest margin.
- **Action:** Design a harder relational task (more classes, longer paths, more distractors) at N=5000+.
- Needs Phase 31 compute budget to run efficiently.
- Status: [ ] Not started (roadmap Phase 33)

---

## Compute Options

Ordered by cost-effectiveness for the immediate next steps (Phase 31 full-scale FB15k-237).

| Option | Cost | VRAM | Best For | Action |
|---|---|---|---|---|
| **Google Colab Pro+** | ~$50/month | A100 40GB | First Phase 31 validation — low commitment | [ ] Evaluate |
| **RunPod / vast.ai** | ~$1–3/hour | A100 80GB or H100 | On-demand bursts for benchmark runs ($20–50 total for a solid run) | [ ] Evaluate |
| **Lambda Labs** | ~$1.25–2/hour | A100 80GB | Longer multi-day training runs; more stable than vast.ai | [ ] Evaluate |
| **TPU Research Cloud** | Free | TPU v4/v5 | Serious compute for free; requires researcher application | [ ] Apply |

### Recommendation (from external review)
> "Don't spend significant money yet. Spend $30–50 on Colab Pro+ or a few RunPod hours to validate that Phase 31 mini-batching works and to get full-scale FB15k-237 results. If those results hold or improve, *then* it's worth pursuing larger compute — either through a cloud budget or a TPU Research Cloud application."

### TPU Research Cloud application notes
- Google gives free TPU access to researchers
- Frame as: novel graph architecture for knowledge graph reasoning / relational composition
- 30-phase validation history and GitHub repo are strong supporting evidence
- Worth applying once full-scale results exist to include in the application
- Status: [ ] Not started (defer until Phase 31 results available)

---

## Publication Pathway Checklist

### Already Have ✅
- [x] Novel architecture with theoretical motivation (edge-first dual attention)
- [x] 31 validation phases with honest failure documentation
- [x] Multi-seed statistical validation (Phase 29: 5 seeds, low variance)
- [x] Real dataset results beating established baselines (FB15k-237, 5 seeds)
- [x] Clean ablation showing component contributions (Phase 28: +24% at extreme noise)
- [x] Identified failure modes with actionable fixes (Phase 27b confession + correction)
- [x] Graph adjacency optimization (vectorized fast path, caching in `graph.py`)

### Still Needs ❌
- [ ] Full-scale dataset evaluation (Phase 31 — requires compute upgrade)
- [ ] Comparison vs at least one post-2021 baseline (GraphGPS or GRIT)
- [ ] Cross-domain transfer results (Phase 32 — WN18RR)
- [ ] Stronger replication of Phase 27b's graph-vs-transformer finding on a harder task
- [ ] Clear positioning narrative (see below)

### Positioning Narrative (key framing decision)
The clearest differentiation is **not** raw accuracy — it's compositional relational reasoning with efficiency via sparsity:

> *"DELTA is architecturally superior for tasks requiring relational composition under noise."*

The two-sentence evidence:
- Phase 28: dual attention +24% over vanilla at extreme noise = noise robustness
- Phase 27b: fixed graph +4.4% over pure transformer on relational task = structural advantage

This frames DELTA as specialized (right tool for relational composition) rather than general — which is honest and defensible. The Phase 5 result (Transformer ≈ DELTA on non-relational tasks) supports this framing rather than contradicting it.

---

## Phase Roadmap (Next 5)

| Phase | Goal | Blocked By | Priority |
|---|---|---|---|
| **31** | Mini-batching: full FB15k-237 without subsampling | Compute (A100 recommended) | 🔴 Highest |
| **32** | Cross-graph transfer: train FB15k-237, eval WN18RR | Phase 31 compute | 🟡 High |
| **33** | Task-aware graph construction: preserve path ordering in constructor | Phase 31 or gradient-accum workaround | 🟡 High |
| **34** | GraphGPS / GRIT comparison on full FB15k-237 | Phase 31 + baseline implementations | 🔴 Highest (for publication) |
| **35** | Harder relational task: replicate Phase 27b finding at scale | Phase 31 compute | 🟠 Medium |

---

## External Review Summary (Gemini cross-analysis, March 26, 2026)

> "This is no longer an alpha research prototype. It's a credible, statistically validated research contribution with a clear publication path. The Phase 29 multi-seed confirmation is what pushed it over that threshold — 97.4% ± 0.1% on real FB15k-237 across 5 seeds is not a lucky result."

> "You should be thinking about an arXiv preprint within the next 5–10 phases. The story is coherent enough to write. The remaining experiments would strengthen it, not create it."

> "DELTA's clearest differentiation isn't raw accuracy — it's compositional relational reasoning with efficiency via sparsity."

Full review text available on request.

---

*Last updated: March 26, 2026 — after Phase 30 + Phase 27b correction*
