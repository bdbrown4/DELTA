# Adaptive Architecture: DELTA's Self-Modifying Capacity

## The Epiphany

DELTA-Matched (157K params) beats DELTA-Full (293K params) on multi-hop reasoning. Smaller model, harder task, better result. This isn't noise — Phase 44 shows the advantage *accelerates* with depth:

| Depth | DELTA-Matched (157K) | DELTA-Full (293K) | GraphGPS (228K) |
|-------|---------------------|-------------------|-----------------|
| 2p | 0.758 | 0.711 | 0.754 |
| 3p | 0.753 | 0.692 | 0.727 |
| 5p | 0.790 | — | 0.690 |

**The capacity constraint is a feature, not a limitation.** The smaller model can't memorize local edge statistics, so it's forced to learn generalizable relational abstractions that compose across hops. The question: what if DELTA could *discover* this optimal capacity from the data itself, rather than us finding it through hyperparameter search?

This is **synaptic pruning** — the brain's mechanism of starting with excess connectivity and selectively eliminating connections that don't contribute to function. DELTA already has most of the infrastructure to do this.

---

## What Already Exists

DELTA has four components that can be composed into an adaptive architecture:

### 1. Router / Importance Scores
`PostAttentionPruner` (in `delta/router.py`) produces **continuous [0,1] importance gates** per edge and node, computed from observed attention weights. These are already differentiable and already drive memory tier assignment:
- Gate > 0.6 → HOT (full resolution)
- Gate 0.2–0.6 → WARM (compressed)
- Gate < 0.2 → COLD (archived, no attention)

**What it can tell us:** Which components are consistently underutilized.

### 2. Memory Tiers
`TieredMemory` already implements variational compression (warm tier) and node absorption (cold tier merges similar nodes, redirects edges). This is structural modification — the graph literally shrinks when cold nodes are absorbed.

**What it can tell us:** The compression bottleneck already exists and is differentiable.

### 3. Self-Bootstrap
DELTA can construct its own graph from scratch. A lightweight DELTA pass enriches features, a Gumbel-sigmoid constructor builds edges, and a full DELTA stack processes the result. This achieved 157% of FixedChain.

**What it enables:** Re-bootstrapping with a smaller architecture that inherits structural knowledge from the larger one.

### 4. Learned Attention Dropout
`LearnedAttentionDropout` gives each edge a *learned* dropout probability based on its features. Edges the model wants to regularize get high dropout; structural edges get low dropout.

**What it can tell us:** Which edges are structural vs. noisy.

---

## The Adaptive Architecture Proposal

The original plan (Phases 46–49: capacity signals → importance pruning → curriculum compression → bidirectional adaptive) was superseded by a more direct discovery: **learnable per-head temperature** revealed the attention asymmetry that drives DELTA's behavior, without requiring the full adaptive infrastructure.

### What Actually Happened (Phases 46–48)

Instead of measuring capacity signals and pruning, the temperature experiments answered a more fundamental question: **why does DELTA's attention underperform?**

**Phase 46: Learnable Temperature** — Added per-head `_log_temp` parameter. Uniform init=4.0 reduced dead heads from 83%→38% and revealed that edge and node attention want *opposite* sharpness: edge temps drift UP, node temps drift DOWN.

**Phase 47: Selective Sharpening** — Tested layer-specific vs attention-type temperature. B (L0=1, L1+L2=4) achieved best LP MRR (0.4783). Node attention needs explicit initialization to activate — edge-only sharpening leaves node heads dead.

**Phase 48: Asymmetric Temperature** — Independent node/edge init across L1+L2. E (node=2, edge=6) achieved **LP MRR 0.4856** (new record). Node temps are "set and forget" (±0.01 drift); edge temps always drift UP. L2 edge drifts more than L1.

### Connection to the Adaptive Vision

The temperature findings actually *support* the adaptive architecture thesis:

1. **The capacity signal exists** — dead head counts and temperature drift patterns are clear signals about what the model needs
2. **Edge/node asymmetry is structural** — not an artifact. The model consistently learns different optimal temperatures for edge vs node attention
3. **Layer-specific behavior is real** — L0 is always dead (encoding pass), L1+L2 have different drift patterns

The adaptive pruning/expansion vision (Phases 47–50 as originally planned) remains valid as a future direction. The temperature work provides the *mechanistic understanding* that would make adaptive decisions more principled.

### Remaining Adaptive Architecture Path

| Phase | Goal | Status |
|-------|------|--------|
| 49 | L0 temperature + asymmetric L1+L2 (close LP/3p gap) | Planned |
| 50 | Multi-scale adaptive (depth-conditioned routing) | Future |
| 51+ | Importance-driven pruning (informed by temperature findings) | Future |

---

## Original Adaptive Vision (Preserved for Reference)

The following concepts remain valid future directions, now informed by the temperature findings:

### Importance-Driven Pruning
The router's importance scores + temperature drift patterns provide clear signals for which components to prune. Dead heads at L0 (regardless of temperature) and stable node temps (±0.01 drift) suggest fixed architectural decisions can replace learned flexibility in some cases.

### Curriculum Compression
Train at full capacity → apply increasing sparsity pressure → physically remove low-utilization components. The temperature schedule could be combined with sparsity: sharp temperature + low gate → candidate for removal.

### Depth-Conditioned Architecture
Phase 44 showed the multi-hop advantage accelerates with depth. A mixture-of-experts at the architectural level — routing 1p queries to a compact branch and 3p+ queries to a full branch — could exploit the capacity paradox directly.

---

*See [The Brain](the-brain.md) for the long-term vision. See [Key Findings](key-findings.md#attention-temperature-phases-4648) for temperature experiment results.*

The self-bootstrap + adaptive capacity mechanism is what bridges DELTA (a fixed architecture) to The Brain (a self-modifying system). It's the plasticity layer.

---

## Implementation Priority

| Phase | Experiment | Prereqs | Risk | Value |
|-------|-----------|---------|------|-------|
| **46** | Capacity signal measurement | Phase 45 complete | Low | Foundational — tells us if the signal exists |
| **47** | Static importance pruning | Phase 46 data | Low | Direct test of the hypothesis |
| **48** | Curriculum compression | Phase 47 works | Medium | Training-integrated pruning |
| **49** | Bidirectional adaptive | Phase 48 works | High | The Brain prototype |
| **50** | Multi-scale adaptive | Phase 49 works | High | Full vision |

**Phase 46 is the gating experiment.** If importance scores don't contain a clear capacity signal, the idea needs rethinking. If they do, Phases 47-49 follow naturally.

---

## Connection to the Broader Research Program

This is a second paper. The first paper (Phases 40-45) establishes:
- DELTA's edge-first architecture works on real KGs
- Multi-hop compositional advantage grows with depth
- The advantage is structural (Phase 43 robustness), not a hyperparameter accident

The second paper extends with:
- The capacity paradox (smaller beats larger on composition)
- Router-driven structural self-modification
- DELTA discovering its own optimal capacity from data
- Connection to biological synaptic pruning

Together they tell the story: DELTA doesn't just reason over graphs better — it can reason about *itself* and restructure accordingly. That's The Brain.

---

*See [The Brain](the-brain.md) for the long-term vision. See [Bootstrap Strategy](bootstrap-strategy.md) for the self-bootstrap mechanism that enables structural transitions.*
