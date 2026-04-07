# The Brain: DELTA's End Goal

## The Vision

The human brain doesn't process information as flat sequences. It builds, strengthens, and prunes synaptic connections -- dynamically constructing the graph over which reasoning flows. Every thought creates new pathways; every pathway shapes the next thought.

**DELTA is building toward this:** a system that dynamically constructs its own relational graphs and reasons over them, without relying on pre-defined topology or transformer scaffolding.

---

## The Three Paradigms

| Paradigm | Structure | Edges | Graph Source | Limitation |
|----------|-----------|-------|-------------|------------|
| **Transformer** | Flat sequence | Implicit (attention weights) | None -- reconstructs structure from sequences | Quadratic cost to rediscover what a graph already encodes |
| **GNN** | Static graph | Passive scalar wires | Pre-given, fixed | Can't learn new structure; edges carry no computation |
| **DELTA / The Brain** | Dynamic graph | First-class computational citizens | Self-constructed, continuously refined | *This is what we're building* |

The key insight: **edges should think.** In a brain, synapses aren't just wires -- they have their own plasticity, their own state, their own role in computation. DELTA's edge-to-edge attention makes edges computational citizens that attend to each other and collectively discover relational patterns that node-only systems miss.

See [Visual Explainer](ARCHITECTURE_VISUAL.md) for an interactive diagram of this paradigm gap.

---

## What's Been Proven

Five capabilities validate the path toward The Brain:

- **Edge-first attention works** -- 100% on derived relations via 2-hop edge adjacency (Phase 11), +24% noise robustness over vanilla GNN (Phase 28)
- **Competitive on real KGs** -- SelfBootstrapHybrid MRR 0.5089, within 0.004 of GraphGPS, beats it on H@10 (Phase 40)
- **Multi-hop advantage accelerates with depth** -- 5p MRR 0.790 vs GraphGPS 0.690; DELTA is the only model that improves with reasoning depth (Phase 44)
- **Self-bootstrap removes the transformer scaffold** -- DELTA bootstraps DELTA at 157% of FixedChain (Phase 39)
- **Temperature reveals edge/node asymmetry** -- Edge attention wants sharper, node wants softer; asymmetric init yields LP MRR 0.4856 (Phases 46-48)

See [Key Findings](key-findings.md) for detailed evidence. See [Validation Phases](validation-phases.md) for complete results.

---

## The Capacity Paradox

DELTA-Matched (157K params) beats DELTA-Full (293K params) on multi-hop reasoning. Smaller model, harder task, better result:

| Depth | DELTA-Matched (157K) | DELTA-Full (293K) | GraphGPS (228K) |
|-------|---------------------|-------------------|-----------------|
| 2p | 0.758 | 0.711 | 0.754 |
| 3p | 0.753 | 0.692 | 0.727 |
| 5p | 0.790 | -- | 0.690 |

**The capacity constraint is a feature, not a limitation.** The smaller model can't memorize local edge statistics, so it's forced to learn generalizable relational abstractions that compose across hops. This is **synaptic pruning** -- the brain's mechanism of starting with excess connectivity and selectively eliminating connections that don't contribute to function.

DELTA already has the infrastructure for adaptive capacity:

- **PostAttentionPruner** -- continuous [0,1] importance gates per edge and node, driven by observed attention weights
- **TieredMemory** -- variational compression (warm tier) and node absorption (cold tier). The graph literally shrinks when cold nodes are absorbed
- **Self-Bootstrap** -- re-bootstrapping with a smaller architecture that inherits structural knowledge from the larger one
- **LearnedAttentionDropout** -- per-edge learned dropout probability. Structural edges get low dropout; noisy edges get high dropout

The open question: can DELTA discover its own optimal capacity from data, rather than requiring hyperparameter search?

---

## Roadmap to The Brain

### Horizon 1: Core Proven (Phases 41-45) -- Complete

Multi-hop compositional advantage validated across depths, seeds, and regularization regimes. Inference cost is deployment-friendly. See [Validation Phases](validation-phases.md#phase-42-multi-hop-path-query-evaluation).

### Horizon 2: Adaptive Architecture (Phases 46-50) -- Active

Learnable temperature revealed the edge/node asymmetry that drives DELTA's behavior (Phases 46-48). Phase 49 tests whether combining L0 temperature with asymmetric L1+L2 closes the LP/3p trade-off. Phase 50 addresses the three constructor deficiencies identified in [Architecture Overview](architecture.md#graph-constructor).

### Horizon 3: Sequence Domain Generalization (Phases 51-60) -- Planned

The jump from KGs (where structure is given) to sequences (where structure must be discovered) is the critical test. The constructor becomes the entire bet -- if `DifferentiableEdgeSelector` learns the right edges for sequence tasks, DELTA handles the rest. LRA ListOps is the entry point: hierarchical numerical expressions map naturally to DELTA's edge-first reasoning. Requires constructor fixes from Phase 50.

### Horizon 4: Dynamic Reasoning (Phases 61-70) -- Future

- **Iterative graph refinement** -- multi-pass construction where each DELTA pass restructures the graph for the next
- **Temporal reasoning** -- graphs that evolve over time, with edges that strengthen or decay
- **Multi-scale construction** -- hierarchical graphs (entity -> concept -> domain) built bottom-up
- **Online learning** -- graph structure adapts during inference, not just training

### Horizon 5: The Brain (Phases 71+) -- Vision

- **Multi-modal graph construction** -- build relational graphs from text, images, structured data
- **Associative memory** -- long-term graph state that persists across tasks (like synaptic weights)
- **Compositional generalization** -- combine known relations to infer novel ones without retraining
- **Autonomous structure discovery** -- the system discovers what entities and relations exist, not just how they connect

---

## Why Not Just Use Transformers?

Transformers pay a **quadratic tax to rediscover structure** that a graph already encodes. Every self-attention layer recomputes "who should talk to whom" from scratch. A brain doesn't do this -- it has persistent connections that encode which neurons talk to which others.

DELTA's bet is that for **relational reasoning** -- understanding how things connect, not just what things are -- operating on explicit relational structure will be fundamentally more efficient than reconstructing it from flat sequences.

The Brain isn't about replacing transformers everywhere. It's about building something **better suited for relational reasoning** -- and scaling it to see how far that advantage extends.

---

## Open Questions

1. **Can DELTA scale to millions of entities?** Current experiments use <=15K entities. The Brain requires orders of magnitude more.
2. **Can DELTA discover its own optimal capacity?** Phase 42/44 show smaller beats larger on composition -- can the router's importance signals automate this?
3. **Where does explicit structure win vs lose?** The hypothesis: relational tasks benefit from explicit graphs, but sequential/generative tasks may not. Where's the crossover?
4. **Can the constructor learn useful graph structure for sequence domains?** The SelfBootstrap result (Phase 39) shows DELTA can learn structure from scratch. Can this extend to non-relational inputs?
5. **Can temperature optimization close the LP/3p trade-off?** Edge and node attention want opposite sharpness. L0 temperature (Phase 49) may be the missing piece.

---

*See [Architecture Overview](architecture.md) for technical details. See [Status & Roadmap](status-and-roadmap.md) for current progress.*
