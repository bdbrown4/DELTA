# The Brain: DELTA's End Goal

## The Vision

The human brain doesn't process information as flat sequences. It builds, strengthens, and prunes synaptic connections — dynamically constructing the graph over which reasoning flows. Every thought creates new pathways; every pathway shapes the next thought.

**DELTA is building toward this:** a system that dynamically constructs its own relational graphs and reasons over them, without relying on pre-defined topology or transformer scaffolding. Not another GNN paper. Not another attention variant. A fundamentally different computational substrate for relational reasoning.

---

## The Three Paradigms

| Paradigm | Structure | Edges | Graph Source | Limitation |
|----------|-----------|-------|-------------|------------|
| **Transformer** | Flat sequence | Implicit (attention weights) | None — reconstructs structure from sequences | Quadratic cost to rediscover what a graph already encodes |
| **GNN** | Static graph | Passive scalar wires | Pre-given, fixed | Can't learn new structure; edges carry no computation |
| **DELTA / The Brain** | Dynamic graph | First-class computational citizens | Self-constructed, continuously refined | *This is what we're building* |

The key insight: **edges should think.** In a brain, synapses aren't just wires — they have their own plasticity, their own state, their own role in computation. DELTA's edge-to-edge attention makes edges computational citizens that attend to each other, reason about each other, and collectively discover relational patterns that node-only systems miss.

---

## The Evidence Trail

### Foundation: Edges as First-Class Citizens (Phases 1–15)

Edge-centric dual attention works. The core mechanism — edges attending to edges — produces measurable advantages:

- **Phase 11:** 2-hop edge adjacency achieves 100% on derived/transitive relations (+38.9% over 1-hop)
- **Phase 13:** DELTA 100% on all compositional relations (vs Node GNN 87.5% on derived)
- **Phase 2:** Dual parallel attention converges 2.7× faster than sequential

### Scale: Real Data, Real Comparisons (Phases 22–37)

DELTA works on real knowledge graphs and dominates on synthetic relational tasks:

- **Phase 25:** DELTA+Gate 97.6% on real FB15k-237 (vs CompGCN 97.2%)
- **Phase 34:** DELTA 0.880 vs GraphGPS 0.293 vs GRIT 0.307 on edge classification
- **Phase 28:** +24% accuracy over vanilla GNN at 80% feature corruption — relational structure survives noise
- **Phase 35:** Frozen encoder → 0.961 on WN18RR with 100 samples (domain-agnostic transfer)

!!! warning "Phase 37 Leakage Disclosure"
    Phase 37's reported accuracy (0.991–0.994) was invalidated after a leakage audit identified 5 critical issues in the evaluation pipeline. The scale validation (310K edges, GPU training, mini-batching) remains valid. See [Validation Phases](validation-phases.md#phase-37-leakage-audit) for full details. Phase 40 replaces Phase 37 with correct link prediction evaluation.

### The Constructor Bottleneck (Phases 27b, 33, 36, 38)

The original GraphConstructor used non-differentiable hard attention thresholding — task loss couldn't influence which edges were created. This was the philosophical bottleneck: DELTA claimed to "operate on relational structure directly" but couldn't *construct* that structure end-to-end.

- **Phase 27b:** Graph structure helps (+4.4% over Transformer), but the constructor is the bottleneck
- **Phase 33/36:** Task-aware construction adds ≤1.3% — not because it's impossible, but because the constructor can't learn

**Phase 38** (Differentiable Constructor) broke through: Gumbel-sigmoid edge selection with straight-through estimators made construction fully differentiable. The Hybrid variant reached 98% of FixedChain (0.452 ± 0.006 vs 0.461 ± 0.034).

### The Breakthrough: DELTA Bootstraps DELTA (Phase 39)

**This is the moment the transformer scaffold comes down.**

Phase 39 replaced the transformer bootstrap with a FixedChain DELTA layer — DELTA constructs its own graph from trivial sequential input, then processes that self-constructed graph with a full DELTA stack.

| Model | Accuracy | vs FixedChain |
|-------|----------|---------------|
| Transformer | 0.429 ± 0.021 | 89% |
| FixedChain | 0.481 ± 0.015 | 100% |
| P38_Hybrid | 0.459 ± 0.036 | 95% |
| **SelfBootstrap** | **0.757 ± 0.041** | **157%** |
| SelfBootstrapHybrid | 0.716 ± 0.038 | 149% |

The self-bootstrapped DELTA doesn't just match the transformer scaffold — it **obliterates** it. +76% over the transformer, +57% over FixedChain. DELTA-enriched embeddings make edge discovery dramatically more reliable than raw positional embeddings.

### Honest Evaluation: Correct Link Prediction (Phase 40)

Phase 40 rebuilt the evaluation pipeline from scratch, fixing all 5 leakage issues from Phase 37:

- Learned entity/relation embeddings (no label leakage in edge features)
- Train-only graph for message passing
- Filtered MRR/Hits@K metrics
- Proper negative sampling
- Target edge masking

**200-epoch results** (all models still converging):

| Model | MRR | Hits@1 | Hits@10 |
|-------|-----|--------|---------|
| GraphGPS | 0.513 | 0.413 | 0.684 |
| DELTA-Matched | 0.497 | 0.397 | 0.674 |
| SelfBootstrapHybrid | 0.494 | 0.395 | 0.669 |
| DELTA-Full | 0.493 | 0.394 | 0.667 |
| SelfBootstrap | 0.489 | 0.389 | 0.665 |
| GRIT | 0.439 | 0.339 | 0.618 |
| DistMult | 0.047 | 0.020 | 0.098 |

DELTA is competitive with GraphGPS at 200 epochs — but DELTA models were still climbing while GraphGPS had plateaued. The 500-epoch convergence study shows GraphGPS peaked at epoch 200 (MRR 0.530) and began **declining** (overfitting), while DistMult climbed from 0.047 to 0.484. DELTA variants are expected to benefit similarly from extended training.

!!! note "Speed vs Convergence"
    DELTA runs ~10-20s/epoch vs GraphGPS ~0.2s/epoch (43-100× slower per step). In equal wall-clock time, GraphGPS gets far more gradient updates. DELTA's competitiveness at equal epoch count suggests a higher ceiling with optimization.

---

## The Roadmap to The Brain

### Where We Are Now

DELTA has proven three critical capabilities:

1. **Edge-centric reasoning works** — edges as computational citizens produce real advantages on relational tasks
2. **Self-construction works** — DELTA can build its own graph from scratch, no transformer needed (Phase 39)
3. **Competitive LP** — SelfBootstrapHybrid reaches MRR 0.5089 on FB15k-237 link prediction, within 0.004 of GraphGPS (0.5126) and beating it on Hits@10. DELTA-Matched achieves 97% of GraphGPS MRR with 69% of its parameters. (Phase 40)

### What Comes Next

The path from "competitive GNN" to "The Brain" follows three horizons:

#### Horizon 1: Prove the Core (Phases 41–45)

Establish DELTA as a legitimate relational reasoning architecture with publication-quality results.

| Phase | Goal | Status |
|-------|------|--------|
| 41 | Generalization gap investigation | ✅ Negative result (val-set noise) |
| 42 | Multi-hop path queries (1p/2p/3p) | ✅ DELTA-Matched only model improving 2p→3p |
| 43 | DropEdge robustness check | ✅ Advantage holds at all 5 drop rates |
| 44 | Extended depth (4p/5p) | ✅ Advantage accelerates: +0.004 → +0.100 |
| 45 | Inference timing + multi-seed headline | ✅ Per-query inference 0.8-0.9× GraphGPS; 3-seed robust |

#### Horizon 2: Adaptive Architecture (Phases 46–50)

The capacity paradox — smaller DELTA beats larger on composition — is a signal. The model needs to discover its own optimal capacity from the data. See [Adaptive Architecture](adaptive-architecture.md) for the full proposal.

| Phase | Goal | Status |
|-------|------|--------|
| 46 | Attention sharpening via learnable temperature | ✅ Dead heads 83%→38% (full), edge/node asymmetry discovered |
| 47 | Layer-specific temperature initialization | ✅ B (L0 soft, L1+L2 sharp) = best LP MRR (0.4783); node attn needs sharpening to activate |
| 48 | Asymmetric node/edge temperature | ✅ E (node=2,edge=6) = new LP MRR record (0.4856, +1.5%); node temps stable, edge drifts UP; 3p gap persists (L0=1.0 vs D's L0=4.0) |
| 49 | L0 temperature + asymmetric L1+L2 | Planned |
| 50 | Multi-scale adaptive (depth-conditioned routing) | Planned |

#### Horizon 3: Dynamic Reasoning (Phases 51–60)

Move from static evaluation to dynamic graph evolution — the system that builds and refines its own reasoning substrate.

- **Iterative graph refinement:** Multi-pass construction where each DELTA pass restructures the graph for the next
- **Temporal reasoning:** Graphs that evolve over time, with edges that strengthen or decay
- **Multi-scale construction:** Hierarchical graphs (entity → concept → domain) built bottom-up
- **Online learning:** Graph structure adapts during inference, not just training

#### Horizon 4: The Brain (Phases 61+)

The end goal: a general relational reasoning system that rivals transformer-scale language models on tasks where relational structure matters.

- **Multi-modal graph construction:** Build relational graphs from text, images, structured data — any input modality
- **Associative memory:** Long-term graph state that persists across tasks (like synaptic weights)
- **Compositional generalization:** Combine known relations to infer novel ones without retraining
- **Autonomous structure discovery:** The system discovers what entities and relations exist, not just how they connect

---

## Why Not Just Use Transformers?

Transformers work. They scale. They dominate benchmarks. So why build something different?

Because transformers pay a **quadratic tax to rediscover structure** that a graph already encodes. Every self-attention layer recomputes "who should talk to whom" from scratch. A brain doesn't do this — it has persistent connections that encode which neurons talk to which others.

DELTA's bet is that for **relational reasoning** — the kind of reasoning that requires understanding how things connect, not just what things are — operating on explicit relational structure will be fundamentally more efficient than reconstructing it from flat sequences.

The evidence so far supports this:

- **Phase 34:** DELTA 0.880 vs GraphGPS 0.293 on edge classification — a task that *is* about relational structure
- **Phase 28:** +24% noise robustness — relational structure survives feature corruption
- **Phase 39:** Self-bootstrapped DELTA at 157% of FixedChain — DELTA's own pass enriches representations more than any external bootstrap

The Brain isn't about replacing transformers everywhere. It's about building something **better suited for relational reasoning** — and then scaling it to see how far that advantage extends.

---

## Key Open Questions

1. **Does the self-bootstrap advantage hold on real data?** Phase 40 answers this: SelfBootstrapHybrid is the best-performing DELTA variant on FB15k-237 (MRR 0.5089, H@10 0.8158), beating all vanilla DELTA architectures. The self-bootstrap mechanism helps on real graphs, not just synthetic tasks.

2. **Can DELTA discover its own optimal capacity?** Phase 42/44 show smaller DELTA beats larger on composition — the capacity paradox. Phase 46+ will test whether the router's importance signals contain enough information for the model to prune itself to the right size. See [Adaptive Architecture](adaptive-architecture.md).

3. **Can DELTA match transformer-scale reasoning?** Current experiments use ≤15K entities. The Brain requires scaling to millions.

4. **Does iterative refinement help?** Multi-pass graph construction (bootstrap → refine → refine) is theoretically appealing but unvalidated.

5. **What's the speed ceiling?** DELTA's per-epoch cost is 43-100× GraphGPS. Can architectural optimizations (sparse attention, approximate adjacency) close this gap? Phase 45 separated training cost (34×) from inference cost (0.8-0.9× per query) — inference is already comparable.

6. **Where does explicit structure win?** The hypothesis: relational tasks benefit from explicit graphs, but sequential/generative tasks may not. Where's the crossover?

7. **Can temperature optimization close the LP/3p trade-off?** Phase 46–48 discovered that edge and node attention want opposite sharpness: edge temps drift UP, node temps drift DOWN. Asymmetric init (E: node=2, edge=6) yields LP MRR 0.4856 (+1.5%), but D's uniform temp=4.0 still leads 3p (0.4018). L0 temperature (missing from Phases 47–48) may be the key — Phase 49 will test this.

8. **What is the optimal L0 behavior?** L0 attention is dead regardless of temperature in Phases 46–48. All conditions used L0=1.0 while D (all=4.0) is the 3p champion — suggesting L0 temperature matters for compositional reasoning even though attention weights appear unused.

9. **Are node temperatures truly "set and forget"?** Phase 48 showed node temps stable within ±0.01 across all conditions, while edge temps always drift UP. If node attention doesn't learn during training, could it be replaced with a fixed projection — saving parameters and compute?

---

*This document captures the long-term vision. For current experiment results, see [Validation Phases](validation-phases.md). For implementation status, see [Status & Roadmap](status-and-roadmap.md).*
