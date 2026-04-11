# Architecture Overview

## Core Thesis

Reality is a graph. Language is a lossy compression of reality into sequences. Transformers reconstruct relational structure from flat sequences. DELTA operates on relational structure directly.

See [Visual Explainer](ARCHITECTURE_VISUAL.md) for an interactive three-paradigm diagram. The key insight: GNN edges are passive scalar wires; DELTA edges are first-class computational citizens that attend to each other.

---

## Architecture Diagram

```
Raw Input (any modality)
    -> Graph Constructor (transformer-bootstrapped: per-layer edge projections + typed edges)
    -> BFS Graph Partitioner (O(N+E) seed-expansion clustering)
    -> PARALLEL DUAL ATTENTION
        [Node Attention + Edge Attention across all partitions simultaneously]
    -> Post-Attention Pruner (prune based on OBSERVED attention weights)
    -> Learned Attention Dropout (per-edge regularization)
    -> Reconciliation Layer (nodes and edges co-update)
    -> Hierarchical Global Attention (cluster representatives)
    -> Variational Memory Compression (warm nodes -> bottleneck -> KL regularization)
    -> Memory Tier Update (importance-based hot/warm/cold promotion)
    -> Output + Updated Graph State
```

---

## Key Components

### DualParallelAttention

Node attention and edge attention run simultaneously as parallel multi-head self-attention streams over different domains:

- **Node attention:** Nodes attend to neighboring nodes via the edge index.
- **Edge attention:** Edges attend to structurally adjacent edges via the edge adjacency matrix.

The dual parallel design converges 2.7x faster than sequential alternatives (Phase 2). The two streams remain independent until the ReconciliationBridge merges their outputs.

### Multi-Hop Edge Adjacency

The mechanism that enables compositional reasoning. Two edges are "adjacent" if they share a node, creating an edge-to-edge connectivity structure. At 2 hops, an edge can "see" edges two hops away, enabling transitive inference without explicit rule learning.

Phase 11: 2-hop achieves **100% on derived/transitive relations** — a +38.9% jump from 1-hop (61.1%) and beating Node GNN (83.3%). This was the single biggest architectural improvement in the project. Implemented as sparse COO tensors for O(E^0.97) scaling (Phase 17), handling 2500+ edges where the original dense approach timed out at ~500.

### ReconciliationBridge

The cross-stream interaction mechanism. After both attention streams complete, the ReconciliationBridge co-updates nodes and edges in a sequential cascade:

1. **Edges absorb node context:** Each edge concatenates its features with its source and target node features, then projects back to edge dimension.
2. **Nodes absorb edge context:** Each node aggregates updated edge features from all incident edges, concatenates with its own features, then projects back to node dimension.

Both updates use residual connections and LayerNorm, providing direct high-bandwidth gradient flow.

!!! note "Cross-Stream Interaction Comparison"
    A [prototype comparison](https://github.com/bdbrown4/DELTA/blob/main/experiments/prototypes/shared_latent_bottleneck.py) tested ReconciliationBridge (0.889 val) against Cross-Attention Gates (0.218, at majority baseline) and Shared Latent Bottleneck (0.210, at majority baseline). Direct linear mixing is fundamentally superior — gated alternatives never open properly.

### PostAttentionPruner

Prunes graph elements based on *observed* attention weights rather than predicting importance before attention (the original router's chicken-and-egg problem). Uses soft sigmoid gates with per-head attention features, avoiding the non-differentiable hard top-k that caused the original 29% accuracy gap.

Phase 16: **100% accuracy at 50% target sparsity**, matching full attention and beating pre-attention routing by +14.7%.

### Graph Constructor

Transformer-bootstrapped graph construction with per-layer edge projections and typed edges. Solves the bootstrap "chicken-and-egg" problem: a lightweight transformer builds an initial graph, then DELTA refines it.

!!! warning "Limited Contribution (KG domain)"
    Phase 36: GraphConstructor adds <=1.3% over fixed topology. DELTA's core architecture is powerful enough that the given topology suffices for KG tasks. For sequence domain generalization (where the constructor must discover structure from scratch), the constructor becomes the critical component — and has three identified deficiencies.

!!! bug "Three Constructor Deficiencies (Active)"
    **1. Gradient wall at edge selection** (`constructor.py:127`): Hard binary `attn > threshold` blocks gradient flow. Fix: replace with Gumbel-sigmoid (proven in Phase 38/39).

    **2. Dead `edge_type_weights`** (`constructor.py:173-175`): Softmax output is computed then discarded — never included in the returned `DeltaGraph`. Fix: fold into `edge_features` before returning.

    **3. One token = one node (no conceptual compression)**: "New York City" produces three nodes. Acceptable for short sequences; caused the Phase 25 VRAM problem for LRA (4096 tokens).

    Fixes 1 and 2 are backward-compatible and verifiable on FB15k-237 LP.

### Variational Memory Compression

Tiered memory system (hot/warm/cold) with a variational bottleneck and KL regularization. Preserves accuracy while compressing node representations. KL converges during training (0.126 to 0.026), confirming proper latent space regularization.

### BFS Graph Partitioner

O(N+E) BFS seed-expansion clustering, replacing the original O(N^3) spectral partitioning. Balance ratio of 0.79 with importance-aware seeding (Phase 20).

### Learnable Temperature

Per-head temperature scaling reveals an edge/node asymmetry in attention sharpness (Phases 46–52). Edge heads converge to sharper temperatures while node heads prefer softer distributions. Temperature annealing (high node temp early, anneal down) breaks the 3p ceiling (Phase 50: K achieves 3p MRR 0.4148). Edge sharpness boosts LP further (Phase 52: Q achieves LP MRR 0.4905). Multi-seed validation (Phase 53) confirmed LP improvements are robust but multi-hop claims are seed-dependent. After 9 phases, three operating modes emerged: LP-optimized, balanced-3p, and deep-reasoning.

### BrainEncoder

Differentiable graph construction via Gumbel-sigmoid edge selection (Phases 55–57). BrainEncoder uses a `BrainConstructor` to build new edges from learned node representations, then runs DELTA message passing over the augmented graph (original + constructed edges).

- **brain_hybrid mode**: Preserves original KG edges and adds constructed edges. 311K params.
- **Gumbel-sigmoid selection**: Differentiable edge selection with configurable target density. No gradient wall — task loss flows through edge selection.
- **Density control**: `target_density=0.01` produces 2,435 edges and strictly dominates `0.02` (4,870 edges) on MRR, H@3, and H@10 (Phase 56).
- **Results**: MRR 0.4818 (matches delta_full) with H@10 **0.8076** (+4.7% over delta_full). Constructed edges genuinely add recall.
- **Temperature**: Annealing is counterproductive on brain_hybrid — baseline temp=1.0 at 200 epochs is optimal (Phase 57).

See `delta/brain.py` for implementation. See [The Brain](the-brain.md) for the long-term vision.

---

## Self-Bootstrap: Removing the Transformer Scaffold

Graph construction faces a bootstrapping dilemma: you need to understand the input to build the graph, but the graph is how you understand input.

**The journey:**

- **Phases 5-27b (Transformer scaffold):** A lightweight transformer bootstraps the initial graph. Phase 27b confirms graph structure helps (+4.4% over Transformer), but the hard-thresholded constructor is the bottleneck.
- **Phase 36 (Constructor at scale):** GraphConstructor adds <=1.3% over fixed topology — hard thresholding blocks gradient flow, not the concept of learned construction.
- **Phase 38 (Differentiable construction):** Gumbel-sigmoid replaces the hard threshold. Hybrid variant (base topology + learned edges) reaches **98% of FixedChain** with minimal variance.
- **Phase 39 (Self-bootstrap):** Replace the transformer with a FixedChain DELTA layer. DELTA bootstraps DELTA — no transformer anywhere. SelfBootstrap reaches **157% of FixedChain** (0.757 vs 0.481 accuracy, 3-seed average).

**Real-data confirmation:** Phase 40 validates transfer to FB15k-237: SelfBootstrapHybrid MRR 0.5089, within 0.004 of GraphGPS — the best-performing DELTA variant on real data.

**Why it works:** DELTA-enriched embeddings contain relational information that raw positional embeddings lack. The constructor builds a graph informed by DELTA's own understanding of the input structure — each pass makes the next pass smarter. This is the mechanism behind [The Brain](the-brain.md).

See [Validation Phases](validation-phases.md) for full results.

---

## How DELTA Differs

The gap between GNN and DELTA is most visible under stress: at 80% corrupted features (Phase 28), DELTA's edge-aware attention maintains +24% accuracy over standard GNN approaches. Nodes can reason about their neighbors' *relationships*, not just their neighbors' *values*. When node features are noisy, the relational structure (edge-to-edge patterns) remains intact. See [Key Findings](key-findings.md) for detailed analysis.

| Feature | Transformer | GNN | DELTA |
|---------|------------|-----|-------|
| Input structure | Flat sequence | Graph (nodes + scalar edges) | Graph (nodes + rich edges) |
| Edge representation | None (implicit via attention) | Scalar weight | First-class learned embedding |
| Edge-to-edge reasoning | No | No | Yes (via edge adjacency) |
| Compositional reasoning | Must learn from position | Limited by message passing | Native via multi-hop edge attention |
| Noise robustness | Moderate | Degrades at high noise | **+24% at 80% noise** |
| Relational inductive bias | None | Moderate (message passing) | Strong (edge-first dual attention) |
| Scaling | O(N^2) | O(N+E) | O(E^0.97) with sparse ops |

---

## Development Timeline

DELTA has gone through eight development stages, each building on validated results from the previous stage.

| Stage | Phases | Milestone |
|-------|--------|-----------|
| 1. Core Validation | 1–15 | Edge-first dual attention proven: 2-hop edge adjacency hits 100% on derived relations, O(n^0.81) scaling |
| 2. Pitfall Fixes | 16–21 | Six architectural fixes: post-attention pruning, sparse COO, BFS partitioning, variational memory, per-layer constructor, learned dropout |
| 3. Soft Gating | 16 redesign | Soft sigmoid gating achieves 100% at 50% sparsity, resolving the non-differentiable hard top-k problem |
| 4. Scale Validation | 22–24 | Full architecture validated at 10x scale; DELTA matches CompGCN, crushes TransE/RotatE on realistic KG benchmark |
| 5. Real-World GPU | 25–37 | Real-data benchmarks: DELTA+Gate 97.4% on FB15k-237, noise robustness (+24%), self-bootstrap (157%), frozen encoder transfer (0.961) |
| 6. Compositional Reasoning | 38–45 | Multi-hop dominance: 5p MRR 0.790 vs GraphGPS 0.690; only model improving with depth; inference 0.8–0.9x GraphGPS |
| 7. Temperature Optimization | 46–54 | Learnable temperature reveals edge/node asymmetry; LP MRR 0.4905; LP/3p trade-off characterized as fundamental |
| 8. Brain Architecture | 55–57 | Differentiable graph construction via BrainEncoder; matches delta_full MRR with +4.7% H@10 |

See [Validation Phases](validation-phases.md) for complete results. See [Status & Roadmap](status-and-roadmap.md) for current priorities.

---

## Backward Compatibility

All 6 architectural fixes are additive — they extend existing classes or add new ones without modifying interfaces used by earlier phases. No regression was introduced during the fix implementation cycle.

**Verification against critical phases:**

| Phase | Metric | Original | After Fixes | Status |
|-------|--------|----------|-------------|--------|
| 1 | Edge Attention accuracy | 100% | 100% | Match |
| 7 | Gumbel routing at 60% sparsity | 62.5% | 62.5% | Match |
| 9 | DELTA Edge multi-hop | 84.4% | 84.4% | Match |
| 13 | DELTA 2-hop on derived | 100% | 100% | Match |
| 15 | Full / Router@50% | 100% / 65.3% | 100% / 74.7% | Improved |

Phase 15's improvement (65.3% to 74.7%) results from the legacy `ImportanceRouter` wrapper now delegating to `PostAttentionPruner.prune()` with `min()` safety bounds.

**Fix inventory:**

| Fix | Files | Impact |
|-----|-------|--------|
| PostAttentionPruner | `router.py` | New class; `ImportanceRouter` delegates to it |
| BFS Partitioner | `partition.py` | New function; old spectral method still accessible |
| Variational Memory | `memory.py` | New bottleneck option; default behavior unchanged |
| Per-Layer Constructor | `constructor.py` | New parameters; default single-layer preserved |
| Sparse COO | `graph.py` | Drop-in replacement for dense adjacency |
| Learned Dropout | `router.py` | New class; uniform dropout still available |

---

*See [The Brain](the-brain.md) for long-term vision. See [Validation Phases](validation-phases.md) for all experiment results.*
