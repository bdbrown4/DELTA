# Architecture Overview

## Core Thesis

Reality is a graph. Language is a lossy compression of reality into sequences. Transformers reconstruct relational structure from flat sequences. DELTA operates on relational structure directly.

**The three-paradigm gap — visual explainer:** The [Visual Explainer](ARCHITECTURE_VISUAL.md) walks through Transformer → GNN → DELTA with an interactive diagram embedded directly in the page. The key insight: GNN edges are passive scalar wires; DELTA edges are first-class computational citizens that attend to each other. That edge-to-edge attention is what produces the Phase 28 +24% noise robustness gap.

---

## Architecture Diagram

```
Raw Input (any modality)
    → Graph Constructor (transformer-bootstrapped: per-layer edge projections + typed edges)
    → BFS Graph Partitioner (O(N+E) seed-expansion clustering)
    → PARALLEL DUAL ATTENTION
        [Node Attention + Edge Attention across all partitions simultaneously]
    → Post-Attention Pruner (prune based on OBSERVED attention weights)
    → Learned Attention Dropout (per-edge regularization)
    → Reconciliation Layer (nodes and edges co-update)
    → Hierarchical Global Attention (cluster representatives)
    → Variational Memory Compression (warm nodes → bottleneck → KL regularization)
    → Memory Tier Update (importance-based hot/warm/cold promotion)
    → Output + Updated Graph State
```

---

## Key Architectural Components

### DualParallelAttention

Node attention and edge attention run simultaneously in parallel streams. Both operate as standard multi-head self-attention but over different domains:

- **Node attention:** Nodes attend to neighboring nodes via the edge index
- **Edge attention:** Edges attend to structurally adjacent edges via the edge adjacency matrix

The dual parallel design converges 2.7× faster than sequential alternatives (Phase 2).

### Multi-Hop Edge Adjacency

The mechanism that enables compositional reasoning. Two edges are "adjacent" if they share a node — this creates an edge-to-edge connectivity structure. At 2 hops, an edge can "see" edges two hops away, enabling transitive inference.

**Phase 11 result:** 2-hop edge adjacency achieves **100% accuracy on derived/transitive relations** — a +38.9% jump from 1-hop (61.1%) and beating Node GNN (83.3%). This was the biggest architectural improvement in the project.

Implemented as sparse COO tensors for O(E^0.97) scaling (Phase 17), handling 2500+ edges where the original dense approach timed out at ~500 edges.

### ReconciliationBridge

The cross-stream interaction mechanism. After both attention streams complete, the ReconciliationBridge co-updates nodes and edges:

1. **Edges absorb node context:** Each edge concatenates its features with its source and target node features, then projects back to edge dimension
2. **Nodes absorb edge context:** Each node aggregates updated edge features from all incident edges, concatenates with its own features, then projects back to node dimension

Both updates use residual connections and LayerNorm. This sequential cascade (concat → linear → LayerNorm) provides direct high-bandwidth gradient flow.

!!! note "Cross-Stream Interaction Comparison"
    A [prototype comparison](https://github.com/bdbrown4/DELTA/blob/main/experiments/prototypes/shared_latent_bottleneck.py) tested ReconciliationBridge against two alternatives:

    - **ReconciliationBridge:** 0.889 ± 0.050 val — solved the multi-hop task
    - **Cross-Attention Gates:** 0.218 ± 0.009 — at majority baseline (0.214), learned nothing
    - **Shared Latent Bottleneck:** 0.210 ± 0.010 — at majority baseline, learned nothing

    ReconciliationBridge's direct linear mixing is fundamentally superior to gated/bottlenecked alternatives. The gates in cross-attention designs never open properly — the cross-stream path is a dead end from epoch 1.

### PostAttentionPruner

Prunes graph elements based on *observed* attention weights rather than predicting importance before attention (the original router's chicken-and-egg problem). Uses soft sigmoid gates with per-head attention features.

**Phase 16 result:** Soft gating achieves **100% accuracy at 50% target sparsity** — matching full attention and beating pre-attention routing by +14.7%.

### Graph Constructor

Transformer-bootstrapped graph construction with per-layer edge projections and typed edges. Solves the "chicken-and-egg" problem: use a lightweight transformer to bootstrap an initial graph, then DELTA refines it.

!!! warning "Limited Contribution"
    Phase 36 showed the GraphConstructor adds ≤1.3% over fixed topology across all tested configurations. DELTA's core architecture (edge-centric dual attention + 2-hop adjacency) is powerful enough that the given topology is sufficient. The constructor is de-emphasized in favor of the core architectural contributions.

### Variational Memory Compression

Tiered memory system (hot/warm/cold) with a variational bottleneck and KL regularization. Preserves accuracy while compressing node representations. KL converges during training (0.126 → 0.026), confirming the latent space is being regularized.

### BFS Graph Partitioner

O(N+E) BFS seed-expansion clustering, replacing the original O(N³) spectral partitioning. Balance ratio of 0.79 with importance-aware seeding across partitions (Phase 20).

---

## Where the Gap Shows Up

The gap between GNN and DELTA is where the **Phase 28 result** lives. At extreme noise levels (80% corrupted features), DELTA's edge-aware attention maintains **+24% accuracy** over standard GNN approaches.

Why? Because nodes can reason about their neighbors' *relationships*, not just their neighbors' *values*. When node features are noisy, the relational structure (edge-to-edge patterns) is still intact — and DELTA can leverage it.

| Noise Level | Standard GNN | DELTA | Gap |
|------------|-------------|-------|-----|
| 0% (clean) | ~95% | ~97% | +2% |
| 20% | ~88% | ~94% | +6% |
| 50% | ~72% | ~86% | +14% |
| 80% | ~54% | ~78% | **+24%** |

*Results from Phase 28 (noise robustness), synthetic benchmark.*

---

## How DELTA Differs

| Feature | Transformer | GNN | DELTA |
|---------|------------|-----|-------|
| Input structure | Flat sequence | Graph (nodes + scalar edges) | Graph (nodes + rich edges) |
| Edge representation | None (implicit via attention) | Scalar weight | First-class learned embedding |
| Edge-to-edge reasoning | No | No | Yes (via edge adjacency) |
| Compositional reasoning | Must learn from position | Limited by message passing | Native via multi-hop edge attention |
| Noise robustness | Moderate | Degrades at high noise | **+24% at 80% noise** |
| Relational inductive bias | None | Moderate (message passing) | Strong (edge-first dual attention) |
| Scaling | O(N²) | O(N+E) | O(E^0.97) with sparse ops |

---

*See also: [Visual Explainer](ARCHITECTURE_VISUAL.md) for the interactive three-paradigm diagram, [Key Findings](key-findings.md) for detailed experimental evidence.*
