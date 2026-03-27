# Why DELTA? — Transformer vs Graph Neural Net vs DELTA

Interactive visual: [assets/transformer_vs_graph_vs_delta.html](assets/transformer_vs_graph_vs_delta.html)

---

## The Three Paradigms

### 1. Transformer

Tokens in a flat sequence, every token attending to every other. The model has no native concept of relationships — it has to discover that "Paris" and "France" are related by reading the words in order and learning the pattern. That's the O(N²) problem: as sequences get longer, the attention cost explodes.

**What it does well:** Flexible, learns arbitrary patterns from data.
**What it lacks:** No structural inductive bias. Relationships must be reconstructed from position alone.

### 2. Graph Neural Network

Relationships are explicit as edges, which is better. But edges are just scalar weights — they're passive wires, not thinkers. The edge connecting Paris→France just says "0.91 strong" and moves signal along. It can't reason about *what kind* of relationship that is, or how it relates to the Berlin→Germany edge.

**What it does well:** Exploits relational structure, message-passing is efficient.
**What it lacks:** Edges are passive conduits. No mechanism for reasoning about relationships between relationships.

### 3. DELTA (Dual Edge-Level Transformer Architecture)

DELTA promotes edges to **first-class computational citizens**. Nodes and edges both carry rich representations and attend to each other simultaneously in parallel streams.

The key mechanism: **edge-to-edge attention**. The "capital of" edge between Paris and France can *attend* to the "capital of" edge between Berlin and Germany — and recognize they're the same relationship type. That's structural analogy, compositional reasoning, and relational inference all in one mechanism.

On top of that:
- **Tiered memory** on every node (hot / warm / cold) — the router manages what's active vs archived
- **Importance router** — scores nodes and edges to guide sparse attention
- **Parallel dual streams** — node attention and edge attention run simultaneously

**What it does well:** Everything above, plus it degrades gracefully under noise.
**What it lacks:** Higher per-layer cost than vanilla GNN (offset by fewer layers needed).

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

## The Visual Explained

Open `assets/transformer_vs_graph_vs_delta.html` in a browser. Click through all three tabs:

1. **Transformer tab** — A flat sequence of tokens with O(N²) self-attention. "Paris" must discover its relationship to "France" purely through attention weights. The green lines show where attention concentrates, but every token must attend to every other token to find these patterns.

2. **Graph Neural Net tab** — Nodes connected by edges with scalar weights. Paris→France has weight 0.91, but that edge is just a wire — it carries signal, it doesn't compute. The edge can't look at the Berlin→Germany edge and recognize they encode the same relationship.

3. **DELTA tab** — The red dashed arrow is the key. Edge nodes (green rounded rectangles) represent "capital of" and "located in" as rich learned representations. The edge-to-edge attention arrow means the "capital of" edge can attend to other edges and discover structural analogies. Below that, the importance router and tiered memory complete the architecture.

---

## Connection to DELTA's Experiment Phases

| Phase | What it validates | Paradigm gap addressed |
|-------|-------------------|----------------------|
| Phase 28 | Noise robustness (+24% at 80% noise) | Edge-to-edge attention preserves relational structure when node features degrade |
| Phase 27b | Attention-topology interaction | Shows how edge attention interacts with graph structure |
| Phase 33 | Task-aware graph construction | Hybrid constructor preserves base topology while learning long-range edges |
| Phase 34 | DELTA vs GraphGPS vs GRIT | Head-to-head comparison with state-of-the-art graph transformers |
| Phase 25 | Full-scale FB15k-237 | Validates edge-aware reasoning at knowledge graph scale (14,505 entities) |

---

*Visual and writeup created March 2026.*
