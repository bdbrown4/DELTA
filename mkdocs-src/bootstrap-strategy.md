# Bootstrap Strategy

## The Chicken-and-Egg Problem

Graph construction faces a fundamental bootstrapping challenge: you need to understand the input to build the graph, but the graph is how you understand input. DELTA solves this pragmatically:

1. **Use a lightweight transformer** to bootstrap an initial graph from raw input
2. **DELTA processes and refines** the graph
3. **Over time**, trained DELTA models can replace the transformer bootstrap — using their own graph representations to construct graphs for new input
4. **The transformer is scaffolding**, not a permanent dependency

Phase 5 confirms the pipeline preserves accuracy: with equal training (150 epochs each), the transformer→DELTA pipeline matches the transformer alone (98.3%) on a non-relational task.

---

## Phase 27b: The Real Test

Phase 27b clarified the bootstrap's true role. On a 2-hop path composition task (16 relational classes, N=1000), with properly trained models (gradient accumulation, LR scheduler):

| Model | Accuracy | vs. Random (6.2%) |
|---|---|---|
| Fixed Chain DELTA | **40.7%** | 6.6× |
| Transformer | 36.3% | 5.9× |
| Bootstrap DELTA (GraphConstructor) | 34.3% | 5.5× |

---

## Key Conclusions

### Graph structure genuinely helps relational tasks
Fixed Chain DELTA beats the pure Transformer by +4.4% using the same transformer embeddings but with explicit adjacency structure.

### The GraphConstructor is the bottleneck, not graph processing
Attention-thresholded construction (Bootstrap) *underperforms* the fixed chain — the constructor discards sequential connections essential for path composition.

### Phase 27's original result was entirely training-confounded
Phase 27 (DELTA << Transformer) was caused by batch-1 gradient updates with Adam producing chaotic, non-converging updates specifically for deeper DELTA models. Transformer was less affected because it processes the full batch in one shot regardless.

### The fix is task-aware construction
A constructor that preserves positional/path ordering for sequential tasks would give Bootstrap DELTA the benefits of both worlds. This was tested in Phase 33/36.

---

## Phase 36: Constructor at Scale

Phase 36 scaled the constructor test to 500–5000 nodes with three experiment types:

- **Sparse path graphs** (33% edges removed): Both fixed and augmented reach 1.000 at all scales
- **Cross-cluster reasoning** (bottleneck bridges): Best case +1.3% on smallest/sparsest config; effect vanishes with more nodes
- **Edge threshold sweep**: Aggressive threshold (0.05) actively hurts accuracy

**Conclusion:** The GraphConstructor adds no measurable value. DELTA's core architecture (edge-centric dual attention + 2-hop adjacency) is powerful enough that the given topology is sufficient. The constructor is de-emphasized in favor of the core architectural contributions.

---

## Long-Term Vision

Replacing the transformer bootstrap entirely requires:

- **Phase 35** (domain-agnostic transfer) — ✅ Encoder already domain-invariant (probe 0.961 on WN18RR)
- **Phase 36** (constructor at scale) — ❌ Constructor adds ≤1.3%

**Current assessment: not yet ready.** Zero-shot transfer is solved at the encoder level, but the constructor doesn't add sufficient value to justify the complexity. The transformer bootstrap remains the pragmatic choice for graph construction from raw input.

---

*See [Validation Phases](validation-phases.md) for Phase 27b and 36 result tables, [Colab Results](COLAB_RESULTS.md) for Phase 36 detailed analysis.*
