# Bootstrap Strategy

## The Chicken-and-Egg Problem

Graph construction faces a fundamental bootstrapping challenge: you need to understand the input to build the graph, but the graph is how you understand input. DELTA's journey from external scaffolding to full self-bootstrap:

1. **Phase 5–27b:** Use a lightweight transformer to bootstrap an initial graph
2. **Phase 33/36:** Try to make the constructor contribute — hard thresholding blocks gradients
3. **Phase 38:** Differentiable construction with Gumbel-sigmoid — Hybrid reaches 98% of FixedChain
4. **Phase 39:** **Self-bootstrapped DELTA** — no transformer anywhere. DELTA bootstraps DELTA at 157% of FixedChain.

**The transformer scaffold is down.**

---

## Phase 27b: The Real Test

Phase 27b clarified the bootstrap's true role. On a 2-hop path composition task (16 relational classes, N=1000):

| Model | Accuracy | vs. Random (6.2%) |
|---|---|---|
| Fixed Chain DELTA | **40.7%** | 6.6× |
| Transformer | 36.3% | 5.9× |
| Bootstrap DELTA (GraphConstructor) | 34.3% | 5.5× |

**Key insight:** Graph structure helps (+4.4% over Transformer), but the hard-thresholded constructor is the bottleneck, not graph processing.

---

## Phase 36: Constructor at Scale

Phase 36 scaled the constructor test to 500–5000 nodes. Result: the hard-thresholded GraphConstructor adds ≤1.3% over fixed topology. Not because learned construction is impossible, but because **hard thresholding blocks gradient flow**.

---

## Phase 38: Differentiable Construction

*(Was Phase 46. Experiment file: `experiments/phase46_differentiable_constructor.py`)*

Phase 38 identified the root cause: `attn > threshold` is non-differentiable — task loss cannot flow back through edge decisions. Three Gumbel-sigmoid variants were tested. **Full 3-seed results:**

| Variant | Accuracy (3 seeds) | vs FixedChain |
|---------|-------------------|---------------|
| Transformer (control) | 0.387 ± 0.031 | 84% |
| **FixedChain** (control) | **0.461 ± 0.034** | 100% |
| DifferentiableConstructor | 0.393 ± 0.017 | 85% |
| TaskConditionedConstructor | 0.397 ± 0.005 | 86% |
| **HybridConstructor** | **0.452 ± 0.006** | **98%** |

**Hybrid wins:** Preserving base topology + learning additional edges reaches 98% of FixedChain with minimal variance. Pure differentiable construction is harder — learning everything from scratch plateaus at ~85%.

---

## Phase 39: Self-Bootstrapped DELTA — The Scaffold Comes Down

*(Was Phase 46b. Experiment file: `experiments/phase46b_self_bootstrapped.py`)*

The breakthrough: replace the transformer bootstrap with a FixedChain DELTA layer. The system is now **DELTA all the way down** — a lightweight DELTA pass bootstraps embeddings, those embeddings feed the constructor, and a full DELTA stack processes the resulting graph.

**Full 3-seed results:**

| Model | Accuracy (3 seeds) | vs FixedChain |
|-------|-------------------|---------------|
| Transformer | 0.429 ± 0.021 | 89% |
| FixedChain | 0.481 ± 0.015 | 100% |
| P38_Hybrid | 0.459 ± 0.036 | 95% |
| **SelfBootstrap** | **0.757 ± 0.041** | **157%** |
| SelfBootstrapHybrid | 0.716 ± 0.038 | 149% |

### Why +57% Over FixedChain?

The self-bootstrap DELTA runs a full edge-attention + reconciliation pass *before* the constructor. These DELTA-enriched embeddings contain relational information that raw positional embeddings lack. The constructor doesn't just build a better graph — it builds a graph informed by DELTA's own understanding of the input structure.

This is the mechanism behind "The Brain": the system uses its own relational reasoning to decide what relationships exist, then reasons over those relationships, then refines its understanding. Each pass makes the next pass smarter.

### Updated Bootstrap Assessment

- ✅ **Encoder transfer** — Phase 35: encoder is domain-invariant (probe 0.961 on WN18RR)
- ✅ **Differentiable construction** — Phase 38: Hybrid matches hand-coded topology (98%)
- ✅ **Self-bootstrap** — Phase 39: DELTA bootstraps DELTA at 157% of FixedChain
- ⏳ **Real-data validation** — Phase 40: Correct LP evaluation on FB15k-237 (in progress)

**The path from external scaffolding to full self-bootstrap is validated on synthetic tasks.** The remaining question is whether the self-bootstrap advantage transfers to real knowledge graphs — Phase 40 is designed to answer this.

---

*See [The Brain](the-brain.md) for the long-term vision. See [Validation Phases](validation-phases.md) for all result tables.*
