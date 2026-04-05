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

### Phase 46: Capacity Signal Measurement

**Goal:** Empirically measure the signals that would drive adaptive architecture decisions, without modifying the architecture yet. This is the "instrument before you intervene" phase.

**What to build:**
```
experiments/phase46_capacity_signal.py
```

Train DELTA-Matched and DELTA-Full with router enabled (`use_router=True`). At each evaluation checkpoint, record:

1. **Per-layer importance entropy** — For each DELTALayer, measure the entropy of the edge gate distribution. Low entropy = model has clear signal about what matters. High entropy = model is uncertain, possibly under-capacity.

2. **Layer-wise gate sparsity** — What fraction of edge gates are below 0.1? Below 0.01? Track across training. If sparsity increases over training, the model is learning what to prune. If stable, the architecture may already be near-optimal.

3. **Attention head utilization** — Per-head mean attention weight. Dead heads (near-zero) are pruning candidates. In DELTA-Matched (2 layers × 4 heads = 8 heads) vs DELTA-Full (3 layers × 4 heads = 12 heads), do the extra heads contribute?

4. **Cross-depth routing consistency** — When evaluating 1p vs 3p vs 5p queries, do the same edges get high importance scores? Or does the model route differently depending on reasoning depth? If routing changes with depth, the model has learned depth-dependent computation paths.

**Measurement protocol:**
- Train both models (delta_matched, delta_full) for 500 epochs with `use_router=True`
- Record importance scores at every eval checkpoint (every 25 epochs)
- Evaluate on 1p/2p/3p/4p/5p queries post-training
- Compare importance distributions between models and across query depths

**Expected outcome:** Empirical evidence for whether the router already contains the capacity signal needed for self-modification. If DELTA-Full's extra heads/layers consistently show low importance while DELTA-Matched's are all high-utilization, the data is telling us the optimal capacity — we just need to listen.

---

### Phase 47: Importance-Driven Pruning

**Goal:** The router tells us what to prune. See if post-training compression preserves (or improves) multi-hop performance.

**What to build:**
```
experiments/phase47_importance_pruning.py
```

**Protocol — Static Pruning (simplest possible test):**

1. Train DELTA-Full (293K params, 3 layers, d_node=64)
2. Collect importance scores across full training run
3. Identify persistently low-importance components:
   - Attention heads where mean attention < threshold across all checkpoints
   - Layer dimensions where gate values are consistently < 0.1
4. Create a **compressed model** by removing identified components:
   - Prune dead heads (reduce `num_heads` per layer)
   - Reduce `d_node`/`d_edge` by removing low-variance dimensions
   - Potentially remove an entire layer if layer-3 gates are consistently near-zero
5. **Knowledge distillation:** Initialize compressed model from surviving parameters of the full model (no random init — inherit the learned weights)
6. Fine-tune for 50 epochs on the same data
7. Evaluate on 1p–5p multi-hop queries

**Success criteria:**
- Compressed model matches or beats DELTA-Full on 3p/5p
- If the compressed model converges to ~157K params and matches DELTA-Matched, the capacity signal is real
- Timing: compressed model should train/infer faster

**Key insight this tests:** Whether DELTA-Full's degradation on multi-hop is because it *has* excess capacity (and pruning fixes it) or because it *learned* something different (and pruning destroys it). If pruning works, the degradation is a capacity problem. If it doesn't, the degradation is a learning dynamics problem.

---

### Phase 48: Curriculum Compression

**Goal:** Instead of train-then-prune, compress *during* training with a schedule.

**What to build:**
```
experiments/phase48_curriculum_compression.py
```

**Protocol — Gradual Pruning During Training:**

This follows the same curriculum warmup pattern that already works in DELTA (Gumbel temperature annealing, GRL λ schedule):

1. **Phase 1 (Epochs 1–100): Full capacity.** Train DELTA-Full normally. The model has maximum capacity to explore the loss landscape.

2. **Phase 2 (Epochs 100–300): Increasing sparsity pressure.** Linearly increase the `target_sparsity` parameter in `PostAttentionPruner.soft_prune()` from 0.0 → 0.5. The sparsity loss pushes edge gates toward binary values. Components the model doesn't need will have their gates driven to zero.

3. **Phase 3 (Epochs 300–400): Structural commitment.** At epoch 300, evaluate which components have gate values below 0.05. **Physically remove them** — reduce tensor dimensions, remove heads, potentially remove layers. Reinitialize the optimizer with the new parameter set.

4. **Phase 4 (Epochs 400–500): Fine-tuning.** Train the compressed model at its new (potentially much smaller) capacity.

**Key schedule parameters:**
```python
sparsity_schedule = {
    'warmup_epochs': 100,      # Full capacity exploration
    'compression_start': 100,   # Begin sparsity pressure
    'compression_end': 300,     # Target sparsity reached
    'commit_epoch': 300,        # Physical pruning
    'final_epochs': 500,        # Fine-tune compressed model
    'target_sparsity': 0.5,     # 50% of edges pruned
}
```

**What makes this different from standard pruning:**
- The router's importance scores are computed **post-attention** (not predicted)
- Pruning decisions are based on *observed* attention patterns, not magnitude
- The sparsity loss is differentiable — the model can learn to route around pruning
- The curriculum gives the model time to reorganize before commitment

**Comparison matrix:**
| Method | 1p | 3p | 5p | Final params | Train time |
|--------|----|----|----|----|------|
| DELTA-Matched (157K, fixed) | baseline | baseline | baseline | 157K | ~3600s |
| DELTA-Full (293K, fixed) | higher | lower | — | 293K | ~? |
| DELTA-Full → Curriculum Compressed | ? | ? | ? | ? | ~? |

If curriculum compression from 293K converges to ~160K params and matches DELTA-Matched on 3p/5p, **the model has discovered the optimal capacity from the data.**

---

### Phase 49: Bidirectional Adaptive Architecture (The Brain Prototype)

**Goal:** The full loop — DELTA monitors its own capacity utilization and triggers structural transitions (both compression and expansion) based on data complexity signals.

This is the mechanism Sonnet identified: **bidirectional adaptive architecture driven by the model's own routing signals.**

**What to build:**
```
delta/adaptive.py
experiments/phase49_adaptive.py
```

**Architecture: AdaptiveDELTA**

```python
class AdaptiveDELTA(nn.Module):
    """DELTA with self-modifying architecture.
    
    Monitors two signals:
    1. Low routing entropy → model has clear signal → maintain or compress
    2. High routing entropy → model is uncertain → consider expansion
    
    Uses self-bootstrap as the mechanism for structural transitions.
    """
    
    def __init__(self, initial_config, data_stats):
        # Start with initial_config (e.g., DELTA-Full or DELTA-Matched)
        self.model = create_delta_from_config(initial_config)
        self.importance_history = ImportanceHistory(window=100)
        self.entropy_tracker = EntropyTracker(window=50)
        
    def forward(self, graph):
        # Standard forward pass with router enabled
        result = self.model(graph, use_router=True)
        
        # Record importance signals
        for layer in self.model.layers:
            self.importance_history.update(layer.pruner)
            self.entropy_tracker.update(layer.pruner)
        
        return result
    
    def should_compress(self):
        """Low entropy + many near-zero gates = excess capacity."""
        return (self.entropy_tracker.mean_entropy() < self.compress_threshold
                and self.importance_history.near_zero_fraction() > 0.3)
    
    def should_expand(self):
        """High entropy = model can't decide what matters = under-capacity."""
        return self.entropy_tracker.mean_entropy() > self.expand_threshold
    
    def restructure(self, data, val_queries):
        """Execute structural transition via self-bootstrap."""
        if self.should_compress():
            # Generate compressed config from importance history
            new_config = self.importance_history.optimal_config()
            candidate = create_delta_from_config(new_config)
            
            # Knowledge distillation: initialize from surviving params
            candidate = distill_from(self.model, candidate)
            
            # Verify on held-out reasoning task
            if evaluate_multihop(candidate, val_queries) >= \
               evaluate_multihop(self.model, val_queries) * 0.98:
                self.model = candidate  # Commit
                
        elif self.should_expand():
            # Bootstrap upward: add capacity where entropy is highest
            new_config = self.entropy_tracker.expansion_config()
            candidate = create_delta_from_config(new_config)
            candidate = bootstrap_from(self.model, candidate)
            self.model = candidate
```

**The ImportanceHistory tracker:**
```python
class ImportanceHistory:
    """Tracks importance scores across batches to identify structural trends."""
    
    def update(self, pruner):
        """Record current gate values per component."""
        # Per-head: mean attention weight per head
        # Per-layer: gate distribution statistics
        # Per-dimension: variance of node feature dimensions
        
    def near_zero_fraction(self):
        """Fraction of tracked components with mean gate < 0.05."""
        
    def optimal_config(self):
        """Generate model config that removes near-zero components."""
        # Keep heads where mean_gate > 0.1
        # Keep dimensions where variance > threshold
        # Keep layers where mean edge_gate > 0.1
        return DELTAConfig(
            d_node=surviving_node_dims,
            d_edge=surviving_edge_dims,
            num_layers=surviving_layers,
            num_heads=surviving_heads_per_layer,
        )
```

**Test protocol:**
1. Start with DELTA-Full (293K)
2. Train for 100 epochs (warmup)
3. Every 50 epochs: check `should_compress()` / `should_expand()`
4. If triggered: execute `restructure()` with knowledge distillation
5. Track: parameter count over time, 3p/5p MRR over time, compression decisions

**Expected behavior:** DELTA-Full starts at 293K, compresses during training to ~150-170K, and matches or beats the manually-tuned DELTA-Matched on multi-hop reasoning. The model finds the optimal capacity without us telling it.

---

### Phase 50: Multi-Scale Adaptive (Toward The Brain)

The full vision: DELTA doesn't just adjust its own capacity — it adjusts its capacity *differently for different reasoning depths*.

**Observation from Phase 44:** The importance routing pattern on 1p vs 5p queries should be different. 1p is local (one relation traversal). 5p requires compositional reasoning across 5 hops. The model might need different architectural capacity at different reasoning scales.

**Concept: Depth-Conditioned Architecture**
```
Query type detected (shallow / deep)
    ↓
Route to appropriate architecture branch:
    1p-2p → Compact branch (DELTA-Matched-like, fast)
    3p-5p → Full branch (more heads, more capacity for composition)
```

This is a mixture-of-experts at the architectural level — not routing tokens to experts, but routing reasoning tasks to architectures.

---

## Naming

The adaptive self-modification mechanism needs a name. Candidates:

- **PRISM** — Pruning Router for Intrinsic Structural Modification
- **SYNAPSE** — Self-organizing Yielding Network for Adaptive Parameter Space Evolution  
- **DELTA-S** — DELTA with Synaptic plasticity

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
