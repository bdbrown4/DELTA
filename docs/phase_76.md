# Phase 76 — DIAGNOSE_5p: the sole surviving positive is CAPACITY, not structure

## Result

```
Phase: 76 — Attribute the 5p edge-attention benefit (tau-sweep + random-adjacency capacity control)
Hypothesis: hops1/hops2 beat node_only by ~+0.02 at 5p because of GENUINE STRUCTURAL-LOCAL edge
  composition — so the gap (a) survives traversal temperature tau->0 and (b) requires the LEARNED
  STRUCTURAL edges (a random adjacency of matched size should NOT reproduce it).
Seeds: [42, 123, 456, 7, 99]; conditions node_only / hops1 / random_struct (5/5 each);
  hops2 OOM'd before completing (corroboration only — verdict is independent of it).
Result: REAL BUT CAPACITY. The 5p benefit is genuine (not a tau artifact) but is reproduced by
  RANDOM edges; the structural-specific contribution is not significant. Structural composition REJECTED.

Metrics (edge-sampled FB15k-237 frac=0.10, 11.5K ents / 18K test / deg 4.7, 5 seeds x 600 epochs;
         multi-hop MRR, paired per-seed, bootstrap 95% CI):
TEST 1 (artifact?) hops1 - node_only, 5p gap by traversal temperature tau:
  tau=0.1 +0.0138 [-0.008,+0.035]   tau=0.5 +0.0282 [+0.004,+0.060]   tau=1.0 +0.0264 [+0.003,+0.050]
  tau=2.0 +0.0135 [-0.001,+0.025]   tau=5.0 +0.0108 [+0.004,+0.018]
  -> positive at every tau, significant at 0.5/1.0/5.0, survives hard traversal => NOT an artifact.

TEST 2 (capacity vs structure?) 5p gap vs node_only (tau=1.0):
  hops1         - node_only  +0.0264 [+0.003,+0.050]   (4/5 seeds)
  random_struct - node_only  +0.0247 [-0.008,+0.058]   (4/5 seeds)  <- random reproduces ~95%
  hops1 - random_struct       +0.0017 [-0.016,+0.020]  (3/5 seeds)  <- NOT significant
LP (1p) MRR: node_only 0.0706, hops1 0.0682, random_struct 0.0680  (tied — no LP benefit)

Key insight: the 5p edge-attention benefit is REAL but is a CAPACITY/connectivity effect (more
  neighbors to pool over at the deepest hop), NOT structural-local composition: a RANDOM adjacency
  of matched size reproduces ~95% of it, and the structural-specific advantage (hops1 - random_struct)
  is +0.002, not significant. This is the same verdict Phase 75 reached for content routing's LP gain.
  Across Phases 66-76, DELTA's edge-to-edge structural composition has NO robust advantage over random
  connectivity at ANY hop depth in this setup.
Next question: the static-embedding eval cannot distinguish structural composition from random edges
  (random ~= structural at every hop) -> the ONLY fair test of the thesis is a query-time ("JIT")
  edge-composition mechanism + matching training objective. That is the pinned next direction.
Status: LOGGED as REJECTED (structural composition). The core thesis has no robust empirical support
  in this setup; the value of edge attention here is generic capacity. next_phase = JIT pivot.
```

## Details

### What was tested

The sole surviving positive after Phases 66-75 was hops1/hops2 beating node_only by ~+0.02 at the
deepest hop (5p). Two pre-registered tests attributed it, on re-trained + checkpointed conditions
(Phase 73 didn't save weights), eval-only sweeps via the ~90x vectorized eval + a precompute-once fix
(the redundant per-query graph walk had made an earlier run ~100x too slow):

- **TEST 1 (artifact):** multi-hop eval at traversal temperature tau in {0.1, 0.5, 1.0, 2.0, 5.0}.
  Genuine if the gap survives tau->0 (near-hard traversal); artifact if it only exists at tau=1.0.
- **TEST 2 (capacity):** a `random_struct` condition — a RANDOM edge adjacency of the same pair count
  as hops1. If random reproduces the 5p gap, the benefit is "having edges" (capacity), not structure.

### Key Observations

1. **Not an artifact.** hops1 - node_only at 5p is positive at every temperature and significant at
   tau=0.5/1.0/5.0, including under sharp traversal (tau=0.1: +0.014). The effect is real.
2. **It is capacity, not structure.** random_struct - node_only at 5p (+0.025) ~ matches hops1 -
   node_only (+0.026); the direct structural-specific contrast hops1 - random_struct is +0.002 and
   not significant (3/5 seeds, CI [-0.016, +0.020]). Random connectivity reproduces ~95% of the benefit.
3. **No LP benefit.** node_only / hops1 / random_struct LP MRR are 0.071 / 0.068 / 0.068 — tied.
4. **Consistent with Phase 75.** content routing's LP gain was also capacity (frozen-random reproduced
   it). Now the structural 5p gain is also capacity (random reproduces it). Same blade, both levers.
5. **hops2 OOM** (corroboration). The run died training hops2 from CUDA memory accumulated across the
   sequential trainings in one process (fixable with torch.cuda.empty_cache() between conditions). The
   verdict does not depend on hops2 — the decisive conditions (node_only/hops1/random_struct) all completed.

### Classification: REJECTED (structural composition)

DELTA's edge-to-edge structural composition provides no robust advantage over random connectivity at
any hop depth in this setup. The measurable value of edge attention is a generic capacity/connectivity
effect. The core thesis is not supported.

### Impact

- **Project:** the empirical arc (Phases 66-76) is complete and the answer is honest and clean — edge
  attention's specific structural mechanism does not beat random wiring on this benchmark/eval. The
  defensible facts are: 2p->3p depth-monotonicity (a within-condition property), and that the "edge
  benefit" reduces to capacity. The "relational composition" thesis is refuted across structural
  (2-hop), learned-content, and now structural-vs-random-capacity tests.
- **Methodological (the durable contribution):** this is the cleanest demonstration that the standard
  soft-traversal-over-static-embeddings multi-hop protocol CANNOT distinguish structural composition
  from random connectivity (random ~= structural at every hop). Any architecture evaluated this way is
  measured only on embedding/capacity quality, never on query-time composition.
- **Forward (pinned):** the query-time ("JIT") pivot — invoke edge-to-edge composition along the query's
  relation chain at inference, with a matching training objective (NBFNet / path-GNN family) — is the
  only fair test of the original intuition, and is now strongly motivated. Tonight's checkpoints +
  testbed + controls are the AOT baseline arm for that comparison.

### Next Steps

1. JIT pivot scoping (see research_state future direction): query-time edge-conditioned inference +
   matching training objective; reuse this testbed, the controls discipline, and these checkpoints as
   the encode-once baseline-to-beat.
2. Paper (camera-ready / rebuttal): correct the submitted draft's 2-hop / composition headline; lead on
   depth-monotonicity + the rigorously-attributed negatives (capacity, not composition).
3. Infra: add torch.cuda.empty_cache() between conditions in long sequential runs (the hops2 OOM).
```
