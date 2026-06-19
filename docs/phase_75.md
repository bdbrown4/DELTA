# Phase 75 — Content-Dependent (Composability-Routed) Edge Adjacency: Full Ablation

## Result

```
Phase: 75 — Content-dependent edge adjacency ablation (edge-sampled FB15k-237, DELTA-Matched, 5 seeds × 600 epochs)
Hypothesis: edges routed by LEARNED COMPOSABILITY (bilinear tail_key·head_key, top-K) give better
  multi-hop reasoning than structural adjacency — gap GROWING with hop depth — and beat both a
  similarity control and a random-routing control. ("DELTA edges as the composition of structure.")
Expected (pre-registered): content_comp − {hops1, content_sim, content_random} > 0.010 on 3p (paired/5 seeds),
  with non-decreasing depth gap and composable_frac rising during training.
Seeds: [42, 123, 456, 7, 99]
Result: REJECTED — composition does not help; the apparent LP gain is capacity, not content, and
  content routing INVERTS the depth benefit (helps 1p, hurts 5p).

Metrics (multi-hop MRR — edge-sampled FB15k-237, 11,563 entities, deg 4.7, 18,159 test; 10k queries/type;
         5 seeds × 600 epochs; content = bilinear ContentRouter K=32 d_r=16 aux_w=1.0; structural = Phase 73, paired):
Condition         LP            2p            3p            5p           composable_frac
node_only      0.071±0.006   0.089±0.007   0.104±0.012   0.150±0.029        —
hops1          0.068±0.002   0.085±0.009   0.103±0.012   0.167±0.030        —
hops2          0.067±0.002   0.086±0.007   0.102±0.013   0.170±0.028        —
content_comp   0.105±0.009   0.084±0.006   0.102±0.010   0.144±0.020      0.185
content_sim    0.108±0.011   0.084±0.005   0.108±0.012   0.149±0.008      0.528
content_random 0.099±0.006   0.079±0.006   0.098±0.010   0.140±0.020      0.150

Pre-registered pass/fail (content_comp beats X by >0.010 on 3p, paired):
  vs hops1          3p mean −0.0015  per-seed [+0.007,+0.008,+0.005,−0.023,−0.005]  FAIL
  vs content_sim    3p mean −0.0063  per-seed [+0.005,−0.003,−0.005,−0.012,−0.017]  FAIL
  vs content_random 3p mean +0.0037  per-seed [−0.003,+0.011,+0.009,−0.004,+0.004]  FAIL
ALL THREE FAIL.

Key insight: composability routing does not improve multi-hop reasoning. (1) content_comp ties
  content_sim and barely edges content_random — and composable_frac spans 0.15→0.53 across the three
  with NO performance difference, so composability is unpredictive. (2) The +0.03–0.04 LP gain over
  structural is reproduced by content_random (frozen RANDOM keys) -> it is generic capacity/global-reach,
  not learned content and not composition. (3) The content−hops1 gap is INVERTED in depth: +0.004 (1p)
  -> −0.023 (5p); content routing trades the local structure the deep soft-traversal needs.
Next question: the multi-hop eval is soft-traversal over STATIC node embeddings (no query-time edge
  chaining), so it can only reward composition via embeddings — which help 1p, not deep hops. Testing
  composition fairly would require a query-time compositional eval. Absent that, the composition lever
  (anchor/content routing over edges) is exhausted.
Status: LOGGED as REJECTED (decisive, fully attributed). RR_prior control moot (no content-specific
  win to attribute). next_phase = paper consolidation or a query-time-compositional-eval pivot.
```

## Details

### Hypothesis & design

After Phase 74 hardened the design (red-team: 4 confounds fixed; bilinear feasibility gate: 0.913
representational ceiling) and Phase 75-A/B built + smoke-validated the `ContentRouter`, this is the
pre-registered factorial ablation. content conditions use the bilinear router (relation-only routing,
top-K by `tail_key·head_key`, composability-shaping aux loss). Controls isolate composition from its
confounds: `content_sim` (tied keys = similarity, same params/STE), `content_random` (frozen random
keys = capacity/reach only, no learning), structural `hops1/node_only/hops2` (paired from Phase 73,
same seeds/testbed). Matched K=32; param counts reported; eval uses one cached final adjacency
(asserted to invoke the content builder — composable_frac differs from structural).

### Key Observations

1. **Composition fails every pre-registered clause.** content_comp does not beat hops1 (−0.0015),
   content_sim (−0.0063), or content_random (+0.0037) on 3p; none clears +0.010; per-seed signs are
   mixed. The asymmetric composability mechanism adds nothing over symmetric similarity.
2. **composable_frac is unpredictive of performance.** content_sim (0.528) ≈ content_comp (0.185) ≈
   content_random (0.150) in MRR despite a 3.5× spread in measured composability. The mechanism
   provably learns composability (Phase 75 gate: composable_frac rises past 0.216), but learning it
   does not help the task.
3. **The LP gain is capacity, not content.** All three content conditions beat structural on LP by
   +0.032–0.040 — including content_random (frozen RANDOM keys). So the win is "each edge gets K
   global neighbors + extra router params," a generic message-passing-receptive-field/capacity effect,
   not composition and not even learned content.
4. **Inverted depth signature.** content_comp − hops1: 1p +0.0044, 2p −0.0011, 3p −0.0015, 4p −0.0116,
   5p −0.0229. The gap *shrinks then goes negative with depth* — the exact inverse of the composition
   prediction (gap should grow with depth). Content routing helps shallow LP and *hurts* the deepest
   hop, because it replaces the local structural neighbors the soft-traversal relies on at 5p.
5. **Diligence worked.** Two real bugs (NaN aux, then divergence) were caught before the run; the
   factorial controls + pre-registration delivered a fully-attributed answer rather than an ambiguous
   one. The red-team's predicted realistic outcome (content_comp ≈ structural; gain = capacity) is
   exactly what occurred.

### Classification: REJECTED

The composition thesis — edges as learned composition of structure improving multi-hop reasoning — is
not supported. It is refuted on all pre-registered clauses, with composability shown unpredictive and
the apparent benefit attributed to capacity. RR_prior was not run: it was to rule out "content_comp is
just a relation-pair prior," but there is no content-specific win to explain (random ties it), so the
claim is dead more cheaply.

### Impact

- **Paper:** the content/composition direction is closed. Combined with Phases 66–73 (structural 2-hop
  rejected; edge attention's only robust gain is the structural-LOCAL +0.02 at 5p), the honest,
  defensible story is now firm: edge-to-edge attention's value is **structural and local**, and
  "relational composition" — whether structural (2-hop) or learned (content routing) — is not the
  engine. Depth-monotonicity (2p→3p) remains the durable property.
- **Mechanistic:** locality matters for deep hops under this eval. Content routing that adds global
  reach helps 1p (capacity) but degrades 5p (lost locality) — a clean structure↔content tradeoff.
- **Methodological (the one live thread):** the eval is soft-traversal over static node embeddings, so
  it cannot reward query-time edge-to-edge composition. Any future attempt to resurrect composition
  must first build a query-time compositional eval; without it, the embedding bottleneck caps any
  composition signal at "better embeddings → better 1p," which is not the claim.
- **Infra delivered & retained:** ContentRouter + bilinear content adjacency + the attention route_prob
  hook are tested and committed (default-off; baselines unaffected) — reusable if a query-time eval is
  built.

### Next Steps (Phase 76)

1. Consolidate the paper around the robust findings (depth-monotonicity; structural-local 5p edge
   benefit), with content routing as a documented negative that strengthens the calibration.
2. If composition is to be pursued at all: build a query-time compositional eval (edge-to-edge chaining
   at inference), the only setting that could give the thesis a fair test — otherwise pivot levers
   (e.g., the BetaE standard-benchmark validation).
```
