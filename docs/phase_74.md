# Phase 74 — Content-Dependent Edge Adjacency: Design, Red-Team & Feasibility Gate

## Result

```
Phase: 74 — Composability-Routed Content Edge Adjacency (design + adversarial red-team + cheap feasibility gates, pre-build)
Hypothesis (to test in 74+): edges routed by LEARNED COMPOSABILITY (not feature similarity, not
  structural proximity) give better multi-hop reasoning — "DELTA edges as the composition of structure".
Method: offline probes only (no LP training yet). Quantify dilution; red-team the design; gate feasibility.
Result: PROCEED TO BUILD — with two design corrections forced by the gates.

Findings:
1. DILUTION QUANTIFIED (phase74): structural 2-hop edge-pairs are only 21.6% composable-relation-type
   (vs 19.6% 1-hop, 15.5% random) — ~78% of 2-hop neighbors are non-composable noise. Mechanistic
   explanation of why 2-hop never helped (Phases 66/71/72/73).
2. HARD-BUCKET ROUTING FAILS (phase74b): best supervised relation-only bucket routing (tail_bucket==
   head_bucket) reaches only composable_frac=0.243 at M=256 (~struct-2hop). Single-bucket assignment
   cannot express many-to-many composability.
3. BILINEAR ROUTING SEPARATES PERFECTLY (phase74c): a rank-16 bilinear affinity tail_key·head_key
   reaches composable_frac=0.913 = the ORACLE ceiling. FB15k-237 composability IS low-rank-separable;
   the mechanism must select by continuous bilinear affinity (top-K), NOT argmax bucket equality.
4. THE STING: the 0.913 ceiling is a purely RELATION-LEVEL computation. Composability lives at the
   relation-pair level, so the edge-attention apparatus risks reducing to an [R,R] relation-prior
   lookup. The decisive control for the build is therefore an [R,R] composability-bias baseline.

Key insight: composability is representationally viable (rank-16 separable) but only via bilinear
  affinity, and it may be a relation-TYPE effect (cheap [R,R] prior) rather than an instance-level
  edge-attention effect — which is exactly what the build must disentangle.
Status: LOGGED. Design hardened + pre-registered (below). next_phase builds the bilinear ContentRouter.
```

## Adversarial red-team — 4 fatal confounds found (all fixed in the build)

A 4-dimension red-team (mechanism / learnability / experiment / implementation) found four
`kills_design` confounds; the build incorporates every fix:

1. **Circularity** — node-feat in the routing key lets the router relearn "edges sharing a node" (=
   structural 1-hop), so a "win" is structure rediscovering structure. → FIX: relation-only routing
   variant as the clean test; report **shared-node-overlap** as the primary attribution diagnostic
   (74b/74c confirm it stays low, ~0.018 — not secretly 1-hop).
2. **Control collapse** — symmetric anchors (Aₕ=Aₜ) make content_comp ≡ content_sim, so attribution
   is impossible. → FIX: content_sim is the literal tied-key special case; report Aₕ/Aₜ asymmetry.
3. **Gradient dead-end** — "EdgeAttention unchanged" is self-contradictory (integer adjacency indices
   carry no gradient → anchors never train); and the LP gradient isn't composability-shaped. → FIX:
   edit attention.py to consume an aligned `route_prob` (STE on the scalar, not the index); add an
   explicit self-supervised middle-consistency aux loss; warm-init keys.
4. **Silent eval bug + attribution** — the `_edge_adj_cache` short-circuit could make eval run
   structural adjacency while only training used content; and content_comp differs from hops1 on 5
   axes. → FIX: assert eval invokes the content builder; factorial ladder with param-matched and
   random-routing controls; match REALIZED neighbor count, not nominal K.

## Pre-registered experiment (locked BEFORE the build)

Conditions (matched realized neighbors-per-target, shared seeds/splits, param counts reported):
`node_only`, `hops1`, `param_matched_hops1`, `content_sim` (tied-key), `content_random` (frozen keys),
`content_comp` (bilinear), `RR_prior` ([R,R] composability bias, no edge attention), `hybrid`.

PASS (claim "composition helps") requires ALL, paired per-seed over 5 seeds, on the edge-sampled
sparse testbed (deg 4.7, 18K test):
1. content_comp − hops1 > 0.010 on 3p (all-positive or Wilcoxon p<0.05);
2. content_comp − content_sim > 0.010 on 3p;
3. content_comp − content_random > 0.010 on 3p;
4. **content_comp − RR_prior > 0.010 on 3p** (instance-level beats relation-type prior — the core claim);
5. 3p gap ≥ 0.5 × 5p gap (excludes a 5p-only-ceiling masquerade);
6. learned adjacency composable_frac RISES during training past struct-2hop's 0.216 (mechanism gate),
   bucket/affinity usage stays non-collapsed, adjacency Jaccard plateaus >0.9 (convergence gate).
ANY clause failing ⇒ REFUTE (and the result is interpretable, not a tuning artifact).

## Feasibility gate verdict

`phase74.py` (dilution), `phase74b.py` (bucket ceiling 0.243 — refuted), `phase74c.py` (bilinear
ceiling 0.913 — proceed). The supervised ceiling proves the signal is *representable*; the build's
open empirical questions are whether LP+aux training reaches it, whether it converts to multi-hop MRR,
and whether it beats the RR_prior. Building is justified; a null remains a real possibility and is
pre-registered as interpretable.

## Next Steps (build)

1. Vectorized `build_content_edge_adjacency` (bilinear affinity, cap-before-materialize, O(E·M+E·K)).
2. Standalone `ContentRouter` (head/tail key projections + low-rank affinity; STE on route_prob;
   middle-consistency + weak anti-collapse aux losses; tau anneal). Reuse only PostAttentionPruner's
   loss-wiring scaffolding, not its post-attention compute.
3. Minimal `attention.py` edit: optional `route_prob` multiplies v_src (gradient path to keys).
4. Harness: the 8-condition factorial ladder incl. RR_prior; adjacency rebuilt per-epoch, route_prob
   per-batch; one fixed cached eval adjacency (assert eval uses it).
5. Smoke (sample_frac 0.03) → confirm keys get gradient, composable_frac rises, eval invokes content
   builder; then the 5-seed run against the pre-registered pass/fail.
```
