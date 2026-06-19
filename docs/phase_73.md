# Phase 73 — High-Power Edge-Sampled Hop Ablation (1p–5p)

## Result

```
Phase: 73 — High-Power Hop Ablation (edge-sampled FB15k-237, DELTA-Matched, 5 seeds × 600 epochs)
Hypothesis: On a sparse, high-statistical-power graph (Phase 67's degree ~4.7 regime, ~18K test
  triples), (a) hops=1 beats node_only on 3p by >0.010, and (b) hops=2 (fair-capped) beats hops=1
  on 3p by >0.010 — the definitive local reproduction of Phase 67's +0.017.
Expected: both gaps >0.010, robust across 5 seeds.
Seeds: [7, 42, 99, 123, 456]
Result: REJECTED — 2-hop adds nothing; 1-hop edge attention helps ONLY at the deepest hop (5p).

Metrics (multi-hop MRR — edge-sampled FB15k-237, 11,563 entities, mean degree 4.71, 27,211 train
         edges, 18,159 test triples; 10k queries/type for 2p-5p; 5 seeds × 600 epochs, no early stop;
         hops=2 FAIR per-target-edge cap K=128: 17.7M -> 3.06M pairs, every edge keeps full 1-hop + sampled 2-hop):
Condition    adj_pairs    LP MRR        2p           3p           4p           5p
node_only        0        0.071±0.006  0.089±0.007  0.104±0.012  0.136±0.024  0.150±0.029
hops=1       2,005,306    0.068±0.002  0.085±0.009  0.103±0.012  0.132±0.020  0.167±0.030
hops=2       3,064,974    0.067±0.002  0.086±0.007  0.102±0.013  0.139±0.021  0.170±0.028

hops=2 − hops=1 (the 2-hop claim):
  2p +0.0007 (3/5)   3p −0.0013 (2/5)   4p +0.0069 (2/5)   5p +0.0030 (2/5)   — all within noise, REJECTED
hops=1 − node_only (does edge attention help at all?):
  2p −0.0040 (1/5)   3p −0.0006 (1/5)   4p −0.0038 (2/5)   5p +0.0171 (4/5) >0.010  — ONLY at 5p
hops=2 − node_only:
  5p +0.0201 (4/5) >0.010   (also only at 5p; 2p/3p/4p ~0)

Reference: Phase 66 (dense deg~20) hops2−hops1 3p=+0.002 REJECTED; Phase 71 (deg 7.3) +0.019 single-seed;
  Phase 72 (deg 5.1, 5 seeds) 3p=−0.009 REJECTED + hops1−node_only 3p=+0.026 (small-graph, n=184 test);
  Phase 67 (deg 4.1, H200, uncommitted) hops2−hops1 3p=+0.017 CONFIRMED — does NOT reproduce here.

Key insight: At full statistical power in Phase 67's own density regime, the 2-hop edge-adjacency
  mechanism provides NO measurable benefit over 1-hop at any depth — the paper's central
  architectural claim does not hold locally. Edge-to-edge attention's only robust benefit is at the
  DEEPEST hop (5p: +0.017 vs node_only, 4/5 seeds); it is absent at 2p/3p/4p. Phase 72's +0.026 on
  3p was small-test-set noise (here 3p is dead-even: 0.104/0.103/0.102). 2p→3p monotonicity holds
  (+0.015..+0.018, Phase 68 reproduced a 4th time).
Next question: Is the 5p-only edge-attention benefit a genuine deep-composition effect worth a
  narrower paper claim, or eval-temperature/traversal noise at the deepest hop? And given 2-hop is
  now rejected across degree 4.7/5/7/20, should the architecture/paper drop the 2-hop framing?
Status: LOGGED as REJECTED (2-hop, definitive at high power). Reframes the contribution. next_phase=74.
```

## Details

### Hypothesis

Phases 66/71/72 failed to show a robust 2-hop benefit, but each had a confound: dense subgraph
(66), single-seed (71), or tiny test set / small-graph noise (72). Phase 73 removes all three by
**edge-sampling** the full graph: `sample_mode='edge', sample_frac=0.10` → 11,563 entities, mean
degree 4.71 (Phase 67's regime), and **18,159 test triples / 10k queries per type** — the
statistical power Phase 54 showed is required for reliable multi-hop conclusions. hops=2 uses a
**fair per-target-edge cap** (`cap_hops2_fair`, K=128): every edge keeps its full 1-hop
neighborhood plus a per-edge random sample of 2-hop neighbors — the principled fix for Phase 66's
*global* random 5.3% cap (which starved many edges).

### Configuration

- Dataset: edge-sampled FB15k-237 (frac=0.10): 11,563 entities, 27,211 train / 18,159 test, deg 4.71
- Model: DELTA-Matched (d_node=48, d_edge=24, 2 layers, 4 heads); lr=0.003, batch=4096, label smoothing 0.1
- Epochs: 600, eval_every=25, **patience=0 (no early stop)**; Seeds: [7, 42, 99, 123, 456]
- hops=1: 2,005,306 pairs (uncapped); hops=2: 3,064,974 pairs (fair cap K=128 from 17.7M natural)
- Queries (leak-free, Phase 44 generator): 1p=18,159, 2p–5p=10,000 each (audit PASSED)
- Hardware: RTX 3080 Ti 12.9GB. Per-seed: node_only 145s, hops=2 2,597s, hops=1 4,715s
  (hops=1 ran during heavy concurrent session activity → inflated; hops=2 ran clean overnight)
- Enabled by the ~90× vectorized eval (Phase 73 infra commit) — without it this run was ~13h of
  eval alone; with it, ~3h total.

### Key Observations

1. **2-hop is definitively rejected.** hops=2 − hops=1 is within noise at every depth (max |gap|
   +0.007 at 4p, positive in only 2/5 seeds). With a fair cap, full power, and Phase 67's density,
   the second hop adds nothing. This completes the picture across degree 4.7 / 5.1 / 7.3 / 20 — the
   2-hop benefit does not reproduce locally anywhere.
2. **Edge attention helps only at 5p.** hops=1 − node_only: 2p −0.004, 3p −0.001, 4p −0.004 (all
   ~0/negative), but 5p **+0.017 (4/5 seeds)**. hops=2 − node_only 5p +0.020 (4/5). The benefit is
   real but confined to the deepest reasoning hop.
3. **Phase 72's 3p benefit was noise.** At n=184 test (Phase 72) hops1−node_only 3p was +0.026; at
   n=18,159 here it is −0.001. Small-test-set variance, not a real 3p effect.
4. **2p→3p monotonicity is the durable property** (+0.015/+0.018/+0.016) — robust in Phases 42/68/71/
   72/73. The most reliable thing DELTA does.
5. **All conditions are tied on LP and 2p/3p/4p** (LP 0.067–0.071) — edge attention is competitive,
   neither clearly helping nor hurting except at 5p.

### Classification: REJECTED (2-hop), PARTIAL-POSITIVE (5p-only edge benefit)

- hops=2 vs hops=1: no depth clears 0.010 robustly → REJECTED, definitively, at high power.
- hops=1/hops=2 vs node_only: clears 0.010 only at 5p (4/5 seeds) → a narrow, deep-hop benefit.

### Impact

- **Paper (important):** the "2-hop edge adjacency enables relational composition" claim is not
  supported at high power in its own density regime. It should be **dropped or heavily qualified**.
  The defensible claims are now: (i) 2p→3p depth-monotonicity (robust, 5 phases); (ii) a small
  deep-hop (5p) edge-attention benefit in the sparse regime (+0.02, 4/5 seeds). Phase 67's +0.017
  (cited, uncommitted, single-source) is contradicted by this higher-powered reproduction and should
  not stand as the headline.
- **Methodology:** the session's pattern held to the end — each increase in rigor (uncap → converge →
  more seeds → high-power test set) shrank the effect. The infra (90× eval, edge-sampler, fair cap)
  makes this caliber of test cheap to repeat.

### Next Steps (Phase 74)

1. Decide the paper reframe: lead with depth-monotonicity + the 5p deep-hop benefit; retire the
   2-hop-composition headline. (Analysis/writing, not compute.)
2. Probe the 5p-only effect: is it genuine deep composition or eval-traversal-temperature artifact?
   (Cheap: vary soft-traversal τ on the trained checkpoints — no retraining.)
3. Optionally: a different lever entirely (content-dependent/anchor-routed edges, or the dead-gate
   repurpose) now that structural 2-hop is exhausted.
```
