# Phase 73 — High-Power Edge-Sampled Hop Ablation (AUTO-GENERATED snapshot)

> Auto-written from `phase73_output.json` on run completion. COMPLETE. A polished `docs/phase_73.md` + `research_state.json` update follow in an interactive session.

## Config
- Edge-sampled FB15k-237: **11563 entities**, 27211 train edges, **18159 test triples**, mean degree **4.71** (sample_frac=0.1)
- hops=1 pairs=2,005,306; hops=2 fair-capped (K=128) pairs=3,064,974
- seeds=[7, 42, 99, 123, 456], conditions=['node_only', 'hops1', 'hops2']
- multi-hop query counts: 1p=18159, 2p=10000, 3p=10000, 4p=10000, 5p=10000

## Results (mean±std across seeds)

| condition | n | LP | 1p | 2p | 3p | 4p | 5p |
|---|---|---|---|---|---|---|---|
| node_only | 5 | 0.071±0.006 | 0.106±0.011 | 0.089±0.007 | 0.104±0.012 | 0.136±0.024 | 0.150±0.029 |
| hops1 | 5 | 0.068±0.002 | 0.100±0.001 | 0.085±0.009 | 0.103±0.012 | 0.132±0.020 | 0.167±0.030 |
| hops2 | 5 | 0.067±0.002 | 0.097±0.004 | 0.086±0.007 | 0.102±0.013 | 0.139±0.021 | 0.170±0.028 |

## Gap analysis (per-seed is what matters — see Phases 71/72)

### hops=2 − hops=1  (the 2-hop claim)

| metric | mean gap | per-seed | >0 |
|---|---|---|---|
| 2p | +0.0007 | -0.005, +0.008, +0.006, +0.002, -0.007 | 3/5 |
| 3p | -0.0013 | -0.015, +0.013, +0.012, -0.006, -0.011 | 2/5 |
| 4p | +0.0069 | -0.011, +0.046, +0.028, -0.010, -0.018 | 2/5 |
| 5p | +0.0030 | -0.010, +0.026, +0.024, -0.001, -0.023 | 2/5 |

### hops=1 − node_only  (does edge attention help at all?)

| metric | mean gap | per-seed | >0 |
|---|---|---|---|
| 2p | -0.0040 | -0.009, -0.006, -0.006, -0.010, +0.011 | 1/5 |
| 3p | -0.0006 | -0.006, -0.011, -0.002, -0.001, +0.017 | 1/5 |
| 4p | -0.0038 | -0.029, -0.020, -0.008, +0.006, +0.032 | 2/5 |
| 5p | +0.0171 **>0.010** | -0.008, +0.005, +0.019, +0.006, +0.063 | 4/5 |

### hops=2 − node_only

| metric | mean gap | per-seed | >0 |
|---|---|---|---|
| 2p | -0.0032 | -0.014, +0.002, +0.000, -0.008, +0.004 | 3/5 |
| 3p | -0.0019 | -0.021, +0.003, +0.010, -0.007, +0.006 | 3/5 |
| 4p | +0.0031 | -0.040, +0.026, +0.020, -0.004, +0.013 | 3/5 |
| 5p | +0.0201 **>0.010** | -0.018, +0.031, +0.043, +0.005, +0.040 | 4/5 |

## Heuristic verdict (auto)

- **2-hop (hops2 vs hops1), 3p:** mean -0.0013, positive in 2/5 seeds → NOT robust.
- **edge attention (hops1 vs node_only), 3p:** mean -0.0006 → inconclusive/none.

_Raw per-(condition,seed) data is in the committed `phase73_output.json`._
