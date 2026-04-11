# Key Findings

38 key findings from 57 experiment phases, organized by research stage. See [Validation Phases](validation-phases.md) for complete result tables.

---

## Core Architecture (Phases 1-15)

### 1. Edge attention is DELTA's strongest signal
Edge-to-edge attention perfectly solves relational classification (100%) where node attention collapses to 26.7%. Edges deserve first-class attention.

### 2. Sub-linear scaling
DELTA scales at O(n^0.81) from 20 to 400 nodes thanks to the importance router's sparse attention, maintaining 100% accuracy.

### 3. Gumbel-softmax routing enables differentiable selection
The straight-through estimator lets all 12 router parameters receive gradients (vs 0 with hard top-k). Temperature annealing provides a curriculum from exploration to exploitation.

### 4. Multi-hop edge adjacency solves compositional reasoning
2-hop edge adjacency achieves **100% on derived/transitive relations** — a +38.9% jump from 1-hop (61.1%) and beating Node GNN (83.3%). The biggest architectural improvement in the project.

### 5. Compositional logic rules are DELTA's sweet spot
Phase 13: DELTA hit 100% on all 7 relation types including derived, while Node GNN achieved only 87.5% on derived.

### 6. Edge embeddings are inherently discriminative
All training methods (classification, contrastive, joint) achieve 100% nearest-neighbor retrieval. Edge attention creates well-clustered embeddings by default.

### 7. Router sparsity needs careful calibration
Aggressive sparsity (40-50%) hurts on relation classification. Works well at moderate sparsity (80%) but needs gentler schedules at higher compression.

### 8. DELTA matches Node GNN at scale, excels on derived relations
On the 500-triple benchmark, both reach 100% — but DELTA's advantage emerges specifically on compositional/transitive tasks.

---

## Architectural Fixes (Phases 16-21)

### 9. Sparse COO multi-hop is the clearest win
O(E^0.97) sub-quadratic scaling — 2500-edge 2-hop completes in 0.18s where the old dense approach timed out at ~500 edges.

### 10. BFS partitioning scales linearly
O(N^0.99) scaling from 50 to 2500 nodes (2ms to 91ms), replacing O(N^3) spectral clustering. Balance ratio of 0.79.

### 11. Variational memory compression preserves accuracy
Identical downstream accuracy (100%) with KL converging (0.126 to 0.026). The learned similarity threshold adapts under gradient.

### 12. Per-layer edge projections increase type diversity
Higher edge-type entropy (1.632 vs 1.562) while matching classification accuracy.

### 13. Soft gating solves post-attention pruning
**100% accuracy at 50% target sparsity**, beating pre-attention routing by +14.7%. The original 29% gap was caused by non-differentiable hard top-k, not a fundamental paradigm problem.

### 14. Learned dropout needs harder benchmarks
All dropout modes reach 0 generalization gap on the current 100-entity KG. The dataset is too easy to differentiate.

---

## Scale & Integration (Phases 22-30)

### 15. Soft gating holds at 10x scale
N=1000 with 15% label noise: soft gating 100%, old router 81.6% (+18.4% gap). Zero generalization gap.

### 16. DELTA matches CompGCN, crushes embedding baselines
Phase 23: DELTA and CompGCN both 100% on FB15k-237-like benchmark. TransE 67.6%, RotatE 70.7%.

### 17. All fixes integrate cleanly at scale
Phase 24: full pipeline on N=1000 noisy benchmark. No fix caused degradation — all ablations matched full DELTA at 100%.

### 18. DELTA+Gate leads on real FB15k-237
Phase 25: 97.6% on actual Freebase triples (2000-entity dense subgraph), narrowly beating CompGCN (97.2%).

### 19. Graph structure genuinely helps — the constructor is the bottleneck
Phase 27b: FixedChain DELTA (40.7%) beat Transformer (36.3%), but the attention-thresholded constructor (34.3%) discards sequential adjacency edges needed for path composition.

### 20. Dual attention's advantage is noise robustness
Phase 28: at 80% feature corruption, Dual Attention 64.2% vs Vanilla EdgeAttention 40.2% (+24%). Node context is the key differentiator at extreme noise. Soft gating adds only +/-0.6% beyond dual attention.

### 21. All results are statistically stable across 5 seeds
Phase 29: DELTA+Gate 97.4% +/- 0.1% on FB15k-237. Soft Gate 100.0% +/- 0.0% vs Old Router 79.7% +/- 1.1%.

---

## Compositional Reasoning (Phases 42-45)

### 22. DELTA-Matched dominates multi-hop compositional reasoning
Phase 42: DELTA-Matched (158K params) is the **only model that improves from 2p to 3p** (MRR 0.733 to 0.738). At 1p it trails GraphGPS by -0.016; at 3p it leads by +0.041. Larger DELTA models (Full, SelfBootstrap) overfit to 1-hop statistics.

### 23. Multi-hop advantage holds across all regularization regimes
Phase 43: DELTA-Matched beats GraphGPS on 3p at **every single DropEdge rate** (0-40%). The advantage narrows at 20% but never flips.

### 24. DELTA's advantage accelerates with reasoning depth
Phase 44: the centerpiece result. DELTA's lead over GraphGPS: +0.004 (2p) -> +0.026 (3p) -> +0.066 (4p) -> **+0.100 (5p)**. DELTA's 5p MRR (0.790) exceeds its own 2p (0.758). No other model improves with depth. GraphGPS and DistMult degrade monotonically.

### 25. Multi-seed confirms statistical robustness
Phase 45: 3p MRR 0.742 +/- 0.009 vs GraphGPS 0.713 +/- 0.007. Std bars don't overlap. DELTA's worst seed (0.731) exceeds GraphGPS's best (0.722).

### 26. Inference cost is deployment-friendly
Phase 45: encoding is 51.8x slower (2-hop edge adjacency, once per graph), but per-query scoring is 0.8-0.9x GraphGPS — DELTA is *faster* per query. Training is 34x slower but one-time.

---

## Attention Temperature (Phases 46-48)

### 27. Learnable temperature reveals edge/node asymmetry
Phase 46: uniform init=4.0 reduces dead heads from 83% to 38%. Edge temps drift UP (4.0 to 4.5), node temps drift DOWN (4.0 to 3.6). L0 attention is always dead regardless of temperature.

### 28. Selective layer-specific sharpening outperforms uniform
Phase 47: B (L0=1, L1+L2=4) achieves best LP MRR (0.4783). Edge temps consistently drift UP (4.4-4.5), node temps drift DOWN (3.5-3.7).

### 29. Asymmetric node/edge temperature yields new LP record
Phase 48: E (node=2, edge=6) achieves **LP MRR 0.4856** (+1.5% over previous best). Node temps are "set and forget" (+/-0.01 drift); edge temps always drift UP. A persistent LP/3p trade-off remains: E leads LP but D (all=4.0) still leads 3p (0.4018 vs 0.3872).

### 30. LP/3p temperature trade-off is fundamental
Phase 49: L0=4.0 does NOT explain D's 3p advantage — H (L0=4,4 + E's asymmetric L1+L2) achieves new LP record (**0.4887**) but 3p only reaches 0.3930 (still below D's 0.4018). After 4 phases (46-49) testing 10+ temperature configurations, D's uniform temp=4.0 remains the only path to 3p≥0.40. Asymmetric init improves LP at the cost of 3p. The trade-off may require dynamic temperature (annealing) rather than static init.

### 31. Temperature annealing breaks the 3p ceiling
Phase 50: K (anneal node 4.0→2.0 fast over 50% of training) achieves **3p MRR 0.4148** — the FIRST configuration in 5 phases (46-50) to beat D's 0.4018 (+0.013). The mechanism: high node temp during early training builds compositional representations, then annealing toward lower temps approaches LP-optimal asymmetry. K's best checkpoint (ep 175) has node temp=2.6, not the anneal target (2.0). M (anneal 4→3) ties H's LP record (**0.4887**) but misses 3p. The Pareto frontier has shifted but LP+3p combined target not yet achieved in a single configuration.

### 32. Training trajectory matters — static init cannot replicate annealing's 3p advantage
Phase 51: N (static node=2.6, matching K's optimal checkpoint value) achieves 3p=0.4001 vs K (annealed to 2.6) 3p=**0.4148** — a +0.015 gap. Early training at high node temp creates 3p-supportive representations that static init cannot replicate. However, N achieves best-ever **4p=0.3426** and **5p=0.3788** (exceeding K's values), suggesting static low node temp uniquely benefits deeper reasoning. P (anneal 4→2.5) achieves a **new LP MRR record (0.4890)** and H@10 record (0.8014) but 3p=0.3823 misses target. K remains closest to the combined target with LP gap of only −0.004.

### 33. Edge sharpness boosts LP but hurts 3p — temperature investigation closed
Phase 52: Q (K's anneal + edge init 7.0) achieves **new LP MRR record (0.4905)** and S achieves **new H@10 record (0.8045)**, but both fail 3p (Q: 0.3927, S: 0.3789). Edge init 7.0 boosts LP by +0.009 but damages 3p by −0.022 — the trade-off extends to the edge temperature axis. After **7 phases (46-52)** testing 20+ configurations, the LP/3p trade-off is confirmed fundamental at the temperature level. Three distinct operating modes emerge: **LP-optimized** (P/Q, LP≥0.4890), **Balanced-3p** (K, 3p=0.4148), and **Deep-reasoning** (N, 4p=0.3426, 5p=0.3788). Temperature controls reasoning depth — a paper-ready contribution about how attention temperature tunes the LP/reasoning depth trade-off in graph neural networks.

### 34. CRITICAL: Multi-hop temperature claims are NOT statistically robust
Phase 53: Multi-seed validation (3 seeds: 42, 123, 456) reveals that **K's 3p advantage and N's deep-hop advantage are seed-dependent artifacts.** K mean 3p=0.3699±0.0200 — **BELOW baseline A (0.3725)**. N mean 4p=0.2354±0.0618, 5p=0.2665±0.0738 — HUGE variance. Even seed=42 re-runs differ due to CUDA non-determinism (K 3p: 0.4148→0.3812). **LP MRR IS robust**: K=0.4832±0.0052, N=0.4842±0.0089. The 500-query multi-hop evaluation is too noisy for single-seed conclusions. All Phases 46-52 claims about 3p/4p/5p advantages from temperature tuning must be treated as unreliable. Only LP improvements are statistically supported.

### 35. Evaluation noise dominates multi-hop variance — investigation CLOSED
Phase 54: 10k-query evaluation reduces cross-seed multi-hop std by **66-84%** (avg K=75%, N=79%), confirming evaluation noise was the dominant variance source. With tight CIs, K and N are **statistically indistinguishable** on multi-hop (3p: 0.2614±0.0032 vs 0.2560±0.0091). 500q and 10kq give dramatically different absolute MRR (K 3p: 0.37→0.26) because larger query samples include harder paths. LP MRR remains robust and protocol-independent (K=0.4845±0.0051, N=0.4865±0.0086). After **9 phases (46-54)**, the conclusion: temperature reliably improves LP but has **no statistically supported effect on multi-hop reasoning depth** in DELTA-Full. Multi-hop investigation CLOSED.

---

## Brain Architecture (Phases 55–58)

### 36. BrainEncoder validates differentiable graph construction for link prediction
Phase 55: BrainEncoder (Gumbel-sigmoid differentiable edge selection) achieves LP MRR 0.4773 on FB15k-237 — within 0.002 of the 0.475 threshold — while delivering **+3.7% H@10** over delta_full (0.7973 vs 0.7603). First successful integration of differentiable graph construction with DELTA. Constructed edges improve recall at the cost of precision.

### 37. Constructor density is a critical hyperparameter — fewer edges strictly dominate
Phase 56: Halving density from 0.02 to 0.01 gains **+0.012 MRR** and **+0.058 H@10** with half the edges (2,435 vs 4,870). Density=0.01 matches delta_full MRR (0.4794 vs 0.4796) while adding +4.7% H@10. The constructor's Gumbel-sigmoid selection is more discriminating at lower density, producing fewer but higher-quality edges.

### 38. Temperature annealing is counterproductive on brain_hybrid — baseline dominates
Phase 57: brain_hybrid exceeds 0.480 MRR (A: 0.4808, B: 0.4818), but the improvement comes from reducing epochs (200 vs 300), not from annealing. K-style annealing adds only +0.001 MRR while **dropping H@10 by -0.046** (0.7613 vs 0.8076). Constructed edges already provide structural information that overlaps with what temperature sharpening achieves, making annealing redundant for MRR and harmful for recall. The optimal brain_hybrid configuration is baseline temp=1.0 at 200 epochs.

### 39. d=0.01 is the optimal density sweet spot — multi-seed validated, sparser fails
Phase 58: Multi-seed validation (3 seeds: 42, 123, 456) across two densities confirms **d=0.01 is statistically robust**: mean MRR=0.4844±0.0097, H@10=0.7994±0.0058. All 3 seeds maintain H@10>0.79; seed=456 achieves a new brain_hybrid **MRR record of 0.4956**. d=0.005 (1,217 edges) **fails to continue the improvement trend**: mean MRR=0.4673 (−0.017 vs d=0.01), and H@10 variance increases 4.4× (std 0.026 vs 0.006). The density curve from d=0.02→d=0.01→d=0.005 shows clear optimum at d=0.01 — halving density further loses recall without compensating precision gains. Brain density optimization CLOSED.

---

*See [Validation Phases](validation-phases.md) for complete result tables. See [Status & Roadmap](status-and-roadmap.md) for current priorities.*
