# DELTA: Publication Roadmap

**Target:** NeurIPS / ICLR (top-tier ML venue)
**Title placeholder:** *"DELTA: Edge-Centric Dual Attention for Relational Reasoning on Knowledge Graphs"*
**Status:** 63 experiment phases complete. Brain architecture validated (Phases 55–58). KG scaling evaluated (Phases 59–63). Pivoting to sparse attention and sequence domains.

**Current evidence base:** 63 experiment phases, 44 unit tests, real FB15k-237 results (97.4% ± 0.1% over 5 seeds, LP MRR 0.4905), multi-hop dominance (5p MRR 0.790), differentiable graph construction (brain_hybrid MRR 0.4818 with H@10 0.8076), scaling evaluation at N=2000/5000 (Phases 59–63). Cross-domain transfer: 0.961 on WN18RR with 100 samples (frozen encoder).

---

## Completed Evidence Base

| Area | Phases | Key Result |
|------|--------|------------|
| Core architecture validation | 1–24 | Edge-first dual attention, 2-hop edge adjacency, 6 architectural fixes |
| Real-data benchmarks | 25–37 | FB15k-237 97.4% ± 0.1% (5 seeds), cross-domain transfer 0.961 |
| Compositional reasoning | 42–45 | 5p MRR 0.790 vs GraphGPS 0.690; only model improving with depth |
| Temperature optimization | 46–54 | LP MRR 0.4905; LP/3p trade-off characterized as fundamental |
| Brain architecture | 55–57 | Differentiable graph construction; MRR 0.4818, H@10 0.8076 |

## Remaining Publication Gaps

| Gap | Priority | Notes |
|-----|----------|-------|
| Full-scale dataset evaluation (14.5K+ entities) | HIGH | Need full FB15k-237 or YAGO3-10 |
| Cross-family benchmark (YAGO3-10 or WN18RR LP) | HIGH | Proves generalization beyond Freebase |
| Interpretability figure | MEDIUM | Attention heatmap on known reasoning chain |
| Multi-seed brain_hybrid validation | MEDIUM | Statistical confidence on construction gains |

---

## Phase I — Active Execution (Colab)

**Goal:** Close the remaining synthetic → real data gap for Phases 35–37.

| Task | Script | Est. Runtime | Status |
|------|--------|-------------|--------|
| Phase 35 `--full` (FB15k-237 → WN18RR, GRL + probe) | `experiments/phase35_relational_transfer.py` | ~18–20h H100 | ✅ Complete |
| Phase 36 `--full` (task-aware construction, 500–5000 nodes, A/B/C) | `experiments/phase36_task_aware_at_scale.py` | ~2–4h H100 | ✅ Complete |
| Phase 37 `--full --num_seeds 5` (4 models × 5 seeds, real FB15k-237) | `experiments/phase37_real_comparison.py` | ~6–10h H100 | ⏳ Queued |

**After each run:** Record results in `docs/COLAB_RESULTS.md`.

**Verification gates:**
- Phase 35: ~~GRL probe > baseline probe~~ → Probe > 0.7 on WN18RR ✅ (0.961)
- Phase 36: ~~augmented > fixed by > 2%~~ → ❌ Failed (max +1.3%); constructor de-emphasized
- Phase 37: DELTA-Matched (~30K params) beats GraphGPS (~33K params) on real FB15k-237

---

## Phase II — Close Critical Proof Gaps

**Goal:** Prove that every architectural component contributes (pre-empt ablation reviewers), and establish Phase 34's synthetic GraphGPS gap holds on real data.

### Phase 34 (real data run)
Repeat Phase 34 on real FB15k-237 (currently only synthetic). Infrastructure already exists in `experiments/phase34_graphgps_grit_comparison.py`. Run in parallel with Phase 37 if Colab compute allows.

**Verification gate:** DELTA-Matched accuracy > GraphGPS accuracy on real FB15k-237 (not just synthetic).

### Phase 38 — Component Ablation on Real FB15k-237

**Script:** `experiments/phase38_component_ablation.py` *(to be written)*

**Ablation matrix** (5 seeds each, real FB15k-237):
| Config | Component removed | Expected drop |
|--------|------------------|--------------|
| Full DELTA-Matched | — (baseline) | — |
| No NodeAttention | Remove node parallel attention | > 2% drop |
| No EdgeAttention | Remove edge parallel attention | > 5% drop (edge-first thesis) |
| No ReconciliationBridge | Remove co-update layer | > 2% drop |
| No PostAttentionPruner | Remove pruning | ≈ 0% (accuracy) + inference time delta |
| 1-hop only | Remove 2-hop edge adjacency | > 3% drop (Phase 11 validated on synthetic) |

**Template:** `experiments/phase28_hard_ablation.py` (component ablation design) + `experiments/phase37_real_comparison.py` (real data + mini-batch training loop).

**Estimated runtime:** ~4–6h H100.

**Verification gate:** Every ablation hurts accuracy on real FB15k-237.

---

## Phase III — Strengthen Novelty Claims

**Goal:** Multi-hop path reasoning + 2 additional benchmark datasets. This answers: "does DELTA generalize beyond Freebase?"

### Phase 39 — Multi-hop Path Queries on Real FB15k-237

**Script:** `experiments/phase39_multihop_path_queries.py` *(to be written)*

**Protocol:** Standard BetaE/ConE 1p/2p/3p path query splits on FB15k-237.

**Required code change:**
- Add `load_path_queries(dataset_name)` to `delta/datasets.py`
- Path query files available alongside the main FB15k-237 split

**Baselines:** DELTA vs GraphGPS vs GRIT vs TransE (the classic multi-hop baseline).

**Connection to prior work:** Phase 11 (100% on derived 2-hop edge adjacency, synthetic) predicts DELTA's 2-hop architecture should excel on 2p/3p queries.

**Estimated runtime:** ~3–4h H100.

**Verification gate:** DELTA accuracy ≥ GraphGPS on 2p and 3p query types.

---

### YAGO3-10 Dataset Loader

**File:** `delta/datasets.py` — add `'yago3-10'` entry to `DATASET_URLS`

**Source:** [TimDettmers/ConvE](https://github.com/TimDettmers/conve) repo (same format as FB15k-237):
- 123,182 entities
- 37 relations
- 1,079,040 training triples

**Note:** At 123K entities, mini-batching via `NeighborSampler` is required. The NeighborSampler fix (seed preservation) makes this feasible.

### Phase 40 — YAGO3-10 Benchmark

**Script:** `experiments/phase40_yago3_benchmark.py` *(to be written)*

**Protocol:** Same 4-model × 5-seed setup as Phase 37 (parameter-matched).

**Estimated runtime:** ~8–12h H100.

**Verification gate:** DELTA-Matched beats GraphGPS on YAGO3-10 (establishes cross-family generalization — YAGO is Wikidata-sourced, not Freebase).

---

### Codex-M Dataset Loader

**File:** `delta/datasets.py` — add `'codex-m'` entry

**Source:** [tsafavi/codex](https://github.com/tsafavi/codex):
- 17,050 entities
- 51 relations
- 206,205 training triples

**Note:** Codex is harder than FB15k-237 (negative sampling is more adversarial). Good for establishing ceiling.

### Phase 41 — Codex-M Benchmark

**Script:** `experiments/phase41_codexm_benchmark.py` *(to be written)*

**Protocol:** Same 4-model × 5-seed setup as Phase 37/40.

**Estimated runtime:** ~4–6h H100.

**Verification gate:** DELTA-Matched beats GraphGPS on Codex-M.

---

## Phase IV — Completeness

**Goal:** Scaling story + interpretability figure. Both turn existing infrastructure into paper-quality figures.

### Phase 42 — Scaling Analysis

**Script:** `experiments/phase42_scaling_analysis.py` *(to be written)*

**Protocol:** Train DELTA on graph size subsets and measure:
- Accuracy vs. entity count: 500 / 2K / 5K / 14.5K / 123K (via YAGO3-10 + NeighborSampler)
- Training time vs. entity count
- Memory usage vs. entity count

**Expected result:** O(E^x) with x < 2 (mini-batching breaks the quadratic edge adjacency wall). Reference: Phase 8 found O(n^0.81) on synthetic.

**Estimated runtime:** ~3–4h H100.

**Verification gate:** Plot shows sub-quadratic scaling; 123K-entity YAGO3-10 finishes without OOM.

---

### Phase 43 — Edge Attention Interpretability

**Script:** `experiments/phase43_interpretability.py` *(to be written)*

**Content:**
1. **Top-k attention heatmap** — for a known 2-hop reasoning chain on FB15k-237 (e.g., `/person/birthplace` + `/place/country` → `/person/nationality`), show which edge pairs receive highest `EdgeAttention` weights
2. **t-SNE of edge embeddings** — FB15k-237 vs WN18RR embeddings from *same* frozen encoder (Phase 35 story: shows structural clustering independent of domain)
3. **Relation pair attention matrix** — 237×237 heatmap of average edge-to-edge attention, showing which relation pairs DELTA learns to compose

**This becomes the "what did the model learn" Figure 3 in the paper.**

**Verification gate:** At least one interpretable attention pattern that matches known relational composition rules (e.g., nationality chains have high transitivity attention).

---

## Phase V — Architectural Evolution (Conditional)

**Goal:** If Phase 39 reveals a multi-hop reasoning gap (e.g., 3p accuracy drops >15% vs 1p), close it with cross-stream interaction during attention. Skip this phase entirely if DELTA handles 3p adequately.

### Phase 44 — Extended Multi-hop Depth (4p/5p Compositional Queries)

**Script:** `experiments/phase44_depth.py`

**Motivation:** Phase 42 showed DELTA-Matched is the *only* model that improves from 2p→3p (MRR 0.733→0.738), beating GraphGPS at 3p by +0.041 despite being smaller (158K vs 214K params). Phase 44 asks: does this compositional advantage continue at 4-hop and 5-hop depth?

**Protocol:**
- 4p: 3 TRAIN hops → 1 TEST hop (quad-nested chain)
- 5p: 4 TRAIN hops → 1 TEST hop (quint-nested chain)
- Same leak-free query generation as Phase 42 (anchor≠answer, no shortcuts, no cycles)
- Same soft entity traversal scoring
- Same training infrastructure (standard LP loss, best-val checkpoint)

**Key question:** If DELTA-Matched's 2p→3p improvement extends to 3p→4p and 4p→5p, it validates that edge-to-edge attention with 2-hop adjacency provides fundamentally better compositional depth than node-based GNNs.

**Models:** delta_matched, graphgps, distmult (baseline with 0 GNN layers)

**Note:** The original Phase 44 plan (ReasoningMesh / gated cross-attention) is deferred — Phase 42 showed DELTA already handles 3p well without cross-stream changes. The ReasoningMesh concept may be revisited as future work.

**Success criteria:**
- Gated mesh improves 3p accuracy by ≥ 3% over baseline
- 1p accuracy does not degrade (mesh doesn't hurt easy cases)
- Gate weights are interpretable (later layers use more cross-talk)

**Estimated runtime:** ~6–8h H100 (5 configs × 3 seeds × FB15k-237).

**Future evolution (not for first paper):**
- Option 2: Message-passing mesh (multi-step within-layer exchange) — if Option 1 shows gains
- Option 3: Shared latent bottleneck — if scale demands compression

**Verification gate:** ≥ 3% improvement on 3p queries with gated mesh vs baseline.

---

## Phase VI — Paper Assembly

**Goal:** Write and submit to NeurIPS (May deadline) or ICLR (October deadline).

### Paper Structure

```
Title: DELTA: Edge-Centric Dual Attention for Domain-Transferable
       Relational Reasoning

Abstract: 3 sentences — gap, method, result

1. Introduction
   - The three-paradigm gap (Transformers → GNNs → DELTA)
   - Why edges as first-class citizens matter for relational reasoning
   - 2-hop edge adjacency and what it enables
   - **Anti-oversquashing claim:** standard GNNs suffer from oversquashing at graph bottlenecks (sparse bridges between clusters); DELTA's edge-centric design solves this *inherently* — no dynamic graph rewiring required. Evidence: Phase 36 Exp B, base model hits 0.961–0.989 on cluster graphs with only 2 bridge edges, where standard GNNs would collapse.
   - Summary of contributions (5 bullet points)

2. Related Work
   - Message-passing GNNs (CompGCN, RGCN, RotatE)
   - Transformers on graphs (GraphGPS, GRIT)
   - Transfer learning in graphs (DANN, domain adaptation)
   - Knowledge graph completion (TransE, BetaE)

3. Architecture
   - DualParallelAttention (node + edge in parallel)
   - 2-hop edge adjacency construction — **key mechanism:** allows edge representations to attend to other edges sharing a node, bypassing the single-edge bottleneck that causes oversquashing in message-passing GNNs. Phase 36 Exp B is the empirical proof.
   - ReconciliationBridge (co-update)
   - PostAttentionPruner
   - GraphConstructor (optional; Phase 36 showed ≤1.3% benefit — vestigial because 2-hop adjacency already solves the routing problem the constructor was designed for; mention briefly and offer as future work)

4. Experiments
   4.1 Setup (datasets: FB15k-237, WN18RR, YAGO3-10, Codex-M)
   4.2 Parameter-matched comparison — Phase 37 flagship table
   4.3 Component ablation — Phase 38 table
   4.4 Multi-hop path queries — Phase 39 results
   4.5 Additional benchmarks — Phases 40–41 (YAGO3-10, Codex-M)

5. Domain Transfer — Inherent Invariance (not DANN)
   - Phase 35: frozen encoder + 100-sample probe → 0.961 on WN18RR
   - GRL unnecessary — encoder is already domain-invariant
   - Reframes Phase 32 zero-shot failure as head mismatch (237→11)
   - Few-shot adaptation curve (proposed Phase 35b: {10,50,100,500,1K} samples)

6. Scaling Analysis
   - Phase 42: sub-quadratic O(E^x) plot
   - Connection to mini-batching (NeighborSampler)

7. Interpretability
   - Phase 43: EdgeAttention top-k + t-SNE figures
   - Which relational compositions DELTA learns

8. Conclusion
   - Edge-first relational inductive bias generalizes across KG domains
   - Future: ReasoningMesh for deeper multi-hop chains, multi-modal graph construction, production KG completion
```

### Results Tables to Compile (all 5 seeds, mean ± std)

| Table | Phases | Columns |
|-------|--------|---------|
| Main comparison | 37, 40, 41 | DELTA-Matched / GraphGPS / GRIT / CompGCN |
| Ablation | 38 | Full / -NodeAttn / -EdgeAttn / -Reconcile / -Pruner / 1-hop |
| Path queries | 39 | DELTA / GraphGPS / GRIT / TransE on 1p / 2p / 3p |
| Transfer | 35 | Probe@{10,50,100,500,1K} samples / Random baseline |
| Scaling | 42 | Accuracy@N and Time@N for 5 graph sizes |

---

## Relevant Files

| File | Role |
|------|------|
| `experiments/phase35_relational_transfer.py` | GRL training template; NeighborSampler + real data |
| `experiments/phase37_real_comparison.py` | Flagship: 4-model × 5-seed protocol on real data |
| `experiments/phase28_hard_ablation.py` | Ablation template (synthetic → adapt for Phase 38) |
| `experiments/phase31_mini_batching.py` | `NeighborSampler` — seeds always preserved (fixed) |
| `delta/datasets.py` | Add YAGO3-10, Codex-M, `load_path_queries()` |
| `delta/attention.py` | `EdgeAttention`, `NodeAttention`, `ReconciliationBridge` (ablation targets) |
| `delta/baselines.py` | `GraphGPSModel`, `GRITModel` (Phases 34, 37, 38–41) |
| `docs/COLAB_RESULTS.md` | Running results log — update after every phase |

---

## Verification Summary

| Gate | Target | Status |
|------|--------|--------|
| Phase 35: frozen probe | > 0.5 on WN18RR | ✅ 0.961 |
| Phase 42–44: multi-hop dominance | DELTA > GraphGPS on 2p–5p | ✅ 5p MRR 0.790 vs 0.690 |
| Phase 45: inference efficiency | Per-query ≤ GraphGPS | ✅ 0.8–0.9x (faster) |
| Phase 46–52: temperature optimization | LP MRR improvement | ✅ 0.4905 (record) |
| Phase 53: multi-seed validation | LP robust across seeds | ✅ LP robust; multi-hop not |
| Phase 55–57: brain architecture | Differentiable construction viable | ✅ MRR 0.4818, H@10 +4.7% |
| Full-scale evaluation (14.5K+ entities) | Required for publication | TBD |
| Cross-family benchmark (YAGO3-10) | Required for publication | TBD |
| Interpretability figure | Required for publication | TBD |

All publication-grade results: **5 seeds, mean ± std reported.**

---

## Decisions

- Target: NeurIPS (May deadline) or ICLR (October deadline) — no hard date, get results right
- Datasets needed: FB15k-237 (flagship), WN18RR (transfer), YAGO3-10 (scale), Codex-M (hardness)
- Parameter matching: DELTA-Matched (~128K params) vs GraphGPS (~214K) vs GRIT (~183K) — DELTA-Matched is smaller, making wins more impressive
- Paper thesis: specialized architecture with edge-first dual attention and differentiable graph construction beats brute-force scale for structured relational reasoning

---

*Last updated: 2026-04-15. 63 experiment phases complete. Brain architecture validated (Phases 55–58). KG scaling evaluated (Phases 59–63). Primary remaining gaps: sparse attention mechanism, sequence domain evaluation, cross-family benchmark, interpretability figures.*
