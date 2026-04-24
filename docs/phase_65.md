# Phase 65 — Brain Hybrid with Sparse Attention at N=5000 (Compute-Deferred)

## Result

```
Phase: 65 — Brain Hybrid with Sparse Attention at N=5000
Hypothesis: BrainConstructor + topk=128 sparse attention at N=5000 achieves test MRR > 0.2472
Expected: brain_hybrid test MRR > Phase 64 Condition B (topk=128) = 0.2472
Seeds: [42]
Result: DEFERRED — experiment aborted due to compute cost (816s/epoch × 75 epochs × 2 conditions ≈ 34hr)

Status: No MRR result. Phase closed as compute-deferred.
Phase 64 baseline remains best N=5000 result: test_MRR=0.2472
```

## Details

### Hypothesis

BrainConstructor learns to add structurally informative edges (Gumbel-sigmoid from O(N²) candidates) at
target_density=0.001 → ~24,765 new edges per batch. Stage 3 DELTA layers then reason over the augmented
graph (152K original + 25K constructed = 177K edges). Combined with topk=128 sparse attention (validated
lossless in Phase 64), the augmented graph stays memory-feasible at N=5000.

Expected outcome: augmented topology provides richer structural shortcuts not present in the training graph,
yielding test MRR > 0.2472.

### Experimental Design

- **Condition B:** brain_hybrid, router=OFF (BrainConstructor + 2 Stage-3 DELTALayers, no PostAttentionPruner)
- **Condition C:** brain_hybrid, router=ON (same + PostAttentionPruner in Stage 3)
- **Architecture:** BrainEncoder: 1 bootstrap DELTALayer (Stage 1) → BrainConstructor (Stage 2) → 2 DELTALayers on augmented graph (Stage 3)
- **Config:** MAX_EPOCHS=75, EVAL_EVERY=25, PATIENCE=3, BS=4096, LR=0.003, SEED=42, SPARSITY_W=0.01
- **Hardware:** RTX PRO 6000 Blackwell 98GB, RunPod

### Why It Was Deferred

#### Cost per epoch
- Phase 64 (1 DELTA layer): 508s/epoch
- Phase 65 (3-stage BrainEncoder): **816.7s/epoch** — 1.6× slower due to 3-stage forward pass
- At 75 epochs × 2 conditions: **~34 hours wall clock**
- Estimated GPU cost: **~$64** on RunPod RTX PRO 6000 ($1.89/hr)

The pod running PID 16242 became unreachable during monitoring after ~2 epochs. With the paper deadline
approaching, the cost/benefit ratio did not justify provisioning a new pod.

#### What epoch 1 showed
Epoch 1 completed successfully in 816.7s with:
- `sp_loss=0.0407` — sparsity loss healthy, constructor is learning
- `constructed_edges=24765` — matches expected ~24,765 from target_density=0.001 at N=5000
- No OOM, no numerical issues — architecture is fully functional at N=5000

### Engineering Contributions

Phase 65 produced significant infrastructure improvements despite no final MRR result:

#### Fix 1: Pure PyTorch fallback in `graph.py`
`build_edge_adjacency()` required `torch_sparse` (not installed on RunPod). Added `try/except ImportError`
with vectorized pure-PyTorch fallback using `torch.repeat_interleave` + sort + unique-dedup. Handles the
E > 500 path fully without `torch_sparse`.

#### Fix 2: Stage 3 E_adj cache update
After subsampling the augmented E_adj in Stage 3, `augmented_graph._edge_adj_cache` was not updated,
causing DELTALayer to rebuild the full (unsubsampled) adj on each subsequent layer call. Fixed by
explicitly setting `augmented_graph._edge_adj_cache = (1, aug_edge_adj)` after subsampling.

#### Fix 3: Vectorized `evaluate_lp_fast` (70K GPU-CPU sync elimination)
Original `evaluate_lp` from phase46c performs 70,656 `.item()` calls (GPU→CPU sync) per evaluation:
one each for `h[i]`, `r[i]`, `t[i]`, `rank` per triple. Under 92GB memory pressure with
`expandable_segments:True`, each sync takes 50–100ms → 60–90 min per eval.

`evaluate_lp_fast` replaces this with vectorized batched scoring: encode once per eval, score all 4977
tails/heads in batch, apply filtered mask as tensor ops. GPU-CPU syncs reduced from 70,656 to ~6
(final rank array `.cpu().numpy()`). Expected eval time: ~23s.

#### Fix 4: Remove `torch.cuda.empty_cache()` from hot path
`brain.py` called `torch.cuda.empty_cache()` in Stage 3 setup (both router=OFF and router=ON paths)
on every forward pass — 37.3×/epoch. With a 92.5GB CUDA memory pool (from `expandable_segments:True`
preserving allocations across batches), each `empty_cache()` call attempted OS-level compaction of the
entire 92.5GB pool → took ~25 minutes per call after epoch 25. Removed both calls entirely.

#### Fix 5: Stage 1 subsample handles `_edge_adj_cache=None`
During evaluation with `cached_edge_adj=None`, `graph._edge_adj_cache` is None, so the original Stage 1
subsample code (guarded by `if graph._edge_adj_cache is not None`) was skipped entirely. Result:
DELTALayer built the full 63M adj → attempted `[63M, 128]` = 30.7GB ctx tensor → only 5.4GB free → 9+ min
CUDA compaction. Fixed: when `_edge_adj_cache is None`, build the original adj fresh then subsample to 30M
before running Stage 1.

### Paper Impact

Phase 65 results are not needed for the NeurIPS 2026 submission. The paper already contains:
- Brain architecture results at N=494 (Appendix B: "Brain Architecture (Preliminary)") — MRR 0.484, H@10 +4.7%
- Future work section referencing BrainEncoder "+4.7% Hits@10 in preliminary experiments"
- Phase 64 sparse attention validation (topk=128 lossless) as the main scaling contribution

Phase 65 would have promoted Brain from appendix to a main N=5000 result. This is deferred to Phase 66+
once compute budget allows, or framed as the natural next experiment post-submission.

### Commits

- `4def6d4` — All 5 infrastructure fixes (graph.py fallback, Stage 3 cache, evaluate_lp_fast, remove empty_cache, Stage 1 eval path)
- `af8d292` — Reduce MAX_EPOCHS 150→75, PATIENCE 2→3 (pivot before abort)
