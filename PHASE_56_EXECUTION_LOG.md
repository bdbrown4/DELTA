# Phase 56: Constructor Density Ablation — Execution Log

## Status: IN PROGRESS (Condition A running)

**Start Time**: 2026-04-09, ~21:00 UTC
**Hardware**: RTX 3080 Ti (12.9GB VRAM) — local fallback due to Colab tunnel expiration
**Expected Duration**: ~20-24 hours (full 4 conditions × 300 epochs)

## Conditions Progress

### Condition A: delta_full (baseline, no construction)
- **Status**: IN PROGRESS
- **Epochs**: 30/300 complete (~10%)
- **Time Elapsed**: ~460s (~7.7 min)
- **Val MRR @ Ep 30**: 0.0151 (still early in training)
- **Expected Completion**: ~40-50 epochs of conditions A × 300 epochs cumulative

### Condition B: brain_hybrid @ density=0.005
- **Status**: QUEUED
- **Expected Start**: After Condition A completes (~4-5 hours)

### Condition C: brain_hybrid @ density=0.01
- **Status**: QUEUED
- **Expected Start**: After Condition B completes (~8-10 hours)

### Condition D: brain_hybrid @ density=0.02
- **Status**: QUEUED
- **Expected Start**: After Condition C completes (~12-14 hours)

## Context

### Why Local Fallback?
1. Colab tunnel 1 (driving-jewelry-creator-defensive.trycloudflare.com): DNS expired
2. Colab tunnel 2 (commons-prefer-correctly-excessive.trycloudflare.com): DNS failed
3. Colab tunnel 3 (murray-band-secret-americas.trycloudflare.com): Connection dropped during Phase 56 execution
4. Decision: Run Phase 56 locally on RTX 3080 Ti as fallback (user absent, unable to provide new Colab hostname)

### Phase 55 Baseline Results
- **Model**: brain_hybrid @ target_density=0.02
- **Test MRR**: 0.4773 (-0.002 vs 0.475 threshold)
- **Test H@10**: 0.7973 (+3.7% over delta_full)
- **Hypothesis**: Reducing density from 0.02 → 0.01 achieves MRR >= 0.480

### Phase 56 Hypothesis
**"Reducing constructor density from 0.02 to 0.01 improves brain_hybrid LP MRR to >= 0.480 by constructing fewer, higher-quality edges."**

**Expected Direction**: 
- Lower density → fewer edges → less noise → higher H@1/MRR at mild H@10 cost
- Optimal density balances recall gain against precision loss

## Expected Thresholds
- **Condition A (delta_full)**: Baseline, expected MRR ~0.4796 (from Phase 55)
- **Condition B (density=0.005)**: Conservative, expected improvement possible
- **Condition C (density=0.01)**: Target density, must achieve MRR >= 0.480 for hypothesis CONFIRMED
- **Condition D (density=0.02)**: Control (matches Phase 55 baseline)

## Data
- **Dataset**: FB15k-237, 500-entity dense subset
- **Entities**: 494
- **Relations**: 160
- **Train Triples**: 9,703
- **Val**: 390
- **Test**: 486

## Configuration
```
--seeds 42
--epochs 300
--eval_every 30
--patience 10
--max_entities 500
--sparsity_weight 0.01
--lr 0.001
--batch_size 512
```

## Next Steps (When Complete)
1. Fetch phase56_output.json from local run
2. Analyze results across all 4 densities
3. Classify hypothesis: CONFIRMED/PARTIAL/REJECTED
4. Create docs/phase_56.md with structured results
5. Update research_state.json
6. Commit Phase 56 results and documentation
7. Plan Phase 57 per continuous loop protocol
