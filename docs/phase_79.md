# Phase 79 — Is the "JIT ≈ 2× AOT" gain the trained readout, or the path mechanism?

## Status: PRE-REGISTERED (design red-teamed, 3 lenses, build_with_fixes). Building.

## Question

In Phase 77 the query-time PEC-pf model (JIT) roughly **doubled** the standard encode-once soft-traversal
(AOT) MRR at every depth — *including 1p, where JIT does zero traversal*. So the 2× is not edge
participation; it is the trained all-N multi-label-BCE pooled readout/objective vs parameter-free
soft-traversal scoring. Phase 79 asks: **is that 2× advantage (a) the trained query-time readout alone**
(mundane — "train a head"; it tightens the negative by showing encode-once soft-traversal *understates*
multi-hop ability), **or (b) does the query-time edge/path-state add MEANINGFUL value beyond a
capacity-matched trained readout** (a genuine positive → motivates Phase-80 co-training)?

## What the red-team caught (2 kills_design, both the same root cause)

The legacy AOT-soft eval (`phase76_diagnose_5p.fast_mh_eval`) is **not** a fair paired control: it scores
in the **64-d decoder space** (`node_proj_out` + the trained `decoder_rel_emb`), whereas JIT operates in
the **48-d encoder-internal** `nf` with its own `rel_emb`. It is also **not parameter-free** (uses the
trained LP DistMult decoder), and it stores no per-query RR. Comparing JIT to it confounds **space +
scorer + objective** with "edge participation." Fix: pin one 48-d space; build a shared-space soft floor
(`AOT-soft48`); the legacy AOT-soft is reported as a **context number only**, never paired.

## Arms (all on the SAME frozen 48-d `nf`/`ef0` from `encode_with_edges`, same 15 seeds, same testbed)

| arm | z-state on the penult frontier | scorer | trained? |
|---|---|---|---|
| **AOT-soft (legacy)** | 64-d soft-traversal over the decoder | trained LP decoder | — (context only) |
| **AOT-soft48** | raw `nf[e]` | **fixed** DistMult (frozen rel) | no (floor) |
| **AOT-readout** | `seed_norm(W_seed(nf[e]))` — node-only | **trained** pooled-DistMult head | yes (PRIMARY control) |
| **JIT-pec** | edge-composed path-state | same trained head | yes |
| **JIT-static** | edge-as-input, composition OFF, params matched | same trained head | yes (guard) |

**Shared frontier (the key fairness control):** all three trained-axis arms pool the **same discrete
train-reachable penult frontier** (parameter-free graph reachability — the `cur` set from
`penult_reachable_mask`). They differ ONLY in the z-state (raw vs node-seed vs edge-composed) and the
scorer (fixed vs trained head). This shares pool-support exactly, so `gap_head` and `gap_path` are clean
one-axis contrasts. (Refinement vs the synthesizer's soft-distribution spec: a uniform discrete frontier
is a *tighter* isolation and avoids an arbitrary soft-traversal relation rep.) **`ef0`/edges never enter
AOT-readout or AOT-soft48** — no edge operator (`W_q/W_k/W_v/W_out/W_ctx` absent), node-only z.

**K=1 identity (the falsifiable plumbing gate):** at 1p there is no traversal — the frontier is the
singleton `{anchor}` and AOT-readout reduces to `seed_norm(W_seed(nf[anchor])) → readout`, byte-identical
to JIT-pec's `forward_query` K=1 path. Capacity of the AOT-readout head is param-for-param matched to
`PathComposerPF`'s readout (`W_seed, seed_norm, rel_emb, W_poolattn, W_read, read_bias`).

## Decision (seed-level paired-t, n=15, primary depth 3p, δ=0.015; reuse `phase78_analyze` machinery)

`gap_total = JIT-pec − AOT-soft48`, `gap_head = AOT-readout − AOT-soft48`, `gap_path = JIT-pec −
AOT-readout`, `path_fraction = gap_path/gap_total`.

PRIMARY = `gap_path` @3p, judged by the same four-quadrant (significant × within-±δ-band) rule as Phase 78:
- **(i) GENUINE POSITIVE** — `gap_path` significant AND CI reaches outside ±δ — *fires only if ALSO
  `JIT-pec > JIT-static` @3p by the margin* (conjunctive guard: rules out "more params" / "ef0-as-input").
  → the path mechanism adds meaningful value → Phase-80 co-training.
- **(ii) SIGNIFICANT-BUT-NEGLIGIBLE** — `gap_path` significant but 90% CI within ±δ → "the 2× is the
  trained readout/objective, not the path." (Expected, given Phases 66–78.)
- **(iii) REFUTE/equivalent** — `gap_path` equivalent within ±δ (with lever) → readout, full stop.
- **(iv) INCONCLUSIVE** — CI wider than ±δ → add seeds.

A PARTIAL (`AOT-readout` recovers most-but-not-all of `gap_total`) is set by `gap_path`'s quadrant, not a
narrative — a tiny-but-significant `gap_path` is "head, not path."

**1p go/no-go gate (runs first):** TOST that `JIT-pec_1p == AOT-readout_1p` at tight tolerance; a 1p
difference means the heads are not form-matched → the 3p comparison is VOID (plumbing bug, not a finding).

## Smallest decisive version

A **1p-only run** across the 15 seeds: at 1p neither arm traverses and K=1 = all train triples is already
generated, so for ~1/5 the compute it settles "is the 2× just a trained head?" (does AOT-readout_1p ≈
JIT-pec_1p ≈ 2× the published AOT-soft, in the shared space vs the floor). Deep hops (3p primary, with the
`gap_path` + `JIT-static` guard) run only as confirmation of whether ANY path signal survives.

## Reuse
`PathComposerPF` (JIT-pec mode='pec', JIT-static mode='static') + `encode_with_edges` + `build_csr` +
`penult_reachable_mask`; `phase77` `train_arm`/`eval_arm`/`generate_train_chains` (verbatim — AOT-readout
exposes the same `score_batch_vec` signature); `phase78` resumable per-(arm,seed) driver + the seed-level
paired-t/TOST/four-quadrant analyzer; the 15 seeds + frozen `p76_hops1` (+ `p76_node_only` for a scope
follow-up: does the readout 2× reproduce on a node-only-pretrained table?).

## Honest prior
Given pec ≈ static ≈ shuffle_full across Phases 66–78, the expected outcome is **(ii)/(iii)**: the 2× is
the trained readout — a clean methodological note (encode-once soft-traversal understates multi-hop
ability), and the negative on edge composition tightens. A genuine positive (i) would be the surprise that
reopens the program.
