# Successor project — "When Does Edge Content Matter?"

**This is not DELTA.** DELTA is a CLOSED project with a closed-negative result
([../docs/DELTA_FINAL.md](../docs/DELTA_FINAL.md)). This directory hosts its kill-gated successor,
scoped in [../docs/SUCCESSOR_SCOPE.md](../docs/SUCCESSOR_SCOPE.md): a theory note (the
feature-determinism boundary for line-graph attention) + a pre-training diagnostic (ρ_E,
Edge-Content Irreducibility) predicting when edge-aware architectures *can* help — with DELTA's
79-phase null as the anchor at the ρ ≈ 0 end of the curve.

| Piece | What |
|---|---|
| [STEP1.md](STEP1.md) | Formal conservativeness statement (Prop. 1), z-free x-spec, label-aware decision — the four blocking Step-1 exit criteria |
| [rho_e/metric.py](rho_e/metric.py) | ρ_E v0.2 reference implementation (key-level evaluation; no ungrouped path) |
| [rho_e/test_metric.py](rho_e/test_metric.py) | 9 gating tests, each tracing to a red-team flaw |
| [rho_e/compute_spectrum.py](rho_e/compute_spectrum.py) | Step-2 spectrum runner (loaders obey the z-free x-spec) |
| [rho_e/compute_spectrum.py](rho_e/compute_spectrum.py), [rho_e/icews_buckets.py](rho_e/icews_buckets.py), [rho_e/report.py](rho_e/report.py) | Step-2 spectrum runner, ICEWS bucket sweep, kill-gate report |
| [rho_e/spectrum.json](rho_e/spectrum.json) | The ratio table (6/7 points computed) |
| [STEP2.md](STEP2.md) | **Step-2 findings — the kill-gate signal (READ THIS)** |

**Pre-registered kill gate (Step 3):** if the mid regime doesn't discriminate better than the binary
"has edge attributes?" schema check (scored on non-empty-z datasets, content-isolating ablation deltas
only), the diagnostic drops to a companion observation and the theory note ships alone.

**⚠ Step-2 signal (see [STEP2.md](STEP2.md)):** the label-free ρ_E is **trending strongly toward that
kill** — it *anti-orders* published gain across the two non-empty-z families (ICEWS ρ=0.94 with *low*
gain; WD50K ρ≈0.6 with gains up to *high*), because irreducibility is neither coverage (the WD50K
qualifier dial is flat in ρ_E) nor task-relevance (ICEWS timestamps are irreducible but unhelpful). The
**one-directional** claim (low ρ_E ⇒ no gain) and the **theory note** survive; the label-aware tier ρ_E^Y
is now motivated, not refuted. Final verdict pending the molecular/ogbl points + the ρ_E^Y test.

Early honest notes: ICEWS14 at day granularity measures ρ_E ≈ 0.94 (expected "mid"), stable across the
v0.2 → v0.2.1 frame fix (decomposition: within-key share 0.31, across-key summaries near-unpredictable,
best Ṽ = 0.09). Time-bucket sensitivity is a live Step-2 item, and high-ρ/low-published-gain is the
task-relevance gap the label-aware tier (ρ_E^Y) exists to dissect. The one-directional law
(low ρ_E ⇒ no content gain) is untouched by either. The v0.2 implementation was adversarially audited
before first use: two metric-breaking bugs (frame mixing, per-call entropy base) were found by numeric
probe and fixed in v0.2.1; the gating suite grew to 15 tests.
