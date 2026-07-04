# Successor project — "When Does Edge Content Matter?"

**This is not DELTA.** DELTA is a CLOSED project with a closed-negative result
([../docs/DELTA_FINAL.md](../docs/DELTA_FINAL.md)). This directory hosts its kill-gated successor,
scoped in [../docs/SUCCESSOR_SCOPE.md](../docs/SUCCESSOR_SCOPE.md): a theory note (the
feature-determinism boundary for line-graph attention) + a pre-training diagnostic (ρ_E,
Edge-Content Irreducibility) predicting when edge-aware architectures *can* help — with DELTA's
79-phase null as the anchor at the ρ ≈ 0 end of the curve.

## ✅ Steps 1–3 COMPLETE — read [STEP3.md](STEP3.md) for the audited verdict

**Verdict (2026-07-03), after a three-referee adversarial audit of the whole Step-2/3 evidence:**
- **Label-free ρ_E — KILLED as a gain predictor.** It is **non-monotone in published gain**
  (band-invariantly: fails under ICEWS scored low/mid/high) and cannot beat the constant "has-z?" check.
  The pre-registered kill fired.
- **Label-free ρ_E — SURVIVES as a one-directional necessary condition** (low ρ_E ⇒ no gain), now supported
  by a real molecular point (bond features ρ_E ≈ 0.06–0.09, low gain; no counterexample on the spectrum).
- **Label-aware ρ_E^Y — PROMISING but UNCONFIRMED.** It looked like a clean rescue (monotone dial), but the
  audit caught that the fine dial is substantially a qualifier-count↔tail-degree **confound**. After
  de-confounding, the *coarse* relevant-vs-irrelevant separation survives (WD50K_100 content signal robust;
  ICEWS/collab correctly ~0), but the *fine gain calibration* does not — deferred to Step 5 under a
  pre-registered de-confounding protocol.
- **Theory note #1 — SHIPS** as the primary contribution (never depended on the diagnostic).

| Piece | What |
|---|---|
| [STEP1.md](STEP1.md) | Formal conservativeness statement (Prop. 1), z-free x-spec, label-aware decision |
| [STEP2.md](STEP2.md) | Step-2-state snapshot (⚠ Finding 3 corrected by Step 3) |
| **[STEP3.md](STEP3.md)** | **The audited Step-3 verdict — READ THIS** |
| [rho_e/metric.py](rho_e/metric.py) | ρ_E v0.2.1 reference implementation (key-level; 16/16 gating tests) |
| [rho_e/rho_e_y.py](rho_e/rho_e_y.py) | label-aware ρ_E^Y (real-vs-corrupted content lift; `--residualize` de-confound) |
| [rho_e/compute_spectrum.py](rho_e/compute_spectrum.py), [rho_e/compute_ogb.py](rho_e/compute_ogb.py) | spectrum runners (KGs; OGB molecular + ogbl-collab) |
| [rho_e/report.py](rho_e/report.py) | assembles all results + runs the gate (A) and de-confounded rescue (B) |
| [rho_e/icews_audit.py](rho_e/icews_audit.py), [rho_e/wd50k_fixed_n.py](rho_e/wd50k_fixed_n.py), [rho_e/rho_e_y_controls.py](rho_e/rho_e_y_controls.py) | the three Step-2/3 firm-up controls |
| [rho_e/spectrum.json](rho_e/spectrum.json), [spectrum_mol.json](rho_e/spectrum_mol.json), [spectrum_collab.json](rho_e/spectrum_collab.json), [rho_e_y.json](rho_e/rho_e_y.json), [rho_e_y_resid.json](rho_e/rho_e_y_resid.json) | computed results |

**What the adversarial audit caught (and this project is defined by catching):** a "band-shopping"
appearance in the anti-ordering wording (fixed → band-invariant non-monotonicity); an `is_in_ring`
z-correlation biasing the molecular ρ_E down (bounded: 0.063 vs 0.085); and, most importantly, that the
ρ_E^Y "rescue" was ~half-to-mostly a tail-degree confound with a leaky positive control (fixed → a
de-confounded, honestly-scoped "promising but unconfirmed" claim). The full transcript of fixes is in
STEP3.md.
