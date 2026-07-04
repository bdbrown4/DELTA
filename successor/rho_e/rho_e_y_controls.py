"""Positive/negative controls for the rho_E^Y construction — is a real ~0 a true negative or a dead metric?

Step-2 firm-up: rho_E^Y came back ~0 on ICEWS (and low on WD50K, whose gain-bearing qualifier ENTITIES are
excluded from z by the tractability rule). Before reading those as "content not task-relevant", we must show
the construction DETECTS relevance when it is present. Two synthetic-z controls on a real graph (ICEWS
skeleton), small budget:
  * POSITIVE: z_synth = leak of the true tail's structural feature (+ noise). Task-relevant BY
    CONSTRUCTION -> rho_E^Y must be clearly > 0.
  * NEGATIVE: z_synth = pure noise, independent of everything -> rho_E^Y must be ~0.
If POS is high and NEG ~0, the metric works and the real ~0 is a genuine "irreducible-but-irrelevant".

Usage: python successor/rho_e/rho_e_y_controls.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from compute_spectrum import load_icews14
from rho_e_y import compute_rho_e_y

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    base = load_icews14()
    x = base["x"]
    src, dst, tau = base["src"], base["dst"], base["tau"]
    rng = np.random.default_rng(0)

    # POSITIVE: z leaks the tail's first structural coordinate (log out-degree) + noise.
    # A relevance-aware measure MUST see this: knowing z narrows which candidate tail is right.
    z_leak = x[dst, 0].astype(float) + 0.25 * rng.standard_normal(len(dst))
    data_pos = dict(src=src, dst=dst, tau=tau, x=x,
                    z_dims=[("synth_tail_leak", z_leak, "continuous")])

    # NEGATIVE: pure noise, independent of the tail.
    z_noise = rng.standard_normal(len(dst))
    data_neg = dict(src=src, dst=dst, tau=tau, x=x,
                    z_dims=[("synth_noise", z_noise, "continuous")])

    rep_pos = compute_rho_e_y(data_pos, n_neg=1, seed=0, pos_budget=8000)
    rep_neg = compute_rho_e_y(data_neg, n_neg=1, seed=0, pos_budget=8000)

    print("=== rho_E^Y construction controls (ICEWS skeleton, 8k positives) ===")
    print(f"  POSITIVE (z = tail-leak):  rho_E^Y = {rep_pos['rho_e_y_joint']:.4f}  "
          f"AUC_W={rep_pos['auc_W']:.3f} -> AUC_Wz={rep_pos['auc_Wz_joint']:.3f}")
    print(f"  NEGATIVE (z = pure noise): rho_E^Y = {rep_neg['rho_e_y_joint']:.4f}  "
          f"AUC_W={rep_neg['auc_W']:.3f} -> AUC_Wz={rep_neg['auc_Wz_joint']:.3f}")
    ok = rep_pos["rho_e_y_joint"] > 0.1 and rep_neg["rho_e_y_joint"] < 0.03
    print(f"\n  VERDICT: construction {'WORKS' if ok else 'SUSPECT'} "
          f"(positive detects relevance, negative stays ~0)" if ok else
          f"\n  VERDICT: SUSPECT — pos={rep_pos['rho_e_y_joint']:.3f} neg={rep_neg['rho_e_y_joint']:.3f}")
    json.dump(dict(pos=rep_pos["rho_e_y_joint"], neg=rep_neg["rho_e_y_joint"],
                   auc_W_pos=rep_pos["auc_W"], auc_Wz_pos=rep_pos["auc_Wz_joint"]),
              open(os.path.join(ROOT, "successor", "rho_e", "rho_e_y_controls.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
