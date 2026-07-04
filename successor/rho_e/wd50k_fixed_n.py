"""Step-2 firm-up: the WD50K qualifier dial at a COMMON fixed edge budget, across seeds.

STEP2.md Finding 3 claimed the dial is "flat within noise (0.567-0.649)" but never computed a noise
band, and two of the four variants were BELOW the 100k budget (n-confound: smaller n -> upward rho_E
bias, SCOPE axiom 4). This control subsamples ALL four variants to a common 31k-edge budget (the size of
the smallest variant, wd50k_100) across several seeds, so the dial is read at matched n with an actual
spread. If the dial is still flat at fixed n, Finding 3's "coverage not irreducibility" conclusion is
firm; if a monotone rise appears at fixed n, the earlier flat reading was the n-confound and we must
revise.

Usage: python successor/rho_e/wd50k_fixed_n.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from metric import compute_rho_e
from compute_spectrum import _load_wd50k_variant

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BUDGET = 31_000
SEEDS = [0, 1, 2]
VARIANTS = [("wd50k_0", "wd50k"), ("wd50k_33", "wd50k_33"),
            ("wd50k_66", "wd50k_66"), ("wd50k_100", "wd50k_100")]


def main():
    out = {}
    for label, variant in VARIANTS:
        data = _load_wd50k_variant(variant)
        rhos = []
        for seed in SEEDS:
            t0 = time.time()
            rep = compute_rho_e(data["src"], data["dst"], data["tau"], data["z_dims"], data["x"],
                                seed=seed, edge_budget=BUDGET)
            rhos.append(rep.rho_e)
            print(f"[{label} seed={seed}] rho_E={rep.rho_e:.4f} n_edges={rep.n_edges} "
                  f"n_keys={rep.n_keys} dup={rep.dup_fraction:.2%} "
                  f"key_p95={rep.key_size_p95:.0f} key_max={rep.key_size_max} [{time.time()-t0:.0f}s]")
        rhos = np.array(rhos)
        out[label] = dict(mean=float(rhos.mean()), std=float(rhos.std(ddof=1)) if len(rhos) > 1 else 0.0,
                          min=float(rhos.min()), max=float(rhos.max()), seeds=SEEDS,
                          rhos=[round(float(r), 4) for r in rhos], budget=BUDGET,
                          n_edges=rep.n_edges, n_keys=rep.n_keys, qual_frac=data["note"])
        print(f"  => {label}: {rhos.mean():.4f} +/- {out[label]['std']:.4f} "
              f"(range {rhos.min():.4f}-{rhos.max():.4f})\n")
    outp = os.path.join(ROOT, "successor", "rho_e", "wd50k_fixed_n.json")
    json.dump(out, open(outp, "w"), indent=2)

    print("\n=== WD50K DIAL AT FIXED n =", BUDGET, "(across seeds) ===")
    means = [out[l]["mean"] for l, _ in VARIANTS]
    within_seed_sd = np.mean([out[l]["std"] for l, _ in VARIANTS])
    across_dial_range = max(means) - min(means)
    for l, _ in VARIANTS:
        o = out[l]
        print(f"  {l:<11} rho_E = {o['mean']:.4f} +/- {o['std']:.4f}  (seeds {o['rhos']})")
    print(f"\n  across-dial range of means:  {across_dial_range:.4f}")
    print(f"  mean within-seed sd:         {within_seed_sd:.4f}")
    monotone = all(means[i] <= means[i+1] + within_seed_sd for i in range(len(means)-1))
    print(f"  monotone-increasing beyond within-seed noise? {'YES' if monotone and across_dial_range > 2*within_seed_sd else 'NO'}")
    print(f"  VERDICT: dial is {'FLAT within noise' if across_dial_range <= 2*within_seed_sd else 'a real rise'} "
          f"at fixed n (range {across_dial_range:.3f} vs 2sigma {2*within_seed_sd:.3f})")
    print("saved", outp)


if __name__ == "__main__":
    main()
