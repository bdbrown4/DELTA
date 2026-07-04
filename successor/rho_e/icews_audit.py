"""Step-2 firm-up: is ICEWS rho_E=0.94 an artifact of the equal-key-weighting estimand caveat?

ICEWS is a multigraph (dup_fraction 0.585): the same (h,r,t) triple recurs on different days, so the
within-key term (share_within=0.305) contributes, and the metric weights keys equally in V but edges in
the dispersion share (STEP1 estimand caveat). If the 0.94 were driven by that frame mismatch on a
heavy-tailed key-size distribution, it would be fragile. Test: restrict to SIZE-1 keys (the simple-graph
subset — NO within-key term, share_within=0 by construction) and recompute. If timestamps are still
irreducible there (rho_E stays high, driven purely by the across-key predictive term V), the high value
is real content-irreducibility, not a multigraph-weighting artifact.

Usage: python successor/rho_e/icews_audit.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from metric import compute_rho_e, _key_ids
from compute_spectrum import load_icews14

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    data = load_icews14()
    src, dst, tau = data["src"], data["dst"], data["tau"]
    z_dims = data["z_dims"]
    x = data["x"]
    key = _key_ids(src, tau, dst)
    sizes = np.bincount(key)
    edge_size = sizes[key]

    print("=== ICEWS key-size distribution ===")
    print(f"  n_edges={len(src)} n_keys={len(sizes)} dup_fraction={np.mean(edge_size>1):.3f}")
    for q in (50, 90, 95, 99, 100):
        print(f"  key_size p{q}: {np.percentile(sizes, q):.0f}")
    print(f"  max key size: {sizes.max()}  (that triple recurs on {sizes.max()} days)")

    # (1) full (reproduce 0.94)
    t0 = time.time()
    rep_full = compute_rho_e(src, dst, tau, z_dims, x, seed=0, edge_budget=100_000)
    print(f"\n[full]        rho_E={rep_full.rho_e:.4f} share_within={rep_full.dims[0].share_within_key:.4f} "
          f"V={rep_full.dims[0].v_summary:.4f} [{time.time()-t0:.0f}s]")

    # (2) SIZE-1 keys only: simple-graph subset, share_within := 0 by construction.
    # rho_E here is PURELY the across-key predictive term (1 - V): can a single event's day be predicted
    # from (x_h, r, x_t)? If no (rho_E high), timestamps are irreducible independent of the multigraph.
    keep = edge_size == 1
    z1 = [(nm, np.asarray(v)[keep], kd) for nm, v, kd in z_dims]
    t0 = time.time()
    rep1 = compute_rho_e(src[keep], dst[keep], tau[keep], z1, x, seed=0, edge_budget=100_000)
    print(f"[size-1 only] rho_E={rep1.rho_e:.4f} share_within={rep1.dims[0].share_within_key:.4f} "
          f"V={rep1.dims[0].v_summary:.4f} n_edges={rep1.n_edges} [{time.time()-t0:.0f}s]")

    # (3) cap the heavy tail: drop keys larger than p95, recompute (removes equal-weighting leverage)
    cap = np.percentile(sizes, 95)
    keepc = edge_size <= cap
    zc = [(nm, np.asarray(v)[keepc], kd) for nm, v, kd in z_dims]
    t0 = time.time()
    repc = compute_rho_e(src[keepc], dst[keepc], tau[keepc], zc, x, seed=0, edge_budget=100_000)
    print(f"[cap<=p95]    rho_E={repc.rho_e:.4f} share_within={repc.dims[0].share_within_key:.4f} "
          f"V={repc.dims[0].v_summary:.4f} n_edges={repc.n_edges} [{time.time()-t0:.0f}s]")

    out = dict(
        key_size_p95=float(np.percentile(sizes, 95)), key_size_max=int(sizes.max()),
        dup_fraction=float(np.mean(edge_size > 1)),
        rho_full=rep_full.rho_e, rho_size1=rep1.rho_e, rho_capped=repc.rho_e,
        V_full=rep_full.dims[0].v_summary, V_size1=rep1.dims[0].v_summary,
        share_within_full=rep_full.dims[0].share_within_key,
    )
    json.dump(out, open(os.path.join(ROOT, "successor", "rho_e", "icews_audit.json"), "w"), indent=2)
    print("\nVERDICT: the anti-ordering (ICEWS high rho_E, low gain) is",
          "ROBUST — high even on the simple-graph subset" if rep1.rho_e > 0.7 else
          "SENSITIVE to the multigraph frame — REVISIT")


if __name__ == "__main__":
    main()
