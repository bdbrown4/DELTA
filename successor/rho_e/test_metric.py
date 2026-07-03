"""Gating tests for rho_E v0.2 (successor/rho_e/metric.py).

Each test traces to a red-team flaw or a Step-1 exit criterion (successor/STEP1.md). All must pass
before Step 2 computes anything on real datasets — failures here mean confidently-wrong ratios later.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from metric import compute_rho_e, _predictability, _key_ids, _summaries

RNG = np.random.default_rng(7)
N_NODES, D_X, N_TYPES = 400, 8, 5


def _nodes():
    return RNG.normal(size=(N_NODES, D_X))


def _simple_edges(n=4000):
    src = RNG.integers(0, N_NODES, n)
    dst = RNG.integers(0, N_NODES, n)
    tau = RNG.integers(0, N_TYPES, n)
    return src, dst, tau


# ── T1 (Prop 1 constant baseline): deterministic z = phi(x_u, tau, x_v) -> rho ~ 0 ──
def test_deterministic_z_gives_low_rho():
    x = _nodes()
    src, dst, tau = _simple_edges()
    z = np.sin(x[src, 0]) + 0.5 * x[dst, 1] + 0.3 * tau + 0.01 * RNG.normal(size=len(src))
    rep = compute_rho_e(src, dst, tau, [("z", z, "continuous")], x, seed=0)
    assert rep.status == "ok"
    assert rep.rho_e < 0.2, f"deterministic z must be ~fully predictable (rho={rep.rho_e:.3f})"


# ── T2: pure-noise z (independent of W) -> rho ~ 1 ──
def test_noise_z_gives_high_rho():
    x = _nodes()
    src, dst, tau = _simple_edges()
    z = RNG.normal(size=len(src))
    rep = compute_rho_e(src, dst, tau, [("z", z, "continuous")], x, seed=0)
    assert rep.rho_e > 0.8, f"independent z must be ~unpredictable (rho={rep.rho_e:.3f})"


# ── T3 (flaw 6): type-determined z -> excluded dim -> rho = 0 exactly, degenerate status ──
def test_type_determined_z_is_degenerate_zero():
    x = _nodes()
    src, dst, tau = _simple_edges()
    lookup = RNG.normal(size=N_TYPES)
    rep = compute_rho_e(src, dst, tau, [("z", lookup[tau], "continuous")], x, seed=0)
    assert rep.status == "degenerate_constant"
    assert rep.rho_e == 0.0
    assert rep.dims[0].excluded


# ── T4 (extended empty-z): no z dims -> schema_empty, rho = 0 ──
def test_schema_empty():
    x = _nodes()
    src, dst, tau = _simple_edges(500)
    rep = compute_rho_e(src, dst, tau, [], x)
    assert rep.status == "schema_empty" and rep.rho_e == 0.0


# ── T5 (flaw 1, THE anti-leakage gate): recurring triples with per-key content. Grouped (key-level)
#    evaluation must report HIGH rho; naive edge-level CV would have reported it LOW via twin lookup. ──
def test_antileakage_recurring_triples():
    x = _nodes()
    n_keys = 1200
    ku = RNG.integers(0, N_NODES, n_keys)
    kv = RNG.integers(0, N_NODES, n_keys)
    kt = RNG.integers(0, N_TYPES, n_keys)
    centers = RNG.uniform(0, 100, n_keys)              # per-key content, INDEPENDENT of x
    reps = 3
    src = np.repeat(ku, reps); dst = np.repeat(kv, reps); tau = np.repeat(kt, reps)
    z = np.repeat(centers, reps) + 0.5 * RNG.normal(size=n_keys * reps)   # tight within-key
    rep = compute_rho_e(src, dst, tau, [("ts", z, "continuous")], x, seed=0)
    assert rep.dup_fraction > 0.9
    assert rep.rho_e > 0.7, f"per-key content independent of W must stay irreducible (rho={rep.rho_e:.3f})"
    # decomposition sanity: within-key term small (tight clusters), across-key term unpredicted
    d = rep.dims[0]
    assert d.share_within_key < 0.2
    assert d.v_summary < 0.3

    # demonstrate the closed trap: EDGE-level rows with twin leakage would look predictable.
    # (internal access on purpose — the public API has no ungrouped path.)
    top_types = np.argsort(-np.bincount(tau))[:50]
    t_onehot = (tau[:, None] == top_types[None, :]).astype(float)
    feats_edges = np.concatenate([x[src], x[dst], t_onehot, tau[:, None].astype(float)], axis=1)
    zs = (z - z.mean()) / z.std()
    v_knn, v_gbt = _predictability(feats_edges, zs, tau, "continuous", 5, 0)   # naive: rows = edges
    naive_rho = 1.0 - max(v_knn, v_gbt)
    assert naive_rho < 0.35, (
        f"naive edge-level CV should exhibit the twin-leakage trap (naive_rho={naive_rho:.3f}); "
        "if this stops reproducing, the gating story needs re-examination")


# ── T6 (flaw 4): mixed deterministic + noise dims -> intermediate rho, per-dim vector separates them ──
def test_mixed_dims_and_per_dim_vector():
    x = _nodes()
    src, dst, tau = _simple_edges()
    z_det = np.sin(x[src, 0]) + 0.5 * x[dst, 1] + 0.05 * RNG.normal(size=len(src))
    z_noise = RNG.normal(size=len(src))
    rep = compute_rho_e(src, dst, tau,
                        [("det", z_det, "continuous"), ("noise", z_noise, "continuous")], x, seed=0)
    by = {d.name: d for d in rep.dims}
    assert by["det"].rho_dim < 0.3 and by["noise"].rho_dim > 0.7
    assert 0.25 < rep.rho_e < 0.75


# ── T8 (flaw 4/8): scale invariance — rescaling a dim by 1000 must not change rho_E materially ──
def test_scale_invariance():
    x = _nodes()
    src, dst, tau = _simple_edges()
    z_det = np.sin(x[src, 0]) + 0.5 * x[dst, 1] + 0.05 * RNG.normal(size=len(src))
    z_noise = RNG.normal(size=len(src))
    r1 = compute_rho_e(src, dst, tau,
                       [("det", z_det, "continuous"), ("noise", z_noise, "continuous")], x, seed=0)
    r2 = compute_rho_e(src, dst, tau,
                       [("det", z_det, "continuous"), ("noise", 1000.0 * z_noise, "continuous")], x, seed=0)
    assert abs(r1.rho_e - r2.rho_e) < 0.05, f"standardization must kill unit dependence ({r1.rho_e:.3f} vs {r2.rho_e:.3f})"


# ── categorical path: predictable vs unpredictable categorical z ──
def test_categorical_dims():
    x = _nodes()
    src, dst, tau = _simple_edges()
    z_pred = (x[src, 0] + x[dst, 1] > 0).astype(int)          # derivable from W
    z_rand = RNG.integers(0, 4, len(src))                      # independent
    rep = compute_rho_e(src, dst, tau,
                        [("pred", z_pred, "categorical"), ("rand", z_rand, "categorical")], x, seed=0)
    by = {d.name: d for d in rep.dims}
    assert by["pred"].rho_dim < 0.35
    assert by["rand"].rho_dim > 0.7


# ── edge budget: subsampling keeps whole keys (never splits twins) and stays in the right regime ──
def test_edge_budget_group_aware():
    x = _nodes()
    n_keys = 800
    ku = RNG.integers(0, N_NODES, n_keys); kv = RNG.integers(0, N_NODES, n_keys)
    kt = RNG.integers(0, N_TYPES, n_keys)
    centers = RNG.uniform(0, 100, n_keys)
    src = np.repeat(ku, 3); dst = np.repeat(kv, 3); tau = np.repeat(kt, 3)
    z = np.repeat(centers, 3) + 0.5 * RNG.normal(size=n_keys * 3)
    rep = compute_rho_e(src, dst, tau, [("ts", z, "continuous")], x, seed=0, edge_budget=900)
    assert rep.n_edges <= 903                                   # whole keys only (multiples of 3)
    assert rep.n_edges % 3 == 0, "budget must subsample whole keys, never split twins"
    assert rep.rho_e > 0.6                                      # regime preserved under budget


# ═══ audit-round tests (each traces to a confirmed v0.2 implementation bug) ═══

def _multigraph(n_keys=1000, reps=3, seed=11):
    rng = np.random.default_rng(seed)
    ku = rng.integers(0, N_NODES, n_keys); kv = rng.integers(0, N_NODES, n_keys)
    kt = rng.integers(0, N_TYPES, n_keys)
    return (np.repeat(ku, reps), np.repeat(kv, reps), np.repeat(kt, reps),
            ku, kv, kt, rng)


# ── audit 9 (frame mixing): adding a big type-determined offset must NOT change rho_E ──
def test_type_mass_invariance():
    x = _nodes()
    src, dst, tau, ku, kv, kt, rng = _multigraph()
    centers = rng.uniform(0, 10, len(ku))
    z = np.repeat(centers, 3) + 0.3 * rng.normal(size=len(src))
    lookup = rng.uniform(0, 500, N_TYPES)                       # huge type-determined mass
    r1 = compute_rho_e(src, dst, tau, [("z", z, "continuous")], x, seed=0)
    r2 = compute_rho_e(src, dst, tau, [("z", z + lookup[tau], "continuous")], x, seed=0)
    assert abs(r1.rho_e - r2.rho_e) < 0.08, (
        f"type-determined mass must not dilute irreducibility ({r1.rho_e:.3f} vs {r2.rho_e:.3f})")


# ── audit 10 (entropy base): categorical multigraph with KNOWN within-key share ──
def test_categorical_multigraph_share_within():
    x = _nodes()
    src, dst, tau, ku, kv, kt, rng = _multigraph(n_keys=1500, reps=4, seed=13)
    n_cls = 4
    # each key uniform over a RANDOM PAIR of the 4 global classes -> true within-key share
    # = log(2)/log(4) = 0.5 under GLOBAL normalization (the buggy per-call version reported ~1.0)
    pair_a = rng.integers(0, n_cls, len(ku))
    pair_b = (pair_a + 1 + rng.integers(0, n_cls - 1, len(ku))) % n_cls
    pick = rng.integers(0, 2, len(src))
    z = np.where(pick == 0, np.repeat(pair_a, 4), np.repeat(pair_b, 4))
    rep = compute_rho_e(src, dst, tau, [("z", z, "categorical")], x, seed=0)
    d = rep.dims[0]
    assert 0.35 < d.share_within_key < 0.7, (
        f"within-key share must use the GLOBAL entropy base (~0.5, got {d.share_within_key:.3f})")
    assert rep.rho_e > 0.6                                       # pair choice independent of W


# ── audit 12 (relative tolerance): jitter-constants excluded at ANY scale; tiny real dims kept ──
def test_jitter_constant_excluded_any_scale():
    x = _nodes()
    src, dst, tau = _simple_edges()
    jit = 5.0 + 1e-8 * RNG.normal(size=len(src))
    for scale in (1.0, 1e6):
        rep = compute_rho_e(src, dst, tau, [("z", scale * jit, "continuous")], x, seed=0)
        assert rep.dims[0].excluded and rep.rho_e == 0.0, f"jitter-constant at x{scale} must be excluded"
    genuine = 1e-6 * RNG.normal(size=len(src))                   # tiny scale, genuine variation
    rep = compute_rho_e(src, dst, tau, [("z", genuine, "continuous")], x, seed=0)
    assert not rep.dims[0].excluded and rep.rho_e > 0.8


# ── audit 13 (summary collapse): summaries EXACTLY type-determined + within-key variation.
#    (Random flips would create genuine beyond-type across-key noise, and rho ~ 1 would be CORRECT
#    under the within-type frame — the collapse must fire only in the exact case, so construct it:
#    every key keeps mode = its type by flipping exactly 2 of 5 reps.) ──
def test_type_determined_summaries_within_key_flips():
    x = _nodes()
    src, dst, tau, ku, kv, kt, rng = _multigraph(n_keys=1200, reps=5, seed=17)
    z = kt.repeat(5).astype(int)                                 # base: all reps = the key's type
    flip_pos = np.zeros(len(src), dtype=bool)
    flip_pos[0::5] = True; flip_pos[1::5] = True                 # exactly 2 of 5 per key -> mode stays
    z[flip_pos] = (z[flip_pos] + 1 + rng.integers(0, N_TYPES - 1, int(flip_pos.sum()))) % N_TYPES
    rep = compute_rho_e(src, dst, tau, [("z", z, "categorical")], x, seed=0)
    d = rep.dims[0]
    assert rep.status == "ok" and not d.excluded
    # summaries (modes) are exactly type-determined -> acc0 = 1 -> V := 1 -> rho_dim == share_within
    assert d.v_summary == 1.0, f"exact type-determined summaries must trigger the collapse (V={d.v_summary})"
    assert abs(d.rho_dim - d.share_within_key) < 0.05, (
        f"rho_dim must reduce to the within-key share "
        f"(rho_dim={d.rho_dim:.3f}, share_within={d.share_within_key:.3f})")


# ── audit 15/16: many types (top-50 overflow), a type seen once, and bitwise determinism ──
def test_many_types_unseen_tau_and_determinism():
    x = _nodes()
    n = 3000
    src = RNG.integers(0, N_NODES, n); dst = RNG.integers(0, N_NODES, n)
    tau = RNG.integers(0, 60, n)                                 # 60 types > top-50 one-hot
    tau[0] = 61                                                  # a type appearing exactly once
    z = np.sin(x[src, 0]) + RNG.normal(size=n)
    r1 = compute_rho_e(src, dst, tau, [("z", z, "continuous")], x, seed=3)
    r2 = compute_rho_e(src, dst, tau, [("z", z, "continuous")], x, seed=3)
    assert np.isfinite(r1.rho_e) and r1.status == "ok"
    assert r1.rho_e == r2.rho_e, "same seed must be bitwise deterministic"


# ── real-data regression: categorical z with SINGLETON classes must not crash (GBT early-stopping) ──
def test_categorical_singleton_classes_no_crash():
    x = _nodes()
    src, dst, tau = _simple_edges(3000)
    # a categorical dim with several classes that appear exactly once (per-key summaries -> singletons)
    z = RNG.integers(0, 8, len(src))
    z[:6] = [100, 101, 102, 103, 104, 105]        # six singleton classes
    rep = compute_rho_e(src, dst, tau, [("z", z, "categorical")], x, seed=0)
    assert rep.status == "ok" and np.isfinite(rep.rho_e)


# ── audit 15: too few keys -> clean 'insufficient_keys' status, no crash ──
def test_insufficient_keys_status():
    x = _nodes()
    src = np.array([0, 1, 2]); dst = np.array([3, 4, 5]); tau = np.array([0, 1, 2])
    z = np.array([0.1, 0.2, 0.3])
    rep = compute_rho_e(src, dst, tau, [("z", z, "continuous")], x, seed=0)
    assert rep.status == "insufficient_keys"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
