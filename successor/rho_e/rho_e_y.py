"""rho_E^Y — the LABEL-AWARE edge-content relevance tier (STEP1 §"label-aware decision"; Step-3 only).

The label-free rho_E measures whether z is DERIVABLE from W=(x_u,tau,x_v). It does NOT measure whether z
is TASK-RELEVANT — and Step 2 showed those come apart (ICEWS: irreducible timestamps, low gain). rho_E^Y
closes that gap: the chance-corrected lift in real-vs-corrupted LINK discrimination that z adds beyond W.
It is the model-free image of the SCOPE §2 kill-gate's own "content-isolating z-on/off feature ablation".

Applies where a corruption target is natural (link prediction): ICEWS, WD50K dial, ogbl-collab. NOT molhiv
(graph classification — no per-edge real/fake target; its evidence is the label-free point + published gain).

CONSTRUCTION (each choice defuses a specific confound; DELTA-lineage rigor):
  * Positives = real edges (u,tau,v) with their z. Negatives = relation-constrained tail corruptions
    (u,tau,v'), v' drawn from tails OBSERVED with relation tau, filtered against the true (u,tau,*) tails.
    Relation-constrained (not uniform) so structure-only negatives are HARD and z has room to show value.
  * The negative SHARES its positive's z (same timestamp/qualifiers). z is therefore marginally identical
    across labels -> it can ONLY help via INTERACTION with the candidate's z-free features x_v'. This kills
    the labeling artifact "real edges carry z, fakes don't" and isolates "does content pick the right tail".
  * Features W=[x_u, x_v, onehot(tau)]; Wz=[W, standardized z]. Grouped CV by the positive key k=(u,tau,v)
    (twins + their negatives never straddle folds). Estimator pair (kNN, GBT); skill = OOF ROC-AUC;
    headline = max of the pair (maximal-selection caveat, as in rho_E).
  * rho_E^Y = clip((AUC_Wz - AUC_W) / (1 - AUC_W), 0, 1): chance-corrected share of the residual
    discriminability that z adds. Per-dim (z_j alone, appended to W) and JOINT (all z) reported.

Read: if rho_E^Y ORDERS published gain where the label-free rho_E anti-orders it, the diagnostic survives
in its label-aware form (#2 lives, label-aware). If rho_E^Y ALSO fails to order gain, #2 is fully dead and
#1 ships alone.

Usage: python successor/rho_e/rho_e_y.py --datasets icews14,wd50k_0,wd50k_100,ogbl_collab
"""
import sys, os, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from compute_spectrum import load_icews14, _load_wd50k_variant
from metric import _key_ids

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load(name):
    if name == "icews14":
        return load_icews14()
    if name.startswith("wd50k"):
        variant = "wd50k" if name == "wd50k_0" else name
        return _load_wd50k_variant(variant)
    if name == "ogbl_collab":
        from compute_ogb import load_ogbl_collab
        d = load_ogbl_collab()
        d["x"] = list(d["x_variants"].values())[0]
        return d
    raise KeyError(name)


def _build_features(src, dst, tau, x):
    top_types = np.argsort(-np.bincount(tau))[:50]
    t_onehot = (tau[:, None] == top_types[None, :]).astype(float)
    return np.concatenate([x[src], x[dst], t_onehot], axis=1)


def _auc_pair(W, y, groups, n_splits, seed):
    """OOF ROC-AUC of real/fake from features W, GroupKFold by `groups`. Returns max over (kNN, GBT)."""
    n = len(y)
    n_splits = int(min(n_splits, max(2, len(np.unique(groups)) // 2)))
    gkf = GroupKFold(n_splits=n_splits)
    oof = {"knn": np.zeros(n), "gbt": np.zeros(n)}
    scaler = StandardScaler()
    for tr, te in gkf.split(W, y, groups):
        Wtr_s = scaler.fit_transform(W[tr]); Wte_s = scaler.transform(W[te])
        k = int(np.clip(np.sqrt(len(tr)), 5, 50))
        knn = KNeighborsClassifier(n_neighbors=k).fit(Wtr_s, y[tr])
        oof["knn"][te] = knn.predict_proba(Wte_s)[:, 1]
        gbt = HistGradientBoostingClassifier(random_state=seed, early_stopping=False).fit(W[tr], y[tr])
        oof["gbt"][te] = gbt.predict_proba(W[te])[:, 1]
    return {m: float(roc_auc_score(y, oof[m])) for m in ("knn", "gbt")}


def _residualize_oof(z, X, groups, n_splits, seed):
    """OOF residual of z regressed on X (GBT), grouped CV. Removes the part of z predictable from X so any
    remaining AUC-lift is orthogonal to X. Used to strip the n_qual<->candidate-tail-degree confound."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    n = len(z)
    n_splits = int(min(n_splits, max(2, len(np.unique(groups)) // 2)))
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.zeros(n)
    for tr, te in gkf.split(X, z, groups):
        r = HistGradientBoostingRegressor(random_state=seed).fit(X[tr], z[tr])
        pred[te] = r.predict(X[te])
    return z - pred


def compute_rho_e_y(data, *, n_neg=1, seed=0, pos_budget=30_000, n_splits=5, residualize=False):
    rng = np.random.default_rng(seed)
    src, dst, tau, x = data["src"], data["dst"], data["tau"], data["x"]
    z_dims = data["z_dims"]
    if not z_dims:
        return dict(status="schema_empty", rho_e_y_joint=0.0, dims={})
    n_edges = len(src)

    # subsample positives to budget (whole edges)
    if n_edges > pos_budget:
        idx = rng.choice(n_edges, pos_budget, replace=False)
        src, dst, tau = src[idx], dst[idx], tau[idx]
        z_vals = {nm: np.asarray(v)[idx] for nm, v, _ in z_dims}
    else:
        z_vals = {nm: np.asarray(v) for nm, v, _ in z_dims}
    P = len(src)
    key = _key_ids(src, tau, dst)

    # relation-constrained corruption pools: tails observed per relation (from the FULL edge set)
    full_tau, full_dst = data["tau"], data["dst"]
    tails_by_rel = {}
    for t in np.unique(tau):
        tails_by_rel[int(t)] = np.unique(full_dst[full_tau == t])
    true_tails = {}  # (u,tau) -> set of true tails, for filtering
    for u, t, v in zip(src, tau, dst):
        true_tails.setdefault((int(u), int(t)), set()).add(int(v))

    # assemble rows: positives then negatives (negatives share parent z + parent group key)
    rows_src, rows_dst, rows_tau, rows_y, rows_grp = [list(src)], [list(dst)], [list(tau)], [[1]*P], [list(key)]
    z_rows = {nm: [list(z_vals[nm])] for nm in z_vals}
    for _ in range(n_neg):
        neg_v = np.empty(P, dtype=int)
        for i in range(P):
            pool = tails_by_rel[int(tau[i])]
            v2 = int(pool[rng.integers(len(pool))])
            tries = 0
            while v2 in true_tails[(int(src[i]), int(tau[i]))] and tries < 8 and len(pool) > 1:
                v2 = int(pool[rng.integers(len(pool))]); tries += 1
            neg_v[i] = v2
        rows_src.append(list(src)); rows_dst.append(list(neg_v)); rows_tau.append(list(tau))
        rows_y.append([0]*P); rows_grp.append(list(key))
        for nm in z_vals:
            z_rows[nm].append(list(z_vals[nm]))   # negative inherits parent z (shared)
    rsrc = np.array(sum(rows_src, [])); rdst = np.array(sum(rows_dst, []))
    rtau = np.array(sum(rows_tau, [])); y = np.array(sum(rows_y, [])); grp = np.array(sum(rows_grp, []))
    zr = {nm: np.array(sum(z_rows[nm], []), dtype=float) for nm in z_vals}

    W = _build_features(rsrc, rdst, rtau, x)
    # de-confound (optional): strip from each z the part predictable from the CANDIDATE TAIL's structural
    # features (the x[rdst] block of W = cols [xd:2*xd]). Kills the n_qual<->tail-degree channel the
    # adversarial audit surfaced; any surviving lift is content orthogonal to candidate structure.
    if residualize:
        xd = x.shape[1]
        tail_block = W[:, xd:2 * xd]
        for nm in zr:
            zr[nm] = _residualize_oof(zr[nm].astype(float), tail_block, grp, n_splits, seed)
    auc_W = _auc_pair(W, y, grp, n_splits, seed)
    base = max(auc_W.values())

    def lift(auc_full):
        best = max(auc_full.values())
        return dict(auc=best, rho=float(np.clip((best - base) / max(1 - base, 1e-9), 0, 1)))

    dims_out = {}
    for nm, vals, kind in z_dims:
        zcol = zr[nm]
        zstd = (zcol - zcol.mean()) / (zcol.std() + 1e-9)
        Wz = np.concatenate([W, zstd[:, None]], axis=1)
        dims_out[nm] = {**lift(_auc_pair(Wz, y, grp, n_splits, seed)), "auc_W": base}
    # joint: all z appended together
    Zall = np.stack([(zr[nm] - zr[nm].mean()) / (zr[nm].std() + 1e-9) for nm in zr], axis=1)
    Wz_all = np.concatenate([W, Zall], axis=1)
    joint = lift(_auc_pair(Wz_all, y, grp, n_splits, seed))

    return dict(status="ok", auc_W=base, auc_W_knn=auc_W["knn"], auc_W_gbt=auc_W["gbt"],
                rho_e_y_joint=joint["rho"], auc_Wz_joint=joint["auc"],
                n_pos=P, n_rows=len(y), n_neg=n_neg, dims=dims_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, required=True)
    ap.add_argument("--n_neg", type=int, default=1)
    ap.add_argument("--pos_budget", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--residualize", action="store_true",
                    help="strip candidate-tail-structure from z (de-confound the n_qual<->degree channel)")
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "successor", "rho_e", "rho_e_y.json"))
    args = ap.parse_args()

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for name in [d.strip() for d in args.datasets.split(",")]:
        t0 = time.time()
        print(f"[{name}] loading + building rho_E^Y {'(residualized)' if args.residualize else ''}...")
        data = _load(name)
        rep = compute_rho_e_y(data, n_neg=args.n_neg, seed=args.seed, pos_budget=args.pos_budget,
                              residualize=args.residualize)
        rep["elapsed_s"] = round(time.time() - t0, 1)
        results[name] = rep
        if rep["status"] == "ok":
            print(f"[{name}] rho_E^Y(joint)={rep['rho_e_y_joint']:.4f}  AUC_W={rep['auc_W']:.4f} "
                  f"AUC_Wz={rep['auc_Wz_joint']:.4f}  n_pos={rep['n_pos']} [{rep['elapsed_s']}s]")
            for nm, d in rep["dims"].items():
                print(f"      {nm:<18} rho_E^Y={d['rho']:.4f} (AUC {d['auc_W']:.3f}->{d['auc']:.3f})")
        else:
            print(f"[{name}] status={rep['status']}")
        json.dump(results, open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
