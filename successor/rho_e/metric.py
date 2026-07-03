"""rho_E — Edge-Content Irreducibility (reference implementation, v0.2.1).

The model-free, pre-training diagnostic from docs/SUCCESSOR_SCOPE.md §2 + successor/STEP1.md.
Estimates the fraction of edge-attribute information NOT derivable from the z-free observables
W = (x_u, tau, x_v). One-directional claim (STEP1 Proposition 1): rho_E ~ 0 => no expressivity-level
content gain is available to a smooth-in-z edge-aware architecture. High rho_E does NOT imply gain.

Design decisions (each traces to a red-team/audit flaw; see SUCCESSOR_SCOPE.md §2 + STEP1.md):
  * NO ungrouped CV path. All prediction runs at the KEY level k=(u,tau,v): KFold over UNIQUE KEYS,
    which is leakage-equivalent to GroupKFold on edges (no key can straddle folds). ESTIMAND CAVEAT
    (documented, v0.3 item): V weights keys equally while dispersion shares are edge-weighted; report
    key-size stats so heavy-tailed datasets can be flagged. [flaw 1 + audit 14]
  * x must be z-free and model-free — the CALLER's obligation, per STEP1 §(ii). [flaws 2, 3b]
  * Per-dim irreducibility, one unified path, BOTH TERMS IN THE SAME WITHIN-TYPE FRAME (audit 9 —
    normalizing share_within by total dispersion let type-determined mass dilute the irreducible term,
    an anti-conservative frame-mixing bug caught by numeric probe):
        rho_dim = share_within(k) + (1 - share_within(k)) * (1 - V_summary)
    with share_within = within-key dispersion / WITHIN-TYPE dispersion, and V_summary the type-chance-
    corrected key-level predictability of per-key summaries E[z|k].
  * Categorical entropies are ALL normalized by the GLOBAL class count of the dim (audit 10 — per-call
    normalization compared incommensurable log bases and inflated share_within).
  * Degenerate conventions, RELATIVE tolerance on the raw scale (audit 12): dims whose raw within-type
    dispersion is <= 1e-6 x raw total dispersion are excluded (V := 1, w := 0); all-excluded or empty z
    => rho_E := 0 with an explicit degenerate status. Summary-level collapse (audit 13): if the type
    baseline explains the summaries (near-)perfectly, V := 1 (type-determined summary variation is not
    irreducible).
  * Weights on globally standardized dims; the per-dim vector is the primary report. [flaw 4]
  * Continuous skill = 1 - MSE_oof(model)/MSE_oof(type-conditional mean), type means out-of-fold,
    pooled THEN clipped. Categorical analog with out-of-fold type-majority accuracy. [flaw 7]
  * Estimator pair fixed and disclosed (kNN + histogram gradient boosting); headline V = max of the two.
    NOTE (STEP1 Prop 1(c)): clip-then-max carries a small maximal-selection (anti-conservative) bias of
    order the CV s.e. — a STATISTICAL channel controlled by fold-CI / fixed-budget reporting, not closed
    by construction.
  * Optional fixed edge budget subsamples whole KEYS (never splits twins). Too few keys => status
    'insufficient_keys' (no crash). [flaw 8 + audit 15]
"""
from __future__ import annotations

import dataclasses
from typing import Literal, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

DispKind = Literal["continuous", "categorical"]
_EPS = 1e-12
_REL_TOL = 1e-6          # relative degenerate-dim tolerance (audit 12)


@dataclasses.dataclass
class DimReport:
    name: str
    kind: DispKind
    excluded: bool                 # ~zero within-type dispersion, RELATIVE to total (degenerate convention)
    weight: float                  # within-type dispersion share on standardized dims (0 if excluded)
    share_within_key: float        # exact irreducible term, within-TYPE frame (0 on simple graphs)
    v_summary_knn: float
    v_summary_gbt: float
    v_summary: float               # max of the pair (headline; see maximal-selection caveat)
    rho_dim: float


@dataclasses.dataclass
class RhoEReport:
    rho_e: float
    status: Literal["ok", "schema_empty", "degenerate_constant", "insufficient_keys"]
    dims: list[DimReport]
    n_edges: int
    n_keys: int
    dup_fraction: float
    key_size_max: int              # estimand-caveat flags (audit 14)
    key_size_p95: float
    n_splits: int
    edge_budget: int | None


# ──────────────────────────────── helpers ────────────────────────────────

def _norm_entropy(values: np.ndarray, n_classes_global: int) -> float:
    """Entropy normalized by log of the GLOBAL class count of the dim (audit 10). 0 if <2 classes."""
    if n_classes_global < 2:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    if len(counts) < 2:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(n_classes_global))


def _dispersion(values: np.ndarray, kind: DispKind, n_classes_global: int = 0) -> float:
    if kind == "continuous":
        return float(np.var(values.astype(float)))
    return _norm_entropy(values, n_classes_global)


def _within_group_dispersion(values: np.ndarray, groups: np.ndarray, kind: DispKind,
                             n_classes_global: int = 0) -> float:
    """Edge-weighted E[D(z | group)] — the exact within-group dispersion term."""
    total, n = 0.0, len(values)
    order = np.argsort(groups, kind="stable")
    gv, gs = values[order], groups[order]
    bounds = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = gv[a:b]
        if len(seg) > 1:
            if kind == "continuous":
                d = float(np.var(seg.astype(float)))
            else:
                d = _norm_entropy(seg, n_classes_global)
            total += d * len(seg)
    return total / max(n, 1)


def _key_ids(src: np.ndarray, tau: np.ndarray, dst: np.ndarray) -> np.ndarray:
    trip = np.stack([src, tau, dst], axis=1)
    _, ids = np.unique(trip, axis=0, return_inverse=True)
    return ids


def _summaries(values: np.ndarray, key_of_edge: np.ndarray, n_keys: int, kind: DispKind) -> np.ndarray:
    if kind == "continuous":
        sums = np.bincount(key_of_edge, weights=values.astype(float), minlength=n_keys)
        cnts = np.bincount(key_of_edge, minlength=n_keys)
        return sums / np.maximum(cnts, 1)
    out = np.empty(n_keys, dtype=values.dtype)
    order = np.argsort(key_of_edge, kind="stable")
    kv, ks = values[order], key_of_edge[order]
    bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = kv[a:b]
        u, c = np.unique(seg, return_counts=True)
        out[ks[a]] = u[np.argmax(c)]
    return out


def _predictability(feats: np.ndarray, target: np.ndarray, tau_rows: np.ndarray,
                    kind: DispKind, n_splits: int, seed: int) -> tuple[float, float]:
    """Chance-corrected out-of-fold predictability (V) of `target` rows from `feats`. Rows are UNIQUE
    KEYS (never edges) => plain KFold carries no twin risk by construction. Baselines (type-majority /
    type-mean) estimated on train folds only. Summary-level degenerate collapse (audit 13): a
    (near-)perfect type baseline yields V := 1, not a 0/EPS blow-up. Returns (v_knn, v_gbt),
    pooled-over-folds then clipped to [0, 1]."""
    n = len(target)
    n_splits = int(min(n_splits, max(2, n // 20))) if n >= 40 else 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = {"knn": np.empty(n, dtype=object), "gbt": np.empty(n, dtype=object)}
    base = np.empty(n, dtype=object)
    scaler = StandardScaler()

    for tr, te in kf.split(feats):
        Xtr, Xte = feats[tr], feats[te]
        ytr = target[tr]
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        k = int(np.clip(np.sqrt(len(tr)), 3, 25))
        k = max(1, min(k, len(tr)))                          # small-n guard (audit 15)
        if kind == "continuous":
            models = {"knn": KNeighborsRegressor(n_neighbors=k),
                      "gbt": HistGradientBoostingRegressor(random_state=seed)}
            gmean = float(np.mean(ytr))
            tmeans = {int(t): float(np.mean(ytr[tau_rows[tr] == t])) for t in np.unique(tau_rows[tr])}
            for i in te:
                base[i] = tmeans.get(int(tau_rows[i]), gmean)   # unseen-tau fallback: global mean
        else:
            models = {"knn": KNeighborsClassifier(n_neighbors=k),
                      # early_stopping=False: its default uses a STRATIFIED internal split that raises
                      # on singleton classes (rare categories in real z, e.g. rare qualifier relations).
                      "gbt": HistGradientBoostingClassifier(random_state=seed, early_stopping=False)}
            u, c = np.unique(ytr, return_counts=True)
            gmaj = u[np.argmax(c)]
            tmaj = {}
            for t in np.unique(tau_rows[tr]):
                ut, ct = np.unique(ytr[tau_rows[tr] == t], return_counts=True)
                tmaj[int(t)] = ut[np.argmax(ct)]
            for i in te:
                base[i] = tmaj.get(int(tau_rows[i]), gmaj)
        for mname, model in models.items():
            X_tr, X_te = (Xtr_s, Xte_s) if mname == "knn" else (Xtr, Xte)
            model.fit(X_tr, ytr)
            yhat = model.predict(X_te)
            for j, i in enumerate(te):
                preds[mname][i] = yhat[j]

    out = []
    for mname in ("knn", "gbt"):
        yhat = np.array([preds[mname][i] for i in range(n)])
        yb = np.array([base[i] for i in range(n)])
        if kind == "continuous":
            yhat = yhat.astype(float); yb = yb.astype(float); yt = target.astype(float)
            mse_m = float(np.mean((yhat - yt) ** 2))
            mse_0 = float(np.mean((yb - yt) ** 2))
            var_t = float(np.var(yt))
            if mse_0 <= _REL_TOL * max(var_t, _EPS):          # type baseline ~perfect on summaries
                v = 1.0                                        # -> across-key term fully type-determined
            else:
                v = 1.0 - mse_m / mse_0                        # pooled, then clipped
        else:
            acc_m = float(np.mean(yhat == target))
            acc_0 = float(np.mean(yb == target))
            if acc_0 >= 1.0 - 1e-12:
                v = 1.0
            else:
                v = (acc_m - acc_0) / (1.0 - acc_0)
        out.append(float(np.clip(v, 0.0, 1.0)))
    return out[0], out[1]


# ──────────────────────────────── main API ────────────────────────────────

def compute_rho_e(src: np.ndarray, dst: np.ndarray, tau: np.ndarray,
                  z_dims: Sequence[tuple[str, np.ndarray, DispKind]],
                  x: np.ndarray, *, n_splits: int = 5, seed: int = 0,
                  edge_budget: int | None = None) -> RhoEReport:
    """Compute rho_E. See module docstring; x MUST be z-free and model-free (caller's obligation)."""
    src = np.asarray(src); dst = np.asarray(dst); tau = np.asarray(tau)
    n_edges = len(src)

    if len(z_dims) == 0:
        return RhoEReport(0.0, "schema_empty", [], n_edges, 0, 0.0, 0, 0.0, n_splits, edge_budget)

    key = _key_ids(src, tau, dst)
    if edge_budget is not None and n_edges > edge_budget:
        rng = np.random.default_rng(seed)
        uk = np.unique(key)
        rng.shuffle(uk)
        sizes = np.bincount(key)
        keep_keys, count = [], 0
        for kid in uk:
            keep_keys.append(kid); count += int(sizes[kid])
            if count >= edge_budget:
                break
        mask = np.isin(key, np.array(keep_keys))
        src, dst, tau, key = src[mask], dst[mask], tau[mask], key[mask]
        z_dims = [(nm, np.asarray(v)[mask], kd) for nm, v, kd in z_dims]
        _, key = np.unique(key, return_inverse=True)
        n_edges = len(src)

    n_keys = int(key.max()) + 1 if n_edges else 0
    sizes = np.bincount(key, minlength=n_keys) if n_keys else np.array([0])
    dup_fraction = float(np.sum(sizes[key] > 1)) / max(n_edges, 1) if n_edges else 0.0
    key_size_max = int(sizes.max()) if n_keys else 0
    key_size_p95 = float(np.percentile(sizes, 95)) if n_keys else 0.0

    if n_keys < 2 * n_splits:                                 # small-n guard (audit 15)
        return RhoEReport(0.0, "insufficient_keys", [], n_edges, n_keys, dup_fraction,
                          key_size_max, key_size_p95, n_splits, edge_budget)

    # per-key features from W = (x_u, tau, x_v): built ONCE per unique key (vectorized; audit 17)
    _, first_edge_of_key = np.unique(key, return_index=True)
    ku, kt, kv = src[first_edge_of_key], tau[first_edge_of_key], dst[first_edge_of_key]
    top_types = np.argsort(-np.bincount(tau))[:50]
    t_onehot = (kt[:, None] == top_types[None, :]).astype(float)
    feats = np.concatenate([x[ku], x[kv], t_onehot, kt[:, None].astype(float)], axis=1)

    dims: list[DimReport] = []
    for name, values, kind in z_dims:
        values = np.asarray(values)
        ncls = int(len(np.unique(values))) if kind == "categorical" else 0
        # degenerate check (audit 12): RELATIVE, on the RAW scale, two scale-free criteria —
        # (a) type-determinism: within-type dispersion negligible vs total;
        # (b) near-constant: coefficient of variation sd/|mean| negligible (catches quantization
        #     jitter at ANY absolute scale without excluding genuinely tiny centered dims).
        raw_total = _dispersion(values, kind, ncls)
        raw_wt = _within_group_dispersion(values, tau, kind, ncls)
        near_constant = (kind == "continuous" and
                         float(np.sqrt(max(raw_total, 0.0))) <= _REL_TOL * abs(float(np.mean(values.astype(float)))))
        if raw_wt <= _REL_TOL * max(raw_total, _EPS) or near_constant:
            dims.append(DimReport(name, kind, True, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0))
            continue
        if kind == "continuous":
            sd = float(np.std(values.astype(float)))
            values_std = (values.astype(float) - float(np.mean(values))) / sd
        else:
            values_std = values
        wt_disp = _within_group_dispersion(values_std, tau, kind, ncls)      # within-TYPE frame
        # share_within in the SAME within-type frame as the chance-corrected V (audit 9)
        wk_disp = _within_group_dispersion(values_std, key, kind, ncls)
        share_within = float(np.clip(wk_disp / max(wt_disp, _EPS), 0.0, 1.0))
        summ = _summaries(values_std, key, n_keys, kind)
        v_knn, v_gbt = _predictability(feats, summ, kt, kind, n_splits, seed)
        v = max(v_knn, v_gbt)
        rho_dim = share_within + (1.0 - share_within) * (1.0 - v)
        dims.append(DimReport(name, kind, False, wt_disp, share_within, v_knn, v_gbt, v, rho_dim))

    included = [d for d in dims if not d.excluded]
    if not included:
        return RhoEReport(0.0, "degenerate_constant", dims, n_edges, n_keys, dup_fraction,
                          key_size_max, key_size_p95, n_splits, edge_budget)
    wsum = sum(d.weight for d in included)
    for d in included:
        d.weight = d.weight / wsum
    rho = float(sum(d.weight * d.rho_dim for d in included))
    return RhoEReport(rho, "ok", dims, n_edges, n_keys, dup_fraction,
                      key_size_max, key_size_p95, n_splits, edge_budget)
