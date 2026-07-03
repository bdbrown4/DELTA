"""Step 2 — compute rho_E across the validation spectrum (docs/SUCCESSOR_SCOPE.md §3).

Each loader must obey the z-free x-specification (successor/STEP1.md §(ii)): delete all z, THEN compute x.
Structural x for featureless KGs: log-degrees, top-R relation-incidence histograms (in/out), PageRank.

Usage:
  python successor/rho_e/compute_spectrum.py --datasets fb15k237
  python successor/rho_e/compute_spectrum.py --datasets fb15k237,icews14 --out successor/rho_e/spectrum.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse, json, time

import numpy as np

from metric import compute_rho_e, RhoEReport

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ───────────────────────── structural x (z-free, model-free) ─────────────────────────

def structural_x(src, dst, tau, n_nodes, n_rels, top_r=50, pr_iters=30):
    """STEP1 §(ii) structural feature set, computed on the (already z-stripped) typed simple graph."""
    src = np.asarray(src); dst = np.asarray(dst); tau = np.asarray(tau)
    # collapse to unique typed edges (multiplicity is structure at the SIMPLE-graph level; timestamps/
    # qualifiers were already stripped by the caller — see STEP1 x-spec)
    trip = np.unique(np.stack([src, tau, dst], axis=1), axis=0)
    s, t, d = trip[:, 0], trip[:, 1], trip[:, 2]
    out_deg = np.bincount(s, minlength=n_nodes).astype(float)
    in_deg = np.bincount(d, minlength=n_nodes).astype(float)
    top = np.argsort(-np.bincount(t, minlength=n_rels))[:top_r]
    rank_of = {int(r): i for i, r in enumerate(top)}
    hist_out = np.zeros((n_nodes, len(top))); hist_in = np.zeros((n_nodes, len(top)))
    for si, ti, di in zip(s, t, d):
        j = rank_of.get(int(ti))
        if j is not None:
            hist_out[si, j] += 1.0
            hist_in[di, j] += 1.0
    hist_out /= np.maximum(hist_out.sum(1, keepdims=True), 1.0)
    hist_in /= np.maximum(hist_in.sum(1, keepdims=True), 1.0)
    # PageRank (alpha=.85) on the untyped simple digraph, power iteration
    alpha, pr = 0.85, np.full(n_nodes, 1.0 / n_nodes)
    outd = np.maximum(out_deg, 1.0)
    for _ in range(pr_iters):
        contrib = np.zeros(n_nodes)
        np.add.at(contrib, d, pr[s] / outd[s])
        pr = (1 - alpha) / n_nodes + alpha * (contrib + pr[out_deg == 0].sum() / n_nodes)
    return np.concatenate([np.log1p(out_deg)[:, None], np.log1p(in_deg)[:, None],
                           hist_out, hist_in, np.log(pr + 1e-12)[:, None]], axis=1)


# ───────────────────────── loaders ─────────────────────────

def load_fb15k237():
    """Symbolic triple store: NO per-edge attributes -> schema-degenerate anchor (rho_E := 0)."""
    ents, rels = {}, {}
    src, dst, tau = [], [], []
    for split in ("train.txt", "valid.txt", "test.txt"):
        with open(os.path.join(ROOT, "data", "fb15k-237", split), encoding="utf-8") as f:
            for line in f:
                h, r, t = line.rstrip("\n").split("\t")
                src.append(ents.setdefault(h, len(ents)))
                tau.append(rels.setdefault(r, len(rels)))
                dst.append(ents.setdefault(t, len(ents)))
    src, dst, tau = np.array(src), np.array(dst), np.array(tau)
    x = structural_x(src, dst, tau, len(ents), len(rels))
    return dict(src=src, dst=dst, tau=tau, x=x, z_dims=[],
                note="symbolic KG; no edge attributes beyond type; schema-degenerate by construction")


def load_icews14():
    """Temporal KG: z = timestamp (day index). Downloads once from a public mirror if absent."""
    ddir = os.path.join(ROOT, "data", "icews14")
    os.makedirs(ddir, exist_ok=True)
    files = ["train.txt", "valid.txt", "test.txt"]
    if not all(os.path.exists(os.path.join(ddir, f)) for f in files):
        import urllib.request
        base = "https://raw.githubusercontent.com/mniepert/mmkb/master/TemporalKGs/icews14/"
        names = {"train.txt": "icews_2014_train.txt", "valid.txt": "icews_2014_valid.txt",
                 "test.txt": "icews_2014_test.txt"}
        for local, remote in names.items():
            urllib.request.urlretrieve(base + remote, os.path.join(ddir, local))
    ents, rels = {}, {}
    src, dst, tau, ts = [], [], [], []
    for split in files:
        with open(os.path.join(ddir, split), encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                h, r, t, d = parts[0], parts[1], parts[2], parts[3]
                src.append(ents.setdefault(h, len(ents)))
                tau.append(rels.setdefault(r, len(rels)))
                dst.append(ents.setdefault(t, len(ents)))
                y, m, dd = d.split("-")
                ts.append(int(y) * 372 + int(m) * 31 + int(dd))   # monotone day index
    src, dst, tau, ts = np.array(src), np.array(dst), np.array(tau), np.array(ts, dtype=float)
    # z-free x: timestamps are z -> x computed on the timestamp-stripped typed graph (structural_x
    # already collapses to unique typed edges, which IS the timestamp-stripped simple graph)
    x = structural_x(src, dst, tau, len(ents), len(rels))
    return dict(src=src, dst=dst, tau=tau, x=x,
                z_dims=[("timestamp_day", ts, "continuous")],
                note="temporal KG; z = event day; x on timestamp-stripped typed simple graph")


LOADERS = {"fb15k237": load_fb15k237, "icews14": load_icews14}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default="fb15k237")
    ap.add_argument("--edge_budget", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "successor", "rho_e", "spectrum.json"))
    args = ap.parse_args()

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for name in [d.strip() for d in args.datasets.split(",")]:
        t0 = time.time()
        print(f"[{name}] loading ...")
        data = LOADERS[name]()
        rep: RhoEReport = compute_rho_e(
            data["src"], data["dst"], data["tau"], data["z_dims"], data["x"],
            seed=args.seed, edge_budget=args.edge_budget)
        row = {
            "rho_e": rep.rho_e, "status": rep.status, "n_edges": rep.n_edges, "n_keys": rep.n_keys,
            "dup_fraction": round(rep.dup_fraction, 4), "edge_budget": rep.edge_budget,
            "note": data["note"],
            "dims": [{"name": d.name, "kind": d.kind, "excluded": d.excluded,
                      "weight": round(d.weight, 4), "share_within_key": round(d.share_within_key, 4),
                      "v_knn": round(d.v_summary_knn, 4), "v_gbt": round(d.v_summary_gbt, 4),
                      "rho_dim": round(d.rho_dim, 4)} for d in rep.dims],
            "elapsed_s": round(time.time() - t0, 1),
        }
        results[name] = row
        print(f"[{name}] rho_E = {rep.rho_e:.4f} ({rep.status}) | edges={rep.n_edges} "
              f"keys={rep.n_keys} dup={rep.dup_fraction:.2%} [{row['elapsed_s']}s]")
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
