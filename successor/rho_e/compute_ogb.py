"""Step-2 spectrum, OGB points: molecular bond features (Finding-1 low end) + ogbl-collab (mid-point).

Framework-agnostic OGB loaders (numpy dicts; no torch_geometric). Both obey the STEP1 §(ii) z-free
x-spec: delete all z, THEN compute x.

MOLECULAR (ogbg-molhiv): a bond is a bond -> tau = CONSTANT (no relation TYPES), and z = the three bond
fields (bond_type, bond_stereo, is_conjugated). This is exactly the D-MPNN question "are bond features
endpoint-inferable from the two atoms?" We run TWO x-specs to guard against x-leakage (which would falsely
LOWER rho_E and flatter our own low-end prediction):
    - clean : atomic-number only + structural (unambiguously z-free)
    - full  : all 9 OGB atom features + structural (atom is_aromatic/is_in_ring are topology-derived and
              could leak the aromatic/conjugated z -> the adversarial-against-ourselves check)
Nodes are global across molecules (per-molecule atom-id offset), so bonds are ~unique keys: rho_E is
essentially the endpoint-predictability term, which is the point.

OGBL-COLLAB: z = (year continuous, weight continuous); tau = CONSTANT (one edge type). x = provided 128-d
node features + structural on the year/weight-stripped simple graph. True multigraph (author pairs recur
across years) -> the within-key term is live, like ICEWS.

Usage:
  python successor/rho_e/compute_ogb.py --datasets molhiv          --out .../spectrum_mol.json
  python successor/rho_e/compute_ogb.py --datasets ogbl_collab     --out .../spectrum_collab.json
"""
import sys, os, argparse, json, time, functools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
# OGB caches its processed graph as a .pt; PyTorch 2.6 flipped torch.load's default to weights_only=True,
# which cannot unpickle OGB's cache on a SECOND run (the first run builds from CSV). The cache is OGB's own
# trusted artifact, so restore weights_only=False.
torch.load = functools.partial(torch.load, weights_only=False)
from metric import compute_rho_e
from compute_spectrum import structural_x

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OGB_ROOT = os.path.join(ROOT, "data", "ogb")


def load_molhiv():
    """ogbg-molhiv: concat all molecules with global atom offsets. tau=const, z=3 bond fields."""
    from ogb.graphproppred import GraphPropPredDataset
    ds = GraphPropPredDataset(name="ogbg-molhiv", root=OGB_ROOT)
    src, dst = [], []
    bond_type, bond_stereo, bond_conj = [], [], []
    node_feat_rows = []
    offset = 0
    for i in range(len(ds)):
        g, _ = ds[i]
        nf = g["node_feat"]                       # (N,9) atom features
        ei = g["edge_index"]                      # (2,E) directed (both directions present in OGB)
        ef = g["edge_feat"]                       # (E,3) bond features
        n = g["num_nodes"]
        # OGB stores each undirected bond twice (u->v and v->u); keep u<v once to avoid double counting
        u, v = ei[0], ei[1]
        keep = u < v
        u, v, ef = u[keep], v[keep], ef[keep]
        src.append(u + offset); dst.append(v + offset)
        bond_type.append(ef[:, 0]); bond_stereo.append(ef[:, 1]); bond_conj.append(ef[:, 2])
        node_feat_rows.append(nf)
        offset += n
    src = np.concatenate(src); dst = np.concatenate(dst)
    bond_type = np.concatenate(bond_type).astype(int)
    bond_stereo = np.concatenate(bond_stereo).astype(int)
    bond_conj = np.concatenate(bond_conj).astype(int)
    node_feat = np.concatenate(node_feat_rows, axis=0).astype(float)   # (Ntot,9)
    n_nodes = node_feat.shape[0]
    tau = np.zeros(len(src), dtype=int)               # single edge type: a bond is a bond
    z_dims = [("bond_type", bond_type, "categorical"),
              ("bond_stereo", bond_stereo, "categorical"),
              ("is_conjugated", bond_conj, "categorical")]
    struct = structural_x(src, dst, tau, n_nodes, 1)   # z-free structural (log-deg, PR; hist degenerate)
    # OGB atom features (9): 0 atomic_num, 1 chirality, 2 degree, 3 formal_charge, 4 numH,
    # 5 num_radical_e, 6 hybridization, 7 is_aromatic, 8 is_in_ring. Per STEP1 x-spec, x may use only
    # intrinsic/topological atom fields VERIFIED not bond-attribute-derived. RDKit derives hybridization
    # (6) and is_aromatic (7) FROM bond orders/aromaticity, and numH (4) from valence/bond order -> these
    # three x-LEAK the bond-attribute z (anti-conservative). The z-free-verified subset excludes them.
    # CAVEAT (adversarial audit): is_in_ring (8) is z-free by PROVENANCE (SSSR ring topology, not a bond
    # attribute) but z-CORRELATED (aromatic/conjugated bonds imply both endpoints in a ring), so including
    # it biases rho_E DOWNWARD. Hence zfree=0.06 is the leak-guarded-but-ring-aided OPTIMISTIC end; the
    # principled value is a RANGE [~0.06 zfree, ~0.46 clean]. (zfree_noring available for bounding.)
    ZFREE_IDX = [0, 1, 2, 3, 5, 8]                      # atomic_num, chirality, degree, charge, radical, in_ring
    ZFREE_NORING_IDX = [0, 1, 2, 3, 5]                 # drop is_in_ring to bound the ring-aided bias
    atomic_num = node_feat[:, 0:1]                     # unambiguously z-free (minimal)
    x_clean = np.concatenate([atomic_num, struct], axis=1)
    x_zfree = np.concatenate([node_feat[:, ZFREE_IDX], struct], axis=1)          # STEP1 canonical
    x_zfree_noring = np.concatenate([node_feat[:, ZFREE_NORING_IDX], struct], axis=1)  # drop ring proxy
    x_full = np.concatenate([node_feat, struct], axis=1)   # incl. hybridization/is_aromatic/numH (leaky)
    return dict(src=src, dst=dst, tau=tau, z_dims=z_dims,
                x_variants={"clean": x_clean, "zfree": x_zfree,
                            "zfree_noring": x_zfree_noring, "full": x_full},
                note="ogbg-molhiv bonds; tau=const; z=bond_type+stereo+conjugated; x-specs "
                     "(clean=atomicnum+struct; zfree=STEP1 non-bond-derived atom subset+struct; "
                     "zfree_noring=zfree minus is_in_ring (bounds ring-aided bias); "
                     "full=all-atom-feats+struct, leaky via hybridization/is_aromatic/numH)")


def load_ogbl_collab():
    """ogbl-collab: z=(year,weight); tau=const; x=128-d node feats + structural (year/weight-stripped)."""
    from ogb.linkproppred import LinkPropPredDataset
    ds = LinkPropPredDataset(name="ogbl-collab", root=OGB_ROOT)
    g = ds[0]
    ei = g["edge_index"]
    src, dst = ei[0].astype(int), ei[1].astype(int)
    # find year + weight regardless of exact key layout
    year = None; weight = None
    for k, val in g.items():
        if val is None:
            continue
        arr = np.asarray(val)
        if "year" in k:
            year = arr.reshape(-1).astype(float)
        elif "weight" in k:
            weight = arr.reshape(-1).astype(float)
        elif k == "edge_feat" and arr.ndim == 2:
            # fallback: [weight, year] or [year, weight]; disambiguate by range (year ~ 1990..2020)
            c0, c1 = arr[:, 0].astype(float), arr[:, 1].astype(float)
            if c0.min() > 1900:
                year, weight = c0, c1
            else:
                year, weight = c1, c0
    node_feat = np.asarray(g["node_feat"]).astype(float)
    n_nodes = node_feat.shape[0]
    tau = np.zeros(len(src), dtype=int)
    z_dims = [("year", year, "continuous"), ("weight", weight, "continuous")]
    struct = structural_x(src, dst, tau, n_nodes, 1)
    x = np.concatenate([node_feat, struct], axis=1)
    return dict(src=src, dst=dst, tau=tau, z_dims=z_dims, x_variants={"nodefeat+struct": x},
                note="ogbl-collab; tau=const; z=year+weight; x=128d nodefeat + structural(stripped)")


LOADERS = {"molhiv": load_molhiv, "ogbl_collab": load_ogbl_collab}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, required=True)
    ap.add_argument("--edge_budget", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for name in [d.strip() for d in args.datasets.split(",")]:
        t0 = time.time()
        print(f"[{name}] loading ...")
        data = LOADERS[name]()
        for xkey, x in data["x_variants"].items():
            rep = compute_rho_e(data["src"], data["dst"], data["tau"], data["z_dims"], x,
                                seed=args.seed, edge_budget=args.edge_budget)
            key = name if len(data["x_variants"]) == 1 else f"{name}[{xkey}]"
            results[key] = {
                "rho_e": rep.rho_e, "status": rep.status, "n_edges": rep.n_edges, "n_keys": rep.n_keys,
                "dup_fraction": round(rep.dup_fraction, 4), "edge_budget": rep.edge_budget,
                "key_size_max": rep.key_size_max, "key_size_p95": rep.key_size_p95,
                "x_spec": xkey, "note": data["note"],
                "dims": [{"name": d.name, "kind": d.kind, "excluded": d.excluded,
                          "weight": round(d.weight, 4), "share_within_key": round(d.share_within_key, 4),
                          "v_knn": round(d.v_summary_knn, 4), "v_gbt": round(d.v_summary_gbt, 4),
                          "rho_dim": round(d.rho_dim, 4)} for d in rep.dims],
                "elapsed_s": round(time.time() - t0, 1),
            }
            print(f"[{key}] rho_E={rep.rho_e:.4f} ({rep.status}) edges={rep.n_edges} keys={rep.n_keys} "
                  f"dup={rep.dup_fraction:.2%} keyp95={rep.key_size_p95:.0f} [{results[key]['elapsed_s']}s]")
            json.dump(results, open(args.out, "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
