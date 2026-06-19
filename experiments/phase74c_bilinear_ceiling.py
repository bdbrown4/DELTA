"""Phase 74c — BILINEAR (rank-d) supervised ceiling for composability routing.

Phase 74b tested HARD-BUCKET routing (tail_bucket==head_bucket), which cannot express
many-to-many composability (each relation gets ONE tail bucket -> can only connect to one
head-bucket group). The real soft mechanism scores edge pairs by a bilinear affinity
tail_key(r_t) . head_key(r_s) and takes top-K — a rank-d factorization that CAN express
many-to-many. This is the correct, more-expressive upper bound; test it before any verdict.

Also report the ORACLE ceiling (select only composable-relation sources) to bound headroom.

Relation-level computation (routing is relation-only, the circularity-free variant), so it's
exact and fast: affinity depends only on (r_t, r_s); fill each target's K source-edge slots
by source relations ranked by affinity, weighted by per-relation edge counts.

GATE: if the bilinear ceiling (best rank-d fit) still can't clear struct-2hop's 0.216 with
real headroom, the composability signal is not low-rank-separable -> refute the lever.

Usage: python experiments/phase74c_bilinear_ceiling.py --sample_frac 0.10 --ds 32,64 --K 32
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
from experiments.phase46c_link_prediction import load_lp_data, build_train_graph_tensors
from experiments.phase74_content_probe import composable_relpairs


def frac_from_relrank(score_rt, relcount, comp_row, K):
    """Given a target relation's affinity to all source relations (score_rt[R]) and
    per-relation edge counts, greedily fill K source-edge slots by descending affinity;
    return (composable_slots, total_slots). comp_row[rs]=1 if (rt,rs) composable."""
    order = torch.argsort(score_rt, descending=True)
    filled = 0; comp_sel = 0
    for rs in order.tolist():
        c = relcount[rs].item()
        if c == 0:
            continue
        take = min(c, K - filled)
        filled += take
        if comp_row[rs]:
            comp_sel += take
        if filled >= K:
            break
    return comp_sel, filled


def weighted_frac(score_mat, relcount, Y, K, relset):
    """Mean composable-fraction over target EDGES (weight target relation by its edge count)."""
    R = relcount.shape[0]
    tot, comp = 0, 0
    for rt in relset:
        w = relcount[rt].item()
        if w == 0:
            continue
        cs, fl = frac_from_relrank(score_mat[rt], relcount, Y[rt], K)
        tot += w * fl; comp += w * cs
    return comp / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample_frac', type=float, default=0.10)
    ap.add_argument('--sample_seed', type=int, default=42)
    ap.add_argument('--ds', type=str, default='32,64')
    ap.add_argument('--K', type=int, default=32)
    ap.add_argument('--steps', type=int, default=800)
    args = ap.parse_args()

    print("Phase 74c — BILINEAR (rank-d) supervised ceiling for composability routing")
    data = load_lp_data('fb15k-237', 'data', sample_mode='edge',
                        sample_frac=args.sample_frac, sample_seed=args.sample_seed, use_cache=True)
    ei, et = build_train_graph_tensors(data['train'])
    R = data['num_relations']; E = ei.shape[1]
    comp = composable_relpairs(data['train'])
    Y = torch.zeros(R, R)
    for i, j in comp:
        Y[i, j] = 1.0
    relcount = torch.bincount(et, minlength=R)
    relset = [r for r in range(R) if relcount[r] > 0]
    print(f"  E={E} R={R} composable_pairs={len(comp)} (density {Y.mean():.4f})")

    # ORACLE ceiling: prefer composable source relations (score = composability itself)
    oracle = weighted_frac(Y.clone(), relcount, Y, args.K, relset)
    # struct-2hop bar reported by 74b for this config = 0.216 (relation-only composable frac)
    print(f"  ORACLE ceiling (select composable sources): composable_frac={oracle:.3f}")
    print(f"  BAR struct-2hop=0.216, hard-bucket(M=256)=0.243\n")

    print(f"  BILINEAR rank-d supervised ceiling:")
    print(f"  {'d':>5} {'comp_frac':>10} {'vs 2hop':>9} {'vs bucket':>10} {'AUC-ish':>8}")
    posw = torch.tensor([(R * R - Y.sum()) / Y.sum().clamp(min=1)])
    best = 0.0
    for d in [int(x) for x in args.ds.split(',')]:
        torch.manual_seed(0)
        hk = torch.randn(R, d, requires_grad=True); tk = torch.randn(R, d, requires_grad=True)
        opt = torch.optim.Adam([hk, tk], lr=0.05)
        lossfn = torch.nn.BCEWithLogitsLoss(pos_weight=posw)
        for _ in range(args.steps):
            aff = tk @ hk.t()
            loss = lossfn(aff, Y)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            aff = (tk @ hk.t())
            # crude separability: mean aff on composable vs non-composable
            pos_m = aff[Y > 0].mean(); neg_m = aff[Y == 0].mean()
            fc = weighted_frac(aff, relcount, Y, args.K, relset)
        best = max(best, fc)
        print(f"  {d:>5} {fc:>10.3f} {fc-0.216:>+9.3f} {fc-0.243:>+10.3f} {float(pos_m-neg_m):>8.2f}")

    print(f"\n  VERDICT: best bilinear ceiling composable_frac = {best:.3f}  (oracle={oracle:.3f}, 2hop=0.216)")
    if best > 0.216 + 0.20:
        print("  -> PROCEED: bilinear routing CAN separate composability (large headroom). Build the")
        print("     full hardened pipeline with SOFT bilinear affinity (not hard buckets) + all controls.")
    elif best > 0.216 + 0.08:
        print("  -> MARGINAL: some separability; build only with eyes open to a tight/possible-null result.")
    else:
        print("  -> REFUTE: even the best rank-d bilinear fit barely beats struct-2hop. FB15k-237")
        print("     composability is NOT low-rank separable; no efficient factorized router captures it.")


if __name__ == '__main__':
    main()
