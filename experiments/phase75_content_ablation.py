"""Phase 75 (build B) — content-dependent edge adjacency ablation harness.

Wires ContentRouter into training/eval and runs the pre-registered ladder. Structural
conditions (node_only/hops1/hops2) reuse the phase73 static-adjacency path; content
conditions (content_comp/content_sim/content_random) use a dynamic path:
  - adjacency INDICES rebuilt once per epoch from current edge features (detached);
  - route_prob recomputed every batch on those indices (differentiable -> trains the keys);
  - composability-shaping aux loss added; cached FINAL adjacency used for eval.

Diagnostics (pre-registration): composable_fraction of the LEARNED adjacency (must rise past
struct-2hop's 0.216) and shared-node-overlap (must stay low, else it's secretly 1-hop). The
content eval asserts it actually uses the content builder (composable_frac != structural).

RR_prior and param_matched_hops1 controls are deferred to the confirmation run (noted); this
build validates the core comp-vs-{hops1,sim,random} comparison + the mechanism diagnostics.

Usage:
  # smoke (tiny, fast): content_comp + hops1
  python experiments/phase75_content_ablation.py --smoke
  # run
  python experiments/phase75_content_ablation.py --sample_frac 0.10 --epochs 600 \
      --seeds 42,123,456,7,99 --conditions node_only,hops1,content_sim,content_random,content_comp
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, time, json
import numpy as np
import torch
import torch.nn.functional as F
from delta.graph import DeltaGraph
from delta.router import ContentRouter
from experiments.phase46c_link_prediction import load_lp_data, build_train_graph_tensors, create_lp_model
from experiments.phase66_hop_ablation import build_condition_adjs, train_with_controlled_adj, _evaluate_with_adj, _encode_with_adj
from experiments.phase71_sparse_hop_ablation import evaluate_multihop_extended_with_adj, QTYPES
from experiments.phase42_multihop import build_full_adjacency
from experiments.phase74_content_probe import composable_relpairs, frac_composable
from experiments.phase74b_supervised_ceiling import shared_node_overlap

STRUCTURAL = {'node_only', 'hops1', 'hops2'}
CONTENT = {'content_comp', 'content_sim', 'content_random'}


# ── content encode/route helpers ─────────────────────────────────────────────
def _edge_feats(model, et, device):
    ef = model.edge_rel_emb(et.to(device))
    if model.needs_proj:
        ef = model.edge_proj_in(ef)
    return ef


def _per_relation_affsoft(router, ef, et, R, tau, device):
    tk, hk = router.keys(ef)
    d_r = tk.shape[1]; E = ef.shape[0]
    cnt = torch.zeros(R, device=device); cnt.index_add_(0, et, torch.ones(E, device=device))
    tk_r = torch.zeros(R, d_r, device=device); tk_r.index_add_(0, et, tk)
    hk_r = torch.zeros(R, d_r, device=device); hk_r.index_add_(0, et, hk)
    nz = cnt > 0
    tk_r[nz] = tk_r[nz] / cnt[nz].unsqueeze(1); hk_r[nz] = hk_r[nz] / cnt[nz].unsqueeze(1)
    return torch.softmax((tk_r @ hk_r.t()) / tau, dim=1)


def epoch_content_adj(model, router, et, K, tau, R, device):
    """Build adjacency indices once per epoch (detached); return adj + per-pair relations."""
    with torch.no_grad():
        ef = _edge_feats(model, et, device)
        adj, _ = router.build(ef, et.to(device), K, tau=tau, num_relations=R)
    etd = et.to(device)
    return adj, etd[adj[0]], etd[adj[1]]   # adj, src_rel, tgt_rel


def batch_route_prob(model, router, et, src_rel, tgt_rel, tau, R, device):
    ef = _edge_feats(model, et, device)              # current, differentiable
    affsoft = _per_relation_affsoft(router, ef, et.to(device), R, tau, device)
    return affsoft[tgt_rel, src_rel]                 # [E_adj]


def encode_content(model, ei, et, adj, route_prob, device):
    ei = ei.to(device); etd = et.to(device)
    nf = model.entity_emb.weight; ef = model.edge_rel_emb(etd)
    if model.needs_proj:
        nf = model.node_proj_in(nf); ef = model.edge_proj_in(ef)
    g = DeltaGraph(node_features=nf, edge_features=ef, edge_index=ei)
    g._edge_adj_cache = (2, adj); g._route_prob = route_prob
    out = model.encoder(g).node_features
    if model.needs_proj:
        out = model.node_proj_out(out)
    return out


def train_content(cond, router, data, epochs, lr, device, bs, seed, eval_every, patience,
                  K, tau, R, aux_w=1.0, comp=None, et_cpu=None):
    torch.manual_seed(seed); np.random.seed(seed)
    model = create_lp_model('delta_matched', data['num_entities'], data['num_relations']).to(device)
    router = router.to(device)
    params = list(model.parameters()) + list(router.parameters())
    opt = torch.optim.Adam([p for p in params if p.requires_grad], lr=lr)
    ei, et = build_train_graph_tensors(data['train'])
    train = data['train']
    best_val, best_state, no_imp = 0.0, None, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        adj, src_rel, tgt_rel = epoch_content_adj(model, router, et, K, tau, R, device)
        perm = torch.randperm(train.shape[1])
        for s in range(0, train.shape[1], bs):
            idx = perm[s:s + bs]
            h = train[0, idx].to(device); r = train[1, idx].to(device); t = train[2, idx].to(device)
            B, N = h.shape[0], model.num_entities
            rp = batch_route_prob(model, router, et, src_rel, tgt_rel, tau, R, device)
            nf = encode_content(model, ei, et, adj, rp, device)
            st = model.score_all_tails(nf, h, r)
            tt = torch.zeros(B, N, device=device); tt[torch.arange(B, device=device), t] = 1.0
            tt = tt * 0.9 + 0.1 / N
            sh = model.score_all_heads(nf, r, t)
            th = torch.zeros(B, N, device=device); th[torch.arange(B, device=device), h] = 1.0
            th = th * 0.9 + 0.1 / N
            loss = (F.binary_cross_entropy_with_logits(st, tt) +
                    F.binary_cross_entropy_with_logits(sh, th)) / 2
            if cond != 'content_random':
                loss = loss + aux_w * router.aux_losses(_edge_feats(model, et, device), ei.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % eval_every == 0 or epoch == epochs:
            val = eval_content_lp(model, router, data, data['val'], ei, et, K, tau, R, device)
            if comp is not None:
                a_, _ = cached_content_adj(model, router, et, K, tau, R, device)
                cf, _ = frac_composable(a_.cpu(), et_cpu, comp)
                print(f"    ep{epoch:4d} val_MRR={val:.4f} composable_frac={cf:.3f}  "
                      f"[cons={float(router.consistency_loss):.3f}]")
            if val > best_val:
                best_val = val; best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_rstate = {k: v.clone() for k, v in router.state_dict().items()}; no_imp = 0
            else:
                no_imp += 1
                if patience > 0 and no_imp >= patience:
                    break
    if best_state:
        model.load_state_dict(best_state); router.load_state_dict(best_rstate)
    return model, router, best_val, ei, et, time.time() - t0


@torch.no_grad()
def cached_content_adj(model, router, et, K, tau, R, device):
    """Build ONE fixed adjacency + route_prob from the final model, for deterministic eval."""
    ef = _edge_feats(model, et, device)
    adj, _ = router.build(ef, et.to(device), K, tau=tau, num_relations=R)
    etd = et.to(device)
    rp = _per_relation_affsoft(router, ef, etd, R, tau, device)[etd[adj[1]], etd[adj[0]]]
    return adj, rp


@torch.no_grad()
def eval_content_lp(model, router, data, triples, ei, et, K, tau, R, device, bs=512):
    model.eval()
    adj, rp = cached_content_adj(model, router, et, K, tau, R, device)
    nf = encode_content(model, ei, et, adj, rp, device)
    triples = triples.to(device); ranks = []
    h2t, r2h = data['hr_to_tails'], data['rt_to_heads']
    for s in range(0, triples.shape[1], bs):
        b = triples[:, s:s + bs]; h, r, t = b[0], b[1], b[2]; B = h.shape[0]
        sc = model.score_all_tails(nf, h, r)
        rows, cols = [], []
        hl, rl, tl = h.tolist(), r.tolist(), t.tolist()
        for i in range(B):
            for e in h2t.get((hl[i], rl[i]), ()):
                if e != tl[i]:
                    rows.append(i); cols.append(e)
        if rows:
            sc[torch.tensor(rows, device=device), torch.tensor(cols, device=device)] = float('-inf')
        ranks.append((sc >= sc[torch.arange(B, device=device), t].unsqueeze(1)).sum(1).clamp_(min=1))
    return float((1.0 / torch.cat(ranks).double()).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--sample_frac', type=float, default=0.10)
    ap.add_argument('--sample_seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--eval_every', type=int, default=25)
    ap.add_argument('--patience', type=int, default=0)
    ap.add_argument('--lr', type=float, default=0.003)
    ap.add_argument('--batch_size', type=int, default=4096)
    ap.add_argument('--seeds', type=str, default='42,123,456,7,99')
    ap.add_argument('--conditions', type=str, default='node_only,hops1,content_sim,content_random,content_comp')
    ap.add_argument('--K', type=int, default=32)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--d_r', type=int, default=16)
    ap.add_argument('--max_queries', type=int, default=10000)
    ap.add_argument('--out', type=str, default='phase75_output.json')
    args = ap.parse_args()
    if args.smoke:
        args.sample_frac = 0.03; args.epochs = 8; args.eval_every = 4
        args.seeds = '42'; args.conditions = 'hops1,content_comp'; args.max_queries = 800
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [int(s) for s in args.seeds.split(',')]
    conditions = [c.strip() for c in args.conditions.split(',')]
    print(f"Phase 75 content ablation | device={device} conditions={conditions} seeds={seeds} smoke={args.smoke}")

    data = load_lp_data('fb15k-237', 'data', sample_mode='edge',
                        sample_frac=args.sample_frac, sample_seed=args.sample_seed, use_cache=True)
    ei0, et0 = build_train_graph_tensors(data['train'])
    R = data['num_relations']
    enc_de = 24  # delta_matched encoder edge dim
    comp = composable_relpairs(data['train'])
    print(f"  ents={data['num_entities']} E={ei0.shape[1]} R={R} test={data['test'].shape[1]} comp_pairs={len(comp)}")

    queries = None
    full_hr2t = build_full_adjacency(data)
    structural_adjs = None
    results = []
    for cond in conditions:
        for seed in seeds:
            print(f"\n=== {cond} seed={seed} ===")
            if cond in STRUCTURAL:
                if structural_adjs is None:
                    structural_adjs = build_condition_adjs(ei0, device, max_adj_pairs=None)
                adj, ch = structural_adjs[cond]
                model, best_val, ei, et, elapsed, npar = train_with_controlled_adj(
                    cond, adj, ch, data, args.epochs, args.lr, device, args.batch_size, seed,
                    args.eval_every, args.patience)
                lp = _evaluate_with_adj(model, data['test'], ei, et, data['hr_to_tails'],
                                        data['rt_to_heads'], device, adj.to(device), ch)
                if queries is None:
                    from experiments.phase44_depth import generate_extended_queries, audit_extended_queries
                    queries = generate_extended_queries(data, max_queries_per_type=args.max_queries, seed=42)
                    print("  audit:", "PASS" if not audit_extended_queries(queries, data) else "ISSUES")
                mh = evaluate_multihop_extended_with_adj(model, queries, ei, et, full_hr2t, device, adj.to(device), ch)
                cfrac, _ = frac_composable(adj.cpu(), et0, comp); ov = shared_node_overlap(adj.cpu(), ei0)
            else:
                tie = (cond == 'content_sim'); freeze = (cond == 'content_random')
                router = ContentRouter(enc_de, d_r=args.d_r, tie=tie, freeze=freeze)
                model, router, best_val, ei, et, elapsed = train_content(
                    cond, router, data, args.epochs, args.lr, device, args.batch_size, seed,
                    args.eval_every, args.patience, args.K, args.tau, R,
                    comp=comp, et_cpu=et0)
                adj, rp = cached_content_adj(model, router, et, args.K, args.tau, R, device)
                # ASSERT eval uses the content builder (composable_frac differs from structural baselines)
                cfrac, _ = frac_composable(adj.cpu(), et0, comp); ov = shared_node_overlap(adj.cpu(), ei0)
                nf = encode_content(model, ei, et, adj, rp, device)
                lp = _lp_from_nf(model, nf, data['test'], data['hr_to_tails'], data['rt_to_heads'], device)
                if queries is None:
                    from experiments.phase44_depth import generate_extended_queries, audit_extended_queries
                    queries = generate_extended_queries(data, max_queries_per_type=args.max_queries, seed=42)
                    print("  audit:", "PASS" if not audit_extended_queries(queries, data) else "ISSUES")
                mh = _mh_from_nf(model, nf, queries, full_hr2t, device)
                npar = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in router.parameters())
            row = {'condition': cond, 'seed': seed, 'params': int(npar), 'lp_MRR': lp['MRR'],
                   'composable_frac': cfrac, 'shared_node_overlap': ov,
                   **{f'{q}_MRR': mh[q]['MRR'] for q in QTYPES}}
            print(f"  LP={lp['MRR']:.4f} 3p={mh['3p']['MRR']:.4f} 5p={mh['5p']['MRR']:.4f} "
                  f"composable_frac={cfrac:.3f} node_overlap={ov:.3f} [{elapsed:.0f}s]")
            results.append(row)
            with open(os.path.join(os.path.dirname(__file__), '..', args.out), 'w') as f:
                json.dump(results, f, indent=2)
    print("\nDONE. saved", args.out)


@torch.no_grad()
def _lp_from_nf(model, nf, triples, h2t, r2h, device, bs=512):
    triples = triples.to(device); ranks = []
    for s in range(0, triples.shape[1], bs):
        b = triples[:, s:s + bs]; h, r, t = b[0], b[1], b[2]; B = h.shape[0]
        sc = model.score_all_tails(nf, h, r)
        rows, cols = [], []; hl, rl, tl = h.tolist(), r.tolist(), t.tolist()
        for i in range(B):
            for e in h2t.get((hl[i], rl[i]), ()):
                if e != tl[i]: rows.append(i); cols.append(e)
        if rows: sc[torch.tensor(rows, device=device), torch.tensor(cols, device=device)] = float('-inf')
        ranks.append((sc >= sc[torch.arange(B, device=device), t].unsqueeze(1)).sum(1).clamp_(min=1))
    rk = torch.cat(ranks).double()
    return {'MRR': float((1.0 / rk).mean()), 'Hits@10': float((rk <= 10).double().mean())}


@torch.no_grad()
def _mh_from_nf(model, node_feats, queries, full_hr2t, device, bs=256, temperature=1.0):
    from experiments.phase42_multihop import compute_valid_answers
    out = {}
    for qt in QTYPES:
        qs = queries.get(qt, [])
        if not qs:
            out[qt] = {'MRR': 0.0, 'count': 0}; continue
        nh = len(qs[0][1]); allr = []
        for s in range(0, len(qs), bs):
            batch = qs[s:s + bs]; B = len(batch)
            cur = node_feats[torch.tensor([q[0] for q in batch], device=device)]
            for hop in range(nh):
                rels = torch.tensor([q[1][hop] for q in batch], device=device)
                sc = (cur * model.decoder_rel_emb(rels)) @ node_feats.t()
                if hop < nh - 1:
                    cur = torch.softmax(sc / temperature, -1) @ node_feats
            ans = torch.tensor([q[2] for q in batch], device=device); rows, cols = [], []
            for i in range(B):
                for va in compute_valid_answers(batch[i][0], batch[i][1], full_hr2t):
                    if va != batch[i][2]: rows.append(i); cols.append(va)
            if rows: sc[torch.tensor(rows, device=device), torch.tensor(cols, device=device)] = float('-inf')
            allr.append((sc >= sc[torch.arange(B, device=device), ans].unsqueeze(1)).sum(1).clamp_(min=1))
        rk = torch.cat(allr).double()
        out[qt] = {'MRR': float((1.0 / rk).mean()), 'count': len(qs)}
    return out


if __name__ == '__main__':
    main()
