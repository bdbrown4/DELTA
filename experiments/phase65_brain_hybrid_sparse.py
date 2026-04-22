"""Phase 65: Brain Hybrid with Sparse Attention at N=5000

Phase 64 validated that topk=128 sparse attention is LOSSLESS at N=5000
(test MRR=0.2472 vs Phase 63 baseline 0.2471 with 30M subsample) while
eliminating the attention dilution bottleneck. But DELTA still learns on a
fixed topology — it cannot reason about edges beyond the training graph.

Phase 65 activates the Brain architecture: BrainConstructor learns to ADD
structurally informative new edges (Gumbel-sigmoid selection from O(N²) candidates),
then Stage 3 DELTA layers reason over the augmented topology. Combined with
topk=128 sparse attention, the augmented graph stays memory-feasible at N=5000.

Hypothesis: brain_hybrid with topk=128 at N=5000 achieves test MRR > 0.2472
(Phase 64 Cond B) because BrainConstructor infers new structural shortcuts
that the original training edges do not capture, providing richer topological
signal for link prediction.

Secondary: enabling PostAttentionPruner (Cond C) further improves MRR by
adaptively discarding low-utility augmented edges inside each DELTA layer.

Architecture changes relative to Phase 64:
  - BrainEncoder replaces DELTAModel: 1 bootstrap layer + Constructor + 2 delta layers
  - target_density=0.001 at N=5000 → ~25K constructed edges (+16% over 152K original)
  - topk=128 applies to ALL DELTALayer calls inside BrainEncoder (bootstrap + delta)
  - E_adj rebuilt each epoch inside BrainEncoder (augmented graph changes with constructor)
  - Original E_adj (63M) pre-built and passed as cached_edge_adj for Stage 1 bootstrap

Design: 3 conditions (Cond A reused from Phase 64):
  A) DELTA 1L, N=5000, topk=128, test_MRR=0.2472 — Phase 64 Cond B REUSED
  B) brain_hybrid, N=5000, topk=128, router=OFF — BrainConstructor + sparse attention
  C) brain_hybrid, N=5000, topk=128, router=ON  — + PostAttentionPruner in Stage 3

Training:
  - Per-batch encoding: fresh Gumbel noise each batch for constructor exploration
  - cached_edge_adj (original 63M) injected for Stage 1 bootstrap speedup
  - Augmented E_adj built from scratch inside BrainEncoder (no external cache)
  - sparsity_weight=0.01 regularizes constructor toward sparse edge selection
  - Same early stopping: PATIENCE=2, EVAL_EVERY=25, MAX_EPOCHS=150, BS=4096, LR=0.003

Cost estimate (per condition):
  - 3-stage BrainEncoder forward: ~3x Phase 64 per-batch encode (~24s/batch)
  - 37 batches/epoch × 24s = ~888s/epoch (vs ~300s for Phase 64)
  - ~125 epochs (early stopping): ~110,000s (~30hr) per condition
  - Cond B + C total: ~60hr. At $1.89/hr: ~$113

VRAM: 
  - Original E_adj: 63M pairs, pre-built once
  - Augmented E_adj: ~84M pairs (177K edges), built per encode inside BrainEncoder
  - topk=128 keeps attention tensors O(E × 128 × d) — bounded
  - Constructor: [5000, 5000, 64] = 6.4GB for Phase 1 (no_grad). Safe on 98GB.
"""
import sys, os, gc, time
import torch
import torch.nn.functional as F
import numpy as np
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.phase46c_link_prediction import (
    load_lp_data, build_train_graph_tensors,
    evaluate_lp, LinkPredictionModel,
)
from delta.model import DELTAModel
from delta.brain import BrainEncoder
from delta.graph import DeltaGraph

device = 'cuda'
d_node, d_edge = 64, 32

# ── Phase 64 Cond B baseline (REUSED — no re-run needed) ──
PHASE64_BASELINE = {
    'label': 'DELTA_1L_topk128',
    'topk': 128,
    'test_MRR': 0.2472,
    'test_H1':  0.1471,
    'test_H10': 0.4575,
    'peak_val': 0.2500,
    'best_ep':  125,
    'time':     None,  # not tracked (reused)
}

# DistMult Phase 62 baseline
DM_TEST_MRR = 0.2244

# ── Conditions B and C ──
CONDITIONS = {
    'B': {
        'label':          'brain_hybrid+topk128+router_OFF',
        'topk':           128,
        'use_router':     False,
        'target_density': 0.001,
        'bootstrap_layers': 1,
        'delta_layers':   2,
    },
    'C': {
        'label':          'brain_hybrid+topk128+router_ON',
        'topk':           128,
        'use_router':     True,
        'target_density': 0.001,
        'bootstrap_layers': 1,
        'delta_layers':   2,
    },
}

# ── Training config ──
MAX_EPOCHS   = 150
EVAL_EVERY   = 25
PATIENCE     = 2
BS           = 4096
LR           = 0.003
SEED         = 42
SPARSITY_W   = 0.01
LABEL_SMOOTH = 0.1


# ═══════════════════════════════════════════════════════════════════════════
# E_adj builder (original graph only — for bootstrap stage cache)
# ═══════════════════════════════════════════════════════════════════════════

def build_edge_adj(N, E_train, ei):
    """Build full edge adjacency for original training graph.
    
    This is passed as cached_edge_adj to speed up Stage 1 bootstrap.
    BrainEncoder builds the augmented E_adj internally for Stage 3.
    """
    t0 = time.time()
    with torch.no_grad():
        tmp = DeltaGraph(
            node_features=torch.zeros(N, d_node, device=device),
            edge_features=torch.zeros(E_train, d_edge, device=device),
            edge_index=ei.to(device),
        )
        tmp.build_edge_adjacency()
        full_edge_adj = tmp._edge_adj_cache[1]
        del tmp
        torch.cuda.empty_cache()
    return full_edge_adj, full_edge_adj.shape[1], time.time() - t0


# ═══════════════════════════════════════════════════════════════════════════
# Fast vectorized LP evaluation (replaces phase46c evaluate_lp for BrainEncoder)
#
# The original evaluate_lp has a Python for-loop with B×4 GPU-CPU .item()
# calls per batch.  Under 90GB+ GPU memory pressure each sync takes ~50-100ms,
# making a single eval take 60-90 minutes for 8788 val triples.
#
# This version reduces syncs from ~70K to ~6 per eval call by:
#   1. torch.no_grad() throughout
#   2. Vectorised rank computation — ranks = (scores >= target).sum(1)
#   3. Single .cpu() transfer per batch instead of B individual .item() calls
#   4. EVAL_BS=4096 → only 3 batches for 8788 val triples
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_lp_fast(model, triples, edge_index, edge_types,
                     hr_to_tails, rt_to_heads, device_,
                     cached_edge_adj=None, eval_bs=4096):
    """Vectorised filtered MRR / Hits@K — fast replacement for evaluate_lp."""
    model.eval()
    n = triples.shape[1]
    if n == 0:
        return {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0}

    ei = edge_index.to(device_)
    et = edge_types.to(device_)

    with torch.no_grad():
        torch.cuda.empty_cache()
        node_feats = model.encode(ei, et, cached_edge_adj=cached_edge_adj)
        N_e = node_feats.shape[0]

        h_all = triples[0]   # stay on CPU for fast .tolist()
        r_all = triples[1]
        t_all = triples[2]

        all_ranks = []

        for start in range(0, n, eval_bs):
            end = min(start + eval_bs, n)
            # Python ints from CPU tensor — no GPU-CPU sync
            h_cpu = h_all[start:end].tolist()
            r_cpu = r_all[start:end].tolist()
            t_cpu = t_all[start:end].tolist()
            B = len(h_cpu)

            h_dev = torch.tensor(h_cpu, dtype=torch.long, device=device_)
            r_dev = torch.tensor(r_cpu, dtype=torch.long, device=device_)
            t_dev = torch.tensor(t_cpu, dtype=torch.long, device=device_)

            # ── Tail prediction ──────────────────────────────────────────
            scores_t = model.score_all_tails(node_feats, h_dev, r_dev)  # [B, N_e]
            # Build filter indices in Python (CPU-side), then bulk-mask GPU tensor
            f_ii, f_jj = [], []
            for i, (hi, ri, ti) in enumerate(zip(h_cpu, r_cpu, t_cpu)):
                for tt in hr_to_tails.get((hi, ri), set()):
                    if tt != ti:
                        f_ii.append(i); f_jj.append(tt)
            if f_ii:
                scores_t[f_ii, f_jj] = float('-inf')
            # Vectorised rank: count entities scoring >= query entity — 1 sync
            tgt_t = scores_t[torch.arange(B, device=device_), t_dev]   # [B]
            ranks_t = (scores_t >= tgt_t.unsqueeze(1)).sum(dim=1)       # [B] GPU
            all_ranks.extend(ranks_t.cpu().numpy().tolist())            # 1 transfer

            # ── Head prediction ──────────────────────────────────────────
            scores_h = model.score_all_heads(node_feats, r_dev, t_dev)  # [B, N_e]
            f_ii, f_jj = [], []
            for i, (hi, ri, ti) in enumerate(zip(h_cpu, r_cpu, t_cpu)):
                for th in rt_to_heads.get((ri, ti), set()):
                    if th != hi:
                        f_ii.append(i); f_jj.append(th)
            if f_ii:
                scores_h[f_ii, f_jj] = float('-inf')
            tgt_h = scores_h[torch.arange(B, device=device_), h_dev]
            ranks_h = (scores_h >= tgt_h.unsqueeze(1)).sum(dim=1)
            all_ranks.extend(ranks_h.cpu().numpy().tolist())

    ranks = np.maximum(np.array(all_ranks, dtype=np.float64), 1.0)
    return {
        'MRR':    float(np.mean(1.0 / ranks)),
        'Hits@1': float(np.mean(ranks <= 1)),
        'Hits@3': float(np.mean(ranks <= 3)),
        'Hits@10':float(np.mean(ranks <= 10)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Training — brain hybrid per-batch encoding with sparsity loss
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch_brain(model, train_triples, edge_index, edge_types,
                      optimizer, device, batch_size=4096,
                      label_smoothing=LABEL_SMOOTH,
                      sparsity_weight=SPARSITY_W,
                      cached_edge_adj=None):
    """One training epoch for brain_hybrid at N=5000.

    Per-batch encoding: BrainConstructor gets fresh Gumbel noise each batch
    for exploration — important for learning non-trivial edge selections.
    cached_edge_adj (original 63M E_adj) injected for Stage 1 bootstrap;
    BrainEncoder subsamples it to 30M internally to keep Stage 1 peak ~12GB
    while retaining full gradient flow throughout all three stages.
    Augmented E_adj (Stage 3) is built fresh inside BrainEncoder each call.
    """
    model.train()
    n = train_triples.shape[1]
    perm = torch.randperm(n)
    total_loss = 0.0
    total_sparsity = 0.0
    num_batches = 0

    ei = edge_index.to(device)
    et = edge_types.to(device)

    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        h = train_triples[0, idx].to(device)
        r = train_triples[1, idx].to(device)
        t = train_triples[2, idx].to(device)
        B = h.shape[0]
        N = model.num_entities

        # Encode — full gradient flow through Stage 1 (30M bootstrap E_adj),
        # BrainConstructor, and Stage 3 (full augmented E_adj)
        node_feats = model.encode(ei, et, cached_edge_adj=cached_edge_adj)

        # Tail prediction
        scores_t = model.score_all_tails(node_feats, h, r)
        targets_t = torch.zeros(B, N, device=device)
        targets_t[torch.arange(B, device=device), t] = 1.0
        if label_smoothing > 0:
            targets_t = targets_t * (1 - label_smoothing) + label_smoothing / N
        loss_t = F.binary_cross_entropy_with_logits(scores_t, targets_t)

        # Head prediction
        scores_h = model.score_all_heads(node_feats, r, t)
        targets_h = torch.zeros(B, N, device=device)
        targets_h[torch.arange(B, device=device), h] = 1.0
        if label_smoothing > 0:
            targets_h = targets_h * (1 - label_smoothing) + label_smoothing / N
        loss_h = F.binary_cross_entropy_with_logits(scores_h, targets_h)

        lp_loss = (loss_t + loss_h) / 2

        # Sparsity regularization from BrainConstructor
        sp_loss = torch.tensor(0.0, device=device)
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'last_sparsity_loss'):
            sp_loss = model.encoder.last_sparsity_loss

        loss = lp_loss + sparsity_weight * sp_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += lp_loss.item()
        total_sparsity += sp_loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1), total_sparsity / max(num_batches, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Runner — brain_hybrid condition
# ═══════════════════════════════════════════════════════════════════════════

def run_brain_hybrid(data, ei, et, cached_edge_adj, cfg, label=''):
    """Run brain_hybrid at N=5000 with topk sparse attention + early stopping."""
    enc = BrainEncoder(
        d_node=d_node, d_edge=d_edge,
        bootstrap_layers=cfg['bootstrap_layers'],
        delta_layers=cfg['delta_layers'],
        num_heads=4,
        target_density=cfg['target_density'],
        hybrid=True,
        init_temp=1.0,
        topk_edges=cfg['topk'],
        use_router_in_delta=cfg['use_router'],
    )
    # Bootstrap Stage 1 on 30M subsampled E_adj to keep peak memory ~50GB.
    # Full 63M E_adj → ~47GB Stage 1 saved activations + ~21GB Stage 3 ctx = OOM.
    # 30M subsample → ~12GB Stage 1 saved + ~21GB Stage 3 ctx = ~50GB. Fits.
    # Full gradient flow maintained — no quality compromise.
    enc.bootstrap_edge_budget = 30_000_000
    model = LinkPredictionModel(enc, data['num_entities'],
                                data['num_relations'], d_node, d_edge).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    best_val = {'MRR': 0}
    best_state = None
    best_ep = 0
    decline_count = 0
    last_val_mrr = 0.0
    constructed_edges_log = []

    for ep in range(1, MAX_EPOCHS + 1):
        loss, sp_loss = train_epoch_brain(
            model, data['train'], ei, et, opt, device,
            batch_size=BS,
            cached_edge_adj=cached_edge_adj,
        )

        # Log epoch timing and constructor stats
        if ep == 1:
            t1 = time.time() - t0
            n_constructed = model.encoder.last_num_constructed_edges if hasattr(model, 'encoder') else 0
            est_total = t1 * MAX_EPOCHS
            print(f'  [{label}] First epoch: {t1:.1f}s, '
                  f'sp_loss={sp_loss:.4f}, '
                  f'constructed_edges={n_constructed}. '
                  f'Est total ({MAX_EPOCHS}ep): {est_total:.0f}s ({est_total/3600:.1f}hr)')
            sys.stdout.flush()

        if ep % EVAL_EVERY == 0 or ep == MAX_EPOCHS:
            n_constructed = getattr(getattr(model, 'encoder', None),
                                    'last_num_constructed_edges', 0)
            constructed_edges_log.append((ep, n_constructed))

            val = evaluate_lp_fast(model, data['val'], ei, et,
                                   data['hr_to_tails'], data['rt_to_heads'], device,
                                   cached_edge_adj=None)
            elapsed = time.time() - t0
            print(f'  [{label}] Ep {ep:4d}  loss={loss:.4f}  sp={sp_loss:.4f}  '
                  f'MRR={val["MRR"]:.4f}  H@1={val["Hits@1"]:.4f}  '
                  f'H@10={val["Hits@10"]:.4f}  edges={n_constructed}  [{elapsed:.0f}s]')
            sys.stdout.flush()

            if val['MRR'] > best_val['MRR']:
                best_val = val.copy()
                best_val['best_epoch'] = ep
                best_ep = ep
                best_state = copy.deepcopy(model.state_dict())
                decline_count = 0
            else:
                if ep > EVAL_EVERY and val['MRR'] < last_val_mrr:
                    decline_count += 1
                    if decline_count >= PATIENCE:
                        print(f'  [{label}] Early stopping at ep {ep} '
                              f'(val MRR declined {decline_count} evals)')
                        sys.stdout.flush()
                        break
            last_val_mrr = val['MRR']

    # Test on best validation checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate_lp_fast(model, data['test'], ei, et,
                             data['hr_to_tails'], data['rt_to_heads'], device,
                             cached_edge_adj=None)
    elapsed = time.time() - t0
    del model, opt, best_state
    gc.collect(); torch.cuda.empty_cache()

    return {
        'val_best':            best_val,
        'test_at_best_val':    test,
        'time':                elapsed,
        'best_ep':             best_ep,
        'constructed_log':     constructed_edges_log,
        # flat fields for summary table
        'test_MRR':   test['MRR'],
        'test_H1':    test['Hits@1'],
        'test_H10':   test['Hits@10'],
        'peak_val':   best_val.get('MRR', 0),
        'topk':       cfg['topk'],
        'router':     cfg['use_router'],
        'density':    cfg['target_density'],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

print()
print('=' * 72)
print('  PHASE 65: Brain Hybrid with Sparse Attention at N=5000')
print('=' * 72)
sys.stdout.flush()

total_t0 = time.time()
results = {}

# ── Load data ──
torch.manual_seed(SEED); np.random.seed(SEED)
data = load_lp_data('fb15k-237', max_entities=5000)
N = data['num_entities']
E_train = data['train'].shape[1]
ei, et = build_train_graph_tensors(data['train'])
print(f'\n  Data: {N} entities, {data["num_relations"]} relations, '
      f'{E_train} train edges')
sys.stdout.flush()

# ── Build original edge adjacency (for Stage 1 bootstrap cache) ──
print('\n  Building original edge adjacency for bootstrap stage cache...')
sys.stdout.flush()
orig_edge_adj, n_orig_pairs, build_time = build_edge_adj(N, E_train, ei)
print(f'  Original E_adj: {n_orig_pairs:,} pairs ({n_orig_pairs/1e6:.1f}M), '
      f'built in {build_time:.1f}s')
sys.stdout.flush()

# ── Run conditions ──
for cond_name, cfg in CONDITIONS.items():
    print(f'\n{"─" * 68}')
    print(f'  Condition {cond_name}: {cfg["label"]}')
    print(f'  Config: topk={cfg["topk"]}, router={cfg["use_router"]}, '
          f'density={cfg["target_density"]}, '
          f'{cfg["bootstrap_layers"]}B+{cfg["delta_layers"]}D layers')
    print(f'{"─" * 68}')
    sys.stdout.flush()

    torch.manual_seed(SEED); np.random.seed(SEED)
    result = run_brain_hybrid(data, ei, et, orig_edge_adj, cfg,
                              label=f'Cond{cond_name}')
    results[cond_name] = result

    print(f'\n  ✓ Cond {cond_name}: test_MRR={result["test_MRR"]:.4f}  '
          f'test_H@1={result["test_H1"]:.4f}  '
          f'test_H@10={result["test_H10"]:.4f}  '
          f'best_ep={result["best_ep"]}  '
          f'time={result["time"]:.0f}s ({result["time"]/3600:.1f}hr)')
    sys.stdout.flush()

# ── Summary ──
total_time = time.time() - total_t0

print(f'\n{"=" * 72}')
print(f'  PHASE 65 SUMMARY')
print(f'{"=" * 72}')
print(f'\n  Total time: {total_time:.0f}s ({total_time/3600:.1f}hr)')
print(f'\n  {"Cond":<6} {"Model":<30} {"topk":<6} {"router":<8} '
      f'{"peak_val":<10} {"best_ep":<8} {"test_MRR":<10} '
      f'{"test_H@1":<10} {"test_H@10":<10} {"Δvs_P64A":<10} {"Time":<10}')
print(f'  {"-"*6} {"-"*30} {"-"*6} {"-"*8} '
      f'{"-"*10} {"-"*8} {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')

# Phase 64 Cond A (reused)
bline = PHASE64_BASELINE
print(f'  {"A(P64)":<6} {"DELTA_1L+topk128":<30} '
      f'{bline["topk"]:<6} {"—":<8} '
      f'{bline["peak_val"]:<10.4f} {bline["best_ep"]:<8} '
      f'{bline["test_MRR"]:<10.4f} {bline["test_H1"]:<10.4f} '
      f'{bline["test_H10"]:<10.4f} '
      f'{"baseline":<10} {"—":<10}')

for cond_name, cfg in CONDITIONS.items():
    r = results.get(cond_name, {})
    if 'test_MRR' in r:
        delta_mrr = r['test_MRR'] - bline['test_MRR']
        model_label = f'brain_hybrid+{"router" if cfg["use_router"] else "no_router"}'
        print(f'  {cond_name:<6} {model_label:<30} '
              f'{r["topk"]:<6} {str(r["router"]):<8} '
              f'{r["peak_val"]:<10.4f} {r["best_ep"]:<8} '
              f'{r["test_MRR"]:<10.4f} {r["test_H1"]:<10.4f} '
              f'{r["test_H10"]:<10.4f} '
              f'{delta_mrr:<+10.4f} {r["time"]:<10.0f}')
    else:
        print(f'  {cond_name:<6} {cfg["label"]:<30} {"—":<6} {"—":<8} '
              f'{"—":<10} {"—":<8} {"FAILED":<10}')

print(f'\n  vs. Phase 64 Baseline (Cond A, DELTA 1L topk=128, test_MRR=0.2472):')
for cond_name in CONDITIONS:
    r = results.get(cond_name, {})
    if 'test_MRR' in r:
        delta = r['test_MRR'] - bline['test_MRR']
        verdict = 'IMPROVE' if delta > 0.001 else ('TIE' if abs(delta) <= 0.001 else 'REGRESS')
        print(f'    {cond_name}: Δ test_MRR={delta:+.4f}  ({verdict})')

print(f'\n  Phase 65 complete.')
sys.stdout.flush()
