"""Phase 77 — PEC-pf: Penultimate-Frontier path-composition Edge-readout (the JIT pivot).

Query-time edge composition: the fair test the AOT eval (soft-traversal over a FROZEN node table)
structurally could not give the edge-to-edge thesis. For a query (anchor, [r1..rK]) we traverse hops
1..K-1 over the TRAIN graph — at each hop letting the path-state at a node compose with the typed
edges leaving it via DELTA's edge-to-edge operator (the shared `EdgeAttention.compose_scores`) — to
build a penultimate per-entity path-state z_{K-1}; then the final relation r_K is applied as an
ALL-N readout (every entity gets a finite score; gold rankable at every depth, unlike full-chain
traversal whose gold-reachability is only 0/3.3/9.9/17.7/21.3% for 1p-5p).

Scope choices for this first fair test (see docs/phase_77.md):
  * FROZEN encoder: all JIT arms train their composer on top of the SAME frozen nf [N,48] / ef0 [E,24]
    recovered from a pre-trained checkpoint -> the "better embeddings" confound is impossible by
    construction. (End-to-end co-training is a follow-up if signal appears.)
  * Pooled-DistMult all-N readout: composition lives in the traversal-built z_{K-1}; the readout pools
    it (r_K-conditioned) and scores all N via q @ nf.t().
  * Four arms share EVERYTHING except the traversal mechanism (asserted by tests):
      pec       — instance edge-key compose (compose_scores) + instance messages W_out(W_v(ej))
      capacity  — SAME but {W_q,W_k,W_v,W_out,W_ctx,log_temp} FROZEN at random init (trained-vs-random)
      rrprior   — uniform attention + RELATION-TYPE-only messages W_out(msg_emb(r)) (instance-vs-type)
      static    — endpoint-only attention W_ctx(cat(nf_src,nf_tgt)), instance messages (isolates Q.K)

Correctness discipline (traps the red-team flagged): compose_scores reads NOTHING off any graph
object (no _route_prob); the DP gathers typed edges fresh per hop (never build_edge_adjacency, so the
stale _edge_adj_cache cannot bite); train vs full hr2t are distinct, leak-asserted in tests.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from delta.graph import DeltaGraph
from delta.attention import EdgeAttention

MODES = ('pec', 'capacity', 'rrprior', 'static')
# Phase-78 lesion arms (all behave like pec on the operator; they differ only in WHICH edge content the
# operator sees, via a within-relation-type permutation set by set_edge_perm):
#   shuffle       — permute the per-edge ej (post endpoint-injection); W_ctx endpoints stay REAL.
#                   Tests the edge's own ej representation (message+key) only.
#   shuffle_full  — permute ej AND the endpoints fed to W_ctx (full edge-identity swap, coherent).
#                   The coherence-clean control: pec~shuffle_full is a clean refutation of edge-instance use.
#   ef0_shuffle   — permute ONLY the frozen ef0 BEFORE injection (real endpoints retained). ef0 is
#                   relation-type dominated (GATE C within-relation var ~0) -> near-noop SANITY control.
LESION_MODES = ('shuffle', 'shuffle_full', 'ef0_shuffle')


@torch.no_grad()
def encode_with_edges(model, edge_index, edge_types, device):
    """Run the (frozen) encoder once and return BOTH the internal node features nf [N, d_enc_node]
    and the internal edge features ef0 [E, d_enc_edge] — the encoder computes both; AOT discards the
    edges at `nf_out = encoded.node_features`. We keep them. Returns encoder-INTERNAL widths (48/24
    for delta_matched), which is what the composer operates in.
    """
    model.eval()
    nf = model.entity_emb.weight
    ef = model.edge_rel_emb(edge_types.to(device))
    if getattr(model, 'needs_proj', False):
        nf = model.node_proj_in(nf)
        ef = model.edge_proj_in(ef)
    graph = DeltaGraph(node_features=nf, edge_features=ef, edge_index=edge_index.to(device))
    encoded = model.encoder(graph)
    return encoded.node_features.detach(), encoded.edge_features.detach()


def build_hr2t(triples):
    """(h, r) -> set(tails) over the GIVEN triples only. Pass data['train'] for the train-only
    adjacency (distinct from build_full_adjacency, which concatenates train+val+test)."""
    from collections import defaultdict
    hr2t = defaultdict(set)
    for i in range(triples.shape[1]):
        hr2t[(triples[0, i].item(), triples[1, i].item())].add(triples[2, i].item())
    return dict(hr2t)


def build_csr(edge_index, edge_types, num_relations, device):
    """CSR-style index for vectorized (src_node, relation) -> edges lookup. Sorts edges by the
    combined key src*R + rel so a batch of (node, rel) queries gathers all matching edges via
    searchsorted ranges. Returns dict consumed by the batched DP."""
    ei = edge_index.to(device); et = edge_types.to(device)
    key = ei[0] * num_relations + et
    order = torch.argsort(key)
    return {'key_sorted': key[order].contiguous(), 'eid_sorted': order.contiguous(),
            'tgt_sorted': ei[1][order].contiguous(), 'R': num_relations}


def _gather_edges(fnode, frel, csr):
    """For frontier entries with node fnode[i] and relation frel[i], gather all matching train edges.
    Returns (seg, eid, tgt): seg[m] = frontier-entry index that matched edge m came from."""
    R = csr['R']; ks = csr['key_sorted']
    qk = fnode * R + frel
    lo = torch.searchsorted(ks, qk, right=False)
    hi = torch.searchsorted(ks, qk, right=True)
    counts = hi - lo
    total = int(counts.sum().item())
    if total == 0:
        z = torch.zeros(0, dtype=torch.long, device=fnode.device)
        return z, z, z
    seg = torch.repeat_interleave(torch.arange(fnode.shape[0], device=fnode.device), counts)
    starts = torch.cumsum(counts, 0) - counts
    within = torch.arange(total, device=fnode.device) - torch.repeat_interleave(starts, counts)
    gpos = lo[seg] + within
    return seg, csr['eid_sorted'][gpos], csr['tgt_sorted'][gpos]


def build_R2E(edge_index, edge_types, num_relations, device):
    """R2E[r] = (src_nodes, edge_ids, tgt_nodes) for all train edges of relation r (sorted by src is
    not required). Used to gather, per hop, the typed edges leaving the current frontier."""
    ei = edge_index.to(device); et = edge_types.to(device)
    R2E = {}
    for r in range(num_relations):
        m = (et == r)
        if not bool(m.any()):
            continue
        eid = m.nonzero(as_tuple=False).squeeze(1)
        R2E[r] = (ei[0, eid], eid, ei[1, eid])
    return R2E


def _scatter_softmax(scores, index, N):
    """Stable softmax of scores [M,H] within groups given by index [M] (group ids in [0,N))."""
    max_vals = torch.full((N, scores.shape[1]), -1e9, device=scores.device)
    idx = index.unsqueeze(-1).expand_as(scores)
    max_vals.scatter_reduce_(0, idx, scores, reduce='amax', include_self=False)
    scores = scores - max_vals.gather(0, idx)
    exp = torch.exp(scores)
    denom = torch.zeros(N, scores.shape[1], device=scores.device)
    denom.scatter_add_(0, idx, exp)
    return exp / (denom.gather(0, idx) + 1e-10)


class PathComposerPF(nn.Module):
    def __init__(self, num_relations, d_node=48, d_edge=24, num_heads=4, mode='pec'):
        super().__init__()
        assert mode in MODES or mode in LESION_MODES, mode   # LESION_MODES = Phase-78 within-type edge-content controls
        assert d_edge % num_heads == 0
        self.mode = mode
        self.d_node = d_node
        self.d_edge = d_edge
        self.num_heads = num_heads
        self.d_head = d_edge // num_heads
        self.num_relations = num_relations
        # Phase-78 controls: a fixed within-relation-type permutation of edge indices, set via
        # set_edge_perm(). None for every arm except LESION_MODES. edge_perm permutes ej (shuffle/
        # shuffle_full) or ef0 (ef0_shuffle); ctx_src/ctx_tgt carry the permuted edge's endpoints for
        # the W_ctx attention context (shuffle_full only — full edge-identity swap).
        self.register_buffer('edge_perm', None, persistent=False)
        self.register_buffer('ctx_src', None, persistent=False)
        self.register_buffer('ctx_tgt', None, persistent=False)

        # instance-signal injection: per-edge DP state from frozen ef0 + endpoint entity features
        self.W_ein = nn.Linear(d_edge + 2 * d_node, d_edge)
        self.ein_norm = nn.LayerNorm(d_edge)
        # seed: anchor path-state from its node features
        self.W_seed = nn.Linear(d_node, d_edge)
        self.seed_norm = nn.LayerNorm(d_edge)

        # the shared edge-to-edge operator (same submodules EdgeAttention uses)
        self.W_q = nn.Linear(d_edge, d_edge)
        self.W_k = nn.Linear(d_edge, d_edge)
        self.W_v = nn.Linear(d_edge, d_edge)
        self.W_out = nn.Linear(d_edge, d_edge)
        self.W_ctx = nn.Linear(2 * d_node, num_heads)
        self._log_temp = nn.Parameter(torch.zeros(num_heads))
        self.hop_norm = nn.LayerNorm(d_edge)

        # relation-type-only message table (rrprior arm)
        self.msg_emb = nn.Embedding(num_relations, d_edge)

        # readout: r_K-conditioned frontier pool -> DistMult against all entities
        self.rel_emb = nn.Embedding(num_relations, d_edge)        # query-relation table (NOT decoder_rel_emb)
        self.W_poolattn = nn.Linear(2 * d_edge, 1)               # frontier attention conditioned on r_K
        self.W_read = nn.Linear(d_edge + d_edge, d_node)          # -> entity space for DistMult
        self.read_bias = nn.Parameter(torch.zeros(1))

        if mode == 'capacity':
            # freeze the operator at random init (ContentRouter(freeze=True) analog); train the rest
            for mod in (self.W_q, self.W_k, self.W_v, self.W_out, self.W_ctx):
                for p in mod.parameters():
                    p.requires_grad_(False)
            self._log_temp.requires_grad_(False)

    def set_edge_perm(self, edge_types, seed, edge_index=None):
        """Phase-78 controls. Fix a within-relation-type permutation of edge indices: each edge is
        reassigned the per-edge content of a RANDOM other edge of the SAME relation. This preserves
        (a) relation-type statistics of the composed messages and (b) the traversal connectivity
        (routing in traverse/_traverse_batch always uses the REAL src/tgt, never the permutation),
        while destroying the binding between an edge's position in the graph and its instance content.
        Singleton-relation edges map to themselves. The permutation is fixed for the whole run (trained
        AND evaluated under it) so the operator gets its best shot at wrong-bound content.

        edge_index is REQUIRED for shuffle_full: ctx_src/ctx_tgt = the permuted edge's endpoints, fed
        to W_ctx so the attention context is coherent with the swapped ej (a full edge-identity swap).
        For 'shuffle'/'ef0_shuffle' the W_ctx endpoints stay real (edge_index optional)."""
        et = edge_types.detach().to('cpu')
        E = et.shape[0]
        g = torch.Generator().manual_seed(int(seed))
        perm = torch.arange(E)
        for r in torch.unique(et):
            idx = (et == r).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() > 1:
                perm[idx] = idx[torch.randperm(idx.numel(), generator=g)]
        dev = edge_types.device
        self.edge_perm = perm.to(dev)
        if self.mode == 'shuffle_full':
            assert edge_index is not None, "shuffle_full needs edge_index to permute the W_ctx endpoints"
        if edge_index is not None:                            # build unconditionally (used only by shuffle_full)
            self.ctx_src = edge_index[0][self.edge_perm].to(dev)  # so a later mode-flip to shuffle_full is safe
            self.ctx_tgt = edge_index[1][self.edge_perm].to(dev)

    @torch.no_grad()
    def scramble_strength(self, nf, ef0, edge_index):
        """Positive-control diagnostic: how much the within-type permutation actually MOVES per-edge
        content on this testbed. Returns (frac_moved, rel_disp): fraction of edges with perm[e]!=e, and
        mean ||ej_lesion - ej_pec|| / mean ||ej_pec||. A near-zero rel_disp means the lesion has no
        lever here (type-degenerate testbed) -> a 'pec~lesion' result is testbed-capacity, NOT a
        refutation. Computed against the SHUFFLE construction (ej permuted post-injection)."""
        assert self.edge_perm is not None
        frac = float((self.edge_perm != torch.arange(self.edge_perm.shape[0], device=self.edge_perm.device)).float().mean())
        src, tgt = edge_index[0], edge_index[1]
        ej = self.ein_norm(self.W_ein(torch.cat([ef0, nf[src], nf[tgt]], -1)))
        rel = float((ej[self.edge_perm] - ej).norm(dim=-1).mean() / (ej.norm(dim=-1).mean() + 1e-9))
        return frac, rel

    # ── per-edge DP state (recompute per step; depends on trained W_ein + frozen nf/ef0) ──
    def build_ej(self, nf, ef0, edge_index):
        src, tgt = edge_index[0], edge_index[1]
        if self.mode == 'ef0_shuffle':                           # permute the type-dominated ef0 BEFORE injection
            assert self.edge_perm is not None, "ef0_shuffle requires set_edge_perm first"
            ef0 = ef0[self.edge_perm]
        x = torch.cat([ef0, nf[src], nf[tgt]], dim=-1)
        ej = self.ein_norm(self.W_ein(x))                        # [E, d_edge]
        if self.mode in ('shuffle', 'shuffle_full'):             # wrong-bind the whole ej within relation type
            assert self.edge_perm is not None, "shuffle/shuffle_full require set_edge_perm first"
            ej = ej[self.edge_perm]
        return ej

    def _ctx_endpoints(self, eid, real_src, real_tgt):
        """Endpoints fed to W_ctx. Real for every arm EXCEPT shuffle_full, which uses the permuted
        edge's endpoints so the attention context matches the swapped ej (coherent full-identity swap)."""
        if self.mode == 'shuffle_full':
            return self.ctx_src[eid], self.ctx_tgt[eid]
        return real_src, real_tgt

    def _hop_logits(self, z_src, ej_e, nf, src, tgt):
        """Per-hop attention logits [M,H] for the selected edges, per arm. (Only re-weights when a
        target node receives >1 incoming edge in a hop; on this sparse graph that is the minority, so
        the COMPOSITION lives in the message, not here.) src/tgt are the W_ctx context endpoints
        (already remapped by _ctx_endpoints for shuffle_full); routing is handled by the caller."""
        if self.mode == 'rrprior':
            return torch.zeros(z_src.shape[0], self.num_heads, device=z_src.device)  # uniform
        # pec / capacity / static / shuffle*: the shared edge-to-edge attention operator
        return EdgeAttention.compose_scores(
            self.W_q, self.W_k, self.W_ctx, self._log_temp,
            z_src, ej_e, nf, src, tgt, self.num_heads, self.d_head)

    def _hop_message(self, z_src, ej_e, rel):
        """Per-edge message [M, d_edge]: the path-state composes with the edge (DistMult-style ⊙) so
        the accumulated path actually propagates. THIS is where edge-to-edge composition happens.
          pec/capacity: z_src ⊙ W_v(ej)            (instance edge, composed with the path)
          rrprior:      z_src ⊙ W_v(msg_emb(r))    (RELATION-TYPE edge — isolates instance edge content)
          static:       W_v(ej)                    (edge lookup, NO path composition — the 66-76 ceiling)
        """
        if self.mode == 'rrprior':
            return z_src * self.W_v(self.msg_emb(rel))
        if self.mode == 'static':
            return self.W_v(ej_e)
        return z_src * self.W_v(ej_e)

    def traverse(self, anchor, rel_chain, nf, ej, R2E):
        """Run hops 1..K-1 over TRAIN edges; return (frontier_ids [F], z_frontier [F, d_edge])."""
        N = nf.shape[0]
        z = self.seed_norm(self.W_seed(nf[anchor:anchor + 1]))    # [1, d_edge]
        frontier = torch.tensor([anchor], device=nf.device)
        K = len(rel_chain)
        for i in range(K - 1):
            r = rel_chain[i]
            if r not in R2E:
                return frontier[:0], z[:0]                        # dead end
            r_src, r_eid, r_tgt = R2E[r]
            fmask = torch.zeros(N, dtype=torch.bool, device=nf.device)
            fmask[frontier] = True
            sel = fmask[r_src]
            if not bool(sel.any()):
                return frontier[:0], z[:0]
            e_src, e_id, e_tgt = r_src[sel], r_eid[sel], r_tgt[sel]
            # map z (indexed by frontier order) to each selected edge's src node
            zpos = torch.full((N,), -1, dtype=torch.long, device=nf.device)
            zpos[frontier] = torch.arange(frontier.shape[0], device=nf.device)
            z_src = z[zpos[e_src]]                                # [M, d_edge]
            ej_e = ej[e_id]                                       # [M, d_edge]
            cs, ct = self._ctx_endpoints(e_id, e_src, e_tgt)     # W_ctx context (perm'd for shuffle_full)
            logits = self._hop_logits(z_src, ej_e, nf, cs, ct)             # [M, H]
            new_nodes, inv = torch.unique(e_tgt, return_inverse=True)
            Fn = new_nodes.shape[0]
            attn = _scatter_softmax(logits, inv, Fn)             # [M, H]
            msg = self._hop_message(z_src, ej_e, torch.full_like(e_id, r))  # [M, d_edge] (composes z_src)
            mh = msg.view(-1, self.num_heads, self.d_head)
            weighted = (mh * attn.unsqueeze(-1)).reshape(-1, self.d_edge)   # [M, d_edge]
            agg = torch.zeros(Fn, self.d_edge, device=nf.device)
            agg.scatter_add_(0, inv.unsqueeze(-1).expand(-1, self.d_edge), weighted)
            z = self.hop_norm(self.W_out(agg))                   # [Fn, d_edge]
            frontier = new_nodes
        return frontier, z

    def readout(self, frontier, z_frontier, rel_K, nf):
        """r_K-conditioned frontier pool -> DistMult over ALL N entities. Returns scores [N]."""
        N = nf.shape[0]
        if frontier.shape[0] == 0:
            return self.read_bias.expand(N)                      # finite fallback (dead-end queries)
        rk = self.rel_emb(torch.tensor([rel_K], device=nf.device))         # [1, d_edge]
        rk_exp = rk.expand(z_frontier.shape[0], -1)
        w = torch.softmax(self.W_poolattn(torch.cat([z_frontier, rk_exp], -1)).squeeze(-1), 0)  # [F]
        pooled = (w.unsqueeze(-1) * z_frontier).sum(0, keepdim=True)        # [1, d_edge]
        q = self.W_read(torch.cat([pooled, rk], -1))             # [1, d_node]
        return (q @ nf.t()).squeeze(0) + self.read_bias         # [N]

    def forward_query(self, anchor, rel_chain, nf, ej, R2E):
        if len(rel_chain) == 1:                                  # 1p: no traversal, seed -> readout
            z = self.seed_norm(self.W_seed(nf[anchor:anchor + 1]))
            return self.readout(torch.tensor([anchor], device=nf.device), z, rel_chain[0], nf)
        frontier, z = self.traverse(anchor, rel_chain, nf, ej, R2E)
        return self.readout(frontier, z, rel_chain[-1], nf)

    def score_batch(self, queries, nf, ef0, edge_index, R2E):
        """queries: list of (anchor, rel_chain). Returns scores [B, N] (per-query DP, batched matmul-free).
        ej is rebuilt once for the whole batch (depends on trained W_ein)."""
        ej = self.build_ej(nf, ef0, edge_index)
        rows = [self.forward_query(int(a), list(rc), nf, ej, R2E) for (a, rc) in queries]
        return torch.stack(rows, 0)                              # [B, N]

    # ── batched / vectorized forward (same math as forward_query; grouped by chain length K) ──
    def _traverse_batch(self, anchors, rel_mat, K, nf, ej, csr):
        """anchors [Q], rel_mat [Q,K]. Runs K-1 hops in lockstep across the batch. Returns the
        penult frontier as flat (fnode [F], fquery [F], fz [F,d_edge])."""
        N = nf.shape[0]
        Q = anchors.shape[0]
        fnode = anchors
        fquery = torch.arange(Q, device=nf.device)
        fz = self.seed_norm(self.W_seed(nf[anchors]))               # [Q, d_edge]
        for i in range(K - 1):
            frel = rel_mat[fquery, i]                               # relation for each entry this hop
            seg, eid, tgt = _gather_edges(fnode, frel, csr)
            if seg.numel() == 0:
                empty = torch.zeros(0, dtype=torch.long, device=nf.device)
                return empty, empty, fz[:0]
            z_src = fz[seg]; q_of = fquery[seg]; src_node = fnode[seg]
            ej_e = ej[eid]; rel_e = frel[seg]
            cs, ct = self._ctx_endpoints(eid, src_node, tgt)                   # W_ctx context (perm'd for shuffle_full)
            logits = self._hop_logits(z_src, ej_e, nf, cs, ct)                 # [M,H]
            msg = self._hop_message(z_src, ej_e, rel_e)                        # [M,d_edge]
            gkey = q_of * N + tgt
            uniq, inv = torch.unique(gkey, return_inverse=True)
            G = uniq.shape[0]
            attn = _scatter_softmax(logits, inv, G)
            mh = msg.view(-1, self.num_heads, self.d_head)
            weighted = (mh * attn.unsqueeze(-1)).reshape(-1, self.d_edge)
            agg = torch.zeros(G, self.d_edge, device=nf.device)
            agg.scatter_add_(0, inv.unsqueeze(-1).expand(-1, self.d_edge), weighted)
            fz = self.hop_norm(self.W_out(agg))
            fnode = uniq % N; fquery = uniq // N
        return fnode, fquery, fz

    def _readout_batch(self, fnode, fquery, fz, rel_K, Q, nf):
        """All-N readout for a batch of Q queries. rel_K [Q]. Returns scores [Q, N]."""
        N = nf.shape[0]
        rk_all = self.rel_emb(rel_K)                                # [Q, d_edge]
        if fnode.numel() == 0:
            return self.read_bias.expand(Q, N)
        rk_e = rk_all[fquery]                                       # [F, d_edge]
        pa = self.W_poolattn(torch.cat([fz, rk_e], -1)).squeeze(-1)        # [F]
        # softmax of pa within each query group
        mx = torch.full((Q,), -1e9, device=nf.device).scatter_reduce(0, fquery, pa, reduce='amax', include_self=False)
        ex = torch.exp(pa - mx[fquery])
        den = torch.zeros(Q, device=nf.device).scatter_add(0, fquery, ex)
        w = ex / (den[fquery] + 1e-10)
        pooled = torch.zeros(Q, self.d_edge, device=nf.device)
        pooled.scatter_add_(0, fquery.unsqueeze(-1).expand(-1, self.d_edge), w.unsqueeze(-1) * fz)
        q_read = self.W_read(torch.cat([pooled, rk_all], -1))      # [Q, d_node]
        return q_read @ nf.t() + self.read_bias                    # [Q, N]

    def score_batch_vec(self, queries, nf, ef0, edge_index, csr):
        """Vectorized scorer. queries: list of (anchor, rel_chain). Groups by K, runs batched
        traverse+readout per group, returns scores [B, N] in input order. ej rebuilt once."""
        ej = self.build_ej(nf, ef0, edge_index)
        N = nf.shape[0]
        out = torch.empty(len(queries), N, device=nf.device)
        by_k = {}
        for i, (a, rc) in enumerate(queries):
            by_k.setdefault(len(rc), []).append(i)
        for K, idxs in by_k.items():
            anchors = torch.tensor([int(queries[i][0]) for i in idxs], device=nf.device)
            rel_mat = torch.tensor([list(queries[i][1]) for i in idxs], device=nf.device)
            fnode, fquery, fz = self._traverse_batch(anchors, rel_mat, K, nf, ej, csr)
            scores = self._readout_batch(fnode, fquery, fz, rel_mat[:, K - 1], len(idxs), nf)
            out[torch.tensor(idxs, device=nf.device)] = scores
        return out

    def penult_reachable_mask(self, anchor, rel_chain, train_hr2t, N, device):
        """Entities reachable from the penult frontier via r_K in the TRAIN graph (the MASK-ONLY /
        within-reached candidate set). gold (a TEST edge at eval) is usually NOT here — that is the
        point: PEC-pf must rank a non-train-reachable answer above this set."""
        cur = {anchor}
        for r in rel_chain[:-1]:
            nxt = set()
            for e in cur:
                nxt |= train_hr2t.get((e, r), set())
            cur = nxt
            if not cur:
                break
        reached = set()
        rK = rel_chain[-1]
        for m in cur:
            reached |= train_hr2t.get((m, rK), set())
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        if reached:
            mask[torch.tensor(sorted(reached), device=device)] = True
        return mask
