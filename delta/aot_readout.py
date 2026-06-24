"""Phase 79 — AOT-readout: the fair, node-only control for the "JIT ~2x AOT" gain.

A query-time model on the SAME frozen 48-d encoder-internal nf JIT uses, with a trained pooled-DistMult
readout head CAPACITY-MATCHED to PathComposerPF's, but NO edge participation (no ef0, no edge operator).
It pools the SAME discrete train-reachable penult frontier JIT traverses to (parameter-free graph
reachability), with a NODE-ONLY z-state z(e)=seed_norm(W_seed(nf[e])) — i.e. each penult entity's frozen
embedding mapped into path-space, with no composed history. The SOLE difference from JIT-pec is the
z-state (node-only vs edge-composed); the frontier, the readout head, the objective, and nf are identical.

  gap_path = JIT-pec - AOT-readout(trained)      -> the value of edge/path composition (one axis).
  gap_head = AOT-readout(trained) - AOT-readout(UNTRAINED)  -> the value of the trained head (= AOT-soft48
             is just this module left at random init, frozen — same structure, no training).

K=1 IDENTITY (the falsifiable plumbing gate): at 1p there is no traversal, the frontier is {anchor}, and
this reduces to seed_norm(W_seed(nf[anchor])) -> readout, byte-identical to PathComposerPF.forward_query's
K=1 path. See docs/phase_79.md.
"""
import torch
import torch.nn as nn

from delta.path_compose import _gather_edges


class AOTReadout(nn.Module):
    """Node-only query-time readout. score_batch_vec(queries, nf, ef0, ei, csr) matches PathComposerPF's
    signature so phase77 train_arm/eval_arm/generate_train_chains are reused verbatim (ef0/ei unused)."""

    def __init__(self, num_relations, d_node=48, d_edge=24, num_heads=4, mode='aot_readout'):
        super().__init__()
        self.mode = mode
        self.d_node = d_node
        self.d_edge = d_edge
        self.num_heads = num_heads
        self.num_relations = num_relations
        # readout head — submodules byte-identical in shape to PathComposerPF's seed + readout block,
        # so K=1 can be made bit-identical and the param budget of the HEAD is matched. NO edge operator
        # (W_q/W_k/W_v/W_out/W_ctx), NO per-edge state builder (W_ein) — edges never participate.
        self.W_seed = nn.Linear(d_node, d_edge)
        self.seed_norm = nn.LayerNorm(d_edge)
        self.rel_emb = nn.Embedding(num_relations, d_edge)
        self.W_poolattn = nn.Linear(2 * d_edge, 1)
        self.W_read = nn.Linear(d_edge + d_edge, d_node)
        self.read_bias = nn.Parameter(torch.zeros(1))

    # ── discrete penult frontier via TRAIN-graph reachability (routing only; NO operator, NO ef0) ──
    def _frontier_batch(self, anchors, rel_mat, K, nf, csr):
        """Same frontier (fnode, fquery) that PathComposerPF._traverse_batch reaches, minus the message/
        attention that builds its z — pure reachability over the chain's TRAIN edges."""
        N = nf.shape[0]
        fnode = anchors
        fquery = torch.arange(anchors.shape[0], device=nf.device)
        for i in range(K - 1):
            frel = rel_mat[fquery, i]
            seg, eid, tgt = _gather_edges(fnode, frel, csr)
            if seg.numel() == 0:
                z = torch.zeros(0, dtype=torch.long, device=nf.device)
                return z, z
            q_of = fquery[seg]
            uniq = torch.unique(q_of * N + tgt)
            fnode = uniq % N
            fquery = uniq // N
        return fnode, fquery

    # ── readout: identical math to PathComposerPF._readout_batch (r_K-conditioned pool -> DistMult) ──
    def _readout_batch(self, fnode, fquery, fz, rel_K, Q, nf):
        N = nf.shape[0]
        rk_all = self.rel_emb(rel_K)                                # [Q, d_edge]
        if fnode.numel() == 0:
            return self.read_bias.expand(Q, N)
        rk_e = rk_all[fquery]
        pa = self.W_poolattn(torch.cat([fz, rk_e], -1)).squeeze(-1)
        mx = torch.full((Q,), -1e9, device=nf.device).scatter_reduce(0, fquery, pa, reduce='amax', include_self=False)
        ex = torch.exp(pa - mx[fquery])
        den = torch.zeros(Q, device=nf.device).scatter_add(0, fquery, ex)
        w = ex / (den[fquery] + 1e-10)
        pooled = torch.zeros(Q, self.d_edge, device=nf.device)
        pooled.scatter_add_(0, fquery.unsqueeze(-1).expand(-1, self.d_edge), w.unsqueeze(-1) * fz)
        q_read = self.W_read(torch.cat([pooled, rk_all], -1))
        return q_read @ nf.t() + self.read_bias                    # [Q, N]

    def _node_z(self, fnode, nf):
        """Node-only path-state: each frontier entity's frozen embedding mapped into path-space. At the
        anchor (K=1) this equals PathComposerPF's seed state exactly."""
        if fnode.numel() == 0:
            return torch.zeros(0, self.d_edge, device=nf.device)
        return self.seed_norm(self.W_seed(nf[fnode]))

    def score_batch_vec(self, queries, nf, ef0, edge_index, csr):
        """[B, N] scores. Grouped by chain length K. ef0 / edge_index accepted for signature parity with
        PathComposerPF but UNUSED (edges never participate)."""
        N = nf.shape[0]
        out = torch.empty(len(queries), N, device=nf.device)
        by_k = {}
        for i, (a, rc) in enumerate(queries):
            by_k.setdefault(len(rc), []).append(i)
        for K, idxs in by_k.items():
            anchors = torch.tensor([int(queries[i][0]) for i in idxs], device=nf.device)
            rel_mat = torch.tensor([list(queries[i][1]) for i in idxs], device=nf.device)
            if K == 1:
                fnode, fquery = anchors, torch.arange(len(idxs), device=nf.device)   # no traversal -> {anchor}
            else:
                fnode, fquery = self._frontier_batch(anchors, rel_mat, K, nf, csr)
            fz = self._node_z(fnode, nf)
            scores = self._readout_batch(fnode, fquery, fz, rel_mat[:, K - 1], len(idxs), nf)
            out[torch.tensor(idxs, device=nf.device)] = scores
        return out


def copy_readout_head(src_pec, dst_aot):
    """Copy PathComposerPF's seed+readout submodule weights into an AOTReadout so their K=1 forwards are
    bit-identical (used by the 1p plumbing gate / tests). The two share these exact submodules."""
    for name in ['W_seed', 'seed_norm', 'rel_emb', 'W_poolattn', 'W_read']:
        getattr(dst_aot, name).load_state_dict(getattr(src_pec, name).state_dict())
    dst_aot.read_bias.data.copy_(src_pec.read_bias.data)
