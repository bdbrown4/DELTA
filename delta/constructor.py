"""
Transformer-based Graph Constructor for DELTA.

Solves the bootstrap problem: you need to understand input to build a graph,
but the graph is how you understand input.

Strategy: use a lightweight transformer to produce initial embeddings and
infer relational structure, then construct a graph that DELTA can process.

The transformer is scaffolding — once DELTA is trained, its own graph
representations can replace the transformer for graph construction.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from delta.graph import DeltaGraph


class MiniTransformerBlock(nn.Module):
    """Minimal transformer block for bootstrapping — not the main architecture."""

    def __init__(self, d_model: int, num_heads: int = 4, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, attention_weights)."""
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x, attn_weights


class GraphConstructor(nn.Module):
    """Converts raw sequential input into a DELTA graph.

    Pipeline:
    1. Embed input tokens
    2. Run through mini-transformer to get contextual embeddings + attention
    3. Cluster tokens into concept nodes (merge related tokens)
    4. Infer typed edges from attention patterns
    5. Output a DeltaGraph ready for DELTA processing

    For Phase 5 experiments this uses a small learned transformer.
    For production, this could be replaced with a pre-trained model's
    embeddings and attention maps.
    """

    def __init__(self, vocab_size: int, d_model: int, d_node: int, d_edge: int,
                 num_layers: int = 2, num_heads: int = 4,
                 num_edge_types: int = 8,
                 attention_threshold: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_node = d_node
        self.d_edge = d_edge
        self.attention_threshold = attention_threshold
        self.num_edge_types = num_edge_types

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            MiniTransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])

        # Project transformer outputs to graph dimensions
        self.to_node = nn.Linear(d_model, d_node)
        # Per-layer edge projections: each transformer layer produces a different edge type
        self.to_edge_per_layer = nn.ModuleList([
            nn.Linear(2 * d_model + 1, d_edge)  # +1 for attention weight
            for _ in range(num_layers)
        ])
        # Final edge combiner: merge per-layer edge features
        self.edge_combiner = nn.Linear(d_edge * num_layers, d_edge)
        # Edge type classifier (now actually used)
        self.edge_type_head = nn.Linear(d_edge, num_edge_types)

    def forward(self, token_ids: torch.Tensor) -> DeltaGraph:
        """
        Args:
            token_ids: [seq_len] or [1, seq_len] token indices

        Returns:
            DeltaGraph constructed from the input
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        x = self.embedding(token_ids)  # [1, S, d_model]
        x = self.pos_encoding(x)

        # Collect per-layer embeddings and attention
        layer_outputs = []
        attn_weights_all = []
        for layer in self.layers:
            x, attn_w = layer(x)
            layer_outputs.append(x.squeeze(0))  # [S, d_model]
            attn_weights_all.append(attn_w.squeeze(0))  # [S, S]

        x = x.squeeze(0)  # [S, d_model] — final layer output

        # --- Create nodes from final layer ---
        node_features = self.to_node(x)  # [S, d_node]

        # --- Create edges: union of all layers' attention patterns ---
        S = x.shape[0]
        # Collect edges from each layer's attention
        all_src = []
        all_tgt = []
        for attn in attn_weights_all:
            attn_no_self = attn.clone()
            attn_no_self.fill_diagonal_(0)
            edge_mask = attn_no_self > self.attention_threshold
            src_idx, tgt_idx = torch.where(edge_mask)
            all_src.append(src_idx)
            all_tgt.append(tgt_idx)

        # Union of all edges
        if all_src:
            union_src = torch.cat(all_src)
            union_tgt = torch.cat(all_tgt)
            # Deduplicate
            edge_pairs = torch.stack([union_src, union_tgt])
            unique_flat = edge_pairs[0] * S + edge_pairs[1]
            unique_vals, inverse = torch.unique(unique_flat, return_inverse=True)
            src_idx = unique_vals // S
            tgt_idx = unique_vals % S
        else:
            src_idx = torch.zeros(0, dtype=torch.long, device=x.device)
            tgt_idx = torch.zeros(0, dtype=torch.long, device=x.device)

        if len(src_idx) == 0:
            # Fallback: keep top-k attention per node from final layer
            k = min(3, S - 1)
            final_attn = attn_weights_all[-1].clone()
            final_attn.fill_diagonal_(0)
            _, top_idx = torch.topk(final_attn, k, dim=1)
            src_idx = torch.arange(S, device=x.device).unsqueeze(1).expand_as(top_idx).reshape(-1)
            tgt_idx = top_idx.reshape(-1)

        edge_index = torch.stack([src_idx, tgt_idx])
        E = edge_index.shape[1]

        # --- Per-layer edge features ---
        per_layer_edge_feats = []
        for layer_idx, (layer_emb, attn) in enumerate(zip(layer_outputs, attn_weights_all)):
            src_emb = layer_emb[src_idx]  # [E, d_model]
            tgt_emb = layer_emb[tgt_idx]  # [E, d_model]
            attn_vals = attn[src_idx, tgt_idx].unsqueeze(-1)  # [E, 1]
            edge_input = torch.cat([src_emb, tgt_emb, attn_vals], dim=-1)
            layer_feats = self.to_edge_per_layer[layer_idx](edge_input)  # [E, d_edge]
            per_layer_edge_feats.append(layer_feats)

        # Combine per-layer features
        combined = torch.cat(per_layer_edge_feats, dim=-1)  # [E, d_edge * num_layers]
        edge_features = self.edge_combiner(combined)  # [E, d_edge]

        # Classify edge types (now used — stored as edge metadata)
        edge_types = self.edge_type_head(edge_features)  # [E, num_edge_types]
        # Soft edge type weighting folded into features
        edge_type_weights = F.softmax(edge_types, dim=-1)  # [E, num_edge_types]

        return DeltaGraph(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
        )

    def construct_from_embeddings(self, embeddings: torch.Tensor,
                                  attention_matrix: torch.Tensor) -> DeltaGraph:
        """Construct graph from pre-computed embeddings and attention.

        Useful when using a pre-trained transformer (BERT, GPT, etc.)
        as the bootstrap — extract embeddings and attention, pass them here.

        Uses the first layer's edge projection replicated across all layers
        since we only have a single attention matrix.
        """
        S = embeddings.shape[0]
        node_features = self.to_node(embeddings)

        attn_no_self = attention_matrix.clone()
        attn_no_self.fill_diagonal_(0)
        edge_mask = attn_no_self > self.attention_threshold
        src_idx, tgt_idx = torch.where(edge_mask)

        if len(src_idx) == 0:
            k = min(3, S - 1)
            _, top_idx = torch.topk(attn_no_self, k, dim=1)
            src_idx = torch.arange(S, device=embeddings.device).unsqueeze(1).expand_as(top_idx).reshape(-1)
            tgt_idx = top_idx.reshape(-1)

        edge_index = torch.stack([src_idx, tgt_idx])
        src_emb = embeddings[src_idx]
        tgt_emb = embeddings[tgt_idx]
        attn_vals = attention_matrix[src_idx, tgt_idx].unsqueeze(-1)
        edge_input = torch.cat([src_emb, tgt_emb, attn_vals], dim=-1)

        # Use all layer projections with the same input (single attention matrix)
        per_layer = [proj(edge_input) for proj in self.to_edge_per_layer]
        combined = torch.cat(per_layer, dim=-1)
        edge_features = self.edge_combiner(combined)

        return DeltaGraph(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
        )


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1]]
