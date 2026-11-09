"""Dataset-independent spatio-temporal core (design note 4.1-4.2): fusion, embeddings, factorised attention.

Input: tokens (B,T,N,M,d), token_avail (B,T,N,M) bool, node_type (N,) int, node_valid (B,N) bool. Output: hidden
(B,T,N,d). The core knows nothing about the dataset's node count or channels.

Masking rules (every one is covered by a test):
  * An unavailable token never contributes: it is replaced by 0 with `torch.where` and its fusion weight is exactly 0.
  * A padded node (node_valid False) is treated as having no available token, is never a key in cross-node attention
    and is reset to 0 after every block, so whatever sits in its slot cannot reach a valid node.
  * Attention rows with no valid key return 0 instead of NaN.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


class ModalityFusion(nn.Module):
    """w_m = sigmoid(Linear(d,1)(h_m)) * avail_m ; fused = sum(w_m h_m) / (sum(w_m) + eps) ; learned vector if none available."""

    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.gate = nn.Linear(d_model, 1)
        self.missing = nn.Parameter(torch.randn(d_model) * 0.02)       # one shared fallback vector
        self.eps = eps

    def forward(self, tokens, avail):
        """tokens (B,T,N,M,d), avail (B,T,N,M) -> fused (B,T,N,d), weights (B,T,N,M)."""
        h = torch.where(avail[..., None], tokens, torch.zeros_like(tokens))
        w = torch.where(avail, torch.sigmoid(self.gate(h)).squeeze(-1), torch.zeros_like(h[..., 0]))
        fused = (w[..., None] * h).sum(-2) / (w.sum(-1, keepdim=True) + self.eps)
        fused = torch.where(avail.any(-1, keepdim=True), fused, self.missing.expand_as(fused))
        return fused, w


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv, self.out = nn.Linear(d_model, 3 * d_model), nn.Linear(d_model, d_model)

    def forward(self, x, key_valid=None, return_weights=False):
        """x (R,L,d); key_valid (R,L) bool or None (all keys valid). Bidirectional: every query sees every valid key."""
        R, L, d = x.shape
        q, k, v = self.qkv(x).view(R, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)        # each (R,h,L,dh)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        if key_valid is None:
            attn = scores.softmax(-1)
        else:
            kv = key_valid[:, None, None, :]
            attn = torch.where(kv, scores, torch.full_like(scores, torch.finfo(scores.dtype).min)).softmax(-1)
            attn = torch.where(key_valid.any(-1)[:, None, None, None], attn, torch.zeros_like(attn))   # no valid key -> 0, not NaN
            attn = attn * kv                                                                     # padded keys: weight exactly 0
            v = torch.where(kv.transpose(-1, -2), v, torch.zeros_like(v))                       # and never 0 * NaN
        y = (attn @ v).transpose(1, 2).reshape(R, L, d)
        y = self.out(y)
        return (y, attn) if return_weights else y


class FactorisedBlock(nn.Module):
    """Pre-LN: temporal attention (per node over T) -> cross-node attention (per minute over N) -> FFN, each with residual."""

    def __init__(self, d_model, n_heads, d_ffn, dropout):
        super().__init__()
        self.ln_t, self.ln_n, self.ln_f = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.temporal, self.cross = MultiHeadAttention(d_model, n_heads), MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ffn, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, node_valid):
        B, T, N, d = h.shape
        t_in = self.ln_t(h).permute(0, 2, 1, 3).reshape(B * N, T, d)                     # one sequence of T minutes per node
        h = h + self.drop(self.temporal(t_in).reshape(B, N, T, d).permute(0, 2, 1, 3))
        n_in = self.ln_n(h).reshape(B * T, N, d)                                          # one set of N nodes per minute
        kv = node_valid[:, None, :].expand(B, T, N).reshape(B * T, N)
        h = h + self.drop(self.cross(n_in, kv).reshape(B, T, N, d))
        h = h + self.drop(self.ffn(self.ln_f(h)))
        return torch.where(node_valid[:, None, :, None], h, torch.zeros_like(h))


class SpatioTemporalCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fusion = ModalityFusion(cfg.d_model, cfg.fusion_eps)
        self.type_emb = nn.Embedding(cfg.n_node_types, cfg.d_model)
        self.time_emb = nn.Embedding(cfg.n_input_steps, cfg.d_model)
        self.blocks = nn.ModuleList(FactorisedBlock(cfg.d_model, cfg.n_heads, cfg.d_ffn, cfg.dropout) for _ in range(cfg.n_layers))
        self.final_ln = nn.LayerNorm(cfg.d_model)
        for e in (self.type_emb, self.time_emb):
            nn.init.normal_(e.weight, std=0.02)

    def forward(self, tokens, token_avail, node_type, node_valid, return_aux=False):
        B, T, N, M, d = tokens.shape
        c = self.cfg
        if (T, M, d) != (c.n_input_steps, c.n_modalities, c.d_model):
            raise ValueError(f"tokens (T,M,d)={(T, M, d)} differ from the configuration {(c.n_input_steps, c.n_modalities, c.d_model)}")
        if token_avail.shape != (B, T, N, M) or token_avail.dtype != torch.bool:
            raise ValueError("token_avail must be bool (B,T,N,M)")
        if node_type.shape != (N,) or node_valid.shape != (B, N) or node_valid.dtype != torch.bool:
            raise ValueError(f"node_type must be ({N},) and node_valid bool ({B},{N})")
        if int(node_type.min()) < 0 or int(node_type.max()) >= c.n_node_types:
            raise ValueError("node_type outside [0, n_node_types)")
        avail = token_avail & node_valid[:, None, :, None]                                # a padded node has no available token
        fused, w = self.fusion(tokens, avail)
        h = fused + self.type_emb(node_type)[None, None] + self.time_emb.weight[None, :, None]
        h = torch.where(node_valid[:, None, :, None], h, torch.zeros_like(h))
        for blk in self.blocks:
            h = blk(h, node_valid)
        h = self.final_ln(h)
        h = torch.where(node_valid[:, None, :, None], h, torch.zeros_like(h))
        return (h, {"fusion_weights": w, "token_avail": avail}) if return_aux else h
