"""D1 input and output adapters (design note 3.3). They are the only modules that know the channel layout.

Input: per (node type, modality) one Linear on `[values * mask, mask]`, so its width is 2 x the number of channels.
Unavailable entries are replaced by 0 with `torch.where` (never by multiplication), so a polluted value (huge, inf, NaN)
cannot reach the network and its gradient is exactly 0. Output: per node type one Linear d -> (its channels), scattered
back to the dataset's channel layout; every other channel is exactly 0.
"""
import torch
from torch import nn


def _nodes_of_type(node_type, k):
    return torch.nonzero(node_type == k, as_tuple=False).flatten()


class InputAdapter(nn.Module):
    def __init__(self, spec, d_model):
        super().__init__()
        self.spec, self.d = spec, d_model
        self.linears = nn.ModuleDict()
        for (k, m), cols in sorted(spec.type_channels.items()):
            self.linears[f"{spec.type_names[k]}_{m}"] = nn.Linear(2 * len(cols), d_model)

    def forward(self, inputs, input_mask, node_type):
        """inputs (B,T,N,C) float, input_mask (B,T,N,C) bool, node_type (N,) -> tokens (B,T,N,M,d), token_avail (B,T,N,M) bool."""
        B, T, N, C = inputs.shape
        if input_mask.shape != inputs.shape or input_mask.dtype != torch.bool:
            raise ValueError(f"input_mask must be bool with shape {tuple(inputs.shape)}, got {input_mask.dtype} {tuple(input_mask.shape)}")
        if C != self.spec.n_channels or node_type.shape != (N,):
            raise ValueError(f"expected {self.spec.n_channels} channels and node_type ({N},), got C={C}, node_type {tuple(node_type.shape)}")
        mods, M = self.spec.modalities, len(self.spec.modalities)
        tokens = inputs.new_zeros(B, T, N, M, self.d)
        avail = torch.zeros(B, T, N, M, dtype=torch.bool, device=inputs.device)
        for k in range(len(self.spec.type_names)):
            nodes = _nodes_of_type(node_type.to(inputs.device), k)
            if nodes.numel() == 0:
                continue
            x_k, m_k = inputs.index_select(2, nodes), input_mask.index_select(2, nodes)
            toks, avs = [], []
            for m in mods:
                cols = self.spec.channels(k, m)
                if cols.size == 0:                              # this node type has no such modality: token never created
                    toks.append(inputs.new_zeros(B, T, nodes.numel(), self.d))
                    avs.append(torch.zeros(B, T, nodes.numel(), dtype=torch.bool, device=inputs.device))
                    continue
                cols = torch.as_tensor(cols, device=inputs.device)
                x, mk = x_k.index_select(3, cols), m_k.index_select(3, cols)
                feat = torch.cat([torch.where(mk, x, torch.zeros_like(x)), mk.to(x.dtype)], dim=-1)
                av = mk.any(-1)
                h = self.linears[f"{self.spec.type_names[k]}_{m}"](feat)
                toks.append(torch.where(av[..., None], h, torch.zeros_like(h)))
                avs.append(av)
            tok_k, av_k = torch.stack(toks, dim=3), torch.stack(avs, dim=3)
            tokens = tokens.index_copy(2, nodes, tok_k)
            avail = avail.index_copy(2, nodes, av_k)
        return tokens, avail


class OutputAdapter(nn.Module):
    """h (B,N,d) -> (B,N,C): type-specific Linear scattered to the schema-defined channels, 0 everywhere else."""

    def __init__(self, spec, d_model):
        super().__init__()
        self.spec = spec
        self.linears = nn.ModuleDict({spec.type_names[k]: nn.Linear(d_model, int(spec.output_channels(k).size))
                                      for k in range(len(spec.type_names)) if spec.output_channels(k).size})

    def forward(self, h, node_type):
        B, N, _ = h.shape
        if node_type.shape != (N,):
            raise ValueError(f"node_type must have shape ({N},), got {tuple(node_type.shape)}")
        out = h.new_zeros(B, N, self.spec.n_channels)
        for k, name in enumerate(self.spec.type_names):
            if name not in self.linears:
                continue
            nodes = _nodes_of_type(node_type.to(h.device), k)
            if nodes.numel() == 0:
                continue
            cols = torch.as_tensor(self.spec.output_channels(k), device=h.device)
            y = self.linears[name](h.index_select(1, nodes))                                   # (B, Nk, len(cols))
            full = h.new_zeros(B, nodes.numel(), self.spec.n_channels).index_copy(2, cols, y)
            out = out.index_copy(1, nodes, full)
        return out
