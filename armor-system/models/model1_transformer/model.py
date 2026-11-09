"""Model 1 v0: input adapter -> fusion -> factorised Transformer core -> fault head and next-step head."""
import hashlib

import numpy as np
import torch
from torch import nn

from .adapters import InputAdapter, OutputAdapter
from .core import SpatioTemporalCore
from .heads import FaultHead, NextStepHead
from .spec import AdapterSpec, ModelConfig


def enable_determinism(threads=1):
    """CPU-only reproducibility: deterministic kernels and one intra-op thread (float summation order is then fixed)."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)


def tensor_digest(t):
    a = np.ascontiguousarray(t.detach().cpu().to(torch.float32).numpy())
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def state_digest(model):
    """SHA-256 over every parameter and buffer (name, dtype, shape, bytes) in state_dict order."""
    h = hashlib.sha256()
    for name, t in model.state_dict().items():
        a = np.ascontiguousarray(t.detach().cpu().numpy())
        h.update(f"{name}|{a.dtype}|{a.shape}|".encode() + a.tobytes())
    return h.hexdigest()


class Model1(nn.Module):
    def __init__(self, spec, cfg=ModelConfig()):
        super().__init__()
        if len(spec.modalities) != cfg.n_modalities or len(spec.type_names) != cfg.n_node_types:
            raise ValueError("spec and configuration disagree on the number of modalities or node types")
        self.spec, self.cfg = spec, cfg
        self.input_adapter = InputAdapter(spec, cfg.d_model)
        self.core = SpatioTemporalCore(cfg)
        self.fault_head = FaultHead(cfg.d_model, cfg.n_classes)
        self.next_step_head = NextStepHead(OutputAdapter(spec, cfg.d_model))
        self.register_buffer("_default_node_type", torch.as_tensor(spec.node_type), persistent=False)

    @classmethod
    def build(cls, spec, cfg=ModelConfig(), seed=0):
        """Seeded construction: the same seed gives bit-identical initial weights."""
        torch.manual_seed(int(seed))
        return cls(spec, cfg)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, inputs, input_mask, node_valid=None, node_type=None, applicable_entry=None, return_aux=False):
        """inputs (B,T,N,C) scaled values, input_mask (B,T,N,C) bool effective mask.

        node_valid (B,N) bool defaults to all True; node_type (N,) defaults to the spec's (pass it to run a different
        node count, e.g. padded batches); applicable_entry ((N,C) or (B,N,C) bool), if given, zeroes predictions at
        structurally unavailable entries. Returns {"logits": (B,K), "prediction": (B,N,C)} (+ aux).
        """
        B, T, N, _ = inputs.shape
        node_type = self._default_node_type if node_type is None else node_type.to(inputs.device)
        if node_type.shape != (N,):
            raise ValueError(f"node_type has shape {tuple(node_type.shape)} but the batch has {N} nodes; pass node_type for a different node count")
        node_valid = torch.ones(B, N, dtype=torch.bool, device=inputs.device) if node_valid is None else node_valid
        if input_mask.dtype != torch.bool or input_mask.shape != inputs.shape:
            raise ValueError(f"input_mask must be bool with shape {tuple(inputs.shape)}, got {input_mask.dtype} {tuple(input_mask.shape)}")
        input_mask = input_mask & node_valid[:, None, :, None]                  # a padded node's slot is dead BEFORE the adapter: NaN/inf there
        tokens, avail = self.input_adapter(inputs, input_mask, node_type)       # would otherwise give 0 * NaN in the adapter weight gradient
        hidden, aux = self.core(tokens, avail, node_type, node_valid, return_aux=True)
        h_last = hidden[:, -1]                                                  # the final historical minute only
        out = {"logits": self.fault_head(h_last, node_valid),
               "prediction": self.next_step_head(h_last, node_type, node_valid)}
        if applicable_entry is not None:
            app = applicable_entry if applicable_entry.ndim == 3 else applicable_entry[None]
            out["prediction"] = torch.where(app, out["prediction"], torch.zeros_like(out["prediction"]))
        if return_aux:
            out.update(aux, hidden=hidden)
        return out
