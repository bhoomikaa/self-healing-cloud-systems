"""Model 1 v1 = the unchanged v0 model plus a binary detection head (V1_PROTOCOL.md sections 1-2; amendment A-2).

A subclass in a new file, so the committed v0 modules and their evidence hashes stay untouched. The detection head is created AFTER the
v0 modules, so for a given seed the shared weights are initialised exactly as in v0. Head = the fault head's pooled last-minute features
(mean and max over valid nodes) -> Linear(128, 64) -> GELU -> Linear(64, 1): one logit per window, +8,321 parameters (176,521 in total).
"""
import torch

from .heads import FaultHead
from .model import Model1
from .spec import ModelConfig


class Model1V1(Model1):
    def __init__(self, spec, cfg=ModelConfig()):
        super().__init__(spec, cfg)
        self.detection_head = FaultHead(cfg.d_model, 1)                 # same pooling as the fault head, one output

    def forward(self, inputs, input_mask, node_valid=None, node_type=None, applicable_entry=None, return_aux=False):
        """As Model1.forward plus out["detection_logit"] (B,). Both heads read the final historical minute only."""
        B, T, N, _ = inputs.shape
        nv = torch.ones(B, N, dtype=torch.bool, device=inputs.device) if node_valid is None else node_valid
        out = super().forward(inputs, input_mask, node_valid=nv, node_type=node_type, applicable_entry=applicable_entry, return_aux=True)
        out["detection_logit"] = self.detection_head(out["hidden"][:, -1], nv)[:, 0]
        if not return_aux:
            for k in ("hidden", "fusion_weights", "token_avail"):
                out.pop(k)
        return out
