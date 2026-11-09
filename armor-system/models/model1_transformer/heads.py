"""Output heads (design note 4.2). Both read the LAST of the six historical minutes only."""
import torch
from torch import nn


class FaultHead(nn.Module):
    """Mean and max over valid nodes -> concat (2d) -> Linear(2d,d) -> GELU -> Linear(d,n_classes)."""

    def __init__(self, d_model, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, n_classes))

    def forward(self, h_last, node_valid):
        """h_last (B,N,d), node_valid (B,N) bool -> logits (B,n_classes). A sample with no valid node pools to zeros."""
        v = node_valid[..., None]
        count = node_valid.sum(1, keepdim=True).clamp(min=1).to(h_last.dtype)
        mean = torch.where(v, h_last, torch.zeros_like(h_last)).sum(1) / count
        mx = torch.where(v, h_last, torch.full_like(h_last, float("-inf"))).max(1).values
        mx = torch.where(node_valid.any(1, keepdim=True), mx, torch.zeros_like(mx))
        return self.mlp(torch.cat([mean, mx], dim=-1))


class NextStepHead(nn.Module):
    """Per node at the last minute through the dataset's output adapter -> (B,N,C) scaled telemetry of the target minute."""

    def __init__(self, output_adapter):
        super().__init__()
        self.adapter = output_adapter

    def forward(self, h_last, node_type, node_valid):
        pred = self.adapter(h_last, node_type)
        return torch.where(node_valid[..., None], pred, torch.zeros_like(pred))
