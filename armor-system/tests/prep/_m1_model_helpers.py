"""Shared helpers for the Model 1 core tests (Linear 298-50): synthetic D1-shaped batches and a seeded model."""
import os
import sys

import _m1_helpers as H  # noqa: F401  (puts armor-system/data on sys.path)

MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
if MODELS not in sys.path:
    sys.path.insert(0, MODELS)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model1_transformer.model import Model1, enable_determinism  # noqa: E402
from model1_transformer.spec import AdapterSpec, ModelConfig  # noqa: E402
from prep import m1_masks  # noqa: E402

enable_determinism()


def synthetic_spec_and_applicable(schema=None):
    schema = schema or H.synthetic_schema()
    absent = H.synthetic_absent_entry(schema)
    applicable, _, _ = m1_masks.applicable_entry_mask(schema, absent)
    return AdapterSpec.from_d1(schema, applicable), applicable, schema


def make_model(spec, seed=0, dtype=torch.float32, cfg=ModelConfig()):
    return Model1.build(spec, cfg, seed=seed).to(dtype).eval()


def make_batch(applicable, B=2, T=6, seed=0, dtype=torch.float32, holes=0.0):
    """Random scaled values at effective entries, 0 elsewhere. `holes` also drops that share of applicable entries (dynamic missing)."""
    g = torch.Generator().manual_seed(seed)
    app = torch.as_tensor(applicable)
    N, C = app.shape
    mask = app[None, None].expand(B, T, N, C).clone()
    if holes:
        mask &= torch.rand(B, T, N, C, generator=g) >= holes
    tmask = mask[:, -1].clone()
    x = torch.where(mask, torch.randn(B, T, N, C, generator=g, dtype=dtype), torch.zeros((), dtype=dtype))
    y = torch.where(tmask, torch.randn(B, N, C, generator=g, dtype=dtype), torch.zeros((), dtype=dtype))
    return {"inputs": x, "input_mask": mask, "target": y, "target_mask": tmask, "applicable_entry": app.clone(),
            "labels": torch.randint(0, 5, (B,), generator=g)}
