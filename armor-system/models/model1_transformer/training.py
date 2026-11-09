"""Minimal two-stage training infrastructure for Model 1 (Linear 298-51). No tuning, no evaluation of real validation data here.

S1: self-supervised next-step masked MSE on fitting `normal_reference` windows. S2: case-balanced class-weighted fault
cross-entropy on fitting `event_associated` windows, encoder initialised from the S1 checkpoint of the same seed.
Gradient steps are only possible on batches from `training_loader` (fitting windows); `check_batch` refuses anything else.
"""
import hashlib
import itertools
import json
import math
import os
from dataclasses import dataclass

import torch

from . import losses as L
from .model import Model1, enable_determinism
from .spec import AdapterSpec, ModelConfig
from .weighting import CaseBalancedWeights, weighted_window_cross_entropy

SEEDS = (0, 1, 2, 3, 4)
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_config_v1.json")
STAGE_OBJECTIVE = {"s1": ("s1_pretrain", "normal_reference"), "s2": ("s2_fault_finetune", "event_associated")}


class TrainingFailure(RuntimeError):
    """A non-finite loss/gradient/parameter or an empty epoch: training stops instead of continuing silently."""


class LeakageError(RuntimeError):
    """A batch holds a window that may not produce a gradient (not a fitting window, or the wrong objective's category)."""


@dataclass(frozen=True)
class StageConfig:
    lr: float
    weight_decay: float
    batch_size: int
    clip_norm: float
    schedule: str
    epochs: int
    patience: int

    def __post_init__(self):
        if self.schedule not in ("cosine", "constant"):
            raise ValueError(f"unknown schedule {self.schedule!r}")
        if not (self.lr > 0 and self.weight_decay >= 0 and self.batch_size > 0 and self.clip_norm > 0 and self.epochs > 0 and self.patience > 0):
            raise ValueError(f"invalid stage configuration {self}")


@dataclass(frozen=True)
class TrainConfig:
    s1: StageConfig
    s2: StageConfig
    seeds: tuple
    sha256: str

    def stage(self, name):
        return {"s1": self.s1, "s2": self.s2}[name]


def canonical_sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_train_config(path=CONFIG_PATH):
    with open(path) as fh:
        raw = json.load(fh)
    fields = ("lr", "weight_decay", "batch_size", "clip_norm", "schedule", "epochs", "patience")
    cfg = TrainConfig(StageConfig(**{k: raw["s1"][k] for k in fields}), StageConfig(**{k: raw["s2"][k] for k in fields}), tuple(raw["seeds"]), canonical_sha256(raw))
    if cfg.seeds != SEEDS:
        raise ValueError(f"the approved seeds are {SEEDS}, the configuration lists {cfg.seeds}")
    return cfg


def validate_seed(seed):
    if seed not in SEEDS:
        raise ValueError(f"seed {seed!r} is not one of the approved seeds {SEEDS}")
    return seed


def select_device(preference="auto"):
    """'cpu', 'cuda' or 'auto' (cuda when available, else cpu). MPS is deliberately not offered: no determinism guarantee."""
    if preference not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device preference must be auto, cpu or cuda, got {preference!r}")
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but not available")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")          # required by deterministic algorithms on CUDA
        return torch.device("cuda")
    return torch.device("cpu")


def make_optimizer(model, sc):
    return torch.optim.AdamW(model.parameters(), lr=sc.lr, weight_decay=sc.weight_decay)


def make_scheduler(optimizer, sc):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sc.epochs) if sc.schedule == "cosine" else None


def check_batch(batch, stage):
    """Only fitting windows of the stage's own category may produce a gradient."""
    roles = set(batch["role"])
    if roles != {"fit"}:
        raise LeakageError(f"{stage}: batch holds roles {sorted(roles)}; only 'fit' windows may update weights")
    cats = set(batch["category"])
    if cats != {STAGE_OBJECTIVE[stage][1]}:
        raise LeakageError(f"{stage}: batch holds categories {sorted(cats)}; expected only {STAGE_OBJECTIVE[stage][1]!r}")


class EarlyStopper:
    """Patience counter on a monitored metric (non-gradient use of validation, design section 6). mode 'min' or 'max'."""

    def __init__(self, patience, mode="min"):
        if mode not in ("min", "max") or patience < 1:
            raise ValueError("mode must be 'min' or 'max' and patience >= 1")
        self.patience, self.mode, self.best, self.best_epoch, self.bad = patience, mode, None, None, 0

    def update(self, epoch, value):
        """Record `value`; return True when training should stop."""
        if not math.isfinite(value):
            raise TrainingFailure(f"non-finite monitored metric at epoch {epoch}")
        better = self.best is None or (value < self.best if self.mode == "min" else value > self.best)
        if better:
            self.best, self.best_epoch, self.bad = value, epoch, 0
        else:
            self.bad += 1
        return self.bad >= self.patience


class _Limited:
    """First `n` batches of every epoch (only used by short smoke runs and tests)."""

    def __init__(self, loader, n):
        self.loader, self.n = loader, n

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.n)


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_epochs(model, loader, loss_fn, optimizer, *, epochs, clip_norm, device, stage="s1", scheduler=None, max_steps=None, on_epoch_end=None):
    """Generic loop. `loss_fn(model, batch) -> loss`. Returns the per-epoch history (no wall-clock values: logs are reproducible).

    Failure checks (each raises TrainingFailure BEFORE the optimizer step it would corrupt): non-finite loss, non-finite
    gradient norm; at the end of an epoch every parameter must be finite; an empty loader is an error.
    `on_epoch_end(epoch, entry, model) -> bool` may stop training (early stopping); `max_steps` truncates (tests/smoke).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    history, step = [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        n_batches = n_windows = clipped = 0
        loss_sum = gn_sum = gn_max = 0.0
        for batch in loader:
            check_batch(batch, stage)
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch)
            if not torch.isfinite(loss):
                raise TrainingFailure(f"{stage}: non-finite loss {loss.item()} at epoch {epoch}, step {step + 1}")
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(params, clip_norm))
            if not math.isfinite(gn):
                raise TrainingFailure(f"{stage}: non-finite gradient norm at epoch {epoch}, step {step + 1}")
            optimizer.step()
            b = len(batch["role"])
            n_batches, n_windows, step = n_batches + 1, n_windows + b, step + 1
            loss_sum, gn_sum, gn_max, clipped = loss_sum + loss.item() * b, gn_sum + gn, max(gn_max, gn), clipped + (gn > clip_norm)
            if max_steps is not None and step >= max_steps:
                break
        if n_batches == 0:
            raise TrainingFailure(f"{stage}: epoch {epoch} had no batch (empty training selection)")
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise TrainingFailure(f"{stage}: non-finite parameter after epoch {epoch}")
        entry = {"stage": stage, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "loss": loss_sum / n_windows, "batches": n_batches, "windows": n_windows,
                 "grad_norm_mean": gn_sum / n_batches, "grad_norm_max": gn_max, "clipped_batches": int(clipped)}
        history.append(entry)
        if scheduler is not None:
            scheduler.step()
        if max_steps is not None and step >= max_steps:
            entry["truncated"] = True
            break
        if on_epoch_end is not None and on_epoch_end(epoch, entry, model):
            entry["stopped_early"] = True
            break
    return history


def evaluate_masked_mse(model, loader, device):
    """Exact dataset masked MSE (sum of squared errors / effective entries over all batches). No gradient, eval mode, mode restored.

    The S1 early-stopping metric when the loader is the validation `normal_reference` objective. NOT run on real
    validation data in milestone 298-51.
    """
    was_training = model.training
    model.eval()
    sq, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            pred = model(batch["inputs"], batch["input_mask"], applicable_entry=batch["applicable_entry"])["prediction"]
            m = batch["target_mask"]
            p64, t64 = pred.double(), batch["target"].double()                   # subtract in float64: exact accumulation
            sq += float(((torch.where(m, p64, torch.zeros_like(p64)) - torch.where(m, t64, torch.zeros_like(t64))) ** 2).sum())
            n += int(m.sum())
    model.train(was_training)
    if n == 0:
        raise TrainingFailure("evaluation loader has no effective entry")
    return sq / n


def s1_loss_fn(model, batch):
    out = model(batch["inputs"], batch["input_mask"], applicable_entry=batch["applicable_entry"])
    return L.masked_mse(out["prediction"], batch["target"], batch["target_mask"])


def make_s2_loss_fn(weights):
    def s2_loss_fn(model, batch):
        out = model(batch["inputs"], batch["input_mask"], applicable_entry=batch["applicable_entry"])
        w = weights.per_window(batch["case_index"].cpu(), normalised=True, dtype=out["logits"].dtype).to(out["logits"].device)
        return weighted_window_cross_entropy(out["logits"], batch["fault_class"], w)
    return s2_loss_fn


def build_model(spec, seed, device):
    enable_determinism()
    torch.manual_seed(validate_seed(seed))
    return Model1.build(spec, ModelConfig(), seed=seed).to(device)


def train_stage1(data, cfg, seed, *, spec=None, device="cpu", max_steps=None, on_epoch_end=None, epochs=None, max_batches_per_epoch=None):
    """S1 from a fresh seeded model. Returns (model, history). Uses only the fitting `normal_reference` windows."""
    from prep import m1_dataset as md
    dev = torch.device(device)
    spec = spec or AdapterSpec.from_d1(data.schema, data.applicable)
    sc = cfg.s1
    model = build_model(spec, seed, dev)
    loader = md.training_loader(md.training_dataset(data, "s1_pretrain"), batch_size=sc.batch_size, seed=seed)
    loader = _Limited(loader, max_batches_per_epoch) if max_batches_per_epoch else loader
    opt = make_optimizer(model, sc)
    hist = train_epochs(model, loader, s1_loss_fn, opt, epochs=epochs or sc.epochs, clip_norm=sc.clip_norm, device=dev, stage="s1",
                        scheduler=make_scheduler(opt, sc), max_steps=max_steps, on_epoch_end=on_epoch_end)
    return model, hist


def train_stage2(model, data, cfg, seed, *, device="cpu", max_steps=None, on_epoch_end=None, epochs=None, weights=None, max_batches_per_epoch=None):
    """S2 fine-tuning of `model` (normally loaded from the S1 checkpoint of the same seed). All weights trainable, fresh AdamW."""
    from prep import m1_dataset as md
    dev = torch.device(device)
    sc = cfg.s2
    weights = weights or CaseBalancedWeights.from_model1_data(data)
    torch.manual_seed(validate_seed(seed))                                # dropout stream of S2 depends on the seed only
    loader = md.training_loader(md.training_dataset(data, "s2_fault_finetune"), batch_size=sc.batch_size, seed=seed)
    loader = _Limited(loader, max_batches_per_epoch) if max_batches_per_epoch else loader
    opt = make_optimizer(model, sc)
    hist = train_epochs(model, loader, make_s2_loss_fn(weights), opt, epochs=epochs or sc.epochs, clip_norm=sc.clip_norm, device=dev, stage="s2",
                        scheduler=make_scheduler(opt, sc), max_steps=max_steps, on_epoch_end=on_epoch_end)
    return model, hist, weights
