"""v1 training (V1_PROTOCOL.md section 3): Stage 1 with simulated missing, Stage 2 with paired batches and L_fault + L_det.

New file: the committed v0 training module (and its evidence hash) is untouched; the loop below is the v0 loop plus (a) the epoch passed to
the loss (the simulated-missing draws depend on it), (b) the paired-batch guard, (c) optional per-epoch loss components in the history.
No wall-clock values are logged, so histories are reproducible. All v0 failure checks apply (non-finite loss or gradient stops before the step).
"""
import math

import torch
import torch.nn.functional as F

from prep import m1_dataset as md

from . import training as T
from .losses import masked_mse
from .model import enable_determinism
from .model_v1 import Model1V1
from .paired import PairedLoader, check_paired_batch
from .sim_missing import MissingBlocks, simulate_missing
from .spec import ModelConfig
from .weighting import CaseBalancedWeights, weighted_window_cross_entropy


def check_batch_v1(batch, stage):
    if stage == "s1":
        return T.check_batch(batch, "s1")
    if stage == "s2":
        return check_paired_batch(batch)
    raise ValueError(f"unknown stage {stage!r}")


def build_model_v1(spec, seed, device):
    enable_determinism()
    torch.manual_seed(T.validate_seed(seed))
    return Model1V1.build(spec, ModelConfig(), seed=seed).to(device)


def train_epochs_v1(model, loader, loss_fn, optimizer, *, epochs, clip_norm, device, stage, scheduler=None, on_epoch_end=None):
    """`loss_fn(model, batch, epoch) -> loss` or `(loss, {name: float})`. Returns the per-epoch history."""
    params = [p for p in model.parameters() if p.requires_grad]
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        n_batches = n_windows = clipped = 0
        loss_sum = gn_sum = gn_max = 0.0
        parts_sum = {}
        for batch in loader:
            check_batch_v1(batch, stage)
            batch = T._to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            res = loss_fn(model, batch, epoch)
            loss, parts = res if isinstance(res, tuple) else (res, {})
            if not torch.isfinite(loss):
                raise T.TrainingFailure(f"{stage}: non-finite loss {loss.item()} at epoch {epoch}")
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(params, clip_norm))
            if not math.isfinite(gn):
                raise T.TrainingFailure(f"{stage}: non-finite gradient norm at epoch {epoch}")
            optimizer.step()
            b = len(batch["role"])
            n_batches, n_windows = n_batches + 1, n_windows + b
            loss_sum, gn_sum, gn_max, clipped = loss_sum + loss.item() * b, gn_sum + gn, max(gn_max, gn), clipped + (gn > clip_norm)
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v * b
        if n_batches == 0:
            raise T.TrainingFailure(f"{stage}: epoch {epoch} had no batch")
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise T.TrainingFailure(f"{stage}: non-finite parameter after epoch {epoch}")
        entry = {"stage": stage, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "loss": loss_sum / n_windows, "batches": n_batches, "windows": n_windows,
                 "grad_norm_mean": gn_sum / n_batches, "grad_norm_max": gn_max, "clipped_batches": int(clipped)}
        entry.update({f"train_{k}": v / n_windows for k, v in parts_sum.items()})
        history.append(entry)
        if scheduler is not None:
            scheduler.step()
        if on_epoch_end is not None and on_epoch_end(epoch, entry, model):
            entry["stopped_early"] = True
            break
    return history


def make_s1_loss(blocks, seed):
    def loss_fn(model, batch, epoch):
        sm = simulate_missing(batch, blocks, seed, "s1", epoch)
        out = model(sm["inputs"], sm["input_mask"], applicable_entry=batch["applicable_entry"])
        return masked_mse(out["prediction"], batch["target"], batch["target_mask"])           # targets untouched
    return loss_fn


def make_s2_loss(blocks, seed, weights):
    def loss_fn(model, batch, epoch):
        sm = simulate_missing(batch, blocks, seed, "s2", epoch)
        out = model(sm["inputs"], sm["input_mask"], applicable_entry=batch["applicable_entry"])
        ev = batch["det_label"] == 1
        w = weights.per_window(batch["case_index"][ev].cpu(), normalised=True, dtype=out["logits"].dtype).to(out["logits"].device)
        l_fault = weighted_window_cross_entropy(out["logits"][ev], batch["fault_class"][ev], w)
        l_det = F.binary_cross_entropy_with_logits(out["detection_logit"], batch["det_label"].to(out["detection_logit"].dtype))
        return l_fault + l_det, {"l_fault": l_fault.item(), "l_det": l_det.item()}
    return loss_fn


def train_stage1_v1(data, cfg, seed, spec, device="cpu", on_epoch_end=None, epochs=None, max_batches_per_epoch=None):
    dev = torch.device(device)
    sc = cfg.s1
    model = build_model_v1(spec, seed, dev)
    loader = md.training_loader(md.training_dataset(data, "s1_pretrain"), batch_size=sc.batch_size, seed=seed)
    loader = T._Limited(loader, max_batches_per_epoch) if max_batches_per_epoch else loader
    opt = T.make_optimizer(model, sc)
    hist = train_epochs_v1(model, loader, make_s1_loss(MissingBlocks(spec), seed), opt, epochs=epochs or sc.epochs, clip_norm=sc.clip_norm, device=dev, stage="s1",
                           scheduler=T.make_scheduler(opt, sc), on_epoch_end=on_epoch_end)
    return model, hist


def train_stage2_v1(model, data, cfg, seed, device="cpu", on_epoch_end=None, epochs=None, max_batches_per_epoch=None, weights=None):
    dev = torch.device(device)
    sc = cfg.s2
    weights = weights or CaseBalancedWeights.from_model1_data(data)
    torch.manual_seed(T.validate_seed(seed))
    loader = PairedLoader(data, batch_size=sc.batch_size, seed=seed)
    loader = T._Limited(loader, max_batches_per_epoch) if max_batches_per_epoch else loader
    opt = T.make_optimizer(model, sc)
    hist = train_epochs_v1(model, loader, make_s2_loss(MissingBlocks(model.spec), seed, weights), opt, epochs=epochs or sc.epochs, clip_norm=sc.clip_norm, device=dev, stage="s2",
                           scheduler=T.make_scheduler(opt, sc), on_epoch_end=on_epoch_end)
    return model, hist, weights
