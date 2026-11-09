"""Masked next-step MSE and class-weighted fault cross-entropy (design note 4.3). No optimizer, no training loop."""
import torch
import torch.nn.functional as F


class LossInputError(ValueError):
    """A loss was given an invalid label, shape or class-count vector."""


def masked_mse(pred, target, mask):
    """Mean squared error over the entries where `mask` is True; 0 (with finite zero gradients) if no entry is effective.

    Both sides are cleaned with `torch.where` BEFORE the subtraction, so a masked target of any size (even inf/NaN)
    and a masked prediction cannot produce inf*0 = NaN in the backward pass.
    """
    if pred.shape != target.shape or mask.shape != pred.shape or mask.dtype != torch.bool:
        raise LossInputError(f"pred, target and a bool mask must share one shape, got {tuple(pred.shape)}, {tuple(target.shape)}, {mask.dtype} {tuple(mask.shape)}")
    zero = torch.zeros_like(pred)
    p, t = torch.where(mask, pred, zero), torch.where(mask, target.to(pred.dtype), zero)
    n = mask.sum()
    return ((p - t) ** 2).sum() / n.clamp(min=1).to(pred.dtype)


def inverse_frequency_weights(case_counts):
    """w_k = N / (K * n_k) from fitting-case counts. Rejects an empty class instead of dividing by zero."""
    n = torch.as_tensor(case_counts, dtype=torch.float64)
    if n.ndim != 1 or (n <= 0).any():
        raise LossInputError(f"case counts must be a 1-D vector of positive numbers, got {case_counts}")
    return (n.sum() / (len(n) * n)).to(torch.float32)


def fit_class_weights(manifest, n_classes=5, n_fit_cases=88):
    """Class weights from the 88 FITTING cases of the approved manifest only."""
    fit = manifest["counts"]["roles"]["fit"]
    counts = fit["cases_by_class"]
    if fit["cases"] != n_fit_cases or sum(counts) != n_fit_cases or len(counts) != n_classes:
        raise LossInputError(f"manifest fit counts {counts} (cases {fit['cases']}) are not the approved {n_fit_cases} cases in {n_classes} classes")
    return inverse_frequency_weights(counts)


def weighted_cross_entropy(logits, labels, class_weights):
    """Weighted CE (weighted mean: sum w_i nll_i / sum w_i). Labels must be real classes; -1 (no label) is rejected."""
    if logits.ndim != 2 or labels.shape != (logits.shape[0],) or logits.shape[0] == 0:
        raise LossInputError(f"logits must be (B,K) with B>0 and labels (B,), got {tuple(logits.shape)} and {tuple(labels.shape)}")
    if labels.dtype != torch.int64:
        raise LossInputError(f"labels must be int64, got {labels.dtype}")
    K = logits.shape[1]
    if class_weights.shape != (K,):
        raise LossInputError(f"class_weights must have shape ({K},), got {tuple(class_weights.shape)}")
    bad = (labels < 0) | (labels >= K)
    if bad.any():
        raise LossInputError(f"invalid fault labels {sorted(set(labels[bad].tolist()))}: only classes 0..{K - 1} are real (-1 means 'no label')")
    return F.cross_entropy(logits, labels, weight=class_weights.to(logits.dtype))
