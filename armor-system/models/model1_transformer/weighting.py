"""Case-balanced class weighting for the S2 fault objective (Linear 298-51).

For every eligible fitting fault window i of case c with class k:

    w_i = class_weight[k] / n_c          class_weight[k] = N / (K * n_k),  n_k = fitting CASES of class k (33/29/9/13/4)
                                         n_c = eligible windows of case c (the ambiguous conflicting-label window excluded)

so the windows of one case sum to that case's class weight, a case counts once whatever its number of windows, and the
class weight is applied exactly once (the per-window cross-entropy below is computed with reduction="none" and NO
`weight=` argument). Over all eligible windows the weights sum to N = 88 (sum_k n_k * N/(K n_k) = N).

Normalisation and minibatches. The epoch objective is  L = (1/M) * sum_i  w~_i * ce_i  with  w~_i = w_i * M / sum_j w_j
(M = eligible windows = 523, sum_j w_j = 88), i.e. weights rescaled to mean 1 over the training windows, so the loss has
the scale of an ordinary mean cross-entropy. A minibatch loss is the mean over the batch of  w~_i * ce_i  with a FIXED
denominator (the batch size): its expectation over a uniformly shuffled batch equals L, and a full pass over all windows
(drop_last=False) reproduces L exactly. Unlike a weighted mean (divide by the batch's own weight sum) the scale of a
batch does not depend on which classes it happens to hold; the price is that a batch of mostly rare-class windows has a
larger gradient, which is exactly what the class weights ask for.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .losses import LossInputError


class WeightError(ValueError):
    """The window table cannot give valid case-balanced weights (wrong role, unknown case, empty class, ...)."""


def class_weights_from_case_counts(counts):
    """float64 vector N / (K * n_k). Rejects a class without a fitting case."""
    n = np.asarray(counts, np.float64)
    if n.ndim != 1 or len(n) < 2 or (n <= 0).any() or (n != np.round(n)).any():
        raise WeightError(f"case counts must be positive integers per class, got {counts}")
    return n.sum() / (len(n) * n)


class CaseBalancedWeights:
    """Per-case window weights from the eligible fitting windows. Arrays are aligned: one entry per eligible window."""

    def __init__(self, case_index, window_class, n_classes=5):
        case_index, window_class = np.asarray(case_index, np.int64), np.asarray(window_class, np.int64)
        if case_index.ndim != 1 or case_index.shape != window_class.shape or len(case_index) == 0:
            raise WeightError("case_index and window_class must be non-empty 1-D arrays of equal length")
        if (case_index < 0).any():
            raise WeightError("a window without a case (case_index < 0) is not eligible for the fault objective")
        if ((window_class < 0) | (window_class >= n_classes)).any():
            raise WeightError(f"window classes must lie in 0..{n_classes - 1} (-1 means no label)")
        cases, inv = np.unique(case_index, return_inverse=True)
        case_class = np.empty(len(cases), np.int64)
        for j in range(len(cases)):
            cls = np.unique(window_class[inv == j])
            if len(cls) != 1:
                raise WeightError(f"case {cases[j]} has windows of several classes {cls.tolist()}")
            case_class[j] = cls[0]
        self.n_classes = n_classes
        self.case_ids = cases
        self.case_class = case_class
        self.case_windows = np.bincount(inv, minlength=len(cases))                              # n_c
        self.class_counts = np.bincount(case_class, minlength=n_classes)                          # n_k (cases)
        self.class_weights = class_weights_from_case_counts(self.class_counts)                    # float64
        self.window_weight_by_case = self.class_weights[case_class] / self.case_windows           # raw w_i of every window of case c
        self.n_windows = int(len(case_index))
        self.total_weight = float((self.window_weight_by_case * self.case_windows).sum())        # = number of cases
        self.norm = self.n_windows / self.total_weight                                            # w~ = w * M / sum w
        self._lookup = {int(c): j for j, c in enumerate(cases)}

    @classmethod
    def from_model1_data(cls, data, expect_manifest=True):
        """From the approved S2 selection: fitting `event_associated` windows with the ambiguous window excluded."""
        from prep import m1_dataset as md
        rows = md.objective_rows(data, "s2_fault_finetune")
        t = data.table
        if not (t.role[rows] == "fit").all() or not (t.category[rows] == "event_associated").all():
            raise WeightError("the S2 selection holds a non-fitting or non-event window")
        if any(t.exclusion_reason[r] is not None or bool(t.ambiguous[r]) for r in rows):
            raise WeightError("the S2 selection holds an ambiguous-label window")
        case, cls_ = t.case_index[rows], t.fault_class[rows]
        if not (cls_ == data.cases["class_id"][case]).all():
            raise WeightError("window labels disagree with the case table")
        w = cls(case, cls_)
        if expect_manifest:
            fit = data.manifest["counts"]["roles"]["fit"]
            if w.class_counts.tolist() != fit["cases_by_class"] or len(w.case_ids) != fit["cases"]:
                raise WeightError(f"case counts {w.class_counts.tolist()} differ from the manifest's {fit['cases_by_class']}")
        return w

    def per_window(self, case_index, normalised=True, dtype=torch.float32):
        """Weight of every window of a batch from its case ids. Unknown cases (validation, calibration, test, -1) raise."""
        ids = torch.as_tensor(case_index).reshape(-1).tolist()
        try:
            j = [self._lookup[int(c)] for c in ids]
        except KeyError as e:
            raise WeightError(f"case {e.args[0]} is not one of the {len(self.case_ids)} fitting cases with eligible windows") from None
        w = self.window_weight_by_case[j] * (self.norm if normalised else 1.0)
        return torch.as_tensor(w, dtype=dtype)

    def summary(self):
        return {"cases": int(len(self.case_ids)), "eligible_windows": self.n_windows, "case_counts_by_class": self.class_counts.tolist(),
                "class_weights": [float(x) for x in self.class_weights], "total_weight": self.total_weight, "normalisation_factor": self.norm,
                "windows_per_case_min_max": [int(self.case_windows.min()), int(self.case_windows.max())]}


def weighted_window_cross_entropy(logits, labels, window_weights):
    """mean_i( w~_i * ce_i ) with per-window unreduced cross-entropy; the class weight lives only in `window_weights`."""
    if logits.ndim != 2 or labels.shape != (logits.shape[0],) or window_weights.shape != labels.shape or logits.shape[0] == 0:
        raise LossInputError(f"logits (B,K), labels (B,) and window_weights (B,) with B>0 expected, got {tuple(logits.shape)}, {tuple(labels.shape)}, {tuple(window_weights.shape)}")
    if labels.dtype != torch.int64:
        raise LossInputError(f"labels must be int64, got {labels.dtype}")
    K = logits.shape[1]
    bad = (labels < 0) | (labels >= K)
    if bad.any():
        raise LossInputError(f"invalid fault labels {sorted(set(labels[bad].tolist()))}: only classes 0..{K - 1} are real (-1 means 'no label')")
    if not torch.isfinite(window_weights).all() or (window_weights <= 0).any():
        raise LossInputError("window weights must be finite and positive")
    ce = F.cross_entropy(logits, labels, reduction="none")
    return (window_weights.to(ce.dtype) * ce).mean()
