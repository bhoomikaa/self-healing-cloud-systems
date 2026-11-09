"""Paired Stage 2 batches for v1 (V1_PROTOCOL.md section 3): 64 fitting event windows + 64 fitting normal windows per step.

Both halves come from `training_dataset` objects (fitting windows only). An epoch is one pass over the eligible event windows
(`s2_fault_finetune`: 523, the ambiguous conflicting-label window excluded); normal windows (`s1_pretrain`, 1,312) are drawn from a
shuffled permutation that is re-shuffled when exhausted. `ignore` windows, ambiguous windows and every calibration, validation or test
window are rejected by `check_paired_batch` before a step is taken. Generators are seeded, so batches are reproducible.
"""
import math

import torch

from prep import m1_dataset as md

from .training import LeakageError


def check_paired_batch(batch):
    """Raise LeakageError unless the batch is exactly n event windows (label 1) + n normal windows (label 0), all fitting, none ambiguous."""
    if set(batch["role"]) != {"fit"}:
        raise LeakageError(f"s2: batch holds roles {sorted(set(batch['role']))}; only 'fit' windows may update weights")
    label = batch["det_label"]
    cats = batch["category"]
    if len(cats) != len(label) or int(label.sum()) * 2 != len(label):
        raise LeakageError("s2: a paired batch must hold equally many event and normal windows")
    for c, y in zip(cats, label.tolist()):
        if c != ("event_associated" if y == 1 else "normal_reference"):
            raise LeakageError(f"s2: category {c!r} does not match detection label {y} (ignore and mislabelled windows are refused)")
    if batch["fault_label_ambiguous"].any():
        raise LeakageError("s2: an ambiguous-label window reached a training batch")
    ev = label == 1
    if not batch["fault_class_present"][ev].all() or not ((batch["fault_class"][ev] >= 0) & (batch["fault_class"][ev] < 5)).all():
        raise LeakageError("s2: an event window has no valid fault label")


class PairedLoader:
    def __init__(self, data, batch_size=64, seed=0):
        self.event_ds = md.training_dataset(data, "s2_fault_finetune")
        self.normal_ds = md.training_dataset(data, "s1_pretrain")
        self.batch_size = batch_size
        self.g_event = torch.Generator().manual_seed(int(seed))
        self.g_normal = torch.Generator().manual_seed(int(seed) + 10_000)
        self._perm, self._pos = None, 0

    def __len__(self):
        return math.ceil(len(self.event_ds) / self.batch_size)

    def _normals(self, n):
        out = []
        while len(out) < n:
            if self._perm is None or self._pos >= len(self._perm):
                self._perm, self._pos = torch.randperm(len(self.normal_ds), generator=self.g_normal).tolist(), 0
            take = min(n - len(out), len(self._perm) - self._pos)
            out += self._perm[self._pos:self._pos + take]
            self._pos += take
        return out

    def __iter__(self):
        perm = torch.randperm(len(self.event_ds), generator=self.g_event).tolist()
        for i in range(0, len(perm), self.batch_size):
            ev = perm[i:i + self.batch_size]
            nm = self._normals(len(ev))
            batch = md.collate_windows([self.event_ds[j] for j in ev] + [self.normal_ds[j] for j in nm])
            batch["det_label"] = torch.tensor([1] * len(ev) + [0] * len(nm), dtype=torch.int64)
            check_paired_batch(batch)
            yield batch
