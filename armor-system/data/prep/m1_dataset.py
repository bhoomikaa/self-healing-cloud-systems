"""M1-03 PyTorch Dataset / DataLoader for Model 1 over the real D1 tensors (design note sections 2.1, 3.2, 5.3, 12).

`Model1Data.load` verifies everything before a single window can be yielded: manifest hash, scaler hash and
fit-sample provenance, feature-cache hash, schema, MI-6 (no non-zero value outside the declared mask), and that the
manifest's window lists equal windows recomputed from the data. A mismatch raises.

Training iterators (`training_dataset`, `training_loader`) accept only objectives with `gradient=True`, which are
restricted to fitting windows. Calibration, validation and test windows are reachable only through
`evaluation_dataset`, which is never accepted by `training_loader`.
"""
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import loader, m1_masks, m1_split as ms, paths, preprocess as pp
from .common import read_json, sha256_file

ROLE_IDS = {r: i for i, r in enumerate(ms.ROLES)}
CATEGORY_IDS = {c: i for i, c in enumerate(ms.CATEGORIES)}
OPTIONAL_INT_KEYS = ("case_index", "fault_class", "group_id")


class LeakageError(RuntimeError):
    """A calibration, validation, test or embargo observation was requested for a gradient step."""


def default_paths(art_root=paths.ART):
    return {"manifest": os.path.join(art_root, "splits", f"{ms.MANIFEST_NAME}.json"),
            "scaler": os.path.join(art_root, "preprocessing", f"{ms.SCALER_NAME}.npz"),
            "schema": os.path.join(art_root, "inspection", "schema.json"),
            "cases": os.path.join(art_root, "splits", "case_split_assignments.csv"),
            "outer": os.path.join(art_root, "splits", "model1_split_101_25_84_v1.json"),
            "train_period_scaler": os.path.join(art_root, "preprocessing", "scaler_model1_train_period_v1.npz"),
            "cache": paths.CACHE_FILE}


class Model1Data:
    """Verified, pre-transformed D1 arrays plus the window table. Immutable by convention."""

    def __init__(self, Xt, ts, table, cases, manifest, scaler, applicable, indicators, schema, observed=None, cfg=ms.SplitConfig(), boundaries=None):
        self.Xt, self.ts, self.table, self.cases = Xt, ts, table, cases
        self.manifest, self.scaler, self.applicable, self.schema = manifest, scaler, applicable, schema
        self.structural_absent, self.observed_all_zero = indicators
        self.observed, self.cfg, self.boundaries = observed, cfg, boundaries
        if observed is not None:
            m1_masks.validate_bool_mask(observed, Xt.shape, "observed_entry")
        m1_masks.validate_bool_mask(applicable, Xt.shape[1:], "applicable_entry")
        if (Xt[:, ~applicable] != 0).any():
            raise m1_masks.MaskError("transformed data holds non-zero values at unavailable entries")

    @classmethod
    def load(cls, art_root=paths.ART, cache=paths.CACHE_FILE, verify_cache_hash=True):
        p = default_paths(art_root)
        manifest = read_json(p["manifest"])
        if manifest.get("content_sha256") != ms.manifest_content_hash(manifest):
            raise ms.ManifestInvariantError("calibration manifest content hash does not match its body (edited by hand?)")
        if manifest.get("status") != "approved":
            raise ms.ManifestInvariantError("calibration manifest is not approved")
        if sha256_file(p["scaler"]) != manifest["fit_only_scaler"]["file_sha256"]:
            raise ms.ManifestInvariantError("fit-only scaler file hash differs from the manifest")
        schema = read_json(p["schema"])
        with open(p["cases"], newline="") as fh:
            cases = ms.load_case_table(list(csv.DictReader(fh)))
        boundaries = ms.approved_boundaries(read_json(p["outer"]), cases)
        pr = manifest["protocol"]
        cfg = ms.SplitConfig(tc=pr["Tc"], embargo_s=pr["embargo_s"], pre_guard_s=pr["policy"]["pre_guard_s"], post_guard_s=pr["policy"]["post_guard_s"])
        ts, X, _ = loader.load_arrays(cache)
        issues = pp.validate_schema(X, ts, schema)
        if issues:
            raise m1_masks.SchemaError("feature cache fails schema validation: " + "; ".join(issues))
        if verify_cache_hash and ms.feature_cache_digest(ts, X) != manifest["inputs"]["feature_cache_content_sha256"]:
            raise ms.ManifestInvariantError("feature cache differs from the one the manifest was generated from")
        z = np.load(p["scaler"])
        arrays = {k: z[k] for k in z.files}
        ms.check_scaler_provenance(arrays, ts, cfg)
        fresh = ms.fit_scaler_fit_only(X, ts, schema["modality_index_ranges"], cfg)      # the file must be exactly what the fitting samples give
        for k in ("mean", "scale", "constant", "absent_block", "absent_entry", "fit_timestamps"):
            if arrays[k].shape != fresh[k].shape or (arrays[k] != fresh[k]).any():
                raise ms.ManifestInvariantError(f"scaler array '{k}' differs from a fit on the fitting samples only (forged or stale scaler)")
        scaler = ms.scaler_from_arrays(arrays)
        applicable, _, report = m1_masks.applicable_entry_mask(schema, scaler["absent_entry"])
        m1_masks.check_expected_counts(report)
        m1_masks.check_mask_stability(X, applicable, {"fit": ts < cfg.tc - cfg.embargo_s, "calibration_region": (ts >= cfg.tc) & (ts < boundaries[0]),
                                                      "validation": (ts >= boundaries[0]) & (ts < boundaries[1]), "test": ts >= boundaries[1]}, schema)
        table = ms.compute_split(ts, cases, boundaries, cfg)
        ms.assert_leakage_free(table, cases, boundaries, cfg)
        ms.assert_approved_counts(ms.summarize(table, cases, ts, boundaries, cfg))      # v0 loads only the approved split
        ms.verify_manifest_against_table(manifest, table)
        Xt, ind = pp.transform(X, scaler, schema["modality_index_ranges"])
        Xt = m1_masks.mask_values(Xt, applicable)
        indicators = (ind["structural_absent"][0].astype(bool), ind["observed_all_zero_not_structural"].astype(bool))
        return cls(Xt, ts, table, cases, manifest, scaler, applicable, indicators, schema, None, cfg, boundaries)

    def selection(self, roles, categories=None, drop_ambiguous=False):
        t = self.table
        sel = np.isin(t.role, list(roles))
        if categories is not None:
            sel &= np.isin(t.category, list(categories))
        if drop_ambiguous:
            sel &= np.array([x is None for x in t.exclusion_reason], bool)
        return np.nonzero(sel)[0]


class Model1WindowDataset(Dataset):
    """Map-style dataset over a fixed selection of windows (row indices of the window table)."""

    def __init__(self, data, rows, name="custom", allowed_roles=None):
        rows = np.asarray(rows, np.int64)
        t = data.table
        if (t.role[rows] == "embargo").any():
            raise LeakageError("embargo windows are never yielded")
        if allowed_roles is not None and not set(t.role[rows]) <= set(allowed_roles):
            raise LeakageError(f"{name}: roles {sorted(set(t.role[rows]) - set(allowed_roles))} are not allowed here")
        if not m1_split_region_ok(data, rows):
            raise ms.ManifestInvariantError(f"{name}: a selected window is not wholly inside its region")
        if not (np.diff(t.T[rows], axis=1) == data.cfg.step_s).all():
            raise ValueError(f"{name}: a window is not made of consecutive one-minute samples")
        self.data, self.rows, self.name = data, rows, name
        self.n_input = data.cfg.window

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        d, k = self.data, int(self.rows[i])
        t, s, w = d.table, int(d.table.start[k]), self.n_input
        obs_all = np.ones((w + 1,) + d.applicable.shape, bool) if d.observed is None else d.observed[s:s + w + 1]
        eff = obs_all & d.applicable                                    # (w+1, 46, 130): applicable AND observed
        vals = m1_masks.mask_values(d.Xt[s:s + w + 1], eff)             # unavailable -> 0 before any model sees it
        role, cat = str(t.role[k]), str(t.category[k])
        return {
            "inputs": torch.from_numpy(vals[:w]), "target": torch.from_numpy(vals[w]),
            "input_mask": torch.from_numpy(eff[:w]), "target_mask": torch.from_numpy(eff[w]),
            "applicable_entry": torch.from_numpy(d.applicable.copy()), "observed_entry": torch.from_numpy(obs_all[:w].copy()),
            "structural_absent": torch.from_numpy(d.structural_absent.copy()),
            "observed_all_zero_not_structural": torch.from_numpy(d.observed_all_zero[s:s + w].copy()),
            "timestamps": torch.from_numpy(t.T[k].astype(np.int64)), "start_index": torch.tensor(s, dtype=torch.int64),
            "role": role, "category": cat, "role_id": torch.tensor(ROLE_IDS[role]), "category_id": torch.tensor(CATEGORY_IDS[cat]),
            "case_index": int(t.case_index[k]), "fault_class": int(t.fault_class[k]),
            "group_id": int(d.cases["group_id"][t.case_index[k]]) if t.case_index[k] >= 0 else -1,
            "fault_label_ambiguous": torch.tensor(bool(t.ambiguous[k])),
            "fault_label_exclusion_reason": t.exclusion_reason[k],
        }


def m1_split_region_ok(data, rows):
    return bool(ms.region_ok(_subtable(data.table, rows), data.boundaries, data.cfg).all())


def _subtable(t, rows):
    return ms.WindowTable(t.start[rows], t.T[rows], t.period[rows], t.role[rows], t.category[rows], t.case_index[rows], t.fault_class[rows], t.ambiguous[rows], t.exclusion_reason[rows], t.match[rows])


def collate_windows(items):
    """Stack tensors, keep strings as lists, and tolerate missing (None) case metadata.

    `case_index`, `fault_class`, `group_id` become int64 tensors with -1 where absent, plus `<key>_present` bool tensors.
    """
    if not items:
        raise ValueError("cannot collate an empty batch")
    batch = {}
    for key in items[0]:
        vals = [it.get(key) for it in items]
        if key in OPTIONAL_INT_KEYS:
            batch[key] = torch.tensor([-1 if v is None else int(v) for v in vals], dtype=torch.int64)
            batch[f"{key}_present"] = torch.tensor([v is not None and int(v) >= 0 for v in vals])
        elif isinstance(vals[0], torch.Tensor):
            batch[key] = torch.stack(vals)
        else:
            batch[key] = list(vals)
    return batch


def objective_rows(data, objective):
    o = ms.OBJECTIVES[objective]
    return data.selection(o["roles"], o["categories"], o["drop_ambiguous"])


def training_dataset(data, objective):
    """Dataset for a gradient objective. Only objectives restricted to fitting windows are accepted."""
    o = ms.OBJECTIVES.get(objective)
    if o is None:
        raise KeyError(f"unknown objective {objective!r}; choose from {sorted(ms.OBJECTIVES)}")
    if not o["gradient"] or set(o["roles"]) != {"fit"}:
        raise LeakageError(f"objective {objective!r} (roles {o['roles']}) is not a gradient objective over fitting windows")
    ds = Model1WindowDataset(data, objective_rows(data, objective), objective, allowed_roles=("fit",))
    ds.training = True
    return ds


def evaluation_dataset(data, objective):
    """Dataset for a non-gradient objective (calibration, validation, test, diagnostics). Not accepted by `training_loader`."""
    o = ms.OBJECTIVES.get(objective)
    if o is None:
        raise KeyError(f"unknown objective {objective!r}; choose from {sorted(ms.OBJECTIVES)}")
    if o["gradient"]:
        raise ValueError(f"objective {objective!r} is a training objective; use training_dataset")
    ds = Model1WindowDataset(data, objective_rows(data, objective), objective)
    ds.training = False
    return ds


def _training_collate(items):
    batch = collate_windows(items)
    if set(batch["role"]) != {"fit"}:
        raise LeakageError(f"training batch holds roles {sorted(set(batch['role']))}; only 'fit' is allowed")
    return batch


def training_loader(dataset, batch_size=64, seed=0, shuffle=True, drop_last=False, num_workers=0):
    """Seeded DataLoader over a `training_dataset`. Every batch is re-checked to contain only fitting windows."""
    if not getattr(dataset, "training", False):
        raise LeakageError("training_loader accepts only datasets created by training_dataset()")
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g, drop_last=drop_last,
                      num_workers=num_workers, collate_fn=_training_collate)


def evaluation_loader(dataset, batch_size=64, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_windows)
