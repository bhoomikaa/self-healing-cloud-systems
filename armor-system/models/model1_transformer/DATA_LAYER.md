# Model 1 data layer (M1-03, Linear 298-48)

Implements section 12 of [MODEL1_V0_DESIGN.md](MODEL1_V0_DESIGN.md). The design note is not changed by this work. Everything here is CPU-only, deterministic and uses no random numbers when it builds artifacts.

## What was added

| File | Purpose |
|---|---|
| `armor-system/data/prep/m1_masks.py` | Declared `applicable_entry` mask (3,152 entries), `observed_entry`, `mask_values`, MI-6 check, strict schema validation |
| `armor-system/data/prep/m1_split.py` | Policy B window table, fit / calibration roles, leakage checks, approved-count checks, fit-only scaler, manifest builder, deterministic NPZ/JSON writers, objective eligibility table |
| `armor-system/data/prep/m1_dataset.py` | `Model1Data.load` (verifies everything), `Model1WindowDataset`, `collate_windows`, `training_dataset`/`training_loader`, `evaluation_dataset`/`evaluation_loader` |
| `armor-system/data/prep/detection_calibration.py` | Median/IQR, epsilon, floor report, three scores, per-method percentile thresholds (no model needed) |
| `armor-system/data/model1_pipeline.py` | Regenerates the two artifacts below (`--check` compares, `--out-dir` writes elsewhere) |
| `armor-system/data/artifacts/d1/splits/model1_calibration_split_v1.json` | New manifest (approved v0, 2026-10-27) |
| `armor-system/data/artifacts/d1/preprocessing/scaler_model1_fit_only_v1.npz` | New scaler fitted on the 2,880 fitting samples only |
| `armor-system/data/m1_integration.py` | One real-data integration command (below) |
| `armor-system/data/prep/m1_evidence.py` | Builds and verifies `evidence_m1_03.json` |
| `armor-system/data/artifacts/d1/evidence/evidence_m1_03.json` | M1-03 evidence (hashes, command, test results, reproducibility, commit reference) |
| `armor-system/tests/prep/test_m1_*.py` | New tests (masks, split, dataset, calibration utility, evidence, required resources) |

Existing scaler files, the 101/25/84 split manifest, `old/`, the ARMOR baseline and Airflow are untouched.

## Regenerate and verify

Integration command (D1 files, clean cache rebuild, deterministic regeneration, all tests with skipping disabled, evidence check). Run from any directory:

```bash
.venv/bin/python -I armor-system/data/m1_integration.py                    # verify only, writes nothing
.venv/bin/python -I armor-system/data/m1_integration.py --write-evidence   # also writes evidence_m1_03.json
.venv/bin/python -I armor-system/data/m1_integration.py --update           # also rewrites the M1-03 artifacts if they differ
```

It stops with a non-zero status if D1 files do not match the approved checksums, the clean rebuild differs from the manifest's cache hash, two independent generation runs differ from each other or from the committed artifacts, any test is skipped, fails or errors, or the evidence no longer matches the tree. `M1_REQUIRE_REAL_DATA=1` (set by the command) turns every "skip if the data is missing" in the tests into a failure. Lower-level commands:

```bash
.venv/bin/python -I armor-system/data/model1_pipeline.py            # needs data/processed/d1_features_v1.npz (pipeline.py --stage cache)
.venv/bin/python -I armor-system/data/model1_pipeline.py --check    # fresh regeneration must equal the committed artifacts
.venv/bin/python -B -I -m unittest discover -s armor-system/tests/prep -p 'test_*.py'
```

**Evidence and commit reference.** `evidence_m1_03.json` has no hash of itself and no commit SHA that its own commit would change. It records the base commit the work started from, `source_set_sha256` (a hash over the content of the M1-03 files, identical in the working tree and in the commit that adds them) and `final_merge_commit: null`. The merge commit is recorded in the Linear 298-48 comment when the PR merges. The scope of the older `evidence.json` stays 298-44 only.

Generation stops with an exception if any leakage invariant, approved count, mask count (3,152), MI-6 check or approved hash (outer split `32ff03fc…`, preprocessing config `e9aa3ccc…`) differs.

## Using the data

```python
from prep import m1_dataset as md
D = md.Model1Data.load()                               # verifies hashes, provenance, MI-6, window lists
train = md.training_dataset(D, "s1_pretrain")          # 1,312 fitting normal_reference windows
loader = md.training_loader(train, batch_size=64, seed=0)
calib = md.evaluation_dataset(D, "s3_calibration_reference")   # 329 windows, no gradient
```

Each item holds `inputs` (6,46,130), `target` (46,130), `input_mask` (6,46,130) and `target_mask` (46,130) (applicable AND observed), `applicable_entry`, `observed_entry`, `structural_absent`, `observed_all_zero_not_structural`, `timestamps` (7,), `start_index`, `role`/`role_id`, `category`/`category_id`, `case_index`, `fault_class` (0..4 or -1), `group_id`, `fault_label_ambiguous`, `fault_label_exclusion_reason`. Unavailable entries are already 0 in `inputs` and `target`; the masks say which zeros are real. `collate_windows` stacks tensors, keeps strings as lists and turns missing case metadata into -1 plus a `<key>_present` mask.

## Which windows each objective may use

`ignore` windows are not confirmed negatives and are never used by any objective. Only the two gradient objectives update weights, and only fitting windows can reach them: `training_dataset` rejects every other objective, `training_loader` accepts only datasets from `training_dataset`, and every training batch is re-checked to hold only `fit` windows.

| Objective | Region | Categories | Gradient | Windows |
|---|---|---|---|---|
| `s1_pretrain` | fit | normal_reference | yes | 1,312 |
| `s2_fault_finetune` | fit | event_associated, minus the 1 ambiguous window | yes | 523 |
| `s1_early_stopping` | validation | normal_reference | no | 411 |
| `s3_calibration_reference` | calibration | normal_reference | no | 329 |
| `calibration_diagnostics` | calibration | event_associated | no (diagnostics only) | 78 |
| `validation_fault_eval` | validation | event_associated, minus ambiguous | no | 150 |
| `validation_all` | validation | all three | no | 869 |
| `test_structural` | test | all three | no (fixed structural checks only, no tuning) | 2,868 |

## Facts worth knowing

- **Ambiguous window and `fault_label_exclusion_reason`.** Every window record carries `fault_label_exclusion_reason` (`null` unless the fault label is withheld). One fitting window (start index 2550) has `t5` inside the windows of cases 78 and 79, which are different classes; its reason is `"overlapping_cases_different_classes"`. The design note counts that window as event_associated (524 in the fitting region) and also says it is `ignore` for the fault objective. Both are kept: the category stays event_associated so the counts match, `fault_class` and `case_index` are -1, and S2 uses 523 windows. Invariants L9/L10 fail the build if any event_associated window has `fault_class = -1` without a documented reason, or if a reason sits on any other window. The manifest records the reason per role under `windows.<role>.fault_label_exclusions`, in `ambiguous_windows` and in `counts.fault_label_exclusions`. No other exclusion reason exists today.
- **Fit-only scaler.** `absent_block` is identical to the train-period scaler. Constant entries are 4,209 versus 4,206. ARMOR's undocumented upstream pre-scaling is still present in every value (design risk R6).
- **Hashes.** The manifest file, manifest `content_sha256`, scaler file and feature-cache content hashes are recorded in `evidence_m1_03.json` and in the manifest itself, not repeated here.
- **Calibration utility.** An alert is `score > tau`. The thresholds it returns belong to the errors passed in; no real detection threshold exists yet.
- **Not in M1-03:** the PyTorch model, training, any real threshold, the XGBoost control, ARMOR-interval validation.
