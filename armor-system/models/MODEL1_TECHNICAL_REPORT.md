# Model 1 technical report: spatio-temporal Transformer for incident detection and fault-type classification on ARMOR D1

Linear 298-51 · branch `khushidonda/298-51-model-1-two-stage-training-and-validation` · status as of 2026-10-31.

Every number below comes from a committed result file; the file is named next to each table.
- **v1** is the model evaluated once on the held-out D1 test period.
- **v2** is the five-seed ensemble of the frozen v1 checkpoints, chosen and confirmed on development data only. Its D1 test numbers are **post-hoc and non-independent**.

## 1. Summary

| | Development (validation / CV) | D1 test (84 cases, 98 official labels) |
|---|---|---|
| v1 fault type, macro-F1 / weighted-F1 | 0.784 ± 0.090 / 0.907 ± 0.043 (validation) | **0.721 ± 0.055 / 0.827 ± 0.039** |
| v1 detection F1 (our labels) | 0.680 ± 0.069 (validation) | **0.727 ± 0.056** |
| v1 detection F1 (official ARMOR scoring) | – | **0.733 ± 0.052** (ARMOR: 0.9608) |
| v2 detection F1, ensemble vs single-seed mean (CV F1 / F2 / F3) | 0.641 vs 0.612 / 0.775 vs 0.708 / 0.708 vs 0.689 | 0.783 (ours) / 0.769 (official), **post-hoc** |
| v2 fault type, exact-correct cases, ensemble vs single-seed mean (CV F1 / F2 / F3) | 37 vs 33.4 / 38 vs 35.0 / 24 vs 22.4 | macro-F1 0.703, weighted-F1 0.826, **post-hoc** |

Bottom line:
- **Against ARMOR:** Model 1 does not match ARMOR's published detection numbers on the official labels (F1 0.733 vs 0.9608), nor its fault-type weighted-F1 (0.827 vs 0.938).
- **Against the fault-type baselines on test:** v1 is statistically indistinguishable from the v0 ablation and from the XGBoost control. Every paired-bootstrap 95% interval includes 0.
- **v2:** a measured, development-confirmed improvement over single seeds, at zero training cost.

## 2. Data, splits and scoring

- **Dataset.** ARMOR's D1 release (Zenodo 21744694, CC-BY-4.0): 7,200 one-minute samples over five recorded days, 46 nodes (40 pods, 6 VMs) × 130 channels (metric 110, trace 9, log 11), and 210 failure cases in 5 classes (Container Hardware, Container Network, Node CPU, Node Disk, Node Memory).
  - Data validation is in `armor-system/docs/d1_validation_report.md`.
  - D2 was not downloaded (section 9).
- **Split.** The split is chronological, by incident group:
  - fit: 88 cases;
  - calibration block: 13 cases, plus the 329 normal calibration windows that set every threshold;
  - validation: 25 cases;
  - test: 84 cases, exactly ARMOR's test cases.

  Windows are 6 input minutes plus 1 target minute, labelled `event_associated` / `normal_reference` / `ignore` (policy B). The manifest is `data/artifacts/d1/splits/model1_calibration_split_v1.json`.
- **Detection scoring.** We use ARMOR's interval rules, ported and verified against ARMOR's code: alert minutes are merged when less than 600 s apart, singletons are dropped, and an incident counts as hit within ±300 s.
  - **"Ours"** scores against the 84 test case timestamps.
  - **"Official"** scores against ARMOR's 98 post-split labels. With it, ARMOR's released intervals reproduce its published P/R/F1 0.9245 / 1.0 / 0.9608 exactly.
- **Fault type.** Case-level: the mean window softmax per case, then argmax. Metrics are macro-F1 (primary), weighted-F1 (ARMOR's FT metric) and accuracy.

## 3. v1 model and training

- **Architecture.** A factorised spatio-temporal Transformer, 176,521 parameters (`model1_transformer/ARCHITECTURE.md`, `model_v1.py`):
  - per-node-type, per-modality input adapters with explicit masks;
  - modality fusion;
  - 2 blocks of temporal attention (per node over the 6 input minutes), cross-node attention (per minute over 46 nodes) and an FFN;
  - d = 64, 4 heads, a node-type embedding (pod/VM) and no node-ID embedding;
  - a fault head (5 classes) and a detection head.
- **Stage 1.** Masked next-step forecasting on fit normal windows, with 20% simulated missing inputs. The epoch is selected on validation normal MSE.
- **Stage 2.** Paired batches of fault cross-entropy (case-balanced class weights) plus detection BCE. The epoch is selected on validation case NLL plus balanced BCE. Training uses AdamW and 5 seeds (`V1_PROTOCOL.md`, `TRAINING.md`), and takes about 50 min per seed on one CPU thread.
- **Primary detector.** The supervised logit, then a causal 60 s rolling maximum, then the empirical 90th percentile of the 329 calibration windows. These diagnostics were predeclared:
  - D1: other percentiles and ARMOR's SPOT;
  - D2: no rolling maximum;
  - D3: label-free forecast error of Stage 1;
  - D4: v0.
- **Controls.** v0 (the same Transformer without the detection head, with v0 training), an XGBoost case-feature control (`model1_xgboost_control/`) and a majority floor.

## 4. v1 results

### 4.1 Validation (25 cases; it also selects checkpoints, so it is optimistic)

Source: `model1_transformer/results/experiment_v1/validation_summary.json`.

| Model | Macro-F1 | Weighted-F1 | Accuracy |
|---|---|---|---|
| v1 | 0.784 ± 0.090 | 0.907 ± 0.043 | 0.912 ± 0.044 |
| v0 | 0.857 ± 0.180 | 0.934 ± 0.076 | 0.936 ± 0.073 |
| XGBoost | 0.814 ± 0.028 | 0.828 ± 0.022 | 0.832 ± 0.018 |

| Detector (25 incidents) | F1 |
|---|---|
| v1 primary | 0.680 ± 0.069 |
| D2 (no rolling max) | 0.745 ± 0.072 |
| D3, M2 @ p90 | 0.746 ± 0.020 |
| D3, M2 @ p99 | 0.376 ± 0.024 |
| D1 SPOT | 0.662 ± 0.108 |
| 5-seed ensemble (secondary row) | 0.735 |

### 4.2 One-shot D1 test (run `dabf6b59…`, results commit `3ed6e97`; seeds 0–3 byte-identical to the aborted attempt 1)

Source: `model1_transformer/results/test_v1/summary.json`. The test plan was frozen before any test access (`TEST_PLAN.md`, `fdea1a5` / `8a4987f`). The attempt-1 incident is in `TEST_INCIDENT_attempt1.md`.

**Fault type (84 cases):**

| Model | Macro-F1 | Weighted-F1 | Accuracy |
|---|---|---|---|
| v1 | 0.721 ± 0.055 | 0.827 ± 0.039 | 0.833 ± 0.038 |
| v0 | 0.746 ± 0.093 | 0.841 ± 0.065 | 0.848 ± 0.065 |
| XGBoost | 0.710 ± 0.017 | 0.846 ± 0.013 | 0.857 ± 0.015 |
| v1 ensemble (secondary) | 0.703 | 0.826 | 0.833 |
| Majority floor | 0.120 | 0.257 | 0.429 |

Paired case bootstrap (1,000 resamples), 95% intervals:

| Comparison | Macro-F1 difference | Weighted-F1 difference |
|---|---|---|
| v1 − XGBoost | +0.012 (−0.083, +0.121) | −0.018 (−0.079, +0.047) |
| v1 − v0 | −0.025 (−0.087, +0.032) | −0.014 (−0.050, +0.020) |

**Detection (our labels, 84 incidents):**

| Detector | P | R | F1 | False-positive intervals |
|---|---|---|---|---|
| v1 primary | 0.769 | 0.693 | 0.727 ± 0.056 | 17.4 |
| ensemble | 0.793 | 0.774 | 0.783 | 17 |
| D2 | 0.834 | 0.726 | 0.775 ± 0.035 | 12.2 |
| D3 @ p90 | 0.766 | 0.812 | 0.788 ± 0.006 | 20.8 |
| D3 @ p99 | 0.914 | 0.205 | 0.335 ± 0.023 | 1.6 |
| D4 (v0) @ p99 | 0.911 | 0.219 | 0.353 ± 0.010 | 1.8 |

**Missing data (v1 primary F1, recalibrated):**

| Condition | F1 |
|---|---|
| Random 10–40% | 0.729–0.739 |
| drop_log | 0.731 |
| drop_trace | 0.696 |
| drop_log+trace | 0.655 |
| drop_metric (hides all VMs) | 0.595 |
| drop_log+metric (hides all VMs) | 0.555 |
| drop_metric+trace (hides all VMs) | 0.406 |

## 5. Comparison with ARMOR (official 98 labels, D1 test)

| Row | P | R | F1 |
|---|---|---|---|
| ARMOR published (Table 1) / our reproduction of its released intervals | 0.9245 | 1.000 | 0.9608 |
| Model 1 v1 primary (A1, supervised) | 0.826 ± 0.024 | 0.661 ± 0.079 | 0.733 ± 0.052 |
| Model 1 D3 @ p99 (A2, label-free) | 1.000 | 0.200 | 0.333 ± 0.013 |
| Model 1 D3 @ p90 (label-free) | 0.863 | 0.837 | 0.850 ± 0.006 |
| v2 ensemble (**post-hoc**) | 0.833 | 0.714 | 0.769 |

- **Fault type:** ARMOR's published weighted-F1 is 0.938 (not reproduced in our harness); Model 1 v1 scores 0.827.
- **Missing data:** ARMOR released no missing-data intervals.
- **ARMOR's calibration pool is larger and in-sample:** 2,269 training minutes, against Model 1's 329 out-of-sample windows.
- **Recall:** every Model 1 row has lower recall than ARMOR's 1.0.

## 6. Post-test analysis (explains the frozen results; never used to select)

Source: `model1_post_test_analysis/`.
- **Failure study:** 202 distinct failure units, with per-incident detail rebuilt exactly from the frozen code.
  - Of v1's 70 wrong fault-type predictions, 51 come from 26 cases whose root cause is a VM.
  - 51 incidents are missed by the primary detector in at least one seed, but only 6 in all five seeds (seed instability).
  - 49 incidents are lost only when VM metrics are hidden.
- **Weak-label ranking diagnostics** (documented fault windows vs normal windows; not anomaly-detection accuracy), test AUC-ROC:
  - v1 logit: 0.835 ± 0.013
  - D3: 0.737 ± 0.002
- **Placement:** D1's sample graphs contain a per-day pod-to-VM placement. The 6 partial graphs are fault-correlated (`PLACEMENT_VALIDITY.md`).

## 7. v2: frozen-v1 seed ensemble

**Choice** (`PLACEMENT_RELEVANCE.md`, `33b532a`).
- Placement-aware attention was deferred: on development data its edges can directly reach only 1 of v1's 11 validation fault-type errors, at a cost of about 36 trainings.
- Seed instability is visible on development data: validation single-seed detection F1 SD is 0.069.
- Averaging the five frozen seeds has no free parameters and needs no retraining.

**Protocol** (`V2_ENSEMBLE_PROTOCOL.md`, `2ac3ab7`, predeclared).
- v2 is exactly `ensemble.py`: mean case probabilities for fault type; mean sigmoid, then rolling maximum, then the 90th percentile of the same 329 calibration windows for detection.
- **Deciding evidence:** a day-blocked development CV.
  - Folds: F1 = recorded day 1 (45 cases), F2 = day 2 (43), F3 = validation (25). The calibration block is isolated.
  - Per held-out fold, 5 v1-recipe models are trained with fixed epochs (77 and 35, v1's median selected epochs).
- **Acceptance:**
  - E1: ensemble F1 above the single-seed mean in both F1 and F2;
  - E2: missed incidents and false-positive intervals no worse (+2 tolerance on intervals);
  - E3: exact-correct cases no worse than the single-seed mean − 1.

**Results** (`model1_v2/results/cv_summary.json`, figures in `model1_v2/results/figures/`).

| Fold | Single-seed F1 (mean ± SD) | Ensemble F1 | Ensemble P / R | Missed (ens / mean) | False-positive intervals (ens / mean) | Exact-correct cases (ens / mean) | Macro-F1 (ens / mean) |
|---|---|---|---|---|---|---|---|
| F1, day 1 (45) | 0.612 ± 0.052 | **0.641** | 0.758 / 0.556 | 20 / 20.0 | 8 / 11.4 | 37 / 33.4 | 0.740 / 0.644 |
| F2, day 2 (43) | 0.708 ± 0.038 | **0.775** | 0.838 / 0.721 | 12 / 15.4 | 6 / 7.4 | 38 / 35.0 | 0.701 / 0.632 |
| F3, validation (25) | 0.689 ± 0.068 | **0.708** | 0.739 / 0.680 | 8 / 9.2 | 6 / 4.8 | 24 / 22.4 | 0.785 / 0.767 |

**Acceptance: all pass → v2 accepted.**

| Criterion | Ensemble | Single-seed mean |
|---|---|---|
| E1, F1 fold | 0.641 | 0.612 |
| E1, F2 fold | 0.775 | 0.708 |
| E2, missed (F1 + F2) | 32 | 35.4 |
| E2, false-positive intervals (F1 + F2) | 14 | 18.8 |
| E3, exact-correct cases (F1 + F2) | 75 | 68.4 |

Reading the results plainly:
- The detection gain is modest and uneven. In F1, all of it comes from fewer false-positive intervals (recall is unchanged at 25 of 45). In F3 the ensemble has more false-positive intervals than the mean (6 vs 4.8).
- The fault-type gain is larger and consistent: +1.6 to +3.6 exact-correct cases per fold.
- In F1 and F2, the held-out day's normal-window alarm rate is about 10%, against 5% on validation. Thresholds set on the calibration block transfer imperfectly across days.

**Ablations**

| Ablation | Result |
|---|---|
| AB1, ensemble size (figure 2) | Mean F1 rises with k in F2 (0.708 → 0.775) and on validation (0.680 → 0.735, peaking at k = 4). In F1 it is flat until k = 5 (0.612 → 0.641); in F3 it peaks at k = 3 (0.729) and ends at 0.708. Fault macro-F1 is higher at k = 5 than at k = 1 in every fold, but not monotone: F2 and F3 dip at k = 5 after peaking at k = 4. |
| AB2, in-sample calibration pool (329 + 1,312 fit windows; validation only; not selectable) | F1 0.755 vs 0.735: 2 more hits and 2 more false-positive intervals; normal alarm rate 8.3% vs 5.4%. |
| AB3, fixed-epoch F3 vs frozen validation-selected v1 | Single-seed F1 0.689 vs 0.680; exact-correct cases 22.4 vs 22.8. Selection on validation did not flatter v1's validation numbers much. |

**Integrity checks** (`model1_v2/results/evidence_model1_v2.json`):
- The v2 code recomputes the committed validation ensemble exactly: predictions byte-identical; detection, fault and threshold equal.
- The single seeds equal v1's committed validation F1.
- The determinism repeat (F3, seed 0) gives identical outputs and identical model state.
- 14 tests pass: fold layout, no test or embargo windows, held-out and calibration windows never trained, inference equal to the evaluation path.
- `test_freeze.py --verify` reports OK.
- No D1 test window was read by v2 code.

**D1 test for v2 (post-hoc, non-independent).** These are the committed one-shot run's ensemble row: detection F1 0.783 (ours), 0.769 (official); fault macro-F1 0.703, weighted-F1 0.826.
- The row was observed before v2 was chosen, so it is not a confirmation of v2.
- No claim of independent superiority over v1 or ARMOR is made.

## 8. Limitations

1. **Small evaluation sets.** Validation has 25 cases (1 Node Memory); the test has 84 (4 Node Memory). The bootstrap intervals for v1 vs XGBoost and v1 vs v0 all include 0.
2. **The D1 test is spent.** After the post-test analysis, any later D1 test number is non-independent. Confirming v2, or any v3, needs D2 (if permitted) or new testbed incidents (`V2_PROTOCOL_NOTE.md`).
3. **Weak labels.** Incident truth is an injection timestamp. Event windows are a ±300 s proxy, and 21 of ARMOR's 98 official labels have no case record.
4. **VM dependence.** Hiding all VM metrics drops v1 primary F1 to 0.406–0.595. v1 never trains with whole-VM dropout, and v2 inherits this (not re-measured for v2).
5. **Node-family errors.** Node-level root causes dominate the fault-type errors. node-1 and node-2 host no pods, and placement is not used.
6. **CV caveat.** The CV keeps the frozen fit-only scaler, so in F1 and F2 the held-out day's unlabelled feature statistics enter normalisation (declared in the protocol).
7. **Thresholds across days.** Thresholds come from 329 overlapping calibration windows (effective size about 130–170), and they transfer imperfectly across days (section 7).

## 9. D2 and further data

- **Licence:** unclear, so D2 was not downloaded. Zenodo declares CC-BY-4.0 for `ARMOR.zip`, but ART, the upstream release, states D2 is under a non-disclosure agreement (`D2_CHECK.md`).
- **Next step:** ask the ARMOR authors or the AIOps Challenge 2021 organisers before any download.
- **ARMOR's published D2 numbers:** AD F1 0.997, FT weighted-F1 0.869.

## 10. Integration readiness

| Item | Status |
|---|---|
| Inference API | `model1_v2/inference.py`: `Model1V2.load()` hash-checks the five frozen checkpoints; `predict_windows`, `alert_flags`, `case_fault`. Tested equal to the evaluation path. |
| Cost (Apple M4, one CPU thread) | Checkpoint load 0.5 s; **9.9 ms per window** for the 5-model ensemble; 550 ms per 64-window batch. |
| Inputs | Windows built and scaled by the frozen Model 1 data layer (`prep/m1_dataset.py`) from D1-format tensors (46 × 130 per minute). There is **no adapter yet for live telemetry** from a testbed. |
| Serving / deployment | `armor-system/backend/`, `deployment/` and `airflow/dags/` are empty placeholders: no service, container or scheduler exists for any model. |
| Artefacts | Checkpoints live outside Git (`~/298A_project/checkpoints/`) with hashes in the freeze record. They are not yet published to shared storage. |
| Not merged | The branch is unmerged and the post-test, v2 and report commits are unpushed. |

## 11. Reproducibility

| Step | Commit | Command |
|---|---|---|
| v1 training and validation | `4b9a0d9` | `run_v1.py` (see `TRAINING.md`) |
| Test freeze and one-shot run | `8a4987f`, `3ed6e97` | `run_test.py --mode test` (one-shot; already used) |
| Post-test analysis | `bcaf296`, `73df581` | `model1_post_test_analysis/*.py` |
| v2 protocol | `2ac3ab7` | – |
| v2 code and tests | `6e06729`, `e30fa44` | `python -m unittest armor-system/tests/model1_v2/*.py` |
| v2 CV and evaluation | `c9e32ea` | `train_cv.py --fold F{1,2,3} --seed {0..4}`; `evaluate.py validation`, `cv`, `figures`, `evidence` |

Each v1 seed takes about 50 min on one thread. The v2 CV took 16 runs at about 15–35 min each, 4 in parallel; with 8 in parallel, 16 GB RAM thrashes.
