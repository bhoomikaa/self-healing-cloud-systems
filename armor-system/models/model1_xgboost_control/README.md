# XGBoost fault-classification control (Linear 298-49)

An experimental **control** for Model 1: it tells us whether the proposed Spatio-Temporal Multimodal Transformer adds measurable value over a simple fault classifier. It is **not** one of the four proposed final-capstone models and **not** a replacement for the published ARMOR baseline. Everything is CPU-only and deterministic.

**Test is evaluated once, together with Model 1, after both configurations are frozen.**

## What is fitted and what is evaluated

| Partition | Cases | Use here |
|---|---:|---|
| Fit | 88 | XGBoost is trained on these cases only (never "101 cases") |
| Calibration | 13 | Excluded from every fit, class weight and feature decision |
| Validation | 25 | Evaluates the fixed configuration. No tuning, no early stopping |
| Test (held-out) | 84 | Locked. Not loaded, predicted or scored by this code |

Class weights are inverse frequency `N / (K * n_k)` from the 88 fitting cases. There is no feature selection and no learned scaling in this layer (the only scaler is the approved fit-only scaler from M1-03).

## Features: one vector per case

`armor-system/data/prep/m1_case_features.py`. For each documented case: take its approved `event_associated` windows (the one window with an ambiguous fault label is excluded), collect the **distinct** one-minute samples they touch (11-12 per case, because neighbouring windows share 6 of 7 minutes), and for each node group (VM, pod; from the declared node types, never from node position) and each channel applicable to that group compute over those minutes: mean of the group average, standard deviation of the group average (ddof 0), largest absolute value across applicable nodes and minutes, and last-minute minus first-minute change. That gives 2 groups x (57 + 73) channels x 4 statistics = **520 features**. Matrices: fit `(88, 520)`, validation `(25, 520)`. Class ids follow Model 1: 0 Container Hardware, 1 Container Network, 2 Node CPU, 3 Node Disk, 4 Node Memory.

Overlapping cases (same incident group) can share a few minutes. The approved protocol keeps those cases on the same side of every split, so no minute crosses fit and validation.

## Observation timing: what these features can and cannot claim

Exact times are saved per case in `results/xgb_control_v1/observation_timing.json` (offsets in seconds relative to the documented case time `t_case`; case times are not minute-aligned, sample times are).

- **Samples used.** Each case vector uses the distinct samples of that case's approved windows: about **5-6 samples before** the documented injection time and **5-6 at or after it**, spanning roughly `t_case - 359 s` to `t_case + 359 s`. Fit cases: first sample between -359 s and -294 s, last between +284 s and +359 s. Validation cases: first between -358 s and -301 s, last between +302 s and +359 s.
- **Pre- and post-injection telemetry.** Both are included. The last-minus-first change feature therefore contrasts the minutes before and after the injection.
- **Earliest possible prediction.** A case prediction cannot be made before its last sample, i.e. **about 4.7 to 6 minutes after the documented injection time** (median about 5.4 for fit cases, 5.5 for validation).
- **Why this is legitimate classification.** The weak label is the documented fault window (`event_associated`), and the question asked is "given the telemetry around a known injected incident, which of the five fault classes was it?". It uses no information from later minutes, from other cases, from calibration or from the held-out set. The runtime check in `observation_timing` also proves that all fit samples precede the fit cutoff, all validation samples lie in the validation period, no sample is shared, and none lies in the calibration region or the held-out period.
- **Why it is not early incident detection.** The windows are anchored on the documented injection time, which a deployed system would not know, and the prediction is only available minutes after onset. These numbers say nothing about how early or how reliably an incident would be noticed. Do not describe them as detection performance.
- **Fair comparison rule for the Transformer.** Model 1's fault classifier must classify the same cases from equivalent observations: the same approved `event_associated` windows of each case (same samples, no sample later than the case's last window), case-level aggregation only after those windows, the same 88 fitting cases for training, the same 25 validation cases, the same label order, ambiguous-label windows excluded, and no calibration or held-out observation in any fit, class weight or feature decision. A Transformer that sees more minutes, better-aligned windows or the case's true onset in some other way is not comparable; one that sees less must say so.

## Fixed configuration

`config_v1.json`: native `xgboost.train`, `multi:softprob`, 5 classes, `hist`, `max_depth` 3, `eta` 0.1, 100 rounds, `subsample` 0.8, `colsample_bytree` 0.5, `min_child_weight` 1, `lambda` 1, one thread. Five independent runs with seeds 0, 1, 2, 3, 4, same features and parameters. No best seed is chosen. Package pin: `requirements.txt` (`xgboost==3.4.1`).

## Run

```bash
.venv/bin/python -m pip install -r armor-system/models/model1_xgboost_control/requirements.txt   # once, into the project .venv
.venv/bin/python -I armor-system/models/model1_xgboost_control/xgb_control.py                    # writes results/xgb_control_v1/
.venv/bin/python -I armor-system/models/model1_xgboost_control/xgb_control.py --write-evidence   # two clean reruns, all tests, evidence file
```

Needs the D1 feature cache and M1-03 artifacts (see `START_HERE.md`). **macOS note:** `torch` and `xgboost` each bundle an OpenMP runtime and abort if loaded into one process. The script therefore extracts the features in one process (torch) and trains in another (xgboost); tests do the same. Do not "fix" this with `KMP_DUPLICATE_LIB_OK`.

## Outputs (`results/xgb_control_v1/`)

`validation_summary.json` (per-seed values, mean and sample SD across seeds, majority control, limitations), `per_seed/seed_*_metrics.json` and `seed_*_predictions.csv` (per-case probabilities), `majority_control.json`, `config_used.json`, `case_features_provenance.json` (shapes, class distributions, class weights, array and input hashes), and `evidence_xgb_control_v1.json` (hashes, versions, commands, reproducibility and test runs; no self-hash, no self-referential commit SHA).

## How to read the results

- Primary metric: five-class macro-F1 over all five labels (zero-division counts as 0). Secondary: accuracy, weighted-F1, per-class precision/recall/support, 5x5 confusion matrix (true classes in rows).
- **Seed SD measures training randomness only.** It is not uncertainty across independent incidents. 25 validation cases give wide uncertainty.
- **Node Memory has one validation case**, so its recall can only be 0 or 1. Node CPU has 3. The "macro-F1 excluding Node Memory" is an extra sensitivity number, never a replacement for the five-class result.
- Fault-family diagnostic: container family (Container Hardware, Container Network) and node family (Node CPU, Node Disk, Node Memory), from the D1 category names. Each family shows its confusion rows against all five predicted classes, so mistakes into the other family stay visible. Family-level accuracy has all of the family's cases as denominator; conditional within-family accuracy has only the cases predicted inside the family as denominator and is not overall accuracy. The families do not prove that a fault physically touches only pods or only VMs.
- Labels are documented fault windows (`event_associated`), not verified minute-level anomalies.

## Future frozen-test pathway (blocked today)

1. Freeze this configuration (`config_sha256`) and Model 1's configuration; record both hashes.
2. Write the decision to evaluate in the Linear issue and a frozen-config file.
3. Run one evaluation on the 84 held-out cases for both models together, with the same case-level protocol.
4. Never re-run after seeing the numbers.

`xgb_control.final_test()` raises `FinalTestLocked`, and `build_case_matrix` raises `RoleLocked` for any role except fit and validation. Neither is to be changed until step 1 is done.
