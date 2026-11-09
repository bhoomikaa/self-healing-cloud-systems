# Model 1 v1: design audit and approved protocol (Linear 298-51)

**Status: all decisions below are approved by the project lead (2026-10-27) and recorded BEFORE any v1 training code exists.** The earlier specified-by-audit marks are removed: every item the audit specified is confirmed as written. Blocker B1 is resolved (section 9). v1 is not trained or evaluated at this commit.

## 0. Predeclaration (unchanged from the v0 protocol)

> Model 1 v0 uses no simulated-missing training and is reported as the ablation. Model 1 v1 uses simulated-missing training and is the predeclared final model for held-out test evaluation, regardless of validation performance.

v0 (five Stage 1 checkpoints, forecast-error detection, results in `results/detection_v0/`) is preserved unchanged.

## 1. Design authority: what is approved and what is new

Source: `MODEL1_V0_DESIGN.md` (approved, merged), sections 3, 4, 7 and the frozen list F1-F11 in section 10. A search of that note finds **no** binary detection head, no supervised detection objective, no SPOT/EVT thresholding and no rolling maximum.

| Element | Approved original design | v1 status |
|---|---|---|
| Encoder, next-step head, fault head (d 64, 2 layers, 4 heads) | F7 "initial default", two heads | **Unchanged** (168,200 parameters: encoder 151,169, fault head 8,581, next-step head 8,450) |
| Binary detection head | **Not specified** (F7: "two heads"; F9: detection = next-step error) | **Explicit v1 extension** (amendment A-2): +8,321 parameters = 176,521 |
| Supervised detection labels | F3 defines the three window categories only for training/validation use; S1 "never sees `event_associated`" | **New use** of F3 categories as weak detection labels (A-2) |
| Detection signal | F9: M1/M2/M3 forecast error, M2 primary | v1 primary = supervised score; forecast-error detection kept as a **diagnostic** (A-2) |
| Threshold | F9: 99th percentile on calibration windows | v1 primary = **empirical 90th percentile** of the rolled calibration scores (**extension A-3**); SPOT is a diagnostic only |
| Trailing rolling maximum | not in design | **Extension A-3** |
| Simulated missing | 4.4 optional switch [D]: probability 0.1, off for v0, "one ablation on validation only" | v1: **0.20**, both stages, and v1 is the final model (**extension A-4** of a [D] default) |
| Case-balanced fault weights, five-class head, case-level aggregation | F8 + amendment A-1 (TRAINING.md section 7) | unchanged |
| Split, scaler, masks, calibration region, ARMOR interval/scoring rules, 25 validation incidents, +/-300 s proxy | F1, F2, F5, F9, F10, F11 | **Unchanged** |
| Seeds, optimizer, learning rates, epochs, clipping | TRAINING.md section 7 | unchanged (v0 values) |

Per design 7.4 item 5, A-2 to A-4 are written amendments; this document is that record.

## 2. Architecture

Existing model unchanged. Detection head: input = the fault head's pooled last-minute features (mean and max over valid nodes, 128 values) -> `Linear(128, 64)` -> GELU -> `Linear(64, 1)`, one **logit per window**, no node identity. Added parameters: 8,256 + 65 = **8,321**; total **176,521**. The shared modules are initialised exactly as in v0 for a given seed (the new head is created after them).

## 3. Training and loss policy (no calibration, validation or test example in any weight update)

Optimizer settings are v0's (AdamW, weight decay 0.01, clip 1.0, batch 64; Stage 1 lr 1e-3 cosine, Stage 2 lr 1e-4 constant, 100 epochs, patience 10, seeds 0-4). Only fitting windows are batched; roles and categories are enforced before every step.

- **Stage 1**: next-step masked MSE on fitting `normal_reference` windows only, with simulated-missing inputs (section 4). No detection loss (Stage 1 contains no event windows, F3); the detection head is not trained here. Validation early stopping uses clean (not masked) validation `normal_reference` masked MSE, as in v0.
- **Stage 2**: initialised from that seed's selected v1 Stage 1 checkpoint, fresh AdamW. Each optimizer step uses a **paired batch**: 64 windows from the 523 eligible fitting event windows (shuffled; the ambiguous conflicting-label window is excluded as in v0) plus 64 fitting `normal_reference` windows (shuffled, cycling through the 1,312). An epoch is one pass over the event windows. `ignore` windows are never used.
  - `L = L_fault + L_det`, weight 1 each (fixed, no search).
  - `L_fault`: case-balanced class-weighted cross-entropy (v0 weighting) on the event half of the batch.
  - `L_det`: binary cross-entropy with logits over the 128 windows, label 1 for event windows and 0 for normal windows; the 1:1 pairing balances the classes, so no class weight is applied.
  - The next-step head is not trained in Stage 2 (as in v0). Both heads backpropagate into the shared encoder.
- **Checkpoint selection** (validation, no gradient, clean inputs, never a detection F1, threshold or interval metric): lowest `case-level fault NLL (25 cases, v0 definition) + balanced window BCE`, the latter the mean of the per-class mean BCE over the 150 validation event windows and the 411 validation normal windows. Patience 10. Validation is used for checkpoint selection and metrics only, never for a gradient update.
- **What the labels mean:** `event_associated` is the documented-fault window under policy B (pre 120 s, post 300 s), a weak proxy; `normal_reference` is far from every case. The head therefore learns the *geometry of documented injections*. **v1 detection is a supervised detector, not label-free like ARMOR's AD**, so a comparison with ARMOR's unsupervised AD must say so (section 6).

## 4. Simulated-missing policy (training only, both stages)

- **Rate and unit:** each applicable and available (node, modality) block is hidden with probability **0.20**, independently per window, input minute, node and modality. A hidden block sets `observed_entry` False for all of that modality's channels of that node at that minute; `effective = applicable & observed`, and the existing adapter path replaces the values by 0 with `torch.where` and receives the mask. **Targets are never hidden** (target values and target mask are untouched).
- **Three states stay distinguishable:** *natural absence* (`applicable` False, layers 1-2) is never droppable and never counted; a *true zero* (`applicable` True, `observed` True, value 0) is data; an *artificially hidden* block (`applicable` True, `observed` False) is logged in a separate boolean `artificial_missing` that the model never receives (the model sees only the effective mask), so tests and logs can tell hidden blocks from real ones.
- **Determinism:** the Bernoulli draws come from a counter-based generator (`numpy` Philox) keyed by (seed, stage, epoch, window start index), independent of batch order and size, with its own stream (it does not consume the shuffle or dropout RNG).
- Never applied to calibration, validation or test scoring unless a missing-data experiment says so (section 7); training-time masks are never reused as evaluation masks.

## 5. Detection pipeline (v1)

1. **Score**: `s_t` = detection logit of the window whose target minute is `t` (inputs end at `t - 60 s`).
2. **Trailing rolling maximum (causal, 60 s):** `s'_t = max(s_t, s_{t-60 s})`, using the previous window only if it exists, is exactly 60 s earlier and belongs to the same period (the same role: calibration or validation); otherwise `s'_t = s_t` (partial window). The output is stamped `t`, the target minute of the later window, and uses nothing later than `t`. The same transform is applied to the calibration scores before the threshold is computed. Consequence, intended and to be reported: one isolated exceeding minute becomes a two-minute alert, so ARMOR's singleton rule no longer drops it. The interval start is unchanged, so detection delay is not shifted.
3. **Primary threshold (decision B1):** the **empirical 90th percentile** (linear interpolation) of the rolled `s'` over the **329 calibration `normal_reference` windows**, with a **7-window moving-block bootstrap interval** (blocks of 7 consecutive windows in time order, 47 blocks resampled with replacement per resample, 1,000 resamples, numpy `default_rng(20261028 + seed)`, 2.5th and 97.5th percentiles of the resampled 90th percentile) and the ranks used (n = 329, p = 90: ascending ranks 296 and 297, 34th and 33rd from the top, weight 0.2). The interval is a diagnostic; it never changes the threshold. No validation or test label is used.
4. **Alerts and scoring**: `s' > threshold`; the ARMOR-equivalent `get_pre_interval` (delay 600 s, singleton drop) and `eval_AD` (+/-300 s proxy) against the **25 documented validation incident timestamps**; the same diagnostics as v0 (precision, recall, F1, detection delay, false-positive intervals, normal-reference false-alarm rate over the 411 windows, ignore-region classes, coverage, no-score handling). Five seeds, mean +/- sample SD. No official ARMOR label or interval file, no test data.

## 6. Primary and diagnostic configurations

- **Primary (frozen):** supervised logit -> causal rolling maximum -> empirical 90th percentile of the rolled calibration scores -> ARMOR intervals -> 25 validation incidents. Fixed before training; never chosen from validation; the same configuration is the test configuration.
- **Diagnostics D1-D4: reported, never selected from, never alternatives to the primary:**
  - D1: the same rolled supervised score with other thresholds: **SPOT** (ARMOR's initialisation step with its D1 parameters `q = 0.1`, `level = 0.95` and its fallback; the "threshold <= 0 invalid" rule dropped because logits can be negative), and the 95th, 97th, 99th and 99.5th percentiles.
  - D2: supervised score **without** the rolling maximum, 90th percentile, isolating the rolling maximum.
  - D3: v1 **Stage 1** checkpoint, forecast error (M1, M2, M3) with the v0 protocol (M2 at the 99th percentile primary; 95/97/99.5 as in v0) and additionally the 90th percentile, isolating the effect of simulated-missing training against v0 and serving as the **label-free** row.
  - D4: v0 as already reported (`results/detection_v0/`).
- **ARMOR comparison (at test time, later):** two rows are reported, never merged: the **v1 primary (supervised)** and **D3 (v1 Stage 1 forecast error, label-free, like-for-like with ARMOR's unsupervised AD)**. No ARMOR number is compared during this validation phase (ARMOR's published detection numbers are on the test period).
- Five simultaneous changes separate v1 from v0 (head, supervision, thresholding, rolling maximum, simulated missing); D1-D3 are the minimum needed to attribute differences.

## 7. Missing-data validation plan (validation only)

**ARMOR's six scenarios.** `main.py::build_missing_cases(include_all_missing=False)` takes every non-empty proper subset of the three modalities {log, metric, trace}: drop {log}, {metric}, {trace}, {log, metric}, {log, trace}, {metric, trace}, plus the full baseline. `get_channel_indices_to_mask` zeroes **all channels of the modality for all nodes and the whole evaluation period**; ARMOR's `metric` includes the 57 VM metrics (spans 0-52 and 73-129). Reproduction on our schema: `observed_entry` False for every channel in `modality_index_ranges[modality]` (metric 110, trace 9, log 11), for every node and minute, input and target (the target mask follows design rule 3). **Dropping `metric` hides all six VMs entirely** (their learned missing vector is used) while dropping `trace` or `log` affects pods only; scenarios containing `metric` are therefore **reported separately** from the pod-only scenarios. Primary: **recalibrate on the same masked calibration windows** (ARMOR-faithful); secondary: **clean thresholds** (from unmasked calibration) applied to the masked validation windows. Models: v0 (forecast error, M2 at the 99th percentile), v1 primary (supervised, 90th percentile) and D3. Five seeds.

**Random missingness.** Rates **0, 10, 20, 30 and 40%** with **10 deterministic masks per rate**. A mask is a Bernoulli draw over (sample minute, node, applicable modality) of the whole sample timeline (Philox keyed by rate index and mask index), so a hidden sample is hidden in every window that uses it, and the target mask follows. Masks depend only on (rate, mask index): identical for every model and seed (paired). At 0% all ten masks are identical (one distinct mask, reported once). Calibration is redone per mask on the masked calibration windows (primary) or clean thresholds are used (secondary), then the validation protocol of section 5 is run; per rate report the mean +/- SD over masks (and seeds) of F1, recall, precision, false-positive intervals and alert coverage, relative to 0%.

Caveat: v1 trains with per-block random drops, so ARMOR's whole-modality scenarios are out of distribution for it; the random-rate experiment matches its training distribution at 20%.

## 8. SPOT feasibility audit (measured, analysis only; `results/v1_audit/`)

ARMOR's original `SPOT` and `get_threshold` were run unchanged on the v0 forecast-error M2 scores of the **329 calibration windows** (five seeds; `v1_audit_spot.py`).

- ARMOR uses only the initialisation step: threshold `t` = ascending order statistic `S[int(level * n)]`, peaks = scores above `t` minus `t`, a Grimshaw GPD maximum-likelihood fit, extreme quantile `t + (sigma / gamma) * ((n q / Nt)^(-gamma) - 1)`; no streaming; fallback `sorted(scores)[int(n * level)]` on any exception.
- With `level = 0.95`, `q = 0.1`: `n q / Nt` is about 2 for any `n`, so the "extreme quantile" lies **below** `t`, near the 90th percentile. ARMOR fits on about 2,270 in-sample training minutes (about 113 peaks); here `Nt = 16`.
- **Dependence:** adjacent windows share 6 of 7 minutes; lag-1 autocorrelation 0.23-0.29, effective sample size **128-170 of 329**, so the 16 peaks carry about 7 independent ones.
- **The fit degenerates:** the GPD shape is estimated as exactly **gamma = 0** (an exponential tail) on the real scores of all five seeds and in all 2,500 block-bootstrap resamples. The threshold is then `t - sigma ln(r)`: **12.5-18.9, against empirical 90th percentiles of 35.9-42.6**; **21-32% of the normal calibration windows exceed it** (nominal 10%).
- **Unstable:** 7-window block bootstrap, threshold IQR/median 0.67-1.15 (plain bootstrap 1.0-2.9); 2.5% quantiles of the resampled thresholds are **negative** in every seed.
- No fit raised an exception; the fallback was never needed. These are v0 forecast-error scores; the v1 supervised scores did not exist when this was measured.

## 9. Decisions, risks and limitations

- **B1 (resolved):** the primary threshold is the **empirical 90th percentile of the rolled calibration scores (329 windows) with the 7-window block-bootstrap interval**. Rationale: with ARMOR's parameters (level 0.95, q 0.1) SPOT reduces to roughly the 90th percentile and, on 329 overlapping windows, is not a tail estimator (section 8: 16 correlated peaks, degenerate exponential fit, thresholds far below the 90th percentile and unstable). SPOT is kept as diagnostic D1 only. **Option (c)** (fitting SPOT on the 1,312 fitting normal windows) is **rejected**: in-sample supervised logits are optimistic, so the threshold would be too low for held-out windows. The choice was made before any v1 score existed and is never revisited from validation results.
- **R1** Supervised vs label-free detection: v1 uses fitting-region incident labels and cannot be compared one-to-one with ARMOR's unsupervised AD; hence the two-row comparison in section 6.
- **R2** The rolling maximum bypasses the singleton rule and changes what an alert is; five changes versus v0 are confounded (D1-D3 mitigate). A 90th-percentile operating point alerts on about 10% of normal calibration minutes by construction: alert coverage and false-positive intervals are reported with every number.
- **R3** The head learns the policy-B label geometry around documented injections.
- **R4** Validation (25 incidents, one Node Memory case) also selects the checkpoint; calibration (329 overlapping windows, effective size about 130-170) is the only source of thresholds; calibration-region drift is unmeasured.
- **R5** ARMOR's whole-modality missing scenarios are out of distribution for v1; random-rate experiments match training.
- **R6** The unexplained `ad_cases.pkl` events cannot be separated from false positives without opening the official labels (test period).

## 10. Guardrails

No hyper-parameter search; no validation-driven choice of the primary detector, threshold, rolling window or rate; v0 checkpoints and results untouched; no calibration, validation or test example in any weight update; no test-period data and no official ARMOR label or interval file; no push, PR or Linear change.

## 11. Test-time reporting rules (added after the validation phase, before any test access)

This section only fixes how already-computed configurations are reported on test. It changes no model, threshold rule or selection and it was written after the validation results were seen, so it is a reporting rule, not a tuned choice. The evidence file `results/experiment_v1/evidence_model1_v1.json` hashes this protocol as committed at the end of Phase 3 and is not rewritten.

1. **The primary detector stays exactly as predeclared in section 6:** supervised logit, causal 60 s rolling maximum, empirical 90th percentile of the rolled calibration scores, ARMOR intervals. The validation results (including that D2 and D3 scored higher F1 than the primary on validation) do not change it.
2. **D1-D4 are reported on test as diagnostics only,** with the same definitions as section 6, and are never promoted to primary, never used to pick between configurations and never described as the model's result. D3 is the only diagnostic that also appears as a second ARMOR comparison row (section 6), labelled label-free.
3. **Seed reporting.** The primary result is the per-seed result for seeds 0-4, reported as mean +/- sample SD (ddof 1) over the five seeds. A secondary row is the **five-seed probability-averaged ensemble:** for fault type, the mean over seeds of the case-level class probabilities (softmax of the case logits, as in `case_eval`), then argmax; for detection, the mean over seeds of `sigmoid(detection logit)` per window, then the same causal 60 s rolling maximum, with the threshold the empirical 90th percentile of the rolled ensemble score on the same 329 calibration windows (no bootstrap interval required), then the same ARMOR interval rules. The ensemble is not chosen from validation or test and is never substituted for the per-seed primary.
4. **Missing-data comparison at one operating point.** The D3 missing-data curves are also reported with M2 at the 90th percentile (`results/experiment_v1/missing_d3_p90_*.json`, same masks and checkpoints, recalibrated on masked calibration windows) so that the primary and D3 curves share a threshold level; the earlier D3 and v0 (D4) curves at the 99th percentile remain as reported. On test, D3 is shown at the 90th and, as in v0, at the 99th percentile.
5. **No further changes.** The test plan (`TEST_PLAN.md`) lists every row evaluated on test. Anything not in it is exploratory, is labelled so, and is not a result.
