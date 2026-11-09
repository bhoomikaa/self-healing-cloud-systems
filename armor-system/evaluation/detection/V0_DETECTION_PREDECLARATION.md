# Model 1 v0 detection experiment: predeclared protocol (Linear 298-51)

**Written and committed BEFORE any Model 1 detection score, calibration threshold or detection metric exists.** Nothing below may be changed after the v0 results are seen; a change would be a written amendment with its reason.

## Predeclaration

> Model 1 v0 uses no simulated-missing training and is reported as the ablation. Model 1 v1 uses simulated-missing training and is the predeclared final model for held-out test evaluation, regardless of validation performance.

v0 is a real, reportable ablation baseline. It is not retrained, tuned or replaced; the five selected Stage 1 checkpoints (seeds 0-4) are used exactly as they are.

## 1. Timestamp alignment (verified from ARMOR's released source before scoring)

`ARMOR/models/unified_representation/armor_model.py::create_dataloader_AR` builds windows of `window_size = 6` input samples plus 1 target sample (`series_len = window_size + 1`, consecutive samples at most 60 s apart) and assigns each window the timestamp `series_sample[-1][0]`, the **target (seventh) sample**. `deviation.py::SLD` appends that timestamp to every score, and `anomaly_detector.py::AD` keeps `eval_slds_sum[["timestamp"]]` alongside the scores before `get_pre_interval`. So an ARMOR score at time `t` is the error of predicting the sample at `t` from the six samples before it. Model 1's alert time is `timestamps[:, 6]`, the same target sample. **The convention is unambiguous, so target-minute scoring is the single primary convention; no alternative offset is scored and none is chosen by performance.**

## 2. v0 calibration (per seed 0-4, from that seed's selected Stage 1 checkpoint)

- Reference set: exactly the 329 calibration `normal_reference` windows (role `calibration`, no case). No optimizer step, no checkpoint change.
- Errors: per window and node, the mean squared error over effective entries at the target minute (`window_errors.collect_window_errors`), all 46 nodes valid.
- Statistics and scores exactly as `prep/detection_calibration.py` (design 4.5): per-node median and IQR, `epsilon = max(1e-6, 0.01 * median(IQR))`, `z = (e - median) / max(IQR, epsilon)`; M1 raw max, M2 max normalised, M3 mean of the 3 largest `z`.
- Thresholds: each method's 99th percentile (linear interpolation) of its calibration scores is **primary**; 95, 97 and 99.5 are sensitivity only.
- **Ranks used:** for `n = 329` and percentile `p`, position `h = (n - 1) p / 100`; the two order statistics at ascending ranks `floor(h) + 1` and `ceil(h) + 1` (also given counted from the top) and the interpolation weight are reported.
- **Block-bootstrap threshold diagnostic (7 windows):** the 329 windows are put in time order; the overlapping blocks of 7 consecutive windows (323 blocks) are resampled with replacement (47 blocks, truncated to 329 windows); each resample is fully recalibrated (statistics, scores, 99th-percentile threshold per method); 1,000 resamples, numpy `default_rng(20261028 + seed)`; the 2.5th and 97.5th percentiles of each threshold are reported. The fraction of adjacent windows that are consecutive minutes is reported, because blocks across a gap in time are an approximation. This is a diagnostic; it never changes a threshold.
- Every calibration result carries the checkpoint file hash, the weights digest and the data/scaler/architecture/config hashes.

## 3. v0 validation detection (per seed)

- Scored: all **869** validation target minutes (contiguous, one window per minute); thresholds come only from calibration; validation never sets or selects a threshold, method or percentile. **Primary = M2 at the 99th percentile.** M1, M3 and the other percentiles are reported as secondary/sensitivity, never as alternatives to pick from.
- Alerts: score strictly above the threshold. Intervals: the ARMOR-equivalent `get_pre_interval(delay = 600 s)`; single-minute alerts are dropped by that rule and their number is reported.
- Event scoring: the ARMOR-equivalent `eval_AD` against the **25 documented validation incident timestamps** with the **+/-300 s proxy hit rule** (a weak proxy: incident end times do not exist). No official ARMOR AD label, interval or test-period information is read.
- Reported per seed and method/percentile: precision, recall, F1, TP/FP/FN, number of intervals, alerts dropped as singletons, detection delay (earliest hitting interval start minus the documented timestamp, hit labels only: median, mean, min, max in seconds), alert coverage (raw and padded by 300 s).
- When ARMOR's scorer has no score (no surviving interval raises `ZeroDivisionError`, or precision = recall = 0 gives a NaN F1) the seed is reported as **"no score"** with precision, recall and F1 counted as **0 for the mean and SD** and the number of such seeds shown next to every mean.
- False alarms over the **411** `normal_reference` validation windows: (a) share whose score exceeds the threshold, (b) share whose target minute lies inside a surviving interval.
- Each surviving interval is classified once: *hit*; *ignore-overlap* (not a hit, but contains the target minute of an `ignore` window); *normal-only false alarm* (neither). The number of intervals overlapping an `ignore` window, hit or not, is reported separately. The ARMOR false-positive count = ignore-overlap + normal-only; the normal-only count is the unambiguous false-alarm count. Ambiguous regions (`ignore` windows and any events without a documented case) are never treated as verified negatives. Alerts near official-AD-only events cannot be separated here because the official label file is not opened (it contains test-period labels); this limitation is stated in the results.
- All five seeds are reported, each with its own calibration; mean +/- **sample** SD (ddof 1). No seed is selected. No significance claim is made (25 events, one Node Memory incident, validation-based Stage 1 checkpoint selection).

## 4. Out of scope

Retraining or tuning v0, simulated-missing training (v1), the official ARMOR comparison (98 AD labels, released intervals; test period), any test-period score, label or metric.
