# Detection scoring harness and preliminary validation protocol (Linear 298-51)

**Status: the scoring harness is implemented and verified. The Model 1 detection protocol in section 4 is DOCUMENTED ONLY; nothing in it has been run.** No model was calibrated, scored or evaluated, and no held-out test label or output was used.

## 1. What exists

| Piece | File | Notes |
|---|---|---|
| ARMOR-equivalent interval construction and event scoring | `armor_scoring.py` | Line-for-line port of ARMOR's `get_pre_interval` and `eval_AD` (pure NumPy). `eval_AD_detail` also returns TP/FP/FN. |
| Synthetic equivalence battery | `equivalence.py` | Runs ARMOR's ORIGINAL functions (from its released source) and the port on synthetic data only. |
| Published-baseline verification | `verify_armor_published_ad.py` | Baseline-only; imports no Model 1 code. |
| Per-window masked prediction errors, alert timestamps | `../../models/model1_transformer/window_errors.py` | Reuses `prep/detection_calibration.node_errors`; scorable roles are calibration and validation only. |
| Evidence | `evidence_armor_scoring_harness_v1.json` (built by `harness_evidence.py`) | Provenance, equivalence results, published-number verification, ambiguities. |

## 2. ARMOR scoring semantics that are preserved (and the ambiguities they carry)

1. Alerts become intervals in the order given (no sorting). A new interval starts when the gap to the previous alert is **>= 600 s** (a 600 s gap splits, 599 s merges). An interval is (first alert, last alert); **an interval with start == end is dropped**, so a single-minute alert (or only repeated identical timestamps) counts as neither a hit nor a false alarm.
2. A label is hit when an interval overlaps `[t - 300, t + 300]` with **closed** boundaries.
3. TP and FN count **labels**; FP counts **intervals** (the two populations differ, so precision and recall are not symmetric).
4. Duplicated labels and duplicated intervals collapse (dict keys): they count once.
5. Precision and recall are rounded to 4 decimals and **F1 is computed from the rounded values**; it can differ from the exact F1 in the fourth decimal.
6. `TP + FP = 0` or `TP + FN = 0` raises `ZeroDivisionError` (a detector with no surviving interval, or no labels, has no score instead of a score of 0). The "density" statistic is NaN when no interval matches.
7. Interval lists must be tuples (`list` pairs raise `TypeError`).
8. In ARMOR the evaluation period is every sample after `split_ts` = the timestamp at 60% of all sample times, and labels are `ad_cases.pkl` entries above it; here the committed `official_armor_v1.json` gives `split_ts = 1651852800`, which is also our validation|test boundary. The origin of `ad_cases.pkl` (21 events without a case record, 7 cases missing) is unexplained (`d1_validation_report.md` 9.2).
9. ARMOR's own threshold is a SPOT quantile on training-period scores; Model 1 uses a percentile on calibration windows. Only the interval/overlap scoring is shared, so a Model 1 F1 and ARMOR's F1 are comparable only on the same population and period.
10. What the "timestamp" of an ARMOR sample means (window start, window end or sample time) is not stated in the released code; Model 1 alerts are placed at the **target minute** of a window. This alignment is an assumption, not a verified equivalence.

## 3. Published-baseline verification (baseline-harness verification, not a Model 1 evaluation)

The released ARMOR alert intervals (sha256 pinned) scored with the port against ARMOR's official scoring labels reproduce the published detection metrics **precision 0.9245, recall 1.0000, F1 0.9608**; ARMOR's original `eval_AD` agrees on the same inputs. The script prints only the three aggregates, PASS/FAIL/BLOCKED and provenance hashes. Required local (Git-ignored) inputs: `armor-system/data/raw/armor_zenodo_21744694/ARMOR/res/D1/20260113_101332/scenarios/full/{tmp/pre_interval.json,res.json}` (the interval file must match sha256 `c8da7bfa44cf18c44403739b98de8e45e98d1f628ae500c3e0ccb57f46c0e9d7`, Zenodo record 21744694) and `.../extracted/D1/cases/ad_cases.pkl` (size and CRC32 from the D1 acquisition manifest). If any is missing or differs, the script reports BLOCKED and substitutes nothing. The ARMOR baseline belongs to the baseline teammates; they should review this harness before it is cited as the baseline reproduction.

## 4. Preliminary Model 1 detection-validation protocol (NOT EXECUTED)

Preconditions: the calibration step (per seed, from the 329 calibration-reference windows and that seed's selected Stage 1 checkpoint) has been run and approved. Thresholds come only from calibration; **validation never sets or changes a threshold**.

1. **Scored minutes.** All **869** validation target minutes (one window per minute: 150 `event_associated`, 411 `normal_reference`, 308 `ignore`; contiguous, 60 s apart; the first six validation samples cannot be a target). Scores are computed with `collect_window_errors` (eval mode, no gradient) and the approved normalisation; M2 (max normalised) is primary, M1 and M3 are reported.
2. **Alerts and intervals.** A minute is an alert when its score exceeds the method's threshold (99th percentile; 95/97/99.5 reported as sensitivity only). `get_pre_interval(alert_target_minutes, delay=600)` builds intervals; the number of alerts dropped as singletons is reported.
3. **Event evaluation (documented incidents only).** Labels are the **25 documented validation incident timestamps** (`cases.csv`), scored with `eval_AD_detail` and the **+/-300 s proxy hit rule**. Report precision, recall, F1, TP/FP/FN, interval count, alert coverage (share of validation minutes inside padded intervals) and detection delay of hits. The proxy rule is a weak definition (incident end times do not exist, E-1) and must be labelled as such.
4. **Normal-reference false-alarm rate.** Over the **411** `normal_reference` windows: (a) the share whose score exceeds the threshold, and (b) the share whose target minute lies inside a surviving alert interval. Intervals are *not* required to avoid the other windows.
5. **Alerts near ignore regions, counted separately.** Each surviving interval is classified exactly once: *hit* (reaches a documented incident's +/-300 s region); *ignore-overlap* (not a hit, but contains the target minute of an `ignore` window); *normal-only false alarm* (neither). The strict ARMOR false-positive count = ignore-overlap + normal-only; the **unambiguous false-alarm count = normal-only** is reported next to it.
6. **Ambiguous is not negative.** `ignore` windows (within the pre/post guards of an incident), the 21 AD-only events without a case record and the 161 uncertain train samples are ambiguous regions: they are never counted as verified negatives or positives. Only `normal_reference` windows (at least 188 s from any case timestamp in the calibration region; the same rule defines them in validation) are treated as negatives. The validation period has no ambiguous-label windows.
7. **Seeds.** Report all five seeds (each with its own calibration and checkpoint hash), each seed's values and mean +/- sample SD. No seed or percentile is selected using validation.
8. **Populations stay apart.** The official ARMOR population (98 AD labels, 84 injection cases, released intervals) lies entirely after `split_ts` = our validation|test boundary, i.e. in the locked test period. It is a later, separate evaluation, never merged with section 4 results.
9. **Uncertainty.** 25 events, one Node Memory incident, validation-based Stage 1 checkpoint selection and about 3-4 calibration windows above the 99th percentile make every number noisy; report counts next to every rate and make no significance claim.

## 5. Not done here

Calibration, threshold values, any Model 1 score, validation detection, test evaluation, simulated-missing training, and the block-bootstrap diagnostic for thresholds (design 4.5).
