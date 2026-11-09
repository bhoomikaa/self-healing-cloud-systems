# Test incident: attempt 1 of the one-shot test run aborted (Linear 298-51)

Status: attempt 1 of `run_test.py --mode test` did not complete. Its partial output is archived unchanged in
`results/test_v1_attempt1_aborted/`. Attempt 2 is launched by a person, not by an agent.

## Timeline (2026-10-27, local time PDT)

| Time | Event |
|---|---|
| before 22:47 | Pre-checks: HEAD `fdea1a5881aac4f5489eb0125adb69f5a6a64b15`, working tree clean, `test_freeze.py --verify` OK, no marker present. |
| 22:47:23 | `run_test.py --mode test` started as a background job of a Claude Code session. Freeze verified; both one-shot markers written (run id `e7304d82465a499a958879a46e9d0410`, head `fdea1a5`). |
| 22:47:24 | Five missing-data workers and the XGBoost-features worker launched. The XGBoost-features worker exited at once with an argument error (cause 2 below). The missing-data workers wrote nothing to their logs. |
| 22:48-22:50:43 | Core step finished seeds 0, 1, 2 and 3 (44-55 s each); the last file was written at 22:50:43. |
| after 22:50:43 | The Claude Code session ended; the run and all its worker processes stopped with it. No Python traceback. Seed 4, the ensemble, the missing-data results, XGBoost and the summary were never written. |
| 23:31 | State inspected: no process alive; output as listed below. No retry. |

## Causes

1. **Primary: the session ended.** The run was a child of the Claude Code process. When that process exited, the operating system stopped the runner and its workers. The run was not resumable.
2. **Separate defect: worker output path check at `run_test.py:481`.** In test mode, `main()` required `--out` to equal `results/test_v1/` exactly. The orchestrator launches the XGBoost-features worker with `--out results/test_v1/xgb_features`, a subfolder, so the worker refused itself: `run_test.py: error: test output goes to …/results/test_v1 only`. The validation dry run did not exercise this check, which applies only in test mode. Even without cause 1, the run would have stopped with "worker(s) failed" after the core step, before any XGBoost, bootstrap or summary output.

## Exactly what was printed (console, `logs/console_test_run.log`)

```
[test core] seed 0 done in 44s: v1 macro-F1 0.6991, primary F1 0.6755
[test core] seed 1 done in 46s: v1 macro-F1 0.7555, primary F1 0.7952
[test core] seed 2 done in 49s: v1 macro-F1 0.7573, primary F1 0.7308
[test core] seed 3 done in 55s: v1 macro-F1 0.7608, primary F1 0.7674
```

## Exactly what was written (now under `results/test_v1_attempt1_aborted/`)

| File | sha256 |
|---|---|
| `TEST_RUN_MARKER.json` (repository marker, read-only) | `51be4f4b512504ca01e588104fff78d8206ebc7d4706b5b28ed4a4a78deabdaa` |
| `EXTERNAL_MARKER_copy_model1_test_run_marker.json` (copy of `~/298A_project/checkpoints/model1_test_run_marker.json`, which stays in place) | `51be4f4b512504ca01e588104fff78d8206ebc7d4706b5b28ed4a4a78deabdaa` |
| `seed_0/result.json` | `8bc6781df1d0545a46bb3d2563ba9b738f34b812bcedfca51f00cacb8ef89925` |
| `seed_0/predictions_v1.csv` | `8ee8b8716ea52f6f6491eb0cbf8e1a7c22bd79aa34d0fd66ecfdf584838313ac` |
| `seed_0/predictions_v0.csv` | `c7d08f56a55ad195ea6e172743b50357168388df87a2c86ad42eff6c26a23ccb` |
| `seed_1/result.json` | `c979fe8a1d2c5e200f47908ae3c35e0ba65b9bd86e0021c9156fdd29fe136ee5` |
| `seed_1/predictions_v1.csv` | `abc70912cc6b5d9c880ac41addd8f7c6e57c7c5697b7832d0aa3adfd2ea9f9c9` |
| `seed_1/predictions_v0.csv` | `7c3acd37ee8fda60d072e16e90f631e9104dff9d99cc6279293315b951740799` |
| `seed_2/result.json` | `1190745b93078262f7db1a97263d132dc3e3cca0b5ac4499c8b17f2ddc48a5a2` |
| `seed_2/predictions_v1.csv` | `b807115dc4144e70b01934a2de50c730e7bd7cf009014a2ca4233cc8c60268a2` |
| `seed_2/predictions_v0.csv` | `0ae2d932076572d79171804506a78440911daaa8af2238f7f723c64632fe286c` |
| `seed_3/result.json` | `53e9ad16fb230e2d020a81da18ff1214b229bf70a3cc3d9baf6c3d1f68a6ee56` |
| `seed_3/predictions_v1.csv` | `0d37ee4430db52c7f009a08e57b2f17ea1dab37efde86b0262af335445a50e81` |
| `seed_3/predictions_v0.csv` | `182e14c4b47354484bd01e6dbd7dafaba36c506ed2e3548fd3cf510d7f4b081e` |
| `logs/xgb_features.log` (the argument error above) | `f939e58f0a367911af7dac15f70a35362ae85803be42bec4937256ffda732f13` |
| `logs/missing_0.log` … `missing_4.log` (empty) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `logs/console_test_run.log` (copy of `~/298A_project/test_run.log`, added when archiving) | `0721d84208f25ce8c8a8e06db37e2a86845e7d03d1d0711032f7b0ebe442f253` |

The `result.json` files hold the full seed-level test results: fault type for v1 and v0, and detection rows for both label populations. Headline values recorded in them:

| Seed | v1 macro-F1 | v1 weighted-F1 | v1 accuracy | v0 macro-F1 | v0 weighted-F1 | v0 accuracy | v1 primary F1 (ours) | v1 primary F1 (official) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6991 | 0.8126 | 0.8214 | 0.7048 | 0.8140 | 0.8095 | 0.6755 | 0.6905 |
| 1 | 0.7555 | 0.8410 | 0.8452 | 0.6714 | 0.8072 | 0.8214 | 0.7952 | 0.7912 |
| 2 | 0.7573 | 0.8504 | 0.8571 | 0.8490 | 0.9118 | 0.9167 | 0.7308 | 0.6783 |
| 3 | 0.7608 | 0.8664 | 0.8690 | 0.8434 | 0.9065 | 0.9167 | 0.7674 | 0.7831 |

These are partial results (4 of 5 seeds, no ensemble, no XGBoost, no missing data). They are not a TEST_PLAN result and are not reported as one.

## What did not change

- **Models and checkpoints:** unchanged. `test_freeze.py --verify` reported OK after the abort. The checkpoint hashes in each attempt-1 `result.json` equal the frozen hashes.
- **Thresholds:** the supervised (primary, D1, D2) and D3 calibration thresholds recorded by attempt 1 equal the committed validation-phase thresholds exactly, for seeds 0-3. All thresholds come from the same 329 calibration windows; none was refitted.
- **Protocol and test plan:** `V1_PROTOCOL.md` (sha256 `d5047e7d…`) and `TEST_PLAN.md` (sha256 `145f4951…`) are unchanged.
- **No retraining, no reselection, no threshold change** happened after these numbers were seen.

## Deviation from the one-shot rule

TEST_PLAN section 1.1 allows fixing a crash only before any number is written. In attempt 1, four seeds of numbers were written and four seed-level values were printed. Attempt 2 is therefore a re-run of an identical, deterministic computation, not a fresh first look. The fix changes only the runner's output-path check and marker handling. Attempt 2 records whether its seed 0-3 files are byte-identical to attempt 1, so any difference would be visible.

## Fix for attempt 2 (freeze commit after this note)

- `run_test.py`: test-mode output paths may be `results/test_v1/` **or any subfolder of it**. Validation runs still may not write there.
- New marker names: `results/test_v1/TEST_RUN_MARKER_attempt2.json` and `~/298A_project/checkpoints/model1_test_run_marker_attempt2.json`. The runner refuses if either exists, and refuses if `results/test_v1/` already holds files. The attempt-1 external marker stays in place, unused.
- After completion the runner compares `seed_0`-`seed_3` (`result.json`, `predictions_v1.csv`, `predictions_v0.csv`) with this archive byte for byte and writes the outcome to `results/test_v1/attempt1_consistency.json`.
- Test: `armor-system/tests/prep/test_m1_run_test.py`.
