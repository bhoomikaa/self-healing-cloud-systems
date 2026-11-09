# Model 1 test plan (Linear 298-51): every row to be evaluated on the test period

Status: **written and committed before any test access.** No test-period window, case label or official ARMOR label or interval file has been read, loaded or scored by any Model 1, v0, v1 or XGBoost code so far (`evidence_model1_v1.json`: `test_data_accessed: false`). This file only lists what will be evaluated and what is frozen. It sets no thresholds, picks nothing and adds no result.

Branch `khushidonda/298-51-model-1-two-stage-training-and-validation`. Protocol: `V1_PROTOCOL.md` (sections 0-10 predeclared, section 11 reporting rules).

## 1. One-shot rule

1. The test period is evaluated **once**, for Model 1 v1, v0 and the XGBoost control **together**, with the frozen artefacts in section 2. Nothing is re-run, re-selected or re-thresholded after any test number is seen. A crash before any number is written may be fixed only if the fix cannot depend on outputs, and the fix is recorded.
2. Every row below is reported, including the ones that look bad, in this order. Rows not listed here are exploratory, labelled so, and are not results.
3. Test labels and official ARMOR files are opened only by the evaluation step after the freeze commit (section 7), never by training, calibration or any threshold code. Calibration stays the same 329 `normal_reference` windows for every row; thresholds are recomputed from them deterministically, not from test data.
4. Test-period numbers are never used to change the primary detector, a threshold rule, the ensemble definition, a checkpoint or an architecture. Validation numbers and test numbers are never pooled.

## 2. What is frozen (hashes)

Commits: protocol (Phase 1) `3f9b4da4d7152782d217a17a950612cf90cd2299`, implementation (Phase 2) `569c5eb878e8ba8300eee227b64d133df5e22d62`, results (Phase 3) `4b9a0d9baac233174b1ea08cb9301174f69af8ae`, D3 p90 missing-data results `dd5b3c29d7c7e0cd388a000a7fc362bcd560601a`, protocol section 11 `dfb7711adaffd8fb1495a5ac124972175699f8f2`.

`V1_PROTOCOL.md` sha256 `d5047e7df208204874e861b697acb2d518b16634673429f2caff4e5d3df1c5ae`.
`training_config_v1.json` sha256 `49c815d095c7cfd5ecbc0c04cbf842da751e0d218de0d9b2e09fc6934d307de6`.
`evidence_model1_v1.json` source-set sha256 (Phase 3) `6d1a65db7af231c9d2eb46d310774fc7cd33470f2e5617dd39db941fca79404f`.

**Model checkpoints** (outside the repository, `~/298A_project/checkpoints/{model1_v1,model1_v0}/seed_N/`; every hash below was recomputed from the file on disk and equals the one recorded in that run's `run_meta.json`; no checkpoint is refitted, reselected or fine-tuned on validation, calibration or test):

| Model | Seed | Stage | File sha256 | State-dict sha256 |
|---|---|---|---|---|
| v1 | 0 | S1 | `5ede5e2d55eeeb5abf20270e9e4bf09b9b05921ff86f14743c2add2f93f0c072` | `` |
| v1 | 0 | S2 | `01940bfb9cd3588e3f3da2f5862f4d9f470095d1f815669e31d827c047343e40` | `d36452a8f4525985afb88c21baac99decdc42c7ade2be32a1c4985f9be8951bc` |
| v1 | 1 | S1 | `0f0c0189355280ddf16f8427f73541c6cea53056dc2033db41c68f7e2110b08b` | `` |
| v1 | 1 | S2 | `671e0fb31f2c19ccff135360c4490c252fdd0eb89fb0955c1ea201758a135fb0` | `f6ae1d2576c0d2610ca2dd4ec9f9988e0df4fc9d48cbbbdfe097bee9c9deb76d` |
| v1 | 2 | S1 | `1c4b68fab7f372db40665e8bae984c07f1df4cef64efd13f8fe56a05a5fc850f` | `` |
| v1 | 2 | S2 | `1c29d48b1e9e975af573f54a69c8b15e542d499b99e8adf08d05f0d5b7a956df` | `2ae8bffe375dcc44d0af845ed86fdd1cde87e4dfbd88859fe576a89bbe2e90fa` |
| v1 | 3 | S1 | `09305071fdaea809eedc7f028cb0628b8b4e496e51b3f287f9cff8f3368f0ae2` | `` |
| v1 | 3 | S2 | `cb2c31a71571b15d89d43903fbea86e7637c1c45b03e08206456d7847f6e96fa` | `3524e963c2c1a7048ffc4c7bbfc5e6ff95f12124fa7a06d81a8097e273cf368c` |
| v1 | 4 | S1 | `9116933fc769a322a2e14994215f81236704dd88508ef29ec924fe2d8c67d709` | `` |
| v1 | 4 | S2 | `c25f2fa09b3a56275c9da3abfce00894cf30d296fa911f62e549131a55f11a02` | `b42294cb25518f251dad1a52e695dbaeb0283d23ac936b0d17c3104fb3db99d5` |
| v0 | 0 | S1 | `507822c6c7e6c6b5f538106aade94da90d4537c5db5d3eb5e8a9088262e920ac` | `` |
| v0 | 0 | S2 | `1c9014b2bff49e01d5efa133688efc23e8ad551408e56c2514537b0068036a1a` | `5cf02ee8e660b673f7f9bd847b06f2bf403eb31bb4bf30ff66cbbbe698ca1b9a` |
| v0 | 1 | S1 | `85129f372d3e09bf6b545246aa4fa494a4e66a3e778c8ff665b97524e410d30e` | `` |
| v0 | 1 | S2 | `6a15dc897f95499cc24c78f0ad90d8d8f0f36380af0484217d49221cdb84a957` | `e4b0461624847e41cea8bd18106cb021a393d7e388f6358c7ba0a02820152b88` |
| v0 | 2 | S1 | `c422ee01a62aca9cc33f0d042f053690fbfcc5acc5542d31076eff9705fd7e7e` | `` |
| v0 | 2 | S2 | `3648c60ce26e7eb55f70564dd335cce85165c9ccd5dee0bbb09236e7a97e51e7` | `f8eb612826c24d1bc8fc08f342e8465c9b3f8c5ac9224299c401e2c5924a3d93` |
| v0 | 3 | S1 | `26578ebeffc121af89c0e874dffa5afcb36ea06c0582fddced5acf24fd6d647b` | `` |
| v0 | 3 | S2 | `fa4f7166401e2bfbfee914a0a66abb7cc6e8288f390497b5b04aa03567379be4` | `ddb8a8c6fa280cea79e925b44289943c1cd92e4168f5865891bb518fc0329295` |
| v0 | 4 | S1 | `f1e71a003d47fbc3438d74215c3a59911c0e3c42f817521619c6ee7a105c60a1` | `` |
| v0 | 4 | S2 | `d4e6890c89fc89bcf1f4a3c6390d5d8aeb8c570207ad4299df652bb6bb1f75b6` | `86ead9ec9f2d5ebaa3fb93f27eb684ef3cb52d445b86964014fdb0322ad6fabe` |

Which checkpoint is used where: v1 primary, D1, D2 and the ensemble use the v1 **S2** checkpoints; D3 uses the v1 **S1** checkpoints; D4 uses the v0 **S1** checkpoints (as in `detection_v0`); v0 fault type uses the v0 **S2** checkpoints; the v0 S2 and v1 S1 files are not otherwise used. State-dict hashes are empty where a stage does not record one.

**XGBoost control** (no checkpoint; five seeds retrained deterministically on the 88 fitting cases from the frozen config, byte-identical across reruns on validation): `config_v1.json` file sha256 `8075f0ebc21a05285946ea3b97d6efd5ba63a3857b588ce791680d727752dc79`, canonical config sha256 `407c55b669bfee4be61712caa6c0e2d477c7af4b5345b76863974599ec5b97e1`, source-set sha256 `f60e088efa7cfdb29bcbf9eb1293c023c7d46beae9548eb97fba636ed9c2561b`. It stays in the same configuration; it is not refitted on validation. The locks (`xgb_control.final_test` / `FinalTestLocked`, `m1_case_features` `RoleLocked`, `window_errors.SCORABLE_ROLES`, `case_eval` `RoleLocked`) stay in their files unchanged, because those files are bound into earlier evidence whose verification hashes the live files. The test role is opened only by `test_freeze.unlocked(token)` for a token that `test_freeze.claim_test_run()` issues after the freeze verified and the one-shot marker was written; the model, features and config do not change.

**Code** (sha256 of files at the moment of this plan; the evaluation code that still has to be written, section 7, is added and hashed in the freeze commit):

| File | sha256 |
|---|---|
| `armor-system/evaluation/detection/armor_scoring.py` | `4c06a1562b46698339292f70e11d158c0e5bc879769667bca265820b76a66d64` |
| `armor-system/evaluation/detection/interval_diagnostics.py` | `304b93d1f0b9bd7b638693d1e36c47a9aba460b66f911a314d4386c5f34b3736` |
| `armor-system/models/model1_transformer/bootstrap_v1.py` | `e3f33d0a9357155465db7813dcc287d82e5dd64777ae604cad1cc27d97597569` |
| `armor-system/models/model1_transformer/case_eval.py` | `fd8197d1752532b8d6a69ad1c309e8e1ee98813bd98c276746931f7defb30d97` |
| `armor-system/models/model1_transformer/checkpoint.py` | `213acdd19d3ce1681deab0358f958a130ed15562e9e9b3639b24500c36bf7e9f` |
| `armor-system/models/model1_transformer/detection_v1.py` | `fcece09f9d416f62f5d4fba367cf51df8afacfe02f6e2c1cd75a2f2e585ccba4` |
| `armor-system/models/model1_transformer/missing_d3_p90.py` | `72aa618b04183ae895229fe4bbfed035904c30f04f9fbcf8a14cd4f943a95b0d` |
| `armor-system/models/model1_transformer/missing_eval.py` | `a9c1b208e79d466c8ba4b95c336e4fb7bcbbb0759f84004235e174a1bbca2776` |
| `armor-system/models/model1_transformer/model_v1.py` | `f30e4ba5d499210534ecdf2d77eb0c5ccc9e1a0e3af443f7d87c8e4096a77b4e` |
| `armor-system/models/model1_transformer/paired.py` | `3ea3801a2d5171e16ed5d1391084172026c558dd263f0c37946eee06f132a902` |
| `armor-system/models/model1_transformer/rolling.py` | `c56d919b2cb98e8c0f9bf01e95eadaadd469ed70a5d707717986e693efaf603f` |
| `armor-system/models/model1_transformer/run_detection_v0.py` | `90d298578da7b9371d68f5fa1189bfa5ce1c0d17311bcd0d7401486ed4e6b2fb` |
| `armor-system/models/model1_transformer/run_v1.py` | `19abd228877d40f617ea3ead86a8d1bf9f8a57af89ee17c97b87d19c8ae8f369` |
| `armor-system/models/model1_transformer/sim_missing.py` | `71f4ca729b298d72f398cadbabc127332217bc2d9b8d71ab7a0e7d20b1a53637` |
| `armor-system/models/model1_transformer/threshold_diagnostics.py` | `c5938f6d1953d7f10786b3f95409542e33610cac2985732e00f131a5ed7e9ddd` |
| `armor-system/models/model1_transformer/training.py` | `425ada84e24b8dda528fda140e6977386019c475bd5dfabfbacdd0be51d24f23` |
| `armor-system/models/model1_transformer/training_config_v1.json` | `49c815d095c7cfd5ecbc0c04cbf842da751e0d218de0d9b2e09fc6934d307de6` |
| `armor-system/models/model1_transformer/training_v1.py` | `8b129fd2182b8030cb9d272b379fc12f4f9ac90d2bce6c27ebc6d78863f621c5` |
| `armor-system/models/model1_transformer/weighting.py` | `11817207c21ae882711eb878fa4f6c9cfdf2f86b4c0a2b268e3dc79661b0f88a` |
| `armor-system/models/model1_transformer/window_errors.py` | `446165fd2d1d2101239b91e6d8408e2c88b01a826ae207d83a2739f106473db4` |
| `armor-system/tests/prep/test_m1_v1_components.py` | `92edd02c2f7db14e16ffb485781cdf7993fdf43856a8f42b6bf6a8e62bf9ccff` |
| `armor-system/tests/prep/test_m1_v1_d3_p90.py` | `a043421f8a37aecd1f31136f829d1567128e79eae75c0bc8a77763bc7f5c9cde` |
| `armor-system/tests/prep/test_m1_v1_detection.py` | `338c1115f580fa0b110eb384185fe19db3d5de95f9360ea996b2094200cff37a` |
| `armor-system/tests/prep/test_m1_v1_results.py` | `b6a29a5ba3210bb090bcd2113057ef0b48550e3c73d86f02bbe3a30e45500fed` |
| `armor-system/tests/prep/test_m1_v1_runner.py` | `20dd8bdcf56bdee065a3b47939962d32a416cd28153b9c7d9939c571698ced28` |

## 3. Fault-type rows (84 test cases, case-level protocol identical to validation: the approved `event_associated` windows of each case, same label order, ambiguous windows excluded)

| # | Row | Detail |
|---|---|---|
| F1 | **v1, per seed 0-4, and mean +/- sample SD (primary)** | macro-F1, weighted-F1, accuracy, per-class precision / recall / F1 / support, confusion matrix per seed and the mean matrix, container-versus-node family diagnostic |
| F2 | v1 five-seed probability-averaged ensemble (secondary) | mean of per-seed case class probabilities, argmax; same metrics as F1 |
| F3 | v0, per seed and mean +/- SD | same metrics (v0 S2 checkpoints) |
| F4 | XGBoost control, per seed and mean +/- SD | same metrics; plus the majority-class control as the floor |
| F5 | Paired case-level bootstrap, **v1 vs XGBoost** | 1,000 resamples of the 84 cases with the same cases drawn for both models, seed 20261028, seed-averaged metrics as in the validation bootstrap; macro-F1 and weighted-F1, each model's interval, the difference interval and the share of positive resamples (not a probability of superiority); resamples missing a class reported |
| F6 | Paired case-level bootstrap, **v1 vs v0** (secondary) | same design |

## 4. Detection rows, our scoring (documented incidents of the test period, ARMOR interval rules: delay 600 s, singleton drop, +/-300 s proxy; windows scored at the target minute; the same scoring set construction as validation)

Reported per seed 0-4 and mean +/- sample SD: precision, recall, F1, false-positive intervals, normal-window alarm rate, padded alert coverage, median delay, and the alerts overlapping `ignore` windows (counted, as in validation). Thresholds come from the same 329 calibration windows. No row has a "best" label.

| # | Row | Definition |
|---|---|---|
| D0 | **v1 primary** | supervised logit, causal 60 s rolling max, empirical 90th percentile of rolled calibration scores (block-bootstrap interval of the threshold reported as in validation) |
| DE | v1 ensemble (secondary) | section 11.3 of the protocol: mean `sigmoid(logit)` over seeds, rolling max, 90th percentile of the rolled ensemble calibration score; one result, no SD |
| D1 | D1 thresholds on the rolled supervised score | SPOT (ARMOR `q = 0.1`, `level = 0.95`), 95th, 97th, 99th, 99.5th percentiles |
| D2 | D2 | supervised score without the rolling max, 90th percentile |
| D3 | D3 (v1 S1 forecast error, label-free) | M2 at the 90th and at the 99th percentile (both reported, neither chosen); M1 and M3 and the 95/97/99.5 percentiles as computed in validation, in an appendix table |
| D4 | D4 (v0 as in `detection_v0`) | M2 at the 99th percentile; the other v0 rules in an appendix table |

## 5. Official ARMOR comparison (98 AD labels, 84 cases, released intervals)

- Population: ARMOR's own scoring labels (`ad_cases.pkl`, 98 post-split labels in 84 injection cases) and ARMOR's released alert intervals (`pre_interval.json`, sha256 `c8da7bfa44cf18c44403739b98de8e45e98d1f628ae500c3e0ccb57f46c0e9d7`), scored with the committed `eval_AD` port (`armor_scoring.py`, verified against ARMOR's original; reproduces ARMOR's published P 0.9245, R 1.0000, F1 0.9608). The origin of 21 label events without a case record is unexplained (`d1_validation_report.md` 9.2) and stays noted. Official and our-scoring populations are never merged.
- **Two ARMOR rows, never merged:**
  - **A1: v1 primary (supervised)** (D0), per seed and mean +/- SD, and the ensemble DE as a secondary line.
  - **A2: D3 (label-free)**, like-for-like with ARMOR's unsupervised detector, M2 at the 99th and at the 90th percentile, both shown.
- Reference rows (not Model 1 results): ARMOR published, ARMOR released-interval reproduction (this repository's own check).
- Alignment caveat, stated beside every row: how ARMOR timestamps a sample (window start, end or sample time) is not stated in its released code, so Model 1 alerts at the target minute are an assumption.
- Reported with the metrics ARMOR reports plus false-positive intervals and alert coverage, and the same deltas (our v1 minus ARMOR).

## 6. Missing-data rows (detection, test period)

- **Scenarios:** ARMOR's six drop scenarios (`log`, `metric`, `trace`, `log+metric`, `log+trace`, `metric+trace`) plus the full baseline, built exactly as in validation. Scenarios that contain `metric` hide all six VMs and are **reported separately** from the pod-only scenarios.
- **Random missingness:** rates 0, 10, 20, 30, 40%, the same 10 deterministic masks per rate and the same Philox keys as in validation; per rate mean +/- SD over masks and seeds.
- **Calibration:** primary = recalibrated on the same masked calibration windows; secondary = clean thresholds from unmasked calibration.
- **Models:** v1 primary; D3 at the 90th (matching the primary) and at the 99th percentile; v0 (D4, M2 at the 99th). Metrics: F1, recall, precision, false-positive intervals, alert coverage, relative to the clean row.
- **Scoring:** our scoring (section 4) for every condition. For the clean condition and the six ARMOR scenarios also the official labels (section 5), for our models only. Random-rate masks are scored with our scoring only.
- **ARMOR's released tree has no missing-data results** (listed before any test access, file names and hashes only; the full Zenodo archive listing is `armor-system/data/manifests/armor_d1/outer_archive_listing.csv`):

  | Released file (D1) | Size | sha256 / CRC32 |
  |---|---|---|
  | `ARMOR/res/D1/20260113_101332/scenarios/full/res.json` | 193 B | sha256 `7bfca3f5a0bf3dd1a5cc2d70f400fa52f20164cd1dbdce0b35f9cda36eae9748` |
  | `ARMOR/res/D1/20260113_101332/scenarios/full/tmp/pre_interval.json` | 1,300 B | sha256 `c8da7bfa44cf18c44403739b98de8e45e98d1f628ae500c3e0ccb57f46c0e9d7` |
  | `ARMOR/res/D1/20260113_101332/scenarios/full/tmp/pre_types.json` (not fetched) | 252 B | CRC32 `729656f5` |
  | `ARMOR/res/D1/20260113_101332/scenarios/full/tmp/rank_df.csv` (not fetched) | 143,951 B | CRC32 `42f19ba7` |
  | `ARMOR/res/D1/20260113_101332/model.pkl` (not fetched) | 15,016,756 B | CRC32 `82aebfc7` |

  `scenarios/` contains only `full` (D2 likewise). There is no directory, interval file or `res.json` for any drop scenario.
- **ARMOR missing-data comparisons that will be made:** none against ARMOR outputs. Our models under ARMOR's six scenarios are scored against ARMOR's official labels and shown beside ARMOR's released **full-scenario** reference row, labelled as a clean reference, not as ARMOR under missing data.
- **ARMOR missing-data comparisons that will not be made:** (a) no ARMOR-under-missing-data row, because no scenario intervals or results were released; (b) ARMOR is not re-run to create them (its released `model.pkl` is a baseline-team artefact, needs an unrestricted pickle load and is out of scope); (c) any missing-data numbers printed in ARMOR's paper are not reproducible from the release and are not tabulated beside ours; (d) the random-rate experiment has no ARMOR counterpart.
- Validation comparison for context: `results/experiment_v1/missing_summary.json`, `missing_d3_p90_summary.json`.

## 7. Test-run machinery (built, tested on validation and synthetic data, part of the freeze)

1. `ensemble.py`: five-seed ensemble (protocol section 11.3). Validation row: `results/ensemble_validation_v1/` (secondary; never chosen from).
2. `run_test.py`: one-shot runner for v1, v0 and XGBoost (`--mode test`), writing every row above and the paired bootstraps to `results/test_v1/`. It verifies the freeze record (`results/test_freeze/evidence_test_freeze.json`) against disk and refuses on any difference, then writes two markers by exclusive create (`results/test_v1/TEST_RUN_MARKER.json` and `~/298A_project/checkpoints/model1_test_run_marker.json`) before reading any test window; an existing marker refuses every later run. Its `--mode validation` dry run reproduced every committed validation result byte for byte (v1 and v0 predictions, v1 detection and thresholds, D3, D4, XGBoost predictions, 48 seed-0 missing-data cells).
3. `scoring_sets.py` (period scoring sets with the validation rules; test needs the run token) and `official_scoring.py` (official labels and released intervals, pinned, token required; Model 1 alerts scored with the validation construction).
4. `test_freeze.py`: freeze record, verification, markers and the gated unlock (section 2).
5. The test is run from the freeze commit; the run records HEAD in its marker. Deciding to run it is a person's decision, recorded in Linear by them.

## 8. Limits stated in advance

- 84 cases give wide intervals; a difference whose bootstrap interval includes 0 is not reported as a win.
- Validation also selected checkpoints, so validation numbers are slightly optimistic; test numbers are the first unselected ones.
- On validation the label-free D3 (p90) and the no-rolling-max D2 scored a higher F1 than the primary. This plan does not change the primary; those results will be shown beside it as diagnostics.
- Alerts near events that appear only in ARMOR's own label file count as false positives under our scoring.
- XGBoost and the Transformer see the same case windows, but seed SDs measure training randomness only, not uncertainty across incidents.
