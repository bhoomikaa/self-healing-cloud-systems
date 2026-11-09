# Model 1 post-test analysis (Linear 298-51)

Status: **post-hoc analysis of the frozen one-shot test run.**
- Run: `dabf6b59c185451e9b3e2a2f930eba30`. Results commit `3ed6e97`, freeze commit `8a4987f`.
- Nothing was retrained, re-selected or re-thresholded. No frozen file was edited, and nothing was written to `results/test_v1/`.
- Every number in `results/test_v1/` stands as committed.

**Rule for anything that follows.** Test-period analyses here may explain the frozen results. They must not change those results, and they must not be used to select, tune or justify any future model. **Any later model (v2) needs a new, untouched evaluation set.** The D1 test period has now been seen, analysed case by case and cannot serve as that set.

This directory sits outside every prefix hashed by `test_freeze.py`. Adding it therefore does not change the frozen tracked-file set.

## 0. Method and provenance

**Fault-type errors** come directly from the saved prediction files: `seed_*/predictions_v1.csv`, `predictions_v0.csv`, `xgboost/seed_*_predictions.csv` and `ensemble/predictions_v1_ensemble.csv`.

**Detection failures needed per-incident detail that the saved results do not hold.** `results/test_v1/` stores only aggregate counts per detector row: TP, FP, interval classes and a delay summary, with no list of which incident was hit. So `reconstruct.py` rebuilt per-window scores and per-incident outcomes:
- **How it ran:** it reused the frozen code paths (`run_test.setup`, `load_models`, `collect`, `collect_cal`, `detection_v1`, `run_detection_v0.evaluate_alerts`) with the frozen checkpoints.
- **Access:** it obtained test access through `test_freeze.resume_token` for the attempt-2 run id. That call re-verified the freeze record and both attempt-2 markers.
- **Reproduction check:** each rebuilt row was compared with the saved row after the same JSON serialisation. **All checks are exact for all five seeds**:
  - the primary, D3 M2 p90 and D3 M2 p99 rows for both label populations (30 rows);
  - the primary and D3 thresholds;
  - the v1-primary and D3-p90 recalibrated cells of the six ARMOR missing-data scenarios, for both populations (120 cells).

  The per-incident lists below therefore break the frozen numbers down; they are not new results.
- **Outputs:**
  - Window scores (`scores_seed_*.npz`) and per-seed detail (`recon_seed_*.json`, `meta.json`) stayed local in the session scratchpad and are not committed. They hold test-period window scores and ARMOR label lists.
  - `analysis.json` holds every table below, including per-label and per-interval lists.

Files:

| File | Role |
|---|---|
| `reconstruct.py` | torch process: per-window scores, per-incident detail, exact reproduction checks |
| `analyze.py` | no torch: failure taxonomy, ranking tables, detection gap, calibration pools → `analysis.json` |
| `bylevel.py` | breakdown by root-cause level and VM |
| `d1_placement.py`, `d1_placement_output.txt` | pod-to-VM edges from the D1 sample graphs, using the non-executing pickle reader |

Families: container = Container Hardware (CH) and Container Network (CN); node = Node CPU, Node Disk, Node Memory. "Ours" = the 84 test case timestamps. "Official" = ARMOR's 98 post-split labels. Late = detection delay > 120 s.

## 1. Failure study

### 1.1 Taxonomy (v1; primary detector for the detection categories)

A **unit** is a distinct failed thing:
- a case, for fault type;
- an incident label, for missed, late and VM-hiding failures;
- a false-alarm episode, meaning overlapping false-positive intervals from any seed merged into one.

An **observation** is one occurrence per seed (and per scenario for VM hiding). The same case failing in several seeds counts as 1 unit and N observations.

| Category | Distinct units | % | Observations | % | Seeds per unit (all 5 / some / one) | Example IDs |
|---|---|---|---|---|---|---|
| Fault type, across family | 18 cases | 21.4% of 84 | 48 | 11.4% of 420 case predictions | 3 / 9 / 6 | case 139 (Node Disk → CN, 5 seeds); case 143 (Node CPU → CH, 5 seeds) |
| Fault type, within family | 11 cases | 13.1% of 84 | 22 | 5.2% of 420 | 0 / 5 / 6 | case 133 (Node Memory → CPU/Disk, 4 seeds); case 172 (Node CPU → Disk, 4 seeds) |
| Missed incident (primary, ours) | 51 labels | 60.7% of 84 | 129 | 30.7% of 420 label-seeds | 6 / 29 / 16 | case 140 (CN, `cartservice`, 5 seeds); case 147 (Node Memory, node-5, 5 seeds) |
| Late detection (> 120 s) | 27 labels | 32.1% of 84 | 45 | 15.5% of 291 hits | 0 / 13 / 14 | case 131 (Node Disk, +246 s in 4 seeds); case 190 (CN, +202–262 s, 3 seeds) |
| False-alarm intervals, all | 46 episodes | – | 87 intervals | 26.5% of 328 intervals | 2 / 19 / 25 | – |
| … in normal regions only | 30 episodes | – | 67 intervals | – | – | 2022-05-06 23:10 UTC (all 5 seeds, 15 normal minutes) |
| … overlapping ignore regions | 15 episodes (+1 mixed) | – | 20 intervals | – | – | 2022-05-07 04:28 UTC (seeds 0, 1); mixed 2022-05-08 18:01 UTC (all 5 seeds) |
| Failure only under VM hiding | 49 labels | 58.3% of 84 | 364 | 28.9% of 1,260 label-seed-scenarios | 15 / 23 / 11 | case 135 (CH, `frontend`, all 5 seeds × 3 scenarios); case 195 (Node Disk, node-3, same) |

**Total.**
- The six categories hold **202 distinct units**: 18 + 11 + 51 + 27 + 46 + 49.
- Some cases appear in several categories. Across the case-based categories the distinct cases number **78 of 84**, and there are 46 false-alarm episodes on top.
- So the distinct units exceed 50 under either count. No failure was invented or padded.

**Definition of "Failure only under VM hiding".** An incident counts when the primary detector hits it on clean data in a seed, misses it in that seed under drop_metric, drop_log+metric or drop_metric+trace (recalibrated thresholds, the reported rows), and is not missed under any of the three non-VM-hiding scenarios in that seed.
- By class and level: CH service 18, CH pod 13, Node Disk 9, CN service 3, Node CPU 3, Node Memory 2, CN pod 1.
- By root-cause level: service 21, pod 14, node 14.

### 1.2 Other models and detectors

**Fault type:**

| Model | Across-family units / obs | Within-family units / obs | Top confusion (obs) |
|---|---|---|---|
| v0 | 18 / 42 | 12 / 22 | Node CPU → CN 11; Node Disk → CN 10 |
| XGBoost | 13 / 40 | 5 / 20 | Node Memory → CN 10; Node CPU → CH 10; Node Disk → CN 10 |
| v1 ensemble | 9 / 9 | 5 / 5 | Node CPU → CN 3 |
| v1 | 18 / 48 | 11 / 22 | Node Disk → CN 11; Node CPU → CN 11; Node CPU → Disk 7 |

Case 143 (Node CPU, root cause node-4) is wrong for every model in every seed.

**Detection:**

| Detector | Missed units / obs | Late units / obs | FP interval obs (normal-only / ignore-overlap) | FP episodes |
|---|---|---|---|---|
| v1 primary | 51 / 129 (6 in all seeds) | 27 / 45 | 87 (67 / 20) | 46 |
| D3 M2 @ p90 | 16 / 79 (15 in all seeds) | 2 / 10 | 104 (89 / 15) | 22 (19 in all seeds) |
| D3 M2 @ p99 | 69 / 334 (64 in all seeds) | 4 / 15 | 8 (8 / 0) | 2 |

D3 misses no Container Hardware incident at p90. Its 16 misses are Node Disk 7, CN 3, Node CPU 3 and Node Memory 3.

**Primary misses by class** (distinct labels / class support):
- CH 20/36
- CN 12/22
- Node CPU 6/8
- Node Disk 10/14
- Node Memory 3/4

**By root-cause level** (`bylevel.py`, observations over 5 seeds):

| Root cause | Cases | Primary miss obs | D3 p90 miss obs | v1 fault-type error obs |
|---|---|---|---|---|
| service (all replicas) | 32 | 31/160 (19%) | 5/160 | 1/160 (0.6%) |
| pod | 26 | 45/130 (35%) | 9/130 | 18/130 (14%) |
| VM node-1 | 4 | 11/20 | 5/20 | 7/20 |
| VM node-2 | 3 | 5/15 | 5/15 | 5/15 |
| VM node-3 | 4 | 2/20 | 20/20 | 1/20 |
| VM node-4 | 4 | 5/20 | 5/20 | 14/20 |
| VM node-5 | 6 | 20/30 | 15/30 | 14/30 |
| VM node-6 | 5 | 10/25 | 15/25 | 10/25 |
| all VM (node level) | 26 | 53/130 (41%) | 65/130 (50%) | 51/130 (39%) |

Node-level root causes hold 26 of 84 cases but 51 of v1's 70 fault-type error observations.

### 1.3 Top three causes and proposed fixes

**Cause 1: node-level faults lack the pod-to-VM link, so v1 confuses them with container faults and with each other.**
- 51 of 70 v1 fault-type error observations have a VM root cause.
- The largest confusions are Node Disk → CN and Node CPU → CN (11 each); v0 and XGBoost show the same pattern.
- v1 treats the 46 nodes without placement edges. Yet D1's own sample graphs carry a per-day pod-to-VM placement (section 3.2).
- node-1 and node-2 host no pod at all. Faults there can show only in those VMs' own metrics.
- *Proposed fix (v2, new evaluation set):* add placement-aware inputs from the D1 graphs, such as VM features joined to their hosted pods and pod aggregates per VM, or graph attention over placement edges.

**Cause 2: the primary's alerting is unstable across seeds, and its threshold depends on few calibration windows.**
- Only 6 of the 51 missed incidents are missed in all five seeds, and none of the 27 late detections is late in all five seeds.
- By contrast, D3 at p90 misses 15 of its 16 in every seed.
- Window-level ranking of the supervised logit is similar across seeds (test AUC-ROC 0.822–0.852, section 2.1). So the alert/no-alert outcome differs mainly at the threshold, not in the overall ranking. This is an observation, not a tested causal claim.
- The p90 threshold sits between the 33rd and 34th highest of 329 calibration windows. ARMOR calibrates on 2,269 samples (section 2.4).
- The seed-averaged ensemble had test F1 0.783, against 0.727 ± 0.056.
- *Proposed fix (v2):* predeclare a seed-averaged detection score as the primary, and calibrate on a larger, time-ordered pool of normal minutes outside the evaluation period.

**Cause 3: detection depends on VM metrics even for container and service faults.**
- 49 incidents hit on clean data are lost only when VMs are hidden; 35 of them are container-class faults (CH/CN) at service or pod level.
- v1 primary F1 falls to 0.595 / 0.555 / 0.406 in the three VM-hiding scenarios, against 0.729–0.739 at 10–40% random missingness.
- v1's training drops random per-block entries, never a whole VM modality (V1_PROTOCOL section 7 caveat).
- *Proposed fix (v2):* train with whole-modality and whole-VM drops, and route pod-level evidence through the placement graph (cause 1) so that container faults remain visible without VM metrics.

## 2. Ranking diagnostics (score vs alerting)

### 2.1 Weak-label ranking diagnostics

These AUCs are **weak-label ranking diagnostics**: the labels are documented fault windows, not verified anomalies. They are **not** anomaly-detection accuracy.
- **Positives and negatives:** `event_associated` windows are positives, `normal_reference` windows are negatives, and `ignore` windows are excluded.
- **Scores:**
  - v1 supervised logit: v1 S2, raw and after the causal 60 s rolling maximum used by the primary;
  - D3 forecast error: M2 max-normalised score of v1 S1, with calibration statistics from the 329 calibration windows.
- **Metrics:** AUC-PR is average precision. Means ± SD are over 5 seeds.

**Validation** (150 event / 411 normal, prevalence 0.267):

| Score | AUC-PR | AUC-ROC | AUC-PR per seed | AUC-ROC per seed |
|---|---|---|---|---|
| v1 supervised logit | 0.715 ± 0.043 | 0.829 ± 0.023 | 0.653, 0.697, 0.749, 0.759, 0.716 | 0.799, 0.820, 0.847, 0.856, 0.824 |
| v1 logit, rolled (primary score) | 0.718 ± 0.046 | 0.835 ± 0.027 | 0.661, 0.694, 0.765, 0.766, 0.706 | 0.800, 0.826, 0.864, 0.862, 0.824 |
| D3 forecast error (M2) | 0.551 ± 0.005 | 0.709 ± 0.004 | 0.550, 0.549, 0.552, 0.559, 0.547 | 0.704, 0.707, 0.709, 0.714, 0.710 |

**Test: post-hoc analysis, not used for any selection** (501 event / 1,400 normal, prevalence 0.264):

| Score | AUC-PR | AUC-ROC | AUC-PR per seed | AUC-ROC per seed |
|---|---|---|---|---|
| v1 supervised logit | 0.677 ± 0.039 | 0.835 ± 0.013 | 0.666, 0.705, 0.697, 0.702, 0.613 | 0.823, 0.839, 0.839, 0.852, 0.822 |
| v1 logit, rolled (primary score) | 0.676 ± 0.048 | 0.838 ± 0.014 | 0.670, 0.719, 0.695, 0.699, 0.596 | 0.828, 0.848, 0.839, 0.855, 0.820 |
| D3 forecast error (M2) | 0.552 ± 0.002 | 0.737 ± 0.002 | 0.554, 0.550, 0.549, 0.554, 0.553 | 0.737, 0.735, 0.736, 0.741, 0.736 |

On this weak-label ranking, D3's window score ranks lower than the supervised logit on both periods. At the alerting level, however, D3 at p90 scored higher incident F1 than the primary (0.788 vs 0.727, ours). Window ranking and interval-level alerting measure different things.

### 2.2 Detection gap against ARMOR's released intervals (official labels)

ARMOR's released full-scenario intervals hit **98 of 98** official labels. The table counts official labels that ARMOR hits but that both the Model 1 primary and D3 miss in the same seed.

Of the 98 labels, 21 have no case record: the nearest case is 463–2,246 s away. This is a known `ad_cases.pkl`/`cases.csv` mismatch inherited from ART (`d1_validation_report.md` section 4). The class breakdown of the 98 labels is CH 35, CN 20, Node Disk 14, Node CPU 7, Node Memory 1, no case record 21.

| Gap | Labels missed in ≥1 seed | in all 5 seeds | Observations | By class, ≥1 seed | By class, all 5 seeds |
|---|---|---|---|---|---|
| Primary and D3 @ p99 (the A1/A2 rows) | 54 | 9 | 147 | CH 15, CN 9, Node Disk 10, Node CPU 5, no record 15 | no record 5, Node CPU 2, CN 1, CH 1 |
| Primary and D3 @ p90 | 10 | 4 | 30 | Node Disk 4, CN 2, Node CPU 1, no record 3 | no record 3, Node CPU 1 |

Labels missed in all 5 seeds by both the primary and D3 @ p90 (UTC):

| Time (UTC) | Case | Fault type / nearest case |
|---|---|---|
| 1651911660 | none | no record; nearest case 1,527 s, CN |
| 1651914243 | none | no record; 1,033 s, Node Disk |
| 1652035716 | case 172 | Node CPU, node-5 |
| 1652104020 | none | no record; 609 s, Node CPU |

### 2.3 Calibration pools

Model 1 thresholds (primary, D1, D2, D3 and D4) all come from the **329 calibration `normal_reference` windows**. That is 329 minutes, and adjacent windows share 6 of 7 minutes; V1_PROTOCOL section 8 estimates the effective sample size at 128–170.

Which calibration windows determine each percentile (linear interpolation, position (n−1)·p):

| Percentile | Position | Interpolates between windows ranked (from the top) | Windows at or above the upper one |
|---|---|---|---|
| p90 (primary, D3 p90) | 295.2 | 34th and 33rd | 34 |
| p95 | 311.6 | 18th and 17th | 18 |
| p97 | 318.16 | 11th and 10th | 11 |
| p99 (D3 A2 row, D4) | 324.72 | 5th and 4th | 5 |
| p99.5 | 326.36 | 3rd and 2nd | 3 |

**ARMOR's pool** (`anomaly_detector.AD`): ARMOR calibrates on `init_samples` = training-file samples with timestamp ≤ the AD split (1651852800).
- Counted from the feature cache, that is **2,269 samples** (minutes). This is what the "about 2,270" in V1_PROTOCOL refers to.
- ARMOR's sequence windowing (`SLD`, `t_value`) may drop a few at run boundaries; that exact count was not measured.
- ARMOR then fits SPOT (q 0.1, level 0.95) on scores of that pool. So ARMOR's pool is about 6.9× Model 1's 329 windows. Model 1's p99 threshold is set by about 5 windows; ARMOR's level-0.95 initial threshold sits about 113 samples from the top.

## 3. Independent data check (nothing downloaded)

### 3.1 D2

**Availability**

| Item | Finding | Status |
|---|---|---|
| In the ARMOR Zenodo package (record 21744694, `ARMOR.zip` 174,195,955 B, CC-BY-4.0, open access) | `ARMOR/data/D2.zip`, **93,883,410 B** (CRC `487315f6`); `ARMOR/config/D2.yaml` 1,379 B | VERIFIED (local outer listing `armor-system/data/manifests/armor_d1/outer_archive_listing.csv`; Zenodo API) |
| ARMOR's released D2 results in the same package | `ARMOR/res/D2/20260114_222628/`: `model.pkl` 4,643,092 B, `scenarios/full/res.json` 194 B, `tmp/pre_interval.json` **156 B**, `tmp/pre_types.json` 162 B, `tmp/rank_df.csv` 29,906 B | VERIFIED (listing); contents not read |
| Labels and intervals | Per ARMOR's README, D2 has the same layout as D1: `cases/ad_cases.pkl` (failure timestamps), `cases/cases.csv` (injection time, level, root cause, failure type), `hash_info/`, `samples/{samples,train_samples,test_samples}.pkl`. So failure labels, root causes and types should be included, and released D2 AD intervals exist (`pre_interval.json`). | README statement VERIFIED; archive contents UNVERIFIED (not downloaded) |
| Elsewhere | The ART GitHub repo (`bbyldebb/ART`) also holds `data/D2.zip`, same size 93,883,410 B, committed 2024-06-06, plus its own `res/D2/` (its `pre_interval.json` is 2,366 B). The ART repo has no licence file. ART's `D1.zip` is byte-identical to ARMOR's (git blob hash checked). | VERIFIED (GitHub tree API); equality of the two D2 files UNVERIFIED (size only) |
| Not available from | aiops-challenge.com (linked, no D2 download found); the raw-D1 anonymous.4open.science repository (D1 only) | VERIFIED as far as searched |
| Licence / NDA | ART's README says of D2: "Due to the non-disclosure agreement, we cannot make it publicly available", yet ART ships `data/D2.zip`, and ARMOR redistributes it under CC-BY-4.0. Project decision (298-43 report; `team_handoff_readiness_audit.md`:147; `fetch_d1.py`): D2 is not downloaded because of this conflict. | VERIFIED |

D2 is described as 18 instances of a bank management system: microservices, servers, databases and dockers. It covers six failure types (Memory, CPU, Network, Disk, JVM-Memory, JVM-CPU), labelled January–June 2021, and was used in the AIOps Challenge 2021.

**ARMOR's published numbers** (arXiv 2603.25538, Table 1, complete data; FT is weighted; three decimals in the paper):

| Dataset | AD P | AD R | AD F1 | FT P | FT R | FT F1 (weighted) | RCL Top1 | RCL Top3 | RCL AVG@5 |
|---|---|---|---|---|---|---|---|---|---|
| D1 | 0.925 | 1.0 | 0.961 | 0.946 | 0.941 | 0.938 | 0.821 | 0.941 | 0.910 |
| D2 | 0.993 | 1.0 | 0.997 | 0.882 | 0.870 | 0.869 | 0.815 | 0.907 | 0.893 |

The D1 four-decimal values used in TEST_PLAN (0.9245 / 1.0 / 0.9608) come from re-scoring ARMOR's released D1 intervals; they round to the paper's values.

### 3.2 D1 pod-to-VM placement

**Yes: D1's preprocessed release contains pod-to-VM placement in the per-sample graphs.** Everything below is VERIFIED by `d1_placement.py`, which uses the repository's non-executing `pickle_symbolic` and `torch_blob` readers; `pickle.load` is never used. The output is in `d1_placement_output.txt`.

- **Edge sets:** 11 distinct sets over the 7,200 samples. Every set has 46 self-loops and the same 142 service-to-service call edges, and VM–VM edges are self-loops only.
- **Placement:** service-to-VM edges appear in both directions. In the 5 complete edge sets (268 edges), which cover 7,194 of the 7,200 samples, **each of the 40 service instances has exactly one VM neighbour**, so the graphs encode a placement map. They hold 3 distinct maps.
- **The placement changes by day** (UTC+8):
  - May 1 and May 9: node-5 hosts 6 pods, node-6 hosts 34.
  - May 3 and May 5: node-6 hosts 7 pods (the `*2-0` services), node-5 hosts 33.
  - May 7: node-3 hosts 11, node-4 hosts 1, node-5 hosts 28.
- **node-1 and node-2 host no pod** in any edge set.
- **Degraded edge sets:** six sets, each used by exactly 1 test sample, have some service-to-VM edges missing. In five of them every pod of one VM loses its link at once; in the sixth, 3 of node-6's 7 pods lose theirs. No service ever has more than one VM neighbour. One reading is a gap in the deployment record rather than a migration; this is UNVERIFIED.
- **Raw telemetry:** the raw-D1 repository listing (anonymous.4open.science) names only `README.md`, which points to an external download of `data/<date>/{log,metric,trace}/` and `groundtruth/`. No placement file is named. Whether the raw metrics carry pod-to-node labels is UNVERIFIED (not downloaded).

Model 1 (v0 and v1) does not use these graph edges.
