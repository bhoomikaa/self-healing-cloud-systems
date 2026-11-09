# Placement relevance on development data, and the v2 choice (Linear 298-51)

Status: analysis of **D1 development data only**: the fit, calibration-block and validation cases, the committed v1 validation predictions, and the placement timeline from `placement_timeline.py`. No D1 test window, label or score is used here. Written before any v2 code.

## 1. How much of v1's development error could placement touch?

Placement adds pod–VM relations. It can act directly only on cases whose root-cause VM **hosts pods** in that period's placement map. In the non-test periods there are two maps (`PLACEMENT_VALIDITY.md`):
- **A** (recorded day 1): node-5 hosts 6 pods, node-6 hosts 34.
- **B** (day 2, the calibration block and validation): node-5 hosts 33, node-6 hosts 7.

node-1 to node-4 host **no pods** in either map.

| Period (cases) | Node-family cases | … on a pod-hosting VM | Cases |
|---|---|---|---|
| Fit, day 1 (0–44) | 15 | 8 | 0, 1, 3, 16, 24, 30, 36, 39 |
| Fit, day 2 (45–87) | 11 | 3 | 81, 83, 84 |
| Calibration block (88–100) | 6 | 1 | 88 |
| Validation (101–125) | 9 | 1 | 117 |
| **All development** | **41** | **13 (32%)** | |

v1 validation errors (committed `results/experiment_v1/seed_*/predictions.csv`, 5 seeds × 25 cases):
- **Container family:** 0 errors in 80 case-predictions.
- **Node family:** 11 errors in 45 case-predictions. By case: 104 (Node CPU, node-3) ×3, 124 (Node Memory, node-4) ×3, 101 (Node Disk, node-4) ×2, 105 (Node Disk, node-1) ×1, 118 (Node CPU, node-4) ×1, 117 (Node CPU, node-6) ×1.
- **Errors on a pod-hosting VM: 1 of 11** (case 117, seed 3).

So on development data, placement edges can directly address at most **1 of v1's 11 validation fault-type errors**. The other 10 are on VMs that host no pods under map B, where the relation bias of `V2_PROTOCOL.md` is exactly zero.

The fit-period errors are in-sample and are not used as evidence. The only development set that would exercise placement more is the day-blocked CV (8 + 3 pod-hosting node cases).

## 2. Cost of the placement experiment as predeclared (`V2_PROTOCOL.md`)

- **Trainings:** 5 main runs plus 30 CV runs (2 arms × 3 folds × 5 seeds) plus 1 reproduction run, about 36 trainings. At v1's measured ~3,090 s per seed that is ~31 CPU-hours.
- **Power:** low. Criterion A2 can be met only through cases 104 and 124, neither of which placement touches directly.

## 3. The other weakness the development data shows: seed instability in detection

These validation results were committed **before** the D1 test run (`1b27b04`, 2026-10-27; test run `3ed6e97`, 2026-10-28):

| Row (validation, 25 incidents) | Detection F1 | P | R | Fault macro-F1 | Fault weighted-F1 |
|---|---|---|---|---|---|
| v1 primary, single seeds (mean ± SD) | 0.680 ± 0.069 | 0.750 | 0.632 | 0.784 ± 0.090 | 0.907 ± 0.043 |
| v1 five-seed ensemble (V1_PROTOCOL 11.3 secondary row) | 0.735 | 0.750 | 0.720 | 0.785 | 0.942 |

- **Detection:** the ensemble's recall (18 of 25 incidents hit) is above the single-seed mean (15.8). Its precision is the same.
- **Cost:** it needs no retraining, adds no parameter and no new threshold rule. It reuses v1's predeclared ensemble definition (`ensemble.py`, frozen).

## 4. Choice for v2

**v2 = the probability-averaged five-seed ensemble of the frozen v1 checkpoints**, with v1's unchanged calibration rule:
- **Detection:** the 60 s causal rolling max of the mean sigmoid, thresholded at the 90th percentile of the 329 calibration windows.
- **Fault type:** the mean of case probabilities.

Why this and not placement:
1. **Measurable value on development data.** It targets the instability that is visible on validation (single-seed F1 SD 0.069), while placement can touch 1 of 11 validation errors.
2. **Minimal complexity and no new free parameters.** It retrains nothing and leaves every frozen v1 file and checkpoint unchanged.
3. **Testable with new development evidence.** A day-blocked CV with retrained v1-recipe models (15 trainings, one arm) gives held-out folds that played no part in this choice.

**Larger calibration pool:** no larger *out-of-sample* normal pool exists in D1 development data. The only larger pool would add the fit-period normal windows, which are in-sample for the detection head (it is trained on them as negatives), as ARMOR's own pool is. It is therefore evaluated only as a non-selectable ablation, not as part of v2.

**Placement** (`V2_PROTOCOL.md`, `9cde327`) is **deferred, not run**. Its protocol stays as committed. It is the candidate for an untouched evaluation set where VMs that host pods carry node faults (D2 if permitted, or testbed incidents).

**Integrity note:**
- This choice is justified here from development evidence only.
- The D1 test row of the same ensemble was **already observed** in the one-shot run (`3ed6e97`, a predeclared secondary row), and the post-test analysis discussed seed instability on test. The choice therefore cannot be called blind to the D1 test.
- Accordingly, v2's D1 test numbers are **post-hoc and non-independent** (`V2_PROTOCOL_NOTE.md`). No claim of independent superiority over v1 or ARMOR is made from them.
