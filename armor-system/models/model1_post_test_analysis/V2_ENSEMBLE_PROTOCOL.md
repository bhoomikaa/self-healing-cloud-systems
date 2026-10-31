# Model 1 v2: frozen-v1 seed-ensemble protocol (predeclared, before any v2 code)

**Status:** written and committed before any v2 code, CV training or v2 output exists. The choice and its development rationale are in `PLACEMENT_RELEVANCE.md` (commit `33b532a`); the reporting rules are in `V2_PROTOCOL_NOTE.md` (`77f4e1e`).

**No D1 test access.** This protocol reads no D1 test window, label or score. The v2 code refuses the `test` role and never calls `test_freeze.unlocked`.

The placement protocol (`V2_PROTOCOL.md`, `9cde327`) stays committed and is **not run**.

## 1. Question

Does averaging the five v1 seeds (probability-averaged ensemble) improve incident detection on held-out development data, compared with a single seed, without losing fault-type accuracy?

## 2. v2 definition (no free parameters)

v2 is built from the five **frozen** v1 Stage 2 checkpoints, seeds 0–4: `~/298A_project/checkpoints/model1_v1/seed_*/s2_best.pt`, hashes in `results/test_freeze/evidence_test_freeze.json`.

- **Fault type:** for each case, the mean over seeds of v1's case probabilities (mean window softmax per case); predict the argmax, ties to the lowest class id.
- **Detection:**
  1. per window, take the mean over seeds of sigmoid(v1 detection logit);
  2. apply the causal 60 s rolling maximum (same period);
  3. threshold at the empirical 90th percentile (linear) of the rolled ensemble score on the **329 calibration `normal_reference` windows**;
  4. score with ARMOR's interval rules: 600 s merging, singletons dropped, ±300 s impact window.

This is exactly `model1_transformer/ensemble.py` (frozen, sha256 `78daf917…`), i.e. V1_PROTOCOL section 11.3's secondary row promoted to the primary. Nothing is retrained, refitted or re-thresholded.

The D3 label-free detector, the missing-data handling and every other v1 component are unchanged. They are reported for v1 only.

## 3. Development evidence

### 3a. Validation (frozen checkpoints, 25 cases / 25 incidents)

v2's validation numbers were computed before the D1 test run (`1b27b04`). They are **recomputed** with the v2 code and must equal `results/ensemble_validation_v1/ensemble_validation.json` (predictions file byte-identical; detection and fault dicts equal). If they differ, the work stops.

Validation is **not deciding**: it was seen when v2 was chosen, and the same 25 cases selected v1's checkpoints. It is reported against the five single seeds, with the paired per-case fault table and the per-incident hit table.

### 3b. Day-blocked development CV (deciding)

Folds follow recorded days and roles. Every fold boundary is a 24-hour data gap or the excluded calibration block, so no window straddles two folds.

| Fold | Held-out windows | Held-out cases (incidents) |
|---|---|---|
| F1 | fit-role windows with target minute ≤ 2022-05-01 15:59 UTC (recorded day 1) | cases 0–44 (45) |
| F2 | fit-role windows with target minute ≥ 2022-05-02 16:00 UTC (recorded day 2) | cases 45–87 (43) |
| F3 | validation-role windows | cases 101–125 (25) |

- **Never used in any fold:** the calibration block (calibration-role windows, cases 88–100), embargo windows and the test period.
- **Training per held-out fold:** five models (seeds 0–4) are trained with the **v1 recipe** on the other two folds' windows:
  - Stage 1: masked MSE with simulated missing on their `normal_reference` windows.
  - Stage 2: from that model's Stage 1, paired batches over their non-ambiguous `event_associated` windows plus `normal_reference` windows, with fault CE plus detection BCE.
  - Same architecture (`Model1V1`), config (`training_config_v1.json`), optimizer, schedule and seeds as v1.
  - Class weights are recomputed from the training folds' cases with v1's `CaseBalancedWeights`.
- **Fixed epochs, no checkpoint selection:** Stage 1 runs 77 epochs and Stage 2 runs 35. These are the medians of v1's selected epochs (`results/experiment_v1/seed_*/run_meta.json`: S1 73, 91, 77, 80, 57; S2 34, 31, 50, 51, 35). The cosine schedule keeps v1's T_max = 100.
- **Scaler:** the frozen fit-only scaler is kept for every fold, because refitting would change the applicable-entry mask and with it the architecture's channel lists. In F1 and F2 the held-out day's **unlabelled feature statistics** therefore enter the normalisation. No label enters it. This is a declared limitation.
- **Per held-out fold and seed:** single-seed fault type (case-level), and single-seed detection with v1's primary rule (thresholds from the same 329 calibration windows, scored on all held-out windows with the held-out cases' timestamps as incident labels).
- **Per held-out fold:** the v2 ensemble of that fold's five models, built with `ensemble.py`.

### 3c. Acceptance (deciding folds F1 and F2; F3 reported)

All three must hold. "Mean single" means the mean over the five single-seed models of that fold.

| # | Criterion |
|---|---|
| E1 | In **both** F1 and F2, ensemble incident F1 > mean single-seed incident F1. |
| E2 | Summed over F1 and F2: ensemble missed incidents ≤ mean single missed incidents, **and** ensemble false-positive intervals ≤ mean single false-positive intervals + 2. |
| E3 | Summed over F1 and F2: ensemble exact-class correct held-out cases ≥ mean single exact-class correct cases − 1. |

- **If all three hold:** v2 = the ensemble of the five frozen v1 checkpoints (section 2).
- **Otherwise:** v2 = v1 unchanged, and the result is reported as negative.

There are no re-runs, no alternative ensembling rules and no threshold changes.

## 4. Ablations (reported, never used to choose)

- **AB1, ensemble size:** for k = 1…5, the mean detection F1 and fault macro-F1 over all C(5, k) seed subsets, on validation (frozen checkpoints) and on each CV fold.
- **AB2, in-sample calibration pool:** the ensemble threshold at the 90th percentile of the 329 calibration windows **plus** the 1,312 fit `normal_reference` windows (in-sample for the detection head, as in ARMOR's pool). Validation only, frozen checkpoints.
- **AB3, F3 vs frozen v1:** F3's fixed-epoch models compared with the frozen, validation-selected v1 on the same 25 cases. This shows how much early stopping on validation flatters v1's validation numbers.

## 5. Leakage and integrity checks (tests that must pass before CV training)

1. **Fold construction:** held-out windows never appear in a training dataset; every training batch holds only training-fold windows (relabelled role `fit`); calibration, embargo and test windows never appear in any training or held-out set.
2. **Fold layout:** each held-out fold's windows lie wholly inside its time block. Case sets are disjoint and equal the lists in 3b.
3. **No test role:** the v2 data objects contain no test-period window. A test asserts that the fold data has no window with target minute ≥ 1651852800.
4. **Frozen state:** `test_freeze.py --verify` is OK before and after. No file under a frozen prefix changes; v2 code lives in `armor-system/models/model1_v2/`.
5. **Frozen-v1 reproduction:** v2's validation ensemble equals the committed one (3a).
6. **Determinism:** one CV model (F3, seed 0) is trained twice. Its held-out outputs must be identical.

## 6. Reporting

**Results:** `model1_v2/results/`:
- per-fold, per-seed JSON;
- ensemble JSON;
- predictions CSVs (case-level, no raw data);
- figures;
- `evidence_model1_v2.json`: code hashes, fold definitions, checkpoint hashes (checkpoints stay outside the repository in `~/298A_project/checkpoints/model1_v2_cv/`), output hashes, the determinism check and the freeze verification.

**D1 test:** v2's D1 test numbers are the committed one-shot run's ensemble row (`results/test_v1/ensemble/`). They are reported **only as post-hoc, non-independent**, because v2 was chosen after that row was observed. No new D1 test computation is made.
