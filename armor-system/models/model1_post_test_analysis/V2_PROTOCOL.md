# Model 1 v2: placement-only experiment protocol (Linear 298-51 follow-up)

**Status:** predeclared. Written and committed **before any v2 code, training or v2 output exists**. It fixes the question, the single intervention, the comparison, the evidence, the acceptance criteria and the leakage tests. Nothing here is tuned to a result.

**Governing note:** `V2_PROTOCOL_NOTE.md` (commit `77f4e1e`). The placement idea came from post-test error analysis on the D1 test period. This experiment uses **no D1 test window, label or score**.

**Scope:** placement only. Ensembling, calibration and dropout changes are later phases (section 8) and are **not** part of this experiment.

## 1. Question

Does causally available pod-to-VM placement improve v1's ability to distinguish **Node CPU / Node Disk / Node Memory** faults from **container** faults (Container Hardware, Container Network), in case-level fault-type classification?

The primary outcome is case-level fault type (exact class) on node-family and container-family cases. Detection is reported as a secondary, non-deciding result (section 4c).

## 2. Intervention

### 2.1 Everything else is v1

The model, data and training are v1's:
- **Model:** v1 as frozen (`model_v1.Model1V1`, `training_config_v1.json`, sha256 `49c815d0…`). That means d = 64, 4 heads, 2 factorised blocks (temporal attention, then cross-node attention, then FFN), node-type embedding (pod/VM), no node-ID embedding, the fault and detection heads, input adapters and masks.
- **Data:** the same windows, roles, masks, fit-only scaler and simulated-missing inputs.
- **Training:**
  - Stage 1: masked MSE plus simulated missing.
  - Stage 2: paired batches with fault CE plus detection BCE, initialised from that seed's selected **v1+placement** Stage 1 checkpoint.
  - AdamW, weight decay 0.01, clip 1.0, batch 64. Stage 1 uses lr 1e-3 cosine; Stage 2 uses lr 1e-4 constant. 100 epochs, patience 10, seeds 0–4.
- **Selection:** as V1_PROTOCOL section 3.
  - Stage 1: clean validation `normal_reference` masked MSE.
  - Stage 2: case-level fault NLL plus balanced window BCE on validation.

The **only** change is the placement input below. No hyperparameter is searched or changed.

### 2.2 Placement rule (per window)

- **Source:** a placement table built only from the per-sample graph edges in D1 `samples/{train,test}_samples.pkl` (pod-to-VM edges), read with the repository's non-executing `pickle_symbolic` / `torch_blob` readers.
  - Each sample minute has its timestamp, a `complete` flag (all 40 pods have exactly one VM neighbour) and, if complete, the pod→VM map.
  - The table is **restricted to minutes before the validation/test boundary 1651852800**. Test-period minutes are never written to it.
- **Rule:** for a window with input minutes t0 … t5 (target t6), the placement is the map of the **last complete sample at or before t0** (causal carry-forward). It is constant across the window's six input minutes. Partial (incomplete) graphs are **never** a source. Nothing at or after t1 is consulted.
- **`placement_available`:** True if such a complete sample exists, otherwise False.
  - In D1 non-test data every window is expected to be True: every recorded day starts with a complete sample, and every partial sample has a complete sample 60 s earlier (`PLACEMENT_VALIDITY.md` sections 3–4).
  - The count of False windows is reported. If False, the window uses plain cross-node attention (section 2.3, all relations 0), i.e. exactly v1's attention.
- **VMs hosting no pods:** in the window's map these (e.g. node-1 and node-2 always, node-3 and node-4 under placements A and B) have **zero placement edges**. Their attention is exactly v1's.

### 2.3 How placement enters the model: a learned relation bias in cross-node attention

In every block's cross-node attention, at every input minute, a relation-dependent scalar is added to the pre-softmax score of query node i and key node j:

`score[l,h,i,j] = q_i·k_j / sqrt(16) + β[l, h, r(i,j)]`

The relation r(i,j) is defined from the window's placement map P:

| r | Meaning |
|---|---|
| 1 | i is a pod and j = P(i) (pod → its host VM) |
| 2 | i is a VM and P(j) = i (VM → a pod it hosts) |
| 3 | i ≠ j are pods with P(i) = P(j) (co-hosted pods) |
| 0 | everything else: self pairs, VM–VM, pods on different VMs, any pair involving a VM that hosts no pod, and every pair when `placement_available` is False |

How the bias is set up:
- **Parameters:** β[l,h,0] ≡ 0 is a constant, not a parameter. β[l,h,1..3] are learnable scalars, so **2 layers × 4 heads × 3 relations = 24 new parameters**.
  - They are **initialised to 0**, so at initialisation v1+placement computes exactly v1's function.
  - They are created after all v1 modules and consume no random numbers, so every other initial weight is identical to v1's for the same seed.
  - They sit in the same AdamW parameter group as the other weights (same lr, weight decay and clipping).
- **Masking:** the bias is added before the existing key masking, so padded or invalid keys still get exactly zero attention weight and fully masked rows still return 0.
- **Not changed:** temporal attention, the node-type embedding, adapters, fusion and heads. No placement feature is added to node tokens.
- **Location:** v2 code lives in a new directory outside every frozen prefix (`armor-system/models/model1_v2/`). It may import frozen v1 modules but must not modify them; `test_freeze.py --verify` must stay OK.

## 3. Comparison

**v1 vs v1+placement**, with identical data, windows, seeds 0–4, objectives, optimizer and schedule, selection rules, class weights and scoring code.

- **v1 arm (validation, section 4a):** the frozen v1 checkpoints and their committed validation predictions (`results/experiment_v1/seed_*/predictions.csv`).
- **Pipeline-identity check, before any v1+placement run:**
  1. With `placement_available` forced False everywhere and β = 0, the v2 forward pass on the frozen v1 seed-0 S2 checkpoint must reproduce v1's seed-0 validation `predictions.csv` byte for byte.
  2. A seed-0 v1 retrain through the v2 training path, with placement disabled, must reproduce v1's selected epochs (S1 73, S2 34) and seed-0 validation predictions.

  If either fails, the experiment stops and the cause is recorded.
- **Pairing:** v1 seed s is paired with v1+placement seed s.
- **One run per seed per arm.** There is no re-run with a changed design. Any change to the bias design, relations, rule or hyperparameters is a new protocol.

## 4. Development evidence (the D1 test is never used)

### 4a. Validation (25 cases: CH 10, CN 6, Node Disk 5, Node CPU 3, Node Memory 1)

Reported per seed and as mean ± SD over seeds, for both arms:
- macro-F1, weighted-F1 and per-class recall;
- **node-family exact-class correct** (out of 9) and **node-family family-level correct** (predicted any node class, out of 9);
- **container-family exact-class correct** (out of 16) and container family-level correct;
- 5×5 confusion matrices per seed and summed over seeds;
- a **paired per-case table per seed**: case, true class, root-cause instance and whether its VM hosts pods in the window's map, v1 prediction, v1+placement prediction, and change (improved / worsened / both correct / both wrong).

Caveat (V1_PROTOCOL R4): the 25 validation cases also select the checkpoints in both arms, so this evidence shares data with selection. 4b uses no selection.

### 4b. Time-blocked cross-validation over non-test cases (development only)

- **Folds** follow recorded days and roles, so fold boundaries fall on 24-hour data gaps or on the isolated calibration block:
  - **F1:** fit cases on recorded day 1 (placement A): cases 0–44 (45 cases; node family 15).
  - **F2:** fit cases on recorded day 2 (placement B): cases 45–87 (43 cases; node family 11).
  - **F3:** the validation block, cases 101–125 (25 cases; node family 9).
- **Calibration block isolated:** cases 88–100 and every calibration-role window are never used for training, selection or evaluation in any fold.
- **Per held-out fold:** both arms are trained from scratch on the other folds' windows with the roles' usual objectives:
  - Stage 1 uses those folds' `normal_reference` windows.
  - Stage 2 uses those folds' non-ambiguous event windows plus `normal_reference` windows.
  - The fit-only scaler and the class weights are recomputed on the training folds only, with the same procedures as v1.
- **No checkpoint selection:** each stage trains for a **fixed number of epochs**, the median of v1's selected epochs over seeds 0–4 as recorded in `results/experiment_v1/seed_*/run_meta.json`: **Stage 1 = 77, Stage 2 = 35**. Same seeds 0–4.
- **Embargo:** not needed beyond the fold layout (24-hour gaps or the excluded calibration block separate every pair of folds). Windows never bridge gaps.
- **Reporting:** the same metrics as 4a on each held-out fold, per seed and mean ± SD, including the paired per-case tables.
- **Status:** CV is **development evidence only**. It is not a test and is never pooled with validation or any test number.

### 4c. Secondary (non-deciding)

Validation detection for both arms is reported with v1's unchanged scoring: primary, D2 and D3, thresholds from the same 329 calibration windows. It is not an acceptance criterion.

## 5. Acceptance criteria (in cases; all four must hold)

N_node(arm, s) is the number of the 9 validation node-family cases classified with the **exact** true class by that arm in seed s. "Improved" means v1 wrong and v1+placement correct, for the same case and seed; "newly wrong" means v1 correct and v1+placement wrong.

| # | Criterion |
|---|---|
| A1 | mean over s of [N_node(v1+placement, s) − N_node(v1, s)] ≥ **+1.0 case** |
| A2 | at least one validation node-family case is **improved in ≥ 3 of 5 seeds** (the same case) |
| A3 | **no** validation container-family case (of 16) is newly wrong in more than 1 seed |
| A4 | CV agrees in direction: Σ over F1, F2 of mean over s of [N_node(v1+placement) − N_node(v1)] on the held-out fold **> 0** (exact-class node-family correct cases). F3 is reported but not counted, because it repeats the validation cases. |

**Outcomes:**
- If all four hold: "placement supported on D1 development data". This is still unconfirmed; an untouched set is required (section 7).
- Otherwise: "not supported". There is no follow-up tuning under this protocol.

**Baseline and power, declared before any run** (from the committed v1 validation predictions):
- v1 gets N_node = 7, 7, 7, 8, 5 (mean 6.8 of 9), so A1 needs a mean of at least 7.8.
- v1 gets all 16 container cases right in every seed.
- **A2 is reachable only through cases 104 (Node CPU, root cause node-3) and 124 (Node Memory, node-4).** These are the only validation node cases v1 gets wrong in 3 seeds; 101 is wrong in 2 seeds, and 105, 117 and 118 in 1. Under placement B, which covers all validation windows, node-3 and node-4 host **no pods**, so placement edges do not touch them directly.
- Of the 9 validation node cases, only **case 117** (node-6, 7 hosted pods) has a root-cause VM with placement edges.
- In the CV fit folds, 8 of 15 (F1) and 3 of 11 (F2) node-family cases have a root-cause VM that hosts pods.

Consequence: the validation criteria have **low power for the direct mechanism**. A pass would have to come from indirect effects; a fail is weak evidence against placement. Both outcomes are recorded as such. This limitation is declared here and does not change the criteria.

## 6. Leakage protections (each is a test that must pass before any training)

1. **Placement source.** The table builder reads only `samples/*_samples.pkl` graph edges and `hash_info/node_hash.pkl`.
   - A test asserts it opens no file under `cases/`, does not read `ad_cases.pkl` or `cases.csv`, imports no label module, and uses no root-cause field.
   - The built table's sha256 is recorded before training.
2. **Causal rule.**
   - Unit tests on synthetic timelines: a placement change at minute m is used only by windows with t0 ≥ m; a window with t0 = m − 60 uses the old map.
   - On the real table, for every window used, the source minute is ≤ t0 and is a complete sample.
   - Any lookup consults nothing after t0.
3. **Fault-correlated partial graphs.** The 6 partial samples are never a placement source; a test asserts their timestamps never appear as a source. These are 2022-05-01 02:55, 2022-05-02 17:30, 2022-05-02 20:20, 2022-05-06 22:18, 2022-05-07 11:23 and 2022-05-08 23:42 UTC (3 in the fit period, 3 in the test period). No per-minute edge or "edge missing" signal enters the model.
4. **No test period.** The table holds no minute ≥ 1651852800, and the v2 code refuses the `test` role (it never calls `test_freeze.unlocked`). No D1 test window, label or score is produced by this experiment.
5. **Frozen state.** `test_freeze.py --verify` is OK before and after every run, and no file under a frozen prefix changes.

## 7. Independent-test limitation

- **D1 test status:** any v1+placement result on the D1 test period may be reported only as **"post-hoc, non-independent"**: the idea came from analysing that period. This experiment computes none.
- **Confirmation** needs an untouched evaluation set under a frozen one-shot rule like v1's TEST_PLAN:
  - **D2**, pending the authors' or data owner's permission (`D2_CHECK.md`: licence unclear; whether D2 has placement edges is unknown); or
  - **new testbed incidents** collected after this protocol.
- **Unseen placements:** placement C (node-3 and node-4 hosting pods) occurs only in the D1 test period, so no development data exercises it.

## 8. Later phases (not part of this experiment)

1. **Ensembling and calibration on the frozen v1 checkpoints first.** A seed-averaged detection score and a larger, time-ordered calibration pool, with no retraining. This needs its own predeclared protocol.
2. **Whole-source dropout** (whole modality / whole VM) as a separate controlled change against v1, with its own protocol. It is never combined with placement in the same comparison.
