# Model 1 v0 design specification

Spatio-Temporal Multimodal Transformer for cloud incident detection and fault-type classification (DATA 298A/298B, Team 1, Topic 6).

| | |
|---|---|
| Status | **Design v0 frozen on 2026-10-26; calibration protocol added the same day (section 5.6, **approved for Model 1 v0** by the project owner on 2026-10-27 after review). No model code exists.** Nothing here is a result. |
| Task | M1-02 (no Linear issue exists yet for it; see section 10). Implementation requirements for the next task, M1-03, are in section 12 |
| Based on | `origin/main` at `b156d55` (PR #35 merged) |
| Authority | The approved split and preprocessing documented in `armor-system/docs/d1_validation_report.md` and `armor-system/docs/d1_data_preparation_runbook.md` win over this note. If they disagree, this note is wrong. |

**Labels used in this note.**
**[V]** = verified against a repository artifact on the date above (appendix A reproduces it).
**[D]** = initial engineering default: a sensible small starting value, *not* a research-proven optimum, and may be changed only through the amendment rule in section 10.
**[A]** = assumption not verified.
**[U]** = unresolved; needs a decision.

---

## 1. Research objective and capstone role

**Objective.** Build one Transformer model that, from one-minute multimodal telemetry of a microservice system, (a) scores how abnormal the next minute looks (detection), and (b) given an incident, names its fault type among five classes: Container Hardware, Container Network, Node CPU, Node Disk, Node Memory.

**Why this is a research question and not only an engineering task.** Metrics, logs and traces arrive as three different kinds of numbers, are missing for some components, and cover only a few hundred labelled incidents. The model must use all three kinds, ignore what does not exist, and still be honest about how little data it has seen.

**Role in the capstone.**

| Workstream | Owner (approved) | Relation to this note |
|---|---|---|
| Model 1 (this note) | Khushi + Bhoomika | Our own model, trained by us on ARMOR D1 now and on our Online Boutique testbed later |
| ARMOR baseline reproduction | three other teammates | Separate. Its official split and scoring are **never changed** by Model 1 (section 7) |
| Airflow / cloud pipeline | Bhoomika | Will later feed live telemetry to the same input contract (section 2) |

Plain English: the baseline tells us what a published method achieves on this data. Model 1 is the method we build and can deploy; the comparison between them must be fair and clearly labelled.

---

## 2. Input and output contracts

All shapes below are for D1 with the approved preprocessing. Dimensions are read from `armor-system/data/artifacts/d1/inspection/schema.json`, never typed into model code.

### 2.1 Window sample (what the data layer must deliver)

Produced by `prepare_window` in `armor-system/data/prep/preprocess.py` [V: signature and return keys].

| Field | Shape | Meaning |
|---|---|---|
| `inputs` | (6, 46, 130) float32 | six consecutive minutes, train-period scaled (`scaler_model1_train_period_v1.npz`) |
| `target` | (46, 130) float32 | the seventh minute, scaled the same way |
| `timestamps` | (7,) int64 | consecutive, exactly 60 s apart (the function raises otherwise) |
| `structural_absent` | (46, 3) bool | existing node-by-modality block mask (section 3.1, layer 1) |
| `observed_all_zero_not_structural` | (6, 46, 3) bool | existing diagnostic: a block that is all zero in a minute, not structurally absent |

The v0 Dataset adds these fields (proposed, not existing):

| Field | Shape | Meaning |
|---|---|---|
| `applicable_entry` | (46, 130) bool | new per-entry mask, section 3.2 (static for D1) |
| `observed_entry` | (6, 46, 130) bool | dynamic availability; **all True for D1**, used for the future testbed |
| `category` | str in {normal_reference, event_associated, ignore} | internal training category, section 5 |
| `case_index`, `fault_class` | int, int in 0..4 or -1 | set only for `event_associated` windows |
| `split` | str | train / validation / test |

**Label mapping.** The raw category ids in `schema.json` (`category_ids`) are 0, 1, 3, 4, 5 (id 2 is unused). The model uses a contiguous order taken from `CLASSES` in `armor-system/data/prep/splits.py`:

| Model class | Name | Raw id |
|---|---|---|
| 0 | Container Hardware | 0 |
| 1 | Container Network | 1 |
| 2 | Node CPU | 3 |
| 3 | Node Disk | 4 |
| 4 | Node Memory | 5 |

The five fault classes and the three training categories are **independent things**. A training category says "how much do we trust the label of this window"; a fault class says "which fault was injected".

### 2.2 Model I/O

| | Shape | Meaning |
|---|---|---|
| Core input: `tokens` | (B, T, N, M, d) | per-node, per-minute, per-modality vectors from the adapter (M = 3 modalities) |
| Core input: `token_avail` | (B, T, N, M) bool | whether each modality token has any usable entry |
| Core input: `node_type` | (N,) int | 0 = pod, 1 = VM (D1); a future dataset may have more types |
| Core input: `node_valid` | (B, N) bool | padding mask so batches may have different node counts |
| Fault output: `logits` | (B, 5) | one score per class |
| Prediction output | (B, N, C) | next-minute estimate in the dataset's own channel space, written by the adapter |
| Anomaly output | (B,) and (B, N) | window score and per-node score, section 4.5 |

B = batch size, T = 6, N = 46, C = 130 for D1. In the core only `d` is a fixed number; N, C, M and node types come from configuration.

---

## 3. D1 feature adapters and mask rules

### 3.1 What was verified about the masks [V]

Counts come from `scaler_model1_train_period_v1.npz` and the cache `armor-system/data/processed/d1_features_v1.npz` (git-ignored; rebuilt by `data/pipeline.py --stage cache`).

**Modalities** (`schema.json`, `modality_index_ranges`): metric = channels 0-52 and 73-129 (110 channels), trace = 53-61 (9), log = 62-72 (11). Total 130.

**Node types**: 40 pod-type nodes and 6 VM nodes (`node-1` ... `node-6`).

**Existing masks (written by `fit_scaler` and `structural_absence` in `preprocess.py`):**

| Mask | Shape | Exact meaning | Verified content |
|---|---|---|---|
| `absent_block` | (46, 3) bool | node-modality block that is **exactly zero in every training-period sample** (3,445 samples before 1651713858) | 6 VM x (trace, log) and 10 pod x log = 22 blocks; metric: none |
| `absent_entry` | (46, 130) bool | `absent_block` copied to every channel of that modality | 230 entries (54 trace + 176 log) |
| `constant` | (46, 130) bool | entry with zero spread in the fit set (kept finite by scale = 1) | 4,206 entries: 4,199 all-zero, 7 constant non-zero |

**Finding 1: the existing masks are correct but too coarse.** They only know whole modality blocks. In the fit period **4,199 more entries are exactly zero for every sample** yet are *not* marked absent, because a block is only absent if *all* its channels are zero. Examples: a VM node has all 53 `kpi_container_*` channels permanently zero; a pod node has all `system.*` channels permanently zero.

**Finding 2: pods and VMs have disjoint channel sets** (observed: "channel is non-zero in at least one sample on at least one node of the type" over the whole Model 1 train period; the declared rule of Finding 3 is wider, 57 VM metric channels and 53 / 9 / 11 pod channels, and contains every observed active channel):

| Node type | Active metric channels | Active trace | Active log | Name pattern |
|---|---|---|---|---|
| VM (6 nodes) | 46 | 0 | 0 | all 46 are `system.*`; none is `kpi_*` |
| Pod (40 nodes) | 28 | 9 | 10 | all 28 are `kpi_container_*`; none is `system.*` |
| Shared by both types | 0 | | | |

The name pattern agrees with the numeric result (host agent metrics versus container metrics), so the difference is a real property of the data, not an accident of zeros.

**Finding 3: a type-by-channel mask exists and is reliable, if it is declared and not learned from zeros.** Define the **declared rule** from the node type and the channel's source: VM nodes (`node-*`) use only the metric channels named `system.*` (57); pod-type nodes use only the metric channels named `kpi_container_*` (53), all 9 trace channels and all 11 log channels; a VM has no trace or log. Then remove the existing layer-1 blocks (`absent_entry`). `applicable_entry` = declared rule AND NOT `absent_entry` has **3,152 applicable entries of 5,980 (52.7%)**: VM 342 (6 x 57), pod metric 2,120 (40 x 53), pod trace 360 (40 x 9), pod log 330 (30 x 11). **0 non-zero values fall outside this mask in any period** (days 1-2, calibration region, validation, test; the test period is a fixed integrity check only). The rule uses channel names, not observed zeros, so it does not depend on which period is used to fit anything.

**Finding 4: learning the type mask from a time period is fragile (a correction to the first version of this note).** The first version derived the type mask from "any non-zero value in the fit period" and reported 2,056 entries with 0 violations. That held only because the whole train period was used. With the strict calibration split of section 5.6, a mask fit on the fit region alone misses the VM channel `system.disk.readonly`: it is zero on every VM during the fit region and is non-zero on `node-6` for 180 minutes in the calibration region [V]. A disk going read-only is plausibly a fault signature, so hiding it would be harmful. Likewise 62 entries are all-zero for a node in the train period but non-zero later (e.g. `kpi_container_network_receive_packets_dropped`). Therefore masks must be **declared by node type and channel source, never inferred from zeros**, and per-node-only masks are forbidden.

**Finding 5: zeros are not missing values.** Among applicable entries 62.7% are exactly zero in the fit region, and 1,604 applicable entries are all-zero there; they are legitimate zeros (counters that did not move), not gaps. ARMOR scaled the features before release, so a zero is roughly the series mean [A: the exact ARMOR scaling recipe is undocumented; the zero-is-not-missing conclusion rests on these fractions and on Finding 4, not on the recipe].

**Finding 6: the log block is absent by node, not by type.** 10 pod nodes (the `*2-0` nodes, e.g. `adservice2-0`) have no log block in the fit region; 30 do. This is layer 1 and stays data-derived (it is identical when fit on the fit region only, section 5.6).

### 3.2 Three mask layers and the per-entry interface

| Layer | Name | Source | Meaning | D1 value |
|---|---|---|---|---|
| 1 | `absent_block` / `absent_entry` | existing (`fit_scaler`) | the block does not exist for this node | unchanged, 230 entries |
| 2 | `applicable_entry` | **new**, **declared** by the dataset adapter configuration (node type x channel source rule of Finding 3), validated against data but not fitted | this channel is not a real measurement for this node type | 3,152 applicable |
| 3 | `observed_entry` (B, T, N, C) | **new**, from the data source | the entry is applicable but was not delivered this minute | all True for D1 |

Effective mask used everywhere: `effective = applicable_entry & observed_entry` (layer 1 is already inside layer 2).

**Rules**

1. A value is "unavailable" only if layer 2 or layer 3 says so. A zero value alone never makes an entry unavailable.
2. Unavailable entries are replaced by 0 before any layer sees them, and the adapter additionally receives the mask itself (so it can tell a masked 0 from a real 0).
3. Unavailable entries contribute nothing to any loss, score, threshold statistic or class decision.
4. Layer 1 is fit on the **fit region only** (section 5.6) and layer 2 is declared; both are applied unchanged to calibration, validation, test and live data. A data check that finds a non-zero value outside the mask (test MI-6) **fails the run and is reported**, it does not silently extend the mask.
5. An applicable entry that is all-zero in the fit region stays applicable (1,604 of them).

### 3.3 Adapters

A D1 adapter turns the (46, 130) minute into tokens for the dataset-independent core.

- **Pod adapter** (node type 0): three small linear layers, input = `[values * mask, mask]` of the type's own channels: metric 53 -> d, trace 9 -> d, log 11 -> d.
- **VM adapter** (node type 1): metric 57 -> d. Trace and log tokens are never created (`token_avail` False).
- Channel index lists per (type, modality) come from `schema.json` plus the layer-2 mask, so adding a node type or a modality changes configuration, not core code.
- **Output adapter** (prediction): type-specific linear d -> 73 (pod: 53 + 9 + 11 channels) or d -> 57 (VM), scattered back to the 130-channel layout. This is the only place that knows 130.

### 3.4 Testable mask-invariance requirements (to be written as unit tests with the model)

| Test | Statement |
|---|---|
| MI-1 value invariance | Replace the value of every unavailable entry (inputs and target) with large random numbers: `logits`, window score, per-node scores and loss are identical (tolerance 1e-6) |
| MI-2 gradient | The gradient of any loss with respect to unavailable input entries is exactly 0 |
| MI-3 not vacuous | Changing a *legitimate zero* at an applicable entry to 0.5 **does** change the output (zeros are data) |
| MI-4 node-type equivariance | Permuting the order of nodes inside one node type permutes the per-node outputs and leaves `logits` unchanged (the model has no node-ID memory) |
| MI-5 padding | Adding padded nodes (`node_valid` False) leaves `logits` and scores unchanged |
| MI-6 mask stability (data) | On the real cache, no value outside `applicable_entry` is non-zero in any period (currently 0 violations); a violation fails the run |
| MI-7 availability handling | If a whole modality is marked unavailable for a node, fusion weights for that modality are exactly 0 and the output still has finite values |

---

## 4. Architecture overview (PyTorch, v0)

All numbers in this section are **[D]** initial engineering defaults: chosen to be small enough for a laptop CPU/MPS (the `.venv` has torch 2.14.0 with MPS, no CUDA) and large enough to be a real Transformer, not tuned for performance.

### 4.1 Pipeline in one picture

```
(B,6,46,130) values + masks
   -> D1 adapter: per node type, per modality linear layers  -> tokens (B,6,46,3,64)
   -> missing-aware modality fusion (gate x availability)    -> (B,6,46,64)
   -> + node-type embedding + learned time-position embedding
   -> 2 x [ temporal attention over 6 minutes (per node)
            -> cross-node attention over 46 nodes (per minute)
            -> feed-forward ]                                 -> H (B,6,46,64)
   -> fault head:  last minute, mean & max over valid nodes -> small MLP -> 5 logits
   -> next-step head: last minute per node -> output adapter -> (B,46,130)
```

### 4.2 Components

| Component | Specification [D] | Why |
|---|---|---|
| Hidden size d | 64 | small, because only 602 fault windows (101 cases) are available for the fault head under the default policy |
| History | T = 6 minutes input, 1 minute target | matches the existing window policy (`make_windows(window=6, horizon=1, step=60)`) |
| Encoder layers | 2, shared across both heads | one encoder serves detection and classification |
| Attention heads | 4 (head size 16) | standard divisor of 64 |
| Attention pattern | temporal (per node) then cross-node (per minute) inside each layer | factorised attention costs 6 x 46 tokens instead of 276 x 276 pairs; cross-node attention lets a node's anomaly be read in the context of the others |
| Normalisation, FFN | pre-LayerNorm, FFN width 256, GELU, dropout 0.1 | common stable defaults |
| Node identity | node-**type** embedding (2 types) only; **no per-node-ID embedding** | a D1 node ID means nothing on another system, and the core must not hard-code 46 |
| Time position | learned embedding of size 6 | six fixed steps |
| Modality fusion | per node and minute: weights `w_m = sigmoid(g(h_m)) * avail_m`; fused = `sum(w_m * h_m) / (sum(w_m) + eps)`; if no modality is available, a learned "missing" vector | an unavailable modality gets exactly zero weight (test MI-7) |
| Fault head | mean and max over valid nodes at the last minute -> concat (128) -> Linear(128,64) -> GELU -> Linear(64,5) | simple; does not need node identity |
| Next-step head | per node, last minute -> output adapter | self-supervised target and basis of anomaly score |
| Parameter count | expected order of 10^5 (about 0.2 million) [A: not measured; measure at implementation] | |

**Reusable core.** The core (fusion, embeddings, attention, heads' shared parts) takes `tokens`, `token_avail`, `node_type`, `node_valid` and configuration (`d`, layers, heads, `n_node_types`, number of modalities, number of classes). It contains no 46, no 130, no D1 channel name. Everything dataset-specific lives in an adapter class with a fixed interface (consumes the dataset window, produces tokens, availability and the output scatter). **Online Boutique adapter** is section 8.

**Known limitation of the design.** Without node IDs the model cannot tell `adservice-0` from `adservice-1` except through their data, and it is blind to the call graph. This costs possible accuracy but keeps transfer possible. Call-graph and ID features are listed as unresolved decisions (section 10).

### 4.3 Losses

- **Next-step loss**: mean squared error between prediction and `target`, **only over `effective` entries**, averaged over those entries.
- **Fault loss**: cross-entropy on 5 classes with class weights proportional to the inverse of the number of **fitting-region cases** per class (33/29/9/13/4 for the five classes under the approved split of section 5.6 [V]), because windows per case are nearly equal (5 or more per case).

### 4.4 Training-time robustness (optional switch)

**Simulated missing** [D]: during training, drop each applicable (node, modality) block of *input* values with probability 0.1 (set layer-3 False). Off for the first v0 run; one ablation on validation only. Rationale: D1 has no real run-time missingness, so this is the only way to test the fusion's missing-awareness before the testbed exists.

### 4.5 Anomaly score and normalisation

For a window with target minute t (the minute being predicted), with N_v valid nodes (46 for D1):

1. **Node error** `e_n` = mean over the effective entries of node n of (prediction - target)^2.
2. **Calibration statistics** (computed once, from the **calibration-reference windows** of section 5.6, never from fitting windows, validation or test): for every node, `median_n` and `IQR_n` = 75th minus 25th percentile of `e_n` over those windows (linear interpolation).
3. **Scale-aware floor**: `epsilon = max(1e-6, 0.01 * median(IQR_n over nodes))`. A node is **floor-affected** if `IQR_n < epsilon`. The calibration report lists every floor-affected node with its IQR, so a node whose error is almost constant does not turn a tiny change into a huge z-score unnoticed.
4. **Normalised node error** `z_n = (e_n - median_n) / max(IQR_n, epsilon)`.
5. **Three window scores**, all reported with their own thresholds:

| Method | Definition | Role |
|---|---|---|
| M1 raw max | `max_n e_n` | no normalisation; shows whether normalisation helps |
| **M2 max normalised** | `max_n z_n` | **primary [D]**: each documented injection hits one node (`cmdb_id` in `case_split_assignments.csv`), so the signal is local |
| M3 top-3 mean | mean of the `min(3, N_v)` largest `z_n` | less sensitive to a single noisy node |

The alert time of a window is the timestamp of its target minute (the seventh sample).

**Threshold (v0, declared before any training):** for each method, `tau = 99th percentile (linear interpolation) of that method's score over the calibration-reference windows`. Every method reports its own numerical `tau` (no shared number across methods), the number of calibration windows, the ranks used, and the floor report. **No threshold exists yet; none is claimed in this note.** Sensitivity at the 95th, 97th, 99th and 99.5th percentiles is planned on validation only (section 7.3). With about 329 calibration windows (section 5.6) the 99th percentile rests on roughly the top three windows and the 99.5th on the top one or two; consecutive windows share 6 of 7 minutes, so the effective sample is smaller still. These thresholds are therefore noisy and are reported with a block-bootstrap interval (block length 7 windows) as a diagnostic.

**Checkpoint rule.** Detection uses the **S1 pretrained checkpoint** (section 6). The calibration statistics and thresholds are valid only for the exact weights that produced the errors; any later weight change (for example fine-tuning for classification) means the calibration must be recomputed, and the fine-tuned weights are not used for detection unless that is done and documented.

---

## 5. Window and weak-label policy

### 5.1 Facts [V]

- 7,200 one-minute samples; windows of 6 + 1 samples; kept windows: train 3,427, validation 869, test 2,868; **0 dropped** by `protect_boundaries` (`window_counts.csv`, `split_counts_by_unit.csv`). Windows within 719 s of a boundary: 12 per period, all lying wholly inside their own period.
- A documented case window is `[t_case - 59, t_case + 300)` (ARMOR's convention, as used in `d1_validation_report.md`). Case times are not minute-aligned (60 different second offsets); samples are. A case window holds **5 or 6 samples**.
- No incident end times exist anywhere in D1. Nothing below invents one.
- All 210 cases have consecutive samples at minutes -9 ... +4 around injection (`eda/incident_aligned_coverage.json`).
- All 21 AD-only events (official anomaly labels without a matching case) lie in the **test** period, so they cannot affect any train or validation window (0 reference windows within 300 s of one).

### 5.2 Evidence about the minutes around injection [V]

Mean absolute scaled value (train-period scaler) over the entries active in the fit period, per minute offset from injection, divided by the mean of samples more than 15 minutes from every case. Train cases (101) and validation cases (25) separately:

| Offset (min) | -10..-1 | 0 | +1 | +2 | +3 | +4 | +5 | +6 | +7 | +8 | +9..+12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 0.92-0.96 | 1.15 | 1.21 | 1.18 | 1.16 | 1.11 | 1.07 | 1.05 | 1.02 | 0.99 | 0.96 |
| validation | 0.94-0.97 | 1.22 | 1.31 | 1.28 | 1.25 | 1.10 | 1.04 | 0.98 | 0.99 | 0.99 | 0.94-0.97 |

Reading it in plain English: before injection the data sits slightly *below* the reference average with no rise at minutes -2 and -1; after injection it rises for roughly five to seven minutes and then returns to the reference level. This is a crude average over everything, not proof for any single case, and it does not show that injection minutes are anomalous in every channel. It does support two guard-band choices: a margin after the documented window of at least the window length (300 s), and a pre-injection margin that is a precaution, not a necessity.

### 5.3 Definitions

Let `W_i = [t_i - 59, t_i + 300)` for case i, and let a window have input minutes `t0..t5` and target minute `t6`.

- **`event_associated`**: the last input minute `t5` lies in some `W_i`. Meaning: the model sees telemetry up to a minute inside a documented injection. It is **not** confirmed minute-level anomaly ground truth. Windows with `t5` inside two case windows of different classes (1 such train window under the default) are set to `ignore` for the fault objective.
- **`normal_reference`**: all seven minutes lie outside every guard region `[t_i - 59 - pre, t_i + 300 + post)`, and the window is not `event_associated`. Meaning: eligible reference telemetry, **not guaranteed normal** (see section 9).
- **`ignore`**: everything else. Excluded from objectives that assume reliable labels.

**Plain-English meaning (fixed wording).** `event_associated` = linked to a documented fault window, not verified anomalous at every minute. `normal_reference` = eligible for learning normal patterns, not guaranteed healthy. `ignore` windows are **not** confirmed negatives, and no anomaly-duration label is created anywhere in this design.

Windows whose target (but not `t5`) is inside `W_i` are `ignore` unless already caught by the normal-reference guard, which also removes them from the reference set.

### 5.4 Policies compared (training and validation only)

Counts of 7-step windows. Train has 3,427 windows from 101 cases, validation has 869 from 25 cases. "EA" = `event_associated`, "NR" = `normal_reference`, "Ign" = `ignore`. Case coverage = cases with at least one EA window.

| Policy | pre guard | post guard | Extra rule | Train EA / NR / Ign | Val EA / NR / Ign |
|---|---|---|---|---|---|
| A: documented window only | 0 s | 0 s | | 602 / 2,243 / 582 | 150 / 570 / 149 |
| B0: post margin only | 0 s | 300 s | | 602 / 1,796 / 1,029 | 150 / 453 / 266 |
| **B: default (excludes final 2 minutes before injection)** | **120 s** | **300 s** | | **602 / 1,641 / 1,184** | **150 / 411 / 308** |
| C: wide margins | 300 s | 600 s | 600 = ARMOR AD merge delay | 602 / 1,147 / 1,678 | 150 / 281 / 438 |
| D: known-normal collection only | 0 s | 0 s | NR windows only from ARMOR's normal sample file (`is_test_file` False) | 602 / 1,329 / 1,496 | 150 / 276 / 443 |
| E: strict EA | 120 s | 300 s | EA needs `t5 >= t_case` (post-onset minutes only) | 505 / 1,641 / 1,281 | 125 / 411 / 333 |

**Case coverage.** All six policies cover **101 of 101** train cases and **25 of 25** validation cases. Minimum EA windows per case: 5 (train) and 6 (validation) for A-D, 5 and 5 for E. EA windows by class under B, train: 219 / 192 / 60 / 108 / 24 (Container Hardware, Container Network, Node CPU, Node Disk, Node Memory); validation: 60 / 36 / 18 / 30 / 6.

**Overlaps.** Seven consecutive-case pairs are closer than 600 s (cases 31-32, 39-40, 78-79, 99-100 in train; 139-140, 142-143, 144-145 in test). Inside train exactly **one** window is in two case windows and the two cases have different classes. Policy E has none.

**Reading the table.** Policy B removes 155 train and 42 validation windows from B0 (the final-two-minute exclusion costs about 9% of B0's train reference windows). Policy D shows that the strict known-normal collection (1,329 / 276) is about 81% / 67% of B's size, so the choice matters for the size of the reference set, not for event coverage.

### 5.5 Frozen choice and sensitivity analysis

- **Default (frozen for v0): policy B.** It uses the documented ARMOR window for event association, excludes the last two minutes before injection and five minutes after the window from the reference set, and keeps every case covered. Choice made from this table, from section 5.2 and from the structural facts only; **no model has been trained.**
- **Sensitivity runs (validation only, report all)**: A (naive), B0 (isolates the effect of excluding the final two minutes before injection: B versus B0 differ only in that), C (wide), D (known-normal only), E (strict event association). If conclusions change between policies, report that as a finding; do not select the policy that scores best on validation without writing the amendment (section 10).
- **Test partition**: used only for fixed integrity checks. Under B the test period has 2,868 windows, 501 EA windows and **84 of 84** cases covered. These numbers were not used to choose anything.

### 5.6 Calibration split inside the training period (approved for v0, feasible)

**Why it exists.** A detection threshold must come from telemetry that the model never learned from. The first version of this note took the threshold from the same windows the model was trained on, which is optimistic (a model scores data it has memorised as "very normal"). This section carves a held-out **calibration-reference** block out of the approved train period. The approved outer split 101 / 25 / 84 and its manifest are **not changed**; validation and test play no part in designing this block.

**Status: APPROVED for Model 1 v0** (project owner, 2026-10-27, after review): 88 fitting cases / 13 calibration cases / 25 validation cases / 84 test cases, cut point `Tc = 1651637521`, `EMBARGO = 360 s`. The approval covers this split only. It does not approve any threshold value, any model result, or the open evaluation definitions of section 7.5, and every limitation in section 9 still applies. Changing `Tc` or the embargo later requires a written amendment (section 7.4 item 5).

**Rules (all enforced by a check, none by convention):**

1. Region by time. One cutoff `Tc` (calibration start) inside the approved train period. A window is a **fitting window** if its target minute `t6 < Tc - EMBARGO`, and a **calibration window** if its first input minute `t0 >= Tc`. Anything else is in the embargo and unused. `EMBARGO = 360 s` (one window of history).
2. **No shared raw timestamp** between any fitting window and any calibration window (inputs or target). Checked as an empty set intersection.
3. **Incident integrity.** `Tc` is chosen only between two cases of *different* `group_id`; no case has event-associated windows on both sides; every `group_id` is on one side. Cases whose windows fall in the calibration region are **not used for weight updates, including classification fine-tuning**; they may be used for diagnostics only.
4. **Policy B categories** (section 5.3) apply unchanged. Fitting windows supply S1 (normal reference) and S2 (event-associated). **Calibration-reference windows are `normal_reference` windows of the calibration region only.** `ignore` windows are in neither role.
5. Calibration windows never update weights, scalers or masks. The only things computed from them are the statistics of section 4.5.
6. The cutoff is chosen only from the train period: among the admissible case gaps, the one whose fitting share of Policy B `normal_reference` windows is closest to 80%. Nothing from validation or test is read.

**Result [V; appendix A.3].** 85 admissible cutoffs, all feasible. Chosen `Tc = 1651637521` (2022-05-04T04:12:01Z), a midpoint of a 90,256 s gap between two cases. Because that gap contains a day-long break in the data, the embargo happens to contain **0 windows** here; the rule is kept for any other cutoff.

| Region | Windows | `event_associated` | `normal_reference` | `ignore` | Cases | Cases by class (Container Hardware / Container Network / Node CPU / Node Disk / Node Memory) |
|---|---|---|---|---|---|---|
| Fitting | 2,868 | 524 | 1,312 | 1,032 | 88 | 33 / 29 / 9 / 13 / 4 |
| Embargo | 0 | 0 | 0 | 0 | | |
| Calibration | 559 | 78 | 329 | 152 | 13 (diagnostics only) | 4 / 3 / 1 / 5 / 0 |
| Train period total | 3,427 | 602 | 1,641 | 1,184 | 101 | 37 / 32 / 10 / 18 / 4 |

- **Fitting share of `normal_reference` windows: 79.95% (1,312), calibration share 20.05% (329).** Strict 80/20 is feasible. By raw minutes the split is 2,880 fitting samples (exactly two collection days) and 565 calibration-region samples (one contiguous run, 16.4%), so "80/20" refers to the `normal_reference` windows that are actually fitted and calibrated.
- Case coverage: all 88 fitting cases have 5 or 6 event-associated windows; **all four Node Memory train cases stay in the fitting region**. The 13 calibration-region cases (no Node Memory) are lost to fine-tuning, which leaves 524 event-associated windows (from 602) for S2.
- Leakage checks on the chosen cutoff: 0 shared timestamps, 0 straddling cases, 0 split groups.
- The calibration region and the approved validation period are adjacent in time on the same collection day; no window crosses the train|validation boundary (`protect_boundaries` dropped 0).
- Next-best cutoffs give 78.7% / 21.3% (14 calibration cases) and 78.1% / 21.9% (15 cases), so the choice is a reasonable member of a family, not a knife-edge.

**What 329 windows can and cannot support.** About 47 non-overlapping windows' worth of independent information. The 99th percentile is determined by roughly the top three windows; see section 4.5. If a larger calibration block is wanted, the only adjustment is a different admissible `Tc` (for example the 78.1% / 21.9% cutoff, 360 windows); no leakage rule is loosened.

**Fit-only preprocessing is required.** The existing `scaler_model1_train_period_v1.npz` was fitted on all 3,445 train-period samples, which includes the 565 calibration-region samples. A calibration holdout that contributes to the scaler is no longer strictly held out. Measured against a scaler refit on the 2,880 fitting samples only [V; appendix A.3]: layer-1 `absent_block` is identical; `constant` entries 4,206 versus 4,209; for 1,541 non-constant entries the mean moves by a median 0.0033 scale units (99th percentile 0.0195, max 0.0315) and the scale changes by a median 0.6% (99th percentile 14%, max 25%). The effect is small, but it is cheap to remove and it makes the claim "calibration observations never touched any fitted quantity" exactly true, so a new fit-only scaler artifact is **required** (exact steps in section 12). The existing artifacts stay untouched.

**Two different kinds of scaling, kept apart.**
- *ARMOR's upstream pre-scaling*: the released `D1` features were already scaled over the **full 5-day timeline** by a recipe nobody documented. Neither the old nor the new scaler can undo it, so some information from validation, test and calibration periods is already present in every value (risk R6). It is disclosed in every report.
- *Our own scaler*: a per-node, per-channel mean and scale that **we** fit. It is the only part we control, and the fit-only version limits it to the fitting region.

---

## 6. Training stages

All stages use fixed seeds (the data pipeline is deterministic; ARMOR's own training is unseeded and is not our reference for reproducibility). Each run records: git commit, the SHA-256 of the split manifest (`content_sha256`, currently `32ff03fcffca09f3d3953777a020514b4af9c882bc85e3dc1693e282607638ac`), the preprocessing config hash (`e9aa3ccc776a8bc9940d151ba3feea3c79f502eff58e12c9946babe0aaefeb8c`), the calibration-split definition (`Tc`, embargo, hash of the window lists), the fit-only scaler hash, policy name, seeds and device.

| Stage | Data | Objective | Notes |
|---|---|---|---|
| S0 data checks | all | none | MI-6, the calibration-split checks of section 5.6, boundary and leakage tests from `armor-system/tests/prep/` plus new window-category tests |
| S1 self-supervised pretraining | **fitting-region `normal_reference` windows only** (1,312) | next-step masked MSE | Never sees `event_associated` or `ignore` windows. Early stopping on validation `normal_reference` masked MSE (documented use of validation) |
| S2 fault fine-tuning | fitting-region `event_associated` windows (524 from 88 cases under B) | 5-class cross-entropy, class-weighted | Encoder initialised from S1; all weights trainable with a smaller learning rate than S1 [D: frozen versus fine-tuned is an ablation] |
| S3 calibration | **calibration-reference windows** (329, section 5.6) | none | No gradient step: median, IQR, epsilon, floor report and thresholds of section 4.5, using the exact S1 checkpoint kept for detection |
| S4 evaluation | validation (all decisions), test (once per frozen configuration) | metrics | Section 7 |

**Consistency with the earlier pretraining decision.** `d1_validation_report.md` section 9.1 item 6 fixes "strict chronological pretraining on samples at or before the AD split (2,269)" and keeps ARMOR's whole-pool (3,714) as a separate sensitivity run. That sentence was written before the 101/25/84 split was approved. Of the 2,269 normal-file samples, 1,847 lie in the Model 1 train period and 421 in the validation period [V]. S1 uses only windows from the **fitting region** (a subset of the train period, hence of the 2,269); validation `normal_reference` windows are used only for early stopping, and calibration windows never update weights (not even through early stopping). All 1,445 normal-file samples after the AD split, including the 161 uncertain ones near AD-only events, are excluded from S1 (they lie in the test period). The whole-pool run stays an optional sensitivity experiment, unchanged.

Initial optimisation defaults [D]: AdamW, batch 64, S1 learning rate 1e-3 with cosine decay, S2 learning rate 1e-4, weight decay 0.01, gradient clipping 1.0, early stopping patience 10 epochs on the S1/S2 validation metric. No data augmentation in v0 (see `armor-system/docs/d1_augmentation_assessment.md`).

---

## 7. Evaluation and comparison protocol

**Two protocols are kept apart.**
Protocol A is ARMOR's official one: split at 60% of samples (`2022-05-06T16:00:00Z`, 1651852800) for anomaly detection, first 126 / last 84 cases for fault typing. Its split manifest is `splits/official_armor_v1.json`; nothing here changes it, its scoring code or `ad_cases.pkl`.
Protocol B is Model 1's: train / validation / test = 101 / 25 / 84 cases from `splits/model1_split_101_25_84_v1.json`.
The validation|test boundary of Protocol B equals ARMOR's AD split, so the 84 Model 1 test cases are the same 84 as ARMOR's test cases [V: `split_counts_by_unit.csv`]. Model 1 trains on 101 cases, ARMOR's fault classifier on 126, so a head-to-head number must say so.

### 7.1 Fault classification (five classes)

- **Primary metric: macro-F1** over the five classes.
- **Secondary:** weighted-F1 (what ARMOR reports, for comparability), per-class recall, confusion matrix, support per class.
- **Case-level aggregation (frozen [D])**: for each case, take every `event_associated` window of that case (5 or 6 under policy B), average the softmax probabilities, take the argmax. Metrics are computed on cases, not windows (84 test cases, 25 validation cases).
- This is **incident-conditioned** classification: the injection time is given. An alert-triggered variant is a later extension [U].
- Report 95% bootstrap intervals over cases. Node Memory has 1 validation case and 4 test cases, so a validation macro-F1 can swing by a whole class; per-class support is always printed next to every number and validation macro-F1 differences are not interpreted as significant.

### 7.2 Controls for the classification result

- **Majority / prior baseline** (floor, no features).
- **XGBoost control (to implement later; `xgboost` is not installed in `.venv`)**: same windows, labels, case-level aggregation, class weights and split, features = for each applicable entry of the 6 input minutes the mean, last value, standard deviation and last-minus-first difference (about 12.6k features = 3,152 x 4 before any reduction). Provides a strong non-neural reference; if it matches the Transformer, the Transformer's added value on D1 is not demonstrated. Dependency to be added in a separate approved change.
- **Official ARMOR numbers** (authors' released outputs, **not** our reproduction): FT weighted-F1 0.938, precision 0.9464, recall 0.9405; AD P 0.9245, R 1.0, F1 0.9608 (`d1_validation_report.md`, `team_handoff_readiness_audit.md`). Our baseline teammates' reproduction, when it exists, replaces these as the comparison.

### 7.3 Detection from prediction error

1. Score every window in a period with section 4.5, alert time = target minute. Three methods (M1 raw max, M2 max normalised, M3 top-3 mean) are always computed; **M2 is primary**.
2. **Threshold (v0, declared before any training):** the 99th percentile of each method's score over the 329 calibration-reference windows. The calibration report prints, per method, the numerical `tau`, the calibration window count, the ranks used, the per-node median and IQR, `epsilon`, and the list of floor-affected nodes. No value is claimed here because no model exists.
3. **Validation-only sensitivity:** the same procedure at the 95th, 97th, 99th and 99.5th percentiles, evaluated on validation windows and the 25 validation cases (false-alarm rate on validation `normal_reference` windows, event hit rate, alert coverage). Validation never sets a threshold in v0. Changing the primary percentile after seeing validation results is an amendment, not a tuning step.
4. **Alert intervals and matching** follow ARMOR's `eval_AD` as documented in `d1_validation_report.md`: alert timestamps closer than `delay = 600` s are merged into intervals, zero-length intervals are dropped, a label is hit when an alert interval overlaps `[t - 300, t + 300]`; TP and FN count labels, FP counts alert intervals.
5. **Two populations, never averaged or merged:**
   - the **98** official post-boundary AD events (`ad_cases.pkl`, includes the 21 uncertain AD-only events);
   - the **84** post-boundary documented injection cases.
6. **Diagnostics always printed:** number of alert intervals, false-alarm intervals, total alert time, share of post-boundary minutes inside padded alert intervals (alert coverage), false-alarm intervals per day on windows far from all cases, detection delay (alert start minus injection time, for hits only).
7. **Validation implementation check:** the scoring function must first reproduce the published ARMOR AD result (98 events, 8 false-alarm intervals, P 0.9245, R 1.0, F1 0.9608; `d1_validation_report.md`) from ARMOR's released alert intervals, before it is applied to Model 1 output. Those interval files are **not committed** (`d1_validation_report.md` section 9.1 notes they were fetched during read-only research), so the validation step needs them fetched again under an approved decision [U].

### 7.4 No-test-tuning rules

1. Hyper-parameters, thresholds, policies, features, early stopping and model selection use **train and validation only**; thresholds use calibration-reference windows only.
2. The test partition is evaluated **once per frozen configuration**; each evaluation is logged (commit, config hash, date). A configuration that changes after seeing test results is a new configuration and its earlier test result stays in the log.
3. Test data never enters any fitted statistic: scaler, mask, per-node normalisation, threshold or class weights. Calibration-reference windows never enter a weight update, a scaler, a mask, early stopping or class weights; validation windows never enter a threshold.
4. Test-period integrity checks (split, boundary and mask tests) are allowed because they cannot change a decision.
5. Any change to a frozen decision in section 10 requires a written amendment to this note (what changed, why, and which data it was based on).

### 7.5 Unresolved evaluation definitions [U]

| ID | Open point | Why it cannot be settled now |
|---|---|---|
| E-1 | What does "detection" mean for the 84 injection cases when incident end times do not exist? | **Open.** Only the ARMOR hit rule (`+-300 s`) is available; it is a weak definition and must be reported as such. No duration label is invented |
| E-2 | How a window score becomes an alert interval (single-minute alerts are dropped by the zero-length rule) | **Open.** ARMOR's source code is not in the local raw folder (only `D1/`), so the rule could not be read this time; section 7.3 item 4 rests on `d1_validation_report.md`. Must be checked against the released code before M1-05 |
| E-3 | Final test report with a refit on 126 cases (train + validation) for an ARMOR-like training size | **Open, now more costly.** A refit changes the weights, so the calibration (section 4.5) must be redone with a new calibration block that the refit has not seen; it cannot reuse the validation set |
| E-4 | Model selection rule given 25 validation cases (Node Memory n = 1) | **Open.** Needs a minimum-difference convention that has not been agreed |
| E-5 | Does per-node normalisation help? | **Resolved as a planned comparison:** M1 (raw) versus M2 and M3 (normalised) are all reported with their own thresholds; M2 stays primary unless an amendment says otherwise |
| E-6 | Alert-triggered (not incident-conditioned) fault classification | **Open.** Needs a defined alert-to-window rule and a live data path |
| E-7 | Approval of the calibration split (section 5.6) and of its cutoff | **Resolved:** approved for v0 on 2026-10-27; amendments still require a written note |

---

## 8. Online Boutique adaptation strategy

Status: **the Online Boutique testbed does not exist yet** [V: nothing in the repository trains or collects from it]. This section is a plan.

1. Keep the **core** unchanged. Write a new dataset adapter that reads the testbed's schema (node list, node types, per-type channel lists and modalities) from a JSON file in the same format as `schema.json`, plus a train-period scaler and `applicable_entry` fit on testbed training telemetry only.
2. Node types are re-derived from the testbed (for example pod and node). Embeddings for new types are learned; **pretrained D1 weights are not assumed to transfer**. Adapters and embeddings are re-initialised; the encoder may be initialised from D1 as one experiment, compared with training from scratch.
3. Real missingness (a container without logs, a delayed scrape) is expressed through `observed_entry`; the model never needed D1 to contain it.
4. Fault classes are mapped explicitly; classes not present stay out of the head, new ones are added with a new head.
5. Live path: the Airflow pipeline (Bhoomika) must deliver the **window sample contract of section 2.1** (6 consecutive one-minute samples with schema and masks). Agreeing this contract early is the main dependency between the workstreams.
6. Same evaluation code and the same strict chronological, incident-group-aware split rules apply; the testbed needs its own approved split before any training.

---

## 9. Risks, assumptions and limitations

| # | Risk or assumption | Effect | Handling |
|---|---|---|---|
| R1 | Very small supervised set: 101 train cases, 602 EA windows from those 101 cases (windows of one case are strongly correlated) | High variance, overfitting | Small model, case-level metrics, bootstrap, XGBoost control |
| R2 | Node Memory: 4 / 1 / 4 cases (train / validation / test) | Validation macro-F1 unstable; test per-class recall coarse | Report supports, no significance claims on validation |
| R3 | `normal_reference` is not proof of normal operation. Fault end times are unknown; faults may persist after `t + 300` | Reference windows may contain faint abnormal behaviour | Post margin; policies A-E sensitivity; section 5.2 evidence (aggregate only) |
| R4 | Event association is not minute-level ground truth. Some minutes inside `W_i` may be normal | Label noise in the fault objective | Case-level aggregation; policy E sensitivity |
| R5 | Root node shows higher metric activity than other nodes *before* injection in an earlier analysis (cause unknown) | Possible hidden pre-injection signal [A] | Not resolved here; section 5.2 shows no aggregate rise before injection, but per-node effects are not ruled out |
| R6 | ARMOR's features were scaled over the full timeline by an unknown recipe, before our train-only scaler | Residual scaling information from validation/test time in the inputs [A] | Cannot be undone; disclosed in every report |
| R7 | No node IDs and no call graph in v0 | Possible accuracy loss; cannot localise the faulty service | Deliberate (transfer); later ablation |
| R8 | Calibration block is small: 329 windows from one contiguous run (about 47 independent windows) | Thresholds at the 99th and 99.5th percentile are noisy | Block-bootstrap diagnostic; sensitivity at four percentiles; a larger admissible cutoff exists |
| R13 | 13 train-period cases (no Node Memory) are lost to fine-tuning by the calibration split | 524 instead of 602 event-associated windows for S2 | Documented; they remain usable for diagnostics |
| R14 | A data-derived mask can hide a real rare channel (`system.disk.readonly`) | Silent loss of fault signal | Masks are declared by node type and channel source; MI-6 fails a run on any violation |
| R9 | The 98 official AD events include 21 uncertain AD-only events | Detection metrics on that population are partly uncertain | Always reported separately from the 84 cases |
| R10 | D1 is one benchmark, one run structure (5 collection days) | Weak evidence of generalisation | Online Boutique testbed |
| R11 | Official ARMOR numbers are the authors' outputs, not reproduced here | Comparison is provisional | Replace with our baseline team's reproduction |
| R12 | The numbers in this note were measured on one machine (macOS, Python 3.14) | | Appendix A reruns them |

---

## 10. Frozen versus unresolved decisions

### Frozen for v0 (changes only by written amendment to this note)

| ID | Decision |
|---|---|
| F1 | Approved split 101 / 25 / 84 unchanged. Scaler and layer-1 mask fit on the **fitting region** only (section 5.6); the existing scaler artifacts are kept as they are and the new fit-only scaler is a new file |
| F2 | Window = 6 input minutes + 1 target minute, 60 s spacing, never across a gap or a split boundary |
| F3 | Three training categories (`normal_reference`, `event_associated`, `ignore`), separate from the five fault classes |
| F4 | Default window policy B (pre 120 s, post 300 s), sensitivity A, B0, C, D, E reported on validation |
| F5 | Three mask layers; layer 2 is **declared** by node type and channel source (3,152 applicable entries), not learned from zeros; per-node-only masks are forbidden |
| F6 | Class order 0-4 from `splits.CLASSES`, mapping 0->0, 1->1, 3->2, 4->3, 5->4 |
| F7 | Architecture of section 4 (d 64, 2 layers, 4 heads, node-type embedding, no node-ID embedding, factorised attention, shared encoder, two heads) as an *initial* default |
| F8 | Primary fault metric macro-F1; case-level mean-probability aggregation |
| F9 | Detection: next-step error; median / IQR normalisation with `epsilon = max(1e-6, 0.01 * median(IQR))`; methods M1 raw max, **M2 max normalised (primary)**, M3 top-3 mean; threshold = 99th percentile on calibration-reference windows (v0); validation-only sensitivity at 95 / 97 / 99 / 99.5; S1 checkpoint for detection; ARMOR matching rule; 98 and 84 populations separate |
| F10 | Test used once per frozen configuration; official ARMOR split and scoring never changed |
| F11 | Calibration split of section 5.6 (**approved for v0**, 2026-10-27): rules 1-6, `EMBARGO = 360 s`, cutoff chosen from the train period only; calibration windows never touch weights, scalers or masks |

### Unresolved

| ID | Decision | Owner suggestion |
|---|---|---|
| U1 | E-1 to E-4 and E-6 in section 7.5 (E-5 and E-7 resolved) | Khushi + Bhoomika, then team |
| U2 | Call graph or node-ID features (v0 uses neither) | after v0 results |
| U3 | Longer history than 6 minutes | validation ablation |
| U4 | Whether baseline teammates fetch ARMOR's released checkpoint, and whether the released alert-interval files (needed by section 7.3 item 7) are re-fetched | team lead / user |
| U5 | GPU resources, cloud provider, storage, credentials, budget | team |
| U6 | `xgboost` (and `torch`) pins in a requirements file | separate approved change |
| U7 | A Linear issue for Model 1 work, then branch names from it | user |
| U8 | CI and `.gitignore` guards (no CI exists; `*.pkl` is not ignored outside `armor-system/data/raw/`) | team |
| U9 | Hyper-parameter search space and selection rule given 25 validation cases | after first baseline numbers |

---

## 11. Contribution to the course deliverables

[A: the course dates below are from the project notes and should be re-checked against the syllabus.]

- **Workbook 2 (11/11).** This note is the design evidence: a frozen, traceable specification, computed data findings (sections 3 and 5) and a leakage-aware protocol. It contains no performance numbers because none exist; a plausible Workbook claim is the *method and the data audit*, not results.
- **Team Meeting 2 (11/12) / November checkpoint.** The checkpoint asks for a team-trained model, a reproduced baseline and live input. This note defines the input contract (section 2.1) the Airflow work must meet and the evaluation harness the baseline team's numbers will be compared in. It does not by itself deliver a trained model.
- **Final project.** Sections 7 and 8 fix how results will be reported (two protocols, two detection populations, no test tuning) and how the model moves to Online Boutique.

**Recommended next implementation task (not started):** the data layer: a PyTorch `Dataset` and `DataLoader` over `d1_features_v1.npz`, the manifest, the scaler and the new `applicable_entry` mask, with the window categories of section 5, window-category and mask tests (MI-6), and the ARMOR-event scoring function validated against the released result (section 7.3, item 7).

---

## 12. Implementation requirements for M1-03 (data layer)

M1-03 must not change the approved split, the existing scaler artifacts, `old/`, the ARMOR baseline, or Airflow. It writes **new** files only. Requirements:

1. **Calibration-split artifact.** A new manifest (for example `splits/model1_calibration_split_v1.json`) recording `Tc`, `EMBARGO`, Policy B parameters, window start indices per role (fitting / calibration), case ids per role, counts, a content hash, and `status: approved` (v0, 2026-10-27). Built by code (a function in `armor-system/data/prep/`, with unit tests), not by hand. The expected counts are those of section 5.6; any difference is a failure.
2. **Fit-only scaler artifact.** A new file (for example `preprocessing/scaler_model1_fit_only_v1.npz`) fit with the existing `fit_scaler` on exactly the fitting samples (`ts < Tc - EMBARGO`, 2,880 samples). Do **not** overwrite `scaler_model1_train_period_v1.npz`. All calibration, validation, test and live data use the fit-only scaler. Record its hash in the manifest. Expected: `absent_block` identical to the existing one.
3. **Declared mask.** A function that builds `applicable_entry` (3,152 entries) from `schema.json` channel names and node names, configured per dataset, plus `observed_entry` (all True for D1). Test MI-6 on every period; a violation raises.
4. **Dataset / DataLoader** over `d1_features_v1.npz` using `prepare_window`-equivalent logic with the fit-only scaler, yielding the fields of section 2.1 plus the window category, the region (fitting / calibration / validation / test) and the case index. It must refuse to yield windows that are not wholly inside their region.
5. **Leakage tests** (new, in `armor-system/tests/prep/`): no shared timestamp between fitting and calibration windows; no case or `group_id` on both sides; calibration windows excluded from any training iterator; scaler fit set equals the fitting samples; labels map 0->0, 1->1, 3->2, 4->3, 5->4; window counts equal section 5.4 and 5.6; no window crosses a region boundary; Policy B categories reproduced.
6. **Calibration utility** (no model needed to test it): median, IQR, `epsilon = max(1e-6, 0.01 * median(IQR))`, floor-affected node report, the three aggregation methods with `min(3, N_v)`, linear-interpolation percentiles, tested on synthetic errors with known answers (including a node with constant error, a single-node model, and fewer than three nodes).
7. **Reproducibility.** Deterministic output (two runs byte-identical), seeds recorded, no raw telemetry, checkpoints or secrets committed; large arrays go to git-ignored paths like the existing cache.
8. **Not in M1-03:** the PyTorch model, training, any threshold value, the XGBoost control, ARMOR-interval validation.

## Appendix A. Reproducing the evidence

Run from the repository root with the project environment (`.venv/bin/python`), after the D1 cache exists. Both scripts only read files; they write nothing. The expected output is the numbers quoted above.

**A.1 Mask facts (sections 3.1 and 3.2)**

```python
import json, numpy as np
R = "armor-system/data/artifacts/d1/"
sc = np.load(R + "preprocessing/scaler_model1_train_period_v1.npz")
schema = json.load(open(R + "inspection/schema.json"))
nodes, mr = schema["node_order"], schema["modality_index_ranges"]
ch = [c["name"] for c in schema["channel_order"]]
d = np.load("armor-system/data/processed/d1_features_v1.npz")
X, ts = d["X"], d["timestamps"]
B1, B2, TC, EMBARGO = 1651713858, 1651852800, 1651637521, 360   # TC: calibration start (section 5.6)
fit = ts < TC - EMBARGO
cal = (ts >= TC) & (ts < B1)
vm = np.array([n.startswith("node-") for n in nodes])
cm = np.zeros(130, int)
for j, m in enumerate(["metric", "trace", "log"]):
    cm[mr[m]] = j
print("absent_block per modality VM/pod:", sc["absent_block"][vm].sum(0), sc["absent_block"][~vm].sum(0))
print("absent_entry:", int(sc["absent_entry"].sum()), "constant:", int(sc["constant"].sum()))
# declared rule: VM = system.* metrics; pod = kpi_container_* metrics + trace + log
sysch = np.array([n.startswith("system.") for n in ch])
kpi = np.array([n.startswith("kpi_container") for n in ch])
declared = np.zeros((46, 130), bool)
declared[vm] = (cm == 0) & sysch
declared[~vm] = ((cm == 0) & kpi) | (cm == 1) | (cm == 2)
app = declared & ~sc["absent_entry"]
print("applicable:", int(app.sum()), "of", app.size, "| VM:", int(app[vm].sum()),
      "| pod metric/trace/log:", [int(app[~vm][:, cm == j].sum()) for j in range(3)])
for name, m in [("fit region", fit), ("calibration region", cal), ("validation", (ts >= B1) & (ts < B2)),
                ("test", ts >= B2)]:
    print(name, "non-zero outside declared mask:", int(((X[m] != 0).any(0) & ~app).sum()))
print("zero fraction among applicable entries (fit region): %.3f" % float((X[fit][:, app] == 0).mean()))
print("applicable but all-zero in fit region:", int((app & ~(X[fit] != 0).any(0)).sum()))
# why the mask is declared, not learned: a mask learned from the fit region alone misses a real channel
learned = np.zeros((46, 130), bool)
for sel in (vm, ~vm):
    learned[sel] = (X[fit][:, sel] != 0).any((0, 1))[None, :]
miss = (X[cal] != 0).any(0) & ~learned & ~sc["absent_entry"]
print("entries active in calibration region but hidden by a fit-region-learned mask:",
      [(nodes[i], ch[c], int((X[cal][:, i, c] != 0).sum())) for i, c in np.argwhere(miss)])
late = ~(X[ts < B1] != 0).any(0) & (X[ts >= B1] != 0).any(0) & ~sc["absent_entry"]
print("all-zero in train period but non-zero later:", int(late.sum()))
```

**A.2 Window categories (section 5.4)**

```python
import csv, numpy as np
R = "armor-system/data/artifacts/d1/"
d = np.load("armor-system/data/processed/d1_features_v1.npz")
ts, is_test = d["timestamps"], d["is_test_file"]
rows = list(csv.DictReader(open(R + "splits/case_split_assignments.csv")))
cts = np.array([int(r["timestamp"]) for r in rows])
cases_split = np.array([r["model1_split"] for r in rows])
B1, B2 = 1651713858, 1651852800
periods = {"train": (-10**18, B1), "validation": (B1, B2)}

def starts(lo, hi):
    pos = np.where((ts >= lo) & (ts < hi))[0]
    t = ts[pos]
    brk = np.r_[0, np.where(np.diff(t) != 60)[0] + 1, len(pos)]
    out = [pos[a:b - 6] for a, b in zip(brk[:-1], brk[1:]) if b - a >= 7]
    return np.concatenate(out)

def categories(pre, post, strict=False, normal_file_only=False):
    res = {}
    for name, (lo, hi) in periods.items():
        st = starts(lo, hi)
        T = ts[st[:, None] + np.arange(7)]
        t5 = T[:, 5]
        ea = np.zeros(len(st), bool)
        covered = set()
        for i, c in enumerate(cts):
            m = (t5 >= (c if strict else c - 59)) & (t5 < c + 300)
            ea |= m
            if m.any():
                covered.add(i)
        guard = np.zeros(len(st), bool)
        for c in cts:
            guard |= ((T >= c - 59 - pre) & (T < c + 300 + post)).any(1)
        nr = ~guard & ~ea
        if normal_file_only:
            nr &= ~is_test[st[:, None] + np.arange(7)].any(1)
        n_cases = int((cases_split == name).sum())
        cov = sum(1 for i in covered if cases_split[i] == name)
        res[name] = (len(st), int(ea.sum()), int(nr.sum()), int((~(ea | nr)).sum()), f"{cov}/{n_cases} cases")
    return res

policies = {"A": (0, 0, False, False), "B0": (0, 300, False, False), "B": (120, 300, False, False),
            "C": (300, 600, False, False), "D": (0, 0, False, True), "E": (120, 300, True, False)}
for k, v in policies.items():
    print(k, categories(*v))
```

Expected, for example: policy B prints `train (3427, 602, 1641, 1184, '101/101 cases')` and `validation (869, 150, 411, 308, '25/25 cases')`. (Strict-E "EA" requires `t5 >= t_case`; the `ignore` count in the table is `windows - EA - NR`.)

**A.3 Calibration split and fit-only scaler comparison (section 5.6)**

```python
import csv, json, sys, collections, numpy as np
sys.path.insert(0, "armor-system/data")
from prep import preprocess as pp
R = "armor-system/data/artifacts/d1/"
d = np.load("armor-system/data/processed/d1_features_v1.npz")
X, ts = d["X"], d["timestamps"]
rows = list(csv.DictReader(open(R + "splits/case_split_assignments.csv")))
cts = np.array([int(r["timestamp"]) for r in rows])
cls = np.array([r["category"] for r in rows])
cases_split = np.array([r["model1_split"] for r in rows])
gid = np.array([int(r["group_id"]) for r in rows])
B1 = 1651713858                       # train|validation boundary (approved manifest)
EMBARGO = 360                         # seconds: one window length (6 history steps)

pos = np.where(ts < B1)[0]
brk = np.r_[0, np.where(np.diff(ts[pos]) != 60)[0] + 1, len(pos)]
st = np.concatenate([pos[a:b - 6] for a, b in zip(brk[:-1], brk[1:]) if b - a >= 7])
T = ts[st[:, None] + np.arange(7)]
t0, t5, t6 = T[:, 0], T[:, 5], T[:, 6]
ea = np.zeros(len(st), bool)
case_of = np.full(len(st), -1)
for i, c in enumerate(cts):                               # Policy B: event_associated
    m = (t5 >= c - 59) & (t5 < c + 300)
    ea |= m
    case_of[m] = i
guard = np.zeros(len(st), bool)
for c in cts:                                             # Policy B: pre 120 s, post 300 s
    guard |= ((T >= c - 59 - 120) & (T < c + 300 + 300)).any(1)
nr = ~guard & ~ea
train_cases = np.where(cases_split == "train")[0]
order = train_cases[np.argsort(cts[train_cases])]

def evaluate(tc):
    fit, cal = t6 < tc - EMBARGO, t0 >= tc
    fit_cases, cal_cases = set(case_of[fit & ea]), set(case_of[cal & ea])
    side = {i: "fit" if i in fit_cases else "cal" if i in cal_cases else "none" for i in train_cases}
    group_violations = sum(len({side[i] for i in train_cases if gid[i] == g} - {"none"}) > 1 for g in set(gid[train_cases]))
    shared_ts = len(set(np.unique(T[fit])) & set(np.unique(T[cal])))
    return dict(tc=tc, fit=fit, cal=cal, fit_cases=fit_cases, cal_cases=cal_cases, straddle=len(fit_cases & cal_cases),
                groups=group_violations, shared_ts=shared_ts, frac=(fit & nr).sum() / ((fit | cal) & nr).sum())

cands = [int((cts[a] + cts[b]) // 2) for a, b in zip(order[:-1], order[1:]) if gid[a] != gid[b]]
ok = [r for r in map(evaluate, cands) if r["straddle"] == 0 and r["groups"] == 0 and r["shared_ts"] == 0]
best = min(ok, key=lambda r: abs(r["frac"] - 0.8))
fit, cal, tc = best["fit"], best["cal"], best["tc"]
print("feasible cutoffs:", len(ok), "of", len(cands), "| chosen Tc =", tc)
for name, m in (("fitting", fit), ("embargo", ~fit & ~cal), ("calibration", cal)):
    print(name, "windows", int(m.sum()), "event_associated", int((m & ea).sum()), "normal_reference", int((m & nr).sum()),
          "ignore", int((m & ~ea & ~nr).sum()))
print("normal_reference fit fraction: %.4f" % best["frac"], "| shared timestamps:", best["shared_ts"],
      "| straddling cases:", best["straddle"], "| group violations:", best["groups"])
for name, cs in (("fitting", best["fit_cases"]), ("calibration", best["cal_cases"])):
    per = collections.Counter(case_of[(fit if name == "fitting" else cal) & ea])
    print(name, "cases", len(cs), dict(collections.Counter(cls[sorted(cs)])), "min/max EA windows per case", min(per.values()), max(per.values()))
fit_samples = ts < tc - EMBARGO
calib_samples = (ts >= tc) & (ts < B1)
print("fitting samples", int(fit_samples.sum()), "calibration-region samples", int(calib_samples.sum()),
      "contiguous calibration runs", 1 + int((np.diff(ts[calib_samples]) != 60).sum()))
# fit-only scaler versus the existing scaler (written to nothing)
groups = json.load(open(R + "inspection/schema.json"))["modality_index_ranges"]
old = np.load(R + "preprocessing/scaler_model1_train_period_v1.npz")
new = pp.fit_scaler(X, fit_samples, groups)
both = ~old["constant"] & ~old["absent_entry"] & ~new["constant"] & ~new["absent_entry"]
dm = np.abs(new["mean"] - old["mean"])[both] / old["scale"][both]
dsc = np.abs(new["scale"] / old["scale"] - 1)[both]
print("absent_block identical:", bool((new["absent_block"] == old["absent_block"]).all()),
      "| constant entries old/new:", int(old["constant"].sum()), int(new["constant"].sum()))
print("|mean shift|/scale median %.4f p99 %.4f max %.4f | scale change median %.4f p99 %.4f max %.4f" %
      (np.median(dm), np.percentile(dm, 99), dm.max(), np.median(dsc), np.percentile(dsc, 99), dsc.max()))
```

Expected output: `feasible cutoffs: 85 of 85 | chosen Tc = 1651637521`; fitting `2868 / 524 / 1312 / 1032` (windows / event_associated / normal_reference / ignore); calibration `559 / 78 / 329 / 152`; embargo 0; fit fraction 0.7995; 0 shared timestamps, 0 straddling cases, 0 group violations; 88 fitting and 13 calibration cases. The script writes nothing.

## Appendix B. Evidence index

| Claim | Where |
|---|---|
| Split counts, class counts, boundaries, `content_sha256` | `armor-system/data/artifacts/d1/splits/model1_split_101_25_84_v1.json`, `case_split_assignments.csv` |
| Window counts per period | `preprocessing/window_counts.csv`, `preprocessing/split_counts_by_unit.csv` |
| Scaler and `absent_*` masks | `preprocessing/scaler_model1_train_period_v1.npz`, `structural_masks.csv`, `preprocess.py` (`fit_scaler`, `structural_absence`) |
| Channel and modality layout, class ids | `inspection/schema.json` |
| Incident-aligned coverage | `eda/incident_aligned_coverage.json` |
| ARMOR protocol, published AD/FT numbers, label policy | `armor-system/docs/d1_validation_report.md`, `armor-system/docs/team_handoff_readiness_audit.md` |
| Calibration split, fit-only scaler comparison | appendix A.3 (reads `case_split_assignments.csv`, `d1_features_v1.npz`, `scaler_model1_train_period_v1.npz`; writes nothing) |
| Declared mask and the `system.disk.readonly` finding | appendix A.1 |
| Existing tests | `armor-system/tests/data/` (37) and `armor-system/tests/prep/` (30): all passed on 2026-10-26 |
