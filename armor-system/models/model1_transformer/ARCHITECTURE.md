# Model 1 v0 — Transformer core implementation (Linear 298-50)

**Status: implemented and verified on CPU; no training, no validation or test evaluation.**
Authority: `MODEL1_V0_DESIGN.md` (approved, merged). This file records how that design was implemented and the extra v0 implementation decisions. Nothing in the approved design was changed.

## 1. Approved sizes (unchanged)

Width d = 64, 4 attention heads (head size 16), 2 Transformer layers shared by both heads, FFN width 256, GELU, dropout 0.1, 6 historical input minutes (the target minute is never an input), 5 fault classes, node-type embedding (pod = 0, VM = 1; no node-ID embedding), learned time embedding of size 6.

## 2. Implementation decisions recorded for v0

| # | Decision |
|---|---|
| 1 | **Input adapters concatenate masked values and the effective mask**, so each adapter's input width is twice its channel count: pod metric 106, pod trace 18, pod log 22, VM metric 114 (all to d = 64). The design text "53 → d" is read as 53 channels → d. |
| 2 | **Modality gate** `g = sigmoid(Linear(64, 1))`, one gate shared by the three modalities. |
| 3 | **Fusion** `w_m = g(h_m) · avail_m`, `fused = Σ w_m h_m / (Σ w_m + 1e-8)`; weights of unavailable tokens are exactly 0. |
| 4 | **Missing-modality fallback**: one learned shared vector (init N(0, 0.02²)), used when no modality is available for a node and minute. |
| 5 | **Temporal attention is bidirectional** over the six historical input minutes only (every minute attends to every minute). Nothing from the target minute enters the model. |
| 6 | **Pre-LayerNorm blocks** with residual connections, dropout 0.1 on each sublayer output (and inside the FFN), and a **final LayerNorm** after the last block. No dropout on attention weights. |
| 7 | **Next-step head predicts the scaled telemetry of the target minute directly** (not a delta). |
| 8 | **Class weights** `N / (5 · n_class)` from the **88 fitting cases only** (33/29/9/13/4 → 0.533, 0.607, 1.956, 1.354, 4.4). Cross-entropy uses PyTorch's weighted mean (Σ w_i·nll_i / Σ w_i). |

Further implementation choices (not in the design, listed so they can be challenged): 3 modality tokens in the order metric, trace, log; padded nodes output exactly 0; node type is passed to `forward` so any node count works; attention is written out explicitly (no fused kernel) so masked rows behave exactly as documented.

## 3. Tensor flow (D1: B windows, T = 6, N = 46, C = 130)

```
inputs (B,6,46,130), input_mask (B,6,46,130) bool
  InputAdapter  (per node type, per modality: Linear(2·channels, 64) on [where(mask, x, 0), mask])
     -> tokens (B,6,46,3,64), token_avail (B,6,46,3)
  Core: ModalityFusion -> (B,6,46,64) + node-type embedding + time embedding
        2 x [ temporal attention (per node, 6 minutes) -> cross-node attention (per minute, N nodes, key mask = node_valid) -> FFN ]
        final LayerNorm -> hidden (B,6,46,64)
  last minute hidden[:, -1] (B,46,64)
     FaultHead:    mean and max over valid nodes -> (B,128) -> Linear(128,64) -> GELU -> Linear(64,5)   -> logits (B,5)
     NextStepHead: type-specific Linear(64 -> 73 pod | 57 VM) scattered to the schema channels, 0 elsewhere -> prediction (B,46,130)
```

Parameters: **168,200** (design estimate "about 0.2 million"; measured, asserted in the tests).

## 4. Modules

| File | Content |
|---|---|
| `spec.py` | `ModelConfig`; `AdapterSpec.from_d1(schema, applicable)`: node-type vector and channel lists per (type, modality) from the schema and the declared mask rule (names, never data values; counts checked against the approved 53/9/11/57). |
| `adapters.py` | `InputAdapter`, `OutputAdapter` (the only code that knows the channel layout). |
| `core.py` | `ModalityFusion`, `MultiHeadAttention`, `FactorisedBlock`, `SpatioTemporalCore`. Contains no 46, no 130, no channel name (a test greps for it). |
| `heads.py` | `FaultHead`, `NextStepHead`. Both receive the last historical minute only. |
| `losses.py` | `masked_mse`, `weighted_cross_entropy`, `inverse_frequency_weights`, `fit_class_weights`. |
| `model.py` | `Model1`, `enable_determinism`, `state_digest`, `tensor_digest`. |
| `smoke_forward_backward.py` | Real-D1 CPU smoke test, reproducibility check and the evidence writer. |

## 5. Masking rules (each is tested)

- Unavailable entries are replaced by 0 with `torch.where` before any arithmetic, so a huge, infinite or NaN value in an unavailable slot cannot reach the network and its gradient is exactly 0.
- A token with no effective entry is unavailable; its fusion weight is exactly 0. A VM never has trace or log tokens.
- A padded node (`node_valid` False) counts as having no available token, is never an attention key, and is reset to 0 after every block. Attention rows with no valid key return 0, never NaN. A sample with no valid node gives finite outputs.
- Pooling uses −inf for the max and a count-normalised mean over valid nodes only.
- Losses are computed over effective entries only; prediction and target are cleaned with `torch.where` before the subtraction so a masked target of any size cannot create `inf · 0 = NaN` gradients. An empty effective mask gives loss 0 and zero gradients. A fault label outside 0..4 (including −1 "no label") raises.
- A padded node is cleared before the adapter (mask False), so garbage there cannot reach any weight gradient.
- Predictions are exactly 0 on channels a node type does not have; passing `applicable_entry` also zeroes structurally unavailable entries (for example the ten pods without a log block).

## 6. Determinism

`enable_determinism()`: deterministic kernels, one CPU thread. `Model1.build(spec, cfg, seed)` seeds before constructing, so the same seed gives bit-identical initial weights (checked by hash). Dropout is seeded in the smoke test. Evidence records the initial-state hash, the output hashes and the loss values, and compares two in-process runs and two clean-process runs.

## 7. Tests

| File | Covers |
|---|---|
| `tests/prep/test_m1_model_shapes.py` | schema-derived spec, adapter widths, shapes, parameter count, bidirectional temporal attention, last-minute heads, determinism, core independence from 46/130, variable node counts, all-masked rows and nodes, attention finiteness, losses. |
| `tests/prep/test_m1_model_invariance.py` | MI-1 … MI-7 exactly as in design note 3.4 (MI-6 on the real cache). |
| `tests/prep/test_m1_model_real_smoke.py` | real fitting batches (S1 and S2), forward, losses, backward, gradient coverage, reproducibility, leakage guards. |

Evidence: `results/core_v0/evidence_model1_core_v0.json`. Historical evidence (`evidence.json`, `evidence_m1_03.json`, the XGBoost evidence) is not modified; their hashes are recorded and compared with `HEAD`.

## 8. Observations on the approved design (nothing changed)

1. Design 4.2 justifies d = 64 by "602 fault windows (101 cases)". With the approved calibration split the S2 objective has **523 windows from 88 cases**. The size is kept; the rationale is stale.
2. Design 3.4 MI-1 mentions "window score and per-node scores". Those are computed by `detection_calibration.py` from the prediction, target and mask, not by the model; the tests compute them with that module.
3. The model learns nothing about the call graph and has no node IDs (known limitation, design 4.2).

## 9. Out of scope here

Optimizer, training loop, early stopping, simulated-missing training switch (design 4.4), calibration fitting, validation or test evaluation, and the Online Boutique adapter.

## 10. Final review notes (298-50)

**Checked and found correct** (each has a test): temporal attention then cross-node attention then FFN in every block; the forward pass takes only the six historical minutes, so no target-minute observation can enter (the target is used only by the loss); feature, modality and node masking; fully masked attention rows (zeros, never NaN); loss gradients and denominators against closed forms; class order (manifest `label_map.classes`, labels and `cases_by_class` agree) and fit-only class weights; variable node counts; deterministic initialisation and output reproduction; the smoke path builds only `training_dataset` objects (fitting windows), takes no optimizer step and never touches an evaluation dataset.

**Defect found and fixed.** NaN or inf in the slot of a *padded* node whose mask was True produced non-finite gradients for the adapter weights (`0 · NaN` inside the Linear backward), although the forward output was correct. `Model1.forward` now clears the input mask of padded nodes before the adapter, and rejects a non-bool or wrongly shaped mask with `ValueError`. Regression test: `PaddedNodeGradientRegressionTests`.

**Implications of three decisions (unchanged):**
- *Shared modality gate.* One `Linear(64,1)` scores every modality token, so the gate cannot learn "trust logs less than metrics" except through how the three adapters shape their token vectors; that is adequate for v0 (D1 has no run-time missingness) and a per-modality gate is a cheap later ablation.
- *Dropout placement.* Dropout acts on each sublayer output and inside the FFN, never on attention weights or on the fusion weights. In eval mode it is off, so the invariance tests are exact; with dropout on, results are reproducible only when the RNG is seeded (the smoke test seeds it). Dropout does not break the zero-gradient rule for unavailable entries (tested in train mode).
- *Weighted-mean cross-entropy.* The loss divides by the sum of the batch's class weights, so a batch of only rare-class windows has the same loss scale as a batch of only common-class windows; the weights set the relative influence of classes within a batch, not the overall step size. A sampler that over-represents a rare class would count it twice (weights and sampling). `masked_mse` divides by the number of effective entries in the whole batch, so every effective entry weighs the same (on D1 every window has the same effective count).

**Process-wide side effects to know about:** `enable_determinism()` sets deterministic kernels and one CPU thread for the whole process, and `Model1.build` calls `torch.manual_seed`. The test helper calls `enable_determinism()` on import.
