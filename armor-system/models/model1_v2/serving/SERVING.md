# Model 1 v2 inference service

What it is: a local HTTP service that runs the frozen Model 1 v2 ensemble (`../../MODEL1_TECHNICAL_REPORT.md`) on D1-layout telemetry minutes and returns incident alerts and fault-type predictions.

| Status | |
|---|---|
| Works end to end (tested) | Validation, windowing, five-checkpoint ensemble, alerting, incidents and fault type over HTTP. A replay of the D1 validation period reproduces the committed v2 validation result exactly (section 6). |
| Replay vs live | **Only replay has been run.** Every replayed request is labelled `source.kind = "replay"` and the response carries a `replay_data` warning. **No live telemetry source exists yet** (section 8). |
| Not done | Container image, cloud deployment, Airflow DAG, downstream model consumers, authentication. These belong to teammates' components and are proposals here (sections 7–8). |

Code lives in `armor-system/models/model1_v2/serving/`. It modifies no frozen v1 file and no teammate folder (`backend/`, `deployment/`, `airflow/`, other `models/`).

## 1. Files

| File | Role |
|---|---|
| `contract.py` | Request schema `model1.telemetry.v1`: validation and error codes. No torch. |
| `service.py` | Loads artefacts and checkpoints (hash-checked), builds windows, scores the ensemble, alerts, incidents and warnings. |
| `server.py` | Standard-library HTTP server (`/healthz`, `/v1/metadata`, `/v1/predict`). |
| `replay_demo.py` | Replay client: streams recorded D1 validation minutes through the HTTP API. |
| `bench.py` | Latency benchmark (synthetic values). |
| `requirements-serving.txt` | `torch==2.14.0`, `numpy==2.5.3` (Python 3.14.4) |
| `../../../tests/model1_v2/test_serving.py` | 11 tests: contract, data-layer equality, HTTP, end-to-end replay |

## 2. Setup and checkpoint requirements

1. **Code and artefacts.** Check out the repository; the service imports `armor-system/data/prep`, `models/model1_transformer` and `evaluation/detection`. It reads three **committed** artefacts, verified on start:
   - `data/artifacts/d1/inspection/schema.json`: node and channel order.
   - `data/artifacts/d1/splits/model1_calibration_split_v1.json`: the approved manifest; its content hash is checked.
   - `data/artifacts/d1/preprocessing/scaler_model1_fit_only_v1.npz`: must match the manifest hash.

   It does **not** need the D1 dataset. Only the replay demo and two tests read the local D1 feature cache.
2. **Python.** Install with `pip install -r requirements-serving.txt` (or use the project `.venv`).
3. **Checkpoints.** Ten files, kept outside Git: `model1_v1/seed_{0..4}/s2_best.pt` and `s2_best.json`.
   - Put them under `$MODEL1_CHECKPOINT_ROOT` (default `~/298A_project/checkpoints`).
   - At start each file is hashed against `model1_transformer/results/test_freeze/evidence_test_freeze.json`. A missing or altered file stops the service (`ServiceError`).
   - Total size is 3.5 MB.
   - **They are only on Khushi's machine today.** The team must choose shared storage (section 8).
4. **Run:**

   ```
   MODEL1_CHECKPOINT_ROOT=~/298A_project/checkpoints .venv/bin/python -I armor-system/models/model1_v2/serving/server.py --host 127.0.0.1 --port 8081
   curl -s localhost:8081/healthz        # {"status": "ok", "model": "model1-v2-ensemble", "version": "2.0.0"}
   ```

   Environment variables:
   - `MODEL1_CHECKPOINT_ROOT`
   - `MODEL1_HOST`, `MODEL1_PORT`
   - `MODEL1_TORCH_THREADS` (default 1, the setting all numbers were produced with)
5. **Tests:** `.venv/bin/python -m unittest armor-system/tests/model1_v2/test_serving.py`

Container: not built. The Docker daemon was not running on the development machine, so no image or Dockerfile is committed. A container needs only Python 3.14, the two pinned packages, the repository's `armor-system/` tree and the checkpoint directory mounted at `MODEL1_CHECKPOINT_ROOT`.

## 3. API contract

### Endpoints

| Method and path | Response |
|---|---|
| `GET /healthz` | `200 {"status": "ok", "model", "version"}`. The server only starts after artefacts and checkpoints verify. |
| `GET /v1/metadata` | Model name and version, checkpoint sha256s, detection threshold, scaler hash, `layout` fingerprints, `node_order` (46), `channel_order` (130), modality index ranges, segment limits, incident rule. |
| `POST /v1/predict` | `Content-Type: application/json`, body `model1.telemetry.v1`. Returns `200 model1.prediction.v1` or an error. |

### Request `model1.telemetry.v1`

| Field | Required | Meaning |
|---|---|---|
| `schema_version` | yes | `"model1.telemetry.v1"` |
| `source` | yes | `{"kind": "replay" \| "live", "name": str, ...}`. Echoed back. `replay` adds a warning to every response. |
| `layout` | yes | `{"node_order_sha256", "channel_order_sha256"}` from `/v1/metadata`. Prevents silently mis-ordered inputs. |
| `timestamps` | yes | 7–240 integer epoch seconds, strictly consecutive (step 60 s). Split streams at gaps. |
| `values` | yes | `[T][46][130]` finite numbers: D1-layout features **before** Model 1 scaling, in `node_order` × `channel_order`. |
| `observed` | no | `[T][46][130]` booleans (default all true). Missing measurements are `false`; their values are ignored. |
| `previous` | no | `{"target_ts", "detection_score"}` of the window just before this segment's first target minute. It makes the 60 s rolling maximum continuous across requests. |

Each window is 6 input minutes followed by its **target minute** (the alert minute). A segment of T minutes yields T − 6 windows, with target minutes `timestamps[6:]`. The target minute's values are never a model input.

### Errors

All errors have the form `{"error": {"code", "field", "message"}}`; nothing is repaired silently.

| HTTP | Code | When |
|---|---|---|
| 400 | `invalid_json` | Body is not JSON. |
| 400 | `not_an_object` | Body is not a JSON object. |
| 400 | `bad_schema_version` | `schema_version` is wrong or missing. |
| 400 | `bad_source` | `source` kind or name is invalid. |
| 400 | `layout_mismatch` | Node or channel order fingerprint is wrong or missing. |
| 400 | `bad_timestamps` | Timestamps are not a list of integers. |
| 400 | `bad_length` | Fewer than 7 or more than 240 minutes. |
| 400 | `not_consecutive` | Timestamps are not consecutive minutes. |
| 400 | `missing_values`, `malformed_array`, `bad_shape` | `values` is missing, ragged or the wrong shape. |
| 400 | `non_finite` | `values` contains NaN or infinity. |
| 400 | `bad_observed` | `observed` is not booleans of the same shape. |
| 400 | `value_at_structurally_absent_entry` | A non-zero observed value sits at an entry the model never uses. |
| 400 | `bad_previous`, `previous_not_adjacent` | `previous` is malformed or not the preceding minute. |
| 404 | `not_found` | Unknown path. |
| 405 | `method_not_allowed` | Wrong method for the path. |
| 411 | `length_required` | Content-Length is missing or not numeric. |
| 413 | `body_too_large` | Body exceeds 64 MB; refused before reading. |
| 415 | `unsupported_media_type` | Content-Type is not JSON. |
| 500 | `internal_error` | Unexpected failure. It is logged on the server, and no detail is returned. |

### Response `model1.prediction.v1`

```json
{"schema_version": "model1.prediction.v1",
 "model": {"name": "model1-v2-ensemble", "version": "2.0.0", "detection_threshold": 0.7831957951425433},
 "source": {...}, "replay": true, "n_windows": 60,
 "windows": [{"target_ts": 1651715400, "input_start_ts": 1651715040, "detection_score": 0.81, "detection_score_seed_sd": 0.07,
              "rolled_score": 0.83, "alert": true, "fault_probs": {"Container Hardware": 0.03, "...": 0.0}, "fault_pred": "Node Disk"}],
 "alert_minutes": [1651715400, 1651715460],
 "incidents": [{"start_ts": 1651715400, "end_ts": 1651715460, "alert_minutes": 2, "may_continue_after_segment": false,
                "fault_type": {"pred": "Node Disk", "probs": {...}, "basis": "mean ensemble window probabilities over the incident's windows"}}],
 "warnings": [{"code": "replay_data", "message": "..."}], "latency_ms": 499.7}
```

Warnings:
- `replay_data`: the request was replay.
- `vm_metrics_unobserved`: a window has no observed VM metric. Detection degrades sharply without VM metrics; v1's test F1 was 0.41–0.60 with VMs hidden.
- `out_of_distribution_values`: a scaled value above 50 in absolute size. D1 stays within about 14, so check units and channel order.

## 4. Semantics (what the numbers mean)

- **Window score** (`detection_score`): the mean over the five seeds of sigmoid(detection logit). `detection_score_seed_sd` is the spread across seeds, an uncertainty hint that has not been calibrated.
- **Alert:** the causal 60 s rolling maximum of the window score (this minute and the previous one) is greater than **0.7831957951425433**. That threshold is the 90th percentile on the 329 calibration windows; it is frozen and never refitted at serve time.
- **Incidents:** ARMOR's rule. Alert minutes less than 600 s apart merge, and single-minute intervals are dropped. They are computed per request; a streaming caller merges across requests with the same rule, as `replay_demo.incidents` does. `may_continue_after_segment` marks an incident that touches the end of the segment.
- **Fault type:**
  - **Per window:** `fault_probs`.
  - **Per incident:** the mean of the window probabilities over the incident's windows.
  - The evaluated metric (technical report, section 7) averaged over the **documented case windows** (a case-time window), not over alert windows. The serving basis is therefore not the evaluated basis. On the replayed validation period it agreed with the documented class for **16 of 16** incidents that hit a case (section 6). Those are development cases, not an independent estimate.
- **Not provided:** root-cause localisation (no node or service is named), and incident end times beyond the alert interval.
- **Numerics:** scores depend on batch composition at about 1e-8 (float summation order). Alerts can differ only for a score within about 1e-7 of the threshold. The end-to-end replay reproduces the committed alerts and incidents exactly.

## 5. Latency (`../results/serving_latency.json`, `bench.py`)

Measured on an Apple M4 with 1 torch thread, one request at a time, synthetic values. Service start, including checkpoint hashing, takes 0.35 s.

| Segment | Windows | Request JSON | Round trip (median) | Server time | Server time per window |
|---|---|---|---|---|---|
| 7 minutes | 1 | 0.56 MB | 22 ms | 10 ms | 10.4 ms |
| 66 minutes | 60 | 5.2 MB | 606 ms | 500 ms | 8.3 ms |
| 240 minutes | 234 | 19 MB | 2.35 s | 1.96 s | 8.4 ms |

At one request per minute (7-minute segments with `previous`), a single CPU thread is busy about 10 ms per minute, under 0.1% of the time. JSON is the main overhead; a binary encoding is a possible later change to the contract.

## 6. End-to-end replay demonstration (replay, not live)

```
.venv/bin/python -I armor-system/models/model1_v2/serving/replay_demo.py --start-server --url http://127.0.0.1:8081
```

- **Data:** ARMOR's released D1 features (Zenodo 21744694, CC-BY-4.0) for the **validation period** (875 minutes on 2022-05-05). This is development data only, never the D1 test period. It is read from the local D1 feature cache (`fetch_d1.py`, then the data pipeline's cache stage).
- **Flow:** 15 overlapping 66-minute requests, each passing `previous`. The client merges incidents over the stream and scores them against the 25 documented validation incidents with ARMOR's matching.
- **Result** (`../results/replay_demo/replay_summary.json`, `incidents.jsonl`):
  - **22 incidents: 18 incidents hit, 6 false-positive intervals, F1 0.7347.** This is identical to the committed v2 validation ensemble.
  - The serving-path fault type equals the documented class for 16 of 16 incidents that hit a case.
  - Median request time is 561 ms.
- **Interpretation:** these are validation-period numbers that were seen when v2 was chosen. The demo proves the serving path is faithful, not that the model generalises.

## 7. How downstream components can consume predictions (proposals; nothing is wired)

The unit to consume is a `model1.incident.v1` event, one JSON object per line in `incidents.jsonl`: `start_ts`, `end_ts`, `fault_type.pred`, `fault_type.probs`, `detected_by` and `source`.

| Consumer | Owner | Proposed use |
|---|---|---|
| Orchestration (Airflow) | Bhoomika | Each minute, a task posts the latest 7 minutes plus `previous` to `/v1/predict`, stores the windows, merges incidents with the 600 s rule and emits `model1.incident.v1` when an incident closes (no alert for 600 s). |
| Model 2, graph RCA | not yet started | Use `[start_ts − 300, end_ts + 300]` as the analysis window. Use `fault_type.probs` to weight the candidate node family (container vs VM). Model 1 does **not** localise the root cause. |
| Model 3, LLM recovery | not yet started | Put the incident, the fault-type probabilities and the warnings (especially `vm_metrics_unobserved` and `replay_data`) into the diagnosis context, and state the uncertainty. |
| Model 4, action risk | not yet started | Use the top-class probability and `detection_score_seed_sd` as confidence inputs. A low-confidence incident should gate autonomous actions toward human review. |

## 8. What still needs team integration

1. **Live telemetry adapter (the main gap).**
   - The model accepts only D1's layout: 46 named Online Boutique pods/VMs × 130 ARMOR feature channels, in ARMOR's preprocessed units.
   - A live source needs a mapping from the testbed's metrics, traces and logs to exactly those channels and units.
   - If the testbed differs from D1 (other services, nodes or channels), the frozen model **cannot** be applied without retraining, which is out of scope here.
2. **Checkpoint storage and distribution:** a shared bucket or artefact store with the ten files and their hashes.
3. **Container and cloud deployment:** owned by Bhoomika (`deployment/`). It needs Python 3.14 plus `requirements-serving.txt`. Add TLS and authentication in front of the service; it has none.
4. **Airflow DAG and persistence** for windows and incidents (section 7).
5. **Downstream consumers** for Models 2–4: agree on the `model1.incident.v1` fields with their owners.
6. **Operations:** request logging exists. Metrics, alerting on `vm_metrics_unobserved` and out-of-distribution warnings, and the input-drift monitoring the calibration needs all remain to be built.
