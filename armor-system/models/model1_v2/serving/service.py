"""Model 1 v2 serving core: frozen preprocessing + the five-seed ensemble + ARMOR interval rules (SERVING.md section 4).

`Model1Service.load()` needs only committed artefacts (schema, approved split manifest, fit-only scaler) and the five frozen v1
Stage 2 checkpoints; it never reads the D1 feature cache. Windows are built exactly as the frozen data layer builds them:
`preprocess.transform` with the fit-only scaler, `mask_values` with the applicable mask, then per window the effective mask
(applicable AND observed). Detection is v2's rule (mean sigmoid -> causal 60 s rolling max -> frozen calibration threshold);
incidents use ARMOR's interval rule (alert minutes less than 600 s apart merge, single-minute intervals are dropped).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ARMOR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in ("data", "models", os.path.join("evaluation", "detection")):
    if os.path.join(ARMOR, _p) not in sys.path:
        sys.path.insert(0, os.path.join(ARMOR, _p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model1_v2.serving import contract as ct  # noqa: E402

MODEL = {"name": "model1-v2-ensemble", "version": "2.0.0", "protocol": "armor-system/models/model1_post_test_analysis/V2_ENSEMBLE_PROTOCOL.md (2ac3ab7)",
         "members": "five frozen v1 Stage 2 checkpoints (seeds 0-4)", "classes": None}
INCIDENT_GAP_S = 600
BATCH = 64
CLASS_NAMES = ("Container Hardware", "Container Network", "Node CPU", "Node Disk", "Node Memory")


class ServiceError(RuntimeError):
    """The service cannot start or run (missing or altered artefacts / checkpoints)."""


def load_artifacts(art_root=os.path.join(ARMOR, "data", "artifacts", "d1")):
    """Schema, verified fit-only scaler and applicable mask from committed artefacts (the same checks as Model1Data.load)."""
    from prep import m1_masks, m1_split as ms
    from prep.common import read_json, sha256_file
    manifest = read_json(os.path.join(art_root, "splits", f"{ms.MANIFEST_NAME}.json"))
    if manifest.get("content_sha256") != ms.manifest_content_hash(manifest) or manifest.get("status") != "approved":
        raise ServiceError("the Model 1 split manifest is altered or not approved")
    scaler_path = os.path.join(art_root, "preprocessing", f"{ms.SCALER_NAME}.npz")
    if sha256_file(scaler_path) != manifest["fit_only_scaler"]["file_sha256"]:
        raise ServiceError("the fit-only scaler differs from the one recorded in the approved manifest")
    z = np.load(scaler_path)
    scaler = ms.scaler_from_arrays({k: z[k] for k in z.files})
    schema = read_json(os.path.join(art_root, "inspection", "schema.json"))
    applicable, _, report = m1_masks.applicable_entry_mask(schema, scaler["absent_entry"])
    m1_masks.check_expected_counts(report)
    return {"schema": schema, "scaler": scaler, "applicable": applicable, "scaler_sha256": manifest["fit_only_scaler"]["file_sha256"]}


class Model1Service:
    def __init__(self, artifacts, model, checkpoint_sha256):
        self.schema, self.scaler, self.applicable = artifacts["schema"], artifacts["scaler"], artifacts["applicable"]
        self.scaler_sha256 = artifacts["scaler_sha256"]
        self.model, self.checkpoint_sha256 = model, checkpoint_sha256
        self.vm = np.array([n.startswith("node-") for n in self.schema["node_order"]])
        self.metric = np.zeros(len(self.schema["channel_order"]), bool)
        self.metric[self.schema["modality_index_ranges"]["metric"]] = True

    @classmethod
    def load(cls, ckpt_root=None):
        from model1_transformer import test_freeze as tf
        from model1_transformer.spec import AdapterSpec
        from model1_v2.inference import CKPT_ROOT, Model1V2
        torch.set_num_threads(int(os.environ.get("MODEL1_TORCH_THREADS", "1")))
        root = ckpt_root or os.environ.get("MODEL1_CHECKPOINT_ROOT", CKPT_ROOT)
        missing = [p for s in range(5) for p in (f"model1_v1/seed_{s}/s2_best.pt", f"model1_v1/seed_{s}/s2_best.json") if not os.path.isfile(os.path.join(root, p))]
        if missing:
            raise ServiceError(f"checkpoints missing under {root}: {missing[:3]} (see SERVING.md section 2)")
        art = load_artifacts()
        try:
            model = Model1V2.load(root, spec=AdapterSpec.from_d1(art["schema"], art["applicable"]))
        except RuntimeError as e:
            raise ServiceError(str(e)) from e
        shas = [tf.sha(os.path.join(root, f"model1_v1/seed_{s}/s2_best.pt")) for s in range(5)]
        return cls(art, model, shas)

    # ------------------------------------------------------------------ metadata
    def metadata(self):
        return {"model": {**MODEL, "classes": list(CLASS_NAMES), "checkpoint_sha256": self.checkpoint_sha256, "detection_threshold": self.model.threshold,
                          "fit_only_scaler_sha256": self.scaler_sha256},
                "request_schema": ct.REQUEST_SCHEMA, "response_schema": ct.RESPONSE_SCHEMA, "layout": ct.layout(self.schema),
                "node_order": self.schema["node_order"], "channel_order": [c["name"] for c in self.schema["channel_order"]],
                "modality_index_ranges": self.schema["modality_index_ranges"],
                "segment": {"min_minutes": ct.MIN_MINUTES, "max_minutes": ct.MAX_MINUTES, "step_s": ct.STEP_S, "window_input_minutes": ct.WINDOW},
                "applicable_entries": int(self.applicable.sum()),
                "incident_rule": {"merge_gap_s": INCIDENT_GAP_S, "single_minute_intervals": "dropped"}}

    # ------------------------------------------------------------------ prediction
    def scaled(self, values):
        from prep import m1_masks, preprocess as pp
        Xt, _ = pp.transform(values, self.scaler, self.schema["modality_index_ranges"])
        return m1_masks.mask_values(Xt, self.applicable)

    def windows(self, seg):
        """Model inputs for every window of the segment, built as Model1WindowDataset builds them."""
        from prep import m1_masks
        Xt = self.scaled(seg.values)
        eff = seg.observed & self.applicable[None]
        n = len(seg.timestamps) - ct.WINDOW
        inputs = np.stack([m1_masks.mask_values(Xt[k:k + ct.WINDOW], eff[k:k + ct.WINDOW]) for k in range(n)])
        mask = np.stack([eff[k:k + ct.WINDOW] for k in range(n)])
        return inputs, mask, seg.timestamps[ct.WINDOW:], Xt

    def score(self, inputs, mask):
        det, fault, det_seed = [], [], []
        app = torch.from_numpy(self.applicable.copy())
        for i in range(0, len(inputs), BATCH):
            b = {"inputs": torch.from_numpy(inputs[i:i + BATCH]), "input_mask": torch.from_numpy(mask[i:i + BATCH])}
            b["applicable_entry"] = app[None].expand(len(b["inputs"]), *app.shape).contiguous()
            o = self.model.predict_windows(b)
            det.append(o["det_prob"]), fault.append(o["fault_probs"]), det_seed.append(o["det_prob_per_seed"])
        return np.concatenate(det), np.concatenate(fault), np.concatenate(det_seed, axis=1)

    def predict(self, seg):
        import armor_scoring as sc
        from model1_transformer.rolling import rolling_max_causal
        t0 = time.perf_counter()
        inputs, mask, target_ts, Xt = self.windows(seg)
        det, fault, det_seed = self.score(inputs, mask)
        if seg.previous is not None:
            r = rolling_max_causal(np.r_[seg.previous["detection_score"], det], np.r_[seg.previous["target_ts"], target_ts], np.zeros(len(det) + 1, int))[1:]
        else:
            r = rolling_max_causal(det, target_ts, np.zeros(len(det), int))
        alert = r > self.model.threshold
        alerts = [int(t) for t in target_ts[alert]]
        incidents = []
        for s, e in sc.get_pre_interval(alerts, INCIDENT_GAP_S):
            sel = (target_ts >= s) & (target_ts <= e)
            p = fault[sel].mean(0)
            incidents.append({"start_ts": int(s), "end_ts": int(e), "alert_minutes": int((alert & sel).sum()), "may_continue_after_segment": bool(e == int(target_ts[-1])),
                              "fault_type": {"pred": CLASS_NAMES[int(p.argmax())], "probs": dict(zip(CLASS_NAMES, map(float, p))),
                                             "basis": "mean ensemble window probabilities over the incident's windows"}})
        warnings = list(seg.warnings)
        vm_metric = mask[:, :, self.vm][..., self.metric]
        if not vm_metric.any(axis=(1, 2, 3)).all():
            warnings.append({"code": "vm_metrics_unobserved", "message": "some windows have no observed VM metric; detection degrades sharply without VM metrics (v1 test F1 0.41-0.60)"})
        big = np.abs(Xt[seg.observed & self.applicable[None]]) > ct.OOD_SCALED_ABS
        if big.any():
            warnings.append({"code": "out_of_distribution_values", "message": f"{int(big.sum())} scaled values exceed |{ct.OOD_SCALED_ABS}| (D1 stays within about 14); check units and channel order"})
        wins = [{"target_ts": int(t), "input_start_ts": int(t) - ct.WINDOW * ct.STEP_S, "detection_score": float(d), "detection_score_seed_sd": float(det_seed[:, i].std(ddof=1)),
                 "rolled_score": float(rr), "alert": bool(a), "fault_probs": dict(zip(CLASS_NAMES, map(float, f))), "fault_pred": CLASS_NAMES[int(f.argmax())]}
                for i, (t, d, rr, a, f) in enumerate(zip(target_ts, det, r, alert, fault))]
        return {"schema_version": ct.RESPONSE_SCHEMA, "model": {"name": MODEL["name"], "version": MODEL["version"], "detection_threshold": self.model.threshold},
                "source": seg.source, "replay": seg.source["kind"] == "replay", "n_windows": len(wins), "windows": wins, "alert_minutes": alerts,
                "incidents": incidents, "warnings": warnings, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    def handle(self, body):
        """Validate + predict. Raises contract.RequestError for client errors."""
        return self.predict(ct.validate(body, self.schema, self.applicable))
