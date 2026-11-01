"""REPLAY demonstration: stream recorded D1 minutes through the Model 1 v2 HTTP service (SERVING.md section 6).

    .venv/bin/python -I armor-system/models/model1_v2/serving/replay_demo.py --start-server [--out DIR]
    .venv/bin/python -I armor-system/models/model1_v2/serving/replay_demo.py --url http://127.0.0.1:8081

This is **replay, not live telemetry**: the minutes are ARMOR's released D1 features (CC-BY-4.0) for the validation period
(2022-05-05 01:24 to 2022-05-05 16:00 UTC, development data only, never the D1 test period), sent as source.kind="replay".
Segments of 66 minutes (60 target minutes each) overlap by 6 minutes; each request passes the previous window's score so the
rolling maximum is continuous across segments. Incidents are built over the whole stream with ARMOR's interval rule and, because
replayed validation incidents have documented timestamps, scored with ARMOR's matching. Expected: exactly the committed v2
validation ensemble (TP 18, FP intervals 6, F1 0.7347).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ARMOR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPO = os.path.dirname(ARMOR)
for _p in ("data", "models", os.path.join("evaluation", "detection")):
    if os.path.join(ARMOR, _p) not in sys.path:
        sys.path.insert(0, os.path.join(ARMOR, _p))

import numpy as np  # noqa: E402

from model1_v2.serving import contract as ct  # noqa: E402

TARGETS_PER_SEGMENT = 60
SOURCE = {"kind": "replay", "name": "armor-d1-validation-period", "dataset": "ARMOR D1 release (Zenodo 21744694, CC-BY-4.0), feature cache",
          "note": "recorded 2022 data replayed for demonstration; not live telemetry"}
OUT = os.path.join(os.path.dirname(HERE), "results", "replay_demo")


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def validation_replay():
    """(timestamps, values) of the validation period from the local D1 feature cache, and the validation incident timestamps."""
    from prep import loader
    from prep.common import read_json
    man = read_json(os.path.join(ARMOR, "data", "artifacts", "d1", "splits", "model1_calibration_split_v1.json"))
    b1, b2 = man["protocol"]["outer_period_boundaries"]["train_validation"], man["protocol"]["outer_period_boundaries"]["validation_test"]
    ts, X, _ = loader.load_arrays()
    sel = (ts >= b1) & (ts < b2)
    labels = sorted(int(a["timestamp"]) for a in man["cases"]["assignments"] if a["role"] == "validation")
    return ts[sel], X[sel], labels


def segments(ts):
    """Start/stop indices of overlapping request segments: each holds 6 context minutes + up to 60 target minutes."""
    out, k = [], 0
    while k + ct.WINDOW < len(ts):
        out.append((k, min(k + ct.WINDOW + TARGETS_PER_SEGMENT, len(ts))))
        k += TARGETS_PER_SEGMENT
    return out


def incidents(windows, gap=600):
    import armor_scoring as sc
    t = np.array([w["target_ts"] for w in windows])
    alerts = [w["target_ts"] for w in windows if w["alert"]]
    probs = np.array([[w["fault_probs"][c] for c in w["fault_probs"]] for w in windows])
    names = list(windows[0]["fault_probs"])
    out = []
    for s, e in sc.get_pre_interval(alerts, gap):
        p = probs[(t >= s) & (t <= e)].mean(0)
        out.append({"schema_version": "model1.incident.v1", "start_ts": int(s), "end_ts": int(e), "detected_by": "model1-v2-ensemble",
                    "fault_type": {"pred": names[int(p.argmax())], "probs": dict(zip(names, map(float, p)))}, "source": SOURCE})
    return out


def run(url, out_dir=OUT, log=print):
    import armor_scoring as sc
    meta = get(url + "/v1/metadata")
    ts, X, labels = validation_replay()
    log(f"REPLAY (not live): {len(ts)} recorded D1 validation minutes, {len(labels)} documented incidents, server {meta['model']['name']} {meta['model']['version']}")
    windows, lat, prev = [], [], None
    for a, b in segments(ts):
        body = {"schema_version": ct.REQUEST_SCHEMA, "source": SOURCE, "layout": meta["layout"], "timestamps": [int(x) for x in ts[a:b]],
                "values": X[a:b].tolist()}
        if prev is not None:
            body["previous"] = prev
        t0 = time.perf_counter()
        code, res = post(url + "/v1/predict", body)
        if code != 200:
            raise RuntimeError(f"segment {a}:{b} rejected: {res}")
        lat.append({"minutes": b - a, "windows": res["n_windows"], "round_trip_ms": round((time.perf_counter() - t0) * 1000, 1), "server_ms": res["latency_ms"]})
        windows += res["windows"]
        prev = {"target_ts": res["windows"][-1]["target_ts"], "detection_score": res["windows"][-1]["detection_score"]}
    inc = incidents(windows)
    score = sc.eval_AD_detail([(i["start_ts"], i["end_ts"]) for i in inc], labels, 300)
    summary = {"replay": True, "not_live_telemetry": True, "source": SOURCE, "model": meta["model"], "minutes": int(len(ts)), "windows": len(windows),
               "requests": len(lat), "alert_minutes": sum(w["alert"] for w in windows), "incidents": len(inc),
               "scoring_against_documented_validation_incidents": {k: (float(v) if not isinstance(v, int) else v) for k, v in score.items()},
               "expected_from_committed_v2_validation": {"TP": 18, "FP": 6, "f1": 0.7347}, "latency": lat}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "replay_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(out_dir, "incidents.jsonl"), "w") as fh:
        for i in inc:
            fh.write(json.dumps(i) + "\n")
    log(f"REPLAY result: {len(inc)} incidents, TP {score['TP']}, FP {score['FP']}, F1 {float(score['f1']):.4f} "
        f"(committed v2 validation: TP 18, FP 6, F1 0.7347); median request {np.median([x['round_trip_ms'] for x in lat]):.0f} ms for 66 minutes")
    return summary, inc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--start-server", action="store_true", help="start serving/server.py as a subprocess on --url's port")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    proc = None
    if a.start_server:
        port = a.url.rsplit(":", 1)[1]
        proc = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "server.py"), "--port", port])
        for _ in range(240):
            try:
                get(a.url + "/healthz", timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        else:
            proc.terminate()
            raise SystemExit("server did not become healthy")
    try:
        run(a.url, a.out)
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=30)


if __name__ == "__main__":
    main()
