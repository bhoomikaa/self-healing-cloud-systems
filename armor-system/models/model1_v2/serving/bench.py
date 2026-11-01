"""Latency of the Model 1 v2 HTTP service for segment sizes 7 / 66 / 240 minutes (synthetic in-layout values; SERVING.md section 5).

    .venv/bin/python -I armor-system/models/model1_v2/serving/bench.py [--out FILE]

Starts the server in-process on a free port, sends each size 5 times (after one warm-up) and records round-trip and server-side
milliseconds. Values are random numbers at the model's applicable entries, so they test speed only, not prediction quality.
"""
import argparse
import json
import os
import platform
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ARMOR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in ("data", "models", os.path.join("evaluation", "detection")):
    if os.path.join(ARMOR, _p) not in sys.path:
        sys.path.insert(0, os.path.join(ARMOR, _p))

import numpy as np  # noqa: E402

from model1_v2.serving import contract as ct  # noqa: E402
from model1_v2.serving.replay_demo import post  # noqa: E402
from model1_v2.serving.server import serve  # noqa: E402
from model1_v2.serving.service import Model1Service  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "results", "serving_latency.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    svc = Model1Service.load()
    load_s = time.perf_counter() - t0
    httpd = serve(svc, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/v1/predict"
    rng = np.random.default_rng(0)
    res = {}
    for T in (7, 66, 240):
        v = np.where(svc.applicable[None], rng.normal(size=(T, 46, 130)), 0.0)
        body = {"schema_version": ct.REQUEST_SCHEMA, "source": {"kind": "live", "name": "latency-benchmark (synthetic values)"}, "layout": ct.layout(svc.schema),
                "timestamps": [1_700_000_000 + 60 * i for i in range(T)], "values": v.tolist()}
        payload = len(json.dumps(body))
        post(url, body)
        rt, sv = [], []
        for _ in range(5):
            s = time.perf_counter()
            code, r = post(url, body)
            rt.append((time.perf_counter() - s) * 1000)
            sv.append(r["latency_ms"])
            assert code == 200
        res[str(T)] = {"windows": T - ct.WINDOW, "request_bytes": payload, "round_trip_ms_median": round(float(np.median(rt)), 1),
                       "server_ms_median": round(float(np.median(sv)), 1), "server_ms_per_window": round(float(np.median(sv)) / (T - ct.WINDOW), 2)}
    httpd.shutdown()
    out = {"service_load_s": round(load_s, 2), "segments": res, "torch_threads": int(os.environ.get("MODEL1_TORCH_THREADS", "1")),
           "machine": {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version()},
           "note": "synthetic values; single process, requests scored one at a time"}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
