"""Model 1 v2 serving: contract validation, equality with the frozen data layer, HTTP behaviour and the end-to-end replay.

Contract tests use synthetic minutes (committed artefacts only). Equality and replay tests need the local D1 feature cache and the
frozen checkpoints and are skipped without them. Only validation-period minutes are replayed; no test-period minute is used.
"""
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

ARMOR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in ("data", "models", os.path.join("evaluation", "detection")):
    if os.path.join(ARMOR, p) not in sys.path:
        sys.path.insert(0, os.path.join(ARMOR, p))

import numpy as np  # noqa: E402

from model1_v2.serving import contract as ct  # noqa: E402
from model1_v2.serving.service import load_artifacts  # noqa: E402
from prep import paths  # noqa: E402

HAVE_CKPT = os.path.exists(os.path.expanduser("~/298A_project/checkpoints/model1_v1/seed_4/s2_best.pt"))
HAVE_DATA = os.path.exists(paths.CACHE_FILE)
ART = load_artifacts()


def body(T=8, seed=0, kind="replay"):
    rng = np.random.default_rng(seed)
    v = np.where(ART["applicable"][None], rng.normal(size=(T, 46, 130)), 0.0)
    return {"schema_version": ct.REQUEST_SCHEMA, "source": {"kind": kind, "name": "synthetic"}, "layout": ct.layout(ART["schema"]),
            "timestamps": [1_700_000_000 + 60 * i for i in range(T)], "values": v.tolist()}


class ContractTests(unittest.TestCase):
    def check(self, b, code):
        with self.assertRaises(ct.RequestError) as cm:
            ct.validate(b, ART["schema"], ART["applicable"])
        self.assertEqual(cm.exception.code, code)

    def test_valid_request_parses_and_replay_is_flagged(self):
        seg = ct.validate(body(), ART["schema"], ART["applicable"])
        self.assertEqual(seg.values.shape, (8, 46, 130))
        self.assertTrue(seg.observed.all())
        self.assertEqual([w["code"] for w in seg.warnings], ["replay_data"])
        self.assertEqual(ct.validate(body(kind="live"), ART["schema"], ART["applicable"]).warnings, [])

    def test_rejections(self):
        cases = []
        b = body(); b["schema_version"] = "x"; cases.append((b, "bad_schema_version"))
        b = body(); b["source"] = {"kind": "simulated", "name": "x"}; cases.append((b, "bad_source"))
        b = body(); b["layout"]["channel_order_sha256"] = "0" * 64; cases.append((b, "layout_mismatch"))
        b = body(); del b["layout"]; cases.append((b, "layout_mismatch"))
        b = body(); b["timestamps"][3] = 1.5; cases.append((b, "bad_timestamps"))
        b = body(T=6); cases.append((b, "bad_length"))
        b = body(); b["timestamps"][5] += 60; b["timestamps"][6] += 60; b["timestamps"][7] += 60; cases.append((b, "not_consecutive"))
        b = body(); b["values"] = b["values"][:-1]; cases.append((b, "bad_shape"))
        b = body(); b["values"][0][0] = b["values"][0][0][:-1]; cases.append((b, "malformed_array"))
        b = body(); b["values"][2][0][0] = float("nan"); cases.append((b, "non_finite"))
        b = body(); del b["values"]; cases.append((b, "missing_values"))
        b = body(); b["observed"] = np.ones((8, 46, 130)).tolist(); cases.append((b, "bad_observed"))
        n, c = map(int, np.argwhere(~ART["applicable"])[0])
        b = body(); b["values"][1][n][c] = 3.0; cases.append((b, "value_at_structurally_absent_entry"))
        b = body(); b["previous"] = {"target_ts": 1, "detection_score": 2.0}; cases.append((b, "bad_previous"))
        b = body(); b["previous"] = {"target_ts": 1, "detection_score": 0.5}; cases.append((b, "previous_not_adjacent"))
        cases.append(([], "not_an_object"))
        for b, code in cases:
            with self.subTest(code=code):
                self.check(b, code)

    def test_unobserved_absent_entries_and_adjacent_previous_are_accepted(self):
        b = body()
        n, c = map(int, np.argwhere(~ART["applicable"])[0])
        b["values"][1][n][c] = 3.0
        obs = np.ones((8, 46, 130), bool)
        obs[1, n, c] = False
        b["observed"] = obs.tolist()
        b["previous"] = {"target_ts": b["timestamps"][6] - 60, "detection_score": 0.25}
        seg = ct.validate(b, ART["schema"], ART["applicable"])
        self.assertFalse(seg.observed[1, n, c])


@unittest.skipUnless(HAVE_CKPT, "frozen checkpoints not present")
class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from model1_v2.serving.service import Model1Service
        cls.svc = Model1Service.load()

    def test_prediction_shape_and_continuity_across_segments(self):
        b = body(T=20, seed=3, kind="live")
        whole = self.svc.handle(b)
        self.assertEqual(whole["n_windows"], 14)
        self.assertFalse(whole["replay"])
        first = dict(b, timestamps=b["timestamps"][:13], values=b["values"][:13])
        r1 = self.svc.handle(first)
        second = dict(b, timestamps=b["timestamps"][7:], values=b["values"][7:],
                      previous={"target_ts": r1["windows"][-1]["target_ts"], "detection_score": r1["windows"][-1]["detection_score"]})
        r2 = self.svc.handle(second)
        joined = r1["windows"] + r2["windows"]
        self.assertEqual([w["target_ts"] for w in joined], [w["target_ts"] for w in whole["windows"]])
        for a, w in zip(joined, whole["windows"]):                      # batch composition changes float summation order: ~1e-8
            self.assertAlmostEqual(a["detection_score"], w["detection_score"], delta=1e-6)
            self.assertAlmostEqual(a["rolled_score"], w["rolled_score"], delta=1e-6)
            self.assertEqual(a["alert"], w["alert"])
            self.assertAlmostEqual(sum(a["fault_probs"].values()), 1.0, places=9)

    def test_vm_metrics_hidden_warning(self):
        b = body(seed=1, kind="live")
        obs = np.ones((8, 46, 130), bool)
        vm, metric = np.nonzero(self.svc.vm)[0], np.nonzero(self.svc.metric)[0]
        obs[:, vm[:, None], metric[None, :]] = False
        b["observed"] = obs.tolist()
        codes = [w["code"] for w in self.svc.handle(b)["warnings"]]
        self.assertIn("vm_metrics_unobserved", codes)

    @unittest.skipUnless(HAVE_DATA, "D1 feature cache not present")
    def test_windows_equal_the_frozen_data_layer_on_validation(self):
        from model1_v2.serving.replay_demo import validation_replay
        from prep import m1_dataset as md
        ts, X, _ = validation_replay()
        seg = ct.validate({"schema_version": ct.REQUEST_SCHEMA, "source": {"kind": "replay", "name": "t"}, "layout": ct.layout(ART["schema"]),
                           "timestamps": [int(t) for t in ts[:200]], "values": X[:200].tolist()}, ART["schema"], ART["applicable"])
        inputs, mask, target_ts, _ = self.svc.windows(seg)
        D = md.Model1Data.load()
        ds = md.evaluation_dataset(D, "validation_all")
        by_target = {int(D.table.T[r, -1]): i for i, r in enumerate(ds.rows)}
        for k, t in enumerate(target_ts):
            item = ds[by_target[int(t)]]
            self.assertTrue(np.array_equal(item["inputs"].numpy(), inputs[k]), int(t))
            self.assertTrue(np.array_equal(item["input_mask"].numpy(), mask[k]), int(t))


@unittest.skipUnless(HAVE_CKPT, "frozen checkpoints not present")
class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from model1_v2.serving.server import serve
        from model1_v2.serving.service import Model1Service
        cls.svc = Model1Service.load()
        cls.httpd = serve(cls.svc, "127.0.0.1", 0)
        cls.url = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def req(self, path, data=None, method=None, headers=None):
        r = urllib.request.Request(self.url + path, data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(r, timeout=300) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_health_and_metadata(self):
        self.assertEqual(self.req("/healthz")[1]["status"], "ok")
        code, m = self.req("/v1/metadata")
        self.assertEqual(code, 200)
        self.assertEqual(m["layout"], ct.layout(ART["schema"]))
        self.assertEqual(len(m["model"]["checkpoint_sha256"]), 5)
        self.assertEqual(len(m["node_order"]), 46)

    def test_predict_ok_and_errors(self):
        js = {"Content-Type": "application/json"}
        code, r = self.req("/v1/predict", json.dumps(body(kind="live")).encode(), "POST", js)
        self.assertEqual(code, 200)
        self.assertEqual(r["schema_version"], ct.RESPONSE_SCHEMA)
        self.assertEqual(r["n_windows"], 2)
        bad = body(); bad["timestamps"][4] += 1
        self.assertEqual(self.req("/v1/predict", json.dumps(bad).encode(), "POST", js), (400, {"error": {"code": "not_consecutive", "field": "timestamps",
                         "message": "timestamps must be strictly consecutive minutes (step 60 s, no gaps); split the stream at gaps"}}))
        self.assertEqual(self.req("/v1/predict", b"{not json", "POST", js)[1]["error"]["code"], "invalid_json")
        self.assertEqual(self.req("/v1/predict", b"{}", "POST", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.req("/nope")[0], 404)
        self.assertEqual(self.req("/v1/predict")[0], 405)
        self.assertEqual(self.req("/v1/predict", b"{}", "PUT", js)[0], 405)

    def test_oversized_body_is_refused_before_reading(self):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=30)
        c.putrequest("POST", "/v1/predict")
        c.putheader("Content-Type", "application/json")
        c.putheader("Content-Length", str(10 ** 9))
        c.endheaders()
        r = c.getresponse()
        self.assertEqual(r.status, 413)
        c.close()

    def test_internal_errors_do_not_leak_tracebacks(self):
        orig = self.svc.handle
        self.svc.handle = lambda b: (_ for _ in ()).throw(RuntimeError("secret detail"))
        try:
            code, r = self.req("/v1/predict", json.dumps(body()).encode(), "POST", {"Content-Type": "application/json"})
        finally:
            self.svc.handle = orig
        self.assertEqual(code, 500)
        self.assertNotIn("secret", json.dumps(r))

    @unittest.skipUnless(HAVE_DATA, "D1 feature cache not present")
    def test_end_to_end_replay_reproduces_the_committed_v2_validation_result(self):
        from model1_v2.serving import replay_demo
        with tempfile.TemporaryDirectory() as tmp:
            summary, inc = replay_demo.run(self.url, tmp, log=lambda *a: None)
        s = summary["scoring_against_documented_validation_incidents"]
        self.assertEqual((s["TP"], s["FP"], s["f1"]), (18, 6, 0.7347))
        self.assertEqual(len(inc), 22)
        self.assertTrue(summary["replay"] and all(i["source"]["kind"] == "replay" for i in inc))


if __name__ == "__main__":
    unittest.main()
