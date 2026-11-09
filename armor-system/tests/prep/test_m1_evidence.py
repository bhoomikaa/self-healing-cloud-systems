import copy
import json
import os
import shutil
import tempfile
import unittest

import _m1_helpers as H
from prep import m1_evidence as ev


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="m1_ev_")
        for rel in ev.SOURCE_FILES:
            p = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write(f"content of {rel}\n")
        manifest = {"counts": {"roles": {r: {"windows": 1, "cases": 1} for r in ("fit", "calibration", "validation", "test")}},
                    "inputs": {"feature_cache_content_sha256": "f" * 64, "outer_split_manifest": {"content_sha256": "o" * 64}, "preprocessing_config_sha256": "p" * 64}}
        manifest["content_sha256"] = ev.manifest_content_hash(manifest)
        with open(os.path.join(self.root, ev.MANIFEST_PATH), "w") as fh:
            json.dump(manifest, fh)
        self.doc = ev.build(
            self.root, d1_verification={"problems": [], "inventory_files_checked": 9}, cache_rebuild={"rebuilt_equals_manifest": True},
            reproducibility={"runs": 2, "manifest_identical": True, "scaler_identical": True, "equal_to_committed": True},
            tests={"ran": 5, "failures": 0, "errors": 0, "skipped": 0, "suites": [{"ran": 5}]}, command="cmd", environment={}, git={"head": "a" * 40, "branch": "b", "clean": False})

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_valid_evidence_verifies(self):
        self.assertEqual(ev.verify(self.doc, self.root), [])

    def test_no_circular_commit_or_self_hash(self):
        cr = self.doc["commit_reference"]
        self.assertIsNone(cr["final_merge_commit"])
        self.assertIn("final_merge_commit_recorded_in", cr)
        self.assertEqual(cr["evidence_base_commit"], "a" * 40)
        # the source set never contains the evidence file itself
        self.assertNotIn(ev.EVIDENCE_PATH, ev.SOURCE_FILES)
        self.assertNotIn(ev.EVIDENCE_PATH, [r["path"] for r in cr["source_files"]])
        # and does not depend on HEAD: changing the recorded base commit does not change the source-set hash
        other = ev.build(self.root, d1_verification={}, cache_rebuild={}, reproducibility={}, tests={}, command="", environment={}, git={"head": "zzz", "branch": "x", "clean": True})
        self.assertEqual(other["commit_reference"]["source_set_sha256"], cr["source_set_sha256"])

    def test_changed_source_file_is_detected(self):
        with open(os.path.join(self.root, ev.SOURCE_FILES[3]), "a") as fh:
            fh.write("edit\n")
        bad = ev.verify(self.doc, self.root)
        self.assertTrue(any("source_set_sha256" in b for b in bad))
        self.assertTrue(any(ev.SOURCE_FILES[3] in b for b in bad))

    def test_missing_source_file_is_detected(self):
        os.remove(os.path.join(self.root, ev.SOURCE_FILES[0]))
        self.assertTrue(any("missing" in b for b in ev.verify(self.doc, self.root)))
        with self.assertRaises(ev.EvidenceError):
            ev.source_set(self.root)

    def test_tampered_hashes_and_results_are_detected(self):
        for mutate, text in ((lambda d: d["hashes"].update(fit_only_scaler_file_sha256="0" * 64), "fit_only_scaler"),
                             (lambda d: d["tests"].update(skipped=1), "skips"),
                             (lambda d: d["reproducibility"].update(manifest_identical=False), "reproducibility"),
                             (lambda d: d["feature_cache_rebuild"].update(rebuilt_equals_manifest=False), "cache rebuild"),
                             (lambda d: d["d1_verification"].update(problems=["x"]), "D1"),
                             (lambda d: d.update(evidence_version="9"), "version"),
                             (lambda d: d["tests"].pop("ran"), "positive count"),
                             (lambda d: d["tests"].pop("skipped"), "positive count"),
                             (lambda d: d["tests"].update(suites=[{"ran": 0}]), "ran no tests"),
                             (lambda d: d["d1_verification"].pop("inventory_files_checked"), "D1"),
                             (lambda d: d["commit_reference"].update(evidence_base_commit=""), "40-hex"),
                             (lambda d: d["hashes"].update(preprocessing_config_sha256="0" * 64), "preprocessing_config")):
            d = copy.deepcopy(self.doc)
            mutate(d)
            bad = ev.verify(d, self.root)
            self.assertTrue(any(text in b for b in bad), (text, bad))

    def test_scope_statement_keeps_298_44_evidence_separate(self):
        self.assertIn("298-48", self.doc["scope"])
        self.assertIn("not touched", self.doc["scope"])
        self.assertNotEqual(ev.EVIDENCE_PATH.rsplit("/", 1)[1], "evidence.json")


class RealTreeTests(unittest.TestCase):
    def test_every_listed_source_file_exists_in_the_repository(self):
        repo = os.path.abspath(os.path.join(H.paths.ROOT, ".."))
        rows, h = ev.source_set(repo)
        self.assertEqual(len(rows), len(ev.SOURCE_FILES))
        self.assertEqual(len(h), 64)


if __name__ == "__main__":
    unittest.main()
