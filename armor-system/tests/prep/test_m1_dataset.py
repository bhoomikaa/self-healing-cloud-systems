import copy
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

import _m1_helpers as H
from prep import m1_masks as mm, m1_split as ms, paths, preprocess as pp

try:
    import torch
    from prep import m1_dataset as md
except ImportError:                                            # pragma: no cover
    torch = None

NEEDS_TORCH = H.needs(torch is not None, "torch is not installed")
NEEDS_REAL = H.needs(torch is not None and H.CACHE_PRESENT and H.ARTIFACTS_PRESENT, "needs torch, the D1 feature cache and the M1-03 artifacts")


@NEEDS_TORCH
class CollateTests(unittest.TestCase):
    def item(self, case, cls, role="fit"):
        return {"inputs": torch.zeros(6, 2, 3), "role": role, "case_index": case, "fault_class": cls, "group_id": case,
                "fault_label_ambiguous": torch.tensor(False)}

    def test_optional_case_metadata_and_strings(self):
        b = md.collate_windows([self.item(5, 2), self.item(None, None), self.item(-1, -1)])
        self.assertEqual(b["inputs"].shape, (3, 6, 2, 3))
        self.assertEqual(b["role"], ["fit", "fit", "fit"])
        self.assertEqual(b["case_index"].tolist(), [5, -1, -1])
        self.assertEqual(b["fault_class"].tolist(), [2, -1, -1])
        self.assertEqual(b["case_index_present"].tolist(), [True, False, False])
        self.assertEqual(b["fault_class"].dtype, torch.int64)

    def test_empty_batch_raises(self):
        with self.assertRaises(ValueError):
            md.collate_windows([])

    def test_objective_table_is_leak_safe(self):
        for name, o in ms.OBJECTIVES.items():
            if o["gradient"]:
                self.assertEqual(o["roles"], ("fit",), name)
        self.assertEqual({n for n, o in ms.OBJECTIVES.items() if o["gradient"]}, {"s1_pretrain", "s2_fault_finetune"})


@NEEDS_REAL
class RealLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = md.Model1Data.load()
        from prep import loader
        cls.ts, cls.X, _ = loader.load_arrays()
        cls.schema = H.read_json(os.path.join(paths.ART, "inspection", "schema.json"))
        cls.groups = cls.schema["modality_index_ranges"]

    # --- counts and shapes ----------------------------------------------------------------------------
    def test_selection_sizes_match_the_approved_counts(self):
        D = self.D
        self.assertEqual(len(md.training_dataset(D, "s1_pretrain")), 1312)
        self.assertEqual(len(md.training_dataset(D, "s2_fault_finetune")), 523)      # 524 event windows minus 1 ambiguous
        self.assertEqual(len(md.evaluation_dataset(D, "s3_calibration_reference")), 329)
        self.assertEqual(len(md.evaluation_dataset(D, "calibration_diagnostics")), 78)
        self.assertEqual(len(md.evaluation_dataset(D, "validation_all")), 869)
        self.assertEqual(len(md.evaluation_dataset(D, "test_structural")), 2868)
        for role, exp in (("fit", (524, 1312, 1032)), ("calibration", (78, 329, 152)), ("validation", (150, 411, 308))):
            got = tuple(len(D.selection([role], [c])) for c in ("event_associated", "normal_reference", "ignore"))
            self.assertEqual(got, exp, role)

    def test_item_fields_shapes_dtypes(self):
        it = md.training_dataset(self.D, "s1_pretrain")[0]
        shapes = {"inputs": (6, 46, 130), "target": (46, 130), "input_mask": (6, 46, 130), "target_mask": (46, 130), "applicable_entry": (46, 130),
                  "observed_entry": (6, 46, 130), "structural_absent": (46, 3), "observed_all_zero_not_structural": (6, 46, 3), "timestamps": (7,)}
        for k, s in shapes.items():
            self.assertEqual(tuple(it[k].shape), s, k)
        self.assertEqual(it["inputs"].dtype, torch.float32)
        self.assertEqual(it["input_mask"].dtype, torch.bool)
        self.assertEqual(it["timestamps"].dtype, torch.int64)
        self.assertEqual((it["role"], it["category"]), ("fit", "normal_reference"))
        self.assertEqual((it["case_index"], it["fault_class"]), (-1, -1))
        self.assertTrue(torch.isfinite(it["inputs"]).all() and torch.isfinite(it["target"]).all())
        self.assertTrue((torch.diff(it["timestamps"]) == 60).all())

    def test_batch_shapes(self):
        b = next(iter(md.training_loader(md.training_dataset(self.D, "s1_pretrain"), batch_size=8)))
        self.assertEqual(tuple(b["inputs"].shape), (8, 6, 46, 130))
        self.assertEqual(tuple(b["target"].shape), (8, 46, 130))
        self.assertEqual(tuple(b["timestamps"].shape), (8, 7))
        self.assertEqual(set(b["role"]), {"fit"})

    # --- masks ----------------------------------------------------------------------------------------
    def test_masked_entries_are_zero_and_applicable_count_is_3152(self):
        D = self.D
        self.assertEqual(int(D.applicable.sum()), 3152)
        it = md.evaluation_dataset(D, "validation_all")[3]
        self.assertTrue((it["inputs"][~it["input_mask"]] == 0).all())
        self.assertTrue((it["target"][~it["target_mask"]] == 0).all())
        self.assertTrue((it["input_mask"] == it["applicable_entry"][None]).all())   # D1: observed_entry is all True
        self.assertTrue(it["observed_entry"].all())

    def test_unavailable_entries_are_excluded_before_model_consumption(self):
        D = self.D
        obs = np.ones(D.Xt.shape, bool)
        row = int(D.selection(["fit"], ["normal_reference"])[0])
        s = int(D.table.start[row])
        obs[s + 2, 10, 5] = False                                 # one minute-2 entry of a pod channel is not delivered
        obs[s + 6, 10, 5] = False                                 # and the target entry
        D2 = md.Model1Data(D.Xt, D.ts, D.table, D.cases, D.manifest, D.scaler, D.applicable, (D.structural_absent, D.observed_all_zero), D.schema, obs, D.cfg, D.boundaries)
        it = md.Model1WindowDataset(D2, [row])[0]
        self.assertFalse(it["input_mask"][2, 10, 5])
        self.assertFalse(it["target_mask"][10, 5])
        self.assertEqual(float(it["inputs"][2, 10, 5]), 0.0)
        self.assertEqual(float(it["target"][10, 5]), 0.0)
        self.assertTrue(it["input_mask"][1, 10, 5])

    def test_non_zero_value_at_an_unavailable_entry_is_rejected_at_construction(self):
        Xbad = self.D.Xt.copy()
        i, j = np.argwhere(~self.D.applicable)[0]
        Xbad[0, i, j] = 1.0
        with self.assertRaises(mm.MaskError):
            md.Model1Data(Xbad, self.D.ts, self.D.table, self.D.cases, self.D.manifest, self.D.scaler, self.D.applicable,
                          (self.D.structural_absent, self.D.observed_all_zero), self.D.schema, None, self.D.cfg, self.D.boundaries)

    def test_malformed_masks_are_rejected(self):
        D = self.D
        for bad_app in (D.applicable.astype(np.uint8), D.applicable[:, :100]):
            with self.assertRaises(mm.MaskError):
                md.Model1Data(D.Xt, D.ts, D.table, D.cases, D.manifest, D.scaler, bad_app, (D.structural_absent, D.observed_all_zero), D.schema, None, D.cfg, D.boundaries)
        with self.assertRaises(mm.MaskError):
            md.Model1Data(D.Xt, D.ts, D.table, D.cases, D.manifest, D.scaler, D.applicable, (D.structural_absent, D.observed_all_zero), D.schema, np.ones(D.Xt.shape, np.uint8), D.cfg, D.boundaries)

    # --- equivalence with the existing preprocessing -------------------------------------------------
    def test_items_equal_prepare_window_where_available(self):
        D = self.D
        sc = ms.scaler_from_arrays({k: np.load(os.path.join(paths.ART, "preprocessing", f"{ms.SCALER_NAME}.npz"))[k] for k in ("mean", "scale", "constant", "absent_block", "absent_entry")} | {"n_fit_samples": np.array(2880)})
        ds = md.evaluation_dataset(D, "validation_all")
        for i in (0, 100, 400, len(ds) - 1):
            it = ds[i]
            ref = pp.prepare_window(self.X, self.ts, sc, self.groups, int(it["start_index"]))
            m = it["input_mask"].numpy()
            np.testing.assert_array_equal(it["inputs"].numpy()[m], ref["inputs"][m])
            np.testing.assert_array_equal(it["inputs"].numpy()[~m], 0)
            np.testing.assert_array_equal(it["target"].numpy()[it["target_mask"].numpy()], ref["target"][it["target_mask"].numpy()])
            np.testing.assert_array_equal(it["timestamps"].numpy(), ref["timestamps"])
            np.testing.assert_array_equal(it["structural_absent"].numpy(), ref["structural_absent"].astype(bool))
            np.testing.assert_array_equal(it["observed_all_zero_not_structural"].numpy(), ref["observed_all_zero_not_structural"].astype(bool))

    # --- labels ---------------------------------------------------------------------------------------
    def test_fault_labels_are_consecutive_0_4_and_match_the_case_table(self):
        D = self.D
        ds = md.training_dataset(D, "s2_fault_finetune")
        seen = set()
        for i in range(len(ds)):
            it = ds[i]
            seen.add(it["fault_class"])
            self.assertIn(it["fault_class"], range(5))
            self.assertEqual(it["fault_class"], int(D.cases["class_id"][it["case_index"]]))
            t5 = int(it["timestamps"][5])
            c = int(D.cases["timestamp"][it["case_index"]])
            self.assertTrue(c - 59 <= t5 < c + 300)                # event_associated: t5 inside the case window
            self.assertEqual(it["category"], "event_associated")
            self.assertFalse(bool(it["fault_label_ambiguous"]))
        self.assertEqual(seen, {0, 1, 2, 3, 4})
        self.assertEqual(ms.raw_to_model_map(self.schema), {0: 0, 1: 1, 3: 2, 4: 3, 5: 4})

    def test_exclusion_reason_on_every_unlabelled_event_window(self):
        D = self.D
        t = D.table
        unlabelled = (t.category == "event_associated") & (t.fault_class == -1)
        self.assertEqual(int(unlabelled.sum()), 1)
        self.assertTrue(all(r == "overlapping_cases_different_classes" for r in t.exclusion_reason[unlabelled]))
        row = int(np.nonzero(t.start == 2550)[0][0])
        it = md.Model1WindowDataset(D, [row])[0]
        self.assertEqual((it["category"], it["fault_class"], it["case_index"]), ("event_associated", -1, -1))
        self.assertEqual(it["fault_label_exclusion_reason"], "overlapping_cases_different_classes")
        self.assertTrue(bool(it["fault_label_ambiguous"]))
        # every other window carries null
        others = np.nonzero(t.start != 2550)[0]
        self.assertTrue(all(r is None for r in t.exclusion_reason[others]))
        self.assertEqual(sum(1 for r in t.exclusion_reason if r is not None), 1)
        # records of all kinds expose the field
        for objective in ("s1_pretrain", "s2_fault_finetune"):
            self.assertIsNone(md.training_dataset(D, objective)[0]["fault_label_exclusion_reason"])
        batch = md.collate_windows([it, md.training_dataset(D, "s1_pretrain")[0]])
        self.assertEqual(batch["fault_label_exclusion_reason"], ["overlapping_cases_different_classes", None])

    def test_the_excluded_window_is_not_in_the_fault_objective_and_counts_are_preserved(self):
        D = self.D
        self.assertEqual(len(D.selection(["fit"], ["event_associated"])), 524)
        s2 = md.training_dataset(D, "s2_fault_finetune")
        self.assertEqual(len(s2), 523)
        self.assertNotIn(2550, set(int(x) for x in D.table.start[s2.rows]))
        self.assertEqual(len(D.selection(["validation"], ["event_associated"], drop_ambiguous=True)), 150)

    def test_non_event_windows_have_no_fault_label(self):
        ds = md.training_dataset(self.D, "s1_pretrain")
        for i in range(0, len(ds), 50):
            self.assertEqual((ds[i]["case_index"], ds[i]["fault_class"]), (-1, -1))

    def test_mixed_batch_with_and_without_case_metadata_collates(self):
        D = self.D
        a = md.training_dataset(D, "s1_pretrain")[0]
        b = md.training_dataset(D, "s2_fault_finetune")[0]
        batch = md.collate_windows([a, b])
        self.assertEqual(batch["fault_class_present"].tolist(), [False, True])
        self.assertEqual(batch["category"], ["normal_reference", "event_associated"])

    # --- leakage --------------------------------------------------------------------------------------
    def test_calibration_validation_test_never_enter_training_batches(self):
        D = self.D
        non_fit = set()
        for r in ("calibration", "validation", "test", "embargo"):
            non_fit |= set(np.unique(D.table.T[D.table.role == r]).tolist())
        for objective in ("s1_pretrain", "s2_fault_finetune"):
            ds = md.training_dataset(D, objective)
            n = 0
            for b in md.training_loader(ds, batch_size=128, seed=3):
                self.assertEqual(set(b["role"]), {"fit"})
                self.assertFalse(non_fit & set(b["timestamps"].flatten().tolist()), "a non-fit timestamp reached a training batch")
                self.assertTrue((b["role_id"] == md.ROLE_IDS["fit"]).all())
                n += len(b["role"])
            self.assertEqual(n, len(ds))

    def test_training_iterator_rejects_everything_else(self):
        D = self.D
        for objective in ("s3_calibration_reference", "calibration_diagnostics", "validation_all", "validation_fault_eval", "s1_early_stopping", "test_structural"):
            with self.assertRaises(md.LeakageError):
                md.training_dataset(D, objective)
        with self.assertRaises(KeyError):
            md.training_dataset(D, "not_an_objective")
        for objective in ("s1_pretrain", "s2_fault_finetune"):
            with self.assertRaises(ValueError):
                md.evaluation_dataset(D, objective)
        with self.assertRaises(md.LeakageError):
            md.training_loader(md.evaluation_dataset(D, "s3_calibration_reference"))
        for role in ("calibration", "validation", "test"):
            with self.assertRaises(md.LeakageError):
                md.Model1WindowDataset(D, D.selection([role]), "x", allowed_roles=("fit",))

    def test_embargo_windows_are_never_yielded(self):
        D = self.D
        t = copy.copy(D.table)
        t.role = t.role.copy()
        t.role[int(D.selection(["fit"])[0])] = "embargo"
        D2 = md.Model1Data(D.Xt, D.ts, t, D.cases, D.manifest, D.scaler, D.applicable, (D.structural_absent, D.observed_all_zero), D.schema, None, D.cfg, D.boundaries)
        with self.assertRaises(md.LeakageError):
            md.Model1WindowDataset(D2, np.nonzero(t.role == "embargo")[0])

    def test_a_window_outside_its_region_is_refused(self):
        D = self.D
        t = copy.copy(D.table)
        t.role = t.role.copy()
        k = int(D.selection(["fit"])[0])
        t.role[k] = "calibration"                                  # an early window claimed by the calibration region
        D2 = md.Model1Data(D.Xt, D.ts, t, D.cases, D.manifest, D.scaler, D.applicable, (D.structural_absent, D.observed_all_zero), D.schema, None, D.cfg, D.boundaries)
        with self.assertRaises(ms.ManifestInvariantError):
            md.Model1WindowDataset(D2, [k])

    def test_fit_and_calibration_share_no_timestamp_in_the_dataset_views(self):
        D = self.D
        fit = md.training_dataset(D, "s1_pretrain")
        cal = md.evaluation_dataset(D, "s3_calibration_reference")
        ft = np.unique(np.concatenate([fit[i]["timestamps"].numpy() for i in range(0, len(fit), 7)]))
        ct = np.unique(np.concatenate([cal[i]["timestamps"].numpy() for i in range(0, len(cal), 3)]))
        self.assertEqual(len(np.intersect1d(ft, ct)), 0)
        self.assertLess(ft.max(), 1651637521 - 360)
        self.assertGreaterEqual(ct.min(), 1651637521)

    def test_scaler_fit_set_is_the_fitting_samples(self):
        D = self.D
        z = np.load(os.path.join(paths.ART, "preprocessing", f"{ms.SCALER_NAME}.npz"))
        fit_rows = np.unique(D.table.T[D.table.role == "fit"])
        # every fitting window timestamp lies in the scaler's fit set; no calibration/validation/test timestamp does
        self.assertTrue(np.isin(fit_rows, z["fit_timestamps"]).all())
        for r in ("calibration", "validation", "test"):
            self.assertFalse(np.isin(np.unique(D.table.T[D.table.role == r]), z["fit_timestamps"]).any(), r)

    # --- determinism ----------------------------------------------------------------------------------
    def test_loader_order_is_deterministic_for_a_seed(self):
        ds = md.training_dataset(self.D, "s1_pretrain")

        def order(seed):
            return [int(s) for b in md.training_loader(ds, batch_size=64, seed=seed) for s in b["start_index"]]

        self.assertEqual(order(0), order(0))
        self.assertNotEqual(order(0), order(1))
        self.assertEqual(sorted(order(0)), sorted(order(5)))


@NEEDS_REAL
class LoudFailureTests(unittest.TestCase):
    """Tampered artifacts or data must stop `Model1Data.load`."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="m1_art_")
        self.root = os.path.join(self.tmp, "d1")
        shutil.copytree(paths.ART, self.root, ignore=shutil.ignore_patterns("eda", "figures", "evidence", "quality"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def manifest(self):
        return os.path.join(self.root, "splits", f"{ms.MANIFEST_NAME}.json")

    def rewrite(self, mutate, rehash=True):
        m = H.read_json(self.manifest())
        mutate(m)
        if rehash:
            m["content_sha256"] = ms.manifest_content_hash(m)
        with open(self.manifest(), "w") as fh:
            fh.write(ms.dumps_manifest(m))

    def load(self, **kw):
        return md.Model1Data.load(art_root=self.root, **kw)

    def test_untouched_copy_loads(self):
        self.assertEqual(len(self.load().table.start), 3427 + 869 + 2868)

    def test_hand_edited_manifest_is_rejected(self):
        self.rewrite(lambda m: m["counts"]["roles"]["fit"].update(windows=1), rehash=False)
        with self.assertRaises(ms.ManifestInvariantError):
            self.load()

    def test_unapproved_manifest_is_rejected(self):
        self.rewrite(lambda m: m.update(status="draft"))
        with self.assertRaises(ms.ManifestInvariantError):
            self.load()

    def test_manifest_window_list_that_disagrees_with_the_data_is_rejected(self):
        self.rewrite(lambda m: m["windows"]["calibration"]["start_index"]["normal_reference"].pop())
        with self.assertRaises(ms.ManifestInvariantError):
            self.load()

    def test_scaler_file_that_differs_from_the_manifest_is_rejected(self):
        p = os.path.join(self.root, "preprocessing", f"{ms.SCALER_NAME}.npz")
        z = np.load(p)
        arrays = {k: z[k].copy() for k in z.files}
        arrays["mean"][0, 0] += 0.5
        with open(p, "wb") as fh:
            fh.write(ms.npz_bytes(arrays))
        with self.assertRaises(ms.ManifestInvariantError):
            self.load()

    def test_forged_scaler_values_with_a_rehashed_manifest_are_rejected(self):
        p = os.path.join(self.root, "preprocessing", f"{ms.SCALER_NAME}.npz")
        z = np.load(p)
        arrays = {k: z[k].copy() for k in z.files}
        arrays["mean"][10, 3] += 5.0                              # e.g. a scaler fit on all samples with fit_timestamps forged
        data = ms.npz_bytes(arrays)
        with open(p, "wb") as fh:
            fh.write(data)
        self.rewrite(lambda m: m["fit_only_scaler"].update(file_sha256=ms.sha256_bytes(data)))
        with self.assertRaises(ms.ManifestInvariantError):
            self.load()

    def test_scaler_fit_on_calibration_samples_is_rejected(self):
        p = os.path.join(self.root, "preprocessing", f"{ms.SCALER_NAME}.npz")
        z = np.load(p)
        arrays = {k: z[k].copy() for k in z.files}
        arrays["fit_timestamps"][-1] = 1651637521 + 600           # claims a calibration sample was used
        data = ms.npz_bytes(arrays)
        with open(p, "wb") as fh:
            fh.write(data)
        self.rewrite(lambda m: m["fit_only_scaler"].update(file_sha256=ms.sha256_bytes(data)))
        with self.assertRaises(ms.ManifestInvariantError):
            self.load()

    def test_a_different_cutoff_is_rejected_even_with_a_consistent_manifest_hash(self):
        self.rewrite(lambda m: m["protocol"].update(Tc=m["protocol"]["Tc"] + 120000))
        with self.assertRaises(ms.ManifestInvariantError):
            self.load()

    def test_feature_cache_that_differs_from_the_manifest_is_rejected(self):
        self.rewrite(lambda m: m["inputs"].update(feature_cache_content_sha256="0" * 64))
        with self.assertRaises(ms.ManifestInvariantError):
            self.load()

    def test_invalid_schema_is_rejected(self):
        p = os.path.join(self.root, "inspection", "schema.json")
        s = H.read_json(p)
        s["tensor_shape_per_sample"] = [46, 129]
        with open(p, "w") as fh:
            json.dump(s, fh)
        with self.assertRaises((mm.SchemaError, ms.ManifestInvariantError)):
            self.load()

    def test_data_with_a_value_outside_the_declared_mask_is_rejected(self):
        d = np.load(paths.CACHE_FILE)
        X = d["X"].copy()
        sc = np.load(os.path.join(self.root, "preprocessing", f"{ms.SCALER_NAME}.npz"))
        schema = H.read_json(os.path.join(self.root, "inspection", "schema.json"))
        app, _, _ = mm.applicable_entry_mask(schema, sc["absent_entry"])
        i, j = np.argwhere(~app)[0]
        X[5000, i, j] = 0.25
        cache = os.path.join(self.tmp, "bad_cache.npz")
        np.savez(cache, timestamps=d["timestamps"], X=X, is_test_file=d["is_test_file"])
        with self.assertRaises(mm.MaskError):
            self.load(cache=cache, verify_cache_hash=False)
        with self.assertRaises(ms.ManifestInvariantError):         # and the hash check catches it first by default
            self.load(cache=cache)


if __name__ == "__main__":
    unittest.main()
