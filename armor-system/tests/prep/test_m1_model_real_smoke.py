"""Real ARMOR D1 CPU forward/backward smoke test of the Model 1 core (Linear 298-50). Fitting batches only, no optimizer step."""
import unittest

import _m1_model_helpers as MH  # noqa: F401
import torch

import _m1_helpers as H
from model1_transformer import losses as L, smoke_forward_backward as sm
from model1_transformer.model import Model1
from model1_transformer.spec import AdapterSpec
from prep import m1_dataset as md

NEEDS_REAL = H.needs(H.CACHE_PRESENT and H.ARTIFACTS_PRESENT, "needs the D1 feature cache and the M1-03 artifacts")


@NEEDS_REAL
class RealSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = md.Model1Data.load()
        cls.r = sm.run_smoke(cls.D)

    def test_batches_are_fitting_windows_of_the_right_objectives(self):
        b = self.r["batches"]
        self.assertEqual((b["s1"]["roles"], b["s1"]["categories"]), (["fit"], ["normal_reference"]))
        self.assertEqual((b["s2"]["roles"], b["s2"]["categories"]), (["fit"], ["event_associated"]))
        for k in ("s1", "s2"):
            self.assertEqual(b[k]["inputs_shape"], [8, 6, 46, 130])
            self.assertEqual(b[k]["target_shape"], [8, 46, 130])
            self.assertTrue(b[k]["inputs_finite"])
        self.assertEqual(len(md.training_dataset(self.D, "s1_pretrain")), 1312)
        self.assertEqual(len(md.training_dataset(self.D, "s2_fault_finetune")), 523)

    def test_no_validation_calibration_or_test_batch_can_be_requested(self):
        for obj in ("s1_early_stopping", "s3_calibration_reference", "calibration_diagnostics", "validation_fault_eval", "validation_all"):
            with self.assertRaises(md.LeakageError):
                md.training_dataset(self.D, obj)

    def test_forward_shapes_and_finite_values(self):
        for tag in ("eval_s1", "eval_s2"):
            self.assertEqual(self.r[tag]["logits_shape"], [8, 5])
            self.assertEqual(self.r[tag]["prediction_shape"], [8, 46, 130])
            self.assertEqual(self.r[tag]["hidden_shape"], [8, 6, 46, 64])
            self.assertTrue(self.r[tag]["prediction_outside_applicable_all_zero"])

    def test_losses_are_finite_and_gradients_are_finite_on_the_objective_path(self):
        s1, s2 = self.r["s1_masked_mse"], self.r["s2_weighted_ce"]
        self.assertTrue(0 < s1["loss"] < 1e3 and 0 < s2["loss"] < 1e3)
        self.assertEqual((s1["parameters_with_gradient"], s1["parameters_without_gradient"]), (55, 4))       # fault head is off the S1 path
        self.assertEqual((s2["parameters_with_gradient"], s2["parameters_without_gradient"]), (55, 4))       # next-step head is off the S2 path
        self.assertEqual(s1["without_gradient_prefix"], "fault_head.")
        self.assertEqual(s2["without_gradient_prefix"], "next_step_head.")
        self.assertGreater(s1["global_grad_norm"], 0)
        self.assertGreater(s2["global_grad_norm"], 0)
        self.assertEqual(self.r["optimizer_steps_taken"], 0)

    def test_unlabelled_s1_windows_are_rejected_by_the_fault_loss(self):
        self.assertTrue(self.r["s1_unlabelled_labels_rejected_by_fault_loss"])

    def test_s2_labels_are_real_classes_and_class_weights_use_the_88_fit_cases(self):
        self.assertTrue(all(0 <= y < 5 for y in self.r["s2_weighted_ce"]["labels"]))
        w = L.fit_class_weights(self.D.manifest)
        self.assertEqual([round(float(x), 6) for x in w], self.r["class_weights"])
        self.assertEqual(self.D.manifest["counts"]["roles"]["fit"]["cases"], 88)

    def test_parameter_count_and_spec_come_from_the_real_schema(self):
        spec = AdapterSpec.from_d1(self.D.schema, self.D.applicable)
        self.assertEqual(self.r["parameters"], 168200)
        self.assertEqual(self.r["spec_fingerprint"], spec.fingerprint())
        self.assertEqual(spec.n_nodes, 46)
        self.assertEqual(int((spec.node_type == 1).sum()), 6)

    def test_class_order_matches_the_manifest_labels_and_the_fit_case_counts(self):
        names = ["Container Hardware", "Container Network", "Node CPU", "Node Disk", "Node Memory"]
        self.assertEqual(self.D.manifest["label_map"]["classes"], names)
        ds = md.training_dataset(self.D, "s2_fault_finetune")
        t = self.D.table
        cls, case = t.fault_class[ds.rows], t.case_index[ds.rows]
        self.assertTrue((cls == self.D.cases["class_id"][case]).all())
        counts = [len(set(case[cls == k].tolist())) for k in range(5)]
        self.assertEqual(counts, self.D.manifest["counts"]["roles"]["fit"]["cases_by_class"])
        self.assertEqual(L.fit_class_weights(self.D.manifest).tolist(), L.inverse_frequency_weights(counts).tolist())

    def test_no_optimizer_and_no_evaluation_data_in_the_smoke_path(self):
        import re
        for mod in (sm, ):
            with open(mod.__file__) as fh:
                src = fh.read()
            run = src[src.index("def run_smoke"):src.index("def hashes_of")]
            self.assertIsNone(re.search(r"torch\.optim|optim\.|\.step\(|evaluation_dataset|evaluation_loader", run))
        self.assertEqual(self.r["optimizer_steps_taken"], 0)

    def test_deterministic_across_independent_runs(self):
        again = sm.run_smoke(self.D)
        self.assertEqual(sm.hashes_of(self.r), sm.hashes_of(again))
        self.assertEqual(self.r["initial_state_sha256"], sm.state_digest(Model1.build(AdapterSpec.from_d1(self.D.schema, self.D.applicable), seed=0)))


if __name__ == "__main__":
    unittest.main()
