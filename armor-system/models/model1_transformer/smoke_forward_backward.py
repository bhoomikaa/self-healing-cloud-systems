"""Model 1 core smoke test on real ARMOR D1 fitting batches (Linear 298-50). CPU only, no optimizer step, no training.

    .venv/bin/python -I armor-system/models/model1_transformer/smoke_forward_backward.py                     # run twice, compare hashes
    .venv/bin/python -I armor-system/models/model1_transformer/smoke_forward_backward.py --write-evidence    # + clean subprocess reruns, tests, integration, evidence file
    .venv/bin/python -I armor-system/models/model1_transformer/smoke_forward_backward.py --verify-evidence   # check the evidence against the tree and a fresh run

Only the two gradient objectives' fitting windows are requested (`training_dataset` rejects every other role): one S1
batch (normal-reference windows, masked next-step MSE) and one S2 batch (event-associated windows with a real fault
label, class-weighted cross-entropy). No validation, calibration or test window is ever turned into a batch.
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [os.path.join(REPO, "armor-system", "data"), os.path.join(REPO, "armor-system", "models")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model1_transformer import losses as L  # noqa: E402
from model1_transformer.model import Model1, enable_determinism, state_digest, tensor_digest  # noqa: E402
from model1_transformer.spec import AdapterSpec, ModelConfig  # noqa: E402
from prep import m1_dataset as md, m1_evidence as m1ev  # noqa: E402

EVIDENCE_REL = "armor-system/models/model1_transformer/results/core_v0/evidence_model1_core_v0.json"
EVIDENCE_NAME = "evidence_model1_core_v0"
COMMAND = ".venv/bin/python -I armor-system/models/model1_transformer/smoke_forward_backward.py --write-evidence"
INTEGRATION = [".venv/bin/python", "-I", "armor-system/data/m1_integration.py"]
SOURCE_FILES = tuple(f"armor-system/models/model1_transformer/{f}" for f in
                     ("__init__.py", "spec.py", "adapters.py", "core.py", "heads.py", "losses.py", "model.py", "smoke_forward_backward.py", "ARCHITECTURE.md")) + tuple(
    f"armor-system/tests/prep/{f}" for f in ("_m1_model_helpers.py", "test_m1_model_invariance.py", "test_m1_model_shapes.py", "test_m1_model_real_smoke.py"))
# Historical evidence that this milestone must leave byte-identical.
HISTORICAL = ("armor-system/data/artifacts/d1/evidence/evidence.json", "armor-system/data/artifacts/d1/evidence/evidence_m1_03.json",
              "armor-system/models/model1_xgboost_control/results/xgb_control_v1/evidence_xgb_control_v1.json")
MI_CLASSES = {"MI1ValueInvariance": "MI-1", "MI2Gradient": "MI-2", "MI3NotVacuous": "MI-3", "MI4NodeTypeEquivariance": "MI-4",
              "MI5Padding": "MI-5", "MI6MaskStability": "MI-6", "MI7AvailabilityHandling": "MI-7"}


class SmokeError(RuntimeError):
    pass


def _grad_report(model, expect_none_prefix):
    """All parameters on the objective's path must have a finite gradient; the others must have none."""
    got, none = 0, []
    sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            none.append(name)
            continue
        if not torch.isfinite(p.grad).all():
            raise SmokeError(f"non-finite gradient in {name}")
        got += 1
        sq += float((p.grad.double() ** 2).sum())
    expected = {n for n, _ in model.named_parameters() if n.startswith(expect_none_prefix)}
    if set(none) != expected:
        raise SmokeError(f"unexpected gradient coverage: parameters without gradient {sorted(none)}, expected exactly {sorted(expected)}")
    return {"parameters_with_gradient": got, "parameters_without_gradient": len(none), "without_gradient_prefix": expect_none_prefix,
            "global_grad_norm": round(sq ** 0.5, 6)}


def _batch_facts(b, kind):
    roles, cats = sorted(set(b["role"])), sorted(set(b["category"]))
    return {"kind": kind, "size": len(b["role"]), "roles": roles, "categories": cats, "inputs_shape": list(b["inputs"].shape),
            "target_shape": list(b["target"].shape), "inputs_finite": bool(torch.isfinite(b["inputs"]).all()),
            "effective_mask_coverage": round(float(b["input_mask"].float().mean()), 6), "inputs_sha256": tensor_digest(b["inputs"]),
            "start_index": [int(i) for i in b["start_index"]]}


def run_smoke(D, spec=None, seed=0, batch_size=8):
    """One S1 and one S2 fitting batch: eval forward (hashes), train-mode forward + backward per objective. No optimizer."""
    enable_determinism()
    spec = spec or AdapterSpec.from_d1(D.schema, D.applicable)
    weights = L.fit_class_weights(D.manifest)
    b1 = next(iter(md.training_loader(md.training_dataset(D, "s1_pretrain"), batch_size=batch_size, seed=seed)))
    b2 = next(iter(md.training_loader(md.training_dataset(D, "s2_fault_finetune"), batch_size=batch_size, seed=seed)))
    if set(b1["role"]) != {"fit"} or set(b2["role"]) != {"fit"}:
        raise SmokeError("a smoke batch holds a non-fitting window")
    if set(b1["category"]) != {"normal_reference"} or set(b2["category"]) != {"event_associated"}:
        raise SmokeError("S1/S2 batch categories are not the approved ones")
    if not (b2["fault_class_present"].all() and ((b2["fault_class"] >= 0) & (b2["fault_class"] < 5)).all()):
        raise SmokeError("an S2 window has no valid fault label")
    model = Model1.build(spec, ModelConfig(), seed=seed)
    out = {"seed": seed, "batch_size": batch_size, "parameters": model.n_parameters(), "initial_state_sha256": state_digest(model),
           "spec_fingerprint": spec.fingerprint(), "class_weights": [round(float(w), 6) for w in weights],
           "batches": {"s1": _batch_facts(b1, "s1_pretrain"), "s2": _batch_facts(b2, "s2_fault_finetune")}}
    fwd = lambda b: dict(inputs=b["inputs"], input_mask=b["input_mask"], applicable_entry=b["applicable_entry"])   # noqa: E731
    model.eval()
    with torch.no_grad():
        for tag, b in (("s1", b1), ("s2", b2)):
            o = model(**fwd(b), return_aux=True)
            B = b["inputs"].shape[0]
            if tuple(o["logits"].shape) != (B, 5) or tuple(o["prediction"].shape) != (B, 46, 130):
                raise SmokeError(f"{tag}: unexpected output shapes {tuple(o['logits'].shape)}, {tuple(o['prediction'].shape)}")
            if not all(torch.isfinite(v.float()).all() for v in o.values()):
                raise SmokeError(f"{tag}: non-finite forward output")
            out[f"eval_{tag}"] = {"logits_shape": list(o["logits"].shape), "prediction_shape": list(o["prediction"].shape),
                                  "hidden_shape": list(o["hidden"].shape), "logits_sha256": tensor_digest(o["logits"]),
                                  "prediction_sha256": tensor_digest(o["prediction"]), "fusion_weights_sha256": tensor_digest(o["fusion_weights"]),
                                  "prediction_outside_applicable_all_zero": bool((o["prediction"][:, ~b["applicable_entry"][0]] == 0).all())}
    model.train()                                                   # dropout 0.1 active, seeded
    torch.manual_seed(seed)
    model.zero_grad(set_to_none=True)
    o = model(**fwd(b1))
    loss1 = L.masked_mse(o["prediction"], b1["target"], b1["target_mask"])
    loss1.backward()
    out["s1_masked_mse"] = {"loss": loss1.item(), "effective_target_entries": int(b1["target_mask"].sum()), **_grad_report(model, "fault_head.")}
    try:
        L.weighted_cross_entropy(o["logits"].detach(), b1["fault_class"], weights)                   # S1 windows carry no label: must be rejected
        raise SmokeError("a -1 (no label) fault label was accepted by the loss")
    except L.LossInputError:
        out["s1_unlabelled_labels_rejected_by_fault_loss"] = True
    torch.manual_seed(seed)
    model.zero_grad(set_to_none=True)
    o = model(**fwd(b2))
    loss2 = L.weighted_cross_entropy(o["logits"], b2["fault_class"], weights)
    loss2.backward()
    out["s2_weighted_ce"] = {"loss": loss2.item(), "labels": b2["fault_class"].tolist(), **_grad_report(model, "next_step_head.")}
    if not (torch.isfinite(loss1) and torch.isfinite(loss2)):
        raise SmokeError("non-finite loss")
    out["optimizer_steps_taken"] = 0
    return out


def hashes_of(r):
    flat = {"parameters": r["parameters"], "initial_state_sha256": r["initial_state_sha256"], "s1_loss": r["s1_masked_mse"]["loss"], "s2_loss": r["s2_weighted_ce"]["loss"],
            "s1_grad_norm": r["s1_masked_mse"]["global_grad_norm"], "s2_grad_norm": r["s2_weighted_ce"]["global_grad_norm"]}
    for tag in ("s1", "s2"):
        e = r[f"eval_{tag}"]
        flat.update({f"{tag}_logits": e["logits_sha256"], f"{tag}_prediction": e["prediction_sha256"], f"{tag}_fusion_weights": e["fusion_weights_sha256"],
                     f"{tag}_inputs": r["batches"][tag]["inputs_sha256"]})
    return flat


def _run(cmd, env=None, timeout=3600):
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, p.stdout + p.stderr


def _git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def _unittest_file(name):
    """Run one test module in a clean process; return {ran, skipped, failures, errors, per_test: [(class, test, status)]}."""
    env = {**os.environ, "M1_REQUIRE_REAL_DATA": "1"}
    rc, text = _run([".venv/bin/python", "-I", "-m", "unittest", "discover", "-s", "armor-system/tests/prep", "-p", name, "-v"], env=env)
    ran = int(re.search(r"Ran (\d+) tests", text).group(1))
    per = re.findall(r"^(test_\w+) \(([\w.]+)\.(\w+)\.\w+\) \.\.\. (.+)$", text, flags=re.M)
    last = text.strip().splitlines()[-1]
    return {"ran": ran, "skipped": int((re.search(r"skipped=(\d+)", last) or [0, 0])[1]), "failures": int((re.search(r"failures=(\d+)", last) or [0, 0])[1]),
            "errors": int((re.search(r"errors=(\d+)", last) or [0, 0])[1]), "ok": rc == 0 and last.startswith("OK"),
            "per_test": [{"class": c, "test": t, "status": s} for t, _m, c, s in per]}


def _environment():
    return {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}", "numpy": np.__version__, "torch": torch.__version__,
            "device": "cpu", "torch_num_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def write_evidence(D):
    t0 = time.time()
    status = _git("status", "--porcelain", "--untracked-files=all")
    r1, r2 = run_smoke(D), run_smoke(D)
    in_process = hashes_of(r1) == hashes_of(r2)
    subs = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "smoke.json")
            rc, text = _run([".venv/bin/python", "-I", os.path.relpath(__file__, REPO), "--json-out", path, "--once"])
            if rc != 0:
                raise SmokeError("clean-process smoke rerun failed:\n" + text[-2000:])
            subs.append(hashes_of(json.load(open(path))))
    cross = all(s == hashes_of(r1) for s in subs)
    tests = {n: _unittest_file(f"test_m1_model_{n}.py") for n in ("shapes", "invariance", "real_smoke")}
    for n, t in tests.items():
        if not t["ok"] or t["skipped"]:
            raise SmokeError(f"model test module {n} not clean: {t}")
    mi = {}
    for t in tests["invariance"]["per_test"]:
        mi.setdefault(MI_CLASSES[t["class"]], []).append({"test": t["test"], "status": t["status"]})
    if sorted(mi) != sorted(MI_CLASSES.values()) or any(x["status"] != "ok" for v in mi.values() for x in v):
        raise SmokeError(f"MI-1..MI-7 not all passing: {mi}")
    rc, text = _run(INTEGRATION, env={**os.environ, "M1_REQUIRE_REAL_DATA": "1"})
    m = re.search(r"SUCCESS in ([\d.]+) s: tests ran (\d+), skipped (\d+), failures (\d+), errors (\d+)", text)
    suites = {s: {"ran": int(a), "failures": int(f), "errors": int(e), "skipped": int(k)} for s, a, f, e, k in
              re.findall(r"^\s+(\w+): ran (\d+), failures (\d+), errors (\d+), skipped (\d+)", text, flags=re.M)}
    if rc != 0 or not m or int(m[3]) or int(m[4]) or int(m[5]) or "evidence verification: OK" not in text:
        raise SmokeError("M1-03 integration command did not succeed with zero skips:\n" + text[-3000:])
    rows, set_hash = m1ev.source_set(REPO, SOURCE_FILES)
    hist = {p: {"sha256": m1ev.sha256_file(os.path.join(REPO, p)), "unchanged_vs_HEAD": subprocess.run(["git", "diff", "--quiet", "HEAD", "--", p], cwd=REPO).returncode == 0}
            for p in HISTORICAL}
    ev = {"evidence_name": EVIDENCE_NAME, "evidence_version": "1.0", "linear_issue": "298-50",
          "scope": "Model 1 v0 spatio-temporal Transformer core (architecture, masked losses, MI-1..MI-7, real-D1 CPU forward/backward). No training, no validation or test evaluation.",
          "command": COMMAND, "environment": _environment(),
          "model": {"parameters": r1["parameters"], "config": ModelConfig().__dict__, "spec_fingerprint": r1["spec_fingerprint"], "initial_state_sha256": r1["initial_state_sha256"],
                    "class_weights": r1["class_weights"]},
          "smoke": {k: r1[k] for k in ("seed", "batch_size", "batches", "eval_s1", "eval_s2", "s1_masked_mse", "s2_weighted_ce", "s1_unlabelled_labels_rejected_by_fault_loss", "optimizer_steps_taken")},
          "reproducibility": {"in_process_two_runs_identical": in_process, "clean_process_reruns_identical": cross, "hashes": hashes_of(r1)},
          "invariance_tests": mi,
          "tests": {"model_modules": {n: {k: v for k, v in t.items() if k != "per_test"} for n, t in tests.items()},
                    "m1_integration": {"command": " ".join(INTEGRATION), "tests_ran": int(m[2]), "skipped": int(m[3]), "failures": int(m[4]), "errors": int(m[5]), "suites": suites,
                                       "seconds": float(m[1]), "evidence_m1_03_verified": True}},
          "historical_evidence": hist,
          "commit_reference": {"scheme": "No self-referential hash: source_set_sha256 identifies the exact content of the files below (identical in the working tree and in a commit that adds them).",
                               "evidence_base_commit": _git("rev-parse", "HEAD"), "origin_main_at_generation": _git("rev-parse", "origin/main"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                               "uncommitted_changes_when_generated": [l[3:] for l in status.splitlines()],
                               "source_set_sha256": set_hash, "source_files": rows, "final_merge_commit": None},
          "generation_seconds": round(time.time() - t0, 1)}
    if not (in_process and cross) or not all(h["unchanged_vs_HEAD"] for h in hist.values()):
        raise SmokeError(f"reproducibility or historical-evidence check failed: {ev['reproducibility']}, {hist}")
    path = os.path.join(REPO, EVIDENCE_REL)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(ev, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote", EVIDENCE_REL, "source_set_sha256", set_hash)
    return ev


def verify_evidence(D):
    ev = json.load(open(os.path.join(REPO, EVIDENCE_REL)))
    _, set_hash = m1ev.source_set(REPO, SOURCE_FILES)
    bad = []
    if ev["commit_reference"]["source_set_sha256"] != set_hash:
        bad.append("source_set_sha256 differs from the tree (code or tests changed after the evidence was generated)")
    if hashes_of(run_smoke(D)) != ev["reproducibility"]["hashes"]:
        bad.append("a fresh smoke run does not reproduce the recorded hashes")
    for p, h in ev["historical_evidence"].items():
        if m1ev.sha256_file(os.path.join(REPO, p)) != h["sha256"]:
            bad.append(f"historical evidence changed: {p}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-evidence", action="store_true")
    ap.add_argument("--verify-evidence", action="store_true")
    ap.add_argument("--json-out", help="write one smoke result as JSON (used by clean-process reruns)")
    ap.add_argument("--once", action="store_true", help="run the smoke once instead of twice")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    enable_determinism()
    D = md.Model1Data.load()
    if a.write_evidence:
        write_evidence(D)
        return 0
    if a.verify_evidence:
        bad = verify_evidence(D)
        print("evidence verification:", bad or "OK")
        return 1 if bad else 0
    r = run_smoke(D, seed=a.seed)
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(r, fh, sort_keys=True)
    if not a.once:
        again = hashes_of(run_smoke(D, seed=a.seed))
        if again != hashes_of(r):
            raise SmokeError(f"two runs differ: {hashes_of(r)} vs {again}")
        print("two runs identical in-process")
    print(json.dumps({k: r[k] for k in ("parameters", "initial_state_sha256", "s1_masked_mse", "s2_weighted_ce")}, indent=2))
    print(json.dumps(hashes_of(r), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
