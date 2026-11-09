"""Checkpoints with provenance and integrity checks (Linear 298-51). Checkpoints are git-ignored (*.pt): never commit them.

Two files: `<name>.pt` (model weights and training metadata as plain tensors/strings, loadable with weights_only=True) and
`<name>.json` (provenance, the model state digest and the SHA-256 of the .pt file). Loading verifies the file hash, the state
digest, the architecture and the data provenance the caller expects; any mismatch raises CheckpointError.
"""
import hashlib
import json
import os
import platform
import subprocess

import numpy as np
import torch

from .model import state_digest


class CheckpointError(RuntimeError):
    """Corrupt, tampered or incompatible checkpoint."""


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git_head(repo):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def data_provenance(data, spec, train_cfg, repo=None):
    """Facts a checkpoint must be tied to: split manifest, fit-only scaler, feature cache, architecture, training config."""
    m = data.manifest
    return {"manifest_content_sha256": m["content_sha256"], "fit_only_scaler_file_sha256": m["fit_only_scaler"]["file_sha256"],
            "feature_cache_content_sha256": m["inputs"]["feature_cache_content_sha256"], "preprocessing_config_sha256": m["inputs"]["preprocessing_config_sha256"],
            "spec_fingerprint": spec.fingerprint(), "train_config_sha256": train_cfg.sha256, "git_head": _git_head(repo) if repo else None,
            "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()}


def save_checkpoint(path, model, *, stage, seed, epoch, provenance, history=None, optimizer=None, extra=None):
    """Write `<path>.pt` and `<path>.json` (path without extension). Returns the metadata dict."""
    pt, js = path + ".pt", path + ".json"
    os.makedirs(os.path.dirname(os.path.abspath(pt)), exist_ok=True)
    meta = {"stage": stage, "seed": int(seed), "epoch": int(epoch), "parameters": model.n_parameters(), "model_config": dict(model.cfg.__dict__),
            "state_sha256": state_digest(model), "provenance": provenance, "history": history or [], "extra": extra or {}}
    blob = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "meta_json": json.dumps(meta, sort_keys=True)}
    if optimizer is not None:
        blob["optimizer_state"] = optimizer.state_dict()
    torch.save(blob, pt)
    meta_file = dict(meta, pt_file_sha256=_sha256_file(pt))
    with open(js, "w") as fh:
        json.dump(meta_file, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return meta_file


def load_checkpoint(path, model, *, expect=None, stage=None, seed=None):
    """Verify and load `<path>.pt` into `model` (strict). `expect`: provenance keys that must match exactly. Returns the metadata."""
    pt, js = path + ".pt", path + ".json"
    if not (os.path.isfile(pt) and os.path.isfile(js)):
        raise CheckpointError(f"checkpoint pair {path}.pt/.json is incomplete")
    with open(js) as fh:
        meta = json.load(fh)
    if _sha256_file(pt) != meta.get("pt_file_sha256"):
        raise CheckpointError("checkpoint file hash differs from its provenance file (corrupt or modified)")
    try:
        blob = torch.load(pt, map_location="cpu", weights_only=True)
    except Exception as e:                                                                 # noqa: BLE001
        raise CheckpointError(f"checkpoint cannot be read safely: {e}") from None
    inner = json.loads(blob["meta_json"])
    core = {k: v for k, v in meta.items() if k != "pt_file_sha256"}
    if inner != core:
        raise CheckpointError("provenance file differs from the metadata stored inside the checkpoint")
    for name, want in (("stage", stage), ("seed", seed)):
        if want is not None and meta[name] != want:
            raise CheckpointError(f"checkpoint {name} {meta[name]!r} != expected {want!r}")
    if meta["model_config"] != dict(model.cfg.__dict__) or meta["provenance"].get("spec_fingerprint") != model.spec.fingerprint():
        raise CheckpointError("checkpoint architecture differs from the model it is loaded into")
    for k, v in (expect or {}).items():
        if meta["provenance"].get(k) != v:
            raise CheckpointError(f"provenance mismatch for {k}: checkpoint {meta['provenance'].get(k)!r} != expected {v!r}")
    try:
        model.load_state_dict(blob["model_state"], strict=True)
    except RuntimeError as e:
        raise CheckpointError(f"state dict does not fit the model: {e}") from None
    if state_digest(model) != meta["state_sha256"]:
        raise CheckpointError("loaded weights do not match the recorded state digest (tampered weights)")
    return meta
