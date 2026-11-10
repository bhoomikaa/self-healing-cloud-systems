# Model 1 v2 inference service, containerised

Runs the service described in
`armor-system/models/model1_v2/serving/SERVING.md` in a container, so that
anyone with Docker and a copy of the checkpoints can start it, instead of only
the machine it was trained on.

Verified on 2026-11-10: the container reproduces the committed validation
results exactly. Evidence in `results/container_verification.json`.

## Run it

```
cp .env.example .env            # then point MODEL1_CHECKPOINT_DIR at the checkpoints
docker compose up -d --build
curl localhost:8082/healthz     # {"status": "ok", "model": "model1-v2-ensemble", "version": "2.0.0"}
```

`docker compose down` stops it. The first build takes about 10 minutes, almost
all of it installing torch.

## Checkpoints

The ten files in `model1_v1/seed_{0..4}/s2_best.{pt,json}` are **not** in Git
and **not** in the image. They are mounted read-only from
`MODEL1_CHECKPOINT_DIR` at run time.

Two reasons for keeping them out of the image. Baking 3.5 MB of weights into a
layer would put model weights into something we may later push to a registry,
and the service hashes each file at start-up against
`armor-system/models/model1_transformer/results/test_freeze/evidence_test_freeze.json`,
so a retrain would silently invalidate a baked image.

The start-up check means a container that reaches `healthy` has already proved
which weights it loaded. `/v1/metadata` reports the five `checkpoint_sha256`
values, so a consumer can confirm it is talking to the frozen model rather than
trusting the tag.

## What was verified

**Weights.** All ten mounted files hash-match the frozen test evidence. Zero
mismatches.

**Accuracy.** `replay_demo.py` streamed 875 recorded D1 validation minutes
through the container's HTTP API:

| | TP | FP | F1 |
|---|---|---|---|
| Container | 18 | 6 | 0.7347 |
| Committed v2 validation | 18 | 6 | 0.7347 |

Fault type agreed with the documented class on 16 of the 16 incidents that hit
a documented case.

**Latency.** `bench.py`, the same script and synthetic inputs as the native
run, executed inside the container:

| Segment | Windows | Container, server ms/window | Native reference (M4) |
|---|---|---|---|
| 7 minutes | 1 | 100.9 | 10.4 |
| 66 minutes | 60 | 69.6 | 8.3 |
| 240 minutes | 234 | 62.9 | 8.4 |

Start-up, including checkpoint hashing, is 6.7 s against 0.35 s native.

Six to ten times slower per window. The honest reading: the two runs differ in
both CPU and virtualisation (WSL2 on x86 against macOS on an M4), so this does
**not** isolate container overhead from the hardware difference. Treat it as an
upper bound for this host.

It does not threaten the design either way. The service is called once a
minute, so the worst figure above spends 101 ms of a 60,000 ms budget, under
0.2%. Latency only becomes worth optimising if we move to sub-minute detection.

## Build context

`Dockerfile.dockerignore` trims the context from about 500 MB to the few MB the
image actually copies. Without it, Docker ships `data/raw` and fails on the
symlink Airflow leaves at `armor-system/airflow/logs/scheduler/latest`, which
Windows cannot open. BuildKit prefers a `<dockerfile>.dockerignore` over the
context root's, so this file affects this image only and leaves the Airflow
build and anyone else building from the repository root alone.

## Not done yet

- **Shared checkpoint storage.** The weights still travel by hand. SERVING.md
  section 8 leaves this open and the readiness audit lists it as blocking.
- **Cloud deployment.** Blocked on team decisions: provider, object storage,
  how credentials are handled given the repository is public, and budget.
- **Wiring into Airflow.** The compose file publishes the service on the
  `model1-net` network so the Airflow stack can join it and reach the service by
  hostname rather than through the host's published port. SERVING.md section 7
  sketches the per-minute flow. Not built.
