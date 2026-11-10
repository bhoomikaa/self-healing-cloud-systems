---
title: ARMOR Model 1 Detection API
emoji: 📡
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
short_description: Multimodal transformer detecting and classifying cloud incidents from ARMOR D1 telemetry
---

# ARMOR Model 1 v2 detection API

Detects and classifies infrastructure incidents from one minute of multimodal
cloud telemetry. Five-member ensemble of frozen transformer checkpoints, CPU
only.

SJSU DATA 298A, Team 1, Topic 6: AI-Driven Incident Response and Self-Healing
Cloud Systems.

## Endpoints

| | |
|---|---|
| `GET /healthz` | `{"status": "ok", ...}` once the checkpoints and artefacts are verified |
| `GET /v1/metadata` | model identity, the five checkpoint hashes, detection threshold, node and channel order |
| `POST /v1/predict` | `model1.telemetry.v1` in, `model1.prediction.v1` out |

A request carries at least 7 consecutive one-minute samples of D1-layout
features, 46 nodes by 130 channels, before Model 1 scaling. Every window of 6
input minutes plus 1 target minute inside the segment is scored. Validation never
repairs data: anything outside the contract is rejected with a code, a field and
a message. Full contract in `armor-system/models/model1_v2/serving/SERVING.md`.

## What the numbers are

Detection F1 **0.733 ± 0.052** against ARMOR's published 0.9608; fault-type
weighted F1 **0.827 ± 0.039** against 0.938. The gap is reported rather than
tuned away, and the analysis of why sits in the repository.

Replaying 875 recorded D1 validation minutes through this API reproduces the
committed validation result exactly: TP 18, FP 6, F1 0.7347.

`/v1/metadata` publishes the SHA-256 of all five checkpoints, so a caller can
confirm it is talking to the frozen model rather than trusting a tag.

## Weights

Not in this Space. They live in a private model repo and are pulled at start-up
with the `HF_TOKEN` secret. The service hashes each file against the frozen test
evidence and refuses to start if any is missing or altered.

Set as Space settings:

| | |
|---|---|
| `MODEL1_HF_REPO` (variable) | the private repo holding `model1_v1/seed_{0..4}/s2_best.{pt,json}` |
| `HF_TOKEN` (secret) | a read token for that repo |

## This is replay, not live telemetry

The D1 release is recorded 2022 data from a microservice testbed. Nothing here
is monitoring a running system. Source: ARMOR D1, Zenodo record 21744694,
CC-BY-4.0.
