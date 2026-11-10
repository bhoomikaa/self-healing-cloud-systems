# Deployment

| What | Status |
|---|---|
| [`model1-serving/`](model1-serving/) | **Built and verified locally.** Model 1 v2 inference service in a container, 321 MB. Reproduces the committed validation results exactly; see `model1-serving/results/container_verification.json`. |
| [`hf-space/`](hf-space/) | **Packaged, not yet pushed.** The same service as a free Hugging Face Docker Space, for a public demo URL. Weights stay in a private model repo and are pulled at start-up. Needs an account and a token, so the push is a manual step; see `hf-space/DEPLOY.md`. |
| Cloud deployment | **Blocked.** A free Space is a demo URL, not deployment: it sleeps when idle, has no SLA and serves one request at a time. Provider, object storage, credential handling for a public repository, and budget are all undecided team questions. |
| CI/CD | **Not started.** |

Local orchestration for the data pipeline lives in
[`../airflow/`](../airflow/), not here.
