# Deployment

| What | Status |
|---|---|
| [`model1-serving/`](model1-serving/) | **Built and verified locally.** Model 1 v2 inference service in a container. Reproduces the committed validation results exactly; see `model1-serving/results/container_verification.json`. |
| Cloud deployment | **Blocked.** Provider, object storage, credential handling for a public repository, and budget are all undecided team questions. |
| CI/CD | **Not started.** |

Local orchestration for the data pipeline lives in
[`../airflow/`](../airflow/), not here.
