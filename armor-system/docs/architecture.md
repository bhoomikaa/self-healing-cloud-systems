# ARMOR system architecture

What runs, where it runs, and how data moves between the pieces. Diagrams are
Mermaid, which GitHub renders in place, so they stay editable rather than
becoming a screenshot nobody can update.

Everything below is built and running unless a box says otherwise.

## 1. Data flow, source to prediction

```mermaid
flowchart TD
    subgraph ingest["Ingestion"]
        Z["Zenodo 21744694<br/>ARMOR D1 release, CC-BY-4.0"]
        F["fetch_d1.py<br/>idempotent download<br/>CRC32 + size vs central directory"]
        V["validate_d1.py<br/>checksum gate<br/>SHA-256 vs file_inventory.csv"]
        R[("data/raw/<br/>466 MB extracted<br/>Git-ignored")]
        Z --> F --> R --> V
    end

    subgraph prep["Preparation — pipeline.py"]
        C["cache<br/>decode to float32"]
        I["inspect<br/>schema, counts, leakage"]
        S["splits<br/>case-disjoint, chronological"]
        P["preprocess<br/>train-only scaler"]
        E["eda + figures"]
        A["augmentation<br/>assessed, not adopted"]
        Q["quality<br/>nine-stage validation"]
        NPZ[("data/processed/<br/>d1_features_v1.npz<br/>7200 x 46 x 130")]
        ART[("data/artifacts/d1/<br/>62 committed artifacts")]
        V --> C --> NPZ
        C --> I --> S --> P --> E --> A --> Q
        Q --> ART
    end

    subgraph warehouse["Warehouse"]
        W["build_warehouse.py<br/>star schema"]
        LAKE[("data/warehouse/<br/>Parquet + zstd, 30.6 MB<br/>Hive-partitioned by utc_date")]
        L["load_bigquery.py<br/>load + verify"]
        BQ[("BigQuery armor-d1.armor_d1<br/>fact_telemetry 43,056,000 rows<br/>clustered on timestamp")]
        NPZ --> W --> LAKE --> L --> BQ
        ART --> W
    end

    subgraph model["Model 1"]
        CK[("5 frozen Stage 2 checkpoints<br/>private HF model repo<br/>hashed against the test freeze")]
        SVC["Model1Service<br/>scaler, windowing, ensemble<br/>causal rolling max, ARMOR intervals"]
        CK --> SVC
    end

    subgraph serve["Serving"]
        API["model1-serving container<br/>/healthz /v1/metadata /v1/predict<br/>321 MB, CPU only"]
        UI["Streamlit app<br/>armor-d1-model1.streamlit.app<br/>public, audience picks the input"]
        SVC --> API
        SVC --> UI
    end

    BQ --> UI
    ART --> SVC

    AF["Airflow<br/>18 tasks, weekly, LocalExecutor"]
    AF -.orchestrates.-> ingest
    AF -.orchestrates.-> prep
    AF -.orchestrates.-> warehouse
```

The dotted edges are orchestration, not data. Airflow runs the same committed
scripts a developer runs by hand, so the DAG and a manual run cannot drift.

## 2. Where each piece runs

| Component | Runs on | Environment | State |
|---|---|---|---|
| Ingestion and preparation | Airflow, Docker, local | Python 3.12, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, matplotlib 3.11.2 | Built |
| Orchestration | Airflow 2.10.5, LocalExecutor, Postgres 16 | Docker Compose, repo mounted read-only | Built |
| Lakehouse | Local disk, Parquet + zstd | pyarrow 23.0.0 | Built |
| Warehouse | BigQuery sandbox, US | free tier, no billing account | Built |
| Model 1 training | Apple M4, CPU only, 1 torch thread | Python 3.14.4, torch 2.14.0 | Done, frozen |
| Inference service | Docker, CPU only | Python 3.14-slim, torch 2.14.0+cpu | Built |
| Prototype UI | Streamlit Community Cloud | Python 3.14, torch 2.14.0+cpu | Deployed |
| Checkpoint storage | Private Hugging Face model repo | pulled at start-up, hash-checked | Built |
| Models 2, 3, 4 | — | — | 298B |

The training environment is Python 3.14 and the Airflow image is Python 3.12,
the newest Airflow 2.10.5 supports. That gap is real: it is why regenerated
artifacts compare by content rather than by bytes.

## 3. Model 1 internals

Five frozen checkpoints voting. Each is the same architecture, trained from a
different seed.

```mermaid
flowchart LR
    IN["6 historical minutes<br/>46 nodes x 130 channels<br/>+ observed mask"]

    subgraph adapt["Input adapters"]
        PM["pod metric 53ch"]
        PT["pod trace 9ch"]
        PL["pod log 11ch"]
        VM["VM metric 57ch"]
    end

    FUSE["Modality fusion<br/>gated, availability-weighted<br/>learned fallback when all absent"]
    CORE["2 x factorised block<br/>temporal attention over 6 minutes<br/>then cross-node attention over 46 nodes<br/>d=64, 4 heads, FFN 256"]
    H["hidden, last minute<br/>46 x 64"]
    FH["Fault head<br/>mean+max over valid nodes<br/>-> 5 classes"]
    NH["Next-step head<br/>-> target minute telemetry"]

    IN --> adapt --> FUSE --> CORE --> H
    H --> FH
    H --> NH
```

Each adapter takes the masked values **and** the mask, so its input width is
twice its channel count. Unavailable entries are zeroed with `torch.where`
before any arithmetic, so a NaN in an unobserved slot cannot reach a weight.

Detection, on top of the five models:

```mermaid
flowchart LR
    S5["5 x detection logit"] --> M["mean sigmoid"] --> RM["causal 60 s rolling max"]
    RM --> TH{"> 0.7832 ?"}
    TH -->|yes| AL["alert minute"]
    TH -->|no| OK["quiet"]
    AL --> INC["ARMOR interval rule<br/>merge gaps under 600 s<br/>drop single-minute intervals"]
```

The threshold is the empirical 90th percentile of rolled calibration scores,
fixed once on the calibration split. It is not retuned per run; the serving
metadata publishes it so a caller can check.

## 4. What the numbers are

| | Ours | ARMOR published |
|---|---|---|
| Detection F1 | 0.733 ± 0.052 | 0.9608 |
| Fault type, weighted F1 | 0.827 ± 0.039 | 0.938 |
| Fault type, macro F1 | 0.721 ± 0.055 | — |

The gap is reported rather than tuned away. Training consumed **14.01
CPU-hours** across 20 recorded runs on an Apple M4, CPU only, one torch thread,
deterministic algorithms on.

## 5. Trust boundaries

Three places where something could silently become wrong, and what stops it.

| Boundary | Risk | Control |
|---|---|---|
| Zenodo to disk | a truncated or altered release | CRC32 and size at fetch, SHA-256 against the committed inventory at verify, and a negative control that corrupts a scratch copy and asserts the gate rejects it |
| Pipeline to warehouse | a load that reports success but drops rows | row counts per table and per partition, split totals derived independently from `dim_split`, and a value checksum; any mismatch exits non-zero |
| Checkpoints to inference | serving a different model than was evaluated | every checkpoint hashed against `evidence_test_freeze.json` at start-up; the service refuses to run on a mismatch and publishes the hashes over `/v1/metadata` |

The repository is mounted read-only in Airflow. An earlier version mounted it
read-write and a scheduled run overwrote the team's reviewed artifacts, which is
why the DAG now regenerates into a disposable workspace and diffs the result
against the read-only original.
