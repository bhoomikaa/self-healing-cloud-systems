"""ARMOR Model 1: the deployed prototype.

One complete path through the system, which is what the demo asks for: a minute
chosen by whoever is watching, pulled live from the BigQuery warehouse the
pipeline wrote, scored by the five-seed ensemble, and the result shown.

Nothing here is precomputed. The telemetry is queried at the moment you press
the button, the model runs in this process, and the numbers on screen come out
of that run. If BigQuery is unreachable or the checkpoints fail their hash
check, the page says so and refuses rather than falling back to something
canned.

Two secrets, both set in the Streamlit app settings and neither in the
repository:

    HF_TOKEN              read token for the private checkpoint repo
    MODEL1_HF_REPO        e.g. "bhoomikaa7/armor-model1-checkpoints"
    [gcp_service_account] a service-account key with BigQuery read access

Run locally with the same secrets in .streamlit/secrets.toml:

    streamlit run armor-system/deployment/streamlit-app/app.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
ARMOR = os.path.abspath(os.path.join(HERE, "..", ".."))
if os.path.join(ARMOR, "models") not in sys.path:
    sys.path.insert(0, os.path.join(ARMOR, "models"))

CKPT_DIR = os.environ.get("MODEL1_CHECKPOINT_ROOT", "/tmp/model1_checkpoints")
DATASET = "armor_d1"
N_NODES, N_CHANNELS = 46, 130
WINDOW_MINUTES = 6          # context minutes the model needs before a target minute
STEP_S = 60

st.set_page_config(page_title="ARMOR Model 1", page_icon="📡", layout="wide")


# ----------------------------------------------------------------- resources

@st.cache_resource(show_spinner="Fetching the frozen checkpoints…")
def load_service():
    """Pull the weights, then load the ensemble.

    `Model1Service.load` hashes all ten files against the frozen test evidence
    and raises if any is missing or altered, so a partial download cannot
    quietly serve a different model.
    """
    from huggingface_hub import snapshot_download

    repo = st.secrets.get("MODEL1_HF_REPO", os.environ.get("MODEL1_HF_REPO", "")).strip()
    token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", "")).strip() or None
    if not repo:
        raise RuntimeError("MODEL1_HF_REPO is not set in the app secrets.")

    patterns = [f"model1_v1/seed_{s}/s2_best.{e}" for s in range(5) for e in ("pt", "json")]
    os.makedirs(CKPT_DIR, exist_ok=True)
    snapshot_download(repo_id=repo, repo_type="model", token=token,
                      local_dir=CKPT_DIR, allow_patterns=patterns)

    from model1_v2.serving.service import Model1Service
    return Model1Service.load(ckpt_root=CKPT_DIR)


def secret_names():
    """Top-level secret keys that are set. Names only; values are never shown."""
    try:
        return sorted(st.secrets.keys())
    except Exception:                                             # noqa: BLE001
        return []


@st.cache_resource(show_spinner="Connecting to BigQuery…")
def bq():
    from google.cloud import bigquery
    from google.oauth2 import service_account

    if "gcp_service_account" in st.secrets:
        info = dict(st.secrets["gcp_service_account"])
        missing = [k for k in ("project_id", "private_key", "client_email") if not info.get(k)]
        if missing:
            raise RuntimeError(
                "The [gcp_service_account] secret is present but incomplete; missing: "
                + ", ".join(missing)
                + ". Regenerate the block with make_secrets.py and paste all of it.")
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        return bigquery.Client(credentials=creds, project=info["project_id"])

    if os.environ.get("STREAMLIT_RUNTIME_ENV") or not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""):
        # Say which secret is missing rather than letting the client fail later
        # with "Project was not passed", which points at the wrong thing.
        raise RuntimeError(
            "No [gcp_service_account] section in this app's secrets, so there is no way to "
            "reach BigQuery.\n\n"
            f"Secrets currently set: {secret_names() or 'none at all'}\n\n"
            "Fix it in Manage app -> Settings -> Secrets: paste the whole block that "
            "make_secrets.py prints, including the [gcp_service_account] header and every "
            "line under it, then save and reboot the app. Streamlit takes about a minute "
            "to propagate a secrets change.")

    # Local development uses whatever `gcloud auth application-default login` left.
    return bigquery.Client()


@st.cache_data(ttl=3600, show_spinner=False)
def load_cases(project):
    return bq().query(f"""
        SELECT case_index, timestamp, utc, node_name, failure_type, failure_class, split
        FROM `{project}.{DATASET}.fact_case` ORDER BY timestamp""").to_dataframe()


@st.cache_data(ttl=3600, show_spinner=False)
def load_dims(project):
    nodes = bq().query(f"SELECT node_index, node_name, node_kind FROM `{project}.{DATASET}.dim_node` ORDER BY node_index").to_dataframe()
    chans = bq().query(f"SELECT channel_index, channel_name, modality FROM `{project}.{DATASET}.dim_channel` ORDER BY channel_index").to_dataframe()
    span = bq().query(f"SELECT MIN(timestamp) AS lo, MAX(timestamp) AS hi FROM `{project}.{DATASET}.fact_telemetry`").to_dataframe()
    return nodes, chans, span.iloc[0]


def fetch_window(project, start_ts, n_minutes):
    """Pull n_minutes of telemetry as a (T, 46, 130) array, straight from the warehouse."""
    end_ts = start_ts + (n_minutes - 1) * STEP_S
    sql = f"""
        SELECT timestamp, node_index, channel_index, value
        FROM `{project}.{DATASET}.fact_telemetry`
        WHERE timestamp BETWEEN @lo AND @hi
    """
    from google.cloud import bigquery
    job = bq().query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("lo", "INT64", int(start_ts)),
        bigquery.ScalarQueryParameter("hi", "INT64", int(end_ts)),
    ]))
    df = job.to_dataframe()
    if df.empty:
        raise RuntimeError(f"the warehouse returned no rows between {start_ts} and {end_ts}")

    ts = np.sort(df["timestamp"].unique())
    pos = {t: i for i, t in enumerate(ts)}
    X = np.zeros((len(ts), N_NODES, N_CHANNELS), dtype=np.float32)
    X[df["timestamp"].map(pos).to_numpy(), df["node_index"].to_numpy(), df["channel_index"].to_numpy()] = \
        df["value"].to_numpy(dtype=np.float32)

    expected = len(ts) * N_NODES * N_CHANNELS
    if len(df) != expected:
        raise RuntimeError(f"warehouse returned {len(df):,} rows for {len(ts)} minutes; expected {expected:,}. "
                           "The window is incomplete, so it is not being scored.")
    return ts.astype(np.int64), X, job


def score(service, ts, X, label):
    from model1_v2.serving import contract as ct
    meta = service.metadata()
    body = {
        "schema_version": ct.REQUEST_SCHEMA,
        "source": {"kind": "replay", "name": label,
                   "note": "recorded 2022 D1 telemetry read from BigQuery; not live production data"},
        "layout": meta["layout"],
        "timestamps": [int(t) for t in ts],
        "values": X,
    }
    t0 = time.perf_counter()
    res = service.handle(body)
    res["_client_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return res


# ----------------------------------------------------------------------- UI

st.title("ARMOR Model 1 — incident detection")
st.caption("SJSU DATA 298A · Team 1 · Topic 6 — AI-Driven Incident Response and Self-Healing Cloud Systems")

project = st.secrets.get("gcp_service_account", {}).get("project_id") or os.environ.get("GCP_PROJECT", "armor-d1")

with st.sidebar:
    st.header("Pick a minute to score")
    st.caption("Telemetry is read from BigQuery when you press Score. Nothing is precomputed.")

    try:
        cases = load_cases(project)
        nodes, chans, span = load_dims(project)
    except Exception as exc:                                      # noqa: BLE001
        st.error(f"Cannot reach the warehouse `{project}.{DATASET}`.\n\n{type(exc).__name__}: {exc}")
        st.stop()

    mode = st.radio("Input", ["A documented incident", "Any minute in the dataset"])

    if mode == "A documented incident":
        split = st.selectbox("Split", ["test", "validation", "train"], index=0)
        sel = cases[cases["split"] == split].reset_index(drop=True)
        labels = [f"{r.utc}  ·  {r.failure_class}  ·  {r.node_name}" for r in sel.itertuples()]
        i = st.selectbox("Case", range(len(sel)), format_func=lambda k: labels[k])
        target_ts = int(sel.loc[i, "timestamp"])
        truth = {"class": sel.loc[i, "failure_class"], "type": sel.loc[i, "failure_type"],
                 "node": sel.loc[i, "node_name"], "split": split}
        # D1 case timestamps are not on a minute boundary; the warehouse is.
        target_ts -= target_ts % STEP_S
    else:
        lo, hi = int(span["lo"]), int(span["hi"])
        picked = st.slider("Minute", min_value=lo, max_value=hi, value=(lo + hi) // 2, step=STEP_S,
                           format="%d")
        target_ts = int(picked) - int(picked) % STEP_S
        truth = None
        st.caption(pd.to_datetime(target_ts, unit="s", utc=True).strftime("%Y-%m-%d %H:%M UTC"))

    minutes_after = st.slider("Minutes to score from there", 1, 30, 5,
                              help="Each scored minute needs the 6 minutes before it as context.")
    go = st.button("Score", type="primary", use_container_width=True)

if not go:
    st.info("Choose an input on the left and press **Score**. "
            "The window is fetched from BigQuery and run through the five-seed ensemble live.")
    with st.expander("What this is"):
        st.markdown(
            "- **Warehouse**: `fact_telemetry`, 43,056,000 rows, partitioned by UTC date.\n"
            "- **Model**: five frozen Stage 2 checkpoints, mean sigmoid, causal 60 s rolling max, "
            "frozen calibration threshold.\n"
            "- **Data**: ARMOR D1 (Zenodo 21744694, CC-BY-4.0), recorded 2022 telemetry replayed. "
            "Not a live production system.\n"
            "- Detection F1 is **0.733 ± 0.052** against ARMOR's published 0.9608. The gap is reported, not tuned away.")
    st.stop()

start_ts = target_ts - WINDOW_MINUTES * STEP_S
n_minutes = WINDOW_MINUTES + minutes_after

try:
    service = load_service()
except Exception as exc:                                          # noqa: BLE001
    st.error(f"The model would not load, so nothing is being scored.\n\n{type(exc).__name__}: {exc}")
    st.stop()

with st.status("Running", expanded=True) as status:
    st.write(f"Querying `{project}.{DATASET}.fact_telemetry` for {n_minutes} minutes…")
    try:
        ts, X, job = fetch_window(project, start_ts, n_minutes)
    except Exception as exc:                                      # noqa: BLE001
        status.update(label="Warehouse query failed", state="error")
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()
    scanned = (job.total_bytes_processed or 0) / 1e6
    st.write(f"{len(ts)} minutes · {len(ts) * N_NODES * N_CHANNELS:,} values · {scanned:,.1f} MB scanned")
    st.write("Scoring with the five-seed ensemble…")
    try:
        res = score(service, ts, X, f"bigquery:{project}.{DATASET}")
    except Exception as exc:                                      # noqa: BLE001
        status.update(label="Scoring failed", state="error")
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()
    status.update(label=f"Done in {res['_client_ms']:.0f} ms", state="complete")

wins = pd.DataFrame(res["windows"])
wins["utc"] = pd.to_datetime(wins["target_ts"], unit="s", utc=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Minutes scored", res["n_windows"])
c2.metric("Alert minutes", int(wins["alert"].sum()))
c3.metric("Incidents", len(res["incidents"]))
c4.metric("Server time", f"{res['latency_ms']:.0f} ms")

if truth:
    pred = res["incidents"][0]["fault_type"]["pred"] if res["incidents"] else None
    fired = bool(wins["alert"].any())
    cols = st.columns(3)
    cols[0].markdown(f"**Documented fault**  \n{truth['class']}  \n`{truth['type']}`")
    cols[1].markdown(f"**Detected**  \n{'yes' if fired else 'no'}  \non `{truth['node']}` ({truth['split']})")
    cols[2].markdown(f"**Predicted type**  \n{pred or '—'}")
    if fired and pred == truth["class"]:
        st.success("The model detected this incident and named the right fault type.")
    elif fired:
        st.warning(f"Detected, but called it {pred} rather than {truth['class']}.")
    else:
        st.error("The model did not raise an alert for this documented incident. That is a miss, and misses are "
                 "why detection F1 is 0.733 rather than 0.96.")

st.subheader("Detection score per minute")
chart = wins.set_index("utc")[["detection_score", "rolled_score"]]
st.line_chart(chart)
st.caption(f"Alert fires when the causal rolling max crosses the frozen threshold "
           f"{res['model']['detection_threshold']:.4f}. The threshold was calibrated once and is not tuned per run.")

st.subheader("Per-minute detail")
show = wins[["utc", "detection_score", "detection_score_seed_sd", "rolled_score", "alert", "fault_pred"]]
st.dataframe(show, use_container_width=True, hide_index=True)

if res["warnings"]:
    st.subheader("Warnings from the service")
    for w in res["warnings"]:
        st.warning(f"`{w['code']}` — {w['message']}")

with st.expander("Provenance"):
    meta = service.metadata()
    st.markdown(
        f"- **Warehouse**: `{project}.{DATASET}` · {scanned:,.1f} MB scanned for this query\n"
        f"- **Model**: {meta['model']['name']} {meta['model']['version']}\n"
        f"- **Checkpoints**: {', '.join(h[:8] for h in meta['model']['checkpoint_sha256'])}\n"
        f"- **Threshold**: {meta['model']['detection_threshold']}\n"
        f"- **Scaler**: `{meta['model']['fit_only_scaler_sha256'][:16]}…` fitted on the training period only\n")
    st.caption("The checkpoint hashes match `evidence_test_freeze.json`. The service refuses to start if they do not.")
