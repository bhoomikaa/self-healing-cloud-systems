# Evidence files

| File | Issue | What it records |
|---|---|---|
| `evidence.json` | 298-44 | Test run, PR state, artifact index and reproducibility check of the D1 data-preparation work, used by the dashboard's evidence page. |
| `dashboard_browser_check.json` | 298-44 | Result of the browser check of the Streamlit dashboard. |
| `evidence_m1_03.json` | 298-48 | Hashes of the calibration manifest, fit-only scaler and D1 feature cache, the integration command, test results (with skips), the two-run reproducibility comparison and the commit reference. Written by `.venv/bin/python -I armor-system/data/m1_integration.py --write-evidence`. |

**Scope: 298-44 data-preparation artifacts only.** This applies to `evidence.json` and `dashboard_browser_check.json`. They are historical records; later issues do not rewrite them. In particular, the `artifact_set_sha256` in `evidence.json` describes the artifact set as it was when that run finished. Files that later issues add under `splits/` and `preprocessing/` (the two Model 1 M1-03 artifacts) are listed in `evidence_m1_03.json` and are not part of it, so recomputing that hash on a newer checkout gives a different value without anything having been altered.

`evidence_m1_03.json` avoids self-reference: it contains no hash of itself and no commit SHA of the commit that adds it. It records the base commit the work started from, a hash over the content of the M1-03 source files (identical in the working tree and in the commit that holds them) and a `final_merge_commit` field that stays `null` until the merge commit is written into the Linear 298-48 comment. `m1_integration.py` verifies the file against the tree on every run.
