# D2 licence and schema check (Linear 298-51, pre-v2)

Status: **licence unclear, so D2 was not downloaded.** Part 2 stopped at the licence step. No byte of `D2.zip` was fetched; only public metadata pages and README files were read. The schema notes in section 3 come from files already on disk (ARMOR's README and `config/D2.yaml`, fetched with D1) and are marked as documentation, not inspection.

## 1. Licence evidence (exact wording)

| Source | Exact wording | Read on |
|---|---|---|
| Zenodo record 21744694, API metadata (`https://zenodo.org/api/records/21744694`) | `"license": {"id": "cc-by-4.0"}`, `"access_right": "open"`, `"rights": null`, `"notes": null`; description: "This is the source code and data of ARMOR."; one file `ARMOR.zip` (174,195,955 B, md5 `b0f1068c2a32d76f30630c6217cbaab0`), which contains `ARMOR/data/D2.zip` (93,883,410 B) | 2026-10-28 |
| ARMOR README (inside the Zenodo archive) | Describes D2 ("collected from the management system of a top-tier commercial bank … D2 has been used in the International AIOps Challenge 2021"). It contains **no licence or NDA statement**. | local copy |
| ART README (`raw.githubusercontent.com/bbyldebb/ART/HEAD/README.md`, line 11), the original release of the same preprocessed D2 | "Dataset D2 is collected from the management system of a top-tier commercial bank … D2 has been used in the International AIOps Challenge 2021 https://aiops-challenge.com. **Due to the non-disclosure agreement, we cannot make it publicly available.**" | 2026-10-28 |
| ART repository metadata (GitHub API) | `"license": null` (no licence file). The repository nevertheless contains `data/D2.zip` (93,883,410 B). | 2026-10-28 |
| AIOps Challenge 2021 (original data owner/organiser) | No published terms of use for the 2021 bank dataset found. A web search returned only the 2020 challenge data, whose licence restricts use to "non-commercial purposes such as scientific research or classroom teaching" ([AIOps-Challenge-2020-Data](https://github.com/NetManAIOps/AIOps-Challenge-2020-Data)). | 2026-10-28 |

## 2. Finding

The licence is **unclear**:
- The Zenodo record declares CC-BY-4.0 at **record level**. It covers the single file `ARMOR.zip`, so nominally it covers the D2 bytes inside. No per-file rights statement or exception exists.
- The upstream release of the same data (ART, an overlapping research line; the ART repo also ships `D2.zip`) states that D2 is under a **non-disclosure agreement** and "cannot [be made] publicly available".
- D2 originates from a commercial bank's system via the AIOps Challenge 2021. Neither the ARMOR nor the ART authors document that the data owner granted redistribution rights. A depositor-declared CC-BY licence cannot grant rights the depositor may not hold.
- The project's existing decision (298-43; `team_handoff_readiness_audit.md`:147; `fetch_d1.py`) already defers D2 for this conflict.

Per the Part 2 rule ("if the licence is unclear, do not download"), D2 was not downloaded and its schema was not inspected. To settle it, ask the ARMOR authors (Zenodo creators: Qian, Zhao, Wang, Gao, Chen, Ling, Deng) and/or the AIOps Challenge 2021 organisers whether D2 may be used for academic evaluation, and record the answer before any download.

## 3. What is documented about D2 (not inspected)

From ARMOR's README and `ARMOR/config/D2.yaml` (on disk since the D1 fetch), and the Zenodo/ART file listings:

| Item | Documented value | Status |
|---|---|---|
| Instances | 18 ("microservices, servers, databases, and dockers"); `model_param.instance_dim: 18` | README / config |
| Node types, pods/VMs, placement edges | Not documented per type. ARMOR's README says graphs come from "call relationships and deployment information" for both datasets. Whether D2 has container-to-server (placement) edges is **unknown** without inspection. | UNVERIFIED |
| Modality channels | `feat_span`: metric `[0, 255]` (256), trace `[256, 257]` (2), log `[258, 265]` (8); 266 channels per instance | config |
| Sampling interval | Not stated for D2. The D2 config uses the same `window_size: 6`. | UNVERIFIED |
| Failure cases | Not stated. Labels documented as `cases/ad_cases.pkl` (timestamps) and `cases/cases.csv` (time, level, root cause, type). | README |
| Fault classes | 6: Memory, CPU, Network, Disk, JVM-Memory, JVM-CPU (D1 has 5, with different definitions) | README |
| Train/test split | `train_samples.pkl` / `test_samples.pkl` exist by README layout; ARMOR's downstream split `split_ratio: 0.6` (AD, FT, RCL), as for D1 | README / config |
| ARMOR released D2 results | `res/D2/20260114_222628/`: `scenarios/full/tmp/pre_interval.json` 156 B, `res.json` 194 B, `model.pkl` 4,643,092 B | Zenodo listing |
| ARMOR published D2 numbers | AD P/R/F1 0.993 / 1.0 / 0.997; FT weighted P/R/F1 0.882 / 0.870 / 0.869; RCL Top1 / Top3 / AVG@5 0.815 / 0.907 / 0.893 (arXiv 2603.25538, Table 1) | paper |

## 4. Can each v2 change be tested on D2?

These are judgements from documentation only. Each needs confirmation by schema inspection once the licence is settled.

| v2 change | Testable on D2? | Reason |
|---|---|---|
| Placement-aware inputs | **Unknown.** Only if D2's graphs contain container-to-server edges. | D2 has "servers" and "dockers" among 18 instances and graphs "from … deployment information", but no placement edge has been seen. With 18 instances the placement structure is much smaller than D1's 40 pods / 6 VMs. |
| Larger calibration pool | **Yes, likely.** | It needs only a time-ordered stretch of normal samples before the evaluation period. D2 has a sample timeline and ARMOR's 60% split, as D1 does. |
| Modality dropout | **Yes, likely.** | D2 has metric, trace and log spans (256 / 2 / 8 channels). Trace has only 2 channels, so trace-drop tests are weak. |
| Whole-VM / host dropout | **Unknown.** It depends on which of the 18 instances are hosts (servers) and whether host metrics exist separately. | Not documented. |

General caveat: D2 is a different system with 6 fault classes and a different instance set. v2 must be fitted and calibrated on D2's training portion and confirmed on D2's held-out portion under a frozen one-shot rule (`V2_PROTOCOL_NOTE.md`). D2 can confirm the v2 *method*, not a D1-trained model.
