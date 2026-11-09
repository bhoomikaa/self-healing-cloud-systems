# D1 pod-to-VM placement validity (Linear 298-51, pre-v2)

Status: **read-only analysis.** It uses the D1 sample graphs (non-executing `pickle_symbolic` / `torch_blob` readers; `pickle.load` is never used) and, for timing only, the case injection timestamps. No model was run. This is a pre-v2 check; any v2 built on it falls under `V2_PROTOCOL_NOTE.md`.

Scripts (in this directory):
- `placement_timeline.py`: per-sample placement. It writes `timeline.json`, which is not committed because it is derived per-sample data.
- `timing.py` → `placement_timing.json`
- `partial_cases.py` and `disk_rate.py`: the partial-sample checks

Times are UTC.

## Summary
- **Placement changes:** the three changes all fall inside 24-hour gaps in the data, so their exact times are unknown. No injection falls inside any gap. One injection lies 1,135 s before the start of the first gap, so the 30-minute flag cannot be fully excluded for change 1.
- **Edge source:** the pod-to-VM edges are not derived from root causes or labels. They are stored per sample in the sample files.
- **Partial samples are fault-correlated:** all 6 samples with missing edges occur 57–203 s after a Node Disk Space Consumption injection. Raw per-minute edges therefore carry a fault signal and must not be used directly.
- **Rule:** per-window causal carry-forward of the last complete placement. In D1 this equals a fixed per-day map for every window.

## 1. Placement changes vs case injection times

D1 has 7,200 samples covering five recorded days, in UTC+8: May 1, 3, 5, 7 and 9. Within a day they are 60 s apart; between days there are four gaps of 86,460 s. There are 3 distinct complete placements:
- **A:** node-5 hosts 6 pods, node-6 hosts 34. Used on May 1 and May 9.
- **B:** node-5 hosts 33 pods, node-6 hosts 7. Used on May 3 and May 5.
- **C:** node-3 hosts 11 pods, node-4 hosts 1, node-5 hosts 28. Used on May 7.

Within every recorded day the complete placement never changes. So **every change falls inside a data gap**: its exact time is unknown, bracketed by the last sample before and the first sample after.

| Change | Bracket (last sample before → first after) | Pods moved | Min distance, injection ↔ bracket (cases.csv / ad_cases) | Median distance | Nearest injection before the change | Injections inside the bracket | Within 30 min after an injection? |
|---|---|---|---|---|---|---|---|
| A → B | 2022-05-01 15:59 → 2022-05-02 16:00 | 29 | 1,135 s / 1,135 s | 213,541 s / 231,550 s | 1,135 s before the bracket start | 0 | **Cannot be excluded:** an injection at 15:40:05 is 1,135 s before the last pre-gap minute, so a change in the first 665 s of the gap would be within 30 min. |
| B → C | 2022-05-05 15:59 → 2022-05-06 16:00 | 15 | 3,823 s / 3,823 s | 199,870 s / 197,867 s | 3,823 s | 0 | No |
| C → A | 2022-05-07 15:59 → 2022-05-08 16:00 | 38 | 1,380 s / 1,380 s (after the bracket) | 218,400 s / 200,390 s | 3,694 s | 0 | No |

Distances are 0 for an injection inside the bracket; here none is. "cases.csv" means the 210 case injections and "ad_cases" the 224 AD timestamps.

Reading:
- No injection exists inside any gap, because no data were recorded there.
- Changes 2 and 3 cannot be within 30 min after any recorded injection.
- Change 1 is flagged as **not excludable**. The data cannot show whether it happened in the first 11 minutes of a 24-hour gap.
- The changes move 15–38 pods at once and line up with day boundaries. That pattern fits per-day redeployment better than fault-driven rescheduling, but this is an inference, not proven.

## 2. Where the pod-to-VM edges come from

- **Storage:** the edges are stored in each sample's DGL graph object inside `samples/{train,test}_samples.pkl`, next to that sample's feature tensor. They are read by `placement_timeline.py` from those files only. The labels (`cases.csv`, `ad_cases.pkl`) do not contain them.
- **Not root-cause driven:**
  - node-1 and node-2, which are the root cause of several Node Disk cases, never host a pod.
  - The 6 partial samples (section 4) lose the links of a VM that is **never** the faulted VM: 0 of 6 match the nearest case's root-cause VM.
  - Placement is constant within each recorded day, across many cases with different root causes.
- **Provenance limit:** neither ART nor ARMOR publishes the code that built the graphs. The only statement is the README's "Graphs indicate the topology … generated from call relationships and deployment information". So "derived from deployment telemetry" is the documented claim, not something we can trace in code.

**Leakage verdict:**
- The **complete placement maps** are not derived from labels or root causes. They are safe to use under the rule in section 6.
- The **raw per-minute edges are not safe.** Section 4 shows the 6 edge dropouts are strongly fault-correlated, so a model fed raw edges could learn "missing VM link → Node Disk fault" from a pipeline artefact.

## 3. Graph timestamps and causal use

- Each sample is a `[timestamp, graph, features]` triple, so each graph carries its sample's timestamp. Graphs exist for all 7,200 sample minutes.
- Model 1 windows are 7 consecutive minutes (6 inputs + target) within one recorded day. For every window, the graph at its **first input minute** exists, and so does a complete graph at or before it.
  - Each day's first minute is complete, and partial samples are never at a day start.
- So placement can be taken **strictly from data at or before the window's first input minute.**
- **ARMOR's practice differs.** It attaches the **target minute's** graph to each window (`create_dataloader_AR`: `series_sample[-1][1]`), which includes the 6 partial graphs when they are the target.

## 4. Samples with missing edges (carry-forward feasibility)

| Sample (UTC) | Pods without a VM | Their VM in the prior complete map | Prior complete sample | Present pods agree with prior map | Next complete map equals prior | Nearest case injection |
|---|---|---|---|---|---|---|
| 2022-05-01 02:55 | 6 | node-5 (all its pods) | 02:54 (60 s before) | yes | yes | 57 s after a Node Disk Space Consumption injection (node-1) |
| 2022-05-02 17:30 | 3 | node-6 (3 of its 7) | 17:29 | yes | yes | 92 s after, same type (node-2) |
| 2022-05-02 20:20 | 33 | node-5 (all its pods) | 20:19 | yes | yes | 172 s after, same type (node-1) |
| 2022-05-06 22:18 | 1 | node-4 (its only pod) | 22:17 | yes | yes | 90 s after, same type (node-1) |
| 2022-05-07 11:23 | 1 | node-4 (its only pod) | 11:22 | yes | yes | 104 s after, same type (node-2) |
| 2022-05-08 23:42 | 34 | node-6 (all its pods) | 23:41 | yes | yes | 203 s after, same type (node-3) |

**Carry-forward is feasible for all 6:**
- Each has a complete sample 60 s earlier, with which all its present edges agree.
- The complete placement after it is the same map.

**These samples are fault-correlated:**
- All 6 fall 57–203 s after a **Node Disk Space Consumption** injection. That is 6 of the 14 such cases; no other fault type is followed by a partial sample, and no partial sample occurs elsewhere.
- Only 19.9% of all 7,200 minutes lie within 203 s of any injection, so six of six by chance is about 0.2⁶ ≈ 6×10⁻⁵.
- The missing links belong to a **different** VM from the faulted one, so this looks like a disruption of the collection pipeline caused by the disk fault rather than real placement.

## 5. node-1 and node-2

Confirmed: across all 7,200 samples and all placements, **node-1 and node-2 host 0 pods**. node-3 hosts 11 pods (placement C only), node-4 1 (C only), node-5 up to 37 distinct pods, node-6 up to 35. Faults on node-1 or node-2 can therefore show only in those VMs' own metrics, never through hosted pods.

## 6. Recommended placement rule

**Per-window causal placement with carry-forward:** for each window, use the **last complete placement at or before the window's first input minute**, and never use a partial graph.

- **Why not raw per-minute (or target-minute) edges:** the 6 partial graphs are fault-correlated artefacts (section 4). Using them, as ARMOR does at the target minute, would let a model shortcut on the pipeline.
- **Why not a fixed per-day map:** in D1 it gives identical values for every window, because placement is constant within each recorded day once partial graphs are ignored. But defining it as "the day's map" uses minutes after the window, which is not causal by construction and would not carry over to a live testbed.
- **Why not a lagged placement (e.g. t − k minutes):** it is unnecessary in D1, because no within-day change exists to lag over. Carry-forward is the lag-0 causal version and already excludes the target minute and later. A lag of ≥ 30 min would be the safeguard if a testbed showed placement changes shortly after faults. Section 1 cannot rule this out only for change 1, which sits in a data gap.
- **Fallback:** carry-forward is feasible for every D1 window. No window starts before a complete graph, and every partial sample has a complete one 60 s earlier.
