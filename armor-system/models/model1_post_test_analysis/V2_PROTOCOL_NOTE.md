# Model 1 v2: protocol note, written before any v2 work (Linear 298-51)

Status: written and committed before any v2 design, code, training or data access. It records how v2 results may be reported, not what v2 is.

1. **Where the v2 idea comes from.** The placement idea (pod-to-VM placement-aware inputs), together with a larger calibration pool and modality/VM dropout, came from **post-test error analysis on the D1 test period** (`POST_TEST_ANALYSIS.md`, commit `bcaf296`). After that analysis, the D1 test period (84 cases, 98 official labels) has been seen case by case and is no longer independent of v2's design.
2. **D1 test reporting.** Any v2 result on the D1 test set may be reported only as **"post-hoc, non-independent"**. It may not be presented as a confirmation, a held-out result, or a comparison with ARMOR's or v1's test numbers on equal terms.
3. **Where confirmation must come from.** v2 can be confirmed only on an untouched evaluation set:
   - **D2**, if its licence clearly permits our use and it is suitable for the v2 changes; otherwise
   - **new testbed incidents** collected after this note.

   That set must be frozen, with a one-shot rule like v1's TEST_PLAN, before any v2 result on it is seen.
4. **What stays as it is.** v1's committed test results (`3ed6e97`) remain the independent test result for Model 1. Nothing in v2 changes them.
