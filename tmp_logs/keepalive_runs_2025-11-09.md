# Keepalive Gate Investigation — 2025-11-09

This note captures the workflow executions we have already inspected during the current debugging session so we can reference them without re-querying GitHub Actions.

| Run ID | Job ID | PR | Trigger | Outcome | Gate Reason | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 19203547405 | 54895441501 | #3410 | issue_comment | skipped | keepalive-label-missing | `tmp_logs/run_19203547405.log` lines 170-210 show gate failing on missing label; detection summary mirrors reason and no activation comment captured. |
| 19203264620 | 54894794713 | #3409 | workflow_run (gate re-check) | skipped | missing-round | `tmp_logs/run_19203264620.log` lines 780-840 confirm gate ok but detection fails with `missing-round` even after refreshed comment payload. |
| 19202939357 | 54893995070 | #3407 | issue_comment | skipped | keepalive-label-missing | Behavior consistent with 192035 (gate short-circuits before detection). Re-log capture still pending if deeper evidence required. |
| 19202170858 | 54891730120 | #3403 | issue_comment | skipped | gate-not-success | `tmp_logs/run_19202170858.log` lines 170-240 capture gate reporting `gate-not-success` and `gate_concluded=false`, preventing detection from running. |
| 19200993841 | 54888899837 | #3400 | issue_comment | skipped | gate-not-success | Historical reference; logs not yet re-pulled this session. |
| 19172856484 | 54809785458 | #3366 | issue_comment | dispatched → manual-round | ok | `tmp_logs/run_19172856484.log` lines 170-270 show gate succeeded and detection hit manual-round guard after markers present. |

## Follow-ups

- [x] Capture the job log for `run 19203264620` (gate-triggered path) to document the missing-round condition alongside the issue_comment skip.
- [x] Locate an earlier run where `detect keepalive round comments` completed with `ok=true` so we can compare the full detection + marker workflow against the current failures. → Run 19172856484 (job 54809785458, PR #3366, 2025-11-07 15:23 UTC).
- [ ] Once a successful run is identified, extract:
  - Activation comment author and text
  - Whether autopatch inserted round markers
  - Step outputs for `has_keepalive_label`, `has_human_activation`, and `round` trace metadata
- [ ] Re-run the workflow to capture new autopatch diagnostics added in `agents_pr_meta_keepalive.js` (logs should reveal why round markers remain missing on workflow_run triggers).
- [ ] Confirm the new human-activation diagnostics in `keepalive_gate.js` report which comment/author satisfied the gate (or explicitly log when none were found).

## Current Hypothesis

Every inspected run since 00:52 UTC fails before the detection step because the gate denies dispatch (either due to missing labels or a gate-not-success state). To make an apples-to-apples comparison we still need a job where the gate passed and detection inserted round markers successfully.

## Regression Signals (ranked)

| Rank | Signal | Working Run (PR #3366 / 19172856484) | Current Failures (Run IDs noted above) | Supporting Evidence | Why it matters |
| --- | --- | --- | --- | --- | --- |
| 1 | Keepalive label presence | `keepalive label: true` in gate summary (`tmp_logs/run_19172856484.log` lines 244-258). | Runs 19203547405 & 19202939357 report `keepalive label: false` and exit (`tmp_logs/run_19203547405.log` lines 200-214). | `.github/scripts/keepalive_gate.js` still hard-requires label before any pending-gate logic executes. | Without the label the workflow never surfaces the activation comment, so detection cannot progress. |
| 2 | Gate success criteria | Legacy run shows `gate concluded: true` and `reason=ok` (same log slice). | Run 19202170858 captures `gate-not-success` with `gate_concluded=false` (`tmp_logs/run_19202170858.log` lines 170-210). | New gate code introduces `requireGateSuccess` / `allowPendingGate` flags (diff vs b1d710f2). | Additional gate sensitivity means transient gate failures now block keepalive outright. |
| 3 | Activation comment propagation | Gate returned a fully populated activation comment object (see `ACTIVATION_COMMENT` env in working run). | Run 19203547405 leaves `ACTIVATION_COMMENT` empty because gate exited early; run 19203264620 populates it but detection later flags missing round. | Workflow now sets `if: steps.gate.outputs.activation_comment != ''` before detection, so missing payload skips detection entirely. | Without comment metadata the detection script cannot re-fetch or patch markers, preventing rocket reaction dedupe. |
| 4 | Round marker handling | Success path hits manual-round guard after detecting existing markers (`tmp_logs/run_19172856484.log` lines 232-268). | Run 19203264620 logs `missing-round` even though gate returns activation comment (`tmp_logs/run_19203264620.log` lines 800-860). | `agents_pr_meta_keepalive.js` still relies on markers or autopatch; no fallback for clean instruction without marker in later rounds. | Indicates autopatch path may not fire during workflow_run trigger, leaving comments untagged. |
| 5 | Gate short-circuit inside detection | Legacy script had no dependency on gate status once detection ran. | Current script short-circuits when `GATE_OK` false, emitting `gate-blocked:*` reasons (new branch around line 210 in `.github/scripts/agents_pr_meta_keepalive.js`). | Diff vs b1d710f2 shows newly added gate guard using env vars from workflow. | Detection can now fail even when markers present if gate outputs a non-ok reason (e.g., pending or failure). |
| 6 | Gate run selection heuristic | Old `fetchGateStatus` picked the newest run by timestamp; working run matched the gate that succeeded. | After refactor we rank by status/conclusion, but failing run 19202170858 still reported `gate_not_success`, implying a non-success gate outranked earlier success. | New ranking logic (status/conclusion scoring) in `keepalive_gate.js` may prefer in-progress/failed runs over older successes. | A mis-ranked gate run can block keepalive even if a prior success exists. |
| 7 | Human activation guard breadth | Working run evaluated human activation because label present but didn't require it once `agents:activated` missing; detection still proceeded. | Current gate evaluates human activation whenever alias list non-empty (lines 424-434 in new script). If `requireHumanActivation` true and no fresh comment, gate fails with `no-human-activation`. | Diff shows guard broadened to always fetch latest human activation, even post-activation. | Heightened guard increases odds of `no-human-activation` blocking first orchestration compared to legacy behaviour. |
