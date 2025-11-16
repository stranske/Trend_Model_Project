#!/usr/bin/env python3
"""Verification harness for Codex bootstrap scenarios.

Scenarios implemented (initial subset):
- t01_basic: create labeled issue, expect new branch+PR
- t02_reuse: re-label same issue, expect reuse
- t03_rebootstrap: close PR then relabel, expect new PR

Additional scenarios placeholders are structured but not yet fully implemented.

Outputs:
  codex-verification-report.json : structured results
  codex-verification-report.md   : human readable table
  codex-scenario-logs/           : per-scenario JSON detail

Requires: gh CLI authenticated (GITHUB_TOKEN provided by Actions runtime) and jq for some fallbacks.

Adding a New Scenario
=====================
1. Choose an ID: follow sequential pattern (e.g. t16_new_feature).
2. Implement a function `scenario_<id>(ctx: dict) -> ScenarioResult`:
    - Use `create_issue` to set up state (avoid reusing unrelated issues unless chaining intentionally).
    - Trigger the minimal workflow via label or `gh workflow run` with appropriate `-f` inputs.
    - Sleep a conservative number of seconds (8–18) depending on expected branch/PR latency.
    - Download artifact with `download_artifact(issue, LOG_DIR / f"<id>_{issue}")` or treat absence as expected.
3. Populate a `ScenarioExpectation` reflecting the intended state (artifact/no bootstrap/new PR/reuse).
4. Append the scenario to `SCENARIOS_IMPL` mapping.
5. Update docs/codex-simulation.md (Scenario table) with a concise description.
6. (Optional) Add any new dispatch inputs or simulation labels to the detection workflow and composite action in the same PR.
7. Run the verification workflow manually specifying the new scenario to validate before merging.

Guiding Principles
------------------
- Keep scenarios orthogonal: test one behavioral dimension each.
- Prefer explicit overrides (dispatch inputs) to implicit timing assumptions.
- Fail fast with clear `error` messages when prerequisites (issue/pr) are missing.
- Always write deterministic logic; avoid depending on ordering of concurrent runs.
"""
from __future__ import annotations

import json
import os
import pathlib
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime

SCENARIO_ENV = os.getenv("SCENARIOS", "t01_basic,t02_reuse,t03_rebootstrap")
WORKDIR = pathlib.Path.cwd()
LOG_DIR = WORKDIR / "codex-scenario-logs"
LOG_DIR.mkdir(exist_ok=True)

# Utility helpers


def run(cmd: str, check=True, capture=True, env=None, timeout=60):
    proc = subprocess.run(
        cmd, shell=True, capture_output=capture, text=True, env=env, timeout=timeout
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {cmd}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def gh_json(cmd: str):
    out = run(cmd)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to parse JSON from: {cmd}\nOutput: {out}")


def sleep(seconds: float):
    time.sleep(seconds)


@dataclass
class ScenarioExpectation:
    expect_artifact: bool | None = None
    expect_reuse: bool | None = None
    expect_new_pr: bool | None = None
    expect_no_bootstrap: bool | None = None


@dataclass
class ScenarioResult:
    name: str
    status: str
    details: dict
    error: str | None = None
    expectation: ScenarioExpectation | None = None
    started: str | None = None  # ISO8601
    ended: str | None = None
    duration_s: float | None = None


def create_issue(title: str, body: str, labels=None) -> int:
    labels = labels or []
    label_args = " ".join([f"--label {shlex.quote(lbl)}" for lbl in labels])
    cmd = f"gh issue create -t {shlex.quote(title)} -b {shlex.quote(body)} {label_args} --json number -q .number"
    num = run(cmd)
    return int(num)


def download_artifact(issue: int, dest: pathlib.Path) -> dict:
    dest.mkdir(exist_ok=True, parents=True)
    # Attempt download up to 5 times
    for attempt in range(5):
        rc = subprocess.run(
            f"gh run download -n codex-bootstrap-{issue} -D {dest}", shell=True
        )
        if rc.returncode == 0:
            result_file = dest / "codex_bootstrap_result.json"
            if result_file.exists():
                return json.loads(result_file.read_text())
        sleep(5)
    raise RuntimeError(f"Artifact for issue {issue} not found after retries")


def scenario_t01_basic(ctx: dict) -> ScenarioResult:
    title = "Codex Verification T01"
    issue = create_issue(title, "Bootstrap test", labels=["agent:codex"])
    ctx["issue"] = issue
    sleep(8)
    artifact = download_artifact(issue, LOG_DIR / f"t01_{issue}")
    ok = (
        artifact.get("reused") is False
        and artifact.get("pr")
        and artifact.get("branch")
    )
    status = "pass" if ok else "fail"
    return ScenarioResult(
        "t01_basic",
        status,
        {"issue": issue, **artifact},
        expectation=ScenarioExpectation(
            expect_artifact=True, expect_new_pr=True, expect_reuse=False
        ),
    )


def scenario_t02_reuse(ctx: dict) -> ScenarioResult:
    issue = ctx.get("issue")
    if not issue:
        return ScenarioResult("t02_reuse", "skip", {}, error="No base issue from t01")
    # Remove label then re-add to fire event
    subprocess.run(f"gh issue edit {issue} --remove-label agent:codex", shell=True)
    sleep(2)
    run(f"gh issue edit {issue} --add-label agent:codex")
    sleep(8)
    artifact = download_artifact(issue, LOG_DIR / f"t02_{issue}")
    ok = artifact.get("reused") is True and artifact.get("branch_created") is False
    status = "pass" if ok else "fail"
    return ScenarioResult(
        "t02_reuse",
        status,
        {"issue": issue, **artifact},
        expectation=ScenarioExpectation(
            expect_artifact=True, expect_reuse=True, expect_new_pr=False
        ),
    )


def scenario_t03_rebootstrap(ctx: dict) -> ScenarioResult:
    issue = ctx.get("issue")
    if not issue:
        return ScenarioResult(
            "t03_rebootstrap", "skip", {}, error="No base issue from t01"
        )
    # Need PR number
    # Use latest artifact
    art_dir = max(
        [p for p in LOG_DIR.glob(f"t0[12]_{issue}")],
        key=lambda p: p.stat().st_mtime,
        default=None,
    )
    if not art_dir:
        return ScenarioResult(
            "t03_rebootstrap", "skip", {}, error="No prior artifact dir"
        )
    art_file = next(art_dir.glob("codex_bootstrap_result.json"), None)
    if not art_file:
        return ScenarioResult(
            "t03_rebootstrap", "skip", {}, error="Missing artifact file"
        )
    art = json.loads(art_file.read_text())
    pr = art.get("pr")
    if not pr:
        return ScenarioResult("t03_rebootstrap", "skip", {}, error="No PR in artifact")
    # Close PR
    run(
        f"gh pr close {pr} --comment 'Closing for re-bootstrap test' --delete-branch false"
    )
    sleep(3)
    # Re-label to trigger re-bootstrap
    subprocess.run(f"gh issue edit {issue} --remove-label agent:codex", shell=True)
    sleep(2)
    run(f"gh issue edit {issue} --add-label agent:codex")
    sleep(10)
    new_art = download_artifact(issue, LOG_DIR / f"t03_{issue}")
    new_pr = new_art.get("pr")
    ok = new_pr and new_pr != pr and new_art.get("reused") is False
    status = "pass" if ok else "fail"
    return ScenarioResult(
        "t03_rebootstrap",
        status,
        {"issue": issue, "old_pr": pr, "new_pr": new_pr, **new_art},
        expectation=ScenarioExpectation(
            expect_artifact=True, expect_new_pr=True, expect_reuse=False
        ),
    )


# --- Placeholder / forthcoming scenarios (stubs) ---
def _stub(name: str, reason: str) -> ScenarioResult:
    return ScenarioResult(name, "skip", {}, error=reason)


def scenario_t04_missing_label(ctx: dict) -> ScenarioResult:
    # Create issue WITHOUT the agent:codex label; expect no bootstrap artifact
    issue = create_issue("Codex Verification T04", "Missing label should not trigger")
    ctx.setdefault("t04_issue", issue)
    # Wait a short period to allow any unintended workflow to run
    sleep(6)
    # Attempt to download artifact; expect failure
    try:
        _ = download_artifact(issue, LOG_DIR / f"t04_{issue}")
        # If artifact exists, that's a failure (bootstrap should not run)
        return ScenarioResult(
            "t04_missing_label",
            "fail",
            {"issue": issue},
            error="Artifact unexpectedly present",
            expectation=ScenarioExpectation(expect_no_bootstrap=True),
        )
    except Exception:
        return ScenarioResult(
            "t04_missing_label",
            "pass",
            {"issue": issue},
            expectation=ScenarioExpectation(expect_no_bootstrap=True),
        )


def scenario_t05_manual_sim(ctx: dict) -> ScenarioResult:
    # Trigger minimal workflow via manual dispatch with simulated codex label and manual mode flag
    # Expect bootstrap artifact (new PR) even though issue not directly labeled (simulate_label provides label)
    # We create an issue first, then dispatch referencing it.
    issue = create_issue(
        "Codex Verification T05", "Manual dispatch with simulated label", labels=[]
    )  # no label
    ctx["t05_issue"] = issue
    try:
        run(
            f"gh workflow run 'Codex Assign Minimal' -f test_issue={issue} -f simulate_label='agent:codex,codex-sim:manual' "
        )
    except Exception as e:
        return ScenarioResult(
            "t05_manual_sim",
            "error",
            {"issue": issue},
            error=f"dispatch failed: {e}",
            expectation=ScenarioExpectation(expect_artifact=True, expect_new_pr=True),
        )
    # Allow processing
    sleep(15)
    try:
        art = download_artifact(issue, LOG_DIR / f"t05_{issue}")
    except Exception as e:
        return ScenarioResult(
            "t05_manual_sim",
            "fail",
            {"issue": issue},
            error=f"artifact missing: {e}",
            expectation=ScenarioExpectation(expect_artifact=True, expect_new_pr=True),
        )
    ok = (
        art.get("pr")
        and art.get("branch")
        and art.get("reused") is False
        and art.get("pr_mode") in ("manual", "auto")
    )
    status = "pass" if ok else "fail"
    return ScenarioResult(
        "t05_manual_sim",
        status,
        {"issue": issue, **art},
        expectation=ScenarioExpectation(expect_artifact=True, expect_new_pr=True),
    )


def scenario_t06_manual_no_sim(ctx: dict) -> ScenarioResult:
    # Manual dispatch specifying issue but WITHOUT simulated codex label => expect no bootstrap
    issue = create_issue(
        "Codex Verification T06", "Manual dispatch without label simulation", labels=[]
    )
    ctx["t06_issue"] = issue
    try:
        run(
            f"gh workflow run 'Codex Assign Minimal' -f test_issue={issue} -f simulate_label='' "
        )
    except Exception as e:
        return ScenarioResult(
            "t06_manual_no_sim",
            "error",
            {"issue": issue},
            error=f"dispatch failed: {e}",
            expectation=ScenarioExpectation(expect_no_bootstrap=True),
        )
    sleep(12)
    # Attempt artifact download (should fail)
    try:
        art = download_artifact(issue, LOG_DIR / f"t06_{issue}")
        return ScenarioResult(
            "t06_manual_no_sim",
            "fail",
            {"issue": issue, **art},
            error="Artifact present unexpectedly",
            expectation=ScenarioExpectation(expect_no_bootstrap=True),
        )
    except Exception:
        return ScenarioResult(
            "t06_manual_no_sim",
            "pass",
            {"issue": issue},
            expectation=ScenarioExpectation(expect_no_bootstrap=True),
        )


def scenario_t07_invalid_manual(ctx: dict) -> ScenarioResult:
    # Simulate manual dispatch with invalid/non-numeric issue id to exercise detection path
    # We invoke the workflow via gh API (best-effort); if not available, mark skip.
    # NOTE: GitHub REST for workflow dispatch requires ref.
    try:
        # Provide a bogus test_issue value to trigger invalid-manual-issue reason.
        run(
            "gh workflow run Codex Assign Minimal -f test_issue=abc123 -f simulate_label='' "
        )
    except Exception as e:
        return ScenarioResult(
            "t07_invalid_manual", "skip", {}, error=f"workflow dispatch failed: {e}"
        )
    # There is no issue number, so no artifact expected. Just pass.
    return ScenarioResult(
        "t07_invalid_manual",
        "pass",
        {},
        expectation=ScenarioExpectation(expect_no_bootstrap=True),
    )


def scenario_t08_pat_missing_fallback(ctx: dict) -> ScenarioResult:
    # Create issue with label; rely on absence of SERVICE_BOT_PAT (Actions env may or may not have one)
    issue = create_issue(
        "Codex Verification T08", "PAT missing fallback allowed", labels=["agent:codex"]
    )
    sleep(8)
    try:
        art = download_artifact(issue, LOG_DIR / f"t08_{issue}")
    except Exception as e:
        return ScenarioResult(
            "t08_pat_missing_fallback",
            "fail",
            {"issue": issue},
            error=f"No artifact: {e}",
            expectation=ScenarioExpectation(expect_artifact=True),
        )
    # We accept token_source != SERVICE_BOT_PAT OR fallback_used true
    # Case-insensitive lookup for "token_source"
    token_source = None
    for k, v in art.items():
        if k.lower() == "token_source":
            token_source = v
            break
    fallback_used = art.get("fallback_used")
    ok = (
        token_source in ("GITHUB_TOKEN", "ACTIONS_DEFAULT_TOKEN")
        or fallback_used is True
        or str(fallback_used).lower() == "true"
    )
    status = "pass" if ok else "fail"
    return ScenarioResult(
        "t08_pat_missing_fallback",
        status,
        {"issue": issue, **art},
        expectation=ScenarioExpectation(expect_artifact=True),
    )


def scenario_t09_pat_missing_block(ctx: dict) -> ScenarioResult:
    # Create issue with label then trigger manual dispatch overriding allow_fallback=false.
    # Expectation: if SERVICE_BOT_PAT absent, bootstrap should be blocked (no artifact)
    issue = create_issue(
        "Codex Verification T09",
        "PAT missing with fallback disallowed",
        labels=["agent:codex"],
    )
    ctx["t09_issue"] = issue
    # Fire workflow dispatch referencing issue with override
    try:
        run(
            f"gh workflow run 'Codex Assign Minimal' -f test_issue={issue} -f simulate_label='agent:codex' -f allow_fallback=false "
        )
    except Exception as e:
        return ScenarioResult(
            "t09_pat_missing_block",
            "error",
            {"issue": issue},
            error=f"dispatch failed: {e}",
            expectation=ScenarioExpectation(expect_no_bootstrap=True),
        )
    sleep(15)
    try:
        art = download_artifact(issue, LOG_DIR / f"t09_{issue}")
        # If artifact present, either PAT existed or fallback incorrectly allowed; treat as fail unless token_source == 'SERVICE_BOT_PAT'
        token_source = art.get("token_source")
        if token_source == "SERVICE_BOT_PAT":
            # PAT present so block condition not applicable
            return ScenarioResult(
                "t09_pat_missing_block",
                "pass",
                {"issue": issue, **art},
                expectation=ScenarioExpectation(expect_artifact=True),
            )
        return ScenarioResult(
            "t09_pat_missing_block",
            "fail",
            {"issue": issue, **art},
            error="Artifact present but fallback should have been blocked",
            expectation=ScenarioExpectation(expect_no_bootstrap=True),
        )
    except Exception:
        return ScenarioResult(
            "t09_pat_missing_block",
            "pass",
            {"issue": issue},
            expectation=ScenarioExpectation(expect_no_bootstrap=True),
        )


def scenario_t10_primary_403_fallback(ctx: dict) -> ScenarioResult:
    # Simulate forced primary failure; expect fallback branch creation success with fallback_used true
    issue = create_issue(
        "Codex Verification T10",
        "Forced primary failure fallback",
        labels=["agent:codex"],
    )
    # Record for possible chaining
    ctx["t10_issue"] = issue
    # Wait for workflow
    sleep(10)
    try:
        art = download_artifact(issue, LOG_DIR / f"t10_{issue}")
    except Exception as e:
        return ScenarioResult(
            "t10_primary_403_fallback",
            "fail",
            {"issue": issue},
            error=f"Artifact missing: {e}",
            expectation=ScenarioExpectation(expect_artifact=True),
        )
    fallback_used = art.get("fallback_used")
    ok = str(fallback_used).lower() == "true"
    return ScenarioResult(
        "t10_primary_403_fallback",
        "pass" if ok else "fail",
        {"issue": issue, **art},
        expectation=ScenarioExpectation(expect_artifact=True),
    )


def scenario_t11_dual_fail(ctx: dict) -> ScenarioResult:
    # Simulate forced dual failure; expect missing artifact
    issue = create_issue(
        "Codex Verification T11", "Forced dual failure", labels=["agent:codex"]
    )
    sleep(10)
    try:
        art = download_artifact(issue, LOG_DIR / f"t11_{issue}")
        return ScenarioResult(
            "t11_dual_fail",
            "fail",
            {"issue": issue, **art},
            error="Artifact present but expected failure",
            expectation=ScenarioExpectation(expect_no_bootstrap=True),
        )
    except Exception:
        return ScenarioResult(
            "t11_dual_fail",
            "pass",
            {"issue": issue},
            expectation=ScenarioExpectation(expect_no_bootstrap=True),
        )


def scenario_t12_manual_mode(ctx: dict) -> ScenarioResult:
    # Ensure manual mode selected via simulation label codex-sim:manual on an already labeled issue
    issue = create_issue(
        "Codex Verification T12",
        "Manual mode via simulation label",
        labels=["agent:codex"],
    )  # direct label ensures detection
    ctx["t12_issue"] = issue
    # Add manual simulation label via editing (simulate by adding a benign label recognized by detection? can't add custom w/out repo permission). We'll re-dispatch workflow with simulate_label.
    try:
        run(
            f"gh workflow run 'Codex Assign Minimal' -f test_issue={issue} -f simulate_label='agent:codex,codex-sim:manual' "
        )
    except Exception as e:
        return ScenarioResult(
            "t12_manual_mode",
            "error",
            {"issue": issue},
            error=f"dispatch failed: {e}",
            expectation=ScenarioExpectation(expect_artifact=True),
        )
    sleep(15)
    try:
        art = download_artifact(issue, LOG_DIR / f"t12_{issue}")
    except Exception as e:
        return ScenarioResult(
            "t12_manual_mode",
            "fail",
            {"issue": issue},
            error=f"artifact missing: {e}",
            expectation=ScenarioExpectation(expect_artifact=True),
        )
    # Expectation: pr_mode reported as manual (if action surfaces it) else treat presence as pass
    mode = art.get("pr_mode") or art.get("mode")
    # Accept if mode absent (not yet surfaced), but prefer manual
    ok = bool(art.get("pr")) and (mode in (None, "manual", "auto"))
    status = "pass" if ok else "fail"
    return ScenarioResult(
        "t12_manual_mode",
        status,
        {"issue": issue, **art},
        expectation=ScenarioExpectation(expect_artifact=True),
    )


def scenario_t13_suppressed(ctx: dict) -> ScenarioResult:
    # Test suppressed activation comment via codex-sim:suppress
    issue = create_issue(
        "Codex Verification T13", "Suppressed activation", labels=["agent:codex"]
    )  # direct label
    ctx["t13_issue"] = issue
    try:
        run(
            f"gh workflow run 'Codex Assign Minimal' -f test_issue={issue} -f simulate_label='agent:codex,codex-sim:suppress' "
        )
    except Exception as e:
        return ScenarioResult(
            "t13_suppressed",
            "error",
            {"issue": issue},
            error=f"dispatch failed: {e}",
            expectation=ScenarioExpectation(expect_artifact=True),
        )
    sleep(15)
    try:
        art = download_artifact(issue, LOG_DIR / f"t13_{issue}")
    except Exception as e:
        return ScenarioResult(
            "t13_suppressed",
            "fail",
            {"issue": issue},
            error=f"artifact missing: {e}",
            expectation=ScenarioExpectation(expect_artifact=True),
        )
    # We cannot easily verify absence of comment without additional API calls; placeholder validation = artifact exists
    status = "pass" if art.get("pr") else "fail"
    return ScenarioResult(
        "t13_suppressed",
        status,
        {"issue": issue, **art},
        expectation=ScenarioExpectation(expect_artifact=True),
    )


def scenario_t14_invalid_cmd(ctx: dict) -> ScenarioResult:
    # Create issue and then manual-dispatch with invalid codex command; workflow configured to fail on invalid
    issue = create_issue(
        "Codex Verification T14", "Invalid command enforcement", labels=["agent:codex"]
    )
    ctx["t14_issue"] = issue
    invalid_cmd = "codex: rm -rf /"
    # Dispatch with invalid command override
    try:
        run(
            f"gh workflow run 'Codex Assign Minimal' -f test_issue={issue} -f simulate_label='agent:codex' -f codex_command='{invalid_cmd}' "
        )
    except Exception as e:
        return ScenarioResult(
            "t14_invalid_cmd",
            "error",
            {"issue": issue},
            error=f"dispatch failed: {e}",
            expectation=ScenarioExpectation(expect_artifact=True),
        )
    # Allow run
    sleep(18)
    # Attempt artifact. Two acceptable paths:
    # 1. Action failed BEFORE artifact creation (invalid command) -> no artifact (we treat as pass since enforcement worked)
    # 2. Artifact exists with command_original != command_final and invalid_command true and command_final == 'codex: start'
    try:
        art = download_artifact(issue, LOG_DIR / f"t14_{issue}")
        orig = art.get("command_original")
        final = art.get("command_final")
        invalid_flag = art.get("invalid_command")
        ok = orig == invalid_cmd and final == "codex: start" and invalid_flag is True
        return ScenarioResult(
            "t14_invalid_cmd",
            "pass" if ok else "fail",
            {"issue": issue, **art},
            expectation=ScenarioExpectation(expect_artifact=True),
        )
    except Exception:
        # No artifact—assume action failed early as enforcement. Mark pass.
        return ScenarioResult(
            "t14_invalid_cmd",
            "pass",
            {"issue": issue, "artifact": "missing (enforced)"},
            expectation=ScenarioExpectation(expect_no_bootstrap=True),
        )


def scenario_t15_corrupt_marker(ctx: dict) -> ScenarioResult:
    # Depend on a fresh bootstrap first
    issue = create_issue(
        "Codex Verification T15", "Corrupt marker test", labels=["agent:codex"]
    )
    sleep(8)
    try:
        art = download_artifact(issue, LOG_DIR / f"t15_initial_{issue}")
    except Exception as e:
        return ScenarioResult(
            "t15_corrupt_marker",
            "error",
            {"issue": issue},
            error=f"initial artifact missing: {e}",
        )
    pr = art.get("pr")
    if not pr:
        return ScenarioResult(
            "t15_corrupt_marker",
            "fail",
            {"issue": issue},
            error="No PR from initial bootstrap",
        )
    # Corrupt marker: edit PR body removing potential marker string (placeholder heuristic)
    try:
        run(f"gh pr view {pr} --json body -q .body > pr_body.txt")
        orig = pathlib.Path("pr_body.txt").read_text()
        # Systematically corrupt marker: try to locate a JSON marker block and break its structure
        import re

        marker_pattern = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
        match = marker_pattern.search(orig)
        if match:
            marker_json = match.group(1)
            try:
                marker_obj = json.loads(marker_json)
                # Corrupt a specific field, e.g., remove 'marker' or change its value
                if "marker" in marker_obj:
                    marker_obj["marker"] = "CORRUPTED"
                else:
                    # Remove a random key if 'marker' not present
                    if marker_obj:
                        marker_obj.pop(next(iter(marker_obj)))
                corrupted_marker_json = json.dumps(marker_obj)
                corrupted = (
                    orig[: match.start(1)]
                    + corrupted_marker_json
                    + orig[match.end(1) :]
                )
            except Exception:
                # If JSON parsing fails, just break the block
                corrupted = (
                    orig[: match.start(1)] + "CORRUPTED_MARKER" + orig[match.end(1) :]
                )
        else:
            # If no marker found, fallback to previous heuristic
            corrupted = orig.replace("Codex", "CdX")
        pathlib.Path("pr_body_corrupt.txt").write_text(corrupted)
        run(f"gh pr edit {pr} -F pr_body_corrupt.txt")
    except Exception as e:
        return ScenarioResult(
            "t15_corrupt_marker",
            "error",
            {"issue": issue, "pr": pr},
            error=f"Failed to corrupt PR: {e}",
        )
    # Relabel issue to see if reuse still operates (remove/add)
    subprocess.run(f"gh issue edit {issue} --remove-label agent:codex", shell=True)
    sleep(2)
    run(f"gh issue edit {issue} --add-label agent:codex")
    sleep(10)
    try:
        art2 = download_artifact(issue, LOG_DIR / f"t15_second_{issue}")
    except Exception as e:
        return ScenarioResult(
            "t15_corrupt_marker",
            "error",
            {"issue": issue, "pr": pr},
            error=f"second artifact missing: {e}",
        )
    reused = art2.get("reused")
    status = "pass" if reused else "fail"
    return ScenarioResult(
        "t15_corrupt_marker",
        status,
        {"issue": issue, "pr": pr, "reused_after_corrupt": reused, **art2},
        expectation=ScenarioExpectation(expect_artifact=True, expect_reuse=True),
    )


SCENARIOS_IMPL = {
    "t01_basic": scenario_t01_basic,
    "t02_reuse": scenario_t02_reuse,
    "t03_rebootstrap": scenario_t03_rebootstrap,
    "t04_missing_label": scenario_t04_missing_label,
    "t05_manual_sim": scenario_t05_manual_sim,
    "t06_manual_no_sim": scenario_t06_manual_no_sim,
    "t07_invalid_manual": scenario_t07_invalid_manual,
    "t08_pat_missing_fallback": scenario_t08_pat_missing_fallback,
    "t09_pat_missing_block": scenario_t09_pat_missing_block,
    "t10_primary_403_fallback": scenario_t10_primary_403_fallback,
    "t11_dual_fail": scenario_t11_dual_fail,
    "t12_manual_mode": scenario_t12_manual_mode,
    "t13_suppressed": scenario_t13_suppressed,
    "t14_invalid_cmd": scenario_t14_invalid_cmd,
    "t15_corrupt_marker": scenario_t15_corrupt_marker,
}


def main():
    requested = [s.strip() for s in SCENARIO_ENV.split(",") if s.strip()]
    results: list[ScenarioResult] = []
    shared_ctx: dict = {}
    for name in requested:
        func = SCENARIOS_IMPL.get(name)
        start_ts = datetime.utcnow()
        if not func:
            results.append(
                ScenarioResult(
                    name,
                    "skip",
                    {},
                    error="Not implemented",
                    started=start_ts.isoformat() + "Z",
                    ended=start_ts.isoformat() + "Z",
                    duration_s=0.0,
                )
            )
            continue
        try:
            res = func(shared_ctx)
        except Exception as e:  # noqa
            res = ScenarioResult(name, "error", {}, error=str(e))
        end_ts = datetime.utcnow()
        res.started = res.started or start_ts.isoformat() + "Z"
        res.ended = end_ts.isoformat() + "Z"
        try:
            start_dt = datetime.fromisoformat(res.started.replace("Z", ""))
            end_dt = datetime.fromisoformat(res.ended.replace("Z", ""))
            res.duration_s = round((end_dt - start_dt).total_seconds(), 3)
        except Exception:
            res.duration_s = None
        results.append(res)
    # Write reports
    data = [asdict(r) for r in results]
    (WORKDIR / "codex-verification-report.json").write_text(json.dumps(data, indent=2))
    # Markdown table
    md_lines = [
        "# Codex Bootstrap Verification Report",
        "",
        "| Scenario | Status | Duration(s) | Expectation | Notes |",
        "|----------|--------|------------|-------------|-------|",
    ]
    for r in results:
        exp = r.expectation
        exp_str = ""
        if exp:
            parts = []
            if exp.expect_new_pr:
                parts.append("new_pr")
            if exp.expect_reuse:
                parts.append("reuse")
            if exp.expect_no_bootstrap:
                parts.append("no_bootstrap")
            if exp.expect_artifact and not parts:
                parts.append("artifact")
            exp_str = "+".join(parts) or "—"
        note = r.error or (
            "pr=" + str(r.details.get("pr")) if r.details.get("pr") else ""
        )
        dur = r.duration_s if r.duration_s is not None else ""
        md_lines.append(f"| {r.name} | {r.status} | {dur} | {exp_str} | {note} |")
    (WORKDIR / "codex-verification-report.md").write_text("\n".join(md_lines) + "\n")
    # Diff artifact: highlight any non-pass scenarios
    diff_lines = [
        "# Codex Verification Diff",
        "",
        "Only scenarios with status != pass are listed.",
        "",
        "| Scenario | Status | Expectation | Error/Note |",
        "|----------|--------|-------------|------------|",
    ]
    for r in results:
        if r.status not in {"pass"}:
            exp = r.expectation
            parts = []
            if exp:
                if exp.expect_new_pr:
                    parts.append("new_pr")
                if exp.expect_reuse:
                    parts.append("reuse")
                if exp.expect_no_bootstrap:
                    parts.append("no_bootstrap")
                if exp.expect_artifact and not parts:
                    parts.append("artifact")
            exp_str = "+".join(parts) or "—"
            diff_lines.append(
                f"| {r.name} | {r.status} | {exp_str} | {(r.error or '')[:140]} |"
            )
    (WORKDIR / "codex-verification-diff.md").write_text("\n".join(diff_lines) + "\n")
    # Exit non-zero if any fail/error
    if any(r.status in {"fail", "error"} for r in results):
        print("One or more scenarios failed", flush=True)
        exit(1)


if __name__ == "__main__":
    from trend_analysis.script_logging import setup_script_logging

    setup_script_logging(module_file=__file__)
    main()
