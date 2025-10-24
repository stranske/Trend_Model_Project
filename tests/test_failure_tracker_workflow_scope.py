"""Regression tests for the failure tracker delegation pipeline."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, cast

import pytest

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKER_PATH = (
    REPO_ROOT / ".github" / "workflows" / "maint-47-check-failure-tracker.yml"
)
POST_CI_PATH = REPO_ROOT / ".github" / "workflows" / "maint-46-post-ci.yml"

pytestmark = pytest.mark.skipif(
    not POST_CI_PATH.exists(),
    reason="Maint 46 post-CI workflow retired; Gate summary now owns status reporting.",
)
GH_SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"


def _load_workflow(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _get_step(job: dict, name: str) -> dict:
    for step in job.get("steps", []):
        if step.get("name") == name:
            return step
    raise AssertionError(f"Step {name!r} not found in workflow")


def _helper_abs_path(name: str) -> str:
    helper_path = GH_SCRIPTS_DIR / name
    assert helper_path.exists(), f"Expected helper script {name} to exist"
    return helper_path.resolve().as_posix()


def _load_helper(name: str) -> str:
    helper_path = GH_SCRIPTS_DIR / name
    assert helper_path.exists(), f"Expected helper script {name} to exist"
    return helper_path.read_text(encoding="utf-8")


def _get_tracker_helper_source() -> str:
    return _load_helper("maint-post-ci.js")


def _rewrite_helper_import(script: str, helper_name: str = "maint-post-ci.js") -> str:
    relative = f"./.github/scripts/{helper_name}"
    absolute = _helper_abs_path(helper_name)
    return script.replace(f"'{relative}'", f"'{absolute}'").replace(
        f'"{relative}"', f'"{absolute}"'
    )


def _get_tracker_script() -> str:
    workflow = _load_workflow(POST_CI_PATH)
    job = workflow["jobs"]["failure-tracker"]
    tracker_step = _get_step(job, "Derive failure signature & update tracking issue")
    script = tracker_step.get("with", {}).get("script")
    assert isinstance(script, str), "Expected inline tracker script to be defined"
    return script


def _get_success_resolution_script() -> str:
    workflow = _load_workflow(POST_CI_PATH)
    job = workflow["jobs"]["failure-tracker"]
    resolve_step = _get_step(job, "Resolve failure issue for recovered PR")
    script = resolve_step.get("with", {}).get("script")
    assert isinstance(
        script, str
    ), "Expected success-path resolution script to be defined"
    return script


def _get_post_comment_script() -> str:
    workflow = _load_workflow(POST_CI_PATH)
    job = workflow["jobs"]["post-comment"]
    comment_step = _get_step(job, "Upsert consolidated PR comment")
    script = comment_step.get("with", {}).get("script")
    assert isinstance(script, str), "Expected comment upsert script to be defined"
    return script


def _run_post_comment_harness(
    tmp_path: Path, node_path: str, scenario: dict[str, object]
) -> list[dict[str, Any]]:
    script = _get_post_comment_script()
    pr_value = scenario.get("pr")
    assert isinstance(pr_value, int), "Scenario must provide integer PR number"
    script = script.replace("'${{ needs.context.outputs.pr }}'", f"'{pr_value}'")
    script_b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    scenario_json = json.dumps(scenario)
    harness = textwrap.dedent(
        f"""
        const actionsLog = [];
        function log(type, payload) {{
          actionsLog.push({{ type, ...(payload ? {{ payload }} : {{}}) }});
        }}

        const scenario = {scenario_json};
        const fs = require('fs');
        fs.writeFileSync('maint_post_ci_comment.md', scenario.body, 'utf8');

        const github = {{
          rest: {{
            issues: {{
              async listComments(args) {{
                log('listComments', args);
                return {{ data: scenario.comments }};
              }},
              async updateComment(args) {{
                log('updateComment', args);
                return {{ data: {{}} }};
              }},
              async createComment(args) {{
                log('createComment', args);
                return {{ data: {{}} }};
              }},
            }},
          }},
          paginate: async (fn, args) => {{
            log('paginate', args);
            const response = await fn(args);
            return response.data;
          }},
        }};

        const context = {{ repo: {{ owner: scenario.owner, repo: scenario.repo }} }};
        const core = {{
          info(message) {{ log('coreInfo', {{ message }}); }},
          warning(message) {{ log('coreWarning', {{ message }}); }},
        }};

        (async () => {{
          const scriptSource = Buffer.from('{script_b64}', 'base64').toString('utf8');
          const vm = require('vm');
          const sandbox = {{
            github,
            context,
            core,
            require,
            console,
            process,
            Buffer,
            setTimeout,
            setInterval,
            clearTimeout,
            clearInterval,
          }};
          const runner = new vm.Script('(async () => {{' + scriptSource + '\\n}})();');
          await runner.runInNewContext(sandbox);
          console.log(JSON.stringify(actionsLog));
        }})().catch((error) => {{
          console.error(error.stack || String(error));
          process.exit(1);
        }});
        """
    )

    script_path = tmp_path / "post_comment_harness.js"
    script_path.write_text(harness, encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("GITHUB_WORKSPACE", str(REPO_ROOT))

    result = subprocess.run(
        [node_path, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout.strip()
    assert output, "Expected harness output to contain action log JSON"
    return json.loads(output)


def test_legacy_tracker_workflow_removed() -> None:
    assert not TRACKER_PATH.exists(), "Legacy Maint 47 shell should be removed"


def test_maint_post_ci_listens_to_gate_run() -> None:
    workflow = _load_workflow(POST_CI_PATH)

    trigger = workflow.get("on")
    assert trigger is not None, "Maint 46 Post CI should define workflow_run trigger"

    workflow_run = trigger.get("workflow_run")
    assert (
        workflow_run is not None
    ), "Maint 46 Post CI should listen to workflow_run events"

    assert workflow_run.get("types") == ["completed"]
    assert workflow_run.get("workflows") == ["Gate"]


def test_post_ci_concurrency_uses_pr_number_lock() -> None:
    workflow = _load_workflow(POST_CI_PATH)

    concurrency = workflow.get("concurrency") or {}
    group = str(concurrency.get("group") or "")
    group_flat = " ".join(group.split())

    assert "maint-46-post-ci-" in group_flat, "Concurrency group should be namespaced"
    assert (
        "pull_requests[0].number" in group_flat
    ), "Concurrency must prioritise the PR number key"
    assert (
        concurrency.get("cancel-in-progress") is True
    ), "Post CI runs should cancel in-progress duplicates"


def test_context_exposes_failure_tracker_skip_for_legacy_prs() -> None:
    workflow = _load_workflow(POST_CI_PATH)
    context_job = workflow["jobs"]["context"]

    outputs = context_job.get("outputs", {})
    assert (
        outputs.get("failure_tracker_skip")
        == "${{ steps.info.outputs.failure_tracker_skip }}"
    )

    info_step = _get_step(context_job, "Resolve workflow context")
    script = info_step.get("with", {}).get("script", "")
    assert "require('./.github/scripts/maint-post-ci.js')" in script

    helper_source = _get_tracker_helper_source()
    assert "new Set([10, 12])" in helper_source


def test_post_ci_failure_tracker_handles_failure_path() -> None:
    workflow = _load_workflow(POST_CI_PATH)
    job = workflow["jobs"]["failure-tracker"]
    condition = " ".join(job.get("if", "").split())
    assert "needs.context.outputs.found == 'true'" in condition
    assert "needs.context.outputs.failure_incomplete != 'true'" in condition
    assert "needs.context.outputs.failure_tracker_skip != 'true'" in condition
    assert "workflow_run.event == 'pull_request'" in condition

    tracker_step = _get_step(job, "Derive failure signature & update tracking issue")
    assert (
        tracker_step["if"].strip()
        == "github.event.workflow_run.conclusion == 'failure'"
    )

    summary_step = _get_step(job, "Emit failure summary")
    assert (
        summary_step["if"].strip()
        == "github.event.workflow_run.conclusion == 'failure'"
    ), "Failure summary should only emit on failing runs"

    label_step = _get_step(job, "Label pull request as ci-failure")
    assert label_step["uses"].startswith("actions/github-script@")
    label_condition = label_step.get("if", "").strip()
    assert "github.event.workflow_run.conclusion == 'failure'" in label_condition
    label_script = label_step.get("with", {}).get("script", "")
    assert "'ci-failure'" in label_script
    assert "github.rest.issues.addLabels" in label_script

    tracker_script = tracker_step.get("with", {}).get("script", "")
    assert "require('./.github/scripts/maint-post-ci.js')" in tracker_script

    helper_source = _get_tracker_helper_source()
    assert "github.rest.issues.update({" in helper_source
    assert "github.rest.issues.createComment({" in helper_source

    artifact_steps = [
        step
        for step in job.get("steps", [])
        if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert len(artifact_steps) == 2, "Failure and success paths should each upload once"

    seen_conditions: set[str] = set()
    for step in artifact_steps:
        with_section = step.get("with", {})
        assert with_section.get("name") == "ci-failures-snapshot"
        assert with_section.get("path") == "artifacts/ci_failures_snapshot.json"

        condition = " ".join(step.get("if", "").split())
        if condition:
            seen_conditions.add(condition)

    assert (
        "github.event.workflow_run.conclusion == 'failure'" in seen_conditions
    ), "Failure artifact upload should be gated on failing runs"
    assert (
        "github.event.workflow_run.conclusion == 'success'" in seen_conditions
    ), "Success artifact upload should mirror failure artifact payload"


def test_post_ci_failure_tracker_handles_success_path() -> None:
    workflow = _load_workflow(POST_CI_PATH)
    job = workflow["jobs"]["failure-tracker"]

    heal_step = _get_step(job, "Auto-heal stale failure issues & note success")
    assert (
        heal_step["if"].strip() == "github.event.workflow_run.conclusion == 'success'"
    )

    remove_label_step = _get_step(job, "Remove ci-failure label from pull request")
    assert remove_label_step["uses"].startswith("actions/github-script@")
    remove_condition = remove_label_step.get("if", "").strip()
    assert "github.event.workflow_run.conclusion == 'success'" in remove_condition
    remove_script = remove_label_step.get("with", {}).get("script", "")
    assert "ci-failure" in remove_script
    assert "github.rest.issues.removeLabel" in remove_script


def test_success_path_resolves_tracked_pr_issue() -> None:
    workflow = _load_workflow(POST_CI_PATH)
    job = workflow["jobs"]["failure-tracker"]

    resolve_step = _get_step(job, "Resolve failure issue for recovered PR")
    condition = " ".join(resolve_step.get("if", "").split())
    assert "needs.context.outputs.pr" in condition
    assert "workflow_run.conclusion == 'success'" in condition
    assert resolve_step.get("uses", "").startswith("actions/github-script@")

    script = resolve_step.get("with", {}).get("script", "")
    assert "require('./.github/scripts/maint-post-ci.js')" in script

    helper_source = _get_tracker_helper_source()
    assert "resolveFailureIssuesForRecoveredPR" in helper_source
    assert "<!-- tracked-pr:" in helper_source
    assert "Resolution: Gate run succeeded" in helper_source
    assert "Closed failure issue" in helper_source


def test_post_ci_requires_issue_permissions() -> None:
    workflow = _load_workflow(POST_CI_PATH)
    permissions = workflow.get("permissions", {})

    assert permissions.get("issues") == "write"
    assert permissions.get("pull-requests") == "write"


def test_post_comment_job_upserts_single_pr_comment() -> None:
    workflow = _load_workflow(POST_CI_PATH)
    job = workflow["jobs"]["post-comment"]

    condition = " ".join(job.get("if", "").split())
    assert "needs.context.result == 'success'" in condition
    assert "needs.context.outputs.found == 'true'" in condition
    assert "needs.context.outputs.failure_tracker_skip != 'true'" in condition
    assert "needs.context.outputs.failure_incomplete != 'true'" in condition

    comment_step = _get_step(job, "Upsert consolidated PR comment")
    script = comment_step.get("with", {}).get("script", "")

    assert "const anchorPattern = /<!--\\s*maint-46-post-ci:" in script
    assert "const extractAnchor = (text) =>" in script
    assert "github.rest.issues.updateComment" in script
    assert "github.rest.issues.createComment" in script


def test_post_comment_script_updates_existing_comment(tmp_path: Path) -> None:
    node_path = shutil.which("node")
    if node_path is None:
        pytest.skip("node runtime not available")

    scenario = {
        "owner": "stranske",
        "repo": "Trend_Model_Project",
        "pr": 321,
        "body": (
            "<!-- maint-46-post-ci: pr=321 head=cafebabedead -->\n"
            "<!-- maint-46-post-ci: DO NOT EDIT -->\n"
            "## Automated Status Summary\nUpdated body\n"
        ),
        "comments": [
            {"id": 7001, "body": "Unrelated comment"},
            {
                "id": 7002,
                "body": (
                    "<!-- maint-46-post-ci: pr=123 head=deadbeefcafe -->\n"
                    "<!-- maint-46-post-ci: DO NOT EDIT -->\nPrevious summary\n"
                ),
            },
        ],
    }

    actions = _run_post_comment_harness(tmp_path, node_path, scenario)
    action_types = [entry.get("type") for entry in actions]

    assert action_types.count("updateComment") == 1
    assert "createComment" not in action_types

    scenario = {
        "owner": "stranske",
        "repo": "Trend_Model_Project",
        "pr": 654,
        "body": (
            "<!-- maint-46-post-ci: pr=654 head=baddadfeed00 -->\n"
            "<!-- maint-46-post-ci: DO NOT EDIT -->\n"
            "## Automated Status Summary\nFresh body\n"
        ),
        "comments": [
            {"id": 8101, "body": "General discussion"},
            {"id": 8102, "body": "Status marker missing"},
        ],
    }

    actions = _run_post_comment_harness(tmp_path, node_path, scenario)
    action_types = [entry.get("type") for entry in actions]

    assert action_types.count("createComment") == 1
    assert "updateComment" not in action_types

    create_payloads = [
        entry.get("payload", {})
        for entry in actions
        if entry.get("type") == "createComment"
    ]
    assert create_payloads, "Expected createComment payload to be recorded"
    create_payload = cast(dict[str, Any], create_payloads[0])
    assert create_payload.get("issue_number") == scenario["pr"]
    created_body = str(create_payload.get("body", ""))
    expected_body = scenario["body"].rstrip("\n")
    assert created_body.rstrip("\n") == expected_body


def test_failure_tracker_signature_uses_slugified_workflow_path() -> None:
    helper_source = _get_tracker_helper_source()

    assert "const slugify = (value) =>" in helper_source
    assert "slugify(run.name || run.display_title || 'Gate')" in helper_source
    assert "const signatureParts = failedJobs.map(job =>" in helper_source
    assert (
        "const title = `${slugify(run.name || run.display_title || 'Gate')} failure:"
        in helper_source
    )
    assert "const descriptionLines = [" in helper_source
    assert "const labels = ['ci-failure'];" in helper_source


def test_failure_tracker_script_tags_pr_numbers() -> None:
    helper_source = _get_tracker_helper_source()

    assert "Tracked PR:" in helper_source
    assert "<!-- tracked-pr:" in helper_source


def test_failure_tracker_script_updates_existing_issue(tmp_path: Path) -> None:
    node_path = shutil.which("node")
    if node_path is None:
        pytest.skip("node runtime not available")

    scenario = {
        "owner": "stranske",
        "repo": "Trend_Model_Project",
        "prNumber": 123,
        "run": {
            "id": 314,
            "workflow_id": 271,
            "name": "Gate",
            "display_title": "Gate",
            "path": ".github/workflows/pr-00-gate.yml",
            "html_url": "https://example.test/run/314",
        },
        "jobs": [
            {
                "id": 42,
                "name": "Gate / core tests (3.11)",
                "conclusion": "failure",
                "status": "completed",
                "html_url": "https://example.test/job/42",
            }
        ],
        "issue": {
            "body": (
                "Occurrences: 2\n"
                "Last seen: 2025-01-14T12:00:00Z\n"
                "<!-- occurrence-history-start -->\n"
                "| Timestamp | Run | Sig Hash | Failed Jobs |\n"
                "|---|---|---|---|\n"
                "| 2025-01-13T11:00:00Z | [run](https://example.test/run/21) | deadbeef | 1 |\n"
                "<!-- occurrence-history-end -->\n"
            ),
            "labels": [{"name": "ci-failure"}, {"name": "ci"}],
        },
        "openIssues": [{"number": 99}],
        "closedIssues": [],
        "comments": [],
    }

    helper_path = _helper_abs_path("maint-post-ci.js")
    helper_require_path = helper_path.replace("\\", "\\\\").replace("'", "\\'")
    scenario_json = json.dumps(scenario)

    harness_template = textwrap.dedent(
        """
        const actionsLog = [];
        function log(type, payload) {
          actionsLog.push({ type, ...(payload ? { payload } : {}) });
        }

        const scenario = __SCENARIO__;
        const helper = require('__HELPER__');
        const { updateFailureTracker } = helper;

        const github = {
          rest: {
            actions: {
              async listJobsForWorkflowRun(args) {
                log('listJobsForWorkflowRun', args);
                return { data: { jobs: scenario.jobs } };
              },
                  async getEnvironmentVariable(args) {
                    log('getEnvironmentVariable', args);
                    throw new Error('env var not configured');
                  },
              async updateEnvironmentVariable(args) {
                log('updateEnvironmentVariable', args);
                return { data: {} };
              },
            },
            issues: {
              async get(args) {
                log('getIssue', args);
                return { data: Object.assign({ number: args.issue_number }, scenario.issue) };
              },
              async update(args) {
                log('updateIssue', args);
                return { data: {} };
              },
              async listComments(args) {
                log('listComments', args);
                return { data: scenario.comments };
              },
              async createComment(args) {
                log('createComment', args);
                return { data: {} };
              },
              async getLabel(args) {
                log('getLabel', args);
                throw new Error('label missing');
              },
              async createLabel(args) {
                log('createLabel', args);
                return { data: {} };
              },
              async addLabels(args) {
                log('addLabels', args);
                return { data: {} };
              },
              async create(args) {
                log('createIssue', args);
                return { data: { number: 777 } };
              },
            },
            search: {
              async issuesAndPullRequests(args) {
                log('searchIssues', args);
                const query = String(args.q || '');
                if (query.includes('is:open')) {
                  return { data: { items: scenario.openIssues } };
                }
                if (query.includes('is:closed')) {
                  return { data: { items: scenario.closedIssues } };
                }
                return { data: { items: [] } };
              },
            },
          },
        };

        const context = {
          repo: { owner: scenario.owner, repo: scenario.repo },
          payload: { workflow_run: scenario.run },
        };

        const core = {
          info(message) { log('coreInfo', { message }); },
          warning(message) { log('coreWarning', { message }); },
          notice(message) { log('coreNotice', { message }); },
          setOutput(key, value) { log('setOutput', { key, value }); },
        };

        Object.assign(process.env, {
          PR_NUMBER: String(scenario.prNumber),
          AUTO_HEAL_INACTIVITY_HOURS: '24',
          RATE_LIMIT_MINUTES: '15',
          STACK_TOKENS_ENABLED: 'false',
          STACK_TOKEN_MAX_LEN: '160',
          FAILURE_INACTIVITY_HEAL_HOURS: '0',
          NEW_ISSUE_COOLDOWN_HOURS: '0',
          COOLDOWN_RETRY_MS: '0',
        });

        (async () => {
          await updateFailureTracker({ github, context, core });
          console.log(JSON.stringify(actionsLog));
        })().catch(error => {
          console.error(error.stack || String(error));
          process.exit(1);
        });
        """
    )

    harness = harness_template.replace("__SCENARIO__", scenario_json).replace(
        "__HELPER__", helper_require_path
    )

    script_path = tmp_path / "tracker_harness.js"
    script_path.write_text(harness, encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("GITHUB_WORKSPACE", str(REPO_ROOT))

    result = subprocess.run(
        [node_path, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout.strip()
    assert output, "Expected harness output to contain action log JSON"
    actions = json.loads(output)
    action_types = [entry.get("type") for entry in actions]

    assert action_types.count("createIssue") == 1
    assert "updateIssue" not in action_types
    assert "createComment" not in action_types

    create_payloads = [
        entry.get("payload", {})
        for entry in actions
        if entry.get("type") == "createIssue"
    ]
    assert create_payloads, "Expected createIssue payload to be recorded"
    create_payload = cast(dict[str, Any], create_payloads[0])
    title = str(create_payload.get("title", ""))
    assert title.lower().startswith("gate failure:")
    body = str(create_payload.get("body", ""))
    assert "Occurrences: 1" in body
    assert "## Failure summary" in body
    assert "Gate / core tests (3.11)" in body
    assert "<!-- tracked-pr: 123 -->" in body
    assert create_payload.get("labels") == ["ci-failure"]


def test_success_path_closes_tracked_issue(tmp_path: Path) -> None:
    node_path = shutil.which("node")
    if node_path is None:
        pytest.skip("node runtime not available")

    script = _rewrite_helper_import(_get_success_resolution_script())
    script_b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    scenario = {
        "owner": "stranske",
        "repo": "Trend_Model_Project",
        "pr": 456,
        "runUrl": "https://example.test/run/success",
        "issueNumber": 314,
        "issueBody": (
            "Tracked PR: #456\n"
            "<!-- tracked-pr: 456 -->\n"
            "Occurrences: 3\n"
            "Last seen: 2025-01-14T12:00:00Z\n"
            "Healing threshold: Auto-heal after 24h stability (success path)\n"
        ),
    }

    harness = textwrap.dedent(
        rf"""
        const actionsLog = [];
        function log(type, payload) {{
          actionsLog.push({{ type, ...(payload ? {{ payload }} : {{}}) }});
        }}

        const scenario = {json.dumps(scenario)};
        process.env.PR_NUMBER = String(scenario.pr);
        process.env.RUN_URL = scenario.runUrl;

        const github = {{
          rest: {{
            search: {{
              async issuesAndPullRequests(args) {{
                log('searchIssues', args);
                return {{ data: {{ items: [{{ number: scenario.issueNumber }}] }} }};
              }},
            }},
            issues: {{
              async get(args) {{
                log('getIssue', args);
                return {{ data: {{ body: scenario.issueBody }} }};
              }},
              async createComment(args) {{
                log('createComment', args);
                return {{ data: {{}} }};
              }},
              async update(args) {{
                log('updateIssue', args);
                return {{ data: {{}} }};
              }},
            }},
          }},
        }};

        const context = {{
          repo: {{ owner: scenario.owner, repo: scenario.repo }},
          payload: {{ workflow_run: {{ html_url: scenario.runUrl }} }},
        }};

        const core = {{
          info(message) {{ log('coreInfo', {{ message }}); }},
        }};

        (async () => {{
          const scriptSource = Buffer.from('{script_b64}', 'base64').toString('utf8');
          const vm = require('vm');
          const sandbox = {{
            github,
            context,
            core,
            process,
            console,
            require,
            Buffer,
            setTimeout,
            setInterval,
            clearTimeout,
            clearInterval,
          }};
          const runner = new vm.Script('(async () => {{' + scriptSource + '\n}})();');
          await runner.runInNewContext(sandbox);
          console.log(JSON.stringify(actionsLog));
        }})().catch((error) => {{
          console.error(error.stack || String(error));
          process.exit(1);
        }});
        """
    )

    script_path = tmp_path / "success_resolution_harness.js"
    script_path.write_text(harness, encoding="utf-8")

    result = subprocess.run(
        [node_path, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout.strip()
    assert output, "Expected harness output to contain action log JSON"

    actions = json.loads(output)
    action_types = [entry.get("type") for entry in actions]

    assert (
        action_types.count("createComment") == 1
    ), "Success path should leave a resolution comment"
    assert (
        action_types.count("updateIssue") == 1
    ), "Success path should update existing issue once"

    update_payloads = [
        entry.get("payload", {})
        for entry in actions
        if entry.get("type") == "updateIssue"
    ]
    assert update_payloads, "Expected updateIssue payload"
    update_payload = update_payloads[0]
    assert (
        update_payload.get("state") == "closed"
    ), "Success path should close the issue"
    assert "Resolved:" in (
        update_payload.get("body") or ""
    ), "Issue body should record resolution timestamp"
    assert "<!-- tracked-pr: 456 -->" in (
        update_payload.get("body") or ""
    ), "Tracked PR tag should be preserved"
