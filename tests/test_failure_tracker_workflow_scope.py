"""Regression tests for the failure tracker delegation pipeline."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKER_PATH = (
    REPO_ROOT / ".github" / "workflows" / "maint-47-check-failure-tracker.yml"
)
POST_CI_PATH = REPO_ROOT / ".github" / "workflows" / "maint-46-post-ci.yml"


def _load_workflow(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _get_step(job: dict, name: str) -> dict:
    for step in job.get("steps", []):
        if step.get("name") == name:
            return step
    raise AssertionError(f"Step {name!r} not found in workflow")


def _get_tracker_script() -> str:
    workflow = _load_workflow(POST_CI_PATH)
    job = workflow["jobs"]["failure-tracker"]
    tracker_step = _get_step(job, "Derive failure signature & update tracking issue")
    script = tracker_step.get("with", {}).get("script")
    assert isinstance(script, str), "Expected inline tracker script to be defined"
    return script


def test_tracker_workflow_is_now_thin_shell() -> None:
    workflow = _load_workflow(TRACKER_PATH)
    assert set(workflow["jobs"].keys()) == {"redirect"}
    redirect_job = workflow["jobs"]["redirect"]
    condition = " ".join(redirect_job.get("if", "").split())
    assert "workflow_run.event == 'pull_request'" in condition
    summary_step = _get_step(redirect_job, "Emit delegation summary")
    summary_body = summary_step.get("run", "")
    assert "maint-46-post-ci.yml" in summary_body


def test_tracker_shell_performs_no_issue_writes() -> None:
    workflow = _load_workflow(TRACKER_PATH)
    redirect_job = workflow["jobs"]["redirect"]

    for step in redirect_job.get("steps", []):
        assert (
            step.get("uses") is None
        ), "Delegation shell should not invoke external actions"
        script = step.get("run", "")
        assert "github.rest.issues" not in script


def test_tracker_workflow_triggers_from_gate_run() -> None:
    workflow = _load_workflow(TRACKER_PATH)

    trigger = workflow.get("on")
    assert trigger is not None, "Expected workflow_run trigger to be defined"

    workflow_run = trigger.get("workflow_run")
    assert (
        workflow_run is not None
    ), "Failure tracker should listen to workflow_run events"

    assert workflow_run.get("types") == ["completed"]
    assert workflow_run.get("workflows") == ["Gate"]


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
    assert "new Set([10, 12])" in script


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
    assert "github.rest.issues.update({" in tracker_script
    assert "github.rest.issues.createComment" in tracker_script

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
    assert "<!-- tracked-pr:" in script
    assert "Resolution: Gate run succeeded" in script
    assert "Closed failure issue" in script


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

    assert "<!-- maint-46-post-ci: DO NOT EDIT -->" in script
    assert "github.rest.issues.updateComment" in script
    assert "github.rest.issues.createComment" in script


def test_failure_tracker_signature_uses_slugified_workflow_path() -> None:
    script = _get_tracker_script()

    assert "const slugify = (value) =>" in script
    assert "const workflowPath = (run.path || '').trim();" in script
    assert (
        "const workflowStem = workflowFile.replace(/\\.ya?ml$/i, '') || workflowFile;"
        in script
    )
    assert (
        "const workflowIdRaw = workflowStem || workflowPath || rawWorkflowName || '';"
        in script
    )
    assert "const workflowId = slugify(workflowIdRaw);" in script
    assert (
        "const fallbackId = run.workflow_id ? `workflow-${run.workflow_id}` : (run.id ? `run-${run.id}` : 'workflow');"
        in script
    )
    assert "const workflowKey = workflowId || fallbackId;" in script
    assert "const signature = `${workflowKey}|${sigHash}`;" in script
    assert "Workflow slug: \\`${workflowId}\\`" in script


def test_failure_tracker_script_tags_pr_numbers() -> None:
    script = _get_tracker_script()

    assert "Tracked PR:" in script
    assert "<!-- tracked-pr:" in script


def test_failure_tracker_script_updates_existing_issue(tmp_path: Path) -> None:
    node_path = shutil.which("node")
    if node_path is None:
        pytest.skip("node runtime not available")

    script = _get_tracker_script()
    slugify_match = re.search(
        r"const slugify = \(value\) => \{(?:.|\n)*?\};",
        script,
    )
    assert slugify_match, "Expected slugify helper to be present in tracker script"
    slugify_code = slugify_match.group(0)

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
                "steps": [
                    {"name": "Checkout", "conclusion": "success"},
                    {"name": "pytest", "conclusion": "failure"},
                ],
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

    harness = textwrap.dedent(
        rf"""
        const actionsLog = [];
        function log(type, payload) {{
          actionsLog.push({{ type, ...(payload ? {{ payload }} : {{}}) }});
        }}

        (async () => {{
        const scenario = {json.dumps(scenario)};
        const newline = '\\n';
        const prNumber = scenario.prNumber;
        const prLine = prNumber ? 'Tracked PR: #' + prNumber : null;
        const prTag = prNumber ? '<!-- tracked-pr: ' + prNumber + ' -->' : null;
        const github = {{
          rest: {{
            search: {{
              async issuesAndPullRequests(args) {{
                log('searchIssues', args);
                if (args.q.includes('is:open')) {{
                  return {{ data: {{ items: scenario.openIssues }} }};
                }}
                if (args.q.includes('is:closed')) {{
                  return {{ data: {{ items: scenario.closedIssues }} }};
                }}
                return {{ data: {{ items: [] }} }};
              }},
            }},
            issues: {{
              async getLabel(args) {{ log('getLabel', args); return {{ data: {{}} }}; }},
              async createLabel(args) {{ log('createLabel', args); return {{ data: {{}} }}; }},
              async get(args) {{ log('getIssue', args); return {{ data: Object.assign({{ number: args.issue_number }}, scenario.issue) }}; }},
              async update(args) {{ log('updateIssue', args); return {{ data: {{}} }}; }},
              async listComments(args) {{ log('listComments', args); return {{ data: scenario.comments }}; }},
              async createComment(args) {{ log('createComment', args); return {{ data: {{}} }}; }},
              async addLabels(args) {{ log('addLabels', args); return {{ data: {{}} }}; }},
              async listForRepo(args) {{ log('listForRepo', args); return {{ data: [] }}; }},
              async create(args) {{ log('createIssue', args); return {{ data: {{ number: 777 }} }}; }},
            }},
          }},
          paginate: async () => scenario.jobs,
          request: async () => {{ throw new Error('Unexpected request invocation'); }},
        }};

        const owner = scenario.owner;
        const repo = scenario.repo;
        const run = scenario.run;
        const failedJobs = scenario.jobs.map(job => Object.assign({{}}, job, {{ __stackToken: 'stacks-off' }}));

        {slugify_code}

        const RATE_LIMIT_MINUTES = 15;
        const HEAL_THRESHOLD_DESC = 'Auto-heal after 24h stability (success path)';
        const workflowName = run.name || run.display_title || 'Workflow';
        const workflowPath = (run.path || '').trim();
        const workflowFile = (workflowPath.split('/').pop() || '').trim();
        const workflowStem = workflowFile.replace(/\\.ya?ml$/i, '') || workflowFile;
        const workflowIdRaw = workflowStem || workflowPath || workflowName || '';
        const workflowId = slugify(workflowIdRaw);
        const fallbackId = run.workflow_id ? `workflow-${{run.workflow_id}}` : (run.id ? `run-${{run.id}}` : 'workflow');
        const workflowKey = workflowId || fallbackId;

        const crypto = require('crypto');
        const sigParts = failedJobs.map(j => {{
          const failingStep = (j.steps || []).find(s => (s.conclusion || '').toLowerCase() !== 'success');
          return `${{j.name}}::${{failingStep ? failingStep.name : 'no-step'}}::${{j.__stackToken}}`;
        }}).sort();
        const sigHash = crypto.createHash('sha256').update(sigParts.join('|')).digest('hex').slice(0, 12);
        const signature = `${{workflowKey}}|${{sigHash}}`;
        const title = `Workflow Failure (${{signature}})`;
        const labels = ['ci-failure', 'ci', 'devops', 'priority: medium'];

        for (const lb of labels) {{
          try {{ await github.rest.issues.getLabel({{ owner, repo, name: lb }}); }}
          catch {{ try {{ await github.rest.issues.createLabel({{ owner, repo, name: lb, color: 'BFDADC' }}); }} catch {{}} }}
        }}

        const bodyParts = ['Workflow: Gate'];
        if (prLine) bodyParts.push(prLine);
        if (prTag) bodyParts.push(prTag);
        bodyParts.push('### Failed Jobs', 'Job table here');
        const bodyBlock = bodyParts.join(newline);
        let issue_number = null;
        let reopened = false;

        const qOpen = `repo:${{owner}}/${{repo}} is:issue is:open in:title "${{signature}}" label:ci-failure`;
        const searchOpen = await github.rest.search.issuesAndPullRequests({{ q: qOpen, per_page: 1 }});
        if (searchOpen.data.items.length) {{
          issue_number = searchOpen.data.items[0].number;
        }} else {{
          const qClosed = `repo:${{owner}}/${{repo}} is:issue is:closed in:title "${{signature}}" label:ci-failure`;
          const searchClosed = await github.rest.search.issuesAndPullRequests({{ q: qClosed, per_page: 1 }});
          if (searchClosed.data.items.length) {{
            issue_number = searchClosed.data.items[0].number;
            try {{
              await github.rest.issues.update({{ owner, repo, issue_number, state: 'open' }});
              reopened = true;
            }} catch (e) {{
              issue_number = null;
            }}
          }}
        }}

        if (issue_number) {{
          const existing = await github.rest.issues.get({{ owner, repo, issue_number }});
          const nowIso = new Date().toISOString();
          const baseBody = existing.data.body || '';
          const occMatch = baseBody.match(/Occurrences:\\s*(\d+)/i);
          const occ = (occMatch ? parseInt(occMatch[1], 10) : 0) + 1;
          const summaryLines = [
            `Occurrences: ${{occ}}`,
            `Last seen: ${{nowIso}}`,
            'Healing threshold: ' + HEAL_THRESHOLD_DESC,
            baseBody.trim(),
          ].filter(Boolean);
          let body = summaryLines.join(newline);
          if (prLine && !body.includes(prLine)) {{
            body = body.replace('Healing threshold: ' + HEAL_THRESHOLD_DESC, 'Healing threshold: ' + HEAL_THRESHOLD_DESC + newline + prLine);
          }}
          if (prTag && !body.includes(prTag)) {{
            body = prTag + newline + body;
          }}
          await github.rest.issues.update({{ owner, repo, issue_number, title, body }});
          const comments = await github.rest.issues.listComments({{ owner, repo, issue_number, per_page: 50 }});
          const alreadyCommented = comments.data.some(c => c.body && c.body.includes(run.html_url));
          let postComment = !alreadyCommented;
          if (postComment && comments.data.length) {{
            const last = comments.data[comments.data.length - 1];
            const lastTs = Date.parse(last.created_at);
            if (!isNaN(lastTs)) {{
              const minutesAgo = (Date.now() - lastTs) / 60000;
              if (minutesAgo < RATE_LIMIT_MINUTES) {{
                postComment = false;
              }}
            }}
          }}
          const commentPayload = reopened ? `Failure reoccurred after auto-heal; issue reopened.\\n\\n${{bodyBlock}}` : bodyBlock;
          if (postComment) {{
            await github.rest.issues.createComment({{ owner, repo, issue_number, body: commentPayload }});
          }}
        }} else {{
          await github.rest.issues.create({{ owner, repo, title, body: bodyBlock, labels }});
        }}

        console.log(JSON.stringify(actionsLog));
        }})().catch((error) => {{
          console.error(error.stack || String(error));
          process.exit(1);
        }});
        """
    )

    script_path = tmp_path / "tracker_harness.js"
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
        "createIssue" not in action_types
    ), "Tracker should update existing issue instead of creating new"
    assert (
        action_types.count("updateIssue") == 1
    ), "Existing issue should be updated exactly once"
    assert (
        action_types.count("createComment") == 1
    ), "Existing issue should receive a new occurrence comment"

    update_payloads = [
        entry.get("payload", {})
        for entry in actions
        if entry.get("type") == "updateIssue"
    ]
    assert update_payloads, "Expected an updateIssue payload to inspect"
    update_body = update_payloads[0].get("body") or ""
    assert "Tracked PR: #123" in update_body
    assert "<!-- tracked-pr: 123 -->" in update_body
