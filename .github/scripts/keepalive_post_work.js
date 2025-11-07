'use strict';

const { setTimeout: sleep } = require('timers/promises');

function normalise(value) {
  return String(value ?? '').trim();
}

function normaliseLower(value) {
  return normalise(value).toLowerCase();
}

function parseNumber(value, fallback, { min = Number.NEGATIVE_INFINITY } = {}) {
  if (value === null || value === undefined) {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < min) {
    return fallback;
  }
  return parsed;
}

function parseBoolean(value, fallback = false) {
  if (typeof value === 'boolean') {
    return value;
  }
  const lowered = normaliseLower(value);
  if (!lowered) {
    return fallback;
  }
  if (['true', '1', 'yes', 'y', 'on'].includes(lowered)) {
    return true;
  }
  if (['false', '0', 'no', 'n', 'off'].includes(lowered)) {
    return false;
  }
  return fallback;
}

async function delay(ms) {
  if (ms <= 0) {
    return;
  }
  await sleep(ms);
}

function parseAgentState(env) {
  const textCandidates = [
    env.AGENT_STATE,
    env.AGENT_STATUS,
    env.AGENT_RESULT,
    env.AGENT_OUTCOME,
    env.AGENT_COMPLETION,
    env.LAST_AGENT_STATE,
    env.LAST_AGENT_STATUS,
    env.LAST_AGENT_RESULT,
  ];

  for (const candidate of textCandidates) {
    const lowered = normaliseLower(candidate);
    if (!lowered) {
      continue;
    }
    if (['done', 'completed', 'complete', 'success'].includes(lowered)) {
      return { done: true, value: lowered };
    }
    if (['failed', 'failure', 'error', 'cancelled', 'canceled'].includes(lowered)) {
      return { done: false, value: lowered };
    }
    if (['working', 'in_progress', 'running', 'pending', 'active'].includes(lowered)) {
      return { done: false, value: lowered };
    }
    return { done: lowered === 'done', value: lowered };
  }

  const booleanCandidates = [env.AGENT_DONE, env.LAST_AGENT_DONE];
  for (const candidate of booleanCandidates) {
    if (candidate === undefined) {
      continue;
    }
    const parsed = parseBoolean(candidate, null);
    if (parsed === true) {
      return { done: true, value: 'done' };
    }
    if (parsed === false) {
      return { done: false, value: 'not-done' };
    }
  }

  return { done: false, value: '' };
}

function selectPreferredToken(env) {
  const actionsToken = normalise(env.ACTIONS_BOT_PAT);
  if (actionsToken) {
    return { source: 'ACTIONS_BOT_PAT', token: actionsToken };
  }
  const serviceToken = normalise(env.SERVICE_BOT_PAT);
  if (serviceToken) {
    return { source: 'SERVICE_BOT_PAT', token: serviceToken };
  }
  return { source: 'none', token: '' };
}

function extractAgentAliasFromLabels(labels, fallback = 'codex') {
  const names = extractLabelNames(labels);
  for (const name of names) {
    if (name.startsWith('agent:')) {
      const alias = name.slice('agent:'.length).trim();
      if (alias) {
        return alias;
      }
    }
  }
  return fallback;
}

function computeIdempotencyKey(prNumber, round, trace) {
  const safeTrace = normalise(trace) || 'trace-missing';
  const safeRound = normalise(round) || '?';
  const safePr = Number.isFinite(prNumber) ? String(prNumber) : normalise(prNumber) || '?';
  return `${safePr}/${safeRound}#${safeTrace}`;
}


function extractLabelNames(labels) {
  if (!Array.isArray(labels)) {
    return [];
  }
  return labels
    .map((label) => {
      if (!label) {
        return '';
      }
      if (typeof label === 'string') {
        return label.trim().toLowerCase();
      }
      if (typeof label.name === 'string') {
        return label.name.trim().toLowerCase();
      }
      return '';
    })
    .filter(Boolean);
}

async function loadPull({ github, owner, repo, prNumber }) {
  const { data } = await github.rest.pulls.get({ owner, repo, pull_number: prNumber });
  return {
    headSha: data?.head?.sha || '',
    headRef: data?.head?.ref || '',
    baseRef: data?.base?.ref || '',
    userLogin: data?.user?.login || '',
    raw: data,
  };
}

async function listLabels({ github, owner, repo, prNumber }) {
  const response = await github.rest.issues.listLabelsOnIssue({
    owner,
    repo,
    issue_number: prNumber,
    per_page: 100,
  });
  return Array.isArray(response?.data) ? response.data : [];
}

async function pollForHeadChange({
  fetchHead,
  initialSha,
  timeoutMs,
  intervalMs,
  label,
  core,
}) {
  const start = Date.now();
  let attempts = 0;
  let lastSha = initialSha;

  while (Date.now() - start <= timeoutMs) {
    attempts += 1;
    try {
      const { headSha } = await fetchHead();
      if (headSha) {
        lastSha = headSha;
      }
      if (headSha && headSha !== initialSha) {
        return { changed: true, headSha, attempts, elapsedMs: Date.now() - start };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      core?.warning?.(`Failed to poll head during ${label || 'poll'}: ${message}`);
    }

    if (Date.now() - start >= timeoutMs) {
      break;
    }

    await delay(intervalMs);
  }

  return { changed: false, headSha: lastSha, attempts, elapsedMs: Date.now() - start };
}

function buildSummaryRecorder(summary) {
  const rows = [];
  const record = (step, outcome) => {
    rows.push([step, outcome]);
  };

  const flush = async (heading) => {
    if (!summary) {
      return rows;
    }
    summary
      .addHeading(heading)
      .addTable([
        [
          { data: 'Step', header: true },
          { data: 'Outcome', header: true },
        ],
        ...rows.map(([step, outcome]) => [step, outcome]),
      ]);
    await summary.write();
    return rows;
  };

  return { record, flush, rows };
}

function buildSyncSummaryLabel(trace) {
  const safeTrace = normalise(trace) || 'trace-missing';
  return `Keepalive sync (${safeTrace})`;
}

async function runKeepalivePostWork({ core, github, context, env = process.env }) {
  const summaryHelper = buildSummaryRecorder(core?.summary);
  const record = summaryHelper.record;

  const trace = normalise(env.TRACE);
  const round = normalise(env.ROUND);
  const prNumber = parseNumber(env.PR_NUMBER, NaN, { min: 1 });
  const issueNumber = parseNumber(env.ISSUE_NUMBER, NaN, { min: 1 });
  const baseBranch = normalise(env.PR_BASE) || normalise(env.BASE_BRANCH) || normalise(env.PR_BASE_BRANCH);
  const headBranchEnv = normalise(env.PR_HEAD) || normalise(env.HEAD_BRANCH) || normalise(env.PR_HEAD_BRANCH);
  let baselineHead =
    normalise(env.PR_HEAD_SHA_PREV) ||
    normalise(env.PREVIOUS_HEAD) ||
    normalise(env.HEAD_SHA_PREV) ||
    '';
  const commentId = parseNumber(env.COMMENT_ID, NaN, { min: 1 });
  const commentUrl = normalise(env.COMMENT_URL);
  const agentAliasEnv = normalise(env.AGENT_ALIAS) || 'codex';
  const syncLabel = normaliseLower(env.SYNC_LABEL) || 'agents:sync-required';
  const debugLabel = normaliseLower(env.DEBUG_LABEL) || 'agents:debug';
  const ttlShort = parseNumber(env.TTL_SHORT_MS, 90_000, { min: 0 });
  const pollShort = parseNumber(env.POLL_SHORT_MS, 5_000, { min: 0 });
  const ttlLong = parseNumber(env.TTL_LONG_MS, 240_000, { min: 0 });
  const pollLong = parseNumber(env.POLL_LONG_MS, 5_000, { min: 0 });

  const { owner, repo } = context.repo || {};
  if (!owner || !repo) {
    record('Initialisation', 'Repository context missing; aborting.');
    await summaryHelper.flush(buildSyncSummaryLabel(trace));
    return;
  }
  if (!Number.isFinite(prNumber)) {
    record('Initialisation', 'PR number missing; aborting.');
    await summaryHelper.flush(buildSyncSummaryLabel(trace));
    return;
  }

  const idempotencyKey = computeIdempotencyKey(prNumber, round, trace);
  record('Idempotency', idempotencyKey);
  core?.setOutput?.('idempotency_key', idempotencyKey);
  if (trace) {
    core?.setOutput?.('trace', trace);
  }

  const fetchHead = async () => loadPull({ github, owner, repo, prNumber });

  let currentLabels = [];
  try {
    currentLabels = await listLabels({ github, owner, repo, prNumber });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    core?.warning?.(`Unable to list labels on PR #${prNumber}: ${message}`);
  }
  const labelNames = extractLabelNames(currentLabels);
  const hasSyncLabel = labelNames.includes(syncLabel);
  const hasDebugLabel = labelNames.includes(debugLabel);
  const agentAlias = extractAgentAliasFromLabels(currentLabels, agentAliasEnv);

  record('Labels', hasSyncLabel ? `${syncLabel} present` : `${syncLabel} absent`);
  record('Agent alias', agentAlias);

  const agentState = parseAgentState(env);
  if (!agentState.done) {
    record(
      'Preconditions',
      `Agent state ${agentState.value || '(unknown)'} does not indicate completion; skipping sync gate.`
    );
    core?.setOutput?.('mode', 'skipped-agent-state');
    core?.setOutput?.('success', 'false');
    await summaryHelper.flush(buildSyncSummaryLabel(trace));
    return;
  }
  record('Preconditions', `Agent reported done (${agentState.value || 'done'}).`);

  let initialHeadInfo;
  try {
    initialHeadInfo = await fetchHead();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    record('Initial head fetch', `Failed: ${message}`);
    await summaryHelper.flush(buildSyncSummaryLabel(trace));
    return;
  }

  const initialHead = initialHeadInfo.headSha || '';
  const headBranch = headBranchEnv || initialHeadInfo.headRef || '';
  const baseRef = baseBranch || initialHeadInfo.baseRef || '';
  if (!baselineHead) {
    baselineHead = initialHead;
  }
  record('Baseline head', baselineHead || '(unavailable)');

  if (baselineHead && initialHead && baselineHead !== initialHead) {
    record('Head check', `Head already advanced to ${initialHead}; skipping sync gate.`);
    core?.setOutput?.('mode', 'already-synced');
    core?.setOutput?.('success', 'true');
    core?.setOutput?.('merged_sha', initialHead || '');
    record('Result', `mode=already-synced sha=${initialHead || '(unknown)'} elapsed=0ms`);
    await summaryHelper.flush(buildSyncSummaryLabel(trace));
    return;
  }

  const startTime = Date.now();
  let success = false;
  let finalHead = initialHead;
  let mode = 'none';

  if (baselineHead) {
    const pollResult = await pollForHeadChange({
      fetchHead,
      initialSha: baselineHead,
      timeoutMs: ttlShort,
      intervalMs: pollShort,
      label: 'initial-wait',
      core,
    });
    if (pollResult.changed) {
      success = true;
      finalHead = pollResult.headSha;
      mode = 'already-synced';
      record('Initial poll', `Branch advanced to ${pollResult.headSha}`);
    } else {
      record('Initial poll', 'No change detected after initial wait.');
    }
  } else {
    record('Initial poll', 'Baseline head unavailable; skipping initial wait.');
  }

  const dispatchEventType = normalise(env.DISPATCH_EVENT_TYPE) || 'codex-pr-comment-command';
  const connectorIssue = Number.isFinite(issueNumber) ? Number(issueNumber) : prNumber;
  const commentNumericId = Number.isFinite(commentId) ? Number(commentId) : undefined;

  const connectorDispatch = async (action, label) => {
    if (!github?.rest?.repos?.createDispatchEvent) {
      record(label, 'GitHub client missing repository dispatch API; skipped.');
      return { dispatched: false };
    }
    if (!Number.isFinite(prNumber)) {
      record(label, 'Skipped: PR number missing.');
      return { dispatched: false };
    }
    const baseForDispatch = baseRef || baseBranch;
    const headForDispatch = headBranch || headBranchEnv;
    if (!baseForDispatch || !headForDispatch) {
      record(label, 'Skipped: base/head branch unavailable.');
      return { dispatched: false };
    }
    const issueForDispatch =
      Number.isFinite(connectorIssue) && connectorIssue > 0 ? Number(connectorIssue) : Number(prNumber);
    const payload = {
      action,
      issue: issueForDispatch,
      base: baseForDispatch,
      head: headForDispatch,
      comment_id: commentNumericId,
      comment_url: commentUrl || '',
      agent: agentAlias,
      trace: trace || '',
      round: normalise(round) || '',
    };
    for (const key of Object.keys(payload)) {
      if (payload[key] === undefined || payload[key] === '') {
        delete payload[key];
      }
    }
    try {
      await github.rest.repos.createDispatchEvent({
        owner,
        repo,
        event_type: dispatchEventType,
        client_payload: payload,
      });
      record(label, `Emitted ${dispatchEventType} (${action}).`);
      return { dispatched: true };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      record(label, `Failed: ${message}`);
      return { dispatched: false, error: message };
    }
  };

  if (!success) {
    const dispatchResult = await connectorDispatch('update-branch', 'Update-branch dispatch');
    if (dispatchResult.dispatched) {
      const updateResult = await pollForHeadChange({
        fetchHead,
        initialSha: baselineHead,
        timeoutMs: ttlLong,
        intervalMs: pollLong,
        label: 'dispatch-update-branch',
        core,
      });
      if (updateResult.changed) {
        success = true;
        finalHead = updateResult.headSha;
        mode = 'dispatch-update-branch';
        record('Update-branch result', `Branch advanced to ${updateResult.headSha}`);
      } else {
        record('Update-branch result', 'Branch unchanged after update-branch dispatch.');
      }
    }
  }

  if (!success) {
    const dispatchResult = await connectorDispatch('create-pr', 'Create-pr dispatch');
    if (dispatchResult.dispatched) {
      const syncResult = await pollForHeadChange({
        fetchHead,
        initialSha: baselineHead,
        timeoutMs: ttlLong,
        intervalMs: pollLong,
        label: 'dispatch-create-pr',
        core,
      });
      if (syncResult.changed) {
        success = true;
        finalHead = syncResult.headSha;
        mode = 'dispatch-create-pr';
        record('Create-pr result', `Branch advanced to ${syncResult.headSha}`);
      } else {
        record('Create-pr result', 'Branch unchanged after create-pr dispatch.');
      }
    }
  }

  const elapsedMs = Date.now() - startTime;
  core?.setOutput?.('merged_sha', finalHead || '');
  core?.setOutput?.('mode', mode);
  core?.setOutput?.('success', success ? 'true' : 'false');

  if (success) {
    if (hasSyncLabel) {
      try {
        await github.rest.issues.removeLabel({
          owner,
          repo,
          issue_number: prNumber,
          name: syncLabel,
        });
        record('Sync label', `Removed ${syncLabel}.`);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        record('Sync label', `Failed to remove ${syncLabel}: ${message}`);
      }
    } else {
      record('Sync label', `${syncLabel} not present.`);
    }
    record('Result', `mode=${mode || 'unknown'} sha=${finalHead || '(unknown)'} elapsed=${elapsedMs}ms`);
    await summaryHelper.flush(buildSyncSummaryLabel(trace));
    return;
  }

  if (!hasSyncLabel) {
    try {
      await github.rest.issues.addLabels({
        owner,
        repo,
        issue_number: prNumber,
        labels: [syncLabel],
      });
      record('Sync label', `Applied ${syncLabel}.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      record('Sync label', `Failed to apply ${syncLabel}: ${message}`);
    }
  } else {
    record('Sync label', `${syncLabel} already present.`);
  }

  if (hasDebugLabel) {
    const traceToken = trace ? `{${trace}}` : '{trace-missing}';
    const escalationMessage =
      `Keepalive ${round || '?'} ${traceToken} escalation: agent "done" but branch unchanged after update-branch/create-pr attempts.`;
    try {
      await github.rest.issues.createComment({
        owner,
        repo,
        issue_number: Number.isFinite(issueNumber) ? Number(issueNumber) : prNumber,
        body: escalationMessage,
      });
      record('Escalation comment', 'Posted debug escalation comment.');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      record('Escalation comment', `Failed to post escalation comment: ${message}`);
    }
  } else {
    record('Escalation comment', 'Debug label absent; no comment posted.');
  }

  record('Result', `mode=sync-timeout baseline=${baselineHead || '(unknown)'} elapsed=${elapsedMs}ms`);
  await summaryHelper.flush(buildSyncSummaryLabel(trace));
}

module.exports = {
  runKeepalivePostWork,
};
