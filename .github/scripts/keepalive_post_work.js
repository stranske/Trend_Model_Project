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

function parseLoginList(value) {
  const raw = normalise(value);
  if (!raw) {
    return [];
  }
  return raw
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map((entry) => entry.replace(/\[bot\]$/i, '').toLowerCase());
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

async function findExistingCommandComment({ github, owner, repo, prNumber, body, core }) {
  if (!github?.rest?.issues?.listComments) {
    return null;
  }
  const targetBody = normalise(body);
  if (!targetBody) {
    return null;
  }

  try {
    const comments = await github.paginate(github.rest.issues.listComments, {
      owner,
      repo,
      issue_number: prNumber,
      per_page: 100,
    });
    for (const comment of comments) {
      if (!comment) {
        continue;
      }
      const candidateBody = normalise(comment.body);
      if (candidateBody === targetBody) {
        return comment;
      }
    }
  } catch (error) {
    // Surface via warning but do not block execution.
    const message = error instanceof Error ? error.message : String(error);
    core?.warning?.(`Unable to scan for existing command comment: ${message}`);
  }
  return null;
}

function computeIdempotencyKey(prNumber, round, trace) {
  const safeTrace = normalise(trace) || 'trace-missing';
  const safeRound = normalise(round) || '?';
  const safePr = Number.isFinite(prNumber) ? String(prNumber) : normalise(prNumber) || '?';
  return `${safePr}/${safeRound}#${safeTrace}`;
}

const DEFAULT_SYNC_WORKFLOW_FILE = 'agents-keepalive-branch-sync.yml';
const SYNC_BRANCH_PREFIX = 'sync/codex-';

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

async function findConnectorPr({
  github,
  owner,
  repo,
  baseBranch,
  headPrefix,
  automationLogins,
  since,
}) {
  const candidates = await github.paginate(github.rest.pulls.list, {
    owner,
    repo,
    state: 'open',
    per_page: 50,
  });

  const lowerAutomation = new Set(
    (automationLogins || []).map((login) => login.replace(/\[bot\]$/i, '').toLowerCase())
  );

  const normalisedBase = normalise(baseBranch);
  const prefixLower = normaliseLower(headPrefix);
  const sinceMs = Number.isFinite(since) ? since : 0;

  const filtered = candidates
    .filter((pr) => {
      if (!pr || !pr.base || !pr.user) {
        return false;
      }
      const baseRef = normalise(pr.base.ref);
      if (normalisedBase && baseRef !== normalisedBase) {
        return false;
      }
      if (prefixLower) {
        const headRef = normaliseLower(pr?.head?.ref);
        if (!headRef || !headRef.startsWith(prefixLower)) {
          return false;
        }
      }
      const login = normaliseLower(pr.user.login.replace(/\[bot\]$/i, ''));
      if (lowerAutomation.size > 0 && !lowerAutomation.has(login)) {
        return false;
      }
      if (sinceMs > 0) {
        const createdAt = new Date(pr.created_at || '').getTime();
        if (Number.isFinite(createdAt) && createdAt + 1000 < sinceMs) {
          return false;
        }
      }
      return true;
    })
    .sort((a, b) => {
      const aTime = new Date(a?.created_at || 0).getTime();
      const bTime = new Date(b?.created_at || 0).getTime();
      return bTime - aTime;
    });

  return filtered.length > 0 ? filtered[0] : null;
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
  const automationLogins = parseLoginList(env.AUTOMATION_LOGINS || 'stranske-automation-bot');
  const mergeMethod = normalise(env.MERGE_METHOD) || 'squash';
  const deleteTempBranch = parseBoolean(env.DELETE_TEMP_BRANCH, true);
  const syncLabel = normaliseLower(env.SYNC_LABEL) || 'agents:sync-required';
  const debugLabel = normaliseLower(env.DEBUG_LABEL) || 'agents:debug';
  const ttlShort = parseNumber(env.TTL_SHORT_MS, 90_000, { min: 0 });
  const pollShort = parseNumber(env.POLL_SHORT_MS, 5_000, { min: 0 });
  const ttlLong = parseNumber(env.TTL_LONG_MS, 240_000, { min: 0 });
  const pollLong = parseNumber(env.POLL_LONG_MS, 5_000, { min: 0 });
  const syncWorkflowFile = normalise(env.SYNC_WORKFLOW) || DEFAULT_SYNC_WORKFLOW_FILE;
  const syncWorkflowRefOverride = normalise(env.SYNC_WORKFLOW_REF);

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
  let commandCommentId = Number.isFinite(commentId) ? Number(commentId) : undefined;

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

  const tokenSelection = selectPreferredToken(env);
  record(
    'Token selection',
    tokenSelection.token ? tokenSelection.source : `${tokenSelection.source} (value missing)`
  );
  core?.setOutput?.('token_source', tokenSelection.source);

  const commandBody = `/update-branch trace:${trace || 'trace-missing'}`;

  if (!success) {
    if (Number.isFinite(prNumber)) {
      let existingCommand = null;
      try {
        existingCommand = await findExistingCommandComment({
          github,
          owner,
          repo,
          prNumber,
          body: commandBody,
          core,
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        record('Command lookup', `Failed to scan existing commands: ${message}`);
      }

      if (existingCommand) {
        commandCommentId = Number(existingCommand.id) || commandCommentId;
        record(
          'Command comment',
          `Existing /update-branch command detected (#${existingCommand.id}).`
        );
      } else {
        try {
          const response = await github.rest.issues.createComment({
            owner,
            repo,
            issue_number: prNumber,
            body: commandBody,
          });
          commandCommentId = Number(response?.data?.id) || commandCommentId;
          record(
            'Command comment',
            `Posted /update-branch command (#${commandCommentId || '?'}).`
          );
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          record('Command comment', `Failed to post /update-branch: ${message}`);
        }
      }

      if (commandCommentId) {
        if (github?.rest?.reactions?.createForIssueComment) {
          try {
            await github.rest.reactions.createForIssueComment({
              owner,
              repo,
              comment_id: commandCommentId,
              content: 'eyes',
            });
            record('Command reaction', 'Added 👀 reaction to command comment.');
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            record('Command reaction', `Failed to add reaction: ${message}`);
          }
        } else {
          record('Command reaction', 'Reactions API unavailable; skipped.');
        }
      }
    } else {
      record('Command comment', 'Skipped command dispatch; PR number missing.');
    }

    const updateResult = await pollForHeadChange({
      fetchHead,
      initialSha: baselineHead,
      timeoutMs: ttlShort,
      intervalMs: pollShort,
      label: 'comment-update-branch',
      core,
    });
    if (updateResult.changed) {
      success = true;
      finalHead = updateResult.headSha;
      mode = 'comment-update-branch';
      record('Update-branch result', `Branch advanced to ${updateResult.headSha}`);
    } else {
      record('Update-branch result', 'Branch unchanged after /update-branch attempt.');
    }
  }

  let connectorPr = null;
  if (!success) {
    const workflowRef =
      syncWorkflowRefOverride || headBranch || headBranchEnv || initialHeadInfo.headRef || initialHead;

    if (syncWorkflowFile && workflowRef && github?.rest?.actions?.createWorkflowDispatch) {
      try {
        await github.rest.actions.createWorkflowDispatch({
          owner,
          repo,
          workflow_id: syncWorkflowFile,
          ref: workflowRef,
          inputs: {
            pr_number: String(prNumber),
            trace: trace || '',
            base_ref: baseRef || '',
            head_ref: headBranch || '',
            head_sha: initialHead || '',
            agent: agentAlias,
            round: normalise(round) || '',
            comment_id: commandCommentId ? String(commandCommentId) : '',
            comment_url: commentUrl || '',
            idempotency_key: idempotencyKey,
          },
        });
        record('Sync workflow', `Dispatched ${syncWorkflowFile} on ${workflowRef}.`);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        record('Sync workflow', `Failed to dispatch ${syncWorkflowFile}: ${message}`);
      }

      const dispatchStarted = Date.now();
      const deadline = dispatchStarted + ttlLong;
      while (Date.now() <= deadline) {
        try {
          connectorPr = await findConnectorPr({
            github,
            owner,
            repo,
            baseBranch: headBranch || headBranchEnv || baseRef,
            headPrefix: SYNC_BRANCH_PREFIX,
            automationLogins,
            since: dispatchStarted,
          });
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          core?.warning?.(`Failed to scan sync PRs: ${message}`);
        }

        if (connectorPr) {
          break;
        }

        if (Date.now() > deadline) {
          break;
        }
        await delay(pollLong);
      }

      if (connectorPr) {
        record(
          'Connector PR',
          `Detected PR #${connectorPr.number} from ${connectorPr.user?.login || 'unknown'}.`
        );
        let merged = false;
        try {
          const mergeResponse = await github.rest.pulls.merge({
            owner,
            repo,
            pull_number: connectorPr.number,
            merge_method: mergeMethod,
            commit_title: `Keepalive sync ${trace || ''}`.trim(),
          });
          merged = Boolean(mergeResponse?.data?.merged);
          if (merged) {
            record('Connector merge', `Merged PR #${connectorPr.number}.`);
          } else {
            record('Connector merge', `PR #${connectorPr.number} not merged (merge API returned false).`);
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          record('Connector merge', `Failed: ${message}`);
        }

        if (merged && deleteTempBranch) {
          const headRef = connectorPr?.head?.ref;
          if (headRef) {
            try {
              await github.rest.git.deleteRef({ owner, repo, ref: `heads/${headRef}` });
              record('Connector cleanup', `Deleted temporary branch ${headRef}.`);
            } catch (error) {
              const message = error instanceof Error ? error.message : String(error);
              record('Connector cleanup', `Failed to delete ${headRef}: ${message}`);
            }
          }
        }

        const syncResult = await pollForHeadChange({
          fetchHead,
          initialSha: baselineHead,
          timeoutMs: ttlLong,
          intervalMs: pollLong,
          label: 'action-sync-pr',
          core,
        });
        if (syncResult.changed) {
          success = true;
          finalHead = syncResult.headSha;
          mode = 'action-sync-pr';
          record('Action sync result', `Branch advanced to ${syncResult.headSha}`);
        } else {
          record('Action sync result', 'Branch unchanged after action sync attempt.');
        }
      } else {
        record('Connector PR', 'No sync PR detected before timeout.');
      }
    } else if (!syncWorkflowFile || !workflowRef) {
      record('Sync workflow', 'Skipped due to missing workflow configuration or head ref.');
    } else {
      record('Sync workflow', 'GitHub client missing actions API; skipped.');
    }
  }

  const elapsedMs = Date.now() - startTime;
  core?.setOutput?.('merged_sha', finalHead || '');
  core?.setOutput?.('mode', mode);
  core?.setOutput?.('success', success ? 'true' : 'false');
  if (commandCommentId) {
    core?.setOutput?.('command_comment_id', String(commandCommentId));
  }

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
      `Keepalive ${round || '?'} ${traceToken} escalation: agent "done" but branch unchanged after /update-branch and action sync attempts.`;
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
