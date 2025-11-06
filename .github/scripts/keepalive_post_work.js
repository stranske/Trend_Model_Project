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

async function dispatchConnector({
  github,
  owner,
  repo,
  eventType,
  payload,
  core,
  action,
}) {
  await github.rest.repos.createDispatchEvent({
    owner,
    repo,
    event_type: eventType,
    client_payload: { ...payload, action },
  });
  core?.info?.(`Dispatch ${action} sent via ${eventType}.`);
}

async function findConnectorPr({
  github,
  owner,
  repo,
  baseBranch,
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
  const baseBranch = normalise(env.BASE_BRANCH) || normalise(env.PR_BASE_BRANCH);
  const headBranch = normalise(env.HEAD_BRANCH) || normalise(env.PR_HEAD_BRANCH);
  let baselineHead = normalise(env.PREVIOUS_HEAD);
  const commentId = parseNumber(env.COMMENT_ID, NaN, { min: 1 });
  const commentUrl = normalise(env.COMMENT_URL);
  const agentAlias = normalise(env.AGENT_ALIAS) || 'codex';
  const eventType = normalise(env.DISPATCH_EVENT_TYPE) || 'codex-pr-comment-command';
  const automationLogins = parseLoginList(env.AUTOMATION_LOGINS || 'stranske-automation-bot');
  const mergeMethod = normalise(env.MERGE_METHOD) || 'squash';
  const deleteTempBranch = parseBoolean(env.DELETE_TEMP_BRANCH, true);
  const syncLabel = normaliseLower(env.SYNC_LABEL) || 'agents:sync-required';
  const debugLabel = normaliseLower(env.DEBUG_LABEL) || 'agents:debug';
  const ttlShort = parseNumber(env.TTL_SHORT_MS, 90_000, { min: 0 });
  const pollShort = parseNumber(env.POLL_SHORT_MS, 10_000, { min: 0 });
  const ttlLong = parseNumber(env.TTL_LONG_MS, 240_000, { min: 0 });
  const pollLong = parseNumber(env.POLL_LONG_MS, 20_000, { min: 0 });

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

  record('Labels', hasSyncLabel ? 'agents:sync-required present' : 'agents:sync-required absent');

  let initialHeadInfo;
  try {
    initialHeadInfo = await fetchHead();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    record('Initial head fetch', `Failed: ${message}`);
    await summaryHelper.flush(buildSyncSummaryLabel(trace));
    return;
  }

  const initialHead = initialHeadInfo.headSha || baselineHead;
  if (!baselineHead) {
    baselineHead = initialHead || '';
  }
  record('Baseline head', baselineHead || '(unavailable)');

  let success = false;
  let finalHead = initialHead;

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
    record('Initial poll', `Branch advanced to ${pollResult.headSha}`);
  } else {
    record('Initial poll', 'No change detected after initial wait.');
  }

  const sharedPayload = {
    issue: Number.isFinite(issueNumber) ? Number(issueNumber) : undefined,
    base: baseBranch,
    head: headBranch,
    comment_id: Number.isFinite(commentId) ? Number(commentId) : undefined,
    comment_url: commentUrl,
    agent: agentAlias,
    trace,
    round: Number.isFinite(Number(round)) ? Number(round) : round,
  };

  if (!success) {
    if (Number.isFinite(issueNumber) && headBranch) {
      try {
        await dispatchConnector({
          github,
          owner,
          repo,
          eventType,
          payload: sharedPayload,
          core,
          action: 'update-branch',
        });
        record('Update-branch dispatch', 'Sent to connector.');
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        record('Update-branch dispatch', `Failed: ${message}`);
      }

      const updateResult = await pollForHeadChange({
        fetchHead,
        initialSha: baselineHead,
        timeoutMs: ttlLong,
        intervalMs: pollLong,
        label: 'update-branch',
        core,
      });
      if (updateResult.changed) {
        success = true;
        finalHead = updateResult.headSha;
        record('Update-branch result', `Branch advanced to ${updateResult.headSha}`);
      } else {
        record('Update-branch result', 'Branch unchanged after update-branch attempt.');
      }
    } else {
      record('Update-branch dispatch', 'Skipped due to missing issue or branch metadata.');
    }
  }

  let connectorPr = null;
  if (!success) {
    if (Number.isFinite(issueNumber) && headBranch) {
      const dispatchStarted = Date.now();
      try {
        await dispatchConnector({
          github,
          owner,
          repo,
          eventType,
          payload: sharedPayload,
          core,
          action: 'create-pr',
        });
        record('Create-pr dispatch', 'Sent to connector.');
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        record('Create-pr dispatch', `Failed: ${message}`);
      }

      const deadline = Date.now() + ttlLong;
      while (Date.now() <= deadline) {
        try {
          connectorPr = await findConnectorPr({
            github,
            owner,
            repo,
            baseBranch: headBranch,
            automationLogins,
            since: dispatchStarted,
          });
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          core?.warning?.(`Failed to scan connector PRs: ${message}`);
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

        const createResult = await pollForHeadChange({
          fetchHead,
          initialSha: baselineHead,
          timeoutMs: ttlLong,
          intervalMs: pollLong,
          label: 'create-pr',
          core,
        });
        if (createResult.changed) {
          success = true;
          finalHead = createResult.headSha;
          record('Create-pr result', `Branch advanced to ${createResult.headSha}`);
        } else {
          record('Create-pr result', 'Branch unchanged after create-pr attempt.');
        }
      } else {
        record('Connector PR', 'No connector PR detected before timeout.');
      }
    } else {
      record('Create-pr dispatch', 'Skipped due to missing issue or branch metadata.');
    }
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
        record('Sync label', 'Removed agents:sync-required.');
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        record('Sync label', `Failed to remove agents:sync-required: ${message}`);
      }
    } else {
      record('Sync label', 'No agents:sync-required label present.');
    }
    record('Result', `Branch advanced to ${finalHead || '(unknown)'}.`);
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
      record('Sync label', 'Applied agents:sync-required.');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      record('Sync label', `Failed to apply agents:sync-required: ${message}`);
    }
  } else {
    record('Sync label', 'agents:sync-required already present.');
  }

  if (hasDebugLabel) {
    const traceToken = trace ? `{${trace}}` : '{trace-missing}';
    const message = `Keepalive ${round || '?'} ${traceToken} escalation: agent "done" but branch unchanged after update-branch/create-pr attempts.`;
    try {
      await github.rest.issues.createComment({
        owner,
        repo,
        issue_number: prNumber,
        body: message,
      });
      record('Escalation comment', 'Posted debug escalation comment.');
    } catch (error) {
      const errMessage = error instanceof Error ? error.message : String(error);
      record('Escalation comment', `Failed to post escalation comment: ${errMessage}`);
    }
  } else {
    record('Escalation comment', 'Debug label absent; no comment posted.');
  }

  record('Result', 'Branch unchanged; awaiting manual intervention.');
  await summaryHelper.flush(buildSyncSummaryLabel(trace));
}

module.exports = {
  runKeepalivePostWork,
};
