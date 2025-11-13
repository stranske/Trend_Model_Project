'use strict';

const { setTimeout: sleep } = require('timers/promises');
const { createKeepaliveStateManager } = require('./keepalive_state.js');

const AGENT_LABEL_PREFIX = 'agent:';
const MERGE_METHODS = new Set(['merge', 'squash', 'rebase']);


function normalise(value) {
  return String(value ?? '').trim();
}

function normaliseLower(value) {
  return normalise(value).toLowerCase();
}

function parseBoolean(value, fallback = false) {
  if (value === undefined || value === null || value === '') {
    return fallback;
  }
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    return value !== 0;
  }
  const lowered = normaliseLower(value);
  if (!lowered) {
    return fallback;
  }
  if (['true', '1', 'yes', 'on'].includes(lowered)) {
    return true;
  }
  if (['false', '0', 'no', 'off'].includes(lowered)) {
    return false;
  }
  return fallback;
}

function parseCommaList(value) {
  if (!value) {
    return [];
  }
  if (Array.isArray(value)) {
    return value
      .map((entry) => normaliseLower(typeof entry === 'string' ? entry : entry?.login || entry?.name))
      .filter(Boolean);
  }
  if (typeof value !== 'string') {
    return [];
  }
  return value
    .split(/[\s,]+/)
    .map((entry) => normaliseLower(entry))
    .filter(Boolean);
}

function clampMergeMethod(method, fallback = 'squash') {
  const candidate = normaliseLower(method);
  if (MERGE_METHODS.has(candidate)) {
    return candidate;
  }
  if (candidate === 'ff' || candidate === 'fast-forward' || candidate === 'fastforward') {
    return 'merge';
  }
  return fallback;
}

function toTimestamp(value) {
  if (!value) {
    return 0;
  }
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) {
    return 0;
  }
  return parsed;
}

async function delay(ms) {
  const value = Number(ms);
  const timeout = Number.isFinite(value) && value > 0 ? value : 0;
  await sleep(timeout);
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

function computeIdempotencyKey(prNumber, round, trace) {
  const safeTrace = normalise(trace) || 'trace-missing';
  const safeRound = normalise(round) || '?';
  const safePr = Number.isFinite(prNumber) ? String(prNumber) : normalise(prNumber) || '?';
  return `${safePr}/${safeRound}#${safeTrace}`;
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

function extractLabelNames(labels) {
  if (!Array.isArray(labels)) {
    return [];
  }
  return labels
    .map((entry) => {
      if (!entry) {
        return '';
      }
      if (typeof entry === 'string') {
        return normaliseLower(entry);
      }
      if (typeof entry.name === 'string') {
        return normaliseLower(entry.name);
      }
      return '';
    })
    .filter(Boolean);
}

function extractAgentAliasFromLabels(labels, fallback) {
  const names = extractLabelNames(labels);
  for (const name of names) {
    if (name.startsWith(AGENT_LABEL_PREFIX)) {
      const alias = normalise(name.slice(AGENT_LABEL_PREFIX.length));
      if (alias) {
        return alias;
      }
    }
  }
  return normalise(fallback) || 'codex';
}

function parseAgentState(env = {}) {
  const response = {
    value: '',
    done: false,
  };

  const jsonCandidate = normalise(env.AGENT_STATE_JSON);
  if (jsonCandidate && /^[{[]/.test(jsonCandidate)) {
    try {
      const parsed = JSON.parse(jsonCandidate);
      if (typeof parsed?.value === 'string') {
        response.value = normalise(parsed.value);
      }
      if (typeof parsed?.done === 'boolean') {
        response.done = parsed.done;
      } else if (typeof parsed?.status === 'string') {
        const lower = normaliseLower(parsed.status);
        response.done = lower === 'done' || lower === 'completed' || lower === 'complete';
        if (!response.value) {
          response.value = normalise(parsed.status);
        }
      }
    } catch (error) {
      // fall through to other inputs when JSON parsing fails
    }
  }

  if (!response.value) {
    const valueOrder = [env.AGENT_STATE, env.AGENT_STATUS, env.AGENT_DONE];
    for (const candidate of valueOrder) {
      const normalised = normalise(candidate);
      if (normalised) {
        response.value = normalised;
        break;
      }
    }
  }

  if (!response.done) {
    const lower = normaliseLower(response.value);
    if (lower) {
      response.done = ['done', 'complete', 'completed', 'success', 'true', 'yes'].includes(lower);
    }
  }

  return response;
}

async function pollForHeadChange({ fetchHead, initialSha, timeoutMs, intervalMs, label, core }) {
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

function isForkPull(initialInfo) {
  const forkFlag = initialInfo?.raw?.head?.repo?.fork;
  return Boolean(forkFlag);
}




function mergeStateShallow(target, updates) {
  if (!updates || typeof updates !== 'object') {
    return target;
  }
  const next = { ...target };
  for (const [key, value] of Object.entries(updates)) {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      next[key] = mergeStateShallow(target[key] && typeof target[key] === 'object' ? target[key] : {}, value);
    } else {
      next[key] = value;
    }
  }
  return next;
}

function parseRoundNumber(value) {
  const parsed = Number(value);
  if (Number.isFinite(parsed) && parsed > 0) {
    return Math.round(parsed);
  }
  const fallback = Number(String(value || '').trim().replace(/[^0-9]/g, ''));
  if (Number.isFinite(fallback) && fallback > 0) {
    return Math.round(fallback);
  }
  return 0;
}

async function dispatchCommand({
  github,
  owner,
  repo,
  eventType,
  action,
  prNumber,
  agentAlias,
  baseRef,
  headRef,
  headSha,
  trace,
  round,
  commentInfo,
  idempotencyKey,
  roundTag = 'round=?',
  record,
}) {
  const safeEvent = normalise(eventType) || 'codex-pr-comment-command';
  if (!safeEvent) {
    record(`Dispatch ${action}`, `skipped: event type missing ${roundTag}`);
    return false;
  }
  if (!github?.rest?.repos?.createDispatchEvent) {
    record(`Dispatch ${action}`, `skipped: GitHub client missing createDispatchEvent ${roundTag}`);
    return false;
  }

  const payload = {
    issue: Number.isFinite(prNumber) ? Number(prNumber) : parseNumber(prNumber, 0, { min: 0 }),
    action,
    agent: agentAlias || 'codex',
    base: baseRef || '',
    head: headRef || '',
    head_sha: headSha || '',
    trace: trace || '',
  };

  const parsedRound = parseRoundNumber(round);
  if (parsedRound > 0) {
    payload.round = parsedRound;
  }
  if (commentInfo?.id) {
    payload.comment_id = Number(commentInfo.id);
  }
  if (commentInfo?.url) {
    payload.comment_url = commentInfo.url;
  }
  if (idempotencyKey) {
    payload.idempotency_key = idempotencyKey;
  }
  payload.quiet = true;
  payload.reply = 'none';

  try {
    await github.rest.repos.createDispatchEvent({
      owner,
      repo,
      event_type: safeEvent,
      client_payload: payload,
    });
    record(`Dispatch ${action}`, `sent action=${action} ${roundTag}`);
    return true;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    record(`Dispatch ${action}`, `failed: ${message} ${roundTag}`);
    return false;
  }
}

async function dispatchFallbackWorkflow({
  github,
  owner,
  repo,
  baseRef,
  dispatchRef,
  prNumber,
  headRef,
  headSha,
  trace,
  round,
  agentAlias,
  commentInfo,
  idempotencyKey,
  roundTag = 'round=?',
  record,
}) {
  if (!baseRef || !headRef || !headSha) {
    record('Fallback dispatch', `skipped: base/head/head_sha missing. ${roundTag}`);
    return { dispatched: false };
  }
  try {
    const inputs = {
      pr_number: String(prNumber),
      trace: trace || '',
      base_ref: baseRef,
      head_ref: headRef,
      head_sha: headSha,
    };
    if (agentAlias) {
      inputs.agent = agentAlias;
    }
    if (round) {
      inputs.round = String(round);
    }
    if (commentInfo?.id) {
      inputs.comment_id = String(commentInfo.id);
    }
    if (commentInfo?.url) {
      inputs.comment_url = commentInfo.url;
    }
    if (idempotencyKey) {
      inputs.idempotency_key = idempotencyKey;
    }

    const response = await github.rest.actions.createWorkflowDispatch({
      owner,
      repo,
      workflow_id: 'agents-keepalive-branch-sync.yml',
      ref: dispatchRef || baseRef,
      inputs,
    });

    record(
      'Fallback dispatch',
      `dispatched=keepalive-branch-sync http=${response?.status ?? 0} trace=${trace || 'missing'} ${roundTag}`,
    );
    return {
      dispatched: true,
      status: response?.status ?? 0,
      dispatchedAt: new Date().toISOString(),
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    record('Fallback dispatch', `failed: ${message} ${roundTag}`);
    return { dispatched: false, error: message };
  }
}

async function findFallbackRun({ github, owner, repo, createdAfter, existingRunId, core }) {
  if (!github?.rest?.actions?.listWorkflowRuns) {
    return null;
  }
  try {
    const response = await github.rest.actions.listWorkflowRuns({
      owner,
      repo,
      workflow_id: 'agents-keepalive-branch-sync.yml',
      event: 'workflow_dispatch',
      per_page: 20,
    });
    const runs = response?.data?.workflow_runs || [];
    const threshold = createdAfter ? new Date(createdAfter).getTime() - 5000 : 0;
    for (const run of runs) {
      if (existingRunId && Number(run?.id) === Number(existingRunId)) {
        return run;
      }
      const created = new Date(run?.created_at || run?.run_started_at || 0).getTime();
      if (!createdAfter || (Number.isFinite(created) && created >= threshold)) {
        return run;
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    core?.warning?.(`Failed to locate keepalive branch-sync run: ${message}`);
  }
  return null;
}

function buildAutomationLoginSet(value, fallback = []) {
  const list = parseCommaList(value);
  if (!list.length && Array.isArray(fallback) && fallback.length) {
    return new Set(parseCommaList(fallback));
  }
  return new Set(list);
}

function containsTrace(text, trace) {
  if (!text || !trace) {
    return false;
  }
  const haystack = normaliseLower(text);
  const needle = normaliseLower(trace);
  if (!haystack || !needle) {
    return false;
  }
  return haystack.includes(needle);
}

function scoreConnectorPr(pr, { trace, baseRef }) {
  let score = 0;
  if (!pr || typeof pr !== 'object') {
    return score;
  }
  if (containsTrace(pr.title, trace)) {
    score += 4;
  }
  if (containsTrace(pr.body, trace)) {
    score += 3;
  }
  const headRef = normaliseLower(pr.head?.ref);
  if (headRef && trace && headRef.includes(normaliseLower(trace))) {
    score += 2;
  }
  if (headRef && baseRef && headRef.includes(normaliseLower(baseRef))) {
    score += 1;
  }
  const createdAt = toTimestamp(pr.created_at || pr.updated_at || pr.closed_at);
  if (Number.isFinite(createdAt) && createdAt > 0) {
    score += 0.000001 * createdAt;
  }
  return score;
}

async function locateConnectorPullRequest({
  github,
  owner,
  repo,
  baseRef,
  trace,
  createdAfter,
  allowedLogins,
}) {
  if (!github?.rest?.pulls?.list) {
    return null;
  }
  try {
    const response = await github.rest.pulls.list({
      owner,
      repo,
      state: 'open',
      base: baseRef,
      sort: 'created',
      direction: 'desc',
      per_page: 50,
    });
    const pulls = Array.isArray(response?.data) ? response.data : [];
    if (!pulls.length) {
      return null;
    }
    const allowed = allowedLogins instanceof Set ? allowedLogins : new Set();
    const threshold = createdAfter ? createdAfter - 30_000 : 0;
    let candidate = null;
    let candidateScore = Number.NEGATIVE_INFINITY;
    for (const pr of pulls) {
      const created = toTimestamp(pr.created_at || pr.updated_at);
      if (threshold && created && created < threshold) {
        break;
      }
      const login = normaliseLower(pr.user?.login);
      if (allowed.size && (!login || !allowed.has(login))) {
        continue;
      }
      const score = scoreConnectorPr(pr, { trace, baseRef });
      if (candidate === null || score > candidateScore) {
        candidate = pr;
        candidateScore = score;
      }
    }
    return candidate;
  } catch (error) {
    return null;
  }
}

async function mergeConnectorPullRequest({
  github,
  owner,
  repo,
  baseRef,
  trace,
  dispatchTimestamp,
  allowedLogins,
  mergeMethod,
  deleteBranch,
  record,
  appendRound,
}) {
  const createdAfter = dispatchTimestamp ? toTimestamp(dispatchTimestamp) : 0;
  const pr = await locateConnectorPullRequest({
    github,
    owner,
    repo,
    baseRef,
    trace,
    createdAfter,
    allowedLogins,
  });
  if (!pr) {
    record('Create-pr auto-merge', appendRound('no connector PR detected.'));
    return { attempted: true, merged: false };
  }

  const prNumber = Number(pr.number);
  if (!Number.isFinite(prNumber) || prNumber <= 0) {
    record('Create-pr auto-merge', appendRound('skipped: invalid PR identifier.'));
    return { attempted: true, merged: false };
  }

  try {
    await github.rest.pulls.merge({
      owner,
      repo,
      pull_number: prNumber,
      merge_method: mergeMethod,
      commit_title: `Keepalive sync ${trace || ''}`.trim(),
    });
    record('Create-pr auto-merge', appendRound(`merged PR #${prNumber} using ${mergeMethod}.`));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    record('Create-pr auto-merge', appendRound(`failed: ${message}`));
    return { attempted: true, merged: false, prNumber, error: message };
  }

  let branchDeleted = false;
  if (deleteBranch && pr.head?.ref && !pr.head?.repo?.fork) {
    try {
      await github.rest.git.deleteRef({
        owner,
        repo,
        ref: `heads/${pr.head.ref}`,
      });
      branchDeleted = true;
      record('Create-pr cleanup', appendRound(`deleted branch ${pr.head.ref}.`));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      record('Create-pr cleanup', appendRound(`failed to delete branch ${pr.head.ref}: ${message}`));
    }
  }

  return { attempted: true, merged: true, prNumber, branchDeleted };
}

function formatCommandBody(action, trace, round) {
  const parts = [`/${normalise(action)}`.trim()];
  if (trace) {
    parts.push(`trace:${trace}`);
  }
  if (round) {
    parts.push(`round:${round}`);
  }
  return parts.filter(Boolean).join(' ');
}

async function postCommandComment({
  github,
  owner,
  repo,
  prNumber,
  action,
  trace,
  round,
  record,
  appendRound,
}) {
  const body = formatCommandBody(action, trace, round);
  try {
    const { data } = await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: prNumber,
      body,
    });
    const commentId = data?.id ? Number(data.id) : 0;
    record('Comment command', appendRound(`posted ${body}.`));
    if (commentId > 0) {
      try {
        await github.rest.reactions.createForIssueComment({
          owner,
          repo,
          comment_id: commentId,
          content: 'eyes',
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        record('Comment reaction', appendRound(`failed to add 👀: ${message}`));
      }
    }
    return {
      posted: true,
      body,
      commentId,
      commentUrl: data?.html_url || '',
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    record('Comment command', appendRound(`failed to post ${body}: ${message}`));
    return { posted: false, error: message };
  }
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
  const commentIdEnv = parseNumber(env.COMMENT_ID, NaN, { min: 1 });
  const commentUrlEnv = normalise(env.COMMENT_URL);
  const commentTraceEnv = normalise(env.COMMENT_TRACE);
  const commentRoundEnv = normalise(env.COMMENT_ROUND);
  const agentAliasEnv = normalise(env.AGENT_ALIAS) || 'codex';
  const syncLabel = normaliseLower(env.SYNC_LABEL) || 'agents:sync-required';
  const debugLabel = normaliseLower(env.DEBUG_LABEL) || 'agents:debug';
  const dispatchEventType = normalise(env.DISPATCH_EVENT_TYPE) || 'codex-pr-comment-command';
  const ttlShort = parseNumber(env.TTL_SHORT_MS, 90_000, { min: 0 });
  const pollShort = parseNumber(env.POLL_SHORT_MS, 5_000, { min: 0 });
  const ttlLong = parseNumber(env.TTL_LONG_MS, 240_000, { min: 0 });
  const pollLong = parseNumber(env.POLL_LONG_MS, 5_000, { min: 0 });
  const automationLogins = buildAutomationLoginSet(env.AUTOMATION_LOGINS);
  const mergeMethod = clampMergeMethod(env.MERGE_METHOD, 'squash');
  const deleteTempBranch = parseBoolean(env.DELETE_TEMP_BRANCH, true);
  const roundTag = `round=${round || '?'}`;
  const appendRound = (message) => `${message} ${roundTag}`;

  const finalState = {
    action: 'skip',
    headMoved: false,
    mode: 'none',
    finalHead: '',
    success: false,
    summaryWritten: false,
  };

  const applyFinalState = (updates = {}) => {
    if (!updates || typeof updates !== 'object') {
      return finalState;
    }
    if (Object.prototype.hasOwnProperty.call(updates, 'action')) {
      const value = normaliseLower(updates.action);
      finalState.action = value || finalState.action;
    }
    if (Object.prototype.hasOwnProperty.call(updates, 'headMoved')) {
      finalState.headMoved = Boolean(updates.headMoved);
    }
    if (Object.prototype.hasOwnProperty.call(updates, 'mode')) {
      const raw = updates.mode;
      finalState.mode = raw ? String(raw) : finalState.mode;
    }
    if (Object.prototype.hasOwnProperty.call(updates, 'finalHead')) {
      const raw = updates.finalHead;
      finalState.finalHead = raw ? String(raw) : '';
    }
    if (Object.prototype.hasOwnProperty.call(updates, 'success')) {
      finalState.success = Boolean(updates.success);
    }
    if (Object.prototype.hasOwnProperty.call(updates, 'summaryWritten')) {
      finalState.summaryWritten = Boolean(updates.summaryWritten);
    }
    return finalState;
  };

  const setOutputsFromFinalState = () => {
    core?.setOutput?.('action', finalState.action || 'skip');
    core?.setOutput?.('changed', finalState.headMoved ? 'true' : 'false');
    core?.setOutput?.('mode', finalState.mode || '');
    core?.setOutput?.('success', finalState.success ? 'true' : 'false');
    core?.setOutput?.('merged_sha', finalState.finalHead || '');
  };

  const flushSummary = async () => {
    if (!finalState.summaryWritten && core?.summary) {
      core.summary
        .addRaw(
          `SYNC: action=${finalState.action || 'skip'} head_changed=${finalState.headMoved ? 'true' : 'false'} trace=${
            trace || 'n/a'
          }`,
        )
        .addEOL();
    }
    await summaryHelper.flush(buildSyncSummaryLabel(trace));
    finalState.summaryWritten = true;
  };

  const complete = async () => {
    setOutputsFromFinalState();
    await flushSummary();
  };

  const { owner, repo } = context.repo || {};
  if (!owner || !repo) {
    record('Initialisation', 'Repository context missing; aborting.');
    applyFinalState({ action: 'skip', success: false, mode: 'initialisation-missing-repo' });
    await complete();
    return;
  }
  if (!Number.isFinite(prNumber)) {
    record('Initialisation', 'PR number missing; aborting.');
    applyFinalState({ action: 'skip', success: false, mode: 'initialisation-missing-pr' });
    await complete();
    return;
  }

  const idempotencyKey = computeIdempotencyKey(prNumber, round, trace);
  record('Idempotency', idempotencyKey);
  core?.setOutput?.('idempotency_key', idempotencyKey);
  if (trace) {
    core?.setOutput?.('trace', trace);
  }

  const stateManager = await createKeepaliveStateManager({ github, context, prNumber, trace, round });
  let state = stateManager.state || {};
  let commandState = state.command_dispatch && typeof state.command_dispatch === 'object' ? state.command_dispatch : {};
  let stateCommentId = stateManager.commentId ? Number(stateManager.commentId) : 0;
  let stateCommentUrl = stateManager.commentUrl || '';

  const applyStateUpdate = async (updates, { forcePersist = false } = {}) => {
    if (!forcePersist && !stateCommentId) {
      state = mergeStateShallow(state, updates);
      commandState = state.command_dispatch && typeof state.command_dispatch === 'object' ? state.command_dispatch : {};
      return { state: { ...state }, commentId: stateCommentId, commentUrl: stateCommentUrl };
    }

    const saved = await stateManager.save(updates);
    state = saved.state || {};
    commandState = state.command_dispatch && typeof state.command_dispatch === 'object' ? state.command_dispatch : {};
    stateCommentId = saved.commentId || stateCommentId;
    stateCommentUrl = saved.commentUrl || stateCommentUrl;
    return saved;
  };

  if (!stateCommentId) {
    const saved = await applyStateUpdate({}, { forcePersist: true });
    stateCommentId = saved.commentId || stateCommentId;
    stateCommentUrl = saved.commentUrl || stateCommentUrl;
    record('State comment', appendRound(`initialised id=${stateCommentId || 0}`));
  } else {
    record('State comment', appendRound(`reused id=${stateCommentId}`));
  }

  const updateCommandState = async (updates) => {
    const merged = mergeStateShallow(commandState, updates);
    await applyStateUpdate({ command_dispatch: merged });
    commandState = state.command_dispatch && typeof state.command_dispatch === 'object' ? state.command_dispatch : {};
    return commandState;
  };

  const getCommandActionState = (action) => {
    if (!action) {
      return {};
    }
    const entry = commandState && typeof commandState === 'object' ? commandState[action] : undefined;
    return entry && typeof entry === 'object' ? entry : {};
  };

  const buildHistoryWith = (action) => {
    const base = Array.isArray(commandState?.history) ? commandState.history.slice() : [];
    if (action && !base.includes(action)) {
      base.push(action);
    }
    return base;
  };

  const actionsAttempted = [];
  const noteActionAttempt = (action) => {
    const key = normalise(action);
    if (!key) {
      return;
    }
    if (!actionsAttempted.includes(key)) {
      actionsAttempted.push(key);
    }
  };

  if (baselineHead && !normalise(state.head_sha)) {
    await applyStateUpdate({ head_sha: baselineHead, head_recorded_at: new Date().toISOString() });
  }
  if (state.idempotency_key !== idempotencyKey) {
    await applyStateUpdate({ idempotency_key: idempotencyKey });
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
      `Agent state ${agentState.value || '(unknown)'} does not indicate completion; skipping sync gate.`,
    );
    applyFinalState({ action: 'skip', success: false, mode: 'skipped-agent-state', headMoved: false });
    await complete();
    return;
  }
  record('Preconditions', `Agent reported done (${agentState.value || 'done'}).`);

  let initialHeadInfo;
  try {
    initialHeadInfo = await fetchHead();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    record('Initial head fetch', `Failed: ${message}`);
    applyFinalState({ action: 'skip', success: false, mode: 'head-fetch-failed', headMoved: false });
    await complete();
    return;
  }

  if (isForkPull(initialHeadInfo)) {
    record('Initialisation', 'PR originates from a fork; skipping sync operations.');
    applyFinalState({ action: 'skip', success: false, mode: 'skipped-fork', headMoved: false });
    await complete();
    return;
  }

  const initialHead = initialHeadInfo.headSha || '';
  const headBranch = headBranchEnv || initialHeadInfo.headRef || '';
  const baseRef = baseBranch || initialHeadInfo.baseRef || '';
  if (!baselineHead) {
    baselineHead = normalise(state.head_sha) || initialHead;
  }
  if (!normalise(state.head_sha) && baselineHead) {
    await applyStateUpdate({ head_sha: baselineHead });
  }
  record('Baseline head', baselineHead || '(unavailable)');

  if (baselineHead && initialHead && baselineHead !== initialHead) {
    record('Head check', `Head already advanced to ${initialHead}; skipping sync gate.`);
    if (hasSyncLabel) {
      try {
        await github.rest.issues.removeLabel({ owner, repo, issue_number: prNumber, name: syncLabel });
        record('Sync label', appendRound(`Removed ${syncLabel}.`));
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        record('Sync label', appendRound(`Failed to remove ${syncLabel}: ${message}`));
      }
    } else {
      record('Sync label', appendRound(`${syncLabel} not present.`));
    }
    const elapsed = 0;
    const finalHead = initialHead;
    await applyStateUpdate({
      result: {
        status: 'success',
        mode: 'already-synced',
        merged_sha: finalHead,
        recorded_at: new Date().toISOString(),
      },
    });
    await persistLastInstruction(finalHead);
    record('Result', appendRound(`mode=already-synced sha=${finalHead || '(unknown)'} elapsed=${elapsed}ms`));
    applyFinalState({ action: 'skip', success: true, headMoved: true, finalHead, mode: 'already-synced' });
    await complete();
    return;
  }
  const instructionComment = Number.isFinite(commentIdEnv)
    ? { id: Number(commentIdEnv), url: commentUrlEnv || '' }
    : null;

  if (instructionComment?.id) {
    record('Instruction comment', appendRound(`id=${instructionComment.id}`));
  } else {
    record('Instruction comment', appendRound('unavailable; proceeding without comment context.'));
  }

  const commentInfo = instructionComment;

  const persistLastInstruction = async (finalHeadValue) => {
    const payload = {
      comment_id: commentInfo?.id ? String(commentInfo.id) : '',
      comment_url: commentInfo?.url || '',
      trace: commentTraceEnv || '',
      round: commentRoundEnv || '',
      head_sha: normalise(finalHeadValue) || '',
      recorded_at: new Date().toISOString(),
    };

    const filtered = Object.fromEntries(
      Object.entries(payload).filter(([key, value]) => {
        if (key === 'recorded_at') {
          return true;
        }
        return normalise(value) !== '';
      }),
    );

    if (Object.keys(filtered).length === 0) {
      return;
    }

    await applyStateUpdate({ last_instruction: filtered });
  };

  let finalAction = 'skip';

  const attemptCommand = async (action, label) => {
    const actionKey = normalise(action);
    if (!actionKey) {
      record(`${label} dispatch`, appendRound('skipped: action missing'));
      return { changed: false, dispatched: false, headSha: '' };
    }

    noteActionAttempt(actionKey);

    const existing = getCommandActionState(actionKey);
    const alreadyForTrace = existing.idempotency_key && existing.idempotency_key === idempotencyKey && existing.dispatched;
    const existingComment = existing.comment && typeof existing.comment === 'object' ? existing.comment : {};
    const alreadyCommentForTrace =
      existingComment.idempotency_key && existingComment.idempotency_key === idempotencyKey && existingComment.body;
    let dispatched = false;
    let dispatchTimestamp = existing.dispatched_at || new Date().toISOString();
    let reused = false;
    let commentPosted = false;
    let commentResult = null;

    if (alreadyForTrace) {
      reused = true;
      record(`Dispatch ${actionKey}`, appendRound('skipped: already dispatched for this trace.'));
    } else {
      dispatchTimestamp = new Date().toISOString();
      dispatched = await dispatchCommand({
        github,
        owner,
        repo,
        eventType: dispatchEventType,
        action: actionKey,
        prNumber,
        agentAlias,
        baseRef: baseRef || initialHeadInfo.baseRef || '',
        headRef: headBranch,
        headSha: baselineHead || initialHead || '',
        trace,
        round,
        commentInfo,
        idempotencyKey,
        roundTag,
        record,
      });
    }

    if (!alreadyForTrace && !dispatched) {
      if (alreadyCommentForTrace) {
        record('Comment command', appendRound('skipped: already posted for this trace.'));
      } else {
        commentResult = await postCommandComment({
          github,
          owner,
          repo,
          prNumber,
          action: actionKey,
          trace,
          round,
          record,
          appendRound,
        });
        commentPosted = Boolean(commentResult?.posted);
      }
    }

    let mergedEntry = mergeStateShallow(existing, {
      dispatched: alreadyForTrace ? existing.dispatched : dispatched,
      dispatched_at: dispatchTimestamp,
      idempotency_key: idempotencyKey,
      trace: trace || '',
      round: round || '',
      reused,
    });

    if (commentPosted && commentResult) {
      mergedEntry = mergeStateShallow(mergedEntry, {
        comment: {
          id: commentResult.commentId || existingComment.id || 0,
          url: commentResult.commentUrl || existingComment.url || '',
          body: commentResult.body || existingComment.body || '',
          posted_at: new Date().toISOString(),
          idempotency_key: idempotencyKey,
        },
      });
    } else if (alreadyCommentForTrace && existingComment) {
      mergedEntry = mergeStateShallow(mergedEntry, {
        comment: existingComment,
      });
    }

    await updateCommandState({
      history: buildHistoryWith(actionKey),
      last_action: actionKey,
      last_dispatched_at: mergedEntry.dispatched_at,
      [actionKey]: mergedEntry,
    });

    const shouldPoll = reused || dispatched || commentPosted || alreadyCommentForTrace;
    let pollResult = null;
    if (shouldPoll) {
      if (actionKey === 'create-pr') {
        const autoState = getCommandActionState(actionKey).auto_merge || {};
        const alreadyMergedForTrace =
          autoState.idempotency_key && autoState.idempotency_key === idempotencyKey && autoState.merged;
        if (alreadyMergedForTrace) {
          record('Create-pr auto-merge', appendRound('skipped: already merged for this trace.'));
        } else {
          const mergeResult = await mergeConnectorPullRequest({
            github,
            owner,
            repo,
            baseRef: headBranch || initialHeadInfo.headRef || '',
            trace,
            dispatchTimestamp,
            allowedLogins: automationLogins,
            mergeMethod,
            deleteBranch: deleteTempBranch,
            record,
            appendRound,
          });
          if (mergeResult.attempted) {
            await updateCommandState({
              [actionKey]: mergeStateShallow(getCommandActionState(actionKey), {
                auto_merge: {
                  merged: Boolean(mergeResult.merged),
                  pr_number: mergeResult.prNumber || autoState.pr_number || 0,
                  merged_at: mergeResult.merged ? new Date().toISOString() : autoState.merged_at,
                  branch_deleted: mergeResult.branchDeleted || false,
                  idempotency_key: idempotencyKey,
                },
              }),
            });
          }
        }
      }

      pollResult = await pollForHeadChange({
        fetchHead,
        initialSha: baselineHead || initialHead,
        timeoutMs: ttlLong,
        intervalMs: pollLong,
        label: `${actionKey}-wait`,
        core,
      });
      const outcomeUpdate = {
        result: pollResult.changed ? 'success' : 'timeout',
        resolved_at: new Date().toISOString(),
        resolved_sha: pollResult.headSha || '',
      };
      await updateCommandState({
        [actionKey]: mergeStateShallow(getCommandActionState(actionKey), outcomeUpdate),
      });
    }

    let message;
    if (pollResult?.changed) {
      message = `Branch advanced to ${pollResult.headSha || '(unknown)'}`;
      success = true;
      finalHead = pollResult.headSha;
      baselineHead = finalHead;
      mode = `dispatch-${actionKey}`;
      finalAction = actionKey;
    } else if (shouldPoll) {
      message = commentPosted && !dispatched ? 'Command comment posted; awaiting connector response.' : 'Branch unchanged after command wait.';
    } else if (!alreadyForTrace && !dispatched && !commentPosted && !alreadyCommentForTrace) {
      message = 'Dispatch unavailable; command not sent.';
    } else {
      message = 'Command already dispatched; awaiting external progress.';
    }

    record(`${label} result`, appendRound(message));

    return {
      changed: Boolean(pollResult?.changed),
      dispatched: dispatched || alreadyForTrace || commentPosted || alreadyCommentForTrace,
      headSha: pollResult?.headSha,
    };
  };

  const startTime = Date.now();
  let success = false;
  let finalHead = initialHead;
  let mode = 'none';

  const shortPoll = await pollForHeadChange({
    fetchHead,
    initialSha: baselineHead,
    timeoutMs: ttlShort,
    intervalMs: pollShort,
    label: 'comment-wait',
    core,
  });
  if (shortPoll.changed) {
    success = true;
    finalHead = shortPoll.headSha;
    baselineHead = finalHead;
    mode = 'already-synced';
    record('Initial poll', `Branch advanced to ${shortPoll.headSha}`);
  } else {
  record('Comment wait', appendRound('Head unchanged after comment TTL.'));
  }

  if (!success) {
    // As a guard, re-fetch the head once more before dispatching commands. The
    // short-poll window can miss a freshly advanced head on faster runners, so
    // this explicit check lets us bail out without emitting redundant commands.
    try {
      const freshHead = await fetchHead();
      if (baselineHead && freshHead?.headSha && freshHead.headSha !== baselineHead) {
        success = true;
        finalHead = freshHead.headSha;
        baselineHead = finalHead;
        mode = 'already-synced';
        record('Pre-dispatch check', appendRound(`Head advanced to ${freshHead.headSha}`));
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      core?.warning?.(`Failed to refresh head before dispatch: ${message}`);
    }
  }

  if (!success) {
    await attemptCommand('update-branch', 'Update-branch');
  }

  if (!success) {
    await attemptCommand('create-pr', 'Create-pr');
  }

  if (!success) {
    const dispatchInfo = state.fallback_dispatch?.dispatched
      ? { dispatched: true, status: state.fallback_dispatch.status, dispatchedAt: state.fallback_dispatch.dispatched_at }
      : await dispatchFallbackWorkflow({
          github,
          owner,
          repo,
          baseRef,
          dispatchRef: baseRef || context.payload?.repository?.default_branch,
          prNumber,
          headRef: headBranch,
          headSha: baselineHead || initialHead,
          trace,
          round,
          agentAlias,
          commentInfo,
          idempotencyKey,
          record,
        });

    if (dispatchInfo.dispatched && !state.fallback_dispatch?.dispatched) {
      await applyStateUpdate({
        fallback_dispatch: {
          dispatched: true,
          status: dispatchInfo.status ?? 0,
          dispatched_at: dispatchInfo.dispatchedAt,
          workflow: 'agents-keepalive-branch-sync.yml',
        },
      });
    } else if (dispatchInfo.dispatched) {
      record('Fallback dispatch', appendRound('reuse previous dispatch record.'));
    }

    const existingRunId = state.fallback_dispatch?.run_id;
    const runInfo = await findFallbackRun({
      github,
      owner,
      repo,
      createdAfter: dispatchInfo.dispatchedAt || state.fallback_dispatch?.dispatched_at,
      existingRunId,
      core,
    });
    if (runInfo) {
      record(
        'Fallback run',
        appendRound(runInfo.html_url ? `run=${runInfo.html_url}` : `run_id=${runInfo.id}`),
      );
      if (!existingRunId || Number(existingRunId) !== Number(runInfo.id)) {
        await applyStateUpdate({
          fallback_dispatch: {
            ...(state.fallback_dispatch || {}),
            dispatched: true,
            status: state.fallback_dispatch?.status ?? dispatchInfo.status ?? 0,
            dispatched_at: state.fallback_dispatch?.dispatched_at || dispatchInfo.dispatchedAt,
            workflow: 'agents-keepalive-branch-sync.yml',
            run_id: runInfo.id,
            run_url: runInfo.html_url || '',
          },
        });
      }
    } else if (dispatchInfo.dispatched) {
      record('Fallback run', appendRound('pending discovery.'));
    }

    if (dispatchInfo.dispatched) {
      const longPoll = await pollForHeadChange({
        fetchHead,
        initialSha: baselineHead,
        timeoutMs: ttlLong,
        intervalMs: pollLong,
        label: 'fallback-wait',
        core,
      });
      if (longPoll.changed) {
        success = true;
        finalHead = longPoll.headSha;
        baselineHead = finalHead;
        mode = 'action-sync-pr';
        finalAction = 'create-pr';
        record('Fallback wait', appendRound(`Branch advanced to ${longPoll.headSha}`));
      } else {
        record('Fallback wait', appendRound('Branch unchanged after fallback TTL.'));
      }
    }
  }

  if (!success && (!mode || mode === 'none')) {
    mode = 'sync-timeout';
  }

  if (success) {
    await applyStateUpdate({
      head_sha: finalHead || '',
      head_recorded_at: new Date().toISOString(),
      result: {
        status: 'success',
        mode,
        merged_sha: finalHead || '',
        recorded_at: new Date().toISOString(),
      },
    });
    await persistLastInstruction(finalHead || baselineHead || initialHead);

    if (hasSyncLabel) {
      try {
        await github.rest.issues.removeLabel({ owner, repo, issue_number: prNumber, name: syncLabel });
        record('Sync label', `Removed ${syncLabel}.`);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        record('Sync label', `Failed to remove ${syncLabel}: ${message}`);
      }
    } else {
      record('Sync label', `${syncLabel} not present.`);
    }
    const elapsed = Date.now() - startTime;
    record('Result', appendRound(`mode=${mode || 'unknown'} sha=${finalHead || '(unknown)'} elapsed=${elapsed}ms`));
    applyFinalState({ action: finalAction || 'skip', success: true, headMoved: true, finalHead, mode });
    await complete();
    return;
  }

  await applyStateUpdate({
    result: {
      status: 'timeout',
      mode: mode || 'sync-timeout',
      merged_sha: finalHead || '',
      recorded_at: new Date().toISOString(),
    },
  });

  if (!hasSyncLabel) {
    try {
      await github.rest.issues.addLabels({ owner, repo, issue_number: prNumber, labels: [syncLabel] });
      record('Sync label', appendRound(`Applied ${syncLabel}.`));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      record('Sync label', appendRound(`Failed to apply ${syncLabel}: ${message}`));
    }
  } else {
    record('Sync label', appendRound(`${syncLabel} already present.`));
  }

  if (hasDebugLabel) {
    const traceToken = trace ? `{${trace}}` : '{trace-missing}';
    const attemptedSummary = actionsAttempted.length ? actionsAttempted.join('/') : 'none';
    const escalationMessage = `Keepalive ${round || '?'} ${traceToken} escalation: agent "done" but branch unchanged after comment + commands=${attemptedSummary} + fallback.`;
    try {
      await github.rest.issues.createComment({
        owner,
        repo,
        issue_number: Number.isFinite(issueNumber) ? Number(issueNumber) : prNumber,
        body: escalationMessage,
      });
      record('Escalation comment', appendRound('Posted debug escalation comment.'));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      record('Escalation comment', appendRound(`Failed to post escalation comment: ${message}`));
    }
  } else {
    record('Escalation comment', appendRound('Debug label absent; no comment posted.'));
  }

  const timeoutMessage = appendRound(
    `mode=${mode || 'sync-timeout'} trace:${trace || 'missing'} elapsed=${Date.now() - startTime}ms`,
  );
  record('Result', timeoutMessage);
  applyFinalState({ action: 'escalate', success: false, headMoved: false, mode, finalHead });
  await complete();
}

module.exports = {
  runKeepalivePostWork,
};
