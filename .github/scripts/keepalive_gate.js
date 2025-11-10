'use strict';

const KEEPALIVE_LABEL = 'agents:keepalive';
const AGENT_LABEL_PREFIX = 'agent:';
const MAX_RUNS_PREFIX = 'agents:max-runs:';
const SYNC_REQUIRED_LABEL = 'agents:sync-required';
const ACTIVATED_LABEL = 'agents:activated';
const DEFAULT_RUN_CAP = 2;
const MIN_RUN_CAP = 1;
const MAX_RUN_CAP = 5;
const RECENT_RUN_LOOKBACK_MINUTES = 5;
const ORCHESTRATOR_WORKFLOW_FILE = 'agents-70-orchestrator.yml';
const WORKER_WORKFLOW_FILE = 'agents-72-codex-belt-worker.yml';
const GATE_WORKFLOW_FILE = 'pr-00-gate.yml';

function normaliseLabelName(name) {
  return String(name || '')
    .trim()
    .toLowerCase();
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
        return normaliseLabelName(entry);
      }
      if (typeof entry.name === 'string') {
        return normaliseLabelName(entry.name);
      }
      return '';
    })
    .filter(Boolean);
}

function extractAgentAliases(labels) {
  const names = extractLabelNames(labels);
  const aliases = new Set();
  for (const name of names) {
    if (!name.startsWith(AGENT_LABEL_PREFIX)) {
      continue;
    }
    const alias = name.slice(AGENT_LABEL_PREFIX.length).trim();
    if (alias) {
      aliases.add(alias);
    }
  }
  return Array.from(aliases);
}

function parseRunCap(labels) {
  const names = extractLabelNames(labels);
  for (const name of names) {
    if (!name.startsWith(MAX_RUNS_PREFIX)) {
      continue;
    }
    const value = name.slice(MAX_RUNS_PREFIX.length).trim();
    const parsed = Number.parseInt(value, 10);
    if (Number.isFinite(parsed)) {
      const clamped = Math.min(MAX_RUN_CAP, Math.max(MIN_RUN_CAP, parsed));
      return clamped;
    }
  }
  return DEFAULT_RUN_CAP;
}

function escapeForRegex(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function buildMentionPatterns(aliases) {
  return aliases.map((alias) => {
    const escaped = escapeForRegex(alias);
    return new RegExp(`(^|[^A-Za-z0-9_-])@${escaped}(?![A-Za-z0-9_-])`, 'i');
  });
}

function isHumanUser(user) {
  if (!user) {
    return false;
  }
  const type = String(user.type || '').toLowerCase();
  if (type === 'bot' || type === 'app') {
    return false;
  }
  const login = String(user.login || '').toLowerCase();
  if (!login) {
    return false;
  }
  if (login.endsWith('[bot]')) {
    return false;
  }
  return true;
}

function hasHumanMention(comment, patterns) {
  if (!comment || patterns.length === 0) {
    return false;
  }
  if (!isHumanUser(comment.user)) {
    return false;
  }
  const body = String(comment.body || '');
  if (!body) {
    return false;
  }
  return patterns.some((pattern) => pattern.test(body));
}

function normaliseTimestamp(value) {
  if (!value) {
    return 0;
  }
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) {
    return 0;
  }
  return parsed;
}

function selectLatestComment(current, candidate) {
  if (!candidate || !candidate.id) {
    return current;
  }
  if (!current) {
    return candidate;
  }
  const currentTime = normaliseTimestamp(current.updated_at || current.created_at);
  const candidateTime = normaliseTimestamp(candidate.updated_at || candidate.created_at);
  if (candidateTime > currentTime) {
    return candidate;
  }
  return current;
}

function sanitiseComment(comment) {
  if (!comment || typeof comment !== 'object') {
    return null;
  }
  const idRaw = comment.id;
  const id = idRaw === undefined || idRaw === null ? '' : String(idRaw);
  const htmlUrl = typeof comment.html_url === 'string' ? comment.html_url : '';
  const userLogin = comment?.user?.login ?? '';
  const userType = comment?.user?.type ?? '';
  return {
    id,
    body: typeof comment.body === 'string' ? comment.body : '',
    html_url: htmlUrl,
    user: userLogin
      ? {
          login: userLogin,
          type: userType,
        }
      : null,
    created_at: comment.created_at || '',
    updated_at: comment.updated_at || '',
  };
}

async function detectHumanActivation({ core, github, owner, repo, prNumber, aliases, comments }) {
  const logPrefix = '[human-activation]';
  const logInfo = (message) => {
    if (core && typeof core.info === 'function') {
      core.info(`${logPrefix} ${message}`);
    }
  };

  if (!Array.isArray(aliases) || aliases.length === 0) {
    logInfo('Skipped scan: no agent aliases provided.');
    return null;
  }
  const patterns = buildMentionPatterns(aliases);
  let latest = null;

  logInfo(
    `Scanning PR #${prNumber ?? 'unknown'} for human activation via aliases [${aliases.join(', ')}] (prefetched comments=${
      Array.isArray(comments) ? comments.length : 0
    }).`
  );

  const considerComment = (comment) => {
    if (!comment) {
      return;
    }
    if (hasHumanMention(comment, patterns)) {
      const updated = selectLatestComment(latest, comment);
      if (updated !== latest) {
        const author = comment?.user?.login || '(unknown)';
        const created = comment?.created_at || comment?.updated_at || '(no timestamp)';
        logInfo(
          `Found activation candidate comment ${comment.id || '(no id)'} by ${author} at ${created}; replacing previous candidate ${
            latest?.id || 'none'
          }.`
        );
        latest = updated;
      }
    }
  };

  if (Array.isArray(comments) && comments.length > 0) {
    for (const comment of comments) {
      considerComment(comment);
    }
  }

  const iterator = github.paginate.iterator(github.rest.issues.listComments, {
    owner,
    repo,
    issue_number: prNumber,
    per_page: 100,
  });

  for await (const page of iterator) {
    const entries = Array.isArray(page?.data) ? page.data : page;
    if (!Array.isArray(entries)) {
      continue;
    }
    for (const comment of entries) {
      considerComment(comment);
    }
  }

  if (latest) {
    const author = latest?.user?.login || '(unknown)';
    const created = latest?.created_at || latest?.updated_at || '(no timestamp)';
    logInfo(`Selected activation comment ${latest.id || '(no id)'} by ${author} at ${created}.`);
  } else {
    logInfo('No qualifying human activation comment found.');
  }

  return latest;
}

const STATUS_RANK = {
  completed: 3,
  in_progress: 2,
  queued: 1,
};

const CONCLUSION_RANK = {
  success: 6,
  neutral: 5,
  failure: 4,
  skipped: 3,
  cancelled: 2,
  timed_out: 1,
  action_required: 0,
  stale: 0,
};

function scoreWorkflowRun(run) {
  if (!run) {
    return null;
  }
  const status = String(run.status || '').toLowerCase();
  const conclusion = String(run.conclusion || '').toLowerCase();
  const startedAt = normaliseTimestamp(run.run_started_at || run.created_at);
  const runNumber = Number(run.run_number || 0) || 0;
  const runId = Number(run.id || 0) || 0;
  return {
    run,
    status,
    conclusion,
    startedAt,
    runNumber,
    runId,
    statusRank: STATUS_RANK[status] ?? 0,
    conclusionRank: CONCLUSION_RANK[conclusion] ?? 0,
  };
}

function compareWorkflowRunScores(a, b) {
  if (!a && !b) {
    return 0;
  }
  if (!a) {
    return 1;
  }
  if (!b) {
    return -1;
  }
  if (a.startedAt !== b.startedAt) {
    return b.startedAt - a.startedAt;
  }
  if (a.statusRank !== b.statusRank) {
    return b.statusRank - a.statusRank;
  }
  if (a.conclusionRank !== b.conclusionRank) {
    return b.conclusionRank - a.conclusionRank;
  }
  if (a.runNumber !== b.runNumber) {
    return b.runNumber - a.runNumber;
  }
  if (a.runId !== b.runId) {
    return b.runId - a.runId;
  }
  return 0;
}

async function fetchGateStatus({ github, owner, repo, headSha }) {
  const normalisedSha = String(headSha || '').trim();
  if (!normalisedSha) {
    return { found: false, success: false, status: '', conclusion: '' };
  }
  try {
    const runs = await github.paginate(github.rest.actions.listWorkflowRuns, {
      owner,
      repo,
      workflow_id: GATE_WORKFLOW_FILE,
      head_sha: normalisedSha,
      per_page: 100,
    });

    const scoredRuns = [];
    for (const run of runs) {
      if (!run) {
        continue;
      }
      const runSha = String(run.head_sha || '').trim();
      if (runSha && runSha !== normalisedSha) {
        continue;
      }
      const scored = scoreWorkflowRun(run);
      if (scored) {
        scoredRuns.push(scored);
      }
    }

    if (scoredRuns.length === 0) {
      return { found: false, success: false, status: '', conclusion: '' };
    }

    scoredRuns.sort(compareWorkflowRunScores);
    const latest = scoredRuns[0]?.run;

    if (!latest) {
      return { found: false, success: false, status: '', conclusion: '' };
    }

    const status = String(latest.status || '').toLowerCase();
    const conclusion = String(latest.conclusion || '').toLowerCase();
    const success = status === 'completed' && conclusion === 'success';

    return {
      found: true,
      success,
      status,
      conclusion,
      run: latest,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { found: false, success: false, status: '', conclusion: '', error: message };
  }
}

async function countActiveRuns({
  github,
  owner,
  repo,
  prNumber,
  headSha,
  headRef,
  currentRunId,
  workflowFile = ORCHESTRATOR_WORKFLOW_FILE,
  recentWindowMinutes = RECENT_RUN_LOOKBACK_MINUTES,
}) {
  const statuses = ['queued', 'in_progress'];
  const inflightRunIds = new Set();
  const activeRunIds = new Set();
  const recentRunIds = new Set();

  const workflowIds = Array.isArray(workflowFile)
    ? workflowFile.filter(Boolean)
    : [workflowFile].filter(Boolean);

  const isSameRun = (run) => {
    if (!currentRunId) {
      return false;
    }
    const runId = Number(run?.id || 0);
    const parsedCurrent = Number(currentRunId);
    if (!Number.isFinite(runId) || !Number.isFinite(parsedCurrent)) {
      return false;
    }
    return runId === parsedCurrent;
  };

  const matchesPull = (run) => {
    if (!run) {
      return false;
    }
    if (Array.isArray(run.pull_requests) && run.pull_requests.length > 0) {
      for (const pr of run.pull_requests) {
        const candidate = Number(pr?.number);
        if (Number.isFinite(candidate) && candidate === prNumber) {
          return true;
        }
      }
    }
    if (headSha && run.head_sha === headSha) {
      return true;
    }
    if (headRef && run.head_branch === headRef) {
      return true;
    }
    return false;
  };

  for (const workflowId of workflowIds) {
    if (!workflowId) {
      continue;
    }
    for (const status of statuses) {
      try {
        const runs = await github.paginate(github.rest.actions.listWorkflowRuns, {
          owner,
          repo,
          workflow_id: workflowId,
          status,
          per_page: 100,
        });
        for (const run of runs) {
          if (isSameRun(run)) {
            continue;
          }
          if (!matchesPull(run)) {
            continue;
          }
          const runId = Number(run?.id || 0);
          if (!Number.isFinite(runId) || runId <= 0) {
            continue;
          }
          inflightRunIds.add(runId);
          activeRunIds.add(runId);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return {
          activeRuns: activeRunIds.size,
          inFlightRuns: inflightRunIds.size,
          recentRuns: recentRunIds.size,
          error: message,
        };
      }
    }
  }

  const windowMinutes = Number.isFinite(recentWindowMinutes) ? Number(recentWindowMinutes) : 0;
  if (windowMinutes > 0) {
    const cutoff = Date.now() - windowMinutes * 60 * 1000;
    for (const workflowId of workflowIds) {
      if (!workflowId) {
        continue;
      }
      try {
        await github.paginate(
          github.rest.actions.listWorkflowRuns,
          {
            owner,
            repo,
            workflow_id: workflowId,
            status: 'completed',
            per_page: 100,
          },
          (response, done) => {
            let shouldStop = false;
            const entries = Array.isArray(response?.data) ? response.data : [];
            for (const run of entries) {
              if (!run) {
                continue;
              }
              const timestamp =
                Date.parse(run.updated_at || run.run_started_at || run.created_at || '') || 0;
              if (timestamp && timestamp < cutoff) {
                shouldStop = true;
                break;
              }
              if (isSameRun(run) || !matchesPull(run)) {
                continue;
              }
              const runId = Number(run?.id || 0);
              if (!Number.isFinite(runId) || runId <= 0) {
                continue;
              }
              activeRunIds.add(runId);
              if (!inflightRunIds.has(runId)) {
                recentRunIds.add(runId);
              }
            }
            if (shouldStop && typeof done === 'function') {
              done();
            }
            return [];
          }
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return {
          activeRuns: activeRunIds.size,
          inFlightRuns: inflightRunIds.size,
          recentRuns: recentRunIds.size,
          error: message,
        };
      }
    }
  }

  return {
    activeRuns: activeRunIds.size,
    inFlightRuns: inflightRunIds.size,
    recentRuns: recentRunIds.size,
    recentWindowMinutes: windowMinutes,
  };
}

async function evaluateKeepaliveGate({ core, github, context, options = {} }) {
  const { owner, repo } = context.repo || {};
  if (!owner || !repo) {
    throw new Error('Repository context missing owner or repo.');
  }

  const {
    prNumber: rawPrNumber,
    headSha: inputHeadSha,
  requireHumanActivation = false,
  allowPendingGate = false,
  requireGateSuccess = false,
    comments,
    pullRequest,
    currentRunId,
    workflowFile,
  } = options;

  let prNumber = Number(rawPrNumber);
  if (!Number.isFinite(prNumber) || prNumber <= 0) {
    const candidate = Number(pullRequest?.number);
    if (Number.isFinite(candidate) && candidate > 0) {
      prNumber = candidate;
    }
  }

  if (!Number.isFinite(prNumber) || prNumber <= 0) {
    return {
      ok: false,
      reason: 'missing-pr-number',
      hasKeepaliveLabel: false,
      hasHumanActivation: false,
      gateConcluded: false,
      gateSucceeded: false,
      underRunCap: false,
      runCap: DEFAULT_RUN_CAP,
      activeRuns: 0,
      agentAliases: [],
      primaryAgent: '',
      headSha: '',
      headRef: '',
      hasSyncRequiredLabel: false,
      hasActivatedLabel: false,
      requireHumanActivation: false,
      activationComment: null,
      gateStatus: { found: false, success: false, status: '', conclusion: '' },
      pendingGate: false,
    };
  }

  let pr = pullRequest || null;
  if (!pr) {
    try {
      const response = await github.rest.pulls.get({ owner, repo, pull_number: prNumber });
      pr = response.data;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return {
        ok: false,
        reason: 'pr-fetch-failed',
        hasKeepaliveLabel: false,
        hasHumanActivation: false,
        gateConcluded: false,
        gateSucceeded: false,
        underRunCap: false,
        runCap: DEFAULT_RUN_CAP,
        activeRuns: 0,
        agentAliases: [],
        primaryAgent: '',
        headSha: '',
        headRef: '',
        hasSyncRequiredLabel: false,
        hasActivatedLabel: false,
        requireHumanActivation: false,
        activationComment: null,
        gateStatus: { found: false, success: false, status: '', conclusion: '' },
        pendingGate: false,
        error: message,
      };
    }
  }

  const headSha = inputHeadSha || pr?.head?.sha || '';
  const headRef = pr?.head?.ref || '';
  const labels = Array.isArray(pr?.labels) ? pr.labels : [];
  const labelNames = extractLabelNames(labels);
  const hasKeepaliveLabel = labelNames.includes(KEEPALIVE_LABEL);
  const hasActivatedLabel = labelNames.includes(ACTIVATED_LABEL);
  const hasSyncRequiredLabel = labelNames.includes(SYNC_REQUIRED_LABEL);
  const agentAliases = extractAgentAliases(labels);
  const primaryAgent = agentAliases[0] || '';

  let humanActivation = false;
  let activationComment = null;
  const shouldCheckHumanActivation = hasKeepaliveLabel && agentAliases.length > 0;
  const requireHuman = Boolean(requireHumanActivation) && !hasActivatedLabel;

  if (shouldCheckHumanActivation) {
    try {
      activationComment = await detectHumanActivation({
        core,
        github,
        owner,
        repo,
        prNumber,
        aliases: agentAliases,
        comments,
      });
      humanActivation = Boolean(activationComment);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      core.warning(`Failed to evaluate human activation comments: ${message}`);
      humanActivation = false;
    }
  }

  const gateStatus = await fetchGateStatus({
    github,
    owner,
    repo,
    headSha,
  });
  if (gateStatus.error) {
    core.warning(`Gate status check failed: ${gateStatus.error}`);
  }

  const gateFound = gateStatus.found === true;
  const gateConcluded = gateStatus.status === 'completed';
  const gateSucceeded = gateConcluded && gateStatus.success === true;

  const runCap = parseRunCap(labels);
  const workflowCandidates = new Set();
  if (Array.isArray(workflowFile)) {
    for (const candidate of workflowFile) {
      if (candidate) {
        workflowCandidates.add(candidate);
      }
    }
  } else if (workflowFile) {
    workflowCandidates.add(workflowFile);
  }
  workflowCandidates.add(ORCHESTRATOR_WORKFLOW_FILE);
  workflowCandidates.add(WORKER_WORKFLOW_FILE);

  const workflowList = Array.from(workflowCandidates);

  const {
    activeRuns,
    inFlightRuns,
    recentRuns,
    recentWindowMinutes,
    error: runError,
  } = await countActiveRuns({
    github,
    owner,
    repo,
    prNumber,
    headSha,
    headRef,
    currentRunId,
    workflowFile: workflowList,
  });
  if (runError) {
    core.warning(`Unable to count active orchestrator runs: ${runError}`);
  }
  const underRunCap = activeRuns < runCap;

  let ok = true;
  let reason = 'ok';
  let pendingGate = false;

  if (hasSyncRequiredLabel) {
    ok = false;
    reason = 'sync-required';
  } else if (!hasKeepaliveLabel) {
    ok = false;
    reason = 'keepalive-label-missing';
  } else if (requireHuman && !humanActivation) {
    ok = false;
    reason = 'no-human-activation';
  }

  if (ok) {
    if (!gateFound) {
      if (allowPendingGate) {
        pendingGate = true;
        if (reason === 'ok') {
          reason = 'gate-pending';
        }
      } else {
        ok = false;
        reason = 'gate-run-missing';
      }
    } else if (!gateConcluded) {
      if (allowPendingGate) {
        pendingGate = true;
        if (reason === 'ok') {
          reason = 'gate-pending';
        }
      } else {
        ok = false;
        reason = 'gate-not-concluded';
      }
    } else if (requireGateSuccess && !gateSucceeded) {
      ok = false;
      reason = 'gate-not-success';
    }
  }

  if (ok && !underRunCap) {
    ok = false;
    reason = 'run-cap-reached';
  }

  return {
    ok,
    reason,
    hasKeepaliveLabel,
    hasHumanActivation: humanActivation,
    gateConcluded,
    gateSucceeded,
    underRunCap,
    runCap,
    activeRuns,
    inFlightRuns,
    recentRuns,
    recentWindowMinutes,
    agentAliases,
    primaryAgent,
    headSha,
    headRef,
    hasSyncRequiredLabel,
    hasActivatedLabel,
    requireHumanActivation: requireHuman,
    activationComment: sanitiseComment(activationComment),
    gateStatus,
    pendingGate,
  };
}

module.exports = {
  evaluateKeepaliveGate,
  countActiveRuns,
  RECENT_RUN_LOOKBACK_MINUTES,
};
