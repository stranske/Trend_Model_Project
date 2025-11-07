'use strict';

const KEEPALIVE_LABEL = 'agents:keepalive';
const AGENT_LABEL_PREFIX = 'agent:';
const MAX_RUNS_PREFIX = 'agents:max-runs:';
const SYNC_REQUIRED_LABEL = 'agents:sync-required';
const DEFAULT_RUN_CAP = 2;
const ORCHESTRATOR_WORKFLOW_FILE = 'agents-70-orchestrator.yml';

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
    if (Number.isFinite(parsed) && parsed > 0) {
      return parsed;
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

async function detectHumanActivation({ github, owner, repo, prNumber, aliases, comments }) {
  if (!Array.isArray(aliases) || aliases.length === 0) {
    return false;
  }
  const patterns = buildMentionPatterns(aliases);
  if (Array.isArray(comments) && comments.length > 0) {
    for (const comment of comments) {
      if (hasHumanMention(comment, patterns)) {
        return true;
      }
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
      if (hasHumanMention(comment, patterns)) {
        return true;
      }
    }
  }

  return false;
}

async function fetchGateStatus({ github, owner, repo, headSha }) {
  if (!headSha) {
    return { concluded: false };
  }
  try {
    const runs = await github.paginate(github.rest.checks.listForRef, {
      owner,
      repo,
      ref: headSha,
      check_name: 'Gate',
      status: 'completed',
      per_page: 100,
    });
    const concluded = runs.some((run) => String(run?.status || '').toLowerCase() === 'completed');
    return { concluded };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { concluded: false, error: message };
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
}) {
  const statuses = ['queued', 'in_progress'];
  let activeRuns = 0;

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

  for (const status of statuses) {
    try {
      const runs = await github.paginate(github.rest.actions.listWorkflowRuns, {
        owner,
        repo,
        workflow_id: workflowFile,
        status,
        per_page: 100,
      });
      for (const run of runs) {
        if (isSameRun(run)) {
          continue;
        }
        if (matchesPull(run)) {
          activeRuns += 1;
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return { activeRuns, error: message };
    }
  }

  return { activeRuns };
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
    requireGateSuccess = true,
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
      underRunCap: false,
      runCap: DEFAULT_RUN_CAP,
      activeRuns: 0,
      agentAliases: [],
      primaryAgent: '',
      headSha: '',
      headRef: '',
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
        underRunCap: false,
        runCap: DEFAULT_RUN_CAP,
        activeRuns: 0,
        agentAliases: [],
        primaryAgent: '',
        headSha: '',
        headRef: '',
        error: message,
      };
    }
  }

  const headSha = inputHeadSha || pr?.head?.sha || '';
  const headRef = pr?.head?.ref || '';
  const labels = Array.isArray(pr?.labels) ? pr.labels : [];
  const labelNames = extractLabelNames(labels);
  const hasKeepaliveLabel = labelNames.includes(KEEPALIVE_LABEL);
  const hasSyncRequiredLabel = labelNames.includes(SYNC_REQUIRED_LABEL);
  const agentAliases = extractAgentAliases(labels);
  const primaryAgent = agentAliases[0] || '';

  let humanActivation = false;
  if (hasKeepaliveLabel && agentAliases.length > 0) {
    try {
      humanActivation = await detectHumanActivation({
        github,
        owner,
        repo,
        prNumber,
        aliases: agentAliases,
        comments,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      core.warning(`Failed to evaluate human activation comments: ${message}`);
      humanActivation = false;
    }
  }

  const { concluded: gateConcluded, error: gateError } = await fetchGateStatus({
    github,
    owner,
    repo,
    headSha,
  });
  if (gateError) {
    core.warning(`Gate status check failed: ${gateError}`);
  }

  const runCap = parseRunCap(labels);
  const { activeRuns, error: runError } = await countActiveRuns({
    github,
    owner,
    repo,
    prNumber,
    headSha,
    headRef,
    currentRunId,
    workflowFile,
  });
  if (runError) {
    core.warning(`Unable to count active orchestrator runs: ${runError}`);
  }
  const underRunCap = activeRuns < runCap;

  let ok = true;
  let reason = '';

  if (hasSyncRequiredLabel) {
    ok = false;
    reason = 'sync-required';
  } else if (!hasKeepaliveLabel) {
    ok = false;
    reason = 'keepalive-label-missing';
  } else if (requireHumanActivation && !humanActivation) {
    ok = false;
    reason = 'no-human-activation';
  } else if (!gateConcluded) {
    ok = false;
    reason = 'gate-not-concluded';
  } else if (!underRunCap) {
    ok = false;
    reason = 'run-cap-reached';
  }

  return {
    ok,
    reason,
    hasKeepaliveLabel,
    hasHumanActivation: humanActivation,
    gateConcluded,
    underRunCap,
    runCap,
    activeRuns,
    agentAliases,
    primaryAgent,
    headSha,
    headRef,
    hasSyncRequiredLabel,
  };
}

module.exports = {
  evaluateKeepaliveGate,
};
