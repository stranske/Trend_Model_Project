'use strict';

const KEEPALIVE_LABEL = 'agents:keepalive';
const AGENT_LABEL_PREFIX = 'agent:';
const KEEPALIVE_MARKER = '<!-- codex-keepalive-marker -->';
const GATE_WORKFLOWS = ['pr-00-gate.yml', 'pr-00-gate.yaml'];
const ORCHESTRATOR_WORKFLOWS = ['agents-70-orchestrator.yml', 'agents-70-orchestrator.yaml'];
const DEFAULT_RUN_CAP = 2;

function normaliseString(value) {
  if (value == null) {
    return '';
  }
  return String(value).trim();
}

function normaliseLabelName(value) {
  return normaliseString(value).toLowerCase();
}

function extractLabels(pr) {
  const rawLabels = pr?.labels;
  const labels = [];
  if (Array.isArray(rawLabels)) {
    for (const entry of rawLabels) {
      if (!entry) {
        continue;
      }
      if (typeof entry === 'string') {
        labels.push(entry);
        continue;
      }
      if (entry.name) {
        labels.push(String(entry.name));
      }
    }
  }
  return labels;
}

function toAliasEntry(label) {
  const trimmed = normaliseString(label);
  if (!trimmed.toLowerCase().startsWith(AGENT_LABEL_PREFIX)) {
    return null;
  }
  const alias = normaliseString(trimmed.slice(AGENT_LABEL_PREFIX.length));
  if (!alias) {
    return null;
  }
  return {
    label: trimmed,
    alias,
    normalised: alias.toLowerCase(),
  };
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function isAutomationUser(login, type) {
  const lowered = normaliseString(login).toLowerCase();
  if (!lowered) {
    return false;
  }
  if (lowered.endsWith('[bot]')) {
    return true;
  }
  if (lowered.endsWith('-bot')) {
    return true;
  }
  const loweredType = normaliseString(type).toLowerCase();
  if (loweredType === 'bot' || loweredType === 'app') {
    return true;
  }
  const automationLogins = new Set([
    'chatgpt-codex-connector',
    'stranske-automation-bot',
    'github-actions',
    'dependabot[bot]',
    'dependabot',
    'copilot',
  ]);
  return automationLogins.has(lowered);
}

function findMentionAlias(body, aliasEntries) {
  if (!aliasEntries.length) {
    return null;
  }
  const source = normaliseString(body).toLowerCase();
  if (!source.includes('@')) {
    return null;
  }
  for (const entry of aliasEntries) {
    const alias = entry.normalised;
    if (!alias) {
      continue;
    }
    const needle = `@${alias}`;
    let index = source.indexOf(needle);
    while (index !== -1) {
      const before = index === 0 ? '' : source[index - 1];
      const afterIndex = index + needle.length;
      const after = afterIndex < source.length ? source[afterIndex] : '';
      const beforeAllowed = !before || /[^a-z0-9_/-]/.test(before);
      const afterAllowed = !after || /[^a-z0-9_/-]/.test(after);
      if (beforeAllowed && afterAllowed) {
        return entry;
      }
      index = source.indexOf(needle, index + needle.length);
    }
  }
  return null;
}

function extractKeepaliveAlias(body, aliasEntries) {
  if (!body || !body.includes('@')) {
    return '';
  }
  const aliasByNormalised = new Map();
  for (const entry of aliasEntries) {
    if (entry.normalised) {
      aliasByNormalised.set(entry.normalised, entry.alias);
    }
  }
  const lines = String(body)
    .replace(/\r\n/g, '\n')
    .split('\n')
    .map((line) => line.trim());
  for (const line of lines) {
    if (!line || line.startsWith('<!--')) {
      continue;
    }
    if (!line.startsWith('@')) {
      continue;
    }
    const mention = line.slice(1).split(/[^A-Za-z0-9_-]/, 1)[0] || '';
    if (!mention) {
      continue;
    }
    const normalised = mention.toLowerCase();
    if (aliasByNormalised.has(normalised)) {
      return aliasByNormalised.get(normalised);
    }
    return mention;
  }
  return '';
}

function analyseComments(comments, aliasEntries) {
  const result = {
    hasHumanActivation: false,
    humanAlias: '',
    hasPriorKeepalive: false,
    keepaliveAlias: '',
    highestRound: 0,
  };
  if (!Array.isArray(comments) || comments.length === 0) {
    return result;
  }

  let latestHumanTimestamp = 0;
  let latestKeepaliveTimestamp = 0;

  for (const comment of comments) {
    if (!comment) {
      continue;
    }
    const body = String(comment.body || '');
    const createdAtRaw = comment.created_at || comment.updated_at || null;
    const createdAt = createdAtRaw ? Date.parse(createdAtRaw) : NaN;
    const timestamp = Number.isFinite(createdAt) ? createdAt : 0;

    if (body.includes(KEEPALIVE_MARKER)) {
      result.hasPriorKeepalive = true;
      if (timestamp >= latestKeepaliveTimestamp) {
        latestKeepaliveTimestamp = timestamp;
        result.keepaliveAlias = extractKeepaliveAlias(body, aliasEntries) || result.keepaliveAlias;
      }
      const roundMatch = body.match(/<!--\s*keepalive-round\s*:?\s*(\d+)\s*-->/i);
      if (roundMatch) {
        const parsed = Number.parseInt(roundMatch[1], 10);
        if (Number.isFinite(parsed) && parsed > result.highestRound) {
          result.highestRound = parsed;
        }
      }
    }

    const user = comment.user || comment.author || {};
    const login = user.login || user.name || '';
    const type = user.type || '';
    if (isAutomationUser(login, type)) {
      continue;
    }

    const mention = findMentionAlias(body, aliasEntries);
    if (!mention) {
      continue;
    }

    if (!result.hasHumanActivation || timestamp >= latestHumanTimestamp) {
      result.hasHumanActivation = true;
      latestHumanTimestamp = timestamp;
      result.humanAlias = mention.alias;
    }
  }

  return result;
}

function parseRunCap(labels) {
  const entries = Array.isArray(labels) ? labels : [];
  for (const entry of entries) {
    const name = normaliseString(entry);
    if (!name.toLowerCase().startsWith('agents:max-runs:')) {
      continue;
    }
    const value = name.slice('agents:max-runs:'.length).trim();
    const parsed = Number.parseInt(value, 10);
    if (Number.isFinite(parsed) && parsed > 0) {
      return parsed;
    }
  }
  return DEFAULT_RUN_CAP;
}

async function fetchComments({ github, owner, repo, prNumber }) {
  try {
    const comments = await github.paginate(
      github.rest.issues.listComments,
      {
        owner,
        repo,
        issue_number: prNumber,
        per_page: 100,
      },
      (response) => Array.isArray(response.data) ? response.data : []
    );
    return comments;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Unable to load PR comments: ${message}`);
  }
}

async function fetchPullRequest({ github, owner, repo, prNumber }) {
  try {
    const response = await github.rest.pulls.get({ owner, repo, pull_number: prNumber });
    return response.data;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Unable to load pull request #${prNumber}: ${message}`);
  }
}

function collectCandidateNumbers(value, bucket) {
  const raw = normaliseString(value);
  if (!raw) {
    return;
  }
  const patterns = [
    /keepalive[_-]?pr\s*[:=#-]?\s*(\d+)/gi,
    /pull\s*request\s*[:=#-]?\s*(\d+)/gi,
    /pr\s*[:=#-]?\s*(\d+)/gi,
    /issue\s*[:=#-]?\s*(\d+)/gi,
    /pull\/(\d+)/gi,
    /#(\d+)/g,
  ];
  for (const pattern of patterns) {
    let match;
    while ((match = pattern.exec(raw)) !== null) {
      const candidate = normaliseString(match[1]);
      if (candidate) {
        bucket.add(candidate);
      }
    }
  }
}

function runMatchesPr(run, target) {
  const normalizedTarget = normaliseString(target);
  if (!normalizedTarget) {
    return false;
  }
  const candidates = new Set();

  if (Array.isArray(run?.pull_requests)) {
    for (const pr of run.pull_requests) {
      const number = pr?.number;
      if (Number.isFinite(number)) {
        candidates.add(String(number));
      }
    }
  }

  collectCandidateNumbers(run?.display_title, candidates);
  collectCandidateNumbers(run?.name, candidates);
  collectCandidateNumbers(run?.head_branch, candidates);
  collectCandidateNumbers(run?.path, candidates);
  collectCandidateNumbers(run?.html_url, candidates);
  collectCandidateNumbers(run?.head_commit && run.head_commit.message, candidates);

  return candidates.has(normalizedTarget);
}

async function countActiveRuns({ github, owner, repo, prNumber, currentRunId }) {
  const statuses = ['in_progress', 'queued'];
  const matching = new Set();

  for (const workflowId of ORCHESTRATOR_WORKFLOWS) {
    for (const status of statuses) {
      try {
        const runs = await github.paginate(
          github.rest.actions.listWorkflowRuns,
          {
            owner,
            repo,
            workflow_id: workflowId,
            status,
            per_page: 100,
          },
          (response) => {
            const data = response.data?.workflow_runs;
            if (Array.isArray(data)) {
              return data;
            }
            return [];
          }
        );
        for (const run of runs) {
          const runId = run?.id ? String(run.id) : '';
          if (runId && currentRunId && String(currentRunId) === runId) {
            continue;
          }
          if (runMatchesPr(run, prNumber)) {
            if (runId) {
              matching.add(runId);
            } else {
              matching.add(JSON.stringify({ workflowId, status, head: run?.head_sha || '' }));
            }
          }
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Unable to query orchestrator runs (${workflowId}, ${status}): ${message}`);
      }
    }
  }

  return matching.size;
}

async function resolveGateRun({ github, owner, repo, headSha }) {
  if (!headSha) {
    return { concluded: false };
  }
  const normalizedHead = headSha.toLowerCase();
  for (const workflowId of GATE_WORKFLOWS) {
    try {
      const runs = await github.paginate(
        github.rest.actions.listWorkflowRuns,
        {
          owner,
          repo,
          workflow_id: workflowId,
          head_sha: headSha,
          per_page: 50,
        },
        (response) => {
          const data = response.data?.workflow_runs;
          if (Array.isArray(data)) {
            return data;
          }
          return [];
        }
      );
      if (!runs.length) {
        continue;
      }
      const match = runs.find((run) => normaliseString(run.head_sha).toLowerCase() === normalizedHead) || runs[0];
      if (!match) {
        continue;
      }
      const status = normaliseString(match.status).toLowerCase();
      return {
        concluded: status === 'completed',
        status,
        conclusion: normaliseString(match.conclusion).toLowerCase(),
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Unable to evaluate Gate workflow (${workflowId}): ${message}`);
    }
  }
  return { concluded: false };
}

function resolvePrNumberFromContext(context) {
  const payload = context?.payload || {};
  if (payload.issue?.number) {
    return payload.issue.number;
  }
  if (payload.pull_request?.number) {
    return payload.pull_request.number;
  }
  const workflowRun = payload.workflow_run;
  if (workflowRun && Array.isArray(workflowRun.pull_requests) && workflowRun.pull_requests.length) {
    const pr = workflowRun.pull_requests.find((entry) => Number.isFinite(entry?.number));
    if (pr && Number.isFinite(pr.number)) {
      return pr.number;
    }
  }
  if (payload.client_payload?.pr) {
    return payload.client_payload.pr;
  }
  if (payload.client_payload?.issue) {
    return payload.client_payload.issue;
  }
  return null;
}

async function evaluateKeepaliveGate({
  core,
  github,
  context,
  env = process.env,
  prNumber: explicitPrNumber,
  headSha: explicitHeadSha,
  prData,
  labels: explicitLabels,
  comments: explicitComments,
  requireHumanActivation = true,
  enforceHumanForSubsequentRounds = false,
  currentRunId = env.CURRENT_RUN_ID || env.GITHUB_RUN_ID || '',
} = {}) {
  const owner = context?.repo?.owner;
  const repo = context?.repo?.repo;
  if (!owner || !repo) {
    throw new Error('Repository context is required to evaluate keepalive gate.');
  }

  const resolvedPrNumber = explicitPrNumber ?? resolvePrNumberFromContext(context);
  const parsedPrNumber = Number.parseInt(resolvedPrNumber, 10);
  if (!Number.isFinite(parsedPrNumber) || parsedPrNumber <= 0) {
    return {
      ok: false,
      reason: 'missing-pr-number',
      hasKeepaliveLabel: false,
      hasHumanActivation: false,
      gateConcluded: false,
      underRunCap: false,
      runCap: DEFAULT_RUN_CAP,
      activeRunCount: 0,
      hasPriorKeepalive: false,
      agentAlias: '',
      agentAliases: [],
      prNumber: null,
    };
  }

  let pull = prData;
  if (!pull) {
    pull = await fetchPullRequest({ github, owner, repo, prNumber: parsedPrNumber });
  }

  const labels = explicitLabels || extractLabels(pull);
  const normalisedLabels = new Set(labels.map(normaliseLabelName));
  const hasKeepaliveLabel = normalisedLabels.has(KEEPALIVE_LABEL);

  const aliasEntries = labels
    .map(toAliasEntry)
    .filter(Boolean);

  const comments = explicitComments || (await fetchComments({ github, owner, repo, prNumber: parsedPrNumber }));
  const commentAnalysis = analyseComments(comments, aliasEntries);

  const runCap = parseRunCap(labels);

  const headSha = normaliseString(explicitHeadSha || pull?.head?.sha);
  const gateStatus = await resolveGateRun({ github, owner, repo, headSha });

  let activeRunCount = 0;
  try {
    activeRunCount = await countActiveRuns({
      github,
      owner,
      repo,
      prNumber: parsedPrNumber,
      currentRunId,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    core?.warning?.(`Unable to count orchestrator runs: ${message}`);
    activeRunCount = Number.POSITIVE_INFINITY;
  }

  const hasPriorKeepalive = commentAnalysis.hasPriorKeepalive;
  const humanRequired = requireHumanActivation && (enforceHumanForSubsequentRounds || !hasPriorKeepalive);
  const hasHumanActivation = commentAnalysis.hasHumanActivation;
  const gateConcluded = gateStatus.concluded === true;
  const underRunCap = Number.isFinite(activeRunCount) ? activeRunCount < runCap : false;

  let agentAlias = '';
  if (commentAnalysis.keepaliveAlias) {
    agentAlias = commentAnalysis.keepaliveAlias;
  } else if (commentAnalysis.humanAlias) {
    agentAlias = commentAnalysis.humanAlias;
  } else if (aliasEntries.length) {
    agentAlias = aliasEntries[0].alias;
  }
  agentAlias = normaliseString(agentAlias);

  let reason = '';
  if (!hasKeepaliveLabel) {
    reason = 'keepalive-label-missing';
  } else if (!aliasEntries.length) {
    reason = 'no-human-activation';
  } else if (humanRequired && !hasHumanActivation) {
    reason = 'no-human-activation';
  } else if (!gateConcluded) {
    reason = 'gate-not-concluded';
  } else if (!underRunCap) {
    reason = 'run-cap-reached';
  }

  const ok = !reason;

  return {
    ok,
    reason,
    hasKeepaliveLabel,
    hasHumanActivation,
    gateConcluded,
    underRunCap,
    runCap,
    activeRunCount: Number.isFinite(activeRunCount) ? activeRunCount : runCap,
    hasPriorKeepalive,
    agentAlias,
    agentAliases: aliasEntries.map((entry) => entry.alias),
    highestRecordedRound: commentAnalysis.highestRound,
    prNumber: parsedPrNumber,
  };
}

module.exports = {
  evaluateKeepaliveGate,
};
