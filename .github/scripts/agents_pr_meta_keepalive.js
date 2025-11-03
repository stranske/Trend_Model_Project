'use strict';

function normaliseLogin(login) {
  return String(login || '')
    .trim()
    .toLowerCase()
    .replace(/\[bot\]$/i, '');
}

function parseAllowedLogins(env) {
  const raw = String(env.ALLOWED_LOGINS || '')
    .split(',')
    .map((value) => normaliseLogin(value))
    .filter(Boolean);
  return new Set(raw);
}

function extractIssueNumberFromPull(pull) {
  if (!pull) {
    return null;
  }

  const candidates = [];

  const branch = pull?.head?.ref || '';
  const branchMatch = branch.match(/issue-+([0-9]+)/i);
  if (branchMatch) {
    candidates.push(branchMatch[1]);
  }

  const title = pull?.title || '';
  const titleMatch = title.match(/#([0-9]+)/);
  if (titleMatch) {
    candidates.push(titleMatch[1]);
  }

  const bodyText = pull?.body || '';
  for (const match of bodyText.matchAll(/#([0-9]+)/g)) {
    if (match[1]) {
      candidates.push(match[1]);
    }
  }

  for (const value of candidates) {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }

  return null;
}

async function detectKeepalive({ core, github, context, env = process.env }) {
  const allowedLogins = parseAllowedLogins(env);
  const keepaliveMarker = env.KEEPALIVE_MARKER || '';

  const outputs = {
    dispatch: 'false',
    reason: 'not-keepalive',
    issue: '',
    round: '',
    branch: '',
    base: '',
  };

  const setBasicOutputs = () => {
    core.setOutput('dispatch', outputs.dispatch);
    core.setOutput('reason', outputs.reason);
  };

  const setAllOutputs = () => {
    setBasicOutputs();
    core.setOutput('issue', outputs.issue);
    core.setOutput('round', outputs.round);
    core.setOutput('branch', outputs.branch);
    core.setOutput('base', outputs.base);
  };

  const { comment, issue } = context.payload || {};
  const { owner, repo } = context.repo || {};
  const body = comment?.body || '';
  const author = normaliseLogin(comment?.user?.login);

  const roundMatch = body.match(/<!--\s*keepalive-round:(\d+)\s*-->/i);
  const hasKeepaliveMarker = body.includes(keepaliveMarker);

  if (!roundMatch) {
    core.info('Comment does not contain keepalive round marker; skipping.');
    setBasicOutputs();
    return outputs;
  }

  if (!allowedLogins.has(author)) {
    outputs.reason = 'unauthorised-author';
    core.info(`Keepalive dispatch skipped: author ${author || '(unknown)'} not in allow list.`);
    setBasicOutputs();
    return outputs;
  }

  if (!hasKeepaliveMarker) {
    outputs.reason = 'missing-sentinel';
    core.info('Keepalive dispatch skipped: comment missing codex keepalive marker.');
    setBasicOutputs();
    return outputs;
  }

  const round = Number.parseInt(roundMatch[1], 10);
  if (!Number.isFinite(round) || round <= 0) {
    outputs.reason = 'invalid-round';
    core.info('Keepalive dispatch skipped: invalid round marker.');
    setBasicOutputs();
    return outputs;
  }

  const prNumber = issue?.number;
  let pull;
  try {
    const response = await github.rest.pulls.get({ owner, repo, pull_number: prNumber });
    pull = response.data;
  } catch (error) {
    outputs.reason = 'pull-fetch-failed';
    const message = error instanceof Error ? error.message : String(error);
    core.warning(`Keepalive dispatch skipped: unable to load PR #${prNumber} (${message}).`);
    setBasicOutputs();
    return outputs;
  }

  const issueNumber = extractIssueNumberFromPull(pull);
  if (!issueNumber) {
    outputs.reason = 'missing-issue-reference';
    core.info('Keepalive dispatch skipped: unable to determine linked issue number.');
    setBasicOutputs();
    return outputs;
  }

  outputs.dispatch = 'true';
  outputs.reason = 'keepalive-detected';
  outputs.issue = String(issueNumber);
  outputs.round = String(round);
  outputs.branch = pull?.head?.ref || '';
  outputs.base = pull?.base?.ref || '';

  setAllOutputs();
  return outputs;
}

module.exports = {
  detectKeepalive,
  normaliseLogin,
  parseAllowedLogins,
  extractIssueNumberFromPull,
};
