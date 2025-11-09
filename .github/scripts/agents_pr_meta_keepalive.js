'use strict';

const { renderInstruction, makeTrace } = require('./keepalive_contract.js');

const DEFAULT_INSTRUCTION_SIGNATURE =
  'keepalive workflow continues nudging until everything is complete';

function normaliseBody(value) {
  return String(value || '').replace(/\r\n/g, '\n').trim();
}

function isLikelyInstruction(body) {
  if (!body) {
    return false;
  }
  const normalised = normaliseBody(body);
  if (!normalised || !normalised.toLowerCase().startsWith('@codex')) {
    return false;
  }
  return normalised.toLowerCase().includes(DEFAULT_INSTRUCTION_SIGNATURE);
}

async function computeHighestRound({ github, owner, repo, prNumber }) {
  let highestRound = 0;
  try {
    const comments = await github.paginate(github.rest.issues.listComments, {
      owner,
      repo,
      issue_number: prNumber,
      per_page: 100,
    });
    for (const entry of comments) {
      const marker = String(entry?.body || '').match(/<!--\s*keepalive-round:\s*(\d+)\s*-->/i);
      if (!marker) {
        continue;
      }
      const parsed = Number.parseInt(marker[1], 10);
      if (Number.isFinite(parsed) && parsed > highestRound) {
        highestRound = parsed;
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Unable to scan keepalive history: ${message}`);
  }
  return highestRound;
}

function resolveAgentAlias(env) {
  const value = String(env.KEEPALIVE_AGENT_ALIAS || '').trim();
  if (value) {
    return value;
  }
  return 'codex';
}

async function autopatchKeepaliveComment({
  core,
  github,
  owner,
  repo,
  prNumber,
  comment,
  currentBody,
  highestRound,
  agent,
}) {
  const trimmed = normaliseBody(currentBody);
  if (!isLikelyInstruction(trimmed)) {
    return null;
  }

  if (!comment?.id) {
    return null;
  }

  let effectiveHighestRound = highestRound;
  if (!Number.isFinite(effectiveHighestRound)) {
    effectiveHighestRound = await computeHighestRound({ github, owner, repo, prNumber });
  }

  if (effectiveHighestRound >= 1) {
    core.info(
      `Skipping keepalive autopatch for comment ${comment.id}: next round (${effectiveHighestRound + 1}) must originate from automation.`
    );
    return {
      blocked: true,
      highestRound: effectiveHighestRound,
    };
  }

  const nextRound = effectiveHighestRound + 1;
  const trace = makeTrace();
  const patchedBody = renderInstruction({ round: nextRound, trace, body: trimmed, agent });

  await github.rest.issues.updateComment({
    owner,
    repo,
    comment_id: comment.id,
    body: patchedBody,
  });

  core.info(
    `Inserted keepalive markers for comment ${comment.id} (round ${nextRound}, trace ${trace}).`
  );

  return {
    body: patchedBody,
    round: nextRound,
    trace,
    blocked: false,
  };
}

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
  const agentAlias = resolveAgentAlias(env);
  const toBool = (value) => String(value || '').trim().toLowerCase() === 'true';
  const gateOk = toBool(env.GATE_OK);
  const gateReasonRaw = String(env.GATE_REASON || '').trim();
  const gatePending = toBool(env.GATE_PENDING);

  const escapeRegExp = (value) => String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  const decodeHtmlEntities = (value) => {
    let previous;
    let current = String(value);
    do {
      previous = current;
      current = current
        .replace(/&lt;/gi, '<')
        .replace(/&gt;/gi, '>')
        .replace(/&amp;/gi, '&');
    } while (current !== previous);
    return current;
  };

  const findFirstMatch = (source, patterns) => {
    for (const pattern of patterns) {
      const match = source.match(pattern);
      if (match) {
        return match;
      }
    }
    return null;
  };

  const canonicalMarkerPatterns = [];
  if (keepaliveMarker) {
    canonicalMarkerPatterns.push(new RegExp(escapeRegExp(keepaliveMarker), 'i'));
  }
  canonicalMarkerPatterns.push(/<!--\s*codex-keepalive-marker\s*-->/i);

  const markerPatterns = canonicalMarkerPatterns.concat([
    /<-\s*After:\s*codex-keepalive-marker\s*-->/i,
    /&lt;-\s*After:\s*codex-keepalive-marker\s*--&gt;/i,
    /&lt;!--\s*codex-keepalive-marker\s*--&gt;/i,
  ]);

  const canonicalRoundPatterns = [/<!--\s*keepalive-round\s*:?#?\s*(\d+)\s*-->/i];
  const roundPatterns = canonicalRoundPatterns.concat([
    /<-\s*After:\s*keepalive-round\s*:?#?\s*(\d+)\s*-->/i,
    /&lt;!--\s*keepalive-round\s*:?#?\s*(\d+)\s*--&gt;/i,
    /&lt;-\s*After:\s*keepalive-round\s*:?#?\s*(\d+)\s*--&gt;/i,
  ]);

  const canonicalTracePatterns = [/<!--\s*keepalive-trace\s*:?#?\s*([^>]+?)\s*-->/i];
  const tracePatterns = canonicalTracePatterns.concat([
    /<-\s*After:\s*keepalive-trace\s*:?#?\s*([^>]+?)\s*-->/i,
    /&lt;!--\s*keepalive-trace\s*:?#?\s*([^>]+?)\s*--&gt;/i,
    /&lt;-\s*After:\s*keepalive-trace\s*:?#?\s*([^>]+?)\s*--&gt;/i,
  ]);


  const outputs = {
    dispatch: 'false',
    reason: 'not-keepalive',
    issue: '',
    round: '',
    branch: '',
    base: '',
    trace: '',
    pr: '',
    author: '',
    comment_id: '',
    comment_url: '',
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
    core.setOutput('trace', outputs.trace);
    core.setOutput('pr', outputs.pr);
    core.setOutput('author', outputs.author);
    core.setOutput('comment_id', outputs.comment_id);
    core.setOutput('comment_url', outputs.comment_url);
  };

  const { comment, issue } = context.payload || {};
  const { owner, repo } = context.repo || {};
  const body = comment?.body || '';
  const authorRaw = comment?.user?.login || '';
  const author = normaliseLogin(authorRaw);
  const contextIssueNumber = issue?.number ? Number.parseInt(issue.number, 10) : NaN;
  const isAuthorAllowed = allowedLogins.has(author);

  outputs.author = authorRaw;
  outputs.comment_id = comment?.id ? String(comment.id) : '';
  outputs.comment_url = comment?.html_url || '';

  let workingBody = body;
  const canonicalRoundMatch = findFirstMatch(body, canonicalRoundPatterns);
  let roundMatch = canonicalRoundMatch;
  if (!roundMatch) {
    const decodedBody = decodeHtmlEntities(body);
    if (decodedBody !== body) {
      workingBody = decodedBody;
    }
    roundMatch = findFirstMatch(workingBody, roundPatterns);
  }

  const usedFallbackRound = !canonicalRoundMatch && Boolean(roundMatch);
  if (usedFallbackRound) {
    const preview = JSON.stringify(body.slice(0, 160));
    core.info(`Keepalive canonical round marker not found; matched fallback against payload prefix ${preview}.`);
  }

  let traceMatch = findFirstMatch(body, canonicalTracePatterns);
  if (!traceMatch) {
    traceMatch = findFirstMatch(workingBody, tracePatterns);
  }

  let hasKeepaliveMarker = Boolean(findFirstMatch(body, canonicalMarkerPatterns));
  if (!hasKeepaliveMarker) {
    hasKeepaliveMarker = Boolean(findFirstMatch(workingBody, markerPatterns));
  }

  const isAutomationStatusComment = () => {
    const trimmedBody = normaliseBody(body);
    if (!trimmedBody) {
      return false;
    }
    if (trimmedBody.includes('<!-- autofix-loop:')) {
      return true;
    }
    if (trimmedBody.toLowerCase().startsWith('autofix attempt')) {
      return true;
    }
    const automationAuthors = new Set(['chatgpt-codex-connector']);
    if (automationAuthors.has(author) && !isLikelyInstruction(trimmedBody)) {
      return true;
    }
    return false;
  };

  if (!roundMatch && !hasKeepaliveMarker && isAutomationStatusComment()) {
    outputs.reason = 'automation-comment';
    core.info('Keepalive dispatch skipped: automation status comment without keepalive markers.');
    setAllOutputs();
    return outputs;
  }

  let blockedByManualRound = false;
  let highestRoundCache = null;
  const ensureHighestRound = async () => {
    if (highestRoundCache === null) {
      highestRoundCache = await computeHighestRound({
        github,
        owner,
        repo,
        prNumber: contextIssueNumber,
      });
    }
    return highestRoundCache;
  };

  const shouldAttemptAutopatch =
    (!roundMatch || !hasKeepaliveMarker) &&
    isAuthorAllowed &&
    Number.isFinite(contextIssueNumber) &&
    contextIssueNumber > 0 &&
    owner &&
    repo;

  if (!shouldAttemptAutopatch && !roundMatch && isAuthorAllowed) {
    core.info(
      `Keepalive autopatch skipped: allowed=${isAuthorAllowed} issue=${Number.isFinite(contextIssueNumber) ? contextIssueNumber : 'n/a'} owner=${Boolean(owner)} repo=${Boolean(repo)} marker=${hasKeepaliveMarker}`
    );
  }

  if (shouldAttemptAutopatch) {
    core.info(
      `Keepalive autopatch attempt: issue=${contextIssueNumber} comment=${comment?.id || outputs.comment_id || 'n/a'} roundMatch=${Boolean(roundMatch)} marker=${hasKeepaliveMarker}`
    );
    const highestRound = await ensureHighestRound();
    if (highestRound >= 1) {
      blockedByManualRound = true;
      core.info(`Keepalive autopatch blocked: highestRound=${highestRound} (manual round required).`);
    } else {
      try {
        const patched = await autopatchKeepaliveComment({
          core,
          github,
          owner,
          repo,
          prNumber: contextIssueNumber,
          comment,
          currentBody: body,
          highestRound,
          agent: agentAlias,
        });
        if (patched) {
          if (patched.blocked) {
            blockedByManualRound = true;
            core.info(`Keepalive autopatch declined by guard: highestRound=${patched.highestRound ?? highestRound}.`);
          } else {
            roundMatch = [null, String(patched.round)];
            traceMatch = [null, patched.trace];
            hasKeepaliveMarker = true;
            core.info(`Keepalive autopatch inserted markers: round=${patched.round} trace=${patched.trace}`);
          }
        } else {
          core.info('Keepalive autopatch returned no changes (likely non-instruction comment).');
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        core.warning(`Auto-inserting keepalive markers failed: ${message}`);
      }
    }
  }

  if (!gateOk && isAuthorAllowed) {
    const gateDetail = gateReasonRaw || (gatePending ? 'gate-pending' : '');
    const reason = gateDetail ? `gate-blocked:${gateDetail}` : 'gate-blocked';
    outputs.reason = reason;
    outputs.dispatch = 'false';
    if (Number.isFinite(contextIssueNumber) && contextIssueNumber > 0) {
      outputs.pr = String(contextIssueNumber);
    }
    if (roundMatch) {
      const gateRound = Number.parseInt(roundMatch[1], 10);
      if (Number.isFinite(gateRound) && gateRound > 0) {
        outputs.round = String(gateRound);
      }
    }
    core.info(`Keepalive dispatch deferred: gate reported ${gateDetail || 'a blocking condition'}.`);
    setAllOutputs();
    return outputs;
  }

  if (blockedByManualRound) {
    outputs.reason = 'manual-round';
    core.info('Keepalive dispatch skipped: subsequent keepalive rounds must be initiated by automation.');
    setAllOutputs();
    return outputs;
  }

  if (!roundMatch) {
    outputs.reason = 'missing-round';
    core.info('Keepalive dispatch skipped: comment missing keepalive round marker.');
    setAllOutputs();
    return outputs;
  }

  if (!isAuthorAllowed) {
    outputs.reason = 'unauthorised-author';
    core.info(`Keepalive dispatch skipped: author ${author || '(unknown)'} not in allow list.`);
    setAllOutputs();
    return outputs;
  }

  if (!hasKeepaliveMarker) {
    outputs.reason = 'missing-sentinel';
    core.info('Keepalive dispatch skipped: comment missing codex keepalive marker.');
    setAllOutputs();
    return outputs;
  }

  const round = Number.parseInt(roundMatch[1], 10);
  if (!Number.isFinite(round) || round <= 0) {
    outputs.reason = 'invalid-round';
    core.info('Keepalive dispatch skipped: invalid round marker.');
    setAllOutputs();
    return outputs;
  }

  const commentId = comment?.id;
  if (!commentId) {
    outputs.reason = 'missing-comment-id';
    core.warning('Keepalive dispatch skipped: unable to determine comment id for dedupe.');
    setAllOutputs();
    return outputs;
  }

  const prNumber = issue?.number;

  outputs.pr = prNumber ? String(prNumber) : '';
  outputs.round = String(round);
  const trace = traceMatch ? traceMatch[1].replace(/--+$/u, '').trim() : '';
  outputs.trace = trace;

  let pull;
  try {
    const response = await github.rest.pulls.get({ owner, repo, pull_number: prNumber });
    pull = response.data;
  } catch (error) {
    outputs.reason = 'pull-fetch-failed';
    const message = error instanceof Error ? error.message : String(error);
    core.warning(`Keepalive dispatch skipped: unable to load PR #${prNumber} (${message}).`);
    setAllOutputs();
    return outputs;
  }

  outputs.branch = pull?.head?.ref || '';
  outputs.base = pull?.base?.ref || '';

  const headRepo = pull?.head?.repo;
  const baseRepo = pull?.base?.repo;
  if (
    headRepo &&
    baseRepo &&
    (headRepo.fork || (headRepo.owner?.login && baseRepo.owner?.login && headRepo.owner.login !== baseRepo.owner.login))
  ) {
    outputs.reason = 'fork-pr';
    core.info('Keepalive dispatch skipped: pull request originates from a fork.');
    setAllOutputs();
    return outputs;
  }

  const issueNumber = extractIssueNumberFromPull(pull);
  if (issueNumber) {
    outputs.issue = String(issueNumber);
  }

  let hasRocket = false;
  try {
    const reactions = await github.paginate(github.rest.reactions.listForIssueComment, {
      owner,
      repo,
      comment_id: commentId,
      per_page: 100,
    });
    hasRocket = reactions.some((reaction) => (reaction?.content || '').toLowerCase() === 'rocket');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    core.warning(`Failed to read keepalive reactions for comment ${commentId}: ${message}`);
  }

  if (hasRocket) {
    outputs.reason = 'duplicate-keepalive';
    core.info(`Keepalive dispatch skipped: rocket reaction already present on comment ${commentId}.`);
    setAllOutputs();
    return outputs;
  }

  if (!issueNumber) {
    outputs.reason = 'missing-issue-reference';
    core.info('Keepalive dispatch skipped: unable to determine linked issue number.');
    setAllOutputs();
    return outputs;
  }

  let reactionStatus = 0;
  let reactionContent = '';
  try {
    const response = await github.rest.reactions.createForIssueComment({
      owner,
      repo,
      comment_id: commentId,
      content: 'rocket',
    });
    reactionStatus = Number(response?.status || 0);
    reactionContent = String(response?.data?.content || '').toLowerCase();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (error && error.status === 409) {
      outputs.dispatch = 'false';
      outputs.reason = 'duplicate-keepalive';
      core.info(`Keepalive dispatch skipped: rocket reaction already present on comment ${commentId} (detected via conflict).`);
      setAllOutputs();
      return outputs;
    }

    let hasRocket = false;
    try {
      const reactions = await github.paginate(github.rest.reactions.listForIssueComment, {
        owner,
        repo,
        comment_id: commentId,
        per_page: 100,
      });
      hasRocket = reactions.some((reaction) => (reaction?.content || '').toLowerCase() === 'rocket');
    } catch (readError) {
      const readMessage = readError instanceof Error ? readError.message : String(readError);
      core.warning(`Failed to read keepalive reactions for comment ${commentId}: ${readMessage}`);
    }

    if (hasRocket) {
      outputs.dispatch = 'false';
      outputs.reason = 'duplicate-keepalive';
      core.info(`Keepalive dispatch skipped: rocket reaction already present on comment ${commentId}.`);
      setAllOutputs();
      return outputs;
    }

    core.warning(`Failed to add rocket reaction for dedupe on comment ${commentId}: ${message}`);
  }

  if (reactionStatus === 200 && reactionContent === 'rocket') {
    outputs.dispatch = 'false';
    outputs.reason = 'duplicate-keepalive';
    core.info(`Keepalive dispatch skipped: rocket reaction already present on comment ${commentId} (detected via status ${reactionStatus}).`);
    setAllOutputs();
    return outputs;
  }

  outputs.dispatch = 'true';
  outputs.reason = 'keepalive-detected';

  setAllOutputs();
  return outputs;
}

module.exports = {
  detectKeepalive,
  normaliseLogin,
  parseAllowedLogins,
  extractIssueNumberFromPull,
};
