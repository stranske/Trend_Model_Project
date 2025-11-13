'use strict';

const DEFAULT_INSTRUCTION_SIGNATURE =
  'keepalive workflow continues nudging until everything is complete';
const {
  extractInstructionSegment,
  computeInstructionByteLength,
} = require('../../scripts/keepalive_instruction_segment.js');

const AUTOMATION_LOGINS = new Set(['chatgpt-codex-connector', 'stranske-automation-bot']);
const INSTRUCTION_REACTION = 'hooray';
const LOCK_REACTION = 'dart';

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
  const toBool = (value) => String(value || '').trim().toLowerCase() === 'true';
  const allowReplay = toBool(env.ALLOW_REPLAY);
  const hasValue = (value) => typeof value === 'string' && value.trim() !== '';
  const gateOk = hasValue(env.GATE_OK) ? toBool(env.GATE_OK) : true;
  const gateReasonRaw = String(env.GATE_REASON || '').trim();
  const gatePending = hasValue(env.GATE_PENDING) ? toBool(env.GATE_PENDING) : false;

  let eventName = String(context?.eventName || context?.event_name || '').toLowerCase();
  if (!eventName && context?.payload?.comment) {
    eventName = 'issue_comment';
  }
  let actionName = String(context?.payload?.action || '').toLowerCase();
  if (!actionName && eventName === 'issue_comment') {
    actionName = 'created';
  }

  const escapeRegExp = (value) => String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

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

  const canonicalRoundPatterns = [/<!--\s*keepalive-round\s*:?#?\s*(\d+)\s*-->/i];

  const canonicalTracePatterns = [/<!--\s*keepalive-trace\s*:?#?\s*([^>]+?)\s*-->/i];

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
    processed_reaction: 'false',
    deduped: 'false',
    instruction_body: '',
    instruction_bytes: '0',
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
    core.setOutput('processed_reaction', outputs.processed_reaction);
    core.setOutput('deduped', outputs.deduped);
    core.setOutput('instruction_body', outputs.instruction_body || '');
    core.setOutput('instruction_bytes', outputs.instruction_bytes || '0');
  };

  const { comment, issue } = context.payload || {};
  const { owner, repo } = context.repo || {};
  const body = comment?.body || '';
  const authorRaw = comment?.user?.login || '';
  const author = normaliseLogin(authorRaw);
  const contextIssueNumber = issue?.number ? Number.parseInt(issue.number, 10) : NaN;
  const isAuthorAllowed = allowedLogins.has(author);

  let instructionSeen = false;
  let traceMatch;

  const normaliseBody = (value) => String(value || '').replace(/\r\n/g, '\n').trim();

  const isLikelyInstruction = (value) => {
    if (!value) {
      return false;
    }
    const normalised = normaliseBody(value);
    if (!normalised || !normalised.toLowerCase().startsWith('@codex')) {
      return false;
    }
    return normalised.toLowerCase().includes(DEFAULT_INSTRUCTION_SIGNATURE);
  };

  const resolveSummarySource = () => {
    if (author === 'stranske') {
      return 'stranske';
    }
    if (AUTOMATION_LOGINS.has(author)) {
      return 'bot';
    }
    if (author) {
      return author;
    }
    if (authorRaw) {
      return String(authorRaw);
    }
    return 'unknown';
  };

  const finalise = (seenOverride) => {
    const seenFlag = (typeof seenOverride === 'boolean' ? seenOverride : instructionSeen) ? 'true' : 'false';
    setAllOutputs();
    const commentId = outputs.comment_id || (comment?.id ? String(comment.id) : '') || 'unknown';
    const traceValueRaw = outputs.trace || (traceMatch && traceMatch[1] ? traceMatch[1].replace(/--+$/u, '').trim() : '');
    const traceValue = traceValueRaw || 'n/a';
    const dedupedFlag = outputs.deduped === 'true' ? 'true' : 'false';
    core.info(
      `INSTRUCTION: comment_id=${commentId} trace=${traceValue} source=${resolveSummarySource()} seen=${seenFlag} deduped=${dedupedFlag}`
    );
    return outputs;
  };

  outputs.author = authorRaw;
  outputs.comment_id = comment?.id ? String(comment.id) : '';
  outputs.comment_url = comment?.html_url || '';

  if (eventName === 'issue_comment' && actionName !== 'created') {
    outputs.reason = 'ignored-comment-action';
    core.info(`Keepalive dispatch skipped: unsupported issue_comment action "${actionName || 'unknown'}".`);
    return finalise(false);
  }

  if (eventName !== 'issue_comment' && !allowReplay) {
    outputs.reason = 'unsupported-event';
    core.info(`Keepalive dispatch skipped: event ${eventName || 'unknown'} not eligible for keepalive detection.`);
    return finalise(false);
  }

  const roundMatch = findFirstMatch(body, canonicalRoundPatterns);

  traceMatch = findFirstMatch(body, canonicalTracePatterns);

  const hasKeepaliveMarker = Boolean(findFirstMatch(body, canonicalMarkerPatterns));

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
    if (AUTOMATION_LOGINS.has(author) && !isLikelyInstruction(trimmedBody)) {
      return true;
    }
    return false;
  };

  if (!roundMatch && !hasKeepaliveMarker && isAutomationStatusComment()) {
    outputs.reason = 'automation-comment';
    core.info('Keepalive dispatch skipped: automation status comment without keepalive markers.');
    return finalise(false);
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
    return finalise(false);
  }

  if (!roundMatch) {
    outputs.reason = 'missing-round';
    core.info('Keepalive dispatch skipped: comment missing keepalive round marker.');
    return finalise(false);
  }

  if (!isAuthorAllowed) {
    outputs.reason = 'unauthorised-author';
    core.info(`Keepalive dispatch skipped: author ${author || '(unknown)'} not in allow list.`);
    return finalise(false);
  }

  if (!hasKeepaliveMarker) {
    outputs.reason = 'missing-sentinel';
    core.info('Keepalive dispatch skipped: comment missing codex keepalive marker.');
    return finalise(false);
  }

  const round = Number.parseInt(roundMatch[1], 10);
  if (!Number.isFinite(round) || round <= 0) {
    outputs.reason = 'invalid-round';
    core.info('Keepalive dispatch skipped: invalid round marker.');
    return finalise(false);
  }

  const commentId = comment?.id;
  if (!commentId) {
    outputs.reason = 'missing-comment-id';
    core.warning('Keepalive dispatch skipped: unable to determine comment id for dedupe.');
    return finalise(false);
  }

  const prNumber = issue?.number;

  outputs.pr = prNumber ? String(prNumber) : '';
  outputs.round = String(round);
  const trace = traceMatch ? traceMatch[1].replace(/--+$/u, '').trim() : '';
  if (!trace) {
    outputs.reason = 'missing-trace';
    core.info('Keepalive dispatch skipped: comment missing keepalive trace marker.');
    return finalise(false);
  }
  outputs.trace = trace;

  instructionSeen = true;

  let pull;
  try {
    const response = await github.rest.pulls.get({ owner, repo, pull_number: prNumber });
    pull = response.data;
  } catch (error) {
    outputs.reason = 'pull-fetch-failed';
    const message = error instanceof Error ? error.message : String(error);
    core.warning(`Keepalive dispatch skipped: unable to load PR #${prNumber} (${message}).`);
    return finalise();
  }

  outputs.branch = pull?.head?.ref || '';
  outputs.base = pull?.base?.ref || '';

  const instructionBody = extractInstructionSegment(body);
  if (!instructionBody) {
    outputs.reason = 'instruction-empty';
    outputs.dispatch = 'false';
    core.setFailed('instruction-empty');
    core.info('Keepalive dispatch blocked: instruction segment missing or empty.');
    return finalise(true);
  }
  outputs.instruction_body = instructionBody;
  outputs.instruction_bytes = String(computeInstructionByteLength(instructionBody));

  const headRepo = pull?.head?.repo;
  const baseRepo = pull?.base?.repo;
  if (
    headRepo &&
    baseRepo &&
    (headRepo.fork || (headRepo.owner?.login && baseRepo.owner?.login && headRepo.owner.login !== baseRepo.owner.login))
  ) {
    outputs.reason = 'fork-pr';
    core.info('Keepalive dispatch skipped: pull request originates from a fork.');
    return finalise();
  }

  const issueNumber = extractIssueNumberFromPull(pull);
  if (issueNumber) {
    outputs.issue = String(issueNumber);
  }

  let reactions = [];
  try {
    reactions = await github.paginate(github.rest.reactions.listForIssueComment, {
      owner,
      repo,
      comment_id: commentId,
      per_page: 100,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    core.warning(`Failed to read keepalive reactions for comment ${commentId}: ${message}`);
    reactions = [];
  }

  const hasInstructionReaction = reactions.some(
    (reaction) => (reaction?.content || '').toLowerCase() === INSTRUCTION_REACTION
  );

  let processedReaction = hasInstructionReaction;
  if (!processedReaction) {
    try {
      const response = await github.rest.reactions.createForIssueComment({
        owner,
        repo,
        comment_id: commentId,
        content: INSTRUCTION_REACTION,
      });
      const status = Number(response?.status || 0);
      const content = String(response?.data?.content || '').toLowerCase();
      if (status === 200 || status === 201 || content === INSTRUCTION_REACTION) {
        processedReaction = true;
      }
    } catch (error) {
      if (error && error.status === 409) {
        processedReaction = true;
      } else {
        const message = error instanceof Error ? error.message : String(error);
        outputs.reason = 'instruction-reaction-failed';
        core.warning(`Failed to add ${INSTRUCTION_REACTION} reaction for keepalive comment ${commentId}: ${message}`);
        return finalise();
      }
    }
  }

  if (!processedReaction) {
    outputs.reason = 'missing-instruction-reaction';
    core.info('Keepalive dispatch skipped: unable to confirm instruction reaction.');
    return finalise();
  }

  outputs.processed_reaction = 'true';

  const hasLockReaction = reactions.some(
    (reaction) => (reaction?.content || '').toLowerCase() === LOCK_REACTION
  );

  if (hasLockReaction) {
    outputs.reason = 'lock-held';
    outputs.dispatch = 'false';
    outputs.deduped = 'true';
    core.info(`Keepalive dispatch skipped: ${LOCK_REACTION} reaction already present on comment ${commentId}.`);
    return finalise(true);
  }

  try {
    const response = await github.rest.reactions.createForIssueComment({
      owner,
      repo,
      comment_id: commentId,
      content: LOCK_REACTION,
    });
    const status = Number(response?.status || 0);
    const content = String(response?.data?.content || '').toLowerCase();
    if (status === 200 || status === 201 || content === LOCK_REACTION) {
      outputs.processed_reaction = 'true';
    }
  } catch (error) {
    if (error && error.status === 409) {
      outputs.reason = 'lock-held';
      outputs.dispatch = 'false';
      outputs.deduped = 'true';
      core.info(
        `Keepalive dispatch skipped: ${LOCK_REACTION} reaction already present on comment ${commentId} (detected via conflict).`
      );
      return finalise(true);
    }

    const message = error instanceof Error ? error.message : String(error);
    outputs.reason = 'lock-held';
    outputs.dispatch = 'false';
    core.warning(`Failed to add ${LOCK_REACTION} reaction for keepalive comment ${commentId}: ${message}`);
    return finalise();
  }

  if (!issueNumber) {
    outputs.reason = 'missing-issue-reference';
    core.info('Keepalive dispatch skipped: unable to determine linked issue number.');
    return finalise();
  }

  outputs.dispatch = 'true';
  outputs.reason = 'keepalive-detected';

  return finalise(true);
}

module.exports = {
  detectKeepalive,
  normaliseLogin,
  parseAllowedLogins,
  extractIssueNumberFromPull,
};
