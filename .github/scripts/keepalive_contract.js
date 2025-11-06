'use strict';

const crypto = require('crypto');

function makeTrace() {
  const timestamp = Date.now().toString(36);
  const random = crypto.randomBytes(5).toString('base64').replace(/[^a-zA-Z0-9]/g, '').slice(0, 6);
  const suffix = random.toLowerCase().padEnd(6, '0');
  const trace = `${timestamp}${suffix}`.slice(0, 16);
  return trace.toLowerCase();
}

function ensureAgentPreface(body, agentAlias) {
  const trimmed = String(body ?? '').replace(/\r\n/g, '\n').trim();
  if (!trimmed) {
    throw new Error('Keepalive instruction body is required.');
  }
  const alias = String(agentAlias || '').trim();
  if (!alias) {
    throw new Error('Keepalive agent alias is required.');
  }
  const aliasPattern = new RegExp(`^@${alias.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\b`, 'i');
  if (aliasPattern.test(trimmed)) {
    const mention = trimmed.match(/^@[^\s]+/);
    if (mention && mention[0] !== `@${alias}`) {
      return `@${alias}${trimmed.slice(mention[0].length)}`;
    }
    return trimmed;
  }
  const withoutLeadingMention = trimmed.replace(/^@[^\s]+\s*/, '').trim();
  if (!withoutLeadingMention) {
    return `@${alias}`;
  }
  return `@${alias} ${withoutLeadingMention}`;
}

function renderInstruction({ round, trace, body, agentAlias }) {
  const parsedRound = Number.parseInt(round, 10);
  if (!Number.isFinite(parsedRound) || parsedRound <= 0) {
    throw new Error('Keepalive round must be a positive integer.');
  }
  const normalisedTrace = String(trace || '').trim();
  if (!normalisedTrace) {
    throw new Error('Keepalive trace token is required.');
  }
  const instructionBody = ensureAgentPreface(body, agentAlias);
  const lines = [
    `<!-- keepalive-round: ${parsedRound} -->`,
    '<!-- codex-keepalive-marker -->',
    `<!-- keepalive-trace: ${normalisedTrace} -->`,
    instructionBody,
  ];
  return `${lines.join('\n')}\n`;
}

module.exports = {
  makeTrace,
  renderInstruction,
};
