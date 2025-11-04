'use strict';

const crypto = require('crypto');

function makeTrace() {
  const timestamp = Date.now().toString(36);
  const random = crypto.randomBytes(5).toString('base64').replace(/[^a-zA-Z0-9]/g, '').slice(0, 6);
  const suffix = random.toLowerCase().padEnd(6, '0');
  const trace = `${timestamp}${suffix}`.slice(0, 16);
  return trace.toLowerCase();
}

function ensureCodexPreface(body) {
  const trimmed = body.replace(/\r\n/g, '\n').trim();
  if (!trimmed) {
    throw new Error('Keepalive instruction body is required.');
  }
  if (/^@codex\b/i.test(trimmed)) {
    return trimmed;
  }
  return `@codex ${trimmed}`;
}

function renderInstruction({ round, trace, body }) {
  const parsedRound = Number.parseInt(round, 10);
  if (!Number.isFinite(parsedRound) || parsedRound <= 0) {
    throw new Error('Keepalive round must be a positive integer.');
  }
  const normalisedTrace = String(trace || '').trim();
  if (!normalisedTrace) {
    throw new Error('Keepalive trace token is required.');
  }
  const instructionBody = ensureCodexPreface(String(body ?? ''));
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
