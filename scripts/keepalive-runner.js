'use strict';

function parseJson(value, fallback) {
  try {
    return value ? JSON.parse(value) : fallback;
  } catch (error) {
    return fallback;
  }
}

function coerceBool(value, fallback) {
  if (value === null || value === undefined) {
    return fallback;
  }

  const normalised = String(value).trim().toLowerCase();
  if (!normalised) {
    return fallback;
  }

  const truthy = new Set(['true', '1', 'yes', 'on']);
  const falsy = new Set(['false', '0', 'no', 'off']);
  if (truthy.has(normalised)) {
    return true;
  }
  if (falsy.has(normalised)) {
    return false;
  }
  return fallback;
}

function coerceNumber(value, fallback, { min } = { min: 0 }) {
  if (value === null || value === undefined) {
    return fallback;
  }
  const num = Number(value);
  if (!Number.isFinite(num) || num <= (min ?? 0)) {
    return fallback;
  }
  return num;
}

function dedupe(values) {
  const seen = new Set();
  const unique = [];
  for (const value of values) {
    if (!seen.has(value)) {
      seen.add(value);
      unique.push(value);
    }
  }
  return unique;
}

function summariseList(items, limit = 20) {
  if (items.length <= limit) {
    return items;
  }
  const hidden = items.length - limit;
  return [
    ...items.slice(0, limit),
    `${hidden} more entries not shown to avoid excessive summary noise.`
  ];
}

async function runKeepalive({ core, github, context, env = process.env }) {
  const rawOptions = env.OPTIONS_JSON || '{}';
  const dryRun = (env.DRY_RUN || '').trim().toLowerCase() === 'true';
  const options = parseJson(rawOptions, {});
  const summary = core.summary;

  const addHeading = () => {
    summary.addHeading('Codex Keepalive');
    summary.addRaw(`Dry run: **${dryRun ? 'enabled' : 'disabled'}**`).addEOL();
  };

  const keepaliveEnabled = coerceBool(
    options.enable_keepalive ?? options.keepalive_enabled,
    true
  );
  if (!keepaliveEnabled) {
    core.info('Codex keepalive disabled via options_json.');
    addHeading();
    summary.addRaw('Skip requested via options_json.');
    await summary.write();
    return;
  }

  const idleMinutes = coerceNumber(options.keepalive_idle_minutes, 10, { min: 0 });
  const repeatMinutes = coerceNumber(options.keepalive_repeat_minutes, 30, { min: 0 });

  const labelSource = options.keepalive_labels ?? options.keepalive_label ?? 'agent:codex';
  let targetLabels = String(labelSource)
    .split(',')
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);
  if (!targetLabels.length) {
    targetLabels = ['agent:codex'];
  }
  targetLabels = dedupe(targetLabels);

  const commandRaw = options.keepalive_command ?? '@codex plan-and-execute';
  const command = String(commandRaw).trim() || '@codex plan-and-execute';
  const commandLower = command.toLowerCase();

  const markerRaw = options.keepalive_marker ?? '<!-- codex-keepalive -->';
  const marker = String(markerRaw);

  const agentSource = options.keepalive_agent_logins ?? 'chatgpt-codex-connector';
  let agentLogins = String(agentSource)
    .split(',')
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);
  if (!agentLogins.length) {
    agentLogins = ['chatgpt-codex-connector'];
  }
  agentLogins = dedupe(agentLogins);

  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const now = Date.now();
  const triggered = [];
  const previews = [];
  let scanned = 0;
  addHeading();
  summary
    .addRaw(`Target labels: ${targetLabels.map((label) => `**${label}**`).join(', ')}`)
    .addEOL();
  summary
    .addRaw(`Agent logins: ${agentLogins.map((login) => `**${login}**`).join(', ')}`)
    .addEOL();

  const paginatePulls = github.paginate.iterator(
    github.rest.pulls.list,
    { owner, repo, state: 'open', per_page: 50 }
  );

  for await (const page of paginatePulls) {
    for (const pr of page.data) {
      scanned += 1;
      const labelNames = (pr.labels || []).map((label) =>
        (typeof label === 'string' ? label : label?.name || '').toLowerCase()
      );
      const hasTargetLabel = targetLabels.some((label) => labelNames.includes(label));
      if (!hasTargetLabel) {
        core.info(`#${pr.number}: skipped – missing required label (${targetLabels.join(', ')}).`);
        continue;
      }

      const prNumber = pr.number;
      const { data: comments } = await github.rest.issues.listComments({
        owner,
        repo,
        issue_number: prNumber,
        per_page: 100,
      });
      if (!comments.length) {
        core.info(`#${prNumber}: skipped – no timeline comments.`);
        continue;
      }

      const commandComments = comments
        .filter((comment) => (comment.body || '').toLowerCase().includes(commandLower))
        .sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
      if (!commandComments.length) {
        core.info(`#${prNumber}: skipped – no ${command} command yet.`);
        continue;
      }

      const botComments = comments
        .filter((comment) => agentLogins.includes((comment.user?.login || '').toLowerCase()))
        .sort((a, b) => new Date(a.updated_at || a.created_at) - new Date(b.updated_at || b.created_at));
      if (!botComments.length) {
        core.info(`#${prNumber}: skipped – Codex has not commented yet.`);
        continue;
      }

      const lastAgentComment = botComments[botComments.length - 1];
      const lastAgentTs = new Date(lastAgentComment.updated_at || lastAgentComment.created_at).getTime();
      if (!Number.isFinite(lastAgentTs)) {
        core.info(`#${prNumber}: skipped – unable to parse Codex timestamp.`);
        continue;
      }

      const minutesSinceAgent = (now - lastAgentTs) / 60000;
      if (minutesSinceAgent < idleMinutes) {
        core.info(`#${prNumber}: skipped – last Codex activity ${minutesSinceAgent.toFixed(1)} minutes ago (< ${idleMinutes}).`);
        continue;
      }

      const latestCommandTs = new Date(commandComments[commandComments.length - 1].created_at).getTime();
      if (latestCommandTs > lastAgentTs) {
        core.info(`#${prNumber}: skipped – waiting for Codex response to the latest command.`);
        continue;
      }

      const checklistComments = botComments
        .map((comment) => {
          const body = comment.body || '';
          const unchecked = (body.match(/- \[ \]/g) || []).length;
          const checked = (body.match(/- \[x\]/gi) || []).length;
          const total = unchecked + checked;
          return { comment, unchecked, total };
        })
        .filter((entry) => entry.total > 0 && entry.unchecked > 0)
        .sort((a, b) => new Date(b.comment.updated_at || b.comment.created_at) - new Date(a.comment.updated_at || a.comment.created_at));

      if (!checklistComments.length) {
        core.info(`#${prNumber}: skipped – no Codex checklist with outstanding tasks.`);
        continue;
      }

      const keepaliveComments = comments
        .filter((comment) => (comment.body || '').includes(marker))
        .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      if (keepaliveComments.length) {
        const lastKeepaliveTs = new Date(keepaliveComments[0].created_at).getTime();
        const minutesSinceKeepalive = (now - lastKeepaliveTs) / 60000;
        if (minutesSinceKeepalive < repeatMinutes) {
          core.info(`#${prNumber}: skipped – keepalive sent ${minutesSinceKeepalive.toFixed(1)} minutes ago (< ${repeatMinutes}).`);
          continue;
        }
      }

      const bodyParts = [command];
      if (marker) {
        bodyParts.push('', marker);
      }
      const body = bodyParts.join('\n');
      const outstanding = checklistComments[0].unchecked;
      if (dryRun) {
        previews.push(`#${prNumber} – keepalive preview (remaining tasks: ${outstanding})`);
        core.info(`#${prNumber}: dry run – keepalive comment not posted (remaining tasks: ${outstanding}).`);
      } else {
        await github.rest.issues.createComment({ owner, repo, issue_number: prNumber, body });
        triggered.push(`#${prNumber} – keepalive posted (remaining tasks: ${outstanding})`);
        core.info(`#${prNumber}: keepalive posted (remaining tasks: ${outstanding}).`);
      }
    }
  }

  if (dryRun) {
    if (previews.length) {
      summary.addDetails('Previewed keepalive comments', summariseList(previews));
    } else {
      summary.addRaw('No unattended Codex tasks detected (dry run).');
    }
    summary.addRaw(`Previewed keepalive count: ${previews.length}`).addEOL();
  } else {
    if (triggered.length) {
      summary.addDetails('Triggered keepalive comments', summariseList(triggered));
    } else {
      summary.addRaw('No unattended Codex tasks detected.');
    }
    summary.addRaw(`Triggered keepalive count: ${triggered.length}`).addEOL();
  }
  summary.addRaw(`Evaluated pull requests: ${scanned}`).addEOL();
  await summary.write();
}

module.exports = { runKeepalive };
