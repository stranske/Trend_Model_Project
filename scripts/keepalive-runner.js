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

function normaliseLogin(login) {
  const base = String(login ?? '').trim().toLowerCase();
  if (!base) {
    return '';
  }
  return base.replace(/\[bot\]$/i, '');
}

function parseAgentLoginEntries(source, fallbackEntries) {
  const rawEntries = String(source ?? '')
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean);

  const entries = rawEntries.length ? rawEntries : fallbackEntries;
  const seen = new Set();
  const result = [];

  for (const entry of entries) {
    const login = entry.trim();
    if (!login) {
      continue;
    }

    const normalized = normaliseLogin(login);
    if (!normalized || seen.has(normalized)) {
      continue;
    }

    seen.add(normalized);
    result.push({ original: login, normalized });
  }

  return result;
}

function extractUncheckedTasks(body, limit = 5) {
  if (!body) {
    return [];
  }

  const lines = String(body)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const tasks = [];
  for (const line of lines) {
    const match = line.match(/^- \[ \] \s*(.+)$/i);
    if (match) {
      const task = match[1].trim();
      if (task) {
        tasks.push(task);
      }
    }
    if (tasks.length >= limit) {
      break;
    }
  }
  return tasks;
}

function escapeRegExp(value) {
  return String(value ?? '').replace(/[\\^$.*+?()[\]{}|]/g, '\\$&');
}

function detectKeepaliveSentinel(comments, { sentinelPattern, headerPattern, agentLogins }) {
  if (!Array.isArray(comments) || !comments.length) {
    return null;
  }

  const codexLogins = new Set(agentLogins.map(normaliseLogin));
  codexLogins.add('stranske-automation-bot');
  const codexMentionPattern = /@codex\b/i;

  const sorted = [...comments].sort(
    (a, b) => new Date(b.updated_at || b.created_at) - new Date(a.updated_at || a.created_at)
  );

  for (const comment of sorted) {
    const body = comment?.body || '';
    if (!body) {
      continue;
    }

    if (!(sentinelPattern.test(body) || headerPattern.test(body))) {
      continue;
    }

    const login = normaliseLogin(comment?.user?.login);
    if (codexLogins.has(login) || codexMentionPattern.test(body)) {
      return { comment, login };
    }
  }

  return null;
}

function detectExistingKeepalive(comments, { marker, agentLogins, headerPattern }) {
  if (!Array.isArray(comments) || !comments.length) {
    return [];
  }

  const markerToken = String(marker || '').trim();
  const automationLogins = new Set(agentLogins.map(normaliseLogin));
  automationLogins.add('stranske-automation-bot');

  const looksLikeKeepalive = (comment, body) => {
    const login = normaliseLogin(comment?.user?.login);
    if (!automationLogins.has(login)) {
      return false;
    }

    if (markerToken && body.includes(markerToken)) {
      return true;
    }

    if (headerPattern.test(body)) {
      return true;
    }

    const lower = body.toLowerCase();
    return (
      lower.includes('keepalive mode:') ||
      (lower.includes('@codex plan-and-execute') && lower.includes('checklist'))
    );
  };

  return comments
    .map((comment) => {
      const body = comment?.body || '';
      if (!body) {
        return null;
      }

      const markerPresent = markerToken && body.includes(markerToken);
      if (!(markerPresent || looksLikeKeepalive(comment, body))) {
        return null;
      }

      return {
        comment,
        id: comment.id,
        body,
        timestamp: new Date(comment.updated_at || comment.created_at).getTime(),
      };
    })
    .filter(Boolean)
    .sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));
}

function extractKeepaliveRound(body) {
  const match = String(body || '').match(/<!--\s*keepalive-round:(\d+)\s*-->/i);
  if (match) {
    const round = Number(match[1]);
    if (Number.isFinite(round) && round > 0) {
      return round;
    }
  }
  return null;
}

function computeNextRound(candidates) {
  if (!Array.isArray(candidates) || !candidates.length) {
    return 1;
  }

  const rounds = candidates
    .map((candidate) => extractKeepaliveRound(candidate.body))
    .filter((value) => Number.isFinite(value) && value > 0);

  if (rounds.length) {
    return Math.max(...rounds) + 1;
  }

  return candidates.length + 1;
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
  const pausedLabel = 'agents:paused';

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
    summary.addRaw('Skip requested via options_json.').addEOL();
    summary.addRaw('Skipped 0 paused PRs.').addEOL();
    summary.addRaw('Evaluated pull requests: 0').addEOL();
    await summary.write();
    return;
  }

  const idleMinutes = coerceNumber(options.keepalive_idle_minutes, 10, { min: 0 });
  const repeatMinutes = coerceNumber(options.keepalive_repeat_minutes, 30, { min: 0 });

  // When orchestrator is triggered by Gate completion, we want immediate keepalive activation
  const triggeredByGate = coerceBool(options.triggered_by_gate, false);
  const effectiveIdleMinutes = triggeredByGate ? 0 : idleMinutes;
  // When checking for recent commands, always use the full idle period even if triggered by Gate
  // This prevents keepalive from interrupting fresh human commands

  const labelSource = options.keepalive_labels ?? options.keepalive_label ?? 'agents:keepalive,agent:codex';
  let targetLabels = String(labelSource)
    .split(',')
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);
  if (!targetLabels.length) {
    targetLabels = ['agents:keepalive', 'agent:codex'];
  }
  targetLabels = dedupe(targetLabels);

  const commandRaw = options.keepalive_command ?? '@codex';
  const command = String(commandRaw).trim() || '@codex';
  const commandLower = command.toLowerCase();

  const markerRaw = options.keepalive_marker ?? '<!-- codex-keepalive-marker -->';
  const marker = String(markerRaw);

  const sentinelRaw = options.keepalive_sentinel ?? '[keepalive]';
  const sentinelPattern = new RegExp(escapeRegExp(sentinelRaw), 'i');
  const keepaliveHeaderPattern = /###\s*Keepalive:\s*(on|enabled)/i;

  const instructionTemplateRaw = options.keepalive_instruction ?? '';
  const instructionTemplate = String(instructionTemplateRaw).trim();

  const agentSource = options.keepalive_agent_logins ?? 'chatgpt-codex-connector[bot],stranske-automation-bot';
  const agentEntries = parseAgentLoginEntries(agentSource, [
    'chatgpt-codex-connector[bot]',
    'stranske-automation-bot',
  ]);
  let agentLogins = agentEntries.map(({ normalized }) => normalized);
  agentLogins = dedupe(agentLogins);

  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const now = Date.now();
  const triggered = [];
  const refreshed = [];
  const previews = [];
  const paused = [];
  const skipped = [];
  let skippedCount = 0;
  let scanned = 0;
  addHeading();
  summary
    .addRaw(`Target labels: ${targetLabels.map((label) => `**${label}**`).join(', ')}`)
    .addEOL();
  summary
    .addRaw(
      `Agent logins: ${agentLogins
        .map((login) => `**${login}**`)
        .join(', ')}`
    )
    .addEOL();

  const paginatePulls = github.paginate.iterator(
    github.rest.pulls.list,
    { owner, repo, state: 'open', per_page: 50 }
  );

  const fetchIssueComments = async (issueNumber) => {
    const comments = [];
    const perPage = 100;
    const hasIterator = Boolean(github.paginate?.iterator);

    if (hasIterator) {
      const iterator = github.paginate.iterator(github.rest.issues.listComments, {
        owner,
        repo,
        issue_number: issueNumber,
        per_page: perPage,
      });

      for await (const page of iterator) {
        const data = Array.isArray(page.data) ? page.data : [];
        if (data.length) {
          comments.push(...data);
        }
      }
    } else {
      let page = 1;
      while (true) {
        const { data } = await github.rest.issues.listComments({
          owner,
          repo,
          issue_number: issueNumber,
          per_page: perPage,
          page,
        });
        if (!Array.isArray(data) || !data.length) {
          break;
        }
        comments.push(...data);
        if (data.length < perPage) {
          break;
        }
        page += 1;
      }
    }

    return comments;
  };

  for await (const page of paginatePulls) {
    for (const pr of page.data) {
      scanned += 1;
      const labelNames = (pr.labels || []).map((label) =>
        (typeof label === 'string' ? label : label?.name || '').toLowerCase()
      );

      const prNumber = pr.number;
      const headRef = String(pr.head?.ref || '').trim();
      const recordSkip = (reason, { paused: pausedEntry = false } = {}) => {
        const entry = `#${prNumber} – ${reason}`;
        skipped.push(entry);
        skippedCount += 1;
        if (pausedEntry) {
          paused.push(entry);
        }
        core.info(`#${prNumber}: skipped – ${reason}`);
      };

      if (labelNames.includes(pausedLabel)) {
        recordSkip('keepalive paused via agents:paused label', { paused: true });
        continue;
      }

      const comments = await fetchIssueComments(prNumber);
      if (!comments.length) {
        recordSkip('no timeline comments');
        continue;
      }

      const hasTargetLabel = targetLabels.some((label) => labelNames.includes(label));

      if (!hasTargetLabel) {
        const sentinel = detectKeepaliveSentinel(comments, {
          sentinelPattern,
          headerPattern: keepaliveHeaderPattern,
          agentLogins,
        });

        if (!sentinel) {
          recordSkip('keepalive opt-in not detected');
          continue;
        }

        core.info(`#${prNumber}: keepalive opted-in via sentinel comment ${sentinel.comment?.html_url || ''}.`);
      }

      const botComments = comments
        .filter((comment) => agentLogins.includes(normaliseLogin(comment.user?.login)))
        .sort((a, b) => new Date(a.updated_at || a.created_at) - new Date(b.updated_at || b.created_at));
      if (!botComments.length) {
        recordSkip('Codex has not commented yet');
        continue;
      }

      const lastAgentComment = botComments[botComments.length - 1];
      const lastAgentTs = new Date(lastAgentComment.updated_at || lastAgentComment.created_at).getTime();
      if (!Number.isFinite(lastAgentTs)) {
        recordSkip('unable to parse Codex timestamp');
        continue;
      }

      const minutesSinceAgent = (now - lastAgentTs) / 60000;
      if (minutesSinceAgent < effectiveIdleMinutes) {
        recordSkip(`last Codex activity ${minutesSinceAgent.toFixed(1)} minutes ago (< ${effectiveIdleMinutes})`);
        continue;
      }

      // Skip the mention guard for Gate-triggered sweeps to keep responsiveness high.
      if (!triggeredByGate) {
        const agentMentionPattern = /@(codex|claude|agent)\b/i;
        const agentMentionComments = comments
          .filter((comment) => agentMentionPattern.test(comment.body || ''))
          .sort((a, b) => new Date(a.created_at) - new Date(b.created_at));

        if (agentMentionComments.length > 0) {
          const latestMentionComment = agentMentionComments[agentMentionComments.length - 1];
          const latestMentionTs = new Date(latestMentionComment.created_at).getTime();

          if (Number.isFinite(latestMentionTs)) {
            const minutesSinceMention = (now - latestMentionTs) / 60000;
            const mentionWindow = Math.max(idleMinutes, 1);

            if (minutesSinceMention <= mentionWindow) {
              let allCommits = [];
              let page = 1;
              let fetched;
              do {
                const { data: commitsPage } = await github.rest.pulls.listCommits({
                  owner,
                  repo,
                  pull_number: prNumber,
                  per_page: 100,
                  page,
                });
                fetched = commitsPage.length;
                allCommits = allCommits.concat(commitsPage);
                page += 1;
              } while (fetched === 100);

              const sortedCommits = allCommits.sort((a, b) => {
                const aDate = new Date(a.commit.committer?.date || a.commit.author?.date || 0);
                const bDate = new Date(b.commit.committer?.date || b.commit.author?.date || 0);
                return bDate - aDate;
              });

              if (sortedCommits.length > 0) {
                const latestCommit = sortedCommits[0];
                const latestCommitTs = new Date(latestCommit.commit.committer?.date || latestCommit.commit.author?.date).getTime();
                if (Number.isFinite(latestCommitTs) && latestMentionTs > latestCommitTs) {
                  recordSkip(`waiting for commit after @agent command (${minutesSinceMention.toFixed(1)} minutes ago)`);
                  continue;
                }
              } else {
                recordSkip(`waiting for first commit after @agent command (${minutesSinceMention.toFixed(1)} minutes ago)`);
                continue;
              }
            }
          }
        }
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

      const latestChecklist = checklistComments[0];
      if (!latestChecklist) {
        recordSkip('no Codex checklist with outstanding tasks');
        continue;
      }

      const keepaliveCandidates = detectExistingKeepalive(comments, {
        marker,
        agentLogins,
        headerPattern: keepaliveHeaderPattern,
      });
      const latestKeepalive = keepaliveCandidates[0];
      if (latestKeepalive && !triggeredByGate) {
        const lastKeepaliveTs = latestKeepalive.timestamp;
        const minutesSinceKeepalive = (now - lastKeepaliveTs) / 60000;
        if (minutesSinceKeepalive < repeatMinutes) {
          recordSkip(`keepalive sent ${minutesSinceKeepalive.toFixed(1)} minutes ago (< ${repeatMinutes})`);
          continue;
        }
      }

      const totalTasks = latestChecklist.total;
      const outstanding = latestChecklist.unchecked;
      const completed = Math.max(0, totalTasks - outstanding);
      const itemWord = outstanding === 1 ? 'item' : 'items';
      const verb = outstanding === 1 ? 'remains' : 'remain';
      const defaultInstruction = `Codex, ${outstanding}/${totalTasks} checklist ${itemWord} ${verb} unchecked (completed ${completed}). Continue executing the plan, write the code and tests needed for the next unchecked tasks, update the checklist, and confirm once everything is complete.`;

      const outstandingTasks = extractUncheckedTasks(latestChecklist.comment.body || '', 5);

      const nextRound = computeNextRound(keepaliveCandidates);
      const roundMarker = `<!-- keepalive-round:${nextRound} -->`;

      let instruction = instructionTemplate || defaultInstruction;
      const replacements = {
        remaining: String(outstanding),
        total: String(totalTasks),
        completed: String(completed),
      };
      for (const [token, value] of Object.entries(replacements)) {
        instruction = instruction.split(`{${token}}`).join(value);
      }

      const bodyParts = [command];
      bodyParts.push('', `**Keepalive Round ${nextRound}**`);
      if (instruction) {
        bodyParts.push('', instruction);
      }
      if (outstandingTasks.length) {
        bodyParts.push('', 'Outstanding tasks to tackle next:');
        for (const task of outstandingTasks) {
          bodyParts.push(`- ${task}`);
        }
      }
      if (marker) {
        bodyParts.push('', marker);
      }
      bodyParts.push('', roundMarker);
      const body = bodyParts.join('\n');
      
      // Ensure agent connectors are assigned before posting keepalive
      // This is critical so the agent actually engages when mentioned
      try {
        // Get the current assignees from the PR data we already have
        const currentLogins = (pr.assignees || []).map((a) => normaliseLogin(a.login));
        const missingAgents = agentEntries
          .filter(({ normalized }) => !currentLogins.includes(normalized))
          .map(({ original }) => original);

        if (missingAgents.length > 0) {
          core.info(`#${prNumber}: adding missing agent assignees: ${missingAgents.join(', ')}`);
          await github.rest.issues.addAssignees({
            owner,
            repo,
            issue_number: prNumber,
            assignees: missingAgents,
          });
        }
      } catch (error) {
        core.warning(`#${prNumber}: failed to ensure agent assignees: ${error.message}`);
      }
      
      if (dryRun) {
        previews.push(`#${prNumber} – keepalive preview (remaining tasks: ${outstanding})`);
        core.info(`#${prNumber}: dry run – keepalive comment not posted (remaining tasks: ${outstanding}).`);
      } else {
        await github.rest.issues.createComment({ owner, repo, issue_number: prNumber, body });
        triggered.push(`#${prNumber} – keepalive posted (remaining tasks: ${outstanding}, round ${nextRound})`);
        core.info(`#${prNumber}: keepalive posted (remaining tasks: ${outstanding}, round ${nextRound}).`);
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
    if (refreshed.length) {
      summary.addDetails('Refreshed keepalive comments', summariseList(refreshed));
    }
    summary.addRaw(`Refreshed keepalive count: ${refreshed.length}`).addEOL();
  }
  if (skipped.length) {
    summary.addDetails('Skipped pull requests', summariseList(skipped));
  }
  summary.addRaw(`Skipped keepalive count: ${skippedCount}`).addEOL();
  if (paused.length) {
    summary.addDetails('Paused pull requests', summariseList(paused));
  }
  const pausedLineTemplate = `Skipped ${paused.length} paused PRs.`;
  const pausedLine = paused.length === 1
    ? pausedLineTemplate.replace('PRs.', 'PR.')
    : pausedLineTemplate;
  summary.addRaw(pausedLine).addEOL();
  summary.addRaw(`Evaluated pull requests: ${scanned}`).addEOL();
  await summary.write();
}

module.exports = { runKeepalive };
