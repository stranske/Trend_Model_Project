// Fund selector smoke test using Playwright
// Starts Streamlit app on a dedicated port, loads the Data page, and exercises bulk selection buttons.
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const fetch = global.fetch;

const ROOT = path.join(__dirname, '..', '..');
const VENV_ACTIVATE = path.join(ROOT, '.venv', 'bin', 'activate');
const APP_CMD = fs.existsSync(VENV_ACTIVATE)
  ? `source ${VENV_ACTIVATE} && exec env TREND_DEMO_PROFILE=public_llm_demo PYTHONPATH="." streamlit run streamlit_app/app.py --server.headless true --server.port 8599`
  : 'exec env TREND_DEMO_PROFILE=public_llm_demo PYTHONPATH="." python -m streamlit run streamlit_app/app.py --server.headless true --server.port 8599';
const APP_URL = 'http://localhost:8599';

async function waitForHealth(url, timeoutMs = 30000, intervalMs = 500) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(`${url}/_stcore/health`);
      if (res.ok) return;
    } catch (err) {
      // ignore and retry
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error('Streamlit health check did not become ready');
}

async function main() {
  // Start app
  const appProc = spawn('bash', ['-lc', `cd ${ROOT} && ${APP_CMD}`], {
    // Arrow 25's mimalloc pool can segfault when Streamlit recreates script
    // threads on rerun. Use Arrow's supported system pool for this UI smoke.
    env: {
      ...process.env,
      ARROW_DEFAULT_MEMORY_POOL: process.env.ARROW_DEFAULT_MEMORY_POOL || 'system',
      PYTHONFAULTHANDLER: '1',
    },
    stdio: 'pipe',
  });

  appProc.on('exit', (code, signal) => {
    if (signal && signal !== 'SIGTERM') console.error(`Streamlit terminated by ${signal}`);
    else if (code) console.error(`Streamlit exited with code ${code}`);
  });

  // Pipe app logs for debugging
  appProc.stdout.on('data', (d) => process.stdout.write(d));
  appProc.stderr.on('data', (d) => process.stderr.write(d));

  let browser;
  try {
    await waitForHealth(APP_URL, 45000, 750);

    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    await page.goto(`${APP_URL}/Data`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1500);

    // Choose Sample dataset (avoids date-correction flow)
    const sampleRadio = page.getByRole('radio', { name: /Sample dataset/i });
    if ((await sampleRadio.count()) > 0) {
      const radio = sampleRadio.first();
      if (!(await radio.isChecked())) {
        await radio.check({ force: true });
      }
    }

    // Wait for Fund Column Selection header and current selection count
    await page.getByText('Fund Column Selection').waitFor({ timeout: 20000 });
    // Require one current, visible status. Reading its text also handles
    // Streamlit's nested <strong> without depending on markdown DOM wrappers.
    const countLocator = page.getByText(/^\d+ of \d+ funds selected$/)
      .filter({ visible: true });
    async function waitForSelection(mode) {
      const deadline = Date.now() + 10000;
      let observed = '';
      while (Date.now() < deadline) {
        // A Streamlit rerun briefly leaves both old and new statuses visible.
        // Wait for the unique settled status instead of reading either copy.
        const statuses = await countLocator.allTextContents();
        if (statuses.length !== 1) {
          observed = `${statuses.length} visible selection statuses during rerun`;
          await page.waitForTimeout(100);
          continue;
        }
        observed = statuses[0].replace(/\s+/g, ' ').trim();
        const match = observed.match(/^(\d+) of (\d+) funds selected$/);
        if (match) {
          const selected = Number(match[1]);
          const total = Number(match[2]);
          if ((mode === 'all' && total > 0 && selected === total)
              || (mode === 'empty' && selected === 0)) return observed;
        }
        await page.waitForTimeout(100);
      }
      throw new Error(`Selection did not become ${mode}; current status: ${observed}`);
    }

    async function clickAndWaitForRerun(name) {
      await page.locator('[data-testid="stApp"][data-test-script-state="notRunning"]')
        .waitFor({ timeout: 20000 });
      // A count can already equal the requested value before the click. Observe
      // the complete server rerun so the next click cannot be lost in that run.
      await page.evaluate(() => {
        const app = document.querySelector('[data-testid="stApp"]');
        window.fundSelectorRerun = new Promise((resolve, reject) => {
          let started = false;
          const observer = new MutationObserver(() => {
            const state = app.getAttribute('data-test-script-state');
            if (state === 'running' || state === 'rerunRequested') started = true;
            if (started && state === 'notRunning') {
              clearTimeout(timer);
              observer.disconnect();
              resolve();
            }
          });
          const timer = setTimeout(() => {
            observer.disconnect();
            reject(new Error('Fund selection rerun did not finish'));
          }, 20000);
          observer.observe(app, { attributes: true, attributeFilter: ['data-test-script-state'] });
        });
      });
      await page.getByRole('button', { name, exact: true }).click();
      await page.evaluate(() => window.fundSelectorRerun);
    }

    // Prove each state transition before clicking the next control. Otherwise
    // the second click can race with Streamlit's pending Select All rerun.
    await clickAndWaitForRerun('✅ Select All');
    console.log('After Select All:', await waitForSelection('all'));

    await clickAndWaitForRerun('❌ Clear All');
    console.log('After Clear All:', await waitForSelection('empty'));

    await clickAndWaitForRerun('✅ Select All');
    console.log('Final count:', await waitForSelection('all'));

  } finally {
    try {
      if (browser) await browser.close();
    } finally {
      appProc.kill('SIGTERM');
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
