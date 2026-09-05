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
    env: { ...process.env },
    stdio: 'pipe',
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
        observed = (await countLocator.innerText({ timeout: 2000 }))
          .replace(/\s+/g, ' ').trim();
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

    // Prove each state transition before clicking the next control. Otherwise
    // the second click can race with Streamlit's pending Select All rerun.
    await page.getByRole('button', { name: '✅ Select All' }).first().click();
    console.log('After Select All:', await waitForSelection('all'));

    await page.getByRole('button', { name: '❌ Clear All' }).first().click();
    console.log('After Clear All:', await waitForSelection('empty'));

    await page.getByRole('button', { name: '✅ Select All' }).first().click();
    console.log('Final count:', await waitForSelection('all'));

  } finally {
    if (browser) await browser.close();
    appProc.kill('SIGTERM');
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
