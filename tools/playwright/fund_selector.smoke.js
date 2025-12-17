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
  ? `source ${VENV_ACTIVATE} && PYTHONPATH="." streamlit run streamlit_app/app.py --server.headless true --server.port 8599`
  : 'PYTHONPATH="." python -m streamlit run streamlit_app/app.py --server.headless true --server.port 8599';
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

  try {
    await waitForHealth(APP_URL, 45000, 750);

    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    await page.goto(`${APP_URL}/Data`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1500);

    // Choose Sample dataset (avoids date-correction flow)
    const sampleRadio = page.getByRole('radio', { name: /Sample dataset/i });
    if ((await sampleRadio.count()) > 0) {
      await sampleRadio.first().click();
    }

    // Wait for Fund Column Selection header and current selection count
    await page.getByText('Fund Column Selection').waitFor({ timeout: 20000 });
    const countLocator = page.getByText(/\d+ of \d+ funds selected/);

    // Select All
    await page.getByRole('button', { name: '✅ Select All' }).first().click();
    const afterSelectAll = await countLocator.textContent({ timeout: 10000 });
    console.log('After Select All:', afterSelectAll);

    // Clear All
    await page.getByRole('button', { name: '❌ Clear All' }).first().click();
    await page.getByText('0 of').waitFor({ timeout: 10000 });

    // Select All again to confirm state can recover
    await page.getByRole('button', { name: '✅ Select All' }).first().click();
    const finalCount = await countLocator.textContent({ timeout: 10000 });
    console.log('Final count:', finalCount);

    await browser.close();
  } finally {
    appProc.kill('SIGKILL');
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
