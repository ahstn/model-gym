// Render the original-image grid only after every image has decoded.
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const directory = dirname(fileURLToPath(import.meta.url));
const executable = process.env.GRID_CHROMIUM_PATH
  || '/home/ahstn/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome';
const output = join(directory, 'originals-grid.png');
const digest = async path => createHash('sha256').update(await readFile(path)).digest('hex');
const selection = JSON.parse(await readFile(join(directory, 'selection.json'), 'utf8'));
if (selection.cases.length !== 36 || selection.arm_id !== 'combined') {
  throw new Error('Expected exactly 36 fixed combined-arm cases');
}
for (const [path, hash] of Object.entries(selection.input_files_sha256)) {
  if (await digest(path) !== hash) throw new Error(`Input changed since grid build: ${path}`);
}
for (const item of selection.cases) {
  if (await digest(item.image_path) !== item.view_sha256) {
    throw new Error(`Original image hash mismatch: ${item.case_id}`);
  }
}
const profile = await mkdtemp(join(tmpdir(), 'prompt-collage-chromium-'));
const browser = spawn(executable, [
  '--headless', '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu',
  '--hide-scrollbars', '--no-first-run', '--no-default-browser-check',
  '--remote-debugging-pipe', `--user-data-dir=${profile}`, 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe'] });

let nextId = 0;
let incoming = '';
let diagnostics = '';
const pending = new Map();
browser.stderr.on('data', chunk => { diagnostics += chunk; });
for (const pipe of [browser.stdio[3], browser.stdio[4]]) {
  pipe.on('error', error => { diagnostics += `\nCDP pipe error: ${error.message}`; });
}
browser.stdio[4].on('data', chunk => {
  incoming += chunk.toString();
  let delimiter;
  while ((delimiter = incoming.indexOf('\0')) !== -1) {
    const message = JSON.parse(incoming.slice(0, delimiter));
    incoming = incoming.slice(delimiter + 1);
    const request = pending.get(message.id);
    if (!request) continue;
    pending.delete(message.id);
    clearTimeout(request.timeout);
    if (message.error) request.reject(new Error(JSON.stringify(message.error)));
    else request.resolve(message.result);
  }
});
function rejectPending(error) {
  for (const request of pending.values()) {
    clearTimeout(request.timeout);
    request.reject(error);
  }
  pending.clear();
}
browser.on('error', error => rejectPending(error));
browser.on('exit', code => rejectPending(new Error(`Chromium exited ${code}: ${diagnostics}`)));

function command(method, params = {}, sessionId) {
  return new Promise((resolve, reject) => {
    const id = ++nextId;
    const timeout = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`CDP command timed out: ${method}`));
    }, 45_000);
    pending.set(id, { resolve, reject, timeout });
    browser.stdio[3].write(`${JSON.stringify({ id, method, params, sessionId })}\0`);
  });
}

try {
  const { targetId } = await command('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await command('Target.attachToTarget', { targetId, flatten: true });
  const send = (method, params) => command(method, params, sessionId);
  await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', {
    width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false,
  });
  const navigation = await send('Page.navigate', {
    url: pathToFileURL(join(directory, 'originals-grid.html')).href,
  });
  if (navigation.errorText) throw new Error(navigation.errorText);
  const decoded = await send('Runtime.evaluate', {
    expression: `(async () => {
      if (document.readyState === 'loading') {
        await new Promise(resolve => document.addEventListener('DOMContentLoaded', resolve, { once: true }));
      }
      const images = [...document.images];
      if (images.length !== 36) throw new Error('Expected 36 images, found ' + images.length);
      for (const image of images) image.loading = 'eager';
      await Promise.all(images.map(image => image.decode()));
      await document.fonts.ready;
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      const checks = images.map(image => ({
        case_id: image.closest('article').dataset.caseId,
        complete: image.complete,
        naturalWidth: image.naturalWidth,
        naturalHeight: image.naturalHeight,
        object_fit: getComputedStyle(image).objectFit,
        score_cells: [...image.closest('article').querySelectorAll('td')].map(cell => ({
          model: cell.dataset.model, task: cell.dataset.task,
          score: cell.dataset.score, status: cell.dataset.status, text: cell.textContent,
        })),
      }));
      if (checks.some(image => !image.complete || image.naturalWidth <= 0 || image.naturalHeight <= 0
          || image.object_fit !== 'contain' || image.score_cells.length !== 10)) {
        throw new Error('Image or score-table check failed: ' + JSON.stringify(checks));
      }
      return {
        images: checks,
        columns: getComputedStyle(document.querySelector('.grid')).gridTemplateColumns.split(' ').length,
        width: document.documentElement.scrollWidth,
        height: Math.ceil(document.documentElement.scrollHeight),
      };
    })()`,
    awaitPromise: true,
    returnByValue: true,
  });
  if (decoded.exceptionDetails) throw new Error(JSON.stringify(decoded.exceptionDetails));
  const checks = decoded.result.value;
  if (checks.width !== 1920 || checks.columns !== 6) {
    throw new Error(`Expected six columns at 1920px, got ${checks.columns} at ${checks.width}px`);
  }
  checks.images.forEach((item, index) => {
    const expected = selection.cases[index];
    if (item.case_id !== expected.case_id || item.naturalWidth !== expected.width || item.naturalHeight !== expected.height) {
      throw new Error(`Image order or dimensions mismatch: ${item.case_id}`);
    }
    const models = item.score_cells.map(cell => cell.model).join(',');
    if (models !== 'reference,reference,deepseek,deepseek,luna,luna,gemini,gemini,haiku,haiku') {
      throw new Error(`Unexpected score-table order: ${item.case_id}`);
    }
  });
  const capture = await send('Page.captureScreenshot', {
    format: 'png', fromSurface: true, captureBeyondViewport: true,
    clip: { x: 0, y: 0, width: 1920, height: checks.height, scale: 1 },
  });
  await writeFile(output, Buffer.from(capture.data, 'base64'));
  const png = await readFile(output);
  const dimensions = { width: png.readUInt32BE(16), height: png.readUInt32BE(20) };
  if (dimensions.width !== 1920 || dimensions.height !== checks.height) {
    throw new Error(`Unexpected PNG dimensions: ${JSON.stringify(dimensions)}`);
  }
  const hashes = {};
  for (const name of ['originals-grid.png', 'originals-grid.html', 'scores.csv', 'selection.json', 'collage.css', 'render.mjs', '../build_collage.py']) {
    hashes[name] = await digest(join(directory, name));
  }
  const report = {
    output, dimensions, decoded_image_count: checks.images.length,
    score_cell_count: checks.images.reduce((total, item) => total + item.score_cells.length, 0),
    columns: checks.columns, files_sha256: hashes, images: checks.images,
  };
  await writeFile(join(directory, 'render-validation.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ output, dimensions, decoded_image_count: report.decoded_image_count,
    score_cell_count: report.score_cell_count, columns: report.columns }, null, 2));
} finally {
  if (browser.pid && browser.exitCode === null && browser.signalCode === null) {
    const exited = new Promise(resolve => browser.once('exit', resolve));
    browser.kill('SIGTERM');
    await exited;
  }
  await rm(profile, { recursive: true, force: true });
}
