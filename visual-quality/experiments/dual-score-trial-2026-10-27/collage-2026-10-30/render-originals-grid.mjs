import { spawn } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const directory = dirname(fileURLToPath(import.meta.url));
const executable = process.env.GRID_CHROMIUM_PATH
  || '/home/ahstn/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome';
const output = join(directory, 'originals-grid-render-v2.png');
const profile = await mkdtemp(join(tmpdir(), 'originals-grid-chromium-'));
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
browser.on('exit', code => {
  for (const request of pending.values()) {
    clearTimeout(request.timeout);
    request.reject(new Error(`Chromium exited ${code}: ${diagnostics}`));
  }
  pending.clear();
});

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
      }));
      if (checks.some(image => !image.complete || image.naturalWidth <= 0 || image.naturalHeight <= 0)) {
        throw new Error('At least one image failed to load: ' + JSON.stringify(checks));
      }
      return {
        images: checks,
        width: document.documentElement.scrollWidth,
        height: Math.ceil(document.documentElement.scrollHeight),
      };
    })()`,
    awaitPromise: true,
    returnByValue: true,
  });
  if (decoded.exceptionDetails) throw new Error(JSON.stringify(decoded.exceptionDetails));
  const checks = decoded.result.value;
  if (checks.width !== 1920) throw new Error(`Unexpected page width: ${checks.width}`);
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
  console.log(JSON.stringify({ output, dimensions, decoded_image_count: checks.images.length, images: checks.images }, null, 2));
} finally {
  browser.kill('SIGTERM');
  await new Promise(resolve => browser.exitCode === null ? browser.once('exit', resolve) : resolve());
  await rm(profile, { recursive: true, force: true });
}
