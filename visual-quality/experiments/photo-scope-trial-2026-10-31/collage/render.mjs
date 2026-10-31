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
const digest = async path => createHash('sha256').update(await readFile(path)).digest('hex');
const selection = JSON.parse(await readFile(join(directory, 'selection.json'), 'utf8'));
if (selection.cases.length !== 72 || selection.arm_id !== 'photo_scope') {
  throw new Error('Expected exactly 72 photography-scope cases');
}
for (const [path, hash] of Object.entries(selection.input_files_sha256)) {
  if (await digest(path) !== hash) throw new Error(`Input changed since grid build: ${path}`);
}
for (const item of selection.cases) {
  if (await digest(item.image_path) !== item.view_sha256) {
    throw new Error(`Original image hash mismatch: ${item.case_id}`);
  }
}
const profile = await mkdtemp(join(tmpdir(), 'photo-scope-collage-chromium-'));
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

function verifyRaster(output, images) {
  const program = `import json, sys
from PIL import Image, ImageStat
payload = json.load(sys.stdin)
image = Image.open(payload["output"]).convert("RGB")
checks = []
for item in payload["images"]:
    box = item["rectangle"]
    scale = min(box["width"] / item["naturalWidth"], box["height"] / item["naturalHeight"])
    width, height = item["naturalWidth"] * scale, item["naturalHeight"] * scale
    left, top = box["left"] + (box["width"] - width) / 2, box["top"] + (box["height"] - height) / 2
    # Inspect the interior of the actual contain-fitted image, excluding letterbox margins.
    inset = max(2, min(width, height) * 0.02)
    bounds = [round(left + inset), round(top + inset), round(left + width - inset), round(top + height - inset)]
    patch = image.crop(bounds)
    if patch.width <= 0 or patch.height <= 0:
        raise ValueError("Empty image pixel patch: " + item["case_id"])
    background = (227, 232, 239)
    different = sum(any(abs(channel - expected) > 3 for channel, expected in zip(pixel, background)) for pixel in patch.getdata())
    fraction = different / (patch.width * patch.height)
    if fraction < 0.01:
        raise ValueError("Image region contains only the empty frame background: " + item["case_id"])
    checks.append({"case_id": item["case_id"], "bounds": bounds, "non_background_fraction": fraction,
                   "rgb_stddev": ImageStat.Stat(patch).stddev})
print(json.dumps(checks))`;
  return new Promise((resolve, reject) => {
    const process = spawn(join(directory, '../../../.venv/bin/python'), ['-c', program],
      { stdio: ['pipe', 'pipe', 'pipe'] });
    let result = '';
    let errors = '';
    process.stdout.on('data', chunk => { result += chunk; });
    process.stderr.on('data', chunk => { errors += chunk; });
    process.on('error', reject);
    process.on('close', code => {
      if (code !== 0) reject(new Error(`Raster validation failed: ${errors}`));
      else {
        try { resolve(JSON.parse(result)); } catch (error) { reject(error); }
      }
    });
    process.stdin.end(JSON.stringify({ output, images }));
  });
}

try {
  const { targetId } = await command('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await command('Target.attachToTarget', { targetId, flatten: true });
  const send = (method, params) => command(method, params, sessionId);
  await send('Page.enable');
  const outputs = [];
  const configurations = [
    { stem: 'originals-grid', width: 2560, columns: 8, cases: selection.cases },
    { stem: 'originals-grid-page-1', width: 1920, columns: 6, cases: selection.cases.slice(0, 36) },
    { stem: 'originals-grid-page-2', width: 1920, columns: 6, cases: selection.cases.slice(36) },
  ];
  for (const config of configurations) {
    await send('Emulation.setDeviceMetricsOverride', {
      width: config.width, height: 1080, deviceScaleFactor: 1, mobile: false,
    });
    const navigation = await send('Page.navigate', {
      url: pathToFileURL(join(directory, config.stem + '.html')).href,
    });
    if (navigation.errorText) throw new Error(navigation.errorText);
    const decoded = await send('Runtime.evaluate', {
      expression: `(async () => {
        if (document.readyState === 'loading') {
          await new Promise(resolve => document.addEventListener('DOMContentLoaded', resolve, { once: true }));
        }
        const images = [...document.images];
        if (images.length !== ${config.cases.length}) throw new Error('Unexpected image count: ' + images.length);
        for (const image of images) image.loading = 'eager';
        await Promise.all(images.map(image => image.decode()));
        await document.fonts.ready;
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        const checks = images.map(image => ({
          case_id: image.closest('article').dataset.caseId,
          eligibility: image.closest('article').dataset.eligibility,
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
          rows: getComputedStyle(document.querySelector('.grid')).gridTemplateRows.split(' ').length,
          width: document.documentElement.scrollWidth,
          height: Math.ceil(document.documentElement.scrollHeight),
        };
      })()`,
      awaitPromise: true,
      returnByValue: true,
    });
    if (decoded.exceptionDetails) throw new Error(JSON.stringify(decoded.exceptionDetails));
    const checks = decoded.result.value;
    if (checks.width !== config.width || checks.columns !== config.columns
        || checks.rows !== config.cases.length / config.columns) {
      throw new Error(`Unexpected layout for ${config.stem}: ${JSON.stringify(checks)}`);
    }
    checks.images.forEach((item, index) => {
      const expected = config.cases[index];
      if (item.case_id !== expected.case_id || item.naturalWidth !== expected.width
          || item.naturalHeight !== expected.height || item.eligibility !== expected.eligibility) {
        throw new Error(`Image order, dimensions or eligibility mismatch: ${item.case_id}`);
      }
      if (item.score_cells.some((cell, cellIndex) =>
          Object.entries(expected.score_cells[cellIndex]).some(([key, value]) => cell[key] !== value))) {
        throw new Error(`Displayed scores differ from recorded assessments: ${item.case_id}`);
      }
    });
    // A large screenshot clip alone does not force offscreen images to rasterize.
    // Expand the real viewport to encompass the entire document before the final paint.
    await send('Emulation.setDeviceMetricsOverride', {
      width: config.width, height: checks.height, deviceScaleFactor: 1, mobile: false,
    });
    const painted = await send('Runtime.evaluate', {
      expression: `(async () => {
        window.scrollTo(0, 0);
        const images = [...document.images];
        for (const image of images) { image.loading = 'eager'; image.decoding = 'sync'; }
        await Promise.all(images.map(image => image.decode()));
        await document.fonts.ready;
        for (let frame = 0; frame < 4; frame++) {
          await new Promise(resolve => requestAnimationFrame(resolve));
        }
        const rectangles = images.map(image => {
          const { left, top, right, bottom, width, height } = image.getBoundingClientRect();
          return { case_id: image.closest('article').dataset.caseId,
            rectangle: { left, top, right, bottom, width, height } };
        });
        if (rectangles.some(({ rectangle }) => rectangle.left < 0 || rectangle.top < 0
            || rectangle.right > window.innerWidth || rectangle.bottom > window.innerHeight)) {
          throw new Error('An image remains outside the full screenshot viewport');
        }
        return { width: window.innerWidth, height: window.innerHeight,
          document_height: Math.ceil(document.documentElement.scrollHeight), rectangles };
      })()`,
      awaitPromise: true, returnByValue: true,
    });
    if (painted.exceptionDetails) throw new Error(JSON.stringify(painted.exceptionDetails));
    const viewport = painted.result.value;
    if (viewport.width !== config.width || viewport.height !== checks.height
        || viewport.document_height !== checks.height) {
      throw new Error(`Document size changed during full-viewport paint: ${JSON.stringify(viewport)}`);
    }
    checks.images.forEach((image, index) => {
      if (image.case_id !== viewport.rectangles[index].case_id) throw new Error('Paint order changed');
      image.rectangle = viewport.rectangles[index].rectangle;
    });
    const capture = await send('Page.captureScreenshot', {
      format: 'png', fromSurface: true, captureBeyondViewport: false,
      clip: { x: 0, y: 0, width: config.width, height: checks.height, scale: 1 },
    });
    const output = join(directory, config.stem + '.png');
    await writeFile(output, Buffer.from(capture.data, 'base64'));
    const png = await readFile(output);
    const dimensions = { width: png.readUInt32BE(16), height: png.readUInt32BE(20) };
    if (dimensions.width !== config.width || dimensions.height !== checks.height) {
      throw new Error(`Unexpected PNG dimensions: ${JSON.stringify(dimensions)}`);
    }
    const raster_checks = await verifyRaster(output, checks.images);
    outputs.push({
      output, dimensions, decoded_image_count: checks.images.length,
      screenshot_viewport: { width: viewport.width, height: viewport.height }, raster_checks,
      score_cell_count: checks.images.reduce((total, item) => total + item.score_cells.length, 0),
      columns: checks.columns, rows: checks.rows, images: checks.images,
    });
  }
  const hashes = {};
  const artifactNames = configurations.flatMap(config => [config.stem + '.png', config.stem + '.html']);
  for (const name of [...artifactNames, 'scores.csv', 'selection.json', 'collage.css', 'render.mjs', '../build_collage.py']) {
    hashes[name] = await digest(join(directory, name));
  }
  const report = { ...outputs[0], files_sha256: hashes, pages: outputs.slice(1) };
  await writeFile(join(directory, 'render-validation.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(outputs.map(({ images, raster_checks, ...summary }) => ({
    ...summary, raster_check_count: raster_checks.length,
    minimum_non_background_fraction: Math.min(...raster_checks.map(check => check.non_background_fraction)),
  })), null, 2));
} finally {
  if (browser.pid && browser.exitCode === null && browser.signalCode === null) {
    const exited = new Promise(resolve => browser.once('exit', resolve));
    browser.kill('SIGTERM');
    await exited;
  }
  await rm(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
}
