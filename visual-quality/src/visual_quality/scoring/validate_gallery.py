"""Validate published gallery artifacts against their frozen corpus and score CSV.

The browser harness is injected into a temporary copy; source galleries are never
modified. ``--interim-layout`` labels a 72-case layout fixture as nonpublishable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import signal
import subprocess
import tempfile
from collections import Counter
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image

from visual_quality.ensemble import replace_ensemble_rows

from .gallery import GALLERY, MODELS, score_text
from .history import rehearsal_root

HERE = rehearsal_root()
CHROMIUM = Path("/home/ahstn/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome")
NUMERIC_FIELDS = (
    "technical_score",
    "aesthetic_score",
    "latency_seconds",
    "price_usd",
    "review_elapsed_seconds",
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class Cards(HTMLParser):
    """Capture the actual static card structure without a rendering dependency."""

    def __init__(self) -> None:
        super().__init__()
        self.cards = []
        self.current = None
        self.row = None
        self.cell = None

    def handle_starttag(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        attrs = dict(attributes)
        if tag == "article":
            self.current = {"id": attrs.get("data-case-id"), "rows": [], "images": [], "links": []}
        if self.current is None:
            return
        if tag == "img":
            self.current["images"].append(attrs.get("src"))
        elif tag == "a":
            self.current["links"].append(attrs.get("href"))
        elif tag == "tr":
            self.row = []
        elif tag == "td":
            self.cell = []

    def handle_data(self, value: str) -> None:
        if self.cell is not None:
            self.cell.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self.cell is not None:
            self.row.append("".join(self.cell))
            self.cell = None
        elif tag == "tr" and self.row:
            self.current["rows"].append(self.row)
            self.row = None
        elif tag == "article" and self.current is not None:
            self.cards.append(self.current)
            self.current = None


def read_csv(path: Path) -> dict:
    records = {}
    with path.open(newline="") as stream:
        for row in replace_ensemble_rows(list(csv.DictReader(stream))):
            key = (row["case_id"], row["model"] + ":" + row["arm"])
            require(key not in records, "Duplicate CSV record")
            records[key] = {
                **{name: row.get(name) for name in ("case_id", "model", "arm", "status")},
                "evidence": (
                    json.loads(row.get("evidence") or "{}")
                    if isinstance(row.get("evidence"), str)
                    else (row.get("evidence") or {})
                ),
                **{
                    name: float(row[name]) if row.get(name) not in (None, "", "null") else None
                    for name in NUMERIC_FIELDS
                },
            }
    by_case = {}
    for (identity, key), record in records.items():
        by_case.setdefault(identity, {})[key] = record
    return by_case


def validate_data(args: argparse.Namespace) -> tuple[list[dict], dict]:
    manifest = json.loads((args.experiment / "manifest.json").read_text())
    cases = {row["case_id"]: row for row in manifest["cases"]}
    selection = json.loads((args.gallery / "selection.json").read_text())
    frozen_path = args.experiment / "collage-selection.json"
    frozen = json.loads(frozen_path.read_text())
    require(selection["case_ids"] == frozen["case_ids"], "Gallery selection changed")
    require(selection["frozen_selection_sha256"] == digest(frozen_path), "Selection pin mismatch")
    require(
        selection["references_sha256"] == digest(args.experiment / "references.jsonl"),
        "Reference pin mismatch",
    )
    source = (args.gallery / "data.js").read_text()
    require(source.startswith("const CASES=") and source.endswith(";\n"), "Unexpected gallery data format")
    data = json.loads(source[len("const CASES=") : -2])
    ids = [row["id"] for row in data]
    expected = set(frozen["case_ids"]) if args.interim_layout else set(cases)
    require(len(ids) == len(set(ids)) and set(ids) == expected, "Gallery case coverage mismatch")
    records = read_csv(args.scores)
    neighbours = json.loads((args.experiment / "selected-neighbours.json").read_text())["neighbours"]
    labels = json.loads((args.experiment / "cluster-labels.json").read_text())["labels"]
    families = {}
    for identity in expected:
        families.setdefault(cases[identity]["family_id"], set()).add(identity)
    preview_hashes, rows_checked = {}, 0
    for row in data:
        case = cases[row["id"]]
        require(row["partition"] == case["benchmark_partition"], "Partition mismatch")
        require(row["family"] == case["family_id"], "Family mismatch")
        require(row["view_sha256"] == case["view_sha256"], "View hash mismatch")
        require(row["original"] == Path(case["image_path"]).as_uri(), "Original link mismatch")
        require(row["cluster"] == str(case["rehearsal_content_cluster"]), "Cluster mismatch")
        require(row["cluster_label"] == labels[row["cluster"]], "Cluster content label mismatch")
        require(row["neighbours"] == neighbours[row["id"]], "Related-image results changed")
        require(all(item["case_id"] in cases for item in row["neighbours"]), "Unknown related image")
        require(set(row["siblings"]) == families[row["family"]] - {row["id"]}, "Family links mismatch")
        expected_rows = records[row["id"]]
        require(row["scores"] == expected_rows, "Gallery scores differ from CSV: " + row["id"])
        rows_checked += len(expected_rows)
        preview = args.gallery / row["preview"]
        require(preview.resolve().is_relative_to(args.gallery.resolve()), "Preview path escapes gallery")
        with Image.open(preview) as image:
            require(image.format == "JPEG" and image.mode == "RGB", "Unexpected preview encoding")
            require(image.width <= 600 and image.height <= 400, "Unexpected preview dimensions")
            image.verify()
        preview_hashes[row["id"]] = digest(preview)
    counts = dict(Counter(row["partition"] for row in data))
    clusters = len({row["cluster"] for row in data})
    if not args.interim_layout:
        require(
            counts == {"development": 100, "representative": 300, "challenge": 100, "rehearsal": 3500},
            "Unexpected final partitions",
        )
        require(clusters == 40, "Expected all 40 content clusters")
    return data, {
        "cases": len(data),
        "partitions": counts,
        "clusters": clusters,
        "csv_records_checked": rows_checked,
        "preview_sha256": preview_hashes,
        "references_sha256": selection["references_sha256"],
        "selection_sha256": digest(args.gallery / "selection.json"),
    }


def expected_cells(label: str, record: dict) -> list[str]:
    values = [
        score_text(record, key)
        if key != "latency_seconds"
        else "—"
        if record[key] is None
        else f"{record[key]:.2f}"
        for key in ("technical_score", "aesthetic_score", "latency_seconds")
    ]
    price = record["price_usd"]
    price_text = "—" if price is None else "$0.00" if price == 0 else f"${price:.6f}"
    return [label, *values, price_text]


def validate_pages(args: argparse.Namespace, data: list[dict]) -> list[dict]:
    by_id = {row["id"]: row for row in data}
    selection = json.loads((args.gallery / "selection.json").read_text())["case_ids"]
    png_record = json.loads((args.gallery / "png-validation.json").read_text())
    require(png_record["scores_sha256"] == digest(args.scores), "PNG scores provenance mismatch")
    require(
        png_record["selection_sha256"] == digest(args.gallery / "selection.json"),
        "PNG selection provenance mismatch",
    )
    require(
        png_record["score_cells"] == 1008 and png_record["models_per_image"] == 7,
        "PNG table dimensions mismatch",
    )
    pages = []
    for index in range(4):
        path = args.gallery / f"comparison-page-{index + 1}.html"
        parser = Cards()
        parser.feed(path.read_text())
        ids = selection[index * 18 : (index + 1) * 18]
        require([card["id"] for card in parser.cards] == ids, "Static page identities mismatch")
        for card in parser.cards:
            case = by_id[card["id"]]
            require(card["images"] == [case["preview"]], "Static preview mismatch")
            require(card["links"] == [case["original"]], "Static original link mismatch")
            expected = [
                expected_cells(
                    label, case["scores"][model + (":reference" if model == "reference" else ":baseline")]
                )
                for model, label in MODELS
            ]
            require(
                card["rows"] == expected, "Static score/latency/price cells differ from CSV: " + card["id"]
            )
        png = args.gallery / f"comparison-page-{index + 1}.png"
        record = png_record["pages"][index]
        require(
            record["case_ids"] == ids and record["sha256"] == digest(png), "PNG hash or selection mismatch"
        )
        with Image.open(png) as image:
            require(image.format == "PNG" and image.size == (1964, 3455), "PNG size/encoding mismatch")
            image.verify()
        pages.append(
            {
                "html": path.name,
                "html_sha256": digest(path),
                "png_sha256": digest(png),
                "cases": 18,
                "assessor_rows": 126,
                "score_cells": 252,
            }
        )
    return pages


BROWSER_SCRIPT = r"""
<script>
(async()=>{
const result={status:'running',checks:[],errors:[],loadedPreviews:0};
const out=document.createElement('pre');out.id='gallery-validation-result';document.body.append(out);
const assert=(x,label)=>{if(!x)throw Error(label);result.checks.push(label)};
const el=id=>document.getElementById(id), ids=()=>[...document.querySelectorAll('#grid article')].map(x=>x.dataset.caseId);
const set=(id,value)=>{el(id).value=value;el(id).dispatchEvent(new Event('input',{bubbles:true}))};
const expectedCount=(p,k='all')=>CASES.filter(c=>(p==='all'||p==='benchmark'&&c.partition!=='rehearsal'||c.partition===p)&&(k==='all'||c.cluster===k)).length;
const assertCount=(p,k='all')=>assert(el('count').textContent.startsWith(expectedCount(p,k)+' images'),'count '+p+'/'+k);
try{
assert(el('partition').value==='benchmark','default benchmark');assertCount('benchmark');
assert(ids().length===Math.min(24,expectedCount('benchmark')),'default first page length');
for(const p of [...el('partition').options].map(o=>o.value).filter(v=>v!=='benchmark')){set('partition',p);assertCount(p)}
set('partition','all');const first=ids();el('next').click();assert(!ids().some(x=>first.includes(x)),'next page distinct');el('prev').click();assert(JSON.stringify(ids())===JSON.stringify(first),'previous page restored');
const clusterValues=[...new Set(CASES.map(c=>c.cluster))];assert(el('cluster').options.length===clusterValues.length+1,'all cluster options');
for(const k of clusterValues){set('cluster',k);assertCount('all',k);assert(ids().every(id=>CASES.find(c=>c.id===id).cluster===k),'cluster membership '+k)}
set('cluster','all');set('partition','benchmark');
const target=CASES.find(c=>c.partition!=='rehearsal');set('query',target.id.toLowerCase());assert(JSON.stringify(ids())===JSON.stringify([target.id]),'case query case-insensitive');
for(const arm of [...el('arm').options].map(o=>o.value)){
set('arm',arm);const rows=[...document.querySelectorAll('#grid article > table > tbody > tr')];assert(rows.length===7,'seven assessor rows '+arm);
rows.forEach((row,i)=>{const [m,label]=MODELS[i],key=m+':'+(m==='reference'?'reference':['luna','deepseek','photo_ensemble'].includes(m)?arm:'baseline'),r=target.scores[key]??{},cells=[...row.children].map(x=>x.textContent);assert(cells[0]===label,'assessor '+arm+'/'+m);for(const [j,key]of [[1,'technical_score'],[2,'aesthetic_score'],[3,'latency_seconds']]){const value=r[key+'_display']||(r[key]==null?'—':Number(r[key]).toFixed(2));assert(cells[j]===value,'display '+arm+'/'+m+'/'+key)}assert(cells[4]===(r.price_usd==null?'—':Number(r.price_usd)===0?'$0.00':'$'+Number(r.price_usd).toFixed(6)),'price '+arm+'/'+m)});
}
set('query','__no_such_image__');assert(ids().length===0&&el('count').textContent==='0 images · page 1/1','empty search');
set('query','');set('arm','baseline');set('sort','technical_score');const expected=CASES.filter(c=>c.partition!=='rehearsal').sort((a,b)=>(a.scores['reference:reference']?.technical_score??99)-(b.scores['reference:reference']?.technical_score??99)||a.id.localeCompare(b.id)).slice(0,24).map(c=>c.id);assert(JSON.stringify(ids())===JSON.stringify(expected),'reference technical sort');
const destination=target.neighbours[0]?.case_id;if(destination){const frame=document.createElement('iframe');frame.src=new URL('index.html?case='+encodeURIComponent(destination),document.baseURI);document.body.append(frame);await new Promise((resolve,reject)=>{frame.onload=resolve;frame.onerror=()=>reject(Error('deep link iframe load'))});const doc=frame.contentDocument;assert(doc.getElementById('partition').value==='all'&&doc.getElementById('query').value===destination,'deep link applied');const expectedPresent=CASES.some(c=>c.id===destination);assert(doc.querySelectorAll('#grid article').length===(expectedPresent?1:0),'deep link result');frame.remove()}
let next=0;await Promise.all(Array.from({length:16},async()=>{while(next<CASES.length){const c=CASES[next++];await new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>{if(img.naturalWidth===0)reject(Error('empty preview '+c.id));else{result.loadedPreviews++;resolve()}};img.onerror=()=>reject(Error('preview failed '+c.id));img.src=c.preview})}}));assert(result.loadedPreviews===CASES.length,'all browser previews loaded');
result.status='passed';
}catch(error){result.status='failed';result.errors.push(String(error.stack??error))}
out.textContent=JSON.stringify(result);document.title='GALLERY_VALIDATION_'+result.status.toUpperCase();
})();
</script>
"""


CDP_SCRIPT = r"""
import {spawn} from 'node:child_process';
import {readFile} from 'node:fs/promises';
const [rawCommand, url, profile] = process.argv.slice(1);
const [executable, ...args] = JSON.parse(rawCommand);
const browser = spawn(executable, args, {stdio:'ignore'});
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
let socket;
try {
  let port;
  for (let i=0; i<100; i++) {
    try {port=(await readFile(profile+'/DevToolsActivePort','utf8')).split('\n')[0]; break}
    catch {await pause(100)}
  }
  if (!port) throw Error('Chromium did not expose its local debugging port');
  const response=await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(url)}`,{method:'PUT'});
  if (!response.ok) throw Error('Chromium could not create the validation page');
  const target=await response.json();
  socket=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject});
  let serial=0;const pending=new Map();
  socket.onmessage=event=>{
    const reply=JSON.parse(event.data), done=pending.get(reply.id);
    if(done){pending.delete(reply.id);done(reply)}
  };
  const call=(method,params)=>new Promise(resolve=>{
    const id=++serial;pending.set(id,resolve);socket.send(JSON.stringify({id,method,params}));
  });
  const deadline=Date.now()+270000;
  while(Date.now()<deadline){
    const reply=await call('Runtime.evaluate',{
      expression:"document.getElementById('gallery-validation-result')?.textContent || ''",
      returnByValue:true
    });
    if(reply.error || reply.result?.exceptionDetails) throw Error(JSON.stringify(reply));
    const value=reply.result.result.value;
    if(value){console.log(value);break}
    await pause(100);
  }
  if(Date.now()>=deadline) throw Error('Browser validation result timed out');
} catch(error){console.error(String(error));process.exitCode=1}
finally{socket?.close();browser.kill('SIGKILL')}
"""


def run_browser(command: list[str], *, timeout: float = 300) -> subprocess.CompletedProcess:
    """Bound the full preview load and stop owned descendants before profile cleanup."""
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        # Killing only the parent leaves Chrome children writing to its profile.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            # A child that detached into another session must not hold cleanup open.
            process.stdout.close()
            process.stderr.close()
            process.wait(timeout=5)


def browser_check(args: argparse.Namespace, *, script: str = BROWSER_SCRIPT) -> dict:
    index = (args.gallery / "index.html").read_text()
    with tempfile.TemporaryDirectory(prefix="rehearsal-gallery-check-") as directory:
        work = Path(directory)
        document = index.replace(
            "<head>", '<head><base href="' + html.escape(args.gallery.as_uri() + "/", quote=True) + '">', 1
        )
        document = document.replace("</body>", script + "</body>", 1)
        harness = work / "validation.html"
        harness.write_text(document)
        command = [
            str(args.chromium),
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--allow-file-access-from-files",
            "--disable-crash-reporter",
            "--no-first-run",
            f"--user-data-dir={work / 'profile'}",
            "--remote-debugging-port=0",
            "about:blank",
        ]
        # CDP waits for the actual asynchronous result; --dump-dom's virtual-time
        # completion can stall even when the 10,000-preview harness would pass.
        completed = run_browser(
            [
                "node",
                "--input-type=module",
                "-e",
                CDP_SCRIPT,
                json.dumps(command),
                harness.as_uri(),
                str(work / "profile"),
            ]
        )
        require(completed.returncode == 0, "Headless browser failed: " + completed.stderr[-800:])
        result = json.loads(completed.stdout)
        require(result.get("status") == "passed", "Browser validation failed: " + json.dumps(result))
        result["chromium_version"] = subprocess.check_output(
            [str(args.chromium), "--version"], text=True, timeout=10
        ).strip()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=HERE)
    parser.add_argument("--gallery", type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--chromium", type=Path, default=CHROMIUM)
    parser.add_argument("--interim-layout", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.experiment = args.experiment.resolve()
    args.gallery = (args.gallery or GALLERY).resolve()
    args.scores = (args.scores or args.experiment / "scores-long.csv").resolve()
    args.output = args.output or args.gallery / (
        "interim-layout-validation.json" if args.interim_layout else "publication-validation.json"
    )
    data, records = validate_data(args)
    pages = validate_pages(args, data)
    browser = browser_check(args)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "passed",
        "scope": "interim layout fixture; incomplete scores; not publishable"
        if args.interim_layout
        else "final artifact validation; scoring coverage is reported separately",
        "final_publication_artifacts": not args.interim_layout,
        "gallery_path": str(args.gallery),
        "scores_path": str(args.scores),
        "scores_sha256": digest(args.scores),
        "checker_sha256": digest(Path(__file__)),
        "data_sha256": digest(args.gallery / "data.js"),
        "index_sha256": digest(args.gallery / "index.html"),
        "records": records,
        "static_pages": pages,
        "browser": browser,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(data),
                "static_score_cells": sum(p["score_cells"] for p in pages),
                "browser_checks": len(browser["checks"]),
                "status": "passed",
            }
        )
    )


if __name__ == "__main__":
    main()
