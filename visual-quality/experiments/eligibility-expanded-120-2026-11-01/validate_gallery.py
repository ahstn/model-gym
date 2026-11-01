"""Check live gallery filters, previews, exact view URLs and response joins."""

from __future__ import annotations

import hashlib
import html
import json
import subprocess
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import analyze_expanded as analysis

HERE = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8765/"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(p, base=BASE):
    with opener.open(base + p, timeout=25) as r:
        return r.read()


def main():
    document = get("gallery/expanded-eligibility.html").decode()
    assert get("gallery/expanded-eligibility.html", "http://192.168.1.37:8765/").decode() == document
    assert b"expanded-eligibility.html" in get("gallery/index.html")
    assert get("gallery/eligibility-trial.html")
    assert get("gallery/expanded-eligibility-report.md")
    for p in (
        "gallery/../serve_gallery.py",
        "expanded-originals/../serve_gallery.py",
        "execution-records/index.json",
    ):
        try:
            get(p)
            raise AssertionError("Access allowed: " + p)
        except urllib.error.HTTPError as e:
            assert e.code == 404
    raw = get("gallery/expanded-eligibility.js").decode()
    assert raw.startswith("const TRIAL=") and raw.endswith(";\n")
    j = json.loads(raw[len("const TRIAL=") : -2])
    cases = {x["case_id"]: x for x in j["cases"]}
    assert len(cases) == 240
    results = analysis.read(HERE / "results.json")
    winner = analysis.read(HERE / "winner.json")
    manifest = analysis.read(HERE / "reference-manifest.json")
    rows = list(results["case_rows"])
    dev = [x for x in manifest["cases"] if x["partition"] == "development"]
    for model in analysis.MODELS:
        for arm, rid in [
            ("original", "original"),
            ("previous", "previous"),
            ("selected", winner["models"][model]["round"]),
        ]:
            for row in analysis.data(rid, model, dev):
                row["display_arm"] = arm
                rows.append(row)
    for r in rows:
        v = cases[r["case_id"]]["results"][r["model"]][r["display_arm"]]
        for key in (
            "status",
            "technical_score",
            "aesthetic_score",
            "technical_native",
            "aesthetic_native",
            "exclusion_probability",
            "scope_floor_applied",
            "latency_seconds",
            "price_usd",
        ):
            assert v[key] == r.get(key), (r["case_id"], r["model"], key)

    def original(case):
        return case["case_id"], hashlib.sha256(get(case["original"].lstrip("/"))).hexdigest() == case[
            "view_sha256"
        ]

    with ThreadPoolExecutor(max_workers=10) as pool:
        checks = list(pool.map(original, cases.values()))
    assert all(ok for _, ok in checks), [id for id, ok in checks if not ok]
    harness = r"""<script>
(async()=>{let out={checks:[]};const ck=(v,n)=>{if(!v)throw Error(n);out.checks.push(n)};try{
const count=()=>document.querySelectorAll('#grid .card').length;const ch=(id,v)=>{let x=document.getElementById(id);x.value=v;x.dispatchEvent(new Event('change'))};
ck(count()===120,'120 heldout views');ch('scope','eligible');ck(count()===60,'60 eligible');ch('scope','excluded');ck(count()===60,'60 excluded');ch('scope','all');
for(const f of ['remaining','improved','regressed','missing'])for(const m of ['liquid','jev']){ch('model',m);ch('filter',f);let expected=TRIAL.cases.filter(c=>c.partition==='heldout'&&matches(c,f,[m])).length;ck(count()===expected,`${m} ${f} filter`)}
ch('model','both');ch('filter','all');ch('partition','development');ck(count()===120,'120 development views');ch('partition','all');ck(count()===240,'240 views');
let items=TRIAL.cases.map(c=>({id:c.case_id,url:'expanded-previews/'+c.case_id+'.jpg'})),fail=[],cursor=0;await Promise.all(Array.from({length:12},async()=>{while(cursor<items.length){let a=items[cursor++];await new Promise(resolve=>{let im=new Image();im.onload=()=>{if(!im.naturalWidth)fail.push(a.id);resolve()};im.onerror=()=>{fail.push(a.id);resolve()};im.src=a.url})}}));ck(fail.length===0,'240 preview images loaded');out.status='passed'}catch(e){out.status='failed';out.error=String(e)}let pre=document.createElement('pre');pre.id='result';pre.textContent=JSON.stringify(out);document.body.append(pre)})();</script>"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test = document.replace("<head>", '<head><base href="' + BASE + 'gallery/">', 1).replace(
            "</body>", harness + "</body>", 1
        )
        (p / "check.html").write_text(test)
        cmd = [
            "/home/ahstn/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome",
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--disable-crash-reporter",
            "--allow-file-access-from-files",
            f"--user-data-dir={p}/profile",
            "--dump-dom",
            "--virtual-time-budget=30000",
            (p / "check.html").as_uri(),
        ]
        text = subprocess.run(cmd, capture_output=True, text=True, timeout=55, check=True).stdout
        proof = json.loads(html.unescape(text.split('<pre id="result">', 1)[1].split("</pre>", 1)[0]))
        assert proof["status"] == "passed", proof
    proof["checks"] += [
        "1440 displayed rows match request records",
        "240 exact served originals match scored view hashes",
        "local and LAN gallery agree",
        "server blocks unrelated paths",
        "prior gallery still available",
    ]
    (HERE / "gallery/validation.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
