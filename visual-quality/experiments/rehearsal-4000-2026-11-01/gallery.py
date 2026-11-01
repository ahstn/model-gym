"""Build a browsable comparison from recorded rows and original image previews."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
GALLERY = HERE / "gallery"
MODELS = [
    ("reference", "Reference"),
    ("luna", "Luna"),
    ("deepseek", "DeepSeek"),
    ("luna_deepseek_mean", "Luna + DeepSeek"),
    ("liquid", "Liquid d1.1"),
    ("jev", "Jev 1.14"),
    ("qrealign_lite", "Q-ReAlign Lite"),
]
FIELDS = ("technical_score", "aesthetic_score", "latency_seconds", "price_usd", "review_elapsed_seconds")

STYLE = """
*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:#172d40;font-family:system-ui,sans-serif}
header{padding:22px 26px;background:#eef2f6}h1{margin:0 0 6px;font-size:25px}p{margin:7px 0;color:#53697c;font-size:13px}
nav{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-top:14px}select,input,button{padding:8px 10px;border:1px solid #b6c6d4;border-radius:5px;background:white;color:#172d40}
main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;padding:0 20px 24px}.card{background:#fff;border:1px solid #bfd0de;border-radius:7px;overflow:hidden}
.caption{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;font-weight:700;font-size:14px}.partition{font-size:10px;letter-spacing:.06em;color:#71889b}
.photo{width:100%;height:235px;display:block;object-fit:contain;background:#e2e8ef}
table{border-collapse:collapse;width:calc(100% - 16px);margin:4px 8px 8px;font-size:12px}th{text-align:right;color:#71889b;font-size:10px;font-weight:500}th:first-child,td:first-child{text-align:left}td{text-align:right;padding:4px 3px;border-bottom:1px solid #e6edf3;font-variant-numeric:tabular-nums}td:first-child{font-weight:600}tr.ref{background:#e4f0fb}tr.mean{background:#f3f6f9}
.source{font-size:10px;padding:0 12px 9px;color:#71889b}.source a{color:#456b8b}.empty{color:#8999a7}.money{font-size:10px}footer{padding:0 24px 20px;font-size:12px;color:#53697c}
details{padding:0 12px 10px;font-size:11px;color:#53697c}summary{cursor:pointer}details p{font-size:11px;line-height:1.4}details b{color:#172d40}
@media(max-width:1000px){main{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:620px){main{grid-template-columns:1fr}}@media print{nav{display:none}.card{break-inside:avoid}}
"""


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows() -> dict:
    grouped = defaultdict(dict)
    with (HERE / "scores-long.csv").open(newline="") as stream:
        for original in csv.DictReader(stream):
            row = dict(original)
            for field in FIELDS:
                value = row.get(field)
                row[field] = float(value) if value not in (None, "", "null") else None
            row["evidence"] = json.loads(row.get("evidence") or "{}")
            key = row["model"] + ":" + row["arm"]
            if key in grouped[row["case_id"]]:
                raise ValueError("Duplicate gallery row")
            grouped[row["case_id"]][key] = row
    return grouped


def make_data(cases: list[dict], rows: dict) -> list[dict]:
    previews = GALLERY / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    neighbours_path = HERE / "selected-neighbours.json"
    neighbours = json.loads(neighbours_path.read_text())["neighbours"] if neighbours_path.exists() else {}
    labels_path = HERE / "cluster-labels.json"
    labels = json.loads(labels_path.read_text())["labels"] if labels_path.exists() else {}
    families = defaultdict(list)
    for case in cases:
        families[case["family_id"]].append(case["case_id"])
    data = []
    for case in cases:
        source = Path(case["image_path"])
        if digest(source) != case["view_sha256"]:
            raise ValueError("Gallery original changed: " + case["case_id"])
        target = previews / (case["case_id"] + ".jpg")
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((600, 400), Image.Resampling.LANCZOS)
            image.save(target, quality=90, optimize=True)
        data.append(
            {
                "id": case["case_id"],
                "partition": case["benchmark_partition"],
                "preview": "previews/" + target.name,
                "original": source.as_uri(),
                "view_sha256": case["view_sha256"],
                "family": case["family_id"],
                "cluster": str(case.get("rehearsal_content_cluster", "unknown")),
                "cluster_label": labels.get(str(case.get("rehearsal_content_cluster")), ""),
                "neighbours": neighbours.get(case["case_id"], []),
                "siblings": [
                    identity for identity in families[case["family_id"]] if identity != case["case_id"]
                ],
                "scores": rows.get(case["case_id"], {}),
            }
        )
    return data


def numeric(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def money(value):
    return "—" if value is None else "$0.00" if value == 0 else f"${value:.6f}"


def static_card(case: dict) -> str:
    cells = []
    for model, title in MODELS:
        row = case["scores"].get(model + (":reference" if model == "reference" else ":baseline"), {})
        kind = "ref" if model == "reference" else "mean" if model == "luna_deepseek_mean" else ""
        cells.append(
            f'<tr class="{kind}"><td>{title}</td><td>{numeric(row.get("technical_score"))}</td>'
            f"<td>{numeric(row.get('aesthetic_score'))}</td><td>{numeric(row.get('latency_seconds'))}</td>"
            f'<td class="money">{money(row.get("price_usd"))}</td></tr>'
        )
    return (
        f'<article class="card" data-case-id="{case["id"]}"><div class="caption">{case["id"]}'
        f'<span class="partition">{html.escape(case["partition"].upper())}</span></div>'
        f'<img class="photo" src="{case["preview"]}" alt="{case["id"]}">'
        "<table><thead><tr><th>Assessor</th><th>Tech</th><th>Aest</th><th>Seconds</th><th>Price</th>"
        "</tr></thead><tbody>" + "".join(cells) + "</tbody></table>"
        f'<div class="source"><a href="{case["original"]}">Open exact delivered view</a></div></article>'
    )


def choose_pages(data: list[dict]) -> list[dict]:
    """Cover the reference score range; use no grader score to select displayed cases."""
    available = [
        case
        for case in data
        if case["scores"].get("reference:reference", {}).get("technical_score") is not None
    ]
    available.sort(key=lambda case: (case["scores"]["reference:reference"]["technical_score"], case["id"]))
    if len(available) < 72:
        raise ValueError("Need at least72 scored references before making comparison pages")
    indices = [round(index * (len(available) - 1) / 71) for index in range(72)]
    return [available[index] for index in indices]


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    rows = read_rows()
    data = make_data(manifest["cases"], rows)
    (GALLERY / "data.js").write_text("const CASES=" + json.dumps(data, ensure_ascii=False) + ";\n")
    (GALLERY / "style.css").write_text(STYLE)
    selected = choose_pages(data)
    selection = {
        "method": "72 equally spaced ranks in reference technical score, case ID breaks ties; no student scores",
        "case_ids": [case["id"] for case in selected],
        "references_sha256": digest(HERE / "references.jsonl"),
    }
    (GALLERY / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    for page in range(4):
        title = f"Photo scoring rehearsal · Page {page + 1} / 4"
        document = (
            '<!doctype html><html><head><meta charset="utf-8"><title>'
            + title
            + '</title><link rel="stylesheet" href="style.css"></head><body><header><h1>'
            + title
            + "</h1><p>Baseline scores · Reference = assistant assessment · HTTP end-to-end seconds</p>"
            + "<p>Lite and Reference prices are assigned $0.00; Runpod infrastructure is accounted separately. "
            + "Mean and Reference have no HTTP latency. Lite uses native scores without a scope gate.</p></header><main>"
            + "".join(static_card(case) for case in selected[page * 18 : (page + 1) * 18])
            + "</main><footer>Previews preserve framing; inspect the exact delivered view for fine defects. "
            + "Selection spans reference technical scores and is not a population-weighted sample.</footer></body></html>"
        )
        (GALLERY / f"comparison-page-{page + 1}.html").write_text(document)
    models = json.dumps(MODELS)
    document = """<!doctype html><html><head><meta charset="utf-8"><title>4,000-image scoring rehearsal</title>
<link rel="stylesheet" href="style.css"></head><body><header><h1>4,000-image scoring rehearsal</h1>
<p>Five scorers · separate technical and aesthetic scores · 500 independent assistant references</p>
<p>Seconds measure HTTP request/response time. Reference has no API latency; Lite and Reference prices are assigned $0.00.
The mean has no independent request or charge. Raw Lite has no scope gate. Prices retain six decimal places.</p>
<nav><select id="partition"><option value="benchmark">500 reference benchmark</option><option value="all">All4,000</option>
<option value="representative">Representative evaluation</option><option value="challenge">Challenge evaluation</option>
<option value="development">Development</option><option value="rehearsal">Other rehearsal</option></select>
<select id="arm"><option value="baseline">Baseline</option><option value="own_stats">Own-image stats</option><option value="swapped_stats">Swapped-image stats</option></select>
<select id="cluster"><option value="all">All content clusters</option></select>
<select id="sort"><option value="id">Image ID</option><option value="technical_score">Reference technical score</option><option value="aesthetic_score">Reference aesthetic score</option></select>
<input id="query" placeholder="Find image ID" aria-label="Find image ID"><button id="prev">Previous</button><button id="next">Next</button><span id="count"></span></nav>
</header><main id="grid"></main><footer>Stats arms apply to Luna and DeepSeek; other models retain their baseline. Missing observations are shown as dashes.
Related links use EmbeddingGemma 2 content similarity, which does not establish equal quality or duplication. Controlled variants inherit their parent's vector.
Previews are resized; use the original link to inspect fine defects. No labels are approved for training.</footer>
<script src="data.js"></script><script>
const MODELS=MODEL_DATA;let page=0;const n=24;
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=x=>x==null?'—':Number(x).toFixed(2);const money=x=>x==null?'—':Number(x)===0?'$0.00':'$'+Number(x).toFixed(6);
function notes(c,a){return '<details><summary>Assessment notes</summary>'+MODELS.filter(([m])=>m!=='luna_deepseek_mean').map(([m,label])=>{const arm=m==='reference'?'reference':['luna','deepseek'].includes(m)?a:'baseline',r=c.scores[m+':'+arm]??{},e=r.evidence??{};return `<p><b>${esc(label)}</b>: ${esc(r.status??'missing')}<br>Technical: ${esc(e.technical_quality??'No text evidence returned')}<br>Aesthetic: ${esc(e.aesthetics??'No text evidence returned')}</p>`}).join('')+'</details>'}
function render(){const p=document.getElementById('partition').value,a=document.getElementById('arm').value,s=document.getElementById('sort').value,q=document.getElementById('query').value.toLowerCase(),k=document.getElementById('cluster').value;
let rows=CASES.filter(c=>(p==='all'||p==='benchmark'&&c.partition!=='rehearsal'||c.partition===p)&&(k==='all'||c.cluster===k)&&c.id.toLowerCase().includes(q));
rows.sort((x,y)=>s==='id'?x.id.localeCompare(y.id):((x.scores['reference:reference']?.[s]??99)-(y.scores['reference:reference']?.[s]??99)||x.id.localeCompare(y.id)));
page=Math.max(0,Math.min(page,Math.ceil(rows.length/n)-1));document.getElementById('count').textContent=`${rows.length} images · page ${page+1}/${Math.max(1,Math.ceil(rows.length/n))}`;
document.getElementById('grid').innerHTML=rows.slice(page*n,(page+1)*n).map(c=>`<article class="card" data-case-id="${esc(c.id)}"><div class="caption">${esc(c.id)}<span class="partition">${esc(c.partition.toUpperCase())}</span></div><img class="photo" loading="lazy" src="${esc(c.preview)}" alt="${esc(c.id)}"><table><thead><tr><th>Assessor</th><th>Tech</th><th>Aest</th><th>Seconds</th><th>Price</th></tr></thead><tbody>${MODELS.map(([m,label])=>{const arm=m==='reference'?'reference':['luna','deepseek','luna_deepseek_mean'].includes(m)?a:'baseline',r=c.scores[m+':'+arm]??{};return `<tr class="${m==='reference'?'ref':m==='luna_deepseek_mean'?'mean':''}" title="${esc(r.status??'missing')}"><td>${esc(label)}</td><td>${num(r.technical_score)}</td><td>${num(r.aesthetic_score)}</td><td>${num(r.latency_seconds)}</td><td class="money">${money(r.price_usd)}</td></tr>`}).join('')}</tbody></table><div class="source"><a href="${esc(c.original)}">Open exact delivered view</a> · cluster ${esc(c.cluster)} ${esc(c.cluster_label)}</div><div class="source">Related: ${c.neighbours.map(v=>`<a href="?case=${encodeURIComponent(v.case_id)}">${esc(v.case_id)}</a> (${Number(v.cosine).toFixed(3)})`).join(' · ')||'unavailable'}${c.siblings.length?'<br>Same family: '+c.siblings.map(id=>`<a href="?case=${encodeURIComponent(id)}">${esc(id)}</a>`).join(' · '):''}</div>${notes(c,a)}</article>`).join('');}
const clusters=[...new Set(CASES.map(c=>c.cluster))].sort((a,b)=>Number(a)-Number(b));
for(const k of clusters){const option=document.createElement('option');option.value=k;const label=CASES.find(c=>c.cluster===k)?.cluster_label;option.textContent=`Cluster ${k}${label?' · '+label:''}`;document.getElementById('cluster').append(option)}
const initial=new URLSearchParams(location.search).get('case');if(initial){document.getElementById('query').value=initial;document.getElementById('partition').value='all'}
for(const id of ['partition','arm','cluster','sort','query'])document.getElementById(id).addEventListener('input',()=>{page=0;render()});
document.getElementById('prev').onclick=()=>{page--;render()};document.getElementById('next').onclick=()=>{page++;render()};render();
</script></body></html>""".replace("MODEL_DATA", models)
    (GALLERY / "index.html").write_text(document)
    print(json.dumps({"gallery_cases": len(data), "comparison_cases": len(selected), "output": str(GALLERY)}))


if __name__ == "__main__":
    main()
