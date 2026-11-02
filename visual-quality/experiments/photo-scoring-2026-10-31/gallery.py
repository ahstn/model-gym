#!/usr/bin/env python3
"""Build a portable side-by-side image and score gallery from the audited CSV."""
from canonical_scores import load_scores
import html
from pathlib import Path

ROOT=Path(__file__).resolve().parent
NAMES=[('Reference','reference'),('Luna','luna'),('DeepSeek','deepseek'),('Weighted ensemble','photo_ensemble'),('Haiku','haiku'),('Jev 1.14','jev'),('Liquid d1.1','liquid')]


def main():
    rows=load_scores()
    if len(rows)!=72 or len({r['case_id'] for r in rows})!=72: raise ValueError('Expected 72 distinct cases')
    rows.sort(key=lambda r:(float(r['reference_technical_quality']),r['case_id']))
    cards=[]
    for r in rows:
        cid=html.escape(r['case_id']); path=html.escape(r['image_path'],quote=True)
        values=''.join('<tr'+(' class="reference"' if key=='reference' else '')+'><th scope="row">'+html.escape(name)+'</th><td>'+html.escape(r[key+'_technical_quality'] or 'invalid')+'</td><td>'+html.escape(r[key+'_aesthetics'] or 'invalid')+'</td></tr>' for name,key in NAMES)
        partition='Known poster' if r['known_anchor']=='True' else ('Held out' if r['partition']=='heldout' else 'Prompt development')
        cards.append(f'<article><div class="caption"><strong>{cid}</strong><span>{partition}</span></div><a href="{path}"><img loading="lazy" src="{path}" alt="Original scored image {cid}"></a><table><thead><tr><th>Assessor</th><th>Tech</th><th>Aest</th></tr></thead><tbody>{values}</tbody></table></article>')
    output='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>72-image decision model comparison</title><style>
    *{box-sizing:border-box} body{margin:0;background:#eef1f6;color:#14283c;font:15px system-ui,sans-serif} header{padding:20px 26px;max-width:1200px} h1{font-size:25px;margin:0 0 8px} p{line-height:1.5;margin:6px 0} a{color:#15568e} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:16px;padding:0 20px 24px} article{background:#fff;border:1px solid #d3dde8;border-radius:9px;overflow:hidden}.caption{padding:12px 14px;display:flex;justify-content:space-between;align-items:center}.caption strong{font-size:20px}.caption span{font-size:12px;color:#536b80}img{display:block;width:100%;height:235px;object-fit:contain;background:#e4e9f0}table{width:100%;border-collapse:collapse;text-align:right;font-size:13px} th,td{padding:7px 10px;border-bottom:1px solid #e5e9ef}th:first-child{text-align:left} thead{color:#5c7183}tbody th{font-weight:550}.reference{background:#e5f0fb;color:#174d82}.note{color:#496174;font-size:13px}
    </style></head><body><header><h1>72-image scoring comparison</h1><p>Original images and two 1–5 scores. Lower scores indicate lower quality or the agreed non-photographic floor.</p><p class="note">Weighted ensemble: 37.5% Luna + 37.5% DeepSeek + 25% Liquid, independently per score. Original model calls and prompts are unchanged. Reference scores are assistant judgments. Prompt selection used development images; compare model rankings on held-out images. Scores use two decimals; underlying decision distributions and exclusion probabilities are retained in the experiment files. <a href="scores-72.csv">Download all scores</a> · <a href="README.md">Read report</a></p></header><main class="grid">'''+''.join(cards)+'''</main></body></html>'''
    (ROOT/'gallery.html').write_text(output)
    print('Built 72 cards and',len(rows)*len(NAMES)*2,'score cells')

if __name__=='__main__': main()
