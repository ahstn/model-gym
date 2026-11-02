import csv,html
from pathlib import Path
H=Path(__file__).resolve().parent
names=[('Reference','reference'),('Q-ReAlign Lite','lite'),('Weighted ensemble','photo_ensemble'),('Luna','luna'),('DeepSeek','deepseek'),('Haiku','haiku'),('Jev','jev'),('Liquid','liquid')]
rows=list(csv.DictReader((H/'scores.csv').open()));assert len(rows)==72
rows.sort(key=lambda r:(float(r['reference_technical_quality']),r['case_id']))
cards=[]
for r in rows:
 cid=html.escape(r['case_id']);kind='Poster control' if r['known_anchor']=='True' else r['partition'].title()
 vals=''.join('<tr><th>'+html.escape(n)+'</th><td>'+html.escape(r[k+'_technical_quality'] or 'invalid')+'</td><td>'+html.escape(r[k+'_aesthetics'] or 'invalid')+'</td></tr>' for n,k in names)
 path='../photo-scoring-2026-10-31/images/'+cid+'.png'
 cards.append(f'<article><header><b>{cid}</b><span>{kind}</span></header><a href="{path}"><img loading="lazy" src="{path}" alt="{cid}"></a><table><thead><tr><th>Assessor</th><th>Technical</th><th>Aesthetic</th></tr></thead><tbody>{vals}</tbody></table></article>')
s='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Q-ReAlign Lite comparison</title><style>body{margin:0;background:#eff3f7;color:#182c41;font:15px system-ui}main{padding:16px;display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}body>header{padding:20px}h1{margin:0;font-size:24px}article{border:1px solid #d3dce6;background:white;border-radius:8px;overflow:hidden}article>header{padding:12px;display:flex;justify-content:space-between}article>header b{font-size:19px}article>header span{color:#607789}img{width:100%;height:220px;object-fit:contain;background:#e1e7ee}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px 10px;border-bottom:1px solid #e9edf1;text-align:right}th:first-child{text-align:left}thead{color:#5b7084}tbody th{font-weight:500}</style></head><body><header><h1>Q-ReAlign Lite: 72-image comparison</h1><p>Ensemble: 37.5% Luna + 37.5% DeepSeek + 25% Liquid independently. Technical / aesthetic scores; Lite is native quality scoring without an eligibility floor. <a href="scores.csv">Download scores</a> · <a href="README.md">Read report</a></p></header><main>'''+''.join(cards)+'''</main></body></html>'''
(H/'gallery.html').write_text(s);print('Gallery: 72 cards')
