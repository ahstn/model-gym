import csv,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent;source=root.parent/'photo-scoring-2026-10-31'
ids=['T0028','T0192','T0714','T0965','T0914','T0724','T2000']
prior={r['case_id']:r for r in csv.DictReader((source/'scores-72.csv').open())}
models=['reference','qrealign','luna','deepseek','average','haiku','jev','liquid']
rows=[]
for cid in ids:
 old=prior[cid];result={'case_id':cid,'role':'poster control' if cid=='T2000' else 'heldout photo','image_sha256':old['sha256']}
 for task,stem in [('technical_quality','quality'),('aesthetics','aesthetics')]:
  obs=json.loads((root/'space-results'/f'{cid}-{stem}.json').read_text())
  if obs.get('event')!='complete' or obs['image_sha256']!=old['sha256'] or abs(obs['score_1_5']-(1+4*obs['native_0_1']))>1e-9:raise ValueError('Invalid observation '+cid)
  result['qrealign_'+task]=obs['score_1_5'];result['qrealign_'+task+'_native_0_1']=obs['native_0_1']
  for model in models:
   if model!='qrealign':result[model+'_'+task]=float(old[model+'_'+task])
 rows.append(result)
with (root/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader()
 for row in rows:w.writerow({k:f'{v:.2f}' if isinstance(v,float) and not k.endswith('_native_0_1') else v for k,v in row.items()})
metrics={}
for model in models[1:]:
 d={}
 for task in ('technical_quality','aesthetics'):
  errs=[r[model+'_'+task]-r['reference_'+task] for r in rows if r['role']=='heldout photo']
  d[task]={'mae':sum(map(abs,errs))/len(errs),'bias':sum(errs)/len(errs)}
 d['mean_mae']=sum(d[t]['mae'] for t in ('technical_quality','aesthetics'))/2;metrics[model]=d
analysis={'eligible_photos':6,'poster_controls':1,'metrics':metrics,'source_revision':'9fb0e64200c1615a962551f002e11c1fc981ec04','scale':'1 + 4 * space score; no floor applied','caveat':'six selected images, assistant references, 768 px resized by Space, no independent exclusion gate'}
(root/'comparison-analysis.json').write_text(json.dumps(analysis,indent=2,sort_keys=True)+'\n')
labels={'reference':'Reference','qrealign':'Q-ReAlign Mini','luna':'Luna','deepseek':'DeepSeek','average':'Luna + DeepSeek','haiku':'Haiku','jev':'Jev','liquid':'Liquid'}
lines=['| Image | Reference | Q-ReAlign | Luna | DeepSeek | Mean | Haiku | Jev | Liquid |','|---|---|---|---|---|---|---|---|---|']
for r in rows:
 vals=[f'{r[m+"_technical_quality"]:.2f} / {r[m+"_aesthetics"]:.2f}' for m in models]
 lines.append('| '+r['case_id']+(' (poster)' if r['role']=='poster control' else '')+' | '+' | '.join(vals)+' |')
ranks=['| Model | Technical MAE | Aesthetic MAE | Mean MAE |','|---|---:|---:|---:|']
for name,d in sorted(metrics.items(),key=lambda item:item[1]['mean_mae']):ranks.append(f'| {labels[name]} | {d["technical_quality"]["mae"]:.3f} | {d["aesthetics"]["mae"]:.3f} | {d["mean_mae"]:.3f} |')
text='''# Q-ReAlign Mini hosted Space: seven-image comparison

The third-party [VOIDER/Q-ReAlign-0.8B Space](https://huggingface.co/spaces/VOIDER/Q-ReAlign-0.8B) scored the seven selected images with separate quality and aesthetics calls. Its source loads `q-future/Q-ReAlign-Mini-0.8B` and resizes the longest edge to 768 pixels. Its reported native [0,1] values are converted to [1,5] using `1 + 4 * score`; native values and responses are preserved in [comparison.csv](comparison.csv) and `space-results/`. Its reported values have four decimal places, so conversions inherit that rounding.

Each cell below shows **technical / aesthetic**. The first six photos are held out in the earlier comparison. T2000 is a known poster and is not part of the photo accuracy metrics. The Space has no photography eligibility gate; its poster scores are raw model outputs and are not replaced with the 1.00/1.00 dataset-policy floor.

'''+ '\n'.join(lines)+'''

The following six-photo errors are descriptive against prior assistant references; they are not human accuracy and are too small or selectively sampled for a reliable model ranking. Inputs are not fully matched: the Space resizes to 768 pixels while the earlier assessors saw up to 1,536-pixel views. No inference about general superiority follows from these six cases.

'''+ '\n'.join(ranks)+'''

The underlying scope rule, references, teachers and full corpus are documented in [the earlier report](../photo-scoring-2026-10-31/README.md). The Q-ReAlign Space model revision, image hashes, per-call status and elapsed times are stored with each response. This is Q-ReAlign Mini, not the preferred Q-Align OneAlign checkpoint. Results are experimental and unapproved for training.
'''
(root/'comparison.md').write_text(text);print(text)
