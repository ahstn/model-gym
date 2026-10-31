"""Run the 72 frozen images against the authenticated Lite scorer, without automatic retries."""
import base64,csv,hashlib,json,os,sys,time
from pathlib import Path
import requests
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent/'photo-scoring-2026-10-31'
url='https://j8qvat8tmzyhgr-8888.proxy.runpod.net';token=Path('/tmp/qrealign-lite-token').read_text().strip();headers={'Authorization':'Bearer '+token}
health=requests.get(url+'/health',headers=headers,timeout=20);health.raise_for_status();identity=health.json()
if identity['revision']!='06add46a6bc4a7dfcf0a59a2fb1ed15a6e3d30b7':raise ValueError('wrong model')
(HERE/'server-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
rows=list(csv.DictReader((SOURCE/'scores-72.csv').open()));assert len(rows)==72 and len({x['case_id'] for x in rows})==72
folder=HERE/'results';folder.mkdir(exist_ok=True)
for index,row in enumerate(rows,1):
 cid=row['case_id'];dest=folder/(cid+'.json')
 if dest.exists():
  prior=json.loads(dest.read_text());
  if 'error' in prior:raise RuntimeError('Prior failure requires inspection')
  continue
 image=(SOURCE/'images'/f'{cid}.png').read_bytes();digest=hashlib.sha256(image).hexdigest()
 if digest!=row['sha256']:raise ValueError('image hash mismatch')
 record={'case_id':cid,'view_sha256':digest,'model':identity['model'],'revision':identity['revision']};start=time.monotonic()
 try:
  response=requests.post(url+'/score',headers=headers,json={'image':base64.b64encode(image).decode()},timeout=(20,180));response.raise_for_status();record['response']=response.json()
  for task in ['quality','aesthetics']:
   d=record['response'][task];p=d['probabilities'];s=d['native_0_1']
   if len(p)!=5 or abs(sum(p)-1)>0.001 or not 0<=s<=1 or abs(d['score_1_5']-(1+4*s))>1e-6:raise ValueError('invalid score')
 except Exception as exc:record['error']=type(exc).__name__+': '+str(exc)[:300]
 record['elapsed_seconds']=round(time.monotonic()-start,3)
 with dest.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(index,cid,record.get('error') or 'ok',record['elapsed_seconds'],flush=True)
 if 'error' in record:raise RuntimeError('Stopped after failed request')
