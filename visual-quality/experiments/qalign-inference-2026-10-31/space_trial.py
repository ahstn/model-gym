import requests,json,time,hashlib,csv,sys
from pathlib import Path
base='https://voider-q-realign-0-8b.hf.space/gradio_api'
root=Path(__file__).resolve().parent
source=root.parent/'photo-scoring-2026-10-31'
ids=['T0028','T0192','T0714','T0965','T0914','T0724','T2000']
rows={r['case_id']:r for r in csv.DictReader((source/'scores-72.csv').open())}
results=root/'space-results'; results.mkdir(exist_ok=True)
for cid in ids:
 p=source/'images'/f'{cid}.png'; digest=hashlib.sha256(p.read_bytes()).hexdigest()
 if digest!=rows[cid]['sha256']:raise ValueError('Image hash mismatch '+cid)
 for task in ['Quality (IQA)','Aesthetics (IAA)']:
  dest=results/(cid+('-quality.json' if task.startswith('Quality') else '-aesthetics.json'))
  if dest.exists():
   old=json.loads(dest.read_text())
   if 'error' in old: raise RuntimeError('Previous attempt failed; inspect '+str(dest))
   continue
  record={'case_id':cid,'task':task,'image_sha256':digest,'space_revision':'9fb0e64200c1615a962551f002e11c1fc981ec04','space':'VOIDER/Q-ReAlign-0.8B'};start=time.monotonic()
  try:
   with p.open('rb') as f: r=requests.post(base+'/upload',files={'files':(p.name,f,'image/png')},timeout=30)
   r.raise_for_status(); upload=r.json()[0]
   r=requests.post(base+'/call/score_image',json={'data':[{'path':upload,'meta':{'_type':'gradio.FileData'}},task]},timeout=30);r.raise_for_status()
   record['event_id']=r.json()['event_id']
   with requests.get(base+'/call/score_image/'+record['event_id'],timeout=(20,240),stream=True) as stream:
    stream.raise_for_status(); event=None
    for line in stream.iter_lines(decode_unicode=True):
     if line.startswith('event: '):event=line[7:]
     if line.startswith('data: ') and event in ('complete','error'):
      record['event']=event;record['data']=json.loads(line[6:]);break
   if record.get('event')!='complete':raise ValueError('Inference did not complete')
   native=float(record['data'][0]);
   if not 0<=native<=1:raise ValueError('Score out of range')
   record['native_0_1']=native;record['score_1_5']=1+4*native
  except Exception as e:record['error']=type(e).__name__+': '+str(e)[:300]
  record['elapsed_seconds']=round(time.monotonic()-start,2)
  with dest.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
  print(json.dumps(record),flush=True)
  if 'error' in record:raise RuntimeError('Stopped after failure; response preserved')
