"""Execute the expanded staged experiment using the previous bounded request runner."""
import argparse,json,os,hashlib
from pathlib import Path
import runner
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'eligibility-optimization-2026-11-01';ROUTES=HERE.parent/'rehearsal-4000-2026-11-01'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['development','holdout']);a=ap.parse_args();freeze=read(HERE/'trial-freeze.json')
 for p,h in freeze['inputs'].items():
  if sha(p)!=h:raise RuntimeError('Frozen input changed: '+p)
 variants=read(HERE/'variants.json');past=read(HERE/'previous-prompts.json');cases=[c for c in read(HERE/'reference-manifest.json')['cases'] if c['partition']==a.stage]
 if a.stage=='development':rounds=['original','previous']+list(variants)
 else:
  winner=read(HERE/'winner.json');wf=read(HERE/'winner-freeze.json');assert sha(HERE/'winner.json')==wf['sha256'];rounds=['holdout-original','holdout-previous','holdout-selected']
 routes=read(ROUTES/'api-routes.json');prompts=read(ROUTES/'api-decision-prompts.json');key=os.environ.get('OPENROUTER_API_KEY');assert key
 runner.DATA.mkdir(parents=True,exist_ok=True);runner.init_ledger()
 for r in rounds:
  if r in ('original','holdout-original'):questions={m:None for m in runner.ROUTES}
  elif r in ('previous','holdout-previous'):questions={m:past[m]['question'] for m in runner.ROUTES}
  elif r=='holdout-selected':questions={m:(past[m]['question'] if winner['models'][m]['round']=='previous' else variants[winner['models'][m]['round']]) for m in runner.ROUTES}
  else:questions={m:variants[r] for m in runner.ROUTES}
  runner.run_round(r,[(m,c) for m in runner.ROUTES for c in cases],questions,routes,prompts,key)
if __name__=='__main__':main()
