"""Compare saved Lite scores with earlier 72-image references and model scores."""
import csv,json,hashlib,statistics,random,sys
from pathlib import Path
H=Path(__file__).resolve().parent; S=H.parent/'photo-scoring-2026-10-31'
sys.path.insert(0, str(S))
from canonical_scores import ARCHIVED, load_scores
source=load_scores();out=[];latencies=[]
for r in source:
 p=H/'results'/(r['case_id']+'.json');o=json.loads(p.read_text())
 if 'error' in o or o['view_sha256']!=r['sha256']:raise ValueError('invalid or mismatched result')
 latencies.append({'case_id':r['case_id'],'end_to_end_seconds':o['elapsed_seconds'],'server_seconds':o['response']['seconds'],'input_tokens_quality':o['response']['quality']['input_tokens'],'input_tokens_aesthetics':o['response']['aesthetics']['input_tokens'],'peak_vram_bytes':o['response']['peak_vram']})
 d={'case_id':r['case_id'],'partition':r['partition'],'reference_eligibility':r['reference_eligibility'],'known_anchor':r['known_anchor'],'view_sha256':r['sha256']}
 d['end_to_end_seconds']=o['elapsed_seconds'];d['server_seconds']=o['response']['seconds'];d['quality_input_tokens']=o['response']['quality']['input_tokens'];d['aesthetics_input_tokens']=o['response']['aesthetics']['input_tokens']
 for task,native in [('technical_quality','quality'),('aesthetics','aesthetics')]:
  d['lite_'+task]=o['response'][native]['score_1_5']
  for name in ('reference','luna','deepseek','photo_ensemble','haiku','jev','liquid'):
   key=name+'_'+task+('_native' if name=='photo_ensemble' else '_policy_native' if name=='liquid' else '')
   d[name+'_'+task]=float(r[key]) if r[key] else None
  d['photo_ensemble_'+task+'_native']=r['photo_ensemble_'+task+'_native']
  d['liquid_'+task+'_policy_native']=r['liquid_'+task+'_policy_native']
  d[ARCHIVED+'_'+task]=r[ARCHIVED+'_'+task]
 out.append(d)
models=['lite','luna','deepseek','photo_ensemble','haiku','jev','liquid'];tasks=['technical_quality','aesthetics'];results={}
for name,items in [('fresh71',[x for x in out if x['known_anchor']=='False']),('heldout48',[x for x in out if x['partition']=='heldout']),('eligible68',[x for x in out if x['reference_eligibility']=='eligible'])]:
 common=[r for r in items if all(r[m+'_'+t] is not None for m in models for t in tasks)];results[name]={}
 for m in models:
  vals={t:[r[m+'_'+t]-r['reference_'+t] for r in common] for t in tasks}
  results[name][m]={'n':len(common),'technical_mae':statistics.mean(map(abs,vals[tasks[0]])),'aesthetic_mae':statistics.mean(map(abs,vals[tasks[1]])),'technical_bias':statistics.mean(vals[tasks[0]]),'aesthetic_bias':statistics.mean(vals[tasks[1]])}
  x=results[name][m];x['mean_mae']=(x['technical_mae']+x['aesthetic_mae'])/2
with (H/'scores.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=out[0].keys(),lineterminator='\n');w.writeheader()
 for r in out:w.writerow({k:f'{v:.2f}' if isinstance(v,float) else v for k,v in r.items()})
common=[r for r in out if r['partition']=='heldout' and all(r[m+'_'+t] is not None for m in models for t in tasks)]
diffs=[sum(abs(r['lite_'+t]-r['reference_'+t])-abs(r['photo_ensemble_'+t]-r['reference_'+t]) for t in tasks)/2 for r in common]
rng=random.Random(20261031072);samples=sorted(statistics.mean(rng.choices(diffs,k=len(diffs))) for _ in range(5000))
results['paired_lite_minus_photo_ensemble']={'n':len(diffs),'mean_difference':statistics.mean(diffs),'bootstrap_95_interval':[samples[124],samples[4874]],'seed':20261031072,'resamples':5000}
def latency_summary(key):
 vals=sorted(x[key] for x in latencies);return {'n':len(vals),'min':min(vals),'median':statistics.median(vals),'p95_nearest_rank':vals[(95*len(vals)+99)//100-1],'max':max(vals),'mean':statistics.mean(vals)}
results['latency']={'end_to_end_seconds':latency_summary('end_to_end_seconds'),'server_seconds':latency_summary('server_seconds'),'peak_allocated_vram_bytes':max(x['peak_vram_bytes'] for x in latencies),'per_case':latencies,'batch_size':1,'tasks_per_request':2,'concurrency':1,'scope':'Post-load warm and first-request times. End-to-end includes network and upload; server timing includes two processor/forward passes, excludes decode and server startup.'}
results['ensemble_provenance']={'source':'../photo-scoring-2026-10-31/canonical-ensemble-provenance.json','source_sha256':hashlib.sha256((S/'canonical-ensemble-provenance.json').read_bytes()).hexdigest(),'model':'photo_ensemble','precision':'native component and ensemble scores; display rounding only','latency_kind':'derived_no_http_request','price_basis':'no independent charge; original component costs unchanged'}
(H/'analysis.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
lines=['| Model | Tech MAE | Aesthetic MAE | Mean MAE | Tech bias |','|---|---:|---:|---:|---:|']
for model,x in sorted(results['heldout48'].items(),key=lambda p:p[1]['mean_mae']):lines.append(f'| {"Weighted ensemble" if model=="photo_ensemble" else model} | {x["technical_mae"]:.3f} | {x["aesthetic_mae"]:.3f} | {x["mean_mae"]:.3f} | {x["technical_bias"]:+.3f} |')
poster=next(x for x in out if x['known_anchor']=='True')
paired=results['paired_lite_minus_photo_ensemble'];lo,hi=paired['bootstrap_95_interval']
lat=results['latency'];e=lat['end_to_end_seconds'];sv=lat['server_seconds'];startup=json.loads((H/'startup.json').read_text());cost=json.loads((H/'cost-estimate.json').read_text())
mini_path=H.parent/'qalign-inference-2026-10-31/comparison.csv'
mini={r['case_id']:r for r in csv.DictReader(mini_path.open())};six=[r for r in out if r['case_id'] in mini and r['known_anchor']=='False']
assert len(six)==6
lite_six={t:statistics.mean(abs(r['lite_'+t]-r['reference_'+t]) for r in six) for t in tasks}
mini_six={t:statistics.mean(abs(float(mini[r['case_id']]['qrealign_'+t])-r['reference_'+t]) for r in six) for t in tasks}
text='''# Q-ReAlign Lite: 72-image Runpod trial

[All per-image scores](scores.csv) · [Analysis](analysis.json) · [Prior 72-image gallery](../photo-scoring-2026-10-31/gallery.html)

Static image pages with both scores and per-image Lite end-to-end latency: [page 1](comparison-latency-page-1.png) · [page 2](comparison-latency-page-2.png) · [page 3](comparison-latency-page-3.png) · [page 4](comparison-latency-page-4.png). Each page contains 18 original images. The other models' consolidated records do not contain comparable per-image latency measurements.

Q-ReAlign Lite 4B was evaluated on the same 72 original delivered views on a Runpod 24 GB RTX 4090. Checkpoint `q-future/Q-ReAlign-Lite-4B` revision `06add46a6bc4a7dfcf0a59a2fb1ed15a6e3d30b7`; raw responses, level probabilities, input token counts, latency and image SHA-256 are retained under `results/`. Image processor uses model defaults; no external resizing was applied. The GPU model's native five-word weighted 0–1 score is mapped to 1–5 by `1+4*score`. Scores are raw: no eligibility classifier or floor is applied.

The table compares the 48 previously held-out images on common valid cases, against prior assistant references. Error is mean absolute error on the 1–5 scale; positive bias means higher than the reference. These are not human gold and the sample is small. Additional results for all 71 fresh images and 68 reference-eligible images are in `analysis.json`. The canonical weights were adopted after this study, so this is a retrospective comparison.

'''+ '\n'.join(lines)+f'''

On {paired['n']} paired held-out cases, Lite's combined error minus the canonical weighted ensemble is {paired['mean_difference']:+.3f}; a 5,000-resample paired bootstrap interval is [{lo:+.3f}, {hi:+.3f}]. This measures sampling variability conditional on the fixed assistant references and selected images; it is not human accuracy uncertainty.

Known poster T2000 receives native technical {poster['lite_technical_quality']:.2f} and aesthetics {poster['lite_aesthetics']:.2f}, while the agreed photography-eligibility reference is 1.00 / 1.00. The model has no eligibility gate; a separate validated classifier is required for the policy. This control is excluded from the fresh and held-out metrics, and the separate uncertain boundary cases remain in the held-out results as in the prior experiment.

On the six photos previously assessed by the hosted Mini model, Lite's technical/aesthetic MAE is {lite_six[tasks[0]]:.3f} / {lite_six[tasks[1]]:.3f}; Mini's is {mini_six[tasks[0]]:.3f} / {mini_six[tasks[1]]:.3f} using the previously published two-decimal scores. The Mini Space applied a 768-pixel cap, while Lite used its default processor on the supplied views. This tiny targeted subset and differing preprocessing cannot establish a general model difference. [Mini comparison](../qalign-inference-2026-10-31/comparison.md).

Metrics use unrounded Lite outputs, native Liquid policy scores and the shared canonical ensemble; the remaining earlier models and reference values retain their original schema precision. The ensemble is 37.5% Luna + 37.5% DeepSeek + 25% Liquid independently for each outcome, with all three required. It is recomputed from the original observations without new calls, latency or independent charges. Historical 50/50 columns are retained solely for provenance and are excluded from current comparisons. The gallery and primary CSV score columns display two decimals; explicit native columns retain the ensemble and Liquid arithmetic inputs. The earlier Luna/DeepSeek/Haiku/Jev/Liquid assessments and references were not rerun. Their prompts, eligibility handling, reasoning and input processors differ from this scalar Q-ReAlign model; this is a useful task comparison, not a controlled architecture-only ablation. Assistant references remain subjective and no labels are approved for training.

The two question phrasings and their answer stems are present in the author's [task prompt library](https://github.com/Q-Future/Q-ReAlign/blob/main/qalign/prompts.py). The scorer follows the model card's direct Transformers example; this was not a separate run of the author's ms-swift evaluation pipeline. The five space-prefixed level labels were each verified as a distinct single token before the server started.

## Latency

Each request processed one image and both dimensions sequentially. End-to-end time includes upload and proxy/network overhead. Server processing starts after image decode and includes two independent preprocessing and forward passes. Startup and checkpoint download are excluded. The first request is included; the run used one request at a time. These are experimental single-stream timings, not a production throughput benchmark.

The initial cold startup took approximately {startup['approx_seconds']:.1f} seconds from container creation to the model-ready log, including dependency installation and checkpoint download; it is recorded separately in `startup.json` and was not part of the per-image timing.

| Timing | Median | P95 (nearest rank) | Maximum |
|---|---:|---:|---:|
| End to end | {e['median']:.2f} s | {e['p95_nearest_rank']:.2f} s | {e['max']:.2f} s |
| Server processing | {sv['median']:.2f} s | {sv['p95_nearest_rank']:.2f} s | {sv['max']:.2f} s |

Peak reported allocated GPU memory was {lat['peak_allocated_vram_bytes']/1024**3:.2f} GiB. The model reported falling back to PyTorch linear attention because optional fast kernels were not installed; do not compare these timings with optimized kernels or batched serving. Per-image times and token counts are in `analysis.json` and the individual response files.

## Deployment and cost

The authenticated Runpod pod `j8qvat8tmzyhgr` was created at a reported GPU price of ${cost['gpu_hourly_rate_usd']:.2f}/hour. Approximate GPU cost through the final saved result is ${cost['estimated_compute_until_last_result_usd']:.3f}, based on elapsed time, not a reconciled invoice. Storage and subsequent runtime are additional. [Deployment details](deployment.json) and [cost estimate](cost-estimate.json) contain the configuration and assumptions.

At the time of this original trial, the pod was left running as the requested inference server. This historical deployment note does not describe its current lifecycle state. Its HTTP endpoint requires a bearer token; the local trial client reads that token from `/tmp/qrealign-lite-token`, which is outside the repository and may be removed by system cleanup. Treat that file as a credential. Pod stop/restart behavior and pricing should be checked before depending on it as a long-lived service.

## Rebuilding the comparison

Run `python report.py`, `python gallery.py` and `python render_pages.py` in this directory after rebuilding the source comparison with `../photo-scoring-2026-10-31/canonical_scores.py`. These commands use the retained observations and exact local image views; they make no model calls.
'''
(H/'README.md').write_text(text);print('Rebuilt 72-case canonical comparison and analysis')
