"""Content-aware benchmark and rehearsal selection from genuine image embeddings."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
DATA=Path('/home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01')
SEED=202611014000
CLUSTER_SEED=SEED%(2**32)


def key(tag,value):return hashlib.sha256(f'{SEED}:{tag}:{value}'.encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def vectors():
    out={}
    for n in (0,1):
        data=np.load(DATA/f'embeddings-shard-{n}.npz',allow_pickle=False)
        out.update(zip(data['case_ids'].tolist(),data['embeddings']))
    return out


def fit(cases,vec,k,tag):
    matrix=np.asarray([vec[r['candidate_id']] for r in cases])
    labels=KMeans(n_clusters=k,random_state=CLUSTER_SEED,n_init=20).fit_predict(matrix)
    mapping={r['candidate_id']:int(v) for r,v in zip(cases,labels)}
    write(HERE/f'{tag}-clusters.json',{'clusters':mapping,'seed':CLUSTER_SEED,'experiment_seed':SEED,'k':k,'n_init':20,'count':len(cases)})
    return mapping


def proportional(pool,count,clusters,tag):
    groups=defaultdict(list)
    for r in pool:groups[clusters[r['candidate_id']]].append(r)
    exact={c:len(v)*count/len(pool) for c,v in groups.items()}
    quotas={c:int(math.floor(v)) for c,v in exact.items()}
    for c in sorted(groups,key=lambda c:(-(exact[c]-quotas[c]),c))[:count-sum(quotas.values())]:quotas[c]+=1
    result=[]
    for c,values in sorted(groups.items()):
        result.extend(sorted(values,key=lambda r:key(tag,r['candidate_id']))[:quotas[c]])
    assert len(result)==count
    return result


def diverse(pool,count,vec,tag):
    if count==0:return []
    if count<0:raise ValueError('Selection count cannot be negative')
    if len(pool)<count:raise ValueError(f'Insufficient {tag}: {len(pool)} < {count}')
    pool=sorted(pool,key=lambda r:key(tag,r['candidate_id']))
    if len(pool)==count:return pool
    matrix=np.asarray([vec[r['candidate_id']] for r in pool])
    centroid=matrix.mean(axis=0);centroid/=np.linalg.norm(centroid)
    chosen=[int(np.argmax(matrix@centroid))]
    nearest=matrix@matrix[chosen[0]]
    nearest[chosen]=2
    while len(chosen)<count:
        i=int(np.argmin(nearest));chosen.append(i)
        nearest=np.maximum(nearest,matrix@matrix[i]);nearest[chosen]=2
    return [pool[i] for i in chosen]


def duplicate_review(cases,vec):
    images=np.asarray([vec[r['candidate_id']] for r in cases]);sim=images@images.T
    pairs=[]
    for i in range(len(cases)):
        for j in np.flatnonzero(sim[i,i+1:]>=.90)+i+1:
            a,b=cases[i],cases[j]
            pairs.append({'a':a['candidate_id'],'b':b['candidate_id'],'cosine':float(sim[i,j]),
                          'dhash_distance':(int(a['dhash'],16)^int(b['dhash'],16)).bit_count(),
                          'same_category':a['category']==b['category'],'review_status':'pending'})
    pairs.sort(key=lambda p:-p['cosine'])
    write(HERE/'benchmark-duplicate-candidates.json',{'threshold':.90,'pairs':pairs,'note':'Similarity flags review, not proven duplication'})
    # Limit previews per category so repeated chart layouts do not hide photographic candidates.
    lookup={r['candidate_id']:r for r in cases};chosen=[];counts=Counter()
    for p in pairs:
        category='controls' if lookup[p['a']]['category']=='controls' else 'photos'
        if counts[category]<(16 if category=='controls' else 64):chosen.append(p);counts[category]+=1
    for start in range(0,len(chosen),12):
        page=Image.new('RGB',(1200,6*210),(241,244,247));draw=ImageDraw.Draw(page)
        for k,p in enumerate(chosen[start:start+12]):
            x=(k%2)*600;y=(k//2)*210
            for offset,name in ((0,'a'),(290,'b')):
                r=lookup[p[name]]
                with Image.open(r['image_path']) as im:
                    im=im.convert('RGB');im.thumbnail((280,175));page.paste(im,(x+offset+(280-im.width)//2,y+25))
                draw.text((x+offset+5,y+5),p[name],fill=(20,30,40))
            draw.text((x+5,y+193),f'cos={p["cosine"]:.4f}; dHash distance={p["dhash_distance"]}',fill=(20,30,40))
        dest=HERE/'review'/f'benchmark-related-{start//12+1:02d}.jpg';dest.parent.mkdir(exist_ok=True)
        page.save(dest,quality=90)
    return pairs


def historical_review(cases,vec):
    historical=Path('/home/ahstn/git/model-gym/visual-quality/experiments/embeddinggemma2-clustering-2026-10-31')
    prior=json.loads((historical/'manifest.json').read_text())['cases']
    lookup={r['case_id']:r for r in prior}
    old=np.load(historical/'embeddings.npz',allow_pickle=False)
    old_ids=old['case_ids'].tolist()
    matrix=np.asarray([vec[r['candidate_id']] for r in cases])
    similarities=matrix@old['embeddings'].T
    pairs=[]
    for i,row in enumerate(cases):
        j=int(np.argmax(similarities[i]));other=lookup[old_ids[j]]
        pairs.append({'candidate_id':row['candidate_id'],'historical_case_id':old_ids[j],
                      'cosine':float(similarities[i,j]),'historical_family_id':other['family_id'],
                      'dhash_distance':(int(row['dhash'],16)^int(other['dhash'],16)).bit_count()})
    pairs.sort(key=lambda p:-p['cosine'])
    write(HERE/'benchmark-historical-neighbours.json',{
        'historical_manifest_sha256':sha(historical/'manifest.json'),
        'historical_embeddings_sha256':sha(historical/'embeddings.npz'),
        'scope':'384 previously embedded views only; other historical families are guarded by IDs and dHash',
        'pairs':pairs})
    candidates={r['candidate_id']:r for r in cases}
    for start in range(0,min(36,len(pairs)),12):
        page=Image.new('RGB',(1200,6*210),(241,244,247));draw=ImageDraw.Draw(page)
        for k,p in enumerate(pairs[start:start+12]):
            x=(k%2)*600;y=(k//2)*210
            for offset,row,label in ((0,candidates[p['candidate_id']],p['candidate_id']),
                                     (290,lookup[p['historical_case_id']],p['historical_case_id']+' historical')):
                with Image.open(row['image_path']) as im:
                    im=im.convert('RGB');im.thumbnail((280,175));page.paste(im,(x+offset+(280-im.width)//2,y+25))
                draw.text((x+offset+5,y+5),label,fill=(20,30,40))
            draw.text((x+5,y+193),f'cos={p["cosine"]:.4f}; dHash distance={p["dhash_distance"]}',fill=(20,30,40))
        dest=HERE/'review'/f'benchmark-historical-{start//12+1:02d}.jpg';dest.parent.mkdir(exist_ok=True)
        page.save(dest,quality=90)


def preview():
    cases=[r for r in json.loads((HERE/'candidate-manifest.json').read_text())['cases'] if r['benchmark_candidate']]
    vec=vectors()
    assert all(r['candidate_id'] in vec for r in cases),'Benchmark candidate embeddings incomplete'
    duplicate_review(cases,vec)
    historical_review(cases,vec)
    print('Prepared candidate and historical-neighbour review sheets',flush=True)


def benchmark():
    manifest=json.loads((HERE/'candidate-manifest.json').read_text());cases=[r for r in manifest['cases'] if r['benchmark_candidate']]
    vec=vectors();missing=[r['candidate_id'] for r in cases if r['candidate_id'] not in vec]
    if missing:raise ValueError(f'Benchmark embeddings incomplete: {len(missing)}')
    stats={r['case_id']:r for r in map(json.loads,(HERE/'benchmark-candidate-stats.jsonl').read_text().splitlines())}
    pairs=json.loads((HERE/'benchmark-duplicate-candidates.json').read_text())['pairs']
    # Eliminate retained pairs above the similarity threshold; this is not a duplicate assertion.
    review_path=HERE/'benchmark-candidate-review.json'
    review=json.loads(review_path.read_text()) if review_path.exists() else {'excluded_candidates':[]}
    assert review.get('status')=='reviewed','Complete visual candidate review before benchmark selection'
    assert all(sha(HERE/name)==digest for name,digest in review['inputs'].items()),'Reviewed artifacts changed'
    blocked=set(review['excluded_candidates']); by_id={r['candidate_id']:r for r in cases}
    for p in pairs:
        if p['cosine']>=.985 and p['a'] not in blocked and p['b'] not in blocked:
            blocked.add(max((p['a'],p['b']),key=lambda c:key('similarity-diversity',c)))
    eligible=[r for r in cases if r['candidate_id'] not in blocked]
    clusters=fit(cases,vec,16,'benchmark')
    selected=[];used=set()
    def take(pool,n,partition,tag,method='proportional'):
        pool=[r for r in pool if r['candidate_id'] not in used]
        chosen=proportional(pool,n,clusters,tag) if method=='proportional' else diverse(pool,n,vec,tag)
        for r in chosen:
            selected.append({**r,'benchmark_partition':partition,'benchmark_stratum':tag,'content_cluster':clusters[r['candidate_id']]})
            used.add(r['candidate_id'])
    scope_path=HERE/'benchmark-scope-selection.json'
    scope=json.loads(scope_path.read_text())
    scope_ids=set(scope['reserved_candidates'])
    assert len(scope_ids)==6 and not scope_ids.intersection(blocked),'Resolve scope candidate duplicate review before selection'
    take([r for r in eligible if r['candidate_id'] in scope_ids],6,'challenge','scope_boundary_retrieval_review','diverse')
    for partition,quotas in [('development',{'megalith':80,'vizwiz':16,'controls':4}),('representative',{'megalith':240,'vizwiz':48,'controls':12})]:
        for group,n in quotas.items():take([r for r in eligible if r['category']==group],n,partition,f'{partition}:{group}')
    def tail(metric,quantile,high=False,category=None):
        pool=[r for r in eligible if r['category']!='controls' and (category is None or r['category']==category) and r['candidate_id'] not in used]
        values=np.asarray([stats[r['candidate_id']]['analysis']['metrics'][metric] for r in pool]);cut=float(np.quantile(values,quantile))
        return [r for r,v in zip(pool,values) if (v>=cut if high else v<=cut)]
    take(tail('sharpness_peak',.25),15,'challenge','softness','diverse')
    pool=tail('noise_sigma',.8,True)
    median=np.median([stats[r['candidate_id']]['analysis']['metrics']['sharpness_peak'] for r in eligible if r['category']!='controls'])
    take([r for r in pool if stats[r['candidate_id']]['analysis']['metrics']['sharpness_peak']>=median],10,'challenge','sharp_noise','diverse')
    take(tail('luma_mean',.25),10,'challenge','dark','diverse')
    take(tail('highlight_clip',.8,True),10,'challenge','bright_clipped','diverse')
    take(tail('local_contrast',.25),10,'challenge','low_contrast','diverse')
    for component in ('clevr','plotqa'):take([r for r in eligible if r['component']==component],5,'challenge',f'nonphoto:{component}','diverse')
    take([r for r in eligible if r['category']=='vizwiz'],15,'challenge','capture_context','diverse')
    take(tail('sharpness_peak',.75,True),10,'challenge','high_detail','diverse')
    take([r for r in eligible if r['category']=='megalith'],4,'challenge','content_diversity','diverse')
    assert len(selected)==500 and len({r['family_id'] for r in selected})==500
    assert all(r['historically_fresh'] and not r['controlled_parent'] for r in selected)
    order={'development':0,'representative':1,'challenge':2}
    selected.sort(key=lambda r:(order[r['benchmark_partition']],key('benchmark-final-order',r['candidate_id'])))
    related={c:g['group_id'] for g in review.get('related_groups',[]) for c in g['candidates']}
    for i,r in enumerate(selected,1):
        r['case_id']=f'R{i:04d}'
        r['reviewed_related_group']=related.get(r['candidate_id'])
    np.savez_compressed(HERE/'benchmark-candidate-embeddings.npz',
        embeddings=np.asarray([vec[r['candidate_id']] for r in cases]),
        case_ids=np.asarray([r['candidate_id'] for r in cases]))
    payload={'schema_version':1,'seed':SEED,'cases':selected,'counts':{'images':500,'families':500,'partitions':dict(Counter(r['benchmark_partition'] for r in selected))},
             'selection_software':{'numpy':np.__version__,'scikit_learn':sklearn.__version__,'cluster_seed':CLUSTER_SEED},
             'inputs':{'candidate_manifest_sha256':sha(HERE/'candidate-manifest.json'),'selection_script_sha256':sha(__file__),
                       'benchmark_candidate_embeddings_sha256':sha(HERE/'benchmark-candidate-embeddings.npz'),
                       'candidate_stats_sha256':sha(HERE/'benchmark-candidate-stats.jsonl')},
             'selection_policy':{'representative':'Declared 80/16/4 source proportions of the guarded 1000-candidate benchmark frame; proportional content-cluster strata with seeded random sampling',
                                 'challenge':'Predeclared image-stat/source tails with within-tail embedding diversity; no teacher or reference scores',
                                 'near_similarity_guard':'No retained candidate pair has cosine>=.985; flags are not asserted duplicates',
                                 'freeze_exception':'Only objectively confirmed historical exposure or duplicate-family leakage may replace a benchmark case after early reference review; retain superseded identities and blinded review provenance before any benchmark model calls',
                                 'similarity_guard_excluded_candidates':sorted(blocked)}}
    if review_path.exists():payload['inputs']['candidate_review_sha256']=sha(review_path)
    payload['inputs']['scope_selection_sha256']=sha(scope_path)
    write(HERE/'benchmark-manifest.json',payload)
    print(json.dumps(payload['counts']),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['preview','benchmark']);args=parser.parse_args()
    if args.stage=='preview':preview()
    if args.stage=='benchmark':benchmark()


if __name__=='__main__':main()
