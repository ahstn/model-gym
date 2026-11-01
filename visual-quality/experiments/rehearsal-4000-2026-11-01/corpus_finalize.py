"""Select the full4000 rehearsal after frozen500 benchmark and all candidate embeddings."""
from __future__ import annotations
import hashlib
import io
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from PIL import Image
from visual_quality.views import _render
from visual_quality.generate import _parent_view
from corpus_prepare import DATA as SOURCE_DATA, PILOT, OUTPUT, read_lines, sha, write
from corpus_select import HERE, DATA, SEED, CLUSTER_SEED, key, vectors, diverse


def allocations(groups,count):
    """Capacity-capped square-root cluster allocation reduces large subject groups."""
    quotas={c:0 for c in groups}
    for _ in range(count):
        c=max((c for c in groups if quotas[c]<len(groups[c])),key=lambda c:(math.sqrt(len(groups[c]))/(quotas[c]+1),-c))
        quotas[c]+=1
    return quotas


def content_sample(pool,count,clusters,vec,tag):
    if count<0 or len(pool)<count:raise ValueError(f'Insufficient {tag}: {len(pool)} for {count}')
    groups=defaultdict(list)
    for r in pool:groups[clusters[r['candidate_id']]].append(r)
    selected=[]
    for c,n in allocations(groups,count).items():
        selected.extend(diverse(groups[c],n,vec,f'{tag}:{c}'))
    return selected


def family_views(parent,variants):
    original=Path(parent['source_path']).read_bytes()
    assert hashlib.sha256(original).hexdigest()==parent['source_sha256']
    canonical,canonical_prep=_parent_view(Path(parent['source_path']))
    buffer=io.BytesIO();canonical.save(buffer,format='PNG',compress_level=6)
    canonical_bytes=buffer.getvalue();canonical_sha=hashlib.sha256(canonical_bytes).hexdigest()
    canonical_path=OUTPUT/'family-canonical'/f'{canonical_sha}.png'
    canonical_path.parent.mkdir(exist_ok=True)
    if not canonical_path.exists():canonical_path.write_bytes(canonical_bytes)
    for variant in variants:
        recorded=variant['preprocessing']
        assert recorded==canonical_prep,'Canonical parent preprocessing differs from recorded variant generation'
    inputs=[canonical_bytes]+[(SOURCE_DATA/r['image_path']).read_bytes() for r in variants]
    assert all(hashlib.sha256(b).hexdigest()==r['sha256'] for b,r in zip(inputs[1:],variants))
    edge=1536
    while True:
        rendered=[_render(b,edge) for b in inputs]
        if all(len(b)<=2650000 for b,_ in rendered):break
        edge=edge*9//10
        if edge<256:raise ValueError('Family cannot fit common image limits')
    if any(min(p['output_dimensions'])<256 or math.prod(p['output_dimensions'])<78400 for _,p in rendered):raise ValueError('Family dimensions too small')
    hashes=[hashlib.sha256(b).hexdigest() for b,_ in rendered]
    if len(set(hashes))!=4:raise ValueError('Family members have identical delivered PNGs')
    if len({tuple(p['output_dimensions']) for _,p in rendered})!=1:raise ValueError('Family dimensions differ')
    pixel_hashes=[]
    for payload,_ in rendered:
        with Image.open(io.BytesIO(payload)) as im:pixel_hashes.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
    if len(set(pixel_hashes))!=4:raise ValueError('Family members have identical delivered pixels')
    records=[]
    for i,((data,prep),digest) in enumerate(zip(rendered,hashes)):
        path=OUTPUT/'views'/f'{digest}.png'
        if not path.exists():path.write_bytes(data)
        width,height=prep['output_dimensions']
        record={**parent,'image_path':str(path),'view_sha256':digest,'delivered_bytes':len(data),
                'delivered_pixel_sha256':pixel_hashes[i],
                'width':width,'height':height,'preprocessing':prep,'target_edge':edge,
                'benchmark_partition':'rehearsal','stratum':'controlled','role':'source_parent' if i==0 else 'variant',
                'embedding_view_sha256':parent['view_sha256'],'severity':None if i==0 else i,
                'family_preprocessing':{'canonical_parent_path':str(canonical_path),'canonical_parent_sha256':canonical_sha,
                    'canonical_parent':canonical_prep,'final_parent_view_sha256':hashes[0],
                    'shared_final_edge':edge,'source_parent_sha256':parent['source_sha256'],
                    'lineage':'original parent -> recorded canonical parent at1536 -> transform ifvariant -> shared final full-frame cap'}}
        if i:
            row=variants[i-1]
            with Image.open(io.BytesIO(inputs[i])) as source:source_format=source.format
            record.update(image_id=row['id'],source_path=str(SOURCE_DATA/row['image_path']),source_sha256=row['sha256'],
                          source=row['source'],source_id=row['source_id'],source_revision=row['source_revision'],
                          source_format=source_format,source_metadata={'format':source_format,'frames':1,'icc':False,'alpha':False},
                          decoded_sha256=row['decoded_sha256'],dhash=row['dhash'],component='distortions',
                          category='distortions',transform=row['transform'],controlled_parent=False,
                          candidate_id=None,parent_candidate_id=parent['candidate_id'],parent_image_id=parent['image_id'])
        records.append(record)
    return records


def main():
    candidate_path=HERE/'candidate-manifest.json';candidate=json.loads(candidate_path.read_text())
    candidates=candidate['cases'];vec=vectors()
    assert len(vec)==len(candidates)==5500,'Wait for all candidate embeddings before selecting rehearsal'
    review_path=HERE/'cross-boundary-review.json'
    review=json.loads(review_path.read_text())
    assert review['status']=='reviewed','Cross-boundary duplicate review must finish before the final manifest'
    assert review['neighbours_sha256']==sha(HERE/'cross-boundary-neighbours.json')
    neighbours=json.loads((HERE/'cross-boundary-neighbours.json').read_text())
    assert neighbours['candidate_manifest_sha256']==sha(candidate_path)
    assert neighbours['benchmark_manifest_sha256']==sha(HERE/'benchmark-manifest.json')
    assert not review['benchmark_replacements'],'Resolve confirmed benchmark exposure before finalization'
    excluded=set(review['excluded_candidates'])
    matrix=np.asarray([vec[r['candidate_id']] for r in candidates])
    labels=KMeans(n_clusters=40,random_state=CLUSTER_SEED,n_init=20).fit_predict(matrix)
    clusters={r['candidate_id']:int(v) for r,v in zip(candidates,labels)}
    write(HERE/'rehearsal-clusters.json',{'clusters':clusters,'k':40,'n_init':20,'seed':CLUSTER_SEED,'experiment_seed':SEED,
                                      'counts':dict(Counter(map(int,labels)))})
    benchmark_manifest=json.loads((HERE/'benchmark-manifest.json').read_text())
    benchmark=benchmark_manifest['cases']
    early_review=json.loads((HERE/'benchmark-candidate-review.json').read_text())
    assert sha(HERE/'benchmark-candidate-review.json')==benchmark_manifest['inputs']['candidate_review_sha256']
    related_groups={c:g['group_id'] for g in early_review['related_groups'] for c in g['candidates']}
    historical_guards={g['candidate_id']:g for g in early_review['historical_guards']}
    benchmark_candidates={r['candidate_id'] for r in benchmark}
    for group in early_review['related_groups']:
        if set(group['candidates'])&benchmark_candidates:
            excluded.update(set(group['candidates'])-benchmark_candidates)
    assert not excluded&benchmark_candidates,'A benchmark identity is excluded by the background review'
    selected=[{**r,'rehearsal_content_cluster':clusters[r['candidate_id']],
               'stratum':r['category'],'role':'standalone'} for r in benchmark]
    used={r['image_id'] for r in selected};used_families={r['family_id'] for r in selected}
    rows=read_lines(PILOT/'train.jsonl');variants=defaultdict(list)
    for row in rows:
        if row['component']=='distortions':variants[row['parent_id']].append(row)
    family_rejections=[];controls=[]
    for operator,count in [('gaussian_blur',34),('gaussian_noise',33),('jpeg_compression',33)]:
        pool=[r for r in candidates if r['candidate_id'] not in excluded and r['controlled_parent'] and r['operator']==operator and r['family_id'] not in used_families
              and not r['source_metadata']['icc'] and not r['source_metadata']['alpha']]
        ordered=content_sample(pool,min(len(pool),count+20),clusters,vec,f'controlled:{operator}')
        accepted=0
        for parent in ordered:
            members=sorted(variants[parent['image_id']],key=lambda r:r['transform']['severity_index'])
            if len(members)!=3 or [r['transform']['severity_index'] for r in members]!=[1,2,3]:continue
            try:records=family_views(parent,members)
            except (ValueError,OSError) as exc:
                family_rejections.append({'image_id':parent['image_id'],'reason':str(exc)});continue
            for r in records:r['rehearsal_content_cluster']=clusters[parent['candidate_id']]
            controls.extend(records);used.add(parent['image_id']);used_families.add(parent['family_id']);accepted+=1
            if accepted==count:break
        if accepted!=count:raise ValueError(f'Insufficient valid controlled families for {operator}')
    selected.extend(controls)
    for group,target in [('megalith',3000),('vizwiz',600),('controls',100)]:
        already=sum(r['category']==group for r in selected)
        pool=[r for r in candidates if r['candidate_id'] not in excluded and r['category']==group and r['family_id'] not in used_families]
        picked=content_sample(pool,target-already,clusters,vec,f'rehearsal:{group}')
        for r in picked:
            selected.append({**r,'benchmark_partition':'rehearsal','rehearsal_content_cluster':clusters[r['candidate_id']],
                             'stratum':group,'role':'standalone','controlled_parent':False,'operator':None,'severity':None})
            used.add(r['image_id']);used_families.add(r['family_id'])
    others=sorted([r for r in selected if r['benchmark_partition']=='rehearsal'],key=lambda r:key('final-order',r['image_id']))
    for i,r in enumerate(others,501):r['case_id']=f'R{i:04d}'
    selected=selected[:500]+others
    for row in selected:
        candidate_id=row['candidate_id'] or row.get('parent_candidate_id')
        row['reviewed_related_group']=related_groups.get(candidate_id)
        if candidate_id in historical_guards:
            row['historical_review_guard']=historical_guards[candidate_id]
            row['historically_fresh']=False
    assert len(selected)==4000 and len({r['case_id'] for r in selected})==4000
    assert len({r['family_id'] for r in selected})==3700
    assert len({r['view_sha256'] for r in selected})==4000
    assert len({r['image_id'] for r in selected})==4000
    assert Counter(r['benchmark_partition'] for r in selected)=={'rehearsal':3500,'development':100,'representative':300,'challenge':100}
    final_bench={r['case_id']:r for r in selected[:500]}
    for row in benchmark:
        assert all(final_bench[row['case_id']][k]==row[k] for k in ('image_id','view_sha256','family_id','benchmark_partition','image_path'))
    for r in selected:
        assert sha(Path(r['image_path']))==r['view_sha256']
        assert all(v is False for v in r['task_masks'].values())
    np.savez_compressed(HERE/'candidate-embeddings.npz',embeddings=matrix,case_ids=np.asarray([r['candidate_id'] for r in candidates]))
    shard_metadata=[]
    for shard in(0,1):
        source=DATA/f'embeddings-shard-{shard}.json';record=json.loads(source.read_text())
        assert record['status']=='complete' and record['embedded_count']==2750
        assert record['embeddings_sha256']==sha(DATA/f'embeddings-shard-{shard}.npz')
        write(HERE/f'embedding-shard-{shard}.json',record);shard_metadata.append(record)
    write(HERE/'embedding-provenance.json',{
        'candidate_embeddings_sha256':sha(HERE/'candidate-embeddings.npz'),
        'candidate_manifest_sha256':sha(candidate_path),'shards':[{'path':f'embedding-shard-{i}.json',
        'sha256':sha(HERE/f'embedding-shard-{i}.json')} for i in(0,1)],
        'total_embeddings':5500,'reused_identical_views':sum(b['reused'] for m in shard_metadata for b in m['batches']),
        'configuration':shard_metadata[0]['configuration'],'software':shard_metadata[0]['software']})
    payload={'schema_version':1,'seed':SEED,'cases':selected,
             'selection_software':{'numpy':np.__version__,'scikit_learn':sklearn.__version__,'cluster_seed':CLUSTER_SEED},
             'counts':{'images':4000,'families':3700,'benchmark_images':500,'benchmark_families':500,
                       'partitions':dict(Counter(r['benchmark_partition'] for r in selected)),
                       'components':dict(Counter(r['component'] for r in selected))},
             'inputs':{'candidate_manifest_sha256':sha(candidate_path),'benchmark_manifest_sha256':sha(HERE/'benchmark-manifest.json'),
                       'finalize_script_sha256':sha(Path(__file__))},
             'selection_policy':{'source_quotas':'3000Megalith incl100controlledparents,600VizWiz,300derivatives,100nonphotos',
                                 'content':'capacity-capped square-root cluster allocation then farthest-first within source/operator quotas',
                                 'score_information_used':False,'benchmark_freshness':'historical family/dHash guard; independent500families'},
             'preprocessing_policy':candidate['preprocessing_policy'],'family_render_rejections':family_rejections,
             'training_status':'All task masks false; experimental observations only'}
    payload['inputs']['cross_boundary_review_sha256']=sha(review_path)
    payload['inputs']['embedding_provenance_sha256']=sha(HERE/'embedding-provenance.json')
    payload['selection_policy']['cross_boundary_excluded_candidates']=sorted(excluded)
    write(HERE/'manifest.json',payload)
    print(json.dumps(payload['counts']),flush=True)


if __name__=='__main__':main()
