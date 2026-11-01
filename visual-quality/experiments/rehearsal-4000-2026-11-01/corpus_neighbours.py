"""Audit cross-boundary neighbours before freezing the full rehearsal manifest."""
import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
from corpus_select import HERE,vectors,sha,write


def pages(pairs,lookup,prefix,limit):
    for start in range(0,min(limit,len(pairs)),12):
        page=Image.new('RGB',(1200,1260),(241,244,247));draw=ImageDraw.Draw(page)
        for n,pair in enumerate(pairs[start:start+12]):
            x=(n%2)*600;y=(n//2)*210
            for offset,field in((0,'a'),(290,'b')):
                row=lookup[pair[field]]
                with Image.open(row['image_path']) as im:
                    im=im.convert('RGB');im.thumbnail((280,170))
                    page.paste(im,(x+offset+(280-im.width)//2,y+25))
                draw.text((x+offset+3,y+5),pair[field],fill=(15,25,35))
            text=f'cos={pair["cosine"]:.4f}; dHash={pair["dhash_distance"]}; historical={pair["historically_exposed_b"]}'
            draw.text((x+3,y+193),text,fill=(15,25,35))
        destination=HERE/'review'/f'{prefix}-{start//12+1:02d}.jpg';destination.parent.mkdir(exist_ok=True)
        page.save(destination,quality=90)


def preview_pairs(pairs,stats,limit):
    """Avoid filling every review page with repeated uniform-frame neighbours."""
    chosen=[];per_target=Counter();low_information=0
    for pair in pairs:
        flags=stats[pair['a']]['analysis']['flags']
        low=bool(set(flags)&{'uniform_frame','near_black','near_white'})
        if per_target[pair['a']]>=2 or (low and low_information>=12):continue
        chosen.append(pair);per_target[pair['a']]+=1;low_information+=int(low)
        if len(chosen)==limit:break
    return chosen


def main(benchmark_path=None,output_stem='cross-boundary'):
    candidate_path=HERE/'candidate-manifest.json';benchmark_path=benchmark_path or HERE/'benchmark-manifest.json'
    neighbours_path=HERE/f'{output_stem}-neighbours.json'
    review_path=HERE/f'{output_stem}-review.json'
    if neighbours_path.exists() or review_path.exists():
        raise ValueError('Refusing to overwrite a prior neighbour/review artifact; use a new output stem')
    cases=json.loads(candidate_path.read_text())['cases'];benchmark=json.loads(benchmark_path.read_text())['cases']
    early_review=json.loads((HERE/'benchmark-candidate-review.json').read_text())
    historical_guards={r['candidate_id'] for r in early_review['historical_guards']}
    vec=vectors();assert len(vec)==len(cases)==5500
    lookup={r['candidate_id']:r for r in cases};bench_ids={r['candidate_id'] for r in benchmark}
    other=[r for r in cases if r['candidate_id'] not in bench_ids]
    a=np.asarray([vec[r['candidate_id']] for r in benchmark]);b=np.asarray([vec[r['candidate_id']] for r in other])
    sim=a@b.T;all_pairs=[]
    for i,row in enumerate(benchmark):
        indices=set(np.argsort(-sim[i])[:3].tolist())|set(np.flatnonzero(sim[i]>=.94).tolist())
        for j in indices:
            candidate=other[j]
            all_pairs.append({'a':row['candidate_id'],'benchmark_case_id':row['case_id'],'b':candidate['candidate_id'],
                'cosine':float(sim[i,j]),'dhash_distance':(int(row['dhash'],16)^int(candidate['dhash'],16)).bit_count(),
                'historically_exposed_b':not candidate['historically_fresh'] or candidate['candidate_id'] in historical_guards,
                'same_category':row['category']==candidate['category']})
    all_pairs.sort(key=lambda p:-p['cosine'])
    historical=[p for p in all_pairs if p['historically_exposed_b']]
    stats={r['case_id']:r for r in map(json.loads,(HERE/'benchmark-candidate-stats.jsonl').read_text().splitlines())}
    historical_preview=preview_pairs(historical,stats,48);all_preview=preview_pairs(all_pairs,stats,72)
    # Conservative diversity exclusion does not assert photographic identity.
    automatic=sorted({p['b'] for p in all_pairs if p['cosine']>=.985})
    related_exclusions=sorted({candidate for group in early_review['related_groups']
        if set(group['candidates'])&bench_ids for candidate in group['candidates'] if candidate not in bench_ids})
    payload={'schema_version':1,'candidate_manifest_sha256':sha(candidate_path),
        'benchmark_manifest_sha256':sha(benchmark_path),'script_sha256':sha(__file__),
        'pair_policy':'Every benchmark image has its three closest background neighbours, plus all cosine>=0.94 pairs',
        'pairs':all_pairs,'similarity_diversity_excluded_candidates':automatic,
        'reviewed_related_group_excluded_candidates':related_exclusions,
        'early_review_sha256':sha(HERE/'benchmark-candidate-review.json'),
        'preview_policy':'Descending cosine, at most two pairs per target and twelve uniform/near-black/near-white target pairs; all raw candidates retained separately',
        'historical_preview_pairs':historical_preview,'all_preview_pairs':all_preview,
        'warning':'A historically exposed related scene is a benchmark replacement only after visual confirmation of duplicate-family leakage'}
    write(neighbours_path,payload)
    # The review is separate from the early benchmark review so its pinned artifact is never changed.
    review=review_path
    if not review.exists():write(review,{
        'status':'pending','neighbours_sha256':sha(neighbours_path),
        'excluded_candidates':sorted(set(automatic+related_exclusions)),'benchmark_replacements':[], 'decisions':[],
        'instructions':'Visually inspect historical and full cross-boundary sheets; record confirmed duplicate exclusions and any benchmark replacements before finalization'})
    pages(historical_preview,lookup,f'{output_stem}-historical',48)
    pages(all_preview,lookup,f'{output_stem}-all',72)
    print(json.dumps({'pairs':len(all_pairs),'historical_pairs':len(historical),'automatic_diversity_exclusions':len(automatic)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-manifest',type=Path,default=HERE/'benchmark-manifest.json')
    parser.add_argument('--output-stem',default='cross-boundary')
    args=parser.parse_args();main(args.benchmark_manifest,args.output_stem)
