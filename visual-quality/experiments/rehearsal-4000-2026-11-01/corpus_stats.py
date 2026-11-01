"""Image-stats on frozen delivered views; no scores or automatic floors."""
from __future__ import annotations
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import PIL
import image_stats
from image_stats import analyse_file
from image_stats.metrics import ALGORITHM_VERSION

HERE=Path(__file__).resolve().parent
DATA=Path('/home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01')
CONFIGURATION={
    'max_edge':1536,'algorithm_version':ALGORITHM_VERSION,
    'source_code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(image_stats.__file__).parent.glob('*.py'))},
    'software':{'numpy':np.__version__,'Pillow':PIL.__version__,'opencv':cv2.__version__},
}


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def one(row):
    assert file_sha(row['image_path'])==row['view_sha256'],'Delivered bytes changed'
    assert file_sha(row['source_path'])==row['source_sha256'],'Original source bytes changed'
    identity={'view_sha256':row['view_sha256'],'source_sha256':row['source_sha256'],'configuration':CONFIGURATION}
    cache_key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    path=DATA/'stats-cache'/f'{cache_key}.json'
    if path.exists():
        cached=json.loads(path.read_text())
        assert cached['cache_identity']==identity,'Cached statistics identity mismatch'
        assert cached['sha256']==row['view_sha256'] and cached['source_sha256']==row['source_sha256']
        return cached
    value=analyse_file(row['image_path'],max_edge=1536,source_path=row['source_path'])
    assert value['sha256']==row['view_sha256']
    value['source_sha256']=row['source_sha256']
    value['cache_identity']=identity
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,sort_keys=True)+'\n')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=HERE/'candidate-manifest.json')
    parser.add_argument('--benchmark-pool',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(); cv2.setNumThreads(1)
    cases=json.loads(args.manifest.read_text())['cases']
    if args.benchmark_pool: cases=[r for r in cases if r.get('benchmark_candidate')]
    output=HERE/('benchmark-candidate-stats.jsonl' if args.benchmark_pool else 'stats.jsonl')
    with output.open('w') as stream,ThreadPoolExecutor(args.workers) as pool:
        for i,(row,result) in enumerate(zip(cases,pool.map(one,cases)),1):
            stream.write(json.dumps({'case_id':row['case_id'],'family_id':row['family_id'],**result},sort_keys=True)+'\n')
            if i%100==0: print(f'Stats {i}/{len(cases)}',flush=True)
    print(f'COMPLETE {output}',flush=True)


if __name__=='__main__':main()
