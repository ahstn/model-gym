"""Embed fixed candidate images with the pinned local EmbeddingGemma2 model."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = Path('/home/ahstn/git/model-gym/visual-quality')
DATA = ORIGINAL / 'data/rehearsal-4000-2026-11-01'
OLD = ORIGINAL / 'experiments/embeddinggemma2-clustering-2026-10-31'
CACHE = ORIGINAL / 'data/embeddinggemma2-clustering-2026-10-31/.hf_cache'
MODEL = 'google/embeddinggemma-2'
REVISION = 'f9972ac22e696a105c482202646979e33e4b5a1e'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, choices=(0, 1), required=True)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    os.environ.update(HF_HOME=str(CACHE), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false',
                      HF_HUB_DISABLE_PROGRESS_BARS='1')
    import numpy as np
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    from torch.nn import functional
    manifest_path = HERE/'candidate-manifest.json'
    cases = json.loads(manifest_path.read_text())['cases'][args.shard::2]
    npz_path = DATA/f'embeddings-shard-{args.shard}.npz'
    meta_path = DATA/f'embeddings-shard-{args.shard}.json'
    cfg = {'manifest_sha256': digest(manifest_path), 'script_sha256': digest(__file__),
           'model': MODEL, 'revision': REVISION, 'device': 'cpu', 'dtype': 'float32',
           'max_soft_tokens': 280, 'audio_config': None, 'dimensions': 768,
           'shard': args.shard, 'shards': 2, 'threads': args.threads, 'batch_size': args.batch_size}
    software = {n: importlib.metadata.version(n) for n in ('torch','torchvision','transformers','sentence-transformers','numpy','Pillow')}
    metadata = {'configuration': cfg, 'software': software, 'batches': [], 'status': 'in_progress'}
    vectors = np.empty((0,768), dtype=np.float32)
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        assert metadata['configuration'] == cfg and metadata['software'] == software
        assert digest(npz_path) == metadata['embeddings_sha256']
        old = np.load(npz_path,allow_pickle=False)
        vectors = old['embeddings']; assert old['case_ids'].tolist() == [r['case_id'] for r in cases[:len(vectors)]]
    old_manifest = json.loads((OLD/'manifest.json').read_text())
    old_meta = json.loads((OLD/'embedding-run.json').read_text())
    assert old_meta['software'] == software
    old_npz = np.load(OLD/'embeddings.npz',allow_pickle=False)
    assert digest(OLD/'embeddings.npz') == old_meta['embeddings_sha256']
    by_id = {r['case_id']:r for r in old_manifest['cases']}
    reusable = {by_id[c]['view_sha256']:v for c,v in zip(old_npz['case_ids'].tolist(),old_npz['embeddings'])}
    metadata['reused_artifact'] = {'path':str(OLD/'embeddings.npz'),'sha256':digest(OLD/'embeddings.npz'),'eligibility':'identical delivered image SHA256, same pinned checkpoint and inference software'}
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    torch.manual_seed(202611014000); torch.use_deterministic_algorithms(True)
    print(f'Loading shard{args.shard} {MODEL}@{REVISION}',flush=True)
    model=SentenceTransformer(MODEL,revision=REVISION,device='cpu',cache_folder=str(CACHE/'hub'),config_kwargs={'audio_config':None},model_kwargs={'dtype':torch.float32})
    model.eval(); metadata['loaded_parameters']=sum(p.numel() for p in model.parameters())
    print(f'Loaded {metadata["loaded_parameters"]} parameters',flush=True)
    for start in range(len(vectors),len(cases),args.batch_size):
        batch=cases[start:start+args.batch_size]; out=np.empty((len(batch),768),dtype=np.float32)
        missing=[]; image_inputs=[]
        for i,r in enumerate(batch):
            assert digest(r['image_path']) == r['view_sha256']
            if r['view_sha256'] in reusable: out[i]=reusable[r['view_sha256']]
            else:
                missing.append(i)
                with Image.open(r['image_path']) as im: image_inputs.append({'image': im.convert('RGB')})
        started=time.monotonic()
        if image_inputs:
            with torch.inference_mode():
                encoded=model.encode(image_inputs,batch_size=args.batch_size,processing_kwargs={'image':{'max_soft_tokens':280}},convert_to_tensor=True,show_progress_bar=False)
                out[missing]=functional.normalize(encoded.float(),p=2,dim=-1).cpu().numpy()
        assert np.isfinite(out).all()
        np.testing.assert_allclose(np.linalg.norm(out,axis=1),1,atol=1e-5)
        vectors=np.concatenate((vectors,out))
        metadata['batches'].append({'start':start,'count':len(batch),'reused':len(batch)-len(missing),'seconds':round(time.monotonic()-started,3)})
        temporary=npz_path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,embeddings=vectors,case_ids=np.asarray([r['case_id'] for r in cases[:len(vectors)]])); temporary.replace(npz_path)
        metadata.update(embedded_count=len(vectors),embeddings_sha256=digest(npz_path),status='complete' if len(vectors)==len(cases) else 'in_progress')
        meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
        if start%40==0 or len(vectors)==len(cases): print(f'shard{args.shard} embedded{len(vectors)}/{len(cases)}',flush=True)
    print('COMPLETE',flush=True)


if __name__=='__main__': main()
