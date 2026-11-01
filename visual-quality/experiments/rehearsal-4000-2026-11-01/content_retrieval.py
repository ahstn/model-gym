"""Retrieve scope-boundary candidates for visual review, never for score assignment."""
import argparse
import json
import os
from pathlib import Path
from corpus_embed import CACHE,MODEL,REVISION,HERE,DATA,digest

QUERIES=[
    'a screenshot of a computer screen or software interface',
    'a poster or graphic design with text and artwork',
    'a painting or drawn illustration rather than a photograph',
    'a computer generated rendering or synthetic scene',
    'a photograph of a printed poster hanging on a wall',
    'a photograph of a printed document or product package',
]


def encode():
    os.environ.update(HF_HOME=str(CACHE),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false')
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    model=SentenceTransformer(MODEL,revision=REVISION,device='cpu',cache_folder=str(CACHE/'hub'),
                              config_kwargs={'audio_config':None},model_kwargs={'dtype':torch.float32})
    model.eval()
    with torch.inference_mode():
        vectors=model.encode(QUERIES,prompt_name='Retrieval-query',normalize_embeddings=True,
                             batch_size=6,show_progress_bar=False)
    np.savez_compressed(HERE/'scope-query-embeddings.npz',embeddings=vectors)
    (HERE/'scope-query-provenance.json').write_text(json.dumps({
        'model':MODEL,'revision':REVISION,'queries':QUERIES,'prompt_name':'Retrieval-query',
        'purpose':'Visual review only; similarity does not classify photographic eligibility or assign scores',
        'script_sha256':digest(__file__),'embeddings_sha256':digest(HERE/'scope-query-embeddings.npz')},indent=2)+'\n')


def review():
    import numpy as np
    from PIL import Image,ImageDraw
    cases=[r for r in json.loads((HERE/'candidate-manifest.json').read_text())['cases']
           if r['benchmark_candidate'] and r['category']!='controls']
    vectors={}
    for shard in(0,1):
        item=np.load(DATA/f'embeddings-shard-{shard}.npz',allow_pickle=False)
        vectors.update(zip(item['case_ids'].tolist(),item['embeddings']))
    assert all(r['candidate_id'] in vectors for r in cases)
    queries=np.load(HERE/'scope-query-embeddings.npz',allow_pickle=False)['embeddings']
    similarities=queries@np.asarray([vectors[r['candidate_id']] for r in cases]).T
    audit=[]
    for qi,query in enumerate(QUERIES):
        selected=np.argsort(-similarities[qi])[:16]
        page=Image.new('RGB',(1200,1040),(244,246,248));draw=ImageDraw.Draw(page)
        draw.text((12,10),query,fill=(10,20,30))
        for n,index in enumerate(selected):
            row=cases[index];x=(n%4)*300;y=40+(n//4)*250
            with Image.open(row['image_path']) as im:
                im=im.convert('RGB');im.thumbnail((290,210));page.paste(im,(x+(300-im.width)//2,y))
            draw.text((x+6,y+215),f'{row["candidate_id"]} similarity={similarities[qi,index]:.4f}',fill=(10,20,30))
            audit.append({'query':query,'candidate_id':row['candidate_id'],'cosine':float(similarities[qi,index])})
        out=HERE/'review'/f'scope-retrieval-{qi+1:02d}.jpg';out.parent.mkdir(exist_ok=True);page.save(out,quality=90)
    (HERE/'scope-retrieval.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['encode','review'])
    args=parser.parse_args()
    if args.stage=='encode':encode()
    else:review()
