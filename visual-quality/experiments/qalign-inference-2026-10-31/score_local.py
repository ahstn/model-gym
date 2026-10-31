"""Score the frozen seven-image subset on a GPU using the author's OneAlign API."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'photo-scoring-2026-10-31'
IDS=['T0028','T0192','T0714','T0965','T0914','T0724','T2000']
REVISION='dcc603b95aa0ebd82afa696d4a1e20d11fc80ddb'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True,help='New JSONL file for raw model scores')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Refusing to overwrite existing results')
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU is required for this configuration')
    rows={r['case_id']:r for r in csv.DictReader((SOURCE/'scores-72.csv').open())}
    for cid in IDS:
        image=SOURCE/'images'/f'{cid}.png'
        if hashlib.sha256(image.read_bytes()).hexdigest()!=rows[cid]['sha256']: raise ValueError('Image hash mismatch: '+cid)
    model=AutoModelForCausalLM.from_pretrained('q-future/one-align',revision=REVISION,trust_remote_code=True,attn_implementation='eager',torch_dtype=torch.float16,device_map='auto').eval()
    with args.output.open('x') as output:
        for cid in IDS:
            started=time.perf_counter()
            with Image.open(SOURCE/'images'/f'{cid}.png') as original: image=original.convert('RGB')
            with torch.inference_mode():
                technical=float(model.score([image],task_='quality',input_='image').flatten()[0].item())
                aesthetics=float(model.score([image],task_='aesthetics',input_='image').flatten()[0].item())
            if not (1<=technical<=5 and 1<=aesthetics<=5): raise ValueError('Out-of-range score')
            row={'case_id':cid,'view_sha256':rows[cid]['sha256'],'checkpoint':'q-future/one-align','revision':REVISION,'technical_native':technical,'aesthetics_native':aesthetics,'reference_technical':float(rows[cid]['reference_technical_quality']),'reference_aesthetics':float(rows[cid]['reference_aesthetics']),'reference_eligibility':rows[cid]['reference_eligibility'],'luna': [float(rows[cid]['luna_technical_quality']),float(rows[cid]['luna_aesthetics'])],'deepseek':[float(rows[cid]['deepseek_technical_quality']),float(rows[cid]['deepseek_aesthetics'])],'seconds':round(time.perf_counter()-started,2),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            output.write(json.dumps(row)+'\n');output.flush();print(cid,technical,aesthetics,flush=True)

if __name__=='__main__': main()
