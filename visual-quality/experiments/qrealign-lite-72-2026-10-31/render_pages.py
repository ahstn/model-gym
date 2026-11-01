#!/usr/bin/env python3
"""Render exact saved photos, scores and measured Lite latency as image pages."""
import csv
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'photo-scoring-2026-10-31'
NAMES=[('Reference','reference'),('Q-ReAlign Lite','lite'),('Luna + DeepSeek','average'),('Luna','luna'),('DeepSeek','deepseek'),('Haiku','haiku'),('Jev 1.14','jev'),('Liquid d1.1','liquid')]
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def render():
    with (ROOT/'scores.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    rows.sort(key=lambda r:(float(r['reference_technical_quality']),r['case_id']))
    if len(rows)!=72 or len({r['case_id'] for r in rows})!=72:raise ValueError('Expected 72 unique images')
    title=ImageFont.truetype(BOLD,31);body=ImageFont.truetype(FONT,20);bold=ImageFont.truetype(BOLD,20);small=ImageFont.truetype(FONT,16)
    pages=[]
    for page in range(4):
        width,height=1604,3700
        canvas=Image.new('RGB',(width,height),'#edf1f6');d=ImageDraw.Draw(canvas)
        d.text((24,17),f'Photo scoring + Lite latency  |  Page {page+1} / 4',font=title,fill='#14283c')
        d.text((26,61),'Tech / Aest: 1.00 to 5.00  |  Reference: assistant  |  Lite: raw scores, no photography eligibility floor',font=small,fill='#486075')
        d.text((26,84),'Lite time: end-to-end seconds for both scores, including upload and proxy; other assessors: comparable timings unavailable',font=small,fill='#486075')
        ids=[]
        for i,row in enumerate(rows[page*18:(page+1)*18]):
            cid=row['case_id'];ids.append(cid);x=24+(i%3)*525;y=115+(i//3)*595;w=505;h=579
            d.rounded_rectangle((x,y,x+w,y+h),radius=10,fill='white',outline='#cbd7e3',width=2)
            d.text((x+15,y+11),cid,font=bold,fill='#132a40')
            badge='POSTER' if row['known_anchor']=='True' else ('HELD OUT' if row['partition']=='heldout' else 'TUNING')
            d.text((x+w-14-d.textlength(badge,font=small),y+13),badge,font=small,fill='#60788c')
            top=y+44;d.rectangle((x+2,top,x+w-2,top+211),fill='#e3e9f0')
            path=SOURCE/'images'/f'{cid}.png';raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['view_sha256']:raise ValueError('Image hash mismatch '+cid)
            with Image.open(path) as im:
                im=im.convert('RGB');im.thumbnail((w-12,207),Image.Resampling.LANCZOS)
                canvas.paste(im,(x+(w-im.width)//2,top+(211-im.height)//2))
            tab_y=top+218
            for text,rx in [('Assessor',x+17),('Tech',x+347),('Aest',x+431)]:d.text((rx,tab_y),text,font=small,fill='#62798c')
            for j,(name,key) in enumerate(NAMES):
                ry=tab_y+23+j*31
                if key=='reference':d.rectangle((x+9,ry-2,x+w-9,ry+27),fill='#e6f0fb')
                elif key=='lite':d.rectangle((x+9,ry-2,x+w-9,ry+27),fill='#e6f7ed')
                else:d.line((x+12,ry+27,x+w-12,ry+27),fill='#e8edf2',width=1)
                d.text((x+17,ry),name,font=bold if key in ('reference','lite') else body,fill='#163e65' if key=='reference' else '#1c2f40')
                for task,rx in [('technical_quality',x+393),('aesthetics',x+477)]:
                    value=row[key+'_'+task] or 'invalid'
                    d.text((rx-d.textlength(value,font=bold),ry),value,font=bold,fill='#183f65')
            footer=tab_y+23+len(NAMES)*31+5
            d.rounded_rectangle((x+10,footer-1,x+w-10,footer+31),radius=5,fill='#f1f5f8')
            d.text((x+17,footer+3),'Lite end-to-end (both scores)',font=small,fill='#405970')
            latency=f'{float(row["end_to_end_seconds"]):.2f} s'
            d.text((x+w-20-d.textlength(latency,font=bold),footer),latency,font=bold,fill='#175c39')
        target=ROOT/f'comparison-latency-page-{page+1}.png';canvas.save(target,optimize=True)
        pages.append({'page':page+1,'filename':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'width':width,'height':height,'case_ids':ids,'scores':len(ids)*len(NAMES)*2,'latencies':len(ids)})
        print(target.name,'18 images / 288 scores / 18 latencies',flush=True)
    (ROOT/'comparison-latency-validation.json').write_text(json.dumps({'pages':pages,'source_scores_sha256':hashlib.sha256((ROOT/'scores.csv').read_bytes()).hexdigest(),'images':72,'score_cells':1152,'latencies':72,'latency_scope':'Q-ReAlign Lite end-to-end, both tasks per request'},indent=2)+'\n')

if __name__=='__main__':render()
