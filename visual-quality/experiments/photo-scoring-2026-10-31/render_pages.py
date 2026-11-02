#!/usr/bin/env python3
"""Render the current derived comparison as four readable PNG pages without redrawing photos."""
from canonical_scores import load_scores
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
NAMES=[('Reference','reference'),('Luna','luna'),('DeepSeek','deepseek'),('Weighted ensemble','photo_ensemble'),('Haiku','haiku'),('Jev 1.14','jev'),('Liquid d1.1','liquid')]
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def render():
    rows=load_scores()
    rows.sort(key=lambda r:(float(r['reference_technical_quality']),r['case_id']))
    if len(rows)!=72 or len({r['case_id'] for r in rows})!=72: raise ValueError('Expected 72 distinct images')
    body=ImageFont.truetype(FONT,21); bold=ImageFont.truetype(BOLD,21)
    title=ImageFont.truetype(BOLD,32); small=ImageFont.truetype(FONT,16)
    page_info=[]
    for page in range(4):
        width,height=1544,3278
        canvas=Image.new('RGB',(width,height),'#edf1f6'); draw=ImageDraw.Draw(canvas)
        draw.text((24,18),f'Photo scoring comparison  ·  Page {page+1} / 4',font=title,fill='#14283c')
        draw.text((26,61),'Scores: technical / aesthetic  ·  Reference = assistant  ·  Ensemble: 37.5% Luna + 37.5% DeepSeek + 25% Liquid',font=small,fill='#4b6175')
        ids=[]
        for i,row in enumerate(rows[page*18:(page+1)*18]):
            ids.append(row['case_id']); x=24+(i%3)*506; y=91+(i//3)*530; w=484; h=512
            draw.rounded_rectangle((x,y,x+w,y+h),radius=9,fill='white',outline='#cbd7e3',width=2)
            draw.text((x+16,y+11),row['case_id'],font=bold,fill='#132a40')
            badge='POSTER' if row['known_anchor']=='True' else ('HELD OUT' if row['partition']=='heldout' else 'TUNING')
            badge_width=draw.textlength(badge,font=small)
            draw.text((x+w-16-badge_width,y+14),badge,font=small,fill='#62798c')
            top=y+44; draw.rectangle((x+2,top,x+w-2,top+203),fill='#e3e9f0')
            path=(ROOT/row['image_path']).resolve(); raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['sha256']: raise ValueError('Image bytes changed: '+row['case_id'])
            with Image.open(path) as im:
                im=im.convert('RGB'); im.thumbnail((w-10,198),Image.Resampling.LANCZOS)
                canvas.paste(im,(x+(w-im.width)//2,top+(203-im.height)//2))
            tab_y=top+211
            draw.text((x+17,tab_y),'Assessor',font=small,fill='#6b7f90')
            draw.text((x+323,tab_y),'Tech',font=small,fill='#6b7f90')
            draw.text((x+406,tab_y),'Aest',font=small,fill='#6b7f90')
            for j,(name,key) in enumerate(NAMES):
                ry=tab_y+25+j*30
                if key=='reference': draw.rectangle((x+9,ry-2,x+w-9,ry+27),fill='#e6f0fb')
                else: draw.line((x+12,ry+27,x+w-12,ry+27),fill='#e8edf2',width=1)
                draw.text((x+17,ry),name,font=body if key!='reference' else bold,fill='#163e65' if key=='reference' else '#1c2f40')
                for label,rx in [('technical_quality',x+369),('aesthetics',x+452)]:
                    value=row[key+'_'+label] or 'invalid'
                    text_width=draw.textlength(value,font=bold)
                    draw.text((rx-text_width,ry),value,font=bold,fill='#183f65')
        name=f'comparison-page-{page+1}.png'; out=ROOT/name; canvas.save(out,optimize=True)
        page_info.append({'page':page+1,'file':name,'width':width,'height':height,'case_ids':ids,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'scores':len(ids)*len(NAMES)*2})
        print(name,canvas.size,'images',len(ids),'score cells',len(ids)*len(NAMES)*2,flush=True)
    (ROOT/'image-pages-validation.json').write_text(json.dumps({'pages':page_info,'total_images':72,'total_score_cells':1008,'source_csv_sha256':hashlib.sha256((ROOT/'scores-72.csv').read_bytes()).hexdigest()},indent=2)+'\n')

if __name__=='__main__': render()
