#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze all audited 20-Nov Cham OCR cases and replay on new production models.

- Freeze manifest from original raw results only once, with sha256 per image.
- Reuse the same set for each new model to keep fair paired comparisons.
- dry-run only validates manifest and never calls a server.
- Replay records request- and model-fingerprint-level provenance.
"""
import argparse, base64, hashlib, io, json
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
import live_website_quality_benchmark as qa

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'REPORTS/live_quality_2026_11_20'
MANIFEST=SOURCE/'frozen_test_manifest.json'
OUT=ROOT/'REPORTS/live_quality_frozen_replay'

def blank_bytes():
    img=Image.new('RGB',(900,300),'white')
    f=io.BytesIO();img.save(f,format='PNG')
    return f.getvalue()

def create_manifest():
    raw=[json.loads(s) for s in (SOURCE/'results.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    ids=sorted({r['id'] for r in raw})
    gt={v['sample_id']:v for v in json.loads(qa.GT.read_text(encoding='utf-8'))}
    real=next((r for r in json.loads(qa.REAL.read_text(encoding='utf-8'))['documents']
               if r['doc_id']=='harum_palei_01'),None)
    cases=[]
    for cid in ids:
        if cid.startswith('synthetic_'):
            item=gt[int(cid.split('_')[-1])]
            rel='ocr-benchmark/'+item['image_path']
            lines=[v['text'] for v in item['lines']]
            tier=item['level'];kind='synthetic';font=item.get('font_name')
        elif cid=='real_harum_palei_01':
            if not real: raise RuntimeError('Real image manifest missing')
            rel=real['page_relpath'];lines=[v['gt'] for v in real['lines']]
            tier=6;kind='real_camera';font=None
        elif cid=='negative_blank':
            rel=None;lines=[];tier=0;kind='negative';font=None
        else:
            raise RuntimeError('Unknown frozen case '+cid)
        blob=(ROOT/rel).read_bytes() if rel else blank_bytes()
        cases.append(dict(id=cid,tier=tier,kind=kind,font=font,path=rel,
             gt_lines=lines,image_sha256=hashlib.sha256(blob).hexdigest()))
    cases.sort(key=lambda c:(c['tier'],c['id']))
    bundle=dict(schema_version='1.0.0',frozen_at=datetime.now(timezone.utc).isoformat(),
          source='2026-11-20 production live OCR results',
          ground_truth_json_sha256=hashlib.sha256(qa.GT.read_bytes()).hexdigest(),
          n_cases=len(cases),cases=cases)
    MANIFEST.write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding='utf-8')
    return bundle

def cases_from_manifest(bundle):
    for item in bundle['cases']:
        blob=(ROOT/item['path']).read_bytes() if item['path'] else blank_bytes()
        sha=hashlib.sha256(blob).hexdigest()
        if sha!=item['image_sha256']:
            raise RuntimeError('Input SHA mismatch on '+item['id'])
        case=dict(id=item['id'],tier=item['tier'],kind=item['kind'],font=item['font'],
                  gt=item['gt_lines'])
        if item['path']:case['path']=ROOT/item['path']
        else:case['blob']=blob
        yield case

def run():
    p=argparse.ArgumentParser()
    p.add_argument('--models',default='v24_1_r4,v24_1_r3_1,v24_1_r3,v24_1_r2_1_6')
    p.add_argument('--url',default='https://ocr.cham.asia')
    p.add_argument('--timeout',type=int,default=90)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--max-new-requests',type=int,default=0)
    p.add_argument('--output',default=str(OUT))
    a=p.parse_args()
    bundle=json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else create_manifest()
    cases=list(cases_from_manifest(bundle))
    print('FROZEN_MANIFEST_OK',len(cases),'cases',len(set(c['id'] for c in cases)),'unique',flush=True)
    if a.dry_run: return
    models=[m.strip() for m in a.models.split(',') if m.strip()]
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    j=output/'results.jsonl';old={}
    if j.exists():
        for s in j.read_text(encoding='utf-8').splitlines():
            try:r=json.loads(s);old[(r['id'],r['model'])]=r
            except ValueError:continue
    rows=list(old.values())
    meta=dict(url=a.url,models=models,seed=20261120,
       frozen_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
       source_cases=bundle['n_cases'],created_utc=datetime.now(timezone.utc).isoformat())
    new=0
    for case in cases:
        for model in models:
            if (case['id'],model) in old:continue
            r=qa.call(case,model,a.url,a.timeout);rows.append(r);old[(r['id'],model)]=r
            with j.open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            new+=1
            print('[%s] %s %s CER=%s error=%s'%(new,case['id'],model,r.get('cer'),r.get('error')),flush=True)
            qa.export(output,rows,models,meta)
            if a.max_new_requests and new>=a.max_new_requests:return
    qa.export(output,rows,models,meta)
    print('COMPLETE',len(rows),'requests',flush=True)
if __name__=='__main__':run()
