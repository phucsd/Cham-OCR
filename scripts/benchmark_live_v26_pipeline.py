#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controlled /ocr segmentation & reading-order comparison on fixed GT pages."""
import base64,collections,concurrent.futures,hashlib,json,statistics,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_all_live_v26 import MODELS as ALL_MODELS,BASE
MODELS=['v24_1_r4','v25','auto','v24_1_r3_1']
import live_website_quality_benchmark as qa
sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
M=json.loads((ROOT/'ocr-benchmark/data/live_geometry_multifont_2026_11_21/manifest.json').read_text(encoding='utf-8'))['cases']
F=json.loads((ROOT/'REPORTS/live_quality_2026_11_20/frozen_test_manifest.json').read_text(encoding='utf-8'))['cases']
s1=next(x for x in F if x['tier']==1)
s2=next(x for x in F if x['tier']==6)
s3=next(x for x in M if x['group']=='mixed_fonts')
C=[dict(id=x['id'],path=ROOT/x['path'],gt=x['gt_lines']) for x in [s1,s2]]
C.append(dict(id=s3['id'],path=ROOT/s3['image_relpath'],gt=s3['gt_lines']))
jobs=[(c,m,method,order) for c in C for m in MODELS for method,order in
      [('valley','rows'),('adaptive','rows'),('dbnet','auto')]]
OUT=BASE/'pipeline';OUT.mkdir(parents=True,exist_ok=True);J=OUT/'results.jsonl'
old={}
if J.exists():
    for l in J.read_text(encoding='utf-8').splitlines():
        try:
            r=json.loads(l);old[r['id'],r['model'],r['method'],r['reading_order']]=r
        except ValueError:pass
def go(c,m,method,order):
    blob=c['path'].read_bytes()
    body={'image':base64.b64encode(blob).decode(),'model':m,'method':method,'reading_order':order}
    req=urllib.request.Request('https://ocr.cham.asia/ocr',data=json.dumps(body).encode(),
      headers={'Content-Type':'application/json','User-Agent':'Mozilla/5.0 (ChamOCR pipeline QA)'})
    r=dict(id=c['id'],model=m,method=method,reading_order=order,image_sha256=hashlib.sha256(blob).hexdigest())
    t=time.monotonic()
    try:
        with urllib.request.urlopen(req,timeout=110) as h:obj=json.load(h)
        if obj.get('error'):raise ValueError(str(obj['error']))
        gt='\n'.join(map(qa.norm,c['gt']))
        pr='\n'.join(qa.norm(x.get('text','')) for x in obj['lines'])
        r.update(ground_truth_lines=len(c['gt']),pred_lines=len(obj['lines']),
            cer_pct=round(100*qa.edit(gt,pr)/max(1,len(gt)),2),
            wer_pct=round(100*qa.edit(gt.split(),pr.split())/max(1,len(gt.split())),2),
            version=(obj.get('model_info') or {}).get('model'),
            backend_s=(obj.get('profiler') or {}).get('backend_total_sec'),error=None)
    except Exception as e:r['error']=repr(e)[:200]
    r['wall_s']=round(time.monotonic()-t,2)
    return r
todo=[x for x in jobs if (x[0]['id'],x[1],x[2],x[3]) not in old]
print('START PIPELINE',len(todo),'new',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
    ff={pool.submit(go,*x):x for x in todo}
    for i,f in enumerate(concurrent.futures.as_completed(ff),1):
        r=f.result();old[(r['id'],r['model'],r['method'],r['reading_order'])]=r
        with J.open('a',encoding='utf-8') as h:h.write(json.dumps(r,ensure_ascii=False)+'\n')
        if i%8==0 or r.get('error'):print('PIPELINE',i,'/',len(todo),r['method'],r['model'],'CER',r.get('cer_pct'),'ERR',r.get('error'),flush=True)
summary={}
for method in ['valley','adaptive','dbnet']:
    rr=[r for r in old.values() if r['method']==method]
    summary[method]={m:{'n':len(a),'errors':sum(bool(x.get('error')) for x in a),
        'mean_cer_pct':round(statistics.mean(x['cer_pct'] for x in a if x.get('cer_pct') is not None),2) if any(x.get('cer_pct') is not None for x in a) else None,
        'median_wall_s':round(statistics.median(x['wall_s'] for x in a),2) if a else None}
        for m in MODELS for a in [[x for x in rr if x['model']==m]]}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('DONE PIPELINE',len(old),flush=True)
