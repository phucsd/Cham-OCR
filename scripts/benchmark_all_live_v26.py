#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed, resumable Cham OCR production API benchmark, eight advertised models."""
import argparse,collections,concurrent.futures,hashlib,json,statistics,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import live_website_quality_benchmark as qa
from live_qa_frozen_replay import cases_from_manifest,MANIFEST as FROZEN
MODELS=['v24_1_r4','v25','auto','v24_1_r3_1','v24_1_r2_1_6','v24','v23','vi']
BASE=ROOT/'REPORTS/live_v26_audit_2026_11_22'
GEO=ROOT/'ocr-benchmark/data/live_geometry_multifont_2026_11_21/manifest.json'
def getcases(suite):
    if suite=='frozen': return list(cases_from_manifest(json.loads(FROZEN.read_text(encoding='utf-8'))))
    if suite=='geometry':
        obj=json.loads(GEO.read_text(encoding='utf-8'))
        return [dict(id=x['id'],tier=x['group'],kind=x['kind'],font=';'.join(x['font_names']),
                     path=ROOT/x['image_relpath'],gt=x['gt_lines'])
                for x in obj['cases']]
    if suite in ('stress','vietnamese'):
        items=json.loads((ROOT/'ocr-benchmark/data/live_v26_2026_11_22/manifest.json').read_text(encoding='utf-8'))['cases']
        return [dict(id=x['id'],tier=x['tier'],kind=suite,font=x['origin'],
                     path=ROOT/x['image_relpath'],gt=x['gt_lines']) for x in items if x['suite']==suite]
    raise ValueError('unknown suite '+suite)
def runone(c,m,timeout):
    r=qa.call(c,m,'https://ocr.cham.asia',timeout)
    if m=='auto' and not r.get('error'):
        r['auto_recognizer_first']=r.get('actual_model')
        r['actual_model']='auto'
    r['suite']=c['suite']
    return r
def summarize(rs,models):
    s={'generated_utc':datetime.now(timezone.utc).isoformat(),
       'method':'POST /ocr, dbnet, rows, all models same image+GT',
       'models':{},'by_tier':{},'total_requests':len(rs),'errors':[]}
    for m in models:
        z=[r for r in rs if r['model']==m]
        good=[r for r in z if not r.get('error') and r.get('actual_model')==m]
        def aggregate(g):
            if not g:return {'n':0}
            chars=sum(r.get('gt_chars',0) for r in g)
            words=sum(r.get('gt_words',0) for r in g)
            return dict(n=len(g),micro_cer_pct=round(100*sum(r['edit_chars'] for r in g)/max(1,chars),2),
              micro_wer_pct=round(100*sum(r['edit_words'] for r in g)/max(1,words),2),
              exact_pct=round(100*sum(r['exact'] for r in g)/len(g),2),
              count_match_pct=round(100*sum(r['gt_count']==r['pred_count'] for r in g)/len(g),2),
              med_backend_s=round(statistics.median(float(r.get('backend_sec') or 0) for r in g),3),
              med_wall_s=round(statistics.median(r['wall_sec'] for r in g),3),
              gt_chars=chars,edits=sum(r['edit_chars'] for r in g))
        s['models'][m]=aggregate([r for r in good if str(r['tier']).lower() not in ('0','negative')])
        s['models'][m].update(calls=len(z),errors=len(z)-len(good),
         model_weights=sorted(set(x['fingerprint'] for x in good if x.get('fingerprint'))),
         blank_correct=sum(r['pred_count']==0 for r in good if str(r['tier']).lower() in ('0','negative')),
         blanks_tested=sum(str(r['tier']).lower() in ('0','negative') for r in good))
        s['by_tier'][m]={str(k):aggregate([r for r in good if str(r['tier'])==str(k)])
            for k in sorted(set(str(r['tier']) for r in good))}
    s['errors']=[{'id':r['id'],'model':r['model'],'reason':r.get('error'),'actual':r.get('actual_model')}
         for r in rs if r.get('error') or r.get('actual_model')!=r['model']]
    return s
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--suite',default='frozen',choices=['frozen','geometry','stress','vietnamese'])
    ap.add_argument('--models',default=','.join(MODELS))
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--timeout',type=int,default=100)
    ap.add_argument('--max-new-requests',type=int,default=0)
    ap.add_argument('--per-tier',type=int,default=0,help='Deterministic representative cases per tier')
    a=ap.parse_args();models=a.models.split(',')
    cases=getcases(a.suite)
    if a.per_tier:
        by_tier={}
        for item in cases:by_tier.setdefault(str(item['tier']),[]).append(item)
        selected=[]
        for tier,items in sorted(by_tier.items()):
            items=sorted(items,key=lambda v:v['id'])
            if tier=='negative':selected+=items;continue
            if len(items)<=a.per_tier:selected+=items
            else:
                positions=[round(i*(len(items)-1)/max(1,a.per_tier-1)) for i in range(a.per_tier)]
                selected += [items[k] for k in dict.fromkeys(positions)]
        cases=selected
    for c in cases:c['suite']=a.suite
    out=BASE/a.suite;out.mkdir(parents=True,exist_ok=True)
    j=out/'results.jsonl'
    old={}
    if j.exists():
        for ln in j.read_text(encoding='utf-8').splitlines():
            try:
                r=json.loads(ln);old[r['id'],r['model']]=r
            except ValueError: pass
    pending=[(c,m) for c in cases for m in models if (c['id'],m) not in old]
    if a.max_new_requests:pending=pending[:a.max_new_requests]
    print('START',a.suite,'cases',len(cases),'models',len(models),'pending',len(pending),flush=True)
    for c in cases:
        if 'path' in c:
            assert c['path'].exists(),str(c['path'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        fut={pool.submit(runone,c,m,a.timeout):(c['id'],m) for c,m in pending}
        for idx,f in enumerate(concurrent.futures.as_completed(fut),1):
            cid,m=fut[f]
            try:r=f.result()
            except Exception as e:
                r={'id':cid,'model':m,'tier':'unknown','error':repr(e),'suite':a.suite,'wall_sec':None}
            old[(cid,m)]=r
            with j.open('a',encoding='utf-8') as h:h.write(json.dumps(r,ensure_ascii=False)+'\n')
            if idx%8==0 or r.get('error'):
                print('PROGRESS',idx,'/',len(pending),cid,m,'CER',round(r.get('cer',0)*100,2) if r.get('cer') is not None else '-', 'ERR',r.get('error'),flush=True)
            if idx%16==0:
                (out/'summary.json').write_text(json.dumps(summarize(list(old.values()),models),ensure_ascii=False,indent=2),encoding='utf-8')
    result=summarize(list(old.values()),models)
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',a.suite,'counts',[(m,result['models'][m]['n']) for m in models],'errors',len(result['errors']),flush=True)
if __name__=='__main__':main()
