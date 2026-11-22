#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen punctuation/hyphen crop regression for all production model IDs."""
import argparse,concurrent.futures,json,sys,statistics
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import live_symbol_crop_benchmark as sc
from benchmark_all_live_v26 import MODELS as ALL_MODELS
OUT=ROOT/'REPORTS/live_v26_audit_2026_11_22/crops'
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--per-group',type=int,default=3)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--models',default=','.join(ALL_MODELS))
    a=p.parse_args();models=a.models.split(',')
    tests=sc.case_set(a.per_group)
    OUT.mkdir(parents=True,exist_ok=True)
    j=OUT/'results.jsonl';done={}
    if j.exists():
        for t in j.read_text(encoding='utf-8').splitlines():
            try:
                v=json.loads(t);done[(v['id'],v['model'])]=v
            except ValueError:pass
    jobs=[(c,m) for c in tests for m in models if (c['id'],m) not in done]
    print('START CROPS',len(tests),'cases',len(models),'models',len(jobs),'requests',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures={ex.submit(sc.call,c,m,'https://ocr.cham.asia',85):(c['id'],m) for c,m in jobs}
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            cid,m=futures[f];r=f.result()
            done[(cid,m)]=r
            with j.open('a',encoding='utf-8') as h:h.write(json.dumps(r,ensure_ascii=False)+'\n')
            if i%15==0 or r.get('error'):
                print('PROGRESS CROPS',i,'/',len(jobs),cid,m,'error',r.get('error'),flush=True)
            if i%42==0:
                sc.output(list(done.values()),ALL_MODELS,OUT)
    res=sc.output(list(done.values()),ALL_MODELS,OUT)
    print('DONE CROPS',json.dumps({m:{k:res[m][k] for k in ('cases','cer_pct','full_line_exact_pct','errors')} for m in ALL_MODELS}),flush=True)
if __name__=='__main__':main()
