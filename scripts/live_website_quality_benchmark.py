#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cham OCR Studio production E2E OCR benchmark. Read-only against /ocr."""
import argparse, base64, collections, csv, hashlib, io, json, random
import statistics, time, unicodedata, urllib.request
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
GT=ROOT/'ocr-benchmark/data/benchmark_gt.json'
REAL=ROOT/'ocr-benchmark/data/realscan_gold_v2/manifest.json'
MODELS=['v24_1_r4','v24_1_r3_1','v24_1_r3','v24_1_r2_1_6']
TIER={0:'Blank negative',1:'Standard',2:'Aged paper',3:'Narrow gap',4:'Wavy',5:'Extreme overlap',6:'Real camera'}
SIGNS=set('!@#$%^&*()\"[]{}\\|<>?;\':/.,-–—꩝꩞꩟') | {chr(c) for c in range(0xAA50,0xAA5A)}

def edit(a,b):
    if not a: return len(b)
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):
            row.append(min(row[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=row
    return prev[-1]

def norm(s):
    return unicodedata.normalize('NFC',str(s or '')).strip()

def samples(count,real):
    src=json.loads(GT.read_text(encoding='utf-8'))
    rng=random.Random(20261120)
    items=[]
    for lv in range(1,6):
        pool=sorted((x for x in src if x['level']==lv),key=lambda x:x['sample_id'])
        for x in rng.sample(pool,min(count,len(pool))):
            items.append(dict(id='synthetic_%03d'%x['sample_id'],tier=lv,kind='synthetic',
                path=ROOT/'ocr-benchmark'/x['image_path'],
                gt=[v['text'] for v in x['lines']],font=x.get('font_name')))
    if real:
        for x in json.loads(REAL.read_text(encoding='utf-8'))['documents']:
            if x['doc_id']=='harum_palei_01':
                items.append(dict(id='real_harum_palei_01',tier=6,kind='real_camera',
                    path=ROOT/x['page_relpath'],gt=[v['gt'] for v in x['lines']],font=None))
                break
    from PIL import Image
    img=Image.new('RGB',(900,300),'white')
    buf=io.BytesIO();img.save(buf,format='PNG')
    items.append(dict(id='negative_blank',tier=0,kind='negative',
                      blob=buf.getvalue(),gt=[],font=None))
    return items

def call(sample,model,url,timeout):
    blob=sample.get('blob') or sample['path'].read_bytes()
    gt=[norm(x) for x in sample['gt']]
    body=dict(image=base64.b64encode(blob).decode(),model=model,method='dbnet',reading_order='rows')
    req=urllib.request.Request(url.rstrip('/')+'/ocr',
        data=json.dumps(body).encode('utf-8'),
        headers={'Content-Type':'application/json','Accept':'application/json',
                 'User-Agent':'Mozilla/5.0 (compatible; ChamOCR-Live-Quality-Test/1.0)'},
        method='POST')
    record=dict(id=sample['id'],tier=sample['tier'],kind=sample['kind'],model=model,
                font=sample['font'],gt_lines=gt,
                input_sha256=hashlib.sha256(blob).hexdigest())
    start=time.monotonic()
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:
            obj=json.load(response)
        if obj.get('error'): raise RuntimeError(obj['error'])
        pred=[norm(v.get('text')) for v in obj.get('lines',[])]
        g='\n'.join(gt);p='\n'.join(pred)
        rinfo=(obj.get('model_info') or {}).get('recognizers') or []
        meta=rinfo[0] if rinfo else {}
        cntg=collections.Counter(ch for ch in g if ch in SIGNS)
        cntp=collections.Counter(ch for ch in p if ch in SIGNS)
        vals=[float(v.get('calibrated_confidence',v.get('confidence',0))) for v in obj.get('lines',[])]
        record.update(pred_lines=pred,gt_count=len(gt),pred_count=len(pred),
            gt_chars=len(g),edit_chars=edit(g,p),gt_words=len(g.split()),
            edit_words=edit(g.split(),p.split()),exact=g==p,
            special_gt=sum(cntg.values()),
            special_hit_upper_bound=sum(min(cnt,cntp[ch]) for ch,cnt in cntg.items()),
            actual_model=meta.get('version'),fingerprint=meta.get('weights_sha256_16'),
            detector_fingerprint=(obj.get('model_info') or {}).get('detector',{}).get('cham_dbnet_sha256_16'),
            backend_sec=(obj.get('profiler') or {}).get('backend_total_sec'),
            confidence_mean=round(statistics.mean(vals),4) if vals else None,
            status=(obj.get('document') or {}).get('status'),error=None)
        record['cer']=record['edit_chars']/max(1,record['gt_chars'])
        record['wer']=record['edit_words']/max(1,record['gt_words'])
    except Exception as e:
        record['error']=type(e).__name__+': '+str(e)[:350]
    record['wall_sec']=round(time.monotonic()-start,3)
    return record

def summarize(rs):
    ok=[r for r in rs if r.get('error') is None and r.get('actual_model')==r['model']]
    if not ok: return dict(n=0,attempts=len(rs),errors=len(rs))
    gchars=sum(r['gt_chars'] for r in ok);gwords=sum(r['gt_words'] for r in ok)
    return dict(n=len(ok),attempts=len(rs),errors=len(rs)-len(ok),
        cer_pct=round(100*sum(r['edit_chars'] for r in ok)/max(1,gchars),2),
        wer_pct=round(100*sum(r['edit_words'] for r in ok)/max(1,gwords),2),
        exact_pct=round(100*sum(r['exact'] for r in ok)/len(ok),2),
        line_count_match_pct=round(100*sum(r['gt_count']==r['pred_count'] for r in ok)/len(ok),2),
        mean_abs_line_delta=round(statistics.mean(abs(r['gt_count']-r['pred_count']) for r in ok),2),
        median_backend_sec=round(statistics.median(float(r['backend_sec'] or 0) for r in ok),2),
        median_wall_sec=round(statistics.median(r['wall_sec'] for r in ok),2),
        gt_chars=gchars,gt_words=gwords)

def export(out,records,models,meta):
    summary={'meta':meta,'overall':{},'tiers':{},'failed':[],'fingerprints':{}}
    for model in models:
        rs=[r for r in records if r['model']==model]
        summary['overall'][model]=summarize([r for r in rs if r['tier']!=0])
        summary['tiers'][model]={str(lv):summarize([r for r in rs if r['tier']==lv])
                                 for lv in range(7) if any(r['tier']==lv for r in rs)}
        summary['fingerprints'][model]=sorted({str(r.get('fingerprint')) for r in rs if r.get('fingerprint')})
    summary['failed']=[dict(id=r['id'],model=r['model'],error=r.get('error'),
                            actual_model=r.get('actual_model')) for r in records
                       if r.get('error') or r.get('actual_model')!=r['model']]
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    cols=['id','kind','tier','model','actual_model','fingerprint','gt_count','pred_count',
          'gt_chars','edit_chars','cer','gt_words','edit_words','wer','exact',
          'special_gt','special_hit_upper_bound','backend_sec','wall_sec',
          'confidence_mean','status','error','input_sha256']
    with (out/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(records)
    lines=['# Live Cham OCR Quality Benchmark — 2026-11-20','',
           'Endpoint: '+meta['url']+' | mode: dbnet | reading order: rows',
           'UTC report time: '+datetime.now(timezone.utc).isoformat(),
           'POST /ocr calls recorded: '+str(len(records)),'',
           '## Overall (excluding blank images)','',
           '| Model | Valid cases | Micro CER | Micro WER | Exact line count | Median backend | Errors |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for m,st in summary['overall'].items():
        lines.append('| %s | %s | %s%% | %s%% | %s%% | %s s | %s |'%(
            m,st['n'],st.get('cer_pct','—'),st.get('wer_pct','—'),
            st.get('line_count_match_pct','—'),st.get('median_backend_sec','—'),st['errors']))
    lines += ['','## Difficulty tiers','',
              '| Model | Tier | Cases | CER | WER | Exact line count |',
              '|---|---|---:|---:|---:|---:|']
    for m,levels in summary['tiers'].items():
        for k,st in levels.items():
            lines.append('| %s | %s: %s | %s | %s%% | %s%% | %s%% |'%(
                m,k,TIER[int(k)],st['n'],st.get('cer_pct','—'),
                st.get('wer_pct','—'),st.get('line_count_match_pct','—')))
    lines += ['','## Method and limitations','',
      '- Micro CER: summed Levenshtein character edits over summed ground-truth characters, with NFC normalization, preserving line breaks.',
      '- Micro WER: tokenized by whitespace; interpretation is weak for Cham scriptio continua.',
      '- Exact line count is not a true line-detection recall metric; split and merge errors can cancel.',
      '- Symbol hits in CSV are upper bounds based on occurrence counts, not position-aligned true positives.',
      '- Synthetic cases are not proven independent from model training distribution. Treat real-camera results separately.',
      '- A valid row requires actual_model == requested model, to avoid silent model fallback.',
      '- Wall latency includes network; backend does not. Initial model calls may include cold start.',
      '- Raw ground truth and predictions are retained per request in results.jsonl.',
      '','## Unexpected errors or model mismatch','']
    for r in summary['failed'][:30]:
        lines.append('- %s / %s: %s (actual=%s)'%(r['model'],r['id'],r.get('error'),r.get('actual_model')))
    if not summary['failed']: lines.append('None recorded.')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',default='https://ocr.cham.asia')
    p.add_argument('--models',default=','.join(MODELS))
    p.add_argument('--per-level',type=int,default=4)
    p.add_argument('--include-real',action='store_true')
    p.add_argument('--timeout',type=int,default=90)
    p.add_argument('--max-new-requests',type=int,default=0)
    p.add_argument('--output',default=str(ROOT/'REPORTS/live_quality_2026_11_20'))
    a=p.parse_args()
    models=[m.strip() for m in a.models.split(',') if m.strip()]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    j=out/'results.jsonl'
    old={}
    if j.exists():
        for line in j.read_text(encoding='utf-8').splitlines():
            try:
                r=json.loads(line);old[(r['id'],r['model'])]=r
            except json.JSONDecodeError:pass
    records=list(old.values())
    meta=dict(url=a.url,seed=20261120,models=models,per_level=a.per_level,
              include_real=a.include_real,ground_truth_sha256=hashlib.sha256(GT.read_bytes()).hexdigest(),
              recorded_utc=datetime.now(timezone.utc).isoformat())
    new=0
    for case in samples(a.per_level,a.include_real):
        for model in models:
            if (case['id'],model) in old:continue
            r=call(case,model,a.url,a.timeout)
            records.append(r);old[(r['id'],r['model'])]=r
            with j.open('a',encoding='utf-8') as f:
                f.write(json.dumps(r,ensure_ascii=False)+'\n')
            new+=1
            print('[%d] %s | %s | CER %s | lines %s | %.1fs | error %s'%(
                new,case['id'],model,r.get('cer'),r.get('pred_count'),
                r['wall_sec'],r.get('error')),flush=True)
            export(out,records,models,meta)
            if a.max_new_requests and new>=a.max_new_requests:
                print('LIMIT REACHED',flush=True);return
    print('DONE',json.dumps(export(out,records,models,meta)['overall'],ensure_ascii=False),flush=True)

if __name__=='__main__':main()
