#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production /ocr_crop character/symbol/hyphen regression suite."""
import argparse,base64,collections,csv,hashlib,json,random,time,unicodedata,urllib.request
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'ocr-benchmark/data/frozen_suite'
MODELS=['v24_1_r4','v24_1_r3_1','v24_1_r3','v24_1_r2_1_6']
OUT=ROOT/'REPORTS/live_quality_2026_11_20/symbol_crop_suite'

def lev(a,b):
    p=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        z=[i]
        for j,y in enumerate(b,1):
            z.append(min(z[-1]+1,p[j]+1,p[j-1]+(x!=y)))
        p=z
    return p[-1]

def norm(s):return unicodedata.normalize('NFC',str(s or '')).strip()

def aligned_symbol_correct(gt, pred, target):
    """Match target symbols along a minimum-edit alignment, not at fixed end position."""
    a,b=norm(gt),norm(pred)
    m,n=len(a),len(b)
    d=[[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):d[i][0]=i
    for j in range(n+1):d[0][j]=j
    for i in range(1,m+1):
        for j in range(1,n+1):
            d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,
                        d[i-1][j-1]+(a[i-1]!=b[j-1]))
    hits=0
    i,j=m,n
    while i or j:
        if i and j and d[i][j]==d[i-1][j-1]+(a[i-1]!=b[j-1]):
            if a[i-1]==b[j-1]==target:hits+=1
            i-=1;j-=1
        elif i and d[i][j]==d[i-1][j]+1:i-=1
        else:j-=1
    return hits==a.count(target)

def case_set(n):
    rng=random.Random(20261120)
    tests=[]
    for fname,field,kind in [('symbol_frozen_14_lines.json','symbol','symbol'),
                              ('hyphen_frozen_multi_context_lines.json','group','hyphen')]:
        groups=collections.defaultdict(list)
        for r in json.loads((BASE/fname).read_text(encoding='utf-8')):
            groups[r[field]].append(r)
        for group,items in sorted(groups.items()):
            for r in rng.sample(items,min(n,len(items))):
                tests.append(dict(id=kind+'_'+str(r['sample_id']),kind=kind,group=group,
                    gt=norm(r['gt']),path=BASE/r['crop_file'],
                    w=r['width'],h=r['height']))
    return tests

def call(test,model,url,timeout):
    im=test['path'].read_bytes()
    body={'image':base64.b64encode(im).decode('ascii'),
          'model':model,'bbox':[0,0,test['w'],test['h']]}
    req=urllib.request.Request(url.rstrip('/')+'/ocr_crop',
       data=json.dumps(body).encode('utf-8'),
       headers={'Content-Type':'application/json',
                'User-Agent':'Mozilla/5.0 (compatible; ChamOCR-Live-Quality-Test/1.0)',
                'Accept':'application/json'},method='POST')
    r=dict(id=test['id'],kind=test['kind'],group=test['group'],model=model,
           gt=test['gt'],sha256=hashlib.sha256(im).hexdigest())
    start=time.monotonic()
    try:
        with urllib.request.urlopen(req,timeout=timeout) as resp:obj=json.load(resp)
        if obj.get('error'): raise RuntimeError(obj['error'])
        pred=norm(obj.get('text'))
        r.update(pred=pred,gt_chars=len(test['gt']),edits=lev(test['gt'],pred),
                 exact=pred==test['gt'],confidence=float(obj.get('calibrated_confidence',0)),
                 symbol_correct=(aligned_symbol_correct(test['gt'],pred,test['group']) if test['kind']=='symbol' else None),
                 hyphen_count_correct=(pred.count('-')==test['gt'].count('-') if test['kind']=='hyphen' else None),
                 error=None)
        r['cer']=r['edits']/max(1,r['gt_chars'])
    except Exception as exc:r['error']=type(exc).__name__+': '+str(exc)[:200]
    r['wall_sec']=round(time.monotonic()-start,3)
    return r

def output(rows,models,out):
    summary={}
    for m in models:
        rs=[r for r in rows if r['model']==m]
        ok=[r for r in rs if not r.get('error')]
        summary[m]={'cases':len(ok),'errors':len(rs)-len(ok),
                    'cer_pct':round(100*sum(r['edits'] for r in ok)/max(1,sum(r['gt_chars'] for r in ok)),2),
                    'full_line_exact_pct':round(100*sum(r['exact'] for r in ok)/max(1,len(ok)),2),
                    'symbols':{},'hyphens':{}}
        for k in sorted({r['group'] for r in ok if r['kind']=='symbol'}):
            q=[r for r in ok if r['kind']=='symbol' and r['group']==k]
            summary[m]['symbols'][k]={'n':len(q),'correct':sum(r['symbol_correct'] for r in q)}
        for k in sorted({r['group'] for r in ok if r['kind']=='hyphen'}):
            q=[r for r in ok if r['kind']=='hyphen' and r['group']==k]
            summary[m]['hyphens'][k]={'n':len(q),'count_correct':sum(r['hyphen_count_correct'] for r in q),
                                      'line_exact':sum(r['exact'] for r in q)}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    cols=['id','kind','group','model','gt','pred','gt_chars','edits','cer','exact',
          'symbol_correct','hyphen_count_correct','confidence','wall_sec','error','sha256']
    with (out/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
    lines=['# Live symbol and hyphen crop test','',
        'Endpoint: /ocr_crop (recognition-only, not line detection)',
        'Report UTC: '+datetime.now(timezone.utc).isoformat(),
        'Requests collected: '+str(len(rows)),'',
        '| Model | Valid crops | Character CER | Exact full lines | Request failures |',
        '|---|---:|---:|---:|---:|']
    for m,s in summary.items():
        lines.append('| %s | %s | %s%% | %s%% | %s |'%(
            m,s['cases'],s['cer_pct'],s['full_line_exact_pct'],s['errors']))
    lines+=['','## Each required symbol (minimum-edit alignment)','',
         '| Symbol | '+ ' | '.join(models)+' |',
         '|---|'+ '|---:|'*len(models)]
    allsym=sorted(set(k for s in summary.values() for k in s['symbols']))
    for sym in allsym:
        v=[('%d/%d'%(summary[m]['symbols'].get(sym,{}).get('correct',0),
                         summary[m]['symbols'].get(sym,{}).get('n',0))) for m in models]
        lines.append('| '+('BACKSLASH' if sym=='\\' else sym)+' | '+' | '.join(v)+' |')
    lines+=['','## Hyphen contexts (correct hyphen count)','',
         '| Context | '+' | '.join(models)+' |',
         '|---|'+'---:|'*len(models)]
    allhy=sorted(set(k for s in summary.values() for k in s['hyphens']))
    for grp in allhy:
        v=['%d/%d'%(summary[m]['hyphens'].get(grp,{}).get('count_correct',0),
                     summary[m]['hyphens'].get(grp,{}).get('n',0)) for m in models]
        lines.append('| '+grp+' | '+' | '.join(v)+' |')
    lines+=['','**Limitations:** Each crop is a synthetic pre-segmented 48px textline.',
       'Symbol correctness is checked by minimum-edit character alignment against ground truth.',
       'Hyphen count correctness does not prove correct character placement.',
       'Results are not interchangeable with full-page end-to-end CER.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',default='https://ocr.cham.asia')
    p.add_argument('--models',default=','.join(MODELS))
    p.add_argument('--per-group',type=int,default=2)
    p.add_argument('--timeout',type=int,default=65)
    a=p.parse_args()
    models=a.models.split(','); OUT.mkdir(parents=True,exist_ok=True)
    j=OUT/'results.jsonl';old={}
    if j.exists():
        for line in j.read_text(encoding='utf-8').splitlines():
            try:
                r=json.loads(line);old[(r['id'],r['model'])]=r
            except ValueError:pass
    rows=list(old.values());new=0
    # Rescore previously saved predictions using the current alignment rubric.
    for r in rows:
        if r.get('kind')=='symbol' and not r.get('error'):
            r['symbol_correct']=aligned_symbol_correct(r['gt'],r['pred'],r['group'])
    if rows:
        with j.open('w',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    for test in case_set(a.per_group):
        for model in models:
            if (test['id'],model) in old:continue
            r=call(test,model,a.url,a.timeout);rows.append(r);new+=1
            with j.open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            print('[%d] %s %s match=%s error=%s'%(new,test['id'],model,
                  r.get('symbol_correct',r.get('hyphen_count_correct')),r['error']),flush=True)
            output(rows,models,OUT)
    print('DONE',json.dumps(output(rows,models,OUT),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
