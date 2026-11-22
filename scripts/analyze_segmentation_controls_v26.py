import json,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'REPORTS/live_v26_audit_2026_11_22'
files=['frozen/results.jsonl','geometry/results.jsonl','pipeline/results.jsonl']
raw=[]
for name in files:
 f=O/name
 if f.exists():raw+=[json.loads(x) for x in f.read_text(encoding='utf-8').splitlines()]
refs=sorted({r['id'] for r in raw if 'method' in r and r['method'] in ('dbnet','valley','adaptive')})
codes=['v24_1_r4','v25','auto','v24_1_r3_1']
out={'cases':refs,'models':{}}
for m in codes:
 v={}
 for id in refs:
  r=[x for x in raw if x['id']==id and x['model']==m and not x.get('error')]
  base=[x for x in r if x.get('method') in (None,'dbnet') and x.get('reading_order') in (None,'rows')]
  alt=[x for x in r if x.get('method')=='dbnet' and x.get('reading_order')=='auto']
  v[id]={'dbnet_rows':round(100*base[-1]['cer'],2) if base else None,
         'dbnet_auto':alt[-1].get('cer_pct') if alt else None}
 valid=[z for z in v.values() if z['dbnet_rows'] is not None and z['dbnet_auto'] is not None]
 out['models'][m]={'cases':v,'matched':len(valid),
   'mean_dbnet_rows_pct':round(statistics.mean(x['dbnet_rows'] for x in valid),2) if valid else None,
   'mean_dbnet_auto_pct':round(statistics.mean(x['dbnet_auto'] for x in valid),2) if valid else None}
 print(m,'paired',len(valid),'dbnet rows',out['models'][m]['mean_dbnet_rows_pct'],
       'dbnet auto',out['models'][m]['mean_dbnet_auto_pct'])
(O/'pipeline/dbnet_rows_vs_auto.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
