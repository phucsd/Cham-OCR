import json,unicodedata,statistics
from pathlib import Path
try:import regex
except ImportError:raise SystemExit('regex module unavailable, no cluster score computed')
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'REPORTS/live_v26_audit_2026_11_22'
rows=[json.loads(l) for l in (OUT/'frozen/results.jsonl').read_text(encoding='utf-8').splitlines()]
def lev(a,b):
 if len(a)<len(b):return lev(b,a)
 prev=list(range(len(b)+1))
 for i,c in enumerate(a,1):
  row=[i]
  for j,d in enumerate(b,1):
   row.append(min(row[-1]+1,prev[j]+1,prev[j-1]+(c!=d)))
  prev=row
 return prev[-1]
result={}
for m in sorted({r['model'] for r in rows}):
 a=[r for r in rows if r['model']==m and not r.get('error') and r['tier']!=0]
 edits=ngt=0
 for r in a:
  g=regex.findall(r'\X',unicodedata.normalize('NFC','\n'.join(r['gt_lines'])))
  p=regex.findall(r'\X',unicodedata.normalize('NFC','\n'.join(r['pred_lines'])))
  edits+=lev(g,p);ngt+=len(g)
 result[m]={'pages':len(a),'cluster_edit_ops':edits,'gt_clusters':ngt,
     'micro_cluster_cer_pct':round(100*edits/max(1,ngt),2)}
 print(m,'pages',len(a),'grapheme_CER_pct',result[m]['micro_cluster_cer_pct'])
(OUT/'grapheme_cluster_cer.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
