import json,math,statistics,collections,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from analyze_live_website_quality import align,category
O=ROOT/'REPORTS/live_v26_audit_2026_11_22'
rows=[json.loads(l) for l in (O/'frozen/results.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
models=list(dict.fromkeys(r['model'] for r in rows))
out={}
for model in models:
 a=[r for r in rows if r['model']==model and not r.get('error') and r['tier']!=0]
 a=sorted(a,key=lambda x:x['cer'],reverse=True)
 conf=[r['confidence_mean'] for r in a if r.get('confidence_mean') is not None]
 paired=[(r['confidence_mean'],r['cer']) for r in a if r.get('confidence_mean') is not None]
 pearson=None
 if len(paired)>=3 and len(set(x for x,y in paired))>1:
  try:pearson=statistics.correlation([x for x,y in paired],[y for x,y in paired])
  except Exception:pass
 insert=collections.Counter();group=collections.Counter();op=collections.Counter()
 for r in a:
  g='\n'.join(r['gt_lines']);p='\n'.join(r['pred_lines'])
  for tp,src,dst in align(g,p):
   op[tp]+=1;group[category(src if src!='∅' else dst)]+=1
 high=[r for r in a if (r.get('confidence_mean') or 0)>=0.90 and r.get('cer',0)>=0.20]
 out[model]=dict(n=len(a),mean_confidence=round(statistics.mean(conf),3) if conf else None,
  median_confidence=round(statistics.median(conf),3) if conf else None,
  pearson_confidence_vs_cer=round(pearson,3) if pearson is not None else None,
  n_confidence_ge_90_cer_ge_20=len(high),
  errors=dict(op),groups=dict(group),
  high_confidence_bad=[{'id':r['id'],'tier':r['tier'],'cer_pct':round(r['cer']*100,2),'mean_conf':r['confidence_mean']} for r in high],
  top_worst=[{'id':r['id'],'cer_pct':round(r['cer']*100,2),'chars_gt':r['gt_chars'],
     'chars_pred':len('\n'.join(r['pred_lines'])),'lines_gt':r['gt_count'],'lines_pred':r['pred_count'],
     'mean_conf':r.get('confidence_mean'),'gt_first':r['gt_lines'][0] if r['gt_lines'] else '',
     'pred_first':r['pred_lines'][0] if r['pred_lines'] else ''} for r in a[:8]])
(O/'confidence_error_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
for m in models:
 v=out[m]
 print(m,'N',v['n'],'mean_conf',v['mean_confidence'],'pearson',v['pearson_confidence_vs_cer'],
 'HIGH_CONF_BAD',v['n_confidence_ge_90_cer_ge_20'],'OPS',v['errors'])
