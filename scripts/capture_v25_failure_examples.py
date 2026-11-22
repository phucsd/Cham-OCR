import json,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'REPORTS/live_v26_audit_2026_11_22'
def get(suite):
 p=OUT/suite/'results.jsonl'
 return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []
summary={}
for s in ['frozen','crops']:
 rows=get(s);r4={x['id']:x for x in rows if x['model']=='v24_1_r4' and not x.get('error')}
 v25={x['id']:x for x in rows if x['model']=='v25' and not x.get('error')}
 joined=[]
 for cid in set(r4)&set(v25):
  a=r4[cid];b=v25[cid]
  if s=='frozen':
   g=' | '.join(a['gt_lines']);rp=' | '.join(a['pred_lines']);vp=' | '.join(b['pred_lines'])
  else:g=a['gt'];rp=a['pred'];vp=b['pred']
  joined.append(dict(id=cid,gt=g,r4=rp,v25=vp,
   cer_r4=round(a['cer']*100,2),cer_v25=round(b['cer']*100,2),
   delta=round(100*(b['cer']-a['cer']),2),
   confidence_r4=a.get('confidence_mean',a.get('confidence')),
   confidence_v25=b.get('confidence_mean',b.get('confidence'))))
 summary[s]=sorted(joined,key=lambda x:x['delta'],reverse=True)[:10]
(OUT/'v25_vs_r4_repro_examples.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
for s,group in summary.items():
 print('GROUP',s,'PAIRED',len(group))
 for ex in group[:3]:
  print('CASE',ex['id'],'R4',ex['cer_r4'],'V25',ex['cer_v25'])
  print('GT',repr(ex['gt'][:120]))
  print('R4',repr(ex['r4'][:120]))
  print('V25',repr(ex['v25'][:120]))
