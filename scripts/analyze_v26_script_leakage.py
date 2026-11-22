import json,collections,unicodedata
from pathlib import Path
R=Path(__file__).resolve().parents[1]
O=R/'REPORTS/live_v26_audit_2026_11_22'
rows=[json.loads(x) for x in (O/'frozen/results.jsonl').read_text(encoding='utf-8').splitlines()]
def latin(c):
 if not c.isalpha():return False
 return 'LATIN' in unicodedata.name(c,'')
out={}
for m in sorted({x['model'] for x in rows}):
 a=[x for x in rows if x['model']==m and x['tier']!=0]
 gc=sum(sum(latin(c) for c in '\n'.join(x['gt_lines'])) for x in a)
 pc=sum(sum(latin(c) for c in '\n'.join(x['pred_lines'])) for x in a)
 plen=sum(len('\n'.join(x['pred_lines'])) for x in a)
 pages=sum(any(latin(c) for c in '\n'.join(x['pred_lines'])) for x in a)
 out[m]={'n':len(a),'gt_latin_letters':gc,'ocr_latin_letters':pc,
         'ocr_latin_letter_pct':round(100*pc/max(1,plen),2),
         'pages_with_latin_letters':pages,
         'tiers':{str(t):{'n':len(z),'latin_ocr':sum(sum(latin(c) for c in '\n'.join(x['pred_lines'])) for x in z)}
                  for t in sorted({x['tier'] for x in a}) for z in [[x for x in a if x['tier']==t]]}}
 print(m,'GT Latin',gc,'OCR Latin',pc,'of chars',plen,'pct',out[m]['ocr_latin_letter_pct'],'pages',pages)
(O/'script_leakage_analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
