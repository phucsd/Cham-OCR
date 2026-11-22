import sys,json,hashlib,collections,unicodedata
from pathlib import Path
try:import yaml
except ImportError:print('NO_YAML');sys.exit(1)
R=Path(__file__).resolve().parents[1]
P=R/'ocr-studio/data'
OUT=R/'REPORTS/live_v26_audit_2026_11_22'
models=[('v25','rec_cham_inference_v25','cham_dict_v25.txt'),
 ('v24_1_r4','rec_cham_inference_v24_1_r4','cham_dict_v24_1_r4.txt'),
 ('v24_1_r3_1','rec_cham_inference_v24_1_r3_1','cham_dict_v24_1_r3_1.txt')]
report={}
for model,dir,filename in models:
 y=P/'output'/dir/'inference.yml';dictfile=P/filename
 if not y.exists():print('NO_YML',model);continue
 config=yaml.safe_load(y.read_text(encoding='utf-8'))
 embedded=config.get('PostProcess',{}).get('character_dict',[])
 external=dictfile.read_text(encoding='utf-8-sig').splitlines()
 a=list(embedded)
 if not a:print('NO embedded',model)
 n=sum(x==y for x,y in zip(a,external))
 mismatch=[{'index':i,'embedded':a[i] if i<len(a) else None,'external':external[i] if i<len(external) else None}
           for i in range(max(len(a),len(external)))
           if (a[i] if i<len(a) else None)!=(external[i] if i<len(external) else None)]
 report[model]={'yml':str(y.relative_to(R)),'external':str(dictfile.relative_to(R)),
  'embedded_len':len(a),'external_len':len(external),
  'identical_order':a==external,'same_multiset':collections.Counter(a)==collections.Counter(external),
  'same_position_count':n,'mismatched_positions':len(mismatch),'examples':mismatch[:35],
  'head_embedded':a[:20],'head_external':external[:20],
  'postprocess_name':config.get('PostProcess',{}).get('name'),
  'preprocessing':config.get('PreProcess',{}).get('transform_ops',[None])}
 print(model,'embedded',len(a),'external',len(external),
       'ORDER_EQUAL',a==external,'COUNTERS_EQUAL',collections.Counter(a)==collections.Counter(external),
       'mismatch',len(mismatch))
 if mismatch:
  print('HEAD',[(q['index'],repr(q['embedded']),repr(q['external'])) for q in mismatch[:12]])
(OUT/'charset_order_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
