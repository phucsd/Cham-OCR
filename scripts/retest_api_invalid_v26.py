import json,base64,time,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
f=json.loads((ROOT/'REPORTS/live_quality_2026_11_20/frozen_test_manifest.json').read_text(encoding='utf-8'))['cases']
c=next(x for x in f if x['tier']==1)
data=base64.b64encode((ROOT/c['path']).read_bytes()).decode()
cases=[('missing_image_retest',{'model':'v24_1_r4'}),
       ('invalid_model_valid_image',{'model':'nonexistent_model_2026','image':data})]
out=[]
for name,payload in cases:
  req=urllib.request.Request('https://ocr.cham.asia/ocr',data=json.dumps(payload).encode(),headers={'User-Agent':'Mozilla/5.0','Content-Type':'application/json'})
  try:
    with urllib.request.urlopen(req,timeout=30) as r:status=r.status;body=r.read(700)
  except urllib.error.HTTPError as ex:status=ex.code;body=ex.read(700)
  except Exception as ex:status=None;body=str(ex).encode()
  out.append(dict(name=name,status=status,body=body.decode('utf-8','replace')))
  print(name,status,repr(body.decode('utf-8','replace')[:200]),flush=True)
p=ROOT/'REPORTS/live_v26_audit_2026_11_22/api_contract/retest.json'
p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
