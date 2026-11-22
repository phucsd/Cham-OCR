import json,base64,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parents[1]
x=next(i for i in json.loads((P/'REPORTS/live_quality_2026_11_20/frozen_test_manifest.json').read_text(encoding='utf-8'))['cases'] if i['tier']==1)
data={'image':base64.b64encode((P/x['path']).read_bytes()).decode(),'model':'nonexistent_model_2026','method':'dbnet','reading_order':'rows'}
req=urllib.request.Request('https://ocr.cham.asia/ocr',data=json.dumps(data).encode(),headers={'Content-Type':'application/json','User-Agent':'Mozilla/5.0'})
with urllib.request.urlopen(req,timeout=50) as f:
 o=json.load(f);status=f.status
print('HTTP',status,'requested',data['model'],'actual',o.get('model_info',{}).get('model'),'first',o.get('model_info',{}).get('recognizers',[{}])[0].get('version'),'line_count',len(o.get('lines',[])))
out={'http':status,'requested':data['model'],'actual':o.get('model_info',{}).get('model'),'recognizers':o.get('model_info',{}).get('recognizers'),'line_count':len(o.get('lines',[]))}
(P/'REPORTS/live_v26_audit_2026_11_22/api_contract/invalid_model_fallback.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
