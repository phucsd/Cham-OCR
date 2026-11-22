#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded API-contract / image-codec / malformed-upload smoke; non-destructive."""
import base64,io,json,sys,time,urllib.request,urllib.error
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'REPORTS/live_v26_audit_2026_11_22/api_contract'
OUT.mkdir(parents=True,exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
base='https://ocr.cham.asia'
rows=[]
def call(name,path,data=None):
    t=time.monotonic()
    headers={'User-Agent':'Mozilla/5.0 (ChamOCR-Contract-QA/1.0)','Accept':'application/json'}
    if data is not None: headers['Content-Type']='application/json'
    req=urllib.request.Request(base+path,data=(json.dumps(data).encode('utf-8') if data is not None else None),headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=60) as r:
            body=r.read(100000);status=r.status
    except urllib.error.HTTPError as e:body=e.read(3000);status=e.code
    except Exception as e:status=None;body=str(e).encode()
    result={'case':name,'endpoint':path,'http':status,'latency_s':round(time.monotonic()-t,2),
            'response_excerpt':body[:350].decode('utf-8','replace')}
    rows.append(result);print(name,status,'seconds',result['latency_s'],flush=True)
    return result
for endpoint in ['/health','/api/models','/api/samples','/static/js/studio.js']:
    call('GET '+endpoint,endpoint)
f=json.loads((ROOT/'REPORTS/live_quality_2026_11_20/frozen_test_manifest.json').read_text(encoding='utf-8'))['cases']
c=next(x for x in f if x['tier']==1)
im=Image.open(ROOT/c['path']).convert('RGB')
for fmt in ['PNG','JPEG','WEBP','GIF']:
    b=io.BytesIO();im.save(b,format=fmt)
    call('format_'+fmt,'/ocr',{'image':base64.b64encode(b.getvalue()).decode(),'model':'v24_1_r4','method':'dbnet','reading_order':'rows'})
valid_png_b64=base64.b64encode((ROOT/c['path']).read_bytes()).decode()
for name,data in [
 ('missing_image',{'model':'v24_1_r4'}),
 ('corrupt_base64',{'image':'not-base64!!!','model':'v24_1_r4'}),
 ('invalid_model',{'image':valid_png_b64,'model':'nonexistent_model_2026'}),
 ('empty_image',{'image':'','model':'v24_1_r4'}),
 ('invalid_segmentation',{'image':valid_png_b64,'model':'v24_1_r4','method':'not_a_method'}),
 ('invalid_reading_order',{'image':valid_png_b64,'model':'v24_1_r4','method':'dbnet','reading_order':'invalid'})]:
    call(name,'/ocr',data)
# Invalid bbox and empty crop
blob=valid_png_b64
call('invalid_bbox_zero','/ocr_crop',{'image':blob,'model':'v24_1_r4','bbox':[0,0,0,0]})
call('invalid_bbox_negative','/ocr_crop',{'image':blob,'model':'v24_1_r4','bbox':[-200,-200,-1,-1]})
(OUT/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print('DONE CONTRACT',len(rows),flush=True)
