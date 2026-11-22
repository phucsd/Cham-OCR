#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect whether Auto exposes per-line selected recognizer provenance."""
import base64,json,urllib.request,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'REPORTS/live_v26_audit_2026_11_22'
cases=json.loads((ROOT/'ocr-benchmark/data/live_v26_2026_11_22/manifest.json').read_text(encoding='utf-8'))['cases']
chosen=[next(c for c in cases if c['id']==id) for id in ['vietnamese_00','mixed_vi_cham_00']]
out={}
for x in chosen:
 blob=(ROOT/x['image_relpath']).read_bytes()
 obj={'image':base64.b64encode(blob).decode(),'model':'auto','method':'dbnet','reading_order':'rows'}
 req=urllib.request.Request('https://ocr.cham.asia/ocr',data=json.dumps(obj).encode(),headers={'Content-Type':'application/json','User-Agent':'Mozilla/5.0'})
 with urllib.request.urlopen(req,timeout=80) as f:a=json.load(f)
 lines=[]
 for l in a.get('lines',[]):
  lines.append({k:v for k,v in l.items() if k not in ('polygon','bbox','poly','mask','image','crop','img','patch')})
 out[x['id']]={'line_keys':list(a.get('lines',[{}])[0]),'model_info':a.get('model_info'),
    'lines':lines,'gt_lines':x['gt_lines']}
 print(x['id'],'GT',len(x['gt_lines']),'OCR',len(lines),'keys',list(a.get('lines',[{}])[0]),flush=True)
 print('route',[(l.get('model'),l.get('recognizer'),l.get('language'),l.get('script'),l.get('model_used')) for l in lines][:5],flush=True)
(OUT/'auto_route_line_provenance.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
