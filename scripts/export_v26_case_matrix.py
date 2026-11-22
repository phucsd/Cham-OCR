#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified, Excel-friendly real OCR case matrix, no fabricated measurements."""
import csv,json,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'REPORTS/live_v26_audit_2026_11_22'
sources=['frozen','geometry','stress','vietnamese','crops','digits','pipeline']
rows=[]
def load(suite):
 p=O/suite/'results.jsonl'
 if not p.exists():return
 found={}
 for ln in p.read_text(encoding='utf-8').splitlines():
  try:
   r=json.loads(ln)
   key=(r.get('id'),r.get('model'),r.get('method',''),r.get('reading_order',''))
   found[key]=r
  except json.JSONDecodeError:continue
 for x in found.values():
  iscrop=suite in ('crops','digits')
  raw_cer=x.get('cer')
  if raw_cer is None and x.get('cer_pct') is not None:raw_cer=x['cer_pct']/100
  rows.append(dict(suite=suite,case=x.get('id'),difficulty=x.get('tier',x.get('group')),model=x.get('model'),
   actual_model=x.get('actual_model',x.get('version')),fingerprint=x.get('fingerprint',''),
   sha256=x.get('input_sha256',x.get('sha256')),
   gt_line_count=x.get('gt_count',1 if iscrop else x.get('ground_truth_lines','')),
   ocr_line_count=x.get('pred_count',1 if iscrop else x.get('pred_lines','')),
   cer_pct=round(100*raw_cer,3) if raw_cer is not None else '',
   wer_pct=round(100*x['wer'],3) if x.get('wer') is not None else x.get('wer_pct',''),
   exact_match=x.get('exact',''),
   gt_chars=x.get('gt_chars',''),edits=x.get('edit_chars',x.get('edits','')),
   confidence=x.get('confidence_mean',x.get('confidence','')),
   backend_s=x.get('backend_sec',x.get('backend_s','')),
   wall_s=x.get('wall_sec',x.get('wall_s','')),
   segmentation=x.get('method','dbnet'),reading_order=x.get('reading_order','rows'),
   symbol=x.get('group') if suite=='crops' else '',
   symbol_correct=x.get('symbol_correct',''),hyphen_correct=x.get('hyphen_count_correct',''),
   error=x.get('error',''),
   ground_truth=' | '.join(x['gt_lines']) if isinstance(x.get('gt_lines'),list) else x.get('gt',''),
   prediction=' | '.join(x['pred_lines']) if isinstance(x.get('pred_lines'),list) else x.get('pred','')))
for s in sources:load(s)
if rows:
 out=O/'per_case_results.csv'
 with out.open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 counts=collections.Counter(x['suite'] for x in rows)
 print('CASE_MATRIX',len(rows),'saved',out,'by_suite',dict(counts),flush=True)
else:print('No data!',flush=True)
