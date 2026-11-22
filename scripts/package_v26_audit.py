#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible distributable audit ZIP: results, scripts, ground truths, seen images, provenance."""
import hashlib,json,zipfile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REP=ROOT/'REPORTS/live_v26_audit_2026_11_22'
TARGET=ROOT/'REPORTS/Cham_OCR_V26_Live_Benchmark_2026-11-22.zip'
FILES=set()
def add(x):
 x=Path(x)
 if x.exists() and x.is_file():FILES.add(x.resolve())
def manifest(p):
 if not p.exists():return []
 add(p)
 try:return json.loads(p.read_text(encoding='utf-8')).get('cases',[])
 except Exception:return []
for p in REP.rglob('*'):
 if p.is_file():add(p)
script_names=[
 'benchmark_all_live_v26.py','benchmark_live_v26_crops.py',
 'benchmark_live_v26_pipeline.py','benchmark_v26_cham_digits.py',
 'build_v26_live_audit_report.py','plot_v26_live_audit.py',
 'analyze_v26_confidence.py','analyze_v26_script_leakage.py',
 'capture_v25_failure_examples.py','inspect_v26_auto_router.py',
 'generate_v26_audit_cases.py','ui_probe_live_v26.js','ui_e2e_live_v26.js',
 'ui_export_pdf_e2e_v26.js','make_pdf_ui_fixture_v26.py',
 'verify_ui_exports_v26.py','test_live_v26_api_contract.py',
 'probe_invalid_model_fallback_v26.py','retest_api_invalid_v26.py',
 'export_v26_case_matrix.py','package_v26_audit.py',
 'audit_v25_char_dict_order.py','analyze_segmentation_controls_v26.py',
 'live_website_quality_benchmark.py',
 'live_symbol_crop_benchmark.py','live_qa_frozen_replay.py',
 'analyze_live_website_quality.py']
for name in script_names:add(ROOT/'scripts'/name)
frozen=manifest(ROOT/'REPORTS/live_quality_2026_11_20/frozen_test_manifest.json')
geo=manifest(ROOT/'ocr-benchmark/data/live_geometry_multifont_2026_11_21/manifest.json')
svi=manifest(ROOT/'ocr-benchmark/data/live_v26_2026_11_22/manifest.json')
def sample_ids(s):
 p=REP/s/'results.jsonl'
 if not p.exists():return set()
 ids=set()
 for line in p.read_text(encoding='utf-8').splitlines():
  try:ids.add(json.loads(line)['id'])
  except Exception:pass
 return ids
geo_ids=sample_ids('geometry')
extra_ids=sample_ids('stress')|sample_ids('vietnamese')
for x in frozen:
 if x.get('path'):add(ROOT/x['path'])
for x in geo:
 if x['id'] in geo_ids:add(ROOT/x['image_relpath'])
for x in svi:
 if x['id'] in extra_ids:add(ROOT/x['image_relpath'])
froot=ROOT/'ocr-benchmark/data/frozen_suite'
for manifest_name,kind in [('symbol_frozen_14_lines.json','symbol'),('hyphen_frozen_multi_context_lines.json','hyphen')]:
 p=froot/manifest_name
 samples=json.loads(p.read_text(encoding='utf-8'));add(p)
 ids=sample_ids('crops')
 for x in samples:
  if kind+'_'+str(x['sample_id']) in ids:add(froot/x['crop_file'])
meta={'created':'2026-11-22','website':'https://ocr.cham.asia',
      'summary':'All files bundled from the authenticated Cham-OCR repository and live production OCR audit.',
      'count':len(FILES),
      'sha256':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(FILES)}}
with zipfile.ZipFile(TARGET,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
 for p in sorted(FILES):
  z.write(p,str(p.relative_to(ROOT)).replace('\\','/'))
 z.writestr('ARCHIVE_FILE_SHA256_MANIFEST.json',json.dumps(meta,ensure_ascii=False,indent=2))
with zipfile.ZipFile(TARGET) as z:
 bad=z.testzip();n=len(z.namelist())
print('ZIP_CREATED',str(TARGET),'files',n,'bytes',TARGET.stat().st_size,'crc_error',bad)
