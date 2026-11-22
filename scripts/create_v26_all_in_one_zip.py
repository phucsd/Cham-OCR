#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single flattened ZIP containing entire Cham OCR V26 QA, raw + PDF + summaries."""
from pathlib import Path
import zipfile,hashlib,json,datetime
ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'REPORTS'
src=REPORTS/'Cham_OCR_V26_Live_Benchmark_2026-11-22.zip'
dst=REPORTS/'Cham_OCR_V26_TAT_CA_BAO_CAO_2026-11-22.zip'
audit=REPORTS/'live_v26_audit_2026_11_22'
assert src.is_file(),str(src)
assert (audit/'Bao_cao_kiem_thu_OCR_Cham_V26_2026-11-22.pdf').is_file()
newfiles={
 '01_Bao_cao_chi_tiet_OCR_Cham_V26.pdf':audit/'Bao_cao_kiem_thu_OCR_Cham_V26_2026-11-22.pdf',
 '02_Bao_cao_chi_tiet_OCR_Cham_V26.md':audit/'FINAL_REPORT_VI.md',
 '03_Bang_so_sanh_8_model.csv':audit/'model_comparison.csv',
 '04_Bang_ket_qua_tung_luot_OCR.csv':audit/'per_case_results.csv',
 '05_Bieu_do_so_sanh_model.png':audit/'charts'/'frozen_cer_by_model.png',
 '06_Bieu_do_theo_do_kho.png':audit/'charts'/'frozen_cer_by_difficulty.png',
 '07_Bieu_do_anh_suy_giam.png':audit/'charts'/'stress_by_transform.png',
 '08_Bieu_do_dau_cau.png':audit/'charts'/'symbols_aligned_accuracy.png',
 'SCRIPTS/render_v26_report_pc.js':ROOT/'scripts'/'render_v26_report_pc.js',
 'SCRIPTS/create_v26_all_in_one_zip.py':ROOT/'scripts'/'create_v26_all_in_one_zip.py',
}
for name,fp in newfiles.items():assert fp.is_file(),name
readme="""BỘ BÁO CÁO KIỂM THỬ OCR.CHAM.ASIA — 22/11/2026

MỘT FILE ZIP DUY NHẤT / ALL-IN-ONE

MỞ CÁC FILE Ở THƯ MỤC GỐC CỦA ZIP:
  01_Bao_cao_chi_tiet_OCR_Cham_V26.pdf       : Báo cáo PDF (render từ báo cáo Markdown đầy đủ trên PC)
  02_Bao_cao_chi_tiet_OCR_Cham_V26.md        : Báo cáo Markdown đầy đủ
  03_Bang_so_sanh_8_model.csv               : Bảng điểm / benchmark
  04_Bang_ket_qua_tung_luot_OCR.csv         : 947 bản ghi, mỗi lần OCR/crop một hàng
  05_Bieu_do_so_sanh_model.png              : CER 8 model trên ảnh cố định
  06_Bieu_do_theo_do_kho.png                : CER theo độ khó
  07_Bieu_do_anh_suy_giam.png               : CER ảnh chất lượng thấp
  08_Bieu_do_dau_cau.png                    : Dấu câu/dấu gạch nối

THƯ MỤC REPORTS/live_v26_audit_2026_11_22:
  Kết quả .jsonl có ground truth, dự đoán, fingerprint và thời gian,
  test bằng trình duyệt, file export, biểu đồ và báo cáo gốc.
THƯ MỤC ocr-benchmark:
  Test input và manifest ground truth.
THƯ MỤC scripts:
  Kịch bản có thể chạy lại bài test.

CÁC GIỚI HẠN:
  8 model có mặt trong bài test 30 ảnh cố định. Test mở rộng chỉ
  lấy các model chính/nhóm đại diện; xem số lượng thực ở báo cáo.
  Không bao quát mọi tài liệu ngoài đời; cần ảnh thật hold-out.

Tập ZIP này được tạo trên máy tính của người dùng và kiểm tra CRC.
Không bao gồm thay đổi trọng số, không triển khai lại production.
"""
seen=set()
with zipfile.ZipFile(src,'r') as source, zipfile.ZipFile(dst,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
 for item in source.infolist():
  if item.is_dir():continue
  if item.filename in seen:raise RuntimeError('duplicate in source: '+item.filename)
  seen.add(item.filename)
  z.writestr(item,source.read(item))
 for name,p in newfiles.items():
  assert name not in seen
  z.write(p,name)
  seen.add(name)
 assert 'BAT_DAU_O_DAY.txt' not in seen
 z.writestr('BAT_DAU_O_DAY.txt',readme.encode('utf-8-sig'))
with zipfile.ZipFile(dst) as z:
 bad=z.testzip()
 if bad:raise RuntimeError('CRC ERROR: '+bad)
 names=z.namelist()
 pdfs=[n for n in names if n.lower().endswith('.pdf')]
 assert any(n.startswith('01_Bao_cao') for n in pdfs)
 assert len(names)>=237,(len(names),names[-10:])
sha=hashlib.sha256(dst.read_bytes()).hexdigest()
stat=dst.stat()
print('FINAL_PC_ZIP_PATH='+str(dst))
print('FILE_COUNT='+str(len(names))+' SIZE_BYTES='+str(stat.st_size)+' SHA256='+sha)
print('PDF_COUNT='+str(len(pdfs))+' CRC_CHECK=OK')
