#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consolidated Vietnamese QA report from production live benchmark artifacts."""
import json,collections,datetime
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]/'REPORTS/live_quality_2026_11_20'
SUITE=BASE/'symbol_crop_suite'
FULL=json.loads((BASE/'summary.json').read_text(encoding='utf-8'))
DIAG=json.loads((BASE/'error_diagnostics.json').read_text(encoding='utf-8'))
SYM=json.loads((SUITE/'summary.json').read_text(encoding='utf-8')) if (SUITE/'summary.json').exists() else {}
MODELS=list(FULL['overall'])
LV={0:'Ảnh trắng',1:'Ảnh rõ',2:'Giấy cũ/nhiễu',3:'Dòng sát',4:'Dòng uốn lượn',
    5:'Dòng chồng lấn',6:'Ảnh chụp thật'}
rows=[json.loads(s) for s in (BASE/'results.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
valid=[r for r in rows if not r.get('error') and (r.get('actual_model')==r['model'])]
out=['# BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG CHAM OCR — 20/11/2026','',
 '## 1. Phạm vi và tính xác thực','',
 'Trang test trực tiếp: https://ocr.cham.asia, HTTP POST /ocr và /ocr_crop.',
 'Mỗi lượt lưu nhãn chuẩn, văn bản nhận diện, mã SHA-256 ảnh, model, fingerprint và thời gian.',
 'Kết quả là đo mới trên production, KHÔNG lấy các điểm từ báo cáo train cũ.','',
 '### Model thực sự được server tải','',
 '| Model | SHA256 trọng số (16 hex) | Số ảnh có chữ | Lỗi gọi API |',
 '|---|---|---:|---:|']
for m in MODELS:
    st=FULL['overall'][m]
    out.append('| %s | %s | %s | %s |'%(m,', '.join(FULL['fingerprints'][m]),st['n'],st['errors']))
out+=['','## 2. Kết quả nhận dạng toàn trang','',
       'Micro CER/WER tính trên tổng ký tự/từ của ground truth đã chuẩn NFC; giữ nguyên dấu cách và xuống dòng.',
       '| Model | Ảnh | CER | WER | Đúng số dòng | Median backend |',
       '|---|---:|---:|---:|---:|---:|']
for m in MODELS:
    s=FULL['overall'][m]
    out.append('| %s | %s | %.2f%% | %.2f%% | %.2f%% | %.2f s |'%(
        m,s['n'],s['cer_pct'],s['wer_pct'],s['line_count_match_pct'],s['median_backend_sec']))
out+=['','### Kết quả theo nhóm khó','',
       '| Mức | '+' | '.join(MODELS)+' |',
       '|---|'+'---:|'*len(MODELS)]
for lv in range(1,7):
    v=[]
    for m in MODELS:
        r=FULL['tiers'][m].get(str(lv))
        v.append(('%s%% (%s ảnh)'%(r['cer_pct'],r['n'])) if r and r['n'] else '—')
    out.append('| %s: %s | %s |'%(lv,LV[lv],' | '.join(v)))
out+=['','### So sánh cặp trên chính cùng ảnh','',
       '| Đối chứng | Cùng ảnh | R4 thắng | Hòa | R4 thua | Trung bình (CER R3.x trừ R4) |',
       '|---|---:|---:|---:|---:|---:|']
for m,s in DIAG.get('paired',{}).items():
    out.append('| %s | %s | %s | %s | %s | %s điểm %% |'%(
        m,s['compared'],s['r4_wins'],s['ties'],s['r4_loses'],s['mean_cer_gain_points']))
out+=['','## 3. Sai sót phổ biến — truy vết căn chỉnh ký tự','']
for m in MODELS:
    d=DIAG['models'].get(m,{})
    cc=d.get('character_categories',{})
    sorted_cc=sorted(cc.items(),key=lambda z:z[1],reverse=True)
    out.append('**'+m+'**: '+', '.join('%s: %s'%(k,v) for k,v in sorted_cc[:5])+'.')
    top=d.get('top_confusions',[])[:7]
    out.append('Nhầm nhiều: '+', '.join('%s %s→%s (%s)'%(
        z['op'],repr(z['gt']),repr(z['ocr']),z['count']) for z in top)+'.')
    out.append('')
out+=['Các trường hợp có tỷ lệ lỗi cao (R4):','']
for z in DIAG['models'].get('v24_1_r4',{}).get('worst',[])[:5]:
    out.append('- %s / mức %s: CER %.2f%%, %s/%s dòng OCR/GT'%(
        z['id'],z['tier'],z['cer']*100,z['pred_count'],z['gt_count']))
out+=['','## 4. Dấu câu, ký hiệu và dấu gạch nối (/ocr_crop)','',
       'Đây là phép kiểm thử recognizer trên crop 48px, tách biệt với toàn trang. Không gộp CER crop vào CER E2E.']
if SYM:
    out+=['','| Model | Crops | CER crop | Đúng nguyên dòng | Lỗi API |',
          '|---|---:|---:|---:|---:|']
    for m in MODELS:
        s=SYM.get(m)
        if not s:continue
        out.append('| %s | %s | %.2f%% | %.2f%% | %s |'%(
            m,s['cases'],s['cer_pct'],s['full_line_exact_pct'],s['errors']))
    groups=sorted(set(x for s in SYM.values() for x in s['symbols']))
    out+=['','### Dấu bắt buộc — số đúng/số thử theo căn chỉnh ký tự','',
          '| Ký hiệu | '+' | '.join(MODELS)+' |','|---|'+'---:|'*len(MODELS)]
    for sym in groups:
        v=[]
        for m in MODELS:
            s=SYM.get(m,{}).get('symbols',{}).get(sym,{})
            v.append('%s/%s'%(s.get('correct',0),s.get('n',0)))
        out.append('| %s | %s |'%('backslash' if sym=='\\' else sym,' | '.join(v)))
    groups=sorted(set(x for s in SYM.values() for x in s['hyphens']))
    out+=['','### Dấu nối theo ngữ cảnh — số dòng có đúng số dấu','',
          '| Ngữ cảnh | '+' | '.join(MODELS)+' |','|---|'+'---:|'*len(MODELS)]
    for g in groups:
        v=[]
        for m in MODELS:
            s=SYM.get(m,{}).get('hyphens',{}).get(g,{})
            v.append('%s/%s'%(s.get('count_correct',0),s.get('n',0)))
        out.append('| %s | %s |'%(g,' | '.join(v)))
else:
    out.append('Bộ crop chưa hoàn tất ở thời điểm tổng hợp.')
out+=['','## 5. Đánh giá phát hành và việc cần làm','',
 '- Nên giữ R4 phục vụ ảnh chụp thực tế trong khi benchmark ngoài mẫu còn ít; R3.1 nhiều lần tốt hơn trên ảnh tổng hợp, cần xem các kết quả cặp cụ thể.',
 '- Ưu tiên audit đường ảnh và crop ở các trang uốn lượn; ví dụ synthetic_142 có 7/8 dòng và CER cao ở cả bốn model.',
 '- Bổ sung hard-negative và cặp dễ nhầm cho các dấu phụ âm cuối, các nguyên âm phụ, dấu ꩝ và dấu câu ASCII, sau khi kiểm tra nhãn và font render.',
 '- Tăng số trang ảnh thật chưa từng train, chia tách ở cấp tài liệu và font, công bố khoảng tin cậy trước khi quyết định đổi model production.',
 '- Tách kiểm thử segmentation, recognition-only, Unicode order, punctuation, blank-page guard và latency; thiết lập no-regression gate theo từng nhóm.',
 '- Không mặc định rằng R4.1 hoặc R4.2 đã deploy: hai checkpoint này từng được đánh giá không đủ release gates; dropdown production hiện nêu R4.',
 '','## 6. Giới hạn và tiêu chuẩn khoa học','',
 '- Bộ ảnh fullpage có bản tổng hợp, chưa chứng minh hoàn toàn độc lập miền huấn luyện; benchmark trên ảnh thật mới có 1 tài liệu 25 dòng.',
 '- Hai cách tính WER dựa trên khoảng trắng phụ thuộc cách viết scriptio continua; CER nên là thước đo chính.',
 '- Chỉ số đúng số dòng không thay cho IoU/recall ở cấp polygon; số dòng đúng vẫn có thể cắt sai, ghép dòng hoặc nhầm thứ tự.',
 '- So sánh độ trễ phụ thuộc CPU/queue/cache; không coi đây là throughput load test.',
 '- Bảng ký hiệu được đánh giá theo căn chỉnh Levenshtein, còn dấu nối dùng số lượng dấu; vẫn cần kiểm tra định vị dấu nối trong vòng đánh giá sâu.',
 '- Không suy ra kết luận vượt trội thống kê từ bộ mẫu nhỏ.','',
 '## 7. Tập tin tái lập','',
 'scripts/live_website_quality_benchmark.py — fullpage E2E.',
 'scripts/live_symbol_crop_benchmark.py — symbol và hyphen crop.',
 'scripts/analyze_live_website_quality.py — alignment, confusion, cặp model.',
 'scripts/live_qa_frozen_replay.py — tái kiểm thử đúng 30 trường hợp đã chốt, trên các model mới; hỗ trợ --dry-run và --models.',
 'REPORTS/live_quality_2026_11_20/frozen_test_manifest.json — danh sách ảnh, GT, hash ảnh, mức khó được đóng băng.',
 'REPORTS/live_quality_2026_11_20/results.jsonl, results.csv, summary.json, REPORT.md.',
 'REPORTS/live_quality_2026_11_20/ERROR_DIAGNOSTICS.md và symbol_crop_suite/REPORT.md.',
 '','Tạo báo cáo: '+datetime.datetime.now(datetime.timezone.utc).isoformat()]
(BASE/'FINAL_REPORT_VI.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
print('WROTE',BASE/'FINAL_REPORT_VI.md',len(valid),'valid fullpage responses')
