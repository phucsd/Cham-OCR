# BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG CHAM OCR — 20/11/2026

## 1. Phạm vi và tính xác thực

Trang test trực tiếp: https://ocr.cham.asia, HTTP POST /ocr và /ocr_crop.
Mỗi lượt lưu nhãn chuẩn, văn bản nhận diện, mã SHA-256 ảnh, model, fingerprint và thời gian.
Kết quả là đo mới trên production, KHÔNG lấy các điểm từ báo cáo train cũ.

### Model thực sự được server tải

| Model | SHA256 trọng số (16 hex) | Số ảnh có chữ | Lỗi gọi API |
|---|---|---:|---:|
| v24_1_r4 | 85525b0ca091acaf | 29 | 0 |
| v24_1_r3_1 | 5fb0d8e4d512b4c0 | 29 | 0 |
| v24_1_r3 | 6a1f0cc72bfff5cb | 29 | 0 |
| v24_1_r2_1_6 | ad43ddcf19397a94 | 29 | 0 |

## 2. Kết quả nhận dạng toàn trang

Micro CER/WER tính trên tổng ký tự/từ của ground truth đã chuẩn NFC; giữ nguyên dấu cách và xuống dòng.
| Model | Ảnh | CER | WER | Đúng số dòng | Median backend |
|---|---:|---:|---:|---:|---:|
| v24_1_r4 | 29 | 7.63% | 27.85% | 96.55% | 1.50 s |
| v24_1_r3_1 | 29 | 7.41% | 28.29% | 96.55% | 1.57 s |
| v24_1_r3 | 29 | 8.18% | 31.59% | 96.55% | 1.62 s |
| v24_1_r2_1_6 | 29 | 7.95% | 28.36% | 96.55% | 1.52 s |

### Kết quả theo nhóm khó

| Mức | v24_1_r4 | v24_1_r3_1 | v24_1_r3 | v24_1_r2_1_6 |
|---|---:|---:|---:|---:|
| 1: Ảnh rõ | 6.8% (4 ảnh) | 6.8% (4 ảnh) | 7.02% (4 ảnh) | 6.34% (4 ảnh) |
| 2: Giấy cũ/nhiễu | 6.86% (6 ảnh) | 5.11% (6 ảnh) | 5.61% (6 ảnh) | 5.61% (6 ảnh) |
| 3: Dòng sát | 6.3% (6 ảnh) | 5.69% (6 ảnh) | 6.38% (6 ảnh) | 6.53% (6 ảnh) |
| 4: Dòng uốn lượn | 11.99% (6 ảnh) | 11.3% (6 ảnh) | 12.06% (6 ảnh) | 12.13% (6 ảnh) |
| 5: Dòng chồng lấn | 7.95% (6 ảnh) | 8.67% (6 ảnh) | 9.97% (6 ảnh) | 9.39% (6 ảnh) |
| 6: Ảnh chụp thật | 3.55% (1 ảnh) | 5.2% (1 ảnh) | 6.5% (1 ảnh) | 5.67% (1 ảnh) |

### So sánh cặp trên chính cùng ảnh

| Đối chứng | Cùng ảnh | R4 thắng | Hòa | R4 thua | Trung bình (CER R3.x trừ R4) |
|---|---:|---:|---:|---:|---:|
| v24_1_r3_1 | 29 | 8 | 3 | 18 | -0.61 điểm % |
| v24_1_r3 | 29 | 11 | 6 | 12 | 0.02 điểm % |
| v24_1_r2_1_6 | 29 | 13 | 4 | 12 | -0.09 điểm % |

## 3. Sai sót phổ biến — truy vết căn chỉnh ký tự

**v24_1_r4**: Cham vowel / diacritic: 139, Spacing / linebreak: 69, Cham coda: 61, Cham base consonant / vowel: 55, ASCII punctuation: 55.
Nhầm nhiều: I '∅'→' ' (36), D '꩝'→'∅' (31), D 'ꨮ'→'∅' (30), D ' '→'∅' (27), D 'ꩌ'→'∅' (18), D 'ꨰ'→'∅' (15), D '?'→'∅' (15).

**v24_1_r3_1**: Cham vowel / diacritic: 125, Spacing / linebreak: 74, Cham coda: 65, ASCII punctuation: 64, Cham base consonant / vowel: 57.
Nhầm nhiều: I '∅'→' ' (50), D 'ꨮ'→'∅' (29), D ' '→'∅' (21), D '꩝'→'∅' (20), S '–'→'-' (14), D '?'→'∅' (14), D 'ꩌ'→'∅' (12).

**v24_1_r3**: Cham vowel / diacritic: 152, Spacing / linebreak: 87, Cham coda: 65, ASCII punctuation: 64, Cham base consonant / vowel: 54.
Nhầm nhiều: I '∅'→' ' (66), D 'ꨮ'→'∅' (30), I '∅'→'ꨯ' (20), D ' '→'∅' (19), D '꩝'→'∅' (18), S 'ꨩ'→'ꨱ' (18), D '?'→'∅' (15).

**v24_1_r2_1_6**: Cham vowel / diacritic: 146, Spacing / linebreak: 68, Cham base consonant / vowel: 67, ASCII punctuation: 65, Cham coda: 61.
Nhầm nhiều: I '∅'→' ' (44), D 'ꨮ'→'∅' (28), D ' '→'∅' (21), S 'ꨩ'→'ꩃ' (15), S 'ꨮ'→'ꨱ' (14), D '?'→'∅' (14), S 'ꨩ'→'ꨱ' (13).

Các trường hợp có tỷ lệ lỗi cao (R4):

- synthetic_142 / mức 4: CER 25.75%, 7/8 dòng OCR/GT
- synthetic_147 / mức 4: CER 12.31%, 2/2 dòng OCR/GT
- synthetic_196 / mức 5: CER 12.27%, 3/3 dòng OCR/GT
- synthetic_134 / mức 4: CER 11.11%, 3/3 dòng OCR/GT
- synthetic_089 / mức 3: CER 10.11%, 2/2 dòng OCR/GT

## 4. Dấu câu, ký hiệu và dấu gạch nối (/ocr_crop)

Đây là phép kiểm thử recognizer trên crop 48px, tách biệt với toàn trang. Không gộp CER crop vào CER E2E.

| Model | Crops | CER crop | Đúng nguyên dòng | Lỗi API |
|---|---:|---:|---:|---:|
| v24_1_r4 | 42 | 0.69% | 92.86% | 0 |
| v24_1_r3_1 | 42 | 0.69% | 92.86% | 0 |
| v24_1_r3 | 42 | 1.84% | 85.71% | 0 |
| v24_1_r2_1_6 | 42 | 1.84% | 80.95% | 0 |

### Dấu bắt buộc — số đúng/số thử theo căn chỉnh ký tự

| Ký hiệu | v24_1_r4 | v24_1_r3_1 | v24_1_r3 | v24_1_r2_1_6 |
|---|---:|---:|---:|---:|
| ! | 2/2 | 2/2 | 1/2 | 2/2 |
| # | 2/2 | 2/2 | 2/2 | 2/2 |
| & | 2/2 | 2/2 | 2/2 | 2/2 |
| * | 2/2 | 2/2 | 2/2 | 2/2 |
| / | 1/2 | 1/2 | 1/2 | 1/2 |
| < | 2/2 | 2/2 | 2/2 | 2/2 |
| > | 2/2 | 2/2 | 2/2 | 2/2 |
| ? | 2/2 | 2/2 | 2/2 | 2/2 |
| [ | 2/2 | 2/2 | 2/2 | 2/2 |
| backslash | 2/2 | 2/2 | 2/2 | 2/2 |
| ] | 2/2 | 2/2 | 2/2 | 2/2 |
| _ | 1/2 | 1/2 | 1/2 | 2/2 |
| { | 2/2 | 2/2 | 2/2 | 2/2 |
| } | 2/2 | 2/2 | 1/2 | 2/2 |

### Dấu nối theo ngữ cảnh — số dòng có đúng số dấu

| Ngữ cảnh | v24_1_r4 | v24_1_r3_1 | v24_1_r3 | v24_1_r2_1_6 |
|---|---:|---:|---:|---:|
| double hyphen | 2/2 | 2/2 | 2/2 | 2/2 |
| line beginning | 2/2 | 2/2 | 2/2 | 2/2 |
| line ending | 2/2 | 2/2 | 2/2 | 2/2 |
| number-number | 2/2 | 2/2 | 1/2 | 1/2 |
| word - word | 2/2 | 2/2 | 2/2 | 2/2 |
| word-word | 2/2 | 2/2 | 2/2 | 2/2 |
| year-year | 2/2 | 2/2 | 2/2 | 2/2 |

## 5. Đánh giá phát hành và việc cần làm

- Nên giữ R4 phục vụ ảnh chụp thực tế trong khi benchmark ngoài mẫu còn ít; R3.1 nhiều lần tốt hơn trên ảnh tổng hợp, cần xem các kết quả cặp cụ thể.
- Ưu tiên audit đường ảnh và crop ở các trang uốn lượn; ví dụ synthetic_142 có 7/8 dòng và CER cao ở cả bốn model.
- Bổ sung hard-negative và cặp dễ nhầm cho các dấu phụ âm cuối, các nguyên âm phụ, dấu ꩝ và dấu câu ASCII, sau khi kiểm tra nhãn và font render.
- Tăng số trang ảnh thật chưa từng train, chia tách ở cấp tài liệu và font, công bố khoảng tin cậy trước khi quyết định đổi model production.
- Tách kiểm thử segmentation, recognition-only, Unicode order, punctuation, blank-page guard và latency; thiết lập no-regression gate theo từng nhóm.
- Không mặc định rằng R4.1 hoặc R4.2 đã deploy: hai checkpoint này từng được đánh giá không đủ release gates; dropdown production hiện nêu R4.

## 6. Giới hạn và tiêu chuẩn khoa học

- Bộ ảnh fullpage có bản tổng hợp, chưa chứng minh hoàn toàn độc lập miền huấn luyện; benchmark trên ảnh thật mới có 1 tài liệu 25 dòng.
- Hai cách tính WER dựa trên khoảng trắng phụ thuộc cách viết scriptio continua; CER nên là thước đo chính.
- Chỉ số đúng số dòng không thay cho IoU/recall ở cấp polygon; số dòng đúng vẫn có thể cắt sai, ghép dòng hoặc nhầm thứ tự.
- So sánh độ trễ phụ thuộc CPU/queue/cache; không coi đây là throughput load test.
- Bảng ký hiệu được đánh giá theo căn chỉnh Levenshtein, còn dấu nối dùng số lượng dấu; vẫn cần kiểm tra định vị dấu nối trong vòng đánh giá sâu.
- Không suy ra kết luận vượt trội thống kê từ bộ mẫu nhỏ.

## 7. Tập tin tái lập

scripts/live_website_quality_benchmark.py — fullpage E2E.
scripts/live_symbol_crop_benchmark.py — symbol và hyphen crop.
scripts/analyze_live_website_quality.py — alignment, confusion, cặp model.
scripts/live_qa_frozen_replay.py — tái kiểm thử đúng 30 trường hợp đã chốt, trên các model mới; hỗ trợ --dry-run và --models.
REPORTS/live_quality_2026_11_20/frozen_test_manifest.json — danh sách ảnh, GT, hash ảnh, mức khó được đóng băng.
REPORTS/live_quality_2026_11_20/results.jsonl, results.csv, summary.json, REPORT.md.
REPORTS/live_quality_2026_11_20/ERROR_DIAGNOSTICS.md và symbol_crop_suite/REPORT.md.

Tạo báo cáo: 2026-11-20T04:47:07.047672+00:00
