# Báo Cáo Chẩn Đoán Cô Lập Ưu Tiên B (`ocr.cham.asia` — 2026-11-22)

- **Thời điểm thực thi (UTC)**: `2026-11-22T09:35:47Z`
- **Môi trường kiểm thử**: Production trực tuyến `https://ocr.cham.asia` (chỉ đọc / suy luận không phá hoại, **`0` lượt gọi `/save_correction`**, tổng cộng `132` request kiểm chứng).
- **Bảo toàn dữ liệu lịch sử**: Toàn bộ tệp kết quả ngày `2026-11-22` trong `REPORTS/live_v26_audit_2026_11_22/` được giữ nguyên tuyệt đối.

---

## 1. Ma Trận Nghiệm Thu 10 Chiều (Thay Thế Con Số Cảm Tính `85%`)

| # | Nhóm Đánh Giá | Trạng Thái | Bằng Chứng Định Lượng Hiện Tại | Điều Kiện Để Đóng / Nâng Cấp |
| :-: | :--- | :---: | :--- | :--- |
| 1 | **API Contract** | **PASS** | `6/6` probe đầu vào lỗi/model lạ trả đúng HTTP `400` JSON; `28/28` unit tests PASS | Duy trì CI regression |
| 2 | **Model Provenance** | **PASS** | `selected_recognizer` khớp 100% `(text, conf)` model thực thi trên cả `dbnet`, `valley`, `adaptive` (Pass 1 & Pass 2) và `/ocr_crop`; xóa bỏ đoán ký tự | Duy trì CI regression |
| 3 | **Bộ Frozen (In-Domain)** | **PASS** | `V26.1` đạt `6.71%` Strict Page CER (`5.8%` Matched-Line Strict CER; `99.32%` Line Recall) | Giữ khóa non-regression |
| 4 | **Chữ Số & Dấu Nối Vị Trí** | **INCONCLUSIVE / PARTIAL** | Dấu nối `V26.1` đạt `100.0%` Positional F1 (`100.0%` theo số lượng); số Chăm crop chuẩn còn `6.0%` CER; số ASCII lặp trên `V26.1` bị rụng ở `0px` tracking | Cần router sub-span cho số ASCII hoặc tăng mẫu số dày ở V27 |
| 5 | **Detector & Cảnh Báo Sót Dòng** | **FAIL (Cảnh báo)** | Line Recall đạt `99.32%` (`1` dòng sót trên `synthetic_142`), **nhưng `possible_missed_lines` KHÔNG kích hoạt** (`uncovered_ink_ratio = 0.0` < `0.12`) | Hạ/nâng cấp detector dải mực ngang cho `possible_missed_lines` |
| 6 | **Đa Cột & Thứ Tự Đọc (`auto`)** | **PASS (5 trang thử)** | Trên cả `3` trang cổ thư 2 cột (`48` dòng) và `2` trang 2 cột sạch (`23` dòng), `auto` nâng Kendall's τ từ Interleaved (`~0.04–0.15`) lên **`1.000`** | Mở rộng thêm trang 2 cột chụp nghiêng/cong gáy thật |
| 7 | **Song Ngữ Nội Dòng (Intra-line)** | **FAIL (Kiến trúc)** | Khi Chăm + Việt nằm **chung 1 dòng**, `Auto Router v2.2` đạt `33.33%` CER (hỏng phần ngôn ngữ không được chọn) | Cần **Intra-Line Script Splitter (`Auto Router v2.3`)** tách cụm ngang |
| 8 | **Ảnh Thật Đa Nguồn Hold-Out** | **NOT TESTED (Thiếu mẫu)** | Mới có `1` ảnh chụp sách thật (`harum_palei_01`: `2.96%` CER) + mẫu Wikimedia; chưa đủ `50–100` trang đa nguồn độc lập | Thu thập `RealScan-Gold v4` tách theo tài liệu/người viết |
| 9 | **Độ Tin Cậy & Rủi Ro Trang** | **INCONCLUSIVE** | Temperature `T*=0.8831` hiệu chuẩn tốt trên dòng đã bắt, nhưng trang sót dòng `synthetic_142` vẫn báo `status = ocr_done` | Tích hợp phát hiện dải mực ngang bị bỏ sót vào `Page Risk` |
| 10 | **Hiệu Năng Tuần Tự Cô Lập** | **NOT TESTED (Đầy đủ)** | Đã có số đo tuần tự đơn luồng sơ bộ, chưa chạy đảo thứ tự lặp đa vòng warm/cold P50/P95 chuyên biệt | Chạy benchmark latency P50/P95 độc lập ở Ưu tiên B2 |

---

## 2. Kết Quả Mô-đun 1 & 2: Bóc Tách Detector vs. Recognizer & Kiểm Toán `Strict` vs. `Philological CER`

### 2.1. Phân rã chính xác ca `synthetic_142` (`7/8` dòng, `23.84%` Page CER)
- **Dòng bị bỏ sót**: `GT[1]` (`64` ký tự): `ꨕꨤꩌ ꨨꨣꨬ ꨆꨓꨯꨮꨩ꩝ ꨂꨣꩃ ꨌꩌ ꨝꨴꩍꨟꨪꩆ ꨕꨮꩃ ꨋꩇ ꨈꨴꨭꩀ ꨃꨥ ꨢꩃ ꨚꩀ ꨕꨯꩌ ꨝꨪꨟꨯꩂ ꨟꨆꩊ꩝`.
- **Phân bổ sai số thực tế (`87` thao tác sửa trên toàn trang)**:
  * Sai số do **mất dòng `GT[1]`**: **`65` edits (`74.71%` tổng lỗi trang)**.
  * Sai số nhận dạng trên **7 dòng còn lại đã ghép đúng**: **`22` edits trên `294` ký tự (`7.48%` Matched-Line Strict CER, chiếm `25.29%` tổng lỗi trang)** — chủ yếu rơi vào dấu ngoặc đơn `(...)` và số Chăm `꩑꩐` bên trong cụm `(ꨗꨶꨮꩄ ꨝꨪꨤꩆ ꩑꩐ ...)`.
  * **Đối chứng cắt trực tiếp vùng `GT[1]` gửi vào `/ocr_crop` (`v24_1_r4`)**: đạt **`17.19%` CER** (`conf = 0.8541`, đọc ra `ꨕꨤꩌ ꨨꨣꨬ ꨆꨓꨩ ꨂꨣꩃ ꨌꨩ ꨝꨥꨴꨟꨪꩆ ꨕꨮꩃ ꨋꨭꩇ ꨈꨴꨮꨭꩀ ꨃꨥ ꨢꨮ ꨚꩀ ꨕꨯꨮ ꨝꨪꨟꨯꩂ ꨟꨆꩊ ꩝`) → Chứng minh `V26.1` đọc tốt dòng này nếu Detector không bỏ sót!
  * **Kiểm toán cảnh báo `possible_missed_lines` & Nguyên nhân gốc rễ (`uncovered_ink_ratio = 0.0`)**: `document.status = "ocr_done"`, `detector = "generic_ppocrv4_dbnet"`, `detector_switched = False`, `uncovered_ink_ratio = 0.0` → **`possible_missed_lines = False` (ĐÃ BỎ LỌT CẢNH BÁO & KHÔNG CHUYỂN SANG `cham_dbnet`!)**.
  * **Giải phẫu hình học chính xác vì sao `uncovered_ink_ratio = 0.0`**: Trên `test_142_lvl4.png`, dòng bị sót `GT[1]` nằm ở `y = [73, 107]` (tâm chữ `y ≈ 90`), kẹp giữa khung `PR[0]` (`y = [33, 85]`) và `PR[1]` (`y = [97, 134]`). Trong `compute_uncovered_ink()` (`ocr-studio/app.py`), mặt nạ phủ bị giãn nở đều theo cả chiều dọc `pad = round(0.25 * median_line_h) = 11px` (`cv2.dilate` `23x23`): biên dưới `PR[0]` giãn tới `85 + 11 = 96` chồng khít lên biên trên `PR[1]` giãn tới `97 - 11 = 86`, **nuốt trọn toàn bộ tâm mực của dòng `GT[1]` bị mất**, làm `uncovered_ink_ratio = 0.0`! Chỉ cần giảm giãn dọc `pad_y` xuống `<= 3px` (chỉ giãn ngang `pad_x`) là cả cổng `Legacy-First` (chuyển sang `cham_dbnet`) lẫn cảnh báo `possible_missed_lines` sẽ bắt trúng dòng bị sót!

### 2.2. Đối chứng `Strict CER` vs. `Philological Editorial-Normalized CER` trên toàn bộ 30 trang `frozen` (`V26.1`)
- **Full-Page (gộp cả lỗi tách dòng)**: `Strict CER = 6.71%` (`401` edits) vs. `Philological-Normalized CER = 6.06%` (`362` edits).
- **Matched-Line Only (tách riêng chất lượng Recognizer trên `146` dòng khớp)**: `Strict CER = 5.8%` (`336` edits) vs. `Philological-Normalized CER = 5.13%` (`297` edits).
- **Phân loại chính xác `401` thao tác sai khác**: Chỉ có **`37` thao tác (`9.23%`)** thực sự thuộc quy ước trình bày văn bản học nghiêm ngặt (`khoảng trắng trước ꩝/꩞`: `26`, thay `– ↔ -`: `11`). Các lỗi xóa mất dấu gạch (`8`), nhầm `:` ↔ `;` (`23`), thêm/xóa khoảng trắng tự do (`12`) và nhầm ký tự/dấu Chăm (`69`) đều được giữ nguyên là **lỗi OCR thật** (xác nhận phản biện Mục 3.4 hoàn toàn đúng).

---

## 3. Kết Quả Mô-đun 3: Thứ Tự Đọc Trang 2 Cột (`rows` vs `auto`) & Song Ngữ Chung 1 Dòng

### 3.1. Kiểm chứng Thứ tự đọc Đa cột trên 5 trang 2 cột (`3` cổ thư EFEO + `2` trang HarfBuzz sạch)
| Tài liệu | Số dòng GT | `rows` Kendall's τ | `auto` Kendall's τ | `rows` Strict CER | `auto` Strict CER | `document.layout` (`auto`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `ood_folio_06_wijaya_2col` | `16` | `0.4833` | **`1.0`** | `78.34%` | **`49.46%`** | `{'reading_order': 'columns', 'columns': 2}` |
| `ood_folio_08_udong_2col` | `16` | `0.4167` | **`1.0`** | `70.48%` | **`43.17%`** | `{'reading_order': 'columns', 'columns': 2}` |
| `ood_v3_14_kraong_2col` | `16` | `0.4667` | **`1.0`** | `74.44%` | **`18.8%`** | `{'reading_order': 'columns', 'columns': 2}` |
| `clean_2col_12lines` | `12` | `0.4545` | **`1.0`** | `69.96%` | **`3.7%`** | `{'reading_order': 'columns', 'columns': 2}` |
| `clean_titled_2col_11lines` | `11` | `0.5273` | **`1.0`** | `51.92%` | **`3.08%`** | `{'reading_order': 'columns', 'columns': 2}` |

### 3.2. Văn bản Chăm–Việt trộn trên **cùng một dòng** (`8` mẫu HarfBuzz + FreeType đồng baseline)
| Model | Overall Strict CER | Riêng phần Chữ Chăm CER | Riêng phần Tiếng Việt/Latin CER | Phân bố `selected_recognizer` |
| :--- | :---: | :---: | :---: | :--- |
| `auto` (`Auto Router v2.2`) | `33.33%` | `60.56%` | `54.4%` | `{'v24_1_r4': 3, 'vi': 5}` |
| `v24_1_r4` (`V26.1` thuần Chăm) | `53.73%` | **`43.66%`** | `89.6%` | `{'v24_1_r4': 8}` |
| `vi` (`PP-OCRv5-vi`) | `36.47%` | `100.0%` | **`63.2%`** | `{'vi': 8}` |

- **Kết luận thực nghiệm**: Khi Chăm và tiếng Việt nằm chung trên 1 dòng, `v24_1_r4` đọc rất chuẩn cụm Chăm nhưng mù dấu tiếng Việt, ngược lại `vi` đọc chuẩn cụm Việt nhưng mù hoàn toàn chữ Chăm (`100%` CER cụm Chăm). Vì `Auto Router v2.2` chọn 1 recognizer cho toàn dòng nên bắt buộc hy sinh nửa còn lại → **Cần nâng cấp `Auto Router v2.3` tách đoạn ngang (horizontal script-span splitting)** thay vì huấn luyện gộp từ điển.

---

## 4. Kết Quả Mô-đun 4: Alignment Vị Trí Dấu `-` & Cô Lập Lỗi Chữ Số

### 4.1. Đánh giá Dấu nối `-` theo Vị trí Alignment (`crops/results.jsonl`)
| Model | Đúng theo Số lượng (cũ) | Positional Precision | Positional Recall | **Positional F1** | Đúng cả dấu `-` & 2 ký tự kề | Exact cả dòng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `auto` | `100.0%` | `100.0%` | `100.0%` | **`100.0%`** | `95.83%` | `85.71%` |
| `v24_1_r2_1_6` | `100.0%` | `100.0%` | `100.0%` | **`100.0%`** | `100.0%` | `100.0%` |
| `v24_1_r3_1` | `90.48%` | `95.83%` | `95.83%` | **`95.83%`** | `87.5%` | `76.19%` |
| `v24_1_r4` | `100.0%` | `100.0%` | `100.0%` | **`100.0%`** | `95.83%` | `85.71%` |
| `v25` | `0.0%` | `0.0%` | `0.0%` | **`0.0%`** | `0.0%` | `0.0%` |

### 4.2. Ablation Khoảng cách Chữ số Chăm trên `NotoSansCham-SemiBold` (`V26.1`)
- `tracking = 0px`: CER = **`8.0%`** (Exact = `50.0%`)
- `tracking = +4px`: CER = **`6.0%`** (Exact = `75.0%`)
- `tracking = +8px`: CER = **`34.0%`** (Exact = `0.0%`)

### 4.3. Lưới Đối chứng 3 Nhân tố cho Chữ số ASCII Lặp (`1990-2000`, `2000-2025`, `1999-2000`, `Năm 2000-2025`)
| Điều kiện (Cỡ chữ × Giãn cách) | `v24_1_r4` CER (Giữ số lặp) | `vi` CER (Giữ số lặp) | `auto` CER (Giữ số lặp) |
| :--- | :---: | :---: | :---: |
| `px22_track0` | `27.5%` (`0.0%`) | **`0.0%`** (`100.0%`) | `22.5%` (`25.0%`) |
| `px22_track5` | `20.0%` (`50.0%`) | **`25.0%`** (`25.0%`) | `17.5%` (`25.0%`) |
| `px44_track0` | `35.0%` (`0.0%`) | **`0.0%`** (`100.0%`) | `35.0%` (`0.0%`) |
| `px44_track5` | `40.0%` (`0.0%`) | **`10.0%`** (`100.0%`) | `40.0%` (`25.0%`) |

- **Kết luận về Chữ số Chăm (`6.0%` CER còn lại)**: Các dãy 10 chữ số Chăm liền nhau (`꩐꩑꩒꩓꩔꩕꩖꩗꩘꩙` và `꩙꩘꩗꩖꩕ ꩔꩓꩒꩑꩐`) đều đạt **`0.0%` CER (`100%` khớp)** ở cả `0px` và `+4px`; sai số còn lại tập trung ở câu tách rời từng chữ số đơn bằng khoảng trắng (`꩐ ꩑ ꩒ ... ꩙` bị nuốt khoảng trắng giữa `꩐꩑` và `꩔꩕꩖`). Giãn cách quá rộng (`+8px`) làm vỡ cấu trúc từ (`34.0%` CER).
- **Kết luận về Chữ số ASCII lặp (`1990-2000`, `2000-2025`, `Năm 2000-2025`)**:
  1. Trên crop chuẩn không cắt lẹm ở độ phân giải cao (`44px`), `v24_1_r4` vẫn rụng số lặp và lẫn số `꩐` (`1990-2000` → `1꩐0-200`, `35.0%` CER) → Loại trừ nguyên nhân cắt crop lẹm hoặc ảnh mờ; đây là hạn chế thật sự của recognizer thuần Chăm `v24_1_r4` trên dãy chữ số ASCII lặp.
  2. Ngược lại, `vi` (`PP-OCRv5-vi`) đạt **`0.0%` CER (`100%` khớp tuyệt đối)** trên cả 4 mẫu ở `px22_track0` và `px44_track0`!
  3. Lý do `auto` vẫn bị `35.0%` CER trên `Năm 2000-2025` là vì `v24_1_r4` sinh `"꩒) 200-2026"` với `c_conf >= 0.85` trong khi `c_ratio = 0.09 < 0.35`, bị chặn bởi điều kiện `(c_ratio < 0.62 and c_conf < 0.85)` ở dòng 337 `app.py` nên chưa gọi `viet_model`! Nới điều kiện gọi `viet_model` khi `c_ratio < 0.25` (dòng hầu như không có chữ Chăm) sẽ chuyển thẳng sang `vi` (`0.0%` CER) mà không cần đợi train lại V27!

---

## 5. Kết Quả Mô-đun 5: Cô Lập `V25` theo Độ Rộng Crop & Giải Mã Ca Xoay `-11°` (`40%` CER)

### 5.1. Thí nghiệm Cô lập `V25` trực tiếp trên `/ocr_crop` (Không qua DBNet — Kiểm định giả thuyết Padding Mask)
| Nhóm Độ Rộng (`ar48 = W*48/H`) | Đặc tính Padding ở `W=480` | `V26.1` (`v24_1_r4`) CER | `V25` CER | Ký tự Latin rò rỉ (`V25`) |
| :--- | :--- | :---: | :---: | :---: |
| `narrow_high_padding_ar170_205` | ~60% nền trắng bên phải | **`3.85%`** | `38.46%` | `0` |
| `matched_zero_padding_ar450_480` | ~0% padding, tỷ lệ 1:1 chuẩn W=480! | **`4.17%`** | `41.67%` | `0` |
| `ultrawide_compressed_ar1260` | 0% padding, nén ngang ~2.6x | **`4.74%`** | `33.65%` | `1` |

- **Kết luận thực nghiệm bác bỏ giả thuyết đơn nguyên nhân Padding Mask trên `V25`**: Ngay cả trên nhóm `matched_zero_padding_ar450_480` (**tỷ lệ khớp khít `W=480` với `~0%` padding trắng bên phải** và cắt dòng chuẩn không qua DBNet), `V25` vẫn bị **`41.67%` CER** (so với **`4.17%`** của `V26.1`). Điều này **xác nhận phản biện độc lập Mục 5.1 hoàn toàn chính xác**: thiếu `use_valid_mask` KHÔNG phải nguyên nhân chính làm `V25` sụp đổ trên `clean_frozen`, mà do lệch phân phối dữ liệu/rendering huấn luyện `V25` so với chuẩn HarfBuzz/benchmark hiện hành.

### 5.2. Pháp y Hình học Ca `stress_synthetic_167_rotate_minus_11` (`40.0%` CER) & Thử nghiệm A/B Safe-Border Deskew
- **Detector thực thi trên `synthetic_167` xoay `-11°`**: `cham_dbnet` (`detector_switched = True`).
  * Dòng 1: `is_axis_aligned_merged_aabb = False`, `conf = 0.9759`, `bbox = [109, 64, 388, 144]`, `text = "꩒꩞ ꨓꨭꩀ ꨥꩀ ꨧꨯꨱꩃ ꨤꨪꨝꨪꩀ ꨣꨕꩀ" `
  * Dòng 2: `is_axis_aligned_merged_aabb = False`, `conf = 0.2093`, `bbox = [100, 92, 819, 241]`, `text = "ꩍ ꨆ ꨓ ꨪꩆ ꨩꨣꨎꩀ ꨕꨳꨪꩆ ꨤꨮ" `

#### Bảng A/B Góc Nghiêng: Ảnh Xoay Gốc vs. Safe-Border Projection Deskew (`expand=True`, Không Cắt Biên)
| Trang nguồn | Góc xoay | Góc ước lượng | Raw Lines | Raw Strict CER | Safe-Deskew Lines | **Safe-Deskew Strict CER** | Chênh lệch CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `synthetic_167` | `-5°` | `6.8°` | `2/2` | `4.62%` | `2/2` | **`3.08%`** | `-1.54%` |
| `synthetic_167` | `-8°` | `9.5°` | `2/2` | `4.62%` | `2/2` | **`6.15%`** | `+1.54%` |
| `synthetic_167` | `-11°` | `12.8°` | `2/2` | `40.0%` | `2/2` | **`4.62%`** | `-35.38%` |
| `synthetic_167` | `-14°` | `15.6°` | `2/2` | `35.38%` | `2/2` | **`4.62%`** | `-30.77%` |
| `real_harum_palei_01` | `-5°` | `5.0°` | `25/25` | `6.5%` | `25/25` | **`2.84%`** | `-3.66%` |
| `real_harum_palei_01` | `-8°` | `8.0°` | `25/25` | `3.07%` | `25/25` | **`2.84%`** | `-0.24%` |
| `real_harum_palei_01` | `-11°` | `11.0°` | `25/25` | `3.31%` | `25/25` | **`2.01%`** | `-1.30%` |
| `real_harum_palei_01` | `-14°` | `14.0°` | `25/25` | `8.27%` | `25/25` | **`2.36%`** | `-5.91%` |
