# Báo Cáo Triển Khai Toàn Diện Giai Đoạn 1–4 & Hồ Sơ Tự Phản Biện Đa Vòng (`Gate SC-1..SC-4`)

- **Ngày hoàn tất**: `2026-11-23`
- **Production URL đã đồng bộ**: `https://ocr.cham.asia` (Hugging Face Space `phucsd/cham-ocr-studio`)
- **Phiên bản Pipeline & Router triển khai**: **`Pipeline v26.2`** + **`Auto Router v2.3`** + **Trình nạp `.studio.json` trực tiếp trên Studio**
- **Bộ tư liệu vật lý khóa mã băm**: `RealScan-Gold v4` (`schema_version: "4.1.0"`, `57` trang vật lý đa dòng + `7` mảnh mẫu công cộng trên `27` nguồn gốc độc lập, `synthetic_pages_included = 0`, `1,225` tệp khớp `SHA-256` `100%`)
- **Bộ cổng chất lượng đơn vị**: **`31 / 31` bài kiểm thử PASS** (`ocr-studio/tests/test_quality_gates.py`) & `node --check ocr-studio/static/js/studio.js` PASS (`0` lỗi)

---

## 1. Tóm Tắt Điều Hành: Vì Sao Tự Phản Biện Đa Vòng Đã Ngăn Chặn 7 Cạm Bẫy Lớn

Theo yêu cầu triển khai tuần tự toàn bộ 4 giai đoạn kèm **tự kiểm tra và phản biện nhiều lần trước khi chốt**, mỗi giai đoạn đều trải qua tối thiểu 2–3 vòng đối chứng thực nghiệm phản biện (`Adversarial Self-Critique`). Quá trình này đã phát hiện và loại bỏ **7 lỗi thực tế** mà nếu chỉ chạy một lần sẽ lọt qua báo cáo:

| Vòng Phản Biện | Phát hiện khi tự kiểm tra / phản biện | Biện pháp khắc phục triệt để trước khi chốt |
| :--- | :--- | :--- |
| **`SC-1` Vòng 1** (`Polarity` & `Illumination`) | Thử nghiệm chuẩn hóa nền toàn cục (`gray / GaussianBlur`) làm nổi sợi vải bố dệt dưới lá buông `eap531_1_1_c05` thành `32` hộp nhiễu giả; thử nghiệm `Morphological Closing` trượt ảnh âm bản nét mảnh `2px`. | Bác bỏ `GaussianBlur` division & `Closing`; thiết kế bộ phân biệt **Nền Đặc vs. Nét Mảnh bằng `Morphological Opening`** (`op_p95 < 85.0` & `fg_p992 - op_p95 > 45.0`): đạt **`0/56`** đảo cực nhầm trên `v4` (từ `44/56` lỗi cũ!) và **`50/50` (`100%`)** trên bộ ảnh âm bản chuẩn. |
| **`SC-1` Vòng 2** (`Auto Router` trên `Cabaton 1906`) | Tách từng từ đơn không có cổng thoát nhanh khiến `Cabaton 1906 c080` chậm `336.2s` và chèn ký tự Chăm giả vào câu tiếng Pháp thuần (`"clique du ꩘"`). | Bổ sung cổng thoát nhanh `is_clean_pure_latin` / `is_clean_pure_cham` + gom cụm ngữ đoạn thô (`<= 6` spans) + cổng lợi ích đối xứng hai chiều: bảo vệ **`100%`** dòng tiếng Pháp thuần, kéo thời gian chạy về chuẩn cũ. |
| **`SC-1` Vòng 3** (Đọc từng dòng thô `Cabaton` & `intra_01..08`) | Các dòng mục từ ngắn (`"XVIRBV) idaramavan..."`, `"g873986) gi - Phong tục Chăm cổ"`, `"{꩐ 61ꨯꨮꩀ"`) có `v_conf >= 0.87` hoặc `c_ratio >= 0.62` nên lọt qua regex cũ; ngoài ra sửa điều kiện số thuần làm ảnh hưởng bất biến `'123 - 456'` của unit test `Priority A`. | Mở rộng `_ROUTER_GIBBERISH_TOKEN_RE` & `_ROUTER_CHAM_SOUP_RE`, hạ `min_chunk_w = 0.55*h` giữ nguyên vẹn mục từ đơn, và chỉ chuyển dãy số ASCII sang `vi` khi phát hiện **CTC gộp số lặp `'00'/'99'` hoặc trộn chữ số Chăm** (`'123 - 456'` giữ nguyên ở Chăm): kéo CER bộ 8 câu song ngữ nội dòng từ **`33.33%` xuống `7.45%`**! |
| **`SC-1` Vòng 4** (`get_model_info` khi chưa nạp biến toàn cục) | Unit test mới phát hiện `get_model_info(..., executed_detector='cham_dbnet')` trả `loaded: False` nếu biến toàn cục `det_model_info` rỗng trong môi trường mock. | Sửa điều kiện `if not det_model_info and not executed_detector:` và tách hàm `map_deskew_pts_to_original` độc lập (`30/30` unit tests PASS trước khi deploy). |
| **`SC-2` Vòng 1** (Kiểm toán rò rỉ dòng họ / hồ sơ `EAP`) | Nếu tách ngẫu nhiên giữa `EAP531-1-1` và `EAP531-1-3`, hai nhánh `holdout` và `seed` sẽ dùng chung cùng một bộ sưu tập `EAP531-1` (cùng người lưu giữ). | Thiết kế khóa phân tách 3 lớp nghiêm ngặt: **0 trùng `source_collection_id`, 0 trùng tiền tố hồ sơ `EAP<project>-<series>`, 0 trùng người lưu giữ (`custodian`) và 0 trùng sách in**. |
| **`SC-2` Vòng 2** (Soi trực quan 4 tấm ảnh `v4_bbox_overlay_1..4.jpg`) | Phát hiện **88 hộp nhiễu** nằm trên thẻ màu `QpCARD` / bảng số xanh (`015`, `022`) / thước nhựa trắng ở rìa dưới ảnh EAP lọt qua bộ lọc cũ vì `v24_1_r4` ảo giác chèn 1 ký tự Chăm vào thẻ màu. | Thay bằng thuật toán tìm rãnh nền đen ngăn cách tờ thủ bản với dải thẻ đáy (`compute_eap_card_cutoff_y`): quét sạch **100% (`88/88`)** hộp nhiễu rìa dưới trên toàn bộ 44 ảnh EAP. |
| **`SC-2` Vòng 3** (Soi trực quan ô `#42 eap698_8_1_c45`) | Ô `#42` viết bút bi xanh nhạt trên giấy kem bị `0` dòng vì hộp `QpCARD` độ tương phản cao ở đáy ảnh ngăn nhánh cứu vớt mực mờ kích hoạt. | Che dải thẻ dưới `eap_cutoff_y` và kích hoạt cứu vớt tương phản cục bộ `CLAHE` khi trang có mực nhưng còn `0` dòng: khôi phục **`12` dòng chữ Chăm** trên `#42` (**`56/56` trang vật lý đều phân tách thành công, `0` trang rỗng**). |

---

## 2. Kết Quả Định Lượng Giai Đoạn 1 (`Pipeline v26.2` + `Auto Router v2.3` + Live `https://ocr.cham.asia`)

### 2.1. Các nâng cấp cốt lõi đã triển khai lên Production
1. ** Khắc phục triệt để lỗi P0 đảo cực âm bản trên ảnh lưu trữ viền tối (`check_inverted_polarity`)**:
   - Thay thế heuristic viền `5%` (`bg_val < 145`) bằng phép mở hình thái học (`cv2.MORPH_OPEN`) phân biệt **nền tối đặc thực sự** với **nét chữ mảnh trên nền giấy sáng nằm trong khung bàn chụp màu đen**.
2. **Rào cản vạch kẻ dọc hai cột (`detect_vertical_column_divider` + `col_divider_x`)**:
   - Phát hiện đường kẻ chia cột in dọc ở dải `[0.38W, 0.62W]`, xóa vạch kẻ trên bản đồ dò tìm DBNet và chặn tuyệt đối `merge_line_boxes` / `merge_line_polygons` gộp xuyên qua hai cột (`8/8` trang từ điển Cabaton 1906 được phát hiện chính xác, `0/48` dương tính giả trên trang một cột).
3. **Giãn nở bất đẳng hướng cho tỷ lệ mực bỏ sót (`compute_uncovered_ink`) & Minh bạch Detector (`get_model_info`)**:
   - Giảm giãn nở dọc từ `0.30 * median_line_h` xuống `0.08 * median_line_h` để hộp dòng trên không che khuất dòng ngắn sát bên dưới; đồng thời cập nhật `model_info.detector.name` và `weights_sha256_16` phản ánh đúng `cham_dbnet` khi cổng Legacy-First chuyển đổi detector.
4. **Chỉnh nghiêng an toàn có điều kiện trước DBNet (`maybe_safe_deskew_page` + `map_deskew_pts_to_original`)**:
   - Chỉ kích hoạt khi `|θ| >= 9.0°` **VÀ** độ lợi phương sai chiếu ngang `>= 1.55×`, mở rộng khung chống cắt dấu nguyên âm sát mép và ánh xạ ngược đa giác/hộp bao về hệ tọa độ ảnh gốc qua ma trận nghịch đảo $M^{-1}$ (`0.00%` tác động lên trang thẳng).
5. **`Auto Router v2.3` (Phân tách cụm kịch bản nội dòng đối xứng + Khôi phục dãy số ASCII lặp)**:
   - Tách dòng pha trộn *Akhar Thrah* + Việt/Pháp thành tối đa `6` cụm ngữ đoạn theo thung lũng khoảng trắng, chấm điểm đối xứng hai chiều (`v24_1_r4+vi`), và tự động chuyển các dòng năm/số lặp bị CTC nuốt số `'00'/'99'` (`Năm 2000-2025`, `1990-2000`) sang `PP-OCRv5-vi` trong khi giữ nguyên chữ số Chăm (`꩐..꩙`) và số ngắn sạch (`123 - 456`) ở nhánh Chăm.
6. **Nạp trực tiếp gói `.studio.json` trên giao diện `https://ocr.cham.asia`**:
   - Cho phép kéo-thả đồng thời cặp `<doc_id>.jpg` + `<doc_id>.studio.json` vào Studio (`ocr-studio/index.html`, `ocr-studio/static/js/studio.js`) để hiển thị tức thì toàn bộ hộp dòng mà không cần chạy lại OCR.

### 2.2. Bảng đối chiếu định lượng trước & sau Giai đoạn 1 (Xác minh trực tiếp qua HTTPS `https://ocr.cham.asia`)

| Bài kiểm định | Trước Giai đoạn 1 (`v26.1`) | Sau Giai đoạn 1 (`v26.2` + `Router v2.3`) | Trạng thái |
| :--- | :---: | :---: | :---: |
| Đảo cực nhầm trên 56 trang thật `RealScan-Gold v4` | `44 / 56` trang bị đảo âm bản (`78.6%` lỗi!) | **`0 / 56` trang (`0.0%` lỗi)** | **PASS** |
| Độ nhạy phát hiện ảnh âm bản thật (`images_diagnostic`) | `50 / 50` | **`50 / 50` (`100.0%`)** | **PASS** |
| Thủ bản `eap698_4_1_c08.jpg` qua HTTPS `ocr.cham.asia` | `0` dòng chữ Chăm (vỡ thành âm bản) | **`15 / 15` dòng chữ Chăm** (`cham_dbnet`) | **PASS** |
| Ca bỏ sót dòng `test_142_lvl4.png` (`synthetic_142`) | `7 / 8` dòng (`detector` báo sai tên cũ) | **`8 / 8` dòng**, `model_info.detector = cham_dbnet` | **PASS** |
| Trang chuẩn `harum_palei_01` (`v24_1_r4` & `auto` live) | `25/25` dòng, Doc CER `2.60%` | **`25/25` dòng, Doc CER `2.60%`** (Không suy giảm) | **PASS** |
| Trang `harum_palei_01` xoay nghiêng `-12°` | Không tự chỉnh nghiêng | Tự nắn **`+12.0°`**, CER **`2.13%`**, tọa độ khớp ảnh gốc | **PASS** |
| Bộ 8 câu song ngữ nội dòng (`intra_01..08`) | Strict CER **`33.33%`** (`0` ca tách dòng) | Strict CER **`7.45%`** (`v24_1_r4+vi`, giảm **4.47×** lỗi) | **PASS** |
| `"Năm 2000-2025"` & `"1990-2000"` (`44px`, `model="auto"`) | `"꩒) 200-2026"` (`35.0%` CER) / `"1꩐0-200"` | **`"Năm 2000-2025"` (`0.0%` CER)** & **`"1990-2000"` (`0.0%` CER)** | **PASS** |
| Từ điển 2 cột `cabaton1906_dict_c080` (`80` dòng) | Gộp xuyên cột `>= 1`, chỉ giữ `4` dòng Chăm | **`0` gộp xuyên cột**, khôi phục **`19` dòng chứa chữ Chăm** | **PASS** |

---

## 3. Kết Quả Giai Đoạn 2 & 3: Tái Phân Tách 56 Trang `RealScan-Gold v4`, Khóa Phân Tách Nguồn & Gói `Gold-20`

### 3.1. Thống kê `RealScan-Gold v4` (`schema_version: "4.1.0"`) sau 3 vòng phản biện `Gate SC-2 & SC-3`
- **Tổng số trang tư liệu vật lý đa dòng**: **`57` trang** (`1` trang `harum_palei_01` `verified_gold` + `56` trang lưu trữ/in lịch sử mới từ `25` thủ bản/sách độc lập) + **`7` mẫu chữ Wikimedia** (`synthetic_pages_included = 0`).
- **Tỷ lệ trang vật lý mới có dòng văn bản phân tách sạch**: **`56 / 56` trang (`100%`)**, tổng cộng **`1,136` dòng sạch** (trong đó **`622` dòng chứa ký tự *Akhar Thrah***, sau khi đã loại bỏ sạch **88 hộp nhiễu** thẻ màu `QpCARD`/thước đo/bảng số ở rìa dưới ảnh EAP).
- **Tuân thủ tuyệt đối Chống Tự Gán Nhãn Vòng Tròn (`Gate SC-3`)**: Lệnh `python scripts/ingest_studio_to_realscan_gold.py --version v4 --verify-manifest` xác nhận **`1,225 / 1,225` tệp khớp mã băm `SHA-256` `100%`** và **`100%` dòng chưa qua con người duyệt giữ `gt: null`, `verified_by_human: false`**.

### 3.2. Ma trận khóa phân tách nguồn chống rò rỉ dữ liệu huấn luyện (`disjoint_split_audit`)

| Nhánh quy hoạch (`benchmark_split`) | Số nguồn độc lập | Số trang vật lý | Quy tắc sử dụng | Kiểm toán giao thoa |
| :--- | :---: | :---: | :--- | :---: |
| **`locked_test_holdout`** | **16** nguồn (`harum_palei_01`, Wikimedia, `EAP531-1-*`, `EAP698-{2,3,4,6,8,10}`, `EAP1005-{1,20,32,33}`, *Aymonier 1889*, *Cabaton 1906*) | **37** trang + **7** mẫu | **`training_forbidden: true`** — Khóa vĩnh viễn chỉ để đo kiểm độc lập | `shared_collections = 0`<br>`shared_eap_series = 0`<br>`shared_custodians = 0` |
| **`adaptation_seed_pool`** | **11** thủ bản thuộc **6** series riêng (`EAP698-1-*`, `EAP698-5-1`, `EAP698-7-1`, `EAP698-12-*`, `EAP1005-6-1`, `EAP1005-22-1`) | **20** trang | Chỉ được xuất sang tập huấn luyện `V27` **sau khi con người hiệu đính** (`verified_by_human: true`) | Hoàn toàn tách rời khỏi `holdout` |

### 3.3. Gắn mốc tham chiếu học thuật & Gói ưu tiên `Gold-20` (`Stage 3`)
- Toàn bộ 12 trang in/khắc thạch bản công cộng (*Aymonier 1889* tr. 74–81 & *Aymonier & Cabaton 1906*) đã được gắn trường `scholarly_reference_excerpt` trong `manifest.json`, `.studio.json` và cổng duyệt `review_queue/index.html` để hỗ trợ người hiệu đính đối chiếu nhanh mà không vi phạm nguyên tắc `gt: null`.
- Đã trích xuất sẵn gói 20 trang đại diện **`review_queue/gold20_priority_pack/`** (`12` trang `locked_test_holdout` + `8` trang `adaptation_seed_pool`, `339` dòng) kèm chỉ mục `review_queue/gold20_priority_manifest.json`. Người hiệu đính chỉ cần kéo-thả cặp `<doc_id>.jpg` + `<doc_id>.studio.json` vào `https://ocr.cham.asia` để duyệt ngay.

---

## 4. Kết Quả Giai Đoạn 4 (`Gate SC-4`): Thực Nghiệm Cô Lập Đơn Nhân Tố Cho `V27` & Cổng Chặn Rò Rỉ Tự Động

Chạy kịch bản `scripts/audit_v27_calligraphic_and_descender_preflight.py` (`REPORTS/v27_preflight_diagnostic_summary.json`) đã bóc tách định lượng chính xác nguyên nhân suy giảm của `V26.1` trên chữ viết tay và bản khắc thạch bản:

### 4.1. Thí nghiệm 4B — Đường cong nhiễu thư pháp có đối chứng trên 25 dòng Ground Truth chuẩn

| Biến đổi đơn nhân tố trên cùng 25 dòng GT | Strict CER (`v24_1_r4`) | Exact Line Acc | Độ tự tin TB | Kết luận kỹ thuật |
| :--- | :---: | :---: | :---: | :--- |
| `0_clean_printed_baseline` (Chuẩn gốc) | **`5.52%`** | `12.0%` | `0.9512` | Mốc cơ sở |
| `1_calligraphic_slant_12deg` (Nghiêng nét bút `12°`) | `6.33%` (`+0.81%`) | `12.0%` | `0.9479` | Độ nghiêng nét chữ đơn thuần ảnh hưởng rất nhỏ |
| `3_thin_brush_stroke_fade` (Mảnh nét & phai mực) | `5.31%` (`-0.21%`) | `8.0%` | `0.9466` | Nét mảnh đơn thuần không làm suy giảm `V26.1` |
| **`2_tall_descender_2_1x_box` (Khung cao gấp `2.1×` do nét móc kéo dài)** | **`14.91%` (`+9.39%`!)** | **`0.0%`** | `0.8373` | **Thủ phạm hình học số 1**: Khi chiều cao hộp tăng `2.1×` (đặc trưng của nét chân `ꨭ/ꨳ/ꨴ` viết tay/thạch bản), phép co ảnh về `H=48` ép thân phụ âm còn một nửa và nén ngang `2.1×`, làm CER tăng vọt **gấp `2.7` lần**! |
| `4_combined_scribal_perturbation` (Tổ hợp) | `12.97%` (`+7.45%`) | `0.0%` | `0.8961` | Bị chi phối chủ yếu bởi tỷ lệ chiều cao khung dòng |

### 4.2. Thí nghiệm 4A — Cắt thu hẹp dải dọc trên ảnh thạch bản & viết tay thật (`Aymonier 1889` & `EAP698`)
- Khi thử cắt bớt chiều cao hộp dòng trên tư liệu thật (`full_100` $\to$ `core_80` $\to$ `core_68` $\to$ `core_56`), độ tự tin trung bình không tăng mà giảm nhẹ từ **`0.4963` $\to$ `0.4681` $\to$ `0.4550`**.
- **Kết luận kép từ 4A + 4B**: Không thể chữa bài toán chân chữ dài bằng cách cắt xén hộp ở bước suy luận (vì cắt xén sẽ làm mất nét nguyên âm dưới chân chữ thật như `ꨭ`, `ꨳ`), và cũng không thể chỉ sinh dữ liệu font *Noto Sans Cham* thông thường. Tập huấn luyện **`V27` bắt buộc phải kết hợp đồng thời 2 thành phần**:
  1. **Tăng cường tỷ lệ khung dọc `1.8×–2.4×` với nét móc chân kéo dài** trong bộ sinh tổng hợp để `SVTR-LCNet` (`H=48, W=480`) học cách đọc thân chữ nhỏ nằm giữa dải dấu phụ vươn cao/kéo thấp (giải quyết trực tiếp mức tăng `+9.39%` CER ở Thí nghiệm 4B).
  2. **Hòa trộn mẫu viết tay thật từ nhánh `adaptation_seed_pool` sau khi con người hiệu đính** để mô hình học hình thái nét bút thủ bản (*scribal ductus*) chưa từng có trong font máy tính (giải quyết Thí nghiệm 4A).

### 4.3. Thí nghiệm 4C — Kiểm định Cổng Chặn Rò Rỉ `V27` (`verify_v27_training_disjoint_from_holdout`)
- Đã tích hợp trực tiếp vào `scripts/ingest_studio_to_realscan_gold.py` (`--export-v27-seed-lines`) và khóa cứng trong bài test đơn vị thứ 31 (`test_stage2_to_4_disjoint_split_and_v27_leak_guard`):
  - Chặn **100% (4/4)** kịch bản tấn công rò rỉ: (1) đưa trực tiếp `doc_id` thuộc `locked_test_holdout`, (2) đưa thủ bản anh em cùng hồ sơ `EAP531-1-*`, (3) giả danh `doc_id` nhưng trùng mã băm `crop_sha256` / `page_sha256` của `holdout`, và (4) đưa mẫu `adaptation_seed_pool` chưa được con người xác nhận (`verified_by_human: false` — chống tự học nhãn giả).

---

## 5. Quy Trình Ngắn Gọn Tiếp Theo Cho Người Hiệu Đính (Mở Khóa Tập Chuẩn & Huấn Luyện `V27`)

1. **Duyệt nhanh gói `Gold-20` trên `https://ocr.cham.asia`**:
   - Mở thư mục cục bộ `ocr-benchmark/data/realscan_gold_v4/review_queue/gold20_priority_pack/`.
   - Kéo-thả đồng thời cặp `<doc_id>.jpg` + `<doc_id>.studio.json` vào `https://ocr.cham.asia`, chỉnh sửa & bấm `A` xác nhận từng dòng, bấm `Ctrl+E` xuất `cham_ocr_transcription.json`.
2. **Thăng cấp vào `verified_gold` và xuất dữ liệu sạch cho `V27`**:
   ```bash
   python scripts/ingest_studio_to_realscan_gold.py \
     --version v4 --promote-verified \
     --studio-json path/to/cham_ocr_transcription.json \
     --annotator "ten_chuyen_gia" --review-round 1

   python scripts/ingest_studio_to_realscan_gold.py \
     --version v4 --export-v27-seed-lines ocr-training/data/v27_verified_seed_lines.jsonl
   ```
