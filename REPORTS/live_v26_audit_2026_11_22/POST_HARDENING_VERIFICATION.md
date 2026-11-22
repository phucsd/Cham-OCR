# Nghiệm thu Hậu kiểm & Ưu tiên A trên Production (`https://ocr.cham.asia`)
- **Thời điểm (UTC)**: `2026-11-22T08:07:10Z`
- **HF Space Commit**: `1af2d681733bb88c1a6c9bdd1be63fb2dcdabdff`
- **Nguyên tắc bảo toàn lịch sử**: Toàn bộ tệp kết quả gốc ngày `2026-11-22` (`digits/summary.json`, `api_contract/results.json`, ...) được giữ nguyên vẹn làm bằng chứng lịch sử; kết quả kiểm chứng lại xuất riêng tại tệp này.

## 1. Hợp đồng API HTTP `400` & Danh mục Model (`PASS` `6/6` + `V25 Bilingual (Legacy)`)
| Trường hợp | Lịch sử (22/11) | Hiện tại | Thông báo lỗi / Trạng thái |
| :--- | :---: | :---: | :--- |
| `missing_image` | HTTP `502` | **HTTP `400`** | `Missing or empty 'image' field (expected base64-encoded image)` |
| `invalid_model` (`nonexistent_model_2026`) | HTTP `200` (fallback `v23`) | **HTTP `400`** | `Unknown model 'nonexistent_model_2026'. Allowed models: v24_1_r4, auto, v24_1_r3_1, v24_1_r2_1_6, vi, v25, v24, v23` |
| `invalid_segmentation` (`not_a_method`) | HTTP `200` (rơi vào `adaptive`) | **HTTP `400`** | `Invalid segmentation method 'not_a_method'. Allowed methods: dbnet, valley, adaptive` |
| `invalid_reading_order` (`invalid`) | HTTP `200` | **HTTP `400`** | `Invalid reading_order 'invalid'. Allowed values: auto, rows` |
| `invalid_bbox_zero` (`[0,0,0,0]`) | HTTP `200` (crop `1x1`) | **HTTP `400`** | `Invalid 'bbox' [0, 0, 0, 0]: must satisfy x2 > x1 and y2 > y1` |
| `invalid_bbox_negative` (`[-200,-200,-1,-1]`) | HTTP `200` (crop `1x1`) | **HTTP `400`** | `Invalid 'bbox' [-200, -200, -1, -1]: region lies outside image bounds (850x249)` |
| `/api/models` (`v25`) | `V25 SOTA` | **`V25 Bilingual (Legacy)`** | `group = Historical Baselines`, `deprecated = True` |

## 2. `Auto Router v2.2` & Đối chứng `reading_order` (`rows` vs `auto`)
- **`vietnamese_00`**: Lịch sử (`rows`) `26.02%` CER → Hiện tại: `reading_order='rows'` đạt **`4.88%` CER**, `reading_order='auto'` đạt **`4.88%` CER** (cả 3 dòng đều có `selected_recognizer = "vi"`; dòng 3 nhận đúng `0123456789`, còn khác biệt nhỏ ở khoảng trắng/dấu câu). Lưu ý: văn bản Chăm–Latin trộn trên **cùng một dòng** vẫn cần bộ kiểm thử riêng ở Ưu tiên B.
- **`mixed_vi_cham_00`**: `reading_order='rows'` đạt **`0.0%` CER**, `reading_order='auto'` đạt **`0.0%` CER** (dòng 1 → `"vi"`, dòng 2 → `"v24_1_r4"`).

## 3. Ưu tiên A — Provenance `selected_recognizer` Chính xác trên Mọi Nhánh Segmentation (Xóa bỏ Suy đoán Hậu kỳ)
- Đã truyền trực tiếp `routed_version` qua cả **Pass 1**, **Pass 2**, `seen_boxes` và `deferred_copies` trong `segment_lines_advanced()` (`valley`, `adaptive`), đồng thời xóa bỏ hoàn toàn nhánh đoán model từ dải ký tự đầu ra (`U+AA00..U+AA5F`).
- **`mixed_vi_cham_00` (`model='auto'`)**: `dbnet`, `valley`, `adaptive` đều báo cáo chính xác dòng 1 → `"vi"`, dòng 2 → `"v24_1_r4"`.
- **Kiểm tra chống đoán mò ký tự (`"123 - 456"` thuần ASCII, không có chữ cái `a-z` nên `Auto Router` giữ ở nhánh Chăm)**: `/ocr_crop` → `selected_recognizer = "v24_1_r4"`, `dbnet` → `"v24_1_r4"`, `valley` → `"v24_1_r4"`, `adaptive` → `"v24_1_r4"` (không bị đoán nhầm thành `"vi"`).

## 4. Đối chứng Chữ số Chăm `U+AA50..U+AA59` (Nguyên khung `1100x100` vs Cắt sát khung mực)
- **Ảnh nguyên khung lịch sử (`1100x100` nền trắng chưa cắt, `RecResizeImg` nén ngang ~75% khoảng trắng vào `W=480`)**: CER = `14.0%`, Đúng toàn dòng = `33.33%` (`4/12`).
- **Ảnh cắt sát biên mực thực tế (`ImageOps.invert(...).getbbox()`, `pad_x=12, pad_y=8` trên cùng `12` câu Light/Medium/SemiBold)**: CER = **`6.0%`**, Đúng toàn dòng = **`75.0%`** (`9/12`).
- **Diễn giải khoa học chuẩn xác**: Không thể quy toàn bộ `14.0%` là nhiễu giả — có **`8.0` điểm %** đến từ sai số nén ngang khung ảnh `1100x100`, trong khi **`6.0%` CER còn lại (`3/12` dòng chưa khớp tuyệt đối) là lỗi nhận dạng thật sự của `V26.1`** trên các chuỗi chữ số Chăm dày đặc, cần được đưa vào thử nghiệm đối chứng ở Ưu tiên B/D.
