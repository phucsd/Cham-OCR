# Báo cáo Khắc phục & Tái Kiểm định Trực tuyến `Pipeline v26.3` (Đóng 4 Lỗi `P1` & 2 Vấn đề `P2` từ Báo cáo QA Độc lập 23/11/2026)

- **Ngày thực hiện**: `2026-11-23`
- **Môi trường Production**: `https://ocr.cham.asia` (Hugging Face Space `phucsd/cham-ocr-studio`, commit triển khai `2f700c0b7e3b3bd6aa35aab31f08b173dafd18d5`)
- **Tài liệu đối chiếu**: `Cham_OCR_Independent_QA_2026-11-23.zip` (`BAO_CAO_KIEM_THU_DOC_LAP_OCR_CHAM_2026-11-23.md`, `challenge_shaped_ocr_results.json`, `ui_chrome_e2e.json`)
- **Bằng chứng định lượng**: `REPORTS/live_v26_3_qa_reverification.json`, `REPORTS/live_hf_r4_verification.json`, `ocr-studio/tests/test_quality_gates.py` (`32/32` unit tests PASS trên cả runtime có và không có Paddle).

---

## 1. Tóm tắt Kết quả Trước và Sau `Pipeline v26.3` trên Production (`https://ocr.cham.asia`)

| Hạng mục kiểm định độc lập | Trước `v26.3` (Báo cáo QA 23/11) | Sau `v26.3` (Đo trực tiếp `https://ocr.cham.asia`) | Trạng thái |
| :--- | :---: | :---: | :---: |
| **Micro-CER toàn bộ 11 ảnh văn bản `SYNTHETIC_SHAPED_QA_IMAGES` (`auto`)** | `40.42%` (`6/11` exact) | **`0.63%`** (`9/11` exact `0.00%`, `2/11` đạt `1.22%`) | **ĐÃ ĐÓNG** |
| **`P1-1` — Nạp lại `.studio.json` lần 2 trên folio đang active (`JSON A -> JSON B`)** | Giữ nguyên `JSON A` trên UI & IndexedDB; `ui_chrome_e2e` trượt `import_studio_json` | Cập nhật tức thì `JSON B` trên DOM + IndexedDB `ChamOCRStudioDB`, giữ nguyên sau `Reload` | **ĐÃ ĐÓNG** |
| **`P1-2` — Bố cục 2 cột có tiêu đề ngắn căn giữa (`shaped_two_col_divider` / `open`)** | CER `52.44%`, `layout = {reading_order: "rows", columns: 1}` | CER **`1.22%`** (`7/7` dòng đúng thứ tự cột trái $\to$ cột phải), `columns: 2` | **ĐÃ ĐÓNG** |
| **`P1-3` — Ảnh âm bản nền đen thưa nét 1 dòng (`shaped_inverted.png`)** | `0` dòng (`blank_page`, CER `100.00%`) | `1/1` dòng, CER **`0.00%`** (`ꨧꨪ ꨀꨆꨢꨯꨮꩅ ?`), `0/56` đảo nhầm trên `RealScan-Gold v4.1` | **ĐÃ ĐÓNG** |
| **`P1-4` — Dòng song ngữ Việt đứng trước Chăm (`vi_cham_inline.png`, `auto`)** | CER `50.00%` (`route = ["v24_1_r4"]`, rác chữ số Chăm `ꨚ꩗꩐꩐꩖`) | CER **`0.00%`** (`route = ["v24_1_r4+vi"]`, `Người Chăm: ꨚꨗꩍ ꨓꨶꨮꩉ ?`) | **ĐÃ ĐÓNG** |
| **Toàn bộ 4 chiều dòng song ngữ nội dòng (`cham_vi`, `vi_cham`, `cham_fr`, `cham_latin_cham`)** | `3/4` đạt `0.00%`, `1/4` lỗi `50.00%` | **`4/4` đạt `0.00%` CER** (`v24_1_r4+vi`) | **ĐÃ ĐÓNG** |
| **`P2-1` — Tải đồng thời 3 worker quét xoay vòng toàn bộ 8 model (`HTTP 502` OOM)** | `6 / 143` request lỗi `HTTP 502` đồng thời | **`0 / 32` lỗi `HTTP 5xx/502`** (`32/32` đạt `HTTP 200`, giới hạn LRU `max=2` model phụ) | **ĐÃ ĐÓNG** |
| **`P2-2` — `test_quality_gates.py` kiểm tra `v25.installed` đa phiên bản Paddle** | Fail `1/31` trên Paddle `2.6.2` (`False is True`) | **`32/32` PASS** trên cả Python không cài Paddle và Python 3.13 + Paddle `3.3.1` | **ĐÃ ĐÓNG** |
| **Kiểm định chống hồi quy `harum_palei_01` & `ocr-audit-2026-11-04` (18 trang)** | `2.60%` CER (`25/25` dòng) / `8.97%` Doc CER (`134/134` dòng) | **`2.60%` CER (`25/25` dòng)** / **`8.97%` Doc CER (`134/134` dòng)** | **KHÔNG HỒI QUY** |

---

## 2. Phân tích Nguyên nhân Gốc rễ & Giải pháp Kỹ thuật đã Triển khai

### 2.1. `P1-1`: Lỗi ghi đè ngược `state.lines` cũ khi nạp `.studio.json` trên Folio đang mở (`ocr-studio/static/js/studio.js`)
- **Nguyên nhân gốc rễ**: Trong `applyImportedJsonPayload()`, sau khi gán `folio.lines = entry.lines...` và lưu `await dbPutFolio(folio)`, hàm gọi `await selectFolio(folio.id)` để vẽ lại giao diện. Tuy nhiên, dòng đầu tiên của `selectFolio()` lại gọi `syncActiveFolioToQueue()`, chép ngược mảng `state.lines` cũ (từ lần nạp `JSON A` hoặc kết quả `Transcribe` trước đó) đè lên `folio.lines` mới vừa nạp rồi lưu đè lại xuống IndexedDB.
- **Giải pháp**:
  1. Cập nhật đồng bộ `state.lines = folio.lines`, `state.documentMeta` và `state.profilerMeta` ngay trước `await dbPutFolio(folio)` khi `folio.id === state.activeFolioId`.
  2. Bổ sung tùy chọn `{ skipSyncActive: true }` vào `selectFolio(folioId, options)` để bỏ qua `syncActiveFolioToQueue()` khi kích hoạt lại từ luồng import JSON.
  3. Hiển thị thông báo rõ ràng `"No matching folio found in queue for imported JSON"` khi `hydratedFolios === 0` thay vì báo thành công giả.

### 2.2. `P1-2`: Bố cục 2 cột có tiêu đề ngắn căn giữa bị đọc ngang từng hàng (`ocr-studio/app.py`)
- **Nguyên nhân gốc rễ**: `order_lines_reading()` chỉ tách trước các dòng tiêu đề/chân trang có bề rộng `> 0.6 * text_w`. Khi tiêu đề căn giữa ngắn (`* ꨔꩅ ꨚꨗꩍ *`, rộng `144px = 0.175 * text_w`), hộp chữ này rơi vào `_split_columns()` tạo thành một cụm giữa chỉ có 1 dòng và làm điều kiện `any(len(c[2]) < 3 for c in cols)` bác bỏ toàn bộ bố cục 2 cột.
- **Giải pháp**: Bổ sung cơ chế bóc tách tối đa 2 dòng đầu trang (leading headers) và/hoặc 2 dòng cuối trang (trailing footers) có khoảng tách dọc độc lập (`y2 <= body_y1 + 6`) trong `order_lines_reading()`, đồng thời truyền `col_divider_x` từ `segment_lines_dbnet()` sang bộ sắp thứ tự đọc để bảo toàn rãnh phân cột dọc khi phần thân trang tách sạch thành $\ge 2$ cột song song.

### 2.3. `P1-3`: Ảnh âm bản nền đen thưa nét 1 dòng bị bỏ sót ở cấp trang (`ocr-studio/app.py`)
- **Nguyên nhân gốc rễ**: `check_inverted_polarity()` chỉ xét vùng trung tâm `[14%..86%]` và phân vị `p99.2` (`0.8%` diện tích). Với ảnh 1 dòng chữ mảnh ở góc trái (`shaped_inverted.png`, mật độ nét trắng `0.14%`), `p99.2 == 0` khiến ảnh không được đảo cực trước khi qua cổng lọc trang trắng và DBNet.
- **Giải pháp**: Mở rộng kiểm tra song song `roi_mid` và `roi_wide` (`[4%..96%]`), bổ sung nhánh phát hiện nét trắng mảnh bằng phép biến đổi hình thái học Top-Hat (`cv2.MORPH_OPEN` `21×21`: `op_p985 < 65.0` kết hợp `(tophat > 60).sum() >= 28` và `p99.92(tophat) > 55.0`).
- **Kiểm chứng an toàn**: Đạt **`50 / 50`** đúng trên ảnh âm bản chuẩn đoán, **`0 / 56`** đảo cực nhầm trên toàn bộ `RealScan-Gold v4.1` (kể cả các trang viền đen EFEO/EAP) và **`0 / 2,585`** dương tính giả trên tập ảnh chuẩn đoán thường, đồng thời trả `False` trên ảnh đen trơn rỗng.

### 2.4. `P1-4`: Bất đối xứng khi cụm từ tiếng Việt ngắn đứng đầu dòng chữ Chăm (`ocr-studio/app.py`)
- **Nguyên nhân gốc rễ**: Khi từ tiếng Việt ngắn `"Người Chăm:"` đứng đầu dòng, mô hình thuần Chăm `v24_1_r4` giải mã phần chữ Latinh có dấu thành chuỗi phụ âm ghép liền chữ số Chăm `"ꨚ꩗꩐꩐꩖ ꨏ/ꨮ;"` (`U+AA50..U+AA59`), giữ tỷ lệ ký tự Chăm `c_ratio = 0.89 >= 0.75` và lọt qua biểu thức `_ROUTER_CHAM_SOUP_RE` cũ (vốn chỉ kiểm tra chữ số ASCII `[0-9]`).
- **Giải pháp**: Mở rộng `_ROUTER_CHAM_SOUP_RE` để bắt mẫu phụ âm Chăm dính trực tiếp chữ số Chăm (`[\uAA00-\uAA4F][\uAA50-\uAA59]+` / `[\uAA50-\uAA59]+[\uAA00-\uAA4F]`) hoặc dấu gạch chéo `/[\uAA00-\uAA5F]`, đồng thời hạ ngưỡng chấp nhận nhánh tiếng Việt nội dòng xuống `0.74` khi phát hiện chuỗi soup chữ Chăm ở nửa đầu dòng.

### 2.5. `P2-1` & `P2-2`: Quản lý bộ nhớ LRU đa Model & Tương thích Kiểm thử `v25`
- **Giới hạn bộ nhớ C++ Predictor (`P2-1`)**: Tách nhóm model ghim cố định (`v24_1_r4`, `v24_1_r3_1`, `vi`, `auto`) khỏi các model đối chiếu lịch sử (`v23`, `v24`, `v24_1_r2_1_6`, `v25`), áp dụng cơ chế thu hồi **LRU tối đa `2` model lịch sử** kèm `gc.collect()`, và tinh gọn preload lúc khởi động còn `['v24_1_r4', 'v24_1_r3_1']`.
- **Tương thích kiểm thử đa phiên bản Paddle (`P2-2`)**: Cập nhật `test_workbench_samples_models_and_corpus_zip_export` kiểm tra động `ids['v25']['installed'] is (not (0 < paddle_major < 3))` và bổ sung test thứ 32 `test_stage_v26_3_qa_p1_p2_regressions`.

---

## 3. Kết quả Chi tiết 12 Ảnh `SYNTHETIC_SHAPED_QA_IMAGES` trên `https://ocr.cham.asia` (`model="auto"`)

| ID trường hợp | Dòng phát hiện / GT | CER (`auto`) | Bố cục (`layout`) | Định tuyến (`selected_recognizer`) |
| :--- | :---: | :---: | :---: | :--- |
| `shaped_clean` | `1 / 1` | **`0.00%`** | `rows (1 col)` | `["v24_1_r3_1"]` |
| `shaped_inverted` | `1 / 1` | **`0.00%`** | `rows (1 col)` | `["v24_1_r4"]` |
| `shaped_rotate_minus12` | `1 / 1` | **`0.00%`** | `rows (1 col)` | `["v24_1_r4"]` |
| `shaped_darkframe` | `1 / 1` | **`0.00%`** | `rows (1 col)` | `["v24_1_r4"]` |
| `shaped_lowcontrast` | `1 / 1` | **`0.00%`** | `rows (1 col)` | `["v24_1_r4"]` |
| `cham_vi_inline` | `1 / 1` | **`0.00%`** | `rows (1 col)` | `["v24_1_r4+vi"]` |
| `vi_cham_inline` | `1 / 1` | **`0.00%`** | `rows (1 col)` | `["v24_1_r4+vi"]` |
| `cham_fr_inline` | `1 / 1` | **`0.00%`** | `rows (1 col)` | `["v24_1_r4+vi"]` |
| `cham_latin_cham_inline` | `1 / 1` | **`0.00%`** | `rows (1 col)` | `["v24_1_r4+vi"]` |
| `shaped_two_col_divider` | `7 / 7` | **`1.22%`** | `columns (2 cols)` | `["v24_1_r4"] × 7` |
| `shaped_two_col_open` | `7 / 7` | **`1.22%`** | `columns (2 cols)` | `["v24_1_r4"] × 7` |
| `shaped_negative_ruled` | `0 / 0` | **PASS (`0` dòng)** | `no_text` | `[]` |

---

## 4. Rà soát Phản biện Chuyên sâu Vòng 2 (`v26.3.1` Red-Team Edge-Case Audit)

Khi thực hiện rà soát phản biện toàn diện từng dòng mã lần thứ 2, chúng tôi phát hiện và đóng thêm **5 trường hợp biên tiềm ẩn** chưa lộ ra ở bộ test vòng 1:

1. **Tương thích trình duyệt Safari/WebKit & Giữ nguyên mức Zoom khi nạp lại JSON trên cùng Folio (`studio.js`)**:
   - **Phát hiện**: Trong `selectFolio()`, gán lại `imgEl.src` bằng chính chuỗi `data:image/png;base64,...` đang hiển thị có thể không kích hoạt sự kiện `imgEl.onload` trên Safari/WebKit (khiến khung soạn thảo không vẽ lại), đồng thời trên Chrome lại gọi `zoomToFit()` làm mất mức phóng đại hiện tại của người dùng.
   - **Khắc phục**: Khi `imgEl.src === nextSrc && imgEl.complete && imgEl.naturalWidth > 0`, gọi đồng bộ `finishImageRender(false)` để cập nhật ngay lớp phủ và khung văn bản mà không reset mức zoom (`25% -> 25%`).
2. **Bảo toàn chỉnh sửa thủ công chưa lưu trên `folio_1` khi tải đồng thời `folio_2.png + folio_2.studio.json` (`studio.js`)**:
   - **Phát hiện**: Biểu thức cũ `{ skipSyncActive: hydratedFolios > 0 }` bỏ qua `syncActiveFolioToQueue()` ngay cả khi tệp JSON vừa nạp thuộc về `folio_2` mới thêm, dẫn tới nguy cơ mất chỉnh sửa trên `folio_1` đang mở trước khi chuyển sang `folio_2`.
   - **Khắc phục**: Theo dõi tập `hydratedIds` thực tế và chỉ bật `skipSyncActive` khi `hydratedIds.has(state.activeFolioId)`.
3. **Chặn ghi đè nhầm khi hàng đợi có $\ge 2$ Folio và người dùng chọn nhầm tệp JSON khác tên (`studio.js`)**:
   - **Phát hiện**: Cơ chế fallback `entries.length === 1` sang `state.activeFolioId` nếu chạy khi hàng đợi có nhiều trang sẽ ghi đè nhầm `folio_999.studio.json` vào trang đang mở.
   - **Khắc phục**: Giới hạn fallback về `activeFolioId` chỉ khi `!targetStem || state.queue.length === 1`; khi hàng đợi có $\ge 2$ trang và tên không khớp bất kỳ trang nào, hiển thị cảnh báo `"No matching folio found in queue for imported JSON"`.
4. **Đảo cực đồng nhất cho cả `method="valley"` và `method="adaptive"` trên ảnh âm bản (`app.py`)**:
   - **Phát hiện**: `segment_lines_adaptive()` và `segment_lines_valleys()` trước đó cắt dòng trên ảnh gốc chưa đảo cực `img` thay vì `proc_img`, và `segment_lines_advanced()` chưa đảo cực `img` trước bước nhị phân hóa `THRESH_BINARY_INV`.
   - **Khắc phục**: Cắt dòng từ `proc_img` và bổ sung đảo cực đầu hàm `segment_lines_advanced()`, giúp cả 3 phương pháp `dbnet`, `valley` và `adaptive` đều nhận diện thành công ảnh âm bản `shaped_inverted.png`.
5. **Thứ tự bóc tách tối thiểu `(n_top, n_bot)` cho trang 2 cột có đồng thời Tiêu đề ngắn + Số trang cuối (`app.py`) & Khóa luồng `_INFERENCE_LOCK` trực tiếp trong `get_ocr_model()`**:
   - **Khắc phục**: Kiểm tra tổ hợp `(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)` theo thứ tự số dòng bóc tách tăng dần (tránh bóc tham lam dòng đầu cột trái khi trang có cả tiêu đề và số trang chân trang), đồng thời bọc `_INFERENCE_LOCK` ngay trong `_register_loaded_model()` / `get_ocr_model()` để an toàn tuyệt đối đa luồng.

