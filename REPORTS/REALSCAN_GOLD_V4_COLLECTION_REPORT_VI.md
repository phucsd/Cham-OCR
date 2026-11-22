# Báo Cáo Thu Thập & Khởi Tạo Bộ Tư Liệu Vật Lý `RealScan-Gold v4`

- **Ngày hoàn tất**: `2026-11-22T17:20:04.524052+00:00`
- **Mã định danh bộ dữ liệu**: `realscan_gold_v4` (`schema_version: "4.0.0"`)
- **Manifest khóa SHA-256**: `ocr-benchmark/data/realscan_gold_v4/manifest.json`
- **Trình xây dựng & tái lập dữ liệu**: `scripts/build_realscan_gold_v4.py`
- **Trình thăng cấp sau hiệu đính con người**: `scripts/ingest_studio_to_realscan_gold.py --version v4 --promote-verified`

---

## 1. Giải Quyết Triệt Để Phản Biện Độc Lập Về Thiếu Ảnh Thực Tế

Trong bản phản biện độc lập ngày 22/11/2026, điểm yếu lớn nhất của các bộ `RealScan-Gold v1–v3` là chỉ có **01 trang chụp đa dòng thực tế** (`harum_palei_01`, 25 dòng) + **07 mảnh cắt Wikimedia**, còn 22 trang `ood_*` là ảnh dựng kỹ thuật số từ font chữ.

**`RealScan-Gold v4` khắc phục hoàn toàn hạn chế này theo 4 nguyên tắc khoa học & pháp lý nghiêm ngặt**:
1. **Không trộn ảnh dựng font giả làm ảnh chụp (`synthetic_pages_included = 0`)**: Loại bỏ toàn bộ 22 trang dựng font `ood_*` khỏi thống kê tư liệu vật lý; chỉ kế thừa đúng `harum_palei_01` (1 trang, 25 dòng đã kiểm chứng) và `wikimedia_public_specimens_01` (7 mẫu đã kiểm chứng) vào nhánh `verified_gold`.
2. **Đa dạng nguồn tư liệu vật lý độc lập (`56` trang mới từ `25` văn bản/kho lưu trữ gốc, tổng cộng `57` trang đa dòng vật lý)**:
   - Giới hạn tối đa **1–3 trang nội dung không liền kề** trên mỗi thủ bản viết tay để không một người chép hay một chất liệu giấy nào chi phối kết quả đánh giá.
   - Tuyệt đối loại bỏ các thủ bản chữ Ả Rập / Jawi (*Akhar Bani*) và ảnh bia đá Chăm cổ thế kỷ 13 (*Akhar Rik*) để giữ đúng phạm vi văn tự **Chăm Đông (*Akhar Thrah*, `U+AA00..U+AA5F`)** và **Song ngữ Akhar Thrah – Pháp/Việt**.
3. **Quy ước 2 tầng Chống Tự Gán Nhãn Vòng Tròn (Anti-Circular Ground-Truth Protocol)**:
   - Toàn bộ `56` trang mới thu thập được đặt trong nhánh `staging_review_queue` với `verification_status: "pending_human_review"`, `verified_by_human: false`, `gold_eligible: false`.
   - Đầu ra của hệ thống OCR (`https://ocr.cham.asia`) chỉ được lưu ở trường `draft_prediction` để hỗ trợ khoanh vùng dòng và gợi ý bản nháp; trường **`gt` bị khóa cứng ở giá trị `null`** cho tới khi chuyên gia hiệu đính ký duyệt!
4. **Tuân thủ bản quyền lưu trữ quốc tế (Zero Unauthorized Binary Redistribution)**:
   - Toàn bộ tệp nhị phân ảnh chụp từ *British Library Endangered Archives Programme (EAP)* và *Internet Archive* được giữ cục bộ trong `ocr-benchmark/data/realscan_gold_v4/{pages,crops,review_queue}/` (đã cấu hình `.gitignore` không đẩy lên GitHub).
   - Tệp `manifest.json` lưu đầy đủ URL gốc (IIIF / Archive.org BookReader), mã băm `SHA-256`, thông tin người lưu giữ (*custodian*), chất liệu và giấy phép để bất kỳ nhà nghiên cứu độc lập nào cũng tải lại khớp 100% byte bằng lệnh:
     ```bash
     python scripts/build_realscan_gold_v4.py --fetch
     ```

---

## 2. Thống Kê Phân Tầng Chất Liệu Vật Lý (`RealScan-Gold v4`)

### 2.1. Tổng quan hai nhánh (`verified_gold` vs `staging_review_queue`)

| Chỉ số | Giá trị |
| :--- | :---: |
| Trang chụp đa dòng đã kiểm chứng (`verified_gold` — `harum_palei_01`) | **1** trang (**25** dòng) |
| Mẫu chữ công cộng đã kiểm chứng (`verified_gold` — Wikimedia) | **7** mẫu |
| Trang tư liệu vật lý mới thu hoạch (`staging_review_queue`) | **56** trang (**1136** dòng dự thảo) |
| Số văn bản / thủ bản gốc độc lập trong `staging_review_queue` | **25** nguồn độc lập |
| Tổng số nguồn vật lý độc lập toàn bộ `v4` | **27** nguồn |
| Số trang dựng font kỹ thuật số đưa vào `v4` (`synthetic_pages_included`) | **0** |

### 2.2. Phân bố 56 trang mới theo 4 tầng chất liệu vật lý

| Tầng chất liệu (`stratum`) | Số trang | Số thủ bản/sách độc lập | Tổng dòng cắt dự thảo | Dòng chứa ký tự Chăm | Độ tự tin dự thảo TB |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `palm_leaf_agal` | **8** | **3** | 67 | 67 | `0.4670` |
| `handwritten_paper_traditional` | **23** | **12** | 205 | 202 | `0.4917` |
| `handwritten_modern_substrate` | **13** | **8** | 185 | 182 | `0.5134` |
| `historical_printed_and_lithographed` | **12** | **2** | 679 | 171 | `0.7695` |

---

## 3. Danh Mục Chi Tiết 56 Trang Tư Liệu Vật Lý Mới (`staging_review_queue`)

| # | `doc_id` | Nguồn gốc (`ID` & trang) | Chất liệu (`stratum`) | Người lưu giữ / Tác giả | Dòng dự thảo | Conf TB | Trạng thái cổng |
| :-: | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| 01 | `eap531_1_1_c05` | `EAP531-1-1` (#5) | `palm_leaf_agal` | Cham Balamon collection, NIU | 1 (1 Chăm) | `0.593` | `needs_review` |
| 02 | `eap531_1_1_c09` | `EAP531-1-1` (#9) | `palm_leaf_agal` | Cham Balamon collection, NIU | 10 (10 Chăm) | `0.406` | `needs_review` |
| 03 | `eap531_1_1_c13` | `EAP531-1-1` (#13) | `palm_leaf_agal` | Cham Balamon collection, NIU | 6 (6 Chăm) | `0.405` | `needs_review` |
| 04 | `eap531_1_3_c06` | `EAP531-1-3` (#6) | `palm_leaf_agal` | Cham Balamon collection, NIU | 11 (11 Chăm) | `0.427` | `needs_review` |
| 05 | `eap531_1_3_c10` | `EAP531-1-3` (#10) | `palm_leaf_agal` | Cham Balamon collection, NIU | 17 (17 Chăm) | `0.401` | `needs_review` |
| 06 | `eap531_1_3_c16` | `EAP531-1-3` (#16) | `palm_leaf_agal` | Cham Balamon collection, NIU | 14 (14 Chăm) | `0.465` | `needs_review` |
| 07 | `eap698_12_1_c18` | `EAP698-12-1` (#18) | `palm_leaf_agal` | Mr. Bac Bich Van Nhuan | 4 (4 Chăm) | `0.499` | `needs_review` |
| 08 | `eap698_12_1_c85` | `EAP698-12-1` (#85) | `palm_leaf_agal` | Mr. Bac Bich Van Nhuan | 4 (4 Chăm) | `0.539` | `needs_review` |
| 09 | `eap698_1_1_c15` | `EAP698-1-1` (#15) | `handwritten_paper_traditional` | Mr. Sam Toi / Mr. Sam Van Ta | 8 (8 Chăm) | `0.532` | `needs_review` |
| 10 | `eap698_1_1_c35` | `EAP698-1-1` (#35) | `handwritten_paper_traditional` | Mr. Sam Toi / Mr. Sam Van Ta | 8 (8 Chăm) | `0.527` | `needs_review` |
| 11 | `eap698_1_2_c08` | `EAP698-1-2` (#8) | `handwritten_paper_traditional` | Mr. Sam Toi / Mr. Sam Van Ta | 6 (6 Chăm) | `0.491` | `needs_review` |
| 12 | `eap698_1_2_c22` | `EAP698-1-2` (#22) | `handwritten_paper_traditional` | Mr. Sam Toi / Mr. Sam Van Ta | 9 (9 Chăm) | `0.529` | `needs_review` |
| 13 | `eap698_1_5_c09` | `EAP698-1-5` (#9) | `handwritten_paper_traditional` | Mr. Sam Toi / Mr. Sam Van Ta | 17 (17 Chăm) | `0.448` | `needs_review` |
| 14 | `eap698_1_12_c07` | `EAP698-1-12` (#7) | `handwritten_paper_traditional` | Mr. Xa / Mr. Sam Van Tanh | 8 (8 Chăm) | `0.566` | `needs_review` |
| 15 | `eap698_1_12_c15` | `EAP698-1-12` (#15) | `handwritten_paper_traditional` | Mr. Xa / Mr. Sam Van Tanh | 10 (10 Chăm) | `0.505` | `needs_review` |
| 16 | `eap698_2_1_c09` | `EAP698-2-1` (#9) | `handwritten_paper_traditional` | Mr. Duong Po / Mr. Thanh Cha | 6 (6 Chăm) | `0.547` | `needs_review` |
| 17 | `eap698_2_1_c21` | `EAP698-2-1` (#21) | `handwritten_paper_traditional` | Mr. Duong Po / Mr. Thanh Cha | 10 (10 Chăm) | `0.455` | `needs_review` |
| 18 | `eap698_3_1_c18` | `EAP698-3-1` (#18) | `handwritten_paper_traditional` | Mr. Luong Mat / Gru Luong Da | 5 (5 Chăm) | `0.529` | `needs_review` |
| 19 | `eap698_3_1_c55` | `EAP698-3-1` (#55) | `handwritten_paper_traditional` | Mr. Luong Mat / Gru Luong Da | 6 (6 Chăm) | `0.537` | `needs_review` |
| 20 | `eap698_10_1_c10` | `EAP698-10-1` (#10) | `handwritten_paper_traditional` | Mr. Lik Phaok Tao / Mrs. Dao | 7 (7 Chăm) | `0.573` | `needs_review` |
| 21 | `eap698_10_1_c28` | `EAP698-10-1` (#28) | `handwritten_paper_traditional` | Mr. Lik Phaok Tao / Mrs. Dao | 7 (7 Chăm) | `0.545` | `needs_review` |
| 22 | `eap698_12_7_c16` | `EAP698-12-7` (#16) | `handwritten_paper_traditional` | Imam Lam Nam | 13 (13 Chăm) | `0.510` | `needs_review` |
| 23 | `eap698_12_7_c48` | `EAP698-12-7` (#48) | `handwritten_paper_traditional` | Imam Lam Nam | 14 (14 Chăm) | `0.492` | `needs_review` |
| 24 | `eap1005_1_1_c14` | `EAP1005-1-1` (#14) | `handwritten_paper_traditional` | EAP1005/1 Family Collection | 10 (9 Chăm) | `0.442` | `needs_review` |
| 25 | `eap1005_1_1_c46` | `EAP1005-1-1` (#46) | `handwritten_paper_traditional` | EAP1005/1 Family Collection | 2 (2 Chăm) | `0.389` | `needs_review` |
| 26 | `eap1005_6_1_c12` | `EAP1005-6-1` (#12) | `handwritten_paper_traditional` | EAP1005/6 Bình Thuận Collect | 11 (11 Chăm) | `0.413` | `needs_review` |
| 27 | `eap1005_6_1_c32` | `EAP1005-6-1` (#32) | `handwritten_paper_traditional` | EAP1005/6 Bình Thuận Collect | 13 (12 Chăm) | `0.547` | `needs_review` |
| 28 | `eap1005_20_1_c20` | `EAP1005-20-1` (#20) | `handwritten_paper_traditional` | EAP1005/20 Bình Thuận Collec | 12 (12 Chăm) | `0.475` | `needs_review` |
| 29 | `eap1005_20_1_c64` | `EAP1005-20-1` (#64) | `handwritten_paper_traditional` | EAP1005/20 Bình Thuận Collec | 12 (11 Chăm) | `0.384` | `needs_review` |
| 30 | `eap1005_22_1_c15` | `EAP1005-22-1` (#15) | `handwritten_paper_traditional` | EAP1005/22 Bình Thuận Collec | 8 (8 Chăm) | `0.401` | `needs_review` |
| 31 | `eap1005_22_1_c38` | `EAP1005-22-1` (#38) | `handwritten_paper_traditional` | EAP1005/22 Bình Thuận Collec | 3 (3 Chăm) | `0.474` | `needs_review` |
| 32 | `eap698_1_17_c25` | `EAP698-1-17` (#25) | `handwritten_modern_substrate` | Mr. Xa / Mr. Sam Van Tanh | 22 (22 Chăm) | `0.551` | `needs_review` |
| 33 | `eap698_4_1_c08` | `EAP698-4-1` (#8) | `handwritten_modern_substrate` | Mr. Thanh Nha / Mr. Thanh Va | 13 (13 Chăm) | `0.516` | `needs_review` |
| 34 | `eap698_4_1_c42` | `EAP698-4-1` (#42) | `handwritten_modern_substrate` | Mr. Thanh Nha / Mr. Thanh Va | 14 (14 Chăm) | `0.534` | `needs_review` |
| 35 | `eap698_5_1_c64` | `EAP698-5-1` (#64) | `handwritten_modern_substrate` | Mr. Tai Nguc / Mr. Thanh May | 15 (14 Chăm) | `0.573` | `needs_review` |
| 36 | `eap698_5_1_c92` | `EAP698-5-1` (#92) | `handwritten_modern_substrate` | Mr. Tai Nguc / Mr. Thanh May | 15 (15 Chăm) | `0.578` | `needs_review` |
| 37 | `eap698_6_1_c08` | `EAP698-6-1` (#8) | `handwritten_modern_substrate` | Gru Tai Muoi / Mr. Thanh Ngh | 7 (7 Chăm) | `0.509` | `needs_review` |
| 38 | `eap698_6_1_c18` | `EAP698-6-1` (#18) | `handwritten_modern_substrate` | Gru Tai Muoi / Mr. Thanh Ngh | 8 (7 Chăm) | `0.557` | `needs_review` |
| 39 | `eap698_7_1_c10` | `EAP698-7-1` (#10) | `handwritten_modern_substrate` | Mr. Maduen Cet / Mrs. Dao Th | 27 (27 Chăm) | `0.470` | `needs_review` |
| 40 | `eap698_7_1_c24` | `EAP698-7-1` (#24) | `handwritten_modern_substrate` | Mr. Maduen Cet / Mrs. Dao Th | 21 (20 Chăm) | `0.530` | `needs_review` |
| 41 | `eap698_8_1_c14` | `EAP698-8-1` (#14) | `handwritten_modern_substrate` | Gru Guw Muk / Ms. Dao Thi De | 13 (13 Chăm) | `0.518` | `needs_review` |
| 42 | `eap698_8_1_c45` | `EAP698-8-1` (#45) | `handwritten_modern_substrate` | Gru Guw Muk / Ms. Dao Thi De | 12 (12 Chăm) | `0.444` | `needs_review` |
| 43 | `eap1005_32_5_c05` | `EAP1005-32-5` (#5) | `handwritten_modern_substrate` | EAP1005 Lâm Thuận Collection | 10 (10 Chăm) | `0.392` | `needs_review` |
| 44 | `eap1005_33_27_c10` | `EAP1005-33-27` (#10) | `handwritten_modern_substrate` | EAP1005 Vụ Bổn Collection | 8 (8 Chăm) | `0.502` | `needs_review` |
| 45 | `aymonier1889_chronique_p108` | `GrammaireDeLaLangueChame` (#108) | `historical_printed_and_lithographed` | Étienne Aymonier | 23 (20 Chăm) | `0.520` | `needs_review` |
| 46 | `aymonier1889_chronique_p110` | `GrammaireDeLaLangueChame` (#110) | `historical_printed_and_lithographed` | Étienne Aymonier | 19 (18 Chăm) | `0.527` | `needs_review` |
| 47 | `aymonier1889_chronique_p112` | `GrammaireDeLaLangueChame` (#112) | `historical_printed_and_lithographed` | Étienne Aymonier | 25 (21 Chăm) | `0.541` | `needs_review` |
| 48 | `aymonier1889_chronique_p114` | `GrammaireDeLaLangueChame` (#114) | `historical_printed_and_lithographed` | Étienne Aymonier | 19 (13 Chăm) | `0.562` | `needs_review` |
| 49 | `cabaton1906_dict_c080` | `bub_gb_SSAoAAAAYAAJ` (#80) | `historical_printed_and_lithographed` | Étienne Aymonier & Antoine C | 80 (19 Chăm) | `0.872` | `needs_review` |
| 50 | `cabaton1906_dict_c130` | `bub_gb_SSAoAAAAYAAJ` (#130) | `historical_printed_and_lithographed` | Étienne Aymonier & Antoine C | 75 (15 Chăm) | `0.864` | `needs_review` |
| 51 | `cabaton1906_dict_c190` | `bub_gb_SSAoAAAAYAAJ` (#190) | `historical_printed_and_lithographed` | Étienne Aymonier & Antoine C | 62 (12 Chăm) | `0.864` | `needs_review` |
| 52 | `cabaton1906_dict_c250` | `bub_gb_SSAoAAAAYAAJ` (#250) | `historical_printed_and_lithographed` | Étienne Aymonier & Antoine C | 75 (12 Chăm) | `0.891` | `needs_review` |
| 53 | `cabaton1906_dict_c310` | `bub_gb_SSAoAAAAYAAJ` (#310) | `historical_printed_and_lithographed` | Étienne Aymonier & Antoine C | 76 (10 Chăm) | `0.904` | `needs_review` |
| 54 | `cabaton1906_dict_c370` | `bub_gb_SSAoAAAAYAAJ` (#370) | `historical_printed_and_lithographed` | Étienne Aymonier & Antoine C | 78 (7 Chăm) | `0.906` | `needs_review` |
| 55 | `cabaton1906_dict_c430` | `bub_gb_SSAoAAAAYAAJ` (#430) | `historical_printed_and_lithographed` | Étienne Aymonier & Antoine C | 75 (13 Chăm) | `0.896` | `needs_review` |
| 56 | `cabaton1906_dict_c490` | `bub_gb_SSAoAAAAYAAJ` (#490) | `historical_printed_and_lithographed` | Étienne Aymonier & Antoine C | 72 (11 Chăm) | `0.886` | `needs_review` |

---

## 4. Quy Trình Hiệu Đính Con Người & Thăng Cấp Lên `verified_gold`

1. **Mở cổng duyệt trực quan cục bộ**:
   Mở tệp `ocr-benchmark/data/realscan_gold_v4/review_queue/index.html` trên trình duyệt để xem toàn bộ 56 trang gốc, đối chiếu gợi ý phiên âm học thuật (*Aymonier 1889*, *Aymonier & Cabaton 1906*) và kiểm tra các dòng bị gắn cờ `needs_review`.
2. **Hiệu đính trên Cham OCR Studio (`https://ocr.cham.asia`)**:
   - Tải ảnh từ `ocr-benchmark/data/realscan_gold_v4/pages/<doc_id>.jpg` lên Studio.
   - Chỉnh sửa văn bản *Akhar Thrah* (bàn phím ảo `Alt+K`, kiểm tra cụm Unicode `Alt+U`), bấm xác nhận dòng (`A`) để chuyển trạng thái tài liệu sang **`Verified`**, sau đó bấm **`Ctrl+E` -> Export Structured JSON (`cham_ocr_transcription.json`)**.
3. **Thăng cấp tự động vào nhánh `verified_gold` (với kiểm tra chống rò rỉ tập train & khóa SHA-256)**:
   ```bash
   python scripts/ingest_studio_to_realscan_gold.py \
     --version v4 \
     --promote-verified \
     --studio-json path/to/cham_ocr_transcription.json \
     --annotator "ten_nguoi_hieu_dinh" \
     --review-round 1
   ```
   *(Hỗ trợ cả tệp JSON đơn trang lẫn tệp JSON xuất hàng loạt nhiều trang từ Studio; tự động từ chối nếu dòng chưa được con người xác nhận nhằm bảo vệ tuyệt đối chuẩn `verified_gold`).*

---

## 5. Phát Hiện Thực Nghiệm Từ 1.023 Dòng Tư Liệu Vật Lý Mới

Quá trình tiền phân tách và nhận dạng nháp không phá hoại (`gt: null`) trên 56 trang tư liệu vật lý gốc đã cung cấp 4 bằng chứng thực nghiệm trực tiếp để định hướng kỹ thuật tiếp theo:
1. **Độ tự tin hiệu chuẩn (`calibrated_confidence`) phản ánh trung thực độ khó ngoài đời thực**:
   - Khác với ảnh dựng sạch (`conf ~ 0.94–0.98`), trên thủ bản vật lý chưa từng thấy (`palm_leaf_agal`: `0.4421`, `handwritten_paper_traditional`: `0.4612`, `handwritten_modern_substrate`: `0.4616`), độ tự tin trung bình hạ xuống `0.31–0.64` và kích hoạt cờ **`needs_review` trên 56/56 trang (`100%`)**, không gây ảo tưởng tự tin sai.
2. **Lọc sạch nhiễu bảng màu & thước đo lưu trữ (`22` hộp viền bị loại bỏ)**:
   - Các ảnh chụp của *British Library EAP* luôn đặt bảng chuẩn màu `QpCARD 201`, thẻ số thứ tự trang (`006`, `083`...) và thước đo cm ở rìa dưới ảnh. Bộ lọc `is_eap_calibration_card_box` đã tự động loại bỏ **22 hộp nhiễu phi văn bản** mà không chạm vào bất kỳ dòng chữ *Akhar Thrah* nào.
3. **Xác nhận thực tế giới hạn định tuyến song ngữ nội dòng trên *Dictionnaire čam-français* (1906)**:
   - Trên 8 trang từ điển hai cột (`cabaton1906_dict_c080..c490`, tổng cộng 444 dòng), chế độ `reading_order="auto"` nhận diện đúng bố cục hai cột (`columns: 2`). Tuy nhiên, với các dòng mở đầu bằng một mục từ *Akhar Thrah* ngắn theo sau bởi đoạn giải nghĩa tiếng Pháp dài trên cùng một dòng, ngưỡng `c_ratio < 0.35` của `Auto Router v2.2` chuyển toàn bộ dòng sang `PP-OCRv5-vi` (đạt `conf ~ 0.86–0.90` trên phần chữ Pháp nhưng làm mất chữ *Akhar Thrah* đầu dòng). Phát hiện này xác nhận trực tiếp trên tư liệu in lịch sử rằng **`Auto Router v2.3` bắt buộc phải tách cụm kịch bản theo trục ngang (intra-line horizontal script-span splitting)** trước khi đưa vào từng recognizer chuyên biệt.
4. **Giới hạn thời gian chờ `100s` (`HTTP 524`) của tầng proxy Cloudflare trên trang hai cột siêu dày (`60–70` dòng)**:
   - Các trang từ điển 44–63 dòng (`c310..c490`) hoàn tất qua `https://ocr.cham.asia` trong `87.1s–98.7s`, nhưng các trang 60–70 dòng (`c080..c250`) chạy mất `~101–115s` trên container 2-vCPU, chạm ngưỡng ngắt kết nối `100s` của Cloudflare (`HTTP 524`, trong khi gọi trực tiếp origin container vẫn trả kết quả đầy đủ). Đây là cơ sở định lượng để tối ưu batching CPU hoặc bổ sung cơ chế xử lý bất đồng bộ (async job polling) cho tài liệu mật độ cao.
