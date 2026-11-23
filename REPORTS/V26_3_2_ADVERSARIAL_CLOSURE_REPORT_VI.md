# Báo cáo Đóng các Ca Đối kháng Mới & Tối ưu Chịu tải `Pipeline v26.3.2`

- **Ngày thực hiện**: `2026-11-23`
- **Môi trường Production**: `https://ocr.cham.asia` (Hugging Face Space `phucsd/cham-ocr-studio`, commit `a67d1cc08eaf0e40b49ebd70e53af9848856056f`, cache-busting `?v=26.3.2`)
- **Tài liệu đối chiếu**: `Cham_OCR_QA_Independent_v26_3_1_2026-11-23.zip` (`BAO_CAO_QA_OCR_CHAM_v26_3_1_2026-11-23.md`, `new_adversarial_production.jsonl`, `p1_regression_12cases.jsonl`, `digit_and_vietnamese_v2631.jsonl`)
- **Bằng chứng định lượng**: `REPORTS/live_v26_3_2_qa_reverification.json`, `REPORTS/live_hf_r4_verification.json`, `ocr-studio/tests/test_quality_gates.py` (`33/33` unit tests PASS trên cả 2 môi trường Python).

---

## 1. Tổng hợp Kết quả Trước và Sau `Pipeline v26.3.2` trên `https://ocr.cham.asia`

| Ca kiểm thử đối kháng độc lập (`2026-11-23_v26_3_1`) | Trước `v26.3.2` (`v26.3.1`) | Sau `v26.3.2` (Live `https://ocr.cham.asia`) | Trạng thái |
| :--- | :---: | :---: | :---: |
| **Micro-CER toàn bộ 8 ảnh đối kháng mới (`independent_new_adversarial`, `auto` & `v24_1_r4`)** | `22.58%` (`5/8` PASS) | **`0.00%`** (`8/8` đạt `0.00%` CER tuyệt đối) | **ĐÃ ĐÓNG** |
| **`two_col_short_two_rows.png` (`1` tiêu đề ngắn + `2` dòng/cột = `5` dòng)** | CER `36.36%` (`columns: 1`, `rows`) | CER **`0.00%`** (`5/5` dòng, `columns: 2`) | **ĐÃ ĐÓNG** |
| **`inverted_sparsely_bottomleft.png` (`1600×980`, `0.032%` mực trắng ở góc dưới-trái)** | `0` dòng (`no_text`, CER `100.00%`) | `1/1` dòng, CER **`0.00%`** (`ꨧꨪ ꨀꨆꨢꨯꨮꩅ ?`) | **ĐÃ ĐÓNG** |
| **`inverted_sparsely_topright.png` (`1600×980`, `0.032%` mực trắng ở góc trên-phải)** | `0` dòng (`no_text`, CER `100.00%`) | `1/1` dòng, CER **`0.00%`** (`ꨧꨪ ꨀꨆꨢꨯꨮꩅ ?`) | **ĐÃ ĐÓNG** |
| **`shaped_inverted.png` trên `method="valley"` và `method="adaptive"`** | CER `18.18%` (mất dấu `?` cuối dòng) | CER **`0.00%`** (`['ꨧꨪ ꨀꨆꨢꨯꨮꩅ ?']`, đồng nhất 3 chế độ) | **ĐÃ ĐÓNG** |
| **Chống dương tính giả đảo cực (`RealScan-Gold v4.1` 56 trang + `dark_table_bright_page` + hạt bụi trắng)** | `0 / 56` đảo nhầm | **`0 / 56` đảo nhầm**, `0 / 2,585` diagnostic thường, hạt bụi đơn `False` | **AN TOÀN** |
| **Tải hỗn hợp đồng thời 3 luồng (`24` request trộn 2 cột + song ngữ + legacy models)** | `1/32` gặp `HTTP 502` chờ lock dài | **`24/24` đạt `HTTP 200` (`0` lỗi `5xx`)** + auto-retry `502/503/504` trên UI | **ĐÃ ĐÓNG** |
| **Chống hồi quy tập `SYNTHETIC_SHAPED_QA_IMAGES` (12 ảnh) & `harum_palei_01` (25 dòng)** | `0.63%` Micro-CER / `2.60%` CER | **`0.63%` Micro-CER** / **`2.60%` CER (`25/25` dòng)** | **GIỮ VỮNG** |

---

## 2. Chi tiết 4 Giải pháp Kỹ thuật & Tự Phản biện Nội bộ

1. **Nhận diện lưới 2 cột ngắn `2×2` có cổng chống tách cột giả (`_split_columns` & `order_lines_reading`)**:
   - Giữ nguyên hoàn toàn logic cũ cho các cột có $\ge 3$ dòng; chỉ mở nhánh cột 2 dòng (`n >= 4`) khi thỏa mãn **đồng thời 4 ràng buộc hình học chặt**: (a) lề trái mỗi cột thẳng hàng (`spread <= max(18px, 1.2 * median_h)`), (b) rãnh phân cột rộng (`gutter >= max(3.5 * median_h, 0.12 * W)` hoặc có vạch kẻ dọc `col_divider_x`), (c) bề rộng mỗi cột `<= 0.45 * W`, và (d) từng dòng trong cột 2 dòng phải có dòng đối ứng trùng dải Y ở cột bên cạnh.
   - Đã kiểm thử đối kháng với các mảnh câu thơ tách nhịp và dòng đứt khoảng trắng trên trang 1 cột $\to$ giữ nguyên `columns: 1` (`rows`).
2. **Phát hiện âm bản cực thưa bất biến theo diện tích khung ảnh (`check_inverted_polarity` & `is_blank_page`)**:
   - Thay điều kiện phân vị phụ thuộc diện tích khung `np.percentile(tophat, 99.92) > 55.0` bằng kiểm tra đếm pixel nét tương phản cao (`(tophat > 60).sum() >= 24` và `(tophat > 110).sum() >= 10`) kết hợp yêu cầu $\ge 3$ thành phần liên thông (`area >= 4`) trên nền tối thực sự (`op_p985 < 65.0`).
   - *Phát hiện từ bước tự phản biện nội bộ*: Khi mở rộng vùng biên `roi_edge` ra `[1.5%..98.5%]`, nếu áp dụng nhánh đặc (`op_p95 < 85`) trên `roi_edge` sẽ làm `1/56` trang lá buông hẹp trên nền nhung đen (`eap698_1_17_c25.jpg`, `op_p95 = 75.0`, `op_p985 = 163.0`) bị đảo nhầm. Chúng tôi đã tách riêng `allow_dense=False, allow_sparse=True` trên `roi_edge`, đưa tỷ lệ đảo nhầm về tuyệt đối **`0 / 56`** trên `RealScan-Gold v4.1` mà vẫn bắt trúng chữ âm bản ở góc sát mép trên khung `3000×2000`.
3. **Cắt gọn khoảng trắng ngang cho dải dòng `valley` / `adaptive` (`_trim_horizontal_whitespace_for_rec`)**:
   - Cắt bỏ biên trắng hai đầu khi bề rộng dải cắt `> 6.5 * chiều cao` và vùng mực thực tế chiếm `< 72%` chiều ngang trang (giữ lề an toàn `max(14px, 0.55 * H)`), ngăn bộ resize `W <= 480` bóp ngang chữ `2.1×` và bảo toàn trọn vẹn dấu `?` cuối dòng (`18.18% -> 0.00%` CER).
4. **Thu hẹp phạm vi khóa `_INFERENCE_LOCK` & Tự động thử lại `502/503/504` trên Studio (`app.py` & `studio.js`)**:
   - Đưa `_INFERENCE_LOCK` xuống đúng ranh giới lời gọi C++ (`get_ocr_model`, `get_det_model`, `_detect`, `invoke_recognizer_with_provenance`) thay vì khóa toàn bộ `run_ocr_pipeline()`, giúp giải mã ảnh, xoay nghiêng OpenCV và mã hóa PNG base64 chạy song song đa luồng; đồng thời bổ sung `fetchWithTransientRetry()` (backoff `900ms`) cho `/ocr` và `/ocr_crop` trong `studio.js`.

---

## 3. Ghi nhận Định hướng Dữ liệu cho `V27` & `Gold-20` (Ngoài phạm vi heuristic pipeline)

- **Cặp nhầm lẫn chữ số Chăm đứng rời có khoảng trắng**: Trong tập `digit_and_vietnamese_v2631.jsonl`, `9/12` mẫu số viết liền đạt `0.00%` CER, còn `3/12` lỗi đều tập trung ở chuỗi chữ số đơn lẻ cách khoảng trắng (`"꩐ ꩑ ꩒ ꩓ ꩔ ꩕ ꩖ ꩗ ꩘ ꩙"`), nơi `꩑` (số 1 rời) bị nhầm với `ꩍ` (dấu phụ âm cuối `-h`) và `꩖` (số 6 rời) nhầm với `ꨣ` (`ra`). Đã lưu vào sổ tay thiết kế dữ liệu huấn luyện `V27` để bổ sung hard-mining trực tiếp.
- **Hiệu đính Ground Truth con người cho `RealScan-Gold v4.1` (`Gold-20` trước, mở rộng lên `56` trang)**: Giữ nguyên nguyên tắc tuyệt đối không điền nhãn tự động khi chưa qua kiểm duyệt độc lập của người hiệu đính.
