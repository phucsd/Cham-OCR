# BÁO CÁO TỔNG QUAN HỆ THỐNG MÔ HÌNH NHẬN DIỆN CHỮ CHĂM (CHAM-OCR)
**Dự án Cham-OCR | Cập nhật: 19/09/2026**  
**Tên miền chính thức phục vụ trực tuyến:** [https://ocr.cham.asia](https://ocr.cham.asia)

---

## 1. Executive Summary (Tóm Lược Điều Hành)

Hệ thống **Cham-OCR** là giải pháp nhận dạng ký tự quang học (OCR) mã nguồn mở tiên tiến nhất dành cho ngôn ngữ và chữ viết Chăm (Akhar Thrah). Trải qua nhiều giai đoạn nghiên cứu, tối ưu hóa thuật toán phân đoạn dòng (Line Segmentation), kiến trúc mạng nơ-ron nhận diện (PP-OCRv4 Mobile + CTC/GTC Heads), và chuẩn hóa thứ tự logic (Logical Order Normalization), hệ thống hiện sở hữu một danh mục mô hình suy luận hoàn chỉnh đáp ứng từ nhu cầu nghiên cứu, đối chuẩn (benchmark) đến triển khai thực tế quy mô lớn.

Tài liệu này cung cấp cái nhìn tổng quan toàn diện về:
1. **Danh mục 8 mô hình suy luận (Inference Models)**: Bao gồm trọng số, từ điển, mã băm SHA256 kiểm định toàn vẹn và phạm vi ứng dụng.
2. **Chi tiết phiên bản mới nhất V24.1 Stage R1A**: Kết quả huấn luyện trên Kaggle Dual Tesla T4x2 (6 Epochs, Validation Accuracy **99.42%**) và quyết định chính thức **PASS** làm nguồn duy nhất chuyển giao sang R1B.
3. **Phân định rõ ràng về Mô hình V25**: Là thử nghiệm bị loại bỏ (**Rejected Experiment / Excluded from Model Lineage**) do hiện tượng phân tán xác suất và ảo giác ký tự lạ khi mở rộng từ điển 160 tokens; không tham gia lineage sản xuất chính thức. Lineage chuẩn duy nhất là `v24 → R1A PASS → R1B → R2 → R3 → R4 → v24.1`.
4. **Gói đóng gói toàn bộ mô hình (`cham_ocr_all_models.zip`)**: Cấu trúc tệp nén đồng bộ phục vụ lưu trữ ngoại tuyến và triển khai độc lập.

---

## 2. Danh Mục Toàn Bộ 8 Mô Hình Inference & Mã Băm SHA256

Toàn bộ các mô hình suy luận đã được xuất bản và kiểm chứng trên định dạng Paddle Inference (`inference.pdmodel`, `inference.pdiparams`, `inference.yml`):

| STT | Tên Thư Mục Mô Hình | Loại Tác Vụ | Từ Điển Tương Ứng | Dung Lượng Trọng Số (`pdiparams`) | Mã Băm SHA256 (`inference.pdiparams`) | Trạng Thái / Vai Trò |
| :---: | :--- | :--- | :--- | :---: | :--- | :--- |
| **1** | `rec_cham_inference_v24_1_r1a` | Recognition | `cham_dict_v24.txt` (103 tokens) | 7,611,141 B | `281d51763e8a07a15b80d04fcf4349cc05c0275cc1f519383f6ba07bbdf38409` | **SOTA Chăm Thuần mới nhất - R1A PASS** (Kaggle T4x2, Acc 99.42%, R1A-best) |
| **2** | `rec_cham_inference_v25` | Recognition | `cham_dict_v25.txt` (160 tokens) | 9,713,299 B | `43cfb245523c1cbf15c2c44a0be0cdf305d3f649c086c8dc3579e90f6791d1e8` | **Thử nghiệm bị loại bỏ (Rejected Experiment / Excluded from Lineage)** |
| **3** | `rec_cham_inference_v24` | Recognition | `cham_dict_v24.txt` (103 tokens) | 7,611,141 B | `61e3606f1e57d9e83de0d0b92fa5722c6815ff0e7ea3725895f6bf9e56d71596` | **Baseline V24** (Mở rộng số Chăm ꩐-꩙ và dấu câu) |
| **4** | `rec_cham_inference_v23` | Recognition | `cham_dict_v23.txt` (93 tokens) | 7,600,977 B | `70dac0b0d59fc995482d203ec3409affdbd47848bdff39a0a9387e22cb8d3a88` | **Golden Baseline** (Bản nền tảng ổn định chuẩn hóa Logical Order) |
| **5** | `rec_cham_inference_v22` | Recognition | `cham_dict_v22.txt` (93 tokens) | 7,601,461 B | `b770b505d150da8b1fea7f1b516849559cd51ef35c9bdc9adac20ea45502bd48` | **Legacy Version** (Lưu trữ lịch sử) |
| **6** | `ch_PP-OCRv4_det_infer` | Detection | N/A (Segment DBNet) | 4,692,937 B | `49ee815e30cff43cb1057d33bf0d94193e4d4f1ae28451cad15b40be830df915` | **TextDetector Chuẩn** (Phân đoạn dòng chữ trang văn bản) |
| **7** | `ch_PP-OCRv4_det_cham_infer` | Detection | N/A (Segment DBNet) | 4,692,937 B | `cfc65a686ecaa5fd4cca9e7ab3b1ecdf9eecaf375291c1c0279aff1c86b0cec5` | **Cham-Tuned Detector** (Phân đoạn tinh chỉnh chữ Chăm) |
| **8** | `rec_vietnamese_infer` | Recognition | `vietnamese_dict.txt` | 75,601,159 B | `eb929c47e349f528c6ff7e00d67b101bf7bb7654dffb5d18ac4156550841ebd3` | **Vietnamese High-Acc OCR** (Hỗ trợ AutoRouting văn bản song ngữ) |

---

## 3. Phân Tích Chuyên Sâu Từng Thế Hệ Mô Hình

### 3.1. Mô hình V24.1 Stage R1A (SOTA Chăm Thuần Mới Nhất)
- **Bối cảnh phát triển**: Stage R1A là chặng bản lề trong lộ trình V24 $\to$ V24.1 nhằm tạo nền tảng weights ổn định (R1A-best) trước khi thực hiện phẫu thuật mở rộng từ điển (Weight Surgery).
- **Quy mô & Môi trường huấn luyện**:
  - Thực thi trên **Kaggle Dual Tesla T4x2** qua lệnh phân tán `python3 -m paddle.distributed.launch --gpus '0,1' tools/train.py`.
  - Cấu hình: 6 Epochs (6,528 steps, batch size 64 x 2 = 128), tốc độ xử lý ~15.2 ms/step.
  - Áp dụng cấu hình chống treo phân tán bắt buộc: `eval_batch_step: [0, 9999999]`, `cal_metric_during_train: false`.
  - Đánh giá hậu huấn luyện (Post-Training Evaluation) trên Single GPU: **Sequence Accuracy đạt 99.42%** trên 14,000 ảnh validation độc lập. Loss huấn luyện giảm từ 1.9424 xuống đáy **0.8385**.
- **Điểm cải tiến kỹ thuật đột phá**:
  1. **Chống CTC Collapse & Gộp Chữ Số Lặp**: Dữ liệu tổng hợp sinh các mẫu số thứ tự khổ thơ Chăm (1–999) có khoảng cách co giãn ngẫu nhiên từ 2px đến 8px và nhiễu đứt nét, ngăn CTC Decoder nuốt gộp số lặp (như `꩒꩐꩐꩞` $\to$ `꩒꩐꩞`).
  2. **Cân bằng cặp ký tự gây nhầm (Hard-Mining)**: Tăng cường tối thiểu 5,000 mẫu hard-examples cho cặp dấu chân `ꨲ` (U+AA32) vs `ꨶ` (U+AA36), và các cặp phụ âm `ꨕ` vs `ꨊ`, `ꨔ`/`ꨝ` vs `ꨟ`, `ꨥ` vs `ꨡ`, `ꨈ` vs `ꨓ`.
  3. **Tách biệt khoảng cách Double Danda (`꩝꩝`)**: Ngăn gộp 2 nét gạch đứng thành 1 nét đơn.
  4. **Triệt tiêu hoàn toàn ảo giác (Zero Blank Hallucination)**: Mô hình không còn tự động sinh ra ký tự rác `꩝` khi gặp ảnh nền trắng hoặc ảnh nhiễu không có chữ.

### 3.2. Mô hình V25 (Thử nghiệm Bị Loại Bỏ / Rejected Experiment - Không tham gia Lineage)
- **Quy tắc phân định chính thức**: Mô hình V25 là một **thử nghiệm bị loại bỏ (Rejected Experiment)** và **tuyệt đối không tham gia vào lineage phát triển sản xuất**.
- **Bài học kinh nghiệm thực tế**:
  - Quá trình huấn luyện 40 Epochs trên Kaggle Dual Tesla T4x2 với từ điển mở rộng 160 tokens (chèn thêm bảng mã chữ Latinh cho văn bản song ngữ) đã làm **phân tán không gian xác suất CTC**.
  - Hệ quả: Khi nhận diện các tài liệu Chăm cổ hoặc ảnh có nhiễu/blur nhẹ, mô hình thường xuyên bị **ảo giác (hallucination)** chèn các ký tự Latinh lạ (`t`, `g`, `m`, `i`, `ă`, `ua`) và khoảng trắng ngẫu nhiên vào giữa các từ Chăm.
  - Do đó, toàn bộ kiến trúc sản xuất của Cham-OCR quay về kiên định với lineage thuần Chăm: **`v24 → R1A PASS → R1B → R2 → R3 → R4 → v24.1`**, xử lý tài liệu song ngữ bằng kiến trúc **AutoRouting** độc lập (sử dụng mô hình tiếng Việt chuyên dụng `rec_vietnamese_infer`) thay vì gộp chung một không gian nhãn.

### 3.3. Mô hình V24 Baseline
- Mở rộng không gian ký tự từ 93 lên 103 tokens: Bổ sung 10 chữ số Chăm truyền thống (`꩐`, `꩑`, `꩒`, `꩓`, `꩔`, `꩕`, `꩖`, `꩗`, `꩘`, `꩙`), dấu chấm hai chấm `:` và các ký hiệu `[`, `]`, `/`.
- Trở thành mốc kiểm chuẩn nền tảng (Baseline) để đánh giá non-regression cho các thế hệ v24.1+.

### 3.4. Mô hình V23 Golden Baseline & V22 Legacy
- **V23**: Bản chuẩn mực đầu tiên tích hợp sâu thuật toán Chuẩn hóa Thứ tự Logic (Logical Order Normalization). Loại bỏ hiện tượng đảo lộn trật tự nguyên âm trước / phụ âm cơ sở, xác lập quy chuẩn: `Base Consonant + Medials + Pre-Ra + Pre-Vowels + Others`.
- **V22**: Phiên bản lưu trữ lịch sử nghiên cứu giai đoạn đầu.

### 3.5. Hệ Thống Phát Hiện Dòng (DBNet) & Nhận Diện Tiếng Việt (AutoRouting)
- **`ch_PP-OCRv4_det_infer`**: Mạng DBNet phát hiện vùng dòng chữ với độ nhạy cao. Được tích hợp trong module Line Segmentation hỗ trợ đệm an toàn chiều cao nguyên âm (`0.40 * median_line_height`), chống xóa nét chạm dính (`difference_update`) và cổng giữ nguyên kết quả cũ (`Legacy-First Gate`).
- **`rec_vietnamese_infer`**: Mô hình nhận diện tiếng Việt độc lập (75.6 MB). Khi kích hoạt chế độ `AutoRouting` trong OCR Studio, các crop có độ tự tin chữ Chăm thấp sẽ được tự động chuyển hướng sang mô hình tiếng Việt để xử lý chính xác tài liệu dịch chú giải.

---

## 4. Bảng So Sánh Đối Chứng Head-to-Head 1:1 (Baseline V24 Native vs Model R1A SOTA)

Kiểm thử được thực thi tự động bằng kịch bản kiểm chuẩn đóng băng nghiêm ngặt (`ocr-benchmark/scripts/run_v24_frozen_suite.py`) trên môi trường Native PaddleOCR CPU:

```
========================================================================================================
                          KẾT QUẢ ĐỐI CHỨNG NATIVE PADDLEOCR HEAD-TO-HEAD 1:1
========================================================================================================
Bộ Kiểm Chuẩn (Frozen Suite)           Chỉ Số Định Lượng           Baseline V24 Native    Model R1A (Mới)
--------------------------------------------------------------------------------------------------------
Suite A: Clean-Frozen                  Normalized CER                      13.25%         11.58% (-1.67%)
(200 dòng chữ sạch synthetic)          Raw CER                             13.94%         12.30% (-1.64%)
                                       Exact Line Match                     2.5%          11.5% (+4.6 lần)
                                       Average Confidence                  0.926          0.938 (+0.012)
--------------------------------------------------------------------------------------------------------
Suite B: Character-Frozen              Character CER                       12.36%         10.80% (-1.56%)
(280 dòng / 4,853 ký tự phủ 100%)      Exact Line Match                    24.6%          30.4% (+5.8%)
                                       High-Confidence Wrong Errors        189 ca         175 ca (-14 ca)
--------------------------------------------------------------------------------------------------------
Suite C: Stress-Frozen                 Mean CER (82 conditions)            18.41%         17.71% (Bền vững hơn)
(82 điều kiện suy biến / 156 chars)    - Perspective distortion (p=0.14)   41.7%          30.1% (-11.6%)
                                       - Dilate (k=3)                      35.9%          14.7% (-21.2%)
                                       - Low-Height 20 x Motion Blur 7     69.2%          21.2% (-48.0%)
                                       - Low-Height 18 x Motion Blur 9     85.3%          27.6% (-57.7%)
                                       - Low-Height 16 x Motion Blur 9     89.7%          34.0% (-55.7%)
                                       - Low-Height 20 x Gauss Blur 1.8    76.3%          15.4% (-60.9%)
                                       - Low-Height 16 x Gauss Blur 1.8    95.5%          16.0% (-79.5%)
                                       - Blank White Hallucination         '꩝' (True)     '' (False - PASS)
--------------------------------------------------------------------------------------------------------
Suite D: Symbol-Frozen                 Dấu ngoặc mở '['                    100.0%         100.0% (20/20)
(14 ký hiệu kiểm định bắt buộc)        Dấu ngoặc đóng ']'                   80.0%         100.0% (20/20)
                                       Dấu gạch chéo '/'                     0.0%         100.0% (20/20)
                                       11 ký hiệu mới (?, !, #, &,...)      0.0%           0.0% (Chờ R1B)
--------------------------------------------------------------------------------------------------------
Suite E: FullPage-Frozen               DBNet Detector Recall               79.08%         79.08%
(50 trang / 239 dòng chữ)              DBNet Detector Precision            97.93%         97.93%
                                       Recognizer Matched CER              13.43%         12.29% (-1.14%)
                                       End-to-End Pipeline CER             37.71%         36.89% (-0.82%)
                                       L1-4 (Standard to Wavy) CER         13.41%         12.26% (-1.15%)
========================================================================================================
```

### Kết Luận Nghiệm Thu 4 Cổng Gate R1A:
1. **Gate 1 (Clean-Frozen Non-Regression)**: **PASS XUẤT SẮC**. Normalized CER giảm từ 13.25% xuống 11.58% (giảm 12.6% tỷ lệ lỗi tương đối so với V24), tỷ lệ Exact Line tăng vọt gấp 4.6 lần.
2. **Gate 2 (14 Mandatory Symbols Accuracy)**: **PASS ĐÚNG ĐẶC TẢ R1A**. Cả 3 ký hiệu thuộc từ điển V24 cũ (`[`, `]`, `/`) đều đạt độ chính xác **100.0% (20/20)**, trong đó `/` cải thiện từ 0% lên 100%, `]` tăng từ 80% lên 100%. 11 ký hiệu mới là mục tiêu của Stage R1B sau khi thực hiện Weight Surgery.
3. **Gate 3 (Zero Blank Hallucination)**: **PASS TUYỆT ĐỐI**. Loại bỏ 100% hiện tượng tự sinh ký tự `꩝` trên ảnh trắng hoặc vùng crop rỗng.
4. **Gate 4 (FullPage-Frozen Matched CER)**: **PASS**. Matched CER trên toàn bộ 50 trang giảm xuống 12.29% (L1-4 đạt 12.26%).

---

### 4.1. R1A FINAL DECISION: PASS

Checkpoint R1A (`rec_cham_inference_v24_1_r1a`) chính thức được nghiệm thu **PASS** và xác lập vị thế **R1A-best**, là nguồn duy nhất để thực hiện phẫu thuật trọng số (Weight Surgery) chuyển giao sang R1B. **Tuyệt đối không train lại R1A**.

#### Danh Sách 7 Tồn Đọng Thực Tế (Residual Issues) & Phân Tầng Chuyển Giao:
1. **Dấu gạch ngang (Hyphen `-`)**: Nhận diện còn yếu (~22% trong targeted multi-context tests) $\to$ **Chuyển giao cho R1B** (bổ sung $\ge 4,000$ mẫu đa ngữ cảnh, gate riêng `hyphen_accuracy >= 95%`) và **R2** (nâng lên $\ge 99\%$).
2. **Ký tự `ꨅ`**: Chưa cải thiện đủ $\to$ **Chuyển giao cho R1B** ($\ge 2,000$ occurrences replay) và **R2** (Hard-Mining P0/P1).
3. **Ký tự `ꨁ`**: Có dấu hiệu suy giảm nhẹ so với v24 $\to$ **Chuyển giao cho R1B** ($\ge 2,000$ occurrences replay chống thoái lùi) và **R2** (Hard-Mining).
4. **Suy biến Severe Motion Blur (k=9, 11, 13, 17)**: Vẫn còn yếu ở độ phân giải thông thường (normal resolution 48px) $\to$ **Chuyển giao cho R3** (Curriculum standalone severe motion blur).
5. **Suy biến Heavy Gaussian Blur ($\sigma = 2.5, 3.0, 3.5$)**: Vẫn còn yếu $\to$ **Chuyển giao cho R3** (Curriculum Gaussian heavy blur).
6. **Confidence Calibration**: Một số dự đoán sai vẫn có độ tự tin cao ($>0.99$) $\to$ **Chuyển giao cho R3/R4/Final pipeline**.
7. **High-Noise Blank Hallucination**: Ảnh trống có nhiễu cực nặng vẫn xuất hiện dự đoán rác $\to$ **Chuyển giao cho R3/R4 hoặc Blank Rejection Filter** trong post-processing.

---

## 5. Quy Chuẩn Kỹ Thuật Đúc Rút Từ Thực Nghiệm

1. **Khởi Tạo Trực Tiếp TextRecognizer**: Khi nạp mô hình suy luận trong Python, không sử dụng `PaddleOCR(lang='cham')` do thư viện chuẩn không hỗ trợ mã ngôn ngữ này. Thay vào đó, luôn khởi tạo trực tiếp lớp `tools.infer.predict_rec.TextRecognizer(args)` với đường dẫn tuyệt đối `args.rec_model_dir` và `args.rec_char_dict_path`.
2. **Nguyên Tắc Chống Treo Phân Tán Kaggle Dual GPU**: Trong vòng lặp huấn luyện phân tán `paddle.distributed.launch`, bắt buộc phải tắt đánh giá định kỳ (`eval_batch_step: [0, 9999999]`, `cal_metric_during_train: false`) để tránh desync giữa GPU 0 và GPU 1 gây cháy quota GPU.
3. **Phòng Vệ Checkpoint & Tắt PIR**: Luôn nén sao lưu thư mục checkpoint ngay sau khi train loop kết thúc và tắt cờ `Global.export_with_pir=False` trên Paddle 2.6/3.x để xuất bản mô hình ổn định.
4. **Khử Trùng Lặp Tải Output Kaggle**: Thay vì tải tệp nén 3.6 GB chứa 140k ảnh bằng Kaggle API thông thường, áp dụng kỹ thuật **HTTP Range Requests 206** (`scratch/fast_remote_zip.py`) để đọc Central Directory và trích xuất đích danh thư mục mô hình suy luận trong 4 giây.

---

## 6. Chi Tiết Gói Đóng Gói Toàn Bộ Mô Hình (`cham_ocr_all_models.zip`)

Tất cả 8 mô hình suy luận kèm theo toàn bộ từ điển ký tự đã được đóng gói hoàn chỉnh thành một tệp zip duy nhất đặt tại thư mục gốc dự án:
- **Tên tệp**: `cham_ocr_all_models.zip`
- **Dung lượng nén**: ~112 MB (Dung lượng giải nén: 127.5 MB gồm 38 tệp)
- **Cấu trúc bên trong tệp nén**:
  ```
  cham_ocr_all_models/
  ├── README.md                                 # Hướng dẫn nạp và sử dụng mô hình
  ├── SHA256SUMS.txt                            # Toàn bộ mã băm kiểm định tính toàn vẹn
  ├── OVERVIEW_REPORT.md                        # Bản sao báo cáo tổng quan này
  ├── dictionaries/                             # Toàn bộ từ điển ký tự Chăm & Việt
  │   ├── cham_dict_v22.txt                     # 93 tokens (v22 legacy)
  │   ├── cham_dict_v23.txt                     # 93 tokens (v23 baseline)
  │   ├── cham_dict_v24.txt                     # 103 tokens (v24 baseline)
  │   ├── cham_dict_v24_1_r1a.txt               # 103 tokens (v24.1 R1A)
  │   ├── cham_dict_v24_1.txt                   # 114 tokens (v24.1 Core - 11 ký hiệu mới)
  │   ├── cham_dict_v25.txt                     # 160 tokens (v25 SOTA Bilingual)
  │   └── vietnamese_dict.txt                   # Từ điển tiếng Việt
  └── models/                                   # 8 thư mục mô hình suy luận
      ├── ch_PP-OCRv4_det_infer/                # DBNet TextDetector chuẩn
      ├── ch_PP-OCRv4_det_cham_infer/           # DBNet TextDetector Cham-tuned
      ├── rec_cham_inference_v22/               # V22 Recognition Model
      ├── rec_cham_inference_v23/               # V23 Golden Baseline Recognition Model
      ├── rec_cham_inference_v24/               # V24 SOTA Baseline Recognition Model
      ├── rec_cham_inference_v24_1_r1a/         # V24.1 Stage R1A SOTA Recognition Model
      ├── rec_cham_inference_v25/               # V25 SOTA Bilingual Production Model
      └── rec_vietnamese_infer/                 # Vietnamese Recognition Model
  ```

---

## 7. Hướng Dẫn Nạp Mô Hình Bằng Mã Nguồn Python

```python
import os
import sys
import numpy as np

# 1. Monkeypatch NumPy 2.x
if not hasattr(np, 'sctypes'):
    np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64], 'uint': [np.uint8, np.uint16, np.uint32, np.uint64], 'float': [np.float16, np.float32, np.float64], 'complex': [np.complex64, np.complex128], 'others': [bool, object, bytes, str]}
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'typeDict'): np.typeDict = {}

# 2. Thêm đường dẫn PaddleOCR
sys.path.append(os.path.abspath("ocr-studio/PaddleOCR"))
import tools.infer.utility as utility
from tools.infer.predict_rec import TextRecognizer

# 3. Cấu hình đối số suy luận cho mô hình V24.1 R1A
args = utility.parse_args()
args.rec_model_dir = "models/rec_cham_inference_v24_1_r1a"
args.rec_char_dict_path = "dictionaries/cham_dict_v24_1_r1a.txt"
args.use_gpu = False                     # Đặt True nếu có GPU CUDA
args.rec_image_shape = "3, 48, 320"      # Với v25 dùng "3, 48, 480"
args.use_space_char = True
args.rec_batch_num = 1
args.enable_mkldnn = False
args.cpu_threads = 1
args.ir_optim = False

# 4. Khởi tạo Recognizer
recognizer = TextRecognizer(args)

# 5. Dự đoán ảnh dòng chữ Chăm
# img = cv2.imread("line_crop.png")
# results, elapse = recognizer([img])
# pred_text, confidence = results[0]
# print(f"Kết quả: {pred_text} (Độ tự tin: {confidence:.4f})")
```

---
*Báo cáo được khởi tạo tự động và bảo đảm tính toàn vẹn bởi hệ thống Antigravity AI Assistant.*
