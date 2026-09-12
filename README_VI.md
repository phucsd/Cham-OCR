[English](README.md) | **Tiếng Việt**

# Dự án Nhận diện Chữ viết tiếng Chăm (Cham-OCR Monorepo)

Dự án nghiên cứu, huấn luyện và chẩn đoán nhận diện chữ viết tiếng Chăm (Akhar Thrah & West Cham/Cam Srak) dựa trên nền tảng PaddleOCR PP-OCRv4 Multilingual và DBNet.

> 📄 **Báo Cáo Kỹ Thuật Nghiên Cứu (Technical Report / Research Preprint)**: Xem toàn bộ tài liệu nghiên cứu, phương pháp luận, đánh giá thực nghiệm và phân tích cổ tự học chi tiết tại **[RESEARCH.md](RESEARCH.md)** hoặc trực tuyến tại **[https://ocr.cham.asia/research](https://ocr.cham.asia/research)**.
> 
> 🌟 **Trải nghiệm Trực tuyến (Live Demo)**: Truy cập giao diện ứng dụng web Cham OCR Studio tại: **[https://ocr.cham.asia](https://ocr.cham.asia)** (hoặc trên [Hugging Face Spaces](https://huggingface.co/spaces/phucsd/cham-ocr-studio)).

---

## 📁 Cấu trúc Thư mục Dự án

```
Cham-OCR/
├── ocr-studio/                         # 1. Giao diện Chẩn đoán & Review OCR (Diagnostic Studio)
│   ├── app.py                          # Backend HTTP Server & Multi-Crop Orchestrator
│   ├── index.html                      # Giao diện Studio Warm Light Theme (Chuẩn tiếng Anh học thuật)
│   ├── research.html                   # Giao diện Báo cáo Kỹ thuật & Nghiên cứu Khoa học
│   ├── start_studio.py                 # Script chạy nhanh giao diện
│   ├── data/                           # Dictionaries & Model inference (v24, v23, v22)
│   ├── PaddleOCR/                      # Codebase inference PaddleOCR
│   └── README_STUDIO.md                # Hướng dẫn chi tiết sử dụng Studio
│
├── ocr-training/                       # 2. Pipeline Huấn luyện & Fine-tune Mô hình (Kaggle Pipeline)
│   ├── paddleocr_cham_finetune.ipynb  # Notebook chính chạy trên Kaggle GPU T4x2
│   ├── configs/                        # Các file cấu hình YAML huấn luyện (v24, v25)
│   ├── scripts/                        # Scripts sinh dữ liệu tổng hợp & phẫu thuật trọng số
│   ├── data/                           # Dữ liệu huấn luyện (fonts, corpus Chăm đã chuẩn hóa)
│   ├── tests/                          # Bộ kịch bản kiểm thử & đánh giá CER/Accuracy
│   └── README_TRAINING.md              # Hướng dẫn chi tiết quy trình huấn luyện
│
├── Dockerfile                          # Dockerfile đóng gói và triển khai Web App Studio
├── .dockerignore                       # Cấu hình loại trừ file khi đóng gói Docker
├── .gitignore                          # Bộ lọc loại trừ file rác, file tạm và weights nặng
├── DESIGN.md                           # Định nghĩa Design System Tokens cho Studio
├── LICENSE                             # Giấy phép nguồn mở MIT cho mã nguồn phần mềm
├── DATA_PROVENANCE.md                  # Nguồn gốc ngữ liệu văn bản, bản quyền font & dữ liệu tổng hợp
├── RESEARCH.md                         # Toàn văn Báo cáo Kỹ thuật & Khảo sát Cổ tự học Chăm
├── ACADEMIC_AUDIT.md                   # Nhật ký Kiểm định Tài liệu Nội bộ & Đối chiếu Số liệu
├── kernel-metadata.json                # Kaggle Kernel Metadata (tài khoản gustavnguyen)
└── requirements.txt                    # Thư viện phụ thuộc cho toàn dự án
```

---

## 🚀 Hướng dẫn Bắt đầu Nhanh

### A. Chạy thử nghiệm nhận diện & Chẩn đoán OCR (Cham OCR Studio):
1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
2. Khởi chạy studio chẩn đoán cục bộ:
   ```bash
   python ocr-studio/app.py
   ```
   Hoặc chạy qua file khởi động nhanh:
   ```bash
   python ocr-studio/start_studio.py
   ```
3. Mở trình duyệt tại địa chỉ: `http://localhost:7860`
4. Chi tiết tài liệu: xem tại [ocr-studio/README_STUDIO.md](ocr-studio/README_STUDIO.md).

### B. Huấn luyện / Fine-tune mô hình mới trên Kaggle:
1. Đọc hướng dẫn chi tiết tại [ocr-training/README_TRAINING.md](ocr-training/README_TRAINING.md).
2. Sử dụng notebook [paddleocr_cham_finetune.ipynb](ocr-training/paddleocr_cham_finetune.ipynb) và đẩy lên Kaggle GPU thông qua Kaggle API hoặc giao diện web Kaggle.
3. Dự án sử dụng tài khoản Kaggle: `gustavnguyen` với GPU T4x2 song song:
   ```bash
   python3 -m paddle.distributed.launch --gpus '0,1' tools/train.py -c configs/rec_cham_v25.yml
   ```

---

## 📑 Tóm tắt Công nghệ Cốt lõi

1. **Phân đoạn dòng chữ Chăm (Indic Line Segmentation)**:
   - Kết hợp mô hình học sâu **PaddleOCR DBNet** (tỷ lệ unclip 1.8) và thuật toán phân tích đường cắt thung lũng (Indic Valley-Cut Heuristics).
   - Cơ chế bảo vệ nét dính `difference_update` và đệm an toàn tối thiểu `0.40 * median_line_height`.
   - Cổng *Legacy-First* bảo toàn độ chính xác cho tài liệu sạch.
2. **Kiến trúc Nhận diện Chữ viết (Recognition Architecture)**:
   - Mạng nơ-ron **PP-OCRv4 SVTR-LCNet** mở rộng tensor đầu vào lên `[3, 48, 480]`.
   - Cơ chế giải mã kép: CTC Loss kết hợp NRTR Multi-Head Cross-Attention.
   - Chuẩn hóa thứ tự gõ Logic Chăm `normalize_unicode` (Brahmic Logical Order: Phụ âm cơ sở + Dấu phụ dưới chân/quấn RA/LA + Dấu phụ YA/WA + Nguyên âm đứng trước O/AI + Các nguyên âm phụ thuộc khác + Dấu kéo dài âm AA + Phụ âm cuối/Dấu ngắt).
3. **Sinh Dữ liệu Tổng hợp & Tăng cường**:
   - 150,000 dòng dữ liệu tổng hợp qua 4 gói phân tầng.
   - Cơ chế lọc chữ lỗi ô vuông (tofu) bằng `fontTools` cmap và làm mờ động học trong RAM (*On-the-Fly Motion Blur*).
4. **Triển khai Trực tuyến**:
   - Đóng gói container Docker trên **Hugging Face Spaces** (`phucsd/cham-ocr-studio`).
   - Tên miền dịch vụ chính thức: **[https://ocr.cham.asia](https://ocr.cham.asia)**.

---

## 📊 Kết quả Thực nghiệm & Kiểm định Định lượng

Mọi số liệu công bố đều được đối chiếu trực tiếp từ các file kết quả thực nghiệm trong thư mục `ocr-studio/data/` và `ocr-benchmark/results/`:

1. **Bộ Kiểm thử Phân tầng 50 Bài (So sánh V23 vs V24 Validated Baseline)**:
   - CER trung bình toàn bộ 50 bài giảm từ **28.17% (V23)** xuống **13.88% (V24)**.
   - Tỷ lệ vượt qua (Pass Rate, khoảng cách Levenshtein $\le 1$) tăng từ **4.0% lên 50.0%**.
   - Khả năng chống chịu nhiễu hạt (Cat 6: Noise & Grain) cải thiện đột phá từ CER 50.59% xuống còn **8.82%**.
2. **Bộ Stress-Test 200 Trang Tài liệu Tổng hợp Kiểm soát (903 Dòng Chữ)**:
   - Được sinh bởi script `ocr-benchmark/scripts/generate_benchmark_200.py` từ ngữ liệu Chăm thực tế với 5 cấp độ biến dạng hình học và chất lượng giấy.
   - Tỷ lệ phát hiện dòng tổng thể đạt **92.80%**, CER trung bình **23.24%**, WER **51.63%**, tốc độ ~1.33s/trang trên CPU.
   - Xác định chính xác **điểm gãy sụp đổ (Breaking Point) ở Cấp độ 5**: khi biên độ uốn sóng lớn hơn khoảng cách giữa hai dòng ($\text{Amplitude} > \text{Gap}$), tỷ lệ bắt dòng giảm xuống 61.45% do các nét chữ cắt chéo và dính chùm vào nhau.
3. **Phân loại Mô hình V24 vs V25**:
   - **Version 24 (Validated Baseline)**: Mô hình chuẩn chính thức đang phục vụ trên production (CER 16.81%, Pass Rate 44.0% trong bài đối sánh cục bộ).
   - **Version 25 (Experimental Checkpoint)**: Bản thử nghiệm mở rộng từ điển dấu câu và số khổ thơ (đang trong quá trình huấn luyện và tinh chỉnh, CER 51.73% ở checkpoint ban đầu).

---

## 📑 Trích dẫn Khoa học

Nếu bạn sử dụng tài liệu, bộ sinh dữ liệu hoặc công cụ Cham OCR Studio trong nghiên cứu, vui lòng trích dẫn:

```bibtex
@software{cham_ocr_studio,
  author = {Nguyen, Phuc},
  title = {Cham OCR Studio: Deep Learning Pipeline and Paleographic Transcription Workbench for Historical Cham Manuscripts},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/phucsd/Cham-OCR},
  note = {Live web service: https://ocr.cham.asia}
}
```

---

## 📄 Giấy phép & Tuyên bố Bản quyền

- **Mã nguồn phần mềm**: Phát hành theo giấy phép nguồn mở [MIT License](LICENSE). Bản quyền © 2026 Phuc H. Nguyen.
- **Phông chữ & Ngữ liệu Văn bản**: Chi tiết về bản quyền phông chữ Chăm (*Noto Sans Cham* đi kèm repo, các phông chữ tham khảo ngoài), nguồn ngữ liệu văn bản cổ và giấy phép tập dữ liệu tổng hợp được ghi chép đầy đủ tại **[DATA_PROVENANCE.md](DATA_PROVENANCE.md)**.
- **Kiểm định & Đối chiếu Số liệu**: Xem **[ACADEMIC_AUDIT.md](ACADEMIC_AUDIT.md)** để tra cứu nhật ký kiểm định số liệu thực nghiệm và cam kết minh bạch học thuật.
