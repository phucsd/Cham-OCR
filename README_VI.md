[English](README.md) | **Tiếng Việt**

# Dự án Nhận diện Chữ viết tiếng Chăm (Cham-OCR Monorepo)

Dự án nghiên cứu, huấn luyện và chẩn đoán nhận diện chữ viết tiếng Chăm (Akhar Thrah & West Cham/Cam Srak) dựa trên nền tảng PaddleOCR PP-OCRv4 Multilingual và DBNet.

> 🌟 **Trải nghiệm Trực tuyến (Live Demo)**: Truy cập giao diện ứng dụng web Cham OCR Studio tại: **[https://ocr.cham.asia](https://ocr.cham.asia)** (hoặc trên [Hugging Face Spaces](https://huggingface.co/spaces/phucsd/cham-ocr-studio)).

---

## 📁 Cấu trúc Thư mục Dự án

```
Cham-OCR/
├── ocr-studio/                         # 1. Giao diện Chẩn đoán & Review OCR (Diagnostic Studio)
│   ├── app.py                          # Backend HTTP Server & Multi-Crop Orchestrator
│   ├── index.html                      # Giao diện Studio Warm Light Theme (Chuẩn tiếng Anh học thuật)
│   ├── start_studio.py                 # Script chạy nhanh giao diện
│   ├── data/                           # Dictionaries & Model inference (v24, v23, v22, v21, v18)
│   ├── PaddleOCR/                      # Codebase inference PaddleOCR
│   └── README_STUDIO.md                # Hướng dẫn chi tiết sử dụng Studio
│
├── ocr-training/                       # 2. Pipeline Huấn luyện & Fine-tune Mô hình (Kaggle Pipeline)
│   ├── paddleocr_cham_finetune.ipynb  # Notebook chính chạy trên Kaggle GPU T4x2
│   ├── configs/                        # Các file cấu hình YAML huấn luyện
│   ├── scripts/                        # Scripts sinh dữ liệu tổng hợp & phẫu thuật trọng số
│   ├── data/                           # Dữ liệu huấn luyện (fonts, corpus Chăm đã chuẩn hóa)
│   ├── tests/                          # Bộ kịch bản kiểm thử & đánh giá CER/Accuracy
│   └── README_TRAINING.md              # Hướng dẫn chi tiết quy trình huấn luyện
│
├── Dockerfile                          # Dockerfile đóng gói và triển khai Web App Studio
├── .dockerignore                       # Cấu hình loại trừ file khi đóng gói Docker
├── .gitignore                          # Bộ lọc loại trừ file rác, file tạm và weights nặng
├── DESIGN.md                           # Định nghĩa Design System Tokens cho Studio
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
3. Dự án sử dụng tài khoản Kaggle: `gustavnguyen` với GPU T4x2 song song.

---

## 📑 Báo cáo Phương pháp Nghiên cứu & Công nghệ Cốt lõi

1. **Phân đoạn dòng chữ Chăm (Indic Line Segmentation)**:
   - Kết hợp mô hình học sâu **PaddleOCR DBNet** (tỷ lệ mở rộng unclip 1.8) và thuật toán phân tích đường cắt thung lũng (Valley-Cut Heuristics).
   - Cơ chế bảo vệ nét dính `difference_update` và đệm an toàn tối thiểu `0.40 * median_line_height`.
   - Cổng *Legacy-First* bảo toàn độ chính xác cho tài liệu sạch.
2. **Kiến trúc Nhận diện Chữ viết (Recognition Architecture)**:
   - Mạng nơ-ron **PP-OCRv4 SVTR-LCNet** mở rộng tensor đầu vào lên `[3, 48, 480]`.
   - Cơ chế giải mã kép: CTC Loss kết hợp NRTR Multi-Head Cross-Attention.
   - Chuẩn hóa thứ tự gõ Logic Chăm `normalize_unicode` (Brahmic Logical Order).
3. **Sinh Dữ liệu Tổng hợp & Tăng cường**:
   - 150,000 dòng dữ liệu tổng hợp qua 4 gói phân tầng.
   - Cơ chế lọc chữ lỗi ô vuông (tofu) bằng `fontTools` cmap và làm mờ động học trong RAM (*On-the-Fly Motion Blur*).
4. **Triển khai Trực tuyến**:
   - Đóng gói container Docker trên **Hugging Face Spaces** (`phucsd/cham-ocr-studio`).
   - Tên miền sản phẩm chính thức: **[https://ocr.cham.asia](https://ocr.cham.asia)**.
