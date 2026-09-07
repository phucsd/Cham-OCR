---
title: Cham OCR Diagnostic Studio
emoji: 📜
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Dự án Nhận diện Chữ viết tiếng Chăm (Cham-OCR Monorepo)

Dự án nghiên cứu, huấn luyện và chẩn đoán nhận diện chữ viết tiếng Chăm (Akhar Thrah & West Cham) dựa trên nền tảng PaddleOCR PP-OCRv4 Multilingual. Cấu trúc mã nguồn được phân định rõ ràng thành hai phân hệ độc lập:

## 📁 Cấu trúc Thư mục Dự án

```
Cham-OCR/
├── webapp-ui/                          # 1. Ứng dụng Web Chẩn đoán & Review OCR (Diagnostic Studio)
│   ├── app.py                          # Backend HTTP Server & Multi-Crop Orchestrator
│   ├── index.html                      # Giao diện Studio Warm Light Theme
│   ├── start_studio.py                 # Script chạy nhanh giao diện
│   ├── data/                           # Dictionaries & Model inference (v23, v22, v21, v18)
│   ├── PaddleOCR/                      # Codebase inference PaddleOCR
│   └── README_WEBAPP.md                # Hướng dẫn chi tiết sử dụng Web App
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

### A. Chạy thử nghiệm nhận diện & Chẩn đoán OCR (Web App Studio):
1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
2. Khởi chạy studio chẩn đoán:
   ```bash
   python webapp-ui/app.py
   ```
   Hoặc chạy qua file khởi động nhanh:
   ```bash
   python webapp-ui/start_studio.py
   ```
3. Mở trình duyệt tại địa chỉ: `http://localhost:7860`
4. Chi tiết tài liệu: xem tại [webapp-ui/README_WEBAPP.md](webapp-ui/README_WEBAPP.md).

### B. Huấn luyện / Fine-tune mô hình mới trên Kaggle:
1. Đọc hướng dẫn chi tiết tại [ocr-training/README_TRAINING.md](ocr-training/README_TRAINING.md).
2. Sử dụng notebook [paddleocr_cham_finetune.ipynb](ocr-training/paddleocr_cham_finetune.ipynb) và đẩy lên Kaggle GPU thông qua Kaggle API hoặc giao diện web Kaggle.
3. Dự án sử dụng tài khoản Kaggle: `gustavnguyen` với GPU T4x2 song song.
