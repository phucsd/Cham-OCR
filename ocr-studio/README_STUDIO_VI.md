[English](README_STUDIO.md) | **Tiếng Việt**

# Cham OCR Diagnostic Studio (Giao diện Chẩn đoán & Review OCR)

Ứng dụng này cung cấp giao diện web để thử nghiệm, chẩn đoán lỗi phân đoạn dòng (Line Segmentation) và kiểm tra độ chính xác của các mô hình nhận diện chữ viết tiếng Chăm khác nhau.

> 🌟 **Trải nghiệm Trực tuyến**: [https://ocr.cham.asia](https://ocr.cham.asia)

## 📁 Cấu trúc Thư mục Phân hệ Studio

```
ocr-studio/
├── README_STUDIO.md                    # Bản tiếng Anh
├── README_STUDIO_VI.md                 # Hướng dẫn này (Tiếng Việt)
├── app.py                              # Script chạy ứng dụng Web Server & Studio Backend
├── index.html                          # Frontend Claude Warm Light Theme (Giao diện tiếng Anh học thuật)
├── start_studio.py                     # Script khởi động nhanh giao diện
├── PaddleOCR/                          # Bản sao thư viện PaddleOCR phục vụ suy luận cục bộ
└── data/                               # Dữ liệu phục vụ suy luận
    ├── ocr_corrections.txt             # Lưu trữ các chỉnh sửa nhãn của người dùng
    ├── cham_dict_v*.txt                # Từ điển ký tự các phiên bản mô hình
    └── output/                         # Thư mục chứa các mô hình nhận diện đã xuất (inference model)
        ├── rec_cham_inference_v24/
        ├── rec_cham_inference_v23/
        ├── rec_cham_inference_v22/
        └── rec_cham_inference_v21/
```

---

## ⚡ Yêu cầu hệ thống & Cài đặt

1. Cài đặt các thư viện cần thiết từ thư mục gốc:
   ```bash
   pip install -r requirements.txt
   ```
   *Lưu ý*: Đối với môi trường Python mới (như Python 3.13+), ứng dụng đã tự động vá lỗi tương thích NumPy 2.x bằng monkeypatch.

2. Đảm bảo bạn đã có các mô hình inference trong `ocr-studio/data/output/` và file từ điển tương ứng trong `ocr-studio/data/`.

---

## 🚀 Hướng dẫn khởi chạy

Chạy lệnh sau tại thư mục gốc của dự án hoặc trong thư mục `ocr-studio`:

```bash
python ocr-studio/app.py
```

Hoặc sử dụng tệp khởi chạy nhanh:
```bash
python ocr-studio/start_studio.py
```

Ứng dụng sẽ tìm kiếm cổng trống thích hợp (mặc định: `7860`, `8080`, `8081`...) và khởi chạy:
```
🚀 Cham OCR Diagnostic Studio is running at: http://localhost:7860
📁 Corrections will be saved to: ocr-studio/data/ocr_corrections.txt
```

Mở trình duyệt và truy cập `http://localhost:7860` để bắt đầu làm việc.
