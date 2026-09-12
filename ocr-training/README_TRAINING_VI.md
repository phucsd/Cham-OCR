[English](README_TRAINING.md) | **Tiếng Việt**

# Huấn luyện & Fine-tuning PaddleOCR PP-OCRv4 tiếng Chăm Đông (Akhar Thrah)

Dự án này cung cấp quy trình và công cụ tự động hóa để chuẩn bị dữ liệu tổng hợp tiếng Chăm Đông (Akhar Thrah, với định hướng mở rộng Chăm Tây/Cam Srak trong tương lai), tự động hóa việc cấu hình tối ưu và fine-tune mô hình nhận diện chữ viết **PaddleOCR PP-OCRv4 Multilingual** trên môi trường **Kaggle GPU Notebooks** hoặc máy cá nhân chạy CUDA.

> 🌟 **Trải nghiệm Trực tuyến Mô hình Đã Huấn luyện**: [https://ocr.cham.asia](https://ocr.cham.asia)

## 📁 Cấu trúc Thư mục Phân hệ Huấn luyện

```
ocr-training/
├── README_TRAINING.md                  # Bản tiếng Anh
├── README_TRAINING_VI.md               # Hướng dẫn này (Tiếng Việt)
├── paddleocr_cham_finetune.ipynb      # Notebook chính chạy trên Kaggle GPU T4x2
├── configs/                            # Các file cấu hình YAML huấn luyện (v24, v25)
├── scripts/                            # Scripts sinh dữ liệu tổng hợp & phẫu thuật trọng số
│   ├── generate_data.py               # Sinh dữ liệu tổng hợp với Albumentations & Font Validation
│   ├── configure_training.py          # Tải base model, sửa đổi YAML cấu hình thông minh
│   ├── surgery_v25_weights.py         # Phẫu thuật trọng số theo ký tự
│   └── kaggle_auth.py                 # Xác thực Kaggle an toàn qua biến môi trường
├── data/                               # Dữ liệu huấn luyện
│   ├── fonts/                          # Font chữ tiếng Chăm sinh dữ liệu
│   └── corpus/                         # File văn bản tiếng Chăm thô
└── tests/                              # Các script kiểm thử và đánh giá mô hình
```

---

## 🚀 Hướng dẫn Triển khai nhanh trên Kaggle Notebooks

1. Đăng nhập tài khoản Kaggle của dự án: `gustavnguyen`.
2. Sử dụng GPU kép `--accelerator NvidiaTeslaT4` để cấp phát GPU T4x2.
3. Kích hoạt huấn luyện phân tán đa GPU:
   ```bash
   python3 -m paddle.distributed.launch --gpus '0,1' tools/train.py -c configs/rec_cham_v25.yml
   ```
4. **Cơ chế chia chặng 12 tiếng (12-Hour Multi-Stage Checkpoint)**: Chia tiến trình huấn luyện thành 3 chặng an toàn (Epochs 1–12, 13–22, 23–40 theo cấu hình `stage_end_epoch` trong `configs/rec_cham_v25.yml`) để tránh timeout của Kaggle.
