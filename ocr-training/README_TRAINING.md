# Fine-tuning PaddleOCR PP-OCRv4 tiếng Chăm (Akhar Thrah & West Cham)

Dự án này cung cấp quy trình và công cụ tự động hóa để chuẩn bị dữ liệu tổng hợp tiếng Chăm (Akhar Thrah & West Cham/Srak), tự động hóa việc cấu hình tối ưu và fine-tune mô hình nhận diện chữ viết **PaddleOCR PP-OCRv4 Multilingual** trên môi trường **Kaggle GPU Notebooks** hoặc máy cá nhân chạy CUDA.

## 📁 Cấu trúc Thư mục Dự án con Huấn luyện

```
ocr-training/
├── README_TRAINING.md                  # Hướng dẫn này
├── paddleocr_cham_finetune.ipynb      # Notebook chính chạy trên Kaggle GPU
├── data/
│   ├── fonts/                          # Font chữ tiếng Chăm sinh dữ liệu
│   ├── corpus/                         # File văn bản tiếng Chăm thô
│   └── cham_synthetic_images/          # [Thư mục sinh tự động] Dữ liệu ảnh & nhãn
├── configs/                            # Các file cấu hình YAML huấn luyện
├── scripts/
│   ├── generate_data.py               # Sinh dữ liệu tổng hợp với Albumentations & Font Validation
│   └── configure_training.py          # Tải base model, sửa đổi YAML cấu hình thông minh
└── tests/                              # Các script kiểm thử và đánh giá mô hình
```

---

## 🚀 Hướng dẫn Triển khai nhanh trên Kaggle Notebooks (Khuyến nghị)

Để tận dụng sức mạnh GPU miễn phí (Tesla P100 hoặc T4 x2) giúp tăng tốc độ huấn luyện, hãy thực hiện các bước sau:

### 1. Chuẩn bị Font tiếng Chăm
* Tải về các font chữ tiếng Chăm dạng `.ttf` hoặc `.otf` (xem hướng dẫn chi tiết tại `data/fonts/`).
* Trên Kaggle:
  1. Vào trang cá nhân Kaggle của bạn -> chọn **Datasets** -> **New Dataset** -> Upload các file font lên và đặt tên (ví dụ: `cham-fonts`).
  2. Mở Kaggle Notebook của bạn lên và nhấn **+ Add Input** -> Tìm kiếm dataset `cham-fonts` bạn vừa tạo và add vào notebook.

### 2. Import Notebook và Chạy
* Tải tệp tin `paddleocr_cham_finetune.ipynb` về máy.
* Trên Kaggle Notebook mới:
  1. Chọn **File** -> **Import Notebook** -> Upload file `.ipynb` lên.
  2. Tại bảng điều khiển **Settings** phía bên phải của Notebook, đổi mục **Accelerator** thành **GPU** (Ví dụ: GPU T4 x2 hoặc GPU P100).
  3. Kích hoạt mạng Internet (**Internet on** trong tab Settings).
  4. Bỏ chú thích dòng lệnh sao chép font trong Cell của Giai đoạn 2 và điền đúng đường dẫn Kaggle Input font của bạn:
     ```bash
     !cp /kaggle/input/cham-fonts/*.ttf /kaggle/working/paddleocr_cham_finetune/data/fonts/
     ```
  5. Chọn **Run All** để chạy toàn bộ quy trình huấn luyện tự động.

---

## 🛠️ Các đặc tính kỹ thuật nổi bật

### 1. Tránh chữ lỗi ô vuông (Tofu Characters Validation)
Script `generate_data.py` tích hợp thư viện `fontTools` để đọc trực tiếp bản đồ ký tự (`cmap`) của từng tệp font được tải lên. Nếu văn bản tiếng Chăm chứa ký tự mà font đó không hỗ trợ hiển thị, mẫu dữ liệu đó sẽ được tự động bỏ qua (skip), tránh sinh ra các ảnh chứa ô vuông lỗi làm sai lệch mô hình học máy.

### 2. Tăng cường dữ liệu (Data Augmentation) mạnh mẽ
Quy trình sử dụng thư viện `albumentations` để giả lập các điều kiện chụp ảnh thực tế qua 5 bộ biến đổi:
* **Xoay ảnh phối cảnh**: Xoay ngẫu nhiên $\pm 8^\circ$, dịch chuyển tịnh tiến và biến đổi phối cảnh 3D mô phỏng chụp xiên bằng camera điện thoại.
* **Làm mờ ảnh**: Motion Blur (mờ chuyển động), Gaussian Blur mô phỏng out-focus.
* **Nhiễu hạt Gauss**: Mô phỏng nhiễu cảm biến ảnh ban đêm.
* **Nền đa dạng**: Tránh overfit bằng cách tự động sinh màu pastel, gradient tuyến tính và giấy thô có các dòng kẻ tập học sinh mảnh nằm ngoài vùng chữ cái chính.

### 3. Tự động tránh lỗi CUDA OOM & tràn shm trên Kaggle
Script `configure_training.py` tự động phát hiện GPU/CPU của hệ thống, đồng thời cấu hình file YAML huấn luyện phù hợp với cấu hình máy ảo Kaggle:
* Giảm `batch_size_per_card` xuống `64` để tránh lỗi tràn bộ nhớ card đồ họa CUDA Out Of Memory.
* Giới hạn `num_workers` của bộ nạp dữ liệu xuống `2` để tránh tràn phân vùng bộ nhớ chia sẻ `/dev/shm` trong Docker của Kaggle.

### 4. Đóng gói lưu trữ an toàn (Kaggle Persistence)
Khi quá trình huấn luyện hoàn tất, Notebook sẽ tự động nén toàn bộ kết quả mô hình tốt nhất (`best_accuracy`) thành tệp `rec_cham_best_model.zip` và tạo đường dẫn tải xuống trực quan. Bạn có thể lưu trực tiếp về máy tính cá nhân mà không sợ mất file khi session của Kaggle hết giờ.

---

## 👨‍💻 Cách chạy thử nghiệm cục bộ (Local Development)

Nếu bạn muốn chạy thử nghiệm trên máy tính cá nhân của mình:

1. Sao chép font chữ tiếng Chăm của bạn vào thư mục `data/fonts/`.
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install opencv-python albumentations Pillow pyyaml fonttools
   ```
3. Chạy thử quy trình sinh dữ liệu cục bộ (sẽ sinh 100 mẫu thử nghiệm và từ điển `cham_dict.txt` tương ứng):
   ```bash
   python scripts/generate_data.py
   ```
4. Để cấu hình file YAML mẫu, hãy chạy:
   ```bash
   python scripts/configure_training.py
   ```
