#!/usr/bin/env bash
# ==============================================================================
# Cham OCR Model V25 Training Script for Lightning AI Studios
# Supports: NVIDIA L4 (24GB), A10G (24GB), A100 (40/80GB), H100 (80GB)
# ==============================================================================

set -e

echo "======================================================================"
echo "🚀 KHỞI ĐỘNG HUẤN LUYỆN CHAM-OCR MODEL V25 TRÊN LIGHTNING AI"
echo "======================================================================"

# 1. Kiểm tra GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Lỗi: Không tìm thấy GPU NVIDIA. Vui lòng chọn GPU (L4 / A100 / H100) trong Compute Settings!"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "✅ Đang chạy trên phần cứng: $GPU_NAME"

# 2. Cài đặt môi trường & dependencies nếu chưa có
echo "📦 Kiểm tra và cài đặt thư viện..."
pip install -q --upgrade pip
pip install -q pyclipper shapely imgaug lmdb tqdm "numpy<2.0" || true

# 3. Tối ưu hóa I/O: Nạp dữ liệu lên RAM Disk (/dev/shm) để GPU đạt 100% compute
echo "⚡ Chuẩn bị RAM Disk (/dev/shm) để chống nghẽn I/O..."
DATA_SRC="./data/cham_synthetic_v25"
if [ -d "$DATA_SRC" ]; then
    mkdir -p /dev/shm/cham_v25_train
    mkdir -p /dev/shm/cham_v25_val
    echo "  - Sao chép dữ liệu huấn luyện vào /dev/shm..."
    cp -r "$DATA_SRC/train_images" /dev/shm/cham_v25_train/
    cp "$DATA_SRC/train_label.txt" /dev/shm/cham_v25_train/
    cp -r "$DATA_SRC/val_images" /dev/shm/cham_v25_val/
    cp "$DATA_SRC/val_label.txt" /dev/shm/cham_v25_val/
    echo "  ✅ Dữ liệu đã sẵn sàng trên RAM Disk!"
else
    echo "⚠️ Lưu ý: Chưa tìm thấy $DATA_SRC. Vui lòng chạy kịch bản sinh dữ liệu v25 trước!"
fi

# 4. Khởi chạy huấn luyện
echo "======================================================================"
echo "🔥 BẮT ĐẦU HUẤN LUYỆN MODEL V25 (Single-GPU Lightning Config)..."
echo "======================================================================"

# QUY TẮC AN TOÀN TRÊN LIGHTNING AI:
# Dùng python3 -u tools/train.py thay vì distributed.launch để tránh nhân bản tiến trình ngốn VRAM
python3 -u tools/train.py -c configs/rec_cham_v25_lightning.yml

# 5. Xuất mô hình Inference siêu nhẹ (~11MB)
echo "======================================================================"
echo "📦 XUẤT MÔ HÌNH INFERENCE V25 (EXPORT INFERENCE MODEL)..."
echo "======================================================================"
python3 tools/export_model.py \
    -c configs/rec_cham_v25_lightning.yml \
    -o Global.pretrained_model=./output/rec_cham_v25/best_accuracy \
       Global.save_inference_dir=./output/rec_cham_v25_infer

# 6. Đóng gói tệp nén để tải về
echo "🗜️ Đang đóng gói tệp nén rec_cham_v25_infer.zip..."
cd ./output
zip -r ../rec_cham_v25_infer.zip rec_cham_v25_infer/
cd ..

echo "======================================================================"
echo "🎉 HOÀN TẤT HUẤN LUYỆN V25 THÀNH CÔNG!"
echo "   Tệp mô hình tải về: ./rec_cham_v25_infer.zip"
echo "======================================================================"
