#!/bin/bash
set -e

echo "================================================================="
echo "🚀 KHỞI CHẠY HUẤN LUYỆN CHAM-DBNET TRÊN LIGHTNING AI (GPU H100)"
echo "================================================================="

# 1. Kiểm tra GPU
nvidia-smi

# 2. Cài đặt thư viện
echo "📦 Đang cài đặt thư viện phụ thuộc..."
pip install --upgrade pip
pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install pyclipper shapely imgaug lmdb attrdict pillow opencv-python

# 3. Clone PaddleOCR nếu chưa có
if [ ! -d "PaddleOCR" ]; then
    echo "📥 Đang tải PaddleOCR release/2.7..."
    git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleOCR.git
    cd PaddleOCR
    pip install -r requirements.txt
    cd ..
fi

# 4. Tải Fonts Chăm
echo "🔤 Đang tải bộ phông chữ Chăm NotoSansCham..."
mkdir -p data/fonts data/corpus
wget -q -nc -O data/fonts/NotoSansCham-Regular.ttf https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansCham/NotoSansCham-Regular.ttf
wget -q -nc -O data/fonts/NotoSansCham-Bold.ttf https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansCham/NotoSansCham-Bold.ttf
wget -q -nc -O data/fonts/NotoSansCham-Black.ttf https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansCham/NotoSansCham-Black.ttf

# 5. Sinh dữ liệu đặc trị dòng hẹp
echo "⚙️ Đang sinh 15,000 trang huấn luyện và 1,500 trang kiểm thử (Tight lines)..."
python3 scripts/generate_detector_data_v2.py \
    --num_train 15000 \
    --num_val 1500 \
    --output_dir PaddleOCR/data/detector_v2 \
    --workers 8

# 6. Tải pretrained weights
mkdir -p PaddleOCR/pretrain_models
wget -q -nc -O PaddleOCR/pretrain_models/PPLCNetV3_x0_75_ocr_det.pdparams https://paddleocr.bj.bcebos.com/pretrained/PPLCNetV3_x0_75_ocr_det.pdparams

# 7. Sao chép cấu hình H100
mkdir -p PaddleOCR/configs/det
cp configs/det/ch_PP-OCRv4_det_h100.yml PaddleOCR/configs/det/

# 8. Chạy huấn luyện phân tán trên GPU H100
echo "🔥 Bắt đầu huấn luyện mô hình..."
cd PaddleOCR
python3 -m paddle.distributed.launch --gpus '0' tools/train.py -c configs/det/ch_PP-OCRv4_det_h100.yml
cd ..

# 9. Xuất mô hình Inference
echo "📦 Đang xuất mô hình Inference..."
cd PaddleOCR
BEST_MODEL="output/ch_PP-OCRv4_det_cham_h100/best_accuracy"
if [ ! -f "${BEST_MODEL}.pdparams" ]; then
    BEST_MODEL="output/ch_PP-OCRv4_det_cham_h100/latest"
fi

python3 tools/export_model.py \
    -c configs/det/ch_PP-OCRv4_det_h100.yml \
    -o Global.pretrained_model=${BEST_MODEL} \
       Global.save_inference_dir=../output/ch_PP-OCRv4_det_cham_infer
cd ..

# 10. Đóng gói ZIP
echo "🎁 Đang đóng gói tệp cham_dbnet_v1_infer.zip..."
cd output
zip -r ../cham_dbnet_v1_infer.zip ch_PP-OCRv4_det_cham_infer/
cd ..

echo "================================================================="
echo "🎉 HOÀN TẤT TOÀN DIỆN! TỆP NÉN: cham_dbnet_v1_infer.zip"
echo "================================================================="
