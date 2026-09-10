#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Automated Pipeline for Cham-DBNet on Lightning AI (H100).
1. Data Gen (2,000 pages ~40,000 lines on CPU)
2. Switch to H100 GPU
3. Install paddlepaddle-gpu & launch distributed training
4. Stream and monitor training progress
5. Export & package inference model
6. Download locally and stop Studio
"""

import os
import sys
import time
from lightning_sdk import Studio, Machine

if sys.stdout and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def main():
    log("=================================================================")
    log("🚀 KHỞI CHẠY PIPELINE HUẤN LUYỆN CHAM-DBNET TRÊN LIGHTNING AI H100")
    log("=================================================================")

    studio = Studio(name="cham-det-h100", teamspace="phucsd", org="phucsd-org")
    log(f"Studio Status: {studio.status} | Machine: {studio.machine}")

    # Step 1: Check existing dataset
    dataset_exists = studio.run("test -f Cham-OCR/ocr-training/PaddleOCR/data/detector_v2/det_train_label.txt && echo EXISTS || echo NO").strip()
    if "EXISTS" in dataset_exists:
        log("✅ [BƯỚC 1/5] Bộ dữ liệu 2,000 trang (det_train_label.txt) đã có sẵn trên đĩa, bỏ qua bước sinh!")
    else:
        log("\n⚙️ [BƯỚC 1/5] Sinh 2,000 trang dữ liệu đặc trị dòng hẹp (50% gap 3-10px)...")
        studio.run(
            "pkill -9 -f generate_detector_data_v2.py || true",
            "rm -rf Cham-OCR/ocr-training/PaddleOCR/data/detector_v2"
        )
        gen_cmd = "cd Cham-OCR/ocr-training && python3 scripts/generate_detector_data_v2.py --num_train 2000 --num_val 200 --output_dir PaddleOCR/data/detector_v2 --workers 4"
        gen_out = studio.run(gen_cmd)
        for line in gen_out.strip().splitlines()[-4:]:
            log(f"   {line}")
        log("✅ Bộ dữ liệu 2,000 trang đã sinh và tạo nhãn SimpleDataSet thành công!")

    # Step 2: Switch to NVIDIA H100
    log("\n🔥 [BƯỚC 2/5] Nâng cấp phần cứng sang NVIDIA H100 (80GB VRAM SXM5 - lit-h100-80gb-1)...")
    studio.switch_machine("lit-h100-80gb-1")
    log(f"✅ Đã chuyển sang H100! Machine: {studio.machine}")

    # Step 3: Install PaddlePaddle GPU & Verify
    log("\n📦 [BƯỚC 3/5] Cài đặt PaddlePaddle GPU bản CUDA 11.8/12.x tối ưu cho Hopper H100...")
    studio.run("pip install -q paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu118/")
    gpu_check = studio.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    log(f"🔥 GPU H100 đã sẵn sàng: {gpu_check.strip()}")

    # Step 4: Launch Training & Monitor
    log("\n🚀 [BƯỚC 4/5] Khởi chạy huấn luyện Cham-DBNet (batch 64, 150 epochs)...")
    train_cmd = "cd Cham-OCR/ocr-training/PaddleOCR && nohup python3 -m paddle.distributed.launch --gpus '0' tools/train.py -c configs/det/ch_PP-OCRv4_det_h100.yml > train.log 2>&1 &"
    studio.run(train_cmd)
    log("✅ Huấn luyện đã kích hoạt chạy ngầm (background) an toàn trên H100!")

    log("\n📊 Bắt đầu theo dõi chỉ số loss và epoch liên tục...")
    prev_log = ""
    start_train_time = time.time()
    
    while True:
        time.sleep(15)
        # Check running process
        ps_out = studio.run("ps aux | grep tools/train.py | grep -v grep | wc -l").strip()
        is_running = int(ps_out or "0") > 0
        
        try:
            log_tail = studio.run("cd Cham-OCR/ocr-training/PaddleOCR && tail -n 6 train.log").strip()
            if log_tail and log_tail != prev_log:
                prev_log = log_tail
                log("--- TIẾN ĐỘ HUẤN LUYỆN MỚI NHẤT ---")
                for l in log_tail.splitlines()[-4:]:
                    log(f"   {l}")
        except Exception as e:
            log(f"Lưu ý: {e}")
            
        if not is_running:
            elapsed = (time.time() - start_train_time) / 60
            log(f"\n🎉 Tiến trình huấn luyện hoàn tất sau {elapsed:.1f} phút!")
            break

    # Step 5: Export, Zip, Download, and Stop
    log("\n📦 [BƯỚC 5/5] Xuất mô hình Inference nhẹ ~4.5MB & đóng gói ZIP...")
    export_cmd = """cd Cham-OCR/ocr-training/PaddleOCR && \
BEST_MODEL="output/ch_PP-OCRv4_det_cham_h100/best_accuracy" && \
if [ ! -f "${BEST_MODEL}.pdparams" ]; then BEST_MODEL="output/ch_PP-OCRv4_det_cham_h100/latest"; fi && \
python3 tools/export_model.py -c configs/det/ch_PP-OCRv4_det_h100.yml -o Global.pretrained_model=${BEST_MODEL} Global.save_inference_dir=../output/ch_PP-OCRv4_det_cham_infer && \
cd ../output && zip -r ../cham_dbnet_v1_infer.zip ch_PP-OCRv4_det_cham_infer/"""
    studio.run(export_cmd)
    log("✅ Đã xuất mô hình và đóng gói cham_dbnet_v1_infer.zip thành công!")

    # Download to local
    local_zip = os.path.join(os.getcwd(), "ocr-training", "cham_dbnet_v1_infer.zip")
    log(f"⬇️ Đang tải tệp mô hình về máy local: {local_zip}...")
    # Read remote zip in base64 and write locally to ensure 100% reliability
    b64_zip = studio.run("cd Cham-OCR/ocr-training && python3 -c \"import base64, pathlib; print(base64.b64encode(pathlib.Path('cham_dbnet_v1_infer.zip').read_bytes()).decode('ascii'))\"").strip()
    import base64
    with open(local_zip, "wb") as f:
        f.write(base64.b64decode(b64_zip))
    log(f"🎉 ĐÃ TẢI VỀ THÀNH CÔNG: {local_zip} ({os.path.getsize(local_zip)/(1024*1024):.2f} MB)")

    # STOP STUDIO IMMEDIATELY
    log("\n🛑 Tự động dừng Studio để bảo vệ số dư credit...")
    studio.stop()
    log(f"✅ Studio đã DỪNG (Status: {studio.status})! 0 credit bị tiêu tốn thêm.")
    log("=================================================================")
    log("🎊 HOÀN THÀNH TOÀN DIỆN!")
    log("=================================================================")

if __name__ == "__main__":
    main()
