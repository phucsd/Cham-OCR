#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Remote Orchestrator for Cham-DBNet on Lightning AI Studio.
Manages:
1. Phase 1 (CPU): Git clone, dependencies install, dataset generation (12,000 pages, 50% tight lines).
2. Phase 2 (H100 Switch): Switch machine to NVIDIA H100 (80GB VRAM SXM5).
3. Phase 3 (Training & Monitoring): Launch PaddleOCR DBNet distributed training, stream log metrics.
4. Phase 4 (Export & Download): Export lightweight inference model, package ZIP, download locally.
5. Phase 5 (Auto Stop): Stop Studio immediately to prevent credit consumption.
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
    log("🚀 KHỞI ĐỘNG REMOTE ORCHESTRATOR TRÊN LIGHTNING AI")
    log("=================================================================")

    studio = Studio(name="cham-det-h100", teamspace="phucsd", org="phucsd-org")
    log(f"Trạng thái Studio hiện tại: {studio.status} (Machine: {studio.machine})")

    # PHASE 1: CPU SETUP & DATASET GENERATION
    log("\n📦 --- PHASE 1: THIẾT LẬP MÔI TRƯỜNG & SINH DỮ LIỆU TRÊN CPU (0 CREDIT) ---")
    
    setup_cmds = [
        # 1. Clone or Pull Cham-OCR
        "if [ ! -d 'Cham-OCR' ]; then git clone https://github.com/phucsd1/Cham-OCR.git; else cd Cham-OCR && git pull origin main && cd ..; fi",
        # 2. Clone PaddleOCR inside Cham-OCR/ocr-training if not exists
        "cd /teamspace/studios/this_studio/Cham-OCR/ocr-training && if [ ! -d 'PaddleOCR' ]; then git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleOCR.git; fi",
        # 3. Download fonts
        "cd /teamspace/studios/this_studio/Cham-OCR/ocr-training && mkdir -p data/fonts && wget -q -nc -O data/fonts/NotoSansCham-Regular.ttf https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansCham/NotoSansCham-Regular.ttf && wget -q -nc -O data/fonts/NotoSansCham-Bold.ttf https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansCham/NotoSansCham-Bold.ttf",
        # 4. Download pretrained weights
        "cd /teamspace/studios/this_studio/Cham-OCR/ocr-training && mkdir -p PaddleOCR/pretrain_models && wget -q -nc -O PaddleOCR/pretrain_models/PPLCNetV3_x0_75_ocr_det.pdparams https://paddleocr.bj.bcebos.com/pretrained/PPLCNetV3_x0_75_ocr_det.pdparams",
        # 5. Install general pip dependencies
        "pip install -q pyclipper shapely imgaug lmdb attrdict pillow opencv-python pyyaml",
    ]

    for cmd in setup_cmds:
        log(f"Đang thực thi: {cmd[:70]}...")
        out = studio.run(cmd)
        if out.strip():
            log(f"   -> {out.strip()[:200]}")

    log("✅ Thiết lập mã nguồn, phông chữ và pretrained model hoàn tất!")

    # 6. Generate dataset on CPU
    log("⚙️ Bắt đầu sinh 12,000 trang huấn luyện và 1,200 trang kiểm thử (50% cự ly siêu hẹp 3-10px)...")
    gen_cmd = "cd /teamspace/studios/this_studio/Cham-OCR/ocr-training && python3 scripts/generate_detector_data_v2.py --num_train 12000 --num_val 1200 --output_dir PaddleOCR/data/detector_v2 --workers 8"
    out = studio.run(gen_cmd)
    log("Kết quả sinh dữ liệu:")
    for line in out.strip().splitlines()[-6:]:
        log(f"   {line}")

    # Copy config
    studio.run("cd /teamspace/studios/this_studio/Cham-OCR/ocr-training && mkdir -p PaddleOCR/configs/det && cp configs/det/ch_PP-OCRv4_det_h100.yml PaddleOCR/configs/det/")
    log("✅ Bộ dữ liệu và cấu hình H100 đã sẵn sàng!")

    # PHASE 2: SWITCH MACHINE TO H100
    log("\n🔥 --- PHASE 2: CHUYỂN ĐỔI PHẦN CỨNG SANG GPU NVIDIA H100 (80GB VRAM) ---")
    log("Đang gửi lệnh switch_machine sang H100...")
    studio.switch_machine(Machine.H100)
    log(f"Trạng thái sau khi chuyển: {studio.status} (Machine: {studio.machine})")

    # Install paddlepaddle-gpu for Hopper/CUDA
    log("📦 Cài đặt PaddlePaddle GPU bản CUDA 11.8/12.x tối ưu H100...")
    studio.run("pip install -q paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu118/")
    gpu_info = studio.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    log(f"🔥 GPU đã nhận diện: {gpu_info.strip()}")

    # PHASE 3: LAUNCH TRAINING WITH BACKGROUND MONITORING
    log("\n🚀 --- PHASE 3: KHỞI CHẠY HUẤN LUYỆN CHAM-DBNET TRÊN H100 ---")
    train_script = """cd /teamspace/studios/this_studio/Cham-OCR/ocr-training/PaddleOCR && nohup python3 -m paddle.distributed.launch --gpus '0' tools/train.py -c configs/det/ch_PP-OCRv4_det_h100.yml > train.log 2>&1 &"""
    studio.run(train_script)
    log("✅ Lệnh huấn luyện đã kích hoạt chạy ngầm (background) an toàn trên H100!")

    # MONITORING LOOP
    log("📊 Bắt đầu vòng lặp theo dõi liên tục tiến trình huấn luyện...")
    prev_log_tail = ""
    idle_count = 0
    start_time = time.time()

    while True:
        time.sleep(15)
        # Check if process is still running
        ps_out = studio.run("ps aux | grep tools/train.py | grep -v grep | wc -l")
        is_running = int(ps_out.strip() or "0") > 0
        
        # Read log tail
        try:
            log_tail = studio.run("cd /teamspace/studios/this_studio/Cham-OCR/ocr-training/PaddleOCR && tail -n 8 train.log")
            if log_tail.strip() != prev_log_tail:
                prev_log_tail = log_tail.strip()
                log("--- Cập nhật tiến trình mới nhất ---")
                for l in log_tail.strip().splitlines()[-4:]:
                    log(f"   {l}")
        except Exception as e:
            log(f"Không thể đọc log: {e}")

        elapsed_mins = (time.time() - start_time) / 60
        if not is_running:
            log(f"\n🎉 Tiến trình huấn luyện đã kết thúc sau {elapsed_mins:.1f} phút!")
            break

    # PHASE 4: EXPORT MODEL & DOWNLOAD
    log("\n📦 --- PHASE 4: XUẤT MÔ HÌNH INFERENCE & ĐÓNG GÓI ---")
    export_cmd = """cd /teamspace/studios/this_studio/Cham-OCR/ocr-training/PaddleOCR && \
BEST_MODEL="output/ch_PP-OCRv4_det_cham_h100/best_accuracy" && \
if [ ! -f "${BEST_MODEL}.pdparams" ]; then BEST_MODEL="output/ch_PP-OCRv4_det_cham_h100/latest"; fi && \
python3 tools/export_model.py -c configs/det/ch_PP-OCRv4_det_h100.yml -o Global.pretrained_model=${BEST_MODEL} Global.save_inference_dir=../output/ch_PP-OCRv4_det_cham_infer && \
cd ../output && zip -r ../cham_dbnet_v1_infer.zip ch_PP-OCRv4_det_cham_infer/"""
    
    out = studio.run(export_cmd)
    log("Kết quả đóng gói:")
    for l in out.strip().splitlines()[-5:]:
        log(f"   {l}")

    # Download file to local workspace
    local_zip_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cham_dbnet_v1_infer.zip")
    log(f"⬇️ Đang tải tệp mô hình về máy local: {local_zip_path}...")
    studio.download_file("Cham-OCR/ocr-training/cham_dbnet_v1_infer.zip", local_zip_path)
    log(f"✅ Đã tải thành công tệp mô hình ({os.path.getsize(local_zip_path) / (1024*1024):.2f} MB)!")

    # PHASE 5: STOP STUDIO TO SAVE CREDITS
    log("\n🛑 --- PHASE 5: TỰ ĐỘNG DỪNG STUDIO ĐỂ BẢO VỆ CREDIT ---")
    studio.stop()
    log("✅ Studio đã được DỪNG (STOPPED) an toàn! Không còn tiêu tốn bất kỳ credit nào.")
    log("=================================================================")
    log("🎊 HOÀN THÀNH TOÀN BỘ QUY TRÌNH HUẤN LUYỆN CHAM-DBNET TRÊN H100!")
    log("=================================================================")

if __name__ == '__main__':
    main()
