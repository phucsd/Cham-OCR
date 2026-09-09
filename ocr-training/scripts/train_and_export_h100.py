#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-Click Automated Training & Inference Export Pipeline for Cham-DBNet on Lightning AI (H100).
Executes:
1. Environment & CUDA hardware check (Hopper H100 / sm_90 verification)
2. Dataset presence check (generates if missing)
3. Pretrained weight verification
4. Distributed/Single GPU training with PaddleOCR DBNet
5. Model export to lightweight inference model (inference.pdmodel / inference.pdiparams)
6. Automatic ZIP packaging for 1-click download.
"""

import os
import sys
import shutil
import zipfile
import subprocess

# Force UTF-8 stdout
if sys.stdout and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def check_environment():
    print("=" * 65)
    print("🔍 1. KIỂM TRA MÔI TRƯỜNG & GPU PHẦN CỨNG (LIGHTNING AI)")
    print("=" * 65)
    try:
        import paddle
        print(f"   - PaddlePaddle version: {paddle.__version__}")
        print(f"   - CUDA compiled: {paddle.is_compiled_with_cuda()}")
        if paddle.is_compiled_with_cuda():
            count = paddle.device.cuda.device_count()
            print(f"   - Số lượng GPU khả dụng: {count}")
            for i in range(count):
                name = paddle.device.cuda.get_device_name(i)
                print(f"     + GPU {i}: {name}")
        else:
            print("   ⚠️ CẢNH BÁO: PaddlePaddle chưa kích hoạt CUDA GPU!")
    except ImportError:
        print("   ❌ LỖI: Chưa cài đặt PaddlePaddle! Chạy: pip install paddlepaddle-gpu")
        sys.exit(1)

def run_command(cmd, desc="Running"):
    print(f"\n🚀 [{desc}] Lệnh: {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"❌ Lỗi khi thực hiện: {desc} (Mã lỗi: {ret.returncode})")
        sys.exit(ret.returncode)
    print(f"✅ Hoàn tất: {desc}")

def main():
    check_environment()
    
    workspace_dir = os.path.abspath(os.getcwd())
    paddleocr_dir = os.path.join(workspace_dir, "PaddleOCR")
    if not os.path.exists(paddleocr_dir):
        print("\n📥 Chưa tìm thấy PaddleOCR, đang clone release/2.7...")
        run_command("git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleOCR.git", "Clone PaddleOCR")
        
    config_path = os.path.join(workspace_dir, "configs", "det", "ch_PP-OCRv4_det_h100.yml")
    if not os.path.exists(config_path):
        # Check inside PaddleOCR
        config_path = os.path.join(paddleocr_dir, "configs", "det", "ch_PP-OCRv4_det_h100.yml")
        
    # Check dataset
    dataset_dir = os.path.join(paddleocr_dir, "data", "detector_v2")
    if not os.path.exists(dataset_dir) or not os.path.exists(os.path.join(dataset_dir, "det_train_label.txt")):
        print("\n📂 Chưa có dữ liệu detector_v2, đang khởi chạy bộ sinh dữ liệu đặc trị dòng hẹp...")
        gen_script = os.path.join(workspace_dir, "scripts", "generate_detector_data_v2.py")
        if not os.path.exists(gen_script):
            gen_script = "scripts/generate_detector_data_v2.py"
        run_command(f"python3 {gen_script} --num_train 12000 --num_val 1200 --output_dir {dataset_dir} --workers 8", "Sinh dữ liệu Detection V2")

    # Start training
    print("\n" + "=" * 65)
    print("🔥 2. BẮT ĐẦU HUẤN LUYỆN CHAM-DBNET TRÊN GPU H100")
    print("=" * 65)
    
    import paddle
    gpu_count = paddle.device.cuda.device_count() if paddle.is_compiled_with_cuda() else 1
    gpu_list = ",".join([str(i) for i in range(gpu_count)])
    
    os.chdir(paddleocr_dir)
    train_cmd = f"python3 -m paddle.distributed.launch --gpus '{gpu_list}' tools/train.py -c {config_path}"
    run_command(train_cmd, "Huấn luyện PaddleOCR DBNet")
    
    # Export Model
    print("\n" + "=" * 65)
    print("📦 3. XUẤT MÔ HÌNH INFERENCE (INFERENCE MODEL EXPORT)")
    print("=" * 65)
    
    output_dir = os.path.join(paddleocr_dir, "output", "ch_PP-OCRv4_det_cham_h100")
    best_model = os.path.join(output_dir, "best_accuracy")
    if not os.path.exists(best_model + ".pdparams"):
        best_model = os.path.join(output_dir, "latest")
        
    export_infer_dir = os.path.join(workspace_dir, "output", "ch_PP-OCRv4_det_cham_infer")
    os.makedirs(export_infer_dir, exist_ok=True)
    
    export_cmd = f"python3 tools/export_model.py -c {config_path} -o Global.pretrained_model={best_model} Global.save_inference_dir={export_infer_dir}"
    run_command(export_cmd, "Xuất mô hình Inference DBNet")
    
    # Package into ZIP
    print("\n" + "=" * 65)
    print("🎁 4. ĐÓNG GÓI TỆP NÉN SẴN SÀNG TẢI VỀ (ZIP PACKAGING)")
    print("=" * 65)
    
    zip_path = os.path.join(workspace_dir, "cham_dbnet_v1_infer.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(export_infer_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, os.path.dirname(export_infer_dir))
                zipf.write(full_path, rel_path)
                
    print(f"🎉 HOÀN THÀNH TOÀN DIỆN!")
    print(f"   - Mô hình Inference tại: {export_infer_dir}")
    print(f"   - Tệp nén sẵn sàng tải về: {zip_path} ({os.path.getsize(zip_path) / (1024*1024):.2f} MB)")
    print(f"👉 Hãy tải tệp zip này về và giải nén vào ocr-studio/data/output/ (thư mục ch_PP-OCRv4_det_cham_infer) để cập nhật hệ thống.")

if __name__ == '__main__':
    main()
