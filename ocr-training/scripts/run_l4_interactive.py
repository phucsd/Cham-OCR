#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Optimized Pipeline on NVIDIA L4 (1.68 cr/h - g6.4xlarge).
- Cost: 1.68 cr/h (53% cheaper than AWS L40S, cheaper than Nebius 2.14 cr/h)
- In-Place Interactive Pre-Flight Verification:
    1. Train DataLoader (2 batches)
    2. Eval DataLoader (5 batches, batch_size=1)
    3. Forward pass + DBPostProcess + DetMetric
- Only launches 150 epochs training after 100% Pre-flight test passed.
- Model export + local download + guaranteed auto-stop when complete.
"""

import os
import sys
import time
import base64
import zipfile
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
    log("🚀 OPTION A: KHỞI CHẠY PIPELINE TRÊN NVIDIA L4 (1.68 cr/h - 24GB)")
    log("=================================================================")

    studio = Studio(name="cham-det-h100", teamspace="phucsd", org="phucsd-org")
    log(f"Studio hiện tại: Status={studio.status} | Machine={studio.machine}")

    target_machine = Machine.L4
    log(f"\n🔥 [BƯỚC 1/6] Khởi động Studio với NVIDIA L4 (Machine.L4 - 1.68 cr/h)...")
    if str(studio.status).lower() != "running":
        studio.start(machine=target_machine)
    else:
        if str(studio.machine) != str(target_machine):
            log(f"Chuyển máy từ {studio.machine} sang {target_machine}...")
            studio.switch_machine(target_machine)

    log(f"✅ Studio đang chạy! Machine: {studio.machine} | Status: {studio.status}")

    # Giai đoạn Pre-flight: GIỮ MÁY MỞ nếu có lỗi để sửa ngay tại chỗ
    log("\n🔍 [BƯỚC 2/6] Kiểm tra phần cứng GPU L4...")
    gpu_info = studio.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader").strip()
    log(f"🔥 GPU: {gpu_info}")

    log("📦 Kiểm tra PaddlePaddle GPU 3.3.1...")
    paddle_check = studio.run("python3 -c \"import paddle; print('PADDLE_READY', paddle.__version__, paddle.device.cuda.device_count())\" || true").strip()
    if "PADDLE_READY" not in paddle_check or " 0" in paddle_check:
        log("Cài đặt paddlepaddle-gpu cu118...")
        studio.run("pip install -q paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu118/")
        paddle_check = studio.run("python3 -c \"import paddle; print('PADDLE_READY', paddle.__version__, paddle.device.cuda.device_count())\"").strip()
    log(f"✅ PaddlePaddle: {paddle_check}")

    log("📦 Cập nhật gói phụ thuộc tương thích NumPy 2.x...")
    studio.run("python3 -m pip install -q --upgrade 'opencv-python>=4.10' 'opencv-python-headless>=4.10' 'scipy>=1.13' rapidfuzz visualdl pyclipper shapely lmdb attrdict")

    log("🔧 Thiết lập sitecustomize.py monkeypatch...")
    sitecust_cmd = """python3 -c "
import site, pathlib
patch = '''import numpy as np
if not hasattr(np, 'sctypes'): np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64], 'uint': [np.uint8, np.uint16, np.uint32, np.uint64], 'float': [np.float16, np.float32, np.float64], 'complex': [np.complex64, np.complex128], 'others': [bool, object, bytes, str]}
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'typeDict'): np.typeDict = {}
if not hasattr(np, 'long'): np.long = int
if not hasattr(np, 'ulong'): np.ulong = int
if not hasattr(np, 'unicode_'): np.unicode_ = str
if not hasattr(np, 'string_'): np.string_ = bytes
if not hasattr(np, 'object_'): np.object_ = object
if not hasattr(np, 'complex_'): np.complex_ = complex
'''
sp_list = site.getsitepackages()
for sp in sp_list:
    try:
        (pathlib.Path(sp) / 'sitecustomize.py').write_text(patch, encoding='utf-8')
        print(f'Wrote sitecustomize.py to {sp}')
    except Exception as e:
        pass
" """
    studio.run(sitecust_cmd)

    log("🔧 Vá random_crop_data.py loại bỏ size=1 cho NumPy 2.x...")
    crop_patch_cmd = """python3 -c "
import pathlib
p = pathlib.Path('Cham-OCR/ocr-training/PaddleOCR/ppocr/data/imaug/random_crop_data.py')
if p.exists():
    txt = p.read_text(encoding='utf-8')
    if 'size=1' in txt:
        txt = txt.replace('xx = int(np.random.choice(axis, size=1))', 'xx = int(np.random.choice(axis))')
        p.write_text(txt, encoding='utf-8')
        print('Đã vá random_crop_data.py')
    else:
        print('random_crop_data.py đã được vá trước đó')
" """
    log(f"   {studio.run(crop_patch_cmd).strip()}")

    log("\n📄 [BƯỚC 3/6] Đồng bộ cấu hình ch_PP-OCRv4_det_h100.yml trực tiếp từ máy local...")
    local_cfg_path = os.path.join(os.getcwd(), "ocr-training", "configs", "det", "ch_PP-OCRv4_det_h100.yml")
    with open(local_cfg_path, "rb") as f:
        b64_cfg = base64.b64encode(f.read()).decode('ascii')
    studio.run(f"echo '{b64_cfg}' | base64 -d > Cham-OCR/ocr-training/PaddleOCR/configs/det/ch_PP-OCRv4_det_h100.yml")
    studio.run(f"cp Cham-OCR/ocr-training/PaddleOCR/configs/det/ch_PP-OCRv4_det_h100.yml Cham-OCR/ocr-training/configs/det/ch_PP-OCRv4_det_h100.yml")
    log("   CONFIG_SYNCED_SUCCESSFULLY via base64 pipe")

    log("\n🧪 [BƯỚC 4/6] Chạy kiểm thử tại chỗ PRE-FLIGHT TEST (Train + Eval + Metric)...")
    preflight_script = """cd Cham-OCR/ocr-training/PaddleOCR && python3 -c "
import logging
logger = logging.getLogger('preflight')
import yaml, paddle, time
from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric

with open('configs/det/ch_PP-OCRv4_det_h100.yml', 'r') as f:
    config = yaml.safe_load(f)

print('1. Kiểm tra Train DataLoader (batch size 32)...')
train_loader = build_dataloader(config, 'Train', paddle.device.get_device(), logger)
sample_train_batch = None
for idx, batch in enumerate(train_loader):
    print(f'   Train Batch {idx}: image shape = {batch[0].shape}')
    sample_train_batch = batch
    if idx >= 1:
        break

print('2. Kiểm tra Eval DataLoader (batch size 1, dynamic shapes)...')
eval_loader = build_dataloader(config, 'Eval', paddle.device.get_device(), logger)
eval_batches = []
for idx, batch in enumerate(eval_loader):
    print(f'   Eval Batch {idx}: image shape = {batch[0].shape}, polys count = {len(batch[2][0])}')
    eval_batches.append(batch)
    if idx >= 4:
        break

print('3. Kiểm tra Model forward pass + PostProcess + Metric...')
model = build_model(config['Architecture'])
model.eval()
post_process_class = build_post_process(config['PostProcess'])
eval_class = build_metric(config['Metric'])

with paddle.no_grad():
    for batch in eval_batches:
        images = batch[0]
        preds = model(images)
        post_res = post_process_class(preds, batch[1])
        eval_class(post_res, batch)

metric = eval_class.get_metric()
print(f'   Evaluation metric calculated successfully: {metric}')

print('=== PRE_FLIGHT_PASSED_100% ===')
" """
    preflight_out = studio.run(preflight_script).strip()
    log(f"Kết quả Pre-flight Test:\n{preflight_out}")

    if "=== PRE_FLIGHT_PASSED_100% ===" not in preflight_out:
        log("❌ Pre-flight test chưa hoàn thành trọn vẹn! Máy vẫn được GIỮ MỞ để kiểm tra.")
        return

    log("\n🎉 PRE-FLIGHT TEST ĐÃ ĐẠT 100%! BẮT ĐẦU CHẠY HUẤN LUYỆN CHÍNH THỨC 150 EPOCHS...")

    try:
        log("\n🚀 [BƯỚC 5/6] Khởi chạy huấn luyện Cham-DBNet (150 epochs, batch 32, single-GPU L4)...")
        studio.run("pkill -9 -f train.py 2>/dev/null || true")
        studio.run("pkill -9 -f python 2>/dev/null || true")
        time.sleep(2)
        studio.run("cd Cham-OCR/ocr-training/PaddleOCR && rm -f train.log && rm -rf log output/ch_PP-OCRv4_det_cham_h100")
        studio.run("mkdir -p Cham-OCR/ocr-training/output")
        time.sleep(1)

        vram_check = studio.run("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader").strip()
        log(f"   VRAM trước khi khởi chạy: {vram_check}")

        launch_cmd = (
            "cd Cham-OCR/ocr-training/PaddleOCR && "
            "nohup python3 -u tools/train.py -c configs/det/ch_PP-OCRv4_det_h100.yml > train.log 2>&1 &"
        )
        studio.run(launch_cmd)
        time.sleep(5)
        log("✅ Tiến trình huấn luyện đã kích hoạt chạy nền! Đang bắt đầu theo dõi tiến độ...")

        start_time = time.time()
        prev_log = ""

        while True:
            time.sleep(20)
            ps_cnt = studio.run("ps aux | grep '[t]ools/train.py' | wc -l").strip()
            is_alive = int(ps_cnt or "0") > 0

            try:
                latest_lines = studio.run(
                    "cd Cham-OCR/ocr-training/PaddleOCR && tail -n 15 train.log"
                ).strip()
                if latest_lines and latest_lines != prev_log:
                    prev_log = latest_lines
                    elapsed_min = (time.time() - start_time) / 60
                    log(f"⏱️ [{elapsed_min:.1f}m] Tiến độ log:")
                    matched = False
                    for l in latest_lines.splitlines():
                        if any(k in l for k in ["epoch:", "loss:", "hmean:", "save checkpoint", "Error", "Traceback", "MainIndicator", "step:", "eval model:", "PPLCNet", "download"]):
                            log(f"   {l}")
                            matched = True
                    if not matched and latest_lines.strip():
                        log(f"   {latest_lines.splitlines()[-1]}")
            except Exception as e:
                log(f"Log notice: {e}")

            if not is_alive:
                final_log = studio.run(
                    "cd Cham-OCR/ocr-training/PaddleOCR && tail -n 50 train.log"
                ).strip()
                log(f"--- 50 DÒNG LOG CUỐI CÙNG ---\n{final_log}\n-----------------------------")

                if "Traceback (most recent call last)" in final_log or "Exit code 1" in final_log or "SystemError:" in final_log or "FatalError:" in final_log:
                    raise RuntimeError("Tiến trình huấn luyện gặp lỗi ngoại lệ dừng đột ngột!")

                chk_check = studio.run("test -f Cham-OCR/ocr-training/PaddleOCR/output/ch_PP-OCRv4_det_cham_h100/latest.pdparams && echo CHK_FOUND || echo NO_CHK").strip()
                if "CHK_FOUND" not in chk_check:
                    raise RuntimeError("Huấn luyện dừng trước khi lưu checkpoint!")

                log("\n🎉 TIẾN TRÌNH HUẤN LUYỆN 150 EPOCHS ĐÃ KẾT THÚC HOÀN TOÀN THÀNH CÔNG!")
                break


        log("\n📦 [BƯỚC 6/6] Xuất mô hình Inference & Tải về máy local...")
        export_script = """
cd Cham-OCR/ocr-training/PaddleOCR
BEST_MODEL="output/ch_PP-OCRv4_det_cham_h100/best_accuracy"
if [ ! -f "${BEST_MODEL}.pdparams" ]; then
    BEST_MODEL="output/ch_PP-OCRv4_det_cham_h100/latest"
fi
echo "Using checkpoint: $BEST_MODEL"
mkdir -p ../output
python3 tools/export_model.py \
    -c configs/det/ch_PP-OCRv4_det_h100.yml \
    -o Global.pretrained_model=${BEST_MODEL} \
       Global.save_inference_dir=../output/ch_PP-OCRv4_det_cham_infer

cd ../output
rm -f ../cham_dbnet_v1_infer.zip
python3 -c "import shutil; shutil.make_archive('../cham_dbnet_v1_infer', 'zip', '.', 'ch_PP-OCRv4_det_cham_infer')"
ls -lh ../cham_dbnet_v1_infer.zip
"""
        export_out = studio.run(export_script).strip()
        log(f"Export log:\n{export_out}")

        log("⬇️ Đang tải tệp cham_dbnet_v1_infer.zip về thư mục ocr-training/ ...")
        local_dest = os.path.join(os.getcwd(), "ocr-training", "cham_dbnet_v1_infer.zip")
        downloaded = False
        try:
            studio.download_file("Cham-OCR/ocr-training/cham_dbnet_v1_infer.zip", local_dest)
            if os.path.exists(local_dest) and os.path.getsize(local_dest) > 1000:
                downloaded = True
                log("   Đã tải thành công qua native studio.download_file!")
        except Exception as dl_err:
            log(f"   Native download gặp lỗi ({dl_err}), chuyển sang Base64 fallback...")

        if not downloaded:
            remote_b64 = studio.run(
                "python3 -c \"import base64, pathlib; p = pathlib.Path('Cham-OCR/ocr-training/cham_dbnet_v1_infer.zip'); print(base64.b64encode(p.read_bytes()).decode('ascii'))\""
            ).strip()
            with open(local_dest, "wb") as f:
                f.write(base64.b64decode(remote_b64))

        local_size_mb = os.path.getsize(local_dest) / (1024 * 1024)
        log(f"🎉 ĐÃ TẢI THÀNH CÔNG: {local_dest} ({local_size_mb:.2f} MB)")

        infer_extract_dir = os.path.join(os.getcwd(), "ocr-training", "cham_dbnet_v1_infer")
        os.makedirs(infer_extract_dir, exist_ok=True)
        with zipfile.ZipFile(local_dest, "r") as zf:
            zf.extractall(infer_extract_dir)
        log(f"📂 Đã giải nén mô hình suy luận vào: {infer_extract_dir}")

    except Exception as exc:
        log(f"❌ XẢY RA LỖI: {exc}")
        raise exc

    finally:
        log("\n🛑 [AN TOÀN] Đang dừng Studio để bảo vệ số dư tài khoản...")
        try:
            if str(studio.status).lower() == "running":
                studio.stop()
                log(f"✅ Studio đã DỪNG (Status: {studio.status})! 0 credit bị tiêu tốn thêm.")
            else:
                log(f"✅ Studio hiện tại đã ở trạng thái {studio.status}, an toàn.")
        except Exception as stop_err:
            log(f"Cảnh báo khi dừng Studio: {stop_err}")

    log("\n🎊 TOÀN BỘ QUY TRÌNH ĐÃ HOÀN TẤT THÀNH CÔNG RỰC RỠ!")

if __name__ == "__main__":
    main()
