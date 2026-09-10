#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Automated Pipeline for Cham-DBNet on Lightning AI (NVIDIA L40S 48GB).
- Machine: g6e.4xlarge (NVIDIA L40S 48GB Ada Lovelace - 3.54 cr/h)
- 2,000 training pages (det_train_label.txt) + 200 validation pages already prepared on disk.
- 150 epochs, DBNet PP-OCRv4 Mobile, batch size 64.
- Automatic monitoring, live loss streaming, model export, zip download, and guaranteed auto-stop.
"""

import os
import sys
import time
import base64
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
    log("🚀 KHỞI CHẠY PIPELINE HUẤN LUYỆN CHAM-DBNET TRÊN NVIDIA L40S 48GB")
    log("=================================================================")

    studio = Studio(name="cham-det-h100", teamspace="phucsd", org="phucsd-org")
    log(f"Studio hiện tại: Status={studio.status} | Machine={studio.machine}")

    try:
        target_machine = Machine.L40S
        log(f"\n🔥 [BƯỚC 1/5] Khởi động Studio với GPU NVIDIA L40S 48GB (Machine.L40S - 3.54 cr/h)...")
        if str(studio.status).lower() != "running":
            studio.start(machine=target_machine)
        else:
            if str(studio.machine) != str(target_machine):
                log(f"Đang chuyển phần cứng từ {studio.machine} sang {target_machine}...")
                studio.switch_machine(target_machine)
        
        log(f"✅ Studio đang chạy! Machine: {studio.machine} | Status: {studio.status}")

        log("\n🔍 [BƯỚC 2/5] Kiểm tra GPU và môi trường CUDA...")
        gpu_info = studio.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader").strip()
        log(f"🔥 GPU phát hiện: {gpu_info}")

        log("📦 Cài đặt/Kiểm tra PaddlePaddle GPU tương thích CUDA & VisualDL & NumPy...")
        paddle_check = studio.run("python3 -c \"import paddle; print('PADDLE_READY', paddle.__version__, paddle.device.cuda.device_count())\" || true").strip()
        if "PADDLE_READY" not in paddle_check or " 0" in paddle_check:
            log("Đang cài đặt paddlepaddle-gpu cu118...")
            studio.run("pip install -q paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu118/")
            paddle_check = studio.run("python3 -c \"import paddle; print('PADDLE_READY', paddle.__version__, paddle.device.cuda.device_count())\"").strip()
        log(f"✅ PaddlePaddle Status: {paddle_check}")

        log("📦 Cập nhật OpenCV, SciPy, RapidFuzz, VisualDL tương thích hoàn toàn NumPy 2.x...")
        studio.run("python3 -m pip install -q --upgrade 'opencv-python>=4.10' 'opencv-python-headless>=4.10' 'scipy>=1.13' rapidfuzz visualdl pyclipper shapely lmdb attrdict")

        log("🔧 Áp dụng Monkeypatch NumPy 2.x toàn diện qua sitecustomize và tools/...")
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
        sitecust_out = studio.run(sitecust_cmd).strip()
        if sitecust_out:
            log(f"   {sitecust_out}")

        studio.run("cd Cham-OCR/ocr-training/PaddleOCR && git checkout tools/train.py tools/export_model.py 2>/dev/null || true")
        patch_cmd = """python3 -c "
import pathlib
for fn in ['Cham-OCR/ocr-training/PaddleOCR/tools/train.py', 'Cham-OCR/ocr-training/PaddleOCR/tools/export_model.py']:
    p = pathlib.Path(fn)
    if p.exists():
        txt = p.read_text(encoding='utf-8')
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
        if 'np.sctypes' not in txt:
            lines = txt.splitlines(keepends=True)
            insert_idx = 0
            for idx, line in enumerate(lines):
                if line.strip().startswith('from __future__'):
                    insert_idx = idx + 1
            lines.insert(insert_idx, patch + '\\n')
            p.write_text(''.join(lines), encoding='utf-8')
            print(f'Đã patch toàn diện sau __future__: {fn}')
" """
        patch_out = studio.run(patch_cmd).strip()
        if patch_out:
            log(f"   {patch_out}")

        # Patch random_crop_data.py để tương thích hoàn toàn NumPy 2.x
        crop_patch_cmd = """python3 -c "
import pathlib
p = pathlib.Path('Cham-OCR/ocr-training/PaddleOCR/ppocr/data/imaug/random_crop_data.py')
if p.exists():
    txt = p.read_text(encoding='utf-8')
    if 'size=1' in txt:
        txt = txt.replace('xx = int(np.random.choice(axis, size=1))', 'xx = int(np.random.choice(axis))')
        p.write_text(txt, encoding='utf-8')
        print('Đã vá thành công random_crop_data.py (loại bỏ size=1 cho NumPy 2.x)')
    else:
        print('random_crop_data.py đã được vá')
" """
        crop_patch_out = studio.run(crop_patch_cmd).strip()
        if crop_patch_out:
            log(f"   {crop_patch_out}")

        log("🔍 Kiểm tra tính tương thích của toàn bộ thư viện...")
        verify_cmd = """python3 -c "
import numpy as np
if not hasattr(np, 'sctypes'): np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64], 'uint': [np.uint8, np.uint16, np.uint32, np.uint64], 'float': [np.float16, np.float32, np.float64], 'complex': [np.complex64, np.complex128], 'others': [bool, object, bytes, str]}
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'typeDict'): np.typeDict = {}
if not hasattr(np, 'long'): np.long = int
if not hasattr(np, 'ulong'): np.ulong = int
import cv2, imgaug, paddle
print('VERIFY_SUCCESS: numpy', np.__version__, 'cv2', cv2.__version__, 'paddle', paddle.__version__)
" """
        verify_check = studio.run(verify_cmd).strip()
        log(f"   {verify_check}")

        log("\n📂 [BƯỚC 3/5] Xác minh dữ liệu huấn luyện detector_v2...")
        data_stat = studio.run("wc -l Cham-OCR/ocr-training/PaddleOCR/data/detector_v2/det_train_label.txt Cham-OCR/ocr-training/PaddleOCR/data/detector_v2/det_val_label.txt 2>/dev/null || echo NOT_FOUND").strip()
        log(f"Số dòng dữ liệu:\n{data_stat}")
        if "NOT_FOUND" in data_stat:
            raise RuntimeError("Không tìm thấy bộ dữ liệu det_train_label.txt trên đĩa Studio!")

        sample_img_check = studio.run("python3 -c \"import pathlib, json; p = pathlib.Path('Cham-OCR/ocr-training/PaddleOCR/data/detector_v2/det_train_label.txt'); line = p.read_text(encoding='utf-8').splitlines()[0]; img_rel = line.split('\\t')[0]; img_path = pathlib.Path('Cham-OCR/ocr-training/PaddleOCR/data/detector_v2') / img_rel; print('IMG_SAMPLE_CHECK:', img_rel, img_path.exists())\"").strip()
        log(f"   {sample_img_check}")

        # Tạo thư mục output nếu chưa có
        studio.run("mkdir -p Cham-OCR/ocr-training/output")

        log("Đồng bộ tệp cấu hình ch_PP-OCRv4_det_h100.yml từ local vào PaddleOCR/configs/det/...")
        local_cfg_path = os.path.join(os.getcwd(), "ocr-training", "configs", "det", "ch_PP-OCRv4_det_h100.yml")
        with open(local_cfg_path, "r", encoding="utf-8") as f:
            cfg_content = f.read()

        sync_cmd = f"""python3 -c "
import pathlib
content = {repr(cfg_content)}
pathlib.Path('Cham-OCR/ocr-training/PaddleOCR/configs/det/ch_PP-OCRv4_det_h100.yml').write_text(content, encoding='utf-8')
pathlib.Path('Cham-OCR/ocr-training/configs/det/ch_PP-OCRv4_det_h100.yml').write_text(content, encoding='utf-8')
print('CONFIG_SYNCED_SUCCESSFULLY')
" """
        sync_res = studio.run(sync_cmd).strip()
        log(f"   {sync_res}")

        log("\n🚀 [BƯỚC 4/5] Khởi chạy huấn luyện Cham-DBNet (150 epochs, batch 64)...")
        studio.run("pkill -9 -f tools/train.py || true")
        studio.run("cd Cham-OCR/ocr-training/PaddleOCR && rm -f train.log && rm -rf output/ch_PP-OCRv4_det_cham_h100")
        time.sleep(2)

        launch_cmd = (
            "cd Cham-OCR/ocr-training/PaddleOCR && "
            "nohup python3 -m paddle.distributed.launch --gpus '0' tools/train.py -c configs/det/ch_PP-OCRv4_det_h100.yml > train.log 2>&1 &"
        )
        studio.run(launch_cmd)
        log("✅ Tiến trình huấn luyện đã kích hoạt chạy nền! Đang bắt đầu theo dõi tiến độ...")

        start_time = time.time()
        prev_log = ""

        while True:
            time.sleep(15)
            ps_cnt = studio.run("ps aux | grep tools/train.py | grep -v grep | wc -l").strip()
            is_alive = int(ps_cnt or "0") > 0

            try:
                latest_lines = studio.run("cd Cham-OCR/ocr-training/PaddleOCR && tail -n 8 train.log 2>/dev/null").strip()
                if latest_lines and latest_lines != prev_log:
                    prev_log = latest_lines
                    elapsed_min = (time.time() - start_time) / 60
                    log(f"⏱️ [{elapsed_min:.1f}m] Tiến độ log:")
                    matched = False
                    for l in latest_lines.splitlines():
                        if any(k in l for k in ["epoch:", "loss:", "hmean:", "save checkpoint", "Error", "Traceback", "MainIndicator", "epoch: [", "step:"]):
                            log(f"   {l}")
                            matched = True
                    if not matched and latest_lines.strip():
                        log(f"   {latest_lines.splitlines()[-1]}")
            except Exception as e:
                log(f"Log fetch notice: {e}")

            if not is_alive:
                final_log = studio.run("cd Cham-OCR/ocr-training/PaddleOCR && tail -n 40 train.log 2>/dev/null").strip()
                log(f"--- 40 DÒNG LOG CUỐI CÙNG ---\n{final_log}\n-----------------------------")
                
                # Kiểm tra xem có lỗi ngoại lệ khiến tiến trình dừng đột ngột không
                if "Traceback (most recent call last)" in final_log or "Exit code 1" in final_log or "SystemError:" in final_log:
                    raise RuntimeError("Tiến trình huấn luyện gặp lỗi ngoại lệ dừng đột ngột!")

                chk_check = studio.run("test -f Cham-OCR/ocr-training/PaddleOCR/output/ch_PP-OCRv4_det_cham_h100/latest.pdparams && echo CHK_FOUND || echo NO_CHK").strip()
                if "CHK_FOUND" not in chk_check:
                    raise RuntimeError("Huấn luyện dừng bất thường trước khi lưu được checkpoint!")
                
                log("\n🎉 Tiến trình huấn luyện đã kết thúc thành công!")
                break

        log("\n📦 [BƯỚC 5/5] Xuất mô hình Inference & Tải về máy local...")
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

        # Tự động giải nén model vào ocr-training/cham_dbnet_v1_infer
        import zipfile
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
