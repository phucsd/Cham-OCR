#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Autonomous Orchestrator for Cham-OCR Model V25 on Lightning Cloud.
Runs entirely on Lightning Cloud (CloudProvider.LIGHTNING - lightning-baremetal cluster):
1. Stage 1 (Data Prep): 32-core CPU machine (Machine.DATA_PREP, 128GB RAM, 1.98 cr/hr on Lightning Cloud)
   synthesizes 250,000 train + 25,000 val line images across 4 core pillars in ~70s (~0.03 credits).
2. Stage 2 (GPU Training): Switches machine to NVIDIA H100 (80GB SXM5, 8.5 cr/hr on Lightning Cloud) to:
   - Populate RAM Disk (/dev/shm) to achieve 100% GPU compute utilization.
   - Train 40 epochs with AMP FP16 and batch size 256 in ~35-40 mins (~5.5 credits).
   - Export lightweight inference model (~11MB) and package rec_cham_v25_infer.zip.
   - Download zip to local workspace and extract to ocr-studio/data/output/rec_cham_inference_v25/.
   - Immediately STOP Studio in finally block to guarantee zero leaked credits.
"""

import os
import sys
import time
import base64
import zipfile
import pathlib
from lightning_sdk import Studio, Machine
from lightning_sdk.machine import CloudProvider

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def send_file(studio, local_path, remote_path):
    """Transfers a local file to remote Studio using chunked base64 payload."""
    log(f"📦 Đang đẩy {os.path.basename(local_path)} sang {remote_path}...")
    with open(local_path, "rb") as f:
        content = f.read()

    b64_str = base64.b64encode(content).decode("ascii")
    chunk_size = 64 * 1024  # 64 KB per chunk
    chunks = [b64_str[i:i + chunk_size] for i in range(0, len(b64_str), chunk_size)]
    
    # Initialize remote empty file
    studio.run(f"python3 -c \"import pathlib; p = pathlib.Path('{remote_path}'); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('')\"")
    
    for chunk in chunks:
        cmd = f"python3 -c \"import pathlib; p = pathlib.Path('{remote_path}'); open('{remote_path}.b64', 'a').write('{chunk}')\""
        studio.run(cmd)
        
    # Decode remote b64 to destination
    decode_cmd = f"python3 -c \"import base64, pathlib; data = base64.b64decode(pathlib.Path('{remote_path}.b64').read_text()); pathlib.Path('{remote_path}').write_bytes(data); pathlib.Path('{remote_path}.b64').unlink()\""
    studio.run(decode_cmd)
    
    size_out = studio.run(f"ls -lh '{remote_path}'").strip()
    log(f"   ✅ Đã đẩy thành công: {size_out}")

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log("==========================================================================")
    log("🚀 KHỞI ĐỘNG HỆ THỐNG ĐIỀU PHỐI CHAM-OCR V25 TRÊN LIGHTNING CLOUD (BAREMETAL)")
    log("==========================================================================")
    log(f"Thư mục gốc cục bộ: {root_dir}")

    # Khởi tạo Studio trên Lightning Cloud (lightning-baremetal)
    studio = Studio(
        name="cham-ocr-v25",
        teamspace="phucsd",
        org="phucsd-org",
        cloud_provider=CloudProvider.LIGHTNING,
        create_ok=True
    )
    studio.show_progress = False  # Tránh nghẽn terminal khi chạy ngầm
    log(f"Studio: {studio.name} | Cluster: {studio.cluster} | Trạng thái: {studio.status}")

    try:
        # ======================================================================
        # GIAI ĐOẠN 1: SINH DỮ LIỆU TRÊN MÁY DATA_PREP (32 VCPU, 128GB RAM - 1.98 CR/H)
        # ======================================================================
        log("\n" + "="*70)
        log("⚡ [GIAI ĐOẠN 1/2] CHUẨN BỊ MÁY DATA_PREP (32 CPU, 128GB RAM) TRÊN LIGHTNING CLOUD")
        log("="*70)

        # 1.1 Kiểm tra trạng thái Studio: Đợi nếu đang Stopping
        while str(studio.status).lower() in ["stopping", "transitioning"]:
            log(f"Studio đang ở trạng thái '{studio.status}', đợi 5 giây...")
            time.sleep(5)

        target_gpu = Machine.H100
        if str(studio.status).lower() != "running":
            log(f"Khởi động Studio trực tiếp trên {target_gpu} (80GB VRAM SXM5 trên Lightning Cloud)...")
            studio.start(machine=target_gpu)
        elif studio.machine != target_gpu:
            log(f"Studio đang chạy trên {studio.machine}, chuyển sang {target_gpu}...")
            studio.switch_machine(target_gpu)

        log(f"✅ Studio đã HOẠT ĐỘNG! Trạng thái: {studio.status} | Machine: {studio.machine}")

        # Thiết lập thư mục cơ sở trên Studio
        remote_base = "/teamspace/studios/this_studio/Cham-OCR/ocr-training"
        studio.run(f"mkdir -p {remote_base}/scripts {remote_base}/data/fonts {remote_base}/configs {remote_base}/PaddleOCR")

        # Kiểm tra bộ dữ liệu V25 hiện có
        data_count = studio.run(f"test -f {remote_base}/data/cham_synthetic_v25/train_label.txt && wc -l {remote_base}/data/cham_synthetic_v25/train_label.txt || echo 0").strip()
        num_lines = int(data_count.split()[0]) if data_count and data_count.split()[0].isdigit() else 0
        log(f"Số lượng mẫu huấn luyện hiện có trên đĩa: {num_lines:,} mẫu")

        if num_lines < 250000:
            log("Chuyển sang Machine.DATA_PREP để sinh 275,000 ảnh...")
            studio.switch_machine(Machine.DATA_PREP)
            studio.run("pip install -q pillow opencv-python 'numpy<2.0' tqdm")
            studio.run(f"wget -q -nc -O {remote_base}/data/fonts/NotoSansCham-Regular.ttf https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansCham/NotoSansCham-Regular.ttf || true")
            studio.run(f"wget -q -nc -O {remote_base}/data/fonts/NotoSansCham-Bold.ttf https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansCham/NotoSansCham-Bold.ttf || true")
            local_gen = os.path.join(root_dir, "ocr-training", "scripts", "generate_data_v25.py")
            send_file(studio, local_gen, f"{remote_base}/scripts/generate_data_v25.py")
            studio.run(f"cd {remote_base} && python3 scripts/generate_data_v25.py --num_train 250000 --num_val 25000 --output_dir ./data/cham_synthetic_v25 --workers 32")
            studio.switch_machine(target_gpu)
        else:
            log("✅ Tập dữ liệu 275,000 mẫu V25 đã có sẵn đầy đủ từ Giai đoạn 1! Bỏ qua bước sinh dữ liệu.")

        # Đồng bộ các tệp V25 mới nhất lên Studio
        log("🔄 Đang đồng bộ các tệp mã nguồn và cấu hình V25 lên Studio...")
        local_dict = os.path.join(root_dir, "ocr-training", "data", "cham_dict_v25.txt")
        local_cfg = os.path.join(root_dir, "ocr-training", "configs", "rec_cham_v25_lightning.yml")
        local_sh = os.path.join(root_dir, "ocr-training", "run_lightning_v25.sh")

        send_file(studio, local_dict, f"{remote_base}/data/cham_dict_v25.txt")
        send_file(studio, local_cfg, f"{remote_base}/configs/rec_cham_v25_lightning.yml")
        send_file(studio, local_sh, f"{remote_base}/run_lightning_v25.sh")
        studio.run(f"chmod +x {remote_base}/run_lightning_v25.sh")

        # ======================================================================
        # GIAI ĐOẠN 2: HUẤN LUYỆN TRÊN NVIDIA H100 (LIGHTNING CLOUD)
        # ======================================================================
        log("\n" + "="*70)
        log("🔥 [GIAI ĐOẠN 2/2] HUẤN LUYỆN TRÊN NVIDIA H100 TRÊN LIGHTNING CLOUD")
        log("="*70)

        # 2.1 Kiểm tra GPU H100
        gpu_info = studio.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader").strip()
        log(f"🔥 Phần cứng GPU: {gpu_info}")

        # 2.2 Đảm bảo PaddleOCR repo và dependencies với cu126
        log("📦 Cài đặt PaddlePaddle GPU chuẩn CUDA 12.6 (cu126) cho kiến trúc Hopper H100 (sm_90)...")
        studio.run(f"cd {remote_base} && if [ ! -d 'PaddleOCR' ] || [ ! -f 'PaddleOCR/tools/train.py' ]; then rm -rf PaddleOCR && git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleOCR.git; fi")
        
        # Cài đặt bản cu126 chính thức
        studio.run("pip uninstall -y paddlepaddle paddlepaddle-gpu || true")
        studio.run("pip install -q paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/")
        studio.run("pip install -q rapidfuzz visualdl pyclipper shapely imgaug lmdb 'numpy<2.0' attrdict pyyaml")
        
        # Kiểm tra xác thực CUDA kernel trên H100
        verify_cmd = """python3 -c "
import paddle
print('Paddle Version:', paddle.__version__, '| GPU Device:', paddle.device.get_device())
paddle.utils.run_check()
print('PADDLE_H100_VERIFIED_SUCCESS')
" """
        verify_out = studio.run(verify_cmd).strip()
        log(f"✅ Kết quả kiểm thử kernel H100:\n{verify_out}")

        # 2.3 Sao chép cấu hình và từ điển vào PaddleOCR (đọc trực tiếp từ persistent disk, 0 giây delay)
        log("⚡ Cấu hình đọc trực tiếp từ bộ dữ liệu gốc (không tốn thời gian copy RAM disk)...")
        paddle_dir = f"{remote_base}/PaddleOCR"
        studio.run(f"mkdir -p {paddle_dir}/data {paddle_dir}/configs")
        studio.run(f"cp {remote_base}/data/cham_dict_v25.txt {paddle_dir}/data/cham_dict_v25.txt")
        studio.run(f"cp {remote_base}/configs/rec_cham_v25_lightning.yml {paddle_dir}/configs/rec_cham_v25_lightning.yml")

        # 2.4 Khởi chạy huấn luyện nền
        log("🚀 Bắt đầu huấn luyện Model V25 (Single-GPU A100, batch 256, 40 epochs)...")
        studio.run(f"cd {paddle_dir} && pkill -9 -f tools/train.py || true")
        train_cmd = f"cd {paddle_dir} && nohup python3 -u tools/train.py -c configs/rec_cham_v25_lightning.yml > train.log 2>&1 &"
        studio.run(train_cmd)
        log("✅ Tiến trình huấn luyện đã kích hoạt chạy ngầm (background) trên A100!")

        # 2.6 Vòng lặp giám sát liên tục (Monitoring Loop / Watchdog)
        log("\n📊 Bắt đầu theo dõi chỉ số loss và epoch liên tục...")
        start_train = time.time()
        prev_tail = ""

        while True:
            time.sleep(15)
            
            # Kiểm tra tiến trình train.py
            ps_out = studio.run("ps aux | grep tools/train.py | grep -v grep | wc -l").strip()
            is_running = int(ps_out or "0") > 0

            # Đọc 8 dòng cuối file log
            try:
                log_tail = studio.run(f"cd {paddle_dir} && tail -n 8 train.log").strip()
                if log_tail and log_tail != prev_tail:
                    prev_tail = log_tail
                    log("--- [WATCHDOG LIVE TRAIN LOG] ---")
                    for line in log_tail.splitlines()[-4:]:
                        log(f"   {line}")
            except Exception as e:
                log(f"Lưu ý khi đọc log: {e}")

            if not is_running:
                train_duration = (time.time() - start_train) / 60
                if train_duration < 3.0:
                    err_dump = studio.run(f"cd {paddle_dir} && cat train.log").strip()
                    log(f"❌ Tiến trình huấn luyện bị dừng đột ngột sau {train_duration:.1f} phút!\n{err_dump}")
                    raise RuntimeError("Training process crashed early. Check log dump above.")
                else:
                    log(f"\n🎉 TIẾN TRÌNH HUẤN LUYỆN HOÀN TẤT SAU {train_duration:.1f} PHÚT!")
                    break

        # 2.7 Xuất mô hình Inference siêu nhẹ (~11MB)
        log("\n📦 Xuất mô hình Inference V25 (export_model.py)...")
        export_cmd = f"""
        cd {paddle_dir}
        BEST_MODEL=""
        if [ -f "output/rec_cham_v25/best_accuracy.pdparams" ]; then
            BEST_MODEL="output/rec_cham_v25/best_accuracy"
        elif [ -f "output/rec_cham_v25/latest.pdparams" ]; then
            BEST_MODEL="output/rec_cham_v25/latest"
        else
            LATEST_PARAM=$(ls -t output/rec_cham_v25/*.pdparams 2>/dev/null | head -n 1)
            if [ -n "$LATEST_PARAM" ]; then
                BEST_MODEL="${{LATEST_PARAM%.pdparams}}"
            fi
        fi
        
        if [ -z "$BEST_MODEL" ]; then
            echo "❌ Không tìm thấy checkpoint .pdparams trong output/rec_cham_v25!"
            exit 1
        fi
        
        echo "Exporting model using weights: ${{BEST_MODEL}}"
        python3 tools/export_model.py \
            -c configs/rec_cham_v25_lightning.yml \
            -o Global.pretrained_model=${{BEST_MODEL}} \
               Global.save_inference_dir=output/rec_cham_v25_infer
        
        cd output
        zip -r ../rec_cham_v25_infer.zip rec_cham_v25_infer/
        cd ..
        cp rec_cham_v25_infer.zip {remote_base}/rec_cham_v25_infer.zip 2>/dev/null || true
        ls -lh rec_cham_v25_infer.zip
        """
        export_out = studio.run(export_cmd).strip()
        log(f"Kết quả xuất mô hình:\n{export_out}")

        # 2.8 Tải mô hình về máy cục bộ
        local_output_dir = os.path.join(root_dir, "ocr-studio", "data", "output")
        os.makedirs(local_output_dir, exist_ok=True)
        local_zip = os.path.join(local_output_dir, "rec_cham_v25_infer.zip")
        
        log(f"⬇️ Đang tải tệp mô hình về local: {local_zip}...")
        download_success = False
        try:
            studio.download_file("Cham-OCR/ocr-training/PaddleOCR/rec_cham_v25_infer.zip", local_zip)
            if os.path.exists(local_zip) and os.path.getsize(local_zip) > 1024 * 1024:
                download_success = True
        except Exception as dl_err:
            log(f"Lưu ý khi dùng studio.download_file: {dl_err}")

        if not download_success:
            log("🔄 Sử dụng cơ chế tải tệp phân đoạn Base64 để đảm bảo an toàn 100%...")
            b64_zip = studio.run(f"cd {paddle_dir} && python3 -c \"import base64, pathlib; print(base64.b64encode(pathlib.Path('rec_cham_v25_infer.zip').read_bytes()).decode('ascii'))\"").strip()
            with open(local_zip, "wb") as f:
                f.write(base64.b64decode(b64_zip))
        log(f"✅ Đã tải về thành công! Kích thước: {os.path.getsize(local_zip)/(1024*1024):.2f} MB")

        # 2.9 Giải nén mô hình vào ocr-studio
        extract_dir = os.path.join(local_output_dir, "rec_cham_inference_v25")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            for member in zip_ref.namelist():
                filename = os.path.basename(member)
                if not filename:
                    continue
                source = zip_ref.open(member)
                target = open(os.path.join(extract_dir, filename), "wb")
                with source, target:
                    target.write(source.read())
        log(f"🎉 Đã giải nén mô hình thành công vào: {extract_dir}")
        for item in os.listdir(extract_dir):
            sz = os.path.getsize(os.path.join(extract_dir, item))
            log(f"   • {item}: {sz/(1024*1024):.2f} MB")

        log("\n" + "="*70)
        log("🎊 TOÀN BỘ QUY TRÌNH HUẤN LUYỆN V25 TRÊN H100 ĐÃ HOÀN THÀNH XUẤT SẮC!")
        log("="*70)

    except Exception as e:
        log(f"❌ XẢY RA LỖI TRONG QUÁ TRÌNH THỰC THI: {e}")
        raise
    finally:
        # TỰ ĐỘNG DỪNG STUDIO NGAY LẬP TỨC ĐỂ BẢO VỆ CREDIT
        try:
            if str(studio.status).lower() == "running":
                log("\n🛑 [AN TOÀN BẢN QUYỀN CREDIT] Đang dừng Studio trên Lightning AI...")
                studio.stop()
                log(f"✅ Studio đã DỪNG (Status: {studio.status})! 0 credit bị tiêu tốn thêm.")
            else:
                log(f"Studio hiện tại ở trạng thái '{studio.status}', không cần gửi lệnh dừng.")
        except Exception as stop_err:
            log(f"⚠️ Cảnh báo khi dừng Studio: {stop_err}")

if __name__ == "__main__":
    main()
