import os
import sys
import time
import json
import zipfile
import socket
import threading
import queue
from kaggle.api.kaggle_api_extended import KaggleApi

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

socket.setdefaulttimeout(60)

def get_api():
    os.environ["KAGGLE_USERNAME"] = "gustavnguyen"
    os.environ["KAGGLE_KEY"] = "6bf56db7e5c0fa7895d157167961d92b"
    api = KaggleApi()
    api.authenticate()
    return api

def stream_logs_worker(api, kernel_id, log_file_path, stop_event):
    """Lắng nghe luồng SSE log trực tiếp từ Kaggle và lưu file đồng thời phân tích sự kiện (tự động kết nối lại)"""
    while not stop_event.is_set():
        try:
            with open(log_file_path, 'a', encoding='utf-8') as f_log:
                for event in api.kernels_logs_stream(kernel_id):
                    if stop_event.is_set():
                        break
                    data = event.get('data', '')
                    if data:
                        f_log.write(data)
                        f_log.flush()
                        # In các dòng quan trọng
                        for line in data.splitlines():
                            line_str = line.strip()
                            if any(k in line_str for k in [
                                'epoch: [', 'best metric', 'cur metric',
                                'save best model', 'save model in',
                                'Detected GPU Count', 'Place(gpu', 'train dataloader has',
                                'Hoàn thành', 'HOÀN TẤT', 'Error', 'Exception'
                            ]) and not 'MB/s' in line_str and not 'â”' in line_str:
                                print(f"🔥 [KAGGLE LIVE] {line_str}")
        except Exception:
            pass
        if not stop_event.is_set():
            time.sleep(4)

def run_watchdog(user="gustavnguyen", slug="paddleocr-cham-finetune", output_dir="ocr-training/output/v24", poll_interval=30):
    kernel_id = f"{user}/{slug}"
    print("="*80)
    print(f"🛡️  ACTIVE WATCHDOG V24 - LIVE STREAM LOG TỪ KAGGLE DUAL T4x2")
    print(f"🎯 Kernel: {kernel_id}")
    print(f"⏱️  Cơ chế: SSE Live Stream + Heartbeat {poll_interval}s")
    print("="*80)
    
    api = get_api()
    os.makedirs(output_dir, exist_ok=True)
    live_log_path = os.path.join(output_dir, "live_training.log")
    
    # Khởi chạy luồng SSE Live Stream
    stop_stream_event = threading.Event()
    stream_thread = threading.Thread(
        target=stream_logs_worker,
        args=(api, kernel_id, live_log_path, stop_stream_event),
        daemon=True
    )
    stream_thread.start()
    print(f"📡 Đã kết nối kênh Live Log SSE. Ghi nhật ký thời gian thực tại: {live_log_path}\n")
    
    start_time = time.time()
    last_status = None
    fail_count = 0
    
    while True:
        try:
            res = api.kernels_status(kernel_id)
            if hasattr(res, 'status') and hasattr(res.status, 'name'):
                status = res.status.name.lower()
            elif hasattr(res, 'status'):
                status = str(res.status).split('.')[-1].lower()
            else:
                status = 'unknown'
                
            elapsed_sec = int(time.time() - start_time)
            elapsed_min = elapsed_sec // 60
            
            if status != last_status or elapsed_sec % 180 < poll_interval:
                print(f"⏱️  [Heartbeat {elapsed_min:02d}m{elapsed_sec%60:02d}s] Trạng thái: {status.upper()}")
                last_status = status
                
            # 1. THÀNH CÔNG
            if 'complete' in status:
                stop_stream_event.set()
                print("\n" + "="*80)
                print(f"🎉 KERNEL ĐÃ HOÀN TẤT THÀNH CÔNG TRONG {elapsed_min} PHÚT!")
                print("📥 Bắt đầu tải và đồng bộ mô hình (rec_cham_v24_best.zip)...")
                print("="*80)
                
                download_success = download_outputs(api, kernel_id, output_dir)
                if download_success:
                    print("✅ Tải và giải nén mô hình V24 hoàn tất!")
                    return True
                else:
                    print("❌ Lỗi khi tải hoặc giải nén kết quả từ Kaggle.")
                    return False
                    
            # 2. LỖI SỚM
            elif any(k in status for k in ['error', 'cancel', 'abort', 'fail']):
                stop_stream_event.set()
                print("\n" + "!"*80)
                print(f"🚨 BÁO ĐỘNG ĐỎ: KERNEL DỪNG VỚI TRẠNG THÁI LỖI: {status.upper()}!")
                print(f"⏱️  Thời gian chạy: {elapsed_min} phút {elapsed_sec%60} giây.")
                print("!"*80)
                return False
                
            fail_count = 0
            
        except Exception as e:
            fail_count += 1
            if fail_count >= 5:
                print(f"⚠️  Mất kết nối kiểm tra Kaggle API ({fail_count}/5): {e}")
                
        time.sleep(poll_interval)

def download_outputs(api, kernel_id, output_dir):
    try:
        socket.setdefaulttimeout(600)
        api.kernels_output(kernel_id, path=output_dir)
        print("✅ Đã tải file output từ Kaggle.")
        
        zip_files = [f for f in os.listdir(output_dir) if f.endswith('.zip')]
        if not zip_files:
            print("❌ Không tìm thấy file .zip trong output!")
            return False
            
        for zf in zip_files:
            zp = os.path.join(output_dir, zf)
            extract_folder = os.path.join(output_dir, zf.replace('.zip', ''))
            os.makedirs(extract_folder, exist_ok=True)
            print(f"📦 Đang giải nén {zf} ({os.path.getsize(zp) / (1024*1024):.2f} MB)...")
            with zipfile.ZipFile(zp, 'r') as zref:
                zref.extractall(extract_folder)
            print(f"✅ Giải nén xong vào: {extract_folder}")
            
            infer_candidates = [
                os.path.join(extract_folder, 'output', 'rec_cham_inference_v24'),
                os.path.join(extract_folder, 'rec_cham_inference_v24'),
                extract_folder
            ]
            for cand in infer_candidates:
                if os.path.exists(os.path.join(cand, 'inference.pdmodel')) or os.path.exists(os.path.join(cand, 'model.pdmodel')):
                    target_studio_dir = 'ocr-studio/data/output/rec_cham_inference_v24'
                    os.makedirs(target_studio_dir, exist_ok=True)
                    import shutil
                    for item in os.listdir(cand):
                        s_item = os.path.join(cand, item)
                        d_item = os.path.join(target_studio_dir, item)
                        if os.path.isfile(s_item):
                            shutil.copy2(s_item, d_item)
                    print(f"🚀 Đã đồng bộ mô hình inference V24 vào Studio: {target_studio_dir}")
                    break
        return True
    except Exception as e:
        print(f"❌ Lỗi giải nén: {e}")
        return False

if __name__ == '__main__':
    run_watchdog()
