#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Điều phối tự động bài Benchmark Cham-OCR Recognition V24 trên Kaggle Dual GPU Tesla T4x2:
1. Đóng gói Kaggle Notebook
2. Đẩy lên Kaggle với bộ tăng tốc bắt buộc NvidiaTeslaT4 (T4x2 phân tán)
3. Giám sát thời gian thực luồng log SSE và trạng thái Kernel
4. Tự động tải file benchmark_t4_results.zip và hiển thị bảng kết quả phân tích.
"""

import os
import sys
import time
import json
import zipfile
import socket
import threading
import subprocess

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thiết lập timeout cho socket
socket.setdefaulttimeout(60)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from kaggle_auth import init_kaggle_auth
init_kaggle_auth()

from kaggle.api.kaggle_api_extended import KaggleApi

KERNEL_USER = "gustavnguyen"
KERNEL_SLUG = "cham-ocr-t4-benchmark"
KERNEL_ID = f"{KERNEL_USER}/{KERNEL_SLUG}"
SUBMISSION_DIR = os.path.abspath("ocr-training/output/kaggle_benchmark_submission")
OUTPUT_DIR = os.path.abspath("ocr-training/benchmark_results/kaggle_t4")

def get_api():
    api = KaggleApi()
    api.authenticate()
    return api

def stream_logs_worker(api, kernel_id, log_file_path, stop_event):
    """Lắng nghe luồng SSE log trực tiếp từ Kaggle và in các sự kiện quan trọng"""
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
                        for line in data.splitlines():
                            line_str = line.strip()
                            if any(k in line_str for k in [
                                '🔥 BẮT ĐẦU:', 'HOÀN THÀNH:', 'THẤT BẠI', 'Steady Step',
                                'BẢNG TỔNG KẾT', 'CẤU HÌNH TỐI ƯU NHẤT', 'Detected GPU Count',
                                'Place(gpu', 'Hoàn thành sinh dữ liệu', 'OOM', 'Exception', 'Error',
                                'Max Steady-State', 'Thời gian hoàn thành'
                            ]) or ('|' in line_str and any(t in line_str for t in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'Trial Name', 'Thí nghiệm'])):
                                print(f"⚡ [KAGGLE T4x2] {line_str}", flush=True)
        except Exception:
            pass
        if not stop_event.is_set():
            time.sleep(3)

def main():
    skip_push = "--monitor-only" in sys.argv or "--skip-push" in sys.argv
    print("="*80)
    print("🚀 ĐIỀU PHỐI BENCHMARK DUAL GPU TESLA T4x2 TRÊN KAGGLE")
    print(f"🎯 Kernel Target: {KERNEL_ID}")
    print(f"⚙️  Accelerator: NvidiaTeslaT4 (Dual GPU T4x2)")
    print("="*80)
    
    api = get_api()
    print("✅ Xác thực thành công tài khoản:", os.environ["KAGGLE_USERNAME"])

    if not skip_push:
        # 1. Sinh notebook mới nhất
        print("\n📦 Bước 1: Chuẩn bị Kaggle Benchmark Notebook...")
        import prepare_benchmark_notebook
        prepare_benchmark_notebook.main()
        
        # 2. Đẩy Kernel lên Kaggle
        print("\n📤 Bước 2: Đẩy Notebook lên Kaggle với cờ --accelerator NvidiaTeslaT4...")
        push_cmd = f"kaggle kernels push -p \"{SUBMISSION_DIR}\" --accelerator NvidiaTeslaT4"
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        res = subprocess.run(push_cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
        if res.returncode != 0:
            print(f"❌ Lỗi khi đẩy kernel: {res.stderr}")
            sys.exit(1)
        print("✅ Đã đẩy kernel thành công!")
        print(res.stdout.strip())
    else:
        print("ℹ️  Chế độ Monitor-Only: Bỏ qua bước sinh notebook và push, kết nối trực tiếp vào Kernel đang chạy.")
    
    # 3. Giám sát & Stream Live Log
    print("\n📡 Bước 3: Giám sát Kernel & Lắng nghe luồng SSE Live Log...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    live_log = os.path.join(OUTPUT_DIR, "live_benchmark.log")
    if os.path.exists(live_log):
        os.remove(live_log)
        
    stop_event = threading.Event()
    stream_t = threading.Thread(target=stream_logs_worker, args=(api, KERNEL_ID, live_log, stop_event), daemon=True)
    stream_t.start()
    
    start_time = time.time()
    last_status = None
    
    while True:
        try:
            status_obj = api.kernels_status(KERNEL_ID)
            status = getattr(status_obj, 'status', None)
            if hasattr(status, 'name'):
                status_name = status.name.lower()
            elif isinstance(status, str):
                status_name = status.lower()
            else:
                status_name = str(status).lower()
                
            elapsed = int(time.time() - start_time)
            if status_name != last_status:
                print(f"⏱️  [{elapsed//60}m {elapsed%60}s] Trạng thái Kernel: {status_name.upper()}", flush=True)
                last_status = status_name
                
            if status_name in ['complete']:
                print("\n🎉 Kernel đã hoàn thành toàn bộ quá trình chạy!")
                break
            elif status_name in ['error', 'cancel', 'aborted']:
                print(f"\n❌ Kernel kết thúc bất thường với trạng thái: {status_name.upper()}")
                stop_event.set()
                sys.exit(1)
        except Exception as e:
            print(f"⚠️  Cảnh báo kiểm tra trạng thái: {e}")
            
        time.sleep(15)
        
    stop_event.set()
    
    # 5. Tải file kết quả ZIP
    print("\n📥 Bước 5: Tải tệp kết quả benchmark_t4_results.zip về máy cục bộ...")
    time.sleep(5)
    
    try:
        api.kernels_output(KERNEL_ID, path=OUTPUT_DIR)
        print("✅ Tải kết quả hoàn tất!")
        
        # Giải nén kết quả
        zip_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.zip')]
        for z in zip_files:
            z_path = os.path.join(OUTPUT_DIR, z)
            print(f"📦 Đang giải nén: {z}...")
            with zipfile.ZipFile(z_path, 'r') as zipf:
                zipf.extractall(OUTPUT_DIR)
            print("✅ Giải nén xong!")
            
        # Hiển thị tóm tắt nếu có
        summary_md = os.path.join(OUTPUT_DIR, "benchmark_t4_summary.md")
        if os.path.exists(summary_md):
            print("\n" + "="*80)
            print("📄 KẾT QUẢ BÁO CÁO BENCHMARK T4x2:")
            print("="*80)
            with open(summary_md, "r", encoding="utf-8") as f:
                print(f.read())
                
    except Exception as e:
        print(f"❌ Lỗi khi tải hoặc đọc kết quả: {e}")

if __name__ == '__main__':
    main()
