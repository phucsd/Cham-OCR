#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Active Watchdog & SSE Live Stream Logger for Cham-OCR V25 Training on Kaggle.
Monitors the kernel run in real time, streams execution logs, provides status heartbeats,
and automatically downloads checkpoints when complete.
"""

import os
import sys
import time
import json
import zipfile
import shutil
import socket
import threading
import argparse
from kaggle.api.kaggle_api_extended import KaggleApi

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

socket.setdefaulttimeout(60)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)
sys.path.insert(0, SCRIPT_DIR)

from kaggle_auth import init_kaggle_auth


def get_api():
    init_kaggle_auth()
    api = KaggleApi()
    api.authenticate()
    return api


def stream_logs_worker(api, kernel_id, log_file_path, stop_event):
    """Lắng nghe luồng SSE log trực tiếp từ Kaggle và lưu file đồng thời phân tích sự kiện."""
    consecutive_empty = 0
    while not stop_event.is_set():
        try:
            with open(log_file_path, 'a', encoding='utf-8') as f_log:
                for event in api.kernels_logs_stream(kernel_id):
                    if stop_event.is_set():
                        break
                    data = event.get('data', '')
                    if data:
                        consecutive_empty = 0
                        f_log.write(data)
                        f_log.flush()
                        # In các dòng tiến độ quan trọng ra console
                        for line in data.splitlines():
                            line_str = line.strip()
                            if any(k in line_str for k in [
                                'epoch: [', 'best metric', 'cur metric',
                                'save best model', 'save model in', 'loss:', 'ips:', 'lr:',
                                'Detected GPU Count', 'Place(gpu', 'train dataloader has',
                                'Hoàn thành', 'HOÀN TẤT', 'Error', 'Exception',
                                'RuntimeError', 'ImportError', 'Traceback', 'LAUNCH',
                                'Train Progress', 'Val Progress', 'Weight Surgery',
                                'PHẪU THUẬT', 'PREFLIGHT', 'unknown_chars',
                                'Paddle Version', 'PADDLE ENVIRONMENT',
                                'NumPy Version', 'OpenCV Version', 'ImgAug Version'
                            ]) and not 'MB/s' in line_str and not 'â”' in line_str:
                                print(f"🔥 [KAGGLE V25] {line_str}")
                    else:
                        consecutive_empty += 1
        except Exception as e:
            time.sleep(3)

        if not stop_event.is_set():
            time.sleep(5)


import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest


def download_chunked_file(url, target_path, chunk_size=1024*1024*8, max_retries=10):
    """Tải file theo từng chunk có hỗ trợ resume và tự động retry nếu ngắt mạng."""
    for attempt in range(max_retries):
        try:
            existing_bytes = os.path.getsize(target_path) if os.path.exists(target_path) else 0
            headers = {}
            if existing_bytes > 0:
                headers["Range"] = f"bytes={existing_bytes}-"
                print(f"🔄 Lần thử {attempt + 1}/{max_retries}: Tiếp tục tải từ byte {existing_bytes} ({existing_bytes / (1024*1024):.2f} MB)...", flush=True)
            else:
                print(f"📥 Bắt đầu tải mới về: {target_path}...", flush=True)

            resp = requests.get(url, headers=headers, stream=True, timeout=60)
            
            if resp.status_code == 416:
                print("✅ File đã được tải đầy đủ!", flush=True)
                return target_path

            if resp.status_code not in (200, 206):
                resp.raise_for_status()

            total_bytes = existing_bytes + int(resp.headers.get("content-length", 0))
            remaining_mb = (total_bytes - existing_bytes) / (1024*1024)
            print(f"📊 Còn lại: {remaining_mb:.2f} MB / Tổng: {total_bytes / (1024*1024):.2f} MB", flush=True)

            mode = "ab" if existing_bytes > 0 and resp.status_code == 206 else "wb"
            downloaded = existing_bytes
            start_time = time.time()
            last_print = start_time

            with open(target_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_print >= 5:
                            speed = (downloaded - existing_bytes) / (now - start_time) / (1024*1024) if (now - start_time) > 0 else 0
                            percent = (downloaded / total_bytes * 100) if total_bytes > 0 else 0
                            print(f"⚡ Đã tải: {downloaded / (1024*1024):.1f} MB / {total_bytes / (1024*1024):.1f} MB ({percent:.1f}%) | Tốc độ: {speed:.2f} MB/s", flush=True)
                            last_print = now

            if os.path.exists(target_path) and os.path.getsize(target_path) >= total_bytes:
                print(f"\n✅ Tải hoàn tất 100%! Tổng dung lượng: {os.path.getsize(target_path) / (1024*1024):.2f} MB", flush=True)
                return target_path
        except Exception as e:
            print(f"⚠️ Mạng gián đoạn: {e}. Đang tự động kết nối lại sau 3 giây...", flush=True)
            time.sleep(3)

    raise RuntimeError(f"❌ Không thể tải file sau {max_retries} lần thử!")


def download_outputs(api, kernel_id, output_dir):
    """Tải và giải nén an toàn toàn bộ checkpoint output từ Kaggle bằng chunked streaming."""
    try:
        socket.setdefaulttimeout(600)
        print(f"📥 Đang truy vấn URL output từ Kaggle ({kernel_id})...", flush=True)
        owner, slug, _ = api.parse_kernel_string(kernel_id)
        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner
        req.kernel_slug = slug

        client = api.build_kaggle_client()
        res = client.kernels.kernels_api_client.list_kernel_session_output(req)

        if not res.files:
            print("⚠️ Không tìm thấy file output nào trên Kaggle qua session output API, fallback sang api.kernels_output...", flush=True)
            api.kernels_output(kernel_id, path=output_dir)
            return True

        for f_info in res.files:
            target_path = os.path.join(output_dir, f_info.file_name)
            print(f"📦 Tìm thấy output: {f_info.file_name} -> {target_path}", flush=True)
            download_chunked_file(f_info.url, target_path)

            # Giải nén nếu là file zip
            if target_path.endswith(".zip"):
                print(f"📂 Đang kiểm tra nội dung ZIP: {target_path}...", flush=True)
                with zipfile.ZipFile(target_path, 'r') as z:
                    namelist = z.namelist()
                    # Tìm file checkpoint zip con hoặc các file trọng yếu
                    target_files = [n for n in namelist if 'checkpoint.zip' in n or 'best_accuracy' in n or 'cham_dict_v25.txt' in n or 'latest' in n]
                    extract_dir = os.path.join(output_dir, "extracted")
                    os.makedirs(extract_dir, exist_ok=True)
                    for tf in target_files:
                        z.extract(tf, extract_dir)
                        print(f"  ✅ Đã trích xuất: {tf}", flush=True)

                    # Nếu có file checkpoint zip con (như rec_cham_v25_stageX_checkpoint.zip)
                    sub_zips = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('checkpoint.zip')]
                    for szp in sub_zips:
                        final_ckpt_dir = os.path.join(output_dir, "final_checkpoint")
                        os.makedirs(final_ckpt_dir, exist_ok=True)
                        with zipfile.ZipFile(szp, 'r') as subz:
                            subz.extractall(final_ckpt_dir)
                        print(f"🎉 ĐÃ GIẢI NÉN TOÀN BỘ CHECKPOINT VÀO: {final_ckpt_dir}", flush=True)

        return True
    except Exception as e:
        print(f"❌ Lỗi tải output: {e}", flush=True)
        return False


def run_watchdog(user="gustavnguyen", slug="paddleocr-cham-v25-stage1", output_dir=None, poll_interval=30):
    kernel_id = f"{user}/{slug}"
    if output_dir is None:
        output_dir = os.path.join(TRAINING_DIR, "output", "v25_stage1")
    os.makedirs(output_dir, exist_ok=True)
    live_log_path = os.path.join(output_dir, "live_training.log")
    if os.path.exists(live_log_path) and os.path.getsize(live_log_path) > 0:
        backup_path = os.path.join(output_dir, f"live_training_prev_{int(time.time())}.log")
        try:
            shutil.move(live_log_path, backup_path)
            print(f"📦 Đã lưu nhật ký phiên trước sang: {backup_path}")
        except Exception:
            pass

    print("=" * 80)
    print("🛡️  ACTIVE WATCHDOG V25 - SSE LIVE STREAM & LOG MONITOR")
    print(f"🎯 Kernel Mục Tiêu : {kernel_id}")
    print(f"📂 Nhật Ký Ghi Vào  : {live_log_path}")
    print(f"⏱️  Nhịp Heartbeat   : {poll_interval} giây / chu kỳ")
    print("=" * 80)

    api = get_api()

    # Khởi chạy luồng SSE Live Stream
    stop_stream_event = threading.Event()
    stream_thread = threading.Thread(
        target=stream_logs_worker,
        args=(api, kernel_id, live_log_path, stop_stream_event),
        daemon=True
    )
    stream_thread.start()
    print("📡 Kênh SSE Live Stream đã mở. Sẵn sàng nhận logs...\n")

    start_time = time.time()
    last_status = None
    fail_count = 0
    seen_log_offset = 0

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

            if status != last_status or elapsed_sec % 60 < poll_interval:
                print(f"⏱️  [Heartbeat {elapsed_min:02d}m{elapsed_sec%60:02d}s] Trạng thái: {status.upper()}", flush=True)
                last_status = status

            # Bổ sung kiểm tra log trực tiếp từ REST API để phòng ngừa gián đoạn SSE
            try:
                raw_logs = api.kernels_logs(kernel_id)
                if isinstance(raw_logs, str) and len(raw_logs) > seen_log_offset:
                    new_chunk = raw_logs[seen_log_offset:]
                    seen_log_offset = len(raw_logs)
                    with open(live_log_path, 'a', encoding='utf-8') as f_log:
                        f_log.write(new_chunk)
                        f_log.flush()
                    for line in new_chunk.splitlines():
                        line_str = line.strip()
                        if line_str and not 'MB/s' in line_str and not 'â”' in line_str:
                            print(f"🔥 [LOG] {line_str}", flush=True)
            except Exception:
                pass

            # 1. Trạng thái HOÀN TẤT
            if 'complete' in status:
                stop_stream_event.set()
                print("\n" + "=" * 80, flush=True)
                print(f"🎉 KERNEL V25 ĐÃ HOÀN TẤT THÀNH CÔNG TRONG {elapsed_min} PHÚT!", flush=True)
                print("📥 Bắt đầu tải và đồng bộ checkpoint kết quả...", flush=True)
                print("=" * 80, flush=True)
                download_outputs(api, kernel_id, output_dir)
                return True

            # 2. Trạng thái LỖI
            elif any(k in status for k in ['error', 'cancel', 'abort', 'fail']):
                stop_stream_event.set()
                print("\n" + "!" * 80, flush=True)
                print(f"🚨 BÁO ĐỘNG: KERNEL V25 BỊ DỪNG HOẶC GẶP LỖI: {status.upper()}!", flush=True)
                print(f"⏱️  Thời gian chạy: {elapsed_min} phút {elapsed_sec%60} giây.", flush=True)
                print("!" * 80, flush=True)
                return False

            fail_count = 0

        except Exception as e:
            fail_count += 1
            if fail_count >= 5:
                print(f"⚠️  Mất kết nối kiểm tra Kaggle API ({fail_count}/5): {e}", flush=True)

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Active Watchdog for Cham-OCR V25 Training")
    parser.add_argument("--kernel", type=str, default="paddleocr-cham-v25-stage1", help="Slug của kernel")
    parser.add_argument("--user", type=str, default="gustavnguyen", help="Kaggle username")
    parser.add_argument("--poll-interval", type=int, default=30, help="Chu kỳ heartbeat (giây)")
    parser.add_argument("--output-dir", type=str, default=None, help="Thư mục xuất logs & checkpoint")
    args = parser.parse_args()

    run_watchdog(
        user=args.user,
        slug=args.kernel,
        output_dir=args.output_dir,
        poll_interval=args.poll_interval
    )


if __name__ == '__main__':
    main()
