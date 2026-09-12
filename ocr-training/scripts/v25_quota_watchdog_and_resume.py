#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kịch bản tự động thăm dò Kaggle GPU Quota và tự động kích hoạt huấn luyện Cham-OCR V25 Chặng 1.
Chạy ngầm để đảm bảo:
1. Thăm dò liên tục qua GetAcceleratorQuotaStatistics API.
2. Kiểm tra chặt chẽ: chỉ khởi chạy khi Kaggle đã thực sự RESET quota 30 giờ mới (timeUsed < 3600s, remaining >= 28h).
3. Tự động dispatch prepare_v25_kaggle.py --stage 1 và theo dõi tiến trình real-time.
"""

import os
import sys
import time
import json
import socket
import datetime
import requests
import re

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

socket.setdefaulttimeout(30)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from kaggle_auth import init_kaggle_auth
from kaggle_ops import get_authenticated_api

LOG_FILE = os.path.join(os.path.dirname(SCRIPT_DIR), "v25_quota_watchdog.log")


def log(msg):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now_str}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def check_gpu_quota():
    """Truy vấn chính xác số liệu GPU Quota từ Kaggle API."""
    user, key = init_kaggle_auth()
    url = "https://api.kaggle.com/v1/kernels.KernelsApiService/GetAcceleratorQuotaStatistics"
    try:
        res = requests.post(url, auth=(user, key), json={}, timeout=15)
        if res.status_code != 200:
            log(f"⚠️ Lỗi HTTP {res.status_code} khi gọi GetAcceleratorQuotaStatistics")
            return None

        data = res.json()
        gpu = data.get("gpuQuota", {})
        used_str = gpu.get("timeUsed", "0s")
        allowed_str = gpu.get("totalTimeAllowed", "108000s")
        refresh_time = data.get("quotaRefreshTime", "")

        used = float(re.sub(r"[^\d.]", "", used_str))
        allowed = float(re.sub(r"[^\d.]", "", allowed_str))
        remaining = allowed - used

        return {
            "used_sec": used,
            "allowed_sec": allowed,
            "remaining_sec": remaining,
            "used_hours": used / 3600.0,
            "remaining_hours": remaining / 3600.0,
            "refresh_time": refresh_time
        }
    except Exception as e:
        log(f"⚠️ Ngoại lệ khi kiểm tra quota: {e}")
        return None


def main():
    log("=" * 70)
    log("🚀 KHỞI ĐỘNG V25 KAGGLE QUOTA WATCHDOG & AUTO-DISPATCHER")
    log("   • Mục tiêu: Theo dõi đến khi Kaggle nạp 30h GPU mới -> Tự động chạy V25 Stage 2")
    log(f"   • Tệp nhật ký: {LOG_FILE}")
    log("=" * 70)

    # 1. Kiểm tra trạng thái ban đầu
    quota = check_gpu_quota()
    if quota:
        log(f"📊 Trạng thái hiện tại: Đã dùng {quota['used_hours']:.2f}h / {quota['allowed_sec']/3600:.1f}h | Còn lại: {quota['remaining_hours']:.2f}h | Reset: {quota['refresh_time']}")
    else:
        log("⚠️ Không lấy được thông tin quota ban đầu, sẽ thử lại trong vòng lặp.")

    # 2. Vòng lặp chờ reset
    poll_count = 0
    while True:
        poll_count += 1
        quota = check_gpu_quota()

        if quota:
            # Điều kiện xác nhận đã reset: còn lại >= 28 giờ HOẶC đã dùng < 1 giờ
            is_reset = (quota["remaining_hours"] >= 28.0) or (quota["used_hours"] < 1.0)
            
            if is_reset:
                log("🎉🎉🎉 XÁC NHẬN: KAGGLE ĐÃ RESET GPU QUOTA THÀNH CÔNG! 🎉🎉🎉")
                log(f"   • Hạn ngạch mới: {quota['remaining_hours']:.2f} giờ khả dụng (đã dùng: {quota['used_hours']:.2f}h)")
                break
            else:
                log(f"⏳ [Lần {poll_count}] Quota chưa reset (còn {quota['remaining_hours']:.2f}h). Đang chờ mốc reset lúc 07:00 sáng mai (00:00 UTC).")
        else:
            log(f"⚠️ [Lần {poll_count}] Không kết nối được Kaggle API, sẽ thử lại.")

        now_dt = datetime.datetime.now()
        target_time = datetime.datetime(2026, 9, 12, 7, 0, 0)
        time_diff_sec = (target_time - now_dt).total_seconds()

        if time_diff_sec > 1800:
            sleep_time = 900  # 15 phút
        elif time_diff_sec > 0:
            sleep_time = 120  # 2 phút
        else:
            sleep_time = 60   # 1 phút khi đã qua 07:00 mà đang đợi Kaggle cập nhật

        time.sleep(sleep_time)

    # 3. Kích hoạt huấn luyện V25 Stage 2
    log("🚀 BẮT ĐẦU TỰ ĐỘNG DISPATCH V25 CHẶNG 2 LÊN KAGGLE DUAL GPU T4x2...")
    import subprocess
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "prepare_v25_kaggle.py"), "--stage", "2", "--push"]
    
    log(f"   • Lệnh: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=os.path.dirname(SCRIPT_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")

    for line in proc.stdout:
        log(f"   | {line.strip()}")

    proc.wait()
    if proc.returncode == 0:
        log("✅ DISPATCH V25 CHẶNG 2 THÀNH CÔNG!")
        # Khởi chạy monitor cho Stage 2
        monitor_cmd = [sys.executable, os.path.join(SCRIPT_DIR, "monitor_v25_kernel.py"), "gustavnguyen/paddleocr-cham-v25-stage2", "60"]
        log(f"📡 Khởi chạy monitor Stage 2: {' '.join(monitor_cmd)}")
        subprocess.Popen(monitor_cmd, cwd=os.path.dirname(SCRIPT_DIR))
    else:
        log(f"❌ DISPATCH THẤT BẠI với mã lỗi {proc.returncode}")


if __name__ == "__main__":
    main()
