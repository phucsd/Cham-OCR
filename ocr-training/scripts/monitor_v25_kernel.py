#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dedicated Real-Time Monitor for Cham-OCR V25 Kaggle Kernels.
Streams live training logs (Epoch, Step, Loss, Acc, ETA) directly from Kaggle stream API.
"""

import os
import sys
import time
import json
import socket

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

socket.setdefaulttimeout(30)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from kaggle_auth import init_kaggle_auth
init_kaggle_auth()

from kaggle_ops import get_authenticated_api
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest

def fetch_stream_lines(api, user_name, kernel_slug, timeout_sec=6):
    clean_lines = []
    try:
        with api.build_kaggle_client() as kaggle:
            client = kaggle.kernels.kernels_api_client
            req = ApiGetKernelSessionLogsStreamRequest()
            req.user_name = user_name
            req.kernel_slug = kernel_slug
            res = client.get_kernel_session_logs_stream(req)
            t0 = time.time()
            for line in res.iter_lines(decode_unicode=True):
                if line and line.startswith('data: '):
                    try:
                        item = json.loads(line[6:])
                        text = item.get('data', '')
                        for d in text.splitlines():
                            if d.strip():
                                clean_lines.append(d.strip())
                    except Exception:
                        pass
                if time.time() - t0 > timeout_sec:
                    break
    except Exception as e:
        pass
    return clean_lines

def monitor(kernel_id, poll_seconds=60):
    api = get_authenticated_api()
    if not api:
        print("❌ Lỗi: Không thể xác thực Kaggle API.")
        return 1

    parts = kernel_id.split('/')
    user_name = parts[0]
    kernel_slug = parts[1] if len(parts) > 1 else parts[0]

    print("=" * 75)
    print(f"🛰️  BẮT ĐẦU GIÁM SÁT REAL-TIME STREAMING KERNEL: {kernel_id}")
    print(f"   • Tần suất kiểm tra : {poll_seconds} giây / lần")
    print(f"   • Thời gian bắt đầu : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)
    sys.stdout.flush()

    start_t = time.time()
    last_status = None
    last_count = 0
    poll_count = 0

    while True:
        poll_count += 1
        elapsed_min = int((time.time() - start_t) / 60)
        try:
            st_obj = api.kernels_status(kernel_id)
            status = getattr(st_obj, 'status', None)
            if hasattr(status, 'name'):
                status_str = status.name.lower()
            elif isinstance(status, str):
                status_str = status.lower()
            else:
                status_str = str(status).lower()

            failure_msg = getattr(st_obj, 'failure_message', None) or getattr(st_obj, 'failureMessage', None)

            # Lấy các dòng log trực tiếp từ Live Stream API
            lines = fetch_stream_lines(api, user_name, kernel_slug, timeout_sec=6)
            current_count = len(lines)

            if current_count > last_count:
                new_lines = lines[last_count:]
                # Lọc các dòng có ý nghĩa huấn luyện quan trọng
                key_lines = [l for l in new_lines if any(k in l for k in ['ppocr INFO:', 'epoch:', 'cur metric', 'best metric', 'assert', 'Error', 'STAGE EXIT', '🎉'])]
                if not key_lines:
                    key_lines = new_lines[-10:] # in 10 dòng mới nhất nếu không có dòng quan trọng
                
                print(f"\n📡 [Sau {elapsed_min}m | Status: {status_str.upper()} | Tổng {current_count:,} dòng]:")
                for l in key_lines[-15:]:
                    print(f"   | {l}")
                last_count = current_count
                sys.stdout.flush()
            elif status_str != last_status or poll_count % 5 == 0:
                print(f"⏱️  [Sau {elapsed_min}m] Trạng thái: {status_str.upper()} (Đã nhận {current_count:,} dòng log)...")
                sys.stdout.flush()

            last_status = status_str

            if status_str in ('complete', 'completed'):
                print("\n" + "=" * 75)
                print(f"🎉 KERNEL {kernel_id} ĐÃ HOÀN TẤT THÀNH CÔNG sau {elapsed_min} phút!")
                print("=" * 75)
                sys.stdout.flush()
                return 0
            elif status_str in ('error', 'cancel', 'cancelled', 'aborted'):
                print("\n" + "=" * 75)
                print(f"❌ KERNEL {kernel_id} GẶP LỖI ({status_str.upper()}) sau {elapsed_min} phút!")
                if failure_msg:
                    print(f"   • Failure Message: {failure_msg}")
                if lines:
                    print("   • 25 DÒNG LOG CUỐI CÙNG TRƯỚC KHI CRASH:")
                    for l in lines[-25:]:
                        print(f"     ! {l}")
                print("=" * 75)
                sys.stdout.flush()
                return 1

        except Exception as e:
            print(f"⚠️  [Lỗi API tại {elapsed_min}m]: {e}")
            sys.stdout.flush()

        time.sleep(poll_seconds)

if __name__ == '__main__':
    kernel = sys.argv[1] if len(sys.argv) > 1 else 'gustavnguyen/paddleocr-cham-v25-stage1'
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    sys.exit(monitor(kernel, interval))
