#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uploads essential Cham-OCR files directly into Lightning AI Studio via chunked base64 payload.
Bypasses all GitHub private repository restrictions and network throttling.
"""

import os
import sys
import base64
from lightning_sdk import Studio

if sys.stdout and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def send_file(studio, local_path, remote_path):
    print(f"📦 Đang đẩy tệp {local_path} sang {remote_path}...")
    with open(local_path, "rb") as f:
        content = f.read()

    b64_str = base64.b64encode(content).decode("ascii")
    chunk_size = 64 * 1024  # 64 KB per chunk
    chunks = [b64_str[i:i + chunk_size] for i in range(0, len(b64_str), chunk_size)]
    
    # Initialize remote empty file
    studio.run(f"python3 -c \"import pathlib; p = pathlib.Path('{remote_path}'); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('')\"")
    
    for idx, chunk in enumerate(chunks):
        cmd = f"python3 -c \"import pathlib; p = pathlib.Path('{remote_path}'); open('{remote_path}.b64', 'a').write('{chunk}')\""
        studio.run(cmd)
        
    # Decode remote b64 to destination
    decode_cmd = f"python3 -c \"import base64, pathlib; data = base64.b64decode(pathlib.Path('{remote_path}.b64').read_text()); pathlib.Path('{remote_path}').write_bytes(data); pathlib.Path('{remote_path}.b64').unlink()\""
    studio.run(decode_cmd)
    
    size_out = studio.run(f"ls -lh '{remote_path}'").strip()
    print(f"✅ Hoàn thành: {size_out}")

def main():
    studio = Studio(name="cham-det-h100", teamspace="phucsd", org="phucsd-org")
    print("Studio status:", studio.status)
    
    files_to_sync = [
        ("ocr-training/scripts/generate_detector_data_v2.py", "Cham-OCR/ocr-training/scripts/generate_detector_data_v2.py"),
        ("ocr-training/configs/det/ch_PP-OCRv4_det_h100.yml", "Cham-OCR/ocr-training/PaddleOCR/configs/det/ch_PP-OCRv4_det_h100.yml"),
        ("ocr-training/configs/det/ch_PP-OCRv4_det_h100.yml", "Cham-OCR/ocr-training/configs/det/ch_PP-OCRv4_det_h100.yml"),
        ("ocr-training/data/corpus/cham_text.txt", "Cham-OCR/ocr-training/data/corpus/cham_text.txt"),
    ]
    
    for local_p, remote_p in files_to_sync:
        send_file(studio, local_p, remote_p)
        
    print("\n🎉 Tất cả tệp mã nguồn và cấu hình đã được đẩy lên Studio thành công 100%!")

if __name__ == "__main__":
    main()
