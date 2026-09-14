#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Speed Streaming Downloader cho Checkpoint V25 Stage 3 (Final Stage).
Trích xuất trực tiếp các byte range của rec_cham_v25_stage3_checkpoint.zip
qua single persistent HTTP stream with resume capability.
"""

import os
import sys
import io
import time
import struct
import requests
import zipfile
import pickle
import shutil

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
import kaggle_auth
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
from kaggle_ops import get_authenticated_api

class HttpRangeReader(io.RawIOBase):
    def __init__(self, url, size):
        self.url = url
        self.size = size
        self.pos = 0
    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self.pos = offset
        elif whence == io.SEEK_CUR: self.pos += offset
        elif whence == io.SEEK_END: self.pos = self.size + offset
        return self.pos
    def tell(self): return self.pos
    def seekable(self): return True
    def readable(self): return True
    def readinto(self, b):
        if self.pos >= self.size: return 0
        end = min(self.pos + len(b) - 1, self.size - 1)
        r = requests.get(self.url, headers={'Range': f'bytes={self.pos}-{end}'}, timeout=30)
        data = r.content
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)

def stream_download_range(url, start_byte, end_byte, dest_path, desc=""):
    total_bytes = end_byte - start_byte + 1
    existing = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0

    if existing == total_bytes:
        print(f"✅ Đã có sẵn hoàn chỉnh: {desc} ({existing / (1024*1024):.2f} MB)")
        return

    if existing > total_bytes:
        os.remove(dest_path)
        existing = 0

    resume_start = start_byte + existing
    headers = {"Range": f"bytes={resume_start}-{end_byte}"}

    print(f"⬇️ Đang tải {desc} ({total_bytes / (1024*1024):.2f} MB)...")
    if existing > 0:
        print(f"   🔄 Resume từ byte {existing} ({existing / (1024*1024):.2f} MB)...")

    mode = "ab" if existing > 0 else "wb"
    t0 = time.time()
    last_print = t0
    downloaded_this_run = 0

    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest_path, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024 * 4): # 4 MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded_this_run += len(chunk)
                    curr_total = existing + downloaded_this_run
                    now = time.time()
                    if now - last_print >= 5.0 or curr_total == total_bytes:
                        speed = (downloaded_this_run / (1024 * 1024)) / max(now - t0, 0.1)
                        pct = curr_total / total_bytes * 100
                        print(f"   📊 [{pct:5.1f}%] {curr_total / (1024*1024):.1f} / {total_bytes / (1024*1024):.1f} MB ({speed:.1f} MB/s)", flush=True)
                        last_print = now

    el = time.time() - t0
    avg_speed = (downloaded_this_run / (1024 * 1024)) / max(el, 0.1)
    print(f"✅ Hoàn tất tải {desc} trong {el:.1f}s ({avg_speed:.1f} MB/s)!")

def main():
    owner = "gustavnguyen"
    slug = "paddleocr-cham-v25-stage3"
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    training_dir = os.path.dirname(scripts_dir)
    output_dir = os.path.join(training_dir, "output", "v25_stage3")
    final_ckpt_dir = os.path.join(output_dir, "stage3_final_checkpoint")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_ckpt_dir, exist_ok=True)

    print("=" * 80)
    print("🚀 HIGH-SPEED STREAMING EXTRACTOR - CHAM-OCR V25 STAGE 3 (FINAL)")
    print(f"   • Kernel   : {owner}/{slug}")
    print(f"   • Thư mục  : {output_dir}")
    print("=" * 80)

    api = get_authenticated_api()
    client = api.build_kaggle_client()
    req = ApiListKernelSessionOutputRequest()
    req.user_name = owner
    req.kernel_slug = slug

    print("\n🔍 Đang truy vấn danh sách output file trên Kaggle...")
    res = client.kernels.kernels_api_client.list_kernel_session_output(req)
    if not res.files:
        print("❌ Không tìm thấy file output nào!")
        return 1

    url = res.files[0].url
    r = requests.get(url, headers={'Range': 'bytes=0-0'})
    total_bytes = int(r.headers.get('Content-Range', '').split('/')[-1])
    print(f"🌐 Remote _output_.zip size: {total_bytes / (1024*1024):.2f} MB")

    print("\n📖 Đang đọc Central Directory của archive...")
    reader = HttpRangeReader(url, total_bytes)
    zf = zipfile.ZipFile(reader)
    print(f"✅ Đã tải Central Directory: {len(zf.namelist()):,} files trong archive.")

    stored_targets = [
        "rec_cham_v25_stage3_checkpoint.zip"
    ]

    for name in stored_targets:
        if name in zf.namelist():
            info = zf.getinfo(name)
            assert info.compress_type == 0, f"{name} không phải STORED zip!"
            
            # Đọc 30 bytes header cục bộ để tính data_start
            h_r = requests.get(url, headers={'Range': f'bytes={info.header_offset}-{info.header_offset+64}'}, timeout=30)
            sig, ver, flags, comp, mtime, mdate, crc, csize, usize, fname_len, extra_len = struct.unpack('<IHHHHHIIIHH', h_r.content[:30])
            data_start = info.header_offset + 30 + fname_len + extra_len
            data_end = data_start + info.compress_size - 1

            dest_path = os.path.join(output_dir, name)
            stream_download_range(url, data_start, data_end, dest_path, desc=name)

    # Giải nén rec_cham_v25_stage3_checkpoint.zip vào final_ckpt_dir
    stage3_zip = os.path.join(output_dir, "rec_cham_v25_stage3_checkpoint.zip")
    if os.path.exists(stage3_zip):
        print(f"\n📦 Đang giải nén Checkpoint vào {final_ckpt_dir}...")
        with zipfile.ZipFile(stage3_zip, 'r') as sz:
            for m in sz.namelist():
                fname = os.path.basename(m)
                if fname:
                    target_fp = os.path.join(final_ckpt_dir, fname)
                    with sz.open(m) as sf, open(target_fp, 'wb') as df:
                        shutil.copyfileobj(sf, df)
        print("✅ Giải nén toàn bộ Checkpoint Chặng 3 hoàn tất!")

    # Đọc states để xác minh
    states_p = os.path.join(final_ckpt_dir, "latest.states")
    if os.path.exists(states_p):
        with open(states_p, 'rb') as f:
            st = pickle.load(f)
        print("\n" + "=" * 65)
        print("🎯 XÁC MINH TRẠNG THÁI CHECKPOINT RESUME STAGE 3 CỤC BỘ:")
        print(f"   • Epoch đã lưu         : {st.get('epoch')}")
        print(f"   • Global step          : {st.get('global_step')}")
        best_d = st.get('best_model_dict', {})
        print(f"   • Best Accuracy        : {best_d.get('acc', 'N/A'):.4%}")
        print(f"   • Best Edit Distance   : {best_d.get('norm_edit_dis', 'N/A'):.4%}")
        print(f"   • Best Epoch           : {best_d.get('best_epoch')}")
        print("=" * 65)

    print("\n🎉 HOÀN TẤT ĐỒNG BỘ CHECKPOINT V25 STAGE 3 (FINAL) VỀ MÁY CỤC BỘ!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
