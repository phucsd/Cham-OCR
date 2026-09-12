#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tải an toàn và giải nén output V25 Stage 1 từ Kaggle.
Hỗ trợ:
- Stream chunked download (không nghẽn RAM)
- Tự động resume (HTTP Range header) nếu mạng gián đoạn
- Tự động giải nén rec_cham_v25_stage1_checkpoint.zip
"""

import os
import sys
import time
import requests
import zipfile

sys.path.insert(0, os.path.dirname(__file__))
import kaggle_auth
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

def get_remote_file_size(url):
    """Lấy kích thước chuẩn của file remote trên Cloud."""
    try:
        r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=30)
        if r.status_code == 206:
            cr = r.headers.get("Content-Range", "")
            if "/" in cr:
                return int(cr.split("/")[-1])
        if "Content-Length" in r.headers:
            return int(r.headers["Content-Length"])
    except Exception as e:
        print(f"⚠️ Không lấy được kích thước remote: {e}")
    return None

def download_file(url, target_path, chunk_size=1024*1024*8, max_retries=10):
    """Tải file theo từng chunk có hỗ trợ resume và tự động retry nếu ngắt mạng."""
    total_remote_bytes = get_remote_file_size(url)
    if total_remote_bytes:
        print(f"🌐 Kích thước file remote: {total_remote_bytes / (1024*1024):.2f} MB ({total_remote_bytes} bytes)")

    for attempt in range(max_retries):
        try:
            existing_bytes = os.path.getsize(target_path) if os.path.exists(target_path) else 0

            if total_remote_bytes is not None:
                if existing_bytes > total_remote_bytes:
                    print(f"⚠️ File local lớn hơn remote ({existing_bytes} > {total_remote_bytes}). Xóa file cũ để tải mới...")
                    os.remove(target_path)
                    existing_bytes = 0
                elif existing_bytes == total_remote_bytes and existing_bytes > 0:
                    print(f"✅ File đã tồn tại trọn vẹn ({existing_bytes / (1024*1024):.2f} MB)!")
                    return target_path

            headers = {}
            if existing_bytes > 0:
                headers["Range"] = f"bytes={existing_bytes}-"
                print(f"🔄 Lần thử {attempt + 1}/{max_retries}: Tiếp tục tải từ byte {existing_bytes} ({existing_bytes / (1024*1024):.2f} MB)...")
            else:
                print(f"📥 Bắt đầu tải mới về: {target_path}...")

            resp = requests.get(url, headers=headers, stream=True, timeout=60)
            
            if resp.status_code == 416:
                print("✅ File đã được tải đầy đủ (HTTP 416)!")
                return target_path

            if resp.status_code not in (200, 206):
                resp.raise_for_status()

            total_bytes = total_remote_bytes if total_remote_bytes is not None else (existing_bytes + int(resp.headers.get("content-length", 0)))
            remaining_mb = (total_bytes - existing_bytes) / (1024*1024)
            print(f"📊 Còn lại: {remaining_mb:.2f} MB / Tổng: {total_bytes / (1024*1024):.2f} MB")

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
                print(f"\n✅ Tải hoàn tất 100%! Tổng dung lượng: {os.path.getsize(target_path) / (1024*1024):.2f} MB")
                return target_path
        except Exception as e:
            print(f"⚠️ Mạng gián đoạn: {e}. Đang tự động kết nối lại sau 3 giây...")
            time.sleep(3)

    raise RuntimeError(f"❌ Không thể tải file sau {max_retries} lần thử!")


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 80)
    print("🚀 ĐỒNG BỘ VÀ TẢI CHECKPOINT V25 STAGE 1 TỪ KAGGLE")
    print("=" * 80)

    kaggle_auth.init_kaggle_auth()
    api = KaggleApi()
    api.authenticate()

    kernel_id = "gustavnguyen/paddleocr-cham-v25-stage1"
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "v25_stage1"))
    os.makedirs(output_dir, exist_ok=True)

    print(f"🔍 Đang truy vấn URL tải từ kernel: {kernel_id}...")
    owner, slug, _ = api.parse_kernel_string(kernel_id)
    req = ApiListKernelSessionOutputRequest()
    req.user_name = owner
    req.kernel_slug = slug

    client = api.build_kaggle_client()
    res = client.kernels.kernels_api_client.list_kernel_session_output(req)

    if not res.files:
        print("❌ LỖI: Không tìm thấy file output nào trên Kaggle!")
        return 1

    print(f"📦 Tìm thấy {len(res.files)} file(s) trong output của kernel:")
    for i, f in enumerate(res.files):
        print(f"   [{i+1}] {f.file_name} (URL: {f.url[:50]}...)")

    # Tải từng file
    downloaded_zips = []
    for f_info in res.files:
        target_file = os.path.join(output_dir, f_info.file_name)
        print(f"\n⬇️ Đang xử lý tải: {f_info.file_name}...")
        download_file(f_info.url, target_file)
        if target_file.endswith(".zip"):
            downloaded_zips.append(target_file)

    # Giải nén
    extract_dir = os.path.join(output_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    final_ckpt_dir = os.path.join(output_dir, "stage1_final_checkpoint")
    os.makedirs(final_ckpt_dir, exist_ok=True)

    for zip_path in downloaded_zips:
        print(f"\n📂 Đang kiểm tra tệp ZIP: {os.path.basename(zip_path)} ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                namelist = z.namelist()
                print(f"   • Tổng số file bên trong: {len(namelist)}")
                # Nếu chính file này là checkpoint zip
                if "rec_cham_v25_stage1_checkpoint" in os.path.basename(zip_path):
                    print(f"   🎯 Đây là file checkpoint stage 1! Đang giải nén vào {final_ckpt_dir}...")
                    z.extractall(final_ckpt_dir)
                    print(f"   ✅ Đã giải nén xong vào: {final_ckpt_dir}")
                else:
                    for item in namelist:
                        if any(k in item for k in ['rec_cham_v25', 'best_accuracy', 'cham_dict', 'cham_v25_val']):
                            z.extract(item, extract_dir)
                            print(f"     + Đã trích xuất: {item}")
        except Exception as e:
            print(f"   ⚠️ Lỗi giải nén {zip_path}: {e}")

    # Kiểm tra nếu có rec_cham_v25_stage1_checkpoint.zip nằm bên trong extract_dir
    stage1_nested = None
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f == "rec_cham_v25_stage1_checkpoint.zip":
                stage1_nested = os.path.join(root, f)
                break
    if stage1_nested:
        print(f"\n🎉 Tìm thấy file nén con: {stage1_nested} ({os.path.getsize(stage1_nested)/(1024*1024):.2f} MB)")
        with zipfile.ZipFile(stage1_nested, 'r') as sz:
            sz.extractall(final_ckpt_dir)
        print(f"✅ ĐÃ GIẢI NÉN TOÀN BỘ CHECKPOINT VÀO: {final_ckpt_dir}")

    print("\n" + "=" * 80)
    print("🎉 HOÀN TẤT ĐỒNG BỘ CHECKPOINT V25 STAGE 1 VỀ MÁY CỤC BỘ!")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
