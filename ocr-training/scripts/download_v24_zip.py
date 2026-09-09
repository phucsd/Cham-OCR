import os
import sys
import time
import zipfile
import shutil

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.environ["KAGGLE_USERNAME"] = "gustavnguyen"
os.environ["KAGGLE_KEY"] = "6bf56db7e5c0fa7895d157167961d92b"

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

def download_and_deploy():
    print("=" * 70)
    print("🚀 TẢI TRỰC TIẾP MÔ HÌNH V24 TỪ KAGGLE BẰNG API CHUYÊN DỤNG")
    print("=" * 70)

    save_dir = "ocr-training/output/v24"
    os.makedirs(save_dir, exist_ok=True)
    target_zip_name = "rec_cham_v24_best.zip"
    local_zip_path = os.path.join(save_dir, target_zip_name)

    print(f"📡 Đang kết nối tới máy chủ Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    t0 = time.time()
    with api.build_kaggle_client() as kaggle:
        req = ApiDownloadKernelOutputRequest()
        req.owner_slug = "gustavnguyen"
        req.kernel_slug = "paddleocr-cham-finetune"
        req.file_path = target_zip_name
        
        print(f"📥 Đang lấy Direct URL của '{target_zip_name}'...")
        res = kaggle.kernels.kernels_api_client.download_kernel_output(req)
        download_url = res.url
        print(f"🔗 Direct URL: {download_url[:70]}...")
        
    print(f"⚡ Bắt đầu tải tệp về bằng curl (tối đa băng thông, tự động khôi phục)...")
    cmd = f'curl.exe -L --retry 10 --retry-delay 2 --retry-all-errors -o "{local_zip_path}" "{download_url}"'
    ret = os.system(cmd)
    if ret != 0 or not os.path.exists(local_zip_path) or os.path.getsize(local_zip_path) < 100 * 1024 * 1024:
        print(f"❌ Tải thất bại hoặc tệp không đủ kích thước ({os.path.getsize(local_zip_path) if os.path.exists(local_zip_path) else 0} bytes)")
        return False

    print(f"\n✅ Đã tải xong tệp ({os.path.getsize(local_zip_path)/(1024*1024):.2f} MB) trong {time.time()-t0:.1f}s!")

    # Giải nén
    extract_folder = os.path.join(save_dir, "extracted_v24")
    if os.path.exists(extract_folder):
        shutil.rmtree(extract_folder, ignore_errors=True)
    os.makedirs(extract_folder, exist_ok=True)
    
    print(f"\n📦 Đang giải nén vào '{extract_folder}'...")
    with zipfile.ZipFile(local_zip_path, 'r') as zf:
        zf.extractall(extract_folder)
        names = zf.namelist()
        print(f"   Đã giải nén {len(names)} tệp: {names[:10]}")

    # Tìm thư mục inference
    candidates = [
        os.path.join(extract_folder, "rec_cham_inference_v24"),
        os.path.join(extract_folder, "output", "rec_cham_inference_v24"),
        extract_folder
    ]
    
    found_infer = None
    for cand in candidates:
        if os.path.exists(os.path.join(cand, "inference.pdmodel")) or os.path.exists(os.path.join(cand, "model.pdmodel")):
            found_infer = cand
            break
            
    # Thư mục đích trong Studio
    studio_target = "ocr-studio/data/output/rec_cham_inference_v24"
    os.makedirs(studio_target, exist_ok=True)
    
    if found_infer:
        print(f"\n🚀 Đồng bộ mô hình inference vào Web Studio: '{studio_target}'")
        for item in os.listdir(found_infer):
            s_p = os.path.join(found_infer, item)
            d_p = os.path.join(studio_target, item)
            if os.path.isfile(s_p):
                shutil.copy2(s_p, d_p)
                print(f"   + {item} ({os.path.getsize(d_p)/(1024*1024):.2f} MB)")
    else:
        # Nếu tệp nằm phân tán
        for root, dirs, files in os.walk(extract_folder):
            for f in files:
                if f.endswith('.pdmodel') or f.endswith('.pdiparams') or f.endswith('.yml'):
                    s_p = os.path.join(root, f)
                    d_p = os.path.join(studio_target, f)
                    shutil.copy2(s_p, d_p)
                    print(f"   + {f} ({os.path.getsize(d_p)/(1024*1024):.2f} MB)")

    # Đảm bảo copy từ điển V24 (103 ký tự)
    dict_v24_src = "ocr-training/data/cham_dict_v24.txt"
    if os.path.exists(dict_v24_src):
        shutil.copy2(dict_v24_src, os.path.join(studio_target, "cham_dict_v24.txt"))
        print(f"   + Đã đồng bộ từ điển V24: {dict_v24_src} -> {studio_target}/cham_dict_v24.txt")

    print("\n" + "=" * 70)
    print("🎉 TRIỂN KHAI MÔ HÌNH V24 VÀO HỆ THỐNG THÀNH CÔNG 100%!")
    print(f"📁 Thư mục inference: {studio_target}")
    for item in os.listdir(studio_target):
        p = os.path.join(studio_target, item)
        print(f"   - {item} ({os.path.getsize(p)/(1024*1024):.2f} MB)")
    print("=" * 70)
    return True

if __name__ == '__main__':
    download_and_deploy()
