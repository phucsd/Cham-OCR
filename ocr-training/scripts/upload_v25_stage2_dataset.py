import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))
import kaggle_auth
kaggle_auth.init_kaggle_auth()
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    ds_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "v25_stage2_dataset"))
    print(f"📦 Thư mục Dataset Stage 2: {ds_dir}")
    print(f"📋 Các tệp bên trong:")
    total_size = 0
    for f in os.listdir(ds_dir):
        fp = os.path.join(ds_dir, f)
        sz = os.path.getsize(fp)
        total_size += sz
        print(f"   • {f} ({sz/(1024*1024):.2f} MB)")
    print(f"📊 Tổng dung lượng cần tải: {total_size/(1024*1024):.2f} MB")

    api = KaggleApi()
    api.authenticate()

    print("\n🚀 Bắt đầu tạo mới Dataset trên Kaggle: gustavnguyen/cham-ocr-v25-stage2-checkpoint...")
    t0 = time.time()
    try:
        res = api.dataset_create_new(ds_dir, public=False, quiet=False, convert_to_csv=False, dir_mode='skip')
        print(f"🎉 Tải lên hoàn tất trong {(time.time() - t0)/60:.2f} phút!")
        print("Phản hồi từ Kaggle:", res)
    except Exception as e:
        print(f"❌ Lỗi khi tải lên Dataset: {e}")
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
