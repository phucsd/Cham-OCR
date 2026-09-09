import os
import sys
import json

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    print("📦 Bắt đầu đóng gói Kaggle Notebook huấn luyện V24...")
    
    # 1. Đọc mã nguồn các script cục bộ
    dict_extension_code = read_file('ocr-training/scripts/dict_extension_v24.py')
    weight_surgery_code = read_file('ocr-training/scripts/weight_surgery_v24.py')
    generate_data_v24_code = read_file('ocr-training/scripts/generate_data_v24.py')
    config_v24_code = read_file('ocr-training/configs/rec_cham_v24.yml')
    
    kaggle_dict_extension = dict_extension_code
    kaggle_weight_surgery = weight_surgery_code
    
    cells = []
    
    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + '\n' for line in text.splitlines()]
        })
        
    def add_code(code_str):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + '\n' for line in code_str.splitlines()]
        })

    # Tiêu đề
    add_md("""# Huấn Luyện Mô Hình Nhận Diện Chữ Chăm PP-OCRv4 V24 (Dual GPU T4x2)
Mô hình giải quyết triệt để:
- Số thứ tự khổ thơ Chăm 1-99 (`꩑꩞` - `꩙꩙꩞`)
- Dấu kết thúc câu kép Double Danda (`꩝꩝`)
- Dấu phụ đối kháng `ꨲ` (Au, U+AA32) vs `ꨶ` (O, U+AA36)
- Tổ hợp 3 tầng dấu phụ và bảo toàn nguyên âm trước `ꨯ`, `ꨯꨱ`
- Khả năng chống chịu Mờ rung (Motion Blur) và Nhiễu giấy cổ""")

    # Phase 1: Environment & GPU Check
    add_md("## Phase 1: Cài đặt thư viện & Kiểm tra GPU T4x2")
    add_code("""# 1. Cài đặt PaddlePaddle GPU và các thư viện phụ trợ
!pip install --quiet "paddlepaddle-gpu>=2.6.0" "albumentations>=1.4.0" fonttools pyyaml opencv-python Pillow

# 2. Kiểm tra môi trường GPU
import os
import sys
import paddle

print("="*60)
print("PADDLE ENVIRONMENT STATUS:")
print("Paddle Version:", paddle.__version__)
print("Compiled with CUDA:", paddle.is_compiled_with_cuda())
gpu_count = paddle.device.cuda.device_count()
print(f"Detected GPU Count: {gpu_count}")
print("="*60)

if gpu_count < 1:
    print("⚠️ CẢNH BÁO: Không tìm thấy GPU! Vui lòng chọn Accelerator = GPU T4 x2 trong Settings.")
else:
    for i in range(gpu_count):
        print(f" - GPU {i}: {paddle.device.cuda.get_device_name(i)}")
""")

    # Phase 2: Setup Workspace & PaddleOCR
    add_md("## Phase 2: Khởi tạo thư mục làm việc & Clone PaddleOCR")
    add_code("""import os
import shutil

os.makedirs('/kaggle/working/data/fonts', exist_ok=True)
os.makedirs('/kaggle/working/data/output_v24_temp', exist_ok=True)
os.makedirs('/kaggle/working/configs', exist_ok=True)
os.makedirs('/kaggle/working/scripts', exist_ok=True)
os.makedirs('/kaggle/working/output/rec_cham_v24', exist_ok=True)
os.makedirs('/kaggle/working/output/v24_training', exist_ok=True)

# Tự động phát hiện đường dẫn Assets Input trên Kaggle
candidate_asset_dirs = [
    "/kaggle/input/cham-ocr-v5-assets",
    "/kaggle/input/datasets/gustavnguyen/cham-ocr-v5-assets"
]
ASSETS_DIR = None
for d in candidate_asset_dirs:
    if os.path.exists(d):
        ASSETS_DIR = d
        break

if not ASSETS_DIR and os.path.exists('/kaggle/input'):
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'v23_checkpoint' in dirs:
            ASSETS_DIR = root
            break

if not ASSETS_DIR:
    raise FileNotFoundError(f"❌ LỖI: Không tìm thấy thư mục Assets trong /kaggle/input")

print(f"✅ Đã tìm thấy Assets Input tại: {ASSETS_DIR}")

# Copy fonts
fonts_src = os.path.join(ASSETS_DIR, "fonts", "fonts")
if not os.path.exists(fonts_src):
    fonts_src = os.path.join(ASSETS_DIR, "fonts")

for f in os.listdir(fonts_src):
    if f.endswith(('.ttf', '.otf')):
        shutil.copy2(os.path.join(fonts_src, f), os.path.join('/kaggle/working/data/fonts', f))

print("✅ Đã sao chép fonts:", os.listdir('/kaggle/working/data/fonts'))

# Clone PaddleOCR nếu chưa có
if not os.path.exists('/kaggle/working/PaddleOCR'):
    print("⏳ Đang clone PaddleOCR...")
    !git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git /kaggle/working/PaddleOCR
    !pip install --quiet -r /kaggle/working/PaddleOCR/requirements.txt
else:
    print("✅ PaddleOCR đã tồn tại sẵn.")
""")

    # Phase 3: Ghi mã nguồn các script
    add_md("## Phase 3: Nạp kịch bản Sinh dữ liệu, Mở rộng từ điển & Phẫu thuật trọng số")
    add_code(f"%%writefile /kaggle/working/scripts/dict_extension_v24.py\n{kaggle_dict_extension}")
    add_code(f"%%writefile /kaggle/working/scripts/weight_surgery_v24.py\n{kaggle_weight_surgery}")
    add_code(f"%%writefile /kaggle/working/scripts/generate_data_v24.py\n{generate_data_v24_code}")
    add_code(f"%%writefile /kaggle/working/configs/rec_cham_v24.yml\n{config_v24_code}")

    # Phase 4: Chạy chuẩn bị dữ liệu và phẫu thuật trọng số
    add_md("## Phase 4: Mở rộng từ điển, Phẫu thuật trọng số & Sinh 50K dữ liệu V24 đặc trị")
    add_code("""import os, sys, time
%cd /kaggle/working

print("=== BƯỚC 1: MỞ RỘNG TỪ ĐIỂN V24 ===")
!python3 scripts/dict_extension_v24.py

print("\\n=== BƯỚC 2: PHẪU THUẬT TRỌNG SỐ V24 ===")
!python3 scripts/weight_surgery_v24.py

print("\\n=== BƯỚC 3: SINH DỮ LIỆU TỔNG HỢP V24 ĐẶC TRỊ (45K TRAIN, 4.5K VAL) ===")
t_start = time.time()
# Chạy với 4 vCPU
!python3 scripts/generate_data_v24.py --output-dir /kaggle/working/data/cham_synthetic_v24 --fonts-dir /kaggle/working/data/fonts --num-train 45000 --num-val 4500 --num-workers 4
print(f"🎉 Hoàn thành toàn bộ dữ liệu trong {(time.time()-t_start)/60:.2f} phút!")
""")

    # Phase 5: Khởi chạy huấn luyện song song phân tán trên GPU T4x2
    add_md("## Phase 5: Huấn luyện phân tán song song trên Dual GPU Tesla T4x2 (20 Epochs)")
    add_code("""import os
import paddle

gpu_count = paddle.device.cuda.device_count()
print(f"🔥 Khởi chạy huấn luyện với {gpu_count} GPU...")

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.90"

%cd /kaggle/working/PaddleOCR

if gpu_count > 1:
    print("🚀 Sử dụng lệnh phân tán song song đa GPU (paddle.distributed.launch --gpus '0,1'):")
    !python3 -m paddle.distributed.launch --gpus '0,1' tools/train.py -c /kaggle/working/configs/rec_cham_v24.yml
else:
    print("⚡ Huấn luyện trên Single GPU:")
    !python3 tools/train.py -c /kaggle/working/configs/rec_cham_v24.yml
""")

    # Phase 6: Xuất mô hình Inference & Đóng gói ZIP
    add_md("## Phase 6: Xuất mô hình Inference & Đóng gói rec_cham_v24_best.zip")
    add_code("""import os
import shutil
import zipfile

%cd /kaggle/working/PaddleOCR

best_ckpt = "/kaggle/working/output/rec_cham_v24/best_accuracy"
latest_ckpt = "/kaggle/working/output/rec_cham_v24/latest"

target_ckpt = best_ckpt if os.path.exists(best_ckpt + ".pdparams") else latest_ckpt
print(f"🎯 Sử dụng checkpoint: {target_ckpt}")

infer_dir = "/kaggle/working/output/rec_cham_inference_v24"
os.makedirs(infer_dir, exist_ok=True)

# Xuất mô hình inference
!python3 tools/export_model.py -c /kaggle/working/configs/rec_cham_v24.yml -o Global.pretrained_model={target_ckpt} Global.save_inference_dir={infer_dir}

# Copy từ điển vào thư mục inference
shutil.copy2('/kaggle/working/data/cham_dict_v24.txt', os.path.join(infer_dir, 'cham_dict_v24.txt'))

print("✅ Thư mục Inference:", os.listdir(infer_dir))

# Đóng gói toàn bộ kết quả thành file zip trong /kaggle/working
zip_path = "/kaggle/working/rec_cham_v24_best.zip"
print(f"📦 Đang nén kết quả vào {zip_path}...")

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Nén inference model
    for root, dirs, files in os.walk(infer_dir):
        for f in files:
            full_p = os.path.join(root, f)
            rel_p = os.path.relpath(full_p, "/kaggle/working")
            zipf.write(full_p, rel_p)
            
    # Nén best accuracy pdparams
    for ext in [".pdparams", ".pdopt", ".states"]:
        p = target_ckpt + ext
        if os.path.exists(p):
            zipf.write(p, os.path.join("best_checkpoint", os.path.basename(p)))
            
    # Nén từ điển
    zipf.write('/kaggle/working/data/cham_dict_v24.txt', 'cham_dict_v24.txt')

print(f"🎉 HOÀN TẤT TRỌN GÓI! File zip sẵn sàng ({os.path.getsize(zip_path) / (1024*1024):.2f} MB): {zip_path}")
""")

    # 5. Lưu Notebook JSON
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    submission_dir = 'ocr-training/output/kaggle_v24_submission'
    os.makedirs(submission_dir, exist_ok=True)
    
    nb_path = os.path.join(submission_dir, 'paddleocr_cham_v24_train.ipynb')
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_dict, f, indent=1, ensure_ascii=False)
        
    meta_path = os.path.join(submission_dir, 'kernel-metadata.json')
    metadata = {
        "id": "gustavnguyen/paddleocr-cham-finetune",
        "title": "paddleocr-cham-finetune",
        "code_file": "paddleocr_cham_v24_train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            "gustavnguyen/cham-ocr-v5-assets"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"✅ Đã tạo thành công Notebook tại: {nb_path}")
    print(f"✅ Đã tạo kernel-metadata tại: {meta_path}")
    return nb_path, meta_path

if __name__ == '__main__':
    main()
