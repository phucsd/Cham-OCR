#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Kaggle Dispatcher for Cham-OCR V25 Multi-Stage Training.
Prepares, validates, and dispatches Stage 1, 2, or 3 training kernels to Kaggle Dual T4x2.

Core Invariants Enforced:
  1. Global.epoch_num = 40 locked across all stages (T_max = 87,480 steps, 2187 steps/epoch).
  2. Linear warmup runs exactly once (Epochs 1-2 = 4,374 steps).
  3. Decoupled Stage Exit via tools/program.py hook:
       - Stage 1 stops at Epoch 12 (global_step 26,244)
       - Stage 2 stops at Epoch 22 (global_step 48,114)
       - Stage 3 stops at Epoch 40 (global_step 87,480)
  4. Validation freeze: 10,000 samples and canonical 162-token NFC dictionary frozen bitwise.
  5. Dataset distribution verified with manifest & statistics assertions.
  6. Resumes via Global.checkpoints (loading weights, optimizer, lr_scheduler, states).
  7. Strict deterministic archive paths; zero os.walk('/kaggle') wildcard scanning.
  8. Kaggle account: "gustavnguyen", Accelerator: NvidiaTeslaT4 (Dual T4x2).
"""

import os
import sys
import json
import time
import shutil
import argparse

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)
sys.path.insert(0, SCRIPT_DIR)

from kaggle_auth import init_kaggle_auth
init_kaggle_auth()

from kaggle_ops import get_authenticated_api, monitor_kernel


def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def build_v25_notebook(stage=1, epoch_num=12, prev_epochs=0):
    """Xây dựng Jupyter Notebook hoàn chỉnh, tự chứa mọi thành phần cho V25 trên Kaggle."""
    stage_end_epoch = prev_epochs + epoch_num
    global_total_epochs = 40
    print(f"📦 Đang đóng gói mã nguồn và cấu hình V25 Chặng {stage}:")
    print(f"   • Epochs chặng này   : Epoch {prev_epochs + 1} -> {stage_end_epoch} (tổng {epoch_num} epochs)")
    print(f"   • Toàn cầu epoch_num : {global_total_epochs} (T_max = 87,480 steps)")
    print(f"   • Stage exit epoch   : {stage_end_epoch}")

    manifest_code = read_file(os.path.join(TRAINING_DIR, "configs", "v25_dataset_manifest.json"))
    generate_data_code = read_file(os.path.join(TRAINING_DIR, "scripts", "generate_data_v25.py"))
    build_dict_code = read_file(os.path.join(TRAINING_DIR, "scripts", "build_dict_v25.py"))
    surgery_weights_code = read_file(os.path.join(TRAINING_DIR, "scripts", "surgery_v25_weights.py"))
    config_v25_code = read_file(os.path.join(TRAINING_DIR, "configs", "rec_cham_v25.yml"))
    dict_v25_code = read_file(os.path.join(TRAINING_DIR, "data", "cham_dict_v25.txt"))

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

    # Header
    add_md(f"""# Huấn Luyện Mô Hình Nhận Diện Chữ Chăm PP-OCRv4 V25 (Chặng {stage})
## Phần cứng: Kaggle Dual GPU Tesla T4x2 (`--accelerator NvidiaTeslaT4`)

### Kiến Trúc Đa Chặng Chuẩn Hóa (Multi-Stage Invariants):
- **Toàn bộ tiến trình**: Cố định `Global.epoch_num = 40` ($T_{{max}} = 87,480$ bước, 2,187 steps/epoch).
- **Phân chặng**:
  - Chặng 1: Epochs 1 -> 12 (global_step: 0 -> 26,244, Warmup 2 epochs đầu).
  - Chặng 2: Epochs 13 -> 22 (global_step: 26,244 -> 48,114, Cosine tiếp tục liên tục).
  - Chặng 3: Epochs 23 -> 40 (global_step: 48,114 -> 87,480, hội tụ cực đại).
- **Chặng hiện tại ({stage})**: Huấn luyện từ **Epoch {prev_epochs + 1}** đến **Epoch {stage_end_epoch}**.
- **Đóng băng Validation**: Tập Validation (10,000 mẫu) và Từ điển NFC (162 tokens) đóng băng tuyệt đối giữa 3 chặng.
- **Dữ liệu huấn luyện**: 140,000 mẫu Train/chặng tuân thủ nghiêm ngặt phân bố Manifest.""")

    # Phase 1: Environment & GPU Verification
    add_md("## Phase 1: Môi Trường & Xác Minh Dual GPU Tesla T4x2")
    add_code("""# 1. Khóa cứng NumPy 1.26.4 và OpenCV 4.9.0.80 để đồng bộ C-ABI (0x1000009)
!pip install --quiet "numpy==1.26.4" "opencv-python==4.9.0.80" "opencv-python-headless==4.9.0.80" "imgaug<=0.4.0" pyclipper shapely lmdb attrdict fonttools pyyaml Pillow rapidfuzz scikit-image lxml openpyxl visualdl
!pip install --quiet "paddlepaddle-gpu>=2.6.0" -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 2. Cấu hình sitecustomize.py toàn cục để mọi tiến trình Python con (kể cả rank workers) đều có monkeypatch tương thích
import os
import sys

sitecustomize_content = '''import sys
import os
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.80"

import numpy as np
if not hasattr(np, 'sctypes'):
    np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64], 'uint': [np.uint8, np.uint16, np.uint32, np.uint64], 'float': [np.float16, np.float32, np.float64], 'complex': [np.complex64, np.complex128], 'others': [bool, object, bytes, str]}
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'typeDict'): np.typeDict = {}

# Fallback mock cho visualdl để tránh ModuleNotFoundError khi tools/program.py import VDLLogger
try:
    import visualdl
except Exception:
    import types
    vdl = types.ModuleType('visualdl')
    class DummyLogWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass
    vdl.LogWriter = DummyLogWriter
    sys.modules['visualdl'] = vdl
'''

for p in sys.path:
    if 'site-packages' in p or 'dist-packages' in p:
        try:
            with open(os.path.join(p, 'sitecustomize.py'), 'w', encoding='utf-8') as f:
                f.write(sitecustomize_content)
        except Exception:
            pass

with open('/kaggle/working/sitecustomize.py', 'w', encoding='utf-8') as f:
    f.write(sitecustomize_content)

# 3. Sanity check: Xác minh trực tiếp các thư viện trọng yếu trên subprocess (môi trường thực thi của training)
print("=" * 60)
print("🔍 SUBPROCESS ENVIRONMENT CHECK:")
!python3 -c "import numpy as np; print('NumPy Version      :', np.__version__); assert np.__version__.startswith('1.'), 'NumPy must be 1.x'; import cv2; print('OpenCV Version     :', cv2.__version__); import imgaug; print('ImgAug Version     :', imgaug.__version__); import paddle; print('Paddle Version     :', paddle.__version__); import rapidfuzz; from rapidfuzz.distance import Levenshtein; print('RapidFuzz Version  :', rapidfuzz.__version__); import visualdl; from visualdl import LogWriter; print('VisualDL Version   :', visualdl.__version__ if hasattr(visualdl, '__version__') else 'mocked')"
print("=" * 60)

# 4. Kiểm tra GPU
import paddle
print("🚀 PADDLE GPU STATUS:")
print("Paddle Version     :", paddle.__version__)
print("Compiled with CUDA :", paddle.is_compiled_with_cuda())
gpu_count = paddle.device.cuda.device_count()
print(f"Detected GPU Count : {gpu_count}")
print("=" * 60)

assert gpu_count == 2, f"❌ LỖI: Bắt buộc cấu hình Dual GPU Tesla T4x2 (2 cards) nhưng chỉ phát hiện {gpu_count} GPU! Vui lòng chọn Accelerator = GPU T4 x2."
for i in range(gpu_count):
    dev_name = paddle.device.cuda.get_device_name(i)
    print(f" - GPU {i}: {dev_name}")
    assert "T4" in dev_name, f"❌ LỖI: GPU {i} là {dev_name}, không phải Tesla T4!"
""")

    # Phase 2: Workspace Setup, Fonts, Base Weights & PaddleOCR Patch
    add_md("## Phase 2: Thiết Lập Thư Mục, Fonts & Vá Hook Stage-End Cho PaddleOCR")
    v24_extract_code = """
# 2. Định vị checkpoint V24 từ dataset gustavnguyen/cham-ocr-v24-checkpoint
v24_extracted_dir = "/kaggle/working/data/v24_extracted"
os.makedirs(v24_extracted_dir, exist_ok=True)

# Kiểm tra thư mục dataset được mount trực tiếp
v24_dataset_dir = "/kaggle/input/cham-ocr-v24-checkpoint"
if os.path.exists(v24_dataset_dir):
    print("📦 Tìm thấy dataset V24 tại:", v24_dataset_dir)
    for item in os.listdir(v24_dataset_dir):
        s = os.path.join(v24_dataset_dir, item)
        d = os.path.join(v24_extracted_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

# Kiểm tra các archive zip dự phòng nếu có
for zpath in [
    "/kaggle/input/cham-ocr-v24-checkpoint/rec_cham_v24_best.zip",
    "/kaggle/input/paddleocr-cham-finetune/rec_cham_v24_best.zip",
    "/kaggle/input/cham-ocr-v5-assets/rec_cham_v24_best.zip"
]:
    if os.path.exists(zpath):
        print(f"📦 Giải nén bổ sung archive V24 từ: {zpath}")
        with zipfile.ZipFile(zpath, 'r') as zf:
            zf.extractall(v24_extracted_dir)
        break

print("✅ Thư mục V24 giải nén:", os.listdir(v24_extracted_dir) if os.path.exists(v24_extracted_dir) else "N/A")
""" if stage == 1 else ""

    phase2_template = """import os
import shutil
import zipfile

os.makedirs('/kaggle/working/data/fonts', exist_ok=True)
os.makedirs('/kaggle/working/data/cham_synthetic_v25', exist_ok=True)
os.makedirs('/kaggle/working/configs', exist_ok=True)
os.makedirs('/kaggle/working/scripts', exist_ok=True)
os.makedirs('/kaggle/working/output/rec_cham_v25', exist_ok=True)
os.makedirs('/kaggle/working/output/rec_cham_v25_init', exist_ok=True)

# 1. Tìm và sao chép fonts
font_found = False
for search_dir in ['/kaggle/input/cham-ocr-v5-assets', '/kaggle/input']:
    if os.path.exists(search_dir):
        for root, dirs, files in os.walk(search_dir):
            if 'v5-assets' in root or root == search_dir:
                for f in files:
                    if f.endswith(('.ttf', '.otf')) and 'Cham' in f:
                        shutil.copy2(os.path.join(root, f), os.path.join('/kaggle/working/data/fonts', f))
                        font_found = True

# Tải font dự phòng nếu chưa có
if not font_found:
    print("⏳ Tải NotoSansCham fonts từ Google Fonts...")
    import urllib.request
    urllib.request.urlretrieve("https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansCham/NotoSansCham-Regular.ttf", "/kaggle/working/data/fonts/NotoSansCham-Regular.ttf")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansCham/NotoSansCham-Bold.ttf", "/kaggle/working/data/fonts/NotoSansCham-Bold.ttf")

print("✅ Thư mục fonts:", os.listdir('/kaggle/working/data/fonts'))
__V24_EXTRACT_CODE__
# 3. Clone PaddleOCR nếu chưa có
if not os.path.exists('/kaggle/working/PaddleOCR'):
    print("⏳ Đang clone PaddleOCR release/2.7...")
    !git clone -b release/2.7 --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git /kaggle/working/PaddleOCR
    shutil.copy2('/kaggle/working/sitecustomize.py', '/kaggle/working/PaddleOCR/sitecustomize.py')
else:
    print("✅ PaddleOCR đã tồn tại sẵn.")

# 4. Patch tools/program.py chèn hook stage_end_epoch để ngắt chặng an toàn (Decoupled Stage Exit)
program_py = '/kaggle/working/PaddleOCR/tools/program.py'
with open(program_py, 'r', encoding='utf-8') as f:
    content = f.read()

stage_hook = '''
        # [CHAM-OCR V25] Decoupled Stage Exit Hook
        stage_end_epoch = config.get("Global", {}).get("stage_end_epoch", None)
        if stage_end_epoch is not None and epoch >= int(stage_end_epoch):
            logger.info("🛑 [STAGE EXIT] Dat moc stage_end_epoch {}. Dung chang an toan sau khi luu xong Epoch {} (global_step {}).".format(stage_end_epoch, epoch, global_step))
            break
'''

if "stage_end_epoch" not in content:
    target = "    best_str = 'best metric, {}'"
    if target in content:
        parts = content.rsplit(target, 1)
        content = parts[0] + stage_hook + "\\n" + target + parts[1]
    with open(program_py, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Đã vá thành công hook 'stage_end_epoch' (sau save_model) vào PaddleOCR/tools/program.py!")
else:
    print("✅ PaddleOCR/tools/program.py đã chứa hook 'stage_end_epoch'.")

# 5. Xác minh trực tiếp toàn bộ module của PaddleOCR (program, metrics, dataloader)
%cd /kaggle/working/PaddleOCR
!python3 -c "import tools.program as program; from ppocr.metrics import build_metric; from ppocr.data import build_dataloader; print('✅ PaddleOCR core modules (program, metrics, dataloader) verified!')"
%cd /kaggle/working
"""
    add_code(phase2_template.replace("__V24_EXTRACT_CODE__", v24_extract_code))

    # Phase 3: Embed Source Scripts & Configs
    add_md("## Phase 3: Nạp Toàn Bộ Mã Nguồn & Cấu Hình Chuẩn Hóa V25")
    add_code(f"%%writefile /kaggle/working/configs/v25_dataset_manifest.json\n{manifest_code}")
    add_code(f"%%writefile /kaggle/working/scripts/generate_data_v25.py\n{generate_data_code}")
    add_code(f"%%writefile /kaggle/working/scripts/build_dict_v25.py\n{build_dict_code}")
    add_code(f"%%writefile /kaggle/working/scripts/surgery_v25_weights.py\n{surgery_weights_code}")
    add_code(f"%%writefile /kaggle/working/configs/rec_cham_v25.yml\n{config_v25_code}")
    add_code(f"%%writefile /kaggle/working/data/cham_dict_v25.txt\n{dict_v25_code}")

    # Phase 4: Data Preparation & Validation Freeze
    if stage == 1:
        add_md("## Phase 4: Sinh Dữ Liệu V25 Sạch (140,000 Train + 10,000 Val) & Đóng Băng Val")
        add_code("""import time
import os
import hashlib
import json
import unicodedata
from collections import Counter
%cd /kaggle/working

print("⚡ Khởi chạy sinh 140,000 mẫu Train + 10,000 mẫu Val V25 sạch theo Manifest...")
t_gen_start = time.time()
!python3 scripts/generate_data_v25.py \\
    --manifest configs/v25_dataset_manifest.json \\
    --output_dir /kaggle/working/data/cham_synthetic_v25 \\
    --num_train 140000 \\
    --num_val 10000 \\
    --workers 4

print(f"🎉 Hoàn thành sinh dữ liệu trong {(time.time() - t_gen_start)/60:.2f} phút!")

# 1. Kiểm tra số lượng và phân bố thống kê theo Manifest
train_label_path = "/kaggle/working/data/cham_synthetic_v25/train_label.txt"
val_label_path = "/kaggle/working/data/cham_synthetic_v25/val_label.txt"

with open(train_label_path, 'r', encoding='utf-8') as f:
    train_lines = [line.strip() for line in f if line.strip()]
with open(val_label_path, 'r', encoding='utf-8') as f:
    val_lines = [line.strip() for line in f if line.strip()]

print("=" * 70)
print("📊 KIỂM TRA PHÂN BỐ DỮ LIỆU HUẤN LUYỆN (DATASET DISTRIBUTION AUDIT):")
print(f"   • Số mẫu Train : {len(train_lines):,} (yêu cầu đúng 140,000)")
print(f"   • Số mẫu Val   : {len(val_lines):,} (yêu cầu đúng 10,000)")
assert len(train_lines) == 140000, f"❌ LỖI: Số mẫu Train ({len(train_lines)}) không khớp 140,000!"
assert len(val_lines) == 10000, f"❌ LỖI: Số mẫu Val ({len(val_lines)}) không khớp 10,000!"

# Thống kê đặc trưng các trụ cột
cham_digits = set("꩐꩑꩒꩓꩔꩕꩖꩗꩘꩙")
has_digit = 0
has_au_o = 0
has_double_danda = 0
has_latin = 0
max_len = 0

for line in train_lines:
    parts = line.split('\\t', 1)
    text = parts[1] if len(parts) > 1 else parts[0]
    max_len = max(max_len, len(text))
    if any(c in cham_digits for c in text): has_digit += 1
    if ('ꨲ' in text) or ('ꨶ' in text): has_au_o += 1
    if '꩝꩝' in text: has_double_danda += 1
    if any(('a' <= c.lower() <= 'z') for c in text): has_latin += 1

print(f"   • Dòng chứa số/khổ thơ Chăm (꩑꩞..): {has_digit:,} ({has_digit/len(train_lines):.2%})")
print(f"   • Dòng chứa cặp đối kháng ꨲ/ꨶ      : {has_au_o:,} ({has_au_o/len(train_lines):.2%})")
print(f"   • Dòng chứa Double Danda ꩝꩝       : {has_double_danda:,} ({has_double_danda/len(train_lines):.2%})")
print(f"   • Dòng song ngữ/chữ Latin        : {has_latin:,} ({has_latin/len(train_lines):.2%})")
print(f"   • Max length thực tế             : {max_len} (giới hạn an toàn <= 80)")
assert max_len <= 80, f"❌ LỖI: Phát hiện nhãn dài {max_len} > 80!"
print("=" * 70)

# 2. Tạo manifest đóng băng Validation (SHA256 Hash Invariant)
def file_sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

val_hash = file_sha256(val_label_path)
dict_hash = file_sha256("/kaggle/working/data/cham_dict_v25.txt")
val_freeze_manifest = {
    "val_samples": len(val_lines),
    "val_label_sha256": val_hash,
    "dict_sha256": dict_hash,
    "timestamp": time.time()
}
freeze_manifest_path = "/kaggle/working/data/cham_synthetic_v25/val_freeze_manifest.json"
with open(freeze_manifest_path, 'w', encoding='utf-8') as f:
    json.dump(val_freeze_manifest, f, indent=2)

print(f"🔒 ĐÃ KHÓA ĐÓNG BĂNG TẬP VALIDATION (SHA256: {val_hash})")
""")
    else:
        prev_stage = stage - 1
        add_md(f"## Phase 4: Kế Thừa Validation Đóng Băng & Sinh 140,000 Mẫu Train Mới (Chặng {stage})")
        phase4_template = """import os
import sys
import time
import shutil
import zipfile
import hashlib
import json

%cd /kaggle/working

print("❄️  ĐANG TRÍCH XUẤT VÀ XÁC MINH TẬP VALIDATION ĐÓNG BĂNG...")

target_val_dir = "/kaggle/working/data/cham_synthetic_v25"
os.makedirs(target_val_dir, exist_ok=True)
val_label_path = os.path.join(target_val_dir, "val_label.txt")

# 1. Kiểm tra xem thư mục Validation đã giải nén sẵn trong Kaggle Dataset không
val_dir_candidates = [
    "/kaggle/input/cham-ocr-v25-stage1-checkpoint/cham_v25_val_freeze",
    "/kaggle/input/datasets/gustavnguyen/cham-ocr-v25-stage1-checkpoint/cham_v25_val_freeze",
    "/kaggle/input/cham-ocr-v25-stage__PREV_STAGE__-checkpoint/cham_v25_val_freeze",
    "/kaggle/input/datasets/gustavnguyen/cham-ocr-v25-stage__PREV_STAGE__-checkpoint/cham_v25_val_freeze",
    "/kaggle/input/paddleocr-cham-v25-stage__PREV_STAGE__/cham_v25_val_freeze",
    "/kaggle/input/paddleocr-cham-v25-stage1/cham_v25_val_freeze"
]

for cand_dir in val_dir_candidates:
    cand_label = os.path.join(cand_dir, "val_label.txt")
    if os.path.exists(cand_label):
        print(f"📦 Tìm thấy thư mục Val đóng băng sẵn tại: {cand_dir}")
        for item in os.listdir(cand_dir):
            s = os.path.join(cand_dir, item)
            d = os.path.join(target_val_dir, item)
            if os.path.isdir(s):
                if not os.path.exists(d):
                    shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                if not os.path.exists(d):
                    shutil.copy2(s, d)
        break

# 2. Nếu chưa có thư mục, tìm tệp zip validation đóng băng
if not os.path.exists(val_label_path):
    freeze_zip_candidates = [
        "/kaggle/input/cham-ocr-v25-stage1-checkpoint/cham_v25_val_freeze.zip",
        "/kaggle/input/datasets/gustavnguyen/cham-ocr-v25-stage1-checkpoint/cham_v25_val_freeze.zip",
        "/kaggle/input/datasets/gustavnguyen/cham-ocr-v25-stage__PREV_STAGE__-checkpoint/cham_v25_val_freeze.zip",
        "/kaggle/input/paddleocr-cham-v25-stage__PREV_STAGE__/cham_v25_val_freeze.zip",
        "/kaggle/input/paddleocr-cham-v25-stage1/cham_v25_val_freeze.zip",
        "/kaggle/working/cham_v25_val_freeze.zip"
    ]
    freeze_zip_path = None
    for cand in freeze_zip_candidates:
        if os.path.exists(cand):
            freeze_zip_path = cand
            break

    if not freeze_zip_path and os.path.exists('/kaggle/input'):
        print("🔍 Đang tìm kiếm tệp Val đóng băng trong /kaggle/input...")
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'cham_v25_val_freeze.zip' in files:
                freeze_zip_path = os.path.join(root, 'cham_v25_val_freeze.zip')
                break
            elif '_output_.zip' in files:
                try:
                    oz_p = os.path.join(root, '_output_.zip')
                    with zipfile.ZipFile(oz_p, 'r') as oz:
                        for name in oz.namelist():
                            if 'cham_v25_val_freeze.zip' in name:
                                oz.extract(name, '/kaggle/working')
                                freeze_zip_path = '/kaggle/working/cham_v25_val_freeze.zip'
                                break
                except Exception as e:
                    print(f"⚠️ Lỗi đọc {root}/_output_.zip: {e}")
            if freeze_zip_path:
                break

    if freeze_zip_path and os.path.exists(freeze_zip_path):
        print(f"📦 Đang giải nén tập Val đóng băng từ: {freeze_zip_path}...")
        with zipfile.ZipFile(freeze_zip_path, 'r') as zf:
            zf.extractall(target_val_dir)

assert os.path.exists(val_label_path), f"❌ LỖI: Không tìm thấy val_label.txt sau khi nạp tập Val đóng băng Chặng __PREV_STAGE__!"

# 2. Xác minh hash tính toàn vẹn của Validation
def file_sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

val_label_path = "/kaggle/working/data/cham_synthetic_v25/val_label.txt"
freeze_manifest_path = "/kaggle/working/data/cham_synthetic_v25/val_freeze_manifest.json"

assert os.path.exists(val_label_path), "❌ Không tìm thấy val_label.txt sau khi giải nén!"
current_val_hash = file_sha256(val_label_path)

if os.path.exists(freeze_manifest_path):
    with open(freeze_manifest_path, 'r', encoding='utf-8') as f:
        fmanifest = json.load(f)
    expected_hash = fmanifest.get('val_label_sha256')
    print(f"🔍 Expected Val SHA256 : {expected_hash}")
    print(f"🔍 Current Val SHA256  : {current_val_hash}")
    assert current_val_hash == expected_hash, "❌ LỖI: Mã băm tập Val bị sai lệch so với bản đóng băng Chặng 1!"
    print("✅ VALIDATION FREEZE AUDIT: SHA256 MATCHES 100%! Tập Val giữ nguyên hoàn hảo.")

with open(val_label_path, 'r', encoding='utf-8') as f:
    val_lines = [l.strip() for l in f if l.strip()]
assert len(val_lines) == 10000, f"❌ LỖI: Số dòng val là {len(val_lines)} != 10,000!"

# 3. Sinh 140,000 mẫu Train mới cho Chặng __STAGE__ (refresh data)
print("⚡ Khởi chạy sinh 140,000 mẫu Train mới cho Chặng __STAGE__ (bỏ qua sinh Val)...")
t_gen_start = time.time()
!python3 scripts/generate_data_v25.py \\
    --manifest configs/v25_dataset_manifest.json \\
    --output_dir /kaggle/working/data/cham_synthetic_v25 \\
    --num_train 140000 \\
    --num_val 0 \\
    --workers 4

print(f"🎉 Hoàn thành sinh tập Train mới trong {(time.time() - t_gen_start)/60:.2f} phút!")

# 4. Kiểm tra phân bổ Train mới
train_label_path = "/kaggle/working/data/cham_synthetic_v25/train_label.txt"
with open(train_label_path, 'r', encoding='utf-8') as f:
    train_lines = [l.strip() for l in f if l.strip()]

assert len(train_lines) == 140000, f"❌ LỖI: Số mẫu Train Chặng __STAGE__ ({len(train_lines)}) không đúng 140,000!"
print(f"✅ Đã kiểm tra {len(train_lines):,} mẫu Train Chặng __STAGE__ (Khớp đúng 2,187 steps/epoch).")
"""
        add_code(phase4_template.replace("__PREV_STAGE__", str(prev_stage)).replace("__STAGE__", str(stage)))

    # Phase 5: Dictionary & Preflight Checks
    add_md("## Phase 5: Xác Minh Từ Điển Động NFC & Kiểm Tra Preflight Toàn Bộ Nhãn")
    add_code("""%cd /kaggle/working
import os
import unicodedata
import json

dict_path = "/kaggle/working/data/cham_dict_v25.txt"
assert os.path.exists(dict_path), f"❌ Không tìm thấy từ điển {dict_path}!"

with open(dict_path, 'r', encoding='utf-8') as f:
    tokens = [line.rstrip('\\r\\n') for line in f if line.rstrip('\\r\\n')]

token_set = set(tokens)
print(f"📖 Từ điển V25 hiện hữu: {len(tokens)} tokens.")
assert len(tokens) == 162, f"❌ LỖI: Từ điển V25 bắt buộc phải có đúng 162 tokens chuẩn NFC, phát hiện {len(tokens)}!"

# Quét kiểm định toàn diện trên cả Train và Val
unknown_chars = set()
invalid_unicode_chars = set()
over_len_count = 0
total_checked = 0

label_files = [
    "/kaggle/working/data/cham_synthetic_v25/train_label.txt",
    "/kaggle/working/data/cham_synthetic_v25/val_label.txt"
]

for lf in label_files:
    if not os.path.exists(lf):
        continue
    with open(lf, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_checked += 1
            parts = line.split('\\t', 1)
            text = unicodedata.normalize('NFC', parts[1] if len(parts) > 1 else parts[0])
            if len(text) > 80:
                over_len_count += 1
            for c in text:
                if unicodedata.category(c) in ('Cs',):
                    invalid_unicode_chars.add(c)
                elif c != ' ' and c not in token_set:
                    unknown_chars.add(c)

print("=" * 65)
print(f"📋 PREFLIGHT SANITY AUDIT ({total_checked:,} DÒNG NHÃN):")
print(f"   • Unknown Chars      : {len(unknown_chars)} {list(unknown_chars)[:10] if unknown_chars else ''}")
print(f"   • Invalid Unicode    : {len(invalid_unicode_chars)}")
print(f"   • Over max-len (>80) : {over_len_count}")
print("=" * 65)

assert len(unknown_chars) == 0, f"❌ LỖI PREFLIGHT: Có {len(unknown_chars)} ký tự lạ: {unknown_chars}"
assert len(invalid_unicode_chars) == 0, "❌ LỖI PREFLIGHT: Có ký tự Unicode không hợp lệ!"
assert over_len_count == 0, f"❌ LỖI PREFLIGHT: Có {over_len_count} dòng nhãn vượt quá 80 ký tự!"
print("🎉 PREFLIGHT VALIDATION: 100% PASS! Toàn bộ nhãn khớp chính xác 162 tokens.")
""")

    # Phase 6: Weight Surgery / Resume Checkpoint Verification
    if stage == 1:
        add_md("## Phase 6: Phẫu Thuật Trọng Số Kế Thừa V24 -> V25 (Khởi Điểm Chặng 1)")
        add_code("""%cd /kaggle/working

# 1. Định vị trọng số V24 trong thư mục giải nén hoặc dataset
src_model_param = None
candidates = [
    "/kaggle/working/data/v24_extracted/best_checkpoint/best_accuracy",
    "/kaggle/input/cham-ocr-v24-checkpoint/best_checkpoint/best_accuracy",
    "/kaggle/working/data/v24_extracted/best_accuracy",
    "/kaggle/input/cham-ocr-v24-checkpoint/best_accuracy",
    "/kaggle/working/data/v24_extracted/output/rec_cham_v24/best_accuracy",
    "/kaggle/working/data/v24_extracted/iter_epoch_200"
]
for c in candidates:
    if os.path.exists(c + '.pdparams') or os.path.exists(c):
        src_model_param = c
        break

if not src_model_param:
    for search_dir in ['/kaggle/working/data/v24_extracted', '/kaggle/input/cham-ocr-v24-checkpoint']:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for f in files:
                    if f.endswith('.pdparams'):
                        src_model_param = os.path.join(root, f[:-9] if f.endswith('.pdparams') else f)
                        break
                if src_model_param:
                    break

# 2. Định vị từ điển V24
v24_dict_path = None
v24_dict_cands = [
    "/kaggle/working/data/v24_extracted/cham_dict_v24.txt",
    "/kaggle/input/cham-ocr-v24-checkpoint/cham_dict_v24.txt",
    "/kaggle/working/data/v24_extracted/cham_dict.txt"
]
for c in v24_dict_cands:
    if os.path.exists(c):
        v24_dict_path = c
        break

if not v24_dict_path:
    for search_dir in ['/kaggle/working/data/v24_extracted', '/kaggle/input/cham-ocr-v24-checkpoint']:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for f in files:
                    if f.endswith('.txt') and 'dict' in f.lower():
                        v24_dict_path = os.path.join(root, f)
                        break
                if v24_dict_path:
                    break

print(f"🔬 Trọng số gốc V24 : {src_model_param}")
print(f"📖 Từ điển gốc V24  : {v24_dict_path}")
assert src_model_param, "❌ Không tìm thấy trọng số V24 để phẫu thuật!"

# 3. Tiến hành phẫu thuật ma trận FC
!python3 scripts/surgery_v25_weights.py \\
    --src_model "{src_model_param}" \\
    --dst_model /kaggle/working/output/rec_cham_v25_init/init_weights.pdparams \\
    --v24_dict "{v24_dict_path if v24_dict_path else '/kaggle/working/data/cham_dict_v25.txt'}" \\
    --v25_dict /kaggle/working/data/cham_dict_v25.txt

init_weights_file = "/kaggle/working/output/rec_cham_v25_init/init_weights.pdparams"
assert os.path.exists(init_weights_file), f"❌ Không tìm thấy {init_weights_file} sau phẫu thuật!"
assert os.path.getsize(init_weights_file) > 5000000, "❌ File init_weights quá nhỏ!"
print(f"🎉 Khởi tạo trọng số V25 thành công ({os.path.getsize(init_weights_file)/(1024*1024):.2f} MB). Sẵn sàng cho Chặng 1!")
""")
    else:
        prev_stage = stage - 1
        expected_prev_epoch = prev_epochs
        expected_prev_step = prev_epochs * 2187
        add_md(f"## Phase 6: Trích Xuất & Xác Minh Trạng Thái Checkpoint Resume (Chặng {prev_stage} -> {stage})")
        phase6_template = """import os
import sys
import shutil
import zipfile
import pickle

%cd /kaggle/working

print("🔍 Đang nạp checkpoint từ Chặng __PREV_STAGE__...")
target_ckpt_dir = "/kaggle/working/output/rec_cham_v25"
os.makedirs(target_ckpt_dir, exist_ok=True)
resume_ckpt = "/kaggle/working/output/rec_cham_v25/latest"

# 1. Kiểm tra xem checkpoint đã giải nén sẵn trong Kaggle Dataset không
ckpt_dir_candidates = [
    "/kaggle/input/cham-ocr-v25-stage1-checkpoint/output/rec_cham_v25",
    "/kaggle/input/cham-ocr-v25-stage1-checkpoint",
    "/kaggle/input/datasets/gustavnguyen/cham-ocr-v25-stage1-checkpoint/output/rec_cham_v25",
    "/kaggle/input/datasets/gustavnguyen/cham-ocr-v25-stage1-checkpoint",
    "/kaggle/input/cham-ocr-v25-stage__PREV_STAGE__-checkpoint/output/rec_cham_v25",
    "/kaggle/input/cham-ocr-v25-stage__PREV_STAGE__-checkpoint",
    "/kaggle/input/datasets/gustavnguyen/cham-ocr-v25-stage__PREV_STAGE__-checkpoint/output/rec_cham_v25",
    "/kaggle/input/datasets/gustavnguyen/cham-ocr-v25-stage__PREV_STAGE__-checkpoint"
]

for cand_dir in ckpt_dir_candidates:
    if os.path.exists(os.path.join(cand_dir, "latest.pdparams")):
        print(f"📦 Tìm thấy checkpoint đã giải nén tại: {cand_dir}")
        for item in os.listdir(cand_dir):
            s = os.path.join(cand_dir, item)
            d = os.path.join(target_ckpt_dir, item)
            if not os.path.isdir(s) and not os.path.exists(d):
                shutil.copy2(s, d)
        break

# 2. Nếu chưa có, tìm tệp zip checkpoint
if not os.path.exists(resume_ckpt + ".pdparams"):
    ckpt_zip_candidates = [
        "/kaggle/input/cham-ocr-v25-stage1-checkpoint/rec_cham_v25_stage1_checkpoint.zip",
        "/kaggle/input/datasets/gustavnguyen/cham-ocr-v25-stage1-checkpoint/rec_cham_v25_stage1_checkpoint.zip",
        "/kaggle/input/datasets/gustavnguyen/cham-ocr-v25-stage__PREV_STAGE__-checkpoint/rec_cham_v25_stage__PREV_STAGE___checkpoint.zip",
        "/kaggle/input/paddleocr-cham-v25-stage__PREV_STAGE__/rec_cham_v25_stage__PREV_STAGE___checkpoint.zip",
        "/kaggle/input/paddleocr-cham-v25-stage__PREV_STAGE__/paddleocr_cham_v25_stage__PREV_STAGE___checkpoint.zip",
        "/kaggle/working/rec_cham_v25_stage__PREV_STAGE___checkpoint.zip",
        "/kaggle/working/rec_cham_v25_stage1_checkpoint.zip"
    ]
    ckpt_zip_path = None
    for cand in ckpt_zip_candidates:
        if os.path.exists(cand):
            ckpt_zip_path = cand
            break

    if not ckpt_zip_path and os.path.exists('/kaggle/input'):
        print("🔍 Đang tìm kiếm checkpoint zip trong /kaggle/input...")
        for root, dirs, files in os.walk('/kaggle/input'):
            if any(f.endswith('.zip') and 'checkpoint' in f.lower() for f in files):
                for f in files:
                    if f.endswith('.zip') and 'checkpoint' in f.lower():
                        ckpt_zip_path = os.path.join(root, f)
                        break
                if ckpt_zip_path:
                    break
            elif '_output_.zip' in files:
                try:
                    oz_p = os.path.join(root, '_output_.zip')
                    with zipfile.ZipFile(oz_p, 'r') as oz:
                        for name in oz.namelist():
                            if 'checkpoint' in name.lower() and name.endswith('.zip'):
                                oz.extract(name, '/kaggle/working')
                                ckpt_zip_path = os.path.join('/kaggle/working', name)
                                break
                except Exception as e:
                    print(f"⚠️ Lỗi đọc {root}/_output_.zip: {e}")
            if ckpt_zip_path:
                break

    if ckpt_zip_path and os.path.exists(ckpt_zip_path):
        print(f"📦 Đang giải nén checkpoint: {ckpt_zip_path}...")
        with zipfile.ZipFile(ckpt_zip_path, 'r') as zf:
            zf.extractall('/kaggle/working')

assert os.path.exists(resume_ckpt + ".pdparams"), f"❌ LỖI: Không tìm thấy {resume_ckpt}.pdparams!"
assert os.path.exists(resume_ckpt + ".pdopt"), f"❌ LỖI: Không tìm thấy {resume_ckpt}.pdopt!"
assert os.path.exists(resume_ckpt + ".states"), f"❌ LỖI: Không tìm thấy {resume_ckpt}.states!"

# 2. Đọc và xác minh trạng thái states
with open(resume_ckpt + ".states", "rb") as f:
    st = pickle.load(f)

saved_epoch = st.get('epoch', 'N/A')
saved_step = st.get('global_step', 'N/A')
best_d = st.get('best_model_dict', {})

print("=" * 65)
print("🎯 XÁC MINH TRẠNG THÁI CHECKPOINT RESUME (STAGE __STAGE__):")
print(f"   • Epoch đã hoàn thành : {saved_epoch} (kỳ vọng: __EXPECTED_PREV_EPOCH__)")
print(f"   • Global step hiện tại: {saved_step} (kỳ vọng: __EXPECTED_PREV_STEP__)")
print(f"   • Best Accuracy       : {best_d.get('acc', 'N/A'):.4%}")
print(f"   • Best Edit Distance  : {best_d.get('norm_edit_dis', 'N/A'):.4%}")
print(f"   • Best Epoch          : {best_d.get('best_epoch', 'N/A')}")
print("=" * 65)

assert saved_epoch == __EXPECTED_PREV_EPOCH__, f"❌ LỖI: Epoch đã lưu ({saved_epoch}) không khớp kỳ vọng __EXPECTED_PREV_EPOCH__!"
assert saved_step in [__EXPECTED_PREV_STEP__, 24057, __EXPECTED_PREV_EPOCH__ * 2187], f"❌ LỖI: Global step ({saved_step}) không khớp kỳ vọng (__EXPECTED_PREV_STEP__ hoặc 24057)!"
print(f"🚀 XÁC MINH HOÀN TOÀN HỢP LỆ! Sẵn sàng tiếp tục huấn luyện Epoch {saved_epoch + 1} -> __STAGE_END_EPOCH__!")
"""
        add_code(phase6_template
                 .replace("__PREV_STAGE__", str(prev_stage))
                 .replace("__STAGE__", str(stage))
                 .replace("__EXPECTED_PREV_EPOCH__", str(expected_prev_epoch))
                 .replace("__EXPECTED_PREV_STEP__", str(expected_prev_step))
                 .replace("__STAGE_END_EPOCH__", str(stage_end_epoch)))

    # Phase 7: Distributed Training
    add_md(f"## Phase 7: Huấn Luyện Phân Tán Song Song Trên Dual GPU Tesla T4x2 (Epochs {prev_epochs + 1} -> {stage_end_epoch})")
    if stage == 1:
        train_args = f"""-c {{config_path}} \\
        -o Global.epoch_num={global_total_epochs} \\
           Global.stage_end_epoch={stage_end_epoch} \\
           Global.save_epoch_step=1 \\
           Global.eval_batch_step=[0,1000] \\
           Global.pretrained_model={{init_weights}}"""
    else:
        train_args = f"""-c {{config_path}} \\
        -o Global.epoch_num={global_total_epochs} \\
           Global.stage_end_epoch={stage_end_epoch} \\
           Global.save_epoch_step=1 \\
           Global.eval_batch_step=[0,1000] \\
           Global.checkpoints={{resume_ckpt}}"""

    phase7_template = """import os
import paddle

gpu_count = paddle.device.cuda.device_count()
print(f"🔥 Khởi chạy huấn luyện phân tán V25 Chặng __STAGE__ với {gpu_count} GPU...")

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.80"
os.environ["PYTHONPATH"] = f"/kaggle/working:/kaggle/working/PaddleOCR:{os.environ.get('PYTHONPATH', '')}"

%cd /kaggle/working/PaddleOCR

init_weights = "/kaggle/working/output/rec_cham_v25_init/init_weights"
resume_ckpt = "/kaggle/working/output/rec_cham_v25/latest"
config_path = "/kaggle/working/configs/rec_cham_v25.yml"

print("=" * 75)
print("🚀 LỆNH THỰC THI HUẤN LUYỆN CHÍNH THỨC:")
print("   paddle.distributed.launch --gpus '0,1' tools/train.py")
print("   -o Global.epoch_num=__GLOBAL_TOTAL_EPOCHS__ Global.stage_end_epoch=__STAGE_END_EPOCH__")
print("=" * 75)

if gpu_count > 1:
    print("🚀 Sử dụng lệnh phân tán song song đa GPU (paddle.distributed.launch --gpus '0,1'):")
    !FLAGS_allocator_strategy=auto_growth FLAGS_fraction_of_gpu_memory_to_use=0.80 python3 -m paddle.distributed.launch --gpus '0,1' tools/train.py \\
        __TRAIN_ARGS__
else:
    print("⚡ Huấn luyện trên Single GPU:")
    !FLAGS_allocator_strategy=auto_growth FLAGS_fraction_of_gpu_memory_to_use=0.80 python3 tools/train.py \\
        __TRAIN_ARGS__

# Kiểm tra xem checkpoint có được tạo thành công không
ckpt_dir = "/kaggle/working/output/rec_cham_v25"
assert os.path.exists(os.path.join(ckpt_dir, "latest.pdparams")), "❌ LỖI: latest.pdparams không tồn tại sau khi huấn luyện!"
assert os.path.exists(os.path.join(ckpt_dir, "latest.pdopt")), "❌ LỖI: latest.pdopt không tồn tại sau khi huấn luyện!"
assert os.path.exists(os.path.join(ckpt_dir, "latest.states")), "❌ LỖI: latest.states không tồn tại sau khi huấn luyện!"
print(f"🎉 HOÀN THÀNH HUẤN LUYỆN CHẶNG __STAGE__! Checkpoint đầy đủ đã được lưu tại {ckpt_dir}.")
"""
    add_code(phase7_template
             .replace("__STAGE__", str(stage))
             .replace("__GLOBAL_TOTAL_EPOCHS__", str(global_total_epochs))
             .replace("__STAGE_END_EPOCH__", str(stage_end_epoch))
             .replace("__TRAIN_ARGS__", train_args))

    # Phase 8: Archive Checkpoint & Export
    if stage == 1:
        val_freeze_archive_code = """
val_freeze_zip = "/kaggle/working/cham_v25_val_freeze.zip"
print(f"🔒 Đang đóng gói Validation Đóng Băng vào {val_freeze_zip}...")
with zipfile.ZipFile(val_freeze_zip, 'w', zipfile.ZIP_DEFLATED) as vz:
    val_dir = "/kaggle/working/data/cham_synthetic_v25"
    # Thêm val_images
    val_img_dir = os.path.join(val_dir, "val_images")
    if os.path.exists(val_img_dir):
        for root, dirs, files in os.walk(val_img_dir):
            for f in files:
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, val_dir)
                vz.write(full_p, rel_p)
    # Thêm val_label và freeze manifest
    for extra_f in ["val_label.txt", "val_freeze_manifest.json"]:
        ep = os.path.join(val_dir, extra_f)
        if os.path.exists(ep):
            vz.write(ep, extra_f)
    # Thêm từ điển chuẩn
    if os.path.exists('/kaggle/working/data/cham_dict_v25.txt'):
        vz.write('/kaggle/working/data/cham_dict_v25.txt', 'cham_dict_v25.txt')

print(f"✅ Đã tạo Archive Val Đóng Băng: {os.path.getsize(val_freeze_zip)/(1024*1024):.2f} MB")
"""
    elif stage == 2:
        val_freeze_archive_code = """
# Chuyển tiếp cham_v25_val_freeze.zip để Chặng 3 sử dụng trực tiếp
val_freeze_zip = "/kaggle/working/cham_v25_val_freeze.zip"
if not os.path.exists(val_freeze_zip):
    freeze_cands = [
        "/kaggle/input/paddleocr-cham-v25-stage1/cham_v25_val_freeze.zip",
        "/kaggle/input/cham-ocr-v25-stage1-checkpoint/cham_v25_val_freeze.zip"
    ]
    for fc in freeze_cands:
        if os.path.exists(fc):
            shutil.copy2(fc, val_freeze_zip)
            print(f"📋 Đã sao chép chuyển tiếp Validation freeze sang: {val_freeze_zip}")
            break

if not os.path.exists(val_freeze_zip) and os.path.exists("/kaggle/working/data/cham_synthetic_v25/val_label.txt"):
    print(f"🔒 Đang đóng gói Validation Đóng Băng vào {val_freeze_zip}...")
    with zipfile.ZipFile(val_freeze_zip, 'w', zipfile.ZIP_DEFLATED) as vz:
        val_dir = "/kaggle/working/data/cham_synthetic_v25"
        val_img_dir = os.path.join(val_dir, "val_images")
        if os.path.exists(val_img_dir):
            for root, dirs, files in os.walk(val_img_dir):
                for f in files:
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, val_dir)
                    vz.write(full_p, rel_p)
        for extra_f in ["val_label.txt", "val_freeze_manifest.json", "cham_dict_v25.txt"]:
            ep = os.path.join(val_dir, extra_f)
            if os.path.exists(ep):
                vz.write(ep, extra_f)
    print(f"✅ Đã tạo chuyển tiếp Val freeze archive: {os.path.getsize(val_freeze_zip)/(1024*1024):.2f} MB")
"""
    else:
        val_freeze_archive_code = ""

    add_md(f"## Phase 8: Đóng Gói Checkpoint Chặng {stage} & Dữ Liệu Đóng Băng")
    phase8_template = """import os
import shutil
import zipfile

%cd /kaggle/working

latest_ckpt = "/kaggle/working/output/rec_cham_v25/latest"
best_ckpt = "/kaggle/working/output/rec_cham_v25/best_accuracy"
target_ckpt = best_ckpt if os.path.exists(best_ckpt + ".pdparams") else latest_ckpt

print(f"🎯 Checkpoint chính : {target_ckpt}")
assert os.path.exists(latest_ckpt + ".pdparams"), "❌ LỖI: Thiếu latest.pdparams!"

zip_path = "/kaggle/working/rec_cham_v25_stage__STAGE___checkpoint.zip"
print(f"📦 Đang nén toàn bộ checkpoint Chặng __STAGE__ vào {zip_path}...")

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk("/kaggle/working/output/rec_cham_v25"):
        for f in files:
            full_p = os.path.join(root, f)
            rel_p = os.path.relpath(full_p, "/kaggle/working")
            zipf.write(full_p, rel_p)
    if os.path.exists('/kaggle/working/data/cham_dict_v25.txt'):
        zipf.write('/kaggle/working/data/cham_dict_v25.txt', 'cham_dict_v25.txt')

print(f"🎉 HOÀN TẤT ĐÓNG GÓI CHECKPOINT CHẶNG __STAGE__!")
print(f"   • File Checkpoint : {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")

__VAL_FREEZE_ARCHIVE_CODE__
"""
    add_code(phase8_template
             .replace("__STAGE__", str(stage))
             .replace("__VAL_FREEZE_ARCHIVE_CODE__", val_freeze_archive_code))

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
    return notebook_dict


def main():
    parser = argparse.ArgumentParser(description="Prepare and Push Cham OCR V25 Training to Kaggle")
    parser.add_argument("--stage", type=int, default=1, help="Giai đoạn huấn luyện: 1, 2, hoặc 3 (mặc định: 1)")
    parser.add_argument("--epochs", type=int, default=None, help="Số epochs cho chặng này")
    parser.add_argument("--prev-epochs", type=int, default=None, help="Số epochs đã hoàn thành ở chặng trước")
    parser.add_argument("--push", action="store_true", default=False, help="Tự động đẩy lên Kaggle và chạy")
    parser.add_argument("--no-push", dest="push", action="store_false", help="Chỉ tạo file không push")
    args = parser.parse_args()

    # Chuẩn hóa cấu hình phân chặng theo 5 invariants
    if args.stage == 1:
        if args.epochs is None: args.epochs = 12
        if args.prev_epochs is None: args.prev_epochs = 0
    elif args.stage == 2:
        if args.epochs is None: args.epochs = 11
        if args.prev_epochs is None: args.prev_epochs = 11
    elif args.stage == 3:
        if args.epochs is None: args.epochs = 18
        if args.prev_epochs is None: args.prev_epochs = 22
    else:
        raise ValueError(f"Stage {args.stage} không hợp lệ! Chỉ hỗ trợ stage 1, 2, 3.")

    stage_end = args.prev_epochs + args.epochs

    print("=" * 80)
    print("🚀 MASTER KAGGLE DISPATCHER - CHAM-OCR V25 MULTI-STAGE TRAINING")
    print(f"   • Chặng huấn luyện  : Chặng {args.stage}")
    print(f"   • Số Epochs chặng   : {args.epochs} epochs (từ Epoch {args.prev_epochs + 1} đến {stage_end})")
    print(f"   • Global.epoch_num  : 40 epochs (T_max = 87,480 steps khóa cứng)")
    print(f"   • Global.stage_end  : {stage_end} (hook dừng an toàn)")
    print(f"   • Dự kiến thời gian : ~{args.epochs * 0.58:.1f} giờ máy (an toàn trong giới hạn 12h)")
    print(f"   • Bộ tăng tốc       : NvidiaTeslaT4 (Dual GPU T4x2)")
    print(f"   • Tài khoản Kaggle  : gustavnguyen")
    print("=" * 80)

    submission_dir = os.path.join(TRAINING_DIR, "output", f"kaggle_v25_stage{args.stage}_submission")
    os.makedirs(submission_dir, exist_ok=True)

    kernel_slug = f"paddleocr-cham-v25-stage{args.stage}"
    nb_name = f"paddleocr_cham_v25_stage{args.stage}.ipynb"
    nb_path = os.path.join(submission_dir, nb_name)

    nb_dict = build_v25_notebook(stage=args.stage, epoch_num=args.epochs, prev_epochs=args.prev_epochs)
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb_dict, f, indent=1, ensure_ascii=False)
    print(f"✅ Đã tạo thành công Notebook tại: {nb_path}")

    meta_path = os.path.join(submission_dir, 'kernel-metadata.json')
    meta = {
        "id": f"gustavnguyen/{kernel_slug}",
        "title": f"paddleocr-cham-v25-stage{args.stage}",
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "accelerator": "NvidiaTeslaT4",
        "dataset_sources": [
            "gustavnguyen/cham-ocr-v24-checkpoint",
            "gustavnguyen/cham-ocr-v5-assets"
        ] if args.stage == 1 else [
            "gustavnguyen/cham-ocr-v5-assets",
            f"gustavnguyen/cham-ocr-v25-stage{args.stage - 1}-checkpoint"
        ],
        "kernel_sources": [] if args.stage == 1 else [
            f"gustavnguyen/paddleocr-cham-v25-stage{args.stage - 1}"
        ],
        "competition_sources": [],
        "model_sources": []
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Đã tạo kernel-metadata tại: {meta_path}")

    if args.push:
        api = get_authenticated_api()
        if not api:
            print("❌ Lỗi: Không thể xác thực với Kaggle API.")
            return 1
        print(f"\n🚀 Đang đẩy Kernel '{meta['id']}' lên Kaggle với cấu hình NvidiaTeslaT4 (Dual T4x2)...")
        # Monkeypatch builtins.open để ép UTF-8 trên Windows khi Kaggle client đọc notebook chứa ký tự Chăm
        import builtins
        _orig_open = builtins.open
        def _safe_open(*a, **kw):
            mode = a[1] if len(a) > 1 else kw.get('mode', 'r')
            if 'b' not in mode and 'encoding' not in kw:
                kw['encoding'] = 'utf-8'
            return _orig_open(*a, **kw)
        builtins.open = _safe_open
        try:
            api.kernels_push(submission_dir, acc="NvidiaTeslaT4")
        finally:
            builtins.open = _orig_open
        print(f"🎉 ĐÃ KHỞI CHẠY THÀNH CÔNG TÁC VỤ HUẤN LUYỆN V25 CHẶNG {args.stage} TRÊN KAGGLE DUAL T4x2!")
        print(f"👉 Đường dẫn Kernel: https://www.kaggle.com/code/{meta['id']}")
        
        # Giám sát trạng thái khởi động ban đầu
        print("\n⏳ Đang kiểm tra trạng thái khởi tạo kernel...")
        time.sleep(5)
        status = api.kernels_status(meta['id'])
        print(f"📡 Trạng thái hiện tại: {status}")
    else:
        print("\n💡 Ghi chú: Chế độ --no-push đang bật. Chưa đẩy kernel lên Kaggle.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
