import os
import sys

# NumPy 2.x compatibility monkeypatch
import numpy as np
if not hasattr(np, 'sctypes'):
    np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64], 'uint': [np.uint8, np.uint16, np.uint32, np.uint64], 'float': [np.float16, np.float32, np.float64], 'complex': [np.complex64, np.complex128], 'others': [bool, object, bytes, str]}
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'typeDict'): np.typeDict = {}

import yaml
import paddle
import shutil

# Add PaddleOCR to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PADDLEOCR_DIR = os.path.join(BASE_DIR, "ocr-studio", "PaddleOCR")
if PADDLEOCR_DIR not in sys.path:
    sys.path.insert(0, PADDLEOCR_DIR)

from ppocr.utils.export_model import export

def run_export():
    config_path = os.path.join(BASE_DIR, "ocr-training", "configs", "rec_cham_v25.yml")
    ckpt_path = os.path.join(BASE_DIR, "ocr-training", "output", "v25_stage3", "stage3_final_checkpoint", "best_accuracy")
    save_dir = os.path.join(BASE_DIR, "ocr-studio", "data", "output", "rec_cham_inference_v25")
    dict_path = os.path.join(BASE_DIR, "ocr-training", "data", "cham_dict_v25.txt")

    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config["Global"]["pretrained_model"] = ckpt_path
    config["Global"]["checkpoints"] = None
    config["Global"]["character_dict_path"] = dict_path
    config["Global"]["save_inference_dir"] = save_dir
    config["Global"]["export_with_pir"] = True

    print(f"Exporting model from checkpoint: {ckpt_path} ...")
    print(f"Destination inference dir: {save_dir} ...")
    export(config)

    # Copy dict file to save_dir as well
    dict_dest = os.path.join(save_dir, "cham_dict_v25.txt")
    shutil.copy2(dict_path, dict_dest)
    print(f"Copied dictionary to {dict_dest}")

    print("\n✅ Export V25 completed successfully! Listing files in save_dir:")
    for f in os.listdir(save_dir):
        fp = os.path.join(save_dir, f)
        print(f" - {f}: {os.path.getsize(fp):,} bytes")

if __name__ == "__main__":
    run_export()
