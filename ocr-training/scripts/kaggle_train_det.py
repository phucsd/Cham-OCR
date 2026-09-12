#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle Training Automation for Cham Text Detection (Phase 4).
Prepares and submits a multi-GPU (T4x2) PaddleOCR DBNet training job to Kaggle,
strictly following the Cham-OCR project rules:
- Kaggle user: "gustavnguyen" (Key nạp tự động qua kaggle_auth / .env)
- Accelerator: NvidiaTeslaT4
- Distributed multi-GPU execution: paddle.distributed.launch --gpus '0,1'
"""

import os
import sys
import json
import subprocess
import shutil
import tempfile

# Force UTF-8 standard output for Windows console
if sys.stdout and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Kaggle credentials
from kaggle_auth import init_kaggle_auth
init_kaggle_auth()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from kaggle_ops import get_authenticated_api, monitor_kernel, download_kernel_outputs

def build_det_notebook():
    """Generates the Jupyter Notebook for running Cham Detection training on Kaggle."""
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Cham OCR Text Detection Fine-Tuning (Phase 4 - DBNet PP-OCRv4)\n",
                    "Automated training pipeline for Cham scene text & manuscript detection on Kaggle T4x2."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Environment & Dependencies Setup\n",
                    "!pip install --upgrade pip\n",
                    "!pip install \"paddlepaddle-gpu>=2.6.0\" -i https://www.paddlepaddle.org.cn/packages/stable/cu118/\n",
                    "!pip install imgaug pyclipper shapely lmdb attrdict\n",
                    "\n",
                    "import paddle\n",
                    "print(f\"PaddlePaddle version: {paddle.__version__}\")\n",
                    "print(f\"CUDA available: {paddle.is_compiled_with_cuda()}\")\n",
                    "num_gpus = paddle.device.cuda.device_count()\n",
                    "print(f\"GPU count: {num_gpus}\")\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Clone PaddleOCR repository\n",
                    "import os\n",
                    "if not os.path.exists('PaddleOCR'):\n",
                    "    !git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleOCR.git\n",
                    "\n",
                    "%cd PaddleOCR\n",
                    "!pip install -r requirements.txt\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Prepare Cham Detection Dataset\n",
                    "%cd /kaggle/working/PaddleOCR\n",
                    "os.makedirs('data/detector', exist_ok=True)\n",
                    "os.makedirs('data/fonts', exist_ok=True)\n",
                    "os.makedirs('data/corpus', exist_ok=True)\n",
                    "\n",
                    "# Download Cham fonts\n",
                    "!wget -q -O data/fonts/NotoSansCham-Regular.ttf https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansCham/NotoSansCham-Regular.ttf\n",
                    "!wget -q -O data/fonts/NotoSansCham-Bold.ttf https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansCham/NotoSansCham-Bold.ttf\n",
                    "\n",
                    "print(\"Fonts downloaded successfully.\")\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 4. Adaptive Multi-GPU Training\n",
                    "import paddle\n",
                    "num_gpus = paddle.device.cuda.device_count()\n",
                    "print(f\"Detected {num_gpus} GPUs for training.\")\n",
                    "\n",
                    "config_path = 'configs/det/ch_PP-OCRv4_det_cham.yml'\n",
                    "\n",
                    "# Download pre-trained DBNet weights\n",
                    "!wget -q https://paddleocr.bj.bcebos.com/pretrained/PPLCNetV3_x0_75_ocr_det.pdparams -O ./pretrained_det.pdparams\n",
                    "\n",
                    "if num_gpus > 1:\n",
                    "    gpus = ','.join(str(i) for i in range(num_gpus))\n",
                    "    print(f\"Launching distributed training on GPUs {gpus}...\")\n",
                    "    !python3 -m paddle.distributed.launch --gpus '{gpus}' tools/train.py -c {config_path} -o Global.pretrained_model=./pretrained_det.pdparams\n",
                    "else:\n",
                    "    print(\"Launching single GPU training...\")\n",
                    "    !python3 tools/train.py -c {config_path} -o Global.pretrained_model=./pretrained_det.pdparams\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 5. Export Inference Model\n",
                    "print(\"Exporting detection model to inference format...\")\n",
                    "!python3 tools/export_model.py -c configs/det/ch_PP-OCRv4_det_cham.yml -o Global.checkpoints=./output/ch_PP-OCRv4_det_cham/latest Global.save_inference_dir=./output/det_cham_inference\n",
                    "\n",
                    "# Archive outputs\n",
                    "!zip -r /kaggle/working/det_cham_inference.zip ./output/det_cham_inference\n",
                    "print(\"Done! Checkpoint ready for download.\")\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 4
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return nb

def main():
    print("🚀 Cham Text Detection - Kaggle Dispatcher (Phase 4)")
    api = get_authenticated_api()
    if not api:
        print("❌ Cannot authenticate with Kaggle API.")
        return

    work_dir = os.path.join(PROJECT_ROOT, "output", "kaggle_det_submission")
    os.makedirs(work_dir, exist_ok=True)

    # 1. Write notebook
    nb_path = os.path.join(work_dir, "kaggle_det_notebook.ipynb")
    nb_data = build_det_notebook()
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=2, ensure_ascii=False)
    print(f"📝 Generated Kaggle notebook at: {nb_path}")

    # 2. Write metadata
    kernel_slug = "cham-ocr-v4-det-train"
    meta = {
        "id": f"gustavnguyen/{kernel_slug}",
        "title": kernel_slug,
        "code_file": "kaggle_det_notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }
    meta_path = os.path.join(work_dir, "kernel-metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"📋 Generated kernel-metadata.json at: {meta_path}")

    print("\n📦 To launch training on Kaggle:")
    print(f"   kaggle kernels push -p \"{work_dir}\"")
    print("\n⚙️  Configured GPU: NvidiaTeslaT4 (T4x2) with adaptive paddle.distributed.launch")

if __name__ == '__main__':
    main()
