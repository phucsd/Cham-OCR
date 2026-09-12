#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle Training Automation for Cham-OCR Recognition Model V25.
Supports 3-Stage Training to safely execute under Kaggle's 12-hour limit:
  - Stage 1: Epochs 1-14 (~7.8h) with initial surgery weights
  - Stage 2: Epochs 15-27 (~7.3h) resume via Global.checkpoints
  - Stage 3: Epochs 28-40 (~7.3h) resume via Global.checkpoints & export inference

Hardware & Rules:
  - Kaggle user: "gustavnguyen" (nạp tự động qua kaggle_auth / .env)
  - Accelerator: NvidiaTeslaT4 (GPU T4x2)
  - Multi-GPU Distributed Launch: paddle.distributed.launch --gpus '0,1'
  - Save checkpoint: save_epoch_step: 1
"""

import os
import sys
import json
import argparse
import subprocess
import shutil

# Force UTF-8 standard output for Windows console
if sys.stdout and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Kaggle credentials
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)
sys.path.insert(0, SCRIPT_DIR)

from kaggle_auth import init_kaggle_auth
init_kaggle_auth()

from kaggle_ops import get_authenticated_api, monitor_kernel, download_kernel_outputs


def build_rec_v25_notebook(stage=1, previous_kernel=None):
    """Generates the Jupyter Notebook for running Cham Recognition V25 on Kaggle T4x2."""
    
    stage_title = f"# Cham OCR PP-OCRv4 Recognition Model V25 - Stage {stage}/3\n"
    
    # Define stage-specific epoch range
    if stage == 1:
        epoch_start = 1
        epoch_end = 14
        resume_cmd = "# Stage 1 starts from surgically adapted V24 weights"
    elif stage == 2:
        epoch_start = 15
        epoch_end = 27
        resume_cmd = f"# Stage 2 resumes from Stage 1: {previous_kernel}"
    else:
        epoch_start = 28
        epoch_end = 40
        resume_cmd = f"# Stage 3 resumes from Stage 2: {previous_kernel}"

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    stage_title,
                    f"Automated 3-Stage Training Pipeline on Kaggle Dual Tesla T4 (NvidiaTeslaT4).\n",
                    f"- Stage: {stage} of 3\n",
                    f"- Epochs: {epoch_start} to {epoch_end}\n",
                    f"- Architecture: PP-OCRv4 Rec (LCNetV3 + SVTR + NRTR + CTC)\n",
                    f"- Resolution: 48x480 (Image Shape [3, 48, 480])\n",
                    f"- Max Text Length: 80\n",
                    f"- Mode: AMP O1 (Mixed Precision)\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Environment & CUDA Setup\n",
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
                    "# 3. Prepare Working Directories & Fonts\n",
                    "%cd /kaggle/working/PaddleOCR\n",
                    "os.makedirs('data/fonts', exist_ok=True)\n",
                    "os.makedirs('data/cham_synthetic_v25', exist_ok=True)\n",
                    "os.makedirs('/kaggle/working/output/rec_cham_v25', exist_ok=True)\n",
                    "\n",
                    "# Download Cham fonts\n",
                    "!wget -q -O data/fonts/NotoSansCham-Regular.ttf https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansCham/NotoSansCham-Regular.ttf\n",
                    "!wget -q -O data/fonts/NotoSansCham-Bold.ttf https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansCham/NotoSansCham-Bold.ttf\n",
                    "print(\"Fonts downloaded.\")\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 4. Launch Adaptive Multi-GPU Distributed Training\n",
                    f"{resume_cmd}\n",
                    "import paddle\n",
                    "num_gpus = paddle.device.cuda.device_count()\n",
                    "config_path = '/kaggle/working/configs/rec_cham_v25.yml'\n",
                    "\n",
                    "# Multi-GPU launch command\n",
                    "if num_gpus > 1:\n",
                    "    gpus = ','.join(str(i) for i in range(num_gpus))\n",
                    "    print(f\"Launching distributed training across {num_gpus} GPUs: {gpus}...\")\n",
                    f"    !python3 -m paddle.distributed.launch --gpus '{gpus}' tools/train.py -c {{config_path}} -o Global.epoch_num={epoch_end}\n",
                    "else:\n",
                    "    print(\"Launching single GPU training...\")\n",
                    f"    !python3 tools/train.py -c {{config_path}} -o Global.epoch_num={epoch_end}\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 5. Export Stage Output & Checkpoints\n",
                    f"print(\"Packaging Stage {stage} outputs...\")\n",
                    "if os.path.exists('/kaggle/working/output/rec_cham_v25/latest'):\n",
                    f"    !zip -r /kaggle/working/rec_cham_v25_stage{stage}_checkpoint.zip /kaggle/working/output/rec_cham_v25\n",
                    f"    print(\"✅ Stage {stage} checkpoint archived successfully.\")\n",
                    "else:\n",
                    "    print(\"⚠️ Latest checkpoint directory not found.\")\n"
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

    # Add final inference export for Stage 3
    if stage == 3:
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Export Final V25 Inference Model\n",
                "print(\"🚀 Exporting complete V25 Recognition Inference Model...\")\n",
                "!python3 tools/export_model.py -c /kaggle/working/configs/rec_cham_v25.yml \\\n",
                "    -o Global.checkpoints=/kaggle/working/output/rec_cham_v25/best_accuracy \\\n",
                "       Global.save_inference_dir=/kaggle/working/output/rec_cham_inference_v25\n",
                "\n",
                "!zip -r /kaggle/working/rec_cham_inference_v25.zip /kaggle/working/output/rec_cham_inference_v25\n",
                "print(\"🎉 Final V25 Inference Model is ready for download!\")\n"
            ]
        })

    return nb


def main():
    parser = argparse.ArgumentParser(description="Kaggle 3-Stage Training Dispatcher for Cham OCR V25")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=1, help="Giai đoạn huấn luyện (1, 2 hoặc 3)")
    parser.add_argument("--push", action="store_true", help="Tự động đẩy kernel lên Kaggle và chạy")
    parser.add_argument("--monitor", action="store_true", help="Theo dõi log thời gian thực sau khi push")
    args = parser.parse_args()

    print("=" * 75)
    print(f"🚀 CHAM-OCR V25 KAGGLE DISPATCHER - GIAI ĐOẠN {args.stage}/3")
    print("=" * 75)

    api = get_authenticated_api()
    if not api:
        print("❌ Lỗi: Không thể xác thực với Kaggle API.")
        return

    work_dir = os.path.join(PROJECT_ROOT, "output", f"kaggle_rec_v25_stage{args.stage}")
    os.makedirs(work_dir, exist_ok=True)

    kernel_slug = f"cham-ocr-rec-v25-stage{args.stage}"
    nb_path = os.path.join(work_dir, f"kaggle_rec_v25_stage{args.stage}.ipynb")

    prev_kernel = f"gustavnguyen/cham-ocr-rec-v25-stage{args.stage - 1}" if args.stage > 1 else None
    nb_data = build_rec_v25_notebook(stage=args.stage, previous_kernel=prev_kernel)

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=2, ensure_ascii=False)
    print(f"📝 Đã tạo Notebook tại: {nb_path}")

    # Kaggle Kernel Metadata
    meta = {
        "id": f"gustavnguyen/{kernel_slug}",
        "title": f"Cham OCR Rec V25 Stage {args.stage}",
        "code_file": os.path.basename(nb_path),
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "accelerator": "NvidiaTeslaT4",
        "dataset_sources": [],
        "kernel_sources": [prev_kernel] if prev_kernel else []
    }

    meta_path = os.path.join(work_dir, "kernel-metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"📋 Đã tạo Kernel Metadata tại: {meta_path}")

    if args.push:
        print(f"\n🚀 Đang tải kịch bản lên Kaggle ({meta['id']})...")
        api.kernels_push(work_dir)
        print("✅ Đã khởi chạy thành công tác vụ huấn luyện trên Kaggle T4x2!")
        if args.monitor:
            print("📡 Đang kết nối giám sát tiến trình...")
            monitor_kernel(api, "gustavnguyen", kernel_slug)


if __name__ == "__main__":
    main()
