#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R2.1.4C Causal Continuation Runner
========================================================
Manages the end-to-end lifecycle for R2.1.4C Matched-Pair branches on Kaggle Dual Tesla T4:
- Branch A (control_continue): R2.1.4C-CONTROL-CONTINUE
- Branch B (combmark_replay): R2.1.4C-COMBMARK-REPLAY

Execution Protocol & Invariants:
1. Base Checkpoint: Physical CONTROL-E2 iter_epoch_2.pdparams (SHA256: f1cd3d6ebad0...).
2. Weights-Only Continuation: Global.checkpoints: null, Global.pretrained_model, fresh Adam, fresh Cosine.
3. 150-Step Warmup: warmup_epoch=0.5338078292 across both branches.
4. Multi-GPU Deadlock Prevention: eval_batch_step=[0, 9999999], cal_metric_during_train=false.
5. Checkpoint First-Defense Guard: archive outputs to .zip BEFORE eval/export.
6. Disable PIR export (export_with_pir=false).
7. Autonomous Stall & Deadlock Watchdog: 300s stall warning, 600s timeout kill.
8. Sequential Gating: Branch A MUST train, download, export, and evaluate on Frozen-838/blank/Stress-8
   BEFORE Branch B is launched.
"""

import os
import re
import sys
import json
import time
import zipfile
import shutil
import hashlib
import argparse
import subprocess
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import builtins
_orig_open = builtins.open
def _utf8_open(*args, **kwargs):
    mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
    if "b" not in mode and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
        kwargs["errors"] = "replace"
    return _orig_open(*args, **kwargs)
builtins.open = _utf8_open

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
from kaggle_auth import init_kaggle_auth

DATASET_SLUG = "gustavnguyen/cham-ocr-v24-1-r2-1-4c-training-assets"

CANONICAL_BASE_HASH = "f1cd3d6ebad0cc71fda4e3c2b7cb5148068cf77062ae69be9e998d27fd52ac01"
SOURCE_ZIP_HASH = "8aee2e915a01ce52bf48964298bbcc5c359950e607b4892f8a3890a9693e0333"

EXPECTED_RUNTIME_PATCHES = {
    'ppocr/data/imaug/label_ops.py': 'ba50786568c85bfbddb820fab47a20cd0fad0a05fa7ce2386f86070e73c9a49a',
    'ppocr/data/simple_dataset.py': '32ec76dd71d544428282f529fe2256b4f80105d2ddbbc4eee1a1649f640a8f0a',
    'ppocr/losses/__init__.py': '72d3ba2aff991223b08f571356bec0a0a4ba1f897924608a5cceb5dc8d3b196c',
    'ppocr/losses/rec_multi_loss.py': '02bcb07a7efa488c3610739a2177ba0b64ed4123aea4ad61f2a4d579d976b2a5',
    'ppocr/modeling/architectures/__init__.py': 'c6b14b525bc878b82528f37fa36ec85f987c1cc10dff3962d19350a2d1201b8a',
    'ppocr/optimizer/learning_rate.py': '79320cc448ca500bd02ad45f11f6c16b72418b8796d25c9d58d4de7e0248d9f4',
    'tools/train.py': '854fa116aaa25c75ccb346be38f03d1f1caf69c790f702e50eeeabc54af0b92b'
}

BRANCH_SPECS = {
    "control_continue": {
        "slug": "paddleocr-cham-v24-1-r2-1-4c-control-continue",
        "title": "PaddleOCR Cham v24.1 R2.1.4C Control Continue",
        "folder": "cham_synthetic_r2_1_4c_control_continue",
        "dataset_zip": "r2_1_4c_control_continue_dataset.zip",
        "config_file": "rec_cham_v24_1_r2_1_4c_control_continue.yml",
        "train_samples": 18000,
        "val_samples": 2000,
        "epochs": 2,
        "warmup_epoch": 0.5338078292,
        "step_each_epoch": 281,
        "warmup_steps": 150,
        "output_subpath": "rec_cham_v24_1_r2_1_4c_control_continue",
        "base_ckpt_subdir": "r2_1_2_control_epoch2",
        "base_ckpt_filename": "iter_epoch_2.pdparams",
        "base_sha": CANONICAL_BASE_HASH
    },
    "combmark_replay": {
        "slug": "paddleocr-cham-v24-1-r2-1-4c-combmark-replay",
        "title": "PaddleOCR Cham v24.1 R2.1.4C Combmark Replay",
        "folder": "cham_synthetic_r2_1_4c_combmark_replay",
        "dataset_zip": "r2_1_4c_combmark_replay_dataset.zip",
        "config_file": "rec_cham_v24_1_r2_1_4c_combmark_replay.yml",
        "train_samples": 18000,
        "val_samples": 2000,
        "epochs": 2,
        "warmup_epoch": 0.5338078292,
        "step_each_epoch": 281,
        "warmup_steps": 150,
        "output_subpath": "rec_cham_v24_1_r2_1_4c_combmark_replay",
        "base_ckpt_subdir": "r2_1_2_control_epoch2",
        "base_ckpt_filename": "iter_epoch_2.pdparams",
        "base_sha": CANONICAL_BASE_HASH,
        "dataset_slug": DATASET_SLUG
    },
    "phonotactic_combmark": {
        "slug": "paddleocr-cham-v24-1-r2-1-4d-phonotactic-combmark",
        "title": "PaddleOCR Cham v24.1 R2.1.4D Phonotactic Combmark",
        "folder": "cham_synthetic_r2_1_4d_phonotactic_combmark",
        "dataset_zip": "r2_1_4d_phonotactic_combmark_dataset.zip",
        "config_file": "rec_cham_v24_1_r2_1_4d_phonotactic_combmark.yml",
        "train_samples": 18000,
        "val_samples": 2000,
        "epochs": 2,
        "warmup_epoch": 0.5338078292,
        "step_each_epoch": 281,
        "warmup_steps": 150,
        "output_subpath": "rec_cham_v24_1_r2_1_4d_phonotactic_combmark",
        "base_ckpt_subdir": "r2_1_2_control_epoch2",
        "base_ckpt_filename": "iter_epoch_2.pdparams",
        "base_sha": CANONICAL_BASE_HASH,
        "dataset_slug": "gustavnguyen/cham-ocr-v24-1-r2-1-4d-training-assets"
    },
    "short_phonotactic_combmark": {
        "slug": "paddleocr-cham-v24-1-r2-1-4e-short-phono-comb",
        "title": "PaddleOCR Cham v24.1 R2.1.4E Short Phono Comb",
        "folder": "cham_synthetic_r2_1_4e_short_phonotactic_combmark",
        "dataset_zip": "r2_1_4e_short_phonotactic_combmark_dataset.zip",
        "config_file": "rec_cham_v24_1_r2_1_4e_short_phonotactic_combmark.yml",
        "train_samples": 18000,
        "val_samples": 2000,
        "epochs": 2,
        "warmup_epoch": 0.5338078292,
        "step_each_epoch": 281,
        "warmup_steps": 150,
        "output_subpath": "rec_cham_v24_1_r2_1_4e_short_phonotactic_combmark",
        "base_ckpt_subdir": "r2_1_2_control_epoch2",
        "base_ckpt_filename": "iter_epoch_2.pdparams",
        "base_sha": CANONICAL_BASE_HASH,
        "dataset_slug": "gustavnguyen/cham-ocr-v24-1-r2-1-4d-training-assets"
    },
    "multi_stream": {
        "slug": "paddleocr-cham-v24-1-r2-1-5-multi-stream",
        "title": "PaddleOCR Cham v24.1 R2.1.5 Multi Stream",
        "folder": "cham_synthetic_r2_1_5_multi_stream",
        "dataset_zip": "r2_1_5_multi_stream_dataset.zip",
        "config_file": "rec_cham_v24_1_r2_1_5_multi_stream.yml",
        "train_samples": 18000,
        "val_samples": 2000,
        "epochs": 2,
        "warmup_epoch": 0.5338078292,
        "step_each_epoch": 281,
        "warmup_steps": 150,
        "output_subpath": "rec_cham_v24_1_r2_1_5_multi_stream",
        "base_ckpt_subdir": "r2_1_2_control_epoch2",
        "base_ckpt_filename": "iter_epoch_2.pdparams",
        "base_sha": CANONICAL_BASE_HASH,
        "dataset_slug": "gustavnguyen/cham-ocr-v24-1-r2-1-4d-training-assets"
    },
    "surgical_polish": {
        "slug": "paddleocr-cham-v24-1-r2-1-6-surgical-polish",
        "title": "PaddleOCR Cham v24.1 R2.1.6 Surgical Polish",
        "folder": "cham_synthetic_r2_1_6_surgical_polish",
        "dataset_zip": "r2_1_6_surgical_polish_dataset.zip",
        "config_file": "rec_cham_v24_1_r2_1_6_surgical_polish.yml",
        "train_samples": 18000,
        "val_samples": 2000,
        "epochs": 2,
        "warmup_epoch": 0.5338078292,
        "step_each_epoch": 281,
        "warmup_steps": 150,
        "output_subpath": "rec_cham_v24_1_r2_1_6_surgical_polish",
        "base_ckpt_subdir": "r2_1_2_control_epoch2",
        "base_ckpt_filename": "iter_epoch_2.pdparams",
        "base_sha": CANONICAL_BASE_HASH,
        "dataset_slug": "gustavnguyen/cham-ocr-v24-1-r2-1-4d-training-assets"
    },
    "gate_closure_r2_1_7": {
        "slug": "paddleocr-cham-v24-1-r2-1-7-gate-closure",
        "title": "PaddleOCR Cham v24.1 R2.1.7 Gate Closure",
        "folder": "cham_synthetic_r2_1_7_gate_closure",
        "dataset_zip": "r2_1_7_gate_closure_dataset.zip",
        "config_file": "rec_cham_v24_1_r2_1_7_gate_closure.yml",
        "train_samples": 18000,
        "val_samples": 2000,
        "epochs": 3,
        "warmup_epoch": 0.5338078292,
        "step_each_epoch": 281,
        "warmup_steps": 150,
        "lr": 0.00003,
        "eta_min": 5e-6,
        "output_subpath": "rec_cham_v24_1_r2_1_7_gate_closure",
        "base_ckpt_subdir": "r2_1_2_control_epoch2",
        "base_ckpt_filename": "iter_epoch_2.pdparams",
        "base_sha": CANONICAL_BASE_HASH,
        "dataset_slug": "gustavnguyen/cham-ocr-v24-1-r2-1-4d-training-assets"
    },
    "stage_r3_degradation": {
        "slug": "paddleocr-cham-v24-1-r3-scan-degradation",
        "title": "PaddleOCR Cham v24.1 R3 Scan Degradation",
        "folder": "cham_synthetic_r3_degradation",
        "dataset_zip": "r3_degradation_dataset.zip",
        "config_file": "rec_cham_v24_1_r3_degradation.yml",
        "train_samples": 18000,
        "val_samples": 2000,
        "epochs": 3,
        "warmup_epoch": 0.5338078292,
        "step_each_epoch": 281,
        "warmup_steps": 150,
        "lr": 0.000015,
        "eta_min": 3e-6,
        "output_subpath": "rec_cham_v24_1_r3_degradation",
        "base_ckpt_subdir": "r2_1_best",
        "base_ckpt_filename": "best.pdparams",
        "base_sha": "cd0c4838a7942bc050509311ae65cdf0e61fff1c1c0488a033e7df70c80cedb2",
        "dataset_slug": "gustavnguyen/cham-ocr-v24-1-r2-1-4d-training-assets"
    }
}


def file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def build_notebook_cells(b_key: str) -> list:
    spec = BRANCH_SPECS[b_key]
    slug = spec["slug"]
    title = spec["title"]
    folder = spec["folder"]
    dataset_zip = spec["dataset_zip"]
    config_file = spec["config_file"]
    train_samples = spec["train_samples"]
    val_samples = spec["val_samples"]
    epochs = spec["epochs"]
    warmup_epoch = spec["warmup_epoch"]
    warmup_steps = spec["warmup_steps"]
    lr = spec.get("lr", 0.00003)
    eta_min = spec.get("eta_min", 5e-6)
    output_subpath = spec["output_subpath"]
    base_ckpt_subdir = spec["base_ckpt_subdir"]
    base_ckpt_filename = spec["base_ckpt_filename"]
    base_sha = spec["base_sha"]

    patches_json_str = json.dumps(EXPECTED_RUNTIME_PATCHES, indent=4)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# Cham-OCR v24.1 Stage R2.1.4C: {title}\n",
                "- Target Accelerator: Kaggle Dual Tesla T4 x2 (CUDA)\n",
                f"- Training Epochs: {epochs} ({epochs * spec['step_each_epoch']} steps total)\n",
                f"- Base Checkpoint: Stage R2.1.2 CONTROL Epoch 2 ({base_sha})\n",
                "- Continuation Semantics: WEIGHTS-ONLY continuation (fresh Adam + fresh Cosine schedule)\n",
                f"- Dataset: {train_samples:,} train / {val_samples:,} val samples ({train_samples + val_samples:,} total)\n",
                f"- Learning Rate: Peak 3.0e-5, Min 5.0e-6, exactly {warmup_steps} warmup steps (warmup_epoch={warmup_epoch})\n",
                "- Strict 7-file runtime patch overlay & Checkpoint First-Defense Guard\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, glob, json, hashlib, zipfile, shutil, sys, subprocess, platform, time\n",
                "\n",
                f"SOURCE_SHA = '{SOURCE_ZIP_HASH}'\n",
                f"BASE_SHA = '{base_sha}'\n",
                f"BASE_CKPT_FILENAME = '{base_ckpt_filename}'\n",
                f"BASE_CKPT_SUBDIR = '{base_ckpt_subdir}'\n",
                f"BRANCH_FOLDER = '{folder}'\n",
                f"DATASET_ZIP_NAME = '{dataset_zip}'\n",
                f"CONFIG_NAME = '{config_file}'\n",
                f"EXP_TRAIN_LINES = {train_samples}\n",
                f"EXP_VAL_LINES = {val_samples}\n",
                f"EXPECTED_PATCHES = {patches_json_str}\n",
                "\n",
                "def sha(p):\n",
                "    h = hashlib.sha256()\n",
                "    with open(p, 'rb') as f:\n",
                "        for c in iter(lambda: f.read(1024 * 1024), b''):\n",
                "            h.update(c)\n",
                "    return h.hexdigest().lower()\n",
                "\n",
                "work = '/kaggle/working'\n",
                "\n",
                "# 0. FAIL-FAST: verify the branch dataset is mounted BEFORE any source unpack / pip install,\n",
                "#    so a stale dataset version (still processing at push time) costs seconds, not minutes of GPU quota.\n",
                "_ff_zip = glob.glob(f'/kaggle/input/**/{DATASET_ZIP_NAME}', recursive=True)\n",
                "_ff_dir = [p for p in glob.glob(f'/kaggle/input/**/{BRANCH_FOLDER}', recursive=True) if os.path.isdir(p)]\n",
                "if not _ff_zip and not _ff_dir:\n",
                "    _tops = sorted(set('/'.join(p.split('/')[:6]) for p in glob.glob('/kaggle/input/**/*', recursive=True)))[:40]\n",
                "    raise FileNotFoundError(f'FAIL-FAST: dataset for {BRANCH_FOLDER} not mounted (stale dataset version?). /kaggle/input tops: {_tops}')\n",
                "print(f'✅ FAIL-FAST check passed: dataset for {BRANCH_FOLDER} is mounted ({(_ff_zip or _ff_dir)[0]})')\n",
                "\n",
                "paddle_dir = os.path.join(work, 'PaddleOCR')\n",
                "if os.path.exists(paddle_dir):\n",
                "    shutil.rmtree(paddle_dir, ignore_errors=True)\n",
                "\n",
                "# 1. Unpack clean PaddleOCR source\n",
                "source_zips = glob.glob('/kaggle/input/**/paddleocr_source.zip', recursive=True)\n",
                "source_dirs = [p for p in glob.glob('/kaggle/input/**/paddleocr_source', recursive=True) if os.path.isdir(p)]\n",
                "if source_zips:\n",
                "    szip = source_zips[0]\n",
                "    print('Found source zip:', szip, f'({os.path.getsize(szip):,} bytes)', sha(szip))\n",
                "    assert sha(szip) == SOURCE_SHA, f'PaddleOCR source zip hash mismatch: {sha(szip)}'\n",
                "    os.makedirs(paddle_dir, exist_ok=True)\n",
                "    with zipfile.ZipFile(szip) as z:\n",
                "        z.extractall(paddle_dir)\n",
                "elif source_dirs:\n",
                "    sdir = source_dirs[0]\n",
                "    print('Found source dir in /kaggle/input:', sdir)\n",
                "    shutil.copytree(sdir, paddle_dir)\n",
                "else:\n",
                "    raise FileNotFoundError('Could not find paddleocr_source in /kaggle/input')\n",
                "print('✅ PaddleOCR source initialized in', paddle_dir)\n",
                "\n",
                "# 2. Locate Base and Configs\n",
                "base_zips = glob.glob('/kaggle/input/**/r2_1_4c_base_and_configs.zip', recursive=True)\n",
                "base_dirs = [p for p in glob.glob('/kaggle/input/**/r2_1_4c_base_and_configs', recursive=True) if os.path.isdir(p)]\n",
                "assets_base = os.path.join(work, 'assets_base')\n",
                "if base_zips:\n",
                "    bzip = base_zips[0]\n",
                "    print('Found base zip:', bzip)\n",
                "    os.makedirs(assets_base, exist_ok=True)\n",
                "    with zipfile.ZipFile(bzip) as z:\n",
                "        z.extractall(assets_base)\n",
                "elif base_dirs:\n",
                "    assets_base = base_dirs[0]\n",
                "    print('Found base dir in /kaggle/input:', assets_base)\n",
                "else:\n",
                "    patch_dirs = [p for p in glob.glob('/kaggle/input/**/patches', recursive=True) if os.path.isdir(p)]\n",
                "    if patch_dirs:\n",
                "        assets_base = os.path.dirname(patch_dirs[0])\n",
                "        print('Found assets root via patches dir:', assets_base)\n",
                "    else:\n",
                "        raise FileNotFoundError('Could not find r2_1_4c_base_and_configs in /kaggle/input')\n",
                "print('✅ Base and configs located at', assets_base)\n",
                "\n",
                "# 3. MANDATORY 7-FILE RUNTIME OVERLAY BEFORE ANY IMPORT\n",
                "print('\\nApplying all 7 audited R2.1.4C runtime patches onto PaddleOCR...')\n",
                "for rel_path, exp_hash in EXPECTED_PATCHES.items():\n",
                "    src_patch = glob.glob(f'{assets_base}/**/{rel_path}', recursive=True)\n",
                "    assert len(src_patch) > 0, f'Patch source missing: {rel_path}'\n",
                "    dst_patch = os.path.join(paddle_dir, rel_path)\n",
                "    os.makedirs(os.path.dirname(dst_patch), exist_ok=True)\n",
                "    shutil.copy2(src_patch[0], dst_patch)\n",
                "    actual_h = sha(dst_patch)\n",
                "    assert actual_h == exp_hash, f'SHA256 mismatch after overlay for {rel_path}: {actual_h} != {exp_hash}'\n",
                "    print(f'  • Patched & verified: {rel_path} -> {actual_h[:12]}... OK')\n",
                "print('✅ ALL 7 AUDITED RUNTIME PATCHES OVERLAID AND PROVENANCE-VERIFIED.')\n",
                "\n",
                "# 4. Locate & Setup Branch Dataset\n",
                "ds_zips = glob.glob(f'/kaggle/input/**/{DATASET_ZIP_NAME}', recursive=True)\n",
                "if not ds_zips:\n",
                "    ds_zips = [p for p in glob.glob('/kaggle/input/**/*.zip', recursive=True) if DATASET_ZIP_NAME.replace('.zip', '') in os.path.basename(p)]\n",
                "ds_dirs = [p for p in glob.glob(f'/kaggle/input/**/{BRANCH_FOLDER}', recursive=True) if os.path.isdir(p)]\n",
                "branch_data_dir = os.path.join('/kaggle/working/data', BRANCH_FOLDER)\n",
                "os.makedirs('/kaggle/working/data', exist_ok=True)\n",
                "if ds_zips:\n",
                "    ds_zip = ds_zips[0]\n",
                "    print(f'Unpacking dataset zip: {ds_zip}...')\n",
                "    with zipfile.ZipFile(ds_zip) as z:\n",
                "        z.extractall('/kaggle/working/data')\n",
                "elif ds_dirs:\n",
                "    src_data_dir = ds_dirs[0]\n",
                "    print('Found dataset dir in /kaggle/input:', src_data_dir)\n",
                "    try:\n",
                "        os.symlink(src_data_dir, branch_data_dir)\n",
                "        print(f'✅ Symlinked {branch_data_dir} -> {src_data_dir}')\n",
                "    except Exception as e:\n",
                "        shutil.copytree(src_data_dir, branch_data_dir)\n",
                "        print(f'✅ Copied {src_data_dir} -> {branch_data_dir}')\n",
                "else:\n",
                "    raise FileNotFoundError(f'Could not find dataset for {BRANCH_FOLDER} in /kaggle/input')\n",
                "\n",
                "train_lbl = os.path.join(branch_data_dir, 'train_label.txt')\n",
                "val_lbl = os.path.join(branch_data_dir, 'val_label.txt')\n",
                "assert os.path.exists(train_lbl), f'Missing train_label.txt: {train_lbl}'\n",
                "assert os.path.exists(val_lbl), f'Missing val_label.txt: {val_lbl}'\n",
                "train_cnt = sum(1 for _ in open(train_lbl, 'r', encoding='utf-8'))\n",
                "val_cnt = sum(1 for _ in open(val_lbl, 'r', encoding='utf-8'))\n",
                "assert train_cnt == EXP_TRAIN_LINES, f'Train lines mismatch: {train_cnt} != {EXP_TRAIN_LINES}'\n",
                "assert val_cnt == EXP_VAL_LINES, f'Val lines mismatch: {val_cnt} != {EXP_VAL_LINES}'\n",
                "print(f'✅ Dataset verified: {train_cnt:,} train / {val_cnt:,} val samples.')\n",
                "\n",
                "# 5. Setup Dict, Base Checkpoint, and Config\n",
                "dict_cands = glob.glob(f'{assets_base}/**/cham_dict_v24_1.txt', recursive=True) or glob.glob('/kaggle/input/**/cham_dict_v24_1.txt', recursive=True)\n",
                "assert dict_cands, 'Could not find cham_dict_v24_1.txt in /kaggle/input'\n",
                "shutil.copy2(dict_cands[0], '/kaggle/working/data/cham_dict_v24_1.txt')\n",
                "ckpt_cands = glob.glob(f'{assets_base}/**/{BASE_CKPT_FILENAME}', recursive=True) or glob.glob(f'/kaggle/input/**/{BASE_CKPT_FILENAME}', recursive=True)\n",
                "assert ckpt_cands, f'Could not find base checkpoint {BASE_CKPT_FILENAME} in /kaggle/input'\n",
                "os.makedirs(f'/kaggle/working/data/{BASE_CKPT_SUBDIR}', exist_ok=True)\n",
                "dst_base_ckpt = f'/kaggle/working/data/{BASE_CKPT_SUBDIR}/{BASE_CKPT_FILENAME}'\n",
                "shutil.copy2(ckpt_cands[0], dst_base_ckpt)\n",
                "assert sha(dst_base_ckpt) == BASE_SHA, f'Base checkpoint hash mismatch: {sha(dst_base_ckpt)} != {BASE_SHA}'\n",
                "print('✅ Base checkpoint verified:', BASE_SHA)\n",
                "cfg_cands = glob.glob(f'{assets_base}/**/{CONFIG_NAME}', recursive=True) or glob.glob(f'/kaggle/input/**/{CONFIG_NAME}', recursive=True)\n",
                "assert cfg_cands, f'Could not find config {CONFIG_NAME} in /kaggle/input'\n",
                "os.makedirs('/kaggle/working/configs', exist_ok=True)\n",
                "shutil.copy2(cfg_cands[0], f'/kaggle/working/configs/{CONFIG_NAME}')\n",
                "print(f'✅ Config installed: /kaggle/working/configs/{CONFIG_NAME}')\n",
                "\n",
                "# 6. Setup sitecustomize.py for NumPy 2.x and visualdl fallback\n",
                "sitecustomize_code = (\n",
                "    \"import numpy as np\\n\"\n",
                "    \"if not hasattr(np, 'sctypes'):\\n\"\n",
                "    \"    np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64], 'uint': [np.uint8, np.uint16, np.uint32, np.uint64], 'float': [np.float16, np.float32, np.float64], 'complex': [np.complex64, np.complex128], 'others': [bool, object, bytes, str]}\\n\"\n",
                "    \"if not hasattr(np, 'bool'): np.bool = getattr(np, 'bool_', bool)\\n\"\n",
                "    \"if not hasattr(np, 'int'): np.int = getattr(np, 'int_', int)\\n\"\n",
                "    \"if not hasattr(np, 'float'): np.float = getattr(np, 'float64', float)\\n\"\n",
                "    \"if not hasattr(np, 'typeDict'): np.typeDict = {}\\n\"\n",
                "    \"try:\\n\"\n",
                "    \"    import visualdl\\n\"\n",
                "    \"except Exception:\\n\"\n",
                "    \"    import types, sys\\n\"\n",
                "    \"    vdl = types.ModuleType('visualdl')\\n\"\n",
                "    \"    vdl.LogWriter = type('LogWriter', (), {'__init__': lambda self, *a, **k: None, 'add_scalar': lambda self, *a, **k: None, 'close': lambda self: None})\\n\"\n",
                "    \"    sys.modules['visualdl'] = vdl\\n\"\n",
                "    \"try:\\n\"\n",
                "    \"    import paddle\\n\"\n",
                "    \"    _orig_dp = paddle.DataParallel\\n\"\n",
                "    \"    class PatchedDP(_orig_dp):\\n\"\n",
                "    \"        def __init__(self, layers, *args, **kwargs):\\n\"\n",
                "    \"            kwargs['find_unused_parameters'] = True\\n\"\n",
                "    \"            super().__init__(layers, *args, **kwargs)\\n\"\n",
                "    \"    paddle.DataParallel = PatchedDP\\n\"\n",
                "    \"except Exception:\\n\"\n",
                "    \"    pass\\n\"\n",
                "    \"print('[sitecustomize] NumPy 2.x, visualdl, and DataParallel(find_unused_parameters=True) initialized')\\n\"\n",
                ")\n",
                "import site\n",
                "tdirs = [paddle_dir, work]\n",
                "try:\n",
                "    tdirs.extend(site.getsitepackages())\n",
                "except Exception:\n",
                "    pass\n",
                "for td in tdirs:\n",
                "    try:\n",
                "        p = os.path.join(td, 'sitecustomize.py')\n",
                "        with open(p, 'w', encoding='utf-8') as f:\n",
                "            f.write(sitecustomize_code)\n",
                "        print('✅ Created sitecustomize.py in', td)\n",
                "    except Exception as e:\n",
                "        print(f'Warning: could not write sitecustomize to {td}: {e}')\n",
                "\n",
                "# Explicitly inject NumPy 2.x monkeypatch and find_unused_parameters into tools/*.py\n",
                "np_patch_code = (\n",
                "    \"import numpy as np\\n\"\n",
                "    \"if not hasattr(np, 'sctypes'):\\n\"\n",
                "    \"    np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64], 'uint': [np.uint8, np.uint16, np.uint32, np.uint64], 'float': [np.float16, np.float32, np.float64], 'complex': [np.complex64, np.complex128], 'others': [bool, object, bytes, str]}\\n\"\n",
                "    \"if not hasattr(np, 'bool'): np.bool = getattr(np, 'bool_', bool)\\n\"\n",
                "    \"if not hasattr(np, 'int'): np.int = getattr(np, 'int_', int)\\n\"\n",
                "    \"if not hasattr(np, 'float'): np.float = getattr(np, 'float64', float)\\n\"\n",
                "    \"if not hasattr(np, 'typeDict'): np.typeDict = {}\\n\"\n",
                ")\n",
                "for tool_script in ['train.py', 'eval.py', 'export_model.py']:\n",
                "    sp = os.path.join(paddle_dir, 'tools', tool_script)\n",
                "    if os.path.exists(sp):\n",
                "        with open(sp, 'r', encoding='utf-8') as f:\n",
                "            st = f.read()\n",
                "        if 'np.sctypes' not in st:\n",
                "            marker = 'from __future__ import print_function\\n'\n",
                "            if marker in st:\n",
                "                new_st = st.replace(marker, marker + '\\n' + np_patch_code + '\\n', 1)\n",
                "            else:\n",
                "                new_st = np_patch_code + '\\n' + st\n",
                "            if tool_script == 'train.py' and 'model = paddle.DataParallel(model)' in new_st:\n",
                "                new_st = new_st.replace('model = paddle.DataParallel(model)', 'model = paddle.DataParallel(model, find_unused_parameters=True)')\n",
                "                print('✅ Injected find_unused_parameters=True into tools/train.py')\n",
                "            with open(sp, 'w', encoding='utf-8') as f:\n",
                "                f.write(new_st)\n",
                "            print(f'✅ Injected NumPy 2.x patch into tools/{tool_script} after __future__ imports')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Provision compatible Python runtime if host is Python 3.13+\n",
                "import sys, os, subprocess, shutil, glob\n",
                "print('Host Python version:', sys.version)\n",
                "\n",
                "if sys.version_info >= (3, 13):\n",
                "    print('Host Python is 3.13+. Installing uv and provisioning standalone Python 3.11 for paddlepaddle-gpu 2.6...')\n",
                "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'uv'], check=True)\n",
                "    subprocess.run([sys.executable, '-m', 'uv', 'venv', '--seed', '/tmp/py311', '--python', '3.11'], check=True)\n",
                "    PY_EXE = '/tmp/py311/bin/python'\n",
                "    install_cmd = [sys.executable, '-m', 'uv', 'pip', 'install', '--python', PY_EXE, 'paddlepaddle-gpu>=2.6.0,<3.0.0', 'visualdl', 'imgaug', 'scikit-image', 'pyyaml', 'opencv-python', 'Pillow', 'shapely', 'pyclipper', 'lmdb', 'rapidfuzz', 'tqdm']\n",
                "else:\n",
                "    PY_EXE = sys.executable\n",
                "    install_cmd = [PY_EXE, '-m', 'pip', 'install', '--quiet', 'paddlepaddle-gpu>=2.6.0,<3.0.0', 'visualdl', 'imgaug', 'scikit-image', 'pyyaml', 'opencv-python', 'Pillow', 'shapely', 'pyclipper', 'lmdb', 'rapidfuzz', 'tqdm']\n",
                "\n",
                "with open('/kaggle/working/py_exe.txt', 'w') as f:\n",
                "    f.write(PY_EXE)\n",
                "\n",
                "print('Using Python executable for training & dependencies:', PY_EXE)\n",
                "os.environ['PATH'] = f\"{os.path.dirname(PY_EXE)}:{os.environ['PATH']}\"\n",
                "\n",
                "# Install runtime dependencies\n",
                "print('Installing runtime dependencies into active Python environment...')\n",
                "subprocess.run(install_cmd, check=True)\n",
                "\n",
                "# Deploy sitecustomize into active site-packages if py311 was created\n",
                "for sp in glob.glob('/tmp/py311/lib*/python*/site-packages'):\n",
                "    shutil.copy2('/kaggle/working/sitecustomize.py', os.path.join(sp, 'sitecustomize.py'))\n",
                "    print('✅ Deployed sitecustomize.py to', sp)\n",
                "\n",
                "# Preflight test inside active Python environment\n",
                "preflight_test_code = (\n",
                "    \"import sys, paddle, imgaug\\n\"\n",
                "    \"print('Python:', sys.version)\\n\"\n",
                "    \"print('Paddle:', paddle.__version__)\\n\"\n",
                "    \"print('CUDA:', paddle.is_compiled_with_cuda())\\n\"\n",
                "    \"gpu_count = paddle.device.cuda.device_count()\\n\"\n",
                "    \"print('GPU count:', gpu_count)\\n\"\n",
                "    \"for i in range(gpu_count):\\n\"\n",
                "    \"    print(f' - GPU {i}:', paddle.device.cuda.get_device_name(i))\\n\"\n",
                "    \"assert gpu_count >= 2, f'FATAL: Dual GPU required for Stage R2.1.4C, got {gpu_count}'\\n\"\n",
                "    \"\\n\"\n",
                "    \"sys.path.insert(0, '/kaggle/working/PaddleOCR')\\n\"\n",
                "    \"from ppocr.data import build_dataloader\\n\"\n",
                "    \"from ppocr.optimizer import build_lr_scheduler\\n\"\n",
                "    \"from ppocr.data.imaug.label_ops import BlankAwareMultiLabelEncode\\n\"\n",
                "    \"from ppocr.losses import BlankAwareMultiLoss\\n\"\n",
                "    \"import tools.program as program\\n\"\n",
                "    \"\\n\"\n",
                f"    \"lr_cfg = {{'name': 'Cosine', 'learning_rate': {lr}, 'eta_min': {eta_min}, 'warmup_epoch': {warmup_epoch}}}\\n\"\n",
                f"    \"s = build_lr_scheduler(lr_cfg, epochs={epochs}, step_each_epoch={spec['step_each_epoch']})\\n\"\n",
                "    \"inner = getattr(s, 'learning_rate', s)\\n\"\n",
                f"    \"assert getattr(inner, 'eta_min', None) == {eta_min}, f'eta_min not properly wired: {{getattr(inner, \\\"eta_min\\\", None)}}'\\n\"\n",
                f"    \"assert s.warmup_steps == {warmup_steps}, f'Warmup steps mismatch: {{s.warmup_steps}} != {warmup_steps}'\\n\"\n",
                f"    \"print('✅ Preflight imports & LR scheduler verified: eta_min={eta_min}, warmup_steps={warmup_steps} OK!')\\n\"\n",
                ")\n",
                "with open('/kaggle/working/preflight_check.py', 'w', encoding='utf-8') as f:\n",
                "    f.write(preflight_test_code)\n",
                "subprocess.run([PY_EXE, '/kaggle/working/preflight_check.py'], check=True)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# Launch Adaptive Multi-GPU Training for {title}\n",
                "import os, subprocess, time\n",
                "\n",
                "py_exe = open('/kaggle/working/py_exe.txt').read().strip() if os.path.exists('/kaggle/working/py_exe.txt') else 'python3'\n",
                "os.environ['PATH'] = f\"{os.path.dirname(py_exe)}:{os.environ.get('PATH', '')}\"\n",
                "os.environ['PYTHONUNBUFFERED'] = '1'\n",
                "cmd = [\n",
                "    py_exe, '-u', '-m', 'paddle.distributed.launch',\n",
                "    '--gpus', '0,1',\n",
                "    'tools/train.py',\n",
                f"    '-c', '/kaggle/working/configs/{config_file}'\n",
                "]\n",
                "print('Executing command:', ' '.join(cmd))\n",
                "t0 = time.time()\n",
                "res = subprocess.run(cmd, cwd=paddle_dir)\n",
                "train_duration = time.time() - t0\n",
                "print(f'Train process exited with code: {res.returncode} in {train_duration:.1f}s ({train_duration/60:.2f} mins)')\n",
                "assert res.returncode == 0, f'Training failed with return code {res.returncode}'\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# CHECKPOINT FIRST-DEFENSE GUARD: Archive all checkpoints to .zip immediately\n",
                "import os, glob, hashlib, shutil, json\n",
                "\n",
                f"out_dir = '/kaggle/working/output/{output_subpath}'\n",
                f"zip_base = '/kaggle/working/checkpoints_{slug}'\n",
                "print(f'Archiving checkpoints from {out_dir} -> {zip_base}.zip...')\n",
                "shutil.make_archive(zip_base, 'zip', out_dir)\n",
                "zip_path = f'{zip_base}.zip'\n",
                "print(f'✅ Checkpoint defense archive created: {zip_path} ({os.path.getsize(zip_path):,} bytes)')\n",
                "\n",
                "# Compute and log SHA256 of all generated checkpoint pdparams\n",
                "ckpts = sorted(glob.glob(f'{out_dir}/*.pdparams'))\n",
                "ckpt_hashes = {}\n",
                "for c in ckpts:\n",
                "    h = hashlib.sha256(open(c, 'rb').read()).hexdigest()\n",
                "    ckpt_hashes[os.path.basename(c)] = h\n",
                "    print(f'  • {os.path.basename(c)}: {os.path.getsize(c):,} bytes, SHA256: {h}')\n",
                f"with open('/kaggle/working/checkpoint_manifest_{slug}.json', 'w') as f:\n",
                "    json.dump(ckpt_hashes, f, indent=2)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Export inference model (with export_with_pir=False) & Offline Single-GPU Evaluation\n",
                "import os, subprocess, shutil\n",
                "\n",
                "py_exe = open('/kaggle/working/py_exe.txt').read().strip() if os.path.exists('/kaggle/working/py_exe.txt') else 'python3'\n",
                "os.environ['PATH'] = f\"{os.path.dirname(py_exe)}:{os.environ.get('PATH', '')}\"\n",
                f"infer_dir = '/kaggle/working/output/{output_subpath}_infer'\n",
                "export_cmd = [\n",
                "    py_exe, 'tools/export_model.py',\n",
                f"    '-c', '/kaggle/working/configs/{config_file}',\n",
                "    '-o',\n",
                f"    'Global.checkpoints=/kaggle/working/output/{output_subpath}/latest',\n",
                f"    'Global.save_inference_dir=/kaggle/working/output/{output_subpath}_infer',\n",
                "    'Global.export_with_pir=False'\n",
                "]\n",
                "print('Exporting inference model...')\n",
                "exp_res = subprocess.run(export_cmd, cwd=paddle_dir)\n",
                "print('Export returned:', exp_res.returncode)\n",
                "\n",
                "# Offline Single-GPU validation evaluation\n",
                "eval_cmd = [\n",
                "    py_exe, 'tools/eval.py',\n",
                f"    '-c', '/kaggle/working/configs/{config_file}',\n",
                "    '-o',\n",
                f"    'Global.checkpoints=/kaggle/working/output/{output_subpath}/latest',\n",
                "    'Global.distributed=False'\n",
                "]\n",
                "print('Running offline Single-GPU validation evaluation...')\n",
                "eval_res = subprocess.run(eval_cmd, cwd=paddle_dir)\n",
                "print('Eval returned:', eval_res.returncode)\n",
                "# Cleanup heavy source & dataset dirs from /kaggle/working to keep kernel output small\n",
                "for clean_dir in ['/kaggle/working/PaddleOCR', '/kaggle/working/data', '/kaggle/working/configs']:\n",
                "    if os.path.exists(clean_dir):\n",
                "        shutil.rmtree(clean_dir, ignore_errors=True)\n",
                "        print(f'Cleaned up {clean_dir}')\n",
                "print('🎉 STAGE R2.1.4C PIPELINE COMPLETE!')\n"
            ]
        }
    ]
    return cells


def generate_kernel_files(b_key: str, output_dir: str):
    spec = BRANCH_SPECS[b_key]
    os.makedirs(output_dir, exist_ok=True)

    cells = build_notebook_cells(b_key)
    nb = {
        "cells": cells,
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
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    nb_filename = f"{spec['slug']}.ipynb"
    nb_path = os.path.join(output_dir, nb_filename)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    # Pre-Flight Static Scope & AST Validation on generated notebook
    import ast, builtins
    defined = set(dir(builtins))
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        src = "".join(c['source'])
        tree = ast.parse(src)
        cell_assigned = set()
        cell_loaded = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                cell_assigned.add(node.name)
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, (ast.Store, ast.Param)):
                    cell_assigned.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    cell_loaded.add(node.id)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    cell_assigned.add(alias.asname or alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    cell_assigned.add(alias.asname or alias.name)
            elif isinstance(node, ast.ExceptHandler):
                if node.name:
                    cell_assigned.add(node.name)
        undefined = cell_loaded - defined - cell_assigned
        assert not undefined, f"FATAL: Undefined variable(s) {undefined} in notebook cell {i}"
        defined.update(cell_assigned)
    print(f"✅ Pre-Flight Static Scope & AST validation passed 100% for all cells in {spec['slug']}")

    meta = {
        "id": f"gustavnguyen/{spec['slug']}",
        "title": spec["title"],
        "code_file": nb_filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_gpu": True,
        "enable_gpu": True,
        "enable_internet": True,
        "accelerator": "NvidiaTeslaT4",
        "dataset_sources": [spec.get("dataset_slug", DATASET_SLUG)],
        "kernel_sources": []
    }
    meta_path = os.path.join(output_dir, "kernel-metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Generated kernel files for {b_key} at: {output_dir}")
    print(f"  • Notebook: {nb_path} ({os.path.getsize(nb_path):,} bytes)")
    print(f"  • Metadata: {meta_path}")
    return nb_path, meta_path


def upload_training_assets(bundle_name: str = "r2_1_4c_training_bundle", target_slug: str = DATASET_SLUG):
    """Upload or update Kaggle dataset with training assets."""
    u = init_kaggle_auth()
    bundle_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", bundle_name)
    meta_p = os.path.join(bundle_dir, "dataset-metadata.json")
    assert os.path.exists(meta_p), f"dataset-metadata.json missing in {bundle_dir}"

    print("=" * 80)
    print(f"🚀 UPLOADING ASSET BUNDLE ({bundle_name}) TO KAGGLE DATASET ({target_slug})")
    print("=" * 80)

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    actual_user = api.config_values.get("username")
    assert actual_user == u, f"FATAL: KaggleApi authenticated as '{actual_user}', expected '{u}'!"

    datasets = [d.ref for d in api.dataset_list(mine=True)]
    if target_slug in datasets:
        print(f"Dataset {target_slug} already exists. Creating new version...")
        api.dataset_create_version(
            folder=bundle_dir,
            version_notes=f"{bundle_name} training bundle ({datetime.now(timezone.utc).isoformat()})",
            dir_mode="zip"
        )
        print("✅ Dataset version upload submitted successfully!")
    else:
        print(f"Dataset {target_slug} not found. Creating new dataset...")
        api.dataset_create_new(
            folder=bundle_dir,
            dir_mode="zip"
        )
        print("✅ Dataset created successfully!")

    print("Waiting for Kaggle to finish processing dataset...")
    for i in range(60):
        time.sleep(10)
        try:
            st = api.dataset_status(target_slug)
            st_str = getattr(st, 'name', str(st)).lower()
        except Exception as e:
            st_str = f"processing ({e})"
        print(f"  [{ (i+1)*10 }s] Dataset status: {st_str}", flush=True)
        if "ready" in st_str:
            print("✅ Dataset is READY on Kaggle!", flush=True)
            return True
    raise RuntimeError(f"Dataset {target_slug} did not become ready within 600s!")


def push_and_run_kernel(b_key: str):
    """Push kernel to Kaggle and trigger execution."""
    u = init_kaggle_auth()
    spec = BRANCH_SPECS[b_key]
    kernel_dir = os.path.join(PROJECT_ROOT, "ocr-training", "kernels", spec["slug"])
    generate_kernel_files(b_key, kernel_dir)

    print(f"\n🚀 Pushing kernel {spec['slug']} to Kaggle...")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    actual_user = api.config_values.get("username")
    assert actual_user == u, f"FATAL: KaggleApi authenticated as '{actual_user}', expected '{u}'!"

    res = api.kernels_push(kernel_dir)
    err_msg = getattr(res, "error", None)
    if err_msg:
        raise RuntimeError(f"Kaggle kernels_push returned error: {err_msg}")
    pushed_ref = getattr(res, "ref", f"{u}/{spec['slug']}")
    print(f"✅ Kernel pushed successfully: {pushed_ref} (url: {getattr(res, 'url', 'N/A')})")
    assert f"/{u.lower()}/" in f"/{str(pushed_ref).lower()}", f"FATAL: Kernel pushed to wrong owner: {pushed_ref}"


def monitor_kernel(b_key: str, poll_interval: int = 30, max_stall_seconds: int = None):
    """
    Monitor running kernel status with autonomous stall & deadlock detection.
    """
    u = init_kaggle_auth()
    spec = BRANCH_SPECS[b_key]
    kernel_ref = f"{u}/{spec['slug']}"

    total_steps = spec['epochs'] * spec['step_each_epoch']
    est_train_seconds = int(total_steps * 2.0 + 300)
    if max_stall_seconds is None:
        max_stall_seconds = max(1200, int(est_train_seconds * 1.5))

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    actual_user = api.config_values.get("username")
    assert actual_user == u, f"FATAL: KaggleApi authenticated as '{actual_user}', expected '{u}'!"

    print("=" * 80)
    print(f"👀 MONITORING KERNEL: {kernel_ref}")
    print(f"   Branch: {b_key} | Epochs: {spec['epochs']} | Steps: {total_steps} (Est: ~{est_train_seconds/60:.1f}m | Timeout: {max_stall_seconds/60:.1f}m)")
    print("=" * 80, flush=True)

    start_time = time.time()
    last_status = None
    last_log_line_count = 0
    last_heartbeat = 0
    consecutive_none_status = 0
    last_step = -1
    last_step_time = None
    stall_warned = False

    while True:
        now = time.time()
        elapsed = now - start_time
        if elapsed >= max_stall_seconds:
            raise TimeoutError(f"CRITICAL: Kernel {kernel_ref} exceeded maximum allowable runtime ({elapsed:.0f}s > {max_stall_seconds}s). Aborting!")

        status_obj = None
        last_err = None
        for attempt in range(3):
            try:
                status_obj = api.kernels_status(kernel_ref)
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(5)
                else:
                    print(f"[{time.time()-start_time:.0f}s] Warning: error in kernels_status: {e}", flush=True)

        if status_obj is None:
            consecutive_none_status += 1
            if consecutive_none_status >= 3:
                raise RuntimeError(f"FATAL: Cannot access kernel {kernel_ref} after {consecutive_none_status} consecutive attempts: {last_err}")
            time.sleep(poll_interval)
            continue
        consecutive_none_status = 0

        raw_status = getattr(status_obj.status, 'name', str(status_obj.status))
        status = raw_status.lower()
        now = time.time()
        elapsed = now - start_time

        if status != last_status:
            print(f"[{elapsed:.0f}s] Status change: {last_status} -> {raw_status}", flush=True)
            last_status = status

        if status in ("complete", "completed"):
            print(f"\n🎉 Kernel {kernel_ref} COMPLETED SUCCESSFULLY in {elapsed/60:.1f} mins!", flush=True)
            break
        elif status in ("error", "failed"):
            fail_msg = getattr(status_obj, 'failure_message', getattr(status_obj, 'failureMessage', 'Unknown error'))
            print(f"\n❌ Kernel {kernel_ref} FAILED: {fail_msg}", flush=True)
            try:
                logs_str = api.kernels_logs(kernel_ref)
                print("Kernel Log Output:")
                print(logs_str)
            except Exception as e:
                print(f"Could not retrieve error log: {e}")
            raise RuntimeError(f"Kernel {kernel_ref} execution failed: {fail_msg}")

        # Check logs for progress
        try:
            logs_str = api.kernels_logs(kernel_ref)
            if isinstance(logs_str, str) and logs_str:
                try:
                    entries = json.loads(logs_str)
                    new_msgs = [e.get('data', '') for e in entries if isinstance(e, dict) and 'data' in e]
                    lines = ''.join(new_msgs).splitlines()
                except Exception:
                    lines = logs_str.splitlines()

                if len(lines) > last_log_line_count:
                    for nl in lines[last_log_line_count:]:
                        print(f"  | {nl}", flush=True)
                        m_step = re.search(r"global_step:\s*(\d+)", nl)
                        if m_step:
                            step_val = int(m_step.group(1))
                            if step_val > last_step:
                                last_step = step_val
                                last_step_time = time.time()
                                stall_warned = False
                    last_log_line_count = len(lines)
        except Exception:
            pass

        # Step-progress stall & deadlock watchdog (armed only once training steps are visible,
        # so environment setup / data unpacking does not trigger false alarms).
        if status == "running" and last_step_time is not None and last_step < total_steps:
            stall_s = time.time() - last_step_time
            if stall_s >= 600:
                print(f"\n🛑 DEADLOCK: no new global_step for {stall_s:.0f}s (last step {last_step}/{total_steps}). Aborting.", flush=True)
                cancel_fn = getattr(api, "kernels_cancel", None) or getattr(api, "kernel_cancel", None)
                if cancel_fn is not None:
                    try:
                        cancel_fn(kernel_ref)
                        print(f"   Sent cancel request for {kernel_ref}.", flush=True)
                    except Exception as e:
                        print(f"   Cancel request failed ({e}); cancel manually on kaggle.com.", flush=True)
                else:
                    print("   Kaggle API has no cancel method; cancel manually on kaggle.com.", flush=True)
                raise TimeoutError(f"Kernel {kernel_ref} stalled {stall_s:.0f}s at step {last_step}/{total_steps} (is_stalled=true)")
            elif stall_s >= 300 and not stall_warned:
                print(f"\n⚠️  STALL WARNING (is_stalled=true): no new global_step for {stall_s:.0f}s (last step {last_step}/{total_steps}).", flush=True)
                stall_warned = True

        # Periodic heartbeat
        if status == "running" and (now - last_heartbeat >= 60):
            print(f"[{elapsed:.0f}s | {elapsed/60:.1f}m] Status: {raw_status} (Est total: ~{est_train_seconds/60:.1f}m)", flush=True)
            last_heartbeat = now

        time.sleep(poll_interval)


def download_artifacts(b_key: str):
    """Download checkpoints and inference outputs from Kaggle."""
    init_kaggle_auth()
    spec = BRANCH_SPECS[b_key]
    kernel_ref = f"gustavnguyen/{spec['slug']}"
    dl_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"r2_1_4c_{b_key}_kaggle_output")
    os.makedirs(dl_dir, exist_ok=True)

    print("=" * 80)
    print(f"📥 DOWNLOADING ARTIFACTS FOR {b_key} -> {dl_dir}")
    print("=" * 80)

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Attempt {attempt}/{max_retries}] Downloading kernel output for {kernel_ref}...")
            api.kernels_output(kernel_ref, path=dl_dir)
            print("✅ Artifact download complete! Files downloaded:")
            break
        except Exception as e:
            print(f"⚠️ Download attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait_s = attempt * 5
                print(f"Retrying in {wait_s}s...")
                time.sleep(wait_s)
            else:
                raise
    for f in os.listdir(dl_dir):
        fp = os.path.join(dl_dir, f)
        if os.path.isfile(fp):
            print(f"  • {f} ({os.path.getsize(fp):,} bytes)")
        elif os.path.isdir(fp):
            print(f"  📁 {f}/")

    # Unpack checkpoint defense zip if present
    ckpt_zips = [f for f in os.listdir(dl_dir) if f.startswith("checkpoints_") and f.endswith(".zip")]
    if ckpt_zips:
        czip = os.path.join(dl_dir, ckpt_zips[0])
        extracted_ckpt_dir = os.path.join(dl_dir, "extracted_checkpoints")
        os.makedirs(extracted_ckpt_dir, exist_ok=True)
        print(f"Unpacking checkpoint archive {czip} -> {extracted_ckpt_dir}...")
        with zipfile.ZipFile(czip) as z:
            z.extractall(extracted_ckpt_dir)
        print("✅ Checkpoint defense archive extracted successfully!")
    return dl_dir


def main():
    parser = argparse.ArgumentParser(description="Stage R2.1.4C/D Causal Continuation Runner")
    parser.add_argument("--upload-assets", action="store_true", help="Upload R2.1.4C training bundle to Kaggle dataset")
    parser.add_argument("--upload-assets-d", action="store_true", help="Upload R2.1.4D training bundle to Kaggle dataset")
    parser.add_argument("--generate-kernel", choices=list(BRANCH_SPECS.keys()), help="Generate notebook and metadata for branch")
    parser.add_argument("--push-kernel", choices=list(BRANCH_SPECS.keys()), help="Push kernel to Kaggle")
    parser.add_argument("--monitor", choices=list(BRANCH_SPECS.keys()), help="Monitor running kernel on Kaggle")
    parser.add_argument("--download", choices=list(BRANCH_SPECS.keys()), help="Download artifacts from Kaggle")
    parser.add_argument("--run-branch-a", action="store_true", help="Run Branch A (CONTROL-CONTINUE) end-to-end")
    parser.add_argument("--run-branch-b", action="store_true", help="Run Branch B (COMBMARK-REPLAY) end-to-end")
    parser.add_argument("--run-branch-d", action="store_true", help="Run Branch D (PHONOTACTIC-COMBMARK) end-to-end")
    parser.add_argument("--run-branch-e", action="store_true", help="Run Branch E (SHORT-PHONOTACTIC-COMBMARK) end-to-end")
    parser.add_argument("--run-r2-1-5", action="store_true", help="Run Stage R2.1.5 (MULTI-STREAM-SYNTHESIS) end-to-end")
    parser.add_argument("--run-r2-1-6", action="store_true", help="Run Stage R2.1.6 (SURGICAL-SYMBOL-HYPHEN-POLISH) end-to-end")
    parser.add_argument("--run-r2-1-7", action="store_true", help="Run Stage R2.1.7 (SURGICAL-GATE-CLOSURE) end-to-end")
    parser.add_argument("--run-r3", action="store_true", help="Run Stage R3 (SCAN-DEGRADATION) end-to-end")
    args = parser.parse_args()

    if args.upload_assets:
        upload_training_assets("r2_1_4c_training_bundle", DATASET_SLUG)
    elif args.upload_assets_d:
        upload_training_assets("r2_1_4d_training_bundle", "gustavnguyen/cham-ocr-v24-1-r2-1-4d-training-assets")
    elif args.generate_kernel:
        k_dir = os.path.join(PROJECT_ROOT, "ocr-training", "kernels", BRANCH_SPECS[args.generate_kernel]["slug"])
        generate_kernel_files(args.generate_kernel, k_dir)
    elif args.push_kernel:
        push_and_run_kernel(args.push_kernel)
    elif args.monitor:
        monitor_kernel(args.monitor)
    elif args.download:
        download_artifacts(args.download)
    elif args.run_branch_a:
        print("\n🚀 EXECUTING BRANCH A (R2.1.4C-CONTROL-CONTINUE) END-TO-END...")
        push_and_run_kernel("control_continue")
        monitor_kernel("control_continue")
        download_artifacts("control_continue")
    elif args.run_branch_b:
        print("\n🚀 EXECUTING BRANCH B (R2.1.4C-COMBMARK-REPLAY) END-TO-END...")
        push_and_run_kernel("combmark_replay")
        monitor_kernel("combmark_replay")
        download_artifacts("combmark_replay")
    elif args.run_branch_d:
        print("\n🚀 EXECUTING BRANCH D (R2.1.4D-PHONOTACTIC-COMBMARK) END-TO-END...")
        push_and_run_kernel("phonotactic_combmark")
        monitor_kernel("phonotactic_combmark")
        download_artifacts("phonotactic_combmark")
    elif args.run_branch_e:
        print("\n🚀 EXECUTING BRANCH E (R2.1.4E-SHORT-PHONOTACTIC-COMBMARK) END-TO-END...")
        push_and_run_kernel("short_phonotactic_combmark")
        monitor_kernel("short_phonotactic_combmark")
        download_artifacts("short_phonotactic_combmark")
    elif args.run_r2_1_5:
        print("\n🚀 EXECUTING STAGE R2.1.5 (MULTI-STREAM-SYNTHESIS) END-TO-END...")
        push_and_run_kernel("multi_stream")
        monitor_kernel("multi_stream")
        download_artifacts("multi_stream")
    elif args.run_r2_1_6:
        print("\n🚀 EXECUTING STAGE R2.1.6 (SURGICAL-SYMBOL-HYPHEN-POLISH) END-TO-END...")
        push_and_run_kernel("surgical_polish")
        monitor_kernel("surgical_polish")
        download_artifacts("surgical_polish")
    elif args.run_r2_1_7:
        print("\n🚀 EXECUTING STAGE R2.1.7 (SURGICAL-GATE-CLOSURE) END-TO-END...")
        push_and_run_kernel("gate_closure_r2_1_7")
        monitor_kernel("gate_closure_r2_1_7")
        download_artifacts("gate_closure_r2_1_7")
    elif args.run_r3:
        print("\n🚀 EXECUTING STAGE R3 (SCAN-DEGRADATION) END-TO-END...")
        push_and_run_kernel("stage_r3_degradation")
        monitor_kernel("stage_r3_degradation")
        download_artifacts("stage_r3_degradation")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
