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
    },
    "stage_r3_1_closure": {
        "slug": "paddleocr-cham-v24-1-r3-1-gate-closure",
        "title": "PaddleOCR Cham v24.1 R3.1 Gate Closure",
        "folder": "cham_synthetic_r3_1_closure",
        "dataset_zip": "r3_1_closure_dataset.zip",
        "config_file": "rec_cham_v24_1_r3_1_closure.yml",
        "train_samples": 21600,
        "val_samples": 2400,
        "epochs": 3,
        "warmup_epoch": 0.4451038576,
        "step_each_epoch": 337,
        "warmup_steps": 150,
        "lr": 0.000015,
        "eta_min": 2.5e-6,
        "output_subpath": "rec_cham_v24_1_r3_1_closure",
        "base_ckpt_subdir": "r3_best",
        "base_ckpt_filename": "best.pdparams",
        "base_sha": "f00ae55578a2afae18f951677960ba8dc004174e429123d5d3602137f8ac7318",
        "dataset_slug": "gustavnguyen/cham-ocr-v24-1-r2-1-4d-training-assets"
    },
    "stage_r4_w480": {
        "slug": "paddleocr-cham-v24-1-r4-wide-w480",
        "title": "PaddleOCR Cham v24.1 R4 Wide W480",
        "folder": "cham_synthetic_r4_w480",
        "dataset_zip": "r4_w480_dataset.zip",
        "config_file": "rec_cham_v24_1_r4_w480.yml",
        "train_samples": 24000,
        "val_samples": 2400,
        "epochs": 3,
        "warmup_epoch": 0.4,
        "step_each_epoch": 375,
        "warmup_steps": 150,
        "lr": 0.000015,
        "eta_min": 2.5e-6,
        "output_subpath": "rec_cham_v24_1_r4_w480",
        "base_ckpt_subdir": "r3_1_best",
        "base_ckpt_filename": "best.pdparams",
        "base_sha": "AUTO_FROM_R3_1",
        "dataset_slug": "gustavnguyen/cham-ocr-v24-1-r2-1-4d-training-assets"
    },
    "stage_r4_1_polish": {
        "slug": "paddleocr-cham-v24-1-r4-1-gate-polish-w480",
        "title": "PaddleOCR Cham v24.1 R4.1 Gate Polish W480",
        "folder": "cham_synthetic_r4_1_polish",
        "dataset_zip": "r4_1_polish_dataset.zip",
        "config_file": "rec_cham_v24_1_r4_1_polish.yml",
        "train_samples": 19200,
        "val_samples": 1920,
        "epochs": 3,
        "warmup_epoch": 0.5,
        "step_each_epoch": 300,
        "warmup_steps": 150,
        "lr": 0.000010,
        "eta_min": 1.5e-6,
        "output_subpath": "rec_cham_v24_1_r4_1_polish",
        "base_ckpt_subdir": "r4_best",
        "base_ckpt_filename": "best.pdparams",
        "base_sha": "b98e0c3400d520d4ba1d4d80ca513619036fa32c8da1b9b811219c668f1b7de2",
        "dataset_slug": "gustavnguyen/cham-ocr-v24-1-r2-1-4d-training-assets"
    },
    "stage_r4_2_inktight": {
        "slug": "paddleocr-cham-v24-1-r4-2-inktight-w480",
        "title": "PaddleOCR Cham v24 1 R4 2 Inktight W480",
        "folder": "cham_synthetic_r4_2_inktight",
        "dataset_zip": "r4_2_inktight_dataset.zip",
        "config_file": "rec_cham_v24_1_r4_2_inktight.yml",
        "train_samples": 12800,
        "val_samples": 1280,
        "epochs": 2,
        "warmup_epoch": 0.25,
        "step_each_epoch": 200,
        "warmup_steps": 50,
        "lr": 0.000006,
        "eta_min": 1.0e-6,
        "output_subpath": "rec_cham_v24_1_r4_2_inktight",
        "base_ckpt_subdir": "r4_2_base",
        "base_ckpt_filename": "best.pdparams",
        "base_sha": "1e38d5623858c5c9a85888c177da001ff8b1cea6a7588248f1e1036455e8db2f",
        "dataset_slug": "gustavnguyen/cham-ocr-v24-1-r2-1-4d-training-assets"
    },
    # V26: SVTR key-padding mask fine-tune from the production R4.1 WiSE-FT weights, plus an
    # identical-data/identical-schedule UNMASKED control for causal attribution.
    # train/val counts & step_each_epoch are filled from REPORTS/v26_ft_dataset_manifest.json.
    "v26_masked_ft": {
        "slug": "paddleocr-cham-v26-masked-svtr-ft",
        "title": "PaddleOCR Cham V26 Masked SVTR FT",
        "folder": "cham_synthetic_v26_ft",
        "dataset_zip": "v26_ft_dataset.zip",
        "config_file": "rec_cham_v26_masked_ft.yml",
        "train_samples": "AUTO_V26",
        "val_samples": "AUTO_V26",
        "epochs": 6,
        "warmup_epoch": 0.4,
        "step_each_epoch": "AUTO_V26",
        "warmup_steps": "AUTO_V26",
        "lr": 0.000015,
        "eta_min": 2.0e-6,
        "output_subpath": "rec_cham_v26_masked_ft",
        "base_ckpt_subdir": "v26_base",
        "base_ckpt_filename": "best.pdparams",
        "base_sha": "300b3ef5aec4417985bf70c0ac3548216f8c27272207747c77f28a69d50402a3",
        "extra_patches": "AUTO_V26",
        "expect_valid_mask": True,
        "dataset_slug": "gustavnguyen/cham-ocr-v24-1-r2-1-4d-training-assets"
    },
    "v26_control_ft": {
        "slug": "paddleocr-cham-v26-control-unmasked-ft",
        "title": "PaddleOCR Cham V26 Control Unmasked FT",
        "folder": "cham_synthetic_v26_ft",
        "dataset_zip": "v26_ft_dataset.zip",
        "config_file": "rec_cham_v26_control_ft.yml",
        "train_samples": "AUTO_V26",
        "val_samples": "AUTO_V26",
        "epochs": 6,
        "warmup_epoch": 0.4,
        "step_each_epoch": "AUTO_V26",
        "warmup_steps": "AUTO_V26",
        "lr": 0.000015,
        "eta_min": 2.0e-6,
        "output_subpath": "rec_cham_v26_control_ft",
        "base_ckpt_subdir": "v26_base",
        "base_ckpt_filename": "best.pdparams",
        "base_sha": "300b3ef5aec4417985bf70c0ac3548216f8c27272207747c77f28a69d50402a3",
        "extra_patches": "AUTO_V26",
        "expect_valid_mask": False,
        "dataset_slug": "gustavnguyen/cham-ocr-v24-1-r2-1-4d-training-assets"
    },
    # Phase B: V26-Full 30-epoch unfrozen multi-stage training on 64,000 train / 6,400 val (W=480)
    "v26_full_s1": {
        "slug": "paddleocr-cham-v26-full-stage1",
        "remote_slug": "paddleocr-cham-v26-full-stage-1-e1-10",
        "title": "PaddleOCR Cham V26 Full Stage 1 E1-10",
        "folder": "cham_synthetic_v26_full",
        "dataset_zip": "v26_full_dataset.zip",
        "config_file": "rec_cham_v26_full_s1.yml",
        "train_samples": "AUTO_V26_FULL",
        "val_samples": "AUTO_V26_FULL",
        "epochs": 30,
        "stage_epochs": 10,
        "stage_end_epoch": 10,
        "warmup_epoch": 1.0,
        "step_each_epoch": "AUTO_V26_FULL",
        "warmup_steps": "AUTO_V26_FULL",
        "lr": 0.000020,
        "eta_min": 1.0e-6,
        "output_subpath": "rec_cham_v26_full_s1",
        "base_ckpt_subdir": "v26_s1_base",
        "base_ckpt_filename": "best.pdparams",
        "base_ckpt_local_path": "ocr-training/output/v26_masked_checkpoints/iter_epoch_4.pdparams",
        "base_sha": "AUTO_V26_FULL",
        "require_resume_state": False,
        "extra_patches": "AUTO_V26",
        "expect_valid_mask": True,
        "dataset_slug": "gustavnguyen/cham-ocr-v24-1-r2-1-4d-training-assets"
    },
    "v26_full_s2": {
        "slug": "paddleocr-cham-v26-full-stage2",
        "title": "PaddleOCR Cham V26 Full Stage2",
        "folder": "cham_synthetic_v26_full",
        "dataset_zip": "v26_full_dataset.zip",
        "config_file": "rec_cham_v26_full_s2.yml",
        "train_samples": "AUTO_V26_FULL",
        "val_samples": "AUTO_V26_FULL",
        "epochs": 30,
        "stage_epochs": 10,
        "stage_end_epoch": 20,
        "warmup_epoch": 1.0,
        "step_each_epoch": "AUTO_V26_FULL",
        "warmup_steps": "AUTO_V26_FULL",
        "lr": 0.000020,
        "eta_min": 1.0e-6,
        "output_subpath": "rec_cham_v26_full_s2",
        "base_ckpt_subdir": "v26_s1_e10",
        "base_ckpt_filename": "iter_epoch_10.pdparams",
        "base_ckpt_local_path": "ocr-training/output/r2_1_4c_v26_full_s1_kaggle_output/extracted_checkpoints/iter_epoch_10.pdparams",
        "base_sha": "AUTO_V26_FULL",
        "require_resume_state": True,
        "extra_patches": "AUTO_V26",
        "expect_valid_mask": True,
        "dataset_slug": "gustavnguyen/cham-ocr-v24-1-r2-1-4d-training-assets"
    },
    "v26_full_s3": {
        "slug": "paddleocr-cham-v26-full-stage3",
        "title": "PaddleOCR Cham V26 Full Stage3",
        "folder": "cham_synthetic_v26_full",
        "dataset_zip": "v26_full_dataset.zip",
        "config_file": "rec_cham_v26_full_s3.yml",
        "train_samples": "AUTO_V26_FULL",
        "val_samples": "AUTO_V26_FULL",
        "epochs": 30,
        "stage_epochs": 10,
        "stage_end_epoch": 30,
        "warmup_epoch": 1.0,
        "step_each_epoch": "AUTO_V26_FULL",
        "warmup_steps": "AUTO_V26_FULL",
        "lr": 0.000020,
        "eta_min": 1.0e-6,
        "output_subpath": "rec_cham_v26_full_s3",
        "base_ckpt_subdir": "v26_s2_e20",
        "base_ckpt_filename": "iter_epoch_20.pdparams",
        "base_ckpt_local_path": "ocr-training/output/r2_1_4c_v26_full_s2_kaggle_output/extracted_checkpoints/iter_epoch_20.pdparams",
        "base_sha": "AUTO_V26_FULL",
        "require_resume_state": True,
        "extra_patches": "AUTO_V26",
        "expect_valid_mask": True,
        "dataset_slug": "gustavnguyen/cham-ocr-v24-1-r2-1-4d-training-assets"
    }
}

# Files overlaid on top of the 7 audited patches for V26 (shipped under `extra_patches/` in
# r2_1_4c_base_and_configs.zip). Identical code for masked & control: the mask is opt-in via
# `Architecture.use_valid_mask`, so the control run executes the exact pre-V26 graph.
V26_EXTRA_PATCH_FILES = [
    "ppocr/modeling/architectures/base_model.py",
    "ppocr/modeling/heads/rec_multi_head.py",
    "ppocr/modeling/necks/rnn.py",
    "ppocr/modeling/backbones/rec_svtrnet.py",
    "ppocr/modeling/backbones/rec_lcnetv3.py",
    "tools/program.py",
]


def _remote_slug(spec: dict) -> str:
    """Return the Kaggle kernel slug (may differ from artifact slug if Kaggle rewrote the Clean URL)."""
    return spec.get("remote_slug") or spec["slug"]


def _resolve_v26_spec(spec: dict) -> dict:
    """Fills AUTO_V26 / AUTO_V26_FULL fields from the dataset manifest and local patched PaddleOCR files."""
    spec = dict(spec)
    if spec.get("train_samples") in ("AUTO_V26", "AUTO_V26_FULL"):
        man_name = "v26_full_dataset_manifest.json" if spec.get("train_samples") == "AUTO_V26_FULL" else "v26_ft_dataset_manifest.json"
        man = json.load(open(os.path.join(PROJECT_ROOT, "REPORTS", man_name), "r", encoding="utf-8"))
        spec["train_samples"] = int(man["train_samples"])
        spec["val_samples"] = int(man["val_samples"])
        spec["step_each_epoch"] = int(man["train_samples"]) // 2 // 32
        spec["warmup_steps"] = int(round(spec["warmup_epoch"] * spec["step_each_epoch"]))
    if spec.get("base_sha") == "AUTO_V26_FULL":
        local_ckpt = os.path.join(PROJECT_ROOT, *spec["base_ckpt_local_path"].split("/"))
        assert os.path.exists(local_ckpt), f"Missing V26-Full base/resume checkpoint: {local_ckpt}"
        spec["base_sha"] = file_sha256(local_ckpt)
    if spec.get("extra_patches") == "AUTO_V26":
        studio_paddle = os.path.join(PROJECT_ROOT, "ocr-studio", "PaddleOCR")
        spec["extra_patches"] = {rel: file_sha256(os.path.join(studio_paddle, *rel.split("/"))) for rel in V26_EXTRA_PATCH_FILES}
    return spec


def file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def build_notebook_cells(b_key: str) -> list:
    spec = _resolve_v26_spec(BRANCH_SPECS[b_key])
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
    if base_sha == "AUTO_FROM_R3_1":
        r31_eval_p = os.path.join(PROJECT_ROOT, "REPORTS", "r3_1_closure_evaluation.json")
        assert os.path.exists(r31_eval_p), f"Cannot resolve AUTO_FROM_R3_1: missing {r31_eval_p}"
        base_sha = json.load(open(r31_eval_p, "r", encoding="utf-8"))["best_epoch_sha256"].lower()

    patches_json_str = json.dumps(EXPECTED_RUNTIME_PATCHES, indent=4)
    extra_patches_json_str = json.dumps(spec.get("extra_patches") or {}, indent=4)
    expect_valid_mask = spec.get("expect_valid_mask")
    local_cfg_path = os.path.join(PROJECT_ROOT, "ocr-training", "configs", config_file)
    local_cfg_text = open(local_cfg_path, "r", encoding="utf-8").read() if os.path.exists(local_cfg_path) else None

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
                f"EXTRA_PATCHES = {extra_patches_json_str}\n",
                f"REQUIRE_RESUME_STATE = {bool(spec.get('require_resume_state', False))}\n",
                f"LOCAL_CFG_TEXT = {repr(local_cfg_text)}\n",
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
                "_ff_bzips = glob.glob('/kaggle/input/**/r2_1_4c_base_and_configs.zip', recursive=True)\n",
                "if _ff_bzips:\n",
                "    with zipfile.ZipFile(_ff_bzips[0]) as _zf:\n",
                "        _bn = set(_zf.namelist())\n",
                "    _need = f'{BASE_CKPT_SUBDIR}/{BASE_CKPT_FILENAME}'\n",
                "    if _need not in _bn:\n",
                "        raise FileNotFoundError(f'FAIL-FAST: {_need} missing inside {_ff_bzips[0]} (stale dataset mount?)')\n",
                "else:\n",
                "    _ff_ck = glob.glob(f'/kaggle/input/**/{BASE_CKPT_SUBDIR}/{BASE_CKPT_FILENAME}', recursive=True)\n",
                "    if not _ff_ck:\n",
                "        raise FileNotFoundError(f'FAIL-FAST: {BASE_CKPT_SUBDIR}/{BASE_CKPT_FILENAME} not found in /kaggle/input (stale dataset mount?)')\n",
                "print(f'✅ FAIL-FAST check passed: dataset & {BASE_CKPT_SUBDIR}/{BASE_CKPT_FILENAME} mounted ({(_ff_zip or _ff_dir)[0]})')\n",
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
                "# 3b. Dynamic width pooling fix in rec_lcnetv3.py (prevents hardcoded [1, 40] squeeze when W > 320, e.g. W=480 -> T=60)\n",
                "_lcnet_p = os.path.join(paddle_dir, 'ppocr', 'modeling', 'backbones', 'rec_lcnetv3.py')\n",
                "_lcnet_t = open(_lcnet_p, 'r', encoding='utf-8').read().replace('F.adaptive_avg_pool2d(x, [1, 40])', 'F.adaptive_avg_pool2d(x, [1, int(x.shape[3]) // 2])')\n",
                "with open(_lcnet_p, 'w', encoding='utf-8') as _lc_f:\n",
                "    _lc_f.write(_lcnet_t)\n",
                "assert 'int(x.shape[3]) // 2' in open(_lcnet_p, 'r', encoding='utf-8').read()\n",
                "print('✅ Patched rec_lcnetv3.py dynamic width pooling (W/8 CTC frames in train == eval).')\n",
                "\n",
                "# 3c. Optional extra audited overlays (V26 SVTR key-padding mask + padding-aware SE), SHA-256 verified\n",
                "for rel_path, exp_hash in EXTRA_PATCHES.items():\n",
                "    src_patch = glob.glob(f'{assets_base}/**/extra_patches/{rel_path}', recursive=True)\n",
                "    assert len(src_patch) > 0, f'Extra patch source missing: {rel_path}'\n",
                "    dst_patch = os.path.join(paddle_dir, rel_path)\n",
                "    shutil.copy2(src_patch[0], dst_patch)\n",
                "    assert sha(dst_patch) == exp_hash, f'SHA256 mismatch for extra overlay {rel_path}: {sha(dst_patch)} != {exp_hash}'\n",
                "    print(f'  • Extra overlay verified: {rel_path} -> {exp_hash[:12]}... OK')\n",
                "if EXTRA_PATCHES:\n",
                "    assert 'int(x.shape[3]) // 2' in open(_lcnet_p, 'r', encoding='utf-8').read(), 'dynamic pooling fix lost after overlay'\n",
                "    print(f'✅ {len(EXTRA_PATCHES)} extra overlays applied and provenance-verified.')\n",
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
                "ckpt_cands = (\n",
                "    glob.glob(f'{assets_base}/**/{BASE_CKPT_SUBDIR}/{BASE_CKPT_FILENAME}', recursive=True)\n",
                "    or glob.glob(f'/kaggle/input/**/{BASE_CKPT_SUBDIR}/{BASE_CKPT_FILENAME}', recursive=True)\n",
                "    or glob.glob(f'{assets_base}/**/{BASE_CKPT_FILENAME}', recursive=True)\n",
                "    or glob.glob(f'/kaggle/input/**/{BASE_CKPT_FILENAME}', recursive=True)\n",
                ")\n",
                "assert ckpt_cands, f'Could not find base checkpoint {BASE_CKPT_SUBDIR}/{BASE_CKPT_FILENAME} in /kaggle/input'\n",
                "os.makedirs(f'/kaggle/working/data/{BASE_CKPT_SUBDIR}', exist_ok=True)\n",
                "dst_base_ckpt = f'/kaggle/working/data/{BASE_CKPT_SUBDIR}/{BASE_CKPT_FILENAME}'\n",
                "shutil.copy2(ckpt_cands[0], dst_base_ckpt)\n",
                "assert sha(dst_base_ckpt) == BASE_SHA, f'Base checkpoint hash mismatch: {sha(dst_base_ckpt)} != {BASE_SHA}'\n",
                "for _ext in ('.pdopt', '.states'):\n",
                "    _sib = os.path.splitext(ckpt_cands[0])[0] + _ext\n",
                "    if os.path.exists(_sib):\n",
                "        shutil.copy2(_sib, os.path.splitext(dst_base_ckpt)[0] + _ext)\n",
                "        print('  • Copied resume state:', os.path.basename(_sib))\n",
                "    elif REQUIRE_RESUME_STATE:\n",
                "        raise FileNotFoundError(f'Missing required resume state file: {_sib}')\n",
                "print('✅ Base checkpoint verified:', BASE_SHA)\n",
                "os.makedirs('/kaggle/working/configs', exist_ok=True)\n",
                "if LOCAL_CFG_TEXT is not None:\n",
                "    with open(f'/kaggle/working/configs/{CONFIG_NAME}', 'w', encoding='utf-8', newline='\\n') as _cf:\n",
                "        _cf.write(LOCAL_CFG_TEXT)\n",
                "    print(f'✅ Config installed (inline verified): /kaggle/working/configs/{CONFIG_NAME} ({len(LOCAL_CFG_TEXT)} bytes)')\n",
                "else:\n",
                "    cfg_cands = glob.glob(f'{assets_base}/**/{CONFIG_NAME}', recursive=True) or glob.glob(f'/kaggle/input/**/{CONFIG_NAME}', recursive=True)\n",
                "    assert cfg_cands, f'Could not find config {CONFIG_NAME} in /kaggle/input'\n",
                "    shutil.copy2(cfg_cands[0], f'/kaggle/working/configs/{CONFIG_NAME}')\n",
                "    print(f'✅ Config installed: /kaggle/working/configs/{CONFIG_NAME}')\n",
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
                "os.environ['UV_HTTP_TIMEOUT'] = '300'\n",
                "\n",
                "# Install runtime dependencies (retry transient PyPI / uv network timeouts on 724MB paddle wheel)\n",
                "import time as _pip_time\n",
                "print('Installing runtime dependencies into active Python environment...')\n",
                "for _pip_att in range(1, 4):\n",
                "    try:\n",
                "        subprocess.run(install_cmd, check=True)\n",
                "        break\n",
                "    except subprocess.CalledProcessError as _pip_err:\n",
                "        if _pip_att == 3:\n",
                "            raise\n",
                "        print(f'⚠️ Install attempt {_pip_att} failed ({_pip_err}); retrying in 15s...')\n",
                "        _pip_time.sleep(15)\n",
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
            ] + ([
                "    \"import yaml\\n\"\n",
                "    \"from ppocr.modeling.architectures import build_model\\n\"\n",
                f"    \"_cfg = yaml.safe_load(open('/kaggle/working/configs/{config_file}', 'r', encoding='utf-8'))\\n\"\n",
                "    \"_m = build_model(_cfg['Architecture'])\\n\"\n",
                f"    \"assert bool(_m.use_valid_mask) == {bool(expect_valid_mask)} and bool(_m.backbone.use_valid_mask) == {bool(expect_valid_mask)}, 'V26 mask flag mismatch'\\n\"\n",
                "    \"for _m_k in ('Train', 'Eval'):\\n\"\n",
                "    \"    for _l_k in ('batch_size_per_card', 'drop_last', 'shuffle', 'num_workers'):\\n\"\n",
                "    \"        assert _l_k in _cfg[_m_k]['loader'], f'Missing {_m_k}.loader.{_l_k}'\\n\"\n",
                f"    \"print('✅ V26 model & Train/Eval loader schema verified: use_valid_mask={bool(expect_valid_mask)}')\\n\"\n",
            ] if expect_valid_mask is not None else []) + ([
                "    \"assert 'stage_end_epoch' in open('/kaggle/working/PaddleOCR/tools/program.py', 'r', encoding='utf-8').read(), 'Missing stage_end_epoch hook in tools/program.py'\\n\"\n",
                "    \"print('✅ Multi-stage stage_end_epoch hook verified in tools/program.py')\\n\"\n",
            ] if spec.get("stage_end_epoch") is not None else []) + [
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
        "id": f"gustavnguyen/{_remote_slug(spec)}",
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

    import socket
    socket.setdefaulttimeout(60)
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
            print("Waiting 135s for Kaggle backend zip/mount propagation...", flush=True)
            time.sleep(135)
            print("✅ Dataset is READY & PROPAGATED on Kaggle!", flush=True)
            return True
    raise RuntimeError(f"Dataset {target_slug} did not become ready within 600s!")


def push_and_run_kernel(b_key: str):
    """Push kernel to Kaggle and trigger execution."""
    u = init_kaggle_auth()
    spec = BRANCH_SPECS[b_key]
    kernel_dir = os.path.join(PROJECT_ROOT, "ocr-training", "kernels", spec["slug"])
    generate_kernel_files(b_key, kernel_dir)

    print(f"\n🚀 Pushing kernel {spec['slug']} to Kaggle...")
    import socket
    socket.setdefaulttimeout(60)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    actual_user = api.config_values.get("username")
    assert actual_user == u, f"FATAL: KaggleApi authenticated as '{actual_user}', expected '{u}'!"

    res = api.kernels_push(kernel_dir)
    err_msg = getattr(res, "error", None)
    if err_msg:
        raise RuntimeError(f"Kaggle kernels_push returned error: {err_msg}")
    expected_remote_slug = _remote_slug(spec)
    pushed_ref = getattr(res, "ref", None) or f"{u}/{expected_remote_slug}"
    print(f"✅ Kernel pushed successfully: {pushed_ref} (url: {getattr(res, 'url', 'N/A')})")
    assert f"/{u.lower()}/" in f"/{str(pushed_ref).lower()}", f"FATAL: Kernel pushed to wrong owner: {pushed_ref}"
    actual_pushed_slug = str(pushed_ref).rstrip("/").split("/")[-1]
    if actual_pushed_slug != expected_remote_slug:
        raise RuntimeError(
            f"Kaggle Clean URL resolved pushed slug to '{actual_pushed_slug}', expected '{expected_remote_slug}' "
            f"(pushed_ref={pushed_ref}, title={spec['title']!r})"
        )


def monitor_kernel(b_key: str, poll_interval: int = 30, max_stall_seconds: int = None):
    """
    Monitor running kernel status with autonomous stall & deadlock detection.
    """
    u = init_kaggle_auth()
    spec = _resolve_v26_spec(BRANCH_SPECS[b_key])
    kernel_ref = f"{u}/{_remote_slug(spec)}"

    stage_epochs = spec.get('stage_epochs', spec['epochs'])
    total_steps = stage_epochs * spec['step_each_epoch']
    target_end_step = total_steps
    est_train_seconds = int(stage_epochs * spec['step_each_epoch'] * 2.0 + 900)
    if max_stall_seconds is None:
        max_stall_seconds = max(1200, int(est_train_seconds * 1.5))

    import socket
    socket.setdefaulttimeout(60)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    actual_user = api.config_values.get("username")
    assert actual_user == u, f"FATAL: KaggleApi authenticated as '{actual_user}', expected '{u}'!"

    print("=" * 80)
    print(f"👀 MONITORING KERNEL: {kernel_ref}")
    print(f"   Branch: {b_key} | Stage Epochs: {stage_epochs}/{spec['epochs']} | Target End Step: {target_end_step} (Est: ~{est_train_seconds/60:.1f}m | Timeout: {max_stall_seconds/60:.1f}m)")
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
        if status == "running" and last_step_time is not None and last_step < target_end_step:
            stall_s = time.time() - last_step_time
            if stall_s >= 600:
                print(f"\n🛑 DEADLOCK: no new global_step for {stall_s:.0f}s (last step {last_step}/{target_end_step}). Aborting.", flush=True)
                cancel_fn = getattr(api, "kernels_cancel", None) or getattr(api, "kernel_cancel", None)
                if cancel_fn is not None:
                    try:
                        cancel_fn(kernel_ref)
                        print(f"   Sent cancel request for {kernel_ref}.", flush=True)
                    except Exception as e:
                        print(f"   Cancel request failed ({e}); cancel manually on kaggle.com.", flush=True)
                else:
                    print("   Kaggle API has no cancel method; cancel manually on kaggle.com.", flush=True)
                raise TimeoutError(f"Kernel {kernel_ref} stalled {stall_s:.0f}s at step {last_step}/{target_end_step} (is_stalled=true)")
            elif stall_s >= 300 and not stall_warned:
                print(f"\n⚠️  STALL WARNING (is_stalled=true): no new global_step for {stall_s:.0f}s (last step {last_step}/{target_end_step}).", flush=True)
                stall_warned = True

        # Periodic heartbeat
        if status in ("running", "queued") and (now - last_heartbeat >= 60):
            print(f"[{elapsed:.0f}s | {elapsed/60:.1f}m] Status: {raw_status} (Est total: ~{est_train_seconds/60:.1f}m)", flush=True)
            last_heartbeat = now

        time.sleep(poll_interval)


def _verify_ckpt_zip_against_manifest(zip_path: str, manifest_path: str) -> bool:
    """Return True iff zip_path passes ZIP CRC test and matches all manifest SHA-256s."""
    if not (os.path.exists(zip_path) and os.path.exists(manifest_path)):
        return False
    try:
        manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
        if not isinstance(manifest, dict) or not manifest:
            return False
        if not zipfile.is_zipfile(zip_path):
            return False
        with zipfile.ZipFile(zip_path, "r") as z:
            if z.testzip() is not None:
                return False
            for member, expected_sha in manifest.items():
                actual_sha = hashlib.sha256(z.read(member)).hexdigest().lower()
                if actual_sha != str(expected_sha).lower():
                    return False
        return True
    except Exception:
        return False


def download_artifacts(b_key: str):
    """Download essential Kaggle checkpoints + log via filtered 4 MB chunked streaming."""
    init_kaggle_auth()
    spec = BRANCH_SPECS[b_key]
    slug = spec["slug"]
    remote_slug = _remote_slug(spec)
    kernel_ref = f"gustavnguyen/{remote_slug}"
    kernel_url = f"https://www.kaggle.com/code/{kernel_ref}"
    dl_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"r2_1_4c_{b_key}_kaggle_output")
    os.makedirs(dl_dir, exist_ok=True)

    print("=" * 80)
    print(f"📥 DOWNLOADING ARTIFACTS FOR {b_key} -> {dl_dir}")
    print("=" * 80)

    import socket
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    socket.setdefaulttimeout(45)
    api = KaggleApi()
    api.authenticate()

    keep_re = re.compile(r"^(checkpoints_.*\.zip|checkpoint_manifest_.*\.json|[^/\\]+\.log)$")
    items = []
    log_saved = False
    log_out = os.path.join(dl_dir, f"{slug}.log")

    try:
        with api.build_kaggle_client() as kaggle:
            client = kaggle.kernels.kernels_api_client
            token = None
            while True:
                req = ApiListKernelSessionOutputRequest()
                req.user_name = "gustavnguyen"
                req.kernel_slug = remote_slug
                req.page_size = 100
                if token:
                    req.page_token = token
                resp = client.list_kernel_session_output(req)
                log_blob = getattr(resp, "log", None)
                if log_blob and (not log_saved or not os.path.exists(log_out) or os.path.getsize(log_out) == 0):
                    with open(log_out, "w", encoding="utf-8", newline="\n") as lf:
                        lf.write(str(log_blob))
                    log_saved = True
                    print(f"  ✅ Saved kernel log -> {slug}.log ({os.path.getsize(log_out):,} bytes)", flush=True)
                for item in (resp.files or []):
                    fn = str(getattr(item, "file_name", "") or "").replace("\\", "/")
                    if keep_re.match(fn):
                        items.append(item)
                token = getattr(resp, "next_page_token", None) or None
                if not token:
                    break
    except Exception as exc:
        print(f"❌ Kaggle output listing failed on single API attempt: {exc}", flush=True)
        print(f"BROWSER_FALLBACK_REQUIRED: {kernel_url}", flush=True)
        raise RuntimeError("BROWSER_FALLBACK_REQUIRED") from exc

    # Ensure checkpoint_manifest_*.json is downloaded before checkpoints_*.zip
    items.sort(key=lambda it: (0 if str(it.file_name).endswith(".json") else 1, str(it.file_name)))
    manifest_path = os.path.join(dl_dir, f"checkpoint_manifest_{slug}.json")

    for item in items:
        fn = os.path.basename(str(item.file_name).replace("\\", "/"))
        outfile = os.path.join(dl_dir, fn)

        if fn.startswith("checkpoints_") and fn.endswith(".zip"):
            if _verify_ckpt_zip_against_manifest(outfile, manifest_path):
                print(f"  ⏩ Skipping {fn} (existing archive verified against manifest: {os.path.getsize(outfile):,} bytes)", flush=True)
                continue
        elif fn.endswith(".log") and os.path.exists(outfile) and os.path.getsize(outfile) > 0:
            print(f"  ⏩ Skipping {fn} (already exists: {os.path.getsize(outfile):,} bytes)", flush=True)
            continue

        part_file = outfile + ".part"
        try:
            print(f"  ⬇️ [Attempt 1/1] Chunked streaming {fn} (4 MB chunks, timeout=(20, 45))...", flush=True)
            total = 0
            with requests.get(item.url, stream=True, timeout=(20, 45)) as dl_resp:
                dl_resp.raise_for_status()
                with open(part_file, "wb") as out:
                    for chunk in dl_resp.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            out.write(chunk)
                            total += len(chunk)
            if fn.startswith("checkpoints_") and fn.endswith(".zip") and os.path.exists(manifest_path):
                if not _verify_ckpt_zip_against_manifest(part_file, manifest_path):
                    raise RuntimeError(f"Downloaded zip {fn} failed CRC/SHA-256 verification against manifest")
            os.replace(part_file, outfile)
            print(f"  ✅ Downloaded {fn} ({total:,} bytes)", flush=True)
        except Exception as exc:
            if os.path.exists(part_file):
                try:
                    os.remove(part_file)
                except Exception:
                    pass
            print(f"❌ Single API download attempt failed for {fn}: {exc}", flush=True)
            print(f"BROWSER_FALLBACK_REQUIRED: {kernel_url}", flush=True)
            raise RuntimeError("BROWSER_FALLBACK_REQUIRED") from exc

    # Unpack verified checkpoint defense zip -> extracted_checkpoints/ and remove redundant output/
    ckpt_zips = [f for f in os.listdir(dl_dir) if f.startswith("checkpoints_") and f.endswith(".zip")]
    if not ckpt_zips:
        print(f"❌ Missing checkpoints_*.zip for {kernel_ref}", flush=True)
        print(f"BROWSER_FALLBACK_REQUIRED: {kernel_url}", flush=True)
        raise RuntimeError("BROWSER_FALLBACK_REQUIRED")

    czip = os.path.join(dl_dir, ckpt_zips[0])
    if os.path.exists(manifest_path) and not _verify_ckpt_zip_against_manifest(czip, manifest_path):
        print(f"❌ Archive {czip} failed manifest SHA-256 verification", flush=True)
        print(f"BROWSER_FALLBACK_REQUIRED: {kernel_url}", flush=True)
        raise RuntimeError("BROWSER_FALLBACK_REQUIRED")

    extracted_ckpt_dir = os.path.join(dl_dir, "extracted_checkpoints")
    os.makedirs(extracted_ckpt_dir, exist_ok=True)
    print(f"Unpacking verified archive {czip} -> {extracted_ckpt_dir}...", flush=True)
    with zipfile.ZipFile(czip, "r") as z:
        z.extractall(extracted_ckpt_dir)

    # Verify extracted .pdparams on disk against manifest
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
        for member, exp_sha in manifest.items():
            ext_p = os.path.join(extracted_ckpt_dir, member)
            assert os.path.exists(ext_p), f"Missing extracted file: {ext_p}"
            act_sha = hashlib.sha256(open(ext_p, "rb").read()).hexdigest().lower()
            assert act_sha == str(exp_sha).lower(), f"Extracted SHA-256 mismatch for {member}"
        print(f"✅ Verified {len(manifest)}/{len(manifest)} extracted checkpoint SHA-256 hashes!", flush=True)

    # Clean up partial redundant uncompressed output/ directory if created by earlier downloads
    redundant_out = os.path.join(dl_dir, "output")
    if os.path.isdir(redundant_out):
        shutil.rmtree(redundant_out, ignore_errors=True)
        print(f"🧹 Removed redundant uncompressed directory: {redundant_out}", flush=True)

    for f in sorted(os.listdir(dl_dir)):
        fp = os.path.join(dl_dir, f)
        if os.path.isfile(fp):
            print(f"  • {f} ({os.path.getsize(fp):,} bytes)", flush=True)
        elif os.path.isdir(fp):
            print(f"  📁 {f}/", flush=True)
    return dl_dir


def ingest_browser_download(b_key: str, browser_zip: str, browser_log: str = None):
    """Ingest Kaggle browser-downloaded `results.zip` + `download.txt` into dl_dir and verify SHA-256."""
    spec = BRANCH_SPECS[b_key]
    slug = spec["slug"]
    dl_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"r2_1_4c_{b_key}_kaggle_output")
    os.makedirs(dl_dir, exist_ok=True)
    extracted_ckpt_dir = os.path.join(dl_dir, "extracted_checkpoints")
    os.makedirs(extracted_ckpt_dir, exist_ok=True)

    assert os.path.exists(browser_zip) and zipfile.is_zipfile(browser_zip), f"Invalid browser zip: {browser_zip}"
    manifest_name = f"checkpoint_manifest_{slug}.json"
    ckpt_zip_name = f"checkpoints_{slug}.zip"
    manifest_dst = os.path.join(dl_dir, manifest_name)
    ckpt_zip_dst = os.path.join(dl_dir, ckpt_zip_name)

    with zipfile.ZipFile(browser_zip, "r") as bz:
        names = [n.replace("\\", "/") for n in bz.namelist()]
        for n in names:
            bn = os.path.basename(n)
            if bn == manifest_name:
                with open(manifest_dst, "wb") as mf:
                    mf.write(bz.read(n))
            elif bn == ckpt_zip_name:
                with open(ckpt_zip_dst, "wb") as cf:
                    cf.write(bz.read(n))

        if os.path.exists(ckpt_zip_dst):
            if os.path.exists(manifest_dst):
                assert _verify_ckpt_zip_against_manifest(ckpt_zip_dst, manifest_dst), f"Browser inner zip failed manifest check: {ckpt_zip_dst}"
            with zipfile.ZipFile(ckpt_zip_dst, "r") as cz:
                cz.extractall(extracted_ckpt_dir)
        else:
            # Extract flat/nested checkpoint dir files (iter_epoch_*, latest.*, train.log, config.yml)
            sub_prefix = f"output/{spec['output_subpath']}/"
            for n in names:
                if n.endswith("/"):
                    continue
                if n.startswith(sub_prefix):
                    rel = n[len(sub_prefix):]
                elif "/" not in n and (n.endswith((".pdparams", ".pdopt", ".states", ".yml")) or n == "train.log"):
                    rel = n
                else:
                    continue
                out_p = os.path.join(extracted_ckpt_dir, rel)
                os.makedirs(os.path.dirname(out_p), exist_ok=True)
                with open(out_p, "wb") as of:
                    of.write(bz.read(n))

    if os.path.exists(manifest_dst):
        manifest = json.load(open(manifest_dst, "r", encoding="utf-8"))
        for member, exp_sha in manifest.items():
            ext_p = os.path.join(extracted_ckpt_dir, member)
            assert os.path.exists(ext_p), f"Missing extracted file from browser zip: {ext_p}"
            act_sha = hashlib.sha256(open(ext_p, "rb").read()).hexdigest().lower()
            assert act_sha == str(exp_sha).lower(), f"Browser extracted SHA-256 mismatch for {member}"
        print(f"✅ Browser download verified {len(manifest)}/{len(manifest)} checkpoint SHA-256 hashes!", flush=True)

    if browser_log and os.path.exists(browser_log) and os.path.getsize(browser_log) > 0:
        log_dst = os.path.join(dl_dir, f"{slug}.log")
        raw_bytes = open(browser_log, "rb").read().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        with open(log_dst, "wb") as lf:
            lf.write(raw_bytes)
        print(f"✅ Saved browser kernel log -> {log_dst} ({os.path.getsize(log_dst):,} bytes)", flush=True)
    return dl_dir


def main():
    parser = argparse.ArgumentParser(description="Stage R2.1.4C/D Causal Continuation Runner")
    parser.add_argument("--upload-assets", action="store_true", help="Upload R2.1.4C training bundle to Kaggle dataset")
    parser.add_argument("--upload-assets-d", action="store_true", help="Upload R2.1.4D training bundle to Kaggle dataset")
    parser.add_argument("--generate-kernel", choices=list(BRANCH_SPECS.keys()), help="Generate notebook and metadata for branch")
    parser.add_argument("--push-kernel", choices=list(BRANCH_SPECS.keys()), help="Push kernel to Kaggle")
    parser.add_argument("--monitor", choices=list(BRANCH_SPECS.keys()), help="Monitor running kernel on Kaggle")
    parser.add_argument("--download", choices=list(BRANCH_SPECS.keys()), help="Download artifacts from Kaggle")
    parser.add_argument("--ingest-browser-download", choices=list(BRANCH_SPECS.keys()), help="Verify & unpack browser results.zip for branch")
    parser.add_argument("--browser-zip", help="Path to Kaggle browser-downloaded results.zip")
    parser.add_argument("--browser-log", help="Optional path to Kaggle browser-downloaded download.txt kernel log")
    parser.add_argument("--run-branch-a", action="store_true", help="Run Branch A (CONTROL-CONTINUE) end-to-end")
    parser.add_argument("--run-branch-b", action="store_true", help="Run Branch B (COMBMARK-REPLAY) end-to-end")
    parser.add_argument("--run-branch-d", action="store_true", help="Run Branch D (PHONOTACTIC-COMBMARK) end-to-end")
    parser.add_argument("--run-branch-e", action="store_true", help="Run Branch E (SHORT-PHONOTACTIC-COMBMARK) end-to-end")
    parser.add_argument("--run-r2-1-5", action="store_true", help="Run Stage R2.1.5 (MULTI-STREAM-SYNTHESIS) end-to-end")
    parser.add_argument("--run-r2-1-6", action="store_true", help="Run Stage R2.1.6 (SURGICAL-SYMBOL-HYPHEN-POLISH) end-to-end")
    parser.add_argument("--run-r2-1-7", action="store_true", help="Run Stage R2.1.7 (SURGICAL-GATE-CLOSURE) end-to-end")
    parser.add_argument("--run-r3", action="store_true", help="Run Stage R3 (SCAN-DEGRADATION) end-to-end")
    parser.add_argument("--run-r3-1", action="store_true", help="Run Stage R3.1 (GATE-CLOSURE + SESQUISYLLABIC + EROSION) end-to-end")
    parser.add_argument("--run-r4", action="store_true", help="Run Stage R4 (WIDE-CANVAS W=480) end-to-end")
    parser.add_argument("--run-r4-1", action="store_true", help="Run Stage R4.1 (GATE-POLISH W=480) end-to-end")
    parser.add_argument("--run-r4-2", action="store_true", help="Run Stage R4.2 (INKTIGHT-CLOSURE W=480) end-to-end")
    parser.add_argument("--run-v26-masked", action="store_true", help="Run V26 masked-SVTR fine-tune end-to-end")
    parser.add_argument("--run-v26-control", action="store_true", help="Run V26 unmasked control fine-tune end-to-end")
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
    elif args.ingest_browser_download:
        if not args.browser_zip:
            parser.error("--browser-zip is required with --ingest-browser-download")
        ingest_browser_download(args.ingest_browser_download, args.browser_zip, args.browser_log)
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
    elif args.run_r3_1:
        print("\n🚀 EXECUTING STAGE R3.1 (GATE-CLOSURE + SESQUISYLLABIC + EROSION) END-TO-END...")
        push_and_run_kernel("stage_r3_1_closure")
        monitor_kernel("stage_r3_1_closure")
        download_artifacts("stage_r3_1_closure")
    elif args.run_r4:
        print("\n🚀 EXECUTING STAGE R4 (WIDE-CANVAS W=480) END-TO-END...")
        push_and_run_kernel("stage_r4_w480")
        monitor_kernel("stage_r4_w480")
        download_artifacts("stage_r4_w480")
    elif args.run_r4_1:
        print("\n🚀 EXECUTING STAGE R4.1 (GATE-POLISH W=480) END-TO-END...")
        push_and_run_kernel("stage_r4_1_polish")
        monitor_kernel("stage_r4_1_polish")
        download_artifacts("stage_r4_1_polish")
    elif args.run_r4_2:
        print("\n🚀 EXECUTING STAGE R4.2 (INKTIGHT-CLOSURE W=480) END-TO-END...")
        push_and_run_kernel("stage_r4_2_inktight")
        monitor_kernel("stage_r4_2_inktight")
        download_artifacts("stage_r4_2_inktight")
    elif args.run_v26_masked:
        print("\n🚀 EXECUTING V26 MASKED-SVTR FINE-TUNE (W=480) END-TO-END...")
        push_and_run_kernel("v26_masked_ft")
        monitor_kernel("v26_masked_ft")
        download_artifacts("v26_masked_ft")
    elif args.run_v26_control:
        print("\n🚀 EXECUTING V26 UNMASKED CONTROL FINE-TUNE (W=480) END-TO-END...")
        push_and_run_kernel("v26_control_ft")
        monitor_kernel("v26_control_ft")
        download_artifacts("v26_control_ft")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
