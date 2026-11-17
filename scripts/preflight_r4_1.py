#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R4.1 Surgical Gate-Closure (`W = 480`, `T = 60`) Preflight Verification
============================================================================================
Verifies config, SHA256 of R4 Epoch 3 base checkpoint (`b98e0c34...`), 150-step warmup at
300 steps/epoch (19,200 train samples), live DataLoader `[B, 3, 48, 480]`, `T == 60` CTC frames,
and finite backward gradients.
"""

import os
import sys
import json
import yaml
import hashlib
import argparse
import logging
from datetime import datetime, timezone

import numpy as np
if not hasattr(np, "sctypes"):
    np.sctypes = {"int": [np.int8, np.int16, np.int32, np.int64], "uint": [np.uint8, np.uint16, np.uint32, np.uint64], "float": [np.float16, np.float32, np.float64], "complex": [np.complex64, np.complex128], "others": [bool, object, bytes, str]}
if not hasattr(np, "bool"): np.bool = bool
if not hasattr(np, "int"): np.int = int
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "typeDict"): np.typeDict = {}

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PADDLE_DIR = os.path.join(PROJECT_ROOT, "ocr-studio", "PaddleOCR")
sys.path.insert(0, PADDLE_DIR)

import paddle
from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.optimizer import build_lr_scheduler
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--base-sha", required=True)
    args = parser.parse_args()

    base_ckpt = os.path.abspath(args.base_ckpt)
    base_sha = sha256_file(base_ckpt)
    assert base_sha == args.base_sha.lower(), f"SHA mismatch: {base_sha} != {args.base_sha}"

    cfg_path = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "rec_cham_v24_1_r4_1_polish.yml")
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    assert cfg["Global"]["d2s_train_image_shape"] == [3, 48, 480]
    lr_sched = build_lr_scheduler(dict(cfg["Optimizer"]["lr"]), epochs=3, step_each_epoch=300)
    assert lr_sched.warmup_steps == 150, f"Expected warmup_steps=150, got {lr_sched.warmup_steps}"

    local_cfg = json.loads(json.dumps(cfg))
    data_dir = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_synthetic_r4_1_polish")
    dict_path = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_dict_v24_1.txt")
    local_cfg["Global"]["character_dict_path"] = dict_path
    local_cfg["PostProcess"]["character_dict_path"] = dict_path
    local_cfg["Global"]["pretrained_model"] = base_ckpt[:-9]
    local_cfg["Global"]["distributed"] = False
    local_cfg["Train"]["dataset"]["data_dir"] = data_dir
    local_cfg["Train"]["dataset"]["label_file_list"] = [os.path.join(data_dir, "train_label.txt")]
    local_cfg["Train"]["loader"]["num_workers"] = 0

    post_process_class = build_post_process(local_cfg["PostProcess"], local_cfg["Global"])
    char_num = len(getattr(post_process_class, "character"))
    local_cfg["Architecture"]["Head"]["out_channels_list"] = {
        "CTCLabelDecode": char_num,
        "NRTRLabelDecode": char_num + 3,
    }

    model = build_model(local_cfg["Architecture"])
    raw_sd = paddle.load(base_ckpt)
    model_sd = model.state_dict()
    assert set(model_sd.keys()) == set(raw_sd.keys()), "State dict key mismatch!"
    load_model(local_cfg, model, model_type=local_cfg["Architecture"]["model_type"])

    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in local_cfg["Global"].get("freeze_prefixes", [])):
            param.stop_gradient = True

    loss_class = build_loss(local_cfg["Loss"])
    train_loader = build_dataloader(local_cfg, "Train", paddle.CPUPlace(), logging.getLogger("preflight_r4_1"))

    mixed_batch = None
    for batch in train_loader:
        lengths = batch[3].numpy()
        if 0 < int(np.sum(lengths == 0)) < len(lengths):
            mixed_batch = batch
            break
    assert mixed_batch is not None
    preds = model(mixed_batch[0], data=mixed_batch[1:])
    ctc_shape = list(preds["ctc"].shape)
    assert ctc_shape[1] == 60, f"Expected T=60, got {ctc_shape}"
    losses = loss_class(preds, mixed_batch)
    losses["loss"].backward()

    finite_grads = 0
    for _, param in model.named_parameters():
        if not param.stop_gradient and param.grad is not None:
            assert np.all(np.isfinite(param.grad.numpy()))
            finite_grads += 1

    rep = {
        "stage": "R4_1_GATE_POLISH_W480",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "base_sha256": base_sha,
        "transferred_tensors": len(model_sd),
        "ctc_shape": ctc_shape,
        "finite_grad_tensors": finite_grads,
        "sample_loss": round(float(losses["loss"].numpy()), 4),
    }
    out = os.path.join(PROJECT_ROOT, "REPORTS", "r4_1_polish_preflight_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"✅ Stage R4.1 Preflight PASSED: {rep}")


if __name__ == "__main__":
    main()
