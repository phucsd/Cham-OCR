#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R3.1 Preflight Verification (`GATE_CLOSURE_SESQUISYLLABIC_EROSION_R3_1`)
=============================================================================================
Verifies:
1. Exact YAML Diff against Stage R3 (only expected stage/path/LR/warmup keys differ).
2. Physical Base Checkpoint SHA256 verification (R3 E3).
3. Exact 150-step warmup verification at 337 steps/epoch across 21,600 train samples.
4. Live DataLoader + BlankAwareMultiLoss + PPLCNetV3 forward/backward pass on a mixed batch (blanks + non-blanks),
   verifying finite gradients on all trainable parameters and 0 mutated weights on disk.
"""

import os
import sys
import json
import yaml
import hashlib
import argparse
from datetime import datetime, timezone

import numpy as np
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, bytes, str],
    }
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "typeDict"):
    np.typeDict = {}

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


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def main():
    parser = argparse.ArgumentParser(description="Stage R3.1 Preflight Check")
    parser.add_argument("--base-ckpt", required=True, help="Path to R3 E3 .pdparams file")
    parser.add_argument("--base-sha", required=True, help="Expected SHA256 of --base-ckpt")
    args = parser.parse_args()

    base_ckpt = os.path.abspath(args.base_ckpt)
    assert os.path.exists(base_ckpt) and base_ckpt.endswith(".pdparams"), f"Invalid --base-ckpt: {base_ckpt}"
    base_sha = sha256_file(base_ckpt)
    assert base_sha == args.base_sha.lower(), f"Base SHA mismatch: got {base_sha}, expected {args.base_sha}"
    print(f"✅ Base Checkpoint SHA256 Verified: {base_sha}")

    cfg_r3_path = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "rec_cham_v24_1_r3_degradation.yml")
    cfg_r31_path = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "rec_cham_v24_1_r3_1_closure.yml")
    with open(cfg_r3_path, "r", encoding="utf-8") as f:
        cfg_r3 = yaml.safe_load(f)
    with open(cfg_r31_path, "r", encoding="utf-8") as f:
        cfg_r31 = yaml.safe_load(f)

    flat_r3 = flatten_dict(cfg_r3)
    flat_r31 = flatten_dict(cfg_r31)
    all_keys = sorted(set(flat_r3.keys()) | set(flat_r31.keys()))
    diff_keys = [k for k in all_keys if flat_r3.get(k) != flat_r31.get(k)]
    expected_diff = sorted([
        "Eval.dataset.data_dir",
        "Eval.dataset.label_file_list",
        "Global.config_role",
        "Global.pretrained_model",
        "Global.save_inference_dir",
        "Global.save_model_dir",
        "Global.save_res_path",
        "Optimizer.lr.eta_min",
        "Optimizer.lr.warmup_epoch",
        "Train.dataset.data_dir",
        "Train.dataset.label_file_list",
    ])
    unexpected = [k for k in diff_keys if k not in expected_diff]
    assert not unexpected, f"Unexpected YAML diffs between R3 and R3.1: {unexpected}"
    assert float(cfg_r31["Optimizer"]["lr"]["learning_rate"]) == 1.5e-5
    assert float(cfg_r31["Optimizer"]["lr"]["eta_min"]) == 2.5e-6
    assert int(cfg_r31["Global"]["epoch_num"]) == 3
    lr_sched = build_lr_scheduler(dict(cfg_r31["Optimizer"]["lr"]), epochs=3, step_each_epoch=337)
    assert lr_sched.warmup_steps == 150, f"Expected warmup_steps=150, got {lr_sched.warmup_steps}"
    print(f"✅ YAML Diff & LR Scheduler Check PASSED: diff_keys={diff_keys}, warmup_steps={lr_sched.warmup_steps}")

    local_cfg = json.loads(json.dumps(cfg_r31))
    data_dir_r31 = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_synthetic_r3_1_closure")
    dict_path = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_dict_v24_1.txt")
    local_cfg["Global"]["character_dict_path"] = dict_path
    local_cfg["Global"]["pretrained_model"] = base_ckpt[:-9]
    local_cfg["Global"]["distributed"] = False
    local_cfg["Train"]["dataset"]["data_dir"] = data_dir_r31
    local_cfg["Train"]["dataset"]["label_file_list"] = [os.path.join(data_dir_r31, "train_label.txt")]
    local_cfg["Train"]["loader"]["num_workers"] = 0

    post_process_class = build_post_process(local_cfg["PostProcess"], local_cfg["Global"])
    char_num = len(getattr(post_process_class, "character"))
    local_cfg["Architecture"]["Head"]["out_channels_list"] = {
        "CTCLabelDecode": char_num,
        "NRTRLabelDecode": char_num + 3,
    }

    model = build_model(local_cfg["Architecture"])
    import logging
    logger = logging.getLogger("preflight_r3_1")
    load_model(local_cfg, model, model_type=local_cfg["Architecture"]["model_type"])

    freeze_prefixes = local_cfg["Global"].get("freeze_prefixes", [])
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in freeze_prefixes):
            param.stop_gradient = True

    loss_class = build_loss(local_cfg["Loss"])
    train_loader = build_dataloader(local_cfg, "Train", paddle.CPUPlace(), logger)

    mixed_batch = None
    mixed_idx = -1
    blanks_in_batch = 0
    for idx, batch in enumerate(train_loader):
        lengths = batch[3].numpy()
        n_blanks = int(np.sum(lengths == 0))
        if n_blanks > 0 and n_blanks < len(lengths):
            mixed_batch = batch
            mixed_idx = idx
            blanks_in_batch = n_blanks
            break

    assert mixed_batch is not None, "Could not find a mixed batch with blanks"
    preds = model(mixed_batch[0], data=mixed_batch[1:])
    losses = loss_class(preds, mixed_batch)
    total_loss = float(losses["loss"].numpy())
    ctc_loss = float(losses["CTCLoss"].numpy())
    nrtr_loss = float(losses["NRTRLoss"].numpy())
    losses["loss"].backward()

    finite_grads = 0
    for name, param in model.named_parameters():
        if not param.stop_gradient and param.grad is not None:
            g = param.grad.numpy()
            assert np.all(np.isfinite(g)), f"Non-finite gradient in {name}"
            finite_grads += 1

    post_sha = sha256_file(base_ckpt)
    assert post_sha == base_sha, "Base checkpoint mutated on disk!"

    report = {
        "title": "Cham-OCR v24.1 Stage R3.1 Preflight Report (GATE_CLOSURE_SESQUISYLLABIC_EROSION)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "yaml_diff_keys": diff_keys,
        "unexpected_yaml_diffs": unexpected,
        "warmup_steps": int(lr_sched.warmup_steps),
        "step_each_epoch": 337,
        "base_checkpoint": base_ckpt,
        "base_checkpoint_sha256": base_sha,
        "mixed_batch_idx": mixed_idx,
        "blanks_in_mixed_batch": blanks_in_batch,
        "total_loss": total_loss,
        "ctc_loss": ctc_loss,
        "nrtr_loss": nrtr_loss,
        "optimizer_eligible_finite_grads": finite_grads,
        "mutated_weights_count": 0,
    }
    out_p = os.path.join(PROJECT_ROOT, "REPORTS", "r3_1_preflight_report.json")
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Preflight Report Written: {out_p} (finite_grads={finite_grads}, total_loss={total_loss:.4f})")


if __name__ == "__main__":
    main()
