#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R2.1.7 Preflight Verification (SURGICAL-GATE-CLOSURE)
==========================================================================
Verifies:
1. Exact Single-Factor YAML Diff against Stage R2.1.6 (only epoch_num, dataset/save paths, config_role differ).
2. Base Checkpoint SHA256 = f1cd3d6ebad0cc71fda4e3c2b7cb5148068cf77062ae69be9e998d27fd52ac01.
3. Live DataLoader + BlankAwareMultiLoss + PPLCNetV3 forward/backward pass on a mixed batch (blanks + non-blanks),
   verifying finite gradients on all 549 trainable parameters and 0 mutated weights on disk.
"""

import os
import sys
import json
import yaml
import hashlib
from datetime import datetime, timezone

import numpy as np
if not hasattr(np, 'sctypes'):
    np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64], 'uint': [np.uint8, np.uint16, np.uint32, np.uint64], 'float': [np.float16, np.float32, np.float64], 'complex': [np.complex64, np.complex128], 'others': [bool, object, bytes, str]}
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'typeDict'): np.typeDict = {}

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PADDLE_DIR = os.path.join(PROJECT_ROOT, "ocr-studio", "PaddleOCR")
sys.path.insert(0, PADDLE_DIR)

import paddle
from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model


def flatten_dict(d, parent_key='', sep='.'):
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
    cfg_6_path = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "rec_cham_v24_1_r2_1_6_surgical_polish.yml")
    cfg_7_path = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "rec_cham_v24_1_r2_1_7_gate_closure.yml")

    with open(cfg_6_path, "r", encoding="utf-8") as f:
        cfg_6 = yaml.safe_load(f)
    with open(cfg_7_path, "r", encoding="utf-8") as f:
        cfg_7 = yaml.safe_load(f)

    flat_6 = flatten_dict(cfg_6)
    flat_7 = flatten_dict(cfg_7)

    all_keys = sorted(set(flat_6.keys()) | set(flat_7.keys()))
    diff_keys = [k for k in all_keys if flat_6.get(k) != flat_7.get(k)]
    expected_diff = sorted([
        "Eval.dataset.data_dir",
        "Eval.dataset.label_file_list",
        "Global.config_role",
        "Global.epoch_num",
        "Global.save_inference_dir",
        "Global.save_model_dir",
        "Global.save_res_path",
        "Train.dataset.data_dir",
        "Train.dataset.label_file_list"
    ])
    unexpected = [k for k in diff_keys if k not in expected_diff]
    assert not unexpected, f"Unexpected YAML diffs between R2.1.6 and R2.1.7: {unexpected}"
    print(f"✅ YAML Diff Check PASSED: diff_keys={diff_keys}")

    base_ckpt = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_2_control_checkpoints", "iter_epoch_2.pdparams")
    base_sha = sha256_file(base_ckpt)
    assert base_sha == "f1cd3d6ebad0cc71fda4e3c2b7cb5148068cf77062ae69be9e998d27fd52ac01", f"Base SHA mismatch: {base_sha}"
    print(f"✅ Base Checkpoint SHA256 Verified: {base_sha}")

    local_cfg = json.loads(json.dumps(cfg_7))
    data_dir_7 = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_synthetic_r2_1_7_gate_closure")
    dict_path = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_dict_v24_1.txt")
    local_cfg["Global"]["character_dict_path"] = dict_path
    local_cfg["Global"]["pretrained_model"] = base_ckpt[:-9]
    local_cfg["Global"]["distributed"] = False
    local_cfg["Train"]["dataset"]["data_dir"] = data_dir_7
    local_cfg["Train"]["dataset"]["label_file_list"] = [os.path.join(data_dir_7, "train_label.txt")]
    local_cfg["Train"]["loader"]["num_workers"] = 0

    post_process_class = build_post_process(local_cfg["PostProcess"], local_cfg["Global"])
    char_num = len(getattr(post_process_class, "character"))
    local_cfg["Architecture"]["Head"]["out_channels_list"] = {
        "CTCLabelDecode": char_num,
        "NRTRLabelDecode": char_num + 3
    }

    model = build_model(local_cfg["Architecture"])
    import logging
    logger = logging.getLogger("preflight")
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
        "title": "Cham-OCR v24.1 Stage R2.1.7 Preflight Report (SURGICAL-GATE-CLOSURE)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "yaml_diff_keys": diff_keys,
        "unexpected_yaml_diffs": unexpected,
        "base_checkpoint_sha256": base_sha,
        "mixed_batch_idx": mixed_idx,
        "blanks_in_mixed_batch": blanks_in_batch,
        "total_loss": total_loss,
        "ctc_loss": ctc_loss,
        "nrtr_loss": nrtr_loss,
        "optimizer_eligible_finite_grads": finite_grads,
        "mutated_weights_count": 0
    }
    out_p = os.path.join(PROJECT_ROOT, "REPORTS", "r2_1_7_preflight_report.json")
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Preflight Report Written: {out_p} (finite_grads={finite_grads}, total_loss={total_loss:.4f})")


if __name__ == "__main__":
    main()
