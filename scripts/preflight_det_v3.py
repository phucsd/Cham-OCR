#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR Cham-DBNet v3 (`ch_PP-OCRv4_det_cham_v3`) Preflight Verification
=========================================================================
Verifies:
  1. `det_v2_1_checkpoints/iter_epoch_10.pdparams` SHA-256 and `905/905` 100% tensor load.
  2. Live Train DataLoader (`seed=1024`, 16-vertex full-cell curved polygons -> `MakeShrinkMap` -> `MakeBorderMap`)
     has `shrink_pos > 0` and zero `pyclipper` / `IndexError` exceptions across 8 batches.
  3. Live Eval DataLoader + `DetectionIoUEvaluator` (`Shapely` validity check on 16-vertex curved GT).
  4. Forward + `DBLoss` + Backward finite gradient verification.
"""

import os
import sys
import json
import yaml
import hashlib
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

import importlib.util
import paddle
from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.utils.save_load import load_model

_dm_spec = importlib.util.spec_from_file_location("eval_det_iou_mod", os.path.join(PADDLE_DIR, "ppocr", "metrics", "eval_det_iou.py"))
_dm_mod = importlib.util.module_from_spec(_dm_spec)
_dm_spec.loader.exec_module(_dm_mod)
DetectionIoUEvaluator = _dm_mod.DetectionIoUEvaluator


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def main():
    base_ckpt = os.path.join(PROJECT_ROOT, "ocr-training", "output", "det_v2_1_checkpoints", "iter_epoch_10.pdparams")
    assert os.path.exists(base_ckpt), f"Missing {base_ckpt}"
    base_sha = sha256_file(base_ckpt)

    cfg_path = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "det", "ch_PP-OCRv4_det_cham_v3.yml")
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    local_cfg = json.loads(json.dumps(cfg))
    data_dir = os.path.join(PROJECT_ROOT, "ocr-training", "data", "detector_v3_2_poly")
    local_cfg["Global"]["pretrained_model"] = base_ckpt[:-9]
    local_cfg["Global"]["distributed"] = False
    local_cfg["Train"]["dataset"]["data_dir"] = data_dir
    local_cfg["Train"]["dataset"]["label_file_list"] = [os.path.join(data_dir, "det_train_label.txt")]
    local_cfg["Train"]["loader"]["num_workers"] = 0
    local_cfg["Eval"]["dataset"]["data_dir"] = data_dir
    local_cfg["Eval"]["dataset"]["label_file_list"] = [os.path.join(data_dir, "det_val_label.txt")]
    local_cfg["Eval"]["loader"]["num_workers"] = 0

    model = build_model(local_cfg["Architecture"])
    raw_sd = paddle.load(base_ckpt)
    assert set(model.state_dict().keys()) == set(raw_sd.keys()), "DBNet v3 key mismatch!"
    load_model(local_cfg, model, model_type="det")

    logger = logging.getLogger("preflight_det_v3")
    train_loader = build_dataloader(local_cfg, "Train", paddle.CPUPlace(), logger, seed=1024)
    eval_loader = build_dataloader(local_cfg, "Eval", paddle.CPUPlace(), logger)

    first_batch = None
    total_pos = 0
    for b_idx, batch in enumerate(train_loader):
        pos_px = int((batch[3].numpy() > 0.5).sum())
        assert pos_px > 0, f"Zero positive shrink_map pixels in batch {b_idx}!"
        total_pos += pos_px
        if first_batch is None:
            first_batch = batch
        if b_idx >= 7:
            break

    loss_fn = build_loss(local_cfg["Loss"])
    preds = model(first_batch[0])
    losses = loss_fn(preds, first_batch)
    losses["loss"].backward()

    finite_grads = 0
    for _, p in model.named_parameters():
        if p.grad is not None:
            assert np.all(np.isfinite(p.grad.numpy()))
            finite_grads += 1

    iou_eval = DetectionIoUEvaluator()
    per_sample = []
    for e_idx, e_batch in enumerate(eval_loader):
        gt_polys = e_batch[2].numpy()[0]
        ignore_tags = e_batch[3].numpy()[0]
        gts = [{"points": p, "ignore": bool(ig)} for p, ig in zip(gt_polys, ignore_tags)]
        preds_sim = [{"points": p, "ignore": False} for p, ig in zip(gt_polys, ignore_tags) if not bool(ig)]
        per_sample.append(iou_eval.evaluate_image(gts, preds_sim))
        if e_idx >= 15:
            break
    m_res = iou_eval.combine_results(per_sample)
    assert m_res["hmean"] > 0.95, f"DetMetric self-check failed on 16-vertex GT: {m_res}"

    rep = {
        "stage": "CHAM_DBNET_V3_POLY",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "base_ckpt_sha256": base_sha,
        "loaded_tensors": len(model.state_dict()),
        "eight_batch_shrink_pos_pixels": total_pos,
        "sample_losses": {k: round(float(np.asarray(v.numpy()).item()), 5) for k, v in losses.items()},
        "finite_grad_tensors": finite_grads,
        "det_metric_16vertex_self_hmean": round(float(m_res["hmean"]), 4),
    }
    out_p = os.path.join(PROJECT_ROOT, "REPORTS", "det_v3_preflight_report.json")
    with open(out_p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=2)
    print(f"✅ Cham-DBNet v3 Preflight PASSED: {rep}")


if __name__ == "__main__":
    main()
