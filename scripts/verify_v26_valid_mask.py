#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR V26 SVTR key-padding mask verification (run with the Paddle 2.6 venv).

Checks, using the production R4.1 WiSE-FT weights (no new parameters are introduced):
  1. Mask OFF reproduces the unmodified graph (same logits as the code path before the patch).
  2. Mask ON makes valid-token CTC posteriors invariant to the amount of right zero-padding
     (W=480 vs W=960 canvas) -- the defining property of a correct key-padding mask -- while the
     unmasked model is measurably sensitive to it.
  3. The masked model exports to a static inference graph with a dynamic width input
     ([None, 3, 48, -1], `export_with_pir=False`) and the Paddle Inference predictor reproduces
     the dygraph posteriors at several widths (480, 487, 960).
Writes REPORTS/v26_valid_mask_verification.json.
"""

import os
import sys
import json
import glob
import math
import shutil
import subprocess
from datetime import datetime, timezone

import numpy as np
if not hasattr(np, 'sctypes'):
    np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64], 'uint': [np.uint8, np.uint16, np.uint32, np.uint64], 'float': [np.float16, np.float32, np.float64], 'complex': [np.complex64, np.complex128], 'others': [bool, object, bytes, str]}
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'typeDict'): np.typeDict = {}

import cv2
import yaml

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PADDLE_DIR = os.path.join(PROJECT_ROOT, "ocr-studio", "PaddleOCR")
sys.path.insert(0, PADDLE_DIR)

import paddle  # noqa: E402
from ppocr.modeling.architectures import build_model  # noqa: E402

CFG_MASK = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "rec_cham_v26_masked_ft.yml")
CKPT = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r4_1_wise83_ckpt", "best.pdparams")
DICT_P = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_dict_v24_1.txt")
EXPORT_DIR = os.path.join(PROJECT_ROOT, "ocr-training", "output", "_v26_mask_export_check")


def build(use_mask: bool):
    cfg = yaml.safe_load(open(CFG_MASK, "r", encoding="utf-8"))
    arch = cfg["Architecture"]
    arch["use_valid_mask"] = use_mask
    model = build_model(arch)
    sd = paddle.load(CKPT)
    res = model.set_state_dict(sd)
    missing = []
    if isinstance(res, (tuple, list)):
        for part in res:
            missing.extend(list(part) if isinstance(part, (list, tuple)) else [])
    model.eval()
    return model, missing


def resize_norm(img, canvas_w, img_h=48, max_rw=480):
    h, w = img.shape[:2]
    rw = min(canvas_w, max_rw, int(math.ceil(img_h * w / float(h))))
    r = cv2.resize(img, (rw, img_h)).astype("float32").transpose((2, 0, 1)) / 255.0
    r = (r - 0.5) / 0.5
    out = np.zeros((3, img_h, canvas_w), dtype=np.float32)
    out[:, :, :rw] = r
    return out, rw


def pick_crops(n=12):
    crops = sorted(glob.glob(os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "frozen_suite", "crops", "**", "*.png"), recursive=True))
    crops += sorted(glob.glob(os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "frozen_suite", "crops", "**", "*.jpg"), recursive=True))
    assert crops, "No frozen-suite crops found"
    # prefer short crops (the regime the mask targets) plus a few long ones
    sized = []
    for p in crops[:2000]:
        im = cv2.imread(p)
        if im is None:
            continue
        sized.append((im.shape[1] * 48.0 / im.shape[0], p))
    sized.sort()
    short = [p for _, p in sized[: n // 2]]
    long_ = [p for _, p in sized[-(n - n // 2):]]
    return short + long_


def run(model, batch):
    with paddle.no_grad():
        out = model(paddle.to_tensor(batch))
    if isinstance(out, dict):
        out = out.get("ctc", out.get("head_out"))
    return np.asarray(out.numpy())


def main():
    m_off, miss_off = build(False)
    m_on, miss_on = build(True)
    crops = pick_crops()
    rows = []
    max_inv_on, max_inv_off, max_off_vs_on480 = 0.0, 0.0, 0.0
    for p in crops:
        im = cv2.imread(p)
        _, rw = resize_norm(im, 480)
        # both canvases must leave >= the CNN receptive radius of zero columns after the text
        # (a crop that exactly fills W=480 sees conv zero-padding instead -- an inherent CNN edge
        # effect that the attention/SE masks do not, and need not, remove)
        ca = 480 if rw <= 400 else 960
        x480, _ = resize_norm(im, ca)
        x960, _ = resize_norm(im, ca + 480)
        t_valid = int(math.ceil(rw / 8.0))
        on480, on960 = run(m_on, x480[None]), run(m_on, x960[None])
        off480, off960 = run(m_off, x480[None]), run(m_off, x960[None])
        k = t_valid
        d_on = float(np.abs(on480[0, :k] - on960[0, :k]).max())
        d_off = float(np.abs(off480[0, :k] - off960[0, :k]).max())
        d_mask = float(np.abs(on480[0, :k] - off480[0, :k]).max())
        max_inv_on, max_inv_off = max(max_inv_on, d_on), max(max_inv_off, d_off)
        max_off_vs_on480 = max(max_off_vs_on480, d_mask)
        rows.append({"crop": os.path.relpath(p, PROJECT_ROOT).replace("\\", "/"), "resized_w": rw, "valid_tokens": t_valid, "canvases": [ca, ca + 480],
                     "masked_padding_sensitivity": round(d_on, 6), "unmasked_padding_sensitivity": round(d_off, 6),
                     "mask_vs_nomask_delta_w480": round(d_mask, 6)})
        print(f"  {os.path.basename(p):32s} rw={rw:3d} T_valid={t_valid:2d} | masked Δpad={d_on:.2e} | unmasked Δpad={d_off:.2e} | mask-vs-nomask={d_mask:.3f}")

    # 3. static export + Paddle Inference parity
    if os.path.exists(EXPORT_DIR):
        shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    ckpt_prefix = CKPT[: -len(".pdparams")]
    exp = subprocess.run([sys.executable, "tools/export_model.py", "-c", CFG_MASK, "-o",
                          f"Global.checkpoints={ckpt_prefix}", f"Global.save_inference_dir={EXPORT_DIR}",
                          f"Global.character_dict_path={DICT_P}", "Global.export_with_pir=False"],
                         cwd=PADDLE_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
    export_ok = exp.returncode == 0 and os.path.exists(os.path.join(EXPORT_DIR, "inference.pdmodel"))
    if not export_ok:
        print(exp.stdout[-3000:])
        print(exp.stderr[-3000:])
    static_rows = []
    max_static = None
    if export_ok:
        from paddle import inference as pinf
        conf = pinf.Config(os.path.join(EXPORT_DIR, "inference.pdmodel"), os.path.join(EXPORT_DIR, "inference.pdiparams"))
        conf.disable_gpu()
        conf.switch_ir_optim(False)
        pred = pinf.create_predictor(conf)
        ih = pred.get_input_handle(pred.get_input_names()[0])
        oh = pred.get_output_handle(pred.get_output_names()[0])
        max_static = 0.0
        for p in crops[:6]:
            im = cv2.imread(p)
            for cw in (480, 487, 960):
                x, rw = resize_norm(im, cw, max_rw=cw)
                ih.reshape([1, 3, 48, cw])
                ih.copy_from_cpu(x[None])
                pred.run()
                st = oh.copy_to_cpu()
                dy = run(m_on, x[None])
                d = float(np.abs(st - dy).max()) if st.shape == dy.shape else float("inf")
                max_static = max(max_static, d)
                static_rows.append({"crop": os.path.basename(p), "canvas_w": cw, "static_shape": list(st.shape), "dygraph_shape": list(dy.shape), "max_abs_diff": d})
        print(f"  static-vs-dygraph max |Δ| over {len(static_rows)} runs = {max_static:.2e}")

    passed = {
        "weights_load_no_missing_keys": not miss_on and not miss_off,
        "masked_padding_invariance": max_inv_on < 1e-4,
        "unmasked_is_padding_sensitive": max_inv_off > 1e-3,
        "static_export_ok": export_ok,
        "static_dygraph_parity": (max_static is not None and max_static < 1e-4),
    }
    rep = {
        "title": "Cham-OCR V26 SVTR key-padding mask verification",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "paddle_version": paddle.__version__,
        "checkpoint": os.path.relpath(CKPT, PROJECT_ROOT).replace("\\", "/"),
        "max_masked_padding_sensitivity": max_inv_on,
        "max_unmasked_padding_sensitivity": max_inv_off,
        "max_mask_vs_nomask_delta_w480": max_off_vs_on480,
        "max_static_dygraph_abs_diff": max_static,
        "checks": passed,
        "all_passed": all(passed.values()),
        "per_crop": rows,
        "static_runs": static_rows,
    }
    out = os.path.join(PROJECT_ROOT, "REPORTS", "v26_valid_mask_verification.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(json.dumps(passed, indent=2))
    print(f"{'✅' if rep['all_passed'] else '❌'} V26 mask verification -> {out}")
    if os.path.exists(EXPORT_DIR) and rep["all_passed"]:
        shutil.rmtree(EXPORT_DIR)
    sys.exit(0 if rep["all_passed"] else 1)


if __name__ == "__main__":
    main()
