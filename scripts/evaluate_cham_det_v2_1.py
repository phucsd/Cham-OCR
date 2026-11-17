#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR Cham-DBNet v2.1 (`ch_PP-OCRv4_det_cham_v2_1`) Head-to-Head Evaluation & Promotion Gate
===============================================================================================
Benchmarks `Cham-DBNet v1` vs `Cham-DBNet v2.1` head-to-head across ALL 4 detector & E2E benchmarks:
  1. `detector_v3_hb` Val-120 (Multi-layout rectilinear/tilted HarfBuzz pages)
  2. `detector_v3_1_hybrid` Val-160 (50% Sinusoidal Wave Level-5 + Multi-layout pages)
  3. `RealScan-Gold v1 / harum_palei_01` (25-line authentic camera photo) + `RealScan-Gold v2` (12 OOD archival pages)
  4. `Level-5 Full-Page` (`short:5`, `a4:5`, `audit_18` in production `DET_MODE='hybrid'`)
Promotes `Cham-DBNet v2.1` to `ocr-studio/data/output/ch_PP-OCRv4_det_cham_infer/` ONLY IF it passes
strict non-regression across BOTH `Val-120`/`Val-160` AND `Level-5 / audit_18` + `harum_palei_01`!
"""

import os
import re
import sys
import json
import shutil
import subprocess
from datetime import datetime, timezone

import cv2
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIO_DIR = os.path.join(PROJECT_ROOT, "ocr-studio")
PADDLE_DIR = os.path.join(STUDIO_DIR, "PaddleOCR")
PY26 = os.path.join(PROJECT_ROOT, ".venv_paddle26", "Scripts", "python.exe")
PY313 = r"C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe"
if not os.path.exists(PY313):
    PY313 = sys.executable

sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, STUDIO_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts"))

from evaluate_r2_1_7_causal import parse_train_log
from evaluate_cham_det_v2 import sha256_file, build_det, eval_det_val_120, eval_end_to_end_with_detector


def _poly_bbox_quad(pts):
    arr = np.array(pts, dtype=np.float32)
    rect = cv2.minAreaRect(arr)
    return cv2.boxPoints(rect)


def eval_det_val_curved(det_model, val_root: str) -> dict:
    """Evaluates IoU Recall, Precision, H-Mean & Usable Crop % on curved 16-vertex or 4-vertex val labels."""
    from evaluate_cham_det_v2 import _poly_area, _poly_inter
    lbl_p = os.path.join(val_root, "det_val_label.txt")
    lines = [l.strip().split("\t", 1) for l in open(lbl_p, "r", encoding="utf-8") if "\t" in l]
    tot_gt, tot_dt, rec_gt, prec_dt, usable = 0, 0, 0, 0, 0
    for rel, js in lines:
        gt_boxes = json.loads(js)
        img = cv2.imread(os.path.join(val_root, rel.replace("/", os.sep)))
        dt_boxes, _ = det_model(img)
        if dt_boxes is None:
            dt_boxes = []
        gt_quads = [_poly_bbox_quad(g["points"]) for g in gt_boxes]
        for gq in gt_quads:
            ga = _poly_area(gq)
            best_iou, best_in = 0.0, 0.0
            for dp in dt_boxes:
                inter = _poly_inter(gq, dp)
                u = ga + _poly_area(dp) - inter
                iou = (inter / u) if u > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_in = iou, inter
            if best_iou >= 0.50:
                rec_gt += 1
            if ga > 0 and (best_in / ga) >= 0.85:
                usable += 1
        for dp in dt_boxes:
            da = _poly_area(dp)
            best_iou = 0.0
            for gq in gt_quads:
                inter = _poly_inter(gq, dp)
                u = _poly_area(gq) + da - inter
                iou = (inter / u) if u > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
            if best_iou >= 0.50:
                prec_dt += 1
        tot_gt += len(gt_quads)
        tot_dt += len(dt_boxes)
    r = rec_gt / max(1, tot_gt)
    p = prec_dt / max(1, tot_dt)
    hm = (2 * r * p / (r + p)) if (r + p) > 0 else 0.0
    return {
        "pages": len(lines),
        "gt_lines": tot_gt,
        "dt_lines": tot_dt,
        "recall_pct": round(r * 100.0, 2),
        "precision_pct": round(p * 100.0, 2),
        "hmean_pct": round(hm * 100.0, 2),
        "usable_crop_pct": round(100.0 * usable / max(1, tot_gt), 2),
    }


def main():
    dl_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "det_v2_1_kaggle_output")
    ext_dir = os.path.join(dl_dir, "extracted_checkpoints")
    ckpt_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "det_v2_1_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    for fn in os.listdir(ext_dir):
        fp = os.path.join(ext_dir, fn)
        if os.path.isfile(fp):
            shutil.copy2(fp, os.path.join(ckpt_dir, fn))

    accounting = parse_train_log(
        os.path.join(ckpt_dir, "train.log"),
        total_epochs=20,
        kernel_log_path=os.path.join(dl_dir, "paddleocr-cham-det-v2-1-hybrid.log"),
    )

    val_eval_p = os.path.join(dl_dir, "det_v2_1_val_eval.json")
    best_ep, best_hm = "latest", -1.0
    if os.path.exists(val_eval_p):
        ve = json.load(open(val_eval_p, "r", encoding="utf-8"))
        for ep in (10, 15, "latest"):
            matches = re.findall(r"hmean:\s*([0-9.]+)", ve.get(f"epoch_{ep}", ""))
            if matches and float(matches[-1]) >= best_hm:
                best_hm = float(matches[-1])
                best_ep = ep

    ckpt_prefix = os.path.join(ckpt_dir, f"iter_epoch_{best_ep}" if isinstance(best_ep, int) else "latest")
    if not os.path.exists(ckpt_prefix + ".pdparams"):
        ckpt_prefix = os.path.join(ckpt_dir, "latest")
    ckpt_sha = sha256_file(ckpt_prefix + ".pdparams")

    cfg_path = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "det", "ch_PP-OCRv4_det_cham_v2_1.yml")
    v21_infer_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "ch_PP-OCRv4_det_cham_v2_1_dual_infer")
    if os.path.exists(v21_infer_dir):
        shutil.rmtree(v21_infer_dir)
    os.makedirs(v21_infer_dir, exist_ok=True)

    subprocess.run([PY26, "tools/export_model.py", "-c", cfg_path, "-o", f"Global.checkpoints={ckpt_prefix}", f"Global.save_inference_dir={v21_infer_dir}", "Global.export_with_pir=False"], cwd=PADDLE_DIR, check=True)
    tmp_pir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "_tmp_pir_det_v2_1")
    if os.path.exists(tmp_pir):
        shutil.rmtree(tmp_pir)
    os.makedirs(tmp_pir, exist_ok=True)
    subprocess.run([PY313, "tools/export_model.py", "-c", cfg_path, "-o", f"Global.checkpoints={ckpt_prefix}", f"Global.save_inference_dir={tmp_pir}"], cwd=PADDLE_DIR, check=True)
    shutil.copy2(os.path.join(tmp_pir, "inference.json"), os.path.join(v21_infer_dir, "inference.json"))
    if os.path.exists(os.path.join(tmp_pir, "inference.yml")):
        shutil.copy2(os.path.join(tmp_pir, "inference.yml"), os.path.join(v21_infer_dir, "inference.yml"))
    shutil.rmtree(tmp_pir)

    v1_infer_dir = os.path.join(STUDIO_DIR, "data", "output", "ch_PP-OCRv4_det_cham_infer")
    bkp_v1 = os.path.join(PROJECT_ROOT, "ocr-training", "output", "ch_PP-OCRv4_det_cham_v1_backup")
    ref_v1_dir = bkp_v1 if os.path.exists(bkp_v1) else v1_infer_dir

    val_120_root = os.path.join(PROJECT_ROOT, "ocr-training", "data", "detector_v3_hb")
    val_160_root = os.path.join(PROJECT_ROOT, "ocr-training", "data", "detector_v3_1_hybrid")

    v1_det = build_det(ref_v1_dir)
    v21_det = build_det(v21_infer_dir)

    v1_val120 = eval_det_val_120(v1_det, val_120_root)
    v21_val120 = eval_det_val_120(v21_det, val_120_root)
    v1_val160 = eval_det_val_curved(v1_det, val_160_root)
    v21_val160 = eval_det_val_curved(v21_det, val_160_root)

    v1_e2e = eval_end_to_end_with_detector(ref_v1_dir, "v24_1_r4")
    v21_e2e = eval_end_to_end_with_detector(v21_infer_dir, "v24_1_r4")

    v1_h_cer = v1_e2e["harum_palei_01"]["e2e_doc_cer_pct"]
    v21_h_cer = v21_e2e["harum_palei_01"]["e2e_doc_cer_pct"]
    v1_aud = (v1_e2e["level5_and_audit"].get("audit_18") or {}).get("doc_cer_pct", 0.0)
    v21_aud = (v21_e2e["level5_and_audit"].get("audit_18") or {}).get("doc_cer_pct", 0.0)
    v1_a4 = (v1_e2e["level5_and_audit"].get("a4:5") or {}).get("doc_cer_pct", 0.0)
    v21_a4 = (v21_e2e["level5_and_audit"].get("a4:5") or {}).get("doc_cer_pct", 0.0)
    v1_sh = (v1_e2e["level5_and_audit"].get("short:5") or {}).get("doc_cer_pct", 0.0)
    v21_sh = (v21_e2e["level5_and_audit"].get("short:5") or {}).get("doc_cer_pct", 0.0)

    promote_v21 = (
        v21_val120["hmean_pct"] >= v1_val120["hmean_pct"]
        and v21_val160["hmean_pct"] >= v1_val160["hmean_pct"]
        and v21_h_cer <= v1_h_cer + 0.25
        and v21_e2e["harum_palei_01"]["line_recall_pct"] >= 96.0
        and v21_aud <= v1_aud + 0.30
        and v21_a4 <= v1_a4 + 0.50
        and v21_sh <= v1_sh + 0.50
    )

    promoted_action = "kept_v1_no_promotion"
    if promote_v21:
        if not os.path.exists(bkp_v1):
            shutil.copytree(v1_infer_dir, bkp_v1)
        for fn in os.listdir(v21_infer_dir):
            shutil.copy2(os.path.join(v21_infer_dir, fn), os.path.join(v1_infer_dir, fn))
        promoted_action = "promoted_v2_1_to_ch_PP-OCRv4_det_cham_infer"
        print(f"🚀 PROMOTED Cham-DBNet v2.1 -> {v1_infer_dir} (v1 backed up at {bkp_v1})")
    else:
        print(f"🛡️ Kept Cham-DBNet v1 in production (v2.1 archived at {v21_infer_dir})")

    report = {
        "title": "Cham-OCR Cham-DBNet v2.1 Hybrid Wave+MultiLayout Evaluation & Promotion Gate",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "selected_epoch": best_ep,
        "checkpoint_sha256": ckpt_sha,
        "ground_truth_training_accounting": accounting,
        "val_120_harfbuzz": {"cham_dbnet_v1": v1_val120, "cham_dbnet_v2_1": v21_val120},
        "val_160_hybrid_wave": {"cham_dbnet_v1": v1_val160, "cham_dbnet_v2_1": v21_val160},
        "end_to_end": {"cham_dbnet_v1": v1_e2e, "cham_dbnet_v2_1": v21_e2e},
        "promote_v2_1": promote_v21,
        "promotion_action": promoted_action,
    }
    out_j = os.path.join(PROJECT_ROOT, "REPORTS", "cham_dbnet_v2_1_evaluation.json")
    with open(out_j, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md = [
        "# Đánh giá `Cham-DBNet v2.1` (Unified Sinusoidal Wave `Level-5` + HarfBuzz Multi-Layout)",
        f"- **Thời điểm**: `{report['evaluated_at']}`",
        f"- **Epoch tốt nhất**: Epoch `{best_ep}` (`SHA256: {ckpt_sha[:16]}...`)",
        f"- **Train Loop / Overhead / Total Wall-clock**: `{accounting.get('train_loop_duration_seconds')}s` / `{accounting.get('pipeline_container_overhead_seconds')}s` / `{accounting.get('total_kernel_wall_clock_seconds')}s` ({accounting.get('total_kernel_wall_clock_minutes')} phút)",
        f"- **Quyết định Promotion Gate**: `{promoted_action}` (`promote_v2_1={promote_v21}`)",
        "",
        "## 1. Held-Out Validation (`Val-120` Multi-Layout & `Val-160` Hybrid Wave+Multi-Layout)",
        "| Detector | `Val-120` H-Mean (`R / P`) | `Val-160` Wave+Multi H-Mean (`R / P`) | `Val-160` Usable Crop (`>=85%`) |",
        "| :--- | :---: | :---: | :---: |",
        f"| `Cham-DBNet v1` | **{v1_val120['hmean_pct']:.2f}%** ({v1_val120['recall_pct']}% / {v1_val120['precision_pct']}%) | **{v1_val160['hmean_pct']:.2f}%** ({v1_val160['recall_pct']}% / {v1_val160['precision_pct']}%) | {v1_val160['usable_crop_pct']:.2f}% |",
        f"| `Cham-DBNet v2.1` | **{v21_val120['hmean_pct']:.2f}%** ({v21_val120['recall_pct']}% / {v21_val120['precision_pct']}%) | **{v21_val160['hmean_pct']:.2f}%** ({v21_val160['recall_pct']}% / {v21_val160['precision_pct']}%) | {v21_val160['usable_crop_pct']:.2f}% |",
        "",
        "## 2. End-to-End Page Benchmarks (`DET_MODE='hybrid'`, cố định Recognizer `v24_1_r4`)",
        "| Detector | `harum_palei_01` E2E CER | `short:5` Doc CER | `a4:5` Doc CER | `audit_18` Doc CER |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| `Cham-DBNet v1` | {v1_h_cer:.2f}% ({v1_e2e['harum_palei_01']['pred_lines']}/25) | {v1_sh}% ({(v1_e2e['level5_and_audit'].get('short:5') or {}).get('lines')}) | {v1_a4}% ({(v1_e2e['level5_and_audit'].get('a4:5') or {}).get('lines')}) | **{v1_aud}%** ({(v1_e2e['level5_and_audit'].get('audit_18') or {}).get('lines')}) |",
        f"| `Cham-DBNet v2.1` | {v21_h_cer:.2f}% ({v21_e2e['harum_palei_01']['pred_lines']}/25) | {v21_sh}% ({(v21_e2e['level5_and_audit'].get('short:5') or {}).get('lines')}) | {v21_a4}% ({(v21_e2e['level5_and_audit'].get('a4:5') or {}).get('lines')}) | **{v21_aud}%** ({(v21_e2e['level5_and_audit'].get('audit_18') or {}).get('lines')}) |",
    ]
    out_m = os.path.join(PROJECT_ROOT, "REPORTS", "CHAM_DBNET_V2_1_EVALUATION.md")
    with open(out_m, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"✅ Cham-DBNet v2.1 evaluation saved -> {out_m}")


if __name__ == "__main__":
    main()
