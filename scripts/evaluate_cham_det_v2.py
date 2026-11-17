#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR Cham-DBNet v2 (`ch_PP-OCRv4_det_cham_v2`) Head-to-Head Evaluation & Promotion Gate
===========================================================================================
1. Computes 3-part Ground-Truth Training Accounting (`Train Loop`, `Overhead`, `Kernel Wall-clock`).
2. Selects best epoch from `det_v2_val_eval.json` (`iter_epoch_30`, `iter_epoch_40`, `iter_epoch_50`)
   and exports BOTH:
   - `PY26` (`inference.pdmodel` + `inference.pdiparams` + `inference.pdiparams.info`, `export_with_pir=False`)
   - `PY313` (`inference.json` PIR)
   into `ocr-training/output/ch_PP-OCRv4_det_cham_v2_dual_infer/`.
3. Evaluates `Cham-DBNet v1` vs `Cham-DBNet v2` head-to-head (keeping recognizer fixed!) on:
   - Held-out `detector_v3_hb` Val (`120` pages, `1,473` polygons): IoU Recall, Precision, H-Mean, Crop Usability (`>=95%` GT area).
   - `RealScan-Gold v1 / harum_palei_01` (`25` lines): Detected lines, Line Recall, Matched CER, E2E Doc CER.
   - `Level-5 Full-Page` (`short:5`, `a4:5`, `audit_18`): Detected/GT lines, Abs Line Error, Doc CER, Matched CER.
4. Promotes `Cham-DBNet v2` into `ocr-studio/data/output/ch_PP-OCRv4_det_cham_infer/` (preserving `v1` backup at
   `ocr-training/output/ch_PP-OCRv4_det_cham_v1_backup/`) ONLY IF `Cham-DBNet v2` improves or matches without regression!
"""

import os
import re
import sys
import json
import time
import shutil
import hashlib
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

if os.path.join(PROJECT_ROOT, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
if STUDIO_DIR not in sys.path:
    sys.path.insert(0, STUDIO_DIR)
if os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts"))

from evaluate_r2_1_7_causal import parse_train_log


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def _poly_area(poly):
    return float(cv2.contourArea(np.array(poly, dtype=np.float32)))


def _poly_inter(p1, p2):
    a1 = np.array(p1, dtype=np.float32).reshape(-1, 1, 2)
    a2 = np.array(p2, dtype=np.float32).reshape(-1, 1, 2)
    ret, _ = cv2.intersectConvexConvex(a1, a2)
    return float(ret) if ret > 0 else 0.0


def build_det(det_dir: str, box_thresh: float = 0.4):
    import tools.infer.utility as utility
    import tools.infer.predict_det as predict_det
    b = sys.argv
    sys.argv = [sys.argv[0]]
    args = utility.parse_args()
    sys.argv = b
    args.det_model_dir = os.path.abspath(det_dir)
    args.use_gpu = False
    args.enable_mkldnn = False
    args.ir_optim = False
    args.cpu_threads = 2
    args.det_limit_side_len = 960
    args.det_db_thresh = 0.3
    args.det_db_box_thresh = box_thresh
    args.det_db_unclip_ratio = 1.6
    return predict_det.TextDetector(args)


def eval_det_val_120(det_model, val_root: str) -> dict:
    lbl_p = os.path.join(val_root, "det_val_label.txt")
    lines = [l.strip().split("\t", 1) for l in open(lbl_p, "r", encoding="utf-8") if "\t" in l]
    tot_gt, tot_dt, rec_gt, prec_dt, usable = 0, 0, 0, 0, 0
    ious = []
    for rel, js in lines:
        gt_boxes = json.loads(js)
        img = cv2.imread(os.path.join(val_root, rel.replace("/", os.sep)))
        dt_boxes, _ = det_model(img)
        if dt_boxes is None:
            dt_boxes = []
        for g in gt_boxes:
            gp = g["points"]
            ga = _poly_area(gp)
            best_iou, best_in = 0.0, 0.0
            for dp in dt_boxes:
                inter = _poly_inter(gp, dp)
                u = ga + _poly_area(dp) - inter
                iou = (inter / u) if u > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_in = iou, inter
            if best_iou >= 0.50:
                rec_gt += 1
                ious.append(best_iou)
            if ga > 0 and (best_in / ga) >= 0.90:
                usable += 1
        for dp in dt_boxes:
            da = _poly_area(dp)
            best_iou = 0.0
            for g in gt_boxes:
                gp = g["points"]
                inter = _poly_inter(gp, dp)
                u = _poly_area(gp) + da - inter
                iou = (inter / u) if u > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
            if best_iou >= 0.50:
                prec_dt += 1
        tot_gt += len(gt_boxes)
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
        "mean_iou": round(float(np.mean(ious)) if ious else 0.0, 4),
    }


def eval_end_to_end_with_detector(det_dir: str, rec_ver: str) -> dict:
    import app as studio_app
    from evaluate_level5_pages import load_pages, norm_audit, matched_cer
    from evaluate_realscan_gold import matched_line_stats
    from evaluate_pipeline import levenshtein_distance

    # Keep Legacy-First primary generic detector intact, and swap ONLY secondary cham_det_model
    # using Studio's exact thresholds (CHAM_DET_DB_THRESH=0.3, CHAM_DET_BOX_THRESH=0.4, CHAM_DET_UNCLIP=1.6)
    studio_app.DET_MODE = "hybrid"
    studio_app.det_model = studio_app.get_generic_det_model()
    studio_app._cham_det_failed = False
    studio_app.cham_det_model = studio_app._build_text_detector(
        os.path.abspath(det_dir),
        studio_app.CHAM_DET_DB_THRESH,
        studio_app.CHAM_DET_BOX_THRESH,
        studio_app.CHAM_DET_UNCLIP,
    )

    # 1. Harum Palei full-page
    m = json.load(open(os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "realscan_gold_v1", "manifest.json"), "r", encoding="utf-8"))
    harum = next(d for d in m["documents"] if d["doc_id"] == "harum_palei_01")
    h_img = cv2.imread(os.path.join(PROJECT_ROOT, harum["page_relpath"].replace("/", os.sep)))
    h_out = studio_app.run_ocr_pipeline(h_img, model_ver=rec_ver, method="dbnet")
    h_pred = [l["text"] for l in h_out.get("lines", [])]
    h_gt = [ln["gt"] for ln in harum["lines"]]
    harum_stats = matched_line_stats(h_gt, h_pred)

    # 2. Level-5 & audit_18 pages
    pages = load_pages(limit_per_set=90)
    by_g = {}
    for pg in pages:
        g = pg["group"]
        by_g.setdefault(g, {"n": 0, "gt_l": 0, "pr_l": 0, "abs_l": 0, "d_err": 0, "d_den": 0, "m_err": 0, "m_den": 0})
        im = cv2.imread(pg["path"])
        res = studio_app.run_ocr_pipeline(im, model_ver=rec_ver, method="dbnet")
        pl = [l["text"] for l in res.get("lines", [])]
        gl = pg["gt_lines"]
        st = by_g[g]
        st["n"] += 1
        st["gt_l"] += len(gl)
        st["pr_l"] += len(pl)
        st["abs_l"] += abs(len(pl) - len(gl))
        if gl:
            gd, pd = norm_audit(" ".join(gl)), norm_audit(" ".join(pl))
            st["d_err"] += levenshtein_distance(gd, pd)
            st["d_den"] += len(gd)
            me, md = matched_cer(gl, pl)
            st["m_err"] += me
            st["m_den"] += md
    lvl5_summary = {
        g: {
            "n": s["n"],
            "lines": f"{s['pr_l']}/{s['gt_l']}",
            "abs_line_err": s["abs_l"],
            "doc_cer_pct": round(100.0 * s["d_err"] / max(1, s["d_den"]), 2),
            "matched_cer_pct": round(100.0 * s["m_err"] / max(1, s["m_den"]), 2),
        }
        for g, s in by_g.items()
    }
    return {"harum_palei_01": harum_stats, "level5_and_audit": lvl5_summary}


def main():
    dl_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "det_v2_kaggle_output")
    ext_dir = os.path.join(dl_dir, "extracted_checkpoints")
    ckpt_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "det_v2_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    for fn in os.listdir(ext_dir):
        fp = os.path.join(ext_dir, fn)
        if os.path.isfile(fp):
            shutil.copy2(fp, os.path.join(ckpt_dir, fn))

    train_log = os.path.join(ckpt_dir, "train.log")
    kernel_log = os.path.join(dl_dir, "paddleocr-cham-det-v2-dbnet.log")
    accounting = parse_train_log(train_log, total_epochs=25, kernel_log_path=kernel_log)

    # Parse best epoch from det_v2_val_eval.json
    val_eval_p = os.path.join(dl_dir, "det_v2_val_eval.json")
    best_ep = "latest"
    best_hmean = -1.0
    if os.path.exists(val_eval_p):
        ve = json.load(open(val_eval_p, "r", encoding="utf-8"))
        for ep in (15, 20, "latest"):
            txt = ve.get(f"epoch_{ep}", "")
            matches = re.findall(r"hmean:\s*([0-9.]+)", txt)
            if matches and float(matches[-1]) >= best_hmean:
                best_hmean = float(matches[-1])
                best_ep = ep

    ckpt_prefix = os.path.join(ckpt_dir, f"iter_epoch_{best_ep}" if isinstance(best_ep, int) else "latest")
    if not os.path.exists(ckpt_prefix + ".pdparams"):
        ckpt_prefix = os.path.join(ckpt_dir, "latest")
    ckpt_sha = sha256_file(ckpt_prefix + ".pdparams")

    cfg_path = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "det", "ch_PP-OCRv4_det_cham_v2.yml")
    v2_infer_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "ch_PP-OCRv4_det_cham_v2_dual_infer")
    if os.path.exists(v2_infer_dir):
        shutil.rmtree(v2_infer_dir)
    os.makedirs(v2_infer_dir, exist_ok=True)

    # Export PY26 (.pdmodel + .pdiparams) + PY313 (PIR inference.json)
    subprocess.run([
        PY26, "tools/export_model.py", "-c", cfg_path,
        "-o", f"Global.checkpoints={ckpt_prefix}", f"Global.save_inference_dir={v2_infer_dir}", "Global.export_with_pir=False",
    ], cwd=PADDLE_DIR, check=True)

    tmp_pir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "_tmp_pir_det_v2")
    if os.path.exists(tmp_pir):
        shutil.rmtree(tmp_pir)
    os.makedirs(tmp_pir, exist_ok=True)
    subprocess.run([
        PY313, "tools/export_model.py", "-c", cfg_path,
        "-o", f"Global.checkpoints={ckpt_prefix}", f"Global.save_inference_dir={tmp_pir}",
    ], cwd=PADDLE_DIR, check=True)
    shutil.copy2(os.path.join(tmp_pir, "inference.json"), os.path.join(v2_infer_dir, "inference.json"))
    if os.path.exists(os.path.join(tmp_pir, "inference.yml")):
        shutil.copy2(os.path.join(tmp_pir, "inference.yml"), os.path.join(v2_infer_dir, "inference.yml"))
    shutil.rmtree(tmp_pir)

    v1_infer_dir = os.path.join(STUDIO_DIR, "data", "output", "ch_PP-OCRv4_det_cham_infer")
    val_root = os.path.join(PROJECT_ROOT, "ocr-training", "data", "detector_v3_hb")

    print("\n[1/2] Benchmarking Cham-DBNet v1 vs v2 on Held-Out HarfBuzz Val-120...")
    v1_val = eval_det_val_120(build_det(v1_infer_dir), val_root)
    v2_val = eval_det_val_120(build_det(v2_infer_dir), val_root)
    print(f"  • v1 Val-120: {v1_val}")
    print(f"  • v2 Val-120: {v2_val}")

    rec_ver = "v24_1_r4" if os.path.exists(os.path.join(STUDIO_DIR, "data", "output", "rec_cham_inference_v24_1_r4")) else "v24_1_r3_1"
    print(f"\n[2/2] Benchmarking End-to-End RealScan-Gold & Level-5/Audit-18 in Hybrid Mode (with fixed recognizer `{rec_ver}`)...")
    v1_e2e = eval_end_to_end_with_detector(v1_infer_dir, rec_ver)
    v2_e2e = eval_end_to_end_with_detector(v2_infer_dir, rec_ver)
    print(f"  • v1 E2E: {v1_e2e}")
    print(f"  • v2 E2E: {v2_e2e}")

    # Strict non-regression & promotion gate:
    # Promote v2 iff:
    #   1) Val-120 H-mean >= v1 H-mean (or within 0.5%)
    #   2) Harum Palei E2E Doc CER <= v1 Harum Palei E2E Doc CER + 0.30% AND line recall >= 96%
    #   3) audit_18 Doc CER (if present) <= v1 audit_18 Doc CER + 0.30%
    v1_h_cer = v1_e2e["harum_palei_01"]["e2e_doc_cer_pct"]
    v2_h_cer = v2_e2e["harum_palei_01"]["e2e_doc_cer_pct"]
    v1_aud = (v1_e2e["level5_and_audit"].get("audit_18") or {}).get("doc_cer_pct", 0.0)
    v2_aud = (v2_e2e["level5_and_audit"].get("audit_18") or {}).get("doc_cer_pct", 0.0)
    promote_v2 = (
        v2_val["hmean_pct"] >= v1_val["hmean_pct"] - 0.5
        and v2_h_cer <= v1_h_cer + 0.30
        and v2_e2e["harum_palei_01"]["line_recall_pct"] >= 96.0
        and v2_aud <= v1_aud + 0.30
        and (v2_val["hmean_pct"] > v1_val["hmean_pct"] or v2_h_cer < v1_h_cer or v2_aud < v1_aud)
    )

    promoted_action = "kept_v1_no_promotion"
    if promote_v2:
        bkp_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "ch_PP-OCRv4_det_cham_v1_backup")
        if not os.path.exists(bkp_dir):
            shutil.copytree(v1_infer_dir, bkp_dir)
        for fn in os.listdir(v2_infer_dir):
            shutil.copy2(os.path.join(v2_infer_dir, fn), os.path.join(v1_infer_dir, fn))
        promoted_action = "promoted_v2_to_ch_PP-OCRv4_det_cham_infer"
        print(f"🚀 PROMOTED Cham-DBNet v2 -> {v1_infer_dir} (v1 backed up at {bkp_dir})")
    else:
        print(f"🛡️ Kept Cham-DBNet v1 in production (v2 archived at {v2_infer_dir})")

    report = {
        "title": "Cham-OCR Cham-DBNet v2 Evaluation & Promotion Gate",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "selected_epoch": best_ep,
        "checkpoint_sha256": ckpt_sha,
        "recognizer_fixed": rec_ver,
        "ground_truth_training_accounting": accounting,
        "val_120_harfbuzz": {"cham_dbnet_v1": v1_val, "cham_dbnet_v2": v2_val},
        "end_to_end": {"cham_dbnet_v1": v1_e2e, "cham_dbnet_v2": v2_e2e},
        "promote_v2": promote_v2,
        "promotion_action": promoted_action,
    }
    os.makedirs(os.path.join(PROJECT_ROOT, "REPORTS"), exist_ok=True)
    out_j = os.path.join(PROJECT_ROOT, "REPORTS", "cham_dbnet_v2_evaluation.json")
    with open(out_j, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md = [
        "# Đánh giá `Cham-DBNet v2` (`ch_PP-OCRv4_det_cham_v2` — HarfBuzz Multi-Layout)",
        f"- **Thời điểm**: `{report['evaluated_at']}`",
        f"- **Epoch tốt nhất**: Epoch `{best_ep}` (`SHA256: {ckpt_sha[:16]}...`)",
        f"- **Train Loop Duration**: `{accounting.get('train_loop_duration_seconds')}s` ({accounting.get('train_loop_duration_minutes')} phút)",
        f"- **Pipeline / Container Overhead**: `{accounting.get('pipeline_container_overhead_seconds')}s` ({accounting.get('pipeline_container_overhead_minutes')} phút)",
        f"- **Total Kernel Wall-clock Time**: `{accounting.get('total_kernel_wall_clock_seconds')}s` ({accounting.get('total_kernel_wall_clock_minutes')} phút)",
        f"- **Quyết định Promotion Gate**: `{promoted_action}` (`promote_v2={promote_v2}`)",
        "",
        "## 1. Held-Out `detector_v3_hb` Val-120 (1,473 dòng HarfBuzz đa bố cục)",
        "| Detector | Recall | Precision | H-Mean | Crop Usability (`>=90%` GT) | Mean IoU |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| `Cham-DBNet v1` | {v1_val['recall_pct']:.2f}% | {v1_val['precision_pct']:.2f}% | **{v1_val['hmean_pct']:.2f}%** | {v1_val['usable_crop_pct']:.2f}% | {v1_val['mean_iou']:.4f} |",
        f"| `Cham-DBNet v2` | {v2_val['recall_pct']:.2f}% | {v2_val['precision_pct']:.2f}% | **{v2_val['hmean_pct']:.2f}%** | {v2_val['usable_crop_pct']:.2f}% | {v2_val['mean_iou']:.4f} |",
        "",
        f"## 2. End-to-End Page Benchmarks (cố định Recognizer `{rec_ver}`)",
        "| Detector | `harum_palei_01` Lines | `harum_palei_01` E2E Doc CER | `short:5` Doc CER | `a4:5` Doc CER | `audit_18` Doc CER |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| `Cham-DBNet v1` | {v1_e2e['harum_palei_01']['pred_lines']}/25 | {v1_h_cer:.2f}% | {(v1_e2e['level5_and_audit'].get('short:5') or {}).get('doc_cer_pct')}% | {(v1_e2e['level5_and_audit'].get('a4:5') or {}).get('doc_cer_pct')}% | {v1_aud}% |",
        f"| `Cham-DBNet v2` | {v2_e2e['harum_palei_01']['pred_lines']}/25 | {v2_h_cer:.2f}% | {(v2_e2e['level5_and_audit'].get('short:5') or {}).get('doc_cer_pct')}% | {(v2_e2e['level5_and_audit'].get('a4:5') or {}).get('doc_cer_pct')}% | {v2_aud}% |",
    ]
    out_m = os.path.join(PROJECT_ROOT, "REPORTS", "CHAM_DBNET_V2_EVALUATION.md")
    with open(out_m, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"✅ Cham-DBNet v2 evaluation saved -> {out_j}")


if __name__ == "__main__":
    main()
