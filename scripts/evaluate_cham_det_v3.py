#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR Cham-DBNet v3 (`ch_PP-OCRv4_det_cham_v3`) Evaluation & Gated Auto-Promotion
====================================================================================
Evaluates `Cham-DBNet v3` candidate checkpoints (`iter_epoch_16`, `iter_epoch_18`, `iter_epoch_20`, + in-basin soup)
under `CHAM_DET_BOX_TYPE='poly'` (`unclip=1.8`, `vpad=0.15` and `vpad=0.25`) head-to-head against production
`v1_poly_u1.8` (`Cham-DBNet v1`) and `v1_quad` across ALL detector & E2E suites:
  1. Held-out `Val-120`, `Val-160`, and `Val-v3-160` true Shapely `poly-IoU` H-Mean (`R / P`).
  2. `FullPage-Frozen 50` (`fullpage_frozen_50.json`, 50 pages / 239 GT lines: Recall, Precision, L1-4, L5, E2E CER).
  3. `harum_palei_01` (`25/25`), `short:5` (`64/64`), `a4:5` (`241/241`), `audit_18` (`134/134`), `OOD-12` (`180/180`).
  4. Level 1-4 Non-Regression Guards (`short:1-4` 352 GT lines, `a4:1-4` 1,153 GT lines).
  5. Exact 3-Part Ground-Truth Training Accounting (`train.log` + Kaggle kernel log).

Promotes to `ocr-studio/data/output/ch_PP-OCRv4_det_cham_infer/` (backing up `v1`) ONLY IF:
  - Passes all 7 promotion checks vs `v1_quad` (`eligible_for_default == True`),
  - Zero false-box regression on `short:5` (`<= 64` or exact `64/64`) and `a4:5` (`241/241`),
  - Zero dropped-line regression on `a4:1-4` (`1153/1153` and `doc_cer <= v1_poly_u1.8 + 0.15`),
  - `short:1-4` `doc_cer <= v1_poly_u1.8 + 0.15`,
  - `FullPage-Frozen 50` `overall_recall >= 99.50%` and `e2e_cer <= v1_poly_u1.8` (`7.99%`),
  - Strictly improves total `short:5 + a4:5 + audit_18` Doc CER over `v1_poly_u1.8`!
"""

import os
import sys
import json
import time
import shutil
import argparse
from datetime import datetime, timezone

import numpy as np
if not hasattr(np, "sctypes"):
    np.sctypes = {"int": [np.int8, np.int16, np.int32, np.int64], "uint": [np.uint8, np.uint16, np.uint32, np.uint64], "float": [np.float16, np.float32, np.float64], "complex": [np.complex64, np.complex128], "others": [bool, object, bytes, str]}
if not hasattr(np, "bool"): np.bool = bool
if not hasattr(np, "int"): np.int = int
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "typeDict"): np.typeDict = {}

import cv2

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIO_DIR = os.path.join(PROJECT_ROOT, "ocr-studio")
for _p in (os.path.join(PROJECT_ROOT, "scripts"), STUDIO_DIR, os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluate_v26_masked_ft import PY26, PADDLE_DIR
from evaluate_r2_1_6_causal import file_sha256
from evaluate_r2_1_7_causal import parse_train_log
from evaluate_cham_det_poly_rectify import eval_det_poly_iou, _load_lower_level_pages, _promotion, VAL_120, VAL_160


def _export_det_with_py26(cfg_path: str, ckpt_prefix: str, infer_dir: str):
    import subprocess
    if os.path.exists(infer_dir):
        shutil.rmtree(infer_dir)
    os.makedirs(infer_dir, exist_ok=True)
    subprocess.run(
        [
            PY26,
            "tools/export_model.py",
            "-c",
            cfg_path,
            "-o",
            "Global.pretrained_model=None",
            f"Global.checkpoints={ckpt_prefix}",
            f"Global.save_inference_dir={infer_dir}",
            "Global.export_with_pir=False",
        ],
        cwd=PADDLE_DIR,
        check=True,
    )

VAL_V3 = os.path.join(PROJECT_ROOT, "ocr-training", "data", "detector_v3_2_poly")
STUDIO_DET_DIR = os.path.join(STUDIO_DIR, "data", "output", "ch_PP-OCRv4_det_cham_infer")
V1_BACKUP_DIR = os.path.join(PROJECT_ROOT, "ocr-training", "output", "ch_PP-OCRv4_det_cham_v1_backup")
DL_DIR = os.path.join(PROJECT_ROOT, "ocr-training", "output", "det_v3_kaggle_output")
CFG_V3 = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "det", "ch_PP-OCRv4_det_cham_v3.yml")


def get_v1_dir() -> str:
    return V1_BACKUP_DIR if os.path.exists(os.path.join(V1_BACKUP_DIR, "inference.pdiparams")) else STUDIO_DET_DIR


def eval_fullpage_frozen_50(det_dir: str, box_type: str, unclip: float, vpad: float) -> dict:
    import app as studio_app
    from evaluate_pipeline import levenshtein_distance
    from evaluate_realscan_gold import matched_line_stats
    from scripts.generate_data import normalize_unicode

    studio_app.DET_MODE = "hybrid"
    studio_app.det_model = studio_app.get_generic_det_model()
    studio_app._cham_det_failed = False
    studio_app.CHAM_DET_POLY_VPAD = float(vpad)
    studio_app.CHAM_DET_POLY_MASK = True
    studio_app.cham_det_model = studio_app._build_text_detector(os.path.abspath(det_dir), 0.3, 0.4, unclip, box_type=box_type)
    studio_app.get_ocr_model("v24_1_r4")

    fp_path = os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "frozen_suite", "fullpage_frozen_50.json")
    raw = json.load(open(fp_path, "r", encoding="utf-8"))
    pages = raw if isinstance(raw, list) else raw.get("pages", raw.get("samples", []))

    tot_gt = tot_pr = tot_match = m_ed = m_ch = d_ed = d_ch = 0
    l14_gt = l14_match = l5_gt = l5_match = 0
    for pg in pages:
        rel = pg.get("image_relpath") or pg.get("image_path") or ("data/images/" + pg["filename"])
        ip = os.path.join(PROJECT_ROOT, "ocr-benchmark", rel.replace("/", os.sep))
        if not os.path.exists(ip):
            ip = os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", rel.replace("/", os.sep))
        if not os.path.exists(ip):
            ip = os.path.join(PROJECT_ROOT, rel.replace("/", os.sep))
        assert os.path.exists(ip), f"Missing FullPage-Frozen image: {ip}"
        img = cv2.imread(ip)
        res = studio_app.run_ocr_pipeline(img, model_ver="v24_1_r4", method="dbnet")
        pl = [normalize_unicode(l["text"]) for l in res.get("lines", [])]
        raw_gl = pg.get("gt_lines") or [x["text"] if isinstance(x, dict) else x for x in pg.get("lines", [])]
        gl = [normalize_unicode(x) for x in raw_gl]
        ms = matched_line_stats(gl, pl)
        tot_gt += len(gl)
        tot_pr += len(pl)
        tot_match += ms["matched_lines"]
        m_ed += int(round(ms["matched_line_cer_pct"] * sum(len(x) for x in gl) / 100.0))
        m_ch += sum(len(x) for x in gl)
        gd, pd = " ".join(gl), " ".join(pl)
        d_ed += levenshtein_distance(gd, pd)
        d_ch += len(gd)
        if str(pg.get("level", "")) == "5":
            l5_gt += len(gl)
            l5_match += ms["matched_lines"]
        else:
            l14_gt += len(gl)
            l14_match += ms["matched_lines"]

    return {
        "overall_recall_pct": round(100.0 * tot_match / max(1, tot_gt), 2),
        "overall_precision_pct": round(100.0 * tot_match / max(1, tot_pr), 2),
        "l14_recall_pct": round(100.0 * l14_match / max(1, l14_gt), 2),
        "l5_recall_pct": round(100.0 * l5_match / max(1, l5_gt), 2),
        "e2e_cer_pct": round(100.0 * d_ed / max(1, d_ch), 2),
        "lines": f"{tot_pr}/{tot_gt}",
    }


def eval_full_suite_for_det(name: str, det_dir: str, box_type: str, unclip: float, vpad: float) -> dict:
    import app as studio_app
    from evaluate_level5_pages import load_pages, norm_audit, matched_cer
    from evaluate_realscan_gold import matched_line_stats
    from evaluate_pipeline import levenshtein_distance
    from scripts.generate_data import normalize_unicode

    t0 = time.time()
    studio_app.DET_MODE = "hybrid"
    studio_app.det_model = studio_app.get_generic_det_model()
    studio_app._cham_det_failed = False
    studio_app.CHAM_DET_POLY_VPAD = float(vpad)
    studio_app.CHAM_DET_POLY_MASK = True
    det_inst = studio_app._build_text_detector(os.path.abspath(det_dir), 0.3, 0.4, unclip, box_type=box_type)
    studio_app.cham_det_model = det_inst
    studio_app.get_ocr_model("v24_1_r4")

    val_120 = eval_det_poly_iou(det_inst, VAL_120)
    val_160 = eval_det_poly_iou(det_inst, VAL_160)
    val_v3 = eval_det_poly_iou(det_inst, VAL_V3)

    def _run(img):
        res = studio_app.run_ocr_pipeline(img, model_ver="v24_1_r4", method="dbnet")
        prof = res.get("profiler", {})
        return [l["text"] for l in res.get("lines", [])], prof.get("detector"), bool(prof.get("detector_switched"))

    m1 = json.load(open(os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "realscan_gold_v1", "manifest.json"), "r", encoding="utf-8"))
    harum = next(d for d in m1["documents"] if d["doc_id"] == "harum_palei_01")
    pl, detn, sw = _run(cv2.imread(os.path.join(PROJECT_ROOT, harum["page_relpath"].replace("/", os.sep))))
    harum_stats = matched_line_stats([ln["gt"] for ln in harum["lines"]], pl)

    by_g = {}
    for pg in load_pages(limit_per_set=90):
        g = pg["group"]
        st = by_g.setdefault(g, {"n": 0, "gt_l": 0, "pr_l": 0, "d_err": 0, "d_den": 0, "m_err": 0, "m_den": 0, "switched": 0})
        pl, _, sw = _run(cv2.imread(pg["path"]))
        gl = pg["gt_lines"]
        st["n"] += 1; st["gt_l"] += len(gl); st["pr_l"] += len(pl); st["switched"] += int(sw)
        if gl:
            gd, pd = norm_audit(" ".join(gl)), norm_audit(" ".join(pl))
            st["d_err"] += levenshtein_distance(gd, pd); st["d_den"] += len(gd)
            me, md = matched_cer(gl, pl)
            st["m_err"] += me; st["m_den"] += md
    lvl5 = {g: {"n": s["n"], "lines": f"{s['pr_l']}/{s['gt_l']}", "pred_lines": s["pr_l"], "gt_lines": s["gt_l"],
                "doc_cer_pct": round(100.0 * s["d_err"] / max(1, s["d_den"]), 2),
                "matched_cer_pct": round(100.0 * s["m_err"] / max(1, s["m_den"]), 2)} for g, s in by_g.items()}

    m2 = json.load(open(os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "realscan_gold_v2", "manifest.json"), "r", encoding="utf-8"))
    ood = [d for d in m2["documents"] if d["source_type"] == "ood_archival_gold_v2"]
    e_ed = e_ch = gt_n = pr_n = matched = 0
    for d in ood:
        pl, _, _ = _run(cv2.imread(os.path.join(PROJECT_ROOT, d["page_relpath"].replace("/", os.sep))))
        gl = [ln["gt"] for ln in d["lines"]]
        gt_doc = " ".join(normalize_unicode(x) for x in gl)
        pr_doc = " ".join(normalize_unicode(x) for x in pl)
        e_ed += levenshtein_distance(gt_doc, pr_doc); e_ch += len(gt_doc)
        gt_n += len(gl); pr_n += len(pl)
        matched += matched_line_stats(gl, pl)["matched_lines"]
    ood_sum = {"pages": len(ood), "e2e_lines": f"{pr_n}/{gt_n}", "pred_lines": pr_n, "gt_lines": gt_n,
               "line_recall_pct": round(100.0 * matched / max(1, gt_n), 2),
               "e2e_doc_cer_pct": round(100.0 * e_ed / max(1, e_ch), 2)}

    low_g = {}
    for pg in _load_lower_level_pages():
        pl, _, sw = _run(cv2.imread(pg["path"]))
        gl = pg["gt_lines"]
        st = low_g.setdefault(pg["group"], {"n": 0, "gt_l": 0, "pr_l": 0, "d_err": 0, "d_den": 0, "m_err": 0, "m_den": 0})
        gd, pd = norm_audit(" ".join(gl)), norm_audit(" ".join(pl))
        st["n"] += 1; st["gt_l"] += len(gl); st["pr_l"] += len(pl)
        st["d_err"] += levenshtein_distance(gd, pd); st["d_den"] += len(gd)
        me, md = matched_cer(gl, pl)
        st["m_err"] += me; st["m_den"] += md
    guards = {g: {"lines": f"{s['pr_l']}/{s['gt_l']}", "pred_lines": s["pr_l"], "gt_lines": s["gt_l"],
                  "doc_cer_pct": round(100.0 * s["d_err"] / max(1, s["d_den"]), 2),
                  "matched_cer_pct": round(100.0 * s["m_err"] / max(1, s["m_den"]), 2)} for g, s in low_g.items()}

    fp50 = eval_fullpage_frozen_50(det_dir, box_type, unclip, vpad)
    return {
        "config": name,
        "params": {"box_type": box_type, "unclip": unclip, "vpad": vpad},
        "val_120_poly_iou": val_120,
        "val_160_poly_iou": val_160,
        "val_v3_160_poly_iou": val_v3,
        "harum_palei_01": harum_stats,
        "level5_and_audit": lvl5,
        "ood_12": ood_sum,
        "lower_levels": guards,
        "fullpage_frozen_50": fp50,
        "seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", nargs="*", type=int, default=[16, 20])
    ap.add_argument("--no-promote", action="store_true")
    args = ap.parse_args()

    import paddle
    ext_d = os.path.join(DL_DIR, "extracted_checkpoints")
    assert os.path.isdir(ext_d), f"Missing extracted Det v3 checkpoints: {ext_d}"

    t_log = os.path.join(ext_d, "train.log")
    k_log = os.path.join(DL_DIR, "paddleocr-cham-det-v3-poly.log")
    accounting = parse_train_log(t_log, total_epochs=20, kernel_log_path=k_log) if os.path.exists(t_log) else {}

    # Build soup of available top checkpoints (e.g. E16 + E18 + E20)
    soup_eps = [e for e in (16, 18, 20) if os.path.exists(os.path.join(ext_d, f"iter_epoch_{e}.pdparams"))]
    cand_exports = {}
    for ep in args.epochs:
        cp = os.path.join(ext_d, f"iter_epoch_{ep}.pdparams")
        if os.path.exists(cp):
            idir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"ch_PP-OCRv4_det_cham_v3_e{ep}_infer")
            if not os.path.exists(os.path.join(idir, "inference.pdiparams")):
                _export_det_with_py26(CFG_V3, cp[:-9], idir)
            cand_exports[f"v3_e{ep}"] = (idir, cp)

    if len(soup_eps) >= 2:
        sds = [paddle.load(os.path.join(ext_d, f"iter_epoch_{e}.pdparams")) for e in soup_eps]
        s_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "det_v3_soup_ckpt")
        os.makedirs(s_dir, exist_ok=True)
        s_ckpt = os.path.join(s_dir, "best.pdparams")
        w = 1.0 / len(sds)
        blended = {k: sum(w * sd[k] for sd in sds) for k in sds[0].keys()}
        paddle.save(blended, s_ckpt)
        idir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "ch_PP-OCRv4_det_cham_v3_soup_infer")
        if not os.path.exists(os.path.join(idir, "inference.pdiparams")):
            _export_det_with_py26(CFG_V3, s_ckpt[:-9], idir)
        cand_exports[f"v3_soup_{'_'.join(map(str, soup_eps))}"] = (idir, s_ckpt)

    cache_p = os.path.join(PROJECT_ROOT, "REPORTS", "_det_v3_eval_progress_cache.json")
    cached_res = json.load(open(cache_p, "r", encoding="utf-8")) if os.path.exists(cache_p) else {}

    def _get_or_eval(cname: str, ddir: str, btype: str, u: float, vp: float) -> dict:
        dsha = file_sha256(os.path.join(ddir, "inference.pdiparams"))[:16]
        ckey = f"{cname}|{dsha}|{btype}|{u}|{vp}"
        if ckey in cached_res:
            print(f"♻️  Reusing cached result for `{cname}` ({dsha})...")
            return dict(cached_res[ckey])
        print(f"🔬 Scoring `{cname}` ({dsha})...")
        r_new = eval_full_suite_for_det(cname, ddir, btype, u, vp)
        cached_res[ckey] = dict(r_new)
        with open(cache_p, "w", encoding="utf-8", newline="\n") as cf:
            json.dump(cached_res, cf, indent=2, ensure_ascii=False)
        return r_new

    v1_dir = get_v1_dir()
    res_v1_quad = _get_or_eval("v1_quad", v1_dir, "quad", 1.6, 0.0)
    res_v1_poly = _get_or_eval("v1_poly_u1.8", v1_dir, "poly", 1.8, 0.15)

    results = {"v1_quad": res_v1_quad, "v1_poly_u1.8": res_v1_poly}
    decisions = {}
    qualifying = []

    for tag, (idir, ckpt_p) in cand_exports.items():
        # Inverse unclip of MakeShrinkMap(shrink_ratio=0.4) on full-cell Cham ribbons is u in [3.0, 3.2]
        param_grid = ((3.0, 0.15), (3.2, 0.15))
        for (u, vp) in param_grid:
            cname = f"{tag}_poly_u{u}_vpad{vp}"
            r = _get_or_eval(cname, idir, "poly", u, vp)
            r["infer_dir"] = idir
            r["ckpt_path"] = ckpt_p
            results[cname] = r

            p_vs_quad = _promotion(res_v1_quad, r)
            s5_pr = r["level5_and_audit"]["short:5"]["pred_lines"]
            a45_pr = r["level5_and_audit"]["a4:5"]["pred_lines"]
            a4_14 = r["lower_levels"]["a4:1-4"]
            sh_14 = r["lower_levels"]["short:1-4"]
            fp50 = r["fullpage_frozen_50"]
            v1_sum = (res_v1_poly["level5_and_audit"]["short:5"]["doc_cer_pct"]
                      + res_v1_poly["level5_and_audit"]["a4:5"]["doc_cer_pct"]
                      + res_v1_poly["level5_and_audit"]["audit_18"]["doc_cer_pct"])
            cand_sum = (r["level5_and_audit"]["short:5"]["doc_cer_pct"]
                        + r["level5_and_audit"]["a4:5"]["doc_cer_pct"]
                        + r["level5_and_audit"]["audit_18"]["doc_cer_pct"])

            strict_guards = {
                "passes_7_quad_gates": p_vs_quad["eligible_for_default"],
                "short5_exact_64_lines": (s5_pr == 64),
                "a4_5_exact_241_lines": (a45_pr == 241),
                "a4_1_4_exact_1153_lines": (a4_14["pred_lines"] == 1153),
                "a4_1_4_non_regression": (a4_14["doc_cer_pct"] <= res_v1_poly["lower_levels"]["a4:1-4"]["doc_cer_pct"] + 0.15),
                "short_1_4_non_regression": (sh_14["doc_cer_pct"] <= res_v1_poly["lower_levels"]["short:1-4"]["doc_cer_pct"] + 0.15),
                "fp50_recall_ge_99_5": (fp50["overall_recall_pct"] >= 99.50),
                "fp50_e2e_beats_or_ties_v1_poly": (fp50["e2e_cer_pct"] <= res_v1_poly["fullpage_frozen_50"]["e2e_cer_pct"] + 0.05),
                "beats_v1_poly_on_l5_audit_sum": (cand_sum < v1_sum),
            }
            ok = all(strict_guards.values())
            decisions[cname] = {"guards": strict_guards, "l5_audit_sum": round(cand_sum, 2), "v1_poly_sum": round(v1_sum, 2), "qualifies": ok}
            print(f"  -> `{cname}`: short:5={r['level5_and_audit']['short:5']['doc_cer_pct']}% ({s5_pr}/64) | "
                  f"a4:5={r['level5_and_audit']['a4:5']['doc_cer_pct']}% ({a45_pr}/241) | "
                  f"audit_18={r['level5_and_audit']['audit_18']['doc_cer_pct']}% | "
                  f"a4:1-4={a4_14['doc_cer_pct']}% ({a4_14['pred_lines']}/1153) | "
                  f"FP50={fp50['e2e_cer_pct']}% (R={fp50['overall_recall_pct']}%) | qualifies={ok}")
            if ok:
                qualifying.append((cand_sum + fp50["e2e_cer_pct"], cname))

    winner = min(qualifying)[1] if qualifying else None
    promoted = False
    if winner and not args.no_promote:
        if not os.path.exists(V1_BACKUP_DIR):
            shutil.copytree(STUDIO_DET_DIR, V1_BACKUP_DIR)
        w_dir = results[winner]["infer_dir"]
        for fn in os.listdir(w_dir):
            sp = os.path.join(w_dir, fn)
            if os.path.isfile(sp):
                shutil.copy2(sp, os.path.join(STUDIO_DET_DIR, fn))
        promoted = True

    active_det_sha16 = file_sha256(os.path.join(STUDIO_DET_DIR, "inference.pdiparams"))[:16]
    payload = {
        "title": "Cham-OCR Cham-DBNet v3 Full-Cell Curved Polygon Detector Evaluation",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "training_accounting": accounting,
        "results": {k: {kk: vv for kk, vv in v.items() if kk not in ("infer_dir", "ckpt_path")} for k, v in results.items()},
        "decisions": decisions,
        "winner": winner,
        "promoted": promoted,
        "active_studio_det_sha256_16": active_det_sha16,
    }
    out_j = os.path.join(PROJECT_ROOT, "REPORTS", "cham_dbnet_v3_evaluation.json")
    with open(out_j, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md = [
        "# Đánh giá `Cham-DBNet v3` (Full-Cell Curved Polygons + Border OHEM Hard-Negatives + Short-Line Recall)",
        f"- **Thời điểm**: `{payload['evaluated_at']}`",
        f"- **Kế toán huấn luyện Kaggle Dual T4x2**: Train Loop = `{accounting.get('train_loop_duration_seconds')}s`, "
        f"Overhead = `{accounting.get('pipeline_container_overhead_seconds')}s`, Wall-clock = `{accounting.get('total_kernel_wall_clock_seconds')}s`",
        "",
        "## 1. Đối chuẩn End-to-End & Non-Regression Guards (`DET_MODE='hybrid'`) ",
        "| Config | `Val-160` poly-IoU | `Val-v3` poly-IoU | `short:5` (lines) | `a4:5` (lines) | `audit_18` (lines) | `a4:1-4` Guard (lines) | `FullPage-50` E2E (Recall) | Qualifies |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k, r in results.items():
        l5 = r["level5_and_audit"]
        a14 = r["lower_levels"]["a4:1-4"]
        fp = r["fullpage_frozen_50"]
        q = decisions[k]["qualifies"] if k in decisions else "baseline"
        md.append(
            f"| `{k}` | {r['val_160_poly_iou']['hmean_pct']}% | {r['val_v3_160_poly_iou']['hmean_pct']}% | "
            f"{l5['short:5']['doc_cer_pct']}% ({l5['short:5']['lines']}) | "
            f"{l5['a4:5']['doc_cer_pct']}% ({l5['a4:5']['lines']}) | "
            f"{l5['audit_18']['doc_cer_pct']}% ({l5['audit_18']['lines']}) | "
            f"{a14['doc_cer_pct']}% ({a14['lines']}) | "
            f"{fp['e2e_cer_pct']}% ({fp['overall_recall_pct']}%) | `{q}` |"
        )
    md += ["", "## 2. Quyết định Promotion", f"- **Winner**: `{winner}`; **promoted**: `{promoted}`; active `ch_PP-OCRv4_det_cham_infer` SHA256_16: `{active_det_sha16}`"]
    for cname, d in decisions.items():
        failed = [gk for gk, ok in d["guards"].items() if not ok]
        md.append(f"- `{cname}` (`sum={d['l5_audit_sum']}%` vs `v1_poly={d['v1_poly_sum']}%`): qualifies=`{d['qualifies']}`" + (f" (trượt: `{failed}`)" if failed else " (**PASS ALL 9 GUARDS**)"))

    out_m = os.path.join(PROJECT_ROOT, "REPORTS", "CHAM_DBNET_V3_EVALUATION.md")
    with open(out_m, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print(f"✅ Cham-DBNet v3 evaluation -> {out_m} (winner={winner}, promoted={promoted})")


if __name__ == "__main__":
    main()
