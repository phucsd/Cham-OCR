#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR Cham-DBNet polygon detection + curved-line ("ribbon") rectification benchmark
=====================================================================================
Compares, with the recognizer fixed to `v24_1_r4` and the production `DET_MODE='hybrid'`
(Legacy-First: generic PP-OCRv4 DBNet primary, Cham-DBNet only when the gate fires):
  * v1_quad   - production baseline (Cham-DBNet v1, box_type='quad', perspective quad crop)
  * v1_poly   - Cham-DBNet v1, box_type='poly' + ribbon rectification
  * v21_quad  - Cham-DBNet v2.1, box_type='quad'
  * v21_poly  - Cham-DBNet v2.1, box_type='poly' + ribbon rectification
  * a small sweep of unclip ratio / DB thresholds / rectification padding & masking for the poly path
on: RealScan-Gold v1 harum_palei_01 (25 lines), Level-5 `short:5` + `a4:5`, the 18-image audit suite,
RealScan-Gold v2 OOD-12 archival pages (180 lines, full-page E2E), plus detector-only Val-120 / Val-160
H-Mean (true polygon IoU via shapely; the legacy convex-quad protocol is also reported for quad configs).

Usage:
  python scripts/evaluate_cham_det_poly_rectify.py               # run everything (parallel workers) + report
  python scripts/evaluate_cham_det_poly_rectify.py --aggregate   # rebuild the report from saved parts
  python scripts/evaluate_cham_det_poly_rectify.py --worker e2e --config v21_poly   (internal)
Outputs: REPORTS/CHAM_DBNET_POLY_RECTIFY_EVALUATION.md + REPORTS/cham_dbnet_poly_rectify_evaluation.json
Only numbers from actual runs are written; nothing is deployed by this script.
"""

import os
import sys
import json
import time
import argparse
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

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIO_DIR = os.path.join(PROJECT_ROOT, "ocr-studio")
for _p in (os.path.join(PROJECT_ROOT, "scripts"), STUDIO_DIR, os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PY313 = r"C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe"
if not os.path.exists(PY313):
    PY313 = sys.executable

V1_DIR = os.path.join(STUDIO_DIR, "data", "output", "ch_PP-OCRv4_det_cham_infer")
V1_BACKUP = os.path.join(PROJECT_ROOT, "ocr-training", "output", "ch_PP-OCRv4_det_cham_v1_backup")
V21_DIR = os.path.join(PROJECT_ROOT, "ocr-training", "output", "ch_PP-OCRv4_det_cham_v2_1_dual_infer")
V22_DIR = os.path.join(PROJECT_ROOT, "ocr-training", "output", "ch_PP-OCRv4_det_cham_v2_2_dual_infer")
VAL_120 = os.path.join(PROJECT_ROOT, "ocr-training", "data", "detector_v3_hb")
VAL_160 = os.path.join(PROJECT_ROOT, "ocr-training", "data", "detector_v3_1_hybrid")
PARTS_DIR = os.path.join(PROJECT_ROOT, "ocr-benchmark", "results", "cham_det_poly_rectify")
REC_VER = "v24_1_r4"
OUT_JSON = os.path.join(PROJECT_ROOT, "REPORTS", "cham_dbnet_poly_rectify_evaluation.json")
OUT_MD = os.path.join(PROJECT_ROOT, "REPORTS", "CHAM_DBNET_POLY_RECTIFY_EVALUATION.md")


def _v1_dir():
    # If v1 was ever replaced in production, the backup is the true v1 reference
    return V1_BACKUP if os.path.exists(V1_BACKUP) else V1_DIR


def _cfg(det, box, unclip=1.6, box_thresh=0.4, db_thresh=0.3, vpad=0.15, mask=True, note=""):
    return {"det": det, "box_type": box, "unclip": unclip, "box_thresh": box_thresh, "db_thresh": db_thresh,
            "vpad": vpad, "mask": mask, "note": note}


CONFIGS = {
    # Core comparison
    "v1_quad": _cfg("v1", "quad", note="production baseline"),
    "v1_poly": _cfg("v1", "poly"),
    "v21_quad": _cfg("v21", "quad"),
    "v21_poly": _cfg("v21", "poly"),
    # Poly-path sweep (Cham-DBNet v2.1)
    "v21_poly_u1.4": _cfg("v21", "poly", unclip=1.4),
    "v21_poly_u1.8": _cfg("v21", "poly", unclip=1.8),
    "v21_poly_u2.0": _cfg("v21", "poly", unclip=2.0),
    "v21_poly_b0.5": _cfg("v21", "poly", box_thresh=0.5),
    "v21_poly_t0.2": _cfg("v21", "poly", db_thresh=0.2),
    "v21_poly_vpad0": _cfg("v21", "poly", vpad=0.0),
    "v21_poly_vpad0.4": _cfg("v21", "poly", vpad=0.40),
    "v21_poly_nomask": _cfg("v21", "poly", mask=False),
    "v21_poly_u2.2": _cfg("v21", "poly", unclip=2.2),
    "v21_poly_u2.5": _cfg("v21", "poly", unclip=2.5),
    "v21_poly_vpad0.6": _cfg("v21", "poly", vpad=0.60),
    "v21_poly_u2.0_vpad0.4": _cfg("v21", "poly", unclip=2.0, vpad=0.40),
    "v21_poly_u2.5_vpad0.4": _cfg("v21", "poly", unclip=2.5, vpad=0.40),
    # Poly-path sweep (Cham-DBNet v1)
    "v1_poly_u1.4": _cfg("v1", "poly", unclip=1.4),
    "v1_poly_u1.8": _cfg("v1", "poly", unclip=1.8),
    "v1_poly_u2.0": _cfg("v1", "poly", unclip=2.0),
    "v1_poly_u2.2": _cfg("v1", "poly", unclip=2.2),
    "v1_poly_u1.8_vpad0.3": _cfg("v1", "poly", unclip=1.8, vpad=0.30),
    "v1_poly_b0.5": _cfg("v1", "poly", box_thresh=0.5),
    "v1_poly_t0.2": _cfg("v1", "poly", db_thresh=0.2),
    "v1_poly_vpad0": _cfg("v1", "poly", vpad=0.0),
    "v1_poly_vpad0.4": _cfg("v1", "poly", vpad=0.40),
    "v1_poly_nomask": _cfg("v1", "poly", mask=False),
}
CORE = ["v1_quad", "v1_poly", "v21_quad", "v21_poly"]
DET_KEYS = [("v1", "quad", 1.6), ("v1", "poly", 1.6), ("v21", "quad", 1.6), ("v21", "poly", 1.6),
            ("v21", "poly", 1.4), ("v21", "poly", 1.8), ("v21", "poly", 2.0), ("v21", "poly", 2.2),
            ("v1", "poly", 2.0)]


def _det_dir(tag):
    return {"v1": _v1_dir(), "v21": V21_DIR, "v22": V22_DIR}[tag]


def _register_extra_configs():
    """Adds configs for an optional fine-tuned Cham-DBNet v2.2 export when it exists on disk."""
    if os.path.exists(os.path.join(V22_DIR, "inference.pdiparams")):
        CONFIGS.setdefault("v22_quad", _cfg("v22", "quad"))
        CONFIGS.setdefault("v22_poly", _cfg("v22", "poly"))
        for k in (("v22", "quad", 1.6), ("v22", "poly", 1.6)):
            if k not in DET_KEYS:
                DET_KEYS.append(k)


# ------------------------------------------------------------------------------------------------
# Detector-only evaluation (true polygon IoU)
# ------------------------------------------------------------------------------------------------
def _shp(pts):
    from shapely.geometry import Polygon
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        return None
    g = Polygon(p)
    if not g.is_valid:
        g = g.buffer(0)
    return g if (not g.is_empty and g.area > 0) else None


def eval_det_poly_iou(det, val_root, iou_thr=0.5):
    lbl = os.path.join(val_root, "det_val_label.txt")
    rows = [l.rstrip("\n").split("\t", 1) for l in open(lbl, "r", encoding="utf-8") if "\t" in l]
    tot_gt = tot_dt = rec = prec = 0
    ious = []
    for rel, js in rows:
        gts = [_shp(g["points"]) for g in json.loads(js)]
        gts = [g for g in gts if g is not None]
        img = cv2.imread(os.path.join(val_root, rel.replace("/", os.sep)))
        dts, _ = det(img)
        dts = [_shp(d) for d in (dts if dts is not None else [])]
        dts = [d for d in dts if d is not None]
        m = np.zeros((len(gts), len(dts)), np.float64)
        for i, g in enumerate(gts):
            gb = g.bounds
            for j, d in enumerate(dts):
                db = d.bounds
                if db[0] > gb[2] or db[2] < gb[0] or db[1] > gb[3] or db[3] < gb[1]:
                    continue
                inter = g.intersection(d).area
                if inter > 0:
                    m[i, j] = inter / (g.area + d.area - inter)
        if len(gts) and len(dts):
            best_g = m.max(axis=1)
            rec += int((best_g >= iou_thr).sum())
            ious.extend(best_g[best_g >= iou_thr].tolist())
            prec += int((m.max(axis=0) >= iou_thr).sum())
        tot_gt += len(gts)
        tot_dt += len(dts)
    r = rec / max(1, tot_gt)
    p = prec / max(1, tot_dt)
    hm = 2 * r * p / (r + p) if (r + p) > 0 else 0.0
    return {"pages": len(rows), "gt_lines": tot_gt, "dt_lines": tot_dt, "recall_pct": round(100 * r, 2),
            "precision_pct": round(100 * p, 2), "hmean_pct": round(100 * hm, 2),
            "mean_iou": round(float(np.mean(ious)) if ious else 0.0, 4)}


def worker_det(key):
    import app as studio_app
    tag, box, unclip = key
    det = studio_app._build_text_detector(os.path.abspath(_det_dir(tag)), 0.3, 0.4, unclip, box_type=box)
    t0 = time.time()
    out = {"key": list(key), "val_120_poly_iou": eval_det_poly_iou(det, VAL_120),
           "val_160_poly_iou": eval_det_poly_iou(det, VAL_160)}
    if box == "quad" and abs(unclip - 1.6) < 1e-6:
        # Legacy protocol of REPORTS/CHAM_DBNET_V2_1_EVALUATION.md (convex quads / minAreaRect GT)
        from evaluate_cham_det_v2 import eval_det_val_120
        from evaluate_cham_det_v2_1 import eval_det_val_curved
        out["val_120_legacy"] = eval_det_val_120(det, VAL_120)
        out["val_160_legacy"] = eval_det_val_curved(det, VAL_160)
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ------------------------------------------------------------------------------------------------
# End-to-end evaluation through ocr-studio/app.py (hybrid, Legacy-First)
# ------------------------------------------------------------------------------------------------
def worker_e2e(name):
    import app as studio_app
    from evaluate_level5_pages import load_pages, norm_audit, matched_cer
    from evaluate_realscan_gold import matched_line_stats
    from evaluate_pipeline import levenshtein_distance
    from scripts.generate_data import normalize_unicode

    cfg = CONFIGS[name]
    studio_app.DET_MODE = "hybrid"
    studio_app.det_model = studio_app.get_generic_det_model()
    studio_app._cham_det_failed = False
    studio_app.CHAM_DET_POLY_VPAD = float(cfg["vpad"])
    studio_app.CHAM_DET_POLY_MASK = bool(cfg["mask"])
    studio_app.cham_det_model = studio_app._build_text_detector(
        os.path.abspath(_det_dir(cfg["det"])), cfg["db_thresh"], cfg["box_thresh"], cfg["unclip"], box_type=cfg["box_type"])
    studio_app.get_ocr_model(REC_VER)
    t0 = time.time()
    per_page = []

    def _run(img):
        res = studio_app.run_ocr_pipeline(img, model_ver=REC_VER, method="dbnet")
        prof = res.get("profiler", {})
        return [l["text"] for l in res.get("lines", [])], prof.get("detector"), bool(prof.get("detector_switched"))

    # 1. harum_palei_01 full page
    m1 = json.load(open(os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "realscan_gold_v1", "manifest.json"), "r", encoding="utf-8"))
    harum = next(d for d in m1["documents"] if d["doc_id"] == "harum_palei_01")
    pl, detn, sw = _run(cv2.imread(os.path.join(PROJECT_ROOT, harum["page_relpath"].replace("/", os.sep))))
    harum_stats = matched_line_stats([ln["gt"] for ln in harum["lines"]], pl)
    harum_stats["detector"] = detn
    harum_stats["detector_switched"] = sw

    # 2. Level-5 short:5 / a4:5 + audit_18
    by_g = {}
    for pg in load_pages(limit_per_set=90):
        g = pg["group"]
        st = by_g.setdefault(g, {"n": 0, "gt_l": 0, "pr_l": 0, "abs_l": 0, "d_err": 0, "d_den": 0, "m_err": 0, "m_den": 0, "switched": 0})
        pl, detn, sw = _run(cv2.imread(pg["path"]))
        gl = pg["gt_lines"]
        st["n"] += 1
        st["gt_l"] += len(gl)
        st["pr_l"] += len(pl)
        st["abs_l"] += abs(len(pl) - len(gl))
        st["switched"] += int(sw)
        rec = {"group": g, "file": pg["filename"], "lines": f"{len(pl)}/{len(gl)}", "detector": detn, "switched": sw}
        if gl:
            gd, pd = norm_audit(" ".join(gl)), norm_audit(" ".join(pl))
            de = levenshtein_distance(gd, pd)
            st["d_err"] += de
            st["d_den"] += len(gd)
            me, md = matched_cer(gl, pl)
            st["m_err"] += me
            st["m_den"] += md
            rec["doc_cer_pct"] = round(100.0 * de / max(1, len(gd)), 2)
        per_page.append(rec)
    lvl5 = {g: {"n": s["n"], "lines": f"{s['pr_l']}/{s['gt_l']}", "pred_lines": s["pr_l"], "gt_lines": s["gt_l"],
                "abs_line_err": s["abs_l"], "cham_switched_pages": s["switched"],
                "doc_cer_pct": round(100.0 * s["d_err"] / max(1, s["d_den"]), 2),
                "matched_cer_pct": round(100.0 * s["m_err"] / max(1, s["m_den"]), 2)} for g, s in by_g.items()}

    # 3. RealScan-Gold v2 OOD-12 archival pages (full-page E2E)
    m2 = json.load(open(os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "realscan_gold_v2", "manifest.json"), "r", encoding="utf-8"))
    ood = [d for d in m2["documents"] if d["source_type"] == "ood_archival_gold_v2"]
    e_ed = e_ch = gt_n = pr_n = matched = m_ed = m_ch = sw_n = 0
    for d in ood:
        pl, detn, sw = _run(cv2.imread(os.path.join(PROJECT_ROOT, d["page_relpath"].replace("/", os.sep))))
        gl = [ln["gt"] for ln in d["lines"]]
        gt_doc = " ".join(normalize_unicode(x) for x in gl)
        pr_doc = " ".join(normalize_unicode(x) for x in pl)
        ed = levenshtein_distance(gt_doc, pr_doc)
        ms = matched_line_stats(gl, pl)
        e_ed += ed
        e_ch += len(gt_doc)
        gt_n += len(gl)
        pr_n += len(pl)
        matched += ms["matched_lines"]
        sw_n += int(sw)
        per_page.append({"group": "ood_12", "file": d["doc_id"], "lines": f"{len(pl)}/{len(gl)}", "detector": detn,
                         "switched": sw, "doc_cer_pct": round(100.0 * ed / max(1, len(gt_doc)), 2),
                         "line_recall_pct": ms["line_recall_pct"]})
    ood_sum = {"pages": len(ood), "e2e_lines": f"{pr_n}/{gt_n}", "pred_lines": pr_n, "gt_lines": gt_n,
               "matched_lines": matched, "line_recall_pct": round(100.0 * matched / max(1, gt_n), 2),
               "e2e_doc_cer_pct": round(100.0 * e_ed / max(1, e_ch), 2), "cham_switched_pages": sw_n}
    return {"config": name, "params": cfg, "det_dir": os.path.relpath(_det_dir(cfg["det"]), PROJECT_ROOT).replace("\\", "/"),
            "recognizer": REC_VER, "harum_palei_01": harum_stats, "level5_and_audit": lvl5, "ood_12": ood_sum,
            "per_page": per_page, "seconds": round(time.time() - t0, 1)}


def _load_lower_level_pages(limit_per_set=90):
    """Level 1-4 pages of the same seeded 90-page samples used by evaluate_level5_pages.load_pages()."""
    import ast
    import random
    bench = os.path.join(PROJECT_ROOT, "ocr-benchmark", "data")
    pages = []
    for gt_file, tag in (("benchmark_gt.json", "short"), ("benchmark_a4_gt.json", "a4")):
        items = json.loads(open(os.path.join(bench, gt_file), "r", encoding="utf-8").read())
        if len(items) > limit_per_set:
            items = sorted(random.Random(7).sample(list(items), limit_per_set), key=lambda x: x["filename"])
        for it in items:
            if str(it.get("level")) == "5":
                continue
            p = os.path.join(bench, "images", it["filename"])
            if os.path.exists(p):
                lines = it["lines"] if isinstance(it["lines"], list) else ast.literal_eval(it["lines"])
                pages.append({"group": f"{tag}:1-4", "filename": it["filename"], "path": p, "gt_lines": [l["text"] for l in lines]})
    return pages


def worker_guard(name):
    """Non-regression guard on the Level 1-4 pages (clean / mildly degraded) of the same samples."""
    import app as studio_app
    from evaluate_level5_pages import norm_audit, matched_cer
    from evaluate_pipeline import levenshtein_distance

    cfg = CONFIGS[name]
    studio_app.DET_MODE = "hybrid"
    studio_app.det_model = studio_app.get_generic_det_model()
    studio_app._cham_det_failed = False
    studio_app.CHAM_DET_POLY_VPAD = float(cfg["vpad"])
    studio_app.CHAM_DET_POLY_MASK = bool(cfg["mask"])
    studio_app.cham_det_model = studio_app._build_text_detector(
        os.path.abspath(_det_dir(cfg["det"])), cfg["db_thresh"], cfg["box_thresh"], cfg["unclip"], box_type=cfg["box_type"])
    t0 = time.time()
    by_g, per_page = {}, []
    for pg in _load_lower_level_pages():
        res = studio_app.run_ocr_pipeline(cv2.imread(pg["path"]), model_ver=REC_VER, method="dbnet")
        pl = [l["text"] for l in res.get("lines", [])]
        sw = bool(res.get("profiler", {}).get("detector_switched"))
        gl = pg["gt_lines"]
        st = by_g.setdefault(pg["group"], {"n": 0, "gt_l": 0, "pr_l": 0, "d_err": 0, "d_den": 0, "m_err": 0, "m_den": 0, "switched": 0})
        gd, pd = norm_audit(" ".join(gl)), norm_audit(" ".join(pl))
        de = levenshtein_distance(gd, pd)
        me, md = matched_cer(gl, pl)
        st["n"] += 1
        st["gt_l"] += len(gl)
        st["pr_l"] += len(pl)
        st["d_err"] += de
        st["d_den"] += len(gd)
        st["m_err"] += me
        st["m_den"] += md
        st["switched"] += int(sw)
        per_page.append({"group": pg["group"], "file": pg["filename"], "lines": f"{len(pl)}/{len(gl)}", "switched": sw,
                         "doc_cer_pct": round(100.0 * de / max(1, len(gd)), 2)})
    summ = {g: {"n": s["n"], "lines": f"{s['pr_l']}/{s['gt_l']}", "cham_switched_pages": s["switched"],
                "doc_cer_pct": round(100.0 * s["d_err"] / max(1, s["d_den"]), 2),
                "matched_cer_pct": round(100.0 * s["m_err"] / max(1, s["m_den"]), 2)} for g, s in by_g.items()}
    return {"config": name, "params": cfg, "recognizer": REC_VER, "lower_levels": summ, "per_page": per_page,
            "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------------------------
# Orchestration / report
# ------------------------------------------------------------------------------------------------
def _part_path(kind, name):
    safe = name if isinstance(name, str) else "_".join(str(x) for x in name)
    return os.path.join(PARTS_DIR, f"{kind}__{safe}.json")


def run_parallel(jobs, configs, det_keys, force=False, guard_configs=()):
    os.makedirs(PARTS_DIR, exist_ok=True)
    queue = []
    for c in guard_configs:
        if force or not os.path.exists(_part_path("guard", c)):
            queue.append(["--worker", "guard", "--config", c])
    for c in configs:
        if force or not os.path.exists(_part_path("e2e", c)):
            queue.append(["--worker", "e2e", "--config", c])
    for k in det_keys:
        if force or not os.path.exists(_part_path("det", k)):
            queue.append(["--worker", "det", "--det-key", f"{k[0]}:{k[1]}:{k[2]}"])
    running = []
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    while queue or running:
        while queue and len(running) < jobs:
            a = queue.pop(0)
            log = os.path.join(PARTS_DIR, "log__" + "_".join(x.replace(":", "_").replace("-", "") for x in a[1:]) + ".txt")
            fh = open(log, "w", encoding="utf-8")
            p = subprocess.Popen([PY313, os.path.abspath(__file__)] + a, cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
            running.append((p, a, fh))
            print(f"▶️  started {' '.join(a)} (pid {p.pid})", flush=True)
        time.sleep(5)
        for item in list(running):
            p, a, fh = item
            if p.poll() is not None:
                fh.close()
                running.remove(item)
                print(f"{'✅' if p.returncode == 0 else '❌'} finished {' '.join(a)} rc={p.returncode}", flush=True)


def _promotion(base, cand):
    """Promotion criterion vs v1_quad production (see task brief)."""
    def g(r, grp):
        return (r["level5_and_audit"].get(grp) or {}).get("doc_cer_pct", 0.0)
    checks = {
        "harum_cer_le_+0.25": cand["harum_palei_01"]["e2e_doc_cer_pct"] <= base["harum_palei_01"]["e2e_doc_cer_pct"] + 0.25,
        "harum_recall_ge_96": cand["harum_palei_01"]["line_recall_pct"] >= 96.0,
        "audit18_le_+0.30": g(cand, "audit_18") <= g(base, "audit_18") + 0.30,
        "a4_5_le_+0.50": g(cand, "a4:5") <= g(base, "a4:5") + 0.50,
        "short_5_le_+0.50": g(cand, "short:5") <= g(base, "short:5") + 0.50,
        "ood12_cer_le_+0.30": cand["ood_12"]["e2e_doc_cer_pct"] <= base["ood_12"]["e2e_doc_cer_pct"] + 0.30,
        "ood12_no_recall_loss": cand["ood_12"]["line_recall_pct"] >= base["ood_12"]["line_recall_pct"],
    }
    passed = all(checks.values())
    strictly_better = any(g(cand, k) < g(base, k) for k in ("short:5", "a4:5", "audit_18"))
    return {"checks": checks, "passes": passed, "strictly_better_on_lvl5_or_audit": strictly_better,
            "eligible_for_default": passed and strictly_better}


def aggregate():
    e2e, det, guard = {}, {}, {}
    if not os.path.isdir(PARTS_DIR):
        raise SystemExit(f"No parts in {PARTS_DIR}")
    for fn in sorted(os.listdir(PARTS_DIR)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(PARTS_DIR, fn), "r", encoding="utf-8"))
        if fn.startswith("e2e__"):
            e2e[d["config"]] = d
        elif fn.startswith("guard__"):
            guard[d["config"]] = d
        elif fn.startswith("det__"):
            det["{}:{}:{}".format(*d["key"])] = d
    base = e2e.get("v1_quad")
    promo = {k: _promotion(base, v) for k, v in e2e.items() if base is not None and k != "v1_quad"}
    eligible = sorted([k for k, v in promo.items() if v["eligible_for_default"]],
                      key=lambda k: (e2e[k]["level5_and_audit"].get("a4:5", {}).get("doc_cer_pct", 1e9)
                                     + e2e[k]["level5_and_audit"].get("short:5", {}).get("doc_cer_pct", 1e9)
                                     + e2e[k]["level5_and_audit"].get("audit_18", {}).get("doc_cer_pct", 1e9)))
    report = {"title": "Cham-DBNet polygon detection + ribbon rectification evaluation",
              "evaluated_at": datetime.now(timezone.utc).isoformat(), "recognizer_fixed": REC_VER,
              "det_mode": "hybrid", "configs": {k: CONFIGS.get(k) for k in e2e},
              "end_to_end": {k: {kk: vv for kk, vv in v.items() if kk != "per_page"} for k, v in e2e.items()},
              "per_page": {k: v.get("per_page") for k, v in e2e.items()},
              "detector_only": det, "lower_level_guard": {k: {kk: vv for kk, vv in v.items() if kk != "per_page"} for k, v in guard.items()},
              "promotion_vs_v1_quad": promo, "eligible_configs_ranked": eligible}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    def L(r, grp, key="doc_cer_pct"):
        s = r["level5_and_audit"].get(grp) or {}
        return s.get(key)

    order = [c for c in list(CONFIGS) if c in e2e] + [c for c in e2e if c not in CONFIGS]
    md = ["# Đánh giá Cham-DBNet Polygon + Ribbon Rectification (`CHAM_DET_BOX_TYPE=poly`)",
          f"- **Thời điểm**: `{report['evaluated_at']}`",
          f"- **Recognizer cố định**: `{REC_VER}` · **DET_MODE**: `hybrid` (Legacy-First, generic PP-OCRv4 DBNet primary)",
          "- **Baseline production**: `v1_quad` (Cham-DBNet v1, `box_type='quad'`, perspective quad crop).",
          "- `poly` = `DBPostProcess(box_type='poly')` + `merge_line_polygons` + `rectify_polygon_line` (centerline remap, "
          "padding `vpad`×median line thickness, mask ngoài polygon/đè polygon dòng lân cận).",
          "- Doc CER = Levenshtein trên toàn trang (normalize_unicode, nối dòng bằng dấu cách); Matched CER = ghép dòng tham lam GT↔pred.",
          "",
          "## 1. End-to-End (cùng recognizer, hybrid)",
          "| Config | Det | box | unclip / box_th / db_th | vpad / mask | `harum_palei_01` Doc CER (lines, recall) | `short:5` Doc / Matched (lines) | `a4:5` Doc / Matched (lines) | `audit_18` Doc / Matched (lines) | OOD-12 E2E Doc CER (lines, recall) |",
          "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for k in order:
        r = e2e[k]
        p = r["params"]
        h = r["harum_palei_01"]
        o = r["ood_12"]
        md.append(
            f"| `{k}` | {p['det']} | {p['box_type']} | {p['unclip']} / {p['box_thresh']} / {p['db_thresh']} | "
            f"{p['vpad'] if p['box_type'] == 'poly' else '—'} / {('on' if p['mask'] else 'off') if p['box_type'] == 'poly' else '—'} | "
            f"{h['e2e_doc_cer_pct']:.2f}% ({h['pred_lines']}/{h['gt_lines']}, {h['line_recall_pct']:.0f}%) | "
            f"{L(r, 'short:5')}% / {L(r, 'short:5', 'matched_cer_pct')}% ({L(r, 'short:5', 'lines')}) | "
            f"{L(r, 'a4:5')}% / {L(r, 'a4:5', 'matched_cer_pct')}% ({L(r, 'a4:5', 'lines')}) | "
            f"{L(r, 'audit_18')}% / {L(r, 'audit_18', 'matched_cer_pct')}% ({L(r, 'audit_18', 'lines')}) | "
            f"{o['e2e_doc_cer_pct']:.2f}% ({o['e2e_lines']}, {o['line_recall_pct']:.2f}%) |")
    md += ["", "Số trang Legacy-First gate chuyển sang Cham-DBNet (short:5 / a4:5 / audit_18 / OOD-12):", ""]
    for k in order:
        r = e2e[k]
        md.append(f"- `{k}`: {L(r, 'short:5', 'cham_switched_pages')}/{L(r, 'short:5', 'n')} · "
                  f"{L(r, 'a4:5', 'cham_switched_pages')}/{L(r, 'a4:5', 'n')} · "
                  f"{L(r, 'audit_18', 'cham_switched_pages')}/{L(r, 'audit_18', 'n')} · "
                  f"{r['ood_12']['cham_switched_pages']}/{r['ood_12']['pages']} (thời gian chạy {r['seconds']}s)")
    md += ["", "## 2. Detector-only (raw DBNet output, IoU ≥ 0.5)",
           "| Detector | box | unclip | Val-120 H-Mean (R / P) poly-IoU | Val-160 H-Mean (R / P) poly-IoU | Val-120 legacy | Val-160 legacy |",
           "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for key in sorted(det):
        d = det[key]
        a, b = d["val_120_poly_iou"], d["val_160_poly_iou"]
        la, lb = d.get("val_120_legacy"), d.get("val_160_legacy")
        tag, box, u = d["key"]
        md.append(f"| `{tag}` | {box} | {u} | {a['hmean_pct']:.2f}% ({a['recall_pct']} / {a['precision_pct']}) | "
                  f"{b['hmean_pct']:.2f}% ({b['recall_pct']} / {b['precision_pct']}) | "
                  f"{(str(la['hmean_pct']) + '%') if la else '—'} | {(str(lb['hmean_pct']) + '%') if lb else '—'} |")
    md += ["", "> poly-IoU: GT polygon thật (4 hoặc 16 đỉnh) ∩ polygon dự đoán (shapely). Legacy: giao thức của "
           "`CHAM_DBNET_V2_1_EVALUATION.md` (`cv2.intersectConvexConvex`, GT Val-160 quy về minAreaRect) — chỉ hợp lệ cho quad.",
           "", "## 3. Promotion criterion (so với `v1_quad`)",
           "| Config | harum ≤+0.25 | harum recall ≥96% | audit_18 ≤+0.30 | a4:5 ≤+0.50 | short:5 ≤+0.50 | OOD-12 ≤+0.30 | OOD-12 recall | PASS | Tốt hơn hẳn L5/audit |",
           "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    yn = lambda b: "✅" if b else "❌"
    for k in order:
        if k not in promo:
            continue
        c = promo[k]["checks"]
        md.append(f"| `{k}` | {yn(c['harum_cer_le_+0.25'])} | {yn(c['harum_recall_ge_96'])} | {yn(c['audit18_le_+0.30'])} | "
                  f"{yn(c['a4_5_le_+0.50'])} | {yn(c['short_5_le_+0.50'])} | {yn(c['ood12_cer_le_+0.30'])} | "
                  f"{yn(c['ood12_no_recall_loss'])} | **{yn(promo[k]['passes'])}** | {yn(promo[k]['strictly_better_on_lvl5_or_audit'])} |")
    md += ["", f"**Config đủ điều kiện làm mặc định (xếp theo tổng Doc CER L5+audit)**: "
           f"{', '.join('`' + k + '`' for k in eligible) if eligible else 'không có — giữ nguyên production (`v1_quad`)'}"]
    if guard:
        md += ["", "## 4. Guard không hồi quy trên trang Level 1-4 (cùng mẫu 90+90 trang seed=7)",
               "| Config | `short:1-4` Doc / Matched (lines, switched) | `a4:1-4` Doc / Matched (lines, switched) |",
               "| :--- | :---: | :---: |"]
        for k in [c for c in order if c in guard]:
            s = guard[k]["lower_levels"]
            f = lambda g: (f"{s[g]['doc_cer_pct']}% / {s[g]['matched_cer_pct']}% ({s[g]['lines']}, {s[g]['cham_switched_pages']}/{s[g]['n']})"
                           if g in s else "—")
            md.append(f"| `{k}` | {f('short:1-4')} | {f('a4:1-4')} |")
    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print(f"✅ Report -> {OUT_MD}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["det", "e2e", "guard"])
    ap.add_argument("--config")
    ap.add_argument("--det-key")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--configs", nargs="*", help="subset of configs to run (default: all)")
    ap.add_argument("--guard-configs", nargs="*", default=[], help="configs to also run on Level 1-4 pages")
    ap.add_argument("--skip-det", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    _register_extra_configs()
    if args.worker == "e2e":
        out = worker_e2e(args.config)
        os.makedirs(PARTS_DIR, exist_ok=True)
        with open(_part_path("e2e", args.config), "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(json.dumps({k: v for k, v in out.items() if k != "per_page"}, ensure_ascii=False))
        return
    if args.worker == "guard":
        out = worker_guard(args.config)
        os.makedirs(PARTS_DIR, exist_ok=True)
        with open(_part_path("guard", args.config), "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(json.dumps({k: v for k, v in out.items() if k != "per_page"}, ensure_ascii=False))
        return
    if args.worker == "det":
        tag, box, u = args.det_key.split(":")
        key = (tag, box, float(u))
        out = worker_det(key)
        os.makedirs(PARTS_DIR, exist_ok=True)
        with open(_part_path("det", key), "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(json.dumps(out, ensure_ascii=False))
        return
    if not args.aggregate:
        cfgs = args.configs if args.configs is not None else list(CONFIGS)
        run_parallel(args.jobs, cfgs, [] if args.skip_det else DET_KEYS, force=args.force, guard_configs=args.guard_configs)
    aggregate()


if __name__ == "__main__":
    main()
