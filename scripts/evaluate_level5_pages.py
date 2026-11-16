#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR Level-5 Full-Page Degradation & 18-Image Audit Suite Evaluator
=======================================================================
Evaluates multiple recognizer models (default + candidate aliases) through
`ocr-studio/app.py` (`DET_MODE='hybrid'`, Cham-DBNet + Legacy-First) on:
1. `short:5` (`benchmark_gt.json`, level == 5)
2. `a4:5` (`benchmark_a4_gt.json`, level == 5)
3. `ocr-audit-2026-11-04` 18-image audit suite (when present on disk)

Reuses DBNet line detections per page across models (preserving per-model
quality gates & Cham/v24 normalization) so multi-model comparison completes fast.
"""

import os
import sys
import ast
import json
import time
import random
import argparse
import pathlib
from datetime import datetime, timezone

import cv2
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIO_DIR = os.path.join(PROJECT_ROOT, "ocr-studio")
BENCH_DIR = os.path.join(PROJECT_ROOT, "ocr-benchmark", "data")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

if STUDIO_DIR not in sys.path:
    sys.path.insert(0, STUDIO_DIR)
if os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts"))

import app as studio_app
from evaluate_pipeline import levenshtein_distance
from scripts.generate_data import normalize_unicode


DEFAULT_AUDIT_DIR = pathlib.Path(r"C:\Users\admin\Documents\Codex\2026-11-04\ti\outputs\ocr-audit-2026-11-04")


def _build_direct_rec(model_dir: str, dict_path: str):
    from tools.infer.predict_rec import TextRecognizer
    import tools.infer.utility as utility

    backup = sys.argv
    sys.argv = [sys.argv[0]]
    args = utility.parse_args()
    sys.argv = backup
    args.rec_model_dir = os.path.abspath(model_dir)
    args.rec_char_dict_path = os.path.abspath(dict_path)
    args.use_gpu = False
    args.rec_image_shape = "3, 48, 320"
    args.use_space_char = True
    args.rec_batch_num = 1
    args.enable_mkldnn = False
    args.cpu_threads = 1
    args.ir_optim = False
    args.precision = "fp32"
    return TextRecognizer(args)


def _split_model_dict(rest: str):
    if "::" in rest:
        return rest.split("::", 1)
    for idx, ch in enumerate(rest):
        if ch == ":":
            if idx == 1 and rest[0].isalpha() and rest[2:3] in ("\\", "/"):
                continue
            return rest[:idx], rest[idx + 1:]
    raise ValueError(f"Expected 'model_dir:dict_path' in '{rest}'")


def register_model(spec_str: str):
    if "=" in spec_str:
        alias, rest = spec_str.split("=", 1)
        mdir, dpath = _split_model_dict(rest)
        rec = _build_direct_rec(mdir, dpath)
        studio_app.ocr_models[alias] = rec
        studio_app.ocr_model_paths[alias] = {"model_dir": os.path.abspath(mdir), "dict_path": os.path.abspath(dpath)}
        return alias
    studio_app.get_ocr_model(spec_str)
    return spec_str


def norm_audit(s: str) -> str:
    return " ".join(normalize_unicode(s).split())


def matched_cer(gl: list, pl: list):
    pairs = []
    gl_n = [norm_audit(g) for g in gl]
    pl_n = [norm_audit(p) for p in pl]
    for gi, a in enumerate(gl_n):
        for pi, b in enumerate(pl_n):
            pairs.append((levenshtein_distance(a, b) / max(len(a), 1), gi, pi, a, b))
    ug, up, err = set(), set(), 0
    for _, gi, pi, a, b in sorted(pairs):
        if gi in ug or pi in up:
            continue
        ug.add(gi)
        up.add(pi)
        err += levenshtein_distance(a, b)
    err += sum(len(a) for i, a in enumerate(gl_n) if i not in ug)
    err += sum(len(b) for i, b in enumerate(pl_n) if i not in up)
    return err, sum(len(a) for a in gl_n)


def detect_once(img):
    ink = studio_app.analyze_text_ink(img)
    if studio_app.is_blank_page(ink):
        return [], [], {"blank_page": True, "detector": None}, ink
    det = studio_app.get_det_model()
    crops, metas, det_info = studio_app.segment_lines_dbnet(img, det, ink=ink, return_info=True, reading_order="auto")
    if not crops:
        _, coords = studio_app.segment_lines_valleys(img, window_size=25, min_dist=12)
        return [], coords, {"fallback_valley": True, "detector": "valley"}, ink
    rec_crops = [studio_app.preprocess_crop_for_rec(c) for c in crops]
    return rec_crops, metas, det_info, ink


def run_cached_rec_on_page(model_ver: str, img, cached_det):
    rec_crops, metas_or_coords, det_info, ink = cached_det
    if det_info.get("blank_page"):
        return [], "no_text"
    ocr = studio_app.get_ocr_model(model_ver)
    if det_info.get("fallback_valley"):
        # Full fallback path
        res = studio_app.run_ocr_pipeline(img, model_ver=model_ver, method="valley")
        return [l["text"] for l in res.get("lines", [])], (res.get("document") or {}).get("status", "needs_review")

    rec_res, _ = ocr(rec_crops)
    # Copy metadata cleanly so models don't mutate shared cached dict
    metas_copy = []
    for m, (ptxt, pconf) in zip(metas_or_coords, rec_res):
        mc = {
            "line_id": m.get("line_id"),
            "bbox": list(m["bbox"]),
            "core_box": list(m["core_box"]),
            "selected_crop_type": m.get("selected_crop_type", "dbnet"),
            "reason": list(m.get("reason") or []),
            "candidates": {"dbnet": {"prediction": ptxt, "confidence": float(pconf)}},
        }
        metas_copy.append(mc)
    dropped = []
    kept_meta, kept_crops = studio_app.apply_line_quality_gates(metas_copy, rec_crops, dropped)
    is_logical = model_ver == "auto" or model_ver in ("v24", "v25", "v26") or str(model_ver).startswith("v24")
    preds = [
        normalize_unicode(km["candidates"][km["selected_crop_type"]]["prediction"]) if is_logical
        else km["candidates"][km["selected_crop_type"]]["prediction"]
        for km in kept_meta
    ]
    doc_reasons = []
    if dropped:
        doc_reasons.append("unreadable_lines_dropped")
    if not kept_meta:
        doc_reasons.append("no_reliable_text")
    unc = det_info.get("uncovered_ink_ratio")
    if unc is not None and unc >= studio_app.UNCOVERED_INK_REVIEW_RATIO:
        doc_reasons.append("possible_missed_lines")
    status = "needs_review" if (doc_reasons or any(km.get("needs_review") for km in kept_meta)) else "ocr_done"
    return preds, status


def load_pages(limit_per_set: int = 90):
    pages = []
    # Match scratch/bench_remote.py sampling when limit < 200 so subsets include the exact lvl5 benchmark slices
    for gt_file, tag in (("benchmark_gt.json", "short"), ("benchmark_a4_gt.json", "a4")):
        items = json.loads(open(os.path.join(BENCH_DIR, gt_file), "r", encoding="utf-8").read())
        if len(items) > limit_per_set:
            rnd = random.Random(7)
            items = sorted(rnd.sample(list(items), limit_per_set), key=lambda x: x["filename"])
        for it in items:
            if str(it.get("level")) != "5":
                continue
            p = os.path.join(BENCH_DIR, "images", it["filename"])
            if os.path.exists(p):
                lines = it["lines"] if isinstance(it["lines"], list) else ast.literal_eval(it["lines"])
                gl = [l["text"] for l in lines]
                pages.append({"group": f"{tag}:5", "filename": it["filename"], "path": p, "gt_lines": gl})

    # Optional 18-image audit suite
    if (DEFAULT_AUDIT_DIR / "raw-results.json").exists():
        raw = json.loads((DEFAULT_AUDIT_DIR / "raw-results.json").read_text(encoding="utf-8-sig"))
        seen_f = set()
        for r in raw.get("results", []):
            if r.get("kind") not in ("benchmark", "fresh"):
                continue
            s = r["sample"]
            fn = s["filename"]
            if fn in seen_f:
                continue
            p = DEFAULT_AUDIT_DIR / "inputs" / fn
            if p.exists():
                seen_f.add(fn)
                gl = [x["text"] for x in s.get("lines", [])]
                pages.append({"group": "audit_18", "filename": fn, "path": str(p), "gt_lines": gl})
    return pages


def main():
    parser = argparse.ArgumentParser(description="Evaluate Level-5 pages and 18-image audit suite.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=90, help="Sampling limit per full set (matches bench_remote.py=90; use 200 for all lvl5)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    studio_app.get_det_model()
    aliases = [register_model(m) for m in args.models]
    pages = load_pages(limit_per_set=args.limit)
    print(f"Loaded {len(pages)} pages across groups: " + str({g: sum(1 for x in pages if x['group'] == g) for g in sorted({x['group'] for x in pages})}))

    # Detect once per page
    det_cache = {}
    img_cache = {}
    for pg in pages:
        img = cv2.imread(pg["path"])
        img_cache[pg["path"]] = img
        det_cache[pg["path"]] = detect_once(img)

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }
    for alias in aliases:
        by_group = {}
        t0 = time.time()
        for pg in pages:
            g = pg["group"]
            by_group.setdefault(g, {"n": 0, "doc_err": 0, "doc_den": 0, "m_err": 0, "m_den": 0, "gt_lines": 0, "pred_lines": 0, "abs_line_err": 0})
            pl, status = run_cached_rec_on_page(alias, img_cache[pg["path"]], det_cache[pg["path"]])
            gl = pg["gt_lines"]
            st = by_group[g]
            st["n"] += 1
            st["gt_lines"] += len(gl)
            st["pred_lines"] += len(pl)
            st["abs_line_err"] += abs(len(pl) - len(gl))
            if gl:
                gt_doc = norm_audit(" ".join(gl))
                pr_doc = norm_audit(" ".join(pl))
                st["doc_err"] += levenshtein_distance(gt_doc, pr_doc)
                st["doc_den"] += len(gt_doc)
                me, md = matched_cer(gl, pl)
                st["m_err"] += me
                st["m_den"] += md

        summary_m = {}
        for g, st in by_group.items():
            summary_m[g] = {
                "n": st["n"],
                "lines": f"{st['pred_lines']}/{st['gt_lines']}",
                "abs_line_err": st["abs_line_err"],
                "doc_cer_pct": round(100.0 * st["doc_err"] / max(1, st["doc_den"]), 2) if st["doc_den"] else None,
                "matched_cer_pct": round(100.0 * st["m_err"] / max(1, st["m_den"]), 2) if st["m_den"] else None,
            }
        report["models"][alias] = summary_m
        print(f"  [{alias}] ({time.time() - t0:.1f}s) -> {summary_m}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✅ Level-5 page evaluation saved -> {args.output}")


if __name__ == "__main__":
    main()
