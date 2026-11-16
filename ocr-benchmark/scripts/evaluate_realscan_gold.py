#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR RealScan-Gold v1 Evaluation Runner
===========================================
Verifies SHA256 integrity of `ocr-benchmark/data/realscan_gold_v1/manifest.json`
and evaluates candidate recognizers (+ `auto` dual-engine router) on:
1. `harum_palei_01`:
   - 25 line crops (Norm CER, Raw CER, WER, Exact Line Match, Date line #22 `꩘/꩕/꩒꩐꩒꩖`, Hyphen lines #2 & #6).
   - Full-page End-to-End `run_ocr_pipeline` on `original_image.png` (Detected lines, Recall, Matched CER, Document CER).
2. `handoff_multiscript_01`:
   - 76 crops (`cham_script` 27, `vietnamese_english` 24, `latin` 25) testing script separation & `auto` routing vs pure-Cham isolation.
"""

import os
import sys
import json
import hashlib
import argparse
from datetime import datetime, timezone

import cv2
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)
STUDIO_DIR = os.path.join(PROJECT_ROOT, "ocr-studio")

if STUDIO_DIR not in sys.path:
    sys.path.insert(0, STUDIO_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import app as studio_app
from evaluate_pipeline import levenshtein_distance
from scripts.generate_data import normalize_unicode


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def verify_manifest_integrity(manifest_path: str) -> dict:
    m = json.load(open(manifest_path, "r", encoding="utf-8"))
    checked = 0
    for doc in m["documents"]:
        if doc.get("page_relpath"):
            p = os.path.join(PROJECT_ROOT, doc["page_relpath"].replace("/", os.sep))
            assert os.path.exists(p), f"Missing {p}"
            got = sha256_file(p)
            assert got == doc["page_sha256"], f"SHA mismatch {p}: {got} != {doc['page_sha256']}"
            checked += 1
        items = list(doc.get("lines", []))
        for grp_items in doc.get("groups", {}).values():
            items.extend(grp_items)
        for it in items:
            p = os.path.join(PROJECT_ROOT, it["crop_relpath"].replace("/", os.sep))
            assert os.path.exists(p), f"Missing {p}"
            got = sha256_file(p)
            assert got == it["crop_sha256"], f"SHA mismatch {p}: {got} != {it['crop_sha256']}"
            checked += 1
    print(f"✅ RealScan-Gold v1 SHA256 integrity verified: {checked} files match 100%.")
    return m


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


def resolve_model_runner(spec_str: str):
    """
    Accepts either:
      - known studio version string ('v24', 'v24_1_r2_1_6', 'auto', ...)
      - explicit 'alias=model_dir:dict_path'
    Registers custom alias into studio_app.ocr_models so run_ocr_pipeline() also works seamlessly!
    """
    if "=" in spec_str:
        alias, rest = spec_str.split("=", 1)
        mdir, dpath = _split_model_dict(rest)
        rec = _build_direct_rec(mdir, dpath)
        studio_app.ocr_models[alias] = rec
        studio_app.ocr_model_paths[alias] = {"model_dir": os.path.abspath(mdir), "dict_path": os.path.abspath(dpath)}
        is_logical = True
    else:
        alias = spec_str
        rec = studio_app.get_ocr_model(alias)
        is_logical = alias == "auto" or alias in ("v24", "v25", "v26") or alias.startswith("v24")
    return alias, rec, is_logical


def matched_line_stats(gt_lines: list, pred_lines: list):
    pairs = []
    for gi, g in enumerate(gt_lines):
        gn = normalize_unicode(g)
        for pi, p in enumerate(pred_lines):
            pn = normalize_unicode(p)
            d = levenshtein_distance(gn, pn)
            pairs.append((d / max(1, len(gn)), d, gi, pi))
    used_g, used_p = set(), set()
    matched_ed, matched_chars = 0, 0
    for _, d, gi, pi in sorted(pairs):
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched_ed += d
        matched_chars += len(normalize_unicode(gt_lines[gi]))
    unmatched_g_chars = sum(len(normalize_unicode(g)) for i, g in enumerate(gt_lines) if i not in used_g)
    unmatched_p_chars = sum(len(normalize_unicode(p)) for i, p in enumerate(pred_lines) if i not in used_p)
    recall = len(used_g) / max(1, len(gt_lines))
    matched_cer = matched_ed / max(1, matched_chars)
    total_gt_doc = " ".join(normalize_unicode(g) for g in gt_lines)
    total_pred_doc = " ".join(normalize_unicode(p) for p in pred_lines)
    doc_cer = levenshtein_distance(total_gt_doc, total_pred_doc) / max(1, len(total_gt_doc))
    return {
        "gt_lines": len(gt_lines),
        "pred_lines": len(pred_lines),
        "matched_lines": len(used_g),
        "line_recall_pct": round(recall * 100.0, 2),
        "matched_line_cer_pct": round(matched_cer * 100.0, 2),
        "e2e_doc_cer_pct": round(doc_cer * 100.0, 2),
        "unmatched_gt_chars": unmatched_g_chars,
        "unmatched_pred_chars": unmatched_p_chars,
    }


def evaluate_model(alias: str, rec_model, is_logical: bool, harum_doc: dict, handoff_doc: dict) -> dict:
    print(f"\n[Evaluate RealScan-Gold v1] Model `{alias}`...")

    # 1. Harum Palei 25 line crops
    raw_ed, norm_ed, tot_chars = 0, 0, 0
    wer_ed, wer_words, exact_cnt = 0, 0, 0
    raw_confs, cal_confs = [], []
    line_details = []

    for ln in harum_doc["lines"]:
        cp = os.path.join(PROJECT_ROOT, ln["crop_relpath"].replace("/", os.sep))
        img = cv2.imread(cp)
        prep = studio_app.preprocess_crop_for_rec(img)
        res, _ = rec_model([prep])
        pred_raw, conf = res[0] if res else ("", 0.0)
        pred_norm = normalize_unicode(pred_raw) if is_logical else pred_raw
        gt_raw = ln["gt"]
        gt_norm = normalize_unicode(gt_raw)

        re_d = levenshtein_distance(gt_raw, pred_raw)
        ne_d = levenshtein_distance(gt_norm, pred_norm)
        raw_ed += re_d
        norm_ed += ne_d
        tot_chars += len(gt_norm)

        gw, pw = gt_norm.split(), pred_norm.split()
        wer_ed += levenshtein_distance(gw, pw)
        wer_words += len(gw)
        if pred_norm == gt_norm:
            exact_cnt += 1

        cal_c = studio_app.calibrate_line_confidence(conf, alias, pred_norm)
        raw_confs.append(float(conf))
        cal_confs.append(float(cal_c))
        line_details.append({
            "line": ln["line"],
            "gt": gt_norm,
            "pred": pred_norm,
            "cer_pct": round(100.0 * ne_d / max(1, len(gt_norm)), 2),
            "raw_conf": round(float(conf), 4),
            "calibrated_conf": round(float(cal_c), 4),
        })

    # Key diagnostic lines in harum_palei_01: line 2 ('ꨁ-ꨗꨫ'), line 6 ('ꨤꨪ-ꨂꨀꨩ'), line 22 date ('꩘/꩕/꩒꩐꩒꩖')
    l2 = line_details[1]
    l6 = line_details[5]
    l22 = line_details[21]

    # 2. Harum Palei Full-Page End-to-End via ocr-studio/app.py run_ocr_pipeline
    page_path = os.path.join(PROJECT_ROOT, harum_doc["page_relpath"].replace("/", os.sep))
    page_img = cv2.imread(page_path)
    pipe_res = studio_app.run_ocr_pipeline(page_img, model_ver=alias, method="dbnet")
    pred_fp_lines = [l["text"] for l in pipe_res.get("lines", [])]
    gt_fp_lines = [ln["gt"] for ln in harum_doc["lines"]]
    fp_stats = matched_line_stats(gt_fp_lines, pred_fp_lines)
    fp_stats["detector"] = (pipe_res.get("profiler") or {}).get("detector")
    fp_stats["doc_status"] = (pipe_res.get("document") or {}).get("status")

    # 3. Handoff Multi-Script 76 crops
    group_stats = {}
    for gname, gitems in handoff_doc["groups"].items():
        confs = []
        cham_chars_cnt = 0
        non_cham_alnum_cnt = 0
        non_empty = 0
        ge_065 = 0
        for it in gitems:
            cp = os.path.join(PROJECT_ROOT, it["crop_relpath"].replace("/", os.sep))
            img = cv2.imread(cp)
            prep = studio_app.preprocess_crop_for_rec(img)
            res, _ = rec_model([prep])
            ptxt, pconf = res[0] if res else ("", 0.0)
            ptxt = normalize_unicode(ptxt) if is_logical else ptxt
            confs.append(float(pconf))
            if float(pconf) >= 0.65:
                ge_065 += 1
            if ptxt.strip():
                non_empty += 1
            if any("\uAA00" <= ch <= "\uAA5F" for ch in ptxt):
                cham_chars_cnt += 1
            if any(("a" <= ch.lower() <= "z") or ("\u00C0" <= ch <= "\u1EF9") for ch in ptxt):
                non_cham_alnum_cnt += 1
        group_stats[gname] = {
            "count": len(gitems),
            "mean_conf": round(float(np.mean(confs)), 4),
            "pct_ge_065": round(100.0 * ge_065 / max(1, len(gitems)), 1),
            "pct_has_cham_script": round(100.0 * cham_chars_cnt / max(1, len(gitems)), 1),
            "pct_has_latin_viet": round(100.0 * non_cham_alnum_cnt / max(1, len(gitems)), 1),
        }

    crop_summary = {
        "lines": len(harum_doc["lines"]),
        "raw_cer_pct": round(100.0 * raw_ed / max(1, tot_chars), 2),
        "norm_cer_pct": round(100.0 * norm_ed / max(1, tot_chars), 2),
        "wer_pct": round(100.0 * wer_ed / max(1, wer_words), 2),
        "exact_lines": f"{exact_cnt}/{len(harum_doc['lines'])}",
        "exact_line_pct": round(100.0 * exact_cnt / max(1, len(harum_doc["lines"])), 1),
        "mean_raw_conf": round(float(np.mean(raw_confs)), 4),
        "mean_calibrated_conf": round(float(np.mean(cal_confs)), 4),
        "line2_hyphen_ok": ("-" in l2["pred"]),
        "line6_hyphen_ok": ("-" in l6["pred"]),
        "line22_date_pred": l22["pred"],
        "line22_date_exact": (l22["pred"] == l22["gt"]),
    }
    print(
        f"  • Harum Crops: NormCER={crop_summary['norm_cer_pct']:.2f}% WER={crop_summary['wer_pct']:.2f}% Exact={crop_summary['exact_lines']} Date22='{l22['pred']}'\n"
        f"  • Harum Page:  {fp_stats['pred_lines']}/25 lines MatchedCER={fp_stats['matched_line_cer_pct']:.2f}% E2E_CER={fp_stats['e2e_doc_cer_pct']:.2f}% ({fp_stats['detector']})\n"
        f"  • Multiscript: cham_script conf={group_stats['cham_script']['mean_conf']:.3f} ({group_stats['cham_script']['pct_ge_065']}%>=0.65) | "
        f"latin latin/vi={group_stats['latin']['pct_has_latin_viet']}% conf={group_stats['latin']['mean_conf']:.3f} | "
        f"vi_en latin/vi={group_stats['vietnamese_english']['pct_has_latin_viet']}% conf={group_stats['vietnamese_english']['mean_conf']:.3f}"
    )
    return {
        "model": alias,
        "harum_palei_crops": crop_summary,
        "harum_palei_fullpage": fp_stats,
        "harum_palei_line_details": line_details,
        "handoff_multiscript_01": group_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cham-OCR models on RealScan-Gold v1.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names ('v24', 'v24_1_r2_1_6', 'auto') or 'alias=model_dir:dict_path'",
    )
    args = parser.parse_args()

    manifest_path = os.path.join(BENCHMARK_DIR, "data", "realscan_gold_v1", "manifest.json")
    manifest = verify_manifest_integrity(manifest_path)
    harum_doc = next(d for d in manifest["documents"] if d["doc_id"] == "harum_palei_01")
    handoff_doc = next(d for d in manifest["documents"] if d["doc_id"] == "handoff_multiscript_01")

    results = []
    for sp in args.models:
        alias, rec, is_logical = resolve_model_runner(sp)
        results.append(evaluate_model(alias, rec, is_logical, harum_doc, handoff_doc))

    out = {
        "title": "Cham-OCR RealScan-Gold v1 End-to-End & Multiscript Evaluation",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": manifest_path,
        "models": results,
    }
    os.makedirs(os.path.join(PROJECT_ROOT, "REPORTS"), exist_ok=True)
    out_json = os.path.join(PROJECT_ROOT, "REPORTS", "realscan_gold_v1_evaluation.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    md = [
        "# Đánh giá RealScan-Gold v1 (`harum_palei_01` + `handoff_multiscript_01`)",
        f"- **Thời điểm**: `{out['evaluated_at']}`",
        "",
        "## 1. Tài liệu thực tế `harum_palei_01` (25 dòng chụp ảnh thực tế)",
        "| Model | Crop Norm CER | WER | Exact Lines | Line #22 Date (`꩘/꩕/꩒꩐꩒꩖`) | Page Lines | FullPage Matched CER | E2E Doc CER |",
        "| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |",
    ]
    for r in results:
        c = r["harum_palei_crops"]
        fp = r["harum_palei_fullpage"]
        md.append(
            f"| `{r['model']}` | **{c['norm_cer_pct']:.2f}%** | {c['wer_pct']:.2f}% | {c['exact_lines']} | `{c['line22_date_pred']}` | "
            f"{fp['pred_lines']}/{fp['gt_lines']} | {fp['matched_line_cer_pct']:.2f}% | {fp['e2e_doc_cer_pct']:.2f}% |"
        )
    md.extend([
        "",
        "## 2. Tài liệu đa ngữ `handoff_multiscript_01` (76 crops: 27 Chăm, 25 Latin, 24 Việt-Anh)",
        "| Model | Cham `>=0.65` (Conf) | Latin routed to Latin/Vi % (Conf) | Vi-En routed to Latin/Vi % (Conf) |",
        "| :--- | :---: | :---: | :---: |",
    ])
    for r in results:
        g = r["handoff_multiscript_01"]
        md.append(
            f"| `{r['model']}` | {g['cham_script']['pct_ge_065']}% ({g['cham_script']['mean_conf']:.3f}) | "
            f"{g['latin']['pct_has_latin_viet']}% ({g['latin']['mean_conf']:.3f}) | "
            f"{g['vietnamese_english']['pct_has_latin_viet']}% ({g['vietnamese_english']['mean_conf']:.3f}) |"
        )
    out_md = os.path.join(PROJECT_ROOT, "REPORTS", "REALSCAN_GOLD_V1_REPORT.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\n✅ RealScan-Gold v1 reports written:\n  • {out_json}\n  • {out_md}")


if __name__ == "__main__":
    main()
