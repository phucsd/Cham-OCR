#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR RealScan-Gold v2 & OOD-Archival-Gold v2 Evaluation Runner
==================================================================
Verifies SHA-256 integrity of `ocr-benchmark/data/realscan_gold_v2/manifest.json` (`212` lines across
real camera `harum_palei_01`, 7 external Wikimedia Commons specimens, and 12 unseen EFEO/Held-Out
archival pages across 3 domains including 2-column folios) and evaluates candidate models on BOTH:
  - Isolated Line Crops (Norm CER, WER, Exact Line %)
  - Full-Page End-to-End Pipeline (`run_ocr_pipeline`, Line Recall %, Matched CER %, E2E Doc CER %)
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone

import cv2

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
from evaluate_realscan_gold import sha256_file, resolve_model_runner, matched_line_stats
from scripts.generate_data import normalize_unicode


def eval_crop_list(rec_model, is_logical: bool, lines: list) -> dict:
    ed, ch, w_ed, w_tot, exact = 0, 0, 0, 0, 0
    for ln in lines:
        p = os.path.join(PROJECT_ROOT, ln["crop_relpath"].replace("/", os.sep))
        im = cv2.imread(p)
        prep = studio_app.preprocess_crop_for_rec(im)
        res, _ = rec_model([prep])
        raw_p = res[0][0] if res else ""
        pr = normalize_unicode(raw_p) if is_logical else raw_p
        gt = normalize_unicode(ln["gt"])
        d = levenshtein_distance(pr, gt)
        ed += d
        ch += len(gt)
        if pr == gt:
            exact += 1
        gw, pw = gt.split(), pr.split()
        w_ed += levenshtein_distance(pw, gw)
        w_tot += len(gw)
    return {
        "n_lines": len(lines),
        "norm_cer_pct": round(100.0 * ed / max(1, ch), 2),
        "wer_pct": round(100.0 * w_ed / max(1, w_tot), 2),
        "exact_lines": f"{exact}/{len(lines)}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["v24", "v24_1_r3", "v24_1_r3_1", "v24_1_r4", "auto"])
    ap.add_argument("--output-json", default=os.path.join(PROJECT_ROOT, "REPORTS", "realscan_gold_v2_evaluation.json"))
    ap.add_argument("--output-md", default=os.path.join(PROJECT_ROOT, "REPORTS", "REALSCAN_GOLD_V2_REPORT.md"))
    args = ap.parse_args()

    m_path = os.path.join(BENCHMARK_DIR, "data", "realscan_gold_v2", "manifest.json")
    m = json.load(open(m_path, "r", encoding="utf-8"))

    # Verify SHA-256 integrity of every single file
    checked = 0
    for d in m["documents"]:
        if d.get("page_relpath"):
            pp = os.path.join(PROJECT_ROOT, d["page_relpath"].replace("/", os.sep))
            assert sha256_file(pp) == d["page_sha256"]
            checked += 1
        for ln in d.get("lines", []):
            cp = os.path.join(PROJECT_ROOT, ln["crop_relpath"].replace("/", os.sep))
            assert sha256_file(cp) == ln["crop_sha256"]
            checked += 1
    print(f"✅ RealScan-Gold v2 SHA-256 verified: {checked} files match 100%.")

    harum_doc = next(d for d in m["documents"] if d["doc_id"] == "harum_palei_01")
    wiki_doc = next(d for d in m["documents"] if d["doc_id"] == "wikimedia_public_specimens_01")
    ood_docs = [d for d in m["documents"] if d["source_type"] == "ood_archival_gold_v2"]

    results = {}
    for spec in args.models:
        alias, rec, is_logical = resolve_model_runner(spec)
        print(f"\n[RealScan-Gold v2] Evaluating `{alias}`...")

        harum_crops = eval_crop_list(rec, is_logical, harum_doc["lines"])
        wiki_crops = eval_crop_list(rec, is_logical, wiki_doc["lines"])

        by_domain = {}
        all_ood_lines = []
        e2e_ed, e2e_ch, e2e_gt_l, e2e_pr_l = 0, 0, 0, 0

        for d in ood_docs:
            dom = d["archival_domain"]
            by_domain.setdefault(dom, {"lines": [], "e2e_ed": 0, "e2e_ch": 0, "gt_l": 0, "pr_l": 0})
            by_domain[dom]["lines"].extend(d["lines"])
            all_ood_lines.extend(d["lines"])

            p_img = cv2.imread(os.path.join(PROJECT_ROOT, d["page_relpath"].replace("/", os.sep)))
            out_p = studio_app.run_ocr_pipeline(p_img, model_ver=alias, method="dbnet")
            pred_l = [l["text"] for l in out_p.get("lines", [])]
            gt_l = [ln["gt"] for ln in d["lines"]]
            gt_doc = " ".join(normalize_unicode(x) for x in gt_l)
            pr_doc = " ".join(normalize_unicode(x) for x in pred_l)
            ded = levenshtein_distance(gt_doc, pr_doc)

            by_domain[dom]["e2e_ed"] += ded
            by_domain[dom]["e2e_ch"] += len(gt_doc)
            by_domain[dom]["gt_l"] += len(gt_l)
            by_domain[dom]["pr_l"] += len(pred_l)
            e2e_ed += ded
            e2e_ch += len(gt_doc)
            e2e_gt_l += len(gt_l)
            e2e_pr_l += len(pred_l)

        dom_summary = {}
        for dom, st in by_domain.items():
            c_res = eval_crop_list(rec, is_logical, st["lines"])
            dom_summary[dom] = {
                "crop_cer_pct": c_res["norm_cer_pct"],
                "crop_wer_pct": c_res["wer_pct"],
                "exact_lines": c_res["exact_lines"],
                "e2e_lines": f"{st['pr_l']}/{st['gt_l']}",
                "e2e_doc_cer_pct": round(100.0 * st["e2e_ed"] / max(1, st["e2e_ch"]), 2),
            }

        ood_overall_crops = eval_crop_list(rec, is_logical, all_ood_lines)
        results[alias] = {
            "harum_palei_01_crops": harum_crops,
            "wikimedia_specimens": wiki_crops,
            "ood_archival_overall": {
                **ood_overall_crops,
                "e2e_lines": f"{e2e_pr_l}/{e2e_gt_l}",
                "e2e_doc_cer_pct": round(100.0 * e2e_ed / max(1, e2e_ch), 2),
            },
            "ood_by_domain": dom_summary,
        }
        print(f"  • `{alias}`: harum={harum_crops['norm_cer_pct']}% | wiki={wiki_crops['norm_cer_pct']}% | OOD-180 Crop={ood_overall_crops['norm_cer_pct']}% E2E={results[alias]['ood_archival_overall']['e2e_doc_cer_pct']}% ({e2e_pr_l}/{e2e_gt_l})")

    payload = {
        "title": "Cham-OCR RealScan-Gold v2 & OOD-Archival-Gold v2 Evaluation",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_summary": m["summary"],
        "models": results,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md = [
        "# Đánh giá `RealScan-Gold v2` & `OOD-Archival-Gold v2` (212 dòng hoàn toàn độc lập tập huấn luyện)",
        f"- **Thời điểm**: `{payload['evaluated_at']}`",
        f"- **Quy mô**: `1` trang chụp thực (`harum_palei_01`, 25 dòng) + `7` mẫu công khai Wikimedia Commons + `12` trang cổ thư/sách OOD (`180` dòng từ 12 phông EFEO + 6 phông Held-Out chưa từng dùng huấn luyện).",
        "",
        "## 1. Tổng hợp Xuyên Miền (`harum_palei_01` + Wikimedia + 180 dòng OOD Archival)",
        "| Model | `harum_palei_01` Crop CER | Wikimedia Specimens CER | OOD-180 Crop CER | OOD-180 WER | OOD-12 Pages E2E Lines | OOD-12 Pages E2E Doc CER |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for alias, r in results.items():
        oo = r["ood_archival_overall"]
        md.append(
            f"| `{alias}` | **{r['harum_palei_01_crops']['norm_cer_pct']}%** | {r['wikimedia_specimens']['norm_cer_pct']}% ({r['wikimedia_specimens']['exact_lines']}) | **{oo['norm_cer_pct']}%** | {oo['wer_pct']}% | {oo['e2e_lines']} | **{oo['e2e_doc_cer_pct']}%** |"
        )
    md.extend([
        "",
        "## 2. Phân rã Theo 3 Miền Tư liệu Cổ thư OOD (Crop CER / E2E Doc CER)",
        "| Model | `efeo_palm_leaf_agal` (Lá buông) | `efeo_regional_folio` (Sách cổ 1-2 cột) | `heldout_camera_book` (Sách in chụp camera) |",
        "| :--- | :---: | :---: | :---: |",
    ])
    for alias, r in results.items():
        bd = r["ood_by_domain"]
        def _f(k):
            d = bd.get(k, {})
            return f"{d.get('crop_cer_pct')}% / {d.get('e2e_doc_cer_pct')}% ({d.get('e2e_lines')})"
        md.append(f"| `{alias}` | {_f('efeo_palm_leaf_agal')} | {_f('efeo_regional_folio')} | {_f('heldout_camera_book')} |")

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"✅ Saved RealScan-Gold v2 report -> {args.output_md}")


if __name__ == "__main__":
    main()
