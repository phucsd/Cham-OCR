#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR RealScan-Gold v3 Unified Cross-Domain, Curved-Agal & Bilingual Evaluation Runner
=========================================================================================
Verifies 100% SHA-256 integrity of `ocr-benchmark/data/realscan_gold_v3/manifest.json` (`432` total items across
real camera `harum_palei_01`, 7 Wikimedia specimens, 76 multiscript handoff crops, 20 unseen pure-Cham OOD pages
covering 12/12 EFEO fonts + 6/6 held-out weights + curved palm-leaf agal, and 2 bilingual Cham-Vietnamese pages)
and benchmarks candidate models + `auto` router on accuracy AND per-page CPU latency (`< 15s` budget).
"""

import os
import sys
import json
import time
import unicodedata
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
from evaluate_realscan_gold import sha256_file, resolve_model_runner, matched_line_stats
from scripts.generate_data import normalize_unicode


def _norm_any(txt: str, is_logical: bool = True) -> str:
    s = unicodedata.normalize("NFC", (txt or "").strip())
    if is_logical and any("\uAA00" <= c <= "\uAA5F" for c in s):
        s = normalize_unicode(s)
    return s


def eval_crops(rec_model, is_logical: bool, lines: list) -> dict:
    ed, ch, w_ed, w_tot, exact = 0, 0, 0, 0, 0
    for ln in lines:
        p = os.path.join(PROJECT_ROOT, ln["crop_relpath"].replace("/", os.sep))
        im = cv2.imread(p)
        prep = studio_app.preprocess_crop_for_rec(im)
        res, _ = rec_model([prep])
        raw_p = res[0][0] if res else ""
        pr = _norm_any(raw_p, is_logical)
        gt = _norm_any(ln["gt"], True)
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


def eval_doc_group(alias: str, rec, is_logical: bool, docs: list) -> tuple[dict, list[float]]:
    all_lines = []
    e2e_ed, e2e_ch, gt_l_tot, pr_l_tot, match_tot = 0, 0, 0, 0, 0
    latencies = []
    for d in docs:
        all_lines.extend(d["lines"])
        p_img = cv2.imread(os.path.join(PROJECT_ROOT, d["page_relpath"].replace("/", os.sep)))
        t0 = time.time()
        out_p = studio_app.run_ocr_pipeline(p_img, model_ver=alias, method="dbnet")
        latencies.append(time.time() - t0)
        pred_l = [_norm_any(l["text"], is_logical) for l in out_p.get("lines", [])]
        gt_l = [_norm_any(ln["gt"], True) for ln in d["lines"]]
        gt_doc = " ".join(gt_l)
        pr_doc = " ".join(pred_l)
        e2e_ed += levenshtein_distance(gt_doc, pr_doc)
        e2e_ch += len(gt_doc)
        gt_l_tot += len(gt_l)
        pr_l_tot += len(pred_l)
        match_tot += matched_line_stats(gt_l, pred_l)["matched_lines"]
    c_res = eval_crops(rec, is_logical, all_lines)
    return {
        **c_res,
        "pages": len(docs),
        "e2e_lines": f"{pr_l_tot}/{gt_l_tot}",
        "line_recall_pct": round(100.0 * match_tot / max(1, gt_l_tot), 2),
        "e2e_doc_cer_pct": round(100.0 * e2e_ed / max(1, e2e_ch), 2),
    }, latencies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["v24", "v24_1_r3_1", "v24_1_r4", "auto"])
    ap.add_argument("--output-json", default=os.path.join(PROJECT_ROOT, "REPORTS", "realscan_gold_v3_evaluation.json"))
    ap.add_argument("--output-md", default=os.path.join(PROJECT_ROOT, "REPORTS", "REALSCAN_GOLD_V3_EVALUATION.md"))
    args = ap.parse_args()

    m_path = os.path.join(BENCHMARK_DIR, "data", "realscan_gold_v3", "manifest.json")
    m = json.load(open(m_path, "r", encoding="utf-8"))

    checked = 0
    for d in m["documents"]:
        if d.get("page_relpath"):
            pp = os.path.join(PROJECT_ROOT, d["page_relpath"].replace("/", os.sep))
            assert sha256_file(pp) == d["page_sha256"], f"SHA mismatch: {pp}"
            checked += 1
        items = list(d.get("lines", []))
        for grp in (d.get("groups") or {}).values():
            items.extend(grp)
        for ln in items:
            cp = os.path.join(PROJECT_ROOT, ln["crop_relpath"].replace("/", os.sep))
            assert sha256_file(cp) == ln["crop_sha256"], f"SHA mismatch: {cp}"
            checked += 1
    print(f"✅ RealScan-Gold v3 SHA-256 verified: {checked} files match 100%.")

    harum_doc = next(d for d in m["documents"] if d["doc_id"] == "harum_palei_01")
    wiki_doc = next(d for d in m["documents"] if d["doc_id"] == "wikimedia_public_specimens_01")
    handoff_doc = next(d for d in m["documents"] if d["doc_id"] == "handoff_multiscript_01")
    v2_docs = [d for d in m["documents"] if d["source_type"] == "ood_archival_gold_v2"]
    v3_pure_docs = [d for d in m["documents"] if d["source_type"] == "ood_archival_gold_v3"]
    v3_bi_docs = [d for d in m["documents"] if d["source_type"] == "bilingual_archival_gold_v3"]

    results = {}
    for spec in args.models:
        alias, rec, is_logical = resolve_model_runner(spec)
        print(f"\n[RealScan-Gold v3] Evaluating `{alias}`...")

        harum_c = eval_crops(rec, is_logical, harum_doc["lines"])
        h_img = cv2.imread(os.path.join(PROJECT_ROOT, harum_doc["page_relpath"].replace("/", os.sep)))
        t_h = time.time()
        h_out = studio_app.run_ocr_pipeline(h_img, model_ver=alias, method="dbnet")
        h_lat = time.time() - t_h
        h_fp = matched_line_stats([ln["gt"] for ln in harum_doc["lines"]], [l["text"] for l in h_out.get("lines", [])])

        wiki_c = eval_crops(rec, is_logical, wiki_doc["lines"])

        # Multiscript handoff 76 crops
        ms_summary = {}
        for gname, gitems in handoff_doc["groups"].items():
            cham_hits, lat_hits, confs = 0, 0, []
            for it in gitems:
                im = cv2.imread(os.path.join(PROJECT_ROOT, it["crop_relpath"].replace("/", os.sep)))
                res, _ = rec([studio_app.preprocess_crop_for_rec(im)])
                ptxt, pconf = res[0] if res else ("", 0.0)
                confs.append(float(pconf))
                if any("\uAA00" <= ch <= "\uAA5F" for ch in ptxt):
                    cham_hits += 1
                if any(("a" <= ch.lower() <= "z") or ("\u00C0" <= ch <= "\u1EF9") for ch in ptxt):
                    lat_hits += 1
            ms_summary[gname] = {
                "count": len(gitems),
                "mean_conf": round(float(np.mean(confs)), 4),
                "pct_has_cham": round(100.0 * cham_hits / max(1, len(gitems)), 1),
                "pct_has_latin_viet": round(100.0 * lat_hits / max(1, len(gitems)), 1),
            }

        v2_res, lat_v2 = eval_doc_group(alias, rec, is_logical, v2_docs)
        v3_pure_res, lat_v3p = eval_doc_group(alias, rec, is_logical, v3_pure_docs)
        bi_res, lat_bi = eval_doc_group(alias, rec, is_logical, v3_bi_docs)

        # Sub-domain breakdown across v3 new domains
        by_v3_dom = {}
        for dom in ("efeo_unseen_regional_v3", "efeo_curved_agal_v3", "heldout_ultralight_book_v3"):
            sub_d = [d for d in v3_pure_docs if d["archival_domain"] == dom]
            by_v3_dom[dom], _ = eval_doc_group(alias, rec, is_logical, sub_d)

        # Combined 20 pure-Cham OOD pages (`v2` 12 + `v3` 8)
        all_pure_crops = eval_crops(rec, is_logical, [ln for d in (v2_docs + v3_pure_docs) for ln in d["lines"]])
        all_lats = [h_lat] + lat_v2 + lat_v3p + lat_bi

        results[alias] = {
            "harum_palei_01": {**harum_c, "e2e_doc_cer_pct": h_fp["e2e_doc_cer_pct"], "e2e_lines": f"{h_fp['pred_lines']}/{h_fp['gt_lines']}"},
            "wikimedia_specimens": wiki_c,
            "handoff_multiscript_01": ms_summary,
            "ood_v2_12pages": v2_res,
            "ood_v3_new_8pages": v3_pure_res,
            "ood_all_20pages_pure_cham": {
                "pages": len(v2_docs) + len(v3_pure_docs),
                "n_lines": all_pure_crops["n_lines"],
                "crop_cer_pct": all_pure_crops["norm_cer_pct"],
                "crop_wer_pct": all_pure_crops["wer_pct"],
                "e2e_doc_cer_pct": round(
                    (v2_res["e2e_doc_cer_pct"] * v2_res["n_lines"] + v3_pure_res["e2e_doc_cer_pct"] * v3_pure_res["n_lines"])
                    / max(1, v2_res["n_lines"] + v3_pure_res["n_lines"]),
                    2,
                ),
            },
            "v3_by_domain": by_v3_dom,
            "bilingual_commentary_2pages": bi_res,
            "cpu_page_latency_sec": {
                "mean_sec": round(float(np.mean(all_lats)), 3),
                "max_sec": round(float(np.max(all_lats)), 3),
                "within_15s_budget": bool(np.max(all_lats) < 15.0),
            },
        }
        r = results[alias]
        print(
            f"  • `{alias}`: harum={r['harum_palei_01']['norm_cer_pct']}% (E2E {r['harum_palei_01']['e2e_doc_cer_pct']}%) | "
            f"OOD-v2={v2_res['norm_cer_pct']}%/{v2_res['e2e_doc_cer_pct']}% | "
            f"OOD-v3-new={v3_pure_res['norm_cer_pct']}%/{v3_pure_res['e2e_doc_cer_pct']}% | "
            f"OOD-20pg={r['ood_all_20pages_pure_cham']['crop_cer_pct']}%/{r['ood_all_20pages_pure_cham']['e2e_doc_cer_pct']}% | "
            f"Bilingual={bi_res['norm_cer_pct']}%/{bi_res['e2e_doc_cer_pct']}% | "
            f"CPU mean={r['cpu_page_latency_sec']['mean_sec']}s (max={r['cpu_page_latency_sec']['max_sec']}s)"
        )

    payload = {
        "title": "Cham-OCR RealScan-Gold v3 Unified Cross-Domain, Curved-Agal & Bilingual Evaluation",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_summary": m["summary"],
        "models": results,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    tot_items = m["summary"].get("total_evaluated_items", 434)
    pure_lines = sum(len(d["lines"]) for d in (v2_docs + v3_pure_docs))
    bi_lines = sum(len(d["lines"]) for d in v3_bi_docs)
    md = [
        f"# Đánh giá `RealScan-Gold v3` ({tot_items} Mục Kiểm Định Xuyên Miền, Lá Buông Cong & Song Ngữ Chăm–Việt)",
        f"- **Thời điểm**: `{payload['evaluated_at']}`",
        f"- **Quy mô**: `harum_palei_01` (`25` dòng chụp thực) + `7` mẫu Wikimedia Commons + `76` mẫu đa hệ chữ + `20` trang cổ thư/sách thuần Chăm OOD (`{pure_lines}` dòng phủ **12/12 phông EFEO** & **6/6 trọng lượng Held-Out** + lá buông cong) + `2` trang khảo dị song ngữ Chăm–Việt (`{bi_lines}` dòng).",
        "",
        "## 1. Tổng hợp Xuyên Miền & Tốc độ CPU (`< 15s` Budget)",
        "| Model | `harum_palei_01` Crop / E2E | Wikimedia | OOD-v2 (12pg) Crop / E2E | OOD-v3 Mới (8pg) Crop / E2E | Tổng OOD 20 Trang Crop / E2E | Song Ngữ Chăm–Việt Crop / E2E | CPU Mean / Max (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for alias, r in results.items():
        hp = r["harum_palei_01"]
        v2 = r["ood_v2_12pages"]
        v3p = r["ood_v3_new_8pages"]
        all20 = r["ood_all_20pages_pure_cham"]
        bi = r["bilingual_commentary_2pages"]
        lat = r["cpu_page_latency_sec"]
        md.append(
            f"| `{alias}` | **{hp['norm_cer_pct']}%** / {hp['e2e_doc_cer_pct']}% | {r['wikimedia_specimens']['norm_cer_pct']}% | "
            f"{v2['norm_cer_pct']}% / {v2['e2e_doc_cer_pct']}% | **{v3p['norm_cer_pct']}%** / **{v3p['e2e_doc_cer_pct']}%** | "
            f"**{all20['crop_cer_pct']}%** / **{all20['e2e_doc_cer_pct']}%** | "
            f"**{bi['norm_cer_pct']}%** / **{bi['e2e_doc_cer_pct']}%** ({bi['e2e_lines']}) | {lat['mean_sec']}s / {lat['max_sec']}s |"
        )
    md += [
        "",
        "## 2. Phân rã 4 Miền Mở Rộng Mới của `RealScan-Gold v3` (Crop CER / E2E Doc CER)",
        "| Model | `efeo_unseen_regional_v3` (`AiaTrang`/`Kraong`/`Pajai`) | `efeo_curved_agal_v3` (Lá buông cong) | `heldout_ultralight_book_v3` (`Thin`/`ExtraLight`/`Medium`) | `bilingual_cham_viet_commentary_v3` |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for alias, r in results.items():
        bd = r["v3_by_domain"]
        bi = r["bilingual_commentary_2pages"]
        def _fmt(d):
            return f"{d['norm_cer_pct']}% / {d['e2e_doc_cer_pct']}% ({d['e2e_lines']})"
        md.append(
            f"| `{alias}` | {_fmt(bd['efeo_unseen_regional_v3'])} | {_fmt(bd['efeo_curved_agal_v3'])} | "
            f"{_fmt(bd['heldout_ultralight_book_v3'])} | {_fmt(bi)} |"
        )

    with open(args.output_md, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print(f"✅ Saved RealScan-Gold v3 report -> {args.output_md}")


if __name__ == "__main__":
    main()
