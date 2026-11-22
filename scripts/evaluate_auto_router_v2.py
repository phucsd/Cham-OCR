#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR Priority 3 — Intelligent Tri-Specialist `auto` Router v2 Evaluation
===========================================================================
Benchmarks `v24_1_r4` (`W=480`), `v24_1_r3_1` (`W=320`), `auto_v1_legacy`, and `auto` (Tri-Specialist Router v2)
head-to-head across:
  1. `Frozen-838` Release Gates (`Clean`, `Char`, `14 Symbols`, `Hyphen`, `Stress-82`, `Blank`)
  2. `RealScan-Gold v1` (`harum_palei_01` + `handoff_multiscript_01` 76 crops)
  3. `RealScan-Gold v2` (`Wikimedia` + `OOD-180` crops + `OOD-12` E2E Doc CER)
  4. `RealScan-Gold v3` (`20` unseen OOD pages + `2` bilingual Cham-Viet commentary pages + CPU latency `< 15s`)
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime, timezone

import cv2
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIO_DIR = os.path.join(PROJECT_ROOT, "ocr-studio")
for _p in (os.path.join(PROJECT_ROOT, "scripts"), STUDIO_DIR, os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import app as studio_app
from frozen_gate_policy import FrozenGatePolicy
from run_v24_frozen_suite import (
    RecognizerEvaluator,
    run_suite_blank,
    run_suite_clean,
    run_suite_character,
    run_suite_stress,
    run_suite_symbols,
    run_suite_hyphen,
    run_suite_fullpage,
)
from diagnostic_r2_1_4b_ctc_counterfactual import DICT_PATH
from evaluate_v26_masked_ft import PY313, STUDIO_R4


class LegacyAutoRouterV1:
    """Exact pre-upgrade AutoRoutingOCRWrapper (`r4` + `vi` at raw `0.65` threshold) for causal ablation."""
    def __init__(self, cham_model, viet_model):
        self.cham_model = cham_model
        self.viet_model = viet_model

    def __call__(self, img_list):
        if not img_list:
            return [], 0.0
        cham_res, _ = self.cham_model(img_list)
        viet_imgs = [img_list[i] for i, (_, c) in enumerate(cham_res) if c < 0.65]
        viet_res = self.viet_model(viet_imgs)[0] if viet_imgs else []
        out, vp = [], 0
        for t_c, c_c in cham_res:
            if c_c >= 0.65:
                out.append((t_c, c_c))
            else:
                t_v, c_v = viet_res[vp] if vp < len(viet_res) else ("", 0.0)
                vp += 1
                out.append((t_c, c_c) if (c_c >= 0.40 and c_c > c_v - 0.15) else (t_v, c_v))
        return out, 0.0


def score_frozen_suite_for_runner(base_ev: RecognizerEvaluator, rec_runner, policy: FrozenGatePolicy, cached_fp_cer: float = None) -> dict:
    base_ev.app_model = rec_runner
    blanks = run_suite_blank(base_ev)
    clean = run_suite_clean(base_ev)
    char = run_suite_character(base_ev)
    stress = run_suite_stress(base_ev)
    syms = run_suite_symbols(base_ev)
    hyps = run_suite_hyphen(base_ev)
    fp_cer = cached_fp_cer if cached_fp_cer is not None else run_suite_fullpage(base_ev)["matched_line_cer"]
    ge = policy.evaluate_gates(
        clean_cer=clean["norm_cer"],
        char_cer=char["cer"],
        symbol_acc=syms["overall_accuracy"],
        failing_symbols=syms["failing_symbols"],
        hyphen_acc=hyps["overall_accuracy"],
        blank_passed=bool(blanks["passed"]),
        blank_white=str(blanks["blank_white"]["pred"]),
        blank_gray=str(blanks["blank_gray"]["pred"]),
        fullpage_cer=fp_cer,
    )
    gates = sum(1 for g in ge["gates"] if g["status"] == "PASS")
    return {
        "clean_cer_pct": round(float(clean["norm_cer"]) * 100.0, 2),
        "char_cer_pct": round(float(char["cer"]) * 100.0, 2),
        "symbols_acc_pct": round(float(syms["overall_accuracy"]) * 100.0, 2),
        "failing_symbols": syms["failing_symbols"],
        "hyphen_acc_pct": round(float(hyps["overall_accuracy"]) * 100.0, 2),
        "fullpage_cer_pct": round(float(fp_cer) * 100.0, 2),
        "blank_passed": bool(blanks["passed"]),
        "stress_82_cer_pct": round(float(stress["overall_stress_cer"]) * 100.0, 2),
        "gates_passed": gates,
        "valid_for_release": bool(ge["valid_for_release"]),
        "_raw_fp_cer": float(fp_cer),
    }


def main():
    policy = FrozenGatePolicy()
    r4 = studio_app.get_ocr_model("v24_1_r4")
    r31 = studio_app.get_ocr_model("v24_1_r3_1")
    vi = studio_app.get_ocr_model("vi")
    auto_v2 = studio_app.get_ocr_model("auto")
    auto_v1 = LegacyAutoRouterV1(r4, vi)
    studio_app.ocr_models["auto_v1_legacy"] = auto_v1

    base_ev = RecognizerEvaluator(model_dir=STUDIO_R4, dict_path=DICT_PATH, verify_baseline=False, rec_image_shape="3, 48, 480")
    print("🔬 [1/3] Scoring Frozen-838 & Stress-82 across `v24_1_r4`, `v24_1_r3_1`, `auto_v1_legacy`, `auto` (v2)...")
    frozen = {}
    r4_fp = None
    for name, runner in [("v24_1_r4", r4), ("v24_1_r3_1", r31), ("auto_v1_legacy", auto_v1), ("auto", auto_v2)]:
        t0 = time.time()
        use_cached_fp = r4_fp if name in ("v24_1_r3_1", "auto_v1_legacy") else None
        frozen[name] = score_frozen_suite_for_runner(base_ev, runner, policy, cached_fp_cer=use_cached_fp)
        if name == "v24_1_r4":
            r4_fp = frozen[name]["_raw_fp_cer"]
        frozen[name]["seconds"] = round(time.time() - t0, 2)
        print(f"  • `{name}`: Clean={frozen[name]['clean_cer_pct']}% Char={frozen[name]['char_cer_pct']}% "
              f"Sym={frozen[name]['symbols_acc_pct']}% Hyp={frozen[name]['hyphen_acc_pct']}% "
              f"Stress={frozen[name]['stress_82_cer_pct']}% Gates={frozen[name]['gates_passed']}/6 ({frozen[name]['seconds']}s)")

    print("🔬 [2/3] Running `RealScan-Gold v3` unified evaluation (`v24`, `v24_1_r3_1`, `v24_1_r4`, `auto`)...")
    rs3_j = os.path.join(PROJECT_ROOT, "REPORTS", "realscan_gold_v3_evaluation.json")
    rs3_m = os.path.join(PROJECT_ROOT, "REPORTS", "REALSCAN_GOLD_V3_EVALUATION.md")
    subprocess.run([
        PY313, os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts", "evaluate_realscan_gold_v3.py"),
        "--models", "v24", "v24_1_r3_1", "v24_1_r4", "auto",
        "--output-json", rs3_j, "--output-md", rs3_m,
    ], cwd=PROJECT_ROOT, check=True)
    rs3 = json.load(open(rs3_j, "r", encoding="utf-8"))["models"]

    # Also score `auto_v1_legacy` in-process on OOD-v2 12 pages + Bilingual 2 pages for direct causal comparison!
    from evaluate_realscan_gold_v3 import eval_doc_group
    m3 = json.load(open(os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "realscan_gold_v3", "manifest.json"), "r", encoding="utf-8"))
    v2_docs = [d for d in m3["documents"] if d["source_type"] == "ood_archival_gold_v2"]
    bi_docs = [d for d in m3["documents"] if d["source_type"] == "bilingual_archival_gold_v3"]
    v1_ood12, _ = eval_doc_group("auto_v1_legacy", auto_v1, False, v2_docs)
    v1_bi, _ = eval_doc_group("auto_v1_legacy", auto_v1, False, bi_docs)

    report = {
        "title": "Cham-OCR Priority 3 — Tri-Specialist Auto Router v2 (`W=480` + `W=320` + `Vi`) Evaluation",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "frozen_suite": frozen,
        "legacy_auto_v1_ablation": {"ood_v2_12pages": v1_ood12, "bilingual_2pages": v1_bi},
        "realscan_gold_v3": rs3,
    }
    out_j = os.path.join(PROJECT_ROOT, "REPORTS", "auto_router_v2_evaluation.json")
    with open(out_j, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md = [
        "# Đánh giá Bộ Định Tuyến Đa Chuyên Gia `auto` Router v2 (`W=480` + `W=320` + `PP-OCRv5-vi`)",
        f"- **Thời điểm**: `{report['evaluated_at']}`",
        "",
        "## 1. Đối chuẩn `Frozen-838` & `Stress-82`",
        "| Model | Clean CER | Char CER | 14 Symbols | Hyphen | Stress-82 | Blank | Gates |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k, r in frozen.items():
        md.append(f"| `{k}` | {r['clean_cer_pct']}% | {r['char_cer_pct']}% | {r['symbols_acc_pct']}% | {r['hyphen_acc_pct']}% | {r['stress_82_cer_pct']}% | `{r['blank_passed']}` | **{r['gates_passed']}/6** |")

    md += [
        "",
        "## 2. Đối chuẩn Xuyên Miền (`RealScan-Gold v2/v3` + Song Ngữ Chăm–Việt + Tốc độ CPU `< 15s`)",
        "| Model | `harum_palei_01` Crop / E2E | OOD-v2 (12pg) Crop / E2E | Tổng OOD 20 Trang Crop / E2E | Song Ngữ Chăm–Việt Crop / E2E | Multiscript Latin / Vi-En % | CPU Mean / Max (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| `auto_v1_legacy` (Trước nâng cấp) | — | {v1_ood12['norm_cer_pct']}% / **{v1_ood12['e2e_doc_cer_pct']}%** | — | {v1_bi['norm_cer_pct']}% / {v1_bi['e2e_doc_cer_pct']}% | — | — |",
    ]
    for k in ("v24_1_r4", "v24_1_r3_1", "auto"):
        r = rs3[k]
        hp = r["harum_palei_01"]
        v2 = r["ood_v2_12pages"]
        a20 = r["ood_all_20pages_pure_cham"]
        bi = r["bilingual_commentary_2pages"]
        ms = r["handoff_multiscript_01"]
        lat = r["cpu_page_latency_sec"]
        md.append(
            f"| `{k}` | **{hp['norm_cer_pct']}%** / {hp['e2e_doc_cer_pct']}% | **{v2['norm_cer_pct']}%** / **{v2['e2e_doc_cer_pct']}%** | "
            f"**{a20['crop_cer_pct']}%** / **{a20['e2e_doc_cer_pct']}%** | **{bi['norm_cer_pct']}%** / **{bi['e2e_doc_cer_pct']}%** | "
            f"{ms['latin']['pct_has_latin_viet']}% / {ms['vietnamese_english']['pct_has_latin_viet']}% | {lat['mean_sec']}s / {lat['max_sec']}s |"
        )

    out_m = os.path.join(PROJECT_ROOT, "REPORTS", "AUTO_ROUTER_V2_EVALUATION.md")
    with open(out_m, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print(f"✅ Auto Router v2 report saved -> {out_m}")


if __name__ == "__main__":
    main()
