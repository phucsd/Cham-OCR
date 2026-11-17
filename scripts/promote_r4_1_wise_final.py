#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R4.1 WiSE-FT (`0.17 * R4.0_E3 + 0.83 * R4.1_E1`, `W = 480, T = 60`)
Final Verification & In-Place Promotion to `ocr-studio/data/output/rec_cham_inference_v24_1_r4/`
"""

import os
import sys
import json
import shutil
import subprocess
from datetime import datetime, timezone

import paddle

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from evaluate_r4_2_inktight import eval_harum_crops
from evaluate_r2_1_6_causal import file_sha256, evaluate_probe_100
from frozen_gate_policy import FrozenGatePolicy


def main():
    sd40 = paddle.load(os.path.join(PROJECT_ROOT, "ocr-training/output/r4_w480_checkpoints/iter_epoch_3.pdparams"))
    sd41 = paddle.load(os.path.join(PROJECT_ROOT, "ocr-training/output/r4_1_polish_checkpoints/iter_epoch_1.pdparams"))
    wise = {k: 0.17 * sd40[k] + 0.83 * sd41[k] for k in sd40}

    w_dir = os.path.join(PROJECT_ROOT, "ocr-training/output/r4_1_wise83_ckpt")
    os.makedirs(w_dir, exist_ok=True)
    w_p = os.path.join(w_dir, "best.pdparams")
    paddle.save(wise, w_p)
    ckpt_sha = file_sha256(w_p)

    infer_dir = os.path.join(PROJECT_ROOT, "ocr-training/output/r4_1_wise83_w480_infer")
    if os.path.exists(infer_dir):
        shutil.rmtree(infer_dir)
    os.makedirs(infer_dir, exist_ok=True)

    py26 = os.path.join(PROJECT_ROOT, ".venv_paddle26/Scripts/python.exe")
    cfg_p = os.path.join(PROJECT_ROOT, "ocr-training/configs/rec_cham_v24_1_r4_1_polish.yml")
    dict_p = os.path.join(PROJECT_ROOT, "ocr-training/data/cham_dict_v24_1.txt")
    subprocess.run(
        [
            py26,
            "tools/export_model.py",
            "-c",
            cfg_p,
            "-o",
            f"Global.checkpoints={w_dir}/best",
            f"Global.save_inference_dir={infer_dir}",
            f"Global.character_dict_path={dict_p}",
            "Global.export_with_pir=False",
        ],
        cwd=os.path.join(PROJECT_ROOT, "ocr-studio/PaddleOCR"),
        check=True,
    )

    out_json = os.path.join(PROJECT_ROOT, "ocr-benchmark/results/r4_1_wise83_w480_raw/benchmark_summary.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            os.path.join(PROJECT_ROOT, "ocr-benchmark/scripts/run_v24_frozen_suite.py"),
            "--model-dir",
            infer_dir,
            "--dict-path",
            dict_p,
            "--rec-image-shape",
            "3, 48, 480",
            "--output",
            out_json,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    with open(out_json, "r", encoding="utf-8") as f:
        s = json.load(f)
    pol = FrozenGatePolicy()
    fsyms = [k for k, v in s["symbol_suite"]["per_symbol"].items() if v["accuracy"] < pol.symbol_min_class_acc]
    ge = pol.evaluate_gates(
        clean_cer=s["clean_suite"]["norm_cer"],
        char_cer=s["character_suite"]["cer"],
        symbol_acc=s["symbol_suite"]["overall_accuracy"],
        failing_symbols=fsyms,
        hyphen_acc=s["hyphen_suite"]["overall_accuracy"],
        blank_passed=bool(s["blank_suite"]["passed"]),
        blank_white=str(s["blank_suite"].get("white_pred", "")),
        blank_gray=str(s["blank_suite"].get("gray_pred", "")),
        fullpage_cer=s["fullpage_suite"]["matched_line_cer"],
    )
    gp = sum(1 for g in ge["gates"] if g["status"] == "PASS")
    pr = evaluate_probe_100(w_p, target_w=480)
    harum = eval_harum_crops(infer_dir)

    studio_r4 = os.path.join(PROJECT_ROOT, "ocr-studio/data/output/rec_cham_inference_v24_1_r4")
    bkp_r40 = os.path.join(PROJECT_ROOT, "ocr-training/output/rec_cham_inference_v24_1_r4_r40_backup")
    if os.path.exists(studio_r4) and not os.path.exists(bkp_r40):
        shutil.copytree(studio_r4, bkp_r40)

    sym_ok = sum(v["correct"] for v in s["symbol_suite"]["per_symbol"].values())
    metrics_payload = {
        "clean_cer_pct": round(s["clean_suite"]["norm_cer"] * 100.0, 2),
        "char_cer_pct": round(s["character_suite"]["cer"] * 100.0, 2),
        "symbols_acc_pct": round(s["symbol_suite"]["overall_accuracy"] * 100.0, 2),
        "symbols_passed": f"{sym_ok}/280",
        "failing_symbols": fsyms,
        "hyphen_acc_pct": round(s["hyphen_suite"]["overall_accuracy"] * 100.0, 2),
        "fullpage_cer_pct": round(s["fullpage_suite"]["matched_line_cer"] * 100.0, 2),
        "stress_82_cer_pct": round(s["stress_suite"]["overall_stress_cer"] * 100.0, 2),
        "probe_100_w480_cer_pct": pr["cer_pct"],
        "harum_crops_cer_pct": harum["crop_cer_pct"],
        "gates_passed": f"{gp}/6",
        "valid_for_release": ge["valid_for_release"],
    }

    subprocess.run(
        [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts/package_v24_1_model.py"),
            "--model-id",
            "v24_1_r4",
            "--display-name",
            f"Cham-OCR v24.1 Stage R4.1 WiSE-FT (W=480 {gp}/6 Gates Wide-Canvas SOTA)",
            "--stage",
            "R4_1_WISE_FT_W480",
            "--epoch",
            "1",
            "--source-kernel",
            "gustavnguyen/paddleocr-cham-v24-1-r4-1-gate-polish-w480",
            "--ckpt-path",
            w_p,
            "--config-path",
            cfg_p,
            "--rec-image-shape",
            "3, 48, 480",
            "--metrics-json",
            json.dumps(metrics_payload),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    new_pdiparams_sha = file_sha256(os.path.join(studio_r4, "inference.pdiparams"))
    rep = {
        "title": "Cham-OCR v24.1 Stage R4.1 WiSE-FT (0.17*R4.0_E3 + 0.83*R4.1_E1) Final Promotion",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": ckpt_sha,
        "packaged_pdiparams_sha256": new_pdiparams_sha,
        "packaged_pdiparams_sha256_16": new_pdiparams_sha[:16],
        "metrics": metrics_payload,
        "gates": ge["gates"],
        "valid_for_release": ge["valid_for_release"],
    }
    out_rep = os.path.join(PROJECT_ROOT, "REPORTS/r4_1_wise_final_promotion.json")
    with open(out_rep, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"✅ FINAL PROMOTED v24_1_r4 (SHA16={new_pdiparams_sha[:16]}):", metrics_payload)


if __name__ == "__main__":
    main()
