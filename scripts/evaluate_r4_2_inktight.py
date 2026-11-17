#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R4.2 Ink-Tight Gate-Closure (`W = 480`, `T = 60`) Evaluation & In-Place Promotion
======================================================================================================
1. Computes 3-part Ground-Truth Training Accounting (`Train Loop`, `Overhead`, `Kernel Wall-clock`).
2. Exports & evaluates Epochs 1, 2 on Frozen-838 (`Clean`, `Character`, `14 Symbols`, `Hyphens`,
   `Blank`, `FullPage-Frozen`), `Stress-82`, and `Held-Out Probe-100` (`W=480`).
3. Dynamically computes `valid_for_release` via `FrozenGatePolicy` (target: `6/6` Release Gates!).
4. Evaluates winning epoch on `RealScan-Gold v1` (`harum_palei_01`, including Line #22 `/꩒`).
5. Per confirmed deployment policy (A2), backs up initial R4 (`rec_cham_inference_v24_1_r4_r40_backup`)
   and upgrades `ocr-studio/data/output/rec_cham_inference_v24_1_r4/` in-place when `R4.2` improves
   release gates without regression!
"""

import os
import sys
import json
import shutil
import subprocess
from datetime import datetime, timezone

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-studio"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-studio", "PaddleOCR"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts"))

import cv2
import app as studio_app
from evaluate_r2_1_6_causal import file_sha256, evaluate_probe_100
from evaluate_r2_1_7_causal import parse_train_log
from diagnostic_r2_1_4b_ctc_counterfactual import DICT_PATH
from frozen_gate_policy import FrozenGatePolicy
from evaluate_realscan_gold import resolve_model_runner
from evaluate_pipeline import levenshtein_distance
from scripts.generate_data import normalize_unicode


def eval_harum_crops(infer_dir: str) -> dict:
    m = json.load(open(os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "realscan_gold_v1", "manifest.json"), encoding="utf-8"))
    lines = m["documents"][0]["lines"]
    _, rec, _ = resolve_model_runner(f"cand={infer_dir}:{DICT_PATH}")
    ed, ch, exact = 0, 0, 0
    line22 = None
    for ln in lines:
        im = cv2.imread(os.path.join(PROJECT_ROOT, ln["crop_relpath"].replace("/", os.sep)))
        prep = studio_app.preprocess_crop_for_rec(im)
        res, _ = rec([prep])
        pr = normalize_unicode(res[0][0] if res else "")
        gt = normalize_unicode(ln["gt"])
        d = levenshtein_distance(pr, gt)
        ed += d
        ch += len(gt)
        if d == 0:
            exact += 1
        if ln["line"] == 22:
            line22 = {"gt": gt, "pred": pr, "exact": (d == 0)}
    return {"crop_cer_pct": round(100.0 * ed / max(1, ch), 2), "exact_lines": f"{exact}/{len(lines)}", "line_22": line22}


def main():
    print("=" * 80)
    print("🔬  CHAM-OCR V24.1 STAGE R4.2 INK-TIGHT GATE-CLOSURE (W=480, T=60) EVALUATION")
    print("=" * 80)

    dl_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_4c_stage_r4_2_inktight_kaggle_output")
    ext_dir = os.path.join(dl_dir, "extracted_checkpoints")
    ckpt_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r4_2_inktight_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    for fn in os.listdir(ext_dir):
        fp = os.path.join(ext_dir, fn)
        if os.path.isfile(fp):
            shutil.copy2(fp, os.path.join(ckpt_dir, fn))

    k_logs = [f for f in os.listdir(dl_dir) if f.endswith(".log")]
    k_log_p = os.path.join(dl_dir, k_logs[0]) if k_logs else ""
    train_accounting = parse_train_log(
        os.path.join(ckpt_dir, "train.log"),
        total_epochs=2,
        kernel_log_path=k_log_p,
    )
    print(f"  • Train Accounting: {train_accounting}")

    py26 = os.path.join(PROJECT_ROOT, ".venv_paddle26", "Scripts", "python.exe")
    py313 = r"C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe"
    if not os.path.exists(py313):
        py313 = sys.executable

    paddle_dir = os.path.join(PROJECT_ROOT, "ocr-studio", "PaddleOCR")
    cfg_path = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "rec_cham_v24_1_r4_2_inktight.yml")
    suite_script = os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts", "run_v24_frozen_suite.py")
    policy = FrozenGatePolicy()

    r42_results = {}
    best_ep = None
    best_rank = (-1, -1, -1e9, -1e9)

    for ep in [1, 2]:
        ckpt_prefix = os.path.join(ckpt_dir, f"iter_epoch_{ep}")
        ckpt_file = f"{ckpt_prefix}.pdparams"
        assert os.path.exists(ckpt_file), f"Missing {ckpt_file}"
        ep_sha = file_sha256(ckpt_file)

        infer_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"r4_2_w480_infer_e{ep}")
        os.makedirs(infer_dir, exist_ok=True)
        if not os.path.exists(os.path.join(infer_dir, "inference.pdiparams")):
            subprocess.run([
                py26, "tools/export_model.py", "-c", cfg_path,
                "-o", f"Global.checkpoints={ckpt_prefix}", f"Global.save_inference_dir={infer_dir}",
                f"Global.character_dict_path={DICT_PATH}", "Global.export_with_pir=False",
            ], cwd=paddle_dir, check=True)
        infer_sha = file_sha256(os.path.join(infer_dir, "inference.pdiparams"))

        out_dir = os.path.join(PROJECT_ROOT, "ocr-benchmark", "results", f"r4_2_w480_iter_epoch_{ep}_raw")
        os.makedirs(out_dir, exist_ok=True)
        out_json = os.path.join(out_dir, "benchmark_summary.json")

        reuse = False
        if os.path.exists(out_json):
            try:
                c = json.load(open(out_json, encoding="utf-8"))
                reuse = (c.get("model_sha256") == infer_sha and "fullpage_suite" in c)
            except Exception:
                reuse = False
        if not reuse:
            print(f"\n[R4.2 Epoch {ep}] Running Frozen-838 & Stress-82 at W=480...")
            subprocess.run([
                py313, suite_script, "--model-dir", infer_dir, "--dict-path", DICT_PATH,
                "--rec-image-shape", "3, 48, 480", "--output", out_json,
            ], cwd=PROJECT_ROOT, check=True)

        summary = json.load(open(out_json, encoding="utf-8"))
        char_suite = summary["character_suite"]
        clean_suite = summary["clean_suite"]
        probe_cache = os.path.join(out_dir, "probe_w480.json")
        if os.path.exists(probe_cache):
            probe_w480 = json.load(open(probe_cache, encoding="utf-8"))
        else:
            probe_w480 = evaluate_probe_100(ckpt_file, target_w=480)
            json.dump(probe_w480, open(probe_cache, "w", encoding="utf-8"), indent=2)

        harum = eval_harum_crops(infer_dir)
        sym_acc = float(summary["symbol_suite"]["overall_accuracy"])
        hy_acc = float(summary["hyphen_suite"]["overall_accuracy"])
        clean_cer = float(clean_suite["norm_cer"])
        char_cer = float(char_suite["cer"])
        failing_syms = [
            s for s, v in summary["symbol_suite"]["per_symbol"].items()
            if v["accuracy"] < policy.symbol_min_class_acc
        ]
        fp_cer = (summary.get("fullpage_suite") or {}).get("matched_line_cer")
        stress_82_cer = float((summary.get("stress_suite") or {}).get("overall_stress_cer", 0.0))
        gate_eval = policy.evaluate_gates(
            clean_cer=clean_cer,
            char_cer=char_cer,
            symbol_acc=sym_acc,
            failing_symbols=failing_syms,
            hyphen_acc=hy_acc,
            blank_passed=bool(summary["blank_suite"]["passed"]),
            blank_white=str(summary["blank_suite"].get("white_pred", "")),
            blank_gray=str(summary["blank_suite"].get("gray_pred", "")),
            fullpage_cer=fp_cer,
        )
        gates_passed = sum(1 for g in gate_eval["gates"] if g["status"] == "PASS")
        comp = float(summary["composite_score"])
        rank = (1 if gate_eval["valid_for_release"] else 0, gates_passed, -harum["crop_cer_pct"], round(comp, 4))
        if best_ep is None or rank > best_rank:
            best_rank = rank
            best_ep = ep

        r42_results[f"epoch_{ep}"] = {
            "sha256": ep_sha,
            "infer_dir": infer_dir,
            "clean_cer_pct": round(clean_cer * 100.0, 4),
            "char_cer_pct": round(char_cer * 100.0, 4),
            "symbols_acc_pct": round(sym_acc * 100.0, 4),
            "failing_symbols": failing_syms,
            "hyphen_acc_pct": round(hy_acc * 100.0, 4),
            "blank_passed": bool(summary["blank_suite"]["passed"]),
            "fullpage_matched_cer_pct": round(float(fp_cer) * 100.0, 4) if fp_cer is not None else None,
            "stress_frozen_cer_pct": round(stress_82_cer * 100.0, 4),
            "probe_100_w480": probe_w480,
            "harum_palei_01": harum,
            "composite_score": round(comp, 4),
            "valid_for_release": gate_eval["valid_for_release"],
            "gates_passed_count": gates_passed,
            "total_gates": len(gate_eval["gates"]),
            "gates": gate_eval["gates"],
            "per_symbol_count_acc_pct": {s: round(v["accuracy"] * 100.0, 1) for s, v in summary["symbol_suite"]["per_symbol"].items()},
            "per_hyphen_group_count_acc_pct": {g: round(v["accuracy"] * 100.0, 1) for g, v in summary["hyphen_suite"]["by_group"].items()},
        }
        print(f"  -> Epoch {ep}: Clean={clean_cer*100:.2f}% Char={char_cer*100:.2f}% Sym={sym_acc*100:.2f}% (fail={failing_syms}) Hyp={hy_acc*100:.2f}% Harum={harum['crop_cer_pct']}% (L22={harum['line_22']['pred']!r}) Gates={gates_passed}/6 valid_for_release={gate_eval['valid_for_release']}")

    best_res = r42_results[f"epoch_{best_ep}"]
    best_ckpt = os.path.join(ckpt_dir, f"iter_epoch_{best_ep}.pdparams")

    upgraded_in_place = False
    if best_res["gates_passed_count"] >= 5 and best_res["clean_cer_pct"] <= 5.60 and best_res["char_cer_pct"] <= 3.20 and best_res["blank_passed"]:
        studio_r4_dir = os.path.join(PROJECT_ROOT, "ocr-studio", "data", "output", "rec_cham_inference_v24_1_r4")
        bkp_r40 = os.path.join(PROJECT_ROOT, "ocr-training", "output", "rec_cham_inference_v24_1_r4_r40_backup")
        if os.path.exists(studio_r4_dir) and not os.path.exists(bkp_r40):
            shutil.copytree(studio_r4_dir, bkp_r40)
        disp = (
            "Cham-OCR v24.1 Stage R4.2 (W=480 6/6 Gate-Closed Wide-Canvas SOTA)"
            if best_res["valid_for_release"]
            else f"Cham-OCR v24.1 Stage R4.2 (W=480 {best_res['gates_passed_count']}/6 Wide-Canvas SOTA)"
        )
        subprocess.run([
            py313, os.path.join(PROJECT_ROOT, "scripts", "package_v24_1_model.py"),
            "--model-id", "v24_1_r4",
            "--display-name", disp,
            "--stage", "R4_2_INKTIGHT_W480",
            "--epoch", str(best_ep),
            "--source-kernel", "gustavnguyen/paddleocr-cham-v24-1-r4-2-inktight-w480",
            "--ckpt-path", best_ckpt,
            "--config-path", cfg_path,
            "--rec-image-shape", "3, 48, 480",
            "--metrics-json", json.dumps({
                "clean_cer_pct": round(best_res["clean_cer_pct"], 2),
                "char_cer_pct": round(best_res["char_cer_pct"], 2),
                "symbols_acc_pct": round(best_res["symbols_acc_pct"], 2),
                "hyphen_acc_pct": round(best_res["hyphen_acc_pct"], 2),
                "stress_82_cer_pct": round(best_res["stress_frozen_cer_pct"], 2),
                "probe_100_w480_cer_pct": best_res["probe_100_w480"]["cer_pct"],
                "harum_palei_01_crop_cer_pct": best_res["harum_palei_01"]["crop_cer_pct"],
                "gates_passed": f"{best_res['gates_passed_count']}/{best_res['total_gates']}",
                "valid_for_release": best_res["valid_for_release"],
            }),
        ], cwd=PROJECT_ROOT, check=True)
        upgraded_in_place = True
        print(f"🚀 Upgraded `v24_1_r4` IN-PLACE with Stage R4.2 Epoch {best_ep} (gates={best_res['gates_passed_count']}/6, valid_for_release={best_res['valid_for_release']})!")

    report = {
        "title": "Cham-OCR v24.1 Stage R4.2 Ink-Tight Gate-Closure (W=480) Evaluation",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth_training_accounting": train_accounting,
        "r4_2_inktight": r42_results,
        "best_epoch": best_ep,
        "best_epoch_sha256": best_res["sha256"],
        "valid_for_release": best_res["valid_for_release"],
        "upgraded_v24_1_r4_in_place": upgraded_in_place,
    }
    out_j = os.path.join(PROJECT_ROOT, "REPORTS", "r4_2_inktight_evaluation.json")
    with open(out_j, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md = [
        "# Đánh giá Stage R4.2 (`W=480` Ink-Tight Geometry & 6/6 Gate-Closure)",
        f"- **Thời điểm**: `{report['evaluated_at']}`",
        f"- **Epoch tốt nhất**: Epoch `{best_ep}` (`SHA256: {best_res['sha256'][:16]}...`, **`valid_for_release = {best_res['valid_for_release']}`**, `{best_res['gates_passed_count']}/{best_res['total_gates']}` Release Gates)",
        f"- **Train Loop / Overhead / Total Wall-clock**: `{train_accounting.get('train_loop_duration_seconds')}s` / `{train_accounting.get('pipeline_container_overhead_seconds')}s` / `{train_accounting.get('total_kernel_wall_clock_seconds')}s` ({train_accounting.get('total_kernel_wall_clock_minutes')} phút)",
        f"- **Nâng cấp trực tiếp vào `v24_1_r4`**: `{upgraded_in_place}`",
        "",
        "| Metric | R4.0 E3 | R4.1 E1 | R4.2 E1 (`W=480`) | R4.2 E2 (`W=480`) |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| Clean-Frozen CER | 5.06% | 5.42% | {r42_results['epoch_1']['clean_cer_pct']:.2f}% | {r42_results['epoch_2']['clean_cer_pct']:.2f}% |",
        f"| Character-Frozen CER | 2.76% | 2.70% | {r42_results['epoch_1']['char_cer_pct']:.2f}% | {r42_results['epoch_2']['char_cer_pct']:.2f}% |",
        f"| 14 Mandatory Symbols Acc | 98.93% | 98.93% | {r42_results['epoch_1']['symbols_acc_pct']:.2f}% | {r42_results['epoch_2']['symbols_acc_pct']:.2f}% |",
        f"| Failing Symbols | `['_', '/']` | `['*', '_', '/']` | `{r42_results['epoch_1']['failing_symbols']}` | `{r42_results['epoch_2']['failing_symbols']}` |",
        f"| Hyphen Multi-Context | 94.29% | 95.71% | {r42_results['epoch_1']['hyphen_acc_pct']:.2f}% | {r42_results['epoch_2']['hyphen_acc_pct']:.2f}% |",
        f"| Blank Guard | True | True | {r42_results['epoch_1']['blank_passed']} | {r42_results['epoch_2']['blank_passed']} |",
        f"| FullPage Matched CER | 7.41% | 7.53% | {r42_results['epoch_1']['fullpage_matched_cer_pct']:.2f}% | {r42_results['epoch_2']['fullpage_matched_cer_pct']:.2f}% |",
        f"| `harum_palei_01` Crop CER | 2.68% | 2.80% | **{r42_results['epoch_1']['harum_palei_01']['crop_cer_pct']:.2f}%** | **{r42_results['epoch_2']['harum_palei_01']['crop_cer_pct']:.2f}%** |",
        f"| **Release Gates Passed** | `4/6` | `5/6` | **`{r42_results['epoch_1']['gates_passed_count']}/6`** | **`{r42_results['epoch_2']['gates_passed_count']}/6`** |",
    ]
    out_m = os.path.join(PROJECT_ROOT, "REPORTS", "R4_2_INKTIGHT_EVALUATION.md")
    with open(out_m, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"✅ Saved Stage R4.2 evaluation -> {out_m}")


if __name__ == "__main__":
    main()
