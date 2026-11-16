#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R3 Scan Degradation Causal & Level-5 Evaluation Runner
===========================================================================
1. Copies & SHA256-verifies extracted iter_epoch_1..3.pdparams & train.log from Kaggle.
2. Computes 3-part Ground-Truth Training Accounting (Train Loop, Overhead, Kernel Wall-clock).
3. Exports E1..E3 with PY26 (export_with_pir=False).
4. Runs Frozen-838 + Blank + FullPage + Stress-82 (via PY313 for Cham-DBNet PIR support)
   + Held-Out Probe-100 (via PY26) on R3 E1, E2, E3.
5. Evaluates Level-5 Full-Page Degradation (short:5, a4:5, audit_18) comparing v24,
   v24_1_r2_1_6, R2.1.7-best, and R3-best.
6. Emits REPORTS/r3_degradation_evaluation.json & REPORTS/R3_DEGRADATION_EVALUATION.md.
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-studio", "PaddleOCR"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts"))

from evaluate_r2_1_6_causal import file_sha256, evaluate_probe_100
from evaluate_r2_1_7_causal import parse_train_log
from analyze_causal_breakdown import analyze_ops
from diagnostic_r2_1_4b_ctc_counterfactual import DICT_PATH
from frozen_gate_policy import FrozenGatePolicy


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage R3 checkpoints & Level-5 full pages.")
    parser.add_argument("--r217-best-dir", default=None, help="Optional inference dir of R2.1.7-best for Level-5 comparison")
    args = parser.parse_args()

    print("=" * 80)
    print("🔬  CHAM-OCR V24.1 STAGE R3 SCAN DEGRADATION EVALUATION (EPOCHS 1, 2, 3)")
    print("=" * 80)

    dl_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_4c_stage_r3_degradation_kaggle_output")
    dl_extract_dir = os.path.join(dl_dir, "extracted_checkpoints")
    ckpt_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r3_degradation_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    assert os.path.exists(dl_extract_dir), f"Missing extracted R3 checkpoints: {dl_extract_dir}"
    for f in os.listdir(dl_extract_dir):
        src = os.path.join(dl_extract_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(ckpt_dir, f))

    train_log = os.path.join(ckpt_dir, "train.log")
    kernel_logs = [
        os.path.join(dl_dir, fn)
        for fn in ("paddleocr-cham-v24-1-r3-scan-degradation.log", "paddleocr-cham-v24-1-r3-degradation.log")
        if os.path.exists(os.path.join(dl_dir, fn))
    ]
    kernel_log = kernel_logs[0] if kernel_logs else os.path.join(dl_dir, "paddleocr-cham-v24-1-r3-scan-degradation.log")
    train_accounting = parse_train_log(train_log, total_epochs=3, kernel_log_path=kernel_log)
    print(f"  • Train Loop Accounting: {train_accounting}")

    py_exe = os.path.join(PROJECT_ROOT, ".venv_paddle26", "Scripts", "python.exe")
    if not os.path.exists(py_exe):
        py_exe = sys.executable

    py3_candidates = [
        r"C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe",
        shutil.which("python"),
    ]
    suite_py_exe = next((p for p in py3_candidates if p and os.path.exists(p)), py_exe)

    paddle_dir = os.path.join(PROJECT_ROOT, "ocr-studio", "PaddleOCR")
    cfg_path = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "rec_cham_v24_1_r3_degradation.yml")
    suite_script = os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts", "run_v24_frozen_suite.py")
    policy = FrozenGatePolicy()
    comb_marks = set([chr(c) for c in range(0xAA29, 0xAA37)] + [chr(c) for c in range(0xAA43, 0xAA4D + 1) if c in (0xAA43, 0xAA4C, 0xAA4D)])

    r3_results = {}
    best_ep = None
    best_rank = (-1, -1, -1e9, -1e9)

    for ep in [1, 2, 3]:
        ckpt_prefix = os.path.join(ckpt_dir, f"iter_epoch_{ep}")
        ckpt_file = f"{ckpt_prefix}.pdparams"
        assert os.path.exists(ckpt_file), f"Missing {ckpt_file}"
        ep_sha = file_sha256(ckpt_file)
        infer_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"r3_degradation_infer_e{ep}")
        os.makedirs(infer_dir, exist_ok=True)

        if not os.path.exists(os.path.join(infer_dir, "inference.pdiparams")):
            print(f"\n[R3 Epoch {ep}] Exporting inference model -> {infer_dir}...")
            exp_cmd = [
                py_exe, "tools/export_model.py",
                "-c", cfg_path,
                "-o",
                f"Global.checkpoints={ckpt_prefix}",
                f"Global.save_inference_dir={infer_dir}",
                f"Global.character_dict_path={DICT_PATH}",
                "Global.export_with_pir=False",
            ]
            subprocess.run(exp_cmd, cwd=paddle_dir, check=True)
        infer_sha = file_sha256(os.path.join(infer_dir, "inference.pdiparams"))

        out_dir = os.path.join(PROJECT_ROOT, "ocr-benchmark", "results", f"r3_degradation_iter_epoch_{ep}_raw")
        os.makedirs(out_dir, exist_ok=True)
        out_json = os.path.join(out_dir, "benchmark_summary.json")

        reuse_ok = False
        if os.path.exists(out_json):
            try:
                cached = json.load(open(out_json, encoding="utf-8"))
                if cached.get("model_sha256") == infer_sha and "fullpage_suite" in cached:
                    reuse_ok = True
            except Exception:
                reuse_ok = False

        if not reuse_ok:
            print(f"[R3 Epoch {ep}] Running Frozen-838 & Blank/Stress-8 Benchmark Suite -> {out_json}...")
            eval_cmd = [
                suite_py_exe, suite_script,
                "--model-dir", infer_dir,
                "--dict-path", DICT_PATH,
                "--output", out_json,
            ]
            subprocess.run(eval_cmd, cwd=PROJECT_ROOT, check=True)

        summary = json.load(open(out_json, encoding="utf-8"))
        char_suite = summary["character_suite"]
        clean_suite = summary["clean_suite"]
        recs = char_suite.get("all_records", char_suite.get("records", []))
        dels, subs, ins, del_chars, del_cats, sub_cats, ins_cats = analyze_ops(recs)
        assert dels + subs + ins == char_suite["edits"], "DP edit count mismatch!"

        comb_final_del = sum(v for k, v in del_chars.items() if k in comb_marks)
        probe_res = evaluate_probe_100(ckpt_file)

        sym_acc = float(summary["symbol_suite"]["overall_accuracy"])
        hy_acc = float(summary["hyphen_suite"]["overall_accuracy"])
        clean_cer = float(clean_suite["norm_cer"])
        char_cer = float(char_suite["cer"])
        failing_syms = [
            s for s, v in summary["symbol_suite"]["per_symbol"].items()
            if v["accuracy"] < policy.symbol_min_class_acc
        ]
        fp_suite = summary.get("fullpage_suite") or {}
        fp_cer = fp_suite.get("matched_line_cer")
        stress_82_cer = float((summary.get("stress_suite") or {}).get("overall_stress_cer", 0.0))
        stress_base_8_cer = float((summary.get("stress_base_8_suite") or {}).get("cer", 0.0))
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
        gates_passed_count = sum(1 for g in gate_eval["gates"] if g["status"] == "PASS")
        comp = float(summary["composite_score"])
        rank = (1 if gate_eval["valid_for_release"] else 0, gates_passed_count, -float(probe_res["cer_pct"]), comp)
        if best_ep is None or rank > best_rank:
            best_rank = rank
            best_ep = ep

        r3_results[f"epoch_{ep}"] = {
            "sha256": ep_sha,
            "infer_dir": infer_dir,
            "clean_cer_pct": round(clean_cer * 100.0, 4),
            "clean_raw_cer_pct": round(clean_suite["raw_cer"] * 100.0, 4),
            "char_cer_pct": round(char_cer * 100.0, 4),
            "char_edits": char_suite["edits"],
            "char_deletions": dels,
            "char_substitutions": subs,
            "char_insertions": ins,
            "combmark_final_deletions": comb_final_del,
            "del_categories": del_cats,
            "sub_categories": sub_cats,
            "ins_categories": ins_cats,
            "symbols_acc_pct": round(sym_acc * 100.0, 4),
            "failing_symbols": failing_syms,
            "hyphen_acc_pct": round(hy_acc * 100.0, 4),
            "blank_passed": bool(summary["blank_suite"]["passed"]),
            "fullpage_matched_cer_pct": round(float(fp_cer) * 100.0, 4) if fp_cer is not None else None,
            "stress_frozen_cer_pct": round(stress_82_cer * 100.0, 4),
            "stress_base_8_cer_pct": round(stress_base_8_cer * 100.0, 4),
            "composite_score": round(comp, 4),
            "valid_for_release": gate_eval["valid_for_release"],
            "gates_passed_count": gates_passed_count,
            "total_gates": len(gate_eval["gates"]),
            "gates": gate_eval["gates"],
            "per_symbol_count_acc_pct": {s: round(v["accuracy"] * 100.0, 1) for s, v in summary["symbol_suite"]["per_symbol"].items()},
            "per_hyphen_group_count_acc_pct": {g: round(v["accuracy"] * 100.0, 1) for g, v in summary["hyphen_suite"]["by_group"].items()},
            "held_out_probe_100_w320": probe_res,
        }

    # Level-5 Full-Page Comparison (via PY313 for Hybrid Cham-DBNet)
    best_r3_infer = r3_results[f"epoch_{best_ep}"]["infer_dir"]
    r2_1_7_eval_p = os.path.join(PROJECT_ROOT, "REPORTS", "r2_1_7_causal_evaluation.json")
    r2_1_7_eval = json.load(open(r2_1_7_eval_p, encoding="utf-8")) if os.path.exists(r2_1_7_eval_p) else None
    r217_dir = args.r217_best_dir
    if not r217_dir and r2_1_7_eval:
        b_ep = r2_1_7_eval["best_epoch"]
        r217_dir = r2_1_7_eval["r2_1_7_gate_closure"][f"epoch_{b_ep}"]["infer_dir"]

    lvl5_models = ["v24", "v24_1_r2_1_6"]
    if r217_dir and os.path.exists(r217_dir):
        lvl5_models.append(f"r2_1_7_best={r217_dir}:{DICT_PATH}")
    lvl5_models.append(f"r3_best={best_r3_infer}:{DICT_PATH}")

    lvl5_out_json = os.path.join(PROJECT_ROOT, "REPORTS", "r3_level5_fullpage_comparison.json")
    lvl5_script = os.path.join(PROJECT_ROOT, "scripts", "evaluate_level5_pages.py")
    print(f"\n[Level-5] Evaluating Full-Page short:5, a4:5 & audit_18 across {len(lvl5_models)} models...")
    subprocess.run([suite_py_exe, lvl5_script, "--models", *lvl5_models, "--limit", "90", "--output", lvl5_out_json], check=True)
    lvl5_data = json.load(open(lvl5_out_json, encoding="utf-8"))

    best_res = r3_results[f"epoch_{best_ep}"]
    report = {
        "title": "Cham-OCR v24.1 Stage R3 Scan Degradation Evaluation",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "r3_degradation": r3_results,
        "best_epoch": best_ep,
        "best_epoch_checkpoint": os.path.join(ckpt_dir, f"iter_epoch_{best_ep}.pdparams"),
        "best_epoch_infer_dir": best_r3_infer,
        "best_epoch_sha256": best_res["sha256"],
        "valid_for_release": best_res["valid_for_release"],
        "ground_truth_training_accounting": train_accounting,
        "level5_fullpage_comparison": lvl5_data,
    }
    out_json_path = os.path.join(PROJECT_ROOT, "REPORTS", "r3_degradation_evaluation.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    r216_ref = json.load(open(os.path.join(PROJECT_ROOT, "REPORTS", "r2_1_6_causal_evaluation.json"), encoding="utf-8"))["r2_1_6_surgical_polish"]["epoch_2"]
    r217_ref = r2_1_7_eval["r2_1_7_gate_closure"][f"epoch_{r2_1_7_eval['best_epoch']}"] if r2_1_7_eval else None

    md = [
        "# Đánh giá Stage R3 (`SCAN-DEGRADATION`)",
        f"- **Thời điểm**: `{report['evaluated_at']}`",
        f"- **Epoch tốt nhất R3**: Epoch {best_ep} (`SHA256: {best_res['sha256'][:16]}...`, `valid_for_release={best_res['valid_for_release']}`)",
        f"- **Train Loop Duration**: `{train_accounting.get('train_loop_duration_seconds')}s` ({train_accounting.get('train_loop_duration_minutes')} phút)",
        f"- **Pipeline / Container Overhead**: `{train_accounting.get('pipeline_container_overhead_seconds')}s` ({train_accounting.get('pipeline_container_overhead_minutes')} phút)",
        f"- **Total Kernel Wall-clock Time**: `{train_accounting.get('total_kernel_wall_clock_seconds')}s` ({train_accounting.get('total_kernel_wall_clock_minutes')} phút)",
        "",
        "## 1. Frozen-838 & Stress-82 & Probe-100",
        "| Metric | R2.1.6 E2 | R2.1.7 Best | R3 E1 | R3 E2 | R3 E3 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    def _v(d, k, fmt="{:.2f}%"):
        return fmt.format(d[k]) if d and d.get(k) is not None else "N/A"
    md.extend([
        f"| Clean-Frozen CER | {r216_ref['clean_cer_pct']:.2f}% | {_v(r217_ref, 'clean_cer_pct')} | {r3_results['epoch_1']['clean_cer_pct']:.2f}% | {r3_results['epoch_2']['clean_cer_pct']:.2f}% | {r3_results['epoch_3']['clean_cer_pct']:.2f}% |",
        f"| Character-Frozen CER | {r216_ref['char_cer_pct']:.2f}% | {_v(r217_ref, 'char_cer_pct')} | {r3_results['epoch_1']['char_cer_pct']:.2f}% | {r3_results['epoch_2']['char_cer_pct']:.2f}% | {r3_results['epoch_3']['char_cer_pct']:.2f}% |",
        f"| 14 Symbols Acc | {r216_ref['symbols_acc_pct']:.2f}% | {_v(r217_ref, 'symbols_acc_pct')} | {r3_results['epoch_1']['symbols_acc_pct']:.2f}% | {r3_results['epoch_2']['symbols_acc_pct']:.2f}% | {r3_results['epoch_3']['symbols_acc_pct']:.2f}% |",
        f"| Hyphen Multi-Context | {r216_ref['hyphen_acc_pct']:.2f}% | {_v(r217_ref, 'hyphen_acc_pct')} | {r3_results['epoch_1']['hyphen_acc_pct']:.2f}% | {r3_results['epoch_2']['hyphen_acc_pct']:.2f}% | {r3_results['epoch_3']['hyphen_acc_pct']:.2f}% |",
        f"| Blank Guard | {r216_ref['blank_passed']} | {r217_ref['blank_passed'] if r217_ref else 'N/A'} | {r3_results['epoch_1']['blank_passed']} | {r3_results['epoch_2']['blank_passed']} | {r3_results['epoch_3']['blank_passed']} |",
        f"| FullPage Matched CER | 6.21% | {_v(r217_ref, 'fullpage_matched_cer_pct')} | {r3_results['epoch_1']['fullpage_matched_cer_pct']:.2f}% | {r3_results['epoch_2']['fullpage_matched_cer_pct']:.2f}% | {r3_results['epoch_3']['fullpage_matched_cer_pct']:.2f}% |",
        f"| Stress-82 CER | 16.89% | {_v(r217_ref, 'stress_frozen_cer_pct')} | {r3_results['epoch_1']['stress_frozen_cer_pct']:.2f}% | {r3_results['epoch_2']['stress_frozen_cer_pct']:.2f}% | {r3_results['epoch_3']['stress_frozen_cer_pct']:.2f}% |",
        f"| Stress-Base-8 CER | 5.13% | {_v(r217_ref, 'stress_base_8_cer_pct')} | {r3_results['epoch_1']['stress_base_8_cer_pct']:.2f}% | {r3_results['epoch_2']['stress_base_8_cer_pct']:.2f}% | {r3_results['epoch_3']['stress_base_8_cer_pct']:.2f}% |",
        f"| Probe-100 CER | {r216_ref['held_out_probe_100_w320']['cer_pct']:.2f}% | {r217_ref['held_out_probe_100_w320']['cer_pct']:.2f}% | {r3_results['epoch_1']['held_out_probe_100_w320']['cer_pct']:.2f}% | {r3_results['epoch_2']['held_out_probe_100_w320']['cer_pct']:.2f}% | {r3_results['epoch_3']['held_out_probe_100_w320']['cer_pct']:.2f}% |",
        f"| Gates Passed | 4/6 | {r217_ref['gates_passed_count']}/{r217_ref['total_gates']} | {r3_results['epoch_1']['gates_passed_count']}/{r3_results['epoch_1']['total_gates']} | {r3_results['epoch_2']['gates_passed_count']}/{r3_results['epoch_2']['total_gates']} | {r3_results['epoch_3']['gates_passed_count']}/{r3_results['epoch_3']['total_gates']} |",
        "",
        "## 2. Full-Page Level-5 Degradation & 18-Image Audit Suite",
        "| Model | short:5 Doc/Matched CER | a4:5 Doc/Matched CER | audit_18 Doc/Matched CER |",
        "| :--- | :---: | :---: | :---: |",
    ])
    for malias, gm in lvl5_data.get("models", {}).items():
        def _fmt_g(gn):
            d = gm.get(gn) or {}
            if not d:
                return "N/A"
            return f"{d.get('doc_cer_pct')}% / {d.get('matched_cer_pct')}% ({d.get('lines')})"
        md.append(f"| `{malias}` | {_fmt_g('short:5')} | {_fmt_g('a4:5')} | {_fmt_g('audit_18')} |")

    out_md_path = os.path.join(PROJECT_ROOT, "REPORTS", "R3_DEGRADATION_EVALUATION.md")
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n✅ STAGE R3 EVALUATION COMPLETE -> {out_json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
