#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R2.1.7 Causal Evaluation & Gate-Policy Comparison Runner
=============================================================================
Executes full post-training verification for Stage R2.1.7 (`SURGICAL-GATE-CLOSURE`):
1. Extracts & verifies SHA256 of iter_epoch_1..3.pdparams & latest.pdparams.
2. Parses train.log for exact 3-part Ground-Truth Training Accounting (Train Loop Duration, etc.).
3. Exports iter_epoch_1, 2, 3 to Paddle Inference format (export_with_pir=False) using .venv_paddle26.
4. Runs run_v24_frozen_suite.py + Held-Out Probe-100 on all 3 epochs.
5. Computes dynamic FrozenGatePolicy status across all gates and emits:
   - REPORTS/r2_1_7_causal_evaluation.json
   - REPORTS/R2_1_7_CAUSAL_EVALUATION.md
"""

import os
import sys
import re
import json
import yaml
import shutil
import hashlib
import subprocess
from datetime import datetime, timezone
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-studio", "PaddleOCR"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts"))

from evaluate_r2_1_6_causal import file_sha256, evaluate_probe_100
from analyze_causal_breakdown import analyze_ops
from diagnostic_r2_1_4b_ctc_counterfactual import DICT_PATH
from frozen_gate_policy import FrozenGatePolicy


def parse_kernel_wall_clock(kernel_log_path: str) -> float:
    if not os.path.exists(kernel_log_path):
        return None
    raw_txt = open(kernel_log_path, "r", encoding="utf-8", errors="replace").read()
    try:
        entries = json.loads(raw_txt)
        times = [float(e["time"]) for e in entries if isinstance(e, dict) and "time" in e]
        if times:
            return round(max(times), 2)
    except Exception:
        pass
    txt_times = [float(m) for m in re.findall(r"^(\d+(?:\.\d+)?)s\s+\d+\s", raw_txt, flags=re.M)]
    return round(max(txt_times), 2) if txt_times else None


def parse_train_log(log_path: str, total_epochs: int = 3, kernel_log_path: str = None) -> dict:
    if not os.path.exists(log_path):
        return {"error": f"Log missing: {log_path}"}
    first_ts = None
    last_ts = None
    epoch_losses = {}
    ts_re = re.compile(rf"^\[(\d{{4}}/\d{{2}}/\d{{2}} \d{{2}}:\d{{2}}:\d{{2}})\] ppocr INFO: epoch: \[(\d+)/{total_epochs}\], global_step: (\d+),.*loss: ([0-9.]+)")
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ts_re.search(line.strip())
            if m:
                ts_str, ep_str, step_str, loss_str = m.groups()
                dt = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
                if first_ts is None:
                    first_ts = dt
                last_ts = dt
                epoch_losses[f"epoch_{int(ep_str)}_final_loss"] = float(loss_str)
    dur_s = (last_ts - first_ts).total_seconds() if (first_ts and last_ts) else None
    wall_s = parse_kernel_wall_clock(kernel_log_path) if kernel_log_path else None
    overhead_s = round(max(0.0, wall_s - dur_s), 2) if (wall_s is not None and dur_s is not None) else None
    out = {
        "first_batch_timestamp": first_ts.strftime("%Y-%m-%d %H:%M:%S") if first_ts else None,
        "last_batch_timestamp": last_ts.strftime("%Y-%m-%d %H:%M:%S") if last_ts else None,
        "train_loop_duration_seconds": dur_s,
        "train_loop_duration_minutes": round(dur_s / 60.0, 2) if dur_s else None,
        "pipeline_container_overhead_seconds": overhead_s,
        "pipeline_container_overhead_minutes": round(overhead_s / 60.0, 2) if overhead_s is not None else None,
        "total_kernel_wall_clock_seconds": wall_s,
        "total_kernel_wall_clock_minutes": round(wall_s / 60.0, 2) if wall_s is not None else None,
    }
    out.update(epoch_losses)
    return out


def main():
    print("=" * 80)
    print("🔬  CHAM-OCR V24.1 STAGE R2.1.7 CAUSAL EVALUATION (EPOCHS 1, 2, 3)")
    print("=" * 80)

    dl_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_4c_gate_closure_r2_1_7_kaggle_output")
    dl_extract_dir = os.path.join(dl_dir, "extracted_checkpoints")
    ckpt_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_7_gate_closure_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    assert os.path.exists(dl_extract_dir), f"Missing extracted checkpoints dir: {dl_extract_dir}"
    for f in os.listdir(dl_extract_dir):
        src = os.path.join(dl_extract_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(ckpt_dir, f))

    train_log = os.path.join(ckpt_dir, "train.log")
    kernel_log = os.path.join(dl_dir, "paddleocr-cham-v24-1-r2-1-7-gate-closure.log")
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
    print(f"  • Export Python (PIR=False): {py_exe}")
    print(f"  • Benchmark Suite Python (Cham-DBNet PIR support): {suite_py_exe}")

    paddle_dir = os.path.join(PROJECT_ROOT, "ocr-studio", "PaddleOCR")
    cfg_path = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "rec_cham_v24_1_r2_1_7_gate_closure.yml")
    suite_script = os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts", "run_v24_frozen_suite.py")
    policy = FrozenGatePolicy()

    comb_marks = set([chr(c) for c in range(0xAA29, 0xAA37)] + [chr(c) for c in range(0xAA43, 0xAA4D + 1) if c in (0xAA43, 0xAA4C, 0xAA4D)])
    r2_1_7_results = {}
    best_ep = None
    best_rank = (-1, -1e9)

    for ep in [1, 2, 3]:
        ckpt_prefix = os.path.join(ckpt_dir, f"iter_epoch_{ep}")
        ckpt_file = f"{ckpt_prefix}.pdparams"
        assert os.path.exists(ckpt_file), f"Missing {ckpt_file}"
        ep_sha = file_sha256(ckpt_file)
        infer_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"r2_1_7_gate_closure_infer_e{ep}")
        os.makedirs(infer_dir, exist_ok=True)

        print(f"\n[Epoch {ep}] Exporting inference model -> {infer_dir}...")
        if not os.path.exists(os.path.join(infer_dir, "inference.pdiparams")):
            exp_cmd = [
                py_exe, "tools/export_model.py",
                "-c", cfg_path,
                "-o",
                f"Global.checkpoints={ckpt_prefix}",
                f"Global.save_inference_dir={infer_dir}",
                f"Global.character_dict_path={DICT_PATH}",
                "Global.export_with_pir=False"
            ]
            subprocess.run(exp_cmd, cwd=paddle_dir, check=True)
        infer_sha = file_sha256(os.path.join(infer_dir, "inference.pdiparams"))

        out_dir = os.path.join(PROJECT_ROOT, "ocr-benchmark", "results", f"r2_1_7_gate_closure_iter_epoch_{ep}_raw")
        os.makedirs(out_dir, exist_ok=True)
        out_json = os.path.join(out_dir, "benchmark_summary.json")

        reuse_ok = False
        if os.path.exists(out_json):
            try:
                cached = json.load(open(out_json, encoding="utf-8"))
                if cached.get("model_sha256") == infer_sha and "fullpage_suite" in cached:
                    reuse_ok = True
                    print(f"[Epoch {ep}] Reusing verified Frozen-838 summary (model_sha256={infer_sha[:16]}...) -> {out_json}")
            except Exception:
                reuse_ok = False

        if not reuse_ok:
            print(f"[Epoch {ep}] Running Frozen-838 & Blank/Stress-8 Benchmark Suite -> {out_json}...")
            eval_cmd = [
                suite_py_exe, suite_script,
                "--model-dir", infer_dir,
                "--dict-path", DICT_PATH,
                "--output", out_json
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
            fullpage_cer=fp_cer
        )
        gates_passed_count = sum(1 for g in gate_eval["gates"] if g["status"] == "PASS")
        comp = float(summary["composite_score"])
        rank = (1 if gate_eval["valid_for_release"] else 0, gates_passed_count, comp)
        if best_ep is None or rank > best_rank:
            best_rank = rank
            best_ep = ep

        r2_1_7_results[f"epoch_{ep}"] = {
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
            "per_combmark_deletions": {f"U+{ord(ch):04X}_{ch}": del_chars.get(ch, 0) for ch in sorted(comb_marks)},
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
            "held_out_probe_100_w320": probe_res
        }

    r2_1_6_path = os.path.join(PROJECT_ROOT, "REPORTS", "r2_1_6_causal_evaluation.json")
    r2_1_6 = json.load(open(r2_1_6_path, encoding="utf-8"))
    ref_e2 = r2_1_6["r2_1_6_surgical_polish"]["epoch_2"]
    best_res = r2_1_7_results[f"epoch_{best_ep}"]

    report = {
        "title": "Cham-OCR v24.1 Stage R2.1.7 Surgical Gate-Closure Evaluation",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "r2_1_6_epoch_2_reference": ref_e2,
        "r2_1_7_gate_closure": r2_1_7_results,
        "best_epoch": best_ep,
        "best_epoch_checkpoint": os.path.join(ckpt_dir, f"iter_epoch_{best_ep}.pdparams"),
        "best_epoch_sha256": best_res["sha256"],
        "valid_for_release": best_res["valid_for_release"],
        "ground_truth_training_accounting": train_accounting,
        "delta_best_minus_r2_1_6_e2": {
            "clean_cer_pct": round(best_res["clean_cer_pct"] - ref_e2["clean_cer_pct"], 4),
            "char_cer_pct": round(best_res["char_cer_pct"] - ref_e2["char_cer_pct"], 4),
            "symbols_acc_pct": round(best_res["symbols_acc_pct"] - ref_e2["symbols_acc_pct"], 4),
            "hyphen_acc_pct": round(best_res["hyphen_acc_pct"] - ref_e2["hyphen_acc_pct"], 4),
            "composite_score": round(best_res["composite_score"] - ref_e2["composite_score"], 4),
            "probe_100_cer_pct": round(best_res["held_out_probe_100_w320"]["cer_pct"] - ref_e2["held_out_probe_100_w320"]["cer_pct"], 2),
        }
    }

    out_json_path = os.path.join(PROJECT_ROOT, "REPORTS", "r2_1_7_causal_evaluation.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# Đánh giá Stage R2.1.7 (`SURGICAL-GATE-CLOSURE`)",
        f"- **Thời điểm**: `{report['evaluated_at']}`",
        f"- **Epoch tốt nhất**: Epoch {best_ep} (`SHA256: {best_res['sha256'][:16]}...`, `valid_for_release={best_res['valid_for_release']}`)",
        f"- **Train Loop Duration**: `{train_accounting.get('train_loop_duration_seconds')}s` ({train_accounting.get('train_loop_duration_minutes')} phút)",
        f"- **Pipeline / Container Overhead**: `{train_accounting.get('pipeline_container_overhead_seconds')}s` ({train_accounting.get('pipeline_container_overhead_minutes')} phút)",
        f"- **Total Kernel Wall-clock Time**: `{train_accounting.get('total_kernel_wall_clock_seconds')}s` ({train_accounting.get('total_kernel_wall_clock_minutes')} phút)",
        "",
        "| Metric | R2.1.6 E2 | R2.1.7 E1 | R2.1.7 E2 | R2.1.7 E3 |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| Clean-Frozen CER | {ref_e2['clean_cer_pct']:.2f}% | {r2_1_7_results['epoch_1']['clean_cer_pct']:.2f}% | {r2_1_7_results['epoch_2']['clean_cer_pct']:.2f}% | {r2_1_7_results['epoch_3']['clean_cer_pct']:.2f}% |",
        f"| Character-Frozen CER | {ref_e2['char_cer_pct']:.2f}% | {r2_1_7_results['epoch_1']['char_cer_pct']:.2f}% | {r2_1_7_results['epoch_2']['char_cer_pct']:.2f}% | {r2_1_7_results['epoch_3']['char_cer_pct']:.2f}% |",
        f"| 14 Symbols Acc | {ref_e2['symbols_acc_pct']:.2f}% | {r2_1_7_results['epoch_1']['symbols_acc_pct']:.2f}% | {r2_1_7_results['epoch_2']['symbols_acc_pct']:.2f}% | {r2_1_7_results['epoch_3']['symbols_acc_pct']:.2f}% |",
        f"| Hyphen Multi-Context | {ref_e2['hyphen_acc_pct']:.2f}% | {r2_1_7_results['epoch_1']['hyphen_acc_pct']:.2f}% | {r2_1_7_results['epoch_2']['hyphen_acc_pct']:.2f}% | {r2_1_7_results['epoch_3']['hyphen_acc_pct']:.2f}% |",
        f"| Blank Guard | {ref_e2['blank_passed']} | {r2_1_7_results['epoch_1']['blank_passed']} | {r2_1_7_results['epoch_2']['blank_passed']} | {r2_1_7_results['epoch_3']['blank_passed']} |",
        f"| FullPage Matched CER | 6.21% | {r2_1_7_results['epoch_1']['fullpage_matched_cer_pct']:.2f}% | {r2_1_7_results['epoch_2']['fullpage_matched_cer_pct']:.2f}% | {r2_1_7_results['epoch_3']['fullpage_matched_cer_pct']:.2f}% |",
        f"| Stress-82 CER | 16.89% | {r2_1_7_results['epoch_1']['stress_frozen_cer_pct']:.2f}% | {r2_1_7_results['epoch_2']['stress_frozen_cer_pct']:.2f}% | {r2_1_7_results['epoch_3']['stress_frozen_cer_pct']:.2f}% |",
        f"| Stress-Base-8 CER | 5.13% | {r2_1_7_results['epoch_1']['stress_base_8_cer_pct']:.2f}% | {r2_1_7_results['epoch_2']['stress_base_8_cer_pct']:.2f}% | {r2_1_7_results['epoch_3']['stress_base_8_cer_pct']:.2f}% |",
        f"| Probe-100 CER | {ref_e2['held_out_probe_100_w320']['cer_pct']:.2f}% | {r2_1_7_results['epoch_1']['held_out_probe_100_w320']['cer_pct']:.2f}% | {r2_1_7_results['epoch_2']['held_out_probe_100_w320']['cer_pct']:.2f}% | {r2_1_7_results['epoch_3']['held_out_probe_100_w320']['cer_pct']:.2f}% |",
        f"| Composite Score | {ref_e2['composite_score']:.4f} | {r2_1_7_results['epoch_1']['composite_score']:.4f} | {r2_1_7_results['epoch_2']['composite_score']:.4f} | {r2_1_7_results['epoch_3']['composite_score']:.4f} |",
        f"| Gates Passed | 4/6 | {r2_1_7_results['epoch_1']['gates_passed_count']}/{r2_1_7_results['epoch_1']['total_gates']} | {r2_1_7_results['epoch_2']['gates_passed_count']}/{r2_1_7_results['epoch_2']['total_gates']} | {r2_1_7_results['epoch_3']['gates_passed_count']}/{r2_1_7_results['epoch_3']['total_gates']} |",
    ]
    out_md_path = os.path.join(PROJECT_ROOT, "REPORTS", "R2_1_7_CAUSAL_EVALUATION.md")
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n✅ STAGE R2.1.7 EVALUATION COMPLETE -> {out_json_path}")
    print(f"   Best: Epoch {best_ep}, valid_for_release={best_res['valid_for_release']}, deltas={report['delta_best_minus_r2_1_6_e2']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
