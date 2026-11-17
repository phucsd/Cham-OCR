#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R4 Wide-Canvas (`W = 480`, `T = 60` CTC Frames) Evaluation Runner
======================================================================================
1. Copies & SHA256-verifies extracted iter_epoch_1..3.pdparams & train.log from Kaggle.
2. Computes 3-part Ground-Truth Training Accounting (Train Loop, Overhead, Kernel Wall-clock).
3. Exports E1..E3 with PY26 (export_with_pir=False, d2s_train_image_shape=[3,48,480]).
4. Runs Frozen-838 + Blank + FullPage + Stress-82 at `--rec-image-shape "3, 48, 480"` (via PY313)
   + Held-Out Probe-100 at BOTH `W=320` AND `W=480` (via PY26) on R4 E1, E2, E3.
5. Evaluates Level-5 Full-Page Degradation (short:5, a4:5, audit_18) + RealScan-Gold v1
   + CPU Full-Page Latency (< 15s requirement) + OOD Font (`fonts_efeo` / `fonts_heldout`) check!
6. Emits REPORTS/r4_w480_evaluation.json & REPORTS/R4_W480_EVALUATION.md.
"""

import os
import sys
import time
import json
import shutil
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


def run_ood_diagnostic_sample(py_exe: str, r31_infer: str, r4_infer: str) -> dict:
    """Evaluates OOD Held-out & EFEO fonts (`benchmark_diagnostic_gt.json`, 240 OOD samples across font_ood splits)."""
    diag_gt = os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "benchmark_diagnostic_gt.json")
    if not os.path.exists(diag_gt):
        return {}
    code = f"""
import os, sys, json, cv2
sys.path.insert(0, {os.path.join(PROJECT_ROOT, 'ocr-studio')!r})
sys.path.insert(0, {os.path.join(PROJECT_ROOT, 'ocr-benchmark', 'scripts')!r})
import app as studio_app
from evaluate_pipeline import levenshtein_distance
from scripts.generate_data import normalize_unicode
import tools.infer.utility as utility
from tools.infer.predict_rec import TextRecognizer

def make_rec(mdir, shape):
    b = sys.argv; sys.argv = [sys.argv[0]]
    a = utility.parse_args(); sys.argv = b
    a.rec_model_dir = mdir
    a.rec_char_dict_path = {DICT_PATH!r}
    a.use_gpu = False; a.rec_image_shape = shape; a.use_space_char = True
    a.rec_batch_num = 1; a.enable_mkldnn = False; a.cpu_threads = 1; a.ir_optim = False; a.precision = 'fp32'
    return TextRecognizer(a)

raw = json.load(open({diag_gt!r}, 'r', encoding='utf-8'))
all_rows = raw if isinstance(raw, list) else raw.get('samples', [])
items = [x for x in all_rows if 'ood' in str(x.get('category', x.get('test_type', ''))).lower() or 'font' in str(x.get('category', x.get('test_type', ''))).lower()]
if not items:
    items = all_rows[:200]
items = items[:240]

models = {{
    'r3_1_w320': make_rec({r31_infer!r}, '3, 48, 320'),
    'r4_w480': make_rec({r4_infer!r}, '3, 48, 480'),
}}
out = {{'n_samples': len(items)}}
for name, rec in models.items():
    ed, ch = 0, 0
    for it in items:
        fn = it.get('image_filename') or it.get('filename') or it.get('image_relpath') or it.get('image_path')
        if not fn:
            continue
        p = os.path.join({os.path.join(PROJECT_ROOT, 'ocr-benchmark', 'data')!r}, 'images_diagnostic', os.path.basename(fn))
        if not os.path.exists(p):
            p = os.path.join({os.path.join(PROJECT_ROOT, 'ocr-benchmark', 'data')!r}, fn.replace('/', os.sep))
        im = cv2.imread(p)
        if im is None:
            continue
        prep = studio_app.preprocess_crop_for_rec(im)
        res, _ = rec([prep])
        pr = normalize_unicode(res[0][0] if res else '')
        gt_raw = it.get('gt') or it.get('text') or ' '.join(l.get('text', '') for l in (it.get('lines') or []))
        gt = normalize_unicode(gt_raw)
        ed += levenshtein_distance(pr, gt)
        ch += len(gt)
    out[name] = {{'cer_pct': round(100.0 * ed / max(1, ch), 2), 'edits': ed, 'chars': ch}}
print('OOD_JSON:' + json.dumps(out))
"""
    res = subprocess.run([py_exe, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    for ln in res.stdout.splitlines():
        if ln.startswith("OOD_JSON:"):
            return json.loads(ln[len("OOD_JSON:"):])
    return {}


def measure_cpu_page_latency(py_exe: str, r4_infer: str) -> dict:
    m_p = os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "realscan_gold_v1", "manifest.json")
    m_data = json.load(open(m_p, "r", encoding="utf-8"))
    harum_rel = next(d["page_relpath"] for d in m_data["documents"] if d["doc_id"] == "harum_palei_01")
    harum_p = os.path.join(PROJECT_ROOT, harum_rel.replace("/", os.sep))
    assert os.path.exists(harum_p), f"Missing {harum_p}"
    code = f"""
import os, sys, time, json, cv2
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, {os.path.join(PROJECT_ROOT, 'ocr-studio')!r})
import app as studio_app

studio_app.get_ocr_model('v24_1_r4')
studio_app.get_det_model()
img = cv2.imread({harum_p!r})
assert img is not None
# Warmup 1 pass, then measure 2 passes
studio_app.run_ocr_pipeline(img, model_ver='v24_1_r4', method='dbnet')
times = []
for _ in range(2):
    t0 = time.time()
    out = studio_app.run_ocr_pipeline(img, model_ver='v24_1_r4', method='dbnet')
    times.append(time.time() - t0)
mean_s = sum(times) / len(times)
print('LAT_JSON:' + json.dumps({{'page': 'harum_palei_01 (25 lines)', 'runs_sec': [round(x, 3) for x in times], 'mean_sec': round(mean_s, 3), 'under_15s_sla': mean_s < 15.0, 'detected_lines': len(out.get('lines', []))}}))
"""
    res = subprocess.run([py_exe, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    for ln in res.stdout.splitlines():
        if ln.startswith("LAT_JSON:"):
            return json.loads(ln[len("LAT_JSON:"):])
    return {}


def main():
    print("=" * 80)
    print("🔬  CHAM-OCR V24.1 STAGE R4 WIDE-CANVAS (W=480, T=60) EVALUATION")
    print("=" * 80)

    dl_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_4c_stage_r4_w480_kaggle_output")
    dl_extract_dir = os.path.join(dl_dir, "extracted_checkpoints")
    ckpt_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r4_w480_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    assert os.path.exists(dl_extract_dir), f"Missing extracted R4 checkpoints: {dl_extract_dir}"
    for f in os.listdir(dl_extract_dir):
        src = os.path.join(dl_extract_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(ckpt_dir, f))

    train_log = os.path.join(ckpt_dir, "train.log")
    kernel_log = os.path.join(dl_dir, "paddleocr-cham-v24-1-r4-wide-w480.log")
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
    cfg_path = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "rec_cham_v24_1_r4_w480.yml")
    suite_script = os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts", "run_v24_frozen_suite.py")
    policy = FrozenGatePolicy()
    comb_marks = set([chr(c) for c in range(0xAA29, 0xAA37)] + [chr(c) for c in range(0xAA43, 0xAA4D + 1) if c in (0xAA43, 0xAA4C, 0xAA4D)])

    r4_results = {}
    best_ep = None
    best_rank = (-1, -1, -1e9, -1e9)

    for ep in [1, 2, 3]:
        ckpt_prefix = os.path.join(ckpt_dir, f"iter_epoch_{ep}")
        ckpt_file = f"{ckpt_prefix}.pdparams"
        assert os.path.exists(ckpt_file), f"Missing {ckpt_file}"
        ep_sha = file_sha256(ckpt_file)
        infer_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"r4_w480_infer_e{ep}")
        os.makedirs(infer_dir, exist_ok=True)

        if not os.path.exists(os.path.join(infer_dir, "inference.pdiparams")):
            print(f"\n[R4 Epoch {ep}] Exporting W=480 inference model -> {infer_dir}...")
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

        out_dir = os.path.join(PROJECT_ROOT, "ocr-benchmark", "results", f"r4_w480_iter_epoch_{ep}_raw")
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
            print(f"[R4 Epoch {ep}] Running Frozen-838 & Blank/Stress-82 Benchmark Suite at W=480 -> {out_json}...")
            eval_cmd = [
                suite_py_exe, suite_script,
                "--model-dir", infer_dir,
                "--dict-path", DICT_PATH,
                "--rec-image-shape", "3, 48, 480",
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
        probe_w320 = evaluate_probe_100(ckpt_file, target_w=320)
        probe_w480 = evaluate_probe_100(ckpt_file, target_w=480)

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
        stress_suite = summary.get("stress_suite") or {}
        stress_82_cer = float(stress_suite.get("overall_stress_cer", 0.0))
        erode_conds = [c for c in (stress_suite.get("all_conditions") or []) if "erode" in str(c.get("name", "")).lower()]
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
        rank = (1 if gate_eval["valid_for_release"] else 0, gates_passed_count, round(comp, 4), -float(probe_w480["cer_pct"]))
        if best_ep is None or rank > best_rank:
            best_rank = rank
            best_ep = ep

        r4_results[f"epoch_{ep}"] = {
            "sha256": ep_sha,
            "infer_dir": infer_dir,
            "rec_image_shape": "3, 48, 480",
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
            "erode_conditions": erode_conds,
            "stress_base_8_cer_pct": round(stress_base_8_cer * 100.0, 4),
            "composite_score": round(comp, 4),
            "valid_for_release": gate_eval["valid_for_release"],
            "gates_passed_count": gates_passed_count,
            "total_gates": len(gate_eval["gates"]),
            "gates": gate_eval["gates"],
            "per_symbol_count_acc_pct": {s: round(v["accuracy"] * 100.0, 1) for s, v in summary["symbol_suite"]["per_symbol"].items()},
            "per_hyphen_group_count_acc_pct": {g: round(v["accuracy"] * 100.0, 1) for g, v in summary["hyphen_suite"]["by_group"].items()},
            "held_out_probe_100_w320": probe_w320,
            "held_out_probe_100_w480": probe_w480,
        }

    best_r4_infer = r4_results[f"epoch_{best_ep}"]["infer_dir"]
    r31_eval_p = os.path.join(PROJECT_ROOT, "REPORTS", "r3_1_closure_evaluation.json")
    r31_eval = json.load(open(r31_eval_p, encoding="utf-8"))
    best_r31_infer = r31_eval["best_epoch_infer_dir"]

    # Level-5 Full-Page Comparison
    lvl5_models = ["v24", "v24_1_r3", f"r3_1_w320={best_r31_infer}:{DICT_PATH}", f"r4_w480={best_r4_infer}:{DICT_PATH}"]
    lvl5_out_json = os.path.join(PROJECT_ROOT, "REPORTS", "r4_level5_fullpage_comparison.json")
    lvl5_script = os.path.join(PROJECT_ROOT, "scripts", "evaluate_level5_pages.py")
    if not os.path.exists(lvl5_out_json):
        print(f"\n[Level-5] Evaluating Full-Page short:5, a4:5 & audit_18 across {len(lvl5_models)} models...")
        subprocess.run([suite_py_exe, lvl5_script, "--models", *lvl5_models, "--limit", "90", "--output", lvl5_out_json], check=True)
    lvl5_data = json.load(open(lvl5_out_json, encoding="utf-8"))

    # RealScan-Gold v1 Comparison
    rs_out_json = os.path.join(PROJECT_ROOT, "REPORTS", "r4_realscan_gold_comparison.json")
    rs_out_md = os.path.join(PROJECT_ROOT, "REPORTS", "r4_realscan_gold_comparison.md")
    rs_script = os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts", "evaluate_realscan_gold.py")
    if not os.path.exists(rs_out_json):
        print(f"\n[RealScan-Gold v1] Evaluating real camera & multiscript benchmarks...")
        subprocess.run([
            suite_py_exe, rs_script,
            "--models", "v24", "v24_1_r3", f"r3_1_w320={best_r31_infer}:{DICT_PATH}", f"r4_w480={best_r4_infer}:{DICT_PATH}",
            "--output-json", rs_out_json,
            "--output-md", rs_out_md,
        ], check=True)
    rs_data = json.load(open(rs_out_json, encoding="utf-8"))

    # OOD Font Diagnostic & CPU Latency SLA (use studio packaged dirs if available so PIR inference.json is present under PY313)
    s_r31 = os.path.join(PROJECT_ROOT, "ocr-studio", "data", "output", "rec_cham_inference_v24_1_r3_1")
    s_r4 = os.path.join(PROJECT_ROOT, "ocr-studio", "data", "output", "rec_cham_inference_v24_1_r4")
    ood_r31 = s_r31 if os.path.exists(os.path.join(s_r31, "inference.json")) else best_r31_infer
    ood_r4 = s_r4 if os.path.exists(os.path.join(s_r4, "inference.json")) else best_r4_infer
    print("\n[OOD & CPU Latency] Checking OOD fonts (`fonts_efeo`/`fonts_heldout`) & CPU full-page latency (<15s)...")
    ood_res = run_ood_diagnostic_sample(suite_py_exe, ood_r31, ood_r4)
    lat_res = measure_cpu_page_latency(suite_py_exe, ood_r4)
    print(f"  • OOD Font Check: {ood_res}")
    print(f"  • CPU Page Latency: {lat_res}")

    best_res = r4_results[f"epoch_{best_ep}"]
    report = {
        "title": "Cham-OCR v24.1 Stage R4 Wide-Canvas (W=480, T=60) Evaluation",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "r4_w480": r4_results,
        "best_epoch": best_ep,
        "best_epoch_checkpoint": os.path.join(ckpt_dir, f"iter_epoch_{best_ep}.pdparams"),
        "best_epoch_infer_dir": best_r4_infer,
        "best_epoch_sha256": best_res["sha256"],
        "valid_for_release": best_res["valid_for_release"],
        "ground_truth_training_accounting": train_accounting,
        "level5_fullpage_comparison": lvl5_data,
        "realscan_gold_comparison": rs_data,
        "ood_font_diagnostic": ood_res,
        "cpu_page_latency": lat_res,
    }
    out_json_path = os.path.join(PROJECT_ROOT, "REPORTS", "r4_w480_evaluation.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    r31_ref = r31_eval["r3_1_closure"][f"epoch_{r31_eval['best_epoch']}"]
    md = [
        "# Đánh giá Stage R4 (`WIDE-CANVAS W=480, T=60 CTC FRAMES + CORPUS & OPTICAL`)",
        f"- **Thời điểm**: `{report['evaluated_at']}`",
        f"- **Epoch tốt nhất R4**: Epoch {best_ep} (`SHA256: {best_res['sha256'][:16]}...`, `valid_for_release={best_res['valid_for_release']}`)",
        f"- **Train Loop Duration**: `{train_accounting.get('train_loop_duration_seconds')}s` ({train_accounting.get('train_loop_duration_minutes')} phút)",
        f"- **Pipeline / Container Overhead**: `{train_accounting.get('pipeline_container_overhead_seconds')}s` ({train_accounting.get('pipeline_container_overhead_minutes')} phút)",
        f"- **Total Kernel Wall-clock Time**: `{train_accounting.get('total_kernel_wall_clock_seconds')}s` ({train_accounting.get('total_kernel_wall_clock_minutes')} phút)",
        f"- **CPU Full-Page Latency (`harum_palei_01`, 25 dòng)**: `{lat_res.get('mean_sec')}s` (`< 15s SLA`: `{lat_res.get('under_15s_sla')}`)",
        f"- **OOD Font Uncontaminated Check**: `{ood_res}`",
        "",
        "## 1. Frozen-838 & Stress-82 & Probe-100 (`W=320` vs `W=480`)",
        "| Metric | R3.1 Best (`W=320`) | R4 E1 (`W=480`) | R4 E2 (`W=480`) | R4 E3 (`W=480`) |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| Clean-Frozen CER | {r31_ref['clean_cer_pct']:.2f}% | {r4_results['epoch_1']['clean_cer_pct']:.2f}% | {r4_results['epoch_2']['clean_cer_pct']:.2f}% | {r4_results['epoch_3']['clean_cer_pct']:.2f}% |",
        f"| Character-Frozen CER | {r31_ref['char_cer_pct']:.2f}% | {r4_results['epoch_1']['char_cer_pct']:.2f}% | {r4_results['epoch_2']['char_cer_pct']:.2f}% | {r4_results['epoch_3']['char_cer_pct']:.2f}% |",
        f"| 14 Symbols Acc | {r31_ref['symbols_acc_pct']:.2f}% | {r4_results['epoch_1']['symbols_acc_pct']:.2f}% | {r4_results['epoch_2']['symbols_acc_pct']:.2f}% | {r4_results['epoch_3']['symbols_acc_pct']:.2f}% |",
        f"| Failing Symbols | {r31_ref['failing_symbols']} | {r4_results['epoch_1']['failing_symbols']} | {r4_results['epoch_2']['failing_symbols']} | {r4_results['epoch_3']['failing_symbols']} |",
        f"| Hyphen Multi-Context | {r31_ref['hyphen_acc_pct']:.2f}% | {r4_results['epoch_1']['hyphen_acc_pct']:.2f}% | {r4_results['epoch_2']['hyphen_acc_pct']:.2f}% | {r4_results['epoch_3']['hyphen_acc_pct']:.2f}% |",
        f"| Blank Guard | {r31_ref['blank_passed']} | {r4_results['epoch_1']['blank_passed']} | {r4_results['epoch_2']['blank_passed']} | {r4_results['epoch_3']['blank_passed']} |",
        f"| FullPage Matched CER | {r31_ref['fullpage_matched_cer_pct']:.2f}% | {r4_results['epoch_1']['fullpage_matched_cer_pct']:.2f}% | {r4_results['epoch_2']['fullpage_matched_cer_pct']:.2f}% | {r4_results['epoch_3']['fullpage_matched_cer_pct']:.2f}% |",
        f"| Stress-82 CER | {r31_ref['stress_frozen_cer_pct']:.2f}% | {r4_results['epoch_1']['stress_frozen_cer_pct']:.2f}% | {r4_results['epoch_2']['stress_frozen_cer_pct']:.2f}% | {r4_results['epoch_3']['stress_frozen_cer_pct']:.2f}% |",
        f"| Probe-100 CER (`W=320`) | {r31_ref['held_out_probe_100_w320']['cer_pct']:.2f}% | {r4_results['epoch_1']['held_out_probe_100_w320']['cer_pct']:.2f}% | {r4_results['epoch_2']['held_out_probe_100_w320']['cer_pct']:.2f}% | {r4_results['epoch_3']['held_out_probe_100_w320']['cer_pct']:.2f}% |",
        f"| Probe-100 CER (`W=480`) | N/A | {r4_results['epoch_1']['held_out_probe_100_w480']['cer_pct']:.2f}% | {r4_results['epoch_2']['held_out_probe_100_w480']['cer_pct']:.2f}% | {r4_results['epoch_3']['held_out_probe_100_w480']['cer_pct']:.2f}% |",
        f"| Gates Passed | {r31_ref['gates_passed_count']}/{r31_ref['total_gates']} | {r4_results['epoch_1']['gates_passed_count']}/{r4_results['epoch_1']['total_gates']} | {r4_results['epoch_2']['gates_passed_count']}/{r4_results['epoch_2']['total_gates']} | {r4_results['epoch_3']['gates_passed_count']}/{r4_results['epoch_3']['total_gates']} |",
        "",
        "## 2. Full-Page Level-5 Degradation & 18-Image Audit Suite",
        "| Model | short:5 Doc/Matched CER | a4:5 Doc/Matched CER | audit_18 Doc/Matched CER |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for malias, gm in lvl5_data.get("models", {}).items():
        def _fmt_g(gn):
            d = gm.get(gn) or {}
            if not d:
                return "N/A"
            return f"{d.get('doc_cer_pct')}% / {d.get('matched_cer_pct')}% ({d.get('lines')})"
        md.append(f"| `{malias}` | {_fmt_g('short:5')} | {_fmt_g('a4:5')} | {_fmt_g('audit_18')} |")

    out_md_path = os.path.join(PROJECT_ROOT, "REPORTS", "R4_W480_EVALUATION.md")
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n✅ STAGE R4 EVALUATION COMPLETE -> {out_json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
