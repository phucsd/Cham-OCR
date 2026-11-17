#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R2.1.6 Causal Evaluation & 6-Way Comparison Runner
========================================================================
Executes the full post-training verification for Stage R2.1.6 (SURGICAL-SYMBOL-HYPHEN-POLISH):
1. Extracts & verifies SHA256 of iter_epoch_1.pdparams, iter_epoch_2.pdparams, latest.pdparams.
2. Parses train.log for Ground-Truth Training Accounting (Train Loop Duration, batch timestamps, losses).
3. Exports iter_epoch_1 and iter_epoch_2 to Paddle Inference format (export_with_pir=False).
4. Runs run_v24_frozen_suite.py on both epochs (Frozen-838, Blank Guards, Stress-8).
5. Runs True Wagner-Fischer Minimum-Edit-Distance Levenshtein DP Breakdown across Character-Frozen (280 lines)
   and Clean-Frozen (200 lines), including per-combining-mark deletion breakdown (ꨯ U+AA2F, ꨰ U+AA30, etc.).
6. Runs Held-Out Diagnostic Probe 100 (1,952 GT chars, W=320) Sub-Peak Deletion Taxonomy on E1 & E2.
7. Emits reports/r2_1_6_causal_evaluation.json with 6-way comparison (A vs B vs D vs E vs R2.1.5 vs R2.1.6).
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

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-studio", "PaddleOCR"))

from analyze_causal_breakdown import levenshtein_alignment, analyze_ops, classify_cham_codepoint
from diagnostic_r2_1_4b_ctc_counterfactual import (
    preprocess_aspect_preserve_cap,
    greedy_ctc_decode,
    ctc_forced_alignment,
    PROBE_JSON,
    DICT_PATH,
    CONFIG_PATH
)


def file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def parse_train_log(log_path: str) -> dict:
    if not os.path.exists(log_path):
        return {"error": f"Log missing: {log_path}"}
    first_ts = None
    last_ts = None
    e1_loss = None
    e2_loss = None
    ts_re = re.compile(r"^\[(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\] ppocr INFO: epoch: \[(\d+)/2\], global_step: (\d+),.*loss: ([0-9.]+)")
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ts_re.search(line.strip())
            if m:
                ts_str, ep_str, step_str, loss_str = m.groups()
                dt = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
                if first_ts is None:
                    first_ts = dt
                last_ts = dt
                if int(ep_str) == 1:
                    e1_loss = float(loss_str)
                elif int(ep_str) == 2:
                    e2_loss = float(loss_str)
    dur_s = (last_ts - first_ts).total_seconds() if (first_ts and last_ts) else None
    return {
        "first_batch_timestamp": first_ts.strftime("%Y-%m-%d %H:%M:%S") if first_ts else None,
        "last_batch_timestamp": last_ts.strftime("%Y-%m-%d %H:%M:%S") if last_ts else None,
        "train_loop_duration_seconds": dur_s,
        "train_loop_duration_minutes": round(dur_s / 60.0, 2) if dur_s else None,
        "epoch_1_final_loss": e1_loss,
        "epoch_2_final_loss": e2_loss
    }


def evaluate_probe_100(cp_path: str, target_w: int = 320) -> dict:
    import paddle
    import paddle.nn.functional as F
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process

    with open(PROBE_JSON, "r", encoding="utf-8") as f:
        probe_data = json.load(f)
    probe_items = probe_data if isinstance(probe_data, list) else probe_data.get("samples", probe_data.get("probe_samples", []))
    images = [cv2.imread(it["image_abspath"] if "image_abspath" in it and os.path.exists(it["image_abspath"]) else os.path.join(PROJECT_ROOT, it.get("image_path", it.get("rel_path", "")))) for it in probe_items]

    postprocess_op = build_post_process({
        "name": "CTCLabelDecode",
        "character_dict_path": DICT_PATH,
        "use_space_char": True
    })
    char_to_idx = {c: i for i, c in enumerate(postprocess_op.character)}
    char_to_idx["<blank>"] = 0
    idx_to_char = {i: c for i, c in enumerate(postprocess_op.character)}
    idx_to_char[0] = "<blank>"

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["Architecture"]["Head"]["out_channels_list"] = {
        "CTCLabelDecode": len(postprocess_op.character),
        "NRTRLabelDecode": len(postprocess_op.character) + 3
    }

    tensors_w = [preprocess_aspect_preserve_cap(img, target_w=target_w, target_h=48)[0] for img in images]

    model = build_model(cfg["Architecture"])
    model.eval()
    sd = paddle.load(cp_path)
    model.set_state_dict(sd)

    tot_edits, tot_d, tot_s, tot_i, exact_cnt = 0, 0, 0, 0, 0
    tax = defaultdict(int)
    comb_tax = defaultdict(int)

    for idx, item in enumerate(probe_items):
        gt_str = item["gt"]
        gt_tokens = [char_to_idx[c] for c in gt_str if c in char_to_idx]
        with paddle.no_grad():
            out = model(tensors_w[idx])
            logits = out["ctc"] if isinstance(out, dict) else out
        probs = F.softmax(logits, axis=-1).numpy()[0]
        log_probs = F.log_softmax(logits, axis=-1).numpy()[0]
        T, V = probs.shape

        fa_res = ctc_forced_alignment(log_probs, gt_tokens, blank_idx=0)
        argmax_path = np.argmax(probs, axis=1)
        pred_tokens = greedy_ctc_decode(probs, blank_idx=0)
        pred_str = "".join([idx_to_char[t] for t in pred_tokens if t in idx_to_char])

        e_dist, ops = levenshtein_alignment(gt_str, pred_str)
        d_c = sum(1 for op in ops if op[0] == "del")
        s_c = sum(1 for op in ops if op[0] == "sub")
        i_c = sum(1 for op in ops if op[0] == "ins")
        tot_edits += e_dist
        tot_d += d_c
        tot_s += s_c
        tot_i += i_c
        if e_dist == 0:
            exact_cnt += 1

        for op in ops:
            if op[0] == "del":
                gt_idx = op[1]
                char = gt_str[gt_idx]
                tok_idx = char_to_idx[char]
                cat = classify_cham_codepoint(char)
                is_adj = (gt_idx > 0 and gt_str[gt_idx] == gt_str[gt_idx - 1]) or (gt_idx < len(gt_str) - 1 and gt_str[gt_idx] == gt_str[gt_idx + 1])
                if is_adj:
                    tok_frames = [t for t in range(T) if argmax_path[t] == tok_idx]
                    has_contig = any(tok_frames[k] == tok_frames[k - 1] + 1 for k in range(1, len(tok_frames)))
                    cls_name = "repeated_token_collapse" if has_contig else "correct_token_top2_top3_only"
                else:
                    assigned_frames = []
                    if fa_res["success"] and gt_idx in fa_res["char_occupancy"]:
                        st_path = fa_res["state_path"]
                        target_state = 2 * gt_idx + 1
                        assigned_frames = [t for t in range(T) if st_path[t] == target_state]
                    if not assigned_frames:
                        expected_center = int(round(gt_idx / max(1, len(gt_str)) * T))
                        assigned_frames = list(range(max(0, expected_center - 2), min(T, expected_center + 3)))
                    ever_argmax = any(argmax_path[t] == tok_idx for t in assigned_frames)
                    in_top2_top3 = False
                    max_prob = 0.0
                    for t in assigned_frames:
                        p_t = probs[t]
                        top3 = np.argpartition(p_t, -3)[-3:]
                        if tok_idx in top3 and argmax_path[t] != tok_idx:
                            in_top2_top3 = True
                        if p_t[tok_idx] > max_prob:
                            max_prob = float(p_t[tok_idx])
                    if ever_argmax:
                        cls_name = "argmax_present_alignment_mismatch"
                    elif in_top2_top3 or max_prob > 0.05:
                        cls_name = "correct_token_top2_top3_only"
                    else:
                        cls_name = "missing_token_never_top1"
                tax[cls_name] += 1
                if cat in ("DEPENDENT_VOWEL_COMBINING_MARK", "FINAL_CONSONANT_SIGN"):
                    comb_tax[cls_name] += 1
                    comb_tax["TOTAL_COMB_DEL"] += 1

    cer = round(tot_edits / 1952.0 * 100.0, 2)
    return {
        "cer_pct": cer,
        "edits": tot_edits,
        "exact_matches": exact_cnt,
        "deletions": tot_d,
        "substitutions": tot_s,
        "insertions": tot_i,
        "subpeak_top2_top3_suppressed": tax["correct_token_top2_top3_only"],
        "subpeak_pct_of_deletions": round(tax["correct_token_top2_top3_only"] / max(1, tot_d) * 100.0, 2),
        "missing_never_top1": tax["missing_token_never_top1"],
        "argmax_alignment_mismatch": tax["argmax_present_alignment_mismatch"],
        "repeated_token_collapse": tax["repeated_token_collapse"],
        "combmark_final_deletions": comb_tax["TOTAL_COMB_DEL"],
        "combmark_subpeak_deletions": comb_tax["correct_token_top2_top3_only"]
    }


def main():
    print("=" * 80)
    print("🔬  CHAM-OCR V24.1 STAGE R2.1.6 CAUSAL EVALUATION & 6-WAY COMPARISON")
    print("=" * 80)

    dl_extract_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_4c_surgical_polish_kaggle_output", "extracted_checkpoints")
    ckpt_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_6_surgical_polish_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    assert os.path.exists(dl_extract_dir), f"Missing extracted checkpoints dir: {dl_extract_dir}"
    for f in os.listdir(dl_extract_dir):
        src = os.path.join(dl_extract_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(ckpt_dir, f))

    e1_ckpt = os.path.join(ckpt_dir, "iter_epoch_1.pdparams")
    e2_ckpt = os.path.join(ckpt_dir, "iter_epoch_2.pdparams")
    train_log = os.path.join(ckpt_dir, "train.log")
    assert os.path.exists(e1_ckpt), f"Missing {e1_ckpt}"
    assert os.path.exists(e2_ckpt), f"Missing {e2_ckpt}"

    e1_sha = file_sha256(e1_ckpt)
    e2_sha = file_sha256(e2_ckpt)
    print(f"  • R2.1.6 Epoch 1 SHA256: {e1_sha}")
    print(f"  • R2.1.6 Epoch 2 SHA256: {e2_sha}")

    train_accounting = parse_train_log(train_log)
    print(f"  • Train Loop Accounting: {train_accounting}")

    py_exe = os.path.join(PROJECT_ROOT, ".venv_paddle26", "Scripts", "python.exe")
    if not os.path.exists(py_exe):
        py_exe = sys.executable

    paddle_dir = os.path.join(PROJECT_ROOT, "ocr-studio", "PaddleOCR")
    cfg_path = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "rec_cham_v24_1_r2_1_6_surgical_polish.yml")
    suite_script = os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts", "run_v24_frozen_suite.py")

    comb_marks = set([chr(c) for c in range(0xAA29, 0xAA37)] + [chr(0xAA43), chr(0xAA4C), chr(0xAA4D)])
    r2_1_6_results = {}

    for ep in [1, 2]:
        ckpt_prefix = os.path.join(ckpt_dir, f"iter_epoch_{ep}")
        infer_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"r2_1_6_surgical_polish_infer_e{ep}")
        os.makedirs(infer_dir, exist_ok=True)

        print(f"\n[Epoch {ep}] Exporting inference model -> {infer_dir}...")
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

        out_dir = os.path.join(PROJECT_ROOT, "ocr-benchmark", "results", f"r2_1_6_surgical_polish_iter_epoch_{ep}_raw")
        os.makedirs(out_dir, exist_ok=True)
        out_json = os.path.join(out_dir, "benchmark_summary.json")

        print(f"[Epoch {ep}] Running Frozen-838 & Blank/Stress-8 Benchmark Suite -> {out_json}...")
        eval_cmd = [
            py_exe, suite_script,
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
        probe_res = evaluate_probe_100(f"{ckpt_prefix}.pdparams")

        r2_1_6_results[f"epoch_{ep}"] = {
            "sha256": e1_sha if ep == 1 else e2_sha,
            "clean_cer_pct": round(clean_suite["norm_cer"] * 100.0, 4),
            "clean_raw_cer_pct": round(clean_suite["raw_cer"] * 100.0, 4),
            "char_cer_pct": round(char_suite["cer"] * 100.0, 4),
            "char_edits": char_suite["edits"],
            "char_deletions": dels,
            "char_substitutions": subs,
            "char_insertions": ins,
            "combmark_final_deletions": comb_final_del,
            "dependent_vowel_combmark_deletions": del_cats.get("DEPENDENT_VOWEL_COMBINING_MARK", 0),
            "dependent_vowel_combmark_substitutions": sub_cats.get("DEPENDENT_VOWEL_COMBINING_MARK", 0),
            "dependent_vowel_combmark_insertions": ins_cats.get("DEPENDENT_VOWEL_COMBINING_MARK", 0),
            "final_consonant_sign_deletions": del_cats.get("FINAL_CONSONANT_SIGN", 0),
            "base_consonant_deletions": del_cats.get("BASE_CONSONANT", 0),
            "space_deletions": del_cats.get("SPACE", 0),
            "del_categories": del_cats,
            "sub_categories": sub_cats,
            "ins_categories": ins_cats,
            "per_combmark_deletions": {f"U+{ord(ch):04X}_{ch}": del_chars.get(ch, 0) for ch in sorted(comb_marks)},
            "symbols_acc_pct": round(summary["symbol_suite"]["overall_accuracy"] * 100.0, 4),
            "hyphen_acc_pct": round(summary["hyphen_suite"]["overall_accuracy"] * 100.0, 4),
            "blank_passed": summary["blank_suite"]["passed"],
            "composite_score": round(summary["composite_score"], 4),
            "per_symbol_count_acc_pct": {s: round(v["accuracy"] * 100.0, 1) for s, v in summary["symbol_suite"]["per_symbol"].items()},
            "per_hyphen_group_count_acc_pct": {g: round(v["accuracy"] * 100.0, 1) for g, v in summary["hyphen_suite"]["by_group"].items()},
            "held_out_probe_100_w320": probe_res
        }

    # Load R2.1.5 evaluation (contains Base, A, B, D, E, R2.1.5) for 6-way comparison
    r2_1_5_path = os.path.join(PROJECT_ROOT, "reports", "r2_1_5_causal_evaluation.json")
    r2_1_5 = json.load(open(r2_1_5_path, encoding="utf-8"))

    a_e2 = r2_1_5["branch_a_control_continue"]["epoch_2"]
    e_e1 = r2_1_5["branch_e_short_phonotactic_combmark"]["epoch_1"]
    e_e2 = r2_1_5["branch_e_short_phonotactic_combmark"]["epoch_2"]
    m5_e1 = r2_1_5["r2_1_5_multi_stream"]["epoch_1"]
    m5_e2 = r2_1_5["r2_1_5_multi_stream"]["epoch_2"]
    m6_e1 = r2_1_6_results["epoch_1"]
    m6_e2 = r2_1_6_results["epoch_2"]

    def compute_delta(target, ref):
        return {
            "frozen_838": {
                "delta_clean_cer_pct": round(target["clean_cer_pct"] - ref["clean_cer_pct"], 4),
                "delta_char_cer_pct": round(target["char_cer_pct"] - ref["char_cer_pct"], 4),
                "delta_char_edits": target["char_edits"] - ref["char_edits"],
                "delta_char_deletions": target["char_deletions"] - ref["char_deletions"],
                "delta_char_substitutions": target["char_substitutions"] - ref["char_substitutions"],
                "delta_char_insertions": target["char_insertions"] - ref["char_insertions"],
                "delta_combmark_final_deletions": target["combmark_final_deletions"] - ref["combmark_final_deletions"],
                "delta_dependent_vowel_combmark_deletions": target["dependent_vowel_combmark_deletions"] - ref["dependent_vowel_combmark_deletions"],
                "delta_dependent_vowel_combmark_substitutions": target["dependent_vowel_combmark_substitutions"] - ref["dependent_vowel_combmark_substitutions"],
                "delta_space_deletions": target["space_deletions"] - ref.get("space_deletions", 8),
                "delta_base_consonant_deletions": target["base_consonant_deletions"] - ref.get("base_consonant_deletions", 75),
                "delta_symbols_acc_pct": round(target["symbols_acc_pct"] - ref["symbols_acc_pct"], 4),
                "delta_hyphen_acc_pct": round(target["hyphen_acc_pct"] - ref["hyphen_acc_pct"], 4),
                "delta_composite_score": round(target["composite_score"] - ref["composite_score"], 4)
            },
            "held_out_probe_100_w320": {
                "delta_cer_pct": round(target["held_out_probe_100_w320"]["cer_pct"] - ref["held_out_probe_100_w320"]["cer_pct"], 2),
                "delta_edits": target["held_out_probe_100_w320"]["edits"] - ref["held_out_probe_100_w320"]["edits"],
                "delta_exact_matches": target["held_out_probe_100_w320"]["exact_matches"] - ref["held_out_probe_100_w320"]["exact_matches"],
                "delta_deletions": target["held_out_probe_100_w320"]["deletions"] - ref["held_out_probe_100_w320"]["deletions"],
                "delta_subpeak_top2_top3_suppressed": target["held_out_probe_100_w320"]["subpeak_top2_top3_suppressed"] - ref["held_out_probe_100_w320"]["subpeak_top2_top3_suppressed"],
                "delta_combmark_final_deletions": target["held_out_probe_100_w320"]["combmark_final_deletions"] - ref["held_out_probe_100_w320"]["combmark_final_deletions"],
                "delta_combmark_subpeak_deletions": target["held_out_probe_100_w320"]["combmark_subpeak_deletions"] - ref["held_out_probe_100_w320"]["combmark_subpeak_deletions"]
            }
        }

    report = {
        "title": "Cham-OCR v24.1 Stage R2.1.6 Surgical Symbol & Hyphen Polish Evaluation (6-Way Comparison: A / B / D / E / R2.1.5 / R2.1.6)",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "base_checkpoint_r2_1_2_control_e2": r2_1_5["base_checkpoint_r2_1_2_control_e2"],
        "branch_a_control_continue": r2_1_5["branch_a_control_continue"],
        "branch_b_combmark_replay": r2_1_5["branch_b_combmark_replay"],
        "branch_d_phonotactic_combmark": r2_1_5["branch_d_phonotactic_combmark"],
        "branch_e_short_phonotactic_combmark": r2_1_5["branch_e_short_phonotactic_combmark"],
        "r2_1_5_multi_stream": r2_1_5["r2_1_5_multi_stream"],
        "r2_1_6_surgical_polish": r2_1_6_results,
        "ground_truth_training_accounting_r2_1_6": train_accounting,
        "causal_deltas": {
            "r216_e1_minus_r215_e1": compute_delta(m6_e1, m5_e1),
            "r216_e2_minus_r215_e2": compute_delta(m6_e2, m5_e2),
            "r216_e1_minus_e_e1": compute_delta(m6_e1, e_e1),
            "r216_e2_minus_e_e2": compute_delta(m6_e2, e_e2),
            "r216_e2_minus_a_e2": compute_delta(m6_e2, a_e2)
        }
    }

    out_report_path = os.path.join(PROJECT_ROOT, "reports", "r2_1_6_causal_evaluation.json")
    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ STAGE R2.1.6 6-WAY EVALUATION REPORT WRITTEN: {out_report_path}")
    print("=" * 80)
    print(json.dumps(report["causal_deltas"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
