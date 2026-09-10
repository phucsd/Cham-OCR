#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Pipeline Evaluation on 200 Benchmark Test Cases.
Evaluates Cham-DBNet Detector + Line Merging + OCR v24 Recognizer + Logical Order Normalizer.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import difflib
from collections import defaultdict, Counter

# Force UTF-8 stdout encoding on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Environment variables for PaddleOCR on Windows CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# NumPy 2.x compatibility monkeypatch
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str]
    }
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'typeDict'): np.typeDict = {}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)
STUDIO_DIR = os.path.join(PROJECT_ROOT, "ocr-studio")

sys.path.insert(0, STUDIO_DIR)
import app
from scripts.generate_data import normalize_unicode

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def compute_cer(pred, gt):
    if not gt and not pred:
        return 0.0
    if not gt:
        return 1.0
    dist = levenshtein_distance(pred, gt)
    return dist / max(len(gt), 1)

def compute_wer(pred, gt):
    w_pred = pred.split()
    w_gt = gt.split()
    if not w_gt and not w_pred:
        return 0.0
    if not w_gt:
        return 1.0
    dist = levenshtein_distance(w_pred, w_gt)
    return dist / max(len(w_gt), 1)

def find_char_confusions(pred, gt, confusion_counter):
    matcher = difflib.SequenceMatcher(None, gt, pred)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            sub_gt = gt[i1:i2]
            sub_pred = pred[j1:j2]
            confusion_counter[(sub_gt, sub_pred)] += 1
        elif tag == 'delete':
            # GT char dropped in pred
            sub_gt = gt[i1:i2]
            confusion_counter[(sub_gt, "[DELETED]")] += 1
        elif tag == 'insert':
            # Hallucinated / extra char in pred
            sub_pred = pred[j1:j2]
            confusion_counter[("[INSERTED]", sub_pred)] += 1

def run_evaluation(gt_file, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    print("🚀 Initializing OCR Pipeline models...")
    t_init = time.time()
    det_model = app.get_det_model()
    ocr_model = app.get_ocr_model("v24")
    print(f"✅ Models initialized in {time.time() - t_init:.2f}s")

    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    print(f"\n🧪 Starting evaluation across {len(gt_data)} benchmark samples...")

    per_sample_results = []
    level_stats = defaultdict(lambda: {
        "count": 0,
        "total_gt_lines": 0,
        "total_pred_lines": 0,
        "exact_line_matches": 0,
        "cer_list": [],
        "wer_list": [],
        "det_latency_list": [],
        "rec_latency_list": [],
        "total_latency_list": [],
        "conf_list": []
    })

    confusion_counter = Counter()

    for idx, sample in enumerate(gt_data, 1):
        filename = sample["filename"]
        level = sample["level"]
        level_name = sample["level_name"]
        img_path = os.path.join(BENCHMARK_DIR, sample["image_path"])

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read image: {img_path}")
            continue

        t0 = time.time()
        # 1. Line detection via Cham-DBNet
        try:
            crops, metas = app.segment_lines_dbnet(img, det_model)
        except Exception as e:
            print(f"❌ DBNet segmentation error on {filename}: {e}")
            crops, metas = [], []

        det_time = time.time() - t0

        # Fallback to valley segmentation if DBNet missed
        if not crops:
            t_fb = time.time()
            _, coords = app.segment_lines_valleys(img, window_size=25, min_dist=sample["gap_px"])
            crops, metas, _ = app.segment_lines_advanced(img, coords, ocr_model)
            det_time += (time.time() - t_fb)

        # 2. Text recognition via v24
        t_rec0 = time.time()
        if crops:
            rec_res, _ = ocr_model(crops)
        else:
            rec_res = []
        rec_time = time.time() - t_rec0
        total_time = det_time + rec_time

        # Extract predictions with unicode normalization
        pred_lines = []
        conf_values = []
        for r in rec_res:
            raw_text, conf = r if r else ("", 0.0)
            norm_text = normalize_unicode(raw_text.strip())
            pred_lines.append(norm_text)
            conf_values.append(float(conf))

        gt_lines = [l["text"].strip() for l in sample["lines"]]
        num_gt = len(gt_lines)
        num_pred = len(pred_lines)

        # Document-level concatenation evaluation
        gt_doc = "\n".join(gt_lines)
        pred_doc = "\n".join(pred_lines)
        doc_cer = compute_cer(pred_doc, gt_doc)
        doc_wer = compute_wer(pred_doc, gt_doc)

        # Line-by-line alignment (using minimum distance matching)
        line_evals = []
        matched_gt = set()
        matched_pred = set()
        
        # Greedy best-match alignment
        cost_matrix = np.zeros((num_gt, max(1, num_pred)), dtype=np.float32)
        for g_i, g_t in enumerate(gt_lines):
            for p_i, p_t in enumerate(pred_lines):
                cost_matrix[g_i, p_i] = compute_cer(p_t, g_t)

        if num_pred > 0:
            for _ in range(min(num_gt, num_pred)):
                min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                g_i, p_i = min_idx
                if cost_matrix[g_i, p_i] > 100:
                    break
                matched_gt.add(g_i)
                matched_pred.add(p_i)
                c_val = float(cost_matrix[g_i, p_i])
                cost_matrix[g_i, :] = 999.0
                cost_matrix[:, p_i] = 999.0
                
                is_exact = bool(pred_lines[p_i] == gt_lines[g_i])
                find_char_confusions(pred_lines[p_i], gt_lines[g_i], confusion_counter)
                line_evals.append({
                    "gt_idx": int(g_i),
                    "pred_idx": int(p_i),
                    "gt": gt_lines[g_i],
                    "pred": pred_lines[p_i],
                    "confidence": float(conf_values[p_i]),
                    "cer": round(c_val, 4),
                    "exact": is_exact
                })

        # Process unmatched GT lines (deletions)
        for g_i in range(num_gt):
            if g_i not in matched_gt:
                find_char_confusions("", gt_lines[g_i], confusion_counter)
                line_evals.append({
                    "gt_idx": int(g_i),
                    "pred_idx": None,
                    "gt": gt_lines[g_i],
                    "pred": "",
                    "confidence": 0.0,
                    "cer": 1.0,
                    "exact": False
                })

        exact_line_count = sum(1 for le in line_evals if le["exact"])
        avg_sample_conf = float(np.mean(conf_values)) if conf_values else 0.0

        sample_res = {
            "sample_id": sample["sample_id"],
            "filename": filename,
            "level": level,
            "level_name": level_name,
            "gap_px": sample["gap_px"],
            "wave_amp": sample["wave_amp"],
            "perspective_factor": sample["perspective_factor"],
            "num_gt_lines": num_gt,
            "num_pred_lines": num_pred,
            "exact_line_matches": exact_line_count,
            "doc_cer": round(doc_cer, 4),
            "doc_wer": round(doc_wer, 4),
            "avg_confidence": round(avg_sample_conf, 4),
            "det_time_sec": round(det_time, 4),
            "rec_time_sec": round(rec_time, 4),
            "total_time_sec": round(total_time, 4),
            "line_evals": line_evals
        }
        per_sample_results.append(sample_res)

        # Accumulate level stats
        lvl_s = level_stats[level]
        lvl_s["count"] += 1
        lvl_s["total_gt_lines"] += num_gt
        lvl_s["total_pred_lines"] += num_pred
        lvl_s["exact_line_matches"] += exact_line_count
        lvl_s["cer_list"].append(doc_cer)
        lvl_s["wer_list"].append(doc_wer)
        lvl_s["det_latency_list"].append(det_time)
        lvl_s["rec_latency_list"].append(rec_time)
        lvl_s["total_latency_list"].append(total_time)
        if avg_sample_conf > 0:
            lvl_s["conf_list"].append(avg_sample_conf)

        if idx % 20 == 0 or idx == len(gt_data):
            print(f"  [{idx:03d}/{len(gt_data)}] {filename} ({level_name}) | CER: {doc_cer*100:5.2f}% | Latency: {total_time:.2f}s | Lines: {num_pred}/{num_gt}")

    # Aggregate level summaries
    summary_by_level = {}
    for lvl in sorted(level_stats.keys()):
        s = level_stats[lvl]
        mean_cer = float(np.mean(s["cer_list"])) * 100.0 if s["cer_list"] else 0.0
        mean_wer = float(np.mean(s["wer_list"])) * 100.0 if s["wer_list"] else 0.0
        exact_pct = (s["exact_line_matches"] / max(1, s["total_gt_lines"])) * 100.0
        det_recall = min(100.0, (s["total_pred_lines"] / max(1, s["total_gt_lines"])) * 100.0)
        mean_det_lat = float(np.mean(s["det_latency_list"])) if s["det_latency_list"] else 0.0
        mean_rec_lat = float(np.mean(s["rec_latency_list"])) if s["rec_latency_list"] else 0.0
        mean_tot_lat = float(np.mean(s["total_latency_list"])) if s["total_latency_list"] else 0.0
        mean_conf = float(np.mean(s["conf_list"])) if s["conf_list"] else 0.0

        summary_by_level[lvl] = {
            "level": lvl,
            "level_name": gt_data[(lvl-1)*40]["level_name"],
            "samples": s["count"],
            "total_gt_lines": s["total_gt_lines"],
            "total_pred_lines": s["total_pred_lines"],
            "line_detection_rate_pct": round(det_recall, 2),
            "mean_cer_pct": round(mean_cer, 2),
            "mean_wer_pct": round(mean_wer, 2),
            "exact_line_match_pct": round(exact_pct, 2),
            "avg_confidence": round(mean_conf, 4),
            "avg_det_latency_sec": round(mean_det_lat, 3),
            "avg_rec_latency_sec": round(mean_rec_lat, 3),
            "avg_total_latency_sec": round(mean_tot_lat, 3)
        }

    # Overall summary
    all_cer = [r["doc_cer"] * 100.0 for r in per_sample_results]
    all_wer = [r["doc_wer"] * 100.0 for r in per_sample_results]
    all_tot_lat = [r["total_time_sec"] for r in per_sample_results]
    all_gt_lines = sum(s["total_gt_lines"] for s in level_stats.values())
    all_exact_lines = sum(s["exact_line_matches"] for s in level_stats.values())
    all_pred_lines = sum(s["total_pred_lines"] for s in level_stats.values())

    top_confusions = []
    for (gt_c, pred_c), cnt in confusion_counter.most_common(25):
        top_confusions.append({
            "ground_truth": gt_c,
            "prediction": pred_c,
            "occurrences": cnt
        })

    overall_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_pipeline": {
            "detector": "ch_PP-OCRv4_det_cham_infer (Cham-DBNet)",
            "recognizer": "rec_cham_inference_v24 (Mobile PPLCNetV3 Cham V24)",
            "normalizer": "normalize_unicode (Logical Order)"
        },
        "overall_metrics": {
            "total_samples": len(per_sample_results),
            "total_lines": all_gt_lines,
            "total_lines_detected": all_pred_lines,
            "line_detection_rate_pct": round((all_pred_lines / max(1, all_gt_lines)) * 100.0, 2),
            "overall_mean_cer_pct": round(float(np.mean(all_cer)), 2),
            "overall_median_cer_pct": round(float(np.median(all_cer)), 2),
            "overall_mean_wer_pct": round(float(np.mean(all_wer)), 2),
            "overall_exact_line_match_pct": round((all_exact_lines / max(1, all_gt_lines)) * 100.0, 2),
            "overall_avg_latency_sec": round(float(np.mean(all_tot_lat)), 3),
            "fps_pages_per_sec": round(1.0 / max(0.001, float(np.mean(all_tot_lat))), 2)
        },
        "by_difficulty_level": summary_by_level,
        "top_error_confusions": top_confusions,
        "per_sample_details": per_sample_results
    }

    def default_converter(o):
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    report_json_path = os.path.join(results_dir, "evaluation_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(overall_report, f, ensure_ascii=False, indent=2, default=default_converter)

    # Print markdown summary table
    print("\n" + "="*80)
    print("📊 BẢNG TỔNG HỢP ĐÁNH GIÁ 200 BÀI TEST CHÂM-OCR (V24 + CHAM-DBNET)")
    print("="*80)
    print(f"{'Cấp Độ':<32} | {'Số Ảnh':<6} | {'Dòng GT':<7} | {'Det Rate':<9} | {'CER (%)':<8} | {'WER (%)':<8} | {'Exact (%)':<9} | {'Độ trễ':<7}")
    print("-" * 102)
    for lvl in sorted(summary_by_level.keys()):
        s = summary_by_level[lvl]
        print(f"{s['level_name']:<32} | {s['samples']:<6} | {s['total_gt_lines']:<7} | {s['line_detection_rate_pct']:<8}% | {s['mean_cer_pct']:<7}% | {s['mean_wer_pct']:<7}% | {s['exact_line_match_pct']:<8}% | {s['avg_total_latency_sec']:<5.2f}s")
    print("-" * 102)
    ov = overall_report['overall_metrics']
    print(f"{'TOÀN BỘ 200 BÀI TEST':<32} | {ov['total_samples']:<6} | {ov['total_lines']:<7} | {ov['line_detection_rate_pct']:<8}% | {ov['overall_mean_cer_pct']:<7}% | {ov['overall_mean_wer_pct']:<7}% | {ov['overall_exact_line_match_pct']:<8}% | {ov['overall_avg_latency_sec']:<5.2f}s")
    print("="*80)

    print("\n🔍 TOP 10 LỖI SAI KÝ TỰ PHỔ BIẾN NHẤT:")
    for i, err in enumerate(top_confusions[:10], 1):
        print(f"  {i}. GT '{err['ground_truth']}' ➔ Pred '{err['prediction']}' ({err['occurrences']} lần)")

    print(f"\n💾 Toàn bộ kết quả chi tiết đã được lưu tại: {report_json_path}")
    return report_json_path

if __name__ == "__main__":
    gt_file_path = os.path.join(BENCHMARK_DIR, "data", "benchmark_gt.json")
    results_dir_path = os.path.join(BENCHMARK_DIR, "results")
    run_evaluation(gt_file_path, results_dir_path)
