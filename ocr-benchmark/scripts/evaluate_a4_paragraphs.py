#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A4 Paragraph & Sentence Flow Evaluation Script.
Evaluates:
1. Long-document OCR accuracy (Line Detection, CER, WER, Latency on full-page A4)
2. Paragraph & Sentence Flow Reconstruction (Automatic detection of soft line continuation vs hard breaks)
3. Multi-style robustness (Mixing Bold headings & Regular body text)
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
from paragraph_flow import reconstruct_paragraph_flow
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
    return levenshtein_distance(pred, gt) / max(len(gt), 1)

def compute_wer(pred, gt):
    w_pred = pred.split()
    w_gt = gt.split()
    if not w_gt and not w_pred:
        return 0.0
    if not w_gt:
        return 1.0
    return levenshtein_distance(w_pred, w_gt) / max(len(w_gt), 1)

def run_a4_evaluation(gt_file, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    print("🚀 Initializing OCR Pipeline models (Cham-DBNet + Model v24)...")
    t_init = time.time()
    det_model = app.get_det_model()
    ocr_model = app.get_ocr_model("v24")
    print(f"✅ Models initialized in {time.time() - t_init:.2f}s")

    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    print(f"\n🧪 Starting evaluation on {len(gt_data)} A4 multi-paragraph benchmark samples...")

    per_sample_results = []
    level_stats = defaultdict(lambda: {
        "count": 0,
        "total_gt_lines": 0,
        "total_pred_lines": 0,
        "total_gt_paras": 0,
        "total_pred_paras": 0,
        "cer_list": [],
        "wer_list": [],
        "para_cer_list": [],
        "para_wer_list": [],
        "cont_tp": 0,
        "cont_fp": 0,
        "cont_fn": 0,
        "cont_tn": 0,
        "det_latency_list": [],
        "rec_latency_list": [],
        "total_latency_list": [],
        "conf_list": []
    })

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
        # 1. Line Detection via Cham-DBNet
        try:
            crops, metas = app.segment_lines_dbnet(img, det_model)
        except Exception as e:
            crops, metas = [], []

        det_time = time.time() - t0

        # Fallback to valley segmentation if DBNet missed
        if not crops:
            t_fb = time.time()
            _, coords = app.segment_lines_valleys(img, window_size=25, min_dist=sample.get("gap_px", 12))
            crops, metas, _ = app.segment_lines_advanced(img, coords, ocr_model)
            det_time += (time.time() - t_fb)

        # 2. Line Recognition via v24
        t_rec0 = time.time()
        if crops:
            rec_res, _ = ocr_model(crops)
        else:
            rec_res = []
        rec_time = time.time() - t_rec0

        # Format detected lines
        pred_line_objects = []
        pred_lines_text = []
        conf_values = []
        for i, (crop, meta, res) in enumerate(zip(crops, metas, rec_res)):
            raw_text, conf = res if res else ("", 0.0)
            norm_text = normalize_unicode(raw_text.strip())
            bbox = meta.get("bbox", [0, 0, img.shape[1], 30])
            pred_line_objects.append({
                "line_id": i,
                "text": norm_text,
                "bbox": bbox,
                "confidence": float(conf)
            })
            pred_lines_text.append(norm_text)
            conf_values.append(float(conf))

        # 3. Automatic Paragraph & Sentence-Flow Reconstruction
        t_flow0 = time.time()
        pred_paragraphs, pred_boundaries = reconstruct_paragraph_flow(pred_line_objects)
        flow_time = time.time() - t_flow0
        total_time = det_time + rec_time + flow_time

        gt_lines = [l["text"].strip() for l in sample["lines"]]
        gt_paras = [p["text"].strip() for p in sample["paragraphs"]]
        gt_boundaries = sample["flow_boundaries"]

        num_gt_l = len(gt_lines)
        num_pred_l = len(pred_lines_text)

        # Document CER/WER
        doc_cer = compute_cer("\n".join(pred_lines_text), "\n".join(gt_lines))
        doc_wer = compute_wer("\n".join(pred_lines_text), "\n".join(gt_lines))

        # Paragraph CER/WER
        pred_para_texts = [p["text"].strip() for p in pred_paragraphs]
        para_cer = compute_cer("\n\n".join(pred_para_texts), "\n\n".join(gt_paras))
        para_wer = compute_wer("\n\n".join(pred_para_texts), "\n\n".join(gt_paras))

        # Evaluate Paragraph Flow / Continuation Detection
        # Compare ground truth continuation points with predicted continuation points
        sample_cont_tp = 0
        sample_cont_fp = 0
        sample_cont_fn = 0
        sample_cont_tn = 0

        # When line count matches or is close, evaluate boundaries sequentially
        eval_len = min(len(gt_boundaries), len(pred_boundaries))
        for b_idx in range(eval_len):
            gt_is_cont = gt_boundaries[b_idx]["is_continuation"]
            pred_is_cont = pred_boundaries[b_idx]["is_continuation"]

            if gt_is_cont and pred_is_cont:
                sample_cont_tp += 1
            elif not gt_is_cont and pred_is_cont:
                sample_cont_fp += 1
            elif gt_is_cont and not pred_is_cont:
                sample_cont_fn += 1
            else:
                sample_cont_tn += 1

        # Calculate sample continuation F1
        denom_f1 = (2 * sample_cont_tp + sample_cont_fp + sample_cont_fn)
        sample_cont_f1 = (2.0 * sample_cont_tp / denom_f1) if denom_f1 > 0 else 0.0

        sample_res = {
            "sample_id": sample["sample_id"],
            "filename": filename,
            "level": level,
            "level_name": level_name,
            "num_gt_lines": num_gt_l,
            "num_pred_lines": num_pred_l,
            "num_gt_paras": len(gt_paras),
            "num_pred_paras": len(pred_paragraphs),
            "doc_cer": round(doc_cer, 4),
            "doc_wer": round(doc_wer, 4),
            "para_cer": round(para_cer, 4),
            "para_wer": round(para_wer, 4),
            "continuation_f1": round(sample_cont_f1, 4),
            "avg_confidence": round(float(np.mean(conf_values)), 4) if conf_values else 0.0,
            "det_time_sec": round(det_time, 3),
            "rec_time_sec": round(rec_time, 3),
            "flow_time_sec": round(flow_time, 4),
            "total_time_sec": round(total_time, 3)
        }
        per_sample_results.append(sample_res)

        # Accumulate level stats
        s = level_stats[level]
        s["count"] += 1
        s["total_gt_lines"] += num_gt_l
        s["total_pred_lines"] += num_pred_l
        s["total_gt_paras"] += len(gt_paras)
        s["total_pred_paras"] += len(pred_paragraphs)
        s["cer_list"].append(doc_cer)
        s["wer_list"].append(doc_wer)
        s["para_cer_list"].append(para_cer)
        s["para_wer_list"].append(para_wer)
        s["cont_tp"] += sample_cont_tp
        s["cont_fp"] += sample_cont_fp
        s["cont_fn"] += sample_cont_fn
        s["cont_tn"] += sample_cont_tn
        s["det_latency_list"].append(det_time)
        s["rec_latency_list"].append(rec_time)
        s["total_latency_list"].append(total_time)
        if conf_values:
            s["conf_list"].append(float(np.mean(conf_values)))

        if idx % 20 == 0 or idx == len(gt_data):
            print(f"  [{idx:03d}/{len(gt_data)}] {filename} ({level_name}) | Line CER: {doc_cer*100:5.2f}% | Para CER: {para_cer*100:5.2f}% | Flow F1: {sample_cont_f1*100:5.1f}% | Time: {total_time:.2f}s | Lines: {num_pred_l}/{num_gt_l}")

    # Summary by level
    summary_by_level = {}
    for lvl in sorted(level_stats.keys()):
        s = level_stats[lvl]
        tp = s["cont_tp"]
        fp = s["cont_fp"]
        fn = s["cont_fn"]
        tn = s["cont_tn"]
        p_prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        p_rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        p_f1 = (2 * p_prec * p_rec / (p_prec + p_rec)) if (p_prec + p_rec) > 0 else 0.0
        p_acc = ((tp + tn) / max(1, tp + tn + fp + fn)) * 100.0

        summary_by_level[lvl] = {
            "level": lvl,
            "level_name": gt_data[(lvl-1)*40]["level_name"],
            "samples": s["count"],
            "total_gt_lines": s["total_gt_lines"],
            "total_pred_lines": s["total_pred_lines"],
            "line_detection_rate_pct": round(min(100.0, (s["total_pred_lines"] / max(1, s["total_gt_lines"])) * 100.0), 2),
            "total_gt_paras": s["total_gt_paras"],
            "total_pred_paras": s["total_pred_paras"],
            "mean_line_cer_pct": round(float(np.mean(s["cer_list"])) * 100.0, 2),
            "mean_para_cer_pct": round(float(np.mean(s["para_cer_list"])) * 100.0, 2),
            "mean_para_wer_pct": round(float(np.mean(s["para_wer_list"])) * 100.0, 2),
            "flow_continuation_precision_pct": round(p_prec, 2),
            "flow_continuation_recall_pct": round(p_rec, 2),
            "flow_continuation_f1_pct": round(p_f1, 2),
            "flow_boundary_accuracy_pct": round(p_acc, 2),
            "avg_confidence": round(float(np.mean(s["conf_list"])) if s["conf_list"] else 0.0, 4),
            "avg_latency_sec": round(float(np.mean(s["total_latency_list"])), 2)
        }

    # Overall Metrics
    all_cer = [r["doc_cer"] * 100.0 for r in per_sample_results]
    all_para_cer = [r["para_cer"] * 100.0 for r in per_sample_results]
    all_para_wer = [r["para_wer"] * 100.0 for r in per_sample_results]
    all_tot_lat = [r["total_time_sec"] for r in per_sample_results]
    all_gt_l = sum(s["total_gt_lines"] for s in level_stats.values())
    all_pred_l = sum(s["total_pred_lines"] for s in level_stats.values())
    all_gt_p = sum(s["total_gt_paras"] for s in level_stats.values())
    all_pred_p = sum(s["total_pred_paras"] for s in level_stats.values())
    tot_tp = sum(s["cont_tp"] for s in level_stats.values())
    tot_fp = sum(s["cont_fp"] for s in level_stats.values())
    tot_fn = sum(s["cont_fn"] for s in level_stats.values())
    tot_tn = sum(s["cont_tn"] for s in level_stats.values())

    ov_prec = (tot_tp / (tot_tp + tot_fp)) * 100.0 if (tot_tp + tot_fp) > 0 else 0.0
    ov_rec = (tot_tp / (tot_tp + tot_fn)) * 100.0 if (tot_tp + tot_fn) > 0 else 0.0
    ov_f1 = (2 * ov_prec * ov_rec / (ov_prec + ov_rec)) if (ov_prec + ov_rec) > 0 else 0.0
    ov_acc = ((tot_tp + tot_tn) / max(1, tot_tp + tot_tn + tot_fp + tot_fn)) * 100.0

    overall_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_type": "A4 Full-Page Long Document Benchmark (Samples 201 - 400)",
        "model_pipeline": {
            "detector": "ch_PP-OCRv4_det_cham_infer (Cham-DBNet)",
            "recognizer": "rec_cham_inference_v24 (Mobile PPLCNetV3 Cham V24)",
            "normalizer": "normalize_unicode (Logical Order)",
            "flow_reconstructor": "paragraph_flow.py (reconstruct_paragraph_flow)"
        },
        "overall_metrics": {
            "total_samples": len(per_sample_results),
            "total_gt_lines": all_gt_l,
            "total_lines_detected": all_pred_l,
            "line_detection_rate_pct": round(min(100.0, (all_pred_l / max(1, all_gt_l)) * 100.0), 2),
            "total_gt_paragraphs": all_gt_p,
            "total_pred_paragraphs": all_pred_p,
            "overall_mean_line_cer_pct": round(float(np.mean(all_cer)), 2),
            "overall_mean_para_cer_pct": round(float(np.mean(all_para_cer)), 2),
            "overall_mean_para_wer_pct": round(float(np.mean(all_para_wer)), 2),
            "flow_continuation_f1_pct": round(ov_f1, 2),
            "flow_continuation_precision_pct": round(ov_prec, 2),
            "flow_continuation_recall_pct": round(ov_rec, 2),
            "flow_boundary_accuracy_pct": round(ov_acc, 2),
            "overall_avg_latency_sec": round(float(np.mean(all_tot_lat)), 2)
        },
        "by_difficulty_level": summary_by_level,
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

    report_json_path = os.path.join(results_dir, "evaluation_a4_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(overall_report, f, ensure_ascii=False, indent=2, default=default_converter)

    # Print summary table
    print("\n" + "="*95)
    print("📊 BẢNG TỔNG HỢP ĐÁNH GIÁ 200 TRANG A4 & THUẬT TOÁN NỐI ĐOẠN VĂN BẢN")
    print("="*95)
    print(f"{'Cấp Độ A4':<36} | {'Det Rate':<9} | {'Line CER':<9} | {'Para CER':<9} | {'Flow F1':<8} | {'Flow Acc':<9} | {'Độ trễ':<6}")
    print("-" * 95)
    for lvl in sorted(summary_by_level.keys()):
        s = summary_by_level[lvl]
        print(f"{s['level_name']:<36} | {s['line_detection_rate_pct']:<8}% | {s['mean_line_cer_pct']:<8}% | {s['mean_para_cer_pct']:<8}% | {s['flow_continuation_f1_pct']:<7}% | {s['flow_boundary_accuracy_pct']:<8}% | {s['avg_latency_sec']:<5.1f}s")
    print("-" * 95)
    ov = overall_report["overall_metrics"]
    print(f"{'TOÀN BỘ 200 TRANG A4':<36} | {ov['line_detection_rate_pct']:<8}% | {ov['overall_mean_line_cer_pct']:<8}% | {ov['overall_mean_para_cer_pct']:<8}% | {ov['flow_continuation_f1_pct']:<7}% | {ov['flow_boundary_accuracy_pct']:<8}% | {ov['overall_avg_latency_sec']:<5.1f}s")
    print("="*95)
    print(f"\n💾 Báo cáo A4 chi tiết đã được lưu tại: {report_json_path}")
    return report_json_path

if __name__ == "__main__":
    gt_file_path = os.path.join(BENCHMARK_DIR, "data", "benchmark_a4_gt.json")
    results_dir_path = os.path.join(BENCHMARK_DIR, "results")
    run_a4_evaluation(gt_file_path, results_dir_path)
