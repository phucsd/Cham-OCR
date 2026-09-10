#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Evaluator for Advanced Cham OCR Benchmark Suite (90 Test Cases across 3 Directions).
Direction A (adv_001..adv_030): Đa ngữ xen kẽ / Code-Switching (Routing Accuracy & Bilingual CER)
Direction B (adv_031..adv_060): Bố cục 2 Cột A4 (Cross-Column Merge & Reading Order Inversions)
Direction C (adv_061..adv_090): Ảnh chụp di động thực địa (Shadow, Glare, Keystone, Motion Blur)
"""

import os
import sys
import json
import time
import math
import cv2
import numpy as np
from collections import defaultdict

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
    return levenshtein_distance(pred, gt) / max(len(gt), 1)

def compute_wer(pred, gt):
    w_pred = pred.split()
    w_gt = gt.split()
    if not w_gt and not w_pred:
        return 0.0
    if not w_gt:
        return 1.0
    return levenshtein_distance(w_pred, w_gt) / max(len(w_gt), 1)

def compute_iou(box1, box2):
    # box format: [ymin, xmin, ymax, xmax]
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter_area = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def kendall_tau_distance(seq1, seq2):
    """Computes Kendall Tau distance (number of pairwise inversions) between two orderings."""
    if len(seq1) <= 1 or len(seq2) <= 1:
        return 0.0, 0
    pos2 = {item: i for i, item in enumerate(seq2)}
    common = [x for x in seq1 if x in pos2]
    if len(common) <= 1:
        return 0.0, 0
    inversions = 0
    n = len(common)
    for i in range(n):
        for j in range(i + 1, n):
            if pos2[common[i]] > pos2[common[j]]:
                inversions += 1
    max_inversions = n * (n - 1) / 2
    norm_dist = inversions / max_inversions if max_inversions > 0 else 0.0
    return norm_dist, inversions

def main():
    print("=" * 80)
    print("🚀 COMPREHENSIVE ADVANCED BENCHMARK EVALUATION (90 SAMPLES ACROSS 3 DIRECTIONS)")
    print("   Models: Cham-DBNet Detector + Model v24 + AutoRoutingOCRWrapper + PaddleOCR vi")
    print("=" * 80)

    t_start = time.time()
    gt_path = os.path.join(BENCHMARK_DIR, "data", "benchmark_advanced_gt.json")
    img_dir = os.path.join(BENCHMARK_DIR, "data", "images_advanced")
    results_dir = os.path.join(BENCHMARK_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_output_path = os.path.join(results_dir, "evaluation_advanced_report.json")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_dataset = json.load(f)

    # Initialize models
    print("📦 Loading OCR Models...")
    det_model = app.get_det_model()
    cham_model = app.get_ocr_model("v24")
    auto_model = app.get_ocr_model("auto")
    viet_model = app.get_ocr_model("vi")
    print("✅ All models initialized successfully.\n")

    # Metrics Containers
    # Direction A Metrics
    dir_a_stats = {
        "count": 0,
        "total_gt_lines": 0,
        "total_pred_lines": 0,
        "detected_gt_lines": 0,
        "cham_lines_total": 0,
        "cham_routed_cham": 0,
        "cham_routed_viet": 0,
        "viet_lines_total": 0,
        "viet_routed_viet": 0,
        "viet_routed_cham": 0,
        "mixed_lines_total": 0,
        "mixed_routed_cham": 0,
        "mixed_routed_viet": 0,
        "cham_cer_list": [],
        "viet_cer_list": [],
        "mixed_cer_list": [],
        "overall_cer_list": [],
        "subtypes": defaultdict(lambda: {"count": 0, "cer_list": [], "routing_acc": []})
    }

    # Direction B Metrics
    dir_b_stats = {
        "count": 0,
        "total_gt_lines": 0,
        "total_pred_lines": 0,
        "detected_gt_lines": 0,
        "cross_column_merges": 0,
        "col1_lines_total": 0,
        "col1_lines_detected": 0,
        "col2_lines_total": 0,
        "col2_lines_detected": 0,
        "cer_list": [],
        "naive_inversion_dist_list": [],
        "column_aware_inversion_dist_list": [],
        "gutter_tiers": {
            "wide": {"count": 0, "merges": 0, "recall": [], "cer": [], "col_order_err": []},
            "medium": {"count": 0, "merges": 0, "recall": [], "cer": [], "col_order_err": []},
            "narrow": {"count": 0, "merges": 0, "recall": [], "cer": [], "col_order_err": []}
        }
    }

    # Direction C Metrics
    dir_c_stats = {
        "count": 0,
        "total_gt_lines": 0,
        "total_pred_lines": 0,
        "detected_gt_lines": 0,
        "overall_cer_list": [],
        "types": {
            "cast_shadow": {"count": 0, "gt_lines": 0, "det_lines": 0, "cer_list": []},
            "flash_glare": {"count": 0, "gt_lines": 0, "det_lines": 0, "glare_lost_lines": 0, "cer_list": []},
            "perspective_keystone": {"count": 0, "gt_lines": 0, "det_lines": 0, "cer_list": []},
            "motion_blur": {"count": 0, "gt_lines": 0, "det_lines": 0, "cer_list": []}
        }
    }

    sample_eval_records = []

    print(f"🧪 Evaluating {len(gt_dataset)} samples across 3 directions...")
    for idx, sample in enumerate(gt_dataset):
        sample_id = sample["sample_id"]
        direction = sample["direction"]
        img_path = os.path.join(img_dir, sample["filename"])
        img_bgr = cv2.imread(img_path)

        t_img_start = time.time()
        # 1. Run DBNet text line detection
        det_boxes, _ = det_model(img_bgr)
        # Sort boxes initially top-to-bottom
        pred_boxes = []
        for box in det_boxes:
            box_np = np.array(box).astype(np.int32)
            ymin = int(np.min(box_np[:, 1]))
            xmin = int(np.min(box_np[:, 0]))
            ymax = int(np.max(box_np[:, 1]))
            xmax = int(np.max(box_np[:, 0]))
            pred_boxes.append({
                "box": [ymin, xmin, ymax, xmax],
                "polygon": box_np.tolist(),
                "center_x": (xmin + xmax) / 2.0,
                "center_y": (ymin + ymax) / 2.0,
                "width": xmax - xmin,
                "height": ymax - ymin
            })

        # Match predicted boxes to GT lines (Greedy IoU & Center distance)
        gt_lines = sample["lines"]
        matched_gt_to_pred = {}
        matched_pred_to_gt = {}
        
        for g_idx, g_line in enumerate(gt_lines):
            g_box = g_line["box"]
            best_iou = 0.0
            best_p = -1
            for p_idx, p_item in enumerate(pred_boxes):
                if p_idx in matched_pred_to_gt:
                    continue
                iou = compute_iou(g_box, p_item["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_p = p_idx
            # If IoU > 0.15 or close center match
            if best_p != -1 and best_iou >= 0.15:
                matched_gt_to_pred[g_idx] = best_p
                matched_pred_to_gt[best_p] = g_idx
            else:
                # Fallback center overlap check
                g_cy = (g_box[0] + g_box[2]) / 2.0
                g_cx = (g_box[1] + g_box[3]) / 2.0
                best_dist = 999999
                best_cand = -1
                for p_idx, p_item in enumerate(pred_boxes):
                    if p_idx in matched_pred_to_gt:
                        continue
                    # Check vertical overlap
                    p_box = p_item["box"]
                    if min(g_box[2], p_box[2]) > max(g_box[0], p_box[0]):
                        dist = abs(g_cx - p_item["center_x"]) + abs(g_cy - p_item["center_y"])
                        if dist < best_dist and dist < 120:
                            best_dist = dist
                            best_cand = p_idx
                if best_cand != -1:
                    matched_gt_to_pred[g_idx] = best_cand
                    matched_pred_to_gt[best_cand] = g_idx

        # Crop predicted boxes for recognition
        crops = []
        for p_item in pred_boxes:
            box = p_item["box"]
            # Pad slightly
            y1 = max(0, box[0] - 2)
            y2 = min(img_bgr.shape[0], box[2] + 2)
            x1 = max(0, box[1] - 2)
            x2 = min(img_bgr.shape[1], box[3] + 2)
            crop = img_bgr[y1:y2, x1:x2]
            if crop.shape[0] < 5 or crop.shape[1] < 5:
                crop = np.zeros((20, 20, 3), dtype=np.uint8)
            crops.append(crop)

        # =====================================================================
        # DIRECTION A EVALUATION: BILINGUAL & AUTO-ROUTING
        # =====================================================================
        if direction == "A_bilingual":
            dir_a_stats["count"] += 1
            dir_a_stats["total_gt_lines"] += len(gt_lines)
            dir_a_stats["total_pred_lines"] += len(pred_boxes)
            dir_a_stats["detected_gt_lines"] += len(matched_gt_to_pred)

            # Run recognition with Auto-routing and Cham model
            auto_preds, _ = auto_model(crops) if crops else ([], 0.0)
            cham_preds, _ = cham_model(crops) if crops else ([], 0.0)

            sample_cer_list = []
            correct_routing_count = 0
            routed_total = 0

            for g_idx, g_line in enumerate(gt_lines):
                gt_text = g_line["text"]
                gt_script = g_line["script"]

                if g_idx in matched_gt_to_pred:
                    p_idx = matched_gt_to_pred[g_idx]
                    auto_text, auto_conf = auto_preds[p_idx]
                    cham_text, cham_conf = cham_preds[p_idx]

                    # Normalize Unicode if Cham
                    norm_pred = normalize_unicode(auto_text) if gt_script == "cham" else auto_text
                    norm_gt = normalize_unicode(gt_text) if gt_script == "cham" else gt_text
                    cer = compute_cer(norm_pred, norm_gt)
                    sample_cer_list.append(cer)

                    # Determine which route was taken: Cham or Viet
                    route_taken = "cham" if auto_text == cham_text else "vietnamese"

                    if gt_script == "cham":
                        dir_a_stats["cham_lines_total"] += 1
                        dir_a_stats["cham_cer_list"].append(cer)
                        if route_taken == "cham":
                            dir_a_stats["cham_routed_cham"] += 1
                            correct_routing_count += 1
                        else:
                            dir_a_stats["cham_routed_viet"] += 1
                        routed_total += 1
                    elif gt_script == "vietnamese":
                        dir_a_stats["viet_lines_total"] += 1
                        dir_a_stats["viet_cer_list"].append(cer)
                        if route_taken == "vietnamese":
                            dir_a_stats["viet_routed_viet"] += 1
                            correct_routing_count += 1
                        else:
                            dir_a_stats["viet_routed_cham"] += 1
                        routed_total += 1
                    else:  # mixed
                        dir_a_stats["mixed_lines_total"] += 1
                        dir_a_stats["mixed_cer_list"].append(cer)
                        if route_taken == "cham":
                            dir_a_stats["mixed_routed_cham"] += 1
                        else:
                            dir_a_stats["mixed_routed_viet"] += 1
                else:
                    # Missed line
                    sample_cer_list.append(1.0)
                    if gt_script == "cham":
                        dir_a_stats["cham_lines_total"] += 1
                        dir_a_stats["cham_cer_list"].append(1.0)
                    elif gt_script == "vietnamese":
                        dir_a_stats["viet_lines_total"] += 1
                        dir_a_stats["viet_cer_list"].append(1.0)
                    else:
                        dir_a_stats["mixed_lines_total"] += 1
                        dir_a_stats["mixed_cer_list"].append(1.0)

            avg_sample_cer = float(np.mean(sample_cer_list)) if sample_cer_list else 0.0
            dir_a_stats["overall_cer_list"].append(avg_sample_cer)
            subtype = sample["subtype"]
            dir_a_stats["subtypes"][subtype]["count"] += 1
            dir_a_stats["subtypes"][subtype]["cer_list"].append(avg_sample_cer)
            if routed_total > 0:
                dir_a_stats["subtypes"][subtype]["routing_acc"].append(correct_routing_count / routed_total)

            sample_eval_records.append({
                "sample_id": sample_id,
                "direction": direction,
                "subtype": subtype,
                "gt_lines": len(gt_lines),
                "det_lines": len(pred_boxes),
                "matched_lines": len(matched_gt_to_pred),
                "avg_cer": avg_sample_cer,
                "routing_acc": (correct_routing_count / routed_total) if routed_total > 0 else 0.0,
                "latency_sec": round(time.time() - t_img_start, 2)
            })

        # =====================================================================
        # DIRECTION B EVALUATION: 2-COLUMN A4 LAYOUT & READING ORDER
        # =====================================================================
        elif direction == "B_multicol":
            dir_b_stats["count"] += 1
            dir_b_stats["total_gt_lines"] += len(gt_lines)
            dir_b_stats["total_pred_lines"] += len(pred_boxes)
            dir_b_stats["detected_gt_lines"] += len(matched_gt_to_pred)

            gutter_px = sample["gutter_px"]
            if gutter_px >= 55:
                tier_name = "wide"
            elif gutter_px >= 35:
                tier_name = "medium"
            else:
                tier_name = "narrow"
            dir_b_stats["gutter_tiers"][tier_name]["count"] += 1

            # 1. Cross-Column Merge Detection:
            # Check if any predicted box spans across the gutter
            # Column 1 center ~ 300, Column 2 center ~ 800
            mid_x = sample["width"] / 2.0
            gutter_left = (mid_x - gutter_px / 2.0)
            gutter_right = (mid_x + gutter_px / 2.0)

            sample_merges = 0
            for p_item in pred_boxes:
                b = p_item["box"]
                # If box extends significantly into both columns across gutter
                if b[1] < gutter_left - 40 and b[3] > gutter_right + 40:
                    sample_merges += 1
                    dir_b_stats["cross_column_merges"] += 1
                    dir_b_stats["gutter_tiers"][tier_name]["merges"] += 1

            # 2. Recognition with Cham Model
            cham_preds, _ = cham_model(crops) if crops else ([], 0.0)

            sample_cer_list = []
            for g_idx, g_line in enumerate(gt_lines):
                gt_text = g_line["text"]
                col_id = g_line["column_id"]
                if col_id == 1:
                    dir_b_stats["col1_lines_total"] += 1
                else:
                    dir_b_stats["col2_lines_total"] += 1

                if g_idx in matched_gt_to_pred:
                    if col_id == 1:
                        dir_b_stats["col1_lines_detected"] += 1
                    else:
                        dir_b_stats["col2_lines_detected"] += 1
                    p_idx = matched_gt_to_pred[g_idx]
                    pred_text, _ = cham_preds[p_idx]
                    norm_pred = normalize_unicode(pred_text)
                    norm_gt = normalize_unicode(gt_text)
                    cer = compute_cer(norm_pred, norm_gt)
                    sample_cer_list.append(cer)
                else:
                    sample_cer_list.append(1.0)

            avg_sample_cer = float(np.mean(sample_cer_list)) if sample_cer_list else 0.0
            dir_b_stats["cer_list"].append(avg_sample_cer)
            dir_b_stats["gutter_tiers"][tier_name]["cer"].append(avg_sample_cer)
            recall = len(matched_gt_to_pred) / max(len(gt_lines), 1)
            dir_b_stats["gutter_tiers"][tier_name]["recall"].append(recall)

            # 3. Reading Order Evaluation:
            # GT reading sequence: [0, 1, 2, ..., N-1]
            gt_order_seq = list(range(len(gt_lines)))

            # Strategy 1: Naive Top-to-Bottom Sort (Standard Y-sort)
            naive_sorted_p_indices = sorted(range(len(pred_boxes)), key=lambda p: pred_boxes[p]["box"][0])
            naive_matched_gt_order = [matched_pred_to_gt[p] for p in naive_sorted_p_indices if p in matched_pred_to_gt]
            naive_dist, _ = kendall_tau_distance(naive_matched_gt_order, gt_order_seq)
            dir_b_stats["naive_inversion_dist_list"].append(naive_dist)

            # Strategy 2: Column-Aware Sort
            # Split boxes into Column 1 (center_x < mid_x) and Column 2 (center_x >= mid_x)
            col1_p_indices = sorted([p for p in range(len(pred_boxes)) if pred_boxes[p]["center_x"] < mid_x],
                                    key=lambda p: pred_boxes[p]["box"][0])
            col2_p_indices = sorted([p for p in range(len(pred_boxes)) if pred_boxes[p]["center_x"] >= mid_x],
                                    key=lambda p: pred_boxes[p]["box"][0])
            col_aware_p_indices = col1_p_indices + col2_p_indices
            col_aware_matched_gt_order = [matched_pred_to_gt[p] for p in col_aware_p_indices if p in matched_pred_to_gt]
            col_aware_dist, _ = kendall_tau_distance(col_aware_matched_gt_order, gt_order_seq)
            dir_b_stats["column_aware_inversion_dist_list"].append(col_aware_dist)
            dir_b_stats["gutter_tiers"][tier_name]["col_order_err"].append(col_aware_dist)

            sample_eval_records.append({
                "sample_id": sample_id,
                "direction": direction,
                "gutter_px": gutter_px,
                "tier": tier_name,
                "merges": sample_merges,
                "gt_lines": len(gt_lines),
                "det_lines": len(pred_boxes),
                "matched_lines": len(matched_gt_to_pred),
                "avg_cer": avg_sample_cer,
                "naive_inversion_dist": round(naive_dist, 4),
                "col_aware_inversion_dist": round(col_aware_dist, 4),
                "latency_sec": round(time.time() - t_img_start, 2)
            })

        # =====================================================================
        # DIRECTION C EVALUATION: MOBILE FIELD DEGRADATIONS
        # =====================================================================
        else:
            dir_c_stats["count"] += 1
            dir_c_stats["total_gt_lines"] += len(gt_lines)
            dir_c_stats["total_pred_lines"] += len(pred_boxes)
            dir_c_stats["detected_gt_lines"] += len(matched_gt_to_pred)

            deg_meta = sample["degradation"]
            deg_type = deg_meta["type"]
            dir_c_stats["types"][deg_type]["count"] += 1
            dir_c_stats["types"][deg_type]["gt_lines"] += len(gt_lines)
            dir_c_stats["types"][deg_type]["det_lines"] += len(matched_gt_to_pred)

            cham_preds, _ = cham_model(crops) if crops else ([], 0.0)

            sample_cer_list = []
            for g_idx, g_line in enumerate(gt_lines):
                gt_text = g_line["text"]
                if g_idx in matched_gt_to_pred:
                    p_idx = matched_gt_to_pred[g_idx]
                    pred_text, _ = cham_preds[p_idx]
                    norm_pred = normalize_unicode(pred_text)
                    norm_gt = normalize_unicode(gt_text)
                    cer = compute_cer(norm_pred, norm_gt)
                    sample_cer_list.append(cer)
                else:
                    sample_cer_list.append(1.0)
                    if deg_type == "flash_glare":
                        # Check if line was inside glare zone
                        gx = (g_line["box"][1] + g_line["box"][3]) / 2.0
                        gy = (g_line["box"][0] + g_line["box"][2]) / 2.0
                        cx, cy = deg_meta["center"]
                        r = deg_meta["radius"]
                        if math.hypot(gx - cx, gy - cy) <= r * 1.2:
                            dir_c_stats["types"]["flash_glare"]["glare_lost_lines"] += 1

            avg_sample_cer = float(np.mean(sample_cer_list)) if sample_cer_list else 0.0
            dir_c_stats["overall_cer_list"].append(avg_sample_cer)
            dir_c_stats["types"][deg_type]["cer_list"].append(avg_sample_cer)

            sample_eval_records.append({
                "sample_id": sample_id,
                "direction": direction,
                "deg_type": deg_type,
                "gt_lines": len(gt_lines),
                "det_lines": len(pred_boxes),
                "matched_lines": len(matched_gt_to_pred),
                "avg_cer": avg_sample_cer,
                "latency_sec": round(time.time() - t_img_start, 2)
            })

        if (idx + 1) % 15 == 0 or (idx + 1) == len(gt_dataset):
            print(f"  [Evaluation Progress] {idx + 1}/{len(gt_dataset)} samples evaluated.")

    total_eval_time = time.time() - t_start

    # =========================================================================
    # CONSTRUCT FINAL REPORT JSON
    # =========================================================================
    # Direction A aggregates
    cham_total = dir_a_stats["cham_lines_total"]
    cham_correct = dir_a_stats["cham_routed_cham"]
    cham_routing_acc = (cham_correct / cham_total * 100.0) if cham_total > 0 else 0.0

    viet_total = dir_a_stats["viet_lines_total"]
    viet_correct = dir_a_stats["viet_routed_viet"]
    viet_routing_acc = (viet_correct / viet_total * 100.0) if viet_total > 0 else 0.0

    total_routed = cham_total + viet_total
    overall_routing_acc = ((cham_correct + viet_correct) / total_routed * 100.0) if total_routed > 0 else 0.0

    dir_a_report = {
        "sample_count": dir_a_stats["count"],
        "total_gt_lines": dir_a_stats["total_gt_lines"],
        "detected_gt_lines": dir_a_stats["detected_gt_lines"],
        "detection_recall_pct": round(dir_a_stats["detected_gt_lines"] / max(dir_a_stats["total_gt_lines"], 1) * 100.0, 2),
        "overall_cer_pct": round(float(np.mean(dir_a_stats["overall_cer_list"])) * 100.0, 2),
        "cham_line_cer_pct": round(float(np.mean(dir_a_stats["cham_cer_list"])) * 100.0, 2) if dir_a_stats["cham_cer_list"] else 0.0,
        "viet_line_cer_pct": round(float(np.mean(dir_a_stats["viet_cer_list"])) * 100.0, 2) if dir_a_stats["viet_cer_list"] else 0.0,
        "mixed_line_cer_pct": round(float(np.mean(dir_a_stats["mixed_cer_list"])) * 100.0, 2) if dir_a_stats["mixed_cer_list"] else 0.0,
        "routing_performance": {
            "overall_accuracy_pct": round(overall_routing_acc, 2),
            "cham_routing_accuracy_pct": round(cham_routing_acc, 2),
            "cham_routed_to_cham": cham_correct,
            "cham_routed_to_viet": dir_a_stats["cham_routed_viet"],
            "viet_routing_accuracy_pct": round(viet_routing_acc, 2),
            "viet_routed_to_viet": viet_correct,
            "viet_routed_to_cham": dir_a_stats["viet_routed_cham"],
            "mixed_lines_count": dir_a_stats["mixed_lines_total"],
            "mixed_routed_cham": dir_a_stats["mixed_routed_cham"],
            "mixed_routed_viet": dir_a_stats["mixed_routed_viet"]
        },
        "subtypes": {
            k: {
                "count": v["count"],
                "cer_pct": round(float(np.mean(v["cer_list"])) * 100.0, 2) if v["cer_list"] else 0.0,
                "routing_acc_pct": round(float(np.mean(v["routing_acc"])) * 100.0, 2) if v["routing_acc"] else 0.0
            }
            for k, v in dir_a_stats["subtypes"].items()
        }
    }

    # Direction B aggregates
    dir_b_report = {
        "sample_count": dir_b_stats["count"],
        "total_gt_lines": dir_b_stats["total_gt_lines"],
        "detected_gt_lines": dir_b_stats["detected_gt_lines"],
        "detection_recall_pct": round(dir_b_stats["detected_gt_lines"] / max(dir_b_stats["total_gt_lines"], 1) * 100.0, 2),
        "overall_cer_pct": round(float(np.mean(dir_b_stats["cer_list"])) * 100.0, 2),
        "cross_column_merges": dir_b_stats["cross_column_merges"],
        "cross_column_merge_rate_pct": round(dir_b_stats["cross_column_merges"] / max(dir_b_stats["total_pred_lines"], 1) * 100.0, 2),
        "reading_order_inversion_distance": {
            "naive_top_to_bottom_sort": round(float(np.mean(dir_b_stats["naive_inversion_dist_list"])), 4),
            "column_aware_sort": round(float(np.mean(dir_b_stats["column_aware_inversion_dist_list"])), 4),
            "improvement_pct": round((1.0 - float(np.mean(dir_b_stats["column_aware_inversion_dist_list"])) /
                                     max(float(np.mean(dir_b_stats["naive_inversion_dist_list"])), 1e-6)) * 100.0, 2)
        },
        "gutter_tiers": {
            k: {
                "count": v["count"],
                "merges": v["merges"],
                "recall_pct": round(float(np.mean(v["recall"])) * 100.0, 2) if v["recall"] else 0.0,
                "cer_pct": round(float(np.mean(v["cer"])) * 100.0, 2) if v["cer"] else 0.0,
                "column_order_error": round(float(np.mean(v["col_order_err"])), 4) if v["col_order_err"] else 0.0
            }
            for k, v in dir_b_stats["gutter_tiers"].items()
        }
    }

    # Direction C aggregates
    dir_c_report = {
        "sample_count": dir_c_stats["count"],
        "total_gt_lines": dir_c_stats["total_gt_lines"],
        "detected_gt_lines": dir_c_stats["detected_gt_lines"],
        "detection_recall_pct": round(dir_c_stats["detected_gt_lines"] / max(dir_c_stats["total_gt_lines"], 1) * 100.0, 2),
        "overall_cer_pct": round(float(np.mean(dir_c_stats["overall_cer_list"])) * 100.0, 2),
        "degradation_breakdown": {
            k: {
                "count": v["count"],
                "gt_lines": v["gt_lines"],
                "det_lines": v["det_lines"],
                "recall_pct": round(v["det_lines"] / max(v["gt_lines"], 1) * 100.0, 2),
                "cer_pct": round(float(np.mean(v["cer_list"])) * 100.0, 2) if v["cer_list"] else 0.0,
                "extra": {"glare_lost_lines": v["glare_lost_lines"]} if "glare_lost_lines" in v else {}
            }
            for k, v in dir_c_stats["types"].items()
        }
    }

    final_report = {
        "metadata": {
            "title": "Advanced Cham OCR 3-Direction Comprehensive Evaluation Report",
            "total_samples": len(gt_dataset),
            "execution_time_seconds": round(total_eval_time, 2),
            "detector_model": "ch_PP-OCRv4_det_cham_infer",
            "cham_recognizer": "rec_cham_inference_v24",
            "viet_recognizer": "PaddleOCR-vi (PP-OCRv6)",
            "router_wrapper": "AutoRoutingOCRWrapper"
        },
        "direction_a_bilingual": dir_a_report,
        "direction_b_multicolumn": dir_b_report,
        "direction_c_mobile_degradations": dir_c_report,
        "per_sample_records": sample_eval_records
    }

    def default_converter(o):
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        elif isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2, default=default_converter)

    # =========================================================================
    # PRINT PRETTY REPORT
    # =========================================================================
    print("\n" + "=" * 85)
    print("📊 BÁO CÁO ĐÁNH GIÁ NĂNG LỰC TOÀN DIỆN 3 HƯỚNG NÂNG CAO")
    print(f"   Thời gian thực thi: {total_eval_time:.2f}s | Tổng mẫu: 90 ảnh (1,461 dòng Ground Truth)")
    print("=" * 85)

    print("\n[HƯỚNG A: ĐA NGỮ XEN KẼ & AUTO-ROUTING (30 MẪU)]")
    print(f"  • Tỷ lệ phát hiện dòng (Line Recall): {dir_a_report['detection_recall_pct']}% ({dir_a_stats['detected_gt_lines']}/{dir_a_stats['total_gt_lines']})")
    print(f"  • Độ chính xác phân luồng Auto-Routing: {dir_a_report['routing_performance']['overall_accuracy_pct']}%")
    print(f"    - Luồng Chăm: {dir_a_report['routing_performance']['cham_routing_accuracy_pct']}% ({cham_correct}/{cham_total} đúng Chăm)")
    print(f"    - Luồng Việt: {dir_a_report['routing_performance']['viet_routing_accuracy_pct']}% ({viet_correct}/{viet_total} đúng Việt)")
    print(f"  • CER Dòng Chăm: {dir_a_report['cham_line_cer_pct']}% | CER Dòng Việt: {dir_a_report['viet_line_cer_pct']}% | CER Hỗn Hợp: {dir_a_report['mixed_line_cer_pct']}%")
    print("  • Phân loại dạng thức:")
    for sub, vals in dir_a_report["subtypes"].items():
        print(f"    - {sub:<20}: CER {vals['cer_pct']}% | Routing Acc {vals['routing_acc_pct']}%")

    print("\n[HƯỚNG B: BỐ CỤC 2 CỘT A4 & TRẬT TỰ ĐỌC (30 MẪU)]")
    print(f"  • Tỷ lệ phát hiện dòng (Line Recall): {dir_b_report['detection_recall_pct']}% ({dir_b_stats['detected_gt_lines']}/{dir_b_stats['total_gt_lines']})")
    print(f"  • Tỷ lệ dính dòng xuyên rãnh cột (Cross-Column Merges): {dir_b_report['cross_column_merges']} lần ({dir_b_report['cross_column_merge_rate_pct']}%)")
    print(f"  • Sai số đảo trật tự đọc (Reading Order Inversion Distance):")
    print(f"    - Xếp Y ngây thơ (Naive Top-to-Bottom) : {dir_b_report['reading_order_inversion_distance']['naive_top_to_bottom_sort']} (loạn trật tự đọc giữa 2 cột)")
    print(f"    - Phân tách theo cột (Column-Aware Sort) : {dir_b_report['reading_order_inversion_distance']['column_aware_sort']} (cải thiện {dir_b_report['reading_order_inversion_distance']['improvement_pct']}%)")
    print("  • Phân tách theo độ rộng rãnh (Gutter Tiers):")
    for tier, vals in dir_b_report["gutter_tiers"].items():
        print(f"    - {tier:<8} (10 mẫu): Dính rãnh: {vals['merges']} | Recall: {vals['recall_pct']}% | CER: {vals['cer_pct']}% | Sai số thứ tự: {vals['column_order_error']}")

    print("\n[HƯỚNG C: ẢNH THỰC ĐỊA DI ĐỘNG (30 MẪU)]")
    print(f"  • Tỷ lệ phát hiện dòng tổng thể: {dir_c_report['detection_recall_pct']}% | CER tổng thể: {dir_c_report['overall_cer_pct']}%")
    print("  • Bóc tách từng dạng suy biến thực tế:")
    for deg, vals in dir_c_report["degradation_breakdown"].items():
        extra_str = f" (Mất hoàn toàn do lóa: {vals['extra']['glare_lost_lines']} dòng)" if "glare_lost_lines" in vals.get("extra", {}) else ""
        print(f"    - {deg:<22} ({vals['count']} mẫu): Recall: {vals['recall_pct']:>5.1f}% | CER: {vals['cer_pct']:>5.1f}%{extra_str}")

    print("=" * 85)
    print(f"✅ Báo cáo JSON đầy đủ đã lưu tại: {report_output_path}")
    print("=" * 85)

if __name__ == "__main__":
    main()
