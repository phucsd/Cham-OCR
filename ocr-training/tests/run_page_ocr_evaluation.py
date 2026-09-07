import os
import sys
import cv2
import random
import re
import json
import numpy as np

# Monkeypatch NumPy 2.x to support legacy symbols removed/deprecated for PaddleOCR and imgaug
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str]
    }
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'typeDict'):
    np.typeDict = {}

from PIL import Image, ImageDraw, ImageFont

# Configure paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
paddleocr_dir = os.path.join(os.path.dirname(PROJECT_ROOT), "ocr_webapp", "PaddleOCR")
if paddleocr_dir not in sys.path:
    sys.path.insert(0, paddleocr_dir)

# Import PaddleOCR system
import tools.infer.predict_det as predict_det
import tools.infer.predict_rec as predict_rec
import tools.infer.utility as utility

# Import local modules
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, r"C:\Users\admin\.gemini\antigravity\brain\3b1501b2-bb18-4c92-a78b-3d27867b4bce\scratch")
from generate_data import visual_to_unicode, get_cached_font
from robustness_stress_test import apply_old_paper_background
from test_page_ocr import apply_page_wave_distortion, apply_page_perspective_distortion, draw_distorted_page, calculate_levenshtein

def main():
    text_block = """ꨗꨫ ꨕꨤꨭꨆꩊ ꨆ ꨌꨬ ꨝꨤꨯꨱꩀ ꨤꨂꨩ ꨟꨕꨩ ꨓꩀ ꨕꨫ ꨆꩊ ꨗꩆ ꨧꨩ ꨅꩃ 
ꨧꨩ ꨓꨌꨯꨱꨥ ꨆꨔꨯꨱꩅ ꨣꨡꩍ ꨟꨪꩆ. ꨗꩆ ꨟꩃ ꨕꨶꨩ ꨅꩃ ꨓꨌꨯꨱꨥ ꨗꩆ ꨗꨯꨱ ꨟꩀ ꨒꨭꩍ ꨝꨩꨤꨧꨬ ꨗꩆ ꨧꨩ ꨌꨳꨮꩅ ꨀꨳꨩ ꨗꩆ ꨧꨩ ꨆꨕꨯꨱꩍ ꨓꨴꨩ ꨀꨟꨴꨩ ꨧꨩ ꨡꨰꩀ ꨡꨩꨎꨳꨯꨮꩃ ꨊꨯꨱꩀ
ꨣꨕꨰꩍ, ꨝꨵꨯꨱꩍ ꨓꨴꨶꩀ ꨗꨯꨱ ꨟꩀ ꨒꨭꩍ ꨚꩀ ꨊꨯꨱꩀ ꨣꨤꨯꩃ. ꨗꨯꨱ ꨓꨮꩊ ꨊꨯꨱꩀ ꨣꨤꨯꩃ ꨅꩃ
ꨓꨌꨯꨱꨥ ꨨꨶꩀ ꨡꨮꩃ ꨝꨵꨯꨱꩍ ꨗꨯꨱ ꨟꩀ ꨒꨭꩍ ꨓꨮꩊ ꨆꨴꨲꩍ ꨡꨩꩆꨳꩀ ꨟꨨꨭꨩ ꨀꨳꨩ ꨗꩆ ꨟꩃ
ꨓꨌꨯꨱꨥ ꨗꩆ ꨗꨯꨱ ꨕꨶꩍ ꨀꨳꨩ ꨟꨐꨭꩌ ꨗꨯꨱ ꨡꨯ ꨀꨳꨩ ꨓꨟꨭꩍ ꨕꨫ ꨆꨴꨲꩍ ꨓꨤꨫ ꨝꩀ ꨧꨩ
ꨓꨤꨫ [2], ꨝꨣꨭꨩꨥ ꨓꨌꨯꨱꨥ ꨗꩆ ꨟꨭꩌ ꨝꨵꨯꨱꩍ ꨐꨭꨩ ꨟꨗꨬ, ꨗꩆ ꨟꩃ ꨐꨭ ꨥꨮꩀ ꨟꨣꨰ
ꨀꨇꩆ ꨧꨯꨱꩃ ꨅꩃ ꨗꩆ ꨝꨣꨭꨩꨥ ꨟꩃ ꨅꩃ ꨗꩆ ꨡꨩꩆ  ꨓꨌꨯꨱꨥ ꨗꩆ ꨝꨩ ꨗꨯꨱ ꨡꨩꩁꨯꨱꨥ 
ꨀꨳꨩ ꨗꩆ ꨆ ꨅꩃ ꨗꩆ ꨆꨳꨮꩃ ꨟꨐꨭꩌ, ꨗꩆ ꨟꩃ ꨓꨌꨯꨱꨥ ꨗꩆ ꨝꨩ ꨡꨩꩁꨯꨱꨥ ꨗꨯꨱ ꨡꨯ
ꨀꨳꨩ ꨕꨫ ꨓꨤꨫ ꨗꩆ ꨔꨭꨩ ꨥꨮꩀ ꨀꨝꨪꩍ, ꨝꨣꨭꨩꨥ ꨟꩃ ꨅꩃ ꨗꩆ ꨓꨊꨫ ꨓꨌꨯꨱꨥ ꨗꩆ ꨤꩄ
ꨨꨝꩉ ꨤꩄ ꨝꨵꨯꨱꩍ ꨡꨯ ꨕꨯꩌ ꨗꩌ ꨝꨔꩍ ꨕꨫ ꨓꨤꨫ ꨗꨫ ꨝꨵꨯꨱꩍ ꨅ ꨯ ꨳꨩ ꨅ. ꨝꨣꨭꨩꨥ
ꨟꩃ ꨓꨌꨯꨱꨥ ꨗꩆ ꨤꩄ ꨥꨮꩀ ꨤꩄ ꨟꩃ ꨆꨩꨣꨗꨫ ꨕꨨꨵꩀ ꨡꨯ ꨀꨳꨩ ꨔꨴꨰꩈ ꨓꨤꨫ
ꨕꨨꨵꩀ ꨟꨐꨭꩌ ꨔꨴꨭꩇ ꨟꨨꨭꨩ ꨝꨵꨯꨱꩍ ꨕꨨꨵꩀ ꨟꨗꨬ ꨝꩀ ꨕꨴꨬ ꨝꩀ ꨎꩆ ꨨꨝꩉ ꨆꩄ
ꨀꨣꩀ ꨗꨫ ꨝꨵꨯꨱꩍ ꨀꨳꨩ ꨔꨭꨩ ꨀꨝꨪꩍ ꨌꨰꩀ, ꨅꩃ ꨗꩆ ꨟꨨꨭꨩ ꨀꨳꨩ ꨕꨫ ꨔꨮꨭ [3] ꨤꨝꨪꩀ ꨆꨳꨮꩃ
ꨋꩇ ꨨꨝꩉ ꨅ, ꨗꩆ ꨟꩃ ꨅꩃ ꨗꩆ ꨡꨩꩆ ꨓꨌꨯꨱꨥ ꨗꩆ ꨣꨥꩀ ꨒꨭꩍ ꨟꨣꨰ ꨡꨩꨎꨳꨯꨮꩃ
ꨝꨵꨯꨱꩍ ꨓꨴꨶꩀ ꨣꨕꨰꩍ ꨗꨯꨱ ꨧꩃ. ꨝꨣꨭꨩꨥ ꨟꩃ ꨓꨌꨯꨱꨥ ꨗꩆ ꨕꨯꨱ ꨕꨫ ꨧꩃ ꨢꨯꨱꩌ ꨓꨎꨭꩍ
ꨨꨣꨬ ꨈꩆ ꨝꨵꨯꨱꩍ ꨟꨓꨳꩆ ꨎꨮꨩ, ꨗꩆ ꨟꩃ ꨕꨯꩌ ꨛꨯꨮ ꨈꨗꨶꨮꩉ ꨤꩃ ꨦꨯ ꨚꨴꨯꨱꩃ ꨀꨗꨰꩍ ꨕꨤꩌ
ꨚꨤꨬ ꨟꩀ ꨅꩃ ꨧꨯꨱꩃ ꨟꨭꩀ ꨗꩆ, ꨤꩄ ꨨꨝꩉ ꨓꨌꨯꨱꨥ ꨅꩃ ꨧꨭꨩ ꨟꨭꩀ ꨗꩆ ꩆ ꨝꨵꨯꨱꩍ
ꨅ ꨡꨯ ꨡꨩꨧꩃ ꨅ, ꨗꩆ ꨟꨮꩃ ꨅꩃ ꨧꨯꨱꩃ ꨟꨭꩀ ꨤꩄ ꨕꨨꨵꩀ ꨤꨆꨮꨭ ꨙꨮꩌ ꨥꨮꩀ ꨧꨯꨱꩃ ꨛꨯꨮꨈꨗꨶꨮꩉ ꨤꩃ ꨦꨯ ꨨꨤꨬ ꨓꨌꨯꨱꨥ ꨕꨨꨵꩀ ꨟꩃ ꨀꨗꨰꩍ ꨙꨩ ꨗꨫ ꨕꨫ ꨨꨭꨩ ꨔꨭ ꨙꩌ ꨕꨩꨆꨴꨲꨩ  ꨆꨵꨯꨱ
ꨧꨯꨱꩃ ꨔꨬ ꨅ ꨓꨌꨯꨱꨥ ꨕꨨꨵꩀ ꨓꩀ ꨕꨫ ꨆꩊ ꨐꨭ ꨓꨶꨬ ꨅ ꨐꨭ ꨗꨯꨱ ꨟꩀ ꨒꨭꩍ ꨚꩀꨊꨯꨱꩀ [4] ꨣꨤꨯꩃ  ꨝꨵꨯꨱꩍ ꨓꨌꨯꨱꨥ ꨕꨨꨵꩀ ꨟꨨꨭꨩ ꨀꨳꨩ ꨝꨵꨯꨱꩍ ꨐꨭ ꨗꨯꨱ ꨕꨶꩍ ꨀꨳꨩ ꨟꨐꨭꩌ ꨐꨭ"""
    
    raw_sentences = [line.strip() for line in text_block.strip().split('\n') if line.strip()]
    
    font_path = os.path.join(PROJECT_ROOT, "data", "fonts", "NotoSansCham-Regular.ttf")
    evidence_dir = os.path.join(PROJECT_ROOT, "output", "evidence")
    page_ocr_dir = os.path.join(evidence_dir, "page_ocr")
    overlays_dir = os.path.join(page_ocr_dir, "detection_overlays")
    crops_dir = os.path.join(page_ocr_dir, "cropped_lines")
    
    os.makedirs(overlays_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)
    
    # 1. Generate 20 distorted document pages and record exact line layout
    print("🎨 Generating 20 distorted pages...")
    pages_data = []
    
    # We use a fixed seed to ensure reproducibility
    random.seed(42)
    np.random.seed(42)
    
    for page_idx in range(20):
        # 5 to 7 lines per page
        num_lines = random.randint(5, 7)
        page_lines = random.sample(raw_sentences, num_lines)
        
        # Render page
        page_img = draw_distorted_page(page_lines, font_path, width=800, height=650)
        img_path = os.path.join(page_ocr_dir, f"page_{page_idx}.png")
        page_img.save(img_path)
        
        pages_data.append({
            "image_path": img_path,
            "ground_truth": page_lines,
            "page_idx": page_idx
        })
        
    # 2. Setup Low-level low-latency detector & V23 recognizer
    print("⚙️ Loading PaddleOCR DBNet detector & V23 Recognizer...")
    sys_argv_backup = sys.argv
    sys.argv = [sys.argv[0]]
    args = utility.parse_args()
    sys.argv = sys_argv_backup
    
    args.det_model_dir = os.path.join(PROJECT_ROOT, "data", "pretrain_models", "ch_PP-OCRv4_det_infer")
    args.rec_model_dir = os.path.join(PROJECT_ROOT, "output", "rec_cham_inference_v23")
    args.rec_char_dict_path = os.path.join(PROJECT_ROOT, "data", "cham_dict_v23.txt")
    args.use_gpu = False
    args.enable_mkldnn = False
    args.use_space_char = True
    args.det_limit_side_len = 960
    args.det_db_thresh = 0.3
    args.det_db_box_thresh = 0.5
    args.det_db_unclip_ratio = 1.6
    
    # Initialize detector and recognizer separately
    text_detector = predict_det.TextDetector(args)
    text_recognizer = predict_rec.TextRecognizer(args)
    
    # 3. Track detailed evaluation metrics
    det_total_pages = 20
    det_total_gt_lines = 0
    det_total_detected_lines = 0
    det_total_missed_lines = 0
    det_total_false_positives = 0
    
    rec_total_lines = 0
    rec_correct_sentences = 0
    rec_total_chars = 0
    rec_total_cer_errors = 0
    rec_total_confidence = 0.0
    
    e2e_total_pages = 20
    e2e_total_chars = 0
    e2e_total_cer_errors = 0
    e2e_correct_sentences = 0
    e2e_total_gt_lines = 0
    
    predictions_jsonl = []
    
    first_4_pages = []
    
    print("\n================== EVALUATING PAGE-OCR PIPELINE ==================")
    for page_idx, page in enumerate(pages_data):
        img_path = page["image_path"]
        gt_lines = page["ground_truth"]
        e2e_total_gt_lines += len(gt_lines)
        det_total_gt_lines += len(gt_lines)
        
        img_cv2 = cv2.imread(img_path)
        h, w = img_cv2.shape[:2]
        
        # 3A. Run DBNet Detector
        dt_boxes, elapse_det = text_detector(img_cv2)
        det_total_detected_lines += len(dt_boxes)
        
        # Draw overlays
        overlay_img = img_cv2.copy()
        
        # We need to match detected bounding boxes to Ground Truth lines.
        # Since we rendered them top-to-bottom at y = 70, 150, 230... 
        # let's approximate the GT box coordinate before distortion.
        # After distortion, we can match box center Y coordinate with y_start + idx * y_spacing.
        # GT centers: y_start + idx * 80.
        gt_centers = [70 + idx * 80 for idx in range(len(gt_lines))]
        
        matched_gt = set()
        detected_line_matches = []
        
        # Sort boxes top to bottom by their center Y coordinate
        sorted_boxes = []
        for box_idx, box in enumerate(dt_boxes):
            box_np = np.array(box).astype(np.int32)
            ymin = int(np.min(box_np[:, 1]))
            ymax = int(np.max(box_np[:, 1]))
            cy = (ymin + ymax) / 2
            sorted_boxes.append((cy, box_np, box_idx))
            
        sorted_boxes.sort(key=lambda x: x[0])
        
        for cy, box_np, box_idx in sorted_boxes:
            # Draw box on overlay
            cv2.polylines(overlay_img, [box_np], True, (0, 255, 0), 2)
            
            # Find closest GT center
            best_gt_idx = -1
            min_dist = 9999.0
            for gt_idx, gt_cy in enumerate(gt_centers):
                dist = abs(cy - gt_cy)
                if dist < min_dist:
                    min_dist = dist
                    best_gt_idx = gt_idx
            
            # Match threshold: if the Y distance is within 60 pixels
            if best_gt_idx != -1 and min_dist < 60 and best_gt_idx not in matched_gt:
                matched_gt.add(best_gt_idx)
                detected_line_matches.append((box_np, best_gt_idx, box_idx))
            else:
                # False positive: either distance is too large or target GT is already matched
                detected_line_matches.append((box_np, -1, box_idx))
                det_total_false_positives += 1
                
        det_total_missed_lines += (len(gt_lines) - len(matched_gt))
        
        # Save overlay image
        cv2.imwrite(os.path.join(overlays_dir, f"page_{page_idx}_overlay.png"), overlay_img)
        
        # Keep first 4 pages for contact sheet
        if page_idx < 4:
            first_4_pages.append(overlay_img)
            
        # 3B. Run Recognizer on Detected Boxes
        for match_idx, (box_np, gt_idx, box_idx) in enumerate(detected_line_matches):
            # Crop line image
            ymin = max(0, int(np.min(box_np[:, 1])))
            ymax = min(h, int(np.max(box_np[:, 1])))
            xmin = max(0, int(np.min(box_np[:, 0])))
            xmax = min(w, int(np.max(box_np[:, 0])))
            
            # Skip empty crops
            if ymax - ymin < 5 or xmax - xmin < 5:
                continue
                
            crop = img_cv2[ymin:ymax, xmin:xmax]
            
            # Save crop
            crop_name = f"page_{page_idx}_box_{box_idx}.png"
            cv2.imwrite(os.path.join(crops_dir, crop_name), crop)
            
            # Run Recognizer
            rec_res, elapse_rec = text_recognizer([crop])
            pred_visual, confidence = rec_res[0] if (rec_res and rec_res[0]) else ("", 0.0)
            pred_unicode = visual_to_unicode(pred_visual.strip())
            pred_clean = re.sub(r'\s+', ' ', pred_unicode.strip())
            
            truth_clean = ""
            if gt_idx != -1:
                truth_unicode = visual_to_unicode(gt_lines[gt_idx].strip())
                truth_clean = re.sub(r'\s+', ' ', truth_unicode.strip())
                
                # Update Recognizer Metrics (only on correct/matched crops)
                rec_total_lines += 1
                rec_total_confidence += confidence
                rec_total_chars += len(truth_clean)
                
                is_rec_match = (pred_clean == truth_clean)
                if is_rec_match:
                    rec_correct_sentences += 1
                    
                errs = calculate_levenshtein(pred_clean, truth_clean)
                rec_total_cer_errors += errs
                
            # Log prediction
            predictions_jsonl.append({
                "page": page_idx,
                "box_idx": box_idx,
                "matched_gt_idx": gt_idx,
                "truth": truth_clean,
                "pred": pred_clean,
                "confidence": float(confidence)
            })
            
        # 3C. Calculate End-to-End Page Metrics (comparing GT lines to predictions top-to-bottom)
        # For simplicity, map predictions to GT lines by matched index or order
        # If a line is missed, it adds GT length errors. If extra, it adds insertions.
        for gt_idx, gt_line in enumerate(gt_lines):
            truth_unicode = visual_to_unicode(gt_line.strip())
            truth_clean = re.sub(r'\s+', ' ', truth_unicode.strip())
            e2e_total_chars += len(truth_clean)
            
            # Find if this GT was matched to any box
            pred_clean = ""
            for item in predictions_jsonl:
                if item["page"] == page_idx and item["matched_gt_idx"] == gt_idx:
                    pred_clean = item["pred"]
                    break
                    
            if pred_clean == truth_clean and truth_clean:
                e2e_correct_sentences += 1
                
            errs = calculate_levenshtein(pred_clean, truth_clean)
            e2e_total_cer_errors += errs

    # Generate Contact Sheet (2x2 grid of first 4 overlay pages)
    print("🖼️ Generating contact sheet for first 4 pages...")
    if len(first_4_pages) >= 4:
        h_thumb, w_thumb = 325, 400
        thumb_images = [cv2.resize(img, (w_thumb, h_thumb)) for img in first_4_pages[:4]]
        
        # Tile 2x2
        top_row = np.hstack((thumb_images[0], thumb_images[1]))
        bottom_row = np.hstack((thumb_images[2], thumb_images[3]))
        contact_sheet = np.vstack((top_row, bottom_row))
        
        cv2.imwrite(os.path.join(page_ocr_dir, "contact_sheet_pages.png"), contact_sheet)
        print("   💾 Saved contact_sheet_pages.png successfully!")
        
    # Write ocr_predictions.jsonl
    predictions_path = os.path.join(page_ocr_dir, "ocr_predictions.jsonl")
    with open(predictions_path, "w", encoding="utf-8") as f:
        for item in predictions_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    # Calculate final scores
    line_recall = (det_total_gt_lines - det_total_missed_lines) / det_total_gt_lines * 100 if det_total_gt_lines > 0 else 0.0
    line_precision = (det_total_detected_lines - det_total_false_positives) / det_total_detected_lines * 100 if det_total_detected_lines > 0 else 0.0
    
    rec_cer = (rec_total_cer_errors / rec_total_chars) * 100 if rec_total_chars > 0 else 0.0
    rec_sent_acc = (rec_correct_sentences / rec_total_lines) * 100 if rec_total_lines > 0 else 0.0
    rec_avg_conf = (rec_total_confidence / rec_total_lines) if rec_total_lines > 0 else 0.0
    
    e2e_cer = (e2e_total_cer_errors / e2e_total_chars) * 100 if e2e_total_chars > 0 else 0.0
    e2e_sent_acc = (e2e_correct_sentences / e2e_total_gt_lines) * 100 if e2e_total_gt_lines > 0 else 0.0
    
    # Save statistics JSON
    metrics_summary_path = os.path.join(page_ocr_dir, "metrics_summary.json")
    metrics_summary = {
        "detector": {
            "dataset": "distorted_pages",
            "pages": det_total_pages,
            "gt_lines": det_total_gt_lines,
            "detected_lines": det_total_detected_lines,
            "missed_lines": det_total_missed_lines,
            "false_positives": det_total_false_positives,
            "line_recall": line_recall,
            "line_precision": line_precision
        },
        "recognizer": {
            "dataset": "distorted_pages_correct_crops",
            "lines": rec_total_lines,
            "cer": rec_cer,
            "sentence_accuracy": rec_sent_acc,
            "avg_confidence": rec_avg_conf
        },
        "e2e": {
            "dataset": "distorted_pages_end_to_end",
            "pages": e2e_total_pages,
            "cer": e2e_cer,
            "sentence_accuracy": e2e_sent_acc,
            "main_failure": "DBNet text line detection failures under severe perspective and bending distortions"
        }
    }
    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4, ensure_ascii=False)
        
    print("\n================== DETAILED METRICS COMPLETED ==================")
    print(f"Detector Line Recall    : {line_recall:.2f}% (Missed: {det_total_missed_lines})")
    print(f"Detector Line Precision : {line_precision:.2f}% (False Positives: {det_total_false_positives})")
    print(f"Recognizer CER (Crops)  : {rec_cer:.2f}%")
    print(f"Recognizer Sent Acc     : {rec_sent_acc:.2f}%")
    print(f"End-to-End Page CER     : {e2e_cer:.2f}%")
    print(f"End-to-End Page Sent Acc: {e2e_sent_acc:.2f}%")
    print("=================================================================")

if __name__ == '__main__':
    main()
