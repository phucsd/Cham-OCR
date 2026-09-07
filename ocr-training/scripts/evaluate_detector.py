import os
import sys

# Khóa cứng vô hiệu hóa oneDNN / MKLDNN để chống crash trên Windows
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_mkldnn"] = "0"

import cv2
import json
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "PaddleOCR"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

# Import PaddleOCR TextDetector
import tools.infer.predict_det as predict_det
import tools.infer.utility as utility

def get_polygon_area(poly):
    p = np.array(poly, dtype=np.float32)
    return cv2.contourArea(p)

def get_polygon_intersection_area(poly1, poly2):
    p1 = np.array(poly1, dtype=np.float32).reshape(-1, 1, 2)
    p2 = np.array(poly2, dtype=np.float32).reshape(-1, 1, 2)
    ret, intersection = cv2.intersectConvexConvex(p1, p2)
    if ret > 0:
        return float(ret)
    return 0.0

def calculate_iou(poly1, poly2):
    inter = get_polygon_intersection_area(poly1, poly2)
    area1 = get_polygon_area(poly1)
    area2 = get_polygon_area(poly2)
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union

def main():
    print("🔍 Bắt đầu đánh giá hiệu năng DBNet Detector...")
    
    det_root = os.path.join(PROJECT_ROOT, "data", "detector")
    val_label_path = os.path.join(det_root, "det_val_label.txt")
    evidence_dir = os.path.join(PROJECT_ROOT, "output", "evidence", "detector")
    contact_sheets_dir = os.path.join(evidence_dir, "contact_sheets")
    
    os.makedirs(contact_sheets_dir, exist_ok=True)
    
    if not os.path.exists(val_label_path):
        print(f"❌ Không tìm thấy tệp nhãn validation: {val_label_path}")
        return
        
    # Khởi tạo TextDetector
    sys_argv_backup = sys.argv
    sys.argv = [sys.argv[0]]
    args = utility.parse_args()
    sys.argv = sys_argv_backup
    
    args.det_model_dir = os.path.join(PROJECT_ROOT, "data", "pretrain_models", "ch_PP-OCRv4_det_infer")
    args.use_gpu = False
    args.enable_mkldnn = False
    args.det_limit_side_len = 960
    
    print("⚙️  Đang nạp mô hình DBNet...")
    detector = predict_det.TextDetector(args)
    
    # Đọc nhãn val
    val_lines = []
    with open(val_label_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_lines.append(line.strip().split("\t"))
                
    total_gt = 0
    total_dt = 0
    recalled_gt = 0
    precise_dt = 0
    usable_crops = 0
    all_ious = []
    
    print(f"📈 Đang đánh giá trên {len(val_lines)} trang tài liệu...")
    for idx, (img_rel_path, label_str) in enumerate(val_lines):
        img_path = os.path.join(det_root, img_rel_path)
        if not os.path.exists(img_path):
            print(f"  ❌ Không tìm thấy ảnh: {img_path}")
            continue
            
        gt_boxes = json.loads(label_str)
        img_cv2 = cv2.imread(img_path)
        
        # Chạy detector
        dt_boxes, _ = detector(img_cv2)
        
        # PIL Image phục vụ vẽ overlays
        pil_img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(pil_img)
        
        # Mapping GT và DT boxes
        gt_matched = [False] * len(gt_boxes)
        dt_matched = [False] * len(dt_boxes)
        
        print(f"\n📄 Trang {idx+1:02d} ({os.path.basename(img_rel_path)}):")
        
        # Tính IoU và Recall/Precision
        for gt_idx, gt in enumerate(gt_boxes):
            gt_poly = gt["points"]
            best_iou = 0.0
            best_dt_idx = -1
            best_inter = 0.0
            
            for dt_idx, dt_poly in enumerate(dt_boxes):
                iou = calculate_iou(gt_poly, dt_poly)
                if iou > best_iou:
                    best_iou = iou
                    best_dt_idx = dt_idx
                    best_inter = get_polygon_intersection_area(gt_poly, dt_poly)
            
            print(f"   GT {gt_idx:2d} -> Max IoU với DT: {best_iou:.4f}")
            
            # Sử dụng ngưỡng IoU >= 0.50 (chuẩn Pascal VOC) để đánh giá tài liệu biến dạng
            if best_iou >= 0.50:
                recalled_gt += 1
                gt_matched[gt_idx] = True
                dt_matched[best_dt_idx] = True
                all_ious.append(best_iou)
            
            # 2. Crop usability check (GT nằm trọn trong DT: giao >= 95% diện tích GT)
            gt_area = get_polygon_area(gt_poly)
            if gt_area > 0 and (best_inter / gt_area) >= 0.95:
                usable_crops += 1
                
        # Tính Precision
        for dt_idx, dt_poly in enumerate(dt_boxes):
            best_iou = 0.0
            for gt_idx, gt in enumerate(gt_boxes):
                iou = calculate_iou(gt["points"], dt_poly)
                if iou > best_iou:
                    best_iou = iou
            if best_iou >= 0.50:
                precise_dt += 1
                
        total_gt += len(gt_boxes)
        total_dt += len(dt_boxes)
        
        # Vẽ Overlay Contact Sheet
        # - Màu xanh lá: Predicted polygon chính xác (IoU >= 0.5)
        # - Màu đỏ nét đứt: Ground-truth bị bỏ sót (Recall fail)
        # - Màu vàng: Predicted polygon sai lệch (False positive)
        
        for dt_idx, dt_poly in enumerate(dt_boxes):
            pts = [tuple(p) for p in dt_poly]
            pts.append(pts[0])
            color = (0, 180, 0) if dt_matched[dt_idx] else (200, 200, 0)
            draw.line(pts, fill=color, width=2)
            
        for gt_idx, gt in enumerate(gt_boxes):
            if not gt_matched[gt_idx]:
                pts = [tuple(p) for p in gt["points"]]
                pts.append(pts[0])
                draw.line(pts, fill=(200, 0, 0), width=2)
                
        # Lưu ảnh Contact Sheet
        contact_sheet_name = f"contact_sheet_{os.path.basename(img_rel_path)}"
        pil_img.save(os.path.join(contact_sheets_dir, contact_sheet_name))
        
    # Tính tổng hợp chỉ số
    recall_rate = (recalled_gt / total_gt) * 100 if total_gt > 0 else 0.0
    precision_rate = (precise_dt / total_dt) * 100 if total_dt > 0 else 0.0
    usability_rate = (usable_crops / total_gt) * 100 if total_gt > 0 else 0.0
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    
    results = {
        "evaluation_summary": {
            "total_val_pages": len(val_lines),
            "total_ground_truth_lines": total_gt,
            "total_detected_lines": total_dt,
            "mean_iou": round(float(mean_iou), 4),
            "line_recall": round(recall_rate, 2),
            "line_precision": round(precision_rate, 2),
            "crop_usability_rate": round(usability_rate, 2)
        }
    }
    
    # Ghi báo cáo JSON
    results_path = os.path.join(evidence_dir, "iou_matching.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print("\n================== DETECTOR EVALUATION REPORT ==================")
    print(f"Line Recall            : {recall_rate:.2f}% ({recalled_gt}/{total_gt})")
    print(f"Line Precision         : {precision_rate:.2f}% ({precise_dt}/{total_dt})")
    print(f"Mean IoU               : {mean_iou:.4f}")
    print(f"Crop Usability Rate    : {usability_rate:.2f}% ({usable_crops}/{total_gt})")
    print(f"Saved matching report to {results_path}")
    print(f"Saved contact sheets under {contact_sheets_dir}")
    print("================================================================")

if __name__ == '__main__':
    main()
