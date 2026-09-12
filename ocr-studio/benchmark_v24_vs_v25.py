import os
import sys
import json
import time
import re
import difflib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 1. NumPy 2.x Compatibility Patch
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

# Set single thread for Windows stability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIO_DIR = os.path.join(PROJECT_ROOT, "ocr-studio")
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "ocr-benchmark")
FONTS_DIR = os.path.join(PROJECT_ROOT, "ocr-training", "data", "fonts")

sys.path.insert(0, STUDIO_DIR)
import app
from scripts.generate_data import normalize_unicode

FONT_REG = os.path.join(FONTS_DIR, "NotoSansCham-Regular.ttf")
FONT_BOLD = os.path.join(FONTS_DIR, "NotoSansCham-Bold.ttf")

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

def render_test_image(text, desc, category):
    font_path = FONT_BOLD if "Bold" in desc else FONT_REG
    font_size = 26
    if "Large" in desc or "large" in desc:
        font_size = 32
    elif "Low Height" in category:
        font_size = 18

    font = ImageFont.truetype(font_path, font_size)
    dummy = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad_x = 20
    pad_y = 10
    w = max(320, tw + pad_x * 2)
    h = max(48, th + pad_y * 2)

    bg_color = (255, 255, 255)
    ink_color = (10, 10, 10)

    if "Paper Texture" in category:
        bg_color = (245, 240, 225)
        ink_color = (40, 35, 30)

    img = Image.new('RGB', (w, h), color=bg_color)
    d = ImageDraw.Draw(img)
    x = (w - tw) // 2 - bbox[0]
    y = (h - th) // 2 - bbox[1]
    d.text((x, y), text, font=font, fill=ink_color)

    arr = np.array(img)

    import cv2
    if "Blur Degradation" in category:
        arr = cv2.GaussianBlur(arr, (5, 5), 1.2)
    elif "Noise & Grain" in category:
        noise = np.random.normal(0, 12, arr.shape).astype(np.float32)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif "Stroke Degradation" in category:
        k = np.ones((2, 2), np.uint8)
        arr = cv2.erode(arr, k, iterations=1)
    elif "Tilt & Perspective" in category:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 2.5, 1.0)
        arr = cv2.warpAffine(arr, M, (w, h), borderValue=bg_color)

    # Convert RGB to BGR for PaddleOCR
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr

def run_benchmark():
    print("=" * 80)
    print("🔍 KHỞI ĐỘNG HỆ THỐNG ĐÁNH GIÁ ĐỐI SÁNH: CHAM-OCR V24 VS V25")
    print("=" * 80)

    t0 = time.time()
    print("\n[1/4] Đang nạp mô hình V24...")
    m24 = app.get_ocr_model("v24")
    print(f"  ✓ Model V24 sẵn sàng trong {time.time() - t0:.2f}s")

    t1 = time.time()
    print("[2/4] Đang nạp mô hình V25...")
    m25 = app.get_ocr_model("v25")
    print(f"  ✓ Model V25 sẵn sàng trong {time.time() - t1:.2f}s")

    # ==========================================================================
    # PHẦN 1: ĐÁNH GIÁ TRÊN BỘ 50 TEST CASES ĐẶC TRỊ
    # ==========================================================================
    json_50_path = os.path.join(STUDIO_DIR, "data", "benchmark_50_tests_results_v24.json")
    with open(json_50_path, "r", encoding="utf-8") as f:
        data_50 = json.load(f)

    tests_50 = data_50["tests"]
    print(f"\n[3/4] Chạy đánh giá 50 Test Cases Đặc trị (10 Categories)...")

    results_50 = []
    cat_summary = {}

    v24_total_cer = 0.0
    v25_total_cer = 0.0
    v24_pass_count = 0
    v25_pass_count = 0

    for t in tests_50:
        t_id = t["id"]
        cat = t["category"]
        desc = t["desc"]
        gt = t["gt"]

        img = render_test_image(gt, desc, cat)

        # Run V24
        res24, _ = m24([img])
        raw24, conf24 = res24[0] if res24 else ("", 0.0)
        norm24 = normalize_unicode(raw24.strip())
        cer24 = compute_cer(norm24, gt)
        pass24 = (cer24 <= 0.10)

        # Run V25
        res25, _ = m25([img])
        raw25, conf25 = res25[0] if res25 else ("", 0.0)
        norm25 = normalize_unicode(raw25.strip())
        cer25 = compute_cer(norm25, gt)
        pass25 = (cer25 <= 0.10)

        v24_total_cer += cer24
        v25_total_cer += cer25
        if pass24: v24_pass_count += 1
        if pass25: v25_pass_count += 1

        if cat not in cat_summary:
            cat_summary[cat] = {
                "count": 0,
                "v24_cer": 0.0, "v25_cer": 0.0,
                "v24_pass": 0, "v25_pass": 0,
                "v24_conf": 0.0, "v25_conf": 0.0
            }
        cat_summary[cat]["count"] += 1
        cat_summary[cat]["v24_cer"] += cer24
        cat_summary[cat]["v25_cer"] += cer25
        cat_summary[cat]["v24_conf"] += conf24
        cat_summary[cat]["v25_conf"] += conf25
        if pass24: cat_summary[cat]["v24_pass"] += 1
        if pass25: cat_summary[cat]["v25_pass"] += 1

        results_50.append({
            "id": t_id,
            "category": cat,
            "desc": desc,
            "gt": gt,
            "v24_pred": norm24,
            "v24_conf": float(conf24),
            "v24_cer": float(cer24),
            "v24_passed": pass24,
            "v25_pred": norm25,
            "v25_conf": float(conf25),
            "v25_cer": float(cer25),
            "v25_passed": pass25,
        })

    # ==========================================================================
    # PHẦN 2: ĐÁNH GIÁ TRÊN BỘ DỮ LIỆU THỰC TẾ 200 TRANG (MẪU 25 TRANG)
    # ==========================================================================
    print(f"\n[4/4] Chạy đánh giá Pipeline đầy đủ (DBNet + Recognizer) trên 25 trang Benchmark chuẩn...")
    gt_200_path = os.path.join(BENCHMARK_DIR, "data", "benchmark_gt.json")
    with open(gt_200_path, "r", encoding="utf-8") as f:
        data_200 = json.load(f)

    # Chọn 25 mẫu đại diện từ Level 1 đến Level 4
    sample_indices = [i for i in range(0, min(150, len(data_200)), 6)]
    sub_samples = [data_200[i] for i in sample_indices[:25]]

    import cv2
    det_model = app.get_det_model()

    doc_v24_cer = 0.0
    doc_v25_cer = 0.0
    doc_v24_wer = 0.0
    doc_v25_wer = 0.0
    total_lines = 0

    pipeline_results = []

    for s_idx, sample in enumerate(sub_samples, 1):
        fn = sample["filename"]
        lvl = sample["level"]
        lvl_name = sample["level_name"]
        img_p = os.path.join(BENCHMARK_DIR, sample["image_path"])

        img = cv2.imread(img_p)
        if img is None:
            continue

        crops, metas = app.segment_lines_dbnet(img, det_model)
        if not crops:
            continue

        gt_lines = [l["text"].strip() for l in sample["lines"]]
        total_lines += len(gt_lines)

        # Predict V24
        rec24, _ = m24(crops)
        pred24_lines = [normalize_unicode(r[0].strip()) if r else "" for r in rec24]
        pred24_doc = "\n".join(pred24_lines)
        gt_doc = "\n".join(gt_lines)
        cer24_doc = compute_cer(pred24_doc, gt_doc)
        wer24_doc = compute_wer(pred24_doc, gt_doc)

        # Predict V25
        rec25, _ = m25(crops)
        pred25_lines = [normalize_unicode(r[0].strip()) if r else "" for r in rec25]
        pred25_doc = "\n".join(pred25_lines)
        cer25_doc = compute_cer(pred25_doc, gt_doc)
        wer25_doc = compute_wer(pred25_doc, gt_doc)

        doc_v24_cer += cer24_doc
        doc_v25_cer += cer25_doc
        doc_v24_wer += wer24_doc
        doc_v25_wer += wer25_doc

        pipeline_results.append({
            "filename": fn,
            "level": lvl,
            "level_name": lvl_name,
            "gt_doc": gt_doc,
            "v24_pred": pred24_doc,
            "v24_cer": float(cer24_doc),
            "v24_wer": float(wer24_doc),
            "v25_pred": pred25_doc,
            "v25_cer": float(cer25_doc),
            "v25_wer": float(wer25_doc)
        })

    n_samples = max(1, len(pipeline_results))
    avg_v24_cer = (doc_v24_cer / n_samples) * 100
    avg_v25_cer = (doc_v25_cer / n_samples) * 100
    avg_v24_wer = (doc_v24_wer / n_samples) * 100
    avg_v25_wer = (doc_v25_wer / n_samples) * 100

    # ==========================================================================
    # TỔNG HỢP VÀ IN KẾT QUẢ SO SÁNH CHI TIẾT
    # ==========================================================================
    print("\n" + "=" * 80)
    print("📊 BẢNG 1: ĐỐI SÁNH 50 TEST CASES ĐẶC TRỊ THEO TỪNG NHÓM DANH MỤC")
    print("=" * 80)
    print(f"| {'Danh mục Test':<32} | {'V24 CER':<9} | {'V25 CER':<9} | {'V24 Pass':<8} | {'V25 Pass':<8} | {'Tiến bộ':<12} |")
    print("|" + "-"*34 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*14 + "|")

    for cat, stats in cat_summary.items():
        cnt = stats["count"]
        c24 = (stats["v24_cer"] / cnt) * 100
        c25 = (stats["v25_cer"] / cnt) * 100
        p24 = (stats["v24_pass"] / cnt) * 100
        p25 = (stats["v25_pass"] / cnt) * 100
        diff_cer = c24 - c25
        if diff_cer > 0:
            trend = f"✅ -{diff_cer:.1f}% CER"
        elif diff_cer < 0:
            trend = f"⚠️ +{abs(diff_cer):.1f}% CER"
        else:
            trend = f"➖ Hòa"

        print(f"| {cat:<32} | {c24:>7.1f}% | {c25:>7.1f}% | {p24:>6.0f}% | {p25:>6.0f}% | {trend:<12} |")

    v24_overall_cer_50 = (v24_total_cer / len(tests_50)) * 100
    v25_overall_cer_50 = (v25_total_cer / len(tests_50)) * 100
    v24_overall_pass_50 = (v24_pass_count / len(tests_50)) * 100
    v25_overall_pass_50 = (v25_pass_count / len(tests_50)) * 100

    print("|" + "="*34 + "|" + "="*11 + "|" + "="*11 + "|" + "="*10 + "|" + "="*10 + "|" + "="*14 + "|")
    print(f"| {'TỔNG THỂ 50 BÀI TEST':<32} | {v24_overall_cer_50:>7.1f}% | {v25_overall_cer_50:>7.1f}% | {v24_overall_pass_50:>6.0f}% | {v25_overall_pass_50:>6.0f}% | {'✅ -' + str(round(v24_overall_cer_50 - v25_overall_cer_50, 1)) + '% CER' if v24_overall_cer_50 > v25_overall_cer_50 else '➖':<12} |")

    print("\n" + "=" * 80)
    print("📄 BẢNG 2: ĐỐI SÁNH PIPELINE THỰC TẾ TRÊN CÁC TRANG VĂN BẢN (CHAM-DBNET + OCR)")
    print("=" * 80)
    print(f"| {'Chỉ số đánh giá':<35} | {'Mô hình V24':<15} | {'Mô hình V25':<15} | {'Độ chênh lệch':<15} |")
    print("|" + "-"*37 + "|" + "-"*17 + "|" + "-"*17 + "|" + "-"*17 + "|")
    print(f"| {'Tỷ lệ lỗi ký tự (CER)':<35} | {avg_v24_cer:>13.2f}% | {avg_v25_cer:>13.2f}% | {'✅ Giảm ' + str(round(avg_v24_cer - avg_v25_cer, 2)) + '%' if avg_v24_cer > avg_v25_cer else 'Tăng ' + str(round(avg_v25_cer - avg_v24_cer, 2)) + '%':<15} |")
    print(f"| {'Tỷ lệ lỗi từ (WER)':<35} | {avg_v24_wer:>13.2f}% | {avg_v25_wer:>13.2f}% | {'✅ Giảm ' + str(round(avg_v24_wer - avg_v25_wer, 2)) + '%' if avg_v24_wer > avg_v25_wer else 'Tăng ' + str(round(avg_v25_wer - avg_v24_wer, 2)) + '%':<15} |")

    # Lưu file kết quả
    output_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_50_tests": {
            "v24_cer": v24_overall_cer_50,
            "v25_cer": v25_overall_cer_50,
            "v24_pass_rate": v24_overall_pass_50,
            "v25_pass_rate": v25_overall_pass_50
        },
        "category_summary": cat_summary,
        "pipeline_pages": {
            "v24_cer": avg_v24_cer,
            "v25_cer": avg_v25_cer,
            "v24_wer": avg_v24_wer,
            "v25_wer": avg_v25_wer
        },
        "detailed_50_tests": results_50,
        "pipeline_samples": pipeline_results
    }

    report_path = os.path.join(STUDIO_DIR, "data", "benchmark_v24_vs_v25_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(output_report, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Báo cáo chi tiết đã lưu tại: {report_path}")

if __name__ == "__main__":
    run_benchmark()
