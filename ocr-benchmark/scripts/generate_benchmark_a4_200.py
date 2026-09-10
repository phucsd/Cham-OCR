#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A4 Synthetic Benchmark Generator for Cham OCR (Samples 201 - 400).
Generates full-page A4 multi-paragraph documents combining Bold and Regular styles,
with explicit ground truth for paragraph boundaries and soft line continuations.
"""

import os
import sys
import json
import math
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Force UTF-8 stdout encoding on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

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

# Set seed for reproducible benchmark dataset
random.seed(1024)
np.random.seed(1024)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)

FONT_REGULAR = os.path.join(PROJECT_ROOT, "ocr-training", "data", "fonts", "NotoSansCham-Regular.ttf")
FONT_BOLD = os.path.join(PROJECT_ROOT, "ocr-training", "data", "fonts", "NotoSansCham-Bold.ttf")

if not os.path.exists(FONT_REGULAR) or not os.path.exists(FONT_BOLD):
    raise FileNotFoundError("Font files not found!")

_FONT_CACHE = {}

def get_font(path, size):
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]

def create_a4_background(w, h, level):
    """Creates realistic paper texture for A4 document matching difficulty level."""
    if level == 1:
        val = random.randint(250, 255)
        return np.full((h, w, 3), (val, val, val), dtype=np.uint8)
    elif level == 2:
        val_r = random.randint(242, 252)
        val_g = random.randint(238, 248)
        val_b = random.randint(228, 240)
        bg = np.full((h, w, 3), (val_b, val_g, val_r), dtype=np.uint8)
        noise = np.random.normal(0, 2.0, (h, w)).astype(np.float32)
        for c in range(3):
            bg[:, :, c] = np.clip(bg[:, :, c].astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return bg
    elif level == 3:
        val_r = random.randint(232, 246)
        val_g = random.randint(222, 238)
        val_b = random.randint(200, 218)
        bg = np.full((h, w, 3), (val_b, val_g, val_r), dtype=np.uint8)
        gh, gw = max(4, h // 50), max(4, w // 50)
        low_noise = np.random.normal(0, 6.0, (gh, gw)).astype(np.float32)
        low_noise = cv2.resize(low_noise, (w, h), interpolation=cv2.INTER_CUBIC)
        noise = low_noise + np.random.normal(0, 3.0, (h, w)).astype(np.float32)
        for c in range(3):
            bg[:, :, c] = np.clip(bg[:, :, c].astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return bg
    elif level == 4:
        val_r = random.randint(226, 242)
        val_g = random.randint(214, 230)
        val_b = random.randint(182, 205)
        bg = np.full((h, w, 3), (val_b, val_g, val_r), dtype=np.uint8)
        gh, gw = max(4, h // 40), max(4, w // 40)
        low_noise = np.random.normal(0, 8.0, (gh, gw)).astype(np.float32)
        low_noise = cv2.resize(low_noise, (w, h), interpolation=cv2.INTER_CUBIC)
        noise = low_noise + np.random.normal(0, 4.0, (h, w)).astype(np.float32)
        for c in range(3):
            bg[:, :, c] = np.clip(bg[:, :, c].astype(np.float32) + noise, 0, 255).astype(np.uint8)
        # Vignette
        X = np.linspace(-1, 1, w)
        Y = np.linspace(-1, 1, h)
        mx, my = np.meshgrid(X, Y)
        dist = np.sqrt(mx**2 + my**2)
        vignette = 1.0 - np.clip((dist - 0.5) * 0.22, 0, 0.25)
        for c in range(3):
            bg[:, :, c] = (bg[:, :, c].astype(np.float32) * vignette).astype(np.uint8)
        return bg
    else:
        # Level 5: Aged parchment with lighting gradient & vignette
        val_r = random.randint(218, 238)
        val_g = random.randint(204, 222)
        val_b = random.randint(170, 195)
        bg = np.full((h, w, 3), (val_b, val_g, val_r), dtype=np.uint8)
        grad_angle = random.uniform(0, 2 * math.pi)
        X = np.linspace(-1, 1, w)
        Y = np.linspace(-1, 1, h)
        mx, my = np.meshgrid(X, Y)
        lighting = 1.0 + 0.10 * (mx * math.cos(grad_angle) + my * math.sin(grad_angle))
        gh, gw = max(4, h // 30), max(4, w // 30)
        low_noise = np.random.normal(0, 10.0, (gh, gw)).astype(np.float32)
        low_noise = cv2.resize(low_noise, (w, h), interpolation=cv2.INTER_CUBIC)
        noise = low_noise + np.random.normal(0, 4.5, (h, w)).astype(np.float32)
        for c in range(3):
            ch = bg[:, :, c].astype(np.float32) * lighting + noise
            bg[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
        dist = np.sqrt(mx**2 + my**2)
        vignette = 1.0 - np.clip((dist - 0.45) * 0.30, 0, 0.35)
        for c in range(3):
            bg[:, :, c] = (bg[:, :, c].astype(np.float32) * vignette).astype(np.uint8)
        return bg

def apply_a4_warping(img_cv, line_boxes, wave_amp, wave_freq, wave_phase, persp_amt):
    h, w = img_cv.shape[:2]
    if wave_amp > 0:
        x_idx = np.arange(w, dtype=np.float32)
        y_idx = np.arange(h, dtype=np.float32)
        X, Y = np.meshgrid(x_idx, y_idx)
        dy = wave_amp * np.sin(X * wave_freq + wave_phase)
        map_x = X
        map_y = Y + dy
        img_warped = cv2.remap(img_cv, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        img_warped = img_cv

    if persp_amt > 0:
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dx1 = random.uniform(0, w * persp_amt)
        dy1 = random.uniform(0, h * persp_amt)
        dx2 = random.uniform(0, w * persp_amt)
        dy2 = random.uniform(0, h * persp_amt)
        dx3 = random.uniform(0, w * persp_amt)
        dy3 = random.uniform(0, h * persp_amt)
        dx4 = random.uniform(0, w * persp_amt)
        dy4 = random.uniform(0, h * persp_amt)
        pts2 = np.float32([
            [dx1, dy1],
            [w - dx2, dy2],
            [dx3, h - dy3],
            [w - dx4, h - dy4]
        ])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_final = cv2.warpPerspective(img_warped, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    else:
        matrix = None
        img_final = img_warped

    transformed_lines = []
    for l_meta in line_boxes:
        pts = l_meta["box"]
        warped_pts = []
        for pt in pts:
            px, py = pt
            if wave_amp > 0:
                p_dy = wave_amp * math.sin(px * wave_freq + wave_phase)
                py += p_dy
            warped_pts.append([px, py])
            
        if matrix is not None:
            pts_arr = np.array(warped_pts, dtype=np.float32).reshape(-1, 1, 2)
            persp_pts = cv2.perspectiveTransform(pts_arr, matrix)
            final_pts = [[round(float(p[0][0]), 1), round(float(p[0][1]), 1)] for p in persp_pts]
        else:
            final_pts = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in warped_pts]

        transformed_lines.append({
            "line_idx": l_meta["line_idx"],
            "para_idx": l_meta["para_idx"],
            "is_para_start": l_meta["is_para_start"],
            "is_heading": l_meta["is_heading"],
            "is_bold": l_meta["is_bold"],
            "text": l_meta["text"],
            "polygon": final_pts
        })

    return img_final, transformed_lines

def load_a4_corpus_units(corpus_path):
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw = f.read()

    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    headings = []
    paragraphs_pool = []

    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if len(lines) == 1 and (lines[0].startswith(('꩑', '꩒', '꩓', '꩔', '꩕')) or len(lines[0]) < 40):
            headings.append(lines[0])
        else:
            # Full text of the block as a coherent paragraph
            full_block_text = " ".join(lines)
            paragraphs_pool.append(full_block_text)

    return headings, paragraphs_pool

def wrap_paragraph_into_lines(para_text, font, max_first_line_w, max_body_line_w):
    """
    Wraps paragraph text into lines fitting the width.
    First line has max_first_line_w (due to indentation).
    Subsequent continuation lines have max_body_line_w.
    """
    words = para_text.split()
    lines = []
    curr_words = []
    
    is_first = True
    for w in words:
        test_words = curr_words + [w]
        test_str = " ".join(test_words)
        try:
            tb = font.getbbox(test_str)
            tw = tb[2] - tb[0]
        except Exception:
            tw = len(test_str) * 16
            
        limit_w = max_first_line_w if is_first else max_body_line_w
        if tw <= limit_w or not curr_words:
            curr_words.append(w)
        else:
            lines.append(" ".join(curr_words))
            curr_words = [w]
            is_first = False

    if curr_words:
        lines.append(" ".join(curr_words))

    return lines

def generate_a4_benchmark_suite(corpus_path, output_dir, start_id=201, num_samples=200):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    headings_pool, paragraphs_pool = load_a4_corpus_units(corpus_path)
    
    # A4 dimensions at 120 DPI
    PAGE_W = 1050
    PAGE_H = 1485
    MARGIN_LEFT = 70
    MARGIN_RIGHT = 70
    MARGIN_TOP = 80
    MARGIN_BOTTOM = 80
    MAX_BODY_W = PAGE_W - MARGIN_LEFT - MARGIN_RIGHT # 910px
    INDENT_PX = 50
    MAX_FIRST_LINE_W = MAX_BODY_W - INDENT_PX # 860px

    samples_per_tier = num_samples // 5
    gt_records = []

    tier_configs = {
        1: {
            "name": "Level 1 (A4 Clean - Standard Leading)",
            "gap_range": (20, 28),
            "para_gap_add": (22, 35),
            "wave_amp_range": (0.0, 0.0),
            "wave_freq_range": (0.0, 0.0),
            "persp_range": (0.0, 0.0)
        },
        2: {
            "name": "Level 2 (A4 Mild - Real Book Texture)",
            "gap_range": (14, 20),
            "para_gap_add": (18, 28),
            "wave_amp_range": (0.0, 0.0),
            "wave_freq_range": (0.0, 0.0),
            "persp_range": (0.005, 0.010)
        },
        3: {
            "name": "Level 3 (A4 Medium - Tight Leading)",
            "gap_range": (8, 13),
            "para_gap_add": (15, 22),
            "wave_amp_range": (1.0, 1.8),
            "wave_freq_range": (0.005, 0.008),
            "persp_range": (0.012, 0.020)
        },
        4: {
            "name": "Level 4 (A4 Hard - Wavy Pages & Tilt)",
            "gap_range": (6, 10),
            "para_gap_add": (12, 18),
            "wave_amp_range": (2.5, 4.5),
            "wave_freq_range": (0.006, 0.010),
            "persp_range": (0.022, 0.035)
        },
        5: {
            "name": "Level 5 (A4 Extreme - Tight Wavy & Vignette)",
            "gap_range": (4, 7),
            "para_gap_add": (10, 15),
            "wave_amp_range": (4.0, 6.5),
            "wave_freq_range": (0.007, 0.012),
            "persp_range": (0.035, 0.050)
        }
    }

    sample_id = start_id
    for tier in range(1, 6):
        cfg = tier_configs[tier]
        print(f"📦 Generating {samples_per_tier} A4 pages for {cfg['name']} (Sample {sample_id} to {sample_id + samples_per_tier - 1})...")

        for _ in range(samples_per_tier):
            font_size_body = random.randint(20, 22)
            font_size_head = random.randint(24, 27)
            font_body = get_font(FONT_REGULAR, font_size_body)
            font_bold = get_font(FONT_BOLD, font_size_head)
            font_body_bold = get_font(FONT_BOLD, font_size_body)

            gap_px = random.randint(cfg["gap_range"][0], cfg["gap_range"][1])
            y_spacing_body = font_size_body + gap_px

            # Compose 2 to 5 sections/paragraphs
            num_paragraphs = random.randint(2, 5)
            paragraphs_data = []

            # Page title / Heading
            include_heading = random.random() < 0.85
            if include_heading:
                head_text = random.choice(headings_pool)
                paragraphs_data.append({
                    "is_heading": True,
                    "is_bold": True,
                    "text": head_text,
                    "font": font_bold,
                    "font_size": font_size_head,
                    "lines": [head_text]
                })

            # Body paragraphs
            picked_paras = random.sample(paragraphs_pool, min(num_paragraphs, len(paragraphs_pool)))
            for p_text in picked_paras:
                p_lines = wrap_paragraph_into_lines(p_text, font_body, MAX_FIRST_LINE_W, MAX_BODY_W)
                if len(p_lines) > 7:
                    # Truncate to fit page
                    p_lines = p_lines[:random.randint(4, 7)]
                    if not p_lines[-1].endswith(('꩞', '꩝')):
                        p_lines[-1] += random.choice(['꩞', '꩝'])
                
                # Check if paragraph has bold emphasis on first line or keyword
                p_bold_lead = random.random() < 0.25
                paragraphs_data.append({
                    "is_heading": False,
                    "is_bold": p_bold_lead,
                    "text": " ".join(p_lines),
                    "font": font_body,
                    "font_size": font_size_body,
                    "lines": p_lines
                })

            # Check total lines and limit to 25 to fit A4 page
            total_lines_estimate = sum(len(p["lines"]) for p in paragraphs_data)
            while total_lines_estimate > 25 and len(paragraphs_data) > 2:
                paragraphs_data.pop()
                total_lines_estimate = sum(len(p["lines"]) for p in paragraphs_data)

            # Render image
            bg_np = create_a4_background(PAGE_W, PAGE_H, tier)
            bg_pil = Image.fromarray(cv2.cvtColor(bg_np, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(bg_pil)

            line_boxes = []
            curr_y = MARGIN_TOP
            global_line_idx = 0

            gt_paragraphs_meta = []
            flow_boundaries = []

            for para_idx, p_obj in enumerate(paragraphs_data):
                is_head = p_obj["is_heading"]
                p_lines = p_obj["lines"]
                p_font = p_obj["font"]
                p_size = p_obj["font_size"]
                y_spacing = p_size + gap_px

                para_line_indices = []

                for l_in_p_idx, line_text in enumerate(p_lines):
                    is_p_start = (l_in_p_idx == 0)
                    is_line_bold = p_obj["is_bold"] if is_head else (p_obj["is_bold"] and is_p_start)
                    cur_font = font_bold if is_head else (font_body_bold if is_line_bold else font_body)

                    if is_head:
                        # Heading can be centered or indented
                        tb = cur_font.getbbox(line_text)
                        tw = tb[2] - tb[0]
                        th = tb[3] - tb[1]
                        line_x = MARGIN_LEFT + random.randint(10, 40)
                    elif is_p_start:
                        # First line indented
                        line_x = MARGIN_LEFT + INDENT_PX
                        tb = cur_font.getbbox(line_text)
                        tw = tb[2] - tb[0]
                        th = tb[3] - tb[1]
                    else:
                        # Continuation line at left margin
                        line_x = MARGIN_LEFT
                        tb = cur_font.getbbox(line_text)
                        tw = tb[2] - tb[0]
                        th = tb[3] - tb[1]

                    # Ink color
                    if tier == 1:
                        ink = (random.randint(10, 25), random.randint(10, 25), random.randint(10, 25))
                    elif tier in (2, 3):
                        ink = (random.randint(20, 45), random.randint(20, 40), random.randint(20, 40))
                    else:
                        ink = (random.randint(30, 65), random.randint(25, 60), random.randint(25, 55))

                    draw.text((line_x, curr_y), line_text, font=cur_font, fill=ink)

                    pad_x = 3
                    pad_y = 1 if gap_px <= 6 else 2
                    box = [
                        [line_x - pad_x, curr_y - pad_y],
                        [line_x + tw + pad_x, curr_y - pad_y],
                        [line_x + tw + pad_x, curr_y + th + pad_y],
                        [line_x - pad_x, curr_y + th + pad_y]
                    ]

                    line_boxes.append({
                        "line_idx": global_line_idx,
                        "para_idx": para_idx,
                        "is_para_start": is_p_start,
                        "is_heading": is_head,
                        "is_bold": is_line_bold,
                        "text": line_text,
                        "box": box
                    })

                    para_line_indices.append(global_line_idx)
                    global_line_idx += 1
                    curr_y += y_spacing

                # Paragraph gap
                p_gap_add = random.randint(cfg["para_gap_add"][0], cfg["para_gap_add"][1])
                curr_y += p_gap_add

                gt_paragraphs_meta.append({
                    "para_id": para_idx,
                    "is_heading": is_head,
                    "text": " ".join(p_lines),
                    "line_indices": para_line_indices,
                    "num_lines": len(para_line_indices)
                })

            # Create ground-truth boundary continuation labels
            # For each adjacent pair of lines (i, i+1):
            # is_continuation = True if they belong to the same paragraph, False if hard break
            for i in range(len(line_boxes) - 1):
                same_para = (line_boxes[i]["para_idx"] == line_boxes[i + 1]["para_idx"])
                flow_boundaries.append({
                    "curr_line_idx": i,
                    "next_line_idx": i + 1,
                    "is_continuation": same_para,
                    "is_hard_break": not same_para
                })

            # Geometric transforms
            wave_amp = random.uniform(cfg["wave_amp_range"][0], cfg["wave_amp_range"][1])
            wave_freq = random.uniform(cfg["wave_freq_range"][0], cfg["wave_freq_range"][1]) if wave_amp > 0 else 0.0
            wave_phase = random.uniform(0, 2 * math.pi) if wave_amp > 0 else 0.0
            persp_amt = random.uniform(cfg["persp_range"][0], cfg["persp_range"][1])

            cv_img = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)
            warped_img, transformed_lines = apply_a4_warping(
                cv_img, line_boxes, wave_amp, wave_freq, wave_phase, persp_amt
            )

            if tier >= 4 and random.random() < 0.35:
                warped_img = cv2.GaussianBlur(warped_img, (3, 3), 0.4)

            # Save A4 image
            filename = f"test_{sample_id:03d}_a4_lvl{tier}.png"
            file_path = os.path.join(images_dir, filename)
            cv2.imwrite(file_path, warped_img)

            record = {
                "sample_id": sample_id,
                "filename": filename,
                "image_path": os.path.relpath(file_path, BENCHMARK_DIR).replace("\\", "/"),
                "level": tier,
                "level_name": cfg["name"],
                "width": PAGE_W,
                "height": PAGE_H,
                "num_lines": len(transformed_lines),
                "num_paragraphs": len(gt_paragraphs_meta),
                "wave_amp": round(wave_amp, 2),
                "wave_freq": round(wave_freq, 5),
                "perspective_factor": round(persp_amt, 4),
                "gap_px": gap_px,
                "lines": transformed_lines,
                "paragraphs": gt_paragraphs_meta,
                "flow_boundaries": flow_boundaries
            }
            gt_records.append(record)
            sample_id += 1

    gt_file = os.path.join(output_dir, "benchmark_a4_gt.json")
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(gt_records, f, ensure_ascii=False, indent=2)

    print(f"🎉 Generated {len(gt_records)} A4 benchmark test images successfully!")
    print(f"   - Annotations saved to: {gt_file}")
    return gt_file

if __name__ == "__main__":
    corpus_file = os.path.join(BENCHMARK_DIR, "data", "benchmark_corpus.txt")
    output_dir = os.path.join(BENCHMARK_DIR, "data")
    generate_a4_benchmark_suite(corpus_file, output_dir, start_id=201, num_samples=200)
