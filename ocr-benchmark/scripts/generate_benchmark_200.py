#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic Benchmark Generator for Cham OCR (200 Test Cases Across 5 Difficulty Tiers)
Based on authentic Cham document corpus.
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
random.seed(42)
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)

FONT_REGULAR = os.path.join(PROJECT_ROOT, "ocr-training", "data", "fonts", "NotoSansCham-Regular.ttf")
FONT_BOLD = os.path.join(PROJECT_ROOT, "ocr-training", "data", "fonts", "NotoSansCham-Bold.ttf")

if not os.path.exists(FONT_REGULAR):
    raise FileNotFoundError(f"Font not found: {FONT_REGULAR}")

_FONT_CACHE = {}

def get_font(path, size):
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]

def create_background(w, h, level):
    """
    Creates background texture matching the difficulty tier.
    """
    if level == 1:
        # Clean white / off-white
        val = random.randint(248, 255)
        bg = np.full((h, w, 3), (val, val, val), dtype=np.uint8)
        return bg

    elif level == 2:
        # Light cream / slight paper grain
        val_r = random.randint(240, 252)
        val_g = random.randint(238, 248)
        val_b = random.randint(228, 242)
        bg = np.full((h, w, 3), (val_b, val_g, val_r), dtype=np.uint8)
        noise = np.random.normal(0, 2.5, (h, w)).astype(np.float32)
        for c in range(3):
            bg[:, :, c] = np.clip(bg[:, :, c].astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return bg

    elif level == 3:
        # Aged paper with soft low-freq clouds
        val_r = random.randint(230, 246)
        val_g = random.randint(220, 238)
        val_b = random.randint(195, 218)
        bg = np.full((h, w, 3), (val_b, val_g, val_r), dtype=np.uint8)
        gh, gw = max(4, h // 40), max(4, w // 40)
        low_noise = np.random.normal(0, 6.0, (gh, gw)).astype(np.float32)
        low_noise = cv2.resize(low_noise, (w, h), interpolation=cv2.INTER_CUBIC)
        noise = low_noise + np.random.normal(0, 3.0, (h, w)).astype(np.float32)
        for c in range(3):
            bg[:, :, c] = np.clip(bg[:, :, c].astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return bg

    elif level == 4:
        # Antique paper with noticeable vignette & tone variation
        val_r = random.randint(225, 242)
        val_g = random.randint(212, 230)
        val_b = random.randint(180, 205)
        bg = np.full((h, w, 3), (val_b, val_g, val_r), dtype=np.uint8)
        gh, gw = max(4, h // 30), max(4, w // 30)
        low_noise = np.random.normal(0, 9.0, (gh, gw)).astype(np.float32)
        low_noise = cv2.resize(low_noise, (w, h), interpolation=cv2.INTER_CUBIC)
        fine_noise = np.random.normal(0, 4.0, (h, w)).astype(np.float32)
        total_noise = low_noise + fine_noise
        for c in range(3):
            bg[:, :, c] = np.clip(bg[:, :, c].astype(np.float32) + total_noise, 0, 255).astype(np.uint8)
        # Vignette
        X = np.linspace(-1, 1, w)
        Y = np.linspace(-1, 1, h)
        mx, my = np.meshgrid(X, Y)
        dist = np.sqrt(mx**2 + my**2)
        vignette = 1.0 - np.clip((dist - 0.5) * 0.25, 0, 0.30)
        for c in range(3):
            bg[:, :, c] = (bg[:, :, c].astype(np.float32) * vignette).astype(np.uint8)
        return bg

    else:
        # Level 5: Aged parchment with blotches, lighting gradient, vignette
        val_r = random.randint(215, 238)
        val_g = random.randint(200, 222)
        val_b = random.randint(165, 195)
        bg = np.full((h, w, 3), (val_b, val_g, val_r), dtype=np.uint8)
        # Lighting gradient across page (one side slightly brighter)
        grad_angle = random.uniform(0, 2 * math.pi)
        X = np.linspace(-1, 1, w)
        Y = np.linspace(-1, 1, h)
        mx, my = np.meshgrid(X, Y)
        lighting = 1.0 + 0.12 * (mx * math.cos(grad_angle) + my * math.sin(grad_angle))
        
        gh, gw = max(4, h // 20), max(4, w // 20)
        low_noise = np.random.normal(0, 12.0, (gh, gw)).astype(np.float32)
        low_noise = cv2.resize(low_noise, (w, h), interpolation=cv2.INTER_CUBIC)
        fine_noise = np.random.normal(0, 5.0, (h, w)).astype(np.float32)
        total_noise = low_noise + fine_noise
        for c in range(3):
            ch = bg[:, :, c].astype(np.float32) * lighting + total_noise
            bg[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
        # Stronger vignette
        dist = np.sqrt(mx**2 + my**2)
        vignette = 1.0 - np.clip((dist - 0.4) * 0.35, 0, 0.40)
        for c in range(3):
            bg[:, :, c] = (bg[:, :, c].astype(np.float32) * vignette).astype(np.uint8)
        return bg

def apply_geometric_warping(img_cv, line_boxes, wave_amp, wave_freq, wave_phase, persp_amt):
    """
    Applies sinusoidal wave remap and perspective warping to image and tracks polygon vertices.
    """
    h, w = img_cv.shape[:2]
    
    # 1. Wave Remap
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

    # 2. Perspective Warp
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

    # 3. Transform polygon points for each line
    transformed_lines = []
    for text, pts in line_boxes:
        # pts is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        # In remap, point at (x, y) moves to (x, y + dy) where dy = amp * sin(x * freq + phase)
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
            "text": text,
            "polygon": final_pts
        })

    return img_final, transformed_lines

def load_corpus_data(corpus_path):
    """Parses Cham corpus into structured text segments."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    blocks = [b.strip() for b in raw_text.split("\n\n") if b.strip()]
    lines_pool = []
    stanzas_pool = []

    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if len(lines) == 1:
            lines_pool.append(lines[0])
        else:
            stanzas_pool.append(lines)
            for l in lines:
                lines_pool.append(l)

    # Also extract individual sentences ending with ꩞ or ꩝
    sentences_pool = []
    for block in blocks:
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Split on Danda or Section Mark
            tokens = []
            curr = ""
            for ch in line:
                curr += ch
                if ch in ['꩞', '꩝', '?']:
                    if curr.strip():
                        tokens.append(curr.strip())
                    curr = ""
            if curr.strip():
                tokens.append(curr.strip())
            for t in tokens:
                if len(t) > 6:
                    sentences_pool.append(t)

    return lines_pool, stanzas_pool, sentences_pool

def generate_benchmark_suite(corpus_path, output_dir, num_samples=200):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    lines_pool, stanzas_pool, sentences_pool = load_corpus_data(corpus_path)
    
    samples_per_level = num_samples // 5
    gt_records = []
    
    level_configs = {
        1: {
            "name": "Level 1 (Easy - Standard)",
            "gap_range": (25, 35),
            "wave_amp_range": (0.0, 0.0),
            "wave_freq_range": (0.0, 0.0),
            "persp_range": (0.0, 0.0),
            "font_size_range": (22, 26),
            "font_choice": [FONT_REGULAR, FONT_BOLD]
        },
        2: {
            "name": "Level 2 (Mild - Real Book Texture)",
            "gap_range": (15, 24),
            "wave_amp_range": (0.0, 0.0),
            "wave_freq_range": (0.0, 0.0),
            "persp_range": (0.005, 0.012),
            "font_size_range": (21, 25),
            "font_choice": [FONT_REGULAR, FONT_BOLD]
        },
        3: {
            "name": "Level 3 (Medium - Tight Spacing)",
            "gap_range": (8, 14),
            "wave_amp_range": (1.0, 2.2),
            "wave_freq_range": (0.005, 0.008),
            "persp_range": (0.015, 0.025),
            "font_size_range": (20, 24),
            "font_choice": [FONT_REGULAR, FONT_BOLD]
        },
        4: {
            "name": "Level 4 (Hard - Wavy & Distorted)",
            "gap_range": (5, 10),
            "wave_amp_range": (3.0, 5.5),
            "wave_freq_range": (0.007, 0.012),
            "persp_range": (0.030, 0.048),
            "font_size_range": (20, 24),
            "font_choice": [FONT_REGULAR, FONT_BOLD]
        },
        5: {
            "name": "Level 5 (Extreme - Combined Degradations)",
            "gap_range": (2, 6),
            "wave_amp_range": (5.5, 8.0),
            "wave_freq_range": (0.008, 0.015),
            "persp_range": (0.045, 0.065),
            "font_size_range": (19, 23),
            "font_choice": [FONT_REGULAR, FONT_BOLD]
        }
    }

    sample_id = 1
    for level in range(1, 6):
        cfg = level_configs[level]
        print(f"📦 Generating {samples_per_level} samples for {cfg['name']}...")
        
        for i in range(samples_per_level):
            # Page layout dimensions
            # Varied: small snippet (2-3 lines), medium paragraph (4-6 lines), full page (7-10 lines)
            layout_type = random.choice(["snippet", "paragraph", "multisection", "stanza"])
            if layout_type == "snippet":
                target_lines_count = random.randint(2, 3)
                page_w = random.choice([750, 850, 950])
            elif layout_type == "paragraph":
                target_lines_count = random.randint(4, 6)
                page_w = random.choice([800, 900, 1000])
            elif layout_type == "multisection":
                target_lines_count = random.randint(7, 10)
                page_w = random.choice([850, 950, 1050])
            else: # stanza
                target_lines_count = random.randint(3, 5)
                page_w = random.choice([800, 900])

            font_path = random.choice(cfg["font_choice"])
            font_size = random.randint(cfg["font_size_range"][0], cfg["font_size_range"][1])
            font = get_font(font_path, font_size)

            gap_px = random.randint(cfg["gap_range"][0], cfg["gap_range"][1])
            y_spacing = font_size + gap_px

            # Compose text lines
            lines_to_render = []
            if layout_type == "stanza" and stanzas_pool:
                chosen_stanza = random.choice(stanzas_pool)
                lines_to_render = list(chosen_stanza[:target_lines_count])
            else:
                for _ in range(target_lines_count):
                    # 40% full sentence, 30% lines_pool, 30% stanza line
                    r_pick = random.random()
                    if r_pick < 0.4:
                        lines_to_render.append(random.choice(sentences_pool))
                    elif r_pick < 0.7:
                        lines_to_render.append(random.choice(lines_pool))
                    else:
                        st = random.choice(stanzas_pool)
                        lines_to_render.append(random.choice(st))

            # Filter or wrap lines to fit page_w
            processed_lines = []
            max_content_w = page_w - 90
            for l in lines_to_render:
                l = l.strip()
                if not l:
                    continue
                # Truncate if exceeds width
                while len(l) > 4:
                    try:
                        tb = font.getbbox(l)
                        tw = tb[2] - tb[0]
                    except Exception:
                        tw = len(l) * int(font_size * 0.7)
                    if tw <= max_content_w:
                        break
                    parts = l.rsplit(" ", 1)
                    l = parts[0] if len(parts) > 1 else l[:-2]
                if l.strip():
                    processed_lines.append(l.strip())

            if not processed_lines:
                processed_lines = [lines_pool[0]]

            # Compute page height
            margin_top = random.randint(30, 50)
            margin_bottom = random.randint(30, 50)
            content_h = len(processed_lines) * y_spacing
            page_h = margin_top + content_h + margin_bottom

            # Render base image
            bg_np = create_background(page_w, page_h, level)
            bg_pil = Image.fromarray(cv2.cvtColor(bg_np, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(bg_pil)

            line_boxes = []
            curr_y = margin_top
            for line_idx, line_text in enumerate(processed_lines):
                # Determine x position: list items or stanzas might have indentation
                if line_text.startswith(('꩑', '꩒', '꩓', '꩔', '꩕')):
                    curr_x = random.randint(40, 60)
                elif line_text.startswith('–'):
                    curr_x = random.randint(50, 70)
                else:
                    curr_x = random.randint(40, 55)

                try:
                    bbox = font.getbbox(line_text)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    y_offset = bbox[1]
                except Exception:
                    tw = len(line_text) * int(font_size * 0.7)
                    th = font_size
                    y_offset = 0

                # Ink color
                if level == 1:
                    ink = (random.randint(10, 30), random.randint(10, 30), random.randint(10, 30))
                elif level in (2, 3):
                    ink = (random.randint(20, 50), random.randint(20, 45), random.randint(20, 45))
                else:
                    ink = (random.randint(30, 70), random.randint(25, 65), random.randint(25, 60))

                draw.text((curr_x, curr_y), line_text, font=font, fill=ink)

                pad_x = 3
                pad_y = 1 if gap_px <= 6 else 2
                box = [
                    [curr_x - pad_x, curr_y - pad_y],
                    [curr_x + tw + pad_x, curr_y - pad_y],
                    [curr_x + tw + pad_x, curr_y + th + pad_y],
                    [curr_x - pad_x, curr_y + th + pad_y]
                ]
                line_boxes.append((line_text, box))
                curr_y += y_spacing

            # Geometric transforms
            wave_amp = random.uniform(cfg["wave_amp_range"][0], cfg["wave_amp_range"][1])
            wave_freq = random.uniform(cfg["wave_freq_range"][0], cfg["wave_freq_range"][1]) if wave_amp > 0 else 0.0
            wave_phase = random.uniform(0, 2 * math.pi) if wave_amp > 0 else 0.0
            persp_amt = random.uniform(cfg["persp_range"][0], cfg["persp_range"][1])

            cv_img = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)
            warped_img, transformed_lines = apply_geometric_warping(
                cv_img, line_boxes, wave_amp, wave_freq, wave_phase, persp_amt
            )

            # Optional slight Gaussian blur for Level 4 & 5 to simulate lens blur/bleed
            if level >= 4 and random.random() < 0.4:
                ksize = random.choice([3])
                warped_img = cv2.GaussianBlur(warped_img, (ksize, ksize), 0.5)

            # Save image
            lvl_tag = f"lvl{level}"
            filename = f"test_{sample_id:03d}_{lvl_tag}.png"
            file_path = os.path.join(images_dir, filename)
            cv2.imwrite(file_path, warped_img)

            # Record ground truth
            record = {
                "sample_id": sample_id,
                "filename": filename,
                "image_path": os.path.relpath(file_path, BENCHMARK_DIR).replace("\\", "/"),
                "level": level,
                "level_name": cfg["name"],
                "width": page_w,
                "height": page_h,
                "font_name": os.path.basename(font_path),
                "font_size": font_size,
                "gap_px": gap_px,
                "wave_amp": round(wave_amp, 2),
                "wave_freq": round(wave_freq, 5),
                "perspective_factor": round(persp_amt, 4),
                "num_lines": len(transformed_lines),
                "lines": transformed_lines
            }
            gt_records.append(record)
            sample_id += 1

    gt_file = os.path.join(output_dir, "benchmark_gt.json")
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(gt_records, f, ensure_ascii=False, indent=2)

    print(f"🎉 Generated {len(gt_records)} benchmark test images successfully!")
    print(f"   - Images directory: {images_dir}")
    print(f"   - Annotations file: {gt_file}")
    return gt_file

if __name__ == "__main__":
    corpus_file = os.path.join(BENCHMARK_DIR, "data", "benchmark_corpus.txt")
    output_dir = os.path.join(BENCHMARK_DIR, "data")
    generate_benchmark_suite(corpus_file, output_dir, num_samples=200)
