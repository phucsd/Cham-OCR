#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic Text Detection Dataset Generator for Cham OCR (Phase 4).
Generates realistic multi-line Cham document and manuscript pages with
exact polygon bounding box labels in PaddleOCR SimpleDataSet format.
"""

import os
import sys
import cv2
import json
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Force UTF-8 stdout encoding on Windows
if sys.stdout and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Monkeypatch NumPy 2.x compatibility
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

_FONT_CACHE = {}

def get_cached_font(font_path, size):
    key = (font_path, size)
    if key not in _FONT_CACHE:
        if os.path.exists(font_path):
            _FONT_CACHE[key] = ImageFont.truetype(font_path, size)
        else:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]

def apply_old_paper_background(width, height):
    """
    Generates a realistic antique parchment/paper background with varied tones,
    vignetting, and stains.
    """
    base_r = random.randint(220, 245)
    base_g = random.randint(210, 235)
    base_b = random.randint(180, 210)
    
    img = np.full((height, width, 3), (base_b, base_g, base_r), dtype=np.uint8)
    
    # 1. Low frequency gradient/shading
    grid_h, grid_w = max(4, height // 60), max(4, width // 60)
    low_noise = np.random.normal(0, 14, (grid_h, grid_w)).astype(np.float32)
    low_noise = cv2.resize(low_noise, (width, height), interpolation=cv2.INTER_CUBIC)
    
    # 2. Fine grain noise
    fine_noise = np.random.normal(0, 5, (height, width)).astype(np.float32)
    total_noise = low_noise + fine_noise
    for c in range(3):
        channel = img[:, :, c].astype(np.float32) + total_noise
        img[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
    # 3. Vignette effect (darker edges)
    X = np.linspace(-1, 1, width)
    Y = np.linspace(-1, 1, height)
    mesh_x, mesh_y = np.meshgrid(X, Y)
    dist = np.sqrt(mesh_x**2 + mesh_y**2)
    vignette = 1.0 - np.clip((dist - 0.5) * 0.35, 0, 0.45)
    for c in range(3):
        img[:, :, c] = (img[:, :, c].astype(np.float32) * vignette).astype(np.uint8)
        
    # 4. Optional water/ink stain spots
    if random.random() < 0.5:
        num_spots = random.randint(1, 3)
        for _ in range(num_spots):
            cx = random.randint(60, width - 60)
            cy = random.randint(60, height - 60)
            radius = random.randint(20, 50)
            cv2.circle(img, (cx, cy), radius, (max(0, base_b - 25), max(0, base_g - 25), max(0, base_r - 30)), -1)
        img = cv2.GaussianBlur(img, (31, 31), 0)
            
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def warp_points(pts, matrix, amp, freq, phase):
    """
    Applies perspective transformation and non-linear wave deformation
    to ground-truth polygon vertices.
    """
    pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    # 1. Perspective transformation
    warped_persp = cv2.perspectiveTransform(pts_arr, matrix)
    
    # 2. Wave deformation (y_w = y + dy)
    final_pts = []
    for pt in warped_persp:
        x, y = pt[0]
        dy = amp * np.sin(x * freq + phase)
        final_pts.append([round(float(x), 1), round(float(y + dy), 1)])
        
    return final_pts

def draw_distorted_page_with_labels(lines, fonts, width=800, height=650):
    font_path = random.choice(fonts)
    font_size = random.randint(20, 26)
    font = get_cached_font(font_path, font_size)
    
    bg = apply_old_paper_background(width, height)
    draw = ImageDraw.Draw(bg)
    
    y_spacing = random.randint(font_size + 24, font_size + 42)
    y_start = random.randint(40, 70)
    
    line_boxes = []
    for idx, text in enumerate(lines):
        y_pos = y_start + idx * y_spacing
        if y_pos + font_size + 15 > height:
            break
            
        x_pos = random.randint(40, 80)
        ink_r = random.randint(20, 45)
        ink_g = random.randint(20, 40)
        ink_b = random.randint(20, 38)
        draw.text((x_pos, y_pos), text, font=font, fill=(ink_r, ink_g, ink_b))
        
        try:
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = len(text) * 14, font_size
            
        pad_x = 4
        pad_y = 3
        box = [
            [x_pos - pad_x, y_pos - pad_y],
            [x_pos + text_w + pad_x, y_pos - pad_y],
            [x_pos + text_w + pad_x, y_pos + text_h + pad_y],
            [x_pos - pad_x, y_pos + text_h + pad_y]
        ]
        line_boxes.append((text, box))
        
    img_np = np.array(bg)
    h, w = img_np.shape[:2]
    
    amp = random.uniform(3.0, 8.0)
    freq = random.uniform(0.004, 0.010)
    phase = random.uniform(0, 2 * np.pi)
    
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    tx1 = random.uniform(0, w * 0.03)
    ty1 = random.uniform(0, h * 0.03)
    tx2 = random.uniform(0, w * 0.03)
    ty2 = random.uniform(0, h * 0.03)
    tx3 = random.uniform(0, w * 0.03)
    ty3 = random.uniform(0, h * 0.03)
    tx4 = random.uniform(0, w * 0.03)
    ty4 = random.uniform(0, h * 0.03)
    
    pts2 = np.float32([
        [0 + tx1, 0 + ty1],
        [w - tx2, 0 + ty2],
        [0 + tx3, h - ty3],
        [w - tx4, h - ty4]
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    
    x_idx = np.arange(w)
    y_idx = np.arange(h)
    X, Y = np.meshgrid(x_idx, y_idx)
    dy = amp * np.sin(X * freq + phase)
    map_x = X.astype(np.float32)
    map_y = (Y + dy).astype(np.float32)
    img_wave = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    img_warped = cv2.warpPerspective(img_wave, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    warped_labels = []
    for text, box in line_boxes:
        warped_box = warp_points(box, matrix, amp, freq, phase)
        warped_labels.append({
            "transcription": text,
            "points": warped_box
        })
        
    return Image.fromarray(img_warped), warped_labels

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Cham Detection dataset")
    parser.add_argument("--num_train", type=int, default=50, help="Number of training pages")
    parser.add_argument("--num_val", type=int, default=10, help="Number of validation pages")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    print("📂 Starting Cham Text Detection Synthetic Generator (Phase 4)...")
    
    corpus_file = os.path.join(PROJECT_ROOT, "data", "corpus", "cham_text.txt")
    if not os.path.exists(corpus_file):
        corpus_file = os.path.join(PROJECT_ROOT, "output_v22", "paddleocr_cham_finetune", "data", "corpus", "cham_text.txt")
        
    if not os.path.exists(corpus_file):
        print(f"❌ Corpus file not found at {corpus_file}!")
        return
        
    with open(corpus_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    print(f"  Loaded {len(sentences)} corpus lines.")
        
    fonts_dir = os.path.join(PROJECT_ROOT, "data", "fonts")
    if not os.path.exists(fonts_dir):
        fonts_dir = os.path.join(PROJECT_ROOT, "output_v22", "paddleocr_cham_finetune", "data", "fonts")
        
    fonts = [
        os.path.join(fonts_dir, f) for f in ["NotoSansCham-Regular.ttf", "NotoSansCham-Bold.ttf", "NotoSansCham-Black.ttf"]
        if os.path.exists(os.path.join(fonts_dir, f))
    ]
    if not fonts:
        fonts = [os.path.join(fonts_dir, "NotoSansCham-Regular.ttf")]
    print(f"  Using fonts: {[os.path.basename(f) for f in fonts]}")
        
    det_root = args.output_dir or os.path.join(PROJECT_ROOT, "data", "detector")
    train_img_dir = os.path.join(det_root, "train_images")
    val_img_dir = os.path.join(det_root, "val_images")
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    
    print(f"  Sinh {args.num_train} trang huấn luyện (train_images)...")
    train_label_entries = []
    for i in range(args.num_train):
        num_lines = random.randint(4, 9)
        lines = random.sample(sentences, min(num_lines, len(sentences)))
        img, labels = draw_distorted_page_with_labels(lines, fonts)
        
        img_name = f"det_train_{i:05d}.png"
        img.save(os.path.join(train_img_dir, img_name))
        train_label_entries.append(f"train_images/{img_name}\t{json.dumps(labels, ensure_ascii=False)}")
        
    with open(os.path.join(det_root, "det_train_label.txt"), "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write("\n".join(train_label_entries) + "\n")
        
    print(f"  Sinh {args.num_val} trang kiểm thử (val_images)...")
    val_label_entries = []
    for i in range(args.num_val):
        num_lines = random.randint(4, 9)
        lines = random.sample(sentences, min(num_lines, len(sentences)))
        img, labels = draw_distorted_page_with_labels(lines, fonts)
        
        img_name = f"det_val_{i:05d}.png"
        img.save(os.path.join(val_img_dir, img_name))
        val_label_entries.append(f"val_images/{img_name}\t{json.dumps(labels, ensure_ascii=False)}")
        
    with open(os.path.join(det_root, "det_val_label.txt"), "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write("\n".join(val_label_entries) + "\n")
        
    print(f"🎉 Hoàn tất sinh tập dữ liệu Detection tại: {det_root}")
    print(f"   - Train: {len(train_label_entries)} ảnh")
    print(f"   - Val: {len(val_label_entries)} ảnh")

if __name__ == '__main__':
    main()
