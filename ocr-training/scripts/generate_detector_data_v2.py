#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic Text Detection Dataset Generator for Cham OCR (V2 - Tight Line Hard Cases).
Generates realistic multi-line Cham document and manuscript pages with
exact polygon bounding box labels in PaddleOCR SimpleDataSet format.

Specially engineered to fix close line segmentation and tight leading issues:
- 50% Ultra-Tight lines (y_spacing = font_size + 3..10px)
- 30% Medium-Tight lines (y_spacing = font_size + 11..20px)
- 20% Standard lines (y_spacing = font_size + 21..35px)
- Multi-paragraph layouts with indentation and short ending lines
- Multiprocessing for high-throughput generation on Cloud GPU/CPU environments.
"""

import os
import sys
import cv2
import json
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ProcessPoolExecutor

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
            try:
                _FONT_CACHE[key] = ImageFont.truetype(font_path, size)
            except Exception:
                _FONT_CACHE[key] = ImageFont.load_default()
        else:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]

def apply_realistic_background(width, height):
    """
    Generates realistic paper texture: modern book paper, aged newsprint, or antique parchment.
    """
    style = random.random()
    if style < 0.40:
        # 1. Clean / Modern Book Paper (off-white, slight grain)
        base_val = random.randint(240, 253)
        img = np.full((height, width, 3), (base_val, base_val, base_val), dtype=np.uint8)
        noise = np.random.normal(0, 3, (height, width)).astype(np.float32)
        for c in range(3):
            img[:, :, c] = np.clip(img[:, :, c].astype(np.float32) + noise, 0, 255).astype(np.uint8)
    else:
        # 2. Aged Paper / Antique Parchment
        base_r = random.randint(220, 246)
        base_g = random.randint(210, 238)
        base_b = random.randint(185, 215)
        img = np.full((height, width, 3), (base_b, base_g, base_r), dtype=np.uint8)
        
        # Low frequency shading
        grid_h, grid_w = max(4, height // 50), max(4, width // 50)
        low_noise = np.random.normal(0, 10, (grid_h, grid_w)).astype(np.float32)
        low_noise = cv2.resize(low_noise, (width, height), interpolation=cv2.INTER_CUBIC)
        
        fine_noise = np.random.normal(0, 4, (height, width)).astype(np.float32)
        total_noise = low_noise + fine_noise
        for c in range(3):
            channel = img[:, :, c].astype(np.float32) + total_noise
            img[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
            
        # Optional vignette
        if random.random() < 0.5:
            X = np.linspace(-1, 1, width)
            Y = np.linspace(-1, 1, height)
            mesh_x, mesh_y = np.meshgrid(X, Y)
            dist = np.sqrt(mesh_x**2 + mesh_y**2)
            vignette = 1.0 - np.clip((dist - 0.5) * 0.25, 0, 0.35)
            for c in range(3):
                img[:, :, c] = (img[:, :, c].astype(np.float32) * vignette).astype(np.uint8)
                
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def warp_points(pts, matrix, amp, freq, phase):
    """
    Applies perspective transformation and non-linear wave deformation to polygon vertices.
    """
    pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    warped_persp = cv2.perspectiveTransform(pts_arr, matrix)
    
    final_pts = []
    for pt in warped_persp:
        x, y = pt[0]
        dy = amp * np.sin(x * freq + phase) if amp > 0 else 0
        final_pts.append([round(float(x), 1), round(float(y + dy), 1)])
        
    return final_pts

def generate_single_page(args_tuple):
    """
    Worker function to generate a single labeled synthetic page.
    """
    idx, sentences, fonts, out_dir, prefix = args_tuple
    
    width = random.choice([800, 900, 1024])
    height = random.choice([600, 750, 900])
    
    font_path = random.choice(fonts)
    font_size = random.randint(18, 26)
    font = get_cached_font(font_path, font_size)
    
    bg = apply_realistic_background(width, height)
    draw = ImageDraw.Draw(bg)
    
    # Line spacing distribution:
    # 50% Ultra-Tight (3..10px), 30% Medium-Tight (11..20px), 20% Standard (21..35px)
    r_mode = random.random()
    if r_mode < 0.50:
        gap = random.randint(3, 10)
    elif r_mode < 0.80:
        gap = random.randint(11, 20)
    else:
        gap = random.randint(21, 35)
        
    y_spacing = font_size + gap
    y_start = random.randint(30, 60)
    
    # Paragraph structure
    num_paras = random.randint(1, 3)
    lines_per_para = []
    for _ in range(num_paras):
        lines_per_para.append(random.randint(2, 6))
        
    line_boxes = []
    current_y = y_start
    sent_idx = random.randint(0, max(0, len(sentences) - 20))
    
    for para_i, num_l in enumerate(lines_per_para):
        for line_i in range(num_l):
            if current_y + font_size + 15 > height:
                break
                
            text = sentences[sent_idx % len(sentences)]
            sent_idx += 1
            
            # Paragraph formatting:
            # - First line indented
            # - Last line of paragraph may be shorter
            is_first_line = (line_i == 0)
            is_last_line = (line_i == num_l - 1)
            
            x_pos = random.randint(60, 90) if is_first_line else random.randint(30, 45)
            
            # Trim text if too long for width
            while len(text) > 5:
                try:
                    tb = font.getbbox(text)
                    tw = tb[2] - tb[0]
                except Exception:
                    tw = len(text) * int(font_size * 0.7)
                if x_pos + tw < width - 40:
                    break
                parts = text.rsplit(' ', 1)
                text = parts[0] if len(parts) > 1 else text[:-2]
                
            if is_last_line and random.random() < 0.6:
                words = text.split()
                if len(words) > 3:
                    cut = random.randint(3, len(words) - 1)
                    text = ' '.join(words[:cut])
                    if not text.endswith(('꩞', '꩝')):
                        text += random.choice(['꩞', '꩝'])
                        
            if not text.strip():
                continue
                
            ink_r = random.randint(15, 50)
            ink_g = random.randint(15, 45)
            ink_b = random.randint(15, 45)
            draw.text((x_pos, current_y), text, font=font, fill=(ink_r, ink_g, ink_b))
            
            try:
                bbox = font.getbbox(text)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except Exception:
                text_w = len(text) * int(font_size * 0.7)
                text_h = font_size
                
            pad_x = random.randint(2, 4)
            # When lines are ultra tight, pad_y must be tight to avoid overlapping labels
            pad_y = 1 if gap <= 6 else random.randint(2, 3)
            
            box = [
                [x_pos - pad_x, current_y - pad_y],
                [x_pos + text_w + pad_x, current_y - pad_y],
                [x_pos + text_w + pad_x, current_y + text_h + pad_y],
                [x_pos - pad_x, current_y + text_h + pad_y]
            ]
            line_boxes.append((text, box))
            current_y += y_spacing
            
        # Paragraph gap
        current_y += random.randint(12, 24)
        if current_y + font_size + 15 > height:
            break
            
    if not line_boxes:
        return None
        
    img_np = np.array(bg)
    h, w = img_np.shape[:2]
    
    # Geometric transformations: mild perspective & wave
    has_distortion = random.random() < 0.65
    if has_distortion:
        amp = random.uniform(1.5, 4.0)
        freq = random.uniform(0.005, 0.010)
        phase = random.uniform(0, 2 * np.pi)
        
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        skew_amt = 0.02
        pts2 = np.float32([
            [random.uniform(0, w * skew_amt), random.uniform(0, h * skew_amt)],
            [w - random.uniform(0, w * skew_amt), random.uniform(0, h * skew_amt)],
            [random.uniform(0, w * skew_amt), h - random.uniform(0, h * skew_amt)],
            [w - random.uniform(0, w * skew_amt), h - random.uniform(0, h * skew_amt)]
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
            w_box = warp_points(box, matrix, amp, freq, phase)
            warped_labels.append({
                "transcription": text,
                "points": w_box
            })
    else:
        img_warped = img_np
        warped_labels = [{
            "transcription": text,
            "points": box
        } for text, box in line_boxes]
        
    img_name = f"{prefix}_{idx:05d}.png"
    img_path = os.path.join(out_dir, img_name)
    Image.fromarray(img_warped).save(img_path)
    
    return f"{prefix}/{img_name}\t{json.dumps(warped_labels, ensure_ascii=False)}"

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Cham Detection dataset V2")
    parser.add_argument("--num_train", type=int, default=1000, help="Number of training pages")
    parser.add_argument("--num_val", type=int, default=100, help="Number of validation pages")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes")
    args = parser.parse_args()

    print(f"🚀 Cham Detection Generator V2 (Tight Line Hard-Cases)")
    print(f"   - Target Train: {args.num_train} pages")
    print(f"   - Target Val:   {args.num_val} pages")
    print(f"   - Workers:      {args.workers}")
    
    corpus_candidates = [
        os.path.join(PROJECT_ROOT, "data", "corpus", "cham_text.txt"),
        os.path.join(PROJECT_ROOT, "ocr-training", "data", "corpus", "cham_text.txt"),
        os.path.join(os.getcwd(), "data", "corpus", "cham_text.txt"),
        os.path.join(os.getcwd(), "ocr-training", "data", "corpus", "cham_text.txt"),
        os.path.join(PROJECT_ROOT, "output_v22", "paddleocr_cham_finetune", "data", "corpus", "cham_text.txt"),
        os.path.join(PROJECT_ROOT, "data", "cham-ocr-v5-assets", "corpus", "corpus", "cham_text.txt")
    ]
    corpus_file = None
    for cand in corpus_candidates:
        if os.path.exists(cand):
            corpus_file = cand
            break

    if not corpus_file:
        print(f"📥 Chưa tìm thấy corpus cục bộ, đang tải cham_text.txt từ GitHub...")
        corpus_file = os.path.join(PROJECT_ROOT, "data", "corpus", "cham_text.txt")
        os.makedirs(os.path.dirname(corpus_file), exist_ok=True)
        try:
            import urllib.request
            raw_url = "https://raw.githubusercontent.com/phucsd/Cham-OCR/main/ocr-training/data/corpus/cham_text.txt"
            urllib.request.urlretrieve(raw_url, corpus_file)
            print(f"✅ Đã tải corpus thành công về: {corpus_file}")
        except Exception as e:
            print(f"⚠️ Không thể tải từ GitHub ({e}), tự động sinh corpus dự phòng...")
            # Fallback Cham sentence generator
            cham_consonants = list("ꨆꨇꨈꨉꨊꨋꨌꨍꨎꨏꨐꨑꨒꨓꨔꨕꨖꨗꨘꨙꨚꨛꨜꨝꨞꨟꨠꨡꨢꨣꨤꨥꨦꨧꨨꨀ")
            cham_vowels = list("ꨩꨪꨫꨬꨭꨮꨯꨰꨱꨲꨴꨵ")
            cham_finals = list("ꩀꩃꩌꩍꩆꩉꩊꩂꩅ")
            synthetic_corpus = []
            for _ in range(5000):
                words = []
                for _ in range(random.randint(4, 10)):
                    syl = random.choice(cham_consonants)
                    if random.random() < 0.3: syl += "ꨳ"
                    if random.random() < 0.7: syl += random.choice(cham_vowels)
                    if random.random() < 0.5: syl += random.choice(cham_finals)
                    words.append(syl)
                synthetic_corpus.append(" ".join(words) + random.choice([" ꩞", " ꩝"]))
            with open(corpus_file, "w", encoding="utf-8") as f:
                f.write("\n".join(synthetic_corpus) + "\n")
            print(f"✅ Đã tạo {len(synthetic_corpus)} câu corpus dự phòng.")

    with open(corpus_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    print(f"   - Đã nạp {len(sentences)} dòng ngữ liệu Chăm ({corpus_file}).")
    
    fonts_dir = os.path.join(PROJECT_ROOT, "data", "fonts")
    if not os.path.exists(fonts_dir):
        fonts_dir = os.path.join(PROJECT_ROOT, "output_v22", "paddleocr_cham_finetune", "data", "fonts")
        
    font_names = ["NotoSansCham-Regular.ttf", "NotoSansCham-Bold.ttf", "NotoSansCham-Black.ttf"]
    fonts = [os.path.join(fonts_dir, f) for f in font_names if os.path.exists(os.path.join(fonts_dir, f))]
    if not fonts:
        fonts = [os.path.join(fonts_dir, "NotoSansCham-Regular.ttf")]
    print(f"   - Using fonts: {[os.path.basename(f) for f in fonts]}")
    
    det_root = args.output_dir or os.path.join(PROJECT_ROOT, "data", "detector_v2")
    train_dir = os.path.join(det_root, "train_images")
    val_dir = os.path.join(det_root, "val_images")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 1. Generate Training Data
    print(f"\n⏳ Generating {args.num_train} training pages...")
    train_tasks = [(i, sentences, fonts, train_dir, "train_images") for i in range(args.num_train)]
    train_labels = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in executor.map(generate_single_page, train_tasks):
            if res:
                train_labels.append(res)
                if len(train_labels) % 200 == 0:
                    print(f"   ... Generated {len(train_labels)}/{args.num_train} training pages")
                    
    with open(os.path.join(det_root, "det_train_label.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(train_labels) + "\n")
    print(f"✅ Finished training data: {len(train_labels)} pages saved.")
    
    # 2. Generate Validation Data
    print(f"\n⏳ Generating {args.num_val} validation pages...")
    val_tasks = [(i, sentences, fonts, val_dir, "val_images") for i in range(args.num_val)]
    val_labels = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in executor.map(generate_single_page, val_tasks):
            if res:
                val_labels.append(res)
                if len(val_labels) % 50 == 0:
                    print(f"   ... Generated {len(val_labels)}/{args.num_val} validation pages")
                    
    with open(os.path.join(det_root, "det_val_label.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(val_labels) + "\n")
    print(f"✅ Finished validation data: {len(val_labels)} pages saved.")
    
    print(f"\n🎉 Dataset successfully generated at: {det_root}")
    print(f"   - Label format: PaddleOCR SimpleDataSet (points + transcription)")

if __name__ == '__main__':
    main()
