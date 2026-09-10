#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Synthetic Dataset Generator for Cham OCR Model V25.
Optimized for 32-core Data Prep CPU Machine on Lightning AI Studios.

Produces 250,000 Train + 25,000 Val line images across 5 core pillars:
1. Standard & Authentic Cham Literature (Po Klong Garai, 57-stanza poem, administration)
2. Extreme Anti-Motion Blur & Defocus Augmentation (Kernels 7x7 to 13x13, downsampling)
3. Intra-line Bilingual Code-Switching (Cham phrase + Vietnamese diacritics + Latin)
4. Full Stanza Digits (1-99: ꩑꩞..꩙꩙꩞) & Boundary Punctuation (꩞, :, –, ?, !)
5. Minimal Pairs Hard-Examples (ꨲ vs ꨶ, ꨯꨮꨩ vs ꨯꨱ, ꩝꩝ variable gap 2-8px)
"""

import os
import sys
import math
import random
import time
import argparse
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# UTF-8 stdout encoding
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)

# Fonts
FONT_CHAM_REG = os.path.join(TRAINING_DIR, "data", "fonts", "NotoSansCham-Regular.ttf")
FONT_CHAM_BOLD = os.path.join(TRAINING_DIR, "data", "fonts", "NotoSansCham-Bold.ttf")

# ==============================================================================
# 1. Ký Tự & Ngữ Liệu Tiếng Chăm & Song Ngữ V25
# ==============================================================================
CHAM_CONSONANTS = list("ꨆꨇꨈꨉꨊꨋꨌꨍꨎꨏꨐꨑꨒꨓꨔꨕꨖꨗꨘꨙꨚꨛꨜꨝꨞꨟꨠꨡꨢꨣꨤꨥꨦꨧꨨꨀꨁꨂꨃꨄꨅ")
CHAM_DIGITS_LIST = list("꩐꩑꩒꩓꩔꩕꩖꩗꩘꩙")

def to_cham_num(n):
    """Chuyển số nguyên n thành chuỗi chữ số Chăm"""
    return "".join(CHAM_DIGITS_LIST[int(d)] for d in str(n))

# Ngữ liệu từ bài thơ 57 khổ và truyền thuyết Po Klong Garai
CHAM_VOCAB_POOL = [
    "ꨛꨯꨮ", "ꨆꨵꨯꨱꩃ", "ꨈꨣꩈ", "ꨧꨯꨱꩃ", "ꨚꩆ", "ꨆꩀ", "ꨨꨤꨩ", "ꨟꨐꨪꩌ", "ꨕꨶꨩ", "ꨝꨯꨱꩍ",
    "ꨝꨰ", "ꨙꨮ", "ꨎꨤꨭꩀ", "ꨟꨐꨭꩌ", "ꨀꨳꨩ", "ꨝꨵꨯꨱꩍ", "ꨝꨩ", "ꨧꨩ", "ꨝꨵꩍ", "ꨀꨇꩆ",
    "ꨗꨯꨱ", "ꨝꨣꨭꨥ", "ꨟꩃ", "ꨓꨮꩊ", "ꨊꨯꨱꩀ", "ꨣꨈꨵꨰ", "ꨚꨢꨯꨱꩌ", "ꨘꩆ", "ꨤꩄ", "ꨆꨳꨮꩃ",
    "ꨧꨤꨪꩍ", "ꨕꨨꨵꩀ", "ꨨꨕꨯꩌ", "ꨀꨣꩀ", "ꨗꨫ", "ꨢꨭꩅ", "ꨚꨰꩀ", "ꨝꨪ", "ꨝꩀ", "ꨆꨮꨭ",
    "ꨝꨴꨬ", "ꨆꨩ", "ꨨꨲꨩ", "ꨡꨯꩍ", "ꨀꨦꨪꩅ", "ꨟꩀ", "ꨟꨰ", "ꨝꨭꩍ", "ꨕꨫ", "ꨕꨤꩌ",
    "ꨅꨩ", "ꨨꨕꨳꨮꩇ", "ꨀꨗꩀ", "ꨓꨌꨯꨱꨥ", "ꨙꩃ", "ꨀꨝꨪꩍ", "ꨤꨝꨭꩀ", "ꨎꨥ", "ꨥꨮꩀ", "ꨢꨮ",
    "ꨐꨭꨩ", "ꨓꨴꨩ", "ꨚꨤꨬ", "ꨂꩀ", "ꨀꨗꨯꨱꩃ", "ꨧꨝꨭꩃ", "ꨈꨮꩇ", "ꨀꨧꨪꩅ", "ꨢꨮꨭ", "ꨕꨘꩀ",
    "ꨎꨮ", "ꨚꨶꨮꩄ", "ꨚꨆꨴꨲꨩ", "ꨟꨪꩆ", "ꨙꨩ", "ꨝꨳꩀ", "ꨚꨗꨶꨮꩄ", "ꨔꨮꨭ", "ꨕꨯꨱꩀ", "ꨓꨳꩆ",
    "ꨀꨶꨮꩆ", "ꨓꨝꨶꨮꩆ", "ꨚꨎꨳꩀ", "ꨆꨵꨮꨭ", "ꨀꨮꩈ", "ꨧꨝꩀ", "ꨈꨗꨴꨮꩍ", "ꨚꩀ", "ꨨꨚꨭꩍ", "ꨤꨪꨎꩃ",
    "ꨝꨵꨬ", "ꨣꨬ", "ꨈꨴꨮꩇ", "ꨧꩃ", "ꨠꨎꨰꩀ", "ꨚꨙꨳꩀ", "ꨤꨯꨩ", "ꨤꨮꩍ", "ꨚꨕꨬ", "ꨈꨵꨰꩍ",
    "ꨀꨤꨩ", "ꨨꨤꨶꨬ", "ꨆꨭꨢꨮꨭ", "ꨤꨪꨔꨬ", "ꨨꨤꨬ", "ꨇꨪꩀ", "ꨙꨪꩍ", "ꨥꩉ", "ꨕꨴꨬ", "ꨁꨗꨩ",
    "ꨈꨭꨣꩈ", "ꨤꨳꩍ", "ꨣꨭꩇ", "ꨤꨶꩀ", "ꨓꨟꨩ", "ꨓꨗꩍ", "ꨣꨪꨢꨩ", "ꨖꨪꩅ", "ꨀꨧꩉ", "ꨚꨓꨯꨱ",
    "ꨇꨪꩆ", "ꨨꨶꩀ", "ꨃꨥ", "ꨧꨣꨥꨩ", "ꨚꨙꩉ", "ꨢꩍ", "ꨄꨰ", "ꨆꨯꨱꩍ", "ꨚꨓꨬ", "ꨓꨭꩍ",
    "ꨓꨌꨬ", "ꨌꨪꩍ", "ꨈꩆ", "ꨆꨴꨲꩍ", "ꨓꨟꨯꨱꩃ", "ꨨꨭꨩ", "ꨗꩌ", "ꨈꩍ", "ꨆꨕꨯꨱꩍ", "ꨕꨨꨵꨮꨭ",
    "ꨌꨙꨳꩀ", "ꨓꨆꨶꨯꩈ", "ꨀꨥꨰꩅ", "ꨗꨆꨓꨯꨮꨩ", "ꨀꨆꨢꨯꨮꩅ", "ꨓꨆꨴꨲꨩ", "ꨚꨟꨆꨴꨲꨩ", "ꨣꨳꨪꩌ", "ꨨꨈꨴꨮꩌ"
]

# Minimal pairs cho dấu Au ꨲ vs O ꨶ
MINIMAL_PAIRS_AU_O = [
    ("ꨀꨲꩆ", "ꨀꨶꩆ"),
    ("ꨓꨆꨴꨲꨩ", "ꨓꨆꨴꨶꨩ"),
    ("ꨚꨴꨲꨩ", "ꨚꨴꨶꨩ"),
    ("ꨆꨲꩆ", "ꨆꨶꩆ"),
    ("ꨟꨲꩆ", "ꨟꨶꩆ"),
    ("ꨣꨲꨩ", "ꨣꨶꨩ"),
    ("ꨚꨟꨆꨴꨲꨩ", "ꨚꨟꨆꨴꨶꨩ"),
    ("ꨚꨲ", "ꨚꨶ"),
    ("ꨆꨲ", "ꨆꨶ"),
    ("ꨀꨲ", "ꨀꨶ")
]

# Cụm dấu 3 tầng & tổ hợp nguyên âm phức tạp
COMPLEX_CLUSTERS = [
    "ꨣꨳꨪꩌ", "ꨓꨳꨪꩌ", "ꨚꨳꨪꩌ", "ꨕꨨꨵꩀ", "ꨝꨪꨗꨳꨪ", 
    "ꨨꨕꨳꨮꩇ", "ꨞꨯꨚꨓꨪꩍ", "ꨧꨪꩆꨝꨳꨪ", "ꨆꨵꨯꨱꩃ", "ꨚꨗꨴꨯꨱꩃ",
    "ꨗꨆꨓꨯꨮꨩ", "ꨚꨓꨯꨮꨩ", "ꨤꨯꨮꨩ", "ꨨꨝꨳꨯꨮꩆ", "ꨨꨈꨴꨮꩌ", "ꨣꨪꨈꨮꩌ"
]

# Mẫu song ngữ nội dòng (Cham phrase + Viet translation/commentary)
BILINGUAL_TEMPLATES = [
    ("ꨛꨯꨮ ꨆꨵꨯꨱꩃ ꨈꨣꩈ", "Vua Pô Klông Gia-rai"),
    ("ꨛꨯꨮ ꨚꩆ", "Pô Păt - Po Pat"),
    ("ꨚꨰꩀ ꨨꨤꨩ", "đắp bờ đập ngăn nước"),
    ("ꨕꨶꨩ ꨝꨯꨱꩍ ꨝꨰ", "dẫn nguồn nước vào ruộng"),
    ("ꨞꩇ ꨂꨣꩃ ꨌꩌ", "xứ sở người Chăm Panduranga"),
    ("yutꨢꨮ", "bờ đập nước sông lớn"),
    ("ꨁꨗꨩ ꨈꨭꨣꩈ", "Bà Thầy truyền dạy giáo lý"),
    ("ꨤꨪꨔꨬ ꨓꨝꨳꩀ", "cơm nắm gói trong lá chuối"),
    ("ꨝꨣꨭꨥ ꨟꩃ ꨣꨈꨵꨰ", "sau cuộc thi tài kết thúc"),
    ("ꨕꨨꨵꨀ ꨨꨕꨯꩌ", "nước chảy quanh năm trên đồng"),
    ("ꨎꨤꨭꩀ ꨟꨐꨭꩌ", "mở rộng hệ thống mương máng"),
    ("ꨣꨈꨵꨰ ꨚꨤꨬ ꨂꩀ", "thi tài đắp đập ngăn sông"),
    ("ꨓꩀ ꨕꨫ ꨚꨰꩀ ꨨꨤꨩ", "đến lúc đắp bờ đập nước"),
    ("ꨝꨰ ꨘꩆ ꨀꨦꨪꩅ", "ruộng lúa tốt tươi màu mỡ"),
    ("ꨣꨈꨵꨰ ꨧꨝꨭꩃ", "thi tài đắp đập đá lớn")
]

# ==============================================================================
# 2. Bộ Sinh Nội Dung Văn Bản (V25 Text Sampler)
# ==============================================================================
def sample_v25_text(pillar_type):
    """
    Sinh dòng văn bản theo từng trụ cột kỹ thuật:
    - pillar 1: Cham chuẩn (chuẩn bị cho Anti-Motion Blur)
    - pillar 2: Song ngữ nội dòng (Cham + Viet + Latin)
    - pillar 3: Số khổ thơ 1-99 & Dấu ngắt câu (꩑꩞..꩔꩓꩞, :, –, ?)
    - pillar 4: Cặp đối kháng Minimal Pairs (ꨲ vs ꨶ, ꩝꩝ variable gap, ꨯꨮꨩ)
    """
    if pillar_type == "bilingual":
        # Trụ cột 2: Song ngữ nội dòng
        pair = random.choice(BILINGUAL_TEMPLATES)
        style = random.choice(["bracket", "colon", "hyphen", "en_dash"])
        if style == "bracket":
            return f"{pair[0]} ({pair[1]})"
        elif style == "colon":
            return f"{pair[0]}: {pair[1]}"
        elif style == "en_dash":
            return f"{pair[0]} – {pair[1]}"
        else:
            return f"{pair[0]} - {pair[1]}"

    elif pillar_type == "stanza_punct":
        # Trụ cột 3: Số khổ 1-99 & Dấu ngắt câu
        num_val = random.randint(1, 99)
        cham_num_str = to_cham_num(num_val)
        sep = random.choice(["꩞ ", "꩝ "])
        words = random.sample(CHAM_VOCAB_POOL, random.randint(2, 4))
        content = " ".join(words)
        end_mark = random.choice(["꩞", "꩝꩝", "꩝", ":", "?", "!"])
        return f"{cham_num_str}{sep}{content}{end_mark}"

    elif pillar_type == "minimal_pairs":
        # Trụ cột 4: Cặp đối kháng
        sub = random.choice(["au_o", "double_danda", "complex_clusters"])
        if sub == "au_o":
            pair = random.choice(MINIMAL_PAIRS_AU_O)
            target = pair[0] if random.random() < 0.5 else pair[1]
            surrounding = random.sample(CHAM_VOCAB_POOL, random.randint(2, 3))
            return f"{target} " + " ".join(surrounding) + random.choice(["꩝", "꩝꩝", "꩞", ""])
        elif sub == "double_danda":
            words = random.sample(CHAM_VOCAB_POOL, random.randint(3, 5))
            return " ".join(words) + "꩝꩝"
        else:
            c = random.choice(COMPLEX_CLUSTERS)
            words = random.sample(CHAM_VOCAB_POOL, random.randint(2, 3))
            return f"{c} " + " ".join(words) + random.choice(["꩝", "꩞", ""])

    else:
        # Chuẩn (sẽ áp dụng Motion Blur nếu thuộc pillar anti_blur)
        words = random.sample(CHAM_VOCAB_POOL, random.randint(3, 6))
        text = " ".join(words)
        if random.random() < 0.5:
            text += random.choice(["꩝", "꩝꩝", "꩞", ""])
        return text

# ==============================================================================
# 3. Kỹ Thuật Render Ảnh & Tinh Chỉnh Nét Chữ
# ==============================================================================
def render_line_v25(text, is_bold=False, font_size=26, pad_left=12, pad_right=12, danda_gap=3):
    """
    Render dòng chữ Chăm / Song ngữ.
    Hỗ trợ biến thiên khoảng cách Double Danda '꩝꩝' từ 2px đến 8px.
    """
    font_path = FONT_CHAM_BOLD if is_bold else FONT_CHAM_REG
    font = ImageFont.truetype(font_path, font_size)

    # Nếu có ꩝꩝ ở cuối và cần biến thiên khoảng cách
    if text.endswith("꩝꩝") and danda_gap != 3:
        base_text = text[:-2]
        bbox_base = font.getbbox(base_text) if base_text else (0, 0, 0, font_size)
        w_base = (bbox_base[2] - bbox_base[0]) if base_text else 0
        
        bbox_danda = font.getbbox("꩝")
        w_danda = bbox_danda[2] - bbox_danda[0]
        h_line = max(bbox_base[3] - bbox_base[1], bbox_danda[3] - bbox_danda[1]) + 16
        w_total = pad_left + w_base + (6 if base_text else 0) + w_danda + danda_gap + w_danda + pad_right

        img = Image.new("RGB", (max(w_total, 60), max(h_line, 48)), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        cur_x = pad_left
        y = 8
        if base_text:
            draw.text((cur_x, y), base_text, fill=(20, 20, 25), font=font)
            cur_x += w_base + 6
        draw.text((cur_x, y), "꩝", fill=(20, 20, 25), font=font)
        cur_x += w_danda + danda_gap
        draw.text((cur_x, y), "꩝", fill=(20, 20, 25), font=font)
        return img

    # Render thông thường
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    w = tw + pad_left + pad_right
    h = max(th + 16, 48)

    img = Image.new("RGB", (max(w, 60), h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    x = pad_left - bbox[0]
    y = 8 - bbox[1]
    draw.text((x, y), text, fill=(20, 20, 25), font=font)
    return img

# ==============================================================================
# 4. Pipeline Biến Dạng Thực Địa (Augmentations) - Đặc Trị Motion Blur
# ==============================================================================
def apply_heavy_motion_blur(img_np, ksize=9, angle=45.0):
    """Tạo ma trận directional motion blur kernel mô phỏng rung tay máy ảnh di động."""
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    rad = math.radians(angle)
    dx = math.cos(rad)
    dy = math.sin(rad)
    cx, cy = ksize // 2, ksize // 2
    for r in range(-ksize // 2, ksize // 2 + 1):
        kx = int(round(cx + r * dx))
        ky = int(round(cy + r * dy))
        if 0 <= kx < ksize and 0 <= ky < ksize:
            kernel[ky, kx] = 1.0
    if kernel.sum() > 0:
        kernel /= kernel.sum()
    else:
        kernel[cx, cy] = 1.0
    return cv2.filter2D(img_np, -1, kernel)

def augment_image_v25(img_pil, is_motion_blur_pillar=False):
    """
    Áp dụng các phép biến thái thực tế:
    - Nếu is_motion_blur_pillar=True: Bắt buộc áp dụng directional motion blur (7x7 đến 13x13).
    - Thêm nhiễu giấy cổ, contrast jitter, downsampling ngẫu nhiên.
    """
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    if is_motion_blur_pillar:
        # Bắt buộc Motion Blur nặng
        ksize = random.choice([7, 9, 11, 13])
        angle = random.uniform(0.0, 180.0)
        img_bgr = apply_heavy_motion_blur(img_bgr, ksize=ksize, angle=angle)
        
        # Thêm Defocus hoặc Downsampling
        if random.random() < 0.4:
            scale = random.uniform(0.5, 0.75)
            small = cv2.resize(img_bgr, (max(int(w * scale), 20), max(int(h * scale), 16)), interpolation=cv2.INTER_LINEAR)
            img_bgr = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        # Biến dạng nhẹ tiêu chuẩn cho các gói khác
        if random.random() < 0.15:
            ksize = random.choice([3, 5])
            angle = random.uniform(0.0, 180.0)
            img_bgr = apply_heavy_motion_blur(img_bgr, ksize=ksize, angle=angle)
        if random.random() < 0.15:
            sigma = random.uniform(0.5, 1.2)
            img_bgr = cv2.GaussianBlur(img_bgr, (0, 0), sigma)

    # Thêm nhiễu nền giấy cổ nhẹ
    if random.random() < 0.6:
        noise = np.random.normal(0, random.uniform(1.0, 3.5), (h, w, 3)).astype(np.float32)
        img_bgr = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Thay đổi độ tương phản và độ sáng
    if random.random() < 0.5:
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-10, 10)
        img_bgr = np.clip(img_bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Đảm bảo resize về chuẩn height 48px cho V25
    target_h = 48
    scale = target_h / float(h)
    new_w = max(int(round(w * scale)), 32)
    img_resized = cv2.resize(img_bgr, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
    return img_resized

# ==============================================================================
# 5. Hàm Worker Đa Tiến Trình (Multiprocessing Task)
# ==============================================================================
def generate_sample_worker(args):
    """Worker tạo một mẫu ảnh và nhãn."""
    sample_idx, pillar_type, out_img_path = args

    # 1. Sinh văn bản
    text = sample_v25_text(pillar_type)

    # 2. Render ảnh
    is_bold = (random.random() < 0.35)
    pad_l = random.randint(2, 14)
    pad_r = random.randint(2, 14)
    danda_gap = random.randint(2, 8) if "꩝꩝" in text else 3
    
    img_pil = render_line_v25(text, is_bold=is_bold, pad_left=pad_l, pad_right=pad_r, danda_gap=danda_gap)

    # 3. Biến dạng (Augmentation)
    is_blur_pillar = (pillar_type == "anti_blur")
    img_final = augment_image_v25(img_pil, is_motion_blur_pillar=is_blur_pillar)

    # 4. Ghi file ảnh
    cv2.imwrite(out_img_path, img_final)

    # 5. Trả về tên file tương đối và nhãn
    rel_path = os.path.basename(os.path.dirname(out_img_path)) + "/" + os.path.basename(out_img_path)
    return f"{rel_path}\t{text}\n"

# ==============================================================================
# 6. Điều Phối Huấn Luyện Toàn Cục (Master Generator Loop)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate Cham OCR V25 Synthetic Dataset (250,000 samples)")
    parser.add_argument("--output_dir", type=str, default="./data/cham_synthetic_v25", help="Thư mục xuất dữ liệu")
    parser.add_argument("--num_train", type=int, default=250000, help="Số lượng mẫu huấn luyện (mặc định 250k)")
    parser.add_argument("--num_val", type=int, default=25000, help="Số lượng mẫu kiểm thử (mặc định 25k)")
    parser.add_argument("--workers", type=int, default=None, help="Số tiến trình CPU song song (mặc định cpu_count)")
    args = parser.parse_args()

    num_workers = args.workers or cpu_count()
    print("=" * 75)
    print("🚀 MASTER SYNTHETIC DATASET GENERATOR V25 (LIGHTNING AI OPTIMIZED)")
    print(f"   • Số CPU Workers song song : {num_workers} tiến trình")
    print(f"   • Quy mô mẫu Huấn luyện     : {args.num_train:,} ảnh")
    print(f"   • Quy mô mẫu Kiểm thử       : {args.num_val:,} ảnh")
    print(f"   • Thư mục đầu ra           : {args.output_dir}")
    print("=" * 75)

    train_img_dir = os.path.join(args.output_dir, "train_images")
    val_img_dir = os.path.join(args.output_dir, "val_images")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)

    # Phân bổ tỷ lệ các gói dữ liệu theo đúng TRAINING_ROADMAP_V25.md:
    # 1. Chuẩn & Cổ tích: 48% (120k / 250k)
    # 2. Anti-Motion Blur: 18% (45k / 250k)
    # 3. Song ngữ nội dòng: 16% (40k / 250k)
    # 4. Số khổ 1-99 & Dấu ngắt: 10% (25k / 250k)
    # 5. Cặp đối kháng Hard-Examples: 8% (20k / 250k)
    pillars = ["standard", "anti_blur", "bilingual", "stanza_punct", "minimal_pairs"]
    weights = [0.48, 0.18, 0.16, 0.10, 0.08]

    # Chuẩn bị danh sách tham số cho Train
    print(f"📦 Chuẩn bị danh mục tác vụ cho {args.num_train:,} mẫu Train...")
    train_tasks = []
    for i in range(args.num_train):
        p_type = random.choices(pillars, weights=weights)[0]
        img_name = f"train_{i+1:07d}.png"
        out_path = os.path.join(train_img_dir, img_name)
        train_tasks.append((i, p_type, out_path))

    # Chuẩn bị danh sách tham số cho Val
    print(f"📦 Chuẩn bị danh mục tác vụ cho {args.num_val:,} mẫu Val...")
    val_tasks = []
    for i in range(args.num_val):
        p_type = random.choices(pillars, weights=weights)[0]
        img_name = f"val_{i+1:06d}.png"
        out_path = os.path.join(val_img_dir, img_name)
        val_tasks.append((i, p_type, out_path))

    t_start = time.time()

    # 1. Sinh tập Train với đa tiến trình
    print(f"\n⚡ Đang sinh tập Train ({args.num_train:,} ảnh) trên {num_workers} CPU cores...")
    train_label_path = os.path.join(args.output_dir, "train_label.txt")
    with open(train_label_path, "w", encoding="utf-8") as f_train:
        with Pool(processes=num_workers) as pool:
            chunksize = 250
            completed = 0
            for label_line in pool.imap_unordered(generate_sample_worker, train_tasks, chunksize=chunksize):
                f_train.write(label_line)
                completed += 1
                if completed % 25000 == 0 or completed == args.num_train:
                    elapsed = time.time() - t_start
                    fps = completed / max(elapsed, 1e-5)
                    print(f"   [Train Progress] {completed:,} / {args.num_train:,} ảnh ({completed/args.num_train*100:.1f}%) - Tốc độ: {fps:.1f} ảnh/s")

    t_train_done = time.time()
    print(f"✅ Hoàn thành tập Train trong {t_train_done - t_start:.2f}s (~{(t_train_done - t_start)/60:.1f} phút)")

    # 2. Sinh tập Val với đa tiến trình
    print(f"\n⚡ Đang sinh tập Val ({args.num_val:,} ảnh) trên {num_workers} CPU cores...")
    val_label_path = os.path.join(args.output_dir, "val_label.txt")
    with open(val_label_path, "w", encoding="utf-8") as f_val:
        with Pool(processes=num_workers) as pool:
            chunksize = 250
            completed = 0
            for label_line in pool.imap_unordered(generate_sample_worker, val_tasks, chunksize=chunksize):
                f_val.write(label_line)
                completed += 1
                if completed % 5000 == 0 or completed == args.num_val:
                    print(f"   [Val Progress] {completed:,} / {args.num_val:,} ảnh ({completed/args.num_val*100:.1f}%)")

    total_time = time.time() - t_start
    print("=" * 75)
    print(f"🎉 HOÀN TẤT SINH TOÀN BỘ BỘ DỮ LIỆU V25 ({args.num_train + args.num_val:,} ẢNH)!")
    print(f"   • Tổng thời gian thực thi: {total_time:.2f}s (~{total_time/60:.2f} phút)")
    print(f"   • Tệp nhãn Train          : {train_label_path}")
    print(f"   • Tệp nhãn Val            : {val_label_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
