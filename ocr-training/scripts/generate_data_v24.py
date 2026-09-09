import os
import sys
import random
import time
import cv2
import numpy as np
import json
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from fontTools.ttLib import TTFont
import albumentations as A
from multiprocessing import Pool, cpu_count

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# 1. Ký Tự & Cấu Trúc Ngữ Liệu Tiếng Chăm V24
# ==============================================================================
CHAM_CONSONANTS = list("ꨆꨇꨈꨉꨊꨋꨌꨍꨎꨏꨐꨑꨒꨓꨔꨕꨖꨗꨘꨙꨚꨛꨜꨝꨞꨟꨠꨡꨢꨣꨤꨥꨦꨧꨨꨀꨁꨂꨃꨄꨅ")
CHAM_DIGITS_LIST = list("꩐꩑꩒꩓꩔꩕꩖꩗꩘꩙")
CHAM_PUNCT_LIST = list("꩜꩝꩞꩟")
LATIN_DIGITS = list("0123456789")
PUNCT_CHARS = list("[]().,;:/-")

# Dấu phụ đặc biệt
DIACRITIC_AU = 'ꨲ'  # U+AA32 (mới bổ sung vào từ điển V24)
DIACRITIC_O = 'ꨶ'   # U+AA36 (đối kháng với Au)

# Cụm từ khó & Mẫu khổ thơ thực nghiệm từ 57 khổ thơ cổ
CORE_STANZA_WORDS = [
    "ꨀꨆꨢꨯꨮꩅ", "ꨧꨪ", "ꨚꨗꩍ", "ꨓꨶꨮꩉ", "ꨓꨝꨳꩀ", "ꨚꨗꨶꨮꩄ", "ꨆꨭꨖꨩ", "ꨔꩅ",
    "ꨨꨤꨭꩆ", "ꨋꩇ", "ꨝꨩ", "ꨤꨪꨠꩍ", "ꨦꩀꨆꨣꨰ", "ꨝꨪꨗꨴꨪꩀ", "ꨕꨨꨵꩀ", "ꨥꩀ",
    "ꨎꩊ", "ꨈꨴꨭꩀ", "ꨆꨔꨯꨱꩅ", "ꨣꨪꨡꩍ", "ꨚꨴꨯꨱꩃ", "ꨤꨯꨩ", "ꨕꨫ", "ꨣꨳꨪꩌ",
    "ꨚꨆꩉ", "ꨆꨘꩊ", "ꨣꨩ", "ꨓꨨꨩ", "ꨙꨯꩌ", "ꨀꨇꨩꩆ", "ꨕꨤꩌ", "ꨕꨭꨤꨪꨆꩊ",
    "ꨧꨳꩌ", "ꨧꨭꨩ", "ꨏꩀ", "ꨕꨯꨱꩀ", "ꨨꨚꩀ", "ꨕꨯꨮꨝꨪꨓꨩ", "ꨀꨲꩆ", "ꨓꨘꩆ",
    "ꨌꩃ", "ꨆꨩ", "ꨔꨬ", "ꨤꨪꨆꨮꨭ", "ꨎꩃ", "ꨅꨩ", "ꨨꨭꨩ", "ꨟꩃ",
    "ꨕꨯꨮꩍ", "ꨧꨗꨪ", "ꨚꨮꩃ", "ꨙꨯꨱꩌ", "ꨚꨣꩆ", "ꨨꨕꨮꩉ", "ꨝꨪꨣꨭꨥ", "ꨠꨡꨯꩍ",
    "ꨛꨯꨮ", "ꨠꨣꨰ", "ꨀꨝꨪꩍ", "ꨕꨴꨬ", "ꨀꨕꨬ", "ꨧꨀꨰ", "ꨕꨣꨩ", "ꨀꨮꩆ",
    "ꨓꨝꨶꨮꩆ", "ꨚꨣꨪꩀ", "ꨚꨗꨴꩃ", "ꨆꨴꨯꨱꩃ", "ꨚꨎꨰ", "ꨟꨰ", "ꨨꨶꨮꩄ", "ꨆꨙꨩ",
    "ꨢꨶꨮꩆ", "ꨐꨭꨩ", "ꨉꩌ", "ꨌꩌ", "ꨂꨣꩃ", "ꨤꩄ", "ꨆꨳꨮꩃ", "ꨕꨯꨱꩃ",
    "ꨠꨢꩍ", "ꨨꨕꨬ", "ꨎꨳꨯꨮꩃ", "ꨎꨶꨰ", "ꨡꨶꩍ", "ꨆꩉ", "ꨆꨮꨭ", "ꨧꨭꨗꨪꩅ",
    "ꨈꨪꨗꨴꨮꩍ", "ꨤꨪꨝꨰꩍ", "ꨣꩈ", "ꨕꨨꨵꨮꨭ", "ꨨꨤꨬ", "ꨀꨗꩀ", "ꨕꩀ", "ꨓꨆꨴꨲꨩ",
    "ꨤꨪꨨꨪꩀ", "ꨚꨓꨯꨱ", "ꨓꨈꨯꩀ", "ꨧꨴꨪ", "ꨝꨪꨗꩈ", "ꨤꨯꨱꨥ", "ꨧꨯꨱꩃ", "ꨚꨣꨶꨬ",
    "ꨠꨓꨭꨥ", "ꨈꨮꩇ", "ꨓꨟꨩ", "ꨓꨮꩊ", "ꨘꨈꨮꩉ", "ꨚꨴꩃꨕꨣꩃ", "ꨆꨵꨯꨱꩃ", "ꨓꨴꨭꩆ",
    "ꨚꨆꨴꨮꩃ", "ꨚꨣꨯꩀ", "ꨌꨮꩀ", "ꨆꨤꩆ", "ꨚꨟꨆꨴꨲꨩ", "ꨣꨭꩇ", "ꨗꩌ", "ꨟꩀ",
    "ꨣꨯꩀ", "ꨨꨟꨭꨩ", "ꨝꨮꩀ", "ꨝꨪꨗꩀ", "ꨆꨶꨮꩄ", "ꨣꨪꨝꨯꨱꩃ", "ꨨꨟꨪꩅ", "ꨝꨶꨮꩊ",
    "ꨣꨪꨝꨭꨥ", "ꨓꨟꩆ", "ꨙꨪꩀ", "ꨀꨨꨯꨱꩀ", "ꨣꨰ", "ꨠꨧꨭꩍ", "ꨞꨯꨚꨓꨪꩍ", "ꨧꩀꨓꨎꨰ",
    "ꨚꨙꩉ", "ꨓꨤꨬ", "ꨈꨰ", "ꨌꩀ", "ꨚꨯꨱꩍ", "ꨌꨳꨪꩇ", "ꨀꨤꩍ", "ꨅꨩ",
    "ꨎꨰ", "ꨚꨓꨴꨰ", "ꨓꨚꩍ", "ꨧꨣꨪ", "ꨎꨝꩅ", "ꨧꨪꨤꩌ", "ꨥꨮꩀ", "ꨚꨤꨬ",
    "ꨠꨩ", "ꨗꨫ", "ꨧꨩ", "ꨆꨶꩍ", "ꨀꨆꨯꨱꩀ", "ꨘꩆ", "ꨤꨶꨪꩄ", "ꨚꨗꨴꨯꨱꩃ",
    "ꨎꨝꨶꨮꩊ", "ꨧꨪꩆꨝꨳꨪ", "ꨚꨯꩀ", "ꨨꨝꩉ", "ꨝꨪꩊꨔꨶꨮꩉ", "ꨤꨪꨟꩆ", "ꨗꨯꨱ",
    "ꨀꨓꨯꨱꩃ", "ꨎꨮꩀ", "ꨚꨟꨓꨰ", "ꨠꨖꨪꩉ", "ꨕꨣꩍ", "ꨙꨶꨮꩄ", "ꨕꨮꩇ", "ꨕꨗꨯꨱ",
    "ꨧꨭꨟꨭꨩ", "ꨆꨯꨮꩃ", "ꨀꨧꨰꩍ", "ꨇꨶꨰ", "ꨆꨆꨭꩍ", "ꨌꨳꨪꩇ", "ꨓꨚꩀ", "ꨇꨪꩆ",
    "ꨓꨴꨩ", "ꨝꨪꨙꨪ", "ꨝꨪꨗꨳꨪ", "ꨝꨳꩀ", "ꨣꨖꨪ", "ꨖꨶꨮꩆ", "ꨅꩍ", "ꨚꨤꨰ",
    "ꨕꨶꩍ", "ꨆꨭꨟꨬ", "ꨝꨪꨗꨰ", "ꨗꨯꨣꨚꩅ", "ꨓꨯꩀ", "ꨧꩍ", "ꨖꨮꩉ", "ꨜꨶꨮꩊ",
    "ꨚꨵꨮꩀ", "ꨤꨪꨆꨭꩀ", "ꨚꨕꨮꩍ", "ꨚꨕꨬ", "ꨣꨤꨯꨩ", "ꨥꩃ", "ꨝꨪ", "ꨞꨯꨕꨮꩉꨨꨩ",
    "ꨆꨚꩊ", "ꨢꩅꨓꨴꩀ", "ꨈꨪꨗꨶꨮꩉ", "ꨘꨨꨶꨮꩉ", "ꨃꨥ", "ꨉꩀ", "ꨝꨯꨱꩃ", "ꨔꨶꩀ",
    "ꨝꩅ", "ꨚꨤꨪꨕꨯꨱ", "ꨆꨣꨳꩀ", "ꨧꨪꨝꩉ", "ꨒꨮꩉ", "ꨍꨰꩅ", "ꨓꨥꩀ", "ꨀꨤꨩ",
    "ꨠꨕꨬ", "ꨌꨰꩀ", "ꨠꨘꩃ", "ꨝꨪꩊꨘꨔꨶꨮꩉ", "ꨠꨝꨰ", "ꨢꨮꨭ", "ꨢꨶꩀ", "ꨝꨵꨯꨱꩍ",
    "ꨆꨆꨬ", "ꨚꨈꨶꨮꩆ", "ꨆꨵꨮꨭ", "ꨨꨣꨬ", "ꨟꨮꨣꨰ", "ꨡꨯꩍ", "ꨕꨯꨱꨍ"
]

# Các mẫu đối kháng và dấu 3 tầng
DIACRITIC_PAIRS = [
    ("ꨀꨲꩆ", "ꨀꨶꩆ"),
    ("ꨓꨆꨴꨲꨩ", "ꨓꨆꨴꨶꨩ"),
    ("ꨚꨴꨲꨩ", "ꨚꨴꨶꨩ"),
    ("ꨆꨲꩆ", "ꨆꨶꩆ"),
    ("ꨟꨲꩆ", "ꨟꨶꩆ"),
    ("ꨣꨲꨩ", "ꨣꨶꨩ"),
    ("ꨚꨟꨆꨴꨲꨩ", "ꨚꨟꨆꨴꨶꨩ")
]

THREE_TIER_CLUSTERS = [
    "ꨣꨳꨪꩌ", "ꨓꨳꨪꩌ", "ꨚꨳꨪꩌ", "ꨕꨨꨵꩀ", "ꨝꨪꨗꨳꨪ", 
    "ꨨꨕꨳꨮꩇ", "ꨞꨯꨚꨓꨪꩍ", "ꨧꨪꩆꨝꨳꨪ", "ꨆꨵꨯꨱꩃ", "ꨚꨗꨴꨯꨱꩃ"
]

PRE_VOWEL_PATTERNS = [
    "ꨗꨯꨣꨚꨮꩅ", "ꨆꨴꨯꩅ", "ꨛꨯꨮ", "ꨚꨓꨯꨱ", "ꨈꨪꨗꨯꨱꩃ", 
    "ꨕꨯꨮꩍ", "ꨠꨡꨯꩍ", "ꨕꨯꨱꩃ", "ꨞꨯꨚꨓꨪꩍ", "ꨚꨓꨯꨱ"
]

def to_cham_num(n):
    """Chuyển số nguyên n thành chuỗi chữ số Chăm"""
    return "".join(CHAM_DIGITS_LIST[int(d)] for d in str(n))

# ==============================================================================
# 2. Font Validator & Bộ Sinh Mẫu Văn Bản V24
# ==============================================================================
class FontValidator:
    def __init__(self, font_paths):
        self.font_cmaps = {}
        for path in font_paths:
            try:
                tt = TTFont(path)
                cmap = set()
                for table in tt['cmap'].tables:
                    cmap.update(table.cmap.keys())
                self.font_cmaps[path] = cmap
            except Exception as e:
                self.font_cmaps[path] = set()

    def supports_string(self, font_path, text):
        supported_set = self.font_cmaps.get(font_path, set())
        for char in text:
            cp = ord(char)
            if char in " \t\n\r" or cp < 128:
                continue
            if cp not in supported_set:
                return False
        return True

def sample_text_v24():
    """
    Sinh dòng văn bản tiếng Chăm V24 tích hợp 5 nhóm kịch bản:
    1. Stanza numbers 1-99 (꩑꩞ - ꩙꩙꩞)
    2. Double Danda ꩝꩝ kết thúc câu
    3. Cặp dấu đối kháng ꨲ vs ꨶ
    4. Cụm dấu 3 tầng và nguyên âm trước ꨯ, ꨯꨱ
    5. Văn bản thông thường kết hợp số tham chiếu Latin
    """
    cat = random.choices(
        ['stanza', 'double_danda', 'diacritics', 'three_tier', 'normal', 'latin_mixed'],
        weights=[0.25, 0.15, 0.15, 0.15, 0.20, 0.10]
    )[0]
    
    if cat == 'stanza':
        # Số thứ tự khổ thơ Chăm từ 1 đến 99
        num_val = random.randint(1, 99)
        cham_num_str = to_cham_num(num_val)
        sep = random.choice(['꩞ ', '꩝ '])
        # Lấy 2-4 từ nội dung
        content = " ".join(random.sample(CORE_STANZA_WORDS, random.randint(2, 4)))
        end_punct = random.choice(['꩝', '꩝꩝', ''])
        return f"{cham_num_str}{sep}{content}{end_punct}"
        
    elif cat == 'double_danda':
        # Câu kết thúc với Double Danda
        content = " ".join(random.sample(CORE_STANZA_WORDS, random.randint(2, 5)))
        return f"{content}꩝꩝"
        
    elif cat == 'diacritics':
        # Hard examples chứa cặp ꨲ vs ꨶ
        pair = random.choice(DIACRITIC_PAIRS)
        word = pair[0] if random.random() < 0.6 else pair[1]
        surrounding = random.sample(CORE_STANZA_WORDS, random.randint(1, 3))
        return f"{word} " + " ".join(surrounding) + random.choice(['꩝', '꩝꩝', ''])
        
    elif cat == 'three_tier':
        # Cụm 3 tầng hoặc nguyên âm trước
        tier_w = random.choice(THREE_TIER_CLUSTERS + PRE_VOWEL_PATTERNS)
        surrounding = random.sample(CORE_STANZA_WORDS, random.randint(2, 3))
        return f"{tier_w} " + " ".join(surrounding) + random.choice(['꩝', '꩝꩝', ''])
        
    elif cat == 'latin_mixed':
        # Kết hợp số Latin hoặc dấu ngoặc tham chiếu
        w1 = random.choice(CORE_STANZA_WORDS)
        w2 = random.choice(CORE_STANZA_WORDS)
        num = random.randint(1, 999)
        style = random.choice([f"({num})", f"[{num}]", f"{num}", f": {num}"])
        return f"{w1} {w2} {style}"
        
    else: # normal
        n_words = random.randint(2, 5)
        text = " ".join(random.sample(CORE_STANZA_WORDS, n_words))
        if random.random() < 0.5:
            text += random.choice(['꩝', '꩝꩝', ''])
        return text

# ==============================================================================
# 3. Render Dòng Chữ Kèm Tinh Chỉnh Khoảng Cách Double Danda
# ==============================================================================
def render_text_v24(text, font_path, font_size=24, pad_w=15, pad_h=8, bg_color=(255, 255, 255), fg_color=(0, 0, 0)):
    font = ImageFont.truetype(font_path, font_size)
    dummy_img = Image.new('RGB', (100, 100))
    dummy_draw = ImageDraw.Draw(dummy_img)
    
    # Xử lý biến thiên khoảng cách Double Danda nếu có ꩝꩝
    if '꩝꩝' in text:
        # Ngẫu nhiên tách 2 nét gạch từ 1px đến 7px
        gap_px = random.randint(1, 7)
        # Thay thế tạm bằng ký tự giả định hoặc render 2 pass
        # Đơn giản nhất: chèn khoảng trống zero-width hoặc thin space
        if gap_px > 3:
            text = text.replace('꩝꩝', '꩝ ꩝' if gap_px > 5 else '꩝\u200a꩝')
            
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = max(10, bbox[2] - bbox[0])
    text_h = max(10, bbox[3] - bbox[1])
    
    img_w = text_w + pad_w * 2
    img_h = max(48, text_h + pad_h * 2)
    img = Image.new('RGB', (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((pad_w - bbox[0], pad_h - bbox[1]), text, fill=fg_color, font=font)
    return img

# ==============================================================================
# 4. Pipeline Data Augmentation Toàn Diện V24 (Motion Blur, Noise, Paper, Skew)
# ==============================================================================
def apply_augmentations_v24(pil_img, is_train=True):
    arr = np.array(pil_img)
    h, w = arr.shape[:2]
    
    if not is_train:
        # Val set: chỉ áp dụng nhiễu nhẹ
        if random.random() < 0.2:
            arr = np.clip(arr.astype(np.float32) + np.random.normal(0, 3, arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
        
    # --- 1. Giả lập màu nền giấy cổ / ố vàng / tương phản thấp ---
    r_bg = random.random()
    if r_bg < 0.25:
        # Giấy ố vàng (parchment)
        tint = np.array([random.randint(220, 245), random.randint(205, 230), random.randint(170, 200)], dtype=np.float32)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray = np.expand_dims(gray, -1)
        arr = np.clip(gray * tint + (1 - gray) * np.array([30, 20, 15]), 0, 255).astype(np.uint8)
    elif r_bg < 0.40:
        # Tương phản thấp (mực mờ)
        arr = cv2.convertScaleAbs(arr, alpha=random.uniform(0.6, 0.85), beta=random.randint(30, 60))
        
    # --- 2. Vết loang mực / ố nước rải rác ---
    if random.random() < 0.20:
        for _ in range(random.randint(1, 3)):
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            rad = random.randint(6, 25)
            color = (random.randint(180, 220), random.randint(170, 210), random.randint(150, 190))
            cv2.circle(arr, (cx, cy), rad, color, -1)
            
    # --- 3. Làm mờ (Motion Blur & Defocus Gaussian) ---
    r_blur = random.random()
    if r_blur < 0.25:
        # Motion blur (mô phỏng rung tay camera)
        k_size = random.choice([3, 5, 7])
        angle = random.uniform(0, 180)
        M = cv2.getRotationMatrix2D((k_size / 2, k_size / 2), angle, 1)
        kernel = np.zeros((k_size, k_size))
        kernel[int((k_size - 1) / 2), :] = np.ones(k_size)
        kernel = cv2.warpAffine(kernel, M, (k_size, k_size))
        kernel = kernel / (np.sum(kernel) + 1e-5)
        arr = cv2.filter2D(arr, -1, kernel)
    elif r_blur < 0.40:
        # Gaussian blur nhẹ
        arr = cv2.GaussianBlur(arr, (3, 3), random.uniform(0.5, 1.2))
        
    # --- 4. Nhiễu hạt cảm biến & Bụi giấy (Noise & Dust) ---
    r_noise = random.random()
    if r_noise < 0.25:
        # Gaussian noise
        sigma = random.uniform(8, 22)
        arr = np.clip(arr.astype(np.float32) + np.random.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
    elif r_noise < 0.35:
        # Salt & pepper (bụi đen rải rác)
        num_dust = random.randint(5, 20)
        for _ in range(num_dust):
            dx = random.randint(0, w - 1)
            dy = random.randint(0, h - 1)
            arr[dy, dx] = random.choice([0, 255])
            
    # --- 5. Đứt nét / Mòn nét (Morphological Erosion/Dilation) ---
    if random.random() < 0.25:
        k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        if random.random() < 0.5:
            arr = cv2.erode(arr, k_morph, iterations=1)
        else:
            arr = cv2.dilate(arr, k_morph, iterations=1)
            
    # --- 6. Xoay góc nghiêng & Biến đổi phối cảnh (Albumentations) ---
    transform = A.Compose([
        A.Affine(scale=(0.95, 1.05), rotate=(-5, 5), translate_percent=(-0.03, 0.03), p=0.4),
        A.Perspective(scale=(0.02, 0.04), keep_size=True, p=0.3)
    ])
    arr = transform(image=arr)['image']
    
    # --- 7. Biến đổi kích thước nhỏ (Downscaled line height) ---
    if random.random() < 0.20:
        target_h = random.randint(24, 34)
        target_w = int(w * (target_h / h))
        if target_w > 10:
            small = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA)
            arr = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            
    return Image.fromarray(arr)

# ==============================================================================
# 5. Worker Function Đa Tiến Trình (Multi-processing Worker)
# ==============================================================================
def generate_sample_worker(args):
    idx, output_dir, font_paths, is_val = args
    text = sample_text_v24()
    font_path = random.choice(font_paths)
    font_size = random.randint(22, 28)
    
    raw_img = render_text_v24(text, font_path, font_size=font_size)
    aug_img = apply_augmentations_v24(raw_img, is_train=(not is_val))
    
    # Resize chuẩn hóa chiều cao 48px, giữ nguyên tỷ lệ
    w_orig, h_orig = aug_img.size
    h_new = 48
    w_new = max(32, int(w_orig * (h_new / h_orig)))
    final_img = aug_img.resize((w_new, h_new), Image.Resampling.BICUBIC)
    
    sub_dir = "val" if is_val else "train"
    img_filename = f"synth_{idx:07d}.png"
    img_rel_path = f"{sub_dir}/{img_filename}"
    full_save_path = os.path.join(output_dir, sub_dir, img_filename)
    
    final_img.save(full_save_path)
    
    # Nhãn lưu ở chuẩn Unicode Logical Order
    # Lưu ý thay thế thin space \u200a nếu có
    clean_label = text.replace('\u200a', '')
    label_line = f"{img_rel_path}\t{clean_label}\n"
    return (is_val, label_line)

# ==============================================================================
# 6. Chương Trình Chính (Main Runner)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Sinh dữ liệu tổng hợp tiếng Chăm V24")
    parser.add_argument('--output-dir', type=str, default='ocr-training/data/cham_synthetic_v24', help='Thư mục xuất dữ liệu')
    parser.add_argument('--fonts-dir', type=str, default='ocr-training/data/fonts', help='Thư mục chứa font NotoSansCham')
    parser.add_argument('--num-train', type=int, default=150000, help='Số lượng mẫu train')
    parser.add_argument('--num-val', type=int, default=15000, help='Số lượng mẫu val')
    parser.add_argument('--num-workers', type=int, default=4, help='Số tiến trình song song')
    parser.add_argument('--test-run', action='store_true', help='Chạy thử nghiệm nhanh 100 mẫu')
    args = parser.parse_args()
    
    if args.test_run:
        args.num_train = 90
        args.num_val = 10
        print("⚡ Chế độ Test Run kích hoạt: 90 mẫu train, 10 mẫu val.")
        
    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'val'), exist_ok=True)
    
    font_paths = [os.path.join(args.fonts_dir, f) for f in os.listdir(args.fonts_dir) if f.endswith(('.ttf', '.otf'))]
    if not font_paths:
        print(f"❌ LỖI: Không tìm thấy font trong {args.fonts_dir}")
        sys.exit(1)
        
    validator = FontValidator(font_paths)
    valid_fonts = [fp for fp in font_paths if len(validator.font_cmaps.get(fp, set())) > 0]
    print(f"📚 Đã nạp {len(valid_fonts)} font hợp lệ: {[os.path.basename(f) for f in valid_fonts]}")
    
    total_samples = args.num_train + args.num_val
    print(f"🚀 Bắt đầu sinh {total_samples:,} mẫu ({args.num_train:,} train, {args.num_val:,} val)...")
    print(f"⚙️  Số vCPU tiến trình song song: {args.num_workers}")
    
    tasks = []
    for i in range(total_samples):
        is_val = (i >= args.num_train)
        tasks.append((i, args.output_dir, valid_fonts, is_val))
        
    t0 = time.time()
    train_labels = []
    val_labels = []
    
    # Chạy Pool đa tiến trình
    chunksize = max(1, len(tasks) // (args.num_workers * 20))
    with Pool(processes=args.num_workers) as pool:
        for idx, (is_val, label_line) in enumerate(pool.imap_unordered(generate_sample_worker, tasks, chunksize=chunksize)):
            if is_val:
                val_labels.append(label_line)
            else:
                train_labels.append(label_line)
                
            if (idx + 1) % 10000 == 0 or (idx + 1) == total_samples:
                elapsed = time.time() - t0
                speed = (idx + 1) / elapsed
                remaining = (total_samples - (idx + 1)) / max(1e-5, speed)
                print(f"   [{idx+1:,}/{total_samples:,}] {((idx+1)/total_samples*100):.1f}% | Tốc độ: {speed:.1f} mẫu/s | Còn lại: ~{remaining/60:.1f} phút")
                
    # Ghi file nhãn
    train_label_path = os.path.join(args.output_dir, 'train_label.txt')
    val_label_path = os.path.join(args.output_dir, 'val_label.txt')
    
    with open(train_label_path, 'w', encoding='utf-8') as f:
        f.writelines(train_labels)
    with open(val_label_path, 'w', encoding='utf-8') as f:
        f.writelines(val_labels)
        
    t_total = time.time() - t0
    print(f"🎉 Hoàn tất sinh dữ liệu V24 trong {t_total:.1f} giây ({t_total/60:.2f} phút)!")
    print(f"   - File train: {train_label_path} ({len(train_labels):,} dòng)")
    print(f"   - File val:   {val_label_path} ({len(val_labels):,} dòng)")

if __name__ == '__main__':
    main()
