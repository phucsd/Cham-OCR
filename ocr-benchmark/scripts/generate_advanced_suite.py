#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Benchmark Suite Generator for Cham OCR (90 Test Cases across 3 Directions).
Direction A (adv_001 - adv_030): Đa ngữ xen kẽ / Code-Switching (Cham + Vietnamese + Latin)
Direction B (adv_031 - adv_060): Bố cục 2 Cột A4 / Multi-Column Reading Order
Direction C (adv_061 - adv_090): Ảnh chụp điện thoại thực địa / Mobile Field Degradations
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

# Set seed for reproducibility
random.seed(2026)
np.random.seed(2026)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)

DATA_DIR = os.path.join(BENCHMARK_DIR, "data")
OUTPUT_IMG_DIR = os.path.join(DATA_DIR, "images_advanced")
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
GT_OUTPUT_PATH = os.path.join(DATA_DIR, "benchmark_advanced_gt.json")
CORPUS_PATH = os.path.join(DATA_DIR, "poklong_corpus.txt")

# Fonts
FONT_CHAM_REG = os.path.join(PROJECT_ROOT, "ocr-training", "data", "fonts", "NotoSansCham-Regular.ttf")
FONT_CHAM_BOLD = os.path.join(PROJECT_ROOT, "ocr-training", "data", "fonts", "NotoSansCham-Bold.ttf")
FONT_VIET_ARIAL = r"C:\Windows\Fonts\arial.ttf"
FONT_VIET_TIMES = r"C:\Windows\Fonts\times.ttf"

if not os.path.exists(FONT_CHAM_REG) or not os.path.exists(FONT_CHAM_BOLD):
    raise FileNotFoundError("Cham font files not found!")

_FONT_CACHE = {}

def get_font(path, size):
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]

# Load Cham text corpus
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    RAW_CHAM_TEXT = f.read().strip()

CHAM_WORDS = RAW_CHAM_TEXT.split()

def get_cham_line_pool():
    lines = []
    curr = []
    for w in CHAM_WORDS:
        curr.append(w)
        if len(curr) >= random.randint(4, 7) or w.endswith('꩝') or w.endswith('꩞'):
            lines.append(' '.join(curr))
            curr = []
    if curr:
        lines.append(' '.join(curr))
    return lines

CHAM_LINE_POOL = get_cham_line_pool()

VIET_SENTENCES = [
    "Truyền thuyết về Vua Pô Klông Gia-rai và Pô Păt cùng nhau thi tài đắp đập ngăn dòng nước.",
    "Pô Klông Gia-rai dùng trí tuệ và sự mẫn tiệp để dẫn nước vào các con mương phục vụ đồng ruộng.",
    "Hai vị thủ lĩnh cùng thề nguyện đắp bờ đập dẫn thủy nhập điền cứu giúp muôn dân Chăm-pa.",
    "Bà Thầy phán truyền lời khuyên thông thái giúp cho công cuộc trị thủy được công bằng và thuận hòa.",
    "Bờ đập ngăn dòng chảy cuồn cuộn giữ cho dòng nước tưới mát những cánh đồng lúa bạt ngàn.",
    "Người dân khắp nơi cùng nhau gánh đất đá đắp nên công trình đập nước kỳ vĩ lưu truyền hậu thế.",
    "Gói cơm nắm trong lá chuối xanh được chia đều cho những người thợ làm việc không ngừng nghỉ.",
    "Dòng sông Krông Lang cuộn chảy hiền hòa đem phù sa bồi đắp cho từng thửa ruộng ven sông.",
    "Vua Pô Klông Gia-rai đã hoàn thành bờ đập trước trong niềm hân hoan rạng rỡ của muôn làng.",
    "Dấu ấn công trình thủy lợi cổ kính vẫn trường tồn cùng năm tháng trên mảnh đất Ninh Thuận.",
    "Nguồn nước ngọt ngào mát lành chảy qua mương dẫn làm xanh tươi khắp các cánh đồng xứ Chăm.",
    "Pô Păt khâm phục đức độ và tài năng phi thường của Vua Pô Klông Gia-rai sau cuộc thi tài.",
    "Các lễ hội tạ ơn thần linh và tổ tiên được tổ chức trang trọng bên bờ đập thiêng liêng.",
    "Tục lệ phân chia nguồn nước tưới tiêu công bằng được ghi khắc trong luật tục truyền đời.",
    "Đập nước Yut vững chãi đứng sừng sững qua bao mùa mưa lũ che chở cho xóm làng bình yên.",
    "Hệ thống đập mương Chăm cổ là di sản kỹ thuật thủy lợi vô cùng độc đáo của Đông Nam Á.",
    "Dân làng hát vang khúc ca mừng mùa lúa trĩu hạt trên những thửa ruộng bậc thang màu mỡ.",
    "Hình ảnh Vua Pô Klông Gia-rai gắn liền với biểu tượng của lòng vị tha và trí tuệ vô biên.",
    "Những viên đá xếp tầng trên thân đập vẫn kiên cường thách thức thời gian qua nhiều thế kỷ.",
    "Nguyện cầu cho mưa thuận gió hòa, mùa màng tươi tốt trên khắp các buôn làng quê hương."
]

INLINE_MIXED_PAIRS = [
    ("ꨛꨯꨮ ꨆꨵꨯꨱꩃ ꨈꨣꩈ (Vua Pô Klông Gia-rai)", "Vua Chăm trị thủy thông thái"),
    ("ꨛꨯꨮ ꨚꩆ (Pô Păt - Po Pat)", "Người thi tài đắp đập ngăn sông"),
    ("ꨚꨰꩀ ꨨꨤꨩ (đắp bờ đập ngăn nước)", "Công trình dẫn thủy nhập điền"),
    ("ꨕꨶꨩ ꨝꨯꨱꩍ ꨝꨰ (dẫn nguồn nước vào ruộng)", "Hệ thống mương mán tưới tiêu"),
    ("ꨞꩇ ꨂꨣꩃ ꨌꩌ (xứ sở của người Chăm)", "Vùng đất thiêng Panduranga cổ kính"),
    ("yutꨢꨮ (bờ đập nước sông lớn)", "Đập nước cổ truyền chặn lũ"),
    ("ꨁꨗꨩ ꨈꨭꨣꩈ (Bà Thầy truyền dạy giáo lý)", "Bậc cao nhân chỉ dẫn công bằng"),
    ("ꨤꨪꨔꨬ ꨓꨝꨳꩀ (cơm nắm gói lá chuối)", "Lương thảo tiếp sức cho dân làng"),
    ("ꨝꨣꨭꨥ ꨟꩃ ꨣꨈꨵꨰ (sau cuộc thi tài kết thúc)", "Phân định thắng bại vinh quang"),
    ("ꨕꨨꨵꨀ ꨨꨕꨯꩌ (nước chảy quanh năm)", "Đất đai màu mỡ mùa màng bội thu"),
    ("ꨎꨤꨭꩀ ꨟꨐꨭꩌ (mở rộng mương máng)", "Dòng nước trong lành về buôn làng"),
    ("ꨣꨈꨵꨰ ꨚꨤꨬ ꨂꩀ (thi tài đắp đập)", "Cuộc tranh tài lịch sử của hai vị")
]

def make_paper_texture(w, h, texture_level=1):
    """Generates realistic paper texture."""
    if texture_level == 1:
        # Clean white / light beige
        val_r = random.randint(248, 255)
        val_g = random.randint(248, 255)
        val_b = random.randint(245, 252)
        bg = np.full((h, w, 3), (val_b, val_g, val_r), dtype=np.uint8)
        noise = np.random.normal(0, 1.5, (h, w)).astype(np.float32)
        for c in range(3):
            bg[:, :, c] = np.clip(bg[:, :, c].astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return bg
    elif texture_level == 2:
        # Realistic vintage / parchment
        val_r = random.randint(238, 248)
        val_g = random.randint(228, 238)
        val_b = random.randint(205, 220)
        bg = np.full((h, w, 3), (val_b, val_g, val_r), dtype=np.uint8)
        gh, gw = max(4, h // 40), max(4, w // 40)
        low_noise = np.random.normal(0, 6.0, (gh, gw)).astype(np.float32)
        low_noise = cv2.resize(low_noise, (w, h), interpolation=cv2.INTER_CUBIC)
        noise = low_noise + np.random.normal(0, 2.5, (h, w)).astype(np.float32)
        for c in range(3):
            bg[:, :, c] = np.clip(bg[:, :, c].astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return bg
    else:
        # Aged paper with slight vignette
        val_r = random.randint(228, 242)
        val_g = random.randint(215, 228)
        val_b = random.randint(185, 205)
        bg = np.full((h, w, 3), (val_b, val_g, val_r), dtype=np.uint8)
        gh, gw = max(4, h // 30), max(4, w // 30)
        low_noise = np.random.normal(0, 8.0, (gh, gw)).astype(np.float32)
        low_noise = cv2.resize(low_noise, (w, h), interpolation=cv2.INTER_CUBIC)
        noise = low_noise + np.random.normal(0, 3.5, (h, w)).astype(np.float32)
        for c in range(3):
            bg[:, :, c] = np.clip(bg[:, :, c].astype(np.float32) + noise, 0, 255).astype(np.uint8)
        X = np.linspace(-1, 1, w)
        Y = np.linspace(-1, 1, h)
        mx, my = np.meshgrid(X, Y)
        dist = np.sqrt(mx**2 + my**2) / 1.414
        vignette = np.clip(1.0 - 0.20 * (dist ** 2), 0.75, 1.0)
        for c in range(3):
            bg[:, :, c] = (bg[:, :, c].astype(np.float32) * vignette).astype(np.uint8)
        return bg

# =========================================================================
# GENERATOR DIRECTION A: Bilingual & Code-Switching (adv_001 - adv_030)
# =========================================================================
def generate_direction_a(sample_idx):
    """
    Direction A: Đa ngữ xen kẽ
    - adv_001..adv_010: Paragraph-level code-switching (Cham para -> Viet para -> Cham para)
    - adv_011..adv_020: Line-level alternating code-switching (Cham line -> Viet line)
    - adv_021..adv_030: Intra-line mixed code-switching (Cham phrase + Viet annotation)
    """
    sub_id = sample_idx  # 1 to 30
    w = 1000
    h = 750
    bg = make_paper_texture(w, h, texture_level=random.choice([1, 2]))
    img_pil = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font_size = 24
    font_cham = get_font(FONT_CHAM_REG, font_size)
    font_cham_bold = get_font(FONT_CHAM_BOLD, font_size)
    font_viet = get_font(FONT_VIET_ARIAL, font_size)

    lines_gt = []

    if sub_id <= 10:
        subtype = "paragraph_switch"
        # 3 paragraphs: Cham -> Viet -> Cham
        y = 50
        line_id = 0
        
        # Para 1: Cham (3 lines)
        cham_start = (sub_id * 7) % (len(CHAM_LINE_POOL) - 10)
        for i in range(3):
            text = CHAM_LINE_POOL[cham_start + i]
            bbox = font_cham.getbbox(text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = 60
            draw.text((x, y), text, fill=(20, 20, 25), font=font_cham)
            lines_gt.append({
                "line_id": line_id,
                "text": text,
                "script": "cham",
                "box": [int(y + bbox[1]), int(x + bbox[0]), int(y + bbox[3]), int(x + bbox[2])],
                "polygon": [[int(x + bbox[0]), int(y + bbox[1])], [int(x + bbox[2]), int(y + bbox[1])],
                            [int(x + bbox[2]), int(y + bbox[3])], [int(x + bbox[0]), int(y + bbox[3])]]
            })
            y += th + 18
            line_id += 1
            
        y += 25  # Paragraph gap
        
        # Para 2: Vietnamese (3 lines)
        viet_start = (sub_id * 3) % (len(VIET_SENTENCES) - 5)
        for i in range(3):
            text = VIET_SENTENCES[viet_start + i]
            bbox = font_viet.getbbox(text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = 60
            draw.text((x, y), text, fill=(25, 25, 30), font=font_viet)
            lines_gt.append({
                "line_id": line_id,
                "text": text,
                "script": "vietnamese",
                "box": [int(y + bbox[1]), int(x + bbox[0]), int(y + bbox[3]), int(x + bbox[2])],
                "polygon": [[int(x + bbox[0]), int(y + bbox[1])], [int(x + bbox[2]), int(y + bbox[1])],
                            [int(x + bbox[2]), int(y + bbox[3])], [int(x + bbox[0]), int(y + bbox[3])]]
            })
            y += th + 18
            line_id += 1
            
        y += 25  # Paragraph gap
        
        # Para 3: Cham (2 lines)
        for i in range(2):
            text = CHAM_LINE_POOL[cham_start + 3 + i]
            bbox = font_cham.getbbox(text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = 60
            draw.text((x, y), text, fill=(20, 20, 25), font=font_cham)
            lines_gt.append({
                "line_id": line_id,
                "text": text,
                "script": "cham",
                "box": [int(y + bbox[1]), int(x + bbox[0]), int(y + bbox[3]), int(x + bbox[2])],
                "polygon": [[int(x + bbox[0]), int(y + bbox[1])], [int(x + bbox[2]), int(y + bbox[1])],
                            [int(x + bbox[2]), int(y + bbox[3])], [int(x + bbox[0]), int(y + bbox[3])]]
            })
            y += th + 18
            line_id += 1

    elif sub_id <= 20:
        subtype = "line_alternating"
        # Alternating lines: Cham -> Viet -> Cham -> Viet... (total 8 lines)
        y = 55
        line_id = 0
        cham_start = (sub_id * 5) % (len(CHAM_LINE_POOL) - 10)
        viet_start = (sub_id * 4) % (len(VIET_SENTENCES) - 6)
        
        for i in range(4):
            # Cham line
            ch_text = CHAM_LINE_POOL[cham_start + i]
            bbox = font_cham.getbbox(ch_text)
            th = bbox[3] - bbox[1]
            x = 60
            draw.text((x, y), ch_text, fill=(20, 20, 25), font=font_cham)
            lines_gt.append({
                "line_id": line_id,
                "text": ch_text,
                "script": "cham",
                "box": [int(y + bbox[1]), int(x + bbox[0]), int(y + bbox[3]), int(x + bbox[2])],
                "polygon": [[int(x + bbox[0]), int(y + bbox[1])], [int(x + bbox[2]), int(y + bbox[1])],
                            [int(x + bbox[2]), int(y + bbox[3])], [int(x + bbox[0]), int(y + bbox[3])]]
            })
            y += th + 14
            line_id += 1
            
            # Viet line (in italic/smaller or slightly indented)
            vi_text = VIET_SENTENCES[viet_start + i]
            bbox_v = font_viet.getbbox(vi_text)
            th_v = bbox_v[3] - bbox_v[1]
            xv = 80
            draw.text((xv, y), vi_text, fill=(50, 50, 60), font=font_viet)
            lines_gt.append({
                "line_id": line_id,
                "text": vi_text,
                "script": "vietnamese",
                "box": [int(y + bbox_v[1]), int(xv + bbox_v[0]), int(y + bbox_v[3]), int(xv + bbox_v[2])],
                "polygon": [[int(xv + bbox_v[0]), int(y + bbox_v[1])], [int(xv + bbox_v[2]), int(y + bbox_v[1])],
                            [int(xv + bbox_v[2]), int(y + bbox_v[3])], [int(xv + bbox_v[0]), int(y + bbox_v[3])]]
            })
            y += th_v + 22
            line_id += 1

    else:
        subtype = "inline_mixed"
        # Intra-line mixed script: Cham phrase followed by Vietnamese annotation/glossary
        y = 60
        line_id = 0
        pair_start = ((sub_id - 20) * 3) % (len(INLINE_MIXED_PAIRS) - 6)
        
        # Render title
        title_font = get_font(FONT_CHAM_BOLD, 26)
        title_text = "ꨛꨯꨮ ꨆꨵꨯꨱꩃ ꨈꨣꩈ - NGỮ LỤC CHĂM & VIỆT"
        bbox_t = title_font.getbbox(title_text)
        draw.text((60, y), title_text, fill=(10, 10, 20), font=title_font)
        lines_gt.append({
            "line_id": line_id,
            "text": title_text,
            "script": "mixed",
            "box": [int(y + bbox_t[1]), int(60 + bbox_t[0]), int(y + bbox_t[3]), int(60 + bbox_t[2])],
            "polygon": [[int(60 + bbox_t[0]), int(y + bbox_t[1])], [int(60 + bbox_t[2]), int(y + bbox_t[1])],
                        [int(60 + bbox_t[2]), int(y + bbox_t[3])], [int(60 + bbox_t[0]), int(y + bbox_t[3])]]
        })
        y += (bbox_t[3] - bbox_t[1]) + 30
        line_id += 1

        for i in range(6):
            cham_term, vi_trans = INLINE_MIXED_PAIRS[(pair_start + i) % len(INLINE_MIXED_PAIRS)]
            full_line_text = f"{cham_term}: {vi_trans}"
            
            # Render using NotoSansCham (which supports both Cham and Latin)
            bbox = font_cham.getbbox(full_line_text)
            th = bbox[3] - bbox[1]
            x = 60
            draw.text((x, y), full_line_text, fill=(20, 20, 25), font=font_cham)
            lines_gt.append({
                "line_id": line_id,
                "text": full_line_text,
                "script": "mixed",
                "box": [int(y + bbox[1]), int(x + bbox[0]), int(y + bbox[3]), int(x + bbox[2])],
                "polygon": [[int(x + bbox[0]), int(y + bbox[1])], [int(x + bbox[2]), int(y + bbox[1])],
                            [int(x + bbox[2]), int(y + bbox[3])], [int(x + bbox[0]), int(y + bbox[3])]]
            })
            y += th + 24
            line_id += 1

    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_bgr, lines_gt, subtype

# =========================================================================
# GENERATOR DIRECTION B: 2-Column A4 Layout & Reading Order (adv_031 - adv_060)
# =========================================================================
def generate_direction_b(sample_idx):
    """
    Direction B: Bố cục 2 Cột A4
    - adv_031..adv_040: Wide gutter (60-80px), clean separation
    - adv_041..adv_050: Medium gutter (35-50px), realistic paper texture
    - adv_051..adv_060: Narrow gutter (20-30px), tight lines, stress test
    """
    sub_id = sample_idx - 30  # 1 to 30
    w = 1100
    h = 1500

    if sub_id <= 10:
        gutter = random.randint(60, 80)
        line_gap = random.randint(18, 24)
        texture_lvl = 1
        wave_amp = 0.0
    elif sub_id <= 20:
        gutter = random.randint(38, 52)
        line_gap = random.randint(13, 17)
        texture_lvl = 2
        wave_amp = 0.0
    else:
        gutter = random.randint(22, 32)
        line_gap = random.randint(9, 13)
        texture_lvl = 3
        wave_amp = random.uniform(1.2, 2.5)

    bg = make_paper_texture(w, h, texture_level=texture_lvl)
    img_pil = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Optional vertical separator line on some samples
    has_separator = (sub_id % 3 == 0)

    font_size = 22
    font_reg = get_font(FONT_CHAM_REG, font_size)
    font_bold = get_font(FONT_CHAM_BOLD, font_size)

    # Calculate column geometry
    margin_x = 70
    col_width = (w - 2 * margin_x - gutter) // 2
    col1_x = margin_x
    col2_x = margin_x + col_width + gutter

    if has_separator:
        sep_x = margin_x + col_width + gutter // 2
        draw.line([(sep_x, 80), (sep_x, h - 80)], fill=(200, 200, 200), width=1)

    lines_gt = []
    line_id = 0
    pool_idx = (sub_id * 11) % (len(CHAM_LINE_POOL) - 30)

    # Render Column 1 (Left column, reading order 0 to N1-1)
    y1 = 80
    col1_indices = []
    num_lines_c1 = random.randint(14, 18)
    for i in range(num_lines_c1):
        if y1 + 35 > h - 80:
            break
        text = CHAM_LINE_POOL[(pool_idx + i) % len(CHAM_LINE_POOL)]
        # Clip text if it exceeds col_width
        while font_reg.getbbox(text)[2] > col_width and len(text.split()) > 2:
            text = ' '.join(text.split()[:-1])

        bbox = font_reg.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Wave displacement if applicable
        dy = int(math.sin(i * 0.8) * wave_amp) if wave_amp > 0 else 0
        cur_y = y1 + dy

        # First line of section could be bold
        f = font_bold if i == 0 else font_reg
        draw.text((col1_x, cur_y), text, fill=(20, 20, 25), font=f)

        lines_gt.append({
            "line_id": line_id,
            "reading_order_idx": line_id,
            "column_id": 1,
            "text": text,
            "script": "cham",
            "box": [int(cur_y + bbox[1]), int(col1_x + bbox[0]), int(cur_y + bbox[3]), int(col1_x + bbox[2])],
            "polygon": [[int(col1_x + bbox[0]), int(cur_y + bbox[1])], [int(col1_x + bbox[2]), int(cur_y + bbox[1])],
                        [int(col1_x + bbox[2]), int(cur_y + bbox[3])], [int(col1_x + bbox[0]), int(cur_y + bbox[3])]]
        })
        col1_indices.append(line_id)
        line_id += 1
        y1 += th + line_gap

    # Render Column 2 (Right column, reading order N1 to N1+N2-1)
    y2 = 80
    col2_indices = []
    num_lines_c2 = random.randint(14, 18)
    pool_idx2 = (pool_idx + num_lines_c1) % len(CHAM_LINE_POOL)
    for i in range(num_lines_c2):
        if y2 + 35 > h - 80:
            break
        text = CHAM_LINE_POOL[(pool_idx2 + i) % len(CHAM_LINE_POOL)]
        while font_reg.getbbox(text)[2] > col_width and len(text.split()) > 2:
            text = ' '.join(text.split()[:-1])

        bbox = font_reg.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        dy = int(math.sin(i * 0.8 + 1.5) * wave_amp) if wave_amp > 0 else 0
        cur_y = y2 + dy

        f = font_bold if i == 0 else font_reg
        draw.text((col2_x, cur_y), text, fill=(20, 20, 25), font=f)

        lines_gt.append({
            "line_id": line_id,
            "reading_order_idx": line_id,
            "column_id": 2,
            "text": text,
            "script": "cham",
            "box": [int(cur_y + bbox[1]), int(col2_x + bbox[0]), int(cur_y + bbox[3]), int(col2_x + bbox[2])],
            "polygon": [[int(col2_x + bbox[0]), int(cur_y + bbox[1])], [int(col2_x + bbox[2]), int(cur_y + bbox[1])],
                        [int(col2_x + bbox[2]), int(cur_y + bbox[3])], [int(col2_x + bbox[0]), int(cur_y + bbox[3])]]
        })
        col2_indices.append(line_id)
        line_id += 1
        y2 += th + line_gap

    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    metadata = {
        "gutter_px": gutter,
        "col1_lines": col1_indices,
        "col2_lines": col2_indices,
        "has_separator": has_separator,
        "wave_amp": wave_amp
    }
    return img_bgr, lines_gt, metadata

# =========================================================================
# GENERATOR DIRECTION C: Mobile Field Degradations (adv_061 - adv_090)
# =========================================================================
def generate_direction_c(sample_idx):
    """
    Direction C: Ảnh chụp điện thoại thực địa
    - adv_061..adv_068: Cast Shadows (Bóng đổ bàn tay / điện thoại)
    - adv_069..adv_076: Specular Flash Glare (Loá flash quầng sáng)
    - adv_077..adv_083: Perspective Keystone Tilt (Góc chụp xiên/nghiêng 20-32 độ)
    - adv_084..adv_090: Motion Blur / Lens Blur (Rung tay khi bấm máy)
    """
    sub_id = sample_idx - 60  # 1 to 30
    w = 1000
    h = 800
    bg = make_paper_texture(w, h, texture_level=random.choice([1, 2, 3]))
    img_pil = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font_size = 24
    font_reg = get_font(FONT_CHAM_REG, font_size)
    font_bold = get_font(FONT_CHAM_BOLD, font_size)

    lines_gt = []
    line_id = 0
    pool_idx = (sub_id * 7) % (len(CHAM_LINE_POOL) - 12)

    # Render clean base document (8-10 lines)
    y = 60
    num_lines = random.randint(8, 10)
    for i in range(num_lines):
        text = CHAM_LINE_POOL[(pool_idx + i) % len(CHAM_LINE_POOL)]
        bbox = font_reg.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = 70

        f = font_bold if i == 0 else font_reg
        draw.text((x, y), text, fill=(25, 25, 30), font=f)

        lines_gt.append({
            "line_id": line_id,
            "text": text,
            "script": "cham",
            "box": [int(y + bbox[1]), int(x + bbox[0]), int(y + bbox[3]), int(x + bbox[2])],
            "polygon": [[int(x + bbox[0]), int(y + bbox[1])], [int(x + bbox[2]), int(y + bbox[1])],
                        [int(x + bbox[2]), int(y + bbox[3])], [int(x + bbox[0]), int(y + bbox[3])]]
        })
        y += th + 24
        line_id += 1

    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    degradation_meta = {}

    if sub_id <= 8:
        # C1: Cast Shadow
        degradation_meta["type"] = "cast_shadow"
        # Shadow angle and drop
        shadow_intensity = random.uniform(0.35, 0.60)  # fraction of light blocked
        degradation_meta["shadow_intensity"] = round(shadow_intensity, 3)
        
        # Create shadow mask
        mask = np.zeros((h, w), dtype=np.float32)
        # Random shadow edge orientation (top-left or right)
        if sub_id % 2 == 0:
            # Diagonal wedge shadow (like a phone held over top right)
            pts = np.array([[w * 0.2, 0], [w, 0], [w, h * 0.8], [w * 0.5, h * 0.4]], dtype=np.int32)
        else:
            # Left side hand shadow
            pts = np.array([[0, 0], [w * 0.55, 0], [w * 0.35, h], [0, h]], dtype=np.int32)
            
        cv2.fillPoly(mask, [pts], 1.0)
        # Blur shadow boundary to make realistic soft shadow
        blur_k = random.choice([41, 61, 81])
        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
        
        # Apply shadow: I_shadow = I * (1 - intensity * mask)
        shadow_mult = 1.0 - shadow_intensity * mask
        for c in range(3):
            img_bgr[:, :, c] = np.clip(img_bgr[:, :, c].astype(np.float32) * shadow_mult, 0, 255).astype(np.uint8)

    elif sub_id <= 16:
        # C2: Specular Flash Glare
        degradation_meta["type"] = "flash_glare"
        # Flash center near middle or upper quadrant
        cx = random.randint(int(w * 0.35), int(w * 0.65))
        cy = random.randint(int(h * 0.25), int(h * 0.60))
        radius = random.randint(110, 200)
        glare_intensity = random.uniform(160, 240)
        degradation_meta["center"] = [cx, cy]
        degradation_meta["radius"] = radius
        degradation_meta["intensity"] = round(glare_intensity, 1)

        Y, X = np.ogrid[:h, :w]
        dist_sq = (X - cx)**2 + (Y - cy)**2
        # Gaussian falloff
        glare_mask = np.exp(-dist_sq / (2.0 * (radius ** 2))).astype(np.float32)
        
        # Add pure glare to BGR
        for c in range(3):
            boosted = img_bgr[:, :, c].astype(np.float32) + glare_intensity * glare_mask
            img_bgr[:, :, c] = np.clip(boosted, 0, 255).astype(np.uint8)

    elif sub_id <= 23:
        # C3: Perspective Keystone Tilt (20 to 32 degrees)
        degradation_meta["type"] = "perspective_keystone"
        tilt_deg = random.uniform(20.0, 31.0)
        degradation_meta["tilt_degrees"] = round(tilt_deg, 1)
        
        # Calculate source and destination points
        # Keystoning: top is narrower and pushed back
        pad_x = int(w * math.sin(math.radians(tilt_deg)) * 0.45)
        pad_y = int(h * math.sin(math.radians(tilt_deg)) * 0.20)
        
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = np.float32([
            [pad_x, pad_y],
            [w - pad_x, pad_y],
            [w, h],
            [0, h]
        ])
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        img_bgr = cv2.warpPerspective(img_bgr, M, (w, h), borderValue=(240, 240, 240))
        
        # Transform GT line polygons and boxes
        for item in lines_gt:
            poly = np.array(item["polygon"], dtype=np.float32).reshape(-1, 1, 2)
            transformed_poly = cv2.perspectiveTransform(poly, M).reshape(-1, 2)
            item["polygon"] = [[int(pt[0]), int(pt[1])] for pt in transformed_poly]
            xs = [pt[0] for pt in transformed_poly]
            ys = [pt[1] for pt in transformed_poly]
            item["box"] = [int(min(ys)), int(min(xs)), int(max(ys)), int(max(xs))]

    else:
        # C4: Motion Blur / Lens Blur
        degradation_meta["type"] = "motion_blur"
        ksize = random.choice([7, 9, 11, 13])
        angle = random.uniform(10.0, 80.0)
        degradation_meta["kernel_size"] = ksize
        degradation_meta["angle"] = round(angle, 1)
        
        # Build motion blur kernel
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
            
        img_bgr = cv2.filter2D(img_bgr, -1, kernel)

    return img_bgr, lines_gt, degradation_meta

# =========================================================================
# MAIN GENERATOR LOOP: 90 SAMPLES
# =========================================================================
def main():
    print("=" * 70)
    print("🚀 GENERATING ADVANCED CHAM OCR BENCHMARK SUITE (90 TEST CASES)")
    print("   Direction A: adv_001..adv_030 (Bilingual / Code-Switching)")
    print("   Direction B: adv_031..adv_060 (2-Column A4 / Reading Order)")
    print("   Direction C: adv_061..adv_090 (Mobile Field Degradations)")
    print("=" * 70)

    dataset_gt = []

    for i in range(1, 91):
        sample_id = f"adv_{i:03d}"
        filename = f"{sample_id}.png"
        filepath = os.path.join(OUTPUT_IMG_DIR, filename)

        if i <= 30:
            direction = "A_bilingual"
            img_bgr, lines_gt, meta = generate_direction_a(i)
            entry = {
                "sample_id": sample_id,
                "filename": filename,
                "direction": direction,
                "subtype": meta,
                "width": int(img_bgr.shape[1]),
                "height": int(img_bgr.shape[0]),
                "lines": lines_gt
            }
        elif i <= 60:
            direction = "B_multicol"
            img_bgr, lines_gt, meta = generate_direction_b(i)
            entry = {
                "sample_id": sample_id,
                "filename": filename,
                "direction": direction,
                "gutter_px": meta["gutter_px"],
                "col1_lines": meta["col1_lines"],
                "col2_lines": meta["col2_lines"],
                "has_separator": meta["has_separator"],
                "wave_amp": meta["wave_amp"],
                "width": int(img_bgr.shape[1]),
                "height": int(img_bgr.shape[0]),
                "lines": lines_gt
            }
        else:
            direction = "C_mobile_degradation"
            img_bgr, lines_gt, meta = generate_direction_c(i)
            entry = {
                "sample_id": sample_id,
                "filename": filename,
                "direction": direction,
                "degradation": meta,
                "width": int(img_bgr.shape[1]),
                "height": int(img_bgr.shape[0]),
                "lines": lines_gt
            }

        # Save image
        cv2.imwrite(filepath, img_bgr)
        dataset_gt.append(entry)

        if i % 15 == 0 or i == 90:
            print(f"  [Progress] Generated {i}/90 samples -> {sample_id} ({direction})")

    # Save Ground Truth JSON
    def default_converter(o):
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        elif isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(GT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_gt, f, ensure_ascii=False, indent=2, default=default_converter)

    total_lines = sum(len(s["lines"]) for s in dataset_gt)
    print("=" * 70)
    print(f"✅ Generated 90 advanced benchmark images in: {OUTPUT_IMG_DIR}")
    print(f"✅ Saved ground truth metadata for {total_lines} lines to: {GT_OUTPUT_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    main()
