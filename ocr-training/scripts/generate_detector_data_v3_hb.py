#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR Detector Dataset Generator V3 (`Cham-DBNet v2` — 100% HarfBuzz Multi-Layout & Hard-Negative Pages)
===========================================================================================================
Generates 1,120 synthetic Cham document & manuscript pages (1,000 Train / 120 Val) in PaddleOCR
SimpleDataSet polygon format (`points` + `transcription`) using 100% HarfBuzz + FreeType shaping:
  - 45% Single-Column Ultra-Tight Stacked Diacritics (`gap = 2..10px` between ink envelopes, rich in `ꨳ + ꨪ + ꩌ`,
    stanza headers `꩑꩞..꩙꩙꩙꩞`, short ending lines, and bottom dates `꩘/꩕/꩒꩐꩒꩖` / `8/5/2026`).
  - 20% Multi-Column (`2-Column` & `Centered Title + 2-Column`, gutter `26..52px`) to prevent cross-column bridging.
  - 20% Wavy / Curved / 3D Perspective / Handheld Skewed (`±7°`) Camera-Captured Pages.
  - 15% Sparse & Heavy-Degradation Hard-Negative Pages (1–3 lines amidst large stained parchment margins,
    decorative ruling lines, speckle dots, spine shadows, `erode=2..3`, motion/Gaussian blur) for OHEM negative mining.

Strict Invariants:
  - 100% Benchmark Blind & OOD Font Uncontaminated (ONLY `ocr-training/data/fonts/NotoSansCham-*.ttf`).
  - Ground-truth `points` tightly wrap the unified HarfBuzz glyph ink envelope (never splitting upper bindus `ꩌ`
    or subjoined `-ya` `ꨳ` into micro-fragments, and never overlapping adjacent ultra-tight lines).
"""

import os
import io
import sys
import math
import json
import time
import random
import shutil
import hashlib
from datetime import datetime, timezone

import cv2
import numpy as np
import uharfbuzz as hb
import freetype
from PIL import Image

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)
REPORTS_DIR = os.path.join(PROJECT_ROOT, "REPORTS")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from generate_data_v24_1_r3_1 import (
    CORPUS_WORDS_SORTED,
    SESQ_WORDS,
    AA33_WORDS,
    BINDU_WORDS,
    ILLEGAL_MEDIAL_RE,
    normalize_unicode,
    to_cham_num,
    synth_fresh_monosyllable,
    find_asset_path,
)

_FT_HB_CACHE = {}


def get_hb_font(weight: str, font_size: int):
    key = (weight, font_size)
    if key not in _FT_HB_CACHE:
        fpath = find_asset_path(f"NotoSansCham-{weight}.ttf") or find_asset_path("NotoSansCham-Regular.ttf")
        if not fpath or not os.path.exists(fpath):
            raise RuntimeError("Missing NotoSansCham TTF in ocr-training/data/fonts")
        face = freetype.Face(fpath)
        face.set_char_size(font_size * 64)
        hb_face = hb.Face(open(fpath, "rb").read())
        hb_font = hb.Font(hb_face)
        hb_font.scale = (font_size * 64, font_size * 64)
        asc = face.size.ascender / 64.0
        desc = face.size.descender / 64.0
        _FT_HB_CACHE[key] = (face, hb_font, asc, desc)
    return _FT_HB_CACHE[key]


def render_hb_alpha_strip(text: str, weight: str, font_size: int):
    """
    Shapes and rasterizes `text` into an alpha mask `(H_strip, W_strip)` uint8 in `[0..255]`
    and returns `(alpha_mask, ink_x0, ink_y0, ink_x1, ink_y1)` for exact unified ink-envelope polygons.
    """
    face, hb_font, asc, desc = get_hb_font(weight, font_size)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf)

    pad_x, pad_y = 8, int(math.ceil(font_size * 0.55))
    adv_w = sum(pos.x_advance / 64.0 for pos in buf.glyph_positions)
    h_strip = max(24, int(math.ceil(asc - desc)) + pad_y * 2)
    w_strip = max(16, int(math.ceil(adv_w)) + pad_x * 2)
    alpha = np.zeros((h_strip, w_strip), dtype=np.uint8)

    pen_x = float(pad_x)
    pen_y = pad_y + int(math.ceil(asc))
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
        bm = face.glyph.bitmap
        gw, gh = bm.width, bm.rows
        if gw > 0 and gh > 0:
            xp = int(round(pen_x + (pos.x_offset / 64.0) + face.glyph.bitmap_left))
            yp = int(round(pen_y - (pos.y_offset / 64.0) - face.glyph.bitmap_top))
            g_arr = np.frombuffer(bytes(bm.buffer), dtype=np.uint8).reshape((gh, gw))
            x1, y1 = max(0, xp), max(0, yp)
            x2, y2 = min(w_strip, xp + gw), min(h_strip, yp + gh)
            if x2 > x1 and y2 > y1:
                alpha[y1:y2, x1:x2] = np.maximum(alpha[y1:y2, x1:x2], g_arr[y1 - yp:y2 - yp, x1 - xp:x2 - xp])
        pen_x += pos.x_advance / 64.0
        pen_y += pos.y_advance / 64.0

    ys, xs = np.where(alpha > 18)
    if len(xs) == 0:
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return alpha[y0:y1, x0:x1]


def make_safe_cham_line(rng: random.Random, n_words: int, stacked_heavy: bool = False, header_mode: str = "body") -> str:
    n_corp = len(CORPUS_WORDS_SORTED)
    if header_mode == "title":
        w1 = SESQ_WORDS[rng.randint(0, len(SESQ_WORDS) - 1)]
        w2 = synth_fresh_monosyllable(rng, force_bindu=True)
        w3 = CORPUS_WORDS_SORTED[rng.randint(0, n_corp - 1)]
        return normalize_unicode(f"{w1} {w2} {w3}")
    if header_mode == "date":
        d, mo, yr = rng.randint(1, 28), rng.randint(1, 12), rng.choice([2004, 2012, 2020, 2024, 2026])
        return f"{to_cham_num(d)}/{to_cham_num(mo)}/{to_cham_num(yr)}" if rng.random() < 0.65 else f"{d}/{mo}/{yr}"

    words = []
    if rng.random() < 0.22:
        words.append(f"{to_cham_num(rng.randint(1, 999))}꩞")
    for k in range(n_words):
        if k % 2 == 0:
            if stacked_heavy and rng.random() < 0.55:
                pool = AA33_WORDS if (rng.random() < 0.5 and AA33_WORDS) else BINDU_WORDS
                cw = rng.choice(pool) if pool else CORPUS_WORDS_SORTED[rng.randint(0, n_corp - 1)]
            elif rng.random() < 0.35 and SESQ_WORDS:
                cw = rng.choice(SESQ_WORDS)
            else:
                cw = CORPUS_WORDS_SORTED[rng.randint(0, n_corp - 1)]
            words.append(cw)
        else:
            sw = synth_fresh_monosyllable(
                rng,
                force_aa33=(stacked_heavy and rng.random() < 0.45),
                force_bindu=(stacked_heavy and rng.random() < 0.45),
                force_pre=(rng.random() < 0.4),
            )
            if rng.random() < 0.10:
                sw = f"{sw}-{synth_fresh_monosyllable(rng)}"
            words.append(sw)
    tail = rng.choice(["", " ꩝", " ꩞", " ꩝꩝"])
    txt = normalize_unicode((" ".join(words) + tail).strip())
    return txt if not ILLEGAL_MEDIAL_RE.search(txt) else normalize_unicode(" ".join(words[:2]))


def make_page_background(width: int, height: int, rng: random.Random, np_rng: np.random.Generator, hard_neg_decor: bool = False) -> np.ndarray:
    style = rng.random()
    if style < 0.38 and not hard_neg_decor:
        base_v = rng.randint(236, 253)
        bg = np.full((height, width, 3), base_v, dtype=np.float32)
        bg += np_rng.normal(0.0, rng.uniform(1.5, 4.5), size=(height, width, 1)).astype(np.float32)
    else:
        # Aged book / antique Cham manuscript parchment (BGR)
        b_c = rng.randint(178, 222)
        g_c = rng.randint(205, 240)
        r_c = rng.randint(218, 248)
        bg = np.empty((height, width, 3), dtype=np.float32)
        bg[:, :, 0] = b_c
        bg[:, :, 1] = g_c
        bg[:, :, 2] = r_c
        gh, gw = max(4, height // 48), max(4, width // 48)
        shade = cv2.resize(np_rng.normal(0.0, rng.uniform(6.0, 14.0), (gh, gw)).astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
        fine = np_rng.normal(0.0, rng.uniform(2.5, 6.5), (height, width)).astype(np.float32)
        bg += (shade + fine)[:, :, None]

    out = np.clip(bg, 0, 255).astype(np.uint8)

    if hard_neg_decor or rng.random() < 0.22:
        # Draw decorative margin border / faint ruling lines / dust speckles (hard negative OHEM targets!)
        margin = rng.randint(14, 28)
        col_line = (rng.randint(120, 185), rng.randint(130, 190), rng.randint(140, 195))
        if rng.random() < 0.65:
            cv2.rectangle(out, (margin, margin), (width - margin, height - margin), col_line, thickness=rng.choice([1, 2]))
        if hard_neg_decor:
            # Add dust specks / punch holes in margins
            for _ in range(rng.randint(8, 26)):
                cx = rng.randint(8, width - 8)
                cy = rng.randint(8, height - 8)
                rad = rng.randint(1, 3)
                cv2.circle(out, (cx, cy), rad, (rng.randint(70, 150),) * 3, -1)
    return out


def composite_strip(page_bgr: np.ndarray, alpha: np.ndarray, x0: int, y0: int, ink_bgr: tuple):
    ph, pw = page_bgr.shape[:2]
    sh, sw = alpha.shape[:2]
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(pw, x0 + sw), min(ph, y0 + sh)
    if x2 <= x1 or y2 <= y1:
        return None
    sub_a = (alpha[y1 - y0:y2 - y0, x1 - x0:x2 - x0].astype(np.float32) / 255.0)[:, :, None]
    sub_p = page_bgr[y1:y2, x1:x2].astype(np.float32)
    ink_arr = np.array(ink_bgr, dtype=np.float32)[None, None, :]
    page_bgr[y1:y2, x1:x2] = np.clip(sub_p * (1.0 - sub_a) + ink_arr * sub_a, 0, 255).astype(np.uint8)
    return (x1, y1, x2, y2)


def generate_one_page(idx: int, is_val: bool, out_img_dir: str, rel_prefix: str, seed: int):
    rng = random.Random(seed + idx * 97)
    np_rng = np.random.default_rng(seed + idx * 131)

    width = rng.choice([768, 864, 960, 1024])
    height = rng.choice([640, 768, 896, 1024])
    weight = rng.choice(["Regular", "Regular", "Regular", "Bold", "Black"])
    font_size = rng.randint(17, 27)

    # Layout taxonomy:
    # 0..44%  -> ultra-tight single-column stacked diacritics
    # 45..64% -> 2-column or centered-title + 2-column
    # 65..84% -> wavy / 3D perspective / skewed camera page
    # 85..99% -> sparse & heavy-degradation hard-negative page
    bucket = idx % 20
    is_two_col = bucket in (9, 10, 11, 12)
    is_wavy_persp = bucket in (13, 14, 15, 16)
    is_sparse_hard_neg = bucket in (17, 18, 19)
    is_ultra_tight = bucket < 9

    page_bgr = make_page_background(width, height, rng, np_rng, hard_neg_decor=is_sparse_hard_neg)
    ink_bgr = (rng.randint(18, 65), rng.randint(18, 60), rng.randint(18, 58))

    placed_items = []  # list of (text, x1, y1, x2, y2, col_id)

    def place_column_lines(col_x0: int, col_x1: int, y_top: int, y_bot: int, max_lines: int, gap_range: tuple, col_id: int):
        cur_y = y_top
        max_w = col_x1 - col_x0
        lines_done = 0
        while cur_y + font_size + 12 < y_bot and lines_done < max_lines:
            is_short_end = (lines_done == max_lines - 1 and rng.random() < 0.55)
            n_w = rng.randint(2, 4) if (is_two_col or is_short_end) else rng.randint(4, 9)
            txt = make_safe_cham_line(rng, n_words=n_w, stacked_heavy=is_ultra_tight)
            # Trim words until rendered strip fits inside column width
            words = txt.split()
            alpha = render_hb_alpha_strip(txt, weight, font_size)
            while alpha is not None and alpha.shape[1] > max_w - 16 and len(words) > 2:
                words = words[:-1]
                txt = " ".join(words)
                alpha = render_hb_alpha_strip(txt, weight, font_size)
            if alpha is None or alpha.shape[1] > max_w - 4:
                break
            sh, sw = alpha.shape[:2]
            if cur_y + sh >= y_bot:
                break
            indent = rng.randint(18, 38) if (lines_done == 0 and not is_two_col and rng.random() < 0.5) else rng.randint(0, 8)
            xp = min(col_x0 + indent, col_x1 - sw - 2)
            placed = composite_strip(page_bgr, alpha, xp, cur_y, ink_bgr)
            if placed:
                placed_items.append((txt, placed[0], placed[1], placed[2], placed[3], col_id))
            gap = rng.randint(gap_range[0], gap_range[1])
            cur_y += sh + gap
            lines_done += 1
        return cur_y

    y_cursor = rng.randint(36, 68)
    # Optional centered header title (35% of pages, matching `harum_palei_01` title + body + footer date!)
    if not is_sparse_hard_neg and rng.random() < 0.35:
        title_txt = make_safe_cham_line(rng, n_words=3, header_mode="title")
        t_alpha = render_hb_alpha_strip(title_txt, "Bold", min(30, font_size + 3))
        if t_alpha is not None and t_alpha.shape[1] < width - 80:
            tx = (width - t_alpha.shape[1]) // 2
            placed = composite_strip(page_bgr, t_alpha, tx, y_cursor, ink_bgr)
            if placed:
                placed_items.append((title_txt, placed[0], placed[1], placed[2], placed[3], -1))
                y_cursor += t_alpha.shape[0] + rng.randint(16, 28)

    if is_two_col:
        gutter = rng.randint(26, 52)
        left_x0 = rng.randint(36, 54)
        right_x1 = width - left_x0
        col_w = (right_x1 - left_x0 - gutter) // 2
        gap_r = (3, 10) if rng.random() < 0.55 else (10, 20)
        place_column_lines(left_x0, left_x0 + col_w, y_cursor, height - 48, rng.randint(6, 15), gap_r, col_id=0)
        place_column_lines(left_x0 + col_w + gutter, right_x1, y_cursor, height - 48, rng.randint(6, 15), gap_r, col_id=1)
    elif is_sparse_hard_neg:
        # 1 to 3 isolated lines on large noisy/ruled/stained page
        y_sp = rng.randint(90, max(120, height // 3))
        place_column_lines(rng.randint(48, 110), width - 54, y_sp, height - 60, rng.randint(1, 3), (18, 42), col_id=0)
    else:
        gap_r = (2, 8) if is_ultra_tight else (8, 22)
        n_paras = rng.randint(1, 3)
        for _ in range(n_paras):
            y_cursor = place_column_lines(rng.randint(38, 62), width - 42, y_cursor, height - 68, rng.randint(4, 9), gap_r, col_id=0)
            y_cursor += rng.randint(12, 24)
        # Optional footer date line (25% chance)
        if y_cursor + font_size + 24 < height and rng.random() < 0.30:
            dt_txt = make_safe_cham_line(rng, n_words=1, header_mode="date")
            d_alpha = render_hb_alpha_strip(dt_txt, weight, max(16, font_size - 1))
            if d_alpha is not None and d_alpha.shape[1] < width - 100:
                dx = rng.randint(width // 2, width - d_alpha.shape[1] - 48)
                placed = composite_strip(page_bgr, d_alpha, dx, y_cursor + 4, ink_bgr)
                if placed:
                    placed_items.append((dt_txt, placed[0], placed[1], placed[2], placed[3], -2))

    if not placed_items:
        # Guaranteed fallback single line
        fb_txt = make_safe_cham_line(rng, n_words=4)
        fb_a = render_hb_alpha_strip(fb_txt, "Regular", 22)
        placed = composite_strip(page_bgr, fb_a, 50, height // 2, ink_bgr)
        placed_items.append((fb_txt, placed[0], placed[1], placed[2], placed[3], 0))

    # Build non-overlapping polygons per column
    line_boxes = []
    by_col = {}
    for item in placed_items:
        by_col.setdefault(item[5], []).append(item)

    for cid, citems in by_col.items():
        citems.sort(key=lambda z: z[2])
        for k, (txt, x1, y1, x2, y2, _) in enumerate(citems):
            px = 3
            top_lim = 2 if k == 0 else (citems[k - 1][4] + y1) // 2
            bot_lim = (height - 2) if k == len(citems) - 1 else (y2 + citems[k + 1][2]) // 2
            py1 = max(top_lim, y1 - 2)
            py2 = min(bot_lim, y2 + 2)
            px1 = max(2, x1 - px)
            px2 = min(width - 2, x2 + px)
            if py2 - py1 >= 8 and px2 - px1 >= 8:
                poly = [[float(px1), float(py1)], [float(px2), float(py1)], [float(px2), float(py2)], [float(px1), float(py2)]]
                line_boxes.append((txt, poly))

    # Geometric wave + 3D perspective + skew warp
    do_warp = is_wavy_persp or (rng.random() < 0.45)
    if do_warp:
        amp = rng.uniform(1.8, 5.0) if is_wavy_persp else rng.uniform(0.0, 2.2)
        freq = rng.uniform(0.004, 0.010)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        skew = rng.uniform(0.025, 0.065) if is_wavy_persp else rng.uniform(0.005, 0.022)
        pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        pts2 = np.float32([
            [rng.uniform(0, width * skew), rng.uniform(0, height * skew)],
            [width - rng.uniform(0, width * skew), rng.uniform(0, height * skew)],
            [rng.uniform(0, width * skew), height - rng.uniform(0, height * skew)],
            [width - rng.uniform(0, width * skew), height - rng.uniform(0, height * skew)],
        ])
        mat = cv2.getPerspectiveTransform(pts1, pts2)
        if amp > 0.2:
            X, Y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
            # Apply inverse vertical shift on image so forward point shift `y_new = y_old - amp*sin(...)` aligns 100%!
            dy_map = amp * np.sin(X * freq + phase).astype(np.float32)
            warped_wave = cv2.remap(page_bgr, X, Y + dy_map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            warped_wave = page_bgr
        page_warped = cv2.warpPerspective(warped_wave, mat, (width, height), borderMode=cv2.BORDER_REPLICATE)

        final_labels = []
        for txt, poly in line_boxes:
            shifted = []
            for x, y in poly:
                wy = y - (amp * math.sin(x * freq + phase) if amp > 0.2 else 0.0)
                shifted.append([x, wy])
            arr_pts = np.array(shifted, dtype=np.float32).reshape(-1, 1, 2)
            t_pts = cv2.perspectiveTransform(arr_pts, mat).reshape(-1, 2)
            clamped = [[round(float(np.clip(px, 0, width - 1)), 1), round(float(np.clip(py, 0, height - 1)), 1)] for px, py in t_pts]
            final_labels.append({"transcription": txt, "points": clamped})
    else:
        page_warped = page_bgr
        final_labels = [{"transcription": txt, "points": [[round(x, 1), round(y, 1)] for x, y in poly]} for txt, poly in line_boxes]

    # Optical scan/camera degradations (blur, erosion/fade, shadow, JPEG)
    if is_sparse_hard_neg or rng.random() < 0.35:
        deg_m = rng.random()
        if deg_m < 0.35:
            # Uneven shadow + Gaussian defocus
            gx = np.linspace(rng.uniform(0.55, 0.85), 1.0, width, dtype=np.float32)
            if rng.random() < 0.5:
                gx = gx[::-1]
            page_warped = np.clip(page_warped.astype(np.float32) * gx[None, :, None], 0, 255).astype(np.uint8)
            sig = rng.uniform(0.8, 2.2)
            page_warped = cv2.GaussianBlur(page_warped, (0, 0), sigmaX=sig, borderType=cv2.BORDER_REPLICATE)
        elif deg_m < 0.65:
            # Mild morphological dry-ink fade (dilate bright paper -> erode dark ink by 2x2)
            page_warped = cv2.dilate(page_warped, np.ones((2, 2), np.uint8), iterations=1)
        else:
            # JPEG compression + sensor grain
            noise = np_rng.normal(0.0, rng.uniform(3.0, 12.0), size=page_warped.shape).astype(np.float32)
            page_warped = np.clip(page_warped.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    img_name = f"{rel_prefix}_{idx:05d}.jpg"
    img_path = os.path.join(out_img_dir, img_name)
    cv2.imwrite(img_path, page_warped, [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(78, 95)])
    return f"{rel_prefix}/{img_name}\t{json.dumps(final_labels, ensure_ascii=False)}", len(final_labels)


def main():
    t0 = time.time()
    n_train = 1000
    n_val = 120
    det_root = os.path.join(TRAINING_DIR, "data", "detector_v3_hb")
    if os.path.exists(det_root):
        shutil.rmtree(det_root)
    tr_dir = os.path.join(det_root, "train_images")
    va_dir = os.path.join(det_root, "val_images")
    os.makedirs(tr_dir, exist_ok=True)
    os.makedirs(va_dir, exist_ok=True)

    print("=" * 80)
    print(f"🚀 GENERATING CHAM-DBNET V2 HARFBUZZ MULTI-LAYOUT DATASET ({n_train} TRAIN / {n_val} VAL PAGES)")
    print("=" * 80)

    tr_labels = []
    tr_polys = 0
    for i in range(n_train):
        lbl, cnt = generate_one_page(i, False, tr_dir, "train_images", seed=20261130)
        tr_labels.append(lbl)
        tr_polys += cnt
        if (i + 1) % 250 == 0:
            print(f"  [train] {i + 1}/{n_train} pages ({tr_polys:,} line polygons)...")

    va_labels = []
    va_polys = 0
    for i in range(n_val):
        lbl, cnt = generate_one_page(i, True, va_dir, "val_images", seed=20261230)
        va_labels.append(lbl)
        va_polys += cnt

    tr_lbl_p = os.path.join(det_root, "det_train_label.txt")
    va_lbl_p = os.path.join(det_root, "det_val_label.txt")
    with open(tr_lbl_p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(tr_labels) + "\n")
    with open(va_lbl_p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(va_labels) + "\n")

    manifest = {
        "stage": "CHAM_DBNET_V2_DETECTOR_V3_HB",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(time.time() - t0, 2),
        "train_pages": len(tr_labels),
        "val_pages": len(va_labels),
        "train_line_polygons": tr_polys,
        "val_line_polygons": va_polys,
        "train_label_sha256": hashlib.sha256(open(tr_lbl_p, "rb").read()).hexdigest(),
        "val_label_sha256": hashlib.sha256(open(va_lbl_p, "rb").read()).hexdigest(),
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "detector_v3_hb_manifest.json"), "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    print(f"✅ Cham-DBNet v2 dataset generated in {manifest['duration_sec']}s -> {det_root}")
    print(f"   Train: {len(tr_labels)} pages ({tr_polys:,} lines) | Val: {len(va_labels)} pages ({va_polys:,} lines)")


if __name__ == "__main__":
    main()
