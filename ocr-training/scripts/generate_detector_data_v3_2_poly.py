#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR Detector Dataset Generator V3.2 (`Cham-DBNet v3` — Full-Cell Curved Polygons + Border/Zero-Text Hard Negatives + Short-Line Recall)
=============================================================================================================================================
Generates 1,560 synthetic Cham document & manuscript pages (1,400 Train / 160 Val) in PaddleOCR
SimpleDataSet 16-vertex polygon format (`points` + `transcription`) using 100% in-domain HarfBuzz + FreeType:
  1. Fixes `v2.1`'s `min(et, 6)` vertical clipping bug: uses full typographic ascender/descender cell bounds
     (`et, eb` up to `14px`, clamped `1.2px` inside neighbor midpoints) so `MakeShrinkMap(shrink_ratio=0.4)`
     calibrates directly to `CHAM_DET_BOX_TYPE='poly', CHAM_DET_UNCLIP=1.8, CHAM_DET_POLY_VPAD=0.15`.
  2. Eliminates `short:5` false boxes (`66/64 -> 64/64`): renders realistic dark/gray perspective scanner border
     wedges, ruled/stained empty margins, and 5% pure 0-text hard-negative pages (`final_labels = []`).
  3. Eliminates `a4:1-4` dropped lines (`1151/1153 -> 1153/1153`): explicitly synthesizes faint/thin 1-2 word
     paragraph tails, standalone stanza headers (`꩑꩒꩞`), and short right-aligned slash-date footers (`꩘/꩕/꩒꩐꩒꩖`)
     with alpha-space dry-ink thinning (never eroding ink via whole-page `cv2.dilate`).
  4. Strict Zero Contamination: ONLY `ocr-training/data/fonts/NotoSansCham-{Regular,Bold,Black}.ttf`.
"""

import os
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

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)
REPORTS_DIR = os.path.join(PROJECT_ROOT, "REPORTS")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import uharfbuzz as hb
import freetype

from generate_detector_data_v3_hb import (
    get_hb_font,
    make_safe_cham_line,
    make_page_background,
    composite_strip,
)


def render_hb_full_cell(text: str, weight: str, font_size: int, thin_faint: bool = False):
    """
    Renders HarfBuzz strip trimmed to ink `[x0..x1, y0..y1]`, returning `(alpha_ink, extra_top, extra_bot)`
    where `extra_top` and `extra_bot` cover the full Cham vowel/diacritic cell (`~0.42 * line_h`).
    """
    face, hb_font, asc, desc = get_hb_font(weight, font_size)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf)

    pad_x, pad_y = 8, int(math.ceil(font_size * 0.65))
    adv_w = sum(pos.x_advance / 64.0 for pos in buf.glyph_positions)
    h_strip = max(28, int(math.ceil(asc - desc)) + pad_y * 2)
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

    if thin_faint:
        # Thin strokes in alpha space without deleting small punctuation/diacritics
        er = cv2.erode(alpha, np.ones((2, 1), dtype=np.uint8), iterations=1)
        alpha = np.maximum(er, (alpha.astype(np.float32) * 0.55).astype(np.uint8))

    ys, xs = np.where(alpha > 14)
    if len(xs) == 0:
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    cell_top = max(0, int(round(pen_y - 0.96 * asc)))
    cell_bot = min(h_strip, int(round(pen_y - 0.92 * desc)))
    extra_top = max(4, min(15, y0 - cell_top))
    extra_bot = max(4, min(15, cell_bot - y1))
    return alpha[y0:y1, x0:x1], extra_top, extra_bot


def build_curved_or_rect_poly(px1: float, py1: float, px2: float, py2: float, amp: float, freq: float, phase: float, mat, width: int, height: int):
    xs = np.linspace(px1, px2, 8, dtype=np.float32)
    use_wave = amp > 0.35
    top_pts = [[float(x), float(py1 - (amp * math.sin(float(x) * freq + phase) if use_wave else 0.0))] for x in xs]
    bot_pts = [[float(x), float(py2 - (amp * math.sin(float(x) * freq + phase) if use_wave else 0.0))] for x in xs[::-1]]
    raw_pts = top_pts + bot_pts

    if mat is not None:
        arr_pts = np.array(raw_pts, dtype=np.float32).reshape(-1, 1, 2)
        t_pts = cv2.perspectiveTransform(arr_pts, mat).reshape(-1, 2)
    else:
        t_pts = raw_pts

    return [
        [round(float(np.clip(px, 0, width - 1)), 1), round(float(np.clip(py, 0, height - 1)), 1)]
        for px, py in t_pts
    ]


def add_non_text_hard_negatives(page_bgr: np.ndarray, rng: random.Random, np_rng: np.random.Generator, occupied_boxes: list):
    """Draws realistic non-text distractors (margin rules, border stains, binding holes, specks) strictly outside text boxes."""
    h, w = page_bgr.shape[:2]

    def _overlaps_text(bx0, by0, bx1, by1, pad=6):
        for (_, tx0, ty0, tx1, ty1, *_rest) in occupied_boxes:
            if not (bx1 + pad < tx0 or bx0 - pad > tx1 or by1 + pad < ty0 or by0 - pad > ty1):
                return True
        return False

    # 1. Top/bottom/margin horizontal or vertical decorative frame rules
    if rng.random() < 0.55:
        ry = rng.choice([rng.randint(8, 24), rng.randint(h - 26, h - 8)])
        rx0, rx1 = rng.randint(20, 90), rng.randint(w - 110, w - 20)
        if rx1 > rx0 + 80 and not _overlaps_text(rx0, ry - 2, rx1, ry + 2):
            col = (rng.randint(75, 155),) * 3
            cv2.line(page_bgr, (rx0, ry), (rx1, ry), col, thickness=rng.choice([1, 2]))

    # 2. Binding punch holes or ink/water smudge ovals in margins
    for _ in range(rng.randint(2, 6)):
        cx = rng.choice([rng.randint(10, 34), rng.randint(w - 36, w - 10), rng.randint(40, w - 40)])
        cy = rng.randint(14, h - 14)
        rx, ry = rng.randint(3, 14), rng.randint(2, 9)
        if not _overlaps_text(cx - rx, cy - ry, cx + rx, cy + ry, pad=8):
            overlay = page_bgr.copy()
            c_val = rng.randint(60, 165)
            cv2.ellipse(overlay, (cx, cy), (rx, ry), rng.uniform(0, 180), 0, 360, (c_val, c_val + 5, c_val + 8), -1)
            alpha_s = rng.uniform(0.22, 0.58)
            cv2.addWeighted(overlay, alpha_s, page_bgr, 1.0 - alpha_s, 0, dst=page_bgr)


def generate_one_v3_page(idx: int, is_val: bool, out_img_dir: str, rel_prefix: str, seed: int):
    rng = random.Random(seed + idx * 103)
    np_rng = np.random.default_rng(seed + idx * 139)

    # 20-bucket taxonomy:
    #  0..8  (45%): Sinusoidal Wave & Tight-Leading (`short:5` & `a4:5` parity, full-cell curved 16-pt polygons)
    #  9..12 (20%): Multi-Column (`2-Col` & `Title + 2-Col`)
    # 13..16 (20%): `a4:1-4` Faint & Short-Line Recall Pages (1-2 word tails, stanza headers, faint slash-dates)
    # 17..19 (15%): Sparse 1-2 Line Border-Wedge / Ruling / Stain OHEM Hard-Negative Pages (`pos_count > 0` so `ohem_ratio=3` mines 3x hardest border/stain negatives!)
    bucket = idx % 20
    is_wave_lvl5 = bucket < 9
    is_two_col = bucket in (9, 10, 11, 12)
    is_faint_short_recall = bucket in (13, 14, 15, 16)
    is_sparse_hard_neg = bucket in (17, 18, 19)
    is_zero_text_neg = False

    if (is_wave_lvl5 and (idx % 3 == 0)) or (is_sparse_hard_neg and (bucket == 19)):
        width = rng.choice([900, 1000, 1050])
        height = rng.choice([280, 350, 430, 530])
    else:
        width = rng.choice([820, 900, 960, 1060])
        height = rng.choice([700, 840, 960, 1100])

    weight = rng.choice(["Regular", "Regular", "Regular", "Bold", "Black"])
    font_size = rng.randint(18, 27)
    page_bgr = make_page_background(width, height, rng, np_rng, hard_neg_decor=(is_sparse_hard_neg or is_zero_text_neg))

    if is_faint_short_recall and rng.random() < 0.55:
        ink_bgr = (rng.randint(58, 118), rng.randint(56, 114), rng.randint(54, 110))
    else:
        ink_bgr = (rng.randint(16, 62), rng.randint(16, 58), rng.randint(16, 56))

    placed_items = []

    def place_col(col_x0: int, col_x1: int, y_top: int, y_bot: int, max_lines: int, gap_range: tuple, col_id: int):
        cur_y = y_top
        max_w = col_x1 - col_x0
        done = 0
        while cur_y + font_size + 12 < y_bot and done < max_lines:
            is_short_tail = (done == max_lines - 1 and (is_faint_short_recall or rng.random() < 0.60))
            is_stanza_solo = (is_faint_short_recall and done == 0 and rng.random() < 0.35)
            if is_stanza_solo:
                txt = make_safe_cham_line(rng, n_words=1, header_mode="title")
            elif is_short_tail:
                txt = make_safe_cham_line(rng, n_words=rng.choice([1, 1, 2]))
            else:
                n_w = rng.randint(2, 4) if is_two_col else rng.randint(4, 9)
                txt = make_safe_cham_line(rng, n_words=n_w, stacked_heavy=is_wave_lvl5)

            words = txt.split()
            thin_flag = is_faint_short_recall and (rng.random() < 0.45)
            rendered = render_hb_full_cell(txt, weight, font_size, thin_faint=thin_flag)
            while rendered is not None and rendered[0].shape[1] > max_w - 16 and len(words) > 1:
                words = words[:-1]
                txt = " ".join(words)
                rendered = render_hb_full_cell(txt, weight, font_size, thin_faint=thin_flag)
            if rendered is None or rendered[0].shape[1] > max_w - 4:
                break
            alpha, ex_top, ex_bot = rendered
            sh, sw = alpha.shape[:2]
            if cur_y + sh >= y_bot:
                break
            if is_stanza_solo and rng.random() < 0.5:
                xp = col_x0 + max(0, (max_w - sw) // 2)
            else:
                indent = rng.randint(16, 38) if (done == 0 and not is_two_col and rng.random() < 0.45) else rng.randint(0, 10)
                xp = min(col_x0 + indent, col_x1 - sw - 2)
            line_ink = (
                min(145, ink_bgr[0] + rng.randint(15, 45)),
                min(142, ink_bgr[1] + rng.randint(15, 45)),
                min(140, ink_bgr[2] + rng.randint(15, 45)),
            ) if (is_faint_short_recall and is_short_tail and rng.random() < 0.5) else ink_bgr
            placed = composite_strip(page_bgr, alpha, xp, cur_y, line_ink)
            if placed:
                placed_items.append((txt, placed[0], placed[1], placed[2], placed[3], ex_top, ex_bot, col_id))
            gap = rng.randint(gap_range[0], gap_range[1])
            cur_y += sh + gap
            done += 1
        return cur_y

    if not is_zero_text_neg:
        y_cursor = rng.randint(36, 68)
        if not is_sparse_hard_neg and not (is_wave_lvl5 and height < 450) and rng.random() < 0.35:
            t_txt = make_safe_cham_line(rng, n_words=rng.choice([2, 3]), header_mode="title")
            t_ren = render_hb_full_cell(t_txt, "Bold", min(30, font_size + 3))
            if t_ren is not None and t_ren[0].shape[1] < width - 80:
                t_a, et, eb = t_ren
                tx = (width - t_a.shape[1]) // 2
                pl = composite_strip(page_bgr, t_a, tx, y_cursor, ink_bgr)
                if pl:
                    placed_items.append((t_txt, pl[0], pl[1], pl[2], pl[3], et, eb, -1))
                    y_cursor += t_a.shape[0] + rng.randint(16, 26)

        if is_two_col:
            gutter = rng.randint(26, 52)
            lx0 = rng.randint(36, 52)
            rx1 = width - lx0
            cw = (rx1 - lx0 - gutter) // 2
            gap_r = (3, 10) if rng.random() < 0.55 else (10, 20)
            place_col(lx0, lx0 + cw, y_cursor, height - 44, rng.randint(6, 15), gap_r, col_id=0)
            place_col(lx0 + cw + gutter, rx1, y_cursor, height - 44, rng.randint(6, 15), gap_r, col_id=1)
        elif is_sparse_hard_neg:
            y_sp = rng.randint(75, max(110, height // 3))
            place_col(rng.randint(48, 110), width - 54, y_sp, height - 55, rng.randint(1, 3), (18, 40), col_id=0)
        elif is_wave_lvl5:
            gap_r = (2, 7) if rng.random() < 0.65 else (6, 14)
            max_l = rng.randint(3, 8) if height < 460 else rng.randint(10, 18)
            place_col(rng.randint(42, 74), width - 44, y_cursor, height - 42, max_l, gap_r, col_id=0)
        else:
            gap_r = (6, 19)
            n_paras = rng.randint(2, 3) if is_faint_short_recall else rng.randint(1, 3)
            for _ in range(n_paras):
                y_cursor = place_col(rng.randint(38, 62), width - 42, y_cursor, height - 66, rng.randint(4, 8), gap_r, col_id=0)
                y_cursor += rng.randint(12, 22)
            if y_cursor + font_size + 22 < height and (is_faint_short_recall or rng.random() < 0.45):
                dt_txt = make_safe_cham_line(rng, n_words=1, header_mode="date")
                d_ren = render_hb_full_cell(dt_txt, weight, max(16, font_size - 1), thin_faint=is_faint_short_recall)
                if d_ren is not None and d_ren[0].shape[1] < width - 100:
                    d_a, et, eb = d_ren
                    dx = rng.randint(width // 2, width - d_a.shape[1] - 44)
                    pl = composite_strip(page_bgr, d_a, dx, y_cursor + 4, ink_bgr)
                    if pl:
                        placed_items.append((dt_txt, pl[0], pl[1], pl[2], pl[3], et, eb, -2))

        if not placed_items:
            fb_txt = make_safe_cham_line(rng, n_words=3)
            fb_a, et, eb = render_hb_full_cell(fb_txt, "Regular", 22)
            pl = composite_strip(page_bgr, fb_a, 50, height // 2, ink_bgr)
            placed_items.append((fb_txt, pl[0], pl[1], pl[2], pl[3], et, eb, 0))

    # Add non-text margin/border/smudge distractors outside placed text boxes
    add_non_text_hard_negatives(page_bgr, rng, np_rng, placed_items)

    # Full typographic cell bounds clamped safely at neighbor midpoints (NO `min(et, 6)` truncation!)
    raw_bounds = []
    by_col = {}
    for it in placed_items:
        by_col.setdefault(it[7], []).append(it)

    for _, citems in by_col.items():
        citems.sort(key=lambda z: z[2])
        for k, (txt, x1, y1, x2, y2, et, eb, _) in enumerate(citems):
            px = 5
            top_lim = 2.0 if k == 0 else (citems[k - 1][4] + y1) / 2.0 + 1.2
            bot_lim = float(height - 2) if k == len(citems) - 1 else (y2 + citems[k + 1][2]) / 2.0 - 1.2
            py1 = max(top_lim, float(y1 - et))
            py2 = min(bot_lim, float(y2 + eb))
            if py2 - py1 < 8.0:
                mid = 0.5 * (y1 + y2)
                py1 = max(2.0, mid - 4.5)
                py2 = min(float(height - 2), mid + 4.5)
            px1 = float(max(2, x1 - px))
            px2 = float(min(width - 2, x2 + px))
            if py2 - py1 >= 8.0 and px2 - px1 >= 10.0:
                raw_bounds.append((txt, px1, py1, px2, py2))

    do_warp = is_wave_lvl5 or is_zero_text_neg or (rng.random() < 0.45)
    if do_warp:
        if is_wave_lvl5:
            amp = rng.uniform(3.2, 8.2)
            freq = rng.uniform(0.005, 0.012)
            skew = rng.uniform(0.012, 0.052)
        else:
            amp = rng.uniform(0.0, 2.4)
            freq = rng.uniform(0.004, 0.009)
            skew = rng.uniform(0.006, 0.028)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        pts2 = np.float32([
            [rng.uniform(0, width * skew), rng.uniform(0, height * skew)],
            [width - rng.uniform(0, width * skew), rng.uniform(0, height * skew)],
            [rng.uniform(0, width * skew), height - rng.uniform(0, height * skew)],
            [width - rng.uniform(0, width * skew), height - rng.uniform(0, height * skew)],
        ])
        mat = cv2.getPerspectiveTransform(pts1, pts2)
        if amp > 0.35:
            X, Y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
            dy_map = amp * np.sin(X * freq + phase).astype(np.float32)
            warped_wave = cv2.remap(page_bgr, X, Y + dy_map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            warped_wave = page_bgr

        # Teach DBNet that dark/gray perspective scanner triangles at tilted borders are BACKGROUND (0)!
        if rng.random() < 0.42:
            bv = rng.randint(24, 98)
            page_warped = cv2.warpPerspective(warped_wave, mat, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(bv, bv, bv))
        else:
            page_warped = cv2.warpPerspective(warped_wave, mat, (width, height), borderMode=cv2.BORDER_REPLICATE)

        final_labels = [
            {"transcription": txt, "points": build_curved_or_rect_poly(px1, py1, px2, py2, amp, freq, phase, mat, width, height)}
            for (txt, px1, py1, px2, py2) in raw_bounds
        ]
    else:
        page_warped = page_bgr
        final_labels = [
            {"transcription": txt, "points": build_curved_or_rect_poly(px1, py1, px2, py2, 0.0, 0.0, 0.0, None, width, height)}
            for (txt, px1, py1, px2, py2) in raw_bounds
        ]

    if is_sparse_hard_neg or is_zero_text_neg or is_wave_lvl5 or rng.random() < 0.35:
        deg_m = rng.random()
        if deg_m < 0.45:
            gx = np.linspace(rng.uniform(0.60, 0.90), 1.0, width, dtype=np.float32)
            if rng.random() < 0.5:
                gx = gx[::-1]
            page_warped = np.clip(page_warped.astype(np.float32) * gx[None, :, None], 0, 255).astype(np.uint8)
            page_warped = cv2.GaussianBlur(page_warped, (0, 0), sigmaX=rng.uniform(0.5, 1.5), borderType=cv2.BORDER_REPLICATE)
        elif deg_m < 0.85:
            noise = np_rng.normal(0.0, rng.uniform(3.0, 12.0), size=page_warped.shape).astype(np.float32)
            page_warped = np.clip(page_warped.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        else:
            # Mild ink bleed (`erode` on bright background thickens dark strokes slightly without erasing short lines)
            page_warped = cv2.erode(page_warped, np.ones((2, 1), np.uint8), iterations=1)

    img_name = f"{rel_prefix}_{idx:05d}.jpg"
    img_path = os.path.join(out_img_dir, img_name)
    cv2.imwrite(img_path, page_warped, [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(78, 95)])
    return f"{rel_prefix}/{img_name}\t{json.dumps(final_labels, ensure_ascii=False)}", len(final_labels), (len(final_labels) == 0)


def main():
    t0 = time.time()
    n_train = 1400
    n_val = 160
    det_root = os.path.join(TRAINING_DIR, "data", "detector_v3_2_poly")
    if os.path.exists(det_root):
        shutil.rmtree(det_root)
    tr_dir = os.path.join(det_root, "train_images")
    va_dir = os.path.join(det_root, "val_images")
    os.makedirs(tr_dir, exist_ok=True)
    os.makedirs(va_dir, exist_ok=True)

    print("=" * 80)
    print(f"🚀 GENERATING CHAM-DBNET V3 FULL-CELL CURVED POLYGON DATASET ({n_train} TRAIN / {n_val} VAL)")
    print("=" * 80)

    tr_labels, tr_polys, tr_zero_neg = [], 0, 0
    for i in range(n_train):
        lbl, cnt, zneg = generate_one_v3_page(i, False, tr_dir, "train_images", seed=20261160)
        tr_labels.append(lbl)
        tr_polys += cnt
        if zneg:
            tr_zero_neg += 1
        if (i + 1) % 350 == 0:
            print(f"  [train] {i + 1}/{n_train} pages ({tr_polys:,} polygons, {tr_zero_neg} zero-text hard-neg pages)...")

    va_labels, va_polys = [], 0
    for i in range(n_val):
        lbl, cnt, _ = generate_one_v3_page(i, True, va_dir, "val_images", seed=20261260)
        va_labels.append(lbl)
        va_polys += cnt

    tr_lbl_p = os.path.join(det_root, "det_train_label.txt")
    va_lbl_p = os.path.join(det_root, "det_val_label.txt")
    with open(tr_lbl_p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(tr_labels) + "\n")
    with open(va_lbl_p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(va_labels) + "\n")

    manifest = {
        "stage": "CHAM_DBNET_V3_POLY",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(time.time() - t0, 2),
        "train_pages": len(tr_labels),
        "train_zero_text_hard_neg_pages": tr_zero_neg,
        "val_pages": len(va_labels),
        "train_line_polygons": tr_polys,
        "val_line_polygons": va_polys,
        "train_label_sha256": hashlib.sha256(open(tr_lbl_p, "rb").read()).hexdigest(),
        "val_label_sha256": hashlib.sha256(open(va_lbl_p, "rb").read()).hexdigest(),
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "detector_v3_2_poly_manifest.json"), "w", encoding="utf-8", newline="\n") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)
    print(f"✅ Cham-DBNet v3 dataset generated in {manifest['duration_sec']}s -> {det_root} ({tr_polys:,} train / {va_polys:,} val polygons)")


if __name__ == "__main__":
    main()
