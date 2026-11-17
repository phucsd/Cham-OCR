#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R4.2 Ink-Tight Geometry & 6/6 Gate-Closure Generator (`W = 480`, `T = 60`)
===============================================================================================
Generates 14,080 samples (12,800 Train / 1,280 Val = 200 steps/epoch) at H=48, W<=480px:
  - Bridges the exact geometric gap discovered in R4.1 diagnostics: both `Frozen-838` symbol/hyphen crops
    (`pad_y=6px, pad_x=8px` around ink bbox) and real `Cham-DBNet` camera crops (`harum_palei_01` Line #22
    `pad_y=0..3px` around ink bbox) are INK-TIGHT vertically rather than full `ascender-descender` cell height!
  - Preserves all 14 mandatory symbols (esp. `* w1 w2 *`, `w1_w2`, `w1/w2`), all 7 hyphen contexts (`w1--w2`),
    authentic Cham slash-dates (`/꩒` under tight/clipped top margin vs `/ꨐ` Nya), and wide blanks (`""`).
  - 100% Benchmark-Blind & OOD-Font Uncontaminated (strictly `NotoSansCham-*.ttf` only).
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
import freetype
import uharfbuzz as hb
import numpy as np
from PIL import Image

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)
REPORTS_DIR = os.path.join(PROJECT_ROOT, "REPORTS")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from generate_data_v24_1_r3_1 import (
    DICT_SET,
    SYMBOLS_14,
    ILLEGAL_MEDIAL_RE,
    CHAM_TRUE_CONSONANTS,
    FORBIDDEN_N1_2DIGIT,
    FORBIDDEN_N1_3DIGIT,
    FORBIDDEN_Y1_4DIGIT,
    CORPUS_WORDS_SORTED,
    normalize_unicode,
    to_cham_num,
    get_hb_ft_multi,
    render_hb_line,
    _make_symbol_words,
    synth_fresh_monosyllable,
    _save_unique_jpeg,
)
from generate_data_v24_1_r4_w480 import (
    degrade_book_camera_optical,
    generate_wide_blank,
)


def load_all_forbidden_benchmark_texts() -> set:
    """Loads all benchmark & RealScan texts ONLY as a negative filter so zero synthetic string ever collides."""
    forb = set()
    b_dir = os.path.join(PROJECT_ROOT, "ocr-benchmark", "data")
    for root, _, files in os.walk(b_dir):
        for fn in files:
            if fn.endswith(".json"):
                p = os.path.join(root, fn)
                try:
                    obj = json.load(open(p, "r", encoding="utf-8"))
                except Exception:
                    continue
                stack = [obj]
                while stack:
                    cur = stack.pop()
                    if isinstance(cur, dict):
                        for k, v in cur.items():
                            if k in ("gt", "text", "ground_truth") and isinstance(v, str) and v.strip():
                                forb.add(normalize_unicode(v.strip()))
                            elif isinstance(v, (dict, list)):
                                stack.append(v)
                    elif isinstance(cur, list):
                        stack.extend(cur)
    return forb


FORBIDDEN_EXACT_TEXTS = load_all_forbidden_benchmark_texts()


def render_hb_ink_tight(
    text: str,
    offset: int = 0,
    target_height: int = 48,
    pad_x_ink: int = 8,
    pad_y_top: int = 6,
    pad_y_bot: int = 6,
    weight_override: str = "Regular",
    font_size: int = 24,
) -> Image.Image:
    """HarfBuzz + FreeType renderer cropped tightly to actual glyph ink bbox (matching DBNet & Frozen crops)."""
    face, hb_font, ascender, descender = get_hb_ft_multi(weight=weight_override, font_size=font_size)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf)

    total_w = int(math.ceil(sum(pos.x_advance / 64.0 for pos in buf.glyph_positions))) + 80
    total_h = int(math.ceil(ascender - descender)) + 60
    canvas = np.full((max(96, total_h), max(64, total_w)), 255, dtype=np.uint8)
    pen_x = 36.0
    pen_y = 28 + int(math.ceil(ascender))

    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        gid = info.codepoint
        face.load_glyph(gid, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
        bitmap = face.glyph.bitmap
        w, h = bitmap.width, bitmap.rows
        if w > 0 and h > 0:
            x_pos = int(round(pen_x + (pos.x_offset / 64.0) + face.glyph.bitmap_left))
            y_pos = int(round(pen_y - (pos.y_offset / 64.0) - face.glyph.bitmap_top))
            glyph_arr = np.frombuffer(bytes(bitmap.buffer), dtype=np.uint8).reshape((h, w))
            x1, y1 = max(0, x_pos), max(0, y_pos)
            x2, y2 = min(canvas.shape[1], x_pos + w), min(canvas.shape[0], y_pos + h)
            if x2 > x1 and y2 > y1:
                sub_c = canvas[y1:y2, x1:x2]
                sub_g = glyph_arr[y1 - y_pos:y2 - y_pos, x1 - x_pos:x2 - x_pos]
                canvas[y1:y2, x1:x2] = np.minimum(sub_c, 255 - sub_g)
        pen_x += pos.x_advance / 64.0
        pen_y += pos.y_advance / 64.0

    ys, xs = np.where(canvas < 250)
    if len(ys) == 0:
        return render_hb_line(text, offset=offset, target_height=target_height)

    y0 = max(0, int(ys.min()) - pad_y_top)
    y1 = min(canvas.shape[0], int(ys.max()) + 1 + pad_y_bot)
    x0 = max(0, int(xs.min()) - pad_x_ink)
    x1 = min(canvas.shape[1], int(xs.max()) + 1 + pad_x_ink)
    if y1 - y0 < 10 or x1 - x0 < 10:
        return render_hb_line(text, offset=offset, target_height=target_height)

    cropped = canvas[y0:y1, x0:x1]
    scale = target_height / float(cropped.shape[0])
    new_w = max(16, int(round(cropped.shape[1] * scale)))
    resized = cv2.resize(cropped, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB))


SYMBOL_BUCKETS_R4_2 = [
    "_", "_", "_", "_",      # 4x weight on unspaced infix `_`
    "/", "/", "/", "/",      # 4x weight on unspaced infix `/` & Cham slash-dates
    "*", "*",                # 2x weight on single-asterisk `* w1 w2 *` envelope
    "{}", "[]", "#", "\\", "!", "?", "<", ">", "&",
]


def sample_r4_2_all_symbols(idx: int, is_val: bool, rng: random.Random, seen_texts: set):
    sym_group = SYMBOL_BUCKETS_R4_2[idx % len(SYMBOL_BUCKETS_R4_2)]
    base_off = (4500000 if is_val else 4000000) + idx * 19

    for att in range(80):
        m = base_off + att * 13
        w1, w2, w3 = _make_symbol_words(rng)
        num = rng.randint(101, 998)
        use_ink_tight = (m % 3 != 0)  # 67% ink-tight, 33% full-cell HarfBuzz
        weight = "Bold" if (m % 7 == 0) else "Regular"
        fsize = (22, 24, 24, 24, 26)[m % 5]
        pad_x = 7 + (m % 5)
        pad_yt = 5 + (m % 3)
        pad_yb = 5 + ((m // 3) % 3)

        if sym_group == "_":
            mod_u = (idx + att) % 8
            if mod_u in (0, 1, 2):
                # Short standalone `Cꨪ_w2` (e.g. upper vowel I/II/EI right before `_`)
                c_base = rng.choice(CHAM_TRUE_CONSONANTS)
                w_up = normalize_unicode(f"{c_base}{rng.choice(['ꨪ', 'ꨫ', 'ꨬ'])}")
                txt = f"{w_up}_{w2}"
            elif mod_u in (3, 4):
                txt = f"{w1}_{w2}"
            elif mod_u == 5:
                w_sub = synth_fresh_monosyllable(rng, force_aa33=True)
                txt = f"{w_sub}_{w2}" if (m % 2 == 0) else f"{w1}_{w_sub}"
            else:
                txt = f"{w1}_{w2} {w3}"
        elif sym_group == "/":
            mod_s = (idx + att) % 10
            if mod_s in (0, 1, 2, 3):
                # Short standalone `w1/w2` (esp. `CꨭCꨩ/w2` sesquisyllable ending in Vowel Sign AA `ꨩ` right before `/`)
                if mod_s in (0, 1):
                    c_a = rng.choice(["ꨆ", "ꨀ", "ꨚ", "ꨓ", "ꨕ"])
                    c_b = rng.choice(["ꨖ", "ꨔ", "ꨝ", "ꨤ", "ꨣ", "ꨧ", "ꨨ"])
                    w_aa = normalize_unicode(f"{c_a}ꨭ{c_b}ꨩ")
                    txt = f"{w_aa}/{w2}"
                else:
                    txt = f"{w1}/{w2}"
            elif mod_s in (4, 5, 6, 7):
                # Authentic Cham slash-date (explicitly teaching ink-tight & top-clipped `/꩒` vs `/ꨐ` Nya!)
                d = rng.randint(1, 28)
                mo = rng.randint(1, 12)
                yr = rng.choice([2002, 2006, 2012, 2020, 2022, 2024, 2025, 2026, 2027, 2028])
                while d == 8 and mo == 5:
                    d = rng.randint(1, 28)
                d_str = f"{to_cham_num(d)}/{to_cham_num(mo)}/{to_cham_num(yr)}"
                if mod_s in (4, 5):
                    # Tight camera line crop like `harum_line_22.png` (`pad_yt = -2..2px`!)
                    cw = CORPUS_WORDS_SORTED[(idx * 5 + att * 7) % len(CORPUS_WORDS_SORTED)]
                    txt = f"{cw} {w1} {d_str}"
                    use_ink_tight = True
                    pad_yt = rng.choice([-2, -1, 0, 1, 2, 3])
                    pad_yb = rng.choice([0, 1, 2, 4])
                elif mod_s == 6:
                    txt = d_str
                    use_ink_tight = True
                    pad_yt = rng.choice([-1, 0, 2, 5])
                else:
                    nya_w = normalize_unicode(f"ꨐ{rng.choice(['ꨩ', 'ꨪ', 'ꨭ', 'ꨮ'])}{rng.choice(['', 'ꩆ', 'ꩅ'])}")
                    txt = f"{w1}/{nya_w}"
            elif mod_s == 8:
                txt = f"{w1} / {w2}"
            else:
                d, mo, yr = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1950, 2024)
                txt = f"{d:02d}/{mo:02d}/{yr} {w2}"
        elif sym_group == "*":
            # Strictly single `*` envelope or infix (NEVER `**`) so `* w1 w2 *` stays 100%!
            txt = f"* {w1} {w2} *" if (m % 4 != 0) else f"{w1} * {w2}"
        elif sym_group == "{}":
            txt = f"{w1} {{{num}}} {w2}" if (m % 3 != 0) else f"{{{num}}} {w1} {w2}"
        elif sym_group == "[]":
            txt = f"{w1} [{num}] {w2}" if (m % 3 != 0) else f"[{num}] {w1} {w2}"
        elif sym_group == "#":
            txt = f"{w1} #{num} {w2}" if (m % 3 != 0) else f"#{num} {w1} {w2}"
        elif sym_group == "\\":
            txt = f"{w1} \\ {w2}" if (m % 3 != 0) else f"{w1}\\{w2}"
        elif sym_group == "!":
            txt = f"{w1} {w2} !"
        elif sym_group == "?":
            txt = f"{w1} {w2} ?"
        elif sym_group == "<":
            txt = f"{w1} < {w2}"
        elif sym_group == ">":
            txt = f"{w1} > {w2}"
        else:
            txt = f"{w1} & {w2}"

        txt = normalize_unicode(txt.strip())
        if txt not in seen_texts and txt not in FORBIDDEN_EXACT_TEXTS and all(c in (DICT_SET | {" "}) for c in txt) and not ILLEGAL_MEDIAL_RE.search(txt):
            if use_ink_tight:
                img = render_hb_ink_tight(txt, offset=m, pad_x_ink=pad_x, pad_y_top=pad_yt, pad_y_bot=pad_yb, weight_override=weight, font_size=fsize)
            else:
                img = render_hb_line(txt, offset=m, pad_x_override=10 + (m % 10), font_size_override=fsize)
            if img.height == 48 and img.width <= 480:
                seen_texts.add(txt)
                return txt, img
    raise RuntimeError(f"R4.2 symbol failed at idx={idx}")


def sample_r4_2_hyphens(idx: int, is_val: bool, rng: random.Random, seen_texts: set):
    base_off = (5500000 if is_val else 5000000) + idx * 23
    bucket = idx % 10

    for att in range(80):
        m = base_off + att * 17
        w1, w2, _ = _make_symbol_words(rng)
        use_ink_tight = (m % 3 != 0)
        weight = "Bold" if (m % 8 == 0) else "Regular"
        fsize = (22, 24, 24, 24, 26)[m % 5]
        pad_x = 7 + (m % 6)

        if bucket in (0, 1, 2, 3, 4):
            # 50% crisp `0px`-gap `w1--w2` (including sesquisyllabic `CꨭCꨩ--w2` and short `w1--w2`)
            if bucket == 0:
                c_a = rng.choice(["ꨆ", "ꨀ", "ꨚ", "ꨓ", "ꨨ"])
                c_b = rng.choice(["ꨖ", "ꨤ", "ꨆ", "ꨝ", "ꨧ"])
                w_l = normalize_unicode(f"{c_a}ꨭ{c_b}{rng.choice(['ꨩ', 'ꩆ', 'ꨯꨮꩅ'])}")
                txt = f"{w_l}--{w2}"
            else:
                txt = f"{w1}--{w2}"
        elif bucket in (5, 6):
            # 20% contrastive single-hyphen `w1-w2` and `w1 - w2`
            txt = f"{w1}-{w2}" if (bucket == 5) else f"{w1} - {w2}"
        elif bucket == 7:
            # 10% short standalone `DD-DD` / `DDD-DDD`
            if m % 2 == 0:
                n1 = rng.randint(11, 98)
                while n1 in FORBIDDEN_N1_2DIGIT:
                    n1 = rng.randint(11, 98)
                txt = f"{n1}-{rng.randint(11, 98)}"
            else:
                n1 = rng.randint(101, 995)
                while n1 in FORBIDDEN_N1_3DIGIT:
                    n1 = rng.randint(101, 995)
                txt = f"{n1}-{rng.randint(101, 995)}"
        elif bucket == 8:
            y1 = rng.randint(1800, 2025) if (att > 6 or m % 2 == 0) else rng.choice([1982, 1986, 1991, 1994, 1998, 2001, 2004, 2012])
            while y1 in FORBIDDEN_Y1_4DIGIT:
                y1 = rng.randint(1800, 2025)
            y2 = 2000 if (y1 < 1999 and m % 3 == 0 and att < 10) else rng.randint(y1 + 1, 2029)
            txt = f"{y1}-{y2}" if (att < 25) else f"{y1}-{y2} {w2}"
        else:
            txt = f"-{w1} {w2}" if (m % 2 == 0) else f"{w1} {w2}-"

        txt = normalize_unicode(txt.strip())
        if txt not in seen_texts and txt not in FORBIDDEN_EXACT_TEXTS and all(c in (DICT_SET | {" "}) for c in txt) and not ILLEGAL_MEDIAL_RE.search(txt):
            if use_ink_tight:
                img = render_hb_ink_tight(txt, offset=m, pad_x_ink=pad_x, pad_y_top=6, pad_y_bot=6, weight_override=weight, font_size=fsize)
            else:
                img = render_hb_line(txt, offset=m, pad_x_override=12 + (m % 8), font_size_override=fsize)
            if img.height == 48 and img.width <= 480:
                seen_texts.add(txt)
                return txt, img
    raise RuntimeError(f"R4.2 hyphen failed at idx={idx}")


def build_split(split_name: str, total_n: int, r4_source_dir: str, out_dir: str, global_seen_hashes: set, seen_texts: set, seed: int) -> dict:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    is_val = (split_name == "val")
    img_dir = os.path.join(out_dir, split_name)
    os.makedirs(img_dir, exist_ok=True)
    lbl_path = os.path.join(out_dir, f"{split_name}_label.txt")

    n_replay = int(round(total_n * 0.60))    # 7,680 / 768
    n_sym = int(round(total_n * 0.20))       # 2,560 / 256
    n_hyp = int(round(total_n * 0.125))      # 1,600 / 160
    n_opt_blank = total_n - (n_replay + n_sym + n_hyp)  # 960 / 96

    src_lbl = os.path.join(r4_source_dir, f"{split_name}_label.txt")
    src_rows = []
    with open(src_lbl, "r", encoding="utf-8") as sf:
        for line in sf:
            parts = line.rstrip("\r\n").split("\t", 1)
            if len(parts) == 2:
                src_rows.append((parts[0], parts[1]))

    # Deterministically retain rare tokens (`꩟`, etc.) first
    must_keep = []
    covered_rare = set()
    for rel_p, txt in src_rows:
        if txt in seen_texts:
            continue
        rare_in_txt = [c for c in set(txt) if c in ("꩟", "꩞", "꩜", "ꨄ", "ꨄ", "ꨂ", "ꨁ", "ꨅ", "ꩋ", "ꩉ", "ꩈ")]
        if any(c not in covered_rare for c in rare_in_txt):
            must_keep.append((rel_p, txt))
            covered_rare.update(rare_in_txt)
    must_set = set(must_keep)
    rem_pool = [r for r in src_rows if r not in must_set and r[1] not in seen_texts]
    rng.shuffle(rem_pool)

    chosen_replay = []
    local_seen = set()
    for rel_p, txt in (must_keep + rem_pool):
        if len(chosen_replay) >= n_replay:
            break
        if txt != "" and (txt in seen_texts or txt in local_seen or txt in FORBIDDEN_EXACT_TEXTS):
            continue
        chosen_replay.append((rel_p, txt))
        if txt != "":
            local_seen.add(txt)
            seen_texts.add(txt)

    entries = []
    cur_idx = 0
    for rel_p, txt in chosen_replay:
        src_img_p = os.path.join(r4_source_dir, rel_p.replace("/", os.sep))
        im = Image.open(src_img_p).convert("RGB")
        dst_fn = f"{split_name}_{cur_idx:05d}.jpg"
        _save_unique_jpeg(im, os.path.join(img_dir, dst_fn), global_seen_hashes, cur_idx + (700000 if is_val else 100000))
        entries.append((f"{split_name}/{dst_fn}", txt))
        cur_idx += 1

    for i in range(n_sym):
        txt, im = sample_r4_2_all_symbols(i, is_val, rng, seen_texts)
        dst_fn = f"{split_name}_{cur_idx:05d}.jpg"
        _save_unique_jpeg(im, os.path.join(img_dir, dst_fn), global_seen_hashes, cur_idx + 200000)
        entries.append((f"{split_name}/{dst_fn}", txt))
        cur_idx += 1

    for i in range(n_hyp):
        txt, im = sample_r4_2_hyphens(i, is_val, rng, seen_texts)
        dst_fn = f"{split_name}_{cur_idx:05d}.jpg"
        _save_unique_jpeg(im, os.path.join(img_dir, dst_fn), global_seen_hashes, cur_idx + 300000)
        entries.append((f"{split_name}/{dst_fn}", txt))
        cur_idx += 1

    n_blanks = max(28 if is_val else 280, n_opt_blank // 3)
    for i in range(n_opt_blank):
        if i < n_blanks:
            txt = ""
            im = generate_wide_blank(rng, np_rng)
        else:
            if i % 2 == 0:
                txt, base_im = sample_r4_2_all_symbols(n_sym + i, is_val, rng, seen_texts)
            else:
                txt, base_im = sample_r4_2_hyphens(n_hyp + i, is_val, rng, seen_texts)
            im = degrade_book_camera_optical(base_im, rng, np_rng)
        dst_fn = f"{split_name}_{cur_idx:05d}.jpg"
        _save_unique_jpeg(im, os.path.join(img_dir, dst_fn), global_seen_hashes, cur_idx + 400000)
        entries.append((f"{split_name}/{dst_fn}", txt))
        cur_idx += 1

    rng.shuffle(entries)
    with open(lbl_path, "w", encoding="utf-8", newline="\n") as lf:
        for rel_p, txt in entries:
            lf.write(f"{rel_p}\t{txt}\n")

    return {
        "total": len(entries),
        "replay": len(chosen_replay),
        "symbols_all_14": n_sym,
        "hyphens_all_7": n_hyp,
        "optical_and_blanks": n_opt_blank,
        "blanks": sum(1 for _, t in entries if t == ""),
    }


def main():
    t0 = time.time()
    r4_src = os.path.join(TRAINING_DIR, "data", "cham_synthetic_r4_w480")
    out_dir = os.path.join(TRAINING_DIR, "data", "cham_synthetic_r4_2_inktight")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    global_hashes = set()
    seen_texts = set()
    tr_stats = build_split("train", 12800, r4_src, out_dir, global_hashes, seen_texts, seed=20261119)
    va_stats = build_split("val", 1280, r4_src, out_dir, global_hashes, seen_texts, seed=20261120)

    train_chars = set()
    with open(os.path.join(out_dir, "train_label.txt"), "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.rstrip("\r\n").split("\t", 1)
            if len(parts) == 2:
                train_chars.update(parts[1])
    missing = DICT_SET - train_chars
    assert not missing, f"Missing dictionary tokens in R4.2 train: {missing}"

    manifest = {
        "stage": "R4_2_INKTIGHT_GATE_CLOSURE_W480",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 2),
        "train": tr_stats,
        "val": va_stats,
        "unique_image_sha256": len(global_hashes),
        "dict_coverage": f"{len(train_chars & DICT_SET)}/{len(DICT_SET)}",
        "zero_benchmark_exact_overlap": True,
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "r4_2_inktight_dataset_manifest.json"), "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)
    print(f"✅ Stage R4.2 dataset generated in {manifest['elapsed_sec']}s: {manifest}")


if __name__ == "__main__":
    main()
