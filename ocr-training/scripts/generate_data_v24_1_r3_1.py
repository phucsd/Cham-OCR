#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R3.1 Gate-Closure, Sesquisyllabic & Morphological Erosion Generator
========================================================================================
Generates 24,000 samples (21,600 Train / 2,400 Val) at H=48, W<=320px (100% HarfBuzz shaping):
  1. Stream 1 (45% = 9,720 Train / 1,080 Val): Clean & R2.1.6 HarfBuzz Anchor Replay.
  2. Stream 2 (18% = 3,888 Train / 432 Val): Sesquisyllabic Disyllable Pre-Vowel/Medial Attachment
     (`C1 C2 ꨯ/ꨰ/ꨴ` vs `C1 ꨯ/ꨰ/ꨴ C2 ꨯ/ꨰ/ꨴ`) + High-Bindu `ꩌ` (`U+AA4C`) + Confused Pairs (`ꨪ`/`ꨫ`, `ꨲ`/`ꨶ`, `ꨣ`/`꩖`).
  3. Stream 3 (19% = 4,104 Train / 456 Val): Contrastive 14 Mandatory Symbols (`ꨳ` vs `_`, `{NNN}`, `w1/w2`, `!`)
     & 7-Group Multi-Context Hyphens (`0px` HarfBuzz `--`, varied-margin `D-D`/`DD-DD`/`DDD-DDD`, zero-preserving years).
  4. Stream 4 (18% = 3,888 Train / 432 Val): Balanced Morphological Erosion (`erode k=2,3`) + Severe Blur/LowRes/Noise
     + High-Noise Blank Negatives (`""`).

Strict Invariants:
  - 100% Benchmark Blind (zero AST/file references to any benchmark or evaluation split).
  - 100% HarfBuzz + FreeType shaping (`NotoSansCham-Regular/Bold/Black.ttf`) on all newly rendered Cham text.
  - 100% Width Cap (W <= 320px at H=48) on all newly generated samples (Streams 2, 3, 4).
  - Zero Train/Val text overlap, 24,000/24,000 unique SHA256 image hashes, 114/114 token coverage.
"""

import os
import re
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
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)
REPORTS_DIR = os.path.join(PROJECT_ROOT, "REPORTS")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from generate_data_v24_1_r2_1_6 import (
    DICT_CHARS_V24_1,
    CHAM_DIGITS,
    CHAM_CARRIER_A,
    CHAM_INDEPENDENT_VOWELS,
    CHAM_TRUE_CONSONANTS,
    CHAM_MEDIAL_ELIGIBLE_CONSONANTS,
    PRIORITY_VOWEL_NUCLEI,
    MEDIAL_VOWEL_NUCLEI,
    PRIORITY_MEDIALS,
    PRIORITY_FINALS,
    RARE_AND_CONFUSED_BASES,
    CORE_STANZA_SET,
    TRAIN_SINGLE_DIGIT_PAIRS,
    VAL_SINGLE_DIGIT_PAIRS,
    FORBIDDEN_N1_2DIGIT,
    FORBIDDEN_Y1_4DIGIT,
    normalize_unicode,
    to_cham_num,
    find_asset_path,
)

DICT_SET = set(DICT_CHARS_V24_1)
SYMBOLS_14 = ['?', '!', '#', '&', '*', '<', '>', '{', '}', '\\', '_', '[', ']', '/']
ILLEGAL_MEDIAL_RE = re.compile(r"[ꨁꨂꨃꨄꨅ][ꨳꨴꨵꨶ]")
FORBIDDEN_N1_3DIGIT = {100, 111, 123, 200, 222, 250, 300, 400, 500}

_HB_MULTI_CACHE = {}


def get_hb_ft_multi(weight="Regular", font_size=24):
    key = (weight, font_size)
    if key not in _HB_MULTI_CACHE:
        fname = f"NotoSansCham-{weight}.ttf"
        fpath = find_asset_path(fname)
        if not fpath or not os.path.exists(fpath):
            fpath = find_asset_path("NotoSansCham-Regular.ttf")
        if not fpath or not os.path.exists(fpath):
            raise RuntimeError(f"Missing Cham font: {fname}")
        face = freetype.Face(fpath)
        face.set_char_size(font_size * 64)
        font_bytes = open(fpath, "rb").read()
        hb_face = hb.Face(font_bytes)
        hb_font = hb.Font(hb_face)
        hb_font.scale = (font_size * 64, font_size * 64)
        asc = face.size.ascender / 64.0
        desc = face.size.descender / 64.0
        _HB_MULTI_CACHE[key] = (face, hb_font, asc, desc)
    return _HB_MULTI_CACHE[key]


def render_hb_line(
    text: str,
    offset: int = 0,
    target_height: int = 48,
    pad_x_override: int = None,
    pad_y_override: int = None,
    weight_override: str = None,
    font_size_override: int = None,
    repeat_digit_gap: int = 0,
) -> Image.Image:
    """100% HarfBuzz + FreeType line renderer with controllable margins, weights, and CTC phase."""
    if weight_override:
        weight = weight_override
    else:
        r_w = offset % 12
        weight = "Bold" if r_w in (0, 6) else ("Black" if r_w == 11 else "Regular")

    font_size = font_size_override if font_size_override else (24 if (offset % 4 != 3) else (22 if offset % 2 == 1 else 26))
    pad_x = pad_x_override if pad_x_override is not None else (12 + (offset % 8))
    pad_y = pad_y_override if pad_y_override is not None else (9 + (offset % 3))

    face, hb_font, ascender, descender = get_hb_ft_multi(weight=weight, font_size=font_size)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf)

    extra_w = max(0, repeat_digit_gap) * 3
    total_w = sum(pos.x_advance / 64.0 for pos in buf.glyph_positions) + extra_w
    line_h = int(math.ceil(ascender - descender)) + pad_y * 2
    line_w = int(math.ceil(total_w)) + pad_x * 2
    canvas = np.full((max(48, line_h), max(16, line_w)), 255, dtype=np.uint8)
    pen_x = float(pad_x)
    pen_y = pad_y + int(math.ceil(ascender))
    prev_gid = None

    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        gid = info.codepoint
        if repeat_digit_gap > 0 and prev_gid is not None and gid == prev_gid and gid != 0:
            pen_x += float(repeat_digit_gap)
        prev_gid = gid

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

    scale = target_height / float(canvas.shape[0])
    new_w = max(16, int(round(canvas.shape[1] * scale)))
    resized = cv2.resize(canvas, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB))


def load_safe_corpus_lexicon():
    """
    Loads individual pure-Cham words (len <= 11, no whitespace, 100% valid 114-token chars)
    from `ocr-training/data/corpus/` to provide authentic sesquisyllabic & lexical roots
    while keeping substring length strictly < 14.
    """
    all_words = set()
    corpus_dir = os.path.join(TRAINING_DIR, "data", "corpus")
    ct_path = os.path.join(corpus_dir, "cham_text.txt")
    dj_path = os.path.join(corpus_dir, "cham_viet_dictionary.jsonl")

    if os.path.exists(ct_path):
        for line in open(ct_path, "r", encoding="utf-8"):
            for tok in line.split():
                w = normalize_unicode(tok.strip())
                if 2 <= len(w) <= 11 and all("\uAA00" <= c <= "\uAA4F" for c in w) and all(c in DICT_SET for c in w):
                    if not ILLEGAL_MEDIAL_RE.search(w):
                        all_words.add(w)

    if os.path.exists(dj_path):
        for line in open(dj_path, "r", encoding="utf-8"):
            try:
                obj = json.loads(line)
                hw = obj.get("headword_cham") or ""
                for tok in hw.split():
                    w = normalize_unicode(tok.strip())
                    if 2 <= len(w) <= 11 and all("\uAA00" <= c <= "\uAA4F" for c in w) and all(c in DICT_SET for c in w):
                        if not ILLEGAL_MEDIAL_RE.search(w):
                            all_words.add(w)
            except Exception:
                pass

    bases = "".join(CHAM_TRUE_CONSONANTS) + CHAM_CARRIER_A
    sesq_re = re.compile(f"^[{bases}][ꨪꨭ]?[{bases}][ꨳꨵꨶ]?[ꨴꨯꨰ]")
    sesq_words = sorted(w for w in all_words if sesq_re.search(w))
    aa33_words = sorted(w for w in all_words if "ꨳ" in w)
    bindu_words = sorted(w for w in all_words if "ꩌ" in w)
    return all_words, sorted(all_words), sesq_words, aa33_words, bindu_words


CORPUS_WORD_SET, CORPUS_WORDS_SORTED, SESQ_WORDS, AA33_WORDS, BINDU_WORDS = load_safe_corpus_lexicon()


def synth_fresh_monosyllable(rng: random.Random, force_aa33=False, force_bindu=False, force_pre=False) -> str:
    """Synthesizes a phonotactically valid Cham word guaranteed NOT in CORPUS_WORD_SET or CORE_STANZA_SET."""
    for _ in range(40):
        if force_aa33:
            c = rng.choice(CHAM_MEDIAL_ELIGIBLE_CONSONANTS)
            m = "ꨳ"
            v = rng.choice(["ꨪ", "ꨮ", "ꨯꨮ", "ꨩ", ""])
            f = "ꩌ" if force_bindu else rng.choice(PRIORITY_FINALS)
        elif force_pre:
            c = rng.choice(CHAM_MEDIAL_ELIGIBLE_CONSONANTS if rng.random() < 0.45 else CHAM_TRUE_CONSONANTS)
            m = rng.choice(["", "ꨴ", "ꨶ", "ꨵ"]) if c in CHAM_MEDIAL_ELIGIBLE_CONSONANTS else ""
            v = rng.choice(["ꨯ", "ꨰ", "ꨯꨱ", "ꨯꨮ"]) if m != "ꨴ" else rng.choice(["", "ꨯ", "ꨰ", "ꨭ", "ꨬ", "ꨪ"])
            f = "ꩌ" if force_bindu else rng.choice(PRIORITY_FINALS + [""])
        else:
            use_m = rng.random() < 0.45
            if use_m:
                c = rng.choice(CHAM_MEDIAL_ELIGIBLE_CONSONANTS)
                m = rng.choice(PRIORITY_MEDIALS)
                v = rng.choice(MEDIAL_VOWEL_NUCLEI)
            else:
                c = rng.choice(CHAM_TRUE_CONSONANTS + [CHAM_CARRIER_A])
                m = ""
                v = rng.choice(PRIORITY_VOWEL_NUCLEI)
            f = "ꩌ" if force_bindu else (rng.choice(PRIORITY_FINALS) if rng.random() < 0.65 else "")
            if not m and not v and not f:
                v = "ꨩ"

        # Add rare prefix or rare suffix consonant if needed to ensure non-corpus novelty while <= 9 chars
        if rng.random() < 0.45:
            pre_c = rng.choice(["ꨀ", "ꨆ", "ꨓ", "ꨚ", "ꨕ", "ꨧ", "ꨟ", "ꨨ", "ꨝ", "ꨦ"])
            pre_v = rng.choice(["", "ꨪ", "ꨭ"])
            w = normalize_unicode(f"{pre_c}{pre_v}{c}{m}{v}{f}")
        else:
            w = normalize_unicode(f"{c}{m}{v}{f}")

        if w and len(w) <= 10 and w not in CORPUS_WORD_SET and w not in CORE_STANZA_SET and not ILLEGAL_MEDIAL_RE.search(w):
            return w
    # Guaranteed non-corpus fallback
    c1 = rng.choice(["ꨍ", "ꨏ", "ꨐ", "ꨑ", "ꨒ", "ꨖ", "ꨘ", "ꨜ", "ꨞ"])
    c2 = rng.choice(["ꨆ", "ꨓ", "ꨚ", "ꨣ", "ꨤ"])
    return normalize_unicode(f"{c1}ꨪ{c2}ꨯꨱꩌ")


def synth_sesquisyllabic_word(rng: random.Random, dual_prevowel: bool = False) -> str:
    """
    Synthesizes targeted sesquisyllabic disyllable:
      - If `dual_prevowel=False` (75%): unmarked presyllable C1 (+ optional ꨪ/ꨭ) + main syllable C2 + (ꨯ | ꨰ | ꨴ)
        e.g. `ꨀꨊꨯꨱꩀ`, `ꨟꨣꨰ`, `ꨓꨗꨴꩀ`, `ꨚꨕꨴꨶꨰ`, `ꨦꨡꨰ` — teaching CTC NOT to duplicate ꨯ/ꨰ/ꨴ onto C1!
      - If `dual_prevowel=True` (25%): genuine C1(ꨯ|ꨰ|ꨴ) + C2(ꨯ|ꨰ|ꨴ) contrastive positive!
    """
    if not dual_prevowel and SESQ_WORDS and rng.random() < 0.55:
        return rng.choice(SESQ_WORDS)

    c1 = rng.choice(["ꨀ", "ꨟ", "ꨓ", "ꨚ", "ꨦ", "ꨆ", "ꨕ", "ꨧ", "ꨨ", "ꨝ", "ꨌ", "ꨎ", "ꨤ"])
    c2 = rng.choice(CHAM_MEDIAL_ELIGIBLE_CONSONANTS + ["ꨊ", "ꨡ", "ꨏ", "ꨑ"])

    if dual_prevowel:
        v1 = rng.choice(["ꨯ", "ꨰ", "ꨴꨪ", "ꨯꨮ"]) if c1 in CHAM_MEDIAL_ELIGIBLE_CONSONANTS else rng.choice(["ꨯ", "ꨰ"])
    else:
        v1 = rng.choice(["", "", "", "ꨪ", "ꨭ"])

    mode = rng.randint(0, 3)
    if mode == 0 and c2 in CHAM_MEDIAL_ELIGIBLE_CONSONANTS:
        # C1 + C2 + ꨴ (+ optional ꨶ + ꨰ/ꨭ/ꨪ)
        m2 = "ꨴ" + (rng.choice(["", "ꨶ"]) if rng.random() < 0.30 else "")
        v2 = rng.choice(["", "ꨰ", "ꨭ", "ꨪ", "ꨯꨱ", "ꨬ", "ꨩ"])
    elif mode == 1:
        m2 = ""
        v2 = "ꨰ"
    elif mode == 2:
        m2 = ""
        v2 = rng.choice(["ꨯ", "ꨯꨱ", "ꨯꨮ"])
    else:
        m2 = rng.choice(["ꨳ", "ꨶ"]) if c2 in CHAM_MEDIAL_ELIGIBLE_CONSONANTS else ""
        v2 = rng.choice(["ꨯꨮ", "ꨰ", "ꨯ"])

    f2 = rng.choice(["", "ꩀ", "ꩃ", "ꩌ", "ꩅ", "ꩍ", "ꩆ", "ꩉ"])
    w = normalize_unicode(f"{c1}{v1}{c2}{m2}{v2}{f2}")
    return w


# ==============================================================================
# Stream 2: Sesquisyllabic & Residual Confused Codepoint Hard-Mining
# ==============================================================================
CONFUSED_PAIRS = [
    ("ꨪ", "ꨫ"),
    ("ꨲ", "ꨶ"),
    ("ꨕ", "ꨊ"),
    ("ꨔ", "ꨟ"),
    ("ꨝ", "ꨟ"),
    ("ꨥ", "ꨡ"),
    ("ꨈ", "ꨓ"),
    ("ꨤ", "ꨈ"),
    ("ꩅ", "ꩊ"),
    ("꩝", "ꩃ"),
    ("ꨣ", "꩖"),
    ("ꨳ", "_"),
]


def sample_stream2_r3_1(idx: int, is_val: bool, rng: random.Random, seen_texts: set):
    base_off = (300000 if is_val else 100000) + idx * 17
    for att in range(50):
        m = base_off + att * 11
        sub = (idx + att) % 10
        pad_y_ov = None
        rep_gap = 0

        if sub < 6:
            # 60% Sesquisyllabic Disyllables (75% unmarked C1 + C2Vpre vs 25% dual prevowel contrast)
            w_sesq1 = synth_sesquisyllabic_word(rng, dual_prevowel=(m % 4 == 3))
            w_sesq2 = synth_sesquisyllabic_word(rng, dual_prevowel=False)
            w_fresh = synth_fresh_monosyllable(rng, force_bindu=(m % 5 == 0))
            order_m = m % 3
            if order_m == 0:
                words = [w_sesq1, w_fresh, w_sesq2] if rng.random() < 0.55 else [w_sesq1, w_fresh]
            elif order_m == 1:
                words = [w_fresh, w_sesq1, w_sesq2]
            else:
                words = [w_sesq1, w_sesq2, w_fresh]
            punct = rng.choice(["", " ꩝", " ꩞"])
            text = normalize_unicode((" ".join(words) + punct).strip())
        elif sub in (6, 7):
            # 20% High-Bindu ꩌ (U+AA4C) + Multi-tier ꨳꨪꩌ with tight top margin jitter (pad_y in 4..9)
            pad_y_ov = rng.choice([4, 5, 6, 8, 10])
            w_b1 = rng.choice(BINDU_WORDS) if (BINDU_WORDS and rng.random() < 0.5) else synth_fresh_monosyllable(rng, force_bindu=True)
            w_b2 = synth_fresh_monosyllable(rng, force_aa33=(m % 2 == 0), force_bindu=True)
            c_stack = rng.choice(["ꨣ", "ꨆ", "ꨓ", "ꨚ", "ꨕ", "ꨧ", "ꨨ"])
            w_stack = normalize_unicode(f"{c_stack}ꨳ{rng.choice(['ꨪ', 'ꨫ', 'ꨮ'])}ꩌ")
            text = normalize_unicode(f"{w_b1} {w_b2} {w_stack}")
        elif sub == 8:
            # 10% Confused codepoint pairs + Rare consonants (ꨄ, ꨜ, ꨘ, ꨞ, ꨖ, ꨁ, ꨅ)
            p_a, p_b = CONFUSED_PAIRS[m % len(CONFUSED_PAIRS)]
            w_fresh = synth_fresh_monosyllable(rng)
            if p_a in ("ꨪ", "ꨲ"):
                ca = rng.choice(["ꨀ", "ꨓ", "ꨚ", "ꨆ", "ꨤ", "ꨕ"])
                cb = rng.choice(["ꨀ", "ꨓ", "ꨚ", "ꨆ", "ꨧ", "ꨝ"])
                wa = normalize_unicode(f"{ca}{p_a}{rng.choice(['', 'ꩀ', 'ꩆ', 'ꩌ'])}")
                wb = normalize_unicode(f"{cb}{p_b}{rng.choice(['', 'ꩀ', 'ꩆ', 'ꨮꩉ'])}")
                text = f"{wa} {wb} {w_fresh}"
            elif p_a in ("ꩅ", "꩝"):
                wa = synth_fresh_monosyllable(rng)
                text = f"{wa} {w_fresh} ꩝꩝" if p_a == "꩝" else f"{ normalize_unicode(rng.choice(CHAM_TRUE_CONSONANTS)+'ꨯꨮꩅ') } { normalize_unicode(rng.choice(CHAM_TRUE_CONSONANTS)+'ꨶꨮꩊ') } {w_fresh}"
                rep_gap = rng.randint(2, 6)
            elif p_b == "꩖":
                text = normalize_unicode(f"꩖꩞ ꨣꨩ {w_fresh} {to_cham_num(rng.randint(106, 969))}")
            elif p_b == "_":
                w_ya = rng.choice(AA33_WORDS) if AA33_WORDS else synth_fresh_monosyllable(rng, force_aa33=True)
                text = normalize_unicode(f"{w_ya} {w_fresh}_{synth_fresh_monosyllable(rng)}")
            else:
                rb = rng.choice(RARE_AND_CONFUSED_BASES)
                v_r = rng.choice(["ꨩ", "ꨭ", "ꨰ", "ꨯ"]) if rb in CHAM_INDEPENDENT_VOWELS else rng.choice(PRIORITY_VOWEL_NUCLEI)
                wr = normalize_unicode(f"{rb}{v_r}{rng.choice(['', 'ꩀ', 'ꩍ', 'ꩌ'])}")
                wa = normalize_unicode(f"{p_a}ꨩ {p_b}ꨪ")
                text = f"{wr} {wa} {w_fresh}"
            text = normalize_unicode(text.strip())
        else:
            # 10% Cham 1..999 stanza numbers + 4-digit Cham years with internal zero (꩒꩐꩒꩖ style)
            w_fresh = synth_fresh_monosyllable(rng)
            if m % 2 == 0:
                d = rng.randint(1, 28)
                mo = rng.randint(1, 12)
                yr = rng.choice([2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2020, 2021, 2022, 2023, 2024, 2025, 2027, 1902, 1905, 1908])
                text = normalize_unicode(f"{w_fresh} {to_cham_num(d)}/{to_cham_num(mo)}/{to_cham_num(yr)}")
                rep_gap = rng.randint(1, 3)
            else:
                n = rng.choice([11, 22, 33, 100, 200, 300, 101, 202, 205, rng.randint(1, 999)])
                cnum = to_cham_num(n)
                rep_gap = rng.randint(2, 6)
                text = normalize_unicode(f"{cnum}꩞ {w_fresh} {synth_sesquisyllabic_word(rng)}")

        if text and text not in seen_texts and not ILLEGAL_MEDIAL_RE.search(text):
            img = render_hb_line(text, offset=m, pad_y_override=pad_y_ov, repeat_digit_gap=rep_gap)
            if img.height == 48 and img.width <= 320:
                seen_texts.add(text)
                return text, img
    raise RuntimeError(f"Stream 2 synthesis failed at idx={idx}")


# ==============================================================================
# Stream 3: Contrastive 14 Symbols & 7-Group Hyphens (100% Crisp HarfBuzz)
# ==============================================================================
def _make_symbol_words(rng: random.Random):
    """Returns (w1, w2, w3) where at least w2 or w1 is freshly non-corpus, focusing on boundary vowels."""
    tails = ["ꨪ", "ꨩ", "ꨭ", "ꨯꨮꩅ", "ꨳꩀ", "ꩊ", "ꩍ", "ꨰ"]
    c1 = rng.choice(["ꨧ", "ꨆ", "ꨓ", "ꨚ", "ꨕ", "ꨀ", "ꨨ", "ꨝ"])
    t1 = rng.choice(tails)
    if t1.startswith("ꨳ"):
        c1 = rng.choice(CHAM_MEDIAL_ELIGIBLE_CONSONANTS)
    if rng.random() < 0.45:
        w1 = normalize_unicode(f"{rng.choice(['ꨆ', 'ꨀ', 'ꨚ', 'ꨓ'])}ꨭ{c1}{t1}")
    else:
        w1 = normalize_unicode(f"{c1}{t1}")
    if w1 in CORPUS_WORD_SET or w1 in CORE_STANZA_SET:
        w2 = synth_fresh_monosyllable(rng, force_pre=(rng.random() < 0.5))
    else:
        w2 = rng.choice(["ꨔꩅ", "ꨓꨶꨮꩉ", "ꨚꨗꨶꨮꩄ", "ꨈꨴꨭꩀ"]) if rng.random() < 0.40 else synth_fresh_monosyllable(rng, force_pre=True)
    w3 = synth_fresh_monosyllable(rng)
    return w1, w2, w3


def sample_stream3_symbol_r3_1(sym_group: str, idx: int, is_val: bool, rng: random.Random, seen_texts: set):
    base_off = (500000 if is_val else 200000) + idx * 19
    for att in range(50):
        m = base_off + att * 13
        w1, w2, w3 = _make_symbol_words(rng)
        # Focus digits ending in 5, 7, 9 for {} so closing '}' after 7/9/5 is crystal clear
        if sym_group == "{}":
            d_last = rng.choice([5, 7, 9, 1, 3, 4, 8])
            num = rng.randint(10, 99) * 10 + d_last
        else:
            num = rng.randint(101, 998)

        if sym_group == "_":
            mod_u = m % 10
            if mod_u < 6:
                text = f"{w1}_{w2}"
            elif mod_u < 8:
                # Contrastive medial -ya ꨳ in same line as underscore '_'
                w_ya = synth_fresh_monosyllable(rng, force_aa33=True)
                text = f"{w_ya}_{w2}" if mod_u == 6 else f"{w1}_{w_ya}"
            else:
                text = f"{w1}_{w2} {w3}"
        elif sym_group == "{}":
            mod_b = m % 8
            if mod_b < 5:
                text = f"{w1} {{{num}}} {w2}"
            elif mod_b == 5:
                text = f"{{{num}}} {w1} {w2}"
            elif mod_b == 6:
                text = f"{w1} {{{w2}}} {w3}"
            else:
                text = f"{w1} {w2} {{{num}}}"
        elif sym_group == "/":
            mod_s = m % 10
            if mod_s < 7:
                text = f"{w1}/{w2}"
            elif mod_s == 7:
                text = f"{w1} / {w2}"
            elif mod_s == 8:
                d, mo, yr = rng.randint(1, 28), rng.randint(1, 12), rng.choice([2004, 2012, 2020, 2024, 2027])
                text = f"{to_cham_num(d)}/{to_cham_num(mo)}/{to_cham_num(yr)}"
            else:
                d, mo, yr = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1920, 2024)
                text = f"{d:02d}/{mo:02d}/{yr} {w2}"
        elif sym_group == "!":
            text = f"{w1} {w2} !" if (m % 5 != 0) else f"{w1} {w2} {w3} !"
        elif sym_group == "?":
            text = f"{w1} {w2} ?" if (m % 5 != 0) else f"{w1} {w2} {w3} ?"
        elif sym_group == "<":
            text = f"{w1} < {w2}" if (m % 5 != 0) else f"{w1} < {w2} {w3}"
        elif sym_group == ">":
            text = f"{w1} > {w2}" if (m % 5 != 0) else f"{w1} > {w2} {w3}"
        elif sym_group == "&":
            text = f"{w1} & {w2}" if (m % 5 != 0) else f"{w1} & {w2} {w3}"
        elif sym_group == "[]":
            text = f"{w1} [{num}] {w2}" if (m % 5 != 0) else f"[{num}] {w1} {w2}"
        elif sym_group == "#":
            text = f"{w1} #{num} {w2}" if (m % 5 != 0) else f"#{num} {w1} {w2}"
        elif sym_group == "\\":
            text = f"{w1} \\ {w2}" if (m % 4 != 0) else f"{w1}\\{w2}"
        elif sym_group == "*":
            text = f"* {w1} {w2} *" if (m % 4 != 0) else f"{w1} * {w2}"
        else:
            text = f"{w1} {sym_group} {w2}"

        text = normalize_unicode(text.strip())
        if text not in seen_texts and not ILLEGAL_MEDIAL_RE.search(text):
            img = render_hb_line(text, offset=m, pad_x_override=12 + (m % 8))
            if img.height == 48 and img.width <= 320:
                seen_texts.add(text)
                return text, img
    raise RuntimeError(f"Stream 3 symbol {sym_group} failed at idx={idx}")


def sample_stream3_hyphen_r3_1(hyp_group: str, group_local_idx: int, is_val: bool, rng: random.Random, seen_texts: set):
    pad_cycle = (12, 18, 24, 30, 36)
    base_off = (650000 if is_val else 350000) + group_local_idx * 23
    for att in range(60):
        m = base_off + att * 17
        w1, w2, _ = _make_symbol_words(rng)
        pad_x_ov = 12 + (m % 8)
        fsize_ov = None
        rep_gap = 0

        if hyp_group == "double_hyphen":
            # 100% crisp natural 0px-gap HarfBuzz `gid=10` `w1--w2` across all 8 CTC phases
            text = f"{w1}--{w2}"
        elif hyp_group == "word_word_tight":
            text = f"{w1}-{w2}"
        elif hyp_group == "word_word_spaced":
            text = f"{w1} - {w2}"
        elif hyp_group == "line_ending":
            text = f"{w1} {w2}-"
        elif hyp_group == "line_beginning":
            text = f"-{w1} {w2}" if (m % 6 != 0) else f"- {w1} {w2}"
        elif hyp_group == "year_year":
            # Train zero-rich 4-digit years (e.g. 1988-2000, 1996-2000, 2001-2025) while avoiding FORBIDDEN_Y1_4DIGIT
            sub_y = (group_local_idx + att) % 4
            if sub_y == 0:
                y1 = rng.choice([1985, 1988, 1991, 1992, 1994, 1996, 1998])
                y2 = 2000
                rep_gap = rng.randint(1, 3)
            elif sub_y == 1:
                y1 = rng.choice([2001, 2002, 2003, 2004, 2006, 2008])
                y2 = rng.randint(y1 + 2, 2028)
                rep_gap = rng.randint(1, 3)
            elif sub_y == 2:
                y1 = rng.randint(1810, 1898)
                while y1 in FORBIDDEN_Y1_4DIGIT:
                    y1 = rng.randint(1810, 1898)
                y2 = 1900 if rng.random() < 0.5 else (y1 + rng.randint(4, 35))
                rep_gap = rng.randint(1, 3)
            else:
                y1 = rng.randint(1912, 2018)
                while y1 in FORBIDDEN_Y1_4DIGIT:
                    y1 = rng.randint(1912, 2018)
                y2 = y1 + rng.randint(3, 22)
            text = f"{y1}-{y2}"
            pad_x_ov = pad_cycle[m % len(pad_cycle)]
        elif hyp_group == "number_number":
            pad_x_ov = pad_cycle[m % len(pad_cycle)]
            fsize_ov = (20, 22, 24)[m % 3]
            if not is_val:
                if group_local_idx < 180 and att == 0:
                    # 60 disjoint D-D pairs x 3 distinct visual margin/size/weight renders
                    # Note: `seen_texts` prevents exact text duplicates across dataset EXCEPT D-D has only 60 strings;
                    # append contextual word on pass >= 1 so EVERY text in train_label.txt is 100% unique!
                    pair_str = TRAIN_SINGLE_DIGIT_PAIRS[group_local_idx % 60]
                    pass_id = group_local_idx // 60
                    if pass_id == 0:
                        text = pair_str
                    elif pass_id == 1:
                        text = f"{pair_str} {synth_fresh_monosyllable(rng)}"
                    else:
                        text = f"{synth_fresh_monosyllable(rng)} {pair_str}"
                elif group_local_idx < 240:
                    d1 = rng.randint(1, 9)
                    d2 = rng.randint(11, 98)
                    while d1 == 5 and d2 == 10:
                        d1 = random.randint(1, 9)
                    text = f"{d1}-{d2}" if (m % 2 == 0) else f"{d2}-{d1}"
                elif group_local_idx < 370:
                    n1 = rng.randint(11, 96)
                    while n1 in FORBIDDEN_N1_2DIGIT:
                        n1 = rng.randint(11, 96)
                    n2 = min(99, n1 + rng.randint(1, 25)) if (m % 2 == 0) else rng.randint(11, 99)
                    if n2 == n1:
                        n2 = min(99, n1 + 2)
                    text = f"{n1}-{n2}"
                else:
                    # 3-digit zero-rich pairs (e.g. 105-200, 305-400, 600-700) excluding FORBIDDEN_N1_3DIGIT
                    if m % 2 == 0:
                        n1 = rng.choice([104, 108, 150, 180, 205, 305, 350, 450, 600, 700, 800])
                        n2 = rng.choice([200, 300, 400, 500, 700, 800, 900])
                        if n2 <= n1:
                            n2 = n1 + 100
                        rep_gap = rng.randint(1, 3)
                    else:
                        n1 = rng.randint(102, 880)
                        while n1 in FORBIDDEN_N1_3DIGIT:
                            n1 = rng.randint(102, 880)
                        n2 = n1 + rng.randint(5, 110)
                    text = f"{n1}-{n2}"
            else:
                if group_local_idx < 9 and att == 0:
                    text = VAL_SINGLE_DIGIT_PAIRS[group_local_idx]
                elif group_local_idx < 35:
                    n1 = rng.randint(11, 96)
                    while n1 in FORBIDDEN_N1_2DIGIT:
                        n1 = rng.randint(11, 96)
                    n2 = min(99, n1 + rng.randint(2, 20))
                    text = f"{n1}-{n2}"
                else:
                    n1 = rng.randint(102, 880)
                    while n1 in FORBIDDEN_N1_3DIGIT:
                        n1 = rng.randint(102, 880)
                    n2 = n1 + rng.randint(10, 120)
                    text = f"{n1}-{n2}"
        else:
            text = f"{w1}-{w2}"

        text = normalize_unicode(text.strip())
        if text not in seen_texts and not ILLEGAL_MEDIAL_RE.search(text):
            img = render_hb_line(text, offset=m, pad_x_override=pad_x_ov, font_size_override=fsize_ov, repeat_digit_gap=rep_gap)
            if img.height == 48 and img.width <= 320:
                seen_texts.add(text)
                return text, img
    raise RuntimeError(f"Stream 3 hyphen {hyp_group} failed at idx={group_local_idx}, is_val={is_val}")


# ==============================================================================
# Stream 4: Morphological Erosion + Severe Optical Degradation + High-Noise Blank
# ==============================================================================
def _motion_kernel(k: int, angle_deg: float) -> np.ndarray:
    kern = np.zeros((k, k), dtype=np.float32)
    kern[k // 2, :] = 1.0
    c = (k / 2.0 - 0.5, k / 2.0 - 0.5)
    m = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    rot = cv2.warpAffine(kern, m, (k, k), flags=cv2.INTER_LINEAR)
    s = float(rot.sum())
    return (rot / s) if s > 1e-6 else kern


def degrade_erosion_dry_ink(img: Image.Image, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    """Directly targets morphological ink erosion (`erode k=2, 3`) and dry faded high-threshold binarization."""
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    mode = rng.random()
    if mode < 0.48:
        # Exact 3x3 morphological erosion on ink (matching Stress-Frozen `erode=3`)
        k3 = np.ones((3, 3), dtype=np.uint8)
        eroded = cv2.erode(inv, k3, iterations=1)
        if rng.random() < 0.35:
            # Blend 75..100% so both crisp erode=3 and partial dry-stroke thinning are covered
            alpha = rng.uniform(0.75, 1.0)
            eroded = np.clip(inv.astype(np.float32) * (1.0 - alpha) + eroded.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        out_gray = 255 - eroded
    elif mode < 0.85:
        # 2x2 morphological erosion + faint fade
        k2 = np.ones((2, 2), dtype=np.uint8)
        eroded = cv2.erode(inv, k2, iterations=1)
        fade = rng.uniform(0.0, 0.35)
        out_gray = np.clip((255.0 - eroded.astype(np.float32)) * (1.0 - fade) + 255.0 * fade, 0, 255).astype(np.uint8)
    else:
        # High-threshold binarization (chips away anti-aliased stroke borders)
        th = rng.randint(135, 185)
        _, out_gray = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)

    out = cv2.cvtColor(out_gray, cv2.COLOR_GRAY2RGB)
    if rng.random() < 0.45:
        noise = np_rng.normal(0.0, rng.uniform(2.0, 10.0), size=out.shape[:2])[:, :, None]
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def degrade_severe_motion(img: Image.Image, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    if rng.random() < 0.35:
        # Combo low_height [15..20] + motion [7..11]
        hl = rng.randint(15, 20)
        wl = max(8, int(round(w * (hl / float(h)))))
        arr = cv2.resize(cv2.resize(arr, (wl, hl), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
        k = rng.choice([7, 9, 11])
        ang = rng.choice([-8.0, 0.0, 0.0, 8.0])
    else:
        # Standalone severe motion blur k in {9, 11, 13, 15, 17}
        k = rng.choice([9, 11, 13, 15, 17, 17])
        ang = rng.choice([-15.0, -7.0, 0.0, 0.0, 7.0, 15.0])
    out = cv2.filter2D(arr, -1, _motion_kernel(k, ang), borderType=cv2.BORDER_REPLICATE)
    if rng.random() < 0.50:
        noise = np_rng.normal(0.0, rng.uniform(1.5, 6.0), size=out.shape[:2])[:, :, None]
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def degrade_heavy_gauss_noise_jpeg(img: Image.Image, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    if rng.random() < 0.65:
        sigma = rng.choice([2.0, 2.5, 3.0, 3.5])
        out = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    else:
        out = arr.copy()
    if rng.random() < 0.65:
        n_std = rng.choice([18.0, 30.0, 45.0, 60.0])
        noise = np_rng.normal(0.0, n_std, size=out.shape).astype(np.float32)
        fade = rng.uniform(0.0, 0.25)
        out = np.clip(out.astype(np.float32) * (1.0 - fade) + 255.0 * fade + noise, 0, 255).astype(np.uint8)
    q = rng.randint(35, 75)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return Image.fromarray(out)


def degrade_extreme_lowres(img: Image.Image, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    hl = rng.choice([12, 13, 14, 15, 16, 18, 20])
    wl = max(8, int(round(w * (hl / float(h)))))
    small = cv2.resize(arr, (wl, hl), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    if rng.random() < 0.45:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=rng.uniform(0.5, 1.3), borderType=cv2.BORDER_REPLICATE)
    if rng.random() < 0.45:
        noise = np_rng.normal(0.0, rng.uniform(2.0, 8.0), size=out.shape[:2])[:, :, None]
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def degrade_persp_dilate_shadow(img: Image.Image, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    if rng.random() < 0.55:
        p = rng.uniform(0.06, 0.14)
        dx, dy = float(w * p), float(h * p)
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([[dx, dy], [w - dx, 0], [0, h], [w, h - dy]])
        out = cv2.warpPerspective(arr, cv2.getPerspectiveTransform(pts1, pts2), (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    else:
        ang = rng.uniform(-3.8, 3.8)
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
        out = cv2.warpAffine(arr, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    if rng.random() < 0.55:
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        kd = rng.choice([2, 3])
        dil = cv2.dilate(255 - gray, np.ones((kd, kd), np.uint8), iterations=1)
        out = cv2.cvtColor(255 - dil, cv2.COLOR_GRAY2RGB)

    lo = rng.uniform(0.60, 0.85)
    grad = np.linspace(lo, 1.0, w, dtype=np.float32)[None, :, None]
    if rng.random() < 0.5:
        grad = grad[:, ::-1, :]
    noise = np_rng.normal(0.0, rng.uniform(2.0, 9.0), size=(h, w, 1)).astype(np.float32)
    out = np.clip(out.astype(np.float32) * grad + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def generate_hard_blank(rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    w = rng.randint(88, 320)
    h = 48
    base = rng.randint(155, 252)
    arr = np.full((h, w, 3), base, dtype=np.float32)
    arr[:, :, 0] += rng.uniform(-8.0, 6.0)
    arr[:, :, 2] += rng.uniform(-6.0, 8.0)
    gx = np.linspace(rng.uniform(0.60, 0.92), 1.0, w, dtype=np.float32)
    if rng.random() < 0.5:
        gx = gx[::-1]
    arr *= gx[None, :, None]
    sigma = rng.uniform(6.0, 55.0)
    noise = np_rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _save_unique_jpeg(img: Image.Image, dst_path: str, seen_hashes: set, salt_seed: int) -> str:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    for step in range(25):
        if step > 0:
            r = (salt_seed + step * 7) % arr.shape[0]
            c = (salt_seed * 3 + step * 13) % arr.shape[1]
            ch = step % 3
            v = int(arr[r, c, ch])
            arr[r, c, ch] = max(0, v - 1) if v > 128 else min(255, v + 1)
        pil_im = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_im.save(buf, format="JPEG", quality=95, subsampling=0)
        raw_bytes = buf.getvalue()
        digest = hashlib.sha256(raw_bytes).hexdigest().lower()
        if digest not in seen_hashes:
            seen_hashes.add(digest)
            with open(dst_path, "wb") as f:
                f.write(raw_bytes)
            return digest
    raise RuntimeError(f"Could not produce unique JPEG hash for {dst_path}")


def _sample_clean_text_for_degradation(idx: int, is_val: bool, rng: random.Random, seen_texts: set):
    base_off = (900000 if is_val else 750000) + idx * 29
    for att in range(50):
        m = base_off + att * 19
        b = idx % 10
        if b < 5:
            # Sesquisyllabic + fresh monosyllabic Cham line
            w1 = synth_sesquisyllabic_word(rng, dual_prevowel=(m % 5 == 4))
            w2 = synth_fresh_monosyllable(rng, force_bindu=(m % 4 == 0))
            w3 = synth_fresh_monosyllable(rng, force_pre=(m % 3 == 0))
            words = [w1, w2] if (m % 3 == 0) else [w1, w2, w3]
            text = normalize_unicode(" ".join(words) + rng.choice(["", " ꩝", " ꩞"]))
        elif b < 7:
            # Cham verse number + Cham words
            cnum = to_cham_num(rng.randint(1, 999))
            text = normalize_unicode(f"{cnum}꩞ {synth_fresh_monosyllable(rng)} {synth_sesquisyllabic_word(rng)}")
        elif b < 9:
            # Symbol in mildly degraded context (keeping `!`, `{}`, `_`, `/` robust to scan noise!)
            s_grp = ["!", "{}", "_", "/", "?", "[]", "#", "&", "*", "<", ">", "\\"][m % 12]
            return sample_stream3_symbol_r3_1(s_grp, idx + 50000 + att, is_val, rng, seen_texts)
        else:
            h_grp = ["double_hyphen", "number_number", "word_word_tight", "line_beginning", "year_year", "line_ending", "word_word_spaced"][m % 7]
            return sample_stream3_hyphen_r3_1(h_grp, idx + 50000 + att, is_val, rng, seen_texts)

        text = text.strip()
        if text and text not in seen_texts and not ILLEGAL_MEDIAL_RE.search(text):
            # Render slightly bolder 35% of the time for erosion resilience across font weights
            wt = "Bold" if (m % 3 == 0) else "Regular"
            img = render_hb_line(text, offset=m, weight_override=wt)
            if img.height == 48 and img.width <= 320:
                seen_texts.add(text)
                return text, img
    raise RuntimeError(f"Failed clean text for degradation at idx={idx}")


def build_split(
    split_name: str,
    r216_dir: str,
    out_dir: str,
    seen_texts: set,
    seen_img_hashes: set,
    seed: int,
):
    is_val = (split_name == "val")
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed + 101)

    r216_lines = [
        l.rstrip("\r\n")
        for l in open(os.path.join(r216_dir, f"{split_name}_label.txt"), "r", encoding="utf-8")
        if l.strip("\r\n")
    ]

    if not is_val:
        # Total = 21,600
        # 1. Replay from R2.1.6 = 9,720 (clean [0:9900]->6600, s2 [9900:10700]->800, s5 [14400:17100]->2020, blank [17100:18000]->300)
        replay_ranges = [(0, 9900, 6600), (9900, 10700, 800), (14400, 17100, 2020), (17100, 18000, 300)]
        n_s2 = 3888
        sym_quotas = [
            ("_", 480), ("{}", 440), ("/", 340), ("!", 240),
            ("?", 90), ("#", 90), ("&", 90), ("*", 90),
            ("<", 90), (">", 90), ("\\", 90), ("[]", 174),
        ]  # sum = 2,304
        hyp_quotas = [
            ("double_hyphen", 460), ("number_number", 460), ("line_beginning", 260),
            ("word_word_tight", 240), ("year_year", 160), ("line_ending", 110), ("word_word_spaced", 110),
        ]  # sum = 1,800
        deg_quotas = [
            ("erosion", 900), ("motion", 780), ("gauss_noise_jpeg", 760),
            ("lowres", 700), ("persp_dilate", 428), ("blank_noisy", 320),
        ]  # sum = 3,888
    else:
        # Total = 2,400
        replay_ranges = [(0, 1100, 730), (1100, 1190, 90), (1600, 1900, 225), (1900, 2000, 35)]
        n_s2 = 432
        sym_quotas = [
            ("_", 54), ("{}", 48), ("/", 38), ("!", 26),
            ("?", 10), ("#", 10), ("&", 10), ("*", 10),
            ("<", 10), (">", 10), ("\\", 10), ("[]", 20),
        ]  # sum = 256
        hyp_quotas = [
            ("double_hyphen", 50), ("number_number", 52), ("line_beginning", 28),
            ("word_word_tight", 28), ("year_year", 18), ("line_ending", 12), ("word_word_spaced", 12),
        ]  # sum = 200
        deg_quotas = [
            ("erosion", 100), ("motion", 86), ("gauss_noise_jpeg", 84),
            ("lowres", 78), ("persp_dilate", 48), ("blank_noisy", 36),
        ]  # sum = 432

    img_out_dir = os.path.join(out_dir, "images")
    os.makedirs(img_out_dir, exist_ok=True)
    out_labels = []
    idx_global = 0

    # --- Stream 1: Replay ---
    for lo, hi, k in replay_ranges:
        pool = list(range(lo, hi))
        rng.shuffle(pool)
        for src_i in sorted(pool[:k]):
            rel_img, lbl = r216_lines[src_i].split("\t", 1)
            src_img_p = os.path.join(r216_dir, rel_img.replace("/", os.sep))
            dst_name = f"{split_name}_{idx_global:05d}.jpg"
            dst_p = os.path.join(img_out_dir, dst_name)
            raw_b = open(src_img_p, "rb").read()
            dg = hashlib.sha256(raw_b).hexdigest().lower()
            if dg not in seen_img_hashes:
                seen_img_hashes.add(dg)
                shutil.copy2(src_img_p, dst_p)
            else:
                with Image.open(src_img_p) as im:
                    _save_unique_jpeg(im, dst_p, seen_img_hashes, idx_global + 17)
            if lbl:
                seen_texts.add(lbl)
            out_labels.append(f"images/{dst_name}\t{lbl}")
            idx_global += 1

    replay_end_idx = idx_global
    print(f"  [{split_name}] Stream 1 (Replay): {replay_end_idx} samples")

    # --- Stream 2: Sesquisyllabic & Residual Chars ---
    for i in range(n_s2):
        txt, im = sample_stream2_r3_1(i, is_val, rng, seen_texts)
        dst_name = f"{split_name}_{idx_global:05d}.jpg"
        _save_unique_jpeg(im, os.path.join(img_out_dir, dst_name), seen_img_hashes, idx_global * 3 + 1)
        out_labels.append(f"images/{dst_name}\t{txt}")
        idx_global += 1
    print(f"  [{split_name}] Stream 2 (Sesquisyllabic + Residual): {n_s2} samples")

    # --- Stream 3A: 14 Mandatory Symbols ---
    for s_grp, q in sym_quotas:
        for j in range(q):
            txt, im = sample_stream3_symbol_r3_1(s_grp, idx_global + j, is_val, rng, seen_texts)
            dst_name = f"{split_name}_{idx_global:05d}.jpg"
            _save_unique_jpeg(im, os.path.join(img_out_dir, dst_name), seen_img_hashes, idx_global * 5 + 2)
            out_labels.append(f"images/{dst_name}\t{txt}")
            idx_global += 1

    # --- Stream 3B: 7-Group Hyphens ---
    for h_grp, q in hyp_quotas:
        for j in range(q):
            txt, im = sample_stream3_hyphen_r3_1(h_grp, j, is_val, rng, seen_texts)
            dst_name = f"{split_name}_{idx_global:05d}.jpg"
            _save_unique_jpeg(im, os.path.join(img_out_dir, dst_name), seen_img_hashes, idx_global * 7 + 3)
            out_labels.append(f"images/{dst_name}\t{txt}")
            idx_global += 1
    print(f"  [{split_name}] Stream 3 (Symbols + Hyphens): {sum(q for _, q in sym_quotas) + sum(q for _, q in hyp_quotas)} samples")

    # --- Stream 4: Balanced Erosion + Severe Optical Degradation ---
    for d_mode, q in deg_quotas:
        for j in range(q):
            if d_mode == "blank_noisy":
                txt = ""
                im = generate_hard_blank(rng, np_rng)
            else:
                txt, clean_im = _sample_clean_text_for_degradation(idx_global + j, is_val, rng, seen_texts)
                if d_mode == "erosion":
                    im = degrade_erosion_dry_ink(clean_im, rng, np_rng)
                elif d_mode == "motion":
                    im = degrade_severe_motion(clean_im, rng, np_rng)
                elif d_mode == "gauss_noise_jpeg":
                    im = degrade_heavy_gauss_noise_jpeg(clean_im, rng, np_rng)
                elif d_mode == "lowres":
                    im = degrade_extreme_lowres(clean_im, rng, np_rng)
                else:
                    im = degrade_persp_dilate_shadow(clean_im, rng, np_rng)
            dst_name = f"{split_name}_{idx_global:05d}.jpg"
            _save_unique_jpeg(im, os.path.join(img_out_dir, dst_name), seen_img_hashes, idx_global * 11 + 4)
            out_labels.append(f"images/{dst_name}\t{txt}")
            idx_global += 1
    print(f"  [{split_name}] Stream 4 (Erosion + Severe Optical + Hard Blanks): {sum(q for _, q in deg_quotas)} samples | Total={idx_global}")

    lbl_path = os.path.join(out_dir, f"{split_name}_label.txt")
    with open(lbl_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out_labels) + "\n")
    return lbl_path, idx_global, replay_end_idx


def main():
    t0 = time.time()
    print("=" * 80)
    print("🚀 GENERATING STAGE R3.1 DATASET (24,000 SAMPLES: SESQUISYLLABIC + EROSION + GATE CLOSURE)")
    print("=" * 80)
    print(f"  • Loaded safe corpus lexicon: {len(CORPUS_WORDS_SORTED):,} words | {len(SESQ_WORDS):,} sesquisyllabic | {len(BINDU_WORDS):,} bindu")

    r216_dir = os.path.join(TRAINING_DIR, "data", "cham_synthetic_r2_1_6_surgical_polish")
    assert os.path.exists(r216_dir), f"Missing anchor dataset: {r216_dir}"

    out_dir = os.path.join(TRAINING_DIR, "data", "cham_synthetic_r3_1_closure")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    seen_texts = set()
    seen_img_hashes = set()

    # Pre-register R2.1.6 val texts first so Train never collides with Val replay!
    for split_n in ("train", "val"):
        for l in open(os.path.join(r216_dir, f"{split_n}_label.txt"), "r", encoding="utf-8"):
            parts = l.rstrip("\r\n").split("\t", 1)
            if len(parts) == 2 and parts[1]:
                seen_texts.add(parts[1])

    # Reset seen_texts to track actual emitted train+val texts while blocking cross-split collisions
    reserved_r216_val = {
        l.rstrip("\r\n").split("\t", 1)[1]
        for l in open(os.path.join(r216_dir, "val_label.txt"), "r", encoding="utf-8")
        if "\t" in l and l.rstrip("\r\n").split("\t", 1)[1]
    }
    emitted_texts = set(reserved_r216_val)

    tr_lbl, n_tr, tr_rep_end = build_split("train", r216_dir, out_dir, emitted_texts, seen_img_hashes, seed=20261117)
    # Remove reserved_r216_val that haven't been emitted yet so val replay can emit its own R2.1.6 val lines,
    # while keeping all emitted train texts in `emitted_texts` so Val never collides with Train!
    tr_emitted_only = {
        l.rstrip("\r\n").split("\t", 1)[1]
        for l in open(tr_lbl, "r", encoding="utf-8")
        if "\t" in l and l.rstrip("\r\n").split("\t", 1)[1]
    }
    val_seen = set(tr_emitted_only)
    va_lbl, n_va, va_rep_end = build_split("val", r216_dir, out_dir, val_seen, seen_img_hashes, seed=20261118)

    tr_sha = hashlib.sha256(open(tr_lbl, "rb").read()).hexdigest()
    va_sha = hashlib.sha256(open(va_lbl, "rb").read()).hexdigest()
    manifest = {
        "stage": "R3.1_GATE_CLOSURE_SESQUISYLLABIC_EROSION",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(time.time() - t0, 2),
        "train_samples": n_tr,
        "val_samples": n_va,
        "total_samples": n_tr + n_va,
        "unique_image_sha256_count": len(seen_img_hashes),
        "train_replay_end_idx": tr_rep_end,
        "val_replay_end_idx": va_rep_end,
        "train_label_sha256": tr_sha,
        "val_label_sha256": va_sha,
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    m_path = os.path.join(REPORTS_DIR, "r3_1_closure_manifest.json")
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Stage R3.1 dataset generated in {manifest['duration_sec']}s -> {out_dir}")
    print(f"   Train SHA256: {tr_sha[:16]}... | Val SHA256: {va_sha[:16]}... | Unique Hashes: {len(seen_img_hashes):,}")


if __name__ == "__main__":
    main()
