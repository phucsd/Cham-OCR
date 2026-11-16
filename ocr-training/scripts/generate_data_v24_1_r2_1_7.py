#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R2.1.7 Surgical Gate-Closure Dataset Generator
==================================================================
Generates the 20,000-sample (18,000 train / 2,000 val) multi-stream dataset for
Stage R2.1.7 (`SURGICAL-GATE-CLOSURE-R2.1.7`):
  - Shares 15,000 / 20,000 (75.0%) samples BIT-FOR-BIT at the exact same indices with Branch E:
    * Train 0..9899 (9,900) & Val 0..1099 (1,100): Clean Stanza Replay (11,000 = 55.0%)
    * Train 14400..17099 (2,700) & Val 1600..1899 (300): Short-Phonotactic CombMark W<=320px (3,000 = 15.0%)
    * Train 17100..17999 (900) & Val 1900..1999 (100): Blank Guard (1,000 = 5.0%)
  - Replaces the middle 5,000 samples (Train 9900..14399 = 4,500; Val 1100..1599 = 500) using
    cached HarfBuzz + FreeType Fixed-Baseline shaping (`uharfbuzz` + `freetype`) + Direct H=48 Pillow:
    * Stream 2 (Train 9900..10649 = 750; Val 1100..1183 = 84):
      Fixed-Baseline Short Phonotactic, Rare-Char & U+AA56 (6) vs U+AA23 (ra) Confusion Hard (834, W <= 320px)
    * Stream 3 (Train 10650..12899 = 2,250; Val 1184..1433 = 250):
      Fixed-Baseline 14 Mandatory Symbols (2,500, W <= 320px, diversified `{}` / `_` / `/` templates & renderer mix)
    * Stream 4 (Train 12900..14399 = 1,500; Val 1434..1599 = 166):
      Fixed-Baseline 7-Group Multi-Context Hyphen (1,666, W <= 320px, `--`, varied-pad standalone `D-D`, `-word` mix)

Strict Invariants:
  1. 100% Independent Generation: Zero imports, reads, or string literals referencing external eval suites.
  2. 100% Width Cap (W <= 320px at H=48) across all replaced hard samples (Streams 2, 3, 4) and Stream 5.
  3. 100% Phonotactically Valid: Zero medials (`ꨳ..ꨶ`) attached to independent vowels (`ꨁ..ꨅ`).
  4. Zero Train/Val Text Leakage & 114/114 Dictionary Token Coverage.
"""

import os
import sys
import io
import math
import json
import random
import shutil
import hashlib
from datetime import datetime, timezone
import numpy as np
import cv2
import uharfbuzz as hb
import freetype
from PIL import Image, ImageDraw, ImageFont

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)
REPORTS_DIR = os.path.join(PROJECT_ROOT, "REPORTS")


def find_asset_path(filename):
    candidates = [
        os.path.join(TRAINING_DIR, "data", "fonts", filename),
        os.path.join(TRAINING_DIR, "data", filename),
        os.path.join(PROJECT_ROOT, "ocr-studio", "fonts", filename),
        os.path.join(PROJECT_ROOT, "ocr-studio", "data", filename),
        os.path.join(PROJECT_ROOT, "05_Dictionaries_and_Fonts", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def parse_unicode_clusters(text):
    if not text:
        return []
    medials = set(['ꨳ', 'ꨴ', 'ꨵ', 'ꨶ'])
    pre_vowels = set(['ꨯ', 'ꨰ'])
    all_combining = set([chr(c) for c in range(0xAA29, 0xAA37)] + [chr(c) for c in range(0xAA40, 0xAA4E)])
    clusters = []
    current_cluster = {'base': None, 'medials': [], 'pre_ra': [], 'pre_vowels': [], 'others': []}
    for ch in text:
        if ch in all_combining:
            if current_cluster['base'] is None:
                current_cluster['base'] = ''
            if ch == 'ꨴ':
                current_cluster['pre_ra'].append(ch)
            elif ch in medials:
                current_cluster['medials'].append(ch)
            elif ch in pre_vowels:
                current_cluster['pre_vowels'].append(ch)
            else:
                current_cluster['others'].append(ch)
        else:
            if current_cluster['base'] is not None:
                clusters.append(current_cluster)
            current_cluster = {'base': ch, 'medials': [], 'pre_ra': [], 'pre_vowels': [], 'others': []}
    if current_cluster['base'] is not None:
        clusters.append(current_cluster)
    return clusters


def normalize_unicode(text):
    clusters = parse_unicode_clusters(text)
    out = []
    for c in clusters:
        s = (c['base'] or '') + "".join(c['medials']) + "".join(c['pre_ra']) + "".join(c['pre_vowels']) + "".join(c['others'])
        out.append(s)
    return "".join(out)


DICT_PATH = find_asset_path("cham_dict_v24_1.txt")
if not DICT_PATH or not os.path.exists(DICT_PATH):
    raise FileNotFoundError("Cannot locate cham_dict_v24_1.txt")
with open(DICT_PATH, "r", encoding="utf-8") as f:
    DICT_CHARS_V24_1 = [line.strip("\r\n") for line in f if line.strip("\r\n")]
assert len(DICT_CHARS_V24_1) == 114, f"Expected 114 tokens, got {len(DICT_CHARS_V24_1)}"

CHAM_DIGITS = list("꩐꩑꩒꩓꩔꩕꩖꩗꩘꩙")
CHAM_CARRIER_A = 'ꨀ'
CHAM_INDEPENDENT_VOWELS = ['ꨁ', 'ꨂ', 'ꨃ', 'ꨄ', 'ꨅ']
CHAM_TRUE_CONSONANTS = [
    'ꨆ', 'ꨇ', 'ꨈ', 'ꨉ', 'ꨊ', 'ꨋ', 'ꨌ', 'ꨍ', 'ꨎ', 'ꨏ',
    'ꨐ', 'ꨑ', 'ꨒ', 'ꨓ', 'ꨔ', 'ꨕ', 'ꨖ', 'ꨗ', 'ꨘ', 'ꨙ',
    'ꨚ', 'ꨛ', 'ꨜ', 'ꨝ', 'ꨞ', 'ꨟ', 'ꨠ', 'ꨡ', 'ꨢ', 'ꨣ',
    'ꨤ', 'ꨥ', 'ꨦ', 'ꨧ', 'ꨨ'
]
CHAM_MEDIAL_ELIGIBLE_CONSONANTS = [
    'ꨆ', 'ꨇ', 'ꨈ', 'ꨌ', 'ꨍ', 'ꨎ', 'ꨓ', 'ꨔ', 'ꨕ', 'ꨗ',
    'ꨘ', 'ꨙ', 'ꨚ', 'ꨛ', 'ꨜ', 'ꨝ', 'ꨟ', 'ꨠ', 'ꨡ', 'ꨣ',
    'ꨤ', 'ꨧ', 'ꨨ'
]
PRIORITY_VOWEL_NUCLEI = [
    'ꨰ', 'ꨰ', 'ꨰ', 'ꨯ', 'ꨯ', 'ꨯ', 'ꨯꨱ', 'ꨯꨱ', 'ꨯꨮ', 'ꨯꨮ',
    'ꨮ', 'ꨬ', 'ꨭ', 'ꨩ', 'ꨪ', 'ꨫ', 'ꨱ', 'ꨲ'
]
MEDIAL_VOWEL_NUCLEI = [
    'ꨰ', 'ꨰ', 'ꨰ', 'ꨯ', 'ꨯ', 'ꨯꨱ', 'ꨯꨱ', 'ꨯꨮ',
    'ꨩ', 'ꨪ', 'ꨬ', 'ꨭ', 'ꨮ', 'ꨲ', ''
]
PRIORITY_MEDIALS = ['ꨴ', 'ꨴ', 'ꨵ', 'ꨵ', 'ꨶ', 'ꨶ', 'ꨳ']
PRIORITY_FINALS = ['ꩌ', 'ꩌ', 'ꩃ', 'ꩃ', 'ꩍ', 'ꩀ', 'ꩅ', 'ꩆ', 'ꩉ', 'ꩊ', 'ꩄ', 'ꩂ', 'ꩈ', 'ꩇ', 'ꩋ', 'ꩁ']

CORE_STANZA_WORDS = [
    "ꨀꨆꨢꨯꨮꩅ", "ꨧꨪ", "ꨚꨗꩍ", "ꨓꨶꨮꩉ", "ꨓꨝꨳꩀ", "ꨚꨗꨶꨮꩄ", "ꨆꨭꨖꨩ", "ꨔꩅ",
    "ꨨꨤꨭꩆ", "ꨋꩇ", "ꨝꨩ", "ꨤꨪꨠꩍ", "ꨦꩀꨆꨣꨰ", "ꨝꨪꨗꨴꨪꩀ", "ꨕꨨꨵꩀ", "ꨥꩀ",
    "ꨎꩊ", "ꨈꨴꨭꩀ", "ꨆꨔꨯꨱꩅ", "ꨣꨪꨡꩍ", "ꨚꨴꨯꨱꩃ", "ꨤꨯꨩ", "ꨕꨫ", "ꨣꨳꨪꩌ",
    "ꨚꨆꩉ", "ꨆꨘꩊ", "ꨣꨩ", "ꨓꨨꨩ", "ꨙꨯꩌ", "ꨀꨇꨩꩆ", "ꨕꨤꩌ", "ꨕꨭꨤꨪꨆꩊ",
    "ꨧꨳꩌ", "ꨧꨭꨩ", "ꨏꩀ", "ꨕꨯꨱꩀ", "ꨨꨚꩀ", "ꨕꨯꨮꨝꨪꨓꨩ", "ꨀꨲꩆ", "ꨓꨘꩆ",
    "ꨌꩃ", "ꨆꨩ", "ꨔꨬ", "ꨤꨪꨆꨮꨭ", "ꨎꩃ", "ꨅꨩ", "ꨨꨭꨩ", "ꨟꩃ",
    "ꨕꨯꨮꩍ", "ꨧꨗꨪ", "ꨚꨮꩃ", "ꨙꨯꨱꩌ", "ꨚꨣꩆ", "ꨨꨕꨮꩉ", "ꨝꨪꨣꨭꨥ", "ꨠꨡꨯꩍ",
    "ꨛꨯꨱ", "ꨠꨣꨰ", "ꨀꨝꨪꩍ", "ꨕꨴꨬ", "ꨀꨕꨬ", "ꨧꨀꨰ", "ꨕꨣꨩ", "ꨀꨮꩆ",
    "ꨓꨝꨶꨮꩆ", "ꨚꨣꨪꩀ", "ꨚꨗꨴꩃ", "ꨆꨴꨯꨱꩃ", "ꨚꨎꨰ", "ꨟꨰ", "ꨨꨶꨮꩄ", "ꨆꨙꨩ",
    "ꨢꨶꨮꩆ", "ꨐꨭꨩ", "ꨉꩌ", "ꨌꩌ", "ꨂꨣꩃ", "ꨤꩄ", "ꨆꨳꨮꩃ", "ꨕꨯꨱꩃ",
    "ꨠꨢꩍ", "ꨚꨕꨬ", "ꨎꨳꨯꨮꩃ", "ꨎꨶꨰ", "ꨡꨶꩍ", "ꨆꩉ", "ꨆꨮꨭ", "ꨧꨭꨗꨪꩅ",
    "ꨈꨪꨗꨴꨮꩍ", "ꨤꨪꨝꨰꩍ", "ꨣꩈ", "ꨕꨨꨵꨮꨭ", "ꨨꨤꨬ", "ꨀꨗꩀ", "ꨕꩀ", "ꨓꨆꨴꨲꨩ",
    "ꨊꨯꨱꩀ", "ꨤꨪꨊꨪꩀ", "ꨓꨊꨫꩆ", "ꨀꨑꨩ", "ꨝꨑꩀ", "ꨒꨭꩍ", "ꨀꨎꨮꩂ", "ꨀꨟꨳꨯꨮꩂ", "ꨀꨟꨯꩂ"
]
CORE_STANZA_SET = set(CORE_STANZA_WORDS)

ANCHOR_WORDS_TAIL = [
    "ꨧꨪ", "ꨆꨭꨖꨩ", "ꨀꨆꨢꨯꨮꩅ", "ꨎꩊ", "ꨚꨗꩍ",
    "ꨓꨝꨳꩀ", "ꨕꨨꨵꩀ", "ꨆꨔꨯꨱꩅ", "ꨝꨩ", "ꨨꨤꨭꩆ"
]
ANCHOR_WORDS_HEAD = [
    "ꨈꨴꨭꩀ", "ꨓꨶꨮꩉ", "ꨓꨝꨳꩀ", "ꨚꨗꨶꨮꩄ", "ꨧꨪ",
    "ꨔꩅ", "ꨋꩇ", "ꨤꨪꨠꩍ", "ꨥꩀ", "ꨣꨪꨡꩍ"
]

RARE_AND_CONFUSED_BASES = [
    'ꨁ', 'ꨂ', 'ꨃ', 'ꨄ', 'ꨅ', 'ꨉ', 'ꨊ', 'ꨍ', 'ꨏ', 'ꨐ',
    'ꨑ', 'ꨒ', 'ꨕ', 'ꨖ', 'ꨘ', 'ꨜ', 'ꨞ', 'ꨡ', 'ꨥ', 'ꨦ'
]


def to_cham_num(n):
    return "".join(CHAM_DIGITS[int(d)] for d in str(n))


def synthesize_short_phonotactic_word(allow_disyllabic=True):
    for _ in range(10):
        has_medial = random.random() < 0.48
        if has_medial:
            base = random.choice(CHAM_MEDIAL_ELIGIBLE_CONSONANTS + [CHAM_CARRIER_A])
            medial = random.choice(PRIORITY_MEDIALS)
            vowel = random.choice(MEDIAL_VOWEL_NUCLEI)
        else:
            if random.random() < 0.12:
                base = random.choice(CHAM_INDEPENDENT_VOWELS)
                medial = ''
                vowel = random.choice(['ꨰ', 'ꨯ', 'ꨩ', 'ꨭ', ''])
            else:
                base = random.choice(CHAM_TRUE_CONSONANTS + [CHAM_CARRIER_A])
                medial = ''
                vowel = random.choice(PRIORITY_VOWEL_NUCLEI)

        has_final = random.random() < 0.55
        final = random.choice(PRIORITY_FINALS) if has_final else ''

        if not medial and not vowel and not final:
            vowel = random.choice(PRIORITY_VOWEL_NUCLEI)

        main_syl = f"{base}{medial}{vowel}{final}"

        if allow_disyllabic and random.random() < 0.28:
            pre_c = random.choice(CHAM_TRUE_CONSONANTS + [CHAM_CARRIER_A])
            pre_v = random.choice(['', 'ꨪ', 'ꨭ', 'ꨩ', 'ꩌ'])
            main_syl = f"{pre_c}{pre_v}{main_syl}"

        w = normalize_unicode(main_syl)
        if w not in CORE_STANZA_SET:
            return w
    return normalize_unicode(main_syl)


def synthesize_tail_sensitive_w1():
    tails = [
        'ꨪ', 'ꨪ', 'ꨩ', 'ꨩ', 'ꨯꨮꩅ', 'ꨯꨱꩅ', 'ꩊ', 'ꩍ',
        'ꨳꩀ', 'ꨵꩀ', 'ꨭꩆ', 'ꨶꨮꩉ', 'ꨮꩃ', 'ꨰ'
    ]
    for _ in range(15):
        tail = random.choice(tails)
        if tail.startswith(('ꨳ', 'ꨵ', 'ꨶ')):
            c = random.choice(CHAM_MEDIAL_ELIGIBLE_CONSONANTS)
        else:
            c = random.choice(CHAM_TRUE_CONSONANTS)
        if random.random() < 0.35:
            pre_c = random.choice(['ꨀ', 'ꨆ', 'ꨓ', 'ꨚ', 'ꨕ', 'ꨨ', 'ꨝ'])
            pre_v = random.choice(['', 'ꨭ', 'ꨪ'])
            w = normalize_unicode(f"{pre_c}{pre_v}{c}{tail}")
        else:
            w = normalize_unicode(f"{c}{tail}")
        if w not in CORE_STANZA_SET:
            return w
    return synthesize_short_phonotactic_word(allow_disyllabic=False)


def synthesize_head_sensitive_w2():
    for _ in range(15):
        mode = random.randint(0, 3)
        if mode in (0, 1):
            c = random.choice(['ꨈ', 'ꨚ', 'ꨆ', 'ꨓ', 'ꨕ', 'ꨝ', 'ꨗ', 'ꨎ'])
            v = random.choice(['ꨭ', 'ꨪ', 'ꨯꨱ', 'ꨬ', 'ꨮ', 'ꨩ'])
            f = random.choice(['ꩀ', 'ꩃ', 'ꩍ', 'ꩅ', ''])
            w = normalize_unicode(f"{c}ꨴ{v}{f}")
        elif mode == 2:
            c = random.choice(['ꨓ', 'ꨚ', 'ꨎ', 'ꨨ', 'ꨢ', 'ꨆ'])
            m = random.choice(['ꨶ', 'ꨳ'])
            v = random.choice(['ꨮ', 'ꨰ', 'ꨯꨮ', ''])
            f = random.choice(['ꩉ', 'ꩀ', 'ꩄ', 'ꩆ', 'ꩃ'])
            w = normalize_unicode(f"{c}{m}{v}{f}")
        else:
            c = random.choice(CHAM_TRUE_CONSONANTS)
            v = random.choice(['ꨯꨮ', 'ꨯꨱ', 'ꨰ', 'ꨪ', 'ꨩ'])
            f = random.choice(['ꩅ', 'ꩍ', 'ꩀ', 'ꩇ', ''])
            w = normalize_unicode(f"{c}{v}{f}")
        if w not in CORE_STANZA_SET:
            return w
    return synthesize_short_phonotactic_word(allow_disyllabic=False)


def sample_pair_words(tail_head_boost=False):
    if tail_head_boost and random.random() < 0.72:
        r = random.random()
        if r < 0.36:
            w1 = random.choice(ANCHOR_WORDS_TAIL)
            w2 = synthesize_head_sensitive_w2()
        elif r < 0.72:
            w1 = synthesize_tail_sensitive_w1()
            w2 = random.choice(ANCHOR_WORDS_HEAD)
        else:
            w1 = synthesize_tail_sensitive_w1()
            w2 = synthesize_head_sensitive_w2()
        return w1, w2

    if random.random() < 0.55:
        w_lex = random.choice(CORE_STANZA_WORDS)
        w_syn = synthesize_short_phonotactic_word(allow_disyllabic=False)
        return (w_lex, w_syn) if random.random() < 0.5 else (w_syn, w_lex)
    else:
        return (
            synthesize_short_phonotactic_word(allow_disyllabic=False),
            synthesize_short_phonotactic_word(allow_disyllabic=False)
        )


def synthesize_rare_base_word():
    base = random.choice(RARE_AND_CONFUSED_BASES)
    if base in CHAM_INDEPENDENT_VOWELS:
        medial = ''
        vowel = random.choice(['ꨩ', 'ꨰ', 'ꨯ', 'ꨭ', ''])
    elif base in CHAM_MEDIAL_ELIGIBLE_CONSONANTS and random.random() < 0.40:
        medial = random.choice(PRIORITY_MEDIALS)
        vowel = random.choice(MEDIAL_VOWEL_NUCLEI)
    else:
        medial = ''
        vowel = random.choice(PRIORITY_VOWEL_NUCLEI)
    final = random.choice(PRIORITY_FINALS) if random.random() < 0.60 else ''
    if not medial and not vowel and not final:
        vowel = 'ꨩ'
    return normalize_unicode(f"{base}{medial}{vowel}{final}")


_PIL_FONT_CACHE = {}
_HB_FT_CACHE = {}


def get_pil_font(size=20, bold=False):
    key = (size, bold)
    if key in _PIL_FONT_CACHE:
        return _PIL_FONT_CACHE[key]
    font_name = "NotoSansCham-Bold.ttf" if bold else "NotoSansCham-Regular.ttf"
    font_path = find_asset_path(font_name)
    if font_path and os.path.exists(font_path):
        f = ImageFont.truetype(font_path, size)
        _PIL_FONT_CACHE[key] = f
        return f
    raise RuntimeError(f"Could not load mandatory Cham font: {font_name}")


def get_hb_ft(bold=False, font_size=24):
    key = (bold, font_size)
    if key not in _HB_FT_CACHE:
        font_name = "NotoSansCham-Bold.ttf" if bold else "NotoSansCham-Regular.ttf"
        font_path = find_asset_path(font_name)
        if not font_path or not os.path.exists(font_path):
            raise RuntimeError(f"Could not load mandatory Cham font: {font_name}")
        face = freetype.Face(font_path)
        face.set_char_size(font_size * 64)
        with open(font_path, "rb") as f:
            font_data = f.read()
        hb_face = hb.Face(font_data)
        hb_font = hb.Font(hb_face)
        hb_font.scale = (font_size * 64, font_size * 64)
        asc = face.size.ascender / 64.0
        desc = face.size.descender / 64.0
        _HB_FT_CACHE[key] = (face, hb_font, asc, desc)
    return _HB_FT_CACHE[key]


def render_hb_fixed_baseline_line(text, offset=0, double_hyphen_gap=0, target_height=48,
                                  pad_x_override=None, repeat_digit_gap=0):
    bold = (offset % 6 == 0) and (offset >= 10)
    f_sizes = (24, 22, 20)
    font_size = f_sizes[offset % 3]
    pad_x = int(pad_x_override) if pad_x_override is not None else (13 + (offset % 8))
    pad_y = 10 + (offset % 2)
    face, hb_font, ascender, descender = get_hb_ft(bold=bold, font_size=font_size)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf)

    cham6_gid = face.get_char_index(ord('꩖'))
    total_w = (sum(pos.x_advance / 64.0 for pos in buf.glyph_positions)
               + max(0, double_hyphen_gap)
               + (max(0, repeat_digit_gap) if "꩖꩖" in text else 0))
    line_h = int(math.ceil(ascender - descender)) + pad_y * 2
    line_w = int(math.ceil(total_w)) + pad_x * 2
    canvas = np.full((max(48, line_h), max(16, line_w)), 255, dtype=np.uint8)
    pen_x = float(pad_x)
    pen_y = pad_y + int(math.ceil(ascender))
    dash_seen = 0
    cham6_seen = 0

    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        gid = info.codepoint
        if double_hyphen_gap > 0 and "--" in text and gid in (10, 118):
            dash_seen += 1
            if dash_seen == 2:
                pen_x += float(double_hyphen_gap)
        if repeat_digit_gap > 0 and "꩖꩖" in text and cham6_gid and gid == cham6_gid:
            cham6_seen += 1
            if cham6_seen >= 2:
                pen_x += float(repeat_digit_gap)
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
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)), text


def render_pil_direct_h48(text, offset=0, double_hyphen_gap=0, target_height=48, pad_x_override=None):
    bold = (offset % 5 == 0) and (offset >= 10)
    font_size = 20
    font = get_pil_font(font_size, bold)
    y_origin = 7
    pad_left = int(pad_x_override) if pad_x_override is not None else (13 + (offset % 4))
    pad_right = int(pad_x_override) if pad_x_override is not None else (14 + (offset % 4))
    dummy = Image.new('RGB', (10, 10))
    d_dummy = ImageDraw.Draw(dummy)

    if double_hyphen_gap > 0 and "--" in text:
        idx = text.find("--")
        p1 = text[:idx + 1]
        p2 = text[idx + 1:]
        bb1 = d_dummy.textbbox((0, 0), p1, font=font)
        bb2 = d_dummy.textbbox((0, 0), p2, font=font)
        x0 = pad_left - bb1[0]
        x2 = x0 + bb1[2] + double_hyphen_gap - bb2[0]
        total_w = max(24, x2 + bb2[2] + pad_right)
        img = Image.new('RGB', (total_w, target_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((x0, y_origin), p1, font=font, fill=(0, 0, 0))
        draw.text((x2, y_origin), p2, font=font, fill=(0, 0, 0))
    else:
        bb = d_dummy.textbbox((0, 0), text, font=font)
        x0 = pad_left - bb[0]
        total_w = max(24, (bb[2] - bb[0]) + pad_left + pad_right)
        img = Image.new('RGB', (total_w, target_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((x0, y_origin), text, font=font, fill=(0, 0, 0))
    return img, text


def render_fixed_baseline_line(text, offset=0, double_hyphen_gap=0, target_height=48,
                               force_hb=False, pil_mod=10, pil_rem=9,
                               pad_x_override=None, repeat_digit_gap=0):
    use_pil = (not force_hb) and (offset % pil_mod == pil_rem)
    if not use_pil:
        return render_hb_fixed_baseline_line(
            text, offset=offset, double_hyphen_gap=double_hyphen_gap,
            target_height=target_height, pad_x_override=pad_x_override,
            repeat_digit_gap=repeat_digit_gap
        )
    return render_pil_direct_h48(
        text, offset=offset, double_hyphen_gap=double_hyphen_gap,
        target_height=target_height, pad_x_override=pad_x_override
    )


def sample_stream2_phonotactic_rare_item(offset):
    """
    Stream 2: Fixed-Baseline Short Phonotactic, Rare-Char & U+AA56 (`꩖`) vs U+AA23 (`ꨣ`) Minimal-Pair Hard.
    25% of Stream 2 targets `꩖` (Cham digit 6) vs `ꨣ` (ra) minimal pairs & stanza numbers (`2-6px` inter-digit gap).
    """
    for attempt in range(25):
        repeat_gap = 0
        if offset % 4 == 0:
            # Dedicated U+AA56 (꩖) vs U+AA23 (ꨣ) minimal pairs (no '-' so hyphen counts stay exact)
            w_tail = synthesize_short_phonotactic_word(allow_disyllabic=False)
            ra_syl = normalize_unicode(f"ꨣ{random.choice(['ꨩ', 'ꨪ', 'ꨭ', 'ꨰ', 'ꨯꨱ'])}{random.choice(['', 'ꩀ', 'ꩌ', 'ꩍ'])}")
            sub6 = ((offset // 4) + attempt) % 5
            if sub6 == 0:
                text = f"꩖꩞ {ra_syl} {w_tail}"
            elif sub6 == 1:
                num_str = random.choice(["꩑꩖", "꩒꩖", "꩓꩖", "꩖꩑", "꩖꩒", "꩖꩐", "꩑꩐꩖", "꩒꩐꩖"])
                text = f"{num_str}꩞ {ra_syl} {w_tail}"
            elif sub6 == 2:
                text = f"꩖꩖꩞ {ra_syl} {w_tail}"
                repeat_gap = 2 + ((offset + attempt) % 5)  # 2-6px inter-digit gap against CTC collapse
            elif sub6 == 3:
                d = random.choice([6, 16, 26])
                mo = random.choice([1, 3, 5, 6, 9, 11])
                y = random.randint(1920, 2023)
                text = f"{ra_syl} {to_cham_num(d)}/{to_cham_num(mo)}/{to_cham_num(y)}"
            else:
                ra_w2 = normalize_unicode(f"ꨣ{random.choice(['ꨩ', 'ꨮ', 'ꨪꩌ', 'ꨳꨪꩌ'])}")
                text = f"{ra_syl} {ra_w2} {w_tail}"
            img, _ = render_fixed_baseline_line(
                text, offset=offset + attempt, force_hb=True, repeat_digit_gap=repeat_gap
            )
            if img.width <= 320:
                return text, img
            continue

        mode = offset % 3
        if mode == 0:
            n_w = 2 if (offset % 2 == 0) else 3
            words = [synthesize_short_phonotactic_word(allow_disyllabic=(i == 0)) for i in range(n_w)]
            punct = random.choice(['', ' ꩝', ' ꩝꩝', ' ꩞'])
            text = normalize_unicode((" ".join(words) + punct).strip())
        elif mode == 1:
            rw1 = synthesize_rare_base_word()
            rw2 = synthesize_short_phonotactic_word(allow_disyllabic=False)
            sub = (offset // 3) % 6
            if sub == 0:
                num = to_cham_num(random.randint(10, 999))
                text = f"{num}꩞ {rw1} {rw2} ꩝"
            elif sub == 1:
                p_ch = random.choice([',', '.', ':', ';'])
                text = f"{rw1}{p_ch} {rw2} ꩝"
            elif sub == 2:
                text = f"{rw1} ({rw2}) ꩝"
            elif sub == 3:
                sp_ch = random.choice(['꩜', '꩟', 'ꩁ'])
                text = f"{sp_ch} {rw1} {rw2}" if sp_ch == '꩜' else f"{rw1} {sp_ch} {rw2} ꩝"
            else:
                rw3 = synthesize_rare_base_word()
                text = f"{rw1} {rw2} {rw3}"
            text = normalize_unicode(text.strip())
        else:
            n_syl = 3 if (offset % 2 == 0) else 4
            syls = [synthesize_rare_base_word() if (i % 2 == 0) else synthesize_short_phonotactic_word(allow_disyllabic=False) for i in range(n_syl)]
            text = normalize_unicode(" ".join(syls))

        img, _ = render_fixed_baseline_line(text, offset=offset + attempt)
        if img.width <= 320:
            return text, img

    text = normalize_unicode(f"{synthesize_rare_base_word()} {synthesize_short_phonotactic_word(allow_disyllabic=False)}")
    img, _ = render_fixed_baseline_line(text, offset=offset)
    return text, img


def sample_stream3_symbol_item(sym_group, offset):
    """
    Stream 3: Fixed-Baseline 14 Mandatory Symbols (W <= 320px).
    Diversifies `{}`, `_`, and `/` templates + 88% HarfBuzz / 12% Direct H=48 Pillow across sizes (20, 22, 24).
    """
    for attempt in range(25):
        use_boost = sym_group in ('_', '/', '{}')
        w1, w2 = sample_pair_words(tail_head_boost=use_boost)
        w3 = synthesize_short_phonotactic_word(allow_disyllabic=False)
        num = random.randint(100, 998) if sym_group in ('{}', '[]', '#') else random.randint(11, 998)
        m = offset + attempt

        if sym_group == '_':
            sub_u = m % 20
            if sub_u < 13:
                text = f"{w1}_{w2}"
            elif sub_u < 17:
                text = f"{w1}_{num}" if sub_u % 2 == 0 else f"{num}_{w2}"
            else:
                text = f"{w1}_{w2} {w3}"
        elif sym_group == '{}':
            if m % 2 == 0:
                sub_b = (m // 2) % 3
                if sub_b == 0:
                    text = f"{w1} {{{num}}} {w2}"
                elif sub_b == 1:
                    text = f"{{{num}}} {w1} {w2}"
                else:
                    text = f"{w1} {w2} {{{num}}}"
            else:
                sub_b = (m // 2) % 3
                if sub_b == 0:
                    text = f"{w1} {{{w2}}}"
                elif sub_b == 1:
                    text = f"{{{w1}}} {w2}"
                else:
                    text = f"{w1} {{{w2}}} {w3}"
        elif sym_group == '/':
            mod_s = m % 20
            if mod_s < 12:
                text = f"{w1}/{w2}"
            elif mod_s < 15:
                text = f"{w1} / {w2}"
            elif mod_s < 18:
                if mod_s % 2 == 0:
                    d, mo, y = random.randint(1, 28), random.randint(1, 12), random.randint(1910, 2023)
                    text = f"{d:02d}/{mo:02d}/{y} {w2}"
                else:
                    n1 = random.randint(11, 95)
                    n2 = random.randint(12, 99)
                    text = f"{n1}/{n2} {w2}"
            else:
                d = random.randint(1, 28)
                mo = random.randint(1, 12)
                y = random.randint(1910, 2023)
                text = f"{to_cham_num(d)}/{to_cham_num(mo)}/{to_cham_num(y)}"
        elif sym_group == '!':
            text = f"{w1} {w2} !" if m % 5 != 0 else f"{w1} {w2} {w3} !"
        elif sym_group == '?':
            text = f"{w1} {w2} ?" if m % 5 != 0 else f"{w1} {w2} {w3} ?"
        elif sym_group == '<':
            text = f"{w1} < {w2}" if m % 5 != 0 else f"{w1} < {w2} {w3}"
        elif sym_group == '>':
            text = f"{w1} > {w2}" if m % 5 != 0 else f"{w1} > {w2} {w3}"
        elif sym_group == '&':
            text = f"{w1} & {w2}" if m % 5 != 0 else f"{w1} & {w2} {w3}"
        elif sym_group == '[]':
            text = f"{w1} [{num}] {w2}" if m % 6 != 0 else f"[{num}] {w1} {w2}"
        elif sym_group == '#':
            text = f"{w1} #{num} {w2}" if m % 6 != 0 else f"#{num} {w1} {w2}"
        elif sym_group == '\\':
            text = f"{w1} \\ {w2}" if m % 5 != 0 else f"{w1}\\{w2}"
        elif sym_group == '*':
            text = f"* {w1} {w2} *" if m % 5 != 0 else f"{w1} * {w2}"
        else:
            text = f"{w1} {sym_group} {w2}"

        text = normalize_unicode(text.strip())
        # 88% HarfBuzz / 12% Direct H=48 Pillow across all symbol classes (~1 in 8)
        force_hb = False
        img, _ = render_fixed_baseline_line(text, offset=m, force_hb=force_hb, pil_mod=8, pil_rem=7)
        if img.width <= 320:
            return text, img

    w1 = synthesize_short_phonotactic_word(allow_disyllabic=False)
    w2 = synthesize_short_phonotactic_word(allow_disyllabic=False)
    text = normalize_unicode(f"{w1} {w2} ?")
    img, _ = render_fixed_baseline_line(text, offset=offset, force_hb=True)
    return text, img


TRAIN_SINGLE_DIGIT_PAIRS = [
    f"{d1}-{d2}"
    for d1 in range(1, 10)
    for d2 in range(1, 10)
    if d1 != d2 and not (d1 in (1, 3, 7) and d2 == d1 + 1) and (d1 + d2) % 7 != 0
]
VAL_SINGLE_DIGIT_PAIRS = [
    f"{d1}-{d2}"
    for d1 in range(1, 10)
    for d2 in range(1, 10)
    if d1 != d2 and not (d1 in (1, 3, 7) and d2 == d1 + 1) and (d1 + d2) % 7 == 0
]
assert len(TRAIN_SINGLE_DIGIT_PAIRS) == 60
assert len(VAL_SINGLE_DIGIT_PAIRS) == 9
assert not (set(TRAIN_SINGLE_DIGIT_PAIRS) & set(VAL_SINGLE_DIGIT_PAIRS))

FORBIDDEN_N1_2DIGIT = {12, 15, 20, 45, 89}
FORBIDDEN_Y1_4DIGIT = {1802, 1945, 1954, 1975, 1980, 1990, 2000, 2010, 2020, 2025}


def sample_stream4_hyphen_item(hyp_group, offset, is_val=False):
    """
    Stream 4: Fixed-Baseline 7-Group Multi-Context Hyphen (W <= 320px).
    - `double_hyphen`: 75% 0px gap + 25% 1-2px gap + 15% spaced/numeric `w1 -- w2` / `num1--num2`, 88% HB / 12% Pillow.
    - `number_number`: 150 standalone D-D with varied margins pad_x in {12, 18, 26, 34} (85% HB / 15% Pillow),
      40 D-DD/DD-D, 130 DD-DD with varied margins, 30 DDD-DDD, 30 contextual.
    - `line_beginning`: 75% `-{w1} {w2}` + 25% `- {w1} {w2}` across 88% HB / 12% Pillow.
    """
    pad_cycle = (12, 18, 26, 34)
    for attempt in range(25):
        use_boost = hyp_group in ('double_hyphen', 'word_word_tight', 'line_ending', 'line_beginning')
        w1, w2 = sample_pair_words(tail_head_boost=use_boost)
        dh_gap = 0
        pad_x_ov = None
        m = offset + attempt

        if hyp_group == 'double_hyphen':
            sub_dh = offset % 20
            if sub_dh in (17, 18):
                text = f"{w1} -- {w2}"
            elif sub_dh == 19:
                n1 = random.randint(11, 88)
                n2 = n1 + random.randint(2, 10)
                text = f"{n1}--{n2}"
            else:
                text = f"{w1}--{w2}"
            # 75% natural 0px HarfBuzz gap + 25% 1-2px gap across CTC stride phases
            if (offset % 4 == 3) and "--" in text and " -- " not in text:
                dh_gap = 1 + ((offset // 4) % 2)
                force_hb = True
            else:
                force_hb = False
            pad_x_ov = 12 + (m % 8)
        elif hyp_group == 'line_ending':
            text = f"{w1} {w2}-"
            force_hb = False
        elif hyp_group == 'line_beginning':
            if offset % 4 == 3:
                text = f"- {w1} {w2}"
            else:
                text = f"-{w1} {w2}"
            force_hb = False
        elif hyp_group == 'word_word_tight':
            text = f"{w1}-{w2}"
            force_hb = False
        elif hyp_group == 'word_word_spaced':
            text = f"{w1} - {w2}"
            force_hb = False
        elif hyp_group == 'number_number':
            pad_x_ov = pad_cycle[(offset + attempt) % len(pad_cycle)]
            if not is_val:
                if offset < 150:
                    text = TRAIN_SINGLE_DIGIT_PAIRS[offset % 60]
                    # Ensure distinct render params across repeats of the same D-D pair (3 passes max):
                    # pass 0 (0..59), pass 1 (60..119), pass 2 (120..149) always differ in pad_x + bold/size.
                    pass_id = offset // 60
                    pad_x_dd = pad_cycle[(offset + pass_id) % len(pad_cycle)] + pass_id
                    render_off = offset * 7 + pass_id * 11 + (1 if pass_id == 0 else 2)
                    # 85% HB / 15% Pillow (remainder 5, 11, 17 mod 20)
                    use_pil_dd = (offset % 20) in (5, 11, 17)
                    img, _ = render_fixed_baseline_line(
                        text, offset=render_off, force_hb=(not use_pil_dd),
                        pil_mod=1, pil_rem=0, pad_x_override=pad_x_dd
                    )
                    if img.width <= 320:
                        return text, img
                    continue
                elif offset < 190:
                    d1 = random.randint(1, 9)
                    d2 = random.randint(11, 98)
                    while d1 == 5 and d2 == 10:
                        d1 = random.randint(1, 9)
                    text = f"{d1}-{d2}" if (m % 3 != 0) else f"{d2}-{d1}"
                elif offset < 320:
                    n1 = random.randint(11, 96)
                    while n1 in FORBIDDEN_N1_2DIGIT:
                        n1 = random.randint(11, 96)
                    if m % 2 == 0:
                        n2 = min(99, n1 + random.randint(1, 25))
                    else:
                        n2 = random.randint(11, 99)
                        if n2 == n1:
                            n2 = min(99, n1 + 2)
                    text = f"{n1}-{n2}"
                elif offset < 350:
                    n1 = random.randint(102, 880)
                    n2 = n1 + random.randint(5, 110)
                    text = f"{n1}-{n2}"
                else:
                    if m % 2 == 0:
                        dp = random.choice(TRAIN_SINGLE_DIGIT_PAIRS)
                    else:
                        n1 = random.randint(11, 96)
                        while n1 in FORBIDDEN_N1_2DIGIT:
                            n1 = random.randint(11, 96)
                        n2 = min(99, n1 + random.randint(1, 19))
                        dp = f"{n1}-{n2}"
                    w_syn = synthesize_short_phonotactic_word(allow_disyllabic=False)
                    text = f"{dp} {w_syn}" if (offset % 2 == 0) else f"{w_syn} {dp}"
            else:
                v_idx = offset % 42
                if v_idx < 9 and attempt == 0:
                    text = VAL_SINGLE_DIGIT_PAIRS[v_idx]
                elif v_idx < 29:
                    n1 = random.randint(11, 96)
                    while n1 in FORBIDDEN_N1_2DIGIT:
                        n1 = random.randint(11, 96)
                    n2 = min(99, n1 + random.randint(1, 25))
                    text = f"{n1}-{n2}"
                else:
                    n1 = random.randint(102, 880)
                    n2 = n1 + random.randint(5, 110)
                    text = f"{n1}-{n2}"
            force_hb = False
        elif hyp_group == 'year_year':
            if m % 3 == 0:
                y1 = random.randint(1804, 1895)
            else:
                y1 = random.randint(1901, 2022)
            while y1 in FORBIDDEN_Y1_4DIGIT:
                y1 = random.randint(1805, 2022)
            y2 = min(2026, y1 + random.randint(1, 32))
            if m % 8 != 0:
                text = f"{y1}-{y2}"
            else:
                w_syn = synthesize_short_phonotactic_word(allow_disyllabic=False)
                text = f"{y1}-{y2} {w_syn}"
            force_hb = False
        else:
            text = f"{w1}-{w2}"
            force_hb = False

        text = normalize_unicode(text.strip())
        img, _ = render_fixed_baseline_line(
            text, offset=m, double_hyphen_gap=dh_gap,
            force_hb=force_hb, pil_mod=8, pil_rem=7, pad_x_override=pad_x_ov
        )
        if img.width <= 320:
            return text, img

    w_a = synthesize_short_phonotactic_word(allow_disyllabic=False)
    w_b = synthesize_short_phonotactic_word(allow_disyllabic=False)
    text = normalize_unicode(f"{w_a}-{w_b}")
    img, _ = render_fixed_baseline_line(text, offset=offset, force_hb=True)
    return text, img


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def generate_r2_1_7_dataset(seed=43):
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 80)
    print("🎨 CHAM-OCR V24.1 STAGE R2.1.7 SURGICAL GATE-CLOSURE DATASET GENERATION")
    print(f"Seed: {seed}")
    print("=" * 80)

    out_dir_a = os.path.join(TRAINING_DIR, "data", "cham_synthetic_r2_1_4c_control_continue")
    out_dir_b = os.path.join(TRAINING_DIR, "data", "cham_synthetic_r2_1_4c_combmark_replay")
    out_dir_d = os.path.join(TRAINING_DIR, "data", "cham_synthetic_r2_1_4d_phonotactic_combmark")
    out_dir_e = os.path.join(TRAINING_DIR, "data", "cham_synthetic_r2_1_4e_short_phonotactic_combmark")
    out_dir_7 = os.path.join(TRAINING_DIR, "data", "cham_synthetic_r2_1_7_gate_closure")

    imgs_e = os.path.join(out_dir_e, "images")
    imgs_7 = os.path.join(out_dir_7, "images")
    assert os.path.exists(imgs_e), f"Branch E reference dataset missing at {out_dir_e}"

    if os.path.exists(out_dir_7):
        shutil.rmtree(out_dir_7)
    os.makedirs(imgs_7, exist_ok=True)

    with open(os.path.join(out_dir_e, "train_label.txt"), "r", encoding="utf-8") as f:
        raw_e_tr = [line.rstrip("\r\n") for line in f]
    with open(os.path.join(out_dir_e, "val_label.txt"), "r", encoding="utf-8") as f:
        raw_e_va = [line.rstrip("\r\n") for line in f]

    assert len(raw_e_tr) == 18000
    assert len(raw_e_va) == 2000

    seen_texts = set()
    for d_dir in (out_dir_a, out_dir_b, out_dir_d, out_dir_e):
        for fn in ("train_label.txt", "val_label.txt"):
            p = os.path.join(d_dir, fn)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.rstrip("\r\n").split("\t", 1)
                        if len(parts) == 2 and parts[1]:
                            seen_texts.add(parts[1])

    # 1. Stream 2 (750 train + 84 val)
    print("[1/5] Generating Stream 2: 750 train + 84 val Short Phonotactic, Rare-Char & ꩖/ꨣ Pairs...")
    stream2_tr, stream2_va = [], []
    att2 = 0
    while len(stream2_tr) < 750:
        att2 += 1
        t, img = sample_stream2_phonotactic_rare_item(len(stream2_tr) + att2 * 0)
        # Vary offset gently if collision occurs while keeping offset % 4 alternation for ꩖/ꨣ
        if t in seen_texts or img.width > 320:
            t, img = sample_stream2_phonotactic_rare_item(len(stream2_tr) + att2 * 4)
        if t and t not in seen_texts and img.width <= 320:
            seen_texts.add(t)
            stream2_tr.append((t, img))
    att2v = 0
    while len(stream2_va) < 84:
        att2v += 1
        t, img = sample_stream2_phonotactic_rare_item(10000 + len(stream2_va) + att2v * 4)
        if t and t not in seen_texts and img.width <= 320:
            seen_texts.add(t)
            stream2_va.append((t, img))

    # 2. Stream 3 (2,250 train + 250 val)
    print("[2/5] Generating Stream 3: 2,250 train + 250 val 14 Mandatory Symbols (diversified {}, _, /)...")
    sym_tr_quotas = [
        ('_', 380), ('{}', 380), ('/', 260), ('?', 200), ('!', 180),
        ('<', 160), ('&', 150), ('[]', 150), ('#', 130), ('\\', 130),
        ('*', 65), ('>', 65)
    ]
    sym_va_quotas = [
        ('_', 42), ('{}', 42), ('/', 28), ('?', 22), ('!', 20),
        ('<', 18), ('&', 16), ('[]', 16), ('#', 15), ('\\', 15),
        ('*', 8), ('>', 8)
    ]
    assert sum(q for _, q in sym_tr_quotas) == 2250
    assert sum(q for _, q in sym_va_quotas) == 250

    stream3_tr, stream3_va = [], []
    for s_grp, q_cnt in sym_tr_quotas:
        c = 0
        att = 0
        while c < q_cnt:
            att += 1
            t, img = sample_stream3_symbol_item(s_grp, c + att * 7)
            if t and t not in seen_texts and img.width <= 320:
                seen_texts.add(t)
                stream3_tr.append((t, img))
                c += 1

    for s_grp, q_cnt in sym_va_quotas:
        c = 0
        att = 0
        while c < q_cnt:
            att += 1
            t, img = sample_stream3_symbol_item(s_grp, 20000 + c + att * 7)
            if t and t not in seen_texts and img.width <= 320:
                seen_texts.add(t)
                stream3_va.append((t, img))
                c += 1

    random.shuffle(stream3_tr)
    random.shuffle(stream3_va)

    # 3. Stream 4 (1,500 train + 166 val)
    print("[3/5] Generating Stream 4: 1,500 train + 166 val 7-Group Multi-Context Hyphen...")
    hyp_tr_quotas = [
        ('double_hyphen', 500),
        ('number_number', 380),
        ('line_beginning', 140),
        ('word_word_tight', 140),
        ('line_ending', 120),
        ('word_word_spaced', 110),
        ('year_year', 110)
    ]
    hyp_va_quotas = [
        ('double_hyphen', 56),
        ('number_number', 42),
        ('line_beginning', 16),
        ('word_word_tight', 15),
        ('line_ending', 13),
        ('word_word_spaced', 12),
        ('year_year', 12)
    ]
    assert sum(q for _, q in hyp_tr_quotas) == 1500
    assert sum(q for _, q in hyp_va_quotas) == 166

    stream4_tr, stream4_va = [], []
    for h_grp, q_cnt in hyp_tr_quotas:
        c = 0
        att = 0
        while c < q_cnt:
            att += 1
            if h_grp == 'number_number' and c < 150:
                # Guarantee each standalone D-D variant renders with distinct (pad_x, font_size/stride, jpeg_q)
                t, img = sample_stream4_hyphen_item(h_grp, c, is_val=False)
                seen_texts.add(t)
                stream4_tr.append((t, img))
                c += 1
            else:
                t, img = sample_stream4_hyphen_item(h_grp, c if att == 1 else (c + att * 19), is_val=False)
                if t and t not in seen_texts and img.width <= 320:
                    seen_texts.add(t)
                    stream4_tr.append((t, img))
                    c += 1

    for h_grp, q_cnt in hyp_va_quotas:
        c = 0
        att = 0
        while c < q_cnt:
            att += 1
            if h_grp == 'number_number' and c < 9 and att == 1:
                t, img = sample_stream4_hyphen_item(h_grp, c, is_val=True)
                if t and t not in seen_texts and img.width <= 320:
                    seen_texts.add(t)
                    stream4_va.append((t, img))
                    c += 1
                else:
                    # Fallback to DD-DD if single-digit val pair already appeared
                    t, img = sample_stream4_hyphen_item(h_grp, 20 + c + att * 19, is_val=True)
                    if t and t not in seen_texts and img.width <= 320:
                        seen_texts.add(t)
                        stream4_va.append((t, img))
                        c += 1
            else:
                t, img = sample_stream4_hyphen_item(h_grp, c if att == 1 else (30000 + c + att * 19), is_val=True)
                if t and t not in seen_texts and img.width <= 320:
                    seen_texts.add(t)
                    stream4_va.append((t, img))
                    c += 1

    random.shuffle(stream4_tr)
    random.shuffle(stream4_va)

    replaced_tr = stream2_tr + stream3_tr + stream4_tr
    replaced_va = stream2_va + stream3_va + stream4_va
    assert len(replaced_tr) == 4500
    assert len(replaced_va) == 500

    # 4. Assemble Train (18,000 samples) & Val (2,000 samples)
    print("[4/5] Assembling Stage R2.1.7 Train (18,000) & Val (2,000)...")
    lines_7_tr = []

    for idx in range(9900):
        src_name = f"r2_1_4e_sphono_tr_{idx:07d}.jpg"
        dst_name = f"r2_1_7_gclosure_tr_{idx:07d}.jpg"
        shutil.copy2(os.path.join(imgs_e, src_name), os.path.join(imgs_7, dst_name))
        lbl = raw_e_tr[idx].split("\t", 1)[1]
        lines_7_tr.append(f"images/{dst_name}\t{lbl}\n")

    for offset, (text, img) in enumerate(replaced_tr):
        idx = 9900 + offset
        q = 87 + (offset % 9)
        dst_name = f"r2_1_7_gclosure_tr_{idx:07d}.jpg"
        dst_path = os.path.join(imgs_7, dst_name)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        with open(dst_path, "wb") as f:
            f.write(buf.getvalue())
        lines_7_tr.append(f"images/{dst_name}\t{text}\n")

    for idx in range(14400, 18000):
        src_name = f"r2_1_4e_sphono_tr_{idx:07d}.jpg"
        dst_name = f"r2_1_7_gclosure_tr_{idx:07d}.jpg"
        shutil.copy2(os.path.join(imgs_e, src_name), os.path.join(imgs_7, dst_name))
        lbl = raw_e_tr[idx].split("\t", 1)[1]
        lines_7_tr.append(f"images/{dst_name}\t{lbl}\n")

    lines_7_va = []
    for idx in range(1100):
        src_name = f"r2_1_4e_sphono_va_{idx:07d}.jpg"
        dst_name = f"r2_1_7_gclosure_va_{idx:07d}.jpg"
        shutil.copy2(os.path.join(imgs_e, src_name), os.path.join(imgs_7, dst_name))
        lbl = raw_e_va[idx].split("\t", 1)[1]
        lines_7_va.append(f"images/{dst_name}\t{lbl}\n")

    for offset, (text, img) in enumerate(replaced_va):
        idx = 1100 + offset
        q = 87 + (offset % 9)
        dst_name = f"r2_1_7_gclosure_va_{idx:07d}.jpg"
        dst_path = os.path.join(imgs_7, dst_name)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        with open(dst_path, "wb") as f:
            f.write(buf.getvalue())
        lines_7_va.append(f"images/{dst_name}\t{text}\n")

    for idx in range(1600, 2000):
        src_name = f"r2_1_4e_sphono_va_{idx:07d}.jpg"
        dst_name = f"r2_1_7_gclosure_va_{idx:07d}.jpg"
        shutil.copy2(os.path.join(imgs_e, src_name), os.path.join(imgs_7, dst_name))
        lbl = raw_e_va[idx].split("\t", 1)[1]
        lines_7_va.append(f"images/{dst_name}\t{lbl}\n")

    tr_lbl_path = os.path.join(out_dir_7, "train_label.txt")
    va_lbl_path = os.path.join(out_dir_7, "val_label.txt")
    with open(tr_lbl_path, "w", encoding="utf-8") as f:
        f.writelines(lines_7_tr)
    with open(va_lbl_path, "w", encoding="utf-8") as f:
        f.writelines(lines_7_va)

    all_tr_chars = set("".join(l.split("\t", 1)[1].strip() for l in lines_7_tr))
    missing_dict_chars = [ch for ch in DICT_CHARS_V24_1 if ch not in all_tr_chars]
    print(f"  • Train Dictionary Coverage: {114 - len(missing_dict_chars)}/114 tokens (Missing: {missing_dict_chars})")
    assert len(missing_dict_chars) == 0, f"Missing dictionary tokens in train set: {missing_dict_chars}"

    tr_sha = file_sha256(tr_lbl_path)
    va_sha = file_sha256(va_lbl_path)

    manifest = {
        "stage": "R2.1.7-SURGICAL-GATE-CLOSURE",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "output_dir": out_dir_7,
        "train_samples": len(lines_7_tr),
        "val_samples": len(lines_7_va),
        "train_label_sha256": tr_sha,
        "val_label_sha256": va_sha,
        "dict_coverage": "114/114",
        "streams": {
            "stream_1_clean_stanza_shared_with_e": {"train_range": [0, 9899], "val_range": [0, 1099], "total": 11000, "pct": 55.0},
            "stream_2_fixed_baseline_phonotactic_rare_u_aa56_aa23": {"train_range": [9900, 10649], "val_range": [1100, 1183], "total": 834, "pct": 4.17},
            "stream_3_fixed_baseline_14_symbols_gate_closure": {"train_range": [10650, 12899], "val_range": [1184, 1433], "total": 2500, "pct": 12.50},
            "stream_4_fixed_baseline_7group_hyphen_gate_closure": {"train_range": [12900, 14399], "val_range": [1434, 1599], "total": 1666, "pct": 8.33},
            "stream_5_short_phonotactic_combmark_shared_with_e": {"train_range": [14400, 17099], "val_range": [1600, 1899], "total": 3000, "pct": 15.0},
            "stream_6_blank_guard_shared_with_e": {"train_range": [17100, 17999], "val_range": [1900, 1999], "total": 1000, "pct": 5.0}
        }
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    manifest_path = os.path.join(REPORTS_DIR, "r2_1_7_gate_closure_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[5/5] Stage R2.1.7 Dataset Generated Successfully!")
    print(f"  • Train Label SHA256: {tr_sha}")
    print(f"  • Val Label SHA256:   {va_sha}")
    print(f"  • Manifest saved to:  {manifest_path}")


if __name__ == "__main__":
    generate_r2_1_7_dataset(seed=43)
