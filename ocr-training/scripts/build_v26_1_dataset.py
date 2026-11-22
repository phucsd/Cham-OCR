#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds the V26.1 Masked-SVTR dataset (`ocr-training/data/cham_synthetic_v26_1`) at H=48, W<=480:
  - Base replay: all 43,520 train / 4,352 val audited images from `cham_synthetic_v26_ft`
  - Fresh HarfBuzz generation (`NotoSansCham-*.ttf` ONLY, 0 OOD/EFEO/heldout fonts):
      20,480 train / 2,048 val across 4 targeted streams + Paleographic Calligraphic/Elastic Augmentation:
        * Stream A (30% = 6,144 / 614): Stanza 1..999 (repeated-digit 2-8px gaps + breaks) &
          Double Danda `꩝꩝` (2-8px gap + blur), with 35% paleographic oblique/nib/elastic augmentation.
        * Stream B (30% = 6,144 / 614): >=5,000 hard minimal pairs for `ꨲ` vs `ꨶ` on `ꨀ/ꨓ/ꨚ/ꨆ`,
          all 9 AGENTS.md minimal pairs (`ꨕ/ꨊ`, `ꨔ/ꨝ/ꨟ`, `ꨥ/ꨡ`, `ꨈ/ꨓ`, `ꨤ/ꨈ`, `ꨪ/ꨫ`, `ꩅ/ꩊ`, `꩝/ꩃ`),
          and multi-tier stacks (`ꨳ` + `ꨪ`/`ꩌ` e.g. `ꨣꨳꨪꩌ`), with 40% paleographic folio augmentation.
        * Stream C (25% = 5,120 / 512): Crisp Gate-Closure targets — single `*` envelope/infix,
          ultra-short `<140px` `D-D`/`DD-DD`/`DDD-DDD` & zero-preserving `YYYY-YYYY`,
          contrastive `w1-w2` vs `w1--w2`, and 14 mandatory symbols (`_`, `/`, etc.).
        * Stream D (15% = 3,072 / 308): Wide manuscript/folio Cham lines with 45% paleographic
          calligraphic/bamboo-nib/elastic stroke augmentation + optical/erosion/blur/lowres + 15% blanks `""`.
  - Totals: 64,000 train (= 1,000 steps/epoch at 32/card x 2 GPUs) & 6,400 val, 0 SHA-256 overlap,
    0 train/val text overlap on fresh data, 0 benchmark contamination, 114/114 token coverage.
"""

import os
import sys
import math
import json
import time
import random
import shutil
import hashlib
from collections import Counter
from datetime import datetime, timezone

import cv2
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

from generate_data_v24_1_r2_1_6 import (  # noqa: E402
    find_asset_path,
)
from generate_data_v24_1_r3_1 import (  # noqa: E402
    DICT_SET,
    ILLEGAL_MEDIAL_RE,
    CHAM_TRUE_CONSONANTS,
    FORBIDDEN_N1_2DIGIT,
    FORBIDDEN_N1_3DIGIT,
    FORBIDDEN_Y1_4DIGIT,
    TRAIN_SINGLE_DIGIT_PAIRS,
    VAL_SINGLE_DIGIT_PAIRS,
    CORPUS_WORDS_SORTED,
    normalize_unicode,
    to_cham_num,
    render_hb_line,
    synth_fresh_monosyllable,
    synth_sesquisyllabic_word,
    _make_symbol_words,
    degrade_erosion_dry_ink,
    degrade_severe_motion,
    degrade_heavy_gauss_noise_jpeg,
    degrade_extreme_lowres,
    degrade_persp_dilate_shadow,
    _save_unique_jpeg,
)
from generate_data_v24_1_r4_w480 import (  # noqa: E402
    compose_wide_corpus_line,
    degrade_book_camera_optical,
    generate_wide_blank,
)
from generate_data_v24_1_r4_2_inktight import (  # noqa: E402
    FORBIDDEN_EXACT_TEXTS,
    render_hb_ink_tight,
    sample_r4_2_all_symbols,
    sample_r4_2_hyphens,
)

DATA_ROOT = os.path.join(TRAINING_DIR, "data")
V26_FT_DIR = os.path.join(DATA_ROOT, "cham_synthetic_v26_ft")
OUT_DIR = os.path.join(DATA_ROOT, "cham_synthetic_v26_1")

FORBIDDEN_COMPACT_TEXTS = {t.replace(" ", "") for t in FORBIDDEN_EXACT_TEXTS if t}


def verify_fonts_strictly_in_domain():
    fonts_dir = os.path.abspath(os.path.join(TRAINING_DIR, "data", "fonts"))
    for fname in ("NotoSansCham-Regular.ttf", "NotoSansCham-Bold.ttf", "NotoSansCham-Black.ttf"):
        p = find_asset_path(fname)
        assert p and os.path.exists(p), f"Missing Cham training font: {fname}"
        ap = os.path.abspath(p)
        assert ap.startswith(fonts_dir), f"Out-of-domain font resolved: {ap}"
        low = ap.lower()
        assert "efeo" not in low and "heldout" not in low, f"Contaminated font path: {ap}"
    return fonts_dir


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest().lower()


def _is_safe_text(txt: str, forbidden_set: set) -> bool:
    if not txt or len(txt) > 95:
        return False
    nt = normalize_unicode(txt.strip())
    if nt in forbidden_set or nt in FORBIDDEN_EXACT_TEXTS:
        return False
    if nt.replace(" ", "") in FORBIDDEN_COMPACT_TEXTS:
        return False
    if ILLEGAL_MEDIAL_RE.search(nt):
        return False
    if not all(c in (DICT_SET | {" "}) for c in nt):
        return False
    return True


def degrade_paleographic_calligraphic_folio(img: Image.Image, rng: random.Random, np_rng) -> Image.Image:
    """
    In-domain paleographic augmentation on NotoSansCham renders (0 OOD fonts):
      1. Oblique calligraphic slant (`sh in [-0.20, +0.20]`, centered at H/2).
      2. Directional bamboo-pen nib stroke modulation (anisotropic 1x2 / 2x1 / diagonal kernel).
      3. Smooth low-frequency 2D elastic baseline & stroke wave (`cv2.remap`).
      4. Warm antique parchment / palm-leaf fiber toning & light optical softness.
    """
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # 1. Oblique calligraphic shear
    sh = rng.uniform(-0.20, 0.20)
    if abs(sh) > 0.025:
        mat = np.array([[1.0, sh, -sh * (h * 0.5)], [0.0, 1.0, 0.0]], dtype=np.float32)
        gray = cv2.warpAffine(gray, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 2. Directional bamboo-nib anisotropic stroke modulation
    ink = 255 - gray
    nib_mode = rng.randint(0, 4)
    if nib_mode == 0:
        kern = np.array([[1, 1]], dtype=np.uint8)  # horizontal broad nib
        ink = cv2.dilate(ink, kern, iterations=1)
    elif nib_mode == 1:
        kern = np.array([[1], [1]], dtype=np.uint8)  # vertical downstroke nib
        ink = cv2.dilate(ink, kern, iterations=1)
    elif nib_mode == 2:
        kern = np.array([[1, 0], [0, 1]], dtype=np.uint8)  # 45-deg reed nib
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kern)
    elif nib_mode == 3:
        kern = np.array([[0, 1], [1, 0]], dtype=np.uint8)  # 135-deg reed nib + dry-ink fade
        ink = np.clip(ink.astype(np.float32) * rng.uniform(0.76, 0.94), 0, 255).astype(np.uint8)
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kern)
    else:
        # Thin dry calligraphic ink
        ink = cv2.erode(ink, np.ones((2, 1), dtype=np.uint8), iterations=1)

    # 3. Smooth low-frequency 2D elastic stroke & baseline wave
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    ax = rng.uniform(0.45, 1.55)
    ay = rng.uniform(0.55, 1.75)
    lx = rng.uniform(38.0, 78.0)
    ly = rng.uniform(22.0, 44.0)
    ph1 = rng.uniform(0.0, 2.0 * math.pi)
    ph2 = rng.uniform(0.0, 2.0 * math.pi)
    map_x = np.broadcast_to(xs, (h, w)) + ax * np.sin(2.0 * math.pi * ys / ly + ph1)
    map_y = np.broadcast_to(ys, (h, w)) + ay * np.sin(2.0 * math.pi * xs / lx + ph2)
    ink_w = cv2.remap(ink, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 4. Warm regional folio / palm-leaf paper tint + fiber grain + gentle optical blur
    alpha_f = np.clip(ink_w.astype(np.float32) / 255.0, 0.0, 1.0)[:, :, None]
    paper_r = rng.uniform(208.0, 240.0)
    paper_g = paper_r - rng.uniform(8.0, 22.0)
    paper_b = paper_g - rng.uniform(10.0, 26.0)
    paper = np.empty((h, w, 3), dtype=np.float32)
    paper[:, :, 0] = paper_r
    paper[:, :, 1] = paper_g
    paper[:, :, 2] = paper_b
    veins = np_rng.normal(0.0, rng.uniform(2.5, 5.5), size=(h, max(4, w // 36))).astype(np.float32)
    veins = cv2.resize(veins, (w, h), interpolation=cv2.INTER_LINEAR)
    paper += veins[:, :, None]

    ink_lvl = rng.uniform(18.0, 62.0)
    ink_rgb = np.array([ink_lvl + rng.uniform(4.0, 14.0), ink_lvl + rng.uniform(1.0, 7.0), ink_lvl], dtype=np.float32)[None, None, :]
    comp = (1.0 - alpha_f) * paper + alpha_f * ink_rgb
    comp += np_rng.normal(0.0, rng.uniform(1.5, 4.0), size=comp.shape)
    out = np.clip(comp, 0, 255).astype(np.uint8)
    if rng.random() < 0.65:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.35, 0.78))
    return Image.fromarray(out)


def _mild_stroke_break_or_blur(img: Image.Image, rng: random.Random, np_rng) -> Image.Image:
    """Applies mild stroke-break / micro-blur noise to challenge CTC collapse on repeated digits & ꩝꩝."""
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    mode = rng.randint(0, 3)
    if mode == 0:
        k = rng.choice([3, 5])
        sig = rng.uniform(0.55, 0.95)
        arr = cv2.GaussianBlur(arr, (k, k), sig)
    elif mode == 1:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        er = 255 - cv2.erode(255 - gray, np.ones((2, 2), dtype=np.uint8), iterations=1)
        h, w = er.shape
        x_end = min(w - 2, max(20, int(w * 0.35)))
        for _ in range(rng.randint(1, 2)):
            rx = rng.randint(6, x_end)
            ry = rng.randint(10, h - 10)
            er[ry:min(h - 2, ry + 2), rx:min(w - 1, rx + 2)] = rng.randint(185, 245)
        arr = cv2.cvtColor(er, cv2.COLOR_GRAY2RGB)
    elif mode == 2:
        arr = np.clip(arr.astype(np.float32) * rng.uniform(0.92, 1.04) + np_rng.normal(0, 3.5, arr.shape), 0, 255).astype(np.uint8)
    else:
        return degrade_erosion_dry_ink(img, rng, np_rng)
    return Image.fromarray(arr)


def sample_stream_a_stanza_danda(idx: int, is_val: bool, rng: random.Random, np_rng, forbidden_set: set):
    """Stream A (30%): Stanza 1..999 (repeated-digit 2-8px gaps + breaks) & Double Danda `꩝꩝` + 35% paleographic."""
    repeated_nums = [
        11, 22, 33, 44, 55, 66, 77, 88, 99,
        100, 110, 111, 112, 122, 197, 199, 200, 201, 202, 209,
        211, 220, 222, 300, 303, 333, 400, 444, 500, 555,
        600, 666, 700, 777, 800, 888, 900, 999,
    ]
    base_off = (17500000 if is_val else 16000000) + idx * 29

    for att in range(80):
        m = base_off + att * 17
        sub = idx % 10
        fsize = rng.choice([20, 22, 24, 24, 26])
        rep_gap = rng.randint(2, 8)
        use_double_danda = (sub >= 7)

        w1 = CORPUS_WORDS_SORTED[(idx * 5 + att * 11) % len(CORPUS_WORDS_SORTED)]
        w2 = synth_fresh_monosyllable(rng, force_pre=(m % 2 == 0), force_bindu=(m % 5 == 0))
        w3 = synth_sesquisyllabic_word(rng, dual_prevowel=(m % 4 == 1))
        w4 = synth_fresh_monosyllable(rng, force_aa33=(m % 3 == 0))

        if not use_double_danda:
            if sub in (0, 1, 2):
                n_val = repeated_nums[(idx + att) % len(repeated_nums)]
            else:
                slot_k = (idx // 10) * 4 + (sub - 3)
                n_val = 1 + (slot_k % 999)
            cnum = to_cham_num(n_val)
            txt = f"{cnum}꩞ {w1} {w2} {w3}" if (m % 2 == 0) else f"{cnum}꩞ {w2} {w1} {w4}"
            if m % 7 == 0:
                txt = txt + " ꩝"
        else:
            txt = f"{w1} {w2} {w3} ꩝꩝" if (m % 3 != 0) else f"{w2} {w1} ꩝꩝ {w4}"

        txt = normalize_unicode(txt.strip())
        if _is_safe_text(txt, forbidden_set):
            img = render_hb_line(txt, offset=m, font_size_override=fsize, repeat_digit_gap=rep_gap)
            if img.height == 48 and img.width <= 480:
                r_mod = idx % 20
                if r_mod < 7:
                    img = degrade_paleographic_calligraphic_folio(img, rng, np_rng)
                elif r_mod < 13:
                    img = _mild_stroke_break_or_blur(img, rng, np_rng)
                forbidden_set.add(txt)
                return txt, img, (r_mod < 7)
    raise RuntimeError(f"Stream A failed at idx={idx}")


def sample_stream_b_minimal_pairs_multitier(idx: int, is_val: bool, rng: random.Random, np_rng, forbidden_set: set):
    """Stream B (30%): >=5,000 hard minimal pairs (`ꨲ` vs `ꨶ`, all 9 pairs) + multi-tier stacks + 40% paleographic."""
    au_o_bases = ["ꨀ", "ꨓ", "ꨚ", "ꨆ"]
    agent_pairs = [
        ("ꨲ", "ꨶ"),
        ("ꨕ", "ꨊ"),
        ("ꨔ", "ꨟ"),
        ("ꨝ", "ꨟ"),
        ("ꨥ", "ꨡ"),
        ("ꨈ", "ꨓ"),
        ("ꨤ", "ꨈ"),
        ("ꨪ", "ꨫ"),
        ("ꩅ", "ꩊ"),
        ("꩝", "ꩃ"),
    ]
    base_off = (18500000 if is_val else 17000000) + idx * 31

    for att in range(80):
        m = base_off + att * 19
        fsize = rng.choice([20, 22, 24, 24, 26])
        coda = rng.choice(["", "", "ꩀ", "ꩆ", "ꩅ", "ꩉ", "ꩌ", "ꩃ"])

        b_au = au_o_bases[(idx + att) % len(au_o_bases)]
        b_o = au_o_bases[(idx + att + (m % 3)) % len(au_o_bases)]
        if m % 2 == 0:
            w_au = normalize_unicode(f"{b_au}ꨲ{coda}")
            w_o = normalize_unicode(f"{b_o}ꨶ{rng.choice(['ꨮ', 'ꨮꩉ', 'ꩆ', 'ꩀ', ''])}")
        else:
            w_au = normalize_unicode(f"{b_au}ꨲ{rng.choice(['ꩆ', 'ꩀ', 'ꨮ', ''])}")
            w_o = normalize_unicode(f"{b_o}ꨶ{coda}")

        pair_a, pair_b = agent_pairs[(idx * 3 + att) % len(agent_pairs)]
        if (idx % 4) == 0:
            c_stk = rng.choice(["ꨣ", "ꨆ", "ꨓ", "ꨚ", "ꨕ", "ꨧ", "ꨨ", "ꨝ"])
            w_extra1 = normalize_unicode(f"{c_stk}ꨳ{rng.choice(['ꨪ', 'ꨫ', 'ꨪ'])}ꩌ")
            w_extra2 = synth_fresh_monosyllable(rng, force_aa33=True, force_bindu=True)
        elif pair_a in ("ꨪ", "ꨲ"):
            ca = rng.choice(["ꨀ", "ꨓ", "ꨚ", "ꨆ", "ꨤ", "ꨕ"])
            cb = rng.choice(["ꨀ", "ꨓ", "ꨚ", "ꨆ", "ꨧ", "ꨝ"])
            w_extra1 = normalize_unicode(f"{ca}{pair_a}{rng.choice(['', 'ꩀ', 'ꩆ'])}")
            w_extra2 = normalize_unicode(f"{cb}{pair_b}{rng.choice(['', 'ꩀ', 'ꩆ', 'ꨮꩉ'])}")
        elif pair_a in ("ꩅ", "꩝"):
            ca = rng.choice(CHAM_TRUE_CONSONANTS)
            cb = rng.choice(CHAM_TRUE_CONSONANTS)
            if pair_a == "ꩅ":
                w_extra1 = normalize_unicode(f"{ca}ꨯꨮꩅ")
                w_extra2 = normalize_unicode(f"{cb}ꨶꨮꩊ")
            else:
                w_extra1 = normalize_unicode(f"{ca}ꨩꩃ")
                w_extra2 = f"{synth_fresh_monosyllable(rng)} ꩝"
        else:
            va = rng.choice(["ꨩ", "ꨪ", "ꨮ", "ꨭ", "ꨰ"])
            vb = rng.choice(["ꨩ", "ꨪ", "ꨮ", "ꨭ", "ꨰ"])
            w_extra1 = normalize_unicode(f"{pair_a}{va}{rng.choice(['', 'ꩀ', 'ꩆ', 'ꩌ'])}")
            w_extra2 = normalize_unicode(f"{pair_b}{vb}{rng.choice(['', 'ꩅ', 'ꩉ', 'ꩍ'])}")

        cw = CORPUS_WORDS_SORTED[(idx * 7 + att) % len(CORPUS_WORDS_SORTED)]
        parts = [w_au, w_o, w_extra1, cw, w_extra2] if (m % 2 == 0) else [w_extra1, w_au, cw, w_o, w_extra2]
        if m % 5 == 0:
            parts = parts[:4]
        txt = normalize_unicode(" ".join(parts).strip())
        if _is_safe_text(txt, forbidden_set):
            pad_y_ov = rng.choice([6, 7, 9, 10]) if "ꩌ" in txt else None
            img = render_hb_line(txt, offset=m, font_size_override=fsize, pad_y_override=pad_y_ov)
            if img.height == 48 and img.width <= 480:
                r_mod = idx % 10
                if r_mod < 4:
                    img = degrade_paleographic_calligraphic_folio(img, rng, np_rng)
                elif r_mod in (4, 5):
                    img = degrade_book_camera_optical(img, rng, np_rng)
                forbidden_set.add(txt)
                return txt, img, (r_mod < 4)
    raise RuntimeError(f"Stream B failed at idx={idx}")


def _sample_single_asterisk(idx: int, rng: random.Random, np_rng, forbidden_set: set):
    base_off = 19100000 + idx * 17
    for att in range(80):
        m = base_off + att * 13
        w1, w2, _ = _make_symbol_words(rng)
        cw = CORPUS_WORDS_SORTED[(idx * 5 + att) % len(CORPUS_WORDS_SORTED)]
        mode = (idx + att) % 5
        if mode in (0, 1):
            txt = f"* {w1} {w2} *"
        elif mode == 2:
            txt = f"* {cw} {w1} {w2} *"
        elif mode == 3:
            txt = f"* {w1} {cw}"
        else:
            txt = f"{w1} * {w2}"
        txt = normalize_unicode(txt.strip())
        if _is_safe_text(txt, forbidden_set):
            fsize = rng.choice([20, 22, 24, 26])
            if m % 3 != 0:
                img = render_hb_ink_tight(txt, offset=m, pad_x_ink=6 + (m % 6), pad_y_top=5 + (m % 3), pad_y_bot=5 + (m % 3), weight_override=("Bold" if m % 6 == 0 else "Regular"), font_size=fsize)
            else:
                img = render_hb_line(txt, offset=m, pad_x_override=8 + (m % 10), font_size_override=fsize)
            if img.height == 48 and img.width <= 480:
                if (idx % 5) == 4:
                    img = _mild_stroke_break_or_blur(img, rng, np_rng)
                forbidden_set.add(txt)
                return txt, img, False
    raise RuntimeError(f"Single-asterisk synthesis failed at idx={idx}")


def _sample_short_number_hyphen(idx: int, is_val: bool, rng: random.Random, np_rng, forbidden_set: set):
    base_off = (19600000 if is_val else 19300000) + idx * 23
    for att in range(100):
        m = base_off + att * 17
        w_ctx = synth_fresh_monosyllable(rng)
        sub = (idx + att) % 4
        if sub == 0:
            pairs = VAL_SINGLE_DIGIT_PAIRS if is_val else TRAIN_SINGLE_DIGIT_PAIRS
            base = pairs[(idx + att) % len(pairs)]
            txt = base if base not in forbidden_set else f"{base} {w_ctx}"
        elif sub == 1:
            n1 = rng.randint(11, 96)
            while n1 in FORBIDDEN_N1_2DIGIT:
                n1 = rng.randint(11, 96)
            n2 = rng.randint(11, 99)
            if ((n1 * 5 + n2) % 2 == 1) != is_val:
                n2 = 11 + ((n2 - 10) % 88)
            if n2 == n1:
                n2 = min(99, n1 + 2)
            cand = f"{n1}-{n2}"
            txt = cand if cand not in forbidden_set else f"{cand} {w_ctx}"
        elif sub == 2:
            pool3 = [104, 108, 125, 140, 150, 180, 204, 305, 405, 600, 700, 800, 900]
            n1 = pool3[(idx + att) % len(pool3)] if (m % 2 == 0) else rng.randint(102, 885)
            while n1 in FORBIDDEN_N1_3DIGIT:
                n1 = rng.randint(102, 885)
            n2 = rng.choice([200, 300, 400, 500, 600, 700, 800, 900]) if (m % 2 == 0) else (n1 + rng.randint(5, 95))
            if ((n1 + n2) % 2 == 1) != is_val:
                n1 += 1
                if n1 in FORBIDDEN_N1_3DIGIT:
                    n1 += 1
            cand = f"{n1}-{n2}"
            txt = cand if cand not in forbidden_set else f"{w_ctx} {cand}"
        else:
            y1 = rng.randint(1804, 2022)
            while y1 in FORBIDDEN_Y1_4DIGIT or ((y1 % 2 == 1) != is_val):
                y1 = rng.randint(1804, 2022)
            y2 = 2000 if (y1 < 1998 and m % 3 == 0 and att < 12) else (y1 + rng.randint(2, 35))
            cand = f"{y1}-{y2}"
            txt = cand if cand not in forbidden_set else f"{cand} {w_ctx}"

        txt = normalize_unicode(txt.strip())
        if _is_safe_text(txt, forbidden_set):
            fsize = rng.choice([20, 22, 24, 26])
            if m % 3 != 0:
                img = render_hb_ink_tight(txt, offset=m, pad_x_ink=5 + (m % 6), pad_y_top=5 + (m % 3), pad_y_bot=5 + (m % 3), font_size=fsize)
            else:
                img = render_hb_line(txt, offset=m, pad_x_override=8 + (m % 8), font_size_override=fsize, repeat_digit_gap=(1 if m % 4 == 0 else 0))
            if img.height == 48 and img.width <= 480:
                if (idx % 6) == 5:
                    img = _mild_stroke_break_or_blur(img, rng, np_rng)
                forbidden_set.add(txt)
                return txt, img, False
    raise RuntimeError(f"Short number-hyphen failed at idx={idx}")


def _sample_contrastive_single_vs_double_hyphen(idx: int, rng: random.Random, np_rng, forbidden_set: set):
    base_off = 19900000 + idx * 19
    for att in range(80):
        m = base_off + att * 13
        w1, w2, _ = _make_symbol_words(rng)
        cw = CORPUS_WORDS_SORTED[(idx * 9 + att) % len(CORPUS_WORDS_SORTED)]
        is_double = (idx % 2 == 1)
        sep = "--" if is_double else "-"
        sub = (idx + att) % 5
        if sub in (0, 1):
            txt = f"{w1}{sep}{w2}"
        elif sub == 2:
            txt = f"{cw}{sep}{w2}" if (len(cw) <= 8) else f"{w1}{sep}{w2}"
        elif sub == 3:
            txt = f"{w1}{sep}{w2} {cw}"
        else:
            txt = f"-{w1} {w2}" if not is_double else f"{w1} {w2}-"
        txt = normalize_unicode(txt.strip())
        if _is_safe_text(txt, forbidden_set):
            fsize = rng.choice([20, 22, 24, 26])
            gap = rng.choice([0, 0, 2, 3, 5, 6]) if is_double else 0
            if m % 3 != 0 and gap == 0:
                img = render_hb_ink_tight(txt, offset=m, pad_x_ink=6 + (m % 6), pad_y_top=5 + (m % 3), pad_y_bot=5 + (m % 3), font_size=fsize)
            else:
                img = render_hb_line(txt, offset=m, pad_x_override=10 + (m % 10), font_size_override=fsize, repeat_digit_gap=gap)
            if img.height == 48 and img.width <= 480:
                if (idx % 5) == 4:
                    img = _mild_stroke_break_or_blur(img, rng, np_rng)
                forbidden_set.add(txt)
                return txt, img, False
    raise RuntimeError(f"Contrastive hyphen failed at idx={idx}")


def sample_stream_c_gate_closure(idx: int, is_val: bool, rng: random.Random, np_rng, forbidden_set: set):
    """Stream C (25%): Crisp `*` envelope, short numeric hyphens, `-` vs `--`, & 14 mandatory symbols."""
    q = idx % 4
    if q == 0:
        return _sample_single_asterisk(idx, rng, np_rng, forbidden_set)
    if q == 1:
        return _sample_short_number_hyphen(idx, is_val, rng, np_rng, forbidden_set)
    if q == 2:
        return _sample_contrastive_single_vs_double_hyphen(idx, rng, np_rng, forbidden_set)
    if idx % 8 == 3:
        txt, img = sample_r4_2_hyphens(idx + 80000, is_val, rng, forbidden_set)
        return txt, img, False
    txt, img = sample_r4_2_all_symbols(idx + 80000, is_val, rng, forbidden_set)
    return txt, img, False


def sample_stream_d_paleographic_optical_blank(idx: int, is_val: bool, rng: random.Random, np_rng, forbidden_set: set):
    """Stream D (15%): Wide manuscript/folio Cham lines (45% paleographic calligraphic/elastic) + 15% blank negatives."""
    if (idx % 20) < 3:
        return "", generate_wide_blank(rng, np_rng), False

    base_off = (21500000 if is_val else 20500000) + idx * 23
    min_w = 240 if (idx % 3 != 0) else 110
    max_w = 478 if (idx % 3 != 0) else 300
    for att in range(60):
        txt, clean_im = compose_wide_corpus_line(base_off + att * 17, is_val, rng, target_min_w=min_w, target_max_w=max_w)
        txt = normalize_unicode(txt.strip())
        if _is_safe_text(txt, forbidden_set) and clean_im.height == 48 and clean_im.width <= 480:
            mode = idx % 11
            used_paleo = False
            if mode < 5:
                im = degrade_paleographic_calligraphic_folio(clean_im, rng, np_rng)
                used_paleo = True
            elif mode == 5:
                im = degrade_book_camera_optical(clean_im, rng, np_rng)
            elif mode == 6:
                im = degrade_erosion_dry_ink(clean_im, rng, np_rng)
            elif mode == 7:
                im = degrade_severe_motion(clean_im, rng, np_rng)
            elif mode == 8:
                im = degrade_heavy_gauss_noise_jpeg(clean_im, rng, np_rng)
            elif mode == 9:
                im = degrade_extreme_lowres(clean_im, rng, np_rng)
            else:
                im = degrade_persp_dilate_shadow(clean_im, rng, np_rng)
            forbidden_set.add(txt)
            return txt, im, used_paleo
    raise RuntimeError(f"Stream D failed at idx={idx}")


def main():
    t0 = time.time()
    verify_fonts_strictly_in_domain()
    assert os.path.isdir(V26_FT_DIR), f"Missing V26 FT base dataset: {V26_FT_DIR}"

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    split_rows = {"train": [], "val": []}
    seen_img_hashes = set()
    val_img_hashes = set()
    base_texts = {"train": set(), "val": set()}
    base_counts = Counter()

    for split in ("val", "train"):
        src_lbl = os.path.join(V26_FT_DIR, f"{split}_label.txt")
        for line in open(src_lbl, "r", encoding="utf-8"):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            rel, lab = line.split("\t", 1)
            sp = os.path.join(V26_FT_DIR, *rel.split("/"))
            h = _sha256_file(sp)
            if split == "val":
                val_img_hashes.add(h)
            else:
                assert h not in val_img_hashes, f"Base FT leak on {rel}"
            assert h not in seen_img_hashes, f"Duplicate hash in V26 FT: {rel}"
            seen_img_hashes.add(h)
            ext = os.path.splitext(rel)[1].lower() or ".jpg"
            new_rel = f"{split}/ft_{len(split_rows[split]):06d}{ext}"
            dp = os.path.join(OUT_DIR, *new_rel.split("/"))
            os.makedirs(os.path.dirname(dp), exist_ok=True)
            shutil.copy2(sp, dp)
            split_rows[split].append((new_rel, lab))
            if lab:
                base_texts[split].add(normalize_unicode(lab))
            base_counts[split] += 1

    assert base_counts["train"] == 43520 and base_counts["val"] == 4352, f"Unexpected base counts: {base_counts}"
    print(f"Copied V26 FT base: {base_counts['train']:,} train / {base_counts['val']:,} val (unique hashes={len(seen_img_hashes):,})")

    fresh_quotas = {
        "train": (6144, 6144, 5120, 3072),
        "val": (614, 614, 512, 308),
    }
    fresh_texts = {"train": set(), "val": set()}
    fresh_stats = {"train": Counter(), "val": Counter()}
    min_pair_counter = Counter()
    stanza_nums_seen = set()

    forbidden_train = set(FORBIDDEN_EXACT_TEXTS) | base_texts["val"] | base_texts["train"]
    rng_tr = random.Random(20261141)
    np_tr = np.random.default_rng(20261141)
    rng_va = random.Random(20261142)
    np_va = np.random.default_rng(20261142)

    for split, (n_a, n_b, n_c, n_d) in fresh_quotas.items():
        is_val = (split == "val")
        rng = rng_va if is_val else rng_tr
        np_rng = np_va if is_val else np_tr
        f_guard = (forbidden_train | fresh_texts["train"]) if is_val else forbidden_train

        generators = [
            ("stream_a_stanza_danda", n_a, lambda i: sample_stream_a_stanza_danda(i, is_val, rng, np_rng, f_guard)),
            ("stream_b_minimal_multitier", n_b, lambda i: sample_stream_b_minimal_pairs_multitier(i, is_val, rng, np_rng, f_guard)),
            ("stream_c_gate_closure", n_c, lambda i: sample_stream_c_gate_closure(i, is_val, rng, np_rng, f_guard)),
            ("stream_d_paleographic_optical_blank", n_d, lambda i: sample_stream_d_paleographic_optical_blank(i, is_val, rng, np_rng, f_guard)),
        ]
        for s_tag, q_n, fn in generators:
            for i in range(q_n):
                txt, im, used_paleo = fn(i)
                new_rel = f"{split}/v26_1_{len(split_rows[split]):06d}.jpg"
                dp = os.path.join(OUT_DIR, *new_rel.split("/"))
                os.makedirs(os.path.dirname(dp), exist_ok=True)
                _save_unique_jpeg(im, dp, seen_img_hashes, len(split_rows[split]) * 13 + i)
                split_rows[split].append((new_rel, txt))
                fresh_stats[split][s_tag] += 1
                if used_paleo and not is_val:
                    min_pair_counter["train_paleographic_calligraphic_folio"] += 1
                if txt:
                    nt = normalize_unicode(txt)
                    fresh_texts[split].add(nt)
                    if not is_val:
                        if "ꨲ" in nt and "ꨶ" in nt:
                            min_pair_counter["train_au_vs_o_pairs"] += 1
                        if "꩝꩝" in nt:
                            min_pair_counter["train_double_danda"] += 1
                        if "ꨳꨪꩌ" in nt or "ꨳꨫꩌ" in nt:
                            min_pair_counter["train_multitier_aa33_i_bindu"] += 1
                        if "*" in nt and "**" not in nt:
                            min_pair_counter["train_single_asterisk"] += 1
                        if s_tag == "stream_a_stanza_danda" and "꩞" in nt:
                            digs = nt.split("꩞", 1)[0]
                            if digs and all(c in "꩐꩑꩒꩓꩔꩕꩖꩗꩘꩙" for c in digs):
                                arabic = "".join(str("꩐꩑꩒꩓꩔꩕꩖꩗꩘꩙".index(c)) for c in digs)
                                stanza_nums_seen.add(int(arabic))
            print(f"  [{split}] {s_tag}: {fresh_stats[split][s_tag]:,} samples")

    covered_tokens = set()
    for split in ("train", "val"):
        lbl_p = os.path.join(OUT_DIR, f"{split}_label.txt")
        with open(lbl_p, "w", encoding="utf-8", newline="\n") as f:
            for rel, lab in split_rows[split]:
                f.write(f"{rel}\t{lab}\n")
                if split == "train":
                    covered_tokens.update(c for c in lab if c in DICT_SET)

    assert len(split_rows["train"]) == 64000, len(split_rows["train"])
    assert len(split_rows["val"]) == 6400, len(split_rows["val"])
    assert not (fresh_texts["train"] & fresh_texts["val"]), "Fresh train/val text overlap!"
    assert not ((fresh_texts["train"] | fresh_texts["val"]) & FORBIDDEN_EXACT_TEXTS), "Benchmark contamination!"
    assert len(covered_tokens) == 114, f"Missing dict tokens: {len(covered_tokens)}/114"
    assert min_pair_counter["train_au_vs_o_pairs"] >= 5000, min_pair_counter
    assert min_pair_counter["train_paleographic_calligraphic_folio"] >= 5500, min_pair_counter
    assert len(stanza_nums_seen) == 999, f"Expected all 999 stanza numerals 1..999, got {len(stanza_nums_seen)}"

    manifest = {
        "title": "Cham-OCR V26.1 Masked-SVTR W=480 Dataset (Early-Backbone Frozen + Paleographic Calligraphic/Elastic Augmentation)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(time.time() - t0, 2),
        "out_dir": "ocr-training/data/cham_synthetic_v26_1",
        "train_samples": len(split_rows["train"]),
        "val_samples": len(split_rows["val"]),
        "step_each_epoch_dual_t4_bs32": len(split_rows["train"]) // 2 // 32,
        "unique_image_sha256_count": len(seen_img_hashes),
        "fresh_train_by_stream": dict(fresh_stats["train"]),
        "fresh_val_by_stream": dict(fresh_stats["val"]),
        "hard_example_counts": dict(min_pair_counter),
        "distinct_stanza_numerals_1_to_999_covered": len(stanza_nums_seen),
        "fresh_train_val_text_overlap": 0,
        "benchmark_exact_contamination": 0,
        "dict_token_coverage": f"{len(covered_tokens)}/114",
        "train_label_sha256": _sha256_file(os.path.join(OUT_DIR, "train_label.txt")),
        "val_label_sha256": _sha256_file(os.path.join(OUT_DIR, "val_label.txt")),
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    m_out = os.path.join(REPORTS_DIR, "v26_1_dataset_manifest.json")
    with open(m_out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"✅ V26.1 dataset built -> {OUT_DIR}")


if __name__ == "__main__":
    main()
