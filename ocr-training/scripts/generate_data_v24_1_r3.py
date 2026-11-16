#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R3 Scan Degradation Curriculum Dataset Generator
====================================================================
Generates the 20,000-sample (18,000 train / 2,000 val) dataset for Stage R3 (`SCAN-DEGRADATION`):
  - 75% (13,500 train / 1,500 val) Clean & Surgical Anchor Replay sampled from Stage R2.1.6:
    * Clean stanza replay: 8,100 train / 900 val (45.0%)
    * Short-phonotactic combining marks + U+AA56 (꩖) vs U+AA23 (ꨣ): 2,250 train / 250 val (12.5%)
    * 14 mandatory symbols (stratified): 1,575 train / 175 val (8.75%)
    * 7-group multi-context hyphens (stratified): 1,125 train / 125 val (6.25%)
    * Clean blank guards ("" ): 450 train / 50 val (2.5%)
  - 25% (4,500 train / 500 val) Optical Scan Degradation Curriculum on freshly synthesized
    non-benchmark Cham text, numerals, symbols, hyphens, and blank paper negatives (H=48, W<=320):
    * (a) Directional motion blur k in {7,9,11,13,15} at angles {-18,-10,-5,0,+5,+10,+18} deg: 1,100 / 125
    * (b) Heavy Gaussian defocus sigma in {1.8,2.2,2.7,3.2} + JPEG q in [45,75]: 1,000 / 110
    * (c) Low-res downsample cliff H_low in [16,24]px -> H=48px + mild blur sigma in [0.6,1.2]: 1,000 / 110
    * (d) Geometric shear +-3..4 deg / sinusoidal wave + ink bleed + shadow gradient: 1,000 / 110
    * (e) Noisy paper / uneven shadow blank negatives ("") sigma_noise in [8,45]: 400 / 45
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

from generate_data_v24_1_r2_1_6 import (
    sample_stream3_symbol_item,
    sample_stream4_hyphen_item,
)
from generate_data_v24_1_r2_1_7 import (
    DICT_CHARS_V24_1,
    CORE_STANZA_WORDS,
    normalize_unicode,
    to_cham_num,
    synthesize_short_phonotactic_word,
    synthesize_rare_base_word,
    sample_stream2_phonotactic_rare_item,
    render_fixed_baseline_line,
    file_sha256,
)

SYMBOLS_14 = ['?', '!', '#', '&', '*', '<', '>', '{', '}', '\\', '_', '[', ']', '/']


def _motion_blur_kernel(k: int, angle_deg: float) -> np.ndarray:
    kern = np.zeros((k, k), dtype=np.float32)
    kern[k // 2, :] = 1.0
    center = (k / 2.0 - 0.5, k / 2.0 - 0.5)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rot = cv2.warpAffine(kern, m, (k, k), flags=cv2.INTER_LINEAR)
    s = float(rot.sum())
    if s > 1e-6:
        rot /= s
    else:
        rot[k // 2, k // 2] = 1.0
    return rot


def apply_degradation_motion_blur(img: Image.Image, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    k = rng.choice([7, 9, 11, 13, 15])
    angle = rng.choice([-18.0, -10.0, -5.0, 0.0, 5.0, 10.0, 18.0])
    kern = _motion_blur_kernel(k, angle)
    out = cv2.filter2D(arr, -1, kern, borderType=cv2.BORDER_REPLICATE)
    if rng.random() < 0.65:
        bg_scale = rng.uniform(0.88, 0.99)
        noise = np_rng.normal(0.0, rng.uniform(1.5, 5.0), size=out.shape[:2])[:, :, None]
        out = np.clip(out.astype(np.float32) * bg_scale + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def apply_degradation_defocus_jpeg(img: Image.Image, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    sigma = rng.choice([1.8, 2.2, 2.7, 3.2])
    out = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    if rng.random() < 0.60:
        alpha = rng.uniform(0.86, 1.0)
        noise = np_rng.normal(0.0, rng.uniform(1.0, 4.0), size=out.shape[:2])[:, :, None]
        out = np.clip(out.astype(np.float32) * alpha + noise, 0, 255).astype(np.uint8)
    q = rng.randint(45, 75)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        out = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out)


def apply_degradation_lowres(img: Image.Image, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    h_low = rng.randint(16, 24)
    w_low = max(8, int(round(w * (h_low / float(h)))))
    small = cv2.resize(arr, (w_low, h_low), interpolation=cv2.INTER_AREA)
    interp = rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    up = cv2.resize(small, (w, h), interpolation=interp)
    sigma = rng.uniform(0.6, 1.2)
    out = cv2.GaussianBlur(up, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    if rng.random() < 0.50:
        noise = np_rng.normal(0.0, rng.uniform(1.5, 5.5), size=out.shape[:2])[:, :, None]
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def apply_degradation_geom_bleed_shadow(img: Image.Image, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]

    # 1. Geometry: 50% shear +/- 3..4 deg, 50% sinusoidal baseline wave (amplitude 1.5..3.0px)
    if rng.random() < 0.50:
        deg = rng.choice([-4.0, -3.5, -3.0, 3.0, 3.5, 4.0])
        sh_x = math.tan(math.radians(deg))
        dx_shift = -sh_x * (h / 2.0)
        m = np.array([[1.0, sh_x, dx_shift], [0.0, rng.uniform(0.95, 1.03), 0.0]], dtype=np.float32)
        out = cv2.warpAffine(arr, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        amp = rng.uniform(1.5, 3.0)
        periods = rng.uniform(1.0, 2.2)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        xs = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
        ys = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
        dy = (amp * np.sin(2.0 * math.pi * periods * xs / max(1.0, float(w)) + phase)).astype(np.float32)
        out = cv2.remap(arr, xs, ys + dy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 2. Ink bleed / subtle thinning (60% subtle bleed, 20% mild erode-paper, 20% clean)
    r_ink = rng.random()
    k2 = np.ones((2, 2), dtype=np.uint8)
    if r_ink < 0.60:
        bleed = cv2.erode(out, k2, iterations=1)
        blend = rng.uniform(0.35, 0.70)
        out = np.clip(out.astype(np.float32) * (1.0 - blend) + bleed.astype(np.float32) * blend, 0, 255).astype(np.uint8)
    elif r_ink < 0.80:
        thin = cv2.dilate(out, k2, iterations=1)
        blend = rng.uniform(0.25, 0.55)
        out = np.clip(out.astype(np.float32) * (1.0 - blend) + thin.astype(np.float32) * blend, 0, 255).astype(np.uint8)

    # 3. Smooth shadow gradient + mild paper noise
    lo = rng.uniform(0.65, 0.82)
    hi = rng.uniform(0.94, 1.00)
    if rng.random() < 0.5:
        gx = np.linspace(lo, hi, w, dtype=np.float32)[None, :, None]
        gy = np.linspace(rng.uniform(0.90, 1.0), 1.0, h, dtype=np.float32)[:, None, None]
    else:
        gx = np.linspace(hi, lo, w, dtype=np.float32)[None, :, None]
        gy = np.linspace(1.0, rng.uniform(0.88, 1.0), h, dtype=np.float32)[:, None, None]
    grad = gx * gy
    noise = np_rng.normal(0.0, rng.uniform(2.0, 7.0), size=(h, w, 1)).astype(np.float32)
    out = np.clip(out.astype(np.float32) * grad + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def generate_noisy_paper_blank(rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    w = rng.randint(96, 320)
    h = 48
    base_luma = rng.randint(205, 248)
    arr = np.full((h, w, 3), base_luma, dtype=np.float32)
    # Warm or cool paper tint
    arr[:, :, 0] += rng.uniform(-6.0, 4.0)
    arr[:, :, 2] += rng.uniform(-4.0, 6.0)

    # Smooth shadow vignette gradient (0.65..1.0)
    x_dir = rng.choice([-1, 1])
    lo = rng.uniform(0.65, 0.86)
    gx = np.linspace(lo, 1.0, w, dtype=np.float32)
    if x_dir < 0:
        gx = gx[::-1]
    gy = np.linspace(rng.uniform(0.84, 1.0), 1.0, h, dtype=np.float32)
    if rng.random() < 0.5:
        gy = gy[::-1]
    grad = (gy[:, None] * gx[None, :])[:, :, None]
    arr *= grad

    # Gaussian paper grain sigma in [8, 45], blended smoothly
    sigma = rng.uniform(8.0, 45.0)
    noise = np_rng.normal(0.0, sigma, size=(h, w, 1)).astype(np.float32)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _sample_fresh_r3_text_and_clean_img(idx: int, is_val: bool, rng: random.Random, seen_texts: set):
    """Synthesizes a fresh non-benchmark Cham / numeral / symbol / hyphen string with W <= 320px at H=48."""
    offset_base = (80000 if is_val else 40000) + idx * 13
    for attempt in range(40):
        m = offset_base + attempt * 7
        bucket = idx % 20
        random.seed(m * 31 + attempt)

        if bucket < 10:
            # 50% Cham multi-word stanza / phonotactic / Cham numeral + U+AA56 vs U+AA23
            sub = (idx + attempt) % 5
            if sub == 0:
                n_w = rng.choice([2, 3])
                words = [
                    rng.choice(CORE_STANZA_WORDS) if (i == 0 and rng.random() < 0.35) else synthesize_short_phonotactic_word(allow_disyllabic=(i == 0))
                    for i in range(n_w)
                ]
                punct = rng.choice(['', ' ꩝', ' ꩞'])
                text = normalize_unicode((" ".join(words) + punct).strip())
                img, _ = render_fixed_baseline_line(text, offset=m, force_hb=True, double_hyphen_gap=0)
            elif sub == 1:
                # Cham verse number 1..999 + Cham text
                n = rng.randint(1, 999)
                cnum = to_cham_num(n)
                w1 = synthesize_short_phonotactic_word(allow_disyllabic=False)
                w2 = synthesize_short_phonotactic_word(allow_disyllabic=False)
                text = normalize_unicode(f"{cnum}꩞ {w1} {w2}")
                repeat_gap = rng.randint(2, 5) if "꩖꩖" in cnum else 0
                img, _ = render_fixed_baseline_line(text, offset=m, force_hb=True, double_hyphen_gap=0, repeat_digit_gap=repeat_gap)
            elif sub == 2:
                # Cham date or rare consonant + medial + combmark
                if rng.random() < 0.45:
                    d, mo, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1905, 2023)
                    ra_w = normalize_unicode(f"ꨣ{rng.choice(['ꨩ', 'ꨪ', 'ꨭ', 'ꨮ', 'ꨳꨪꩌ'])}")
                    text = normalize_unicode(f"{ra_w} {to_cham_num(d)}/{to_cham_num(mo)}/{to_cham_num(y)}")
                else:
                    text = normalize_unicode(f"{synthesize_rare_base_word()} {synthesize_short_phonotactic_word(allow_disyllabic=True)} {synthesize_short_phonotactic_word(allow_disyllabic=False)}")
                img, _ = render_fixed_baseline_line(text, offset=m, force_hb=True, double_hyphen_gap=0)
            elif sub == 3:
                text, img = sample_stream2_phonotactic_rare_item(m)
            else:
                words = [synthesize_short_phonotactic_word(allow_disyllabic=(i == 1)) for i in range(rng.choice([2, 3]))]
                text = normalize_unicode(" ".join(words))
                img, _ = render_fixed_baseline_line(text, offset=m, force_hb=True, double_hyphen_gap=0)
        elif bucket < 14:
            # 20% Cham numerals + mixed punctuation / parentheses
            sub_n = (idx + attempt) % 3
            if sub_n == 0:
                num = to_cham_num(rng.randint(100, 999))
                w1 = synthesize_short_phonotactic_word(allow_disyllabic=False)
                text = normalize_unicode(f"{num}꩞ {w1} ꩝꩝")
            elif sub_n == 1:
                num = to_cham_num(rng.randint(1, 99))
                text = normalize_unicode(f"{num}꩞ {synthesize_rare_base_word()} ({synthesize_short_phonotactic_word(allow_disyllabic=False)})")
            else:
                text = normalize_unicode(f"꩖꩞ {synthesize_short_phonotactic_word(allow_disyllabic=False)} {to_cham_num(rng.randint(10, 89))}꩞")
            img, _ = render_fixed_baseline_line(text, offset=m, force_hb=True, double_hyphen_gap=0)
        elif bucket < 17:
            # 15% 14 mandatory symbols
            sym_groups = ['_', '{}', '/', '?', '!', '<', '>', '&', '[]', '#', '\\', '*']
            s_grp = sym_groups[(idx + attempt) % len(sym_groups)]
            text, img = sample_stream3_symbol_item(s_grp, m)
        else:
            # 15% 7-group hyphens
            hyp_groups = ['double_hyphen', 'number_number', 'line_beginning', 'word_word_tight', 'line_ending', 'word_word_spaced', 'year_year']
            h_grp = hyp_groups[(idx + attempt) % len(hyp_groups)]
            text, img = sample_stream4_hyphen_item(h_grp, m, is_val=is_val)

        if text and text not in seen_texts and img.height == 48 and img.width <= 320:
            seen_texts.add(text)
            return text, img

    raise RuntimeError(f"Failed to synthesize unique W<=320 sample at idx={idx}, is_val={is_val}")


def _sample_replay_from_r2_1_6(raw_lines: list, images_dir: str, split_name: str, rng: random.Random):
    """Samples 75% Stage R3 replay quotas deterministically from Stage R2.1.6."""
    if split_name == "train":
        # train ranges in r2_1_6: clean [0:9900], stream2 [9900:10700], sym [10700:12950], hyp [12950:14400], stream5 [14400:17100], blank [17100:18000]
        r_clean = (0, 9900, 8100)
        r_s2 = (9900, 10700, 800)       # keep all 800
        r_s5 = (14400, 17100, 1450)     # 800 + 1450 = 2250 short phono/combmark
        r_sym = (10700, 12950, 1575)
        r_hyp = (12950, 14400, 1125)
        r_blk = (17100, 18000, 450)
    else:
        # val ranges in r2_1_6: clean [0:1100], stream2 [1100:1190], sym [1190:1440], hyp [1440:1600], stream5 [1600:1900], blank [1900:2000]
        r_clean = (0, 1100, 900)
        r_s2 = (1100, 1190, 90)         # keep all 90
        r_s5 = (1600, 1900, 160)        # 90 + 160 = 250
        r_sym = (1190, 1440, 175)
        r_hyp = (1440, 1600, 125)
        r_blk = (1900, 2000, 50)

    def _pick_uniform(lo, hi, k):
        indices = list(range(lo, hi))
        rng.shuffle(indices)
        return sorted(indices[:k])

    selected = []
    for tag, (lo, hi, k) in (
        ("clean_stanza_replay", r_clean),
        ("stream2_rare_aa56_replay", r_s2),
        ("stream5_short_phono_replay", r_s5),
        ("symbol_14_replay", r_sym),
        ("hyphen_7group_replay", r_hyp),
        ("clean_blank_guard_replay", r_blk),
    ):
        picks = _pick_uniform(lo, hi, k)
        for idx in picks:
            rel_img, lbl = raw_lines[idx].split("\t", 1)
            src_path = os.path.join(os.path.dirname(images_dir), rel_img)
            assert os.path.exists(src_path), f"Missing replay image {src_path}"
            selected.append((tag, src_path, lbl))
    return selected


def generate_r3_dataset(seed: int = 303):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    print("=" * 80)
    print("🎨 CHAM-OCR V24.1 STAGE R3 SCAN DEGRADATION DATASET GENERATION")
    print(f"Seed: {seed}")
    print("=" * 80)

    data_dir_6 = os.path.join(TRAINING_DIR, "data", "cham_synthetic_r2_1_6_surgical_polish")
    assert os.path.exists(data_dir_6), f"Stage R2.1.6 dataset missing at {data_dir_6}"

    out_dir_r3 = os.path.join(TRAINING_DIR, "data", "cham_synthetic_r3_degradation")
    imgs_r3 = os.path.join(out_dir_r3, "images")
    if os.path.exists(out_dir_r3):
        shutil.rmtree(out_dir_r3)
    os.makedirs(imgs_r3, exist_ok=True)

    with open(os.path.join(data_dir_6, "train_label.txt"), "r", encoding="utf-8") as f:
        raw_6_tr = [line.rstrip("\r\n") for line in f]
    with open(os.path.join(data_dir_6, "val_label.txt"), "r", encoding="utf-8") as f:
        raw_6_va = [line.rstrip("\r\n") for line in f]
    assert len(raw_6_tr) == 18000 and len(raw_6_va) == 2000

    # Pre-populate seen_texts with ALL R2.1.x historical train+val labels so fresh R3 strings cannot cross train/val
    seen_texts = set()
    for folder in (
        "cham_synthetic_r2_1_4c_control_continue",
        "cham_synthetic_r2_1_4c_combmark_replay",
        "cham_synthetic_r2_1_4d_phonotactic_combmark",
        "cham_synthetic_r2_1_4e_short_phonotactic_combmark",
        "cham_synthetic_r2_1_6_surgical_polish",
        "cham_synthetic_r2_1_7_gate_closure",
    ):
        d = os.path.join(TRAINING_DIR, "data", folder)
        for fn in ("train_label.txt", "val_label.txt"):
            lp = os.path.join(d, fn)
            if os.path.exists(lp):
                with open(lp, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.rstrip("\r\n").split("\t", 1)
                        if len(parts) == 2 and parts[1]:
                            seen_texts.add(parts[1])

    # 1. Sample 75% Replay from Stage R2.1.6
    print("[1/4] Selecting 75% Clean & Surgical Anchor Replay from R2.1.6 (13,500 train / 1,500 val)...")
    replay_tr = _sample_replay_from_r2_1_6(raw_6_tr, os.path.join(data_dir_6, "images"), "train", rng)
    replay_va = _sample_replay_from_r2_1_6(raw_6_va, os.path.join(data_dir_6, "images"), "val", rng)
    assert len(replay_tr) == 13500 and len(replay_va) == 1500

    # 2. Generate 25% Optical Scan Degradation Curriculum (4,500 train / 500 val)
    print("[2/4] Rendering & degrading 25% Optical Scan Curriculum (4,500 train / 500 val)...")
    deg_specs_tr = [
        ("deg_a_motion_blur", 1100, apply_degradation_motion_blur),
        ("deg_b_defocus_jpeg", 1000, apply_degradation_defocus_jpeg),
        ("deg_c_lowres_cliff", 1000, apply_degradation_lowres),
        ("deg_d_geom_bleed_shadow", 1000, apply_degradation_geom_bleed_shadow),
        ("deg_e_noisy_blank", 400, None),
    ]
    deg_specs_va = [
        ("deg_a_motion_blur", 125, apply_degradation_motion_blur),
        ("deg_b_defocus_jpeg", 110, apply_degradation_defocus_jpeg),
        ("deg_c_lowres_cliff", 110, apply_degradation_lowres),
        ("deg_d_geom_bleed_shadow", 110, apply_degradation_geom_bleed_shadow),
        ("deg_e_noisy_blank", 45, None),
    ]

    def _build_degraded(specs, is_val):
        items = []
        running_idx = 0
        for tag, count, fn in specs:
            for k in range(count):
                if fn is None:
                    img_deg = generate_noisy_paper_blank(rng, np_rng)
                    text = ""
                else:
                    text, img_clean = _sample_fresh_r3_text_and_clean_img(running_idx, is_val, rng, seen_texts)
                    img_deg = fn(img_clean, rng, np_rng)
                assert img_deg.height == 48 and img_deg.width <= 320, f"Bad shape {img_deg.size} for {tag}"
                items.append((tag, text, img_deg))
                running_idx += 1
        return items

    deg_tr = _build_degraded(deg_specs_tr, is_val=False)
    deg_va = _build_degraded(deg_specs_va, is_val=True)
    assert len(deg_tr) == 4500 and len(deg_va) == 500

    # 3. Write images & label files (keep deterministic bucket ordering for auditability; PaddleOCR shuffles Train loader)
    print("[3/4] Writing 20,000 images & train_label.txt / val_label.txt...")
    bucket_counts = {"train": {}, "val": {}}

    def _emit_split(split_name, prefix, replay_items, deg_items):
        lines = []
        idx = 0
        for tag, src_path, lbl in replay_items:
            dst_name = f"r3_{prefix}_{idx:05d}.jpg"
            shutil.copy2(src_path, os.path.join(imgs_r3, dst_name))
            lines.append(f"images/{dst_name}\t{lbl}\n")
            bucket_counts[split_name][tag] = bucket_counts[split_name].get(tag, 0) + 1
            idx += 1
        for k, (tag, lbl, pil_img) in enumerate(deg_items):
            dst_name = f"r3_{prefix}_{idx:05d}.jpg"
            dst_path = os.path.join(imgs_r3, dst_name)
            q = 78 + (k % 15) if tag != "deg_b_defocus_jpeg" else 88 + (k % 7)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=q)
            with open(dst_path, "wb") as f:
                f.write(buf.getvalue())
            lines.append(f"images/{dst_name}\t{lbl}\n")
            bucket_counts[split_name][tag] = bucket_counts[split_name].get(tag, 0) + 1
            idx += 1
        return lines

    lines_tr = _emit_split("train", "tr", replay_tr, deg_tr)
    lines_va = _emit_split("val", "va", replay_va, deg_va)

    tr_lbl_path = os.path.join(out_dir_r3, "train_label.txt")
    va_lbl_path = os.path.join(out_dir_r3, "val_label.txt")
    with open(tr_lbl_path, "w", encoding="utf-8") as f:
        f.writelines(lines_tr)
    with open(va_lbl_path, "w", encoding="utf-8") as f:
        f.writelines(lines_va)

    # 4. Verify 114/114 dict coverage and save manifest
    all_tr_chars = set("".join(l.split("\t", 1)[1].strip() for l in lines_tr))
    missing_dict = [ch for ch in DICT_CHARS_V24_1 if ch not in all_tr_chars]
    print(f"[4/4] Train dictionary coverage: {114 - len(missing_dict)}/114 tokens (Missing: {missing_dict})")
    assert len(missing_dict) == 0, f"Missing dictionary tokens in R3 train set: {missing_dict}"

    tr_sha = file_sha256(tr_lbl_path)
    va_sha = file_sha256(va_lbl_path)
    manifest = {
        "stage": "R3-SCAN-DEGRADATION",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "output_dir": out_dir_r3,
        "train_samples": len(lines_tr),
        "val_samples": len(lines_va),
        "train_label_sha256": tr_sha,
        "val_label_sha256": va_sha,
        "dict_coverage": "114/114",
        "bucket_counts": bucket_counts,
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    m_path = os.path.join(REPORTS_DIR, "r3_degradation_manifest.json")
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"✅ Stage R3 dataset generated successfully -> {out_dir_r3}")
    print(f"   Train SHA256: {tr_sha}")
    print(f"   Val   SHA256: {va_sha}")
    print(f"   Manifest:     {m_path}")


if __name__ == "__main__":
    generate_r3_dataset(seed=303)
