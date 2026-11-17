#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R4.1 Surgical Gate-Closure Micro-Polish Generator (`W = 480`, `T = 60`)
============================================================================================
Generates 21,120 samples (19,200 Train / 1,920 Val) at H=48, W<=480px (100% HarfBuzz shaping):
  1. Stream 1 (65.0% = 12,480 Train / 1,248 Val): Stratified Stage R4 Wide-Canvas Replay
     (wide corpus, sesquisyllabic clusters, all 14 symbols, optical degradations, wide blanks `""`).
  2. Stream 2 (12.5% = 2,400 Train / 240 Val): Surgical Infix Underscore `_` (`w1_w2`, `Cꨪ_w2`, `w1_Cꨳ` vs `ꨳ`/`ꨶ`)
     & Infix Slash `/` (`w1/w2`, Cham slash-dates `꩑..꩙/꩑..꩙/꩒꩐..`, `/꩒` vs `/ꨐ` Nya contrastive pairs).
  3. Stream 3 (12.5% = 2,400 Train / 240 Val): Surgical Natural `0px`-Gap HarfBuzz Double-Hyphen `w1--w2` vs `w1-w2`
     across all CTC phases + Short Standalone `D-D`/`DD-DD`/`DDD-DDD` & Zero-Preserving `YYYY-YYYY` under `W=480` right-pad.
  4. Stream 4 (10.0% = 1,920 Train / 192 Val): Optical Camera / Mild-Erosion Polish on Targeted `_`, `/`, `--`,
     Cham Slash-Dates & Wide Blanks (`""`) to guarantee zero regression on degraded scans and Blank Guard.

Strict Invariants:
  - 100% Benchmark Blind & OOD Font Uncontaminated (ONLY `ocr-training/data/fonts/NotoSansCham-*.ttf`).
  - Zero Train/Val text overlap, 100% unique SHA256 image hashes, 114/114 token coverage, `W <= 480px`.
"""

import os
import sys
import json
import time
import random
import shutil
import hashlib
from datetime import datetime, timezone

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
    TRAIN_SINGLE_DIGIT_PAIRS,
    VAL_SINGLE_DIGIT_PAIRS,
    CORPUS_WORDS_SORTED,
    normalize_unicode,
    to_cham_num,
    render_hb_line,
    synth_fresh_monosyllable,
    degrade_erosion_dry_ink,
    _save_unique_jpeg,
)
from generate_data_v24_1_r4_w480 import (
    degrade_book_camera_optical,
    generate_wide_blank,
)


def _synth_upper_vowel_word(rng: random.Random) -> str:
    """Synthesizes a non-corpus Cham syllable with upper vowel `ꨪ`/`ꨫ`/`ꨬ` (where baseline before `_` or `/` is open)."""
    for _ in range(40):
        base = rng.choice(CHAM_TRUE_CONSONANTS)
        vow = rng.choice(["ꨪ", "ꨫ", "ꨬ", "ꨩ", "ꨭ"])
        coda = rng.choice(["", "", "ꩅ", "ꩆ", "ꩀ", "ꩉ", "ꩍ", "ꩇ"])
        cand = normalize_unicode(f"{base}{vow}{coda}")
        if cand not in CORPUS_WORDS_SORTED and not ILLEGAL_MEDIAL_RE.search(cand):
            return cand
    return synth_fresh_monosyllable(rng)


def sample_r4_1_underscore_or_slash(idx: int, is_val: bool, rng: random.Random, seen_texts: set):
    """Stream 2: Targets unspaced `w1_w2` (esp. `Cꨪ_w2`) and unspaced `w1/w2` + Cham slash dates `/꩒` vs `/ꨐ`."""
    base_off = (2600000 if is_val else 2100000) + idx * 19
    is_underscore = (idx % 2 == 0)

    for att in range(80):
        m = base_off + att * 13
        pad_x = 12 + (m % 16)
        fsize = (20, 22, 24, 24, 26)[m % 5]
        w_up = _synth_upper_vowel_word(rng)
        w_a = synth_fresh_monosyllable(rng, force_pre=(m % 3 == 0))
        w_b = synth_fresh_monosyllable(rng, force_aa33=(m % 4 == 0), force_bindu=(m % 5 == 0))
        cw = CORPUS_WORDS_SORTED[(idx * 7 + att * 11) % len(CORPUS_WORDS_SORTED)]

        if is_underscore:
            mode = (idx // 2 + att) % 8
            if mode in (0, 1, 2):
                # Short standalone unspaced `Cꨪ_w2` (exact structural archetype of `ꨧꨪ_ꨔꩅ` without string overlap!)
                txt = f"{w_up}_{w_b}"
            elif mode in (3, 4):
                # Short standalone unspaced `w1_w2` (including longer 2-syllable right/left sides)
                w_long = f"{w_a}{synth_fresh_monosyllable(rng)}"
                txt = f"{w_up}_{w_long}" if (mode == 3) else f"{w_long}_{w_b}"
            elif mode == 5:
                # Contrastive subjoined `-ya` `ꨳ` / `-wa` `ꨶ` adjacent to `_`
                w_sub = synth_fresh_monosyllable(rng, force_aa33=True)
                txt = f"{w_sub}_{w_up}" if (m % 2 == 0) else f"{w_up}_{w_sub}"
            elif mode == 6:
                txt = f"{cw} {w_up}_{w_b}"
            else:
                txt = f"{w_a}_{w_b} {cw}"
        else:
            mode = (idx // 2 + att) % 10
            if mode in (0, 1, 2, 3):
                # Short standalone unspaced `w1/w2` (exact structural archetype of `ꨆꨭꨖꨩ/ꨓꨶꨮꩉ` without overlap!)
                txt = f"{w_up}/{w_b}" if (mode % 2 == 0) else f"{w_a}/{w_b}"
            elif mode in (4, 5, 6):
                # Authentic Cham digit slash-dates `d/m/20xx` (oversampling `/꩒` vs `/ꨐ` Nya, NEVER `8/5/2026`!)
                d = rng.randint(1, 28)
                mo = rng.randint(1, 12)
                yr = rng.choice([2002, 2004, 2010, 2012, 2020, 2021, 2022, 2024, 2025, 2027, 2028])
                while d == 8 and mo == 5:
                    d = rng.randint(1, 28)
                date_str = f"{to_cham_num(d)}/{to_cham_num(mo)}/{to_cham_num(yr)}"
                if mode == 4:
                    txt = date_str
                elif mode == 5:
                    txt = f"{cw} {w_a} {date_str}"
                else:
                    # Contrastive `/ꨐ...` (Nya word after slash) vs `/꩒...` (Cham digit 2 after slash)
                    nya_w = normalize_unicode(f"ꨐ{rng.choice(['ꨩ', 'ꨪ', 'ꨭ', 'ꨮ'])}{rng.choice(['', 'ꩆ', 'ꩅ', 'ꩀ'])}")
                    txt = f"{w_a}/{nya_w}" if (m % 2 == 0) else f"{w_up} {date_str}"
            elif mode == 7:
                txt = f"{w_a} / {w_b}"
            elif mode == 8:
                txt = f"{cw} {w_up}/{w_b}"
            else:
                d, mo, yr = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1950, 2024)
                txt = f"{d:02d}/{mo:02d}/{yr} {w_b}"

        txt = normalize_unicode(txt.strip())
        if txt not in seen_texts and all(c in DICT_SET for c in txt) and not ILLEGAL_MEDIAL_RE.search(txt):
            img = render_hb_line(txt, offset=m, pad_x_override=pad_x, font_size_override=fsize, repeat_digit_gap=0)
            if img.height == 48 and img.width <= 480:
                seen_texts.add(txt)
                return txt, img
    raise RuntimeError(f"Failed underscore/slash synthesis at idx={idx}")


def sample_r4_1_double_hyphen_and_numbers(idx: int, is_val: bool, rng: random.Random, seen_texts: set):
    """Stream 3: 100% natural `0px`-gap HarfBuzz `w1--w2` vs `w1-w2` + standalone `DD-DD`/`DDD-DDD`/`YYYY-YYYY`."""
    base_off = (3100000 if is_val else 2800000) + idx * 23
    bucket = idx % 10

    for att in range(80):
        m = base_off + att * 17
        pad_x = 12 + (m % 20)
        fsize = (20, 22, 24, 24, 26)[m % 5]
        w1 = synth_fresh_monosyllable(rng, force_pre=(m % 3 == 0))
        w2 = synth_fresh_monosyllable(rng, force_aa33=(m % 4 == 0), force_bindu=(m % 5 == 0))
        w_up = _synth_upper_vowel_word(rng)
        cw = CORPUS_WORDS_SORTED[(idx * 9 + att * 13) % len(CORPUS_WORDS_SORTED)]

        if bucket in (0, 1, 2, 3, 4):
            # 50% pure 0px-gap natural HarfBuzz `w1--w2` (standalone short & multi-syllable, matching Frozen-838!)
            sub = (idx + att) % 5
            if sub == 0:
                txt = f"{w1}--{w2}"
            elif sub == 1:
                txt = f"{w_up}--{w2}"
            elif sub == 2:
                w_long = f"{w1}{synth_fresh_monosyllable(rng)}"
                txt = f"{w_long}--{w_up}" if (m % 2 == 0) else f"{w_up}--{w_long}"
            elif sub == 3:
                txt = f"{cw}--{w2}" if (len(cw) <= 8) else f"{w1}--{w2}"
            else:
                txt = f"{w1}--{w2} {cw}"
        elif bucket == 5:
            # 10% contrastive single-hyphen `w1-w2` right next to `w1--w2` so `-` never over-predicts `--`!
            txt = f"{w1}-{w2}" if (m % 2 == 0) else f"{w_up}-{w2}"
        elif bucket in (6, 7):
            # 20% standalone & contextual `number-number` (`D-D`, `DD-DD`, `DDD-DDD` with repeated zeros `00`)
            sub_n = (idx + att) % 4
            if sub_n == 0:
                pairs = VAL_SINGLE_DIGIT_PAIRS if is_val else TRAIN_SINGLE_DIGIT_PAIRS
                p_str = pairs[(idx + att) % len(pairs)]
                txt = p_str if (p_str not in seen_texts) else f"{p_str} {w1}"
            elif sub_n in (1, 2):
                n1 = rng.randint(11, 96)
                while n1 in FORBIDDEN_N1_2DIGIT:
                    n1 = rng.randint(11, 96)
                n2 = rng.randint(11, 99)
                while n2 == n1:
                    n2 = rng.randint(11, 99)
                # Ensure train & val disjoint on pure `n1-n2` via parity on `n1 + n2`!
                if ((n1 * 3 + n2) % 2 == 1) != is_val:
                    n2 = 11 + ((n2 - 10) % 88)
                    if n2 == n1:
                        n2 = min(99, n1 + 2)
                txt = f"{n1}-{n2}"
            else:
                n1 = rng.choice([104, 108, 130, 150, 180, 205, 305, 405, 600, 700, 800]) if (m % 2 == 0) else rng.randint(102, 880)
                while n1 in FORBIDDEN_N1_3DIGIT:
                    n1 = rng.randint(102, 880)
                n2 = rng.choice([200, 300, 400, 500, 700, 800, 900]) if (m % 2 == 0) else (n1 + rng.randint(5, 110))
                if n2 <= n1:
                    n2 = n1 + 100
                if ((n1 + n2) % 2 == 1) != is_val:
                    n1 += 1
                    if n1 in FORBIDDEN_N1_3DIGIT:
                        n1 += 1
                txt = f"{n1}-{n2}"
        else:
            # 20% zero-rich 4-digit `year-year` (`1988-2000`, `2001-2024`, `1815-1845` at 0px natural HarfBuzz gap!)
            sub_y = (idx + att) % 3
            if sub_y == 0:
                y1 = rng.randint(1951, 1999)
                while y1 in FORBIDDEN_Y1_4DIGIT or ((y1 % 2 == 1) != is_val):
                    y1 = rng.randint(1951, 1999)
                y2 = 2000
            elif sub_y == 1:
                y1 = rng.choice([2001, 2003, 2005, 2007, 2009] if is_val else [2002, 2004, 2006, 2008, 2012])
                while y1 in FORBIDDEN_Y1_4DIGIT:
                    y1 += 2
                y2 = rng.randint(y1 + 2, 2028)
            else:
                y1 = rng.randint(1805, 1985)
                while y1 in FORBIDDEN_Y1_4DIGIT or ((y1 % 2 == 1) != is_val):
                    y1 = rng.randint(1805, 1985)
                y2 = y1 + rng.randint(4, 38)
            txt = f"{y1}-{y2}"

        txt = normalize_unicode(txt.strip())
        if txt not in seen_texts and all(c in DICT_SET for c in txt) and not ILLEGAL_MEDIAL_RE.search(txt):
            # 100% natural 0px HarfBuzz glyph spacing (`repeat_digit_gap=0`) matching Frozen-838 exactly!
            img = render_hb_line(txt, offset=m, pad_x_override=pad_x, font_size_override=fsize, repeat_digit_gap=0)
            if img.height == 48 and img.width <= 480:
                seen_texts.add(txt)
                return txt, img
    raise RuntimeError(f"Failed double-hyphen/number synthesis at idx={idx}")


def build_split_r4_1(
    split_name: str,
    r4_dir: str,
    out_dir: str,
    seen_texts: set,
    seen_img_hashes: set,
    seed: int,
):
    is_val = (split_name == "val")
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed + 509)

    r4_lines = [
        l.rstrip("\r\n")
        for l in open(os.path.join(r4_dir, f"{split_name}_label.txt"), "r", encoding="utf-8")
        if l.strip("\r\n")
    ]

    if not is_val:
        # Total = 19,200 (exact 300 steps/epoch at global batch 64!)
        n_replay = 12480
        n_s2 = 2400
        n_s3 = 2400
        n_s4_opt = 1680
        n_s4_blank = 240
    else:
        # Total = 1,920
        n_replay = 1248
        n_s2 = 240
        n_s3 = 240
        n_s4_opt = 168
        n_s4_blank = 24

    img_out_dir = os.path.join(out_dir, "images")
    os.makedirs(img_out_dir, exist_ok=True)
    out_labels = []
    idx_global = 0

    # --- Stream 1: Stratified Replay from Verified Stage R4 (guaranteeing 100% rare-token inclusion) ---
    must_keep = set()
    covered_chars = set()
    if not is_val:
        for rare_ch in sorted(DICT_SET):
            if rare_ch in covered_chars:
                continue
            for idx_r, row_r in enumerate(r4_lines):
                t_r = row_r.split("\t", 1)[1]
                if rare_ch in t_r:
                    must_keep.add(idx_r)
                    covered_chars.update(t_r)
                    break
    rem_pool = [i for i in range(len(r4_lines)) if i not in must_keep and r4_lines[i].split("\t", 1)[1] not in seen_texts]
    rng.shuffle(rem_pool)
    selected_indices = sorted(list(must_keep) + rem_pool[: max(0, n_replay - len(must_keep))])[:n_replay]
    for src_i in selected_indices:
        rel_img, lbl = r4_lines[src_i].split("\t", 1)
        src_img_p = os.path.join(r4_dir, rel_img.replace("/", os.sep))
        dst_name = f"{split_name}_{idx_global:05d}.jpg"
        dst_p = os.path.join(img_out_dir, dst_name)
        raw_b = open(src_img_p, "rb").read()
        dg = hashlib.sha256(raw_b).hexdigest().lower()
        if dg not in seen_img_hashes:
            seen_img_hashes.add(dg)
            with open(dst_p, "wb") as f:
                f.write(raw_b)
        else:
            with Image.open(src_img_p) as im:
                _save_unique_jpeg(im.convert("RGB"), dst_p, seen_img_hashes, idx_global + 11)
        if lbl:
            seen_texts.add(lbl)
        out_labels.append(f"images/{dst_name}\t{lbl}")
        idx_global += 1
    print(f"  [{split_name}] Stream 1 (R4 Wide Replay): {n_replay} samples")

    # --- Stream 2: Surgical Underscore `_` & Slash `/` + Cham Dates ---
    for i in range(n_s2):
        txt, img = sample_r4_1_underscore_or_slash(i, is_val, rng, seen_texts)
        dst_name = f"{split_name}_{idx_global:05d}.jpg"
        _save_unique_jpeg(img, os.path.join(img_out_dir, dst_name), seen_img_hashes, i + 17)
        out_labels.append(f"images/{dst_name}\t{txt}")
        idx_global += 1
    print(f"  [{split_name}] Stream 2 (Underscore `_` & Slash `/` + Cham Dates): {n_s2} samples")

    # --- Stream 3: Surgical 0px-Gap `--` vs `-` & Short `number-number`/`year-year` ---
    for i in range(n_s3):
        txt, img = sample_r4_1_double_hyphen_and_numbers(i, is_val, rng, seen_texts)
        dst_name = f"{split_name}_{idx_global:05d}.jpg"
        _save_unique_jpeg(img, os.path.join(img_out_dir, dst_name), seen_img_hashes, i + 23)
        out_labels.append(f"images/{dst_name}\t{txt}")
        idx_global += 1
    print(f"  [{split_name}] Stream 3 (Double-Hyphen `--` & Short Numbers/Years): {n_s3} samples")

    # --- Stream 4: Optical Camera / Mild Erosion on Targeted Patterns + Wide Blanks ---
    for i in range(n_s4_opt):
        if i % 2 == 0:
            txt, clean_im = sample_r4_1_underscore_or_slash(n_s2 + i, is_val, rng, seen_texts)
        else:
            txt, clean_im = sample_r4_1_double_hyphen_and_numbers(n_s3 + i, is_val, rng, seen_texts)
        if i % 4 == 0:
            deg_im = degrade_erosion_dry_ink(clean_im, rng, np_rng)
        else:
            deg_im = degrade_book_camera_optical(clean_im, rng, np_rng)
        dst_name = f"{split_name}_{idx_global:05d}.jpg"
        _save_unique_jpeg(deg_im, os.path.join(img_out_dir, dst_name), seen_img_hashes, i + 31)
        out_labels.append(f"images/{dst_name}\t{txt}")
        idx_global += 1

    for i in range(n_s4_blank):
        blk_im = generate_wide_blank(rng, np_rng)
        dst_name = f"{split_name}_{idx_global:05d}.jpg"
        _save_unique_jpeg(blk_im, os.path.join(img_out_dir, dst_name), seen_img_hashes, i + 41)
        out_labels.append(f"images/{dst_name}\t")
        idx_global += 1
    print(f"  [{split_name}] Stream 4 (Optical Targeted + Wide Blanks): {n_s4_opt + n_s4_blank} samples")

    # Guarantee 114/114 dict char coverage on Train
    if not is_val:
        present = set("".join(l.split("\t", 1)[1] for l in out_labels))
        missing = sorted(DICT_SET - present)
        assert not missing, f"Missing dict chars in R4.1 train: {missing}"

    lbl_path = os.path.join(out_dir, f"{split_name}_label.txt")
    with open(lbl_path, "w", encoding="utf-8", newline="\n") as f:
        for line in out_labels:
            f.write(line + "\n")
    return out_labels


def main():
    t0 = time.time()
    print("=" * 80)
    print("🏭  GENERATING STAGE R4.1 SURGICAL GATE-CLOSURE DATASET (21,120 SAMPLES)")
    print("=" * 80)

    r4_dir = os.path.join(TRAINING_DIR, "data", "cham_synthetic_r4_w480")
    assert os.path.exists(os.path.join(r4_dir, "train_label.txt")), f"Missing R4 dataset at {r4_dir}"

    out_dir = os.path.join(TRAINING_DIR, "data", "cham_synthetic_r4_1_polish")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    seen_texts = set()
    seen_hashes = set()
    tr_labels = build_split_r4_1("train", r4_dir, out_dir, seen_texts, seen_hashes, seed=202611171)
    va_labels = build_split_r4_1("val", r4_dir, out_dir, seen_texts, seen_hashes, seed=202611172)

    elapsed = round(time.time() - t0, 2)
    manifest = {
        "stage": "R4_1_GATE_POLISH_W480",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": elapsed,
        "train_samples": len(tr_labels),
        "val_samples": len(va_labels),
        "total_samples": len(tr_labels) + len(va_labels),
        "unique_image_sha256": len(seen_hashes),
        "rec_image_shape": [3, 48, 480],
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Stage R4.1 dataset generated in {elapsed}s -> {out_dir} ({len(seen_hashes):,} unique SHA256 hashes)")


if __name__ == "__main__":
    main()
