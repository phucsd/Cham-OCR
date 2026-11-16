#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R2.1.7 7-Point Contamination, Positional Identity, Width & Baseline Geometry Audit
=======================================================================================================
1. ZERO exact text overlap with all GT-bearing benchmark sources.
2. ZERO internal train-vs-val text overlap.
3. ZERO long-substring (len >= 14) benchmark leaks.
4. ZERO image SHA256 overlap with benchmark images; exactly 15,000 image hashes shared with Branch E.
5. 15,000 shared positions (train 0..9899 + 14400..17999, val 0..1099 + 1600..1999) bit-for-bit
   identical with Branch E (labels + image SHA256), with positional Merkle digests.
6. Static AST generator blindness (no benchmark path/file literals).
7. Replaced 5,000 samples (Streams 2/3/4):
   a. 100% W <= 320px at H=48, 0 illegal [ꨁ..ꨅ][ꨳ..ꨶ] clusters.
   b. All 14 mandatory symbols + 7 hyphen groups present with nonzero counts, plus U+AA56 (꩖) & U+AA23 (ꨣ) coverage.
   c. Baseline geometry of the renderer: '-' ink centred mid-line, '_' ink at the bottom, '--' both dashes aligned.
"""

import os
import sys
import ast
import re
import json
import time
import hashlib
from datetime import datetime, timezone

import numpy as np
from PIL import Image

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-training", "scripts"))

from audit_r2_1_4c_contamination import (
    hash_directory_files,
    load_label_file_texts,
    load_benchmark_texts,
    BENCHMARK_DIR,
    FROZEN_DIR,
    REPORTS_DIR,
)

DATA_DIR_E = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_synthetic_r2_1_4e_short_phonotactic_combmark")
DATA_DIR_7 = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_synthetic_r2_1_7_gate_closure")
GENERATOR_7 = os.path.join(PROJECT_ROOT, "ocr-training", "scripts", "generate_data_v24_1_r2_1_7.py")

TR_SHARED = list(range(0, 9900)) + list(range(14400, 18000))
VA_SHARED = list(range(0, 1100)) + list(range(1600, 2000))
TR_REPLACED = (9900, 14400)
VA_REPLACED = (1100, 1600)
SYMBOLS_14 = ['?', '!', '#', '&', '*', '<', '>', '{', '}', '\\', '_', '[', ']', '/']


def read_lines(p):
    with open(p, "r", encoding="utf-8") as f:
        return [l.rstrip("\r\n") for l in f]


def classify_hyphen(t):
    if "--" in t:
        return "double_hyphen"
    if "-" not in t:
        return None
    if re.search(r"\b1[89]\d\d-\d{4}\b|\b20\d\d-\d{4}\b", t):
        return "year_year"
    if re.search(r"\d-\d", t):
        return "number_number"
    if t.endswith("-"):
        return "line_ending"
    if t.startswith("-"):
        return "line_beginning"
    if " - " in t:
        return "word_word_spaced"
    return "word_word_tight"


def ink_rows(img_arr, thr=160):
    g = img_arr if img_arr.ndim == 2 else img_arr.mean(axis=2)
    ys = np.where(g < thr)[0]
    return (int(ys.min()), int(ys.max())) if ys.size else (None, None)


def run():
    t0 = time.time()
    print("=" * 80)
    print("🛡️  STAGE R2.1.7 7-POINT CONTAMINATION, POSITIONAL, WIDTH & BASELINE AUDIT")
    print("=" * 80)

    # 1. Text disjointness
    print("\n[Audit 1/7] Exact text disjointness vs benchmarks...")
    tr_texts = load_label_file_texts(os.path.join(DATA_DIR_7, "train_label.txt"))
    va_texts = load_label_file_texts(os.path.join(DATA_DIR_7, "val_label.txt"))
    all_texts = tr_texts | va_texts
    gt_sets, _ = load_benchmark_texts()
    bench_all = set()
    text_viol = {}
    for name, s in gt_sets.items():
        bench_all |= s
        ov = all_texts & s
        if ov:
            text_viol[name] = sorted(ov)
            print(f"  ❌ {name}: {len(ov)} overlaps -> {sorted(ov)[:5]}")
        else:
            print(f"  ✅ {name}: {len(s):,} texts, 0 overlap")
    a1 = not text_viol

    # 2. Train/val disjointness
    print("\n[Audit 2/7] Train ∩ Val text overlap...")
    ov_tv = tr_texts & va_texts
    a2 = not ov_tv
    print(f"  • Overlap: {len(ov_tv)} -> {'✅' if a2 else '❌'}")

    # 3. Long substring
    print("\n[Audit 3/7] Long substring (len >= 14) containment...")
    sub_viol = []
    for bt in (t for t in bench_all if len(t) >= 14):
        for gt in all_texts:
            if bt in gt:
                sub_viol.append({"benchmark": bt, "generated": gt})
                break
    a3 = not sub_viol
    print(f"  • Violations: {len(sub_viol)} -> {'✅' if a3 else '❌'}")

    # 4. Image SHA256
    print("\n[Audit 4/7] Image SHA256 disjointness & shared-with-E count...")
    _, pm_e, h_e = hash_directory_files(os.path.join(DATA_DIR_E, "images"), extensions=[".jpg", ".png"])
    files_7, pm_7, h_7 = hash_directory_files(os.path.join(DATA_DIR_7, "images"), extensions=[".jpg", ".png"])
    shared = h_e & h_7
    print(f"  • R2.1.7 images: {len(files_7):,} (unique {len(h_7):,}); shared with E: {len(shared):,} (expected 15,000)")
    img_viol = {}
    for bname, bdir in {
        "frozen_crops": os.path.join(FROZEN_DIR, "crops"),
        "benchmark_images": os.path.join(BENCHMARK_DIR, "images"),
        "diagnostic_probe_images": os.path.join(BENCHMARK_DIR, "images_diagnostic"),
        "stress_500_images": os.path.join(BENCHMARK_DIR, "images_stress_500"),
        "advanced_images": os.path.join(BENCHMARK_DIR, "images_advanced"),
    }.items():
        if os.path.exists(bdir):
            bf, _, bh = hash_directory_files(bdir, extensions=[".jpg", ".png", ".jpeg"])
            ov = h_7 & bh
            if ov:
                img_viol[bname] = len(ov)
                print(f"  ❌ {bname}: {len(ov)} collisions")
            else:
                print(f"  ✅ {bname}: {len(bf):,} images, 0 collisions")
    a4 = (not img_viol) and len(shared) == 15000 and len(h_7) == 20000

    # 5. Positional identity
    print("\n[Audit 5/7] Positional identity on 15,000 shared positions...")
    le_tr = read_lines(os.path.join(DATA_DIR_E, "train_label.txt"))
    le_va = read_lines(os.path.join(DATA_DIR_E, "val_label.txt"))
    l7_tr = read_lines(os.path.join(DATA_DIR_7, "train_label.txt"))
    l7_va = read_lines(os.path.join(DATA_DIR_7, "val_label.txt"))
    mism = []
    merkle = {}
    for split, le, l7, idxs, de, d7 in (
        ("train", le_tr, l7_tr, TR_SHARED, DATA_DIR_E, DATA_DIR_7),
        ("val", le_va, l7_va, VA_SHARED, DATA_DIR_E, DATA_DIR_7),
    ):
        m = hashlib.sha256()
        for i in idxs:
            pe, te = le[i].split("\t", 1)
            p7, t7 = l7[i].split("\t", 1)
            if te != t7:
                mism.append(f"{split} label {i}")
            he = pm_e.get(os.path.normcase(os.path.abspath(os.path.join(de, pe))))
            h7 = pm_7.get(os.path.normcase(os.path.abspath(os.path.join(d7, p7))))
            if not he or he != h7:
                mism.append(f"{split} image {i}")
            else:
                m.update(he.encode())
        merkle[split] = m.hexdigest()
    a5 = not mism
    print(f"  • Mismatches: {len(mism)} | Train Merkle {merkle['train'][:16]}… | Val Merkle {merkle['val'][:16]}… -> {'✅' if a5 else '❌'}")

    # 6. AST blindness
    print("\n[Audit 6/7] Static AST generator blindness...")
    tree = ast.parse(open(GENERATOR_7, encoding="utf-8").read())
    forbidden = ["held_out_diagnostic_probe", "frozen_suite", "benchmark_gt", "clean_frozen_200",
                 "character_frozen_280", "symbol_frozen_14", "hyphen_frozen", "stress_frozen",
                 "stress_base_8", "images_diagnostic", "images_stress_500", "ocr-benchmark"]
    ast_viol = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value.lower()
            for fl in forbidden:
                if fl in v:
                    ast_viol.append({"literal": fl, "line": node.lineno})
    a6 = not ast_viol
    print(f"  • Violations: {len(ast_viol)} -> {'✅' if a6 else '❌'}")

    # 7. Replaced-slice width, phonotactics, coverage, baseline geometry
    print("\n[Audit 7/7] Replaced 5,000: width cap, phonotactics, coverage & baseline geometry...")
    rep = [(DATA_DIR_7, l) for l in l7_tr[TR_REPLACED[0]:TR_REPLACED[1]]] + \
          [(DATA_DIR_7, l) for l in l7_va[VA_REPLACED[0]:VA_REPLACED[1]]]
    assert len(rep) == 5000
    widths, heights, illegal = [], [], 0
    sym_counts = {s: 0 for s in SYMBOLS_14}
    hyp_counts = {}
    cham6_cnt, ra_cnt = 0, 0
    ill_re = re.compile(r"[ꨁꨂꨃꨄꨅ][ꨳꨴꨵꨶ]")
    for d, l in rep:
        p, t = l.split("\t", 1)
        with Image.open(os.path.join(d, p)) as im:
            widths.append(im.width)
            heights.append(im.height)
        if ill_re.search(t):
            illegal += 1
        for s in SYMBOLS_14:
            if s in t:
                sym_counts[s] += 1
        if "꩖" in t:
            cham6_cnt += 1
        if "ꨣ" in t:
            ra_cnt += 1
        g = classify_hyphen(t)
        if g:
            hyp_counts[g] = hyp_counts.get(g, 0) + 1
    over = sum(1 for w in widths if w > 320)
    bad_h = sum(1 for h in heights if h != 48)
    print(f"  • Width: mean={np.mean(widths):.1f}px max={max(widths)}px >320={over} | H!=48: {bad_h} | illegal medials: {illegal}")
    print(f"  • Symbol line counts: {sym_counts}")
    print(f"  • Hyphen group counts: {hyp_counts}")
    print(f"  • U+AA56 (꩖) lines: {cham6_cnt} | U+AA23 (ꨣ) lines: {ra_cnt}")

    from generate_data_v24_1_r2_1_7 import render_fixed_baseline_line
    geo = []
    geo_fail = []
    for off in range(10):
        for probe, gap, flank in (("ꨆ-ꨆ", 0, "ꨆ"), ("ꨆ_ꨆ", 0, "ꨆ"), ("ꨆ--ꨆ", 0, "ꨆ"),
                                  ("ꨆ--ꨆ", 2, "ꨆ"), ("11-11", 0, "11")):
            img, _ = render_fixed_baseline_line(probe, offset=off, double_hyphen_gap=gap)
            arr = np.array(img.convert("L"))
            y0, y1 = ink_rows(arr)
            ref, _ = render_fixed_baseline_line(flank, offset=off)
            ref_cols = np.where((np.array(ref.convert("L")) < 160).any(axis=0))[0]
            fw = int(ref_cols.max() - ref_cols.min() + 1)
            cols = np.where((arr < 160).any(axis=0))[0]
            x0, x1 = int(cols.min()) + fw + 1, int(cols.max()) - fw - 1
            mid = arr[:, x0:x1] if x1 > x0 else arr[:, 0:0]
            sy0, sy1 = ink_rows(mid) if mid.size else (None, None)
            rec = {"offset": off, "probe": probe, "gap": gap, "sym_cols": [x0, x1],
                   "sym_rows": [sy0, sy1], "line_rows": [y0, y1]}
            geo.append(rec)
            if sy0 is None:
                geo_fail.append(rec)
                continue
            centre = (sy0 + sy1) / 2.0
            if "_" in probe and (centre < 30 or (sy1 - sy0) > 5):
                geo_fail.append(rec)
            if "-" in probe and (not (20 <= centre <= 30) or (sy1 - sy0) > 5):
                geo_fail.append(rec)
    a7 = (over == 0 and bad_h == 0 and illegal == 0 and all(v > 0 for v in sym_counts.values())
          and len(hyp_counts) == 7 and all(v > 0 for v in hyp_counts.values())
          and cham6_cnt >= 150 and ra_cnt >= 150 and not geo_fail)
    print(f"  • Baseline geometry probes: {len(geo)} | failures: {len(geo_fail)}")
    for r in geo_fail[:10]:
        print(f"    ❌ {r}")
    print(f"  • Status: {'✅ PASSED' if a7 else '❌ FAILED'}")

    all_ok = all([a1, a2, a3, a4, a5, a6, a7])
    report = {
        "title": "Cham-OCR v24.1 Stage R2.1.7 7-Point Contamination, Positional, Width & Baseline Audit",
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all_ok else "FAIL",
        "execution_duration_sec": round(time.time() - t0, 2),
        "shared_train_image_merkle": merkle["train"],
        "shared_val_image_merkle": merkle["val"],
        "replaced_5000": {
            "mean_width_px": round(float(np.mean(widths)), 1),
            "max_width_px": int(max(widths)),
            "over_320px": over,
            "height_not_48": bad_h,
            "illegal_indep_vowel_medial": illegal,
            "symbol_line_counts": sym_counts,
            "hyphen_group_counts": hyp_counts,
            "cham_digit6_lines": cham6_cnt,
            "cham_ra_lines": ra_cnt,
        },
        "baseline_geometry_probes": geo,
        "violations": {
            "text_overlap": text_viol,
            "train_val_overlap": sorted(ov_tv)[:50],
            "substring": sub_viol[:50],
            "image_overlap": img_viol,
            "positional": mism[:50],
            "ast": ast_viol,
            "geometry": geo_fail,
        },
        "audits": {
            "audit_1_text_disjointness": a1,
            "audit_2_train_val_disjointness": a2,
            "audit_3_long_substring_disjointness": a3,
            "audit_4_image_sha256_disjointness_and_15000_shared": a4,
            "audit_5_positional_15000_identity": a5,
            "audit_6_ast_generator_blindness": a6,
            "audit_7_width_phonotactic_coverage_baseline": a7,
        },
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = os.path.join(REPORTS_DIR, "r2_1_7_contamination_audit.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 80)
    print(f"{'✅ ALL 7 AUDITS PASSED' if all_ok else '❌ AUDIT FAILED'} -> {out}")
    print("=" * 80)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run())
