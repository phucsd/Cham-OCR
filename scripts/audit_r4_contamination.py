#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R4 Wide-Canvas (`W = 480`) 6-Point Contamination & Quality Audit
====================================================================================
1. ZERO exact text & normalized text overlap with benchmarks, frozen suites, Probe-100, and RealScan-Gold v1.
2. ZERO internal train-vs-val non-empty text overlap.
3. ZERO long-substring (len >= 14) benchmark/RealScan-Gold leaks.
4. ZERO image SHA256 overlap with benchmark images & RealScan-Gold v1 crops; 26,400 / 26,400 unique hashes.
5. Static AST generator blindness (no benchmark/realscan/heldout-font code literals in generate_data_v24_1_r4_w480.py).
6. Newly rendered wide slices (17,160 images): 100% H=48 and W<=480px (with genuine wide >320px coverage),
   max_text_length < 100, 0 illegal [ꨁ..ꨅ][ꨳ..ꨶ], 114/114 dict coverage, all 14 symbols & 7 hyphen groups covered.
"""

import os
import sys
import ast
import re
import json
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-training", "scripts"))

from audit_r2_1_4c_contamination import (
    hash_directory_files,
    load_label_file_texts,
    load_benchmark_texts,
    BENCHMARK_DIR,
    FROZEN_DIR,
)
from audit_r2_1_7_contamination import read_lines, classify_hyphen, SYMBOLS_14
from audit_r3_contamination import load_realscan_texts_and_hashes
from generate_data import normalize_unicode

DATA_DIR_R4 = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_synthetic_r4_w480")
GENERATOR_R4 = os.path.join(PROJECT_ROOT, "ocr-training", "scripts", "generate_data_v24_1_r4_w480.py")
DICT_PATH = os.path.join(PROJECT_ROOT, "ocr-studio", "data", "cham_dict_v24_1.txt")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "REPORTS")


def _collect_non_docstring_str_nodes(tree: ast.AST):
    doc_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                val = node.body[0].value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    doc_nodes.add(id(val))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_nodes:
            out.append(node)
    return out


def run():
    t0 = time.time()
    print("=" * 80)
    print("🛡️  STAGE R4 WIDE-CANVAS (W=480) 6-POINT CONTAMINATION & QUALITY AUDIT")
    print("=" * 80)

    # 1. Text disjointness vs benchmarks + RealScan-Gold v1
    print("\n[Audit 1/6] Exact & normalized text disjointness vs benchmarks & RealScan-Gold v1...")
    tr_texts = load_label_file_texts(os.path.join(DATA_DIR_R4, "train_label.txt"))
    va_texts = load_label_file_texts(os.path.join(DATA_DIR_R4, "val_label.txt"))
    all_texts = tr_texts | va_texts
    all_norm = {normalize_unicode(t) for t in all_texts}

    gt_sets, _ = load_benchmark_texts()
    rs_texts, rs_hashes = load_realscan_texts_and_hashes()
    gt_sets["realscan_gold_v1"] = rs_texts

    bench_all = set()
    text_viol = {}
    for name, s in gt_sets.items():
        bench_all |= s
        ov = (all_texts & s) | (all_norm & {normalize_unicode(x) for x in s})
        if ov:
            text_viol[name] = sorted(ov)
            print(f"  ❌ {name}: {len(ov)} overlaps -> {sorted(ov)[:5]}")
        else:
            print(f"  ✅ {name}: {len(s):,} texts, 0 overlap")
    a1 = not text_viol

    # 2. Train vs Val overlap
    print("\n[Audit 2/6] Train ∩ Val non-empty text overlap...")
    ov_tv = tr_texts & va_texts
    a2 = not ov_tv
    print(f"  • Overlap: {len(ov_tv)} -> {'✅' if a2 else '❌'}")

    # 3. Long substring leaks
    print("\n[Audit 3/6] Long substring (len >= 14) containment...")
    sub_viol = []
    for bt in (t for t in bench_all if len(t) >= 14):
        for gt in all_texts:
            if bt in gt:
                sub_viol.append({"benchmark": bt, "generated": gt})
                break
    a3 = not sub_viol
    print(f"  • Violations: {len(sub_viol)} -> {'✅' if a3 else '❌'}")

    # 4. Image SHA256 disjointness + 26,400 unique hashes
    print("\n[Audit 4/6] Image SHA256 uniqueness & benchmark/RealScan-Gold disjointness...")
    files_r4, _, h_r4 = hash_directory_files(os.path.join(DATA_DIR_R4, "images"), extensions=[".jpg", ".png"])
    print(f"  • R4 images: {len(files_r4):,} (unique {len(h_r4):,}, expected 26,400)")
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
            ov = h_r4 & bh
            if ov:
                img_viol[bname] = len(ov)
                print(f"  ❌ {bname}: {len(ov)} collisions")
            else:
                print(f"  ✅ {bname}: {len(bf):,} images, 0 collisions")
    rs_ov = h_r4 & rs_hashes
    if rs_ov:
        img_viol["realscan_gold_v1"] = len(rs_ov)
        print(f"  ❌ realscan_gold_v1: {len(rs_ov)} collisions")
    else:
        print(f"  ✅ realscan_gold_v1: {len(rs_hashes)} images, 0 collisions")
    a4 = (not img_viol) and len(files_r4) == 26400 and len(h_r4) == 26400

    # 5. Static AST blindness
    print("\n[Audit 5/6] Static AST generator blindness (excluding docstrings)...")
    tree = ast.parse(open(GENERATOR_R4, "r", encoding="utf-8").read())
    forbidden = [
        "held_out_diagnostic_probe", "frozen_suite", "benchmark_gt", "clean_frozen_200",
        "character_frozen_280", "symbol_frozen_14", "hyphen_frozen", "stress_frozen",
        "stress_base_8", "images_diagnostic", "images_stress_500", "ocr-benchmark",
        "realscan_gold", "harum_palei", "handoff_multiscript", "fonts_heldout", "fonts_efeo",
    ]
    ast_viol = []
    for node in _collect_non_docstring_str_nodes(tree):
        v = node.value.lower()
        for fl in forbidden:
            if fl in v:
                ast_viol.append({"literal": fl, "line": node.lineno})
    a5 = not ast_viol
    print(f"  • Violations: {len(ast_viol)} -> {'✅' if a5 else '❌'}")

    # 6. Width cap (W <= 480px), wide (>320px) coverage, max_text_length < 100, phonotactics, 114/114 dict coverage
    print("\n[Audit 6/6] Width cap (W <= 480px), wide (>320px) count, phonotactics & 114/114 coverage...")
    tr_lines = read_lines(os.path.join(DATA_DIR_R4, "train_label.txt"))
    va_lines = read_lines(os.path.join(DATA_DIR_R4, "val_label.txt"))
    assert len(tr_lines) == 24000 and len(va_lines) == 2400, f"Got tr={len(tr_lines)} va={len(va_lines)}"

    ill_re = re.compile(r"[ꨁꨂꨃꨄꨅ][ꨳꨴꨵꨶ]")
    sym_counts = {s: 0 for s in SYMBOLS_14}
    hyp_counts = {}
    tr_chars = set()
    illegal = 0
    bad_h = 0
    over_480 = 0
    wide_gt_320 = 0
    over_len_100 = 0
    new_widths = []
    max_txt_len = 0

    for split_lines, new_start, is_tr in ((tr_lines, 8400, True), (va_lines, 840, False)):
        for i, l in enumerate(split_lines):
            rel, t = l.split("\t", 1)
            max_txt_len = max(max_txt_len, len(t))
            if len(t) >= 100:
                over_len_100 += 1
            if is_tr:
                tr_chars.update(t)
            if ill_re.search(t):
                illegal += 1
            for s in SYMBOLS_14:
                if s in t:
                    sym_counts[s] += 1
            g = classify_hyphen(t)
            if g:
                hyp_counts[g] = hyp_counts.get(g, 0) + 1
            if i >= new_start:
                with Image.open(os.path.join(DATA_DIR_R4, rel)) as im:
                    new_widths.append(im.width)
                    if im.width > 480:
                        over_480 += 1
                    if im.width > 320:
                        wide_gt_320 += 1
                    if im.height != 48:
                        bad_h += 1

    dict_tokens = [l.strip("\r\n") for l in open(DICT_PATH, "r", encoding="utf-8") if l.strip("\r\n")]
    missing_tokens = [tok for tok in dict_tokens if tok not in tr_chars]
    a6 = (
        over_480 == 0 and bad_h == 0 and illegal == 0 and over_len_100 == 0
        and wide_gt_320 >= 6000
        and len(missing_tokens) == 0
        and all(v > 0 for v in sym_counts.values())
        and len(hyp_counts) == 7 and all(v > 0 for v in hyp_counts.values())
    )
    print(f"  • Newly rendered 17,160 width: mean={np.mean(new_widths):.1f}px max={max(new_widths)}px (>320px={wide_gt_320:,}, >480px={over_480}) | H!=48={bad_h}")
    print(f"  • Max text len: {max_txt_len} (>=100={over_len_100}) | Illegal medials: {illegal} | Train dict coverage: {len(dict_tokens) - len(missing_tokens)}/{len(dict_tokens)}")
    print(f"  • Symbol counts min={min(sym_counts.values())} | Hyphen groups ({len(hyp_counts)}/7): {hyp_counts}")
    print(f"  • Status: {'✅ PASSED' if a6 else '❌ FAILED'}")

    all_ok = all([a1, a2, a3, a4, a5, a6])
    report = {
        "title": "Cham-OCR v24.1 Stage R4 Wide-Canvas (W=480) 6-Point Audit",
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all_ok else "FAIL",
        "execution_duration_sec": round(time.time() - t0, 2),
        "new_rendered_17160": {
            "mean_width_px": round(float(np.mean(new_widths)), 1),
            "max_width_px": int(max(new_widths)),
            "wide_gt_320px": wide_gt_320,
            "over_480px": over_480,
            "height_not_48": bad_h,
            "max_text_len": max_txt_len,
            "over_len_100": over_len_100,
        },
        "symbol_line_counts": sym_counts,
        "hyphen_group_counts": hyp_counts,
        "missing_dict_tokens": missing_tokens,
        "audits": {
            "audit_1_text_disjointness": a1,
            "audit_2_train_val_disjointness": a2,
            "audit_3_long_substring_disjointness": a3,
            "audit_4_image_sha256_disjointness_and_26400_unique": a4,
            "audit_5_ast_generator_blindness": a5,
            "audit_6_width_480_wide_coverage_phonotactics": a6,
        },
        "violations": {
            "text_overlap": text_viol,
            "train_val_overlap": sorted(ov_tv)[:50],
            "substring": sub_viol[:50],
            "image_overlap": img_viol,
            "ast": ast_viol,
        },
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = os.path.join(REPORTS_DIR, "r4_w480_contamination_audit.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 80)
    print(f"{'✅ ALL 6 AUDITS PASSED' if all_ok else '❌ AUDIT FAILED'} -> {out}")
    print("=" * 80)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run())
