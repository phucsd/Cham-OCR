#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R3 Contamination, Width, Phonotactic & Coverage Audit
=========================================================================
1. ZERO exact text & normalized text overlap with benchmarks, frozen suites, Probe-100, and RealScan-Gold v1.
2. ZERO internal train-vs-val non-empty text overlap.
3. ZERO long-substring (len >= 14) benchmark/RealScan-Gold leaks.
4. ZERO image SHA256 overlap with benchmark images & RealScan-Gold v1 crops; 20,000 / 20,000 unique hashes.
5. Static AST generator blindness (no benchmark/realscan path or filename literals in generate_data_v24_1_r3.py).
6. Newly rendered 25% degraded slices: 100% H=48 and W<=320px, 0 illegal [ꨁ..ꨅ][ꨳ..ꨶ], 114/114 dict coverage,
   all 14 symbols and 7 hyphen groups covered.
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

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-training", "scripts"))

from audit_r2_1_4c_contamination import (
    hash_directory_files,
    hash_single_file,
    load_label_file_texts,
    load_benchmark_texts,
    BENCHMARK_DIR,
    FROZEN_DIR,
)
from audit_r2_1_7_contamination import read_lines, classify_hyphen, SYMBOLS_14
from generate_data import normalize_unicode

DATA_DIR_R3 = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_synthetic_r3_degradation")
GENERATOR_R3 = os.path.join(PROJECT_ROOT, "ocr-training", "scripts", "generate_data_v24_1_r3.py")
REALSCAN_MANIFEST = os.path.join(BENCHMARK_DIR, "realscan_gold_v1", "manifest.json")
DICT_PATH = os.path.join(PROJECT_ROOT, "ocr-studio", "data", "cham_dict_v24_1.txt")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "REPORTS")


def load_realscan_texts_and_hashes():
    texts = set()
    img_hashes = set()
    if os.path.exists(REALSCAN_MANIFEST):
        m = json.load(open(REALSCAN_MANIFEST, "r", encoding="utf-8"))
        for doc in m.get("documents", []):
            if doc.get("page_relpath"):
                p = os.path.join(PROJECT_ROOT, doc["page_relpath"].replace("/", os.sep))
                if os.path.exists(p):
                    img_hashes.add(hash_single_file(p))
            items = list(doc.get("lines", doc.get("crops", [])))
            for grp_items in doc.get("groups", {}).values():
                if isinstance(grp_items, list):
                    items.extend(grp_items)
            for item in items:
                gt = (item.get("gt") or "").strip()
                if gt:
                    texts.add(gt)
                rp = item.get("crop_relpath")
                if rp:
                    cp = os.path.join(PROJECT_ROOT, rp.replace("/", os.sep))
                    if os.path.exists(cp):
                        img_hashes.add(hash_single_file(cp))
    return texts, img_hashes


def run():
    t0 = time.time()
    print("=" * 80)
    print("🛡️  STAGE R3 SCAN DEGRADATION 6-POINT CONTAMINATION & QUALITY AUDIT")
    print("=" * 80)

    # 1. Text disjointness vs benchmarks + RealScan-Gold v1
    print("\n[Audit 1/6] Exact & normalized text disjointness vs benchmarks & RealScan-Gold v1...")
    tr_texts = load_label_file_texts(os.path.join(DATA_DIR_R3, "train_label.txt"))
    va_texts = load_label_file_texts(os.path.join(DATA_DIR_R3, "val_label.txt"))
    all_texts = tr_texts | va_texts
    all_norm = {normalize_unicode(t) for t in all_texts}

    gt_sets, _ = load_benchmark_texts()
    rs_texts, rs_hashes = load_realscan_texts_and_hashes()
    assert len(rs_texts) == 25, f"Expected 25 RealScan-Gold v1 GT texts, got {len(rs_texts)}"
    assert len(rs_hashes) >= 65, f"Expected >= 65 unique RealScan-Gold v1 image hashes, got {len(rs_hashes)}"
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

    # 4. Image SHA256 disjointness + 20,000 unique hashes
    print("\n[Audit 4/6] Image SHA256 uniqueness & benchmark/RealScan-Gold disjointness...")
    files_r3, _, h_r3 = hash_directory_files(os.path.join(DATA_DIR_R3, "images"), extensions=[".jpg", ".png"])
    print(f"  • R3 images: {len(files_r3):,} (unique {len(h_r3):,}, expected 20,000)")
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
            ov = h_r3 & bh
            if ov:
                img_viol[bname] = len(ov)
                print(f"  ❌ {bname}: {len(ov)} collisions")
            else:
                print(f"  ✅ {bname}: {len(bf):,} images, 0 collisions")
    rs_ov = h_r3 & rs_hashes
    if rs_ov:
        img_viol["realscan_gold_v1"] = len(rs_ov)
        print(f"  ❌ realscan_gold_v1: {len(rs_ov)} collisions")
    else:
        print(f"  ✅ realscan_gold_v1: {len(rs_hashes)} images, 0 collisions")
    a4 = (not img_viol) and len(files_r3) == 20000 and len(h_r3) == 20000

    # 5. Static AST blindness
    print("\n[Audit 5/6] Static AST generator blindness...")
    tree = ast.parse(open(GENERATOR_R3, "r", encoding="utf-8").read())
    forbidden = [
        "held_out_diagnostic_probe", "frozen_suite", "benchmark_gt", "clean_frozen_200",
        "character_frozen_280", "symbol_frozen_14", "hyphen_frozen", "stress_frozen",
        "stress_base_8", "images_diagnostic", "images_stress_500", "ocr-benchmark",
        "realscan_gold", "harum_palei", "handoff_multiscript",
    ]
    ast_viol = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value.lower()
            for fl in forbidden:
                if fl in v:
                    ast_viol.append({"literal": fl, "line": node.lineno})
    a5 = not ast_viol
    print(f"  • Violations: {len(ast_viol)} -> {'✅' if a5 else '❌'}")

    # 6. Width cap (new 25%), phonotactics, 114/114 dict coverage, symbols/hyphens
    print("\n[Audit 6/6] Width cap, phonotactics, 114/114 token coverage & degradation counts...")
    tr_lines = read_lines(os.path.join(DATA_DIR_R3, "train_label.txt"))
    va_lines = read_lines(os.path.join(DATA_DIR_R3, "val_label.txt"))
    assert len(tr_lines) == 18000 and len(va_lines) == 2000

    ill_re = re.compile(r"[ꨁꨂꨃꨄꨅ][ꨳꨴꨵꨶ]")
    sym_counts = {s: 0 for s in SYMBOLS_14}
    hyp_counts = {}
    all_chars = set()
    tr_chars = set()
    illegal = 0
    bad_h = 0
    over_deg = 0
    deg_widths = []

    for split_lines, deg_start, is_tr in ((tr_lines, 13500, True), (va_lines, 1500, False)):
        for i, l in enumerate(split_lines):
            rel, t = l.split("\t", 1)
            all_chars.update(t)
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
            if i >= deg_start:
                with Image.open(os.path.join(DATA_DIR_R3, rel)) as im:
                    deg_widths.append(im.width)
                    if im.width > 320:
                        over_deg += 1
                    if im.height != 48:
                        bad_h += 1

    dict_tokens = [l.strip("\r\n") for l in open(DICT_PATH, "r", encoding="utf-8") if l.strip("\r\n")]
    missing_tokens = [tok for tok in dict_tokens if tok not in tr_chars]
    a6 = (
        over_deg == 0 and bad_h == 0 and illegal == 0
        and len(missing_tokens) == 0
        and all(v > 0 for v in sym_counts.values())
        and len(hyp_counts) == 7 and all(v > 0 for v in hyp_counts.values())
    )
    print(f"  • Degraded 5,000 width: mean={np.mean(deg_widths):.1f}px max={max(deg_widths)}px >320={over_deg} | H!=48={bad_h}")
    print(f"  • Illegal medials: {illegal} | Train dict coverage: {len(dict_tokens) - len(missing_tokens)}/{len(dict_tokens)} (missing={missing_tokens})")
    print(f"  • Symbol counts min={min(sym_counts.values())} | Hyphen groups ({len(hyp_counts)}/7): {hyp_counts}")
    print(f"  • Status: {'✅ PASSED' if a6 else '❌ FAILED'}")

    all_ok = all([a1, a2, a3, a4, a5, a6])
    report = {
        "title": "Cham-OCR v24.1 Stage R3 Scan Degradation 6-Point Audit",
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all_ok else "FAIL",
        "execution_duration_sec": round(time.time() - t0, 2),
        "degraded_5000": {
            "mean_width_px": round(float(np.mean(deg_widths)), 1),
            "max_width_px": int(max(deg_widths)),
            "over_320px": over_deg,
            "height_not_48": bad_h,
        },
        "symbol_line_counts": sym_counts,
        "hyphen_group_counts": hyp_counts,
        "missing_dict_tokens": missing_tokens,
        "audits": {
            "audit_1_text_disjointness": a1,
            "audit_2_train_val_disjointness": a2,
            "audit_3_long_substring_disjointness": a3,
            "audit_4_image_sha256_disjointness_and_20000_unique": a4,
            "audit_5_ast_generator_blindness": a5,
            "audit_6_width_phonotactics_coverage": a6,
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
    out = os.path.join(REPORTS_DIR, "r3_contamination_audit.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 80)
    print(f"{'✅ ALL 6 AUDITS PASSED' if all_ok else '❌ AUDIT FAILED'} -> {out}")
    print("=" * 80)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run())
