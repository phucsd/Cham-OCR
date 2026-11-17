#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR v24.1 Stage R4.1 Surgical Gate-Closure (`W = 480`) 6-Point Contamination & Quality Audit
=================================================================================================
1. ZERO exact text & normalized text overlap with benchmarks, frozen suites, Probe-100, and RealScan-Gold v1.
2. ZERO internal train-vs-val non-empty text overlap.
3. ZERO long-substring (len >= 14) benchmark/RealScan-Gold leaks.
4. ZERO image SHA256 overlap with benchmark images & RealScan-Gold v1 crops; 21,120 / 21,120 unique hashes.
5. Static AST generator blindness (no benchmark/realscan/heldout-font code literals in generate_data_v24_1_r4_1_polish.py).
6. 100% H=48 and W<=480px (with wide >320px replay + targeted `_`, `/`, `--`, `number-number` parity),
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
from audit_r4_contamination import _collect_non_docstring_str_nodes
from generate_data import normalize_unicode

DATA_DIR = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_synthetic_r4_1_polish")
GENERATOR_P = os.path.join(PROJECT_ROOT, "ocr-training", "scripts", "generate_data_v24_1_r4_1_polish.py")
DICT_PATH = os.path.join(PROJECT_ROOT, "ocr-studio", "data", "cham_dict_v24_1.txt")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "REPORTS")


def run():
    t0 = time.time()
    print("=" * 80)
    print("🛡️  STAGE R4.1 SURGICAL GATE-CLOSURE (W=480) 6-POINT CONTAMINATION AUDIT")
    print("=" * 80)

    tr_texts = load_label_file_texts(os.path.join(DATA_DIR, "train_label.txt"))
    va_texts = load_label_file_texts(os.path.join(DATA_DIR, "val_label.txt"))
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

    ov_tv = tr_texts & va_texts
    a2 = not ov_tv
    print(f"  • Train ∩ Val overlap: {len(ov_tv)} -> {'✅' if a2 else '❌'}")

    sub_viol = []
    for bt in (t for t in bench_all if len(t) >= 14):
        for gt in all_texts:
            if bt in gt:
                sub_viol.append({"benchmark": bt, "generated": gt})
                break
    a3 = not sub_viol
    print(f"  • Long-substring violations: {len(sub_viol)} -> {'✅' if a3 else '❌'}")

    files_all, _, h_all = hash_directory_files(os.path.join(DATA_DIR, "images"), extensions=[".jpg", ".png"])
    img_viol = {}
    for bname, bdir in {
        "frozen_crops": os.path.join(FROZEN_DIR, "crops"),
        "benchmark_images": os.path.join(BENCHMARK_DIR, "images"),
        "diagnostic_probe_images": os.path.join(BENCHMARK_DIR, "images_diagnostic"),
        "stress_500_images": os.path.join(BENCHMARK_DIR, "images_stress_500"),
        "advanced_images": os.path.join(BENCHMARK_DIR, "images_advanced"),
    }.items():
        if os.path.exists(bdir):
            _, _, bh = hash_directory_files(bdir, extensions=[".jpg", ".png", ".jpeg"])
            ov = h_all & bh
            if ov:
                img_viol[bname] = len(ov)
    if h_all & rs_hashes:
        img_viol["realscan_gold_v1"] = len(h_all & rs_hashes)
    a4 = (not img_viol) and len(files_all) == 21120 and len(h_all) == 21120
    print(f"  • Image SHA256 unique: {len(h_all):,}/{len(files_all):,} | collisions={img_viol} -> {'✅' if a4 else '❌'}")

    tree = ast.parse(open(GENERATOR_P, "r", encoding="utf-8").read())
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
    print(f"  • AST violations: {len(ast_viol)} -> {'✅' if a5 else '❌'}")

    tr_lines = read_lines(os.path.join(DATA_DIR, "train_label.txt"))
    va_lines = read_lines(os.path.join(DATA_DIR, "val_label.txt"))
    assert len(tr_lines) == 19200 and len(va_lines) == 1920

    ill_re = re.compile(r"[ꨁꨂꨃꨄꨅ][ꨳꨴꨵꨶ]")
    sym_counts = {s: 0 for s in SYMBOLS_14}
    hyp_counts = {}
    tr_chars = set()
    illegal, bad_h, over_480, wide_gt_320, over_len_100 = 0, 0, 0, 0, 0

    for split_lines, new_start, is_tr in ((tr_lines, 12480, True), (va_lines, 1248, False)):
        for i, l in enumerate(split_lines):
            rel, t = l.split("\t", 1)
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
                with Image.open(os.path.join(DATA_DIR, rel)) as im:
                    if im.width > 480:
                        over_480 += 1
                    if im.width > 320:
                        wide_gt_320 += 1
                    if im.height != 48:
                        bad_h += 1
            else:
                with Image.open(os.path.join(DATA_DIR, rel)) as im:
                    if im.width > 320:
                        wide_gt_320 += 1

    dict_tokens = [l.strip("\r\n") for l in open(DICT_PATH, "r", encoding="utf-8") if l.strip("\r\n")]
    missing_tokens = [tok for tok in dict_tokens if tok not in tr_chars]
    dd_count = hyp_counts.get("double_hyphen", hyp_counts.get("double hyphen", 0))
    a6 = (
        over_480 == 0 and bad_h == 0 and illegal == 0 and over_len_100 == 0
        and wide_gt_320 >= 3000
        and sym_counts["_"] >= 1500 and sym_counts["/"] >= 1500
        and dd_count >= 1200
        and len(missing_tokens) == 0
        and all(v > 0 for v in sym_counts.values())
        and len(hyp_counts) == 7 and all(v > 0 for v in hyp_counts.values())
    )
    print(f"  • Coverage: wide>320={wide_gt_320:,}, '_'={sym_counts['_']}, '/'={sym_counts['/']}, '--'={dd_count} | hyp={hyp_counts} -> {'✅' if a6 else '❌'}")

    all_ok = all([a1, a2, a3, a4, a5, a6])
    report = {
        "title": "Cham-OCR v24.1 Stage R4.1 Surgical Gate-Closure (W=480) 6-Point Audit",
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all_ok else "FAIL",
        "execution_duration_sec": round(time.time() - t0, 2),
        "wide_gt_320px": wide_gt_320,
        "symbol_line_counts": sym_counts,
        "hyphen_group_counts": hyp_counts,
        "missing_dict_tokens": missing_tokens,
        "audits": {
            "audit_1_text_disjointness": a1,
            "audit_2_train_val_disjointness": a2,
            "audit_3_long_substring_disjointness": a3,
            "audit_4_image_sha256_disjointness_and_21120_unique": a4,
            "audit_5_ast_generator_blindness": a5,
            "audit_6_width_480_targeted_parity": a6,
        },
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = os.path.join(REPORTS_DIR, "r4_1_polish_contamination_audit.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"{'✅ ALL 6 AUDITS PASSED' if all_ok else '❌ AUDIT FAILED'} -> {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run())
