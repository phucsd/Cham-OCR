#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR Stage R4 Active Learning Export Utility
================================================
Converts structured Studio human-verified corrections (`ocr-studio/data/active_learning/corrections.jsonl`
+ `crops/<sha256_16>.png`) into a leakage-filtered, document-grouped train/val dataset ready for Stage R4 fine-tuning.
"""

import os
import sys
import json
import random
import shutil
import hashlib
import argparse
from datetime import datetime, timezone

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if os.path.join(PROJECT_ROOT, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def _load_allowed_chars(dict_path: str) -> set:
    with open(dict_path, "r", encoding="utf-8") as f:
        chars = {line.strip("\r\n") for line in f if line.strip("\r\n")}
    chars.add(" ")
    return chars


def _load_protected_sets():
    """Loads held-out benchmark, Frozen-838, Probe-100, and RealScan-Gold v1 texts & image hashes to block leakage."""
    protected_texts = set()
    protected_hashes = set()
    try:
        from audit_r2_1_4c_contamination import load_benchmark_texts, hash_directory_files, BENCHMARK_DIR, FROZEN_DIR
        from audit_r3_contamination import load_realscan_texts_and_hashes
        gt_sets, _ = load_benchmark_texts()
        for s in gt_sets.values():
            protected_texts.update(s)
        rs_texts, rs_hashes = load_realscan_texts_and_hashes()
        protected_texts.update(rs_texts)
        protected_hashes.update(rs_hashes)
        for d in (
            os.path.join(FROZEN_DIR, "crops"),
            os.path.join(BENCHMARK_DIR, "images"),
            os.path.join(BENCHMARK_DIR, "images_diagnostic"),
            os.path.join(BENCHMARK_DIR, "images_stress_500"),
            os.path.join(BENCHMARK_DIR, "images_advanced"),
        ):
            if os.path.exists(d):
                _, _, hs = hash_directory_files(d, extensions=[".jpg", ".png", ".jpeg"])
                protected_hashes.update(hs)
    except Exception as e:
        print(f"Warning: could not load full protected benchmark filter ({e}); proceeding with empty external filter.")
    return protected_texts, protected_hashes


def export_active_learning_dataset(
    input_jsonl: str,
    output_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    dict_path: str = None,
    crops_root: str = None,
) -> dict:
    assert os.path.exists(input_jsonl), f"Missing corrections JSONL: {input_jsonl}"
    dict_path = dict_path or os.path.join(PROJECT_ROOT, "ocr-studio", "data", "cham_dict_v24_1.txt")
    allowed_chars = _load_allowed_chars(dict_path)
    crops_root = crops_root or os.path.dirname(os.path.dirname(os.path.abspath(input_jsonl)))

    from generate_data import normalize_unicode

    protected_texts, protected_hashes = _load_protected_sets()

    # Deduplicate by (crop_sha256, normalized_text) or (crop_sha256) keeping the latest correction
    latest_by_crop = {}
    stats = {
        "total_rows": 0,
        "skipped_no_crop": 0,
        "skipped_empty_or_invalid_chars": 0,
        "skipped_protected_leak": 0,
        "dedup_kept": 0,
    }

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            stats["total_rows"] += 1
            rec = json.loads(raw_line)
            crop_rel = rec.get("crop_relpath")
            if not crop_rel:
                stats["skipped_no_crop"] += 1
                continue
            crop_abs = os.path.join(crops_root, crop_rel.replace("/", os.sep))
            if not os.path.exists(crop_abs):
                # Fallback if crop_rel is relative to input_jsonl parent directory
                alt_abs = os.path.join(os.path.dirname(os.path.abspath(input_jsonl)), os.path.basename(crop_rel))
                alt2_abs = os.path.join(os.path.dirname(os.path.abspath(input_jsonl)), "crops", os.path.basename(crop_rel))
                if os.path.exists(alt_abs):
                    crop_abs = alt_abs
                elif os.path.exists(alt2_abs):
                    crop_abs = alt2_abs
                else:
                    stats["skipped_no_crop"] += 1
                    continue

            crop_sha = (rec.get("crop_sha256") or _sha256_file(crop_abs)).lower()
            text = (rec.get("text") or "").strip()
            if any("\uAA00" <= ch <= "\uAA5F" for ch in text):
                text = normalize_unicode(text)
            if not text or any(ch not in allowed_chars for ch in text):
                stats["skipped_empty_or_invalid_chars"] += 1
                continue

            if text in protected_texts or crop_sha in protected_hashes:
                stats["skipped_protected_leak"] += 1
                continue

            # Keep latest correction per crop_sha256 (avoid conflicting multi-labels for identical image crop)
            prev = latest_by_crop.get(crop_sha)
            ts = rec.get("timestamp_utc") or ""
            if prev is None or ts >= prev["timestamp_utc"]:
                latest_by_crop[crop_sha] = {
                    "crop_sha256": crop_sha,
                    "crop_abs": crop_abs,
                    "text": text,
                    "image_name": str(rec.get("image_name") or crop_sha[:16]),
                    "timestamp_utc": ts,
                }

    items = list(latest_by_crop.values())
    stats["dedup_kept"] = len(items)

    # Document-grouped split by image_name so lines from one page never appear in both train and val
    by_doc = {}
    for it in items:
        by_doc.setdefault(it["image_name"], []).append(it)

    doc_names = sorted(by_doc.keys())
    rng = random.Random(seed)
    rng.shuffle(doc_names)

    n_docs = len(doc_names)
    if n_docs <= 1:
        val_docs = set()
    else:
        n_val_docs = max(1, int(round(n_docs * val_ratio)))
        if n_val_docs >= n_docs:
            n_val_docs = n_docs - 1
        val_docs = set(doc_names[:n_val_docs])

    train_items, val_items = [], []
    for d in doc_names:
        if d in val_docs:
            val_items.extend(by_doc[d])
        else:
            train_items.extend(by_doc[d])

    # Avoid identical non-empty text appearing in both splits if another doc has the exact same string
    train_texts = {it["text"] for it in train_items}
    val_items_clean = []
    for it in val_items:
        if it["text"] in train_texts:
            train_items.append(it)
        else:
            val_items_clean.append(it)
    val_items = val_items_clean

    imgs_out = os.path.join(output_dir, "images")
    os.makedirs(imgs_out, exist_ok=True)

    def _write_split(split_items, filename, prefix):
        lines = []
        for idx, it in enumerate(split_items):
            dst_name = f"al_r4_{prefix}_{idx:05d}_{it['crop_sha256'][:12]}.png"
            shutil.copy2(it["crop_abs"], os.path.join(imgs_out, dst_name))
            lines.append(f"images/{dst_name}\t{it['text']}\n")
        out_p = os.path.join(output_dir, filename)
        with open(out_p, "w", encoding="utf-8", newline="\n") as lf:
            lf.writelines(lines)
        return out_p, len(lines)

    tr_path, tr_n = _write_split(train_items, "train_list.txt", "tr")
    va_path, va_n = _write_split(val_items, "val_list.txt", "va")

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": os.path.abspath(input_jsonl),
        "output_dir": os.path.abspath(output_dir),
        "dict_path": os.path.abspath(dict_path),
        "seed": seed,
        "val_ratio": val_ratio,
        "stats": stats,
        "total_documents": n_docs,
        "val_documents": sorted(val_docs),
        "train_samples": tr_n,
        "val_samples": va_n,
        "train_list": tr_path,
        "val_list": va_path,
    }
    m_path = os.path.join(output_dir, "export_manifest.json")
    with open(m_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Export Studio active-learning corrections to Stage R4 dataset.")
    parser.add_argument(
        "--input-jsonl",
        default=os.path.join(PROJECT_ROOT, "ocr-studio", "data", "active_learning", "corrections.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "ocr-training", "data", "active_learning_r4"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = export_active_learning_dataset(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
