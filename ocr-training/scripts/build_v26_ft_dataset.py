#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds the V26 masked-attention fine-tune dataset (`ocr-training/data/cham_synthetic_v26_ft`).

Union of three already-audited W=480 synthetic sets (no new text is generated, so no new
contamination surface):
  * cham_synthetic_r4_w480      (REPORTS/r4_w480_contamination_audit.json)
  * cham_synthetic_r4_1_polish  (REPORTS/r4_1_polish_contamination_audit.json)
  * cham_synthetic_r4_2_inktight (REPORTS/r4_2_inktight_dataset_manifest.json)
Images are de-duplicated by SHA-256 of their bytes (R4.1 replays part of R4), and any image whose
hash appears in a validation split is removed from train (no train/val leakage).
Writes REPORTS/v26_ft_dataset_manifest.json.
"""

import os
import sys
import json
import shutil
import hashlib
from collections import Counter
from datetime import datetime, timezone

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(PROJECT_ROOT, "ocr-training", "data")
OUT_DIR = os.path.join(DATA_ROOT, "cham_synthetic_v26_ft")
SOURCES = [
    ("r4_w480", "cham_synthetic_r4_w480", "REPORTS/r4_w480_contamination_audit.json"),
    ("r4_1_polish", "cham_synthetic_r4_1_polish", "REPORTS/r4_1_polish_contamination_audit.json"),
    ("r4_2_inktight", "cham_synthetic_r4_2_inktight", "REPORTS/r4_2_inktight_dataset_manifest.json"),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def read_split(src_dir, split):
    rows = []
    with open(os.path.join(src_dir, f"{split}_label.txt"), "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n").rstrip("\r")
            if not ln.strip():
                continue
            rel, lab = ln.split("\t", 1)
            rows.append((rel, lab))
    return rows


def main():
    for _, folder, audit in SOURCES:
        assert os.path.isdir(os.path.join(DATA_ROOT, folder)), f"missing source {folder}"
        assert os.path.exists(os.path.join(PROJECT_ROOT, audit)), f"missing audit {audit}"
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    val_hashes = set()
    split_rows = {"train": [], "val": []}
    stats = {"train": Counter(), "val": Counter()}
    dup = {"train": 0, "val": 0}
    leak_removed = 0
    seen = {"train": set(), "val": set()}
    # validation first, so train can exclude anything that is in val
    for split in ("val", "train"):
        for tag, folder, _ in SOURCES:
            src_dir = os.path.join(DATA_ROOT, folder)
            for rel, lab in read_split(src_dir, split):
                sp = os.path.join(src_dir, *rel.split("/"))
                h = sha(sp)
                if h in seen[split]:
                    dup[split] += 1
                    continue
                if split == "train" and h in val_hashes:
                    leak_removed += 1
                    continue
                seen[split].add(h)
                if split == "val":
                    val_hashes.add(h)
                ext = os.path.splitext(rel)[1].lower() or ".jpg"
                new_rel = f"{split}/{tag}_{len(split_rows[split]):06d}{ext}"
                dp = os.path.join(OUT_DIR, *new_rel.split("/"))
                os.makedirs(os.path.dirname(dp), exist_ok=True)
                shutil.copy2(sp, dp)
                split_rows[split].append((new_rel, lab))
                stats[split][tag] += 1
        with open(os.path.join(OUT_DIR, f"{split}_label.txt"), "w", encoding="utf-8", newline="\n") as f:
            for rel, lab in split_rows[split]:
                f.write(f"{rel}\t{lab}\n")
        print(f"{split}: {len(split_rows[split]):,} unique images {dict(stats[split])} (dups skipped {dup[split]:,})")

    assert not (seen["train"] & val_hashes), "train/val leakage"
    n_tr = len(split_rows["train"])
    man = {
        "title": "Cham-OCR V26 masked-attention fine-tune dataset (union of audited R4/R4.1/R4.2 W=480 sets)",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": "ocr-training/data/cham_synthetic_v26_ft",
        "sources": [{"tag": t, "folder": f, "contamination_audit": a} for t, f, a in SOURCES],
        "train_samples": n_tr,
        "val_samples": len(split_rows["val"]),
        "train_by_source": dict(stats["train"]),
        "val_by_source": dict(stats["val"]),
        "duplicates_skipped": dup,
        "train_images_removed_for_val_overlap": leak_removed,
        "train_val_sha256_overlap": 0,
        "step_each_epoch_dual_t4_bs32": n_tr // 2 // 32,
    }
    out = os.path.join(PROJECT_ROOT, "REPORTS", "v26_ft_dataset_manifest.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: man[k] for k in ("train_samples", "val_samples", "duplicates_skipped", "train_images_removed_for_val_overlap", "step_each_epoch_dual_t4_bs32")}, indent=2))
    print(f"✅ V26 FT dataset -> {OUT_DIR}")


if __name__ == "__main__":
    main()
