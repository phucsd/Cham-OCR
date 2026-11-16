#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Packages the Stage R2.1.7 Gate-Closure training bundle for Kaggle dataset upload:
1. Updates `r2_1_4c_base_and_configs.zip` to include `rec_cham_v24_1_r2_1_7_gate_closure.yml`.
2. Creates `r2_1_7_gate_closure_dataset.zip` containing `cham_synthetic_r2_1_7_gate_closure/`.
3. Removes `r2_1_6_surgical_polish_dataset.zip` from `r2_1_4d_training_bundle` to keep upload ~35 MB.
"""

import os
import sys
import zipfile
import shutil

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_DIR = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_4d_training_bundle")
DATA_DIR_7 = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_synthetic_r2_1_7_gate_closure")
CFG_7_PATH = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "rec_cham_v24_1_r2_1_7_gate_closure.yml")


def main():
    assert os.path.exists(BUNDLE_DIR), f"Bundle dir missing: {BUNDLE_DIR}"
    assert os.path.exists(DATA_DIR_7), f"Dataset dir missing: {DATA_DIR_7}"
    assert os.path.exists(CFG_7_PATH), f"Config file missing: {CFG_7_PATH}"

    baszip_path = os.path.join(BUNDLE_DIR, "r2_1_4c_base_and_configs.zip")
    tmp_extract = os.path.join(PROJECT_ROOT, "ocr-training", "output", "_tmp_base_configs_7")
    if os.path.exists(tmp_extract):
        shutil.rmtree(tmp_extract)
    os.makedirs(tmp_extract, exist_ok=True)

    with zipfile.ZipFile(baszip_path, "r") as z:
        z.extractall(tmp_extract)

    configs_sub = os.path.join(tmp_extract, "configs")
    os.makedirs(configs_sub, exist_ok=True)
    shutil.copy2(CFG_7_PATH, os.path.join(configs_sub, os.path.basename(CFG_7_PATH)))

    with zipfile.ZipFile(baszip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(tmp_extract):
            for fn in files:
                full_p = os.path.join(root, fn)
                rel_p = os.path.relpath(full_p, tmp_extract).replace("\\", "/")
                z.write(full_p, rel_p)
    shutil.rmtree(tmp_extract)
    print(f"✅ Updated {baszip_path} ({os.path.getsize(baszip_path):,} bytes)")

    for old_zip in ("r2_1_5_multi_stream_dataset.zip", "r2_1_6_surgical_polish_dataset.zip"):
        p_old = os.path.join(BUNDLE_DIR, old_zip)
        if os.path.exists(p_old):
            os.remove(p_old)
            print(f"🧹 Removed old {p_old}")

    zip_7_path = os.path.join(BUNDLE_DIR, "r2_1_7_gate_closure_dataset.zip")
    parent_data = os.path.dirname(DATA_DIR_7)
    print(f"📦 Creating {zip_7_path}...")
    with zipfile.ZipFile(zip_7_path, "w", zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(DATA_DIR_7):
            for fn in files:
                full_p = os.path.join(root, fn)
                rel_p = os.path.relpath(full_p, parent_data).replace("\\", "/")
                z.write(full_p, rel_p)
    print(f"✅ Created {zip_7_path} ({os.path.getsize(zip_7_path):,} bytes)")

    print("\nBundle contents:")
    for fn in sorted(os.listdir(BUNDLE_DIR)):
        fp = os.path.join(BUNDLE_DIR, fn)
        print(f"  • {fn}: {os.path.getsize(fp):,} bytes")


if __name__ == "__main__":
    main()
