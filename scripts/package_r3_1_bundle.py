#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Packages the Stage R3.1 Gate-Closure + Sesquisyllabic + Erosion bundle for Kaggle dataset upload:
1. Updates `r2_1_4c_base_and_configs.zip` to include:
   - `configs/rec_cham_v24_1_r3_1_closure.yml`
   - `r3_best/best.pdparams` (copied from --base-ckpt)
2. Removes previous dataset zips (`r3_degradation_dataset.zip`, etc.) to keep upload compact.
3. Creates `r3_1_closure_dataset.zip` containing `cham_synthetic_r3_1_closure/`.
"""

import os
import sys
import zipfile
import shutil
import argparse

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_DIR = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_4d_training_bundle")
DATA_DIR_R3_1 = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_synthetic_r3_1_closure")
CFG_R3_1_PATH = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "rec_cham_v24_1_r3_1_closure.yml")


def main():
    parser = argparse.ArgumentParser(description="Package Stage R3.1 Kaggle training bundle")
    parser.add_argument("--base-ckpt", required=True, help="Path to winning R3 E3 .pdparams file")
    args = parser.parse_args()

    base_ckpt = os.path.abspath(args.base_ckpt)
    assert os.path.exists(BUNDLE_DIR), f"Bundle dir missing: {BUNDLE_DIR}"
    assert os.path.exists(DATA_DIR_R3_1), f"Dataset dir missing: {DATA_DIR_R3_1}"
    assert os.path.exists(CFG_R3_1_PATH), f"Config file missing: {CFG_R3_1_PATH}"
    assert os.path.exists(base_ckpt), f"Base checkpoint missing: {base_ckpt}"

    baszip_path = os.path.join(BUNDLE_DIR, "r2_1_4c_base_and_configs.zip")
    tmp_extract = os.path.join(PROJECT_ROOT, "ocr-training", "output", "_tmp_base_configs_r3_1")
    if os.path.exists(tmp_extract):
        shutil.rmtree(tmp_extract)
    os.makedirs(tmp_extract, exist_ok=True)

    with zipfile.ZipFile(baszip_path, "r") as z:
        z.extractall(tmp_extract)

    configs_sub = os.path.join(tmp_extract, "configs")
    os.makedirs(configs_sub, exist_ok=True)
    shutil.copy2(CFG_R3_1_PATH, os.path.join(configs_sub, os.path.basename(CFG_R3_1_PATH)))

    r3_best_sub = os.path.join(tmp_extract, "r3_best")
    os.makedirs(r3_best_sub, exist_ok=True)
    shutil.copy2(base_ckpt, os.path.join(r3_best_sub, "best.pdparams"))

    with zipfile.ZipFile(baszip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(tmp_extract):
            for fn in files:
                full_p = os.path.join(root, fn)
                rel_p = os.path.relpath(full_p, tmp_extract).replace("\\", "/")
                z.write(full_p, rel_p)
    shutil.rmtree(tmp_extract)
    print(f"✅ Updated {baszip_path} ({os.path.getsize(baszip_path):,} bytes)")

    for old_zip in (
        "r2_1_5_multi_stream_dataset.zip",
        "r2_1_6_surgical_polish_dataset.zip",
        "r2_1_7_gate_closure_dataset.zip",
        "r3_degradation_dataset.zip",
    ):
        p_old = os.path.join(BUNDLE_DIR, old_zip)
        if os.path.exists(p_old):
            os.remove(p_old)
            print(f"🧹 Removed old {p_old}")

    zip_r3_1_path = os.path.join(BUNDLE_DIR, "r3_1_closure_dataset.zip")
    parent_data = os.path.dirname(DATA_DIR_R3_1)
    print(f"📦 Creating {zip_r3_1_path}...")
    with zipfile.ZipFile(zip_r3_1_path, "w", zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(DATA_DIR_R3_1):
            for fn in files:
                full_p = os.path.join(root, fn)
                rel_p = os.path.relpath(full_p, parent_data).replace("\\", "/")
                z.write(full_p, rel_p)
    print(f"✅ Created {zip_r3_1_path} ({os.path.getsize(zip_r3_1_path):,} bytes)")

    print("\nBundle contents:")
    for fn in sorted(os.listdir(BUNDLE_DIR)):
        fp = os.path.join(BUNDLE_DIR, fn)
        print(f"  • {fn}: {os.path.getsize(fp):,} bytes")


if __name__ == "__main__":
    main()
