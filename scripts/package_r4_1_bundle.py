#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Packages Stage R4.1 Surgical Gate-Closure (`W = 480`) training bundle for Kaggle dataset upload:
1. Updates `r2_1_4c_base_and_configs.zip` with `configs/rec_cham_v24_1_r4_1_polish.yml` and `r4_best/best.pdparams`.
2. Removes previous `r4_w480_dataset.zip` to keep dataset upload fast & compact.
3. Creates `r4_1_polish_dataset.zip` containing `cham_synthetic_r4_1_polish/`.
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
DATA_DIR = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_synthetic_r4_1_polish")
CFG_PATH = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "rec_cham_v24_1_r4_1_polish.yml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", required=True)
    args = parser.parse_args()

    base_ckpt = os.path.abspath(args.base_ckpt)
    assert os.path.exists(BUNDLE_DIR) and os.path.exists(DATA_DIR) and os.path.exists(CFG_PATH) and os.path.exists(base_ckpt)

    baszip_path = os.path.join(BUNDLE_DIR, "r2_1_4c_base_and_configs.zip")
    tmp_extract = os.path.join(PROJECT_ROOT, "ocr-training", "output", "_tmp_base_configs_r4_1")
    if os.path.exists(tmp_extract):
        shutil.rmtree(tmp_extract)
    os.makedirs(tmp_extract, exist_ok=True)

    with zipfile.ZipFile(baszip_path, "r") as z:
        z.extractall(tmp_extract)

    configs_sub = os.path.join(tmp_extract, "configs")
    os.makedirs(configs_sub, exist_ok=True)
    shutil.copy2(CFG_PATH, os.path.join(configs_sub, os.path.basename(CFG_PATH)))

    r4_best_sub = os.path.join(tmp_extract, "r4_best")
    os.makedirs(r4_best_sub, exist_ok=True)
    shutil.copy2(base_ckpt, os.path.join(r4_best_sub, "best.pdparams"))

    with zipfile.ZipFile(baszip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(tmp_extract):
            for fn in files:
                full_p = os.path.join(root, fn)
                rel_p = os.path.relpath(full_p, tmp_extract).replace("\\", "/")
                z.write(full_p, rel_p)
    shutil.rmtree(tmp_extract)

    for old_zip in ("r4_w480_dataset.zip", "r3_1_closure_dataset.zip"):
        p_old = os.path.join(BUNDLE_DIR, old_zip)
        if os.path.exists(p_old):
            os.remove(p_old)

    zip_out = os.path.join(BUNDLE_DIR, "r4_1_polish_dataset.zip")
    parent_data = os.path.dirname(DATA_DIR)
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(DATA_DIR):
            for fn in files:
                full_p = os.path.join(root, fn)
                rel_p = os.path.relpath(full_p, parent_data).replace("\\", "/")
                z.write(full_p, rel_p)
    print(f"✅ Created {zip_out} ({os.path.getsize(zip_out):,} bytes)")


if __name__ == "__main__":
    main()
