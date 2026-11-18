#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Packages the V26 masked-SVTR fine-tune (+ unmasked control) training bundle for Kaggle:
1. Updates `r2_1_4c_base_and_configs.zip` with
     configs/rec_cham_v26_masked_ft.yml, configs/rec_cham_v26_control_ft.yml,
     v26_base/best.pdparams (production R4.1 WiSE-FT checkpoint, SHA-256 verified),
     extra_patches/<5 patched PaddleOCR modeling files> (SHA-256 recorded for the notebook).
2. Removes the previous stage dataset zip(s) to keep the dataset version compact.
3. Creates `v26_ft_dataset.zip` containing `cham_synthetic_v26_ft/`.
"""

import os
import sys
import json
import zipfile
import shutil
import hashlib

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_DIR = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_4d_training_bundle")
DATA_DIR = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_synthetic_v26_ft")
CFGS = [os.path.join(PROJECT_ROOT, "ocr-training", "configs", n) for n in ("rec_cham_v26_masked_ft.yml", "rec_cham_v26_control_ft.yml")]
BASE_CKPT = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r4_1_wise83_ckpt", "best.pdparams")
BASE_SHA = "300b3ef5aec4417985bf70c0ac3548216f8c27272207747c77f28a69d50402a3"
STUDIO_PADDLE = os.path.join(PROJECT_ROOT, "ocr-studio", "PaddleOCR")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    from build_and_launch_r2_1_4c import V26_EXTRA_PATCH_FILES
    assert os.path.isdir(BUNDLE_DIR) and os.path.isdir(DATA_DIR)
    assert all(os.path.exists(c) for c in CFGS)
    assert sha(BASE_CKPT) == BASE_SHA, "base checkpoint hash mismatch"

    baszip_path = os.path.join(BUNDLE_DIR, "r2_1_4c_base_and_configs.zip")
    tmp = os.path.join(PROJECT_ROOT, "ocr-training", "output", "_tmp_base_configs_v26")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    with zipfile.ZipFile(baszip_path, "r") as z:
        z.extractall(tmp)

    os.makedirs(os.path.join(tmp, "configs"), exist_ok=True)
    for c in CFGS:
        shutil.copy2(c, os.path.join(tmp, "configs", os.path.basename(c)))
    os.makedirs(os.path.join(tmp, "v26_base"), exist_ok=True)
    shutil.copy2(BASE_CKPT, os.path.join(tmp, "v26_base", "best.pdparams"))

    extra_dir = os.path.join(tmp, "extra_patches")
    if os.path.exists(extra_dir):
        shutil.rmtree(extra_dir)
    patch_hashes = {}
    for rel in V26_EXTRA_PATCH_FILES:
        src = os.path.join(STUDIO_PADDLE, *rel.split("/"))
        dst = os.path.join(extra_dir, *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        patch_hashes[rel] = sha(dst)

    with zipfile.ZipFile(baszip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(tmp):
            for fn in files:
                full_p = os.path.join(root, fn)
                z.write(full_p, os.path.relpath(full_p, tmp).replace("\\", "/"))
    shutil.rmtree(tmp)

    for old_zip in ("r4_2_inktight_dataset.zip", "r4_1_polish_dataset.zip", "r4_w480_dataset.zip", "r3_1_closure_dataset.zip"):
        p_old = os.path.join(BUNDLE_DIR, old_zip)
        if os.path.exists(p_old):
            os.remove(p_old)

    zip_out = os.path.join(BUNDLE_DIR, "v26_ft_dataset.zip")
    parent = os.path.dirname(DATA_DIR)
    n = 0
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(DATA_DIR):
            for fn in files:
                full_p = os.path.join(root, fn)
                z.write(full_p, os.path.relpath(full_p, parent).replace("\\", "/"))
                n += 1
    man = {"base_ckpt_sha256": BASE_SHA, "extra_patches": patch_hashes, "dataset_zip": os.path.basename(zip_out),
           "dataset_files": n, "dataset_zip_bytes": os.path.getsize(zip_out),
           "base_and_configs_sha256": sha(baszip_path)}
    with open(os.path.join(PROJECT_ROOT, "REPORTS", "v26_bundle_manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=2)
    print(json.dumps(man, indent=2))
    print(f"✅ V26 bundle ready in {BUNDLE_DIR}")


if __name__ == "__main__":
    main()
