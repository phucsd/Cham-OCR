#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Packages the Phase B V26-Full 30-epoch 3-stage training bundle for Kaggle:
  - Stage 1: warm-start weights-only (`v26_s1_base/best.pdparams` from V26 masked E4) + `v26_full_dataset.zip`
  - Stage 2: state-preserving resume (`v26_s1_e10/iter_epoch_10.{pdparams,pdopt,states}`)
  - Stage 3: state-preserving resume (`v26_s2_e20/iter_epoch_20.{pdparams,pdopt,states}`)
Updates `r2_1_4c_base_and_configs.zip` in `ocr-training/output/r2_1_4d_training_bundle/` with
all 3 V26-Full configs and all 6 SHA-256 verified overlays (incl. `tools/program.py`).
"""

import os
import sys
import json
import zipfile
import shutil
import hashlib
import argparse

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_DIR = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_4d_training_bundle")
DATA_DIR = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_synthetic_v26_full")
STUDIO_PADDLE = os.path.join(PROJECT_ROOT, "ocr-studio", "PaddleOCR")
CFGS = [os.path.join(PROJECT_ROOT, "ocr-training", "configs", f"rec_cham_v26_full_s{s}.yml") for s in (1, 2, 3)]

sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--rebuild-dataset-zip", action="store_true")
    args = ap.parse_args()

    from build_and_launch_r2_1_4c import V26_EXTRA_PATCH_FILES
    assert os.path.isdir(BUNDLE_DIR), f"Missing {BUNDLE_DIR}"
    assert os.path.isdir(DATA_DIR), f"Missing {DATA_DIR}"
    assert all(os.path.exists(c) for c in CFGS), "Missing one or more V26-Full configs"

    baszip_path = os.path.join(BUNDLE_DIR, "r2_1_4c_base_and_configs.zip")
    tmp = os.path.join(PROJECT_ROOT, "ocr-training", "output", "_tmp_base_configs_v26_full")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    with zipfile.ZipFile(baszip_path, "r") as z:
        z.extractall(tmp)

    os.makedirs(os.path.join(tmp, "configs"), exist_ok=True)
    for c in CFGS:
        shutil.copy2(c, os.path.join(tmp, "configs", os.path.basename(c)))

    # Clear older V26 base dirs inside base_and_configs so zip stays small and unambiguous
    for old_sub in ("v26_base", "v26_s1_base", "v26_s1_e10", "v26_s2_e20"):
        od = os.path.join(tmp, old_sub)
        if os.path.exists(od):
            shutil.rmtree(od)

    if args.stage == 1:
        src_ckpt = os.path.join(PROJECT_ROOT, "ocr-training", "output", "v26_masked_checkpoints", "iter_epoch_4.pdparams")
        assert os.path.exists(src_ckpt), f"Missing Stage 1 base ckpt: {src_ckpt}"
        dst_dir = os.path.join(tmp, "v26_s1_base")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src_ckpt, os.path.join(dst_dir, "best.pdparams"))
        ckpt_sha = sha(os.path.join(dst_dir, "best.pdparams"))
        state_files = ["best.pdparams"]
    elif args.stage == 2:
        src_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_4c_v26_full_s1_kaggle_output", "extracted_checkpoints")
        dst_dir = os.path.join(tmp, "v26_s1_e10")
        os.makedirs(dst_dir, exist_ok=True)
        state_files = []
        for ext in (".pdparams", ".pdopt", ".states"):
            fn = f"iter_epoch_10{ext}"
            sp = os.path.join(src_dir, fn)
            assert os.path.exists(sp), f"Missing Stage 1 resume artifact: {sp}"
            shutil.copy2(sp, os.path.join(dst_dir, fn))
            state_files.append(fn)
        ckpt_sha = sha(os.path.join(dst_dir, "iter_epoch_10.pdparams"))
    else:
        src_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_4c_v26_full_s2_kaggle_output", "extracted_checkpoints")
        dst_dir = os.path.join(tmp, "v26_s2_e20")
        os.makedirs(dst_dir, exist_ok=True)
        state_files = []
        for ext in (".pdparams", ".pdopt", ".states"):
            fn = f"iter_epoch_20{ext}"
            sp = os.path.join(src_dir, fn)
            assert os.path.exists(sp), f"Missing Stage 2 resume artifact: {sp}"
            shutil.copy2(sp, os.path.join(dst_dir, fn))
            state_files.append(fn)
        ckpt_sha = sha(os.path.join(dst_dir, "iter_epoch_20.pdparams"))

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

    old_ft = os.path.join(BUNDLE_DIR, "v26_ft_dataset.zip")
    if os.path.exists(old_ft):
        os.remove(old_ft)

    zip_out = os.path.join(BUNDLE_DIR, "v26_full_dataset.zip")
    n_files = None
    if args.rebuild_dataset_zip or not os.path.exists(zip_out):
        parent = os.path.dirname(DATA_DIR)
        n_files = 0
        print(f"Packing {DATA_DIR} -> {zip_out}...")
        with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_STORED) as z:
            for root, _, files in os.walk(DATA_DIR):
                for fn in files:
                    full_p = os.path.join(root, fn)
                    z.write(full_p, os.path.relpath(full_p, parent).replace("\\", "/"))
                    n_files += 1

    man = {
        "stage": args.stage,
        "base_ckpt_sha256": ckpt_sha,
        "staged_checkpoint_files": state_files,
        "extra_patches": patch_hashes,
        "dataset_zip": os.path.basename(zip_out),
        "dataset_zip_bytes": os.path.getsize(zip_out),
        "dataset_files": n_files,
        "base_and_configs_sha256": sha(baszip_path),
    }
    m_out = os.path.join(PROJECT_ROOT, "REPORTS", f"v26_full_s{args.stage}_bundle_manifest.json")
    with open(m_out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=2)
    print(json.dumps(man, indent=2))
    print(f"✅ V26-Full Stage {args.stage} bundle ready in {BUNDLE_DIR}")


if __name__ == "__main__":
    main()
