#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Packages the V26.1 Masked-SVTR (Early-Backbone Frozen + BN-Locked + Paleographic Augmentation)
16-epoch 2-stage training bundle for Kaggle Dual Tesla T4x2:
  - Stage 1: warm-start weights-only (`v26_1_s1_base/best.pdparams` built from V26-Masked `soup_50e3_50e4`)
  - Stage 2: state-preserving resume (`v26_1_s1_e8/iter_epoch_8.{pdparams,pdopt,states}`)
Updates `r2_1_4c_base_and_configs.zip` in `ocr-training/output/r2_1_4d_training_bundle/` with
both V26.1 configs and all 6 SHA-256 verified overlays (incl. BN-locked `tools/program.py`).
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
DATA_DIR = os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_synthetic_v26_1")
STUDIO_PADDLE = os.path.join(PROJECT_ROOT, "ocr-studio", "PaddleOCR")
CFGS = [os.path.join(PROJECT_ROOT, "ocr-training", "configs", f"rec_cham_v26_1_s{s}.yml") for s in (1, 2)]

sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest().lower()


def ensure_v26_1_soup_base() -> str:
    """Builds `ocr-training/output/v26_1_base/best.pdparams` = 0.5*E3 + 0.5*E4 of V26-Masked (`OOD-180=26.33%`, `Symbols=100%`)."""
    out_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "v26_1_base")
    os.makedirs(out_dir, exist_ok=True)
    out_ckpt = os.path.join(out_dir, "best.pdparams")
    if os.path.exists(out_ckpt):
        return out_ckpt

    import paddle
    m_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_4c_v26_masked_ft_kaggle_output", "extracted_checkpoints")
    e3_p = os.path.join(m_dir, "iter_epoch_3.pdparams")
    e4_p = os.path.join(m_dir, "iter_epoch_4.pdparams")
    assert os.path.exists(e3_p) and os.path.exists(e4_p), f"Missing V26-Masked E3/E4 checkpoints in {m_dir}"
    e3 = paddle.load(e3_p)
    e4 = paddle.load(e4_p)
    blended = {k: 0.5 * e3[k] + 0.5 * e4[k] for k in e3.keys()}
    paddle.save(blended, out_ckpt)
    print(f"✅ Built V26.1 warm-start base (`soup_50e3_50e4`): {out_ckpt} (sha={sha(out_ckpt)[:16]}...)")
    return out_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--rebuild-dataset-zip", action="store_true")
    args = ap.parse_args()

    from build_and_launch_r2_1_4c import V26_EXTRA_PATCH_FILES
    assert os.path.isdir(BUNDLE_DIR), f"Missing {BUNDLE_DIR}"
    assert os.path.isdir(DATA_DIR), f"Missing {DATA_DIR}"
    assert all(os.path.exists(c) for c in CFGS), "Missing one or more V26.1 configs"

    baszip_path = os.path.join(BUNDLE_DIR, "r2_1_4c_base_and_configs.zip")
    tmp = os.path.join(PROJECT_ROOT, "ocr-training", "output", "_tmp_base_configs_v26_1")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    with zipfile.ZipFile(baszip_path, "r") as z:
        z.extractall(tmp)

    os.makedirs(os.path.join(tmp, "configs"), exist_ok=True)
    for c in CFGS:
        shutil.copy2(c, os.path.join(tmp, "configs", os.path.basename(c)))

    for old_sub in ("v26_base", "v26_s1_base", "v26_s1_e10", "v26_s2_e20", "v26_1_s1_base", "v26_1_s1_e8"):
        od = os.path.join(tmp, old_sub)
        if os.path.exists(od):
            shutil.rmtree(od)

    if args.stage == 1:
        src_ckpt = ensure_v26_1_soup_base()
        dst_dir = os.path.join(tmp, "v26_1_s1_base")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src_ckpt, os.path.join(dst_dir, "best.pdparams"))
        ckpt_sha = sha(os.path.join(dst_dir, "best.pdparams"))
        state_files = ["best.pdparams"]
    else:
        src_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_4c_v26_1_s1_kaggle_output", "extracted_checkpoints")
        dst_dir = os.path.join(tmp, "v26_1_s1_e8")
        os.makedirs(dst_dir, exist_ok=True)
        state_files = []
        for ext in (".pdparams", ".pdopt", ".states"):
            fn = f"iter_epoch_8{ext}"
            sp = os.path.join(src_dir, fn)
            assert os.path.exists(sp), f"Missing Stage 1 resume artifact: {sp}"
            shutil.copy2(sp, os.path.join(dst_dir, fn))
            state_files.append(fn)
        ckpt_sha = sha(os.path.join(dst_dir, "iter_epoch_8.pdparams"))

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

    for old_z in ("v26_ft_dataset.zip", "v26_full_dataset.zip"):
        op = os.path.join(BUNDLE_DIR, old_z)
        if os.path.exists(op):
            os.remove(op)

    zip_out = os.path.join(BUNDLE_DIR, "v26_1_dataset.zip")
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
    m_out = os.path.join(PROJECT_ROOT, "REPORTS", f"v26_1_s{args.stage}_bundle_manifest.json")
    with open(m_out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=2)
    print(json.dumps(man, indent=2))
    print(f"✅ V26.1 Stage {args.stage} bundle ready in {BUNDLE_DIR}")


if __name__ == "__main__":
    main()
