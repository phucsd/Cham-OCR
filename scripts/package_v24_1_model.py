#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Packages a winning V24.1 114-token checkpoint/inference dir into:
  - ocr-studio/data/output/rec_cham_inference_<model_id>/
      * inference.pdmodel, inference.pdiparams, inference.pdiparams.info (via PY26 export_with_pir=False)
      * inference.json (via PY313 PIR export)
      * inference.yml (copied from 114-token template rec_cham_inference_v24_1_r2_1_6/inference.yml)
      * cham_dict_v24_1.txt
      * model_manifest.json
  - ocr-studio/data/cham_dict_<model_id>.txt
And verifies SHA256s + studio app load under PY313.
"""

import os
import sys
import json
import shutil
import hashlib
import argparse
import subprocess
from datetime import datetime, timezone

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIO_DIR = os.path.join(PROJECT_ROOT, "ocr-studio")
PADDLE_DIR = os.path.join(STUDIO_DIR, "PaddleOCR")
PY26 = os.path.join(PROJECT_ROOT, ".venv_paddle26", "Scripts", "python.exe")
PY313 = r"C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe"
if not os.path.exists(PY313):
    PY313 = sys.executable


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def main():
    parser = argparse.ArgumentParser(description="Package 114-token V24.1 model into ocr-studio/data/output.")
    parser.add_argument("--model-id", required=True, help="e.g. v24_1_r3")
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--source-kernel", required=True)
    parser.add_argument("--ckpt-path", required=True, help="Path to iter_epoch_N.pdparams")
    parser.add_argument("--config-path", required=True, help="Path to training YAML config")
    parser.add_argument("--dict-path", default=os.path.join(PROJECT_ROOT, "ocr-training", "data", "cham_dict_v24_1.txt"))
    parser.add_argument("--rec-image-shape", default=None, help="Optional override e.g. '3, 48, 480' (auto-detected from config YAML)")
    parser.add_argument("--metrics-json", default=None, help="Optional JSON metrics blob")
    args = parser.parse_args()

    import yaml
    cfg_obj = yaml.safe_load(open(os.path.abspath(args.config_path), "r", encoding="utf-8"))
    d2s_shape = (cfg_obj.get("Global") or {}).get("d2s_train_image_shape") or [3, 48, 320]
    auto_rec_shape = f"{int(d2s_shape[0])}, {int(d2s_shape[1])}, {int(d2s_shape[2])}"
    rec_image_shape = args.rec_image_shape or auto_rec_shape
    rec_w = int([x.strip() for x in rec_image_shape.split(",")][2])

    ckpt_abs = os.path.abspath(args.ckpt_path)
    assert ckpt_abs.endswith(".pdparams") and os.path.exists(ckpt_abs), f"Missing {ckpt_abs}"
    ckpt_prefix = ckpt_abs[:-9]
    ckpt_sha = sha256_file(ckpt_abs)
    dict_abs = os.path.abspath(args.dict_path)
    assert os.path.exists(dict_abs), f"Missing dict: {dict_abs}"
    dict_tokens = [l.strip("\r\n") for l in open(dict_abs, "r", encoding="utf-8") if l.strip("\r\n")]
    assert len(dict_tokens) == 114, f"Expected 114 tokens, got {len(dict_tokens)}"

    out_dir = os.path.join(STUDIO_DIR, "data", "output", f"rec_cham_inference_{args.model_id}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1. PY26 export (inference.pdmodel + inference.pdiparams + inference.pdiparams.info)
    print(f"[1/4] Exporting .pdmodel/.pdiparams via PY26 -> {out_dir}...")
    subprocess.run([
        PY26, "tools/export_model.py",
        "-c", os.path.abspath(args.config_path),
        "-o",
        f"Global.checkpoints={ckpt_prefix}",
        f"Global.save_inference_dir={out_dir}",
        f"Global.character_dict_path={dict_abs}",
        "Global.export_with_pir=False",
    ], cwd=PADDLE_DIR, check=True)

    # 2. PY313 PIR export into temp dir -> copy inference.json
    tmp_pir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"_tmp_pir_{args.model_id}")
    if os.path.exists(tmp_pir):
        shutil.rmtree(tmp_pir)
    os.makedirs(tmp_pir, exist_ok=True)
    print(f"[2/4] Exporting PIR inference.json via PY313 -> {tmp_pir}...")
    subprocess.run([
        PY313, "tools/export_model.py",
        "-c", os.path.abspath(args.config_path),
        "-o",
        f"Global.checkpoints={ckpt_prefix}",
        f"Global.save_inference_dir={tmp_pir}",
        f"Global.character_dict_path={dict_abs}",
    ], cwd=PADDLE_DIR, check=True)
    shutil.copy2(os.path.join(tmp_pir, "inference.json"), os.path.join(out_dir, "inference.json"))
    shutil.rmtree(tmp_pir)

    # 3. Copy inference.yml + dictionary copies
    pir_yml = os.path.join(tmp_pir, "inference.yml") if os.path.exists(tmp_pir) else ""
    yml_tpl = os.path.join(STUDIO_DIR, "data", "output", "rec_cham_inference_v24_1_r2_1_6", "inference.yml")
    if os.path.exists(yml_tpl):
        yml_txt = open(yml_tpl, "r", encoding="utf-8").read()
        if rec_w != 320:
            yml_txt = yml_txt.replace("      - 3\n      - 48\n      - 320", f"      - 3\n      - 48\n      - {rec_w}")
            yml_txt = yml_txt.replace("          - 48\n          - 320\n", f"          - 48\n          - {rec_w}\n")
        with open(os.path.join(out_dir, "inference.yml"), "w", encoding="utf-8", newline="\n") as yf:
            yf.write(yml_txt)
    shutil.copy2(dict_abs, os.path.join(out_dir, "cham_dict_v24_1.txt"))
    studio_dict = os.path.join(STUDIO_DIR, "data", f"cham_dict_{args.model_id}.txt")
    shutil.copy2(dict_abs, studio_dict)

    # 4. Build model_manifest.json
    files_meta = {}
    for fn in sorted(os.listdir(out_dir)):
        fp = os.path.join(out_dir, fn)
        if os.path.isfile(fp):
            files_meta[fn] = {"size_bytes": os.path.getsize(fp), "sha256": sha256_file(fp)}

    manifest = {
        "model_id": args.model_id,
        "display_name": args.display_name,
        "stage": args.stage,
        "epoch": args.epoch,
        "packaged_at": datetime.now(timezone.utc).isoformat(),
        "source_kernel": args.source_kernel,
        "source_checkpoint": os.path.relpath(ckpt_abs, PROJECT_ROOT).replace("\\", "/"),
        "source_checkpoint_sha256": ckpt_sha,
        "dictionary_path": os.path.relpath(studio_dict, PROJECT_ROOT).replace("\\", "/"),
        "dictionary_sha256": sha256_file(studio_dict),
        "dictionary_tokens": len(dict_tokens),
        "rec_image_shape": rec_image_shape,
        "logical_order": True,
        "export_with_pir": False,
        "files": files_meta,
    }
    if args.metrics_json:
        manifest["benchmark_metrics"] = json.loads(args.metrics_json)

    manifest_path = os.path.join(out_dir, "model_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    # Verify load inside Studio app via PY313
    print(f"[4/4] Verifying Studio app.get_ocr_model('{args.model_id}') under PY313...")
    subprocess.run([
        PY313, "-c",
        f"import sys; sys.path.insert(0, {STUDIO_DIR!r}); import app; "
        f"m = app.get_ocr_model({args.model_id!r}); print('Loaded OK:', app.get_model_info({args.model_id!r})['recognizers'])"
    ], check=True)
    print(f"✅ Packaged `{args.model_id}` successfully -> {out_dir}")


if __name__ == "__main__":
    main()
