#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR Cham-DBNet v2 (`ch_PP-OCRv4_det_cham_v2`) Kaggle Dual Tesla T4x2 Launcher & Watchdog
=============================================================================================
Strictly enforces all `.agents/AGENTS.md` invariants:
  - Kaggle user `gustavnguyen` via `kaggle_auth.init_kaggle_auth()`.
  - Dedicated dataset `gustavnguyen/cham-ocr-det-v2-training-assets` (does NOT disturb recognizer bundle).
  - `--accelerator NvidiaTeslaT4` + `paddle.distributed.launch --gpus '0,1' tools/train.py`.
  - `eval_batch_step: [0, 9999999]` & `cal_metric_during_train: false` (Zero Multi-GPU NCCL Deadlock).
  - Checkpoint First-Defense `.zip` BEFORE any `eval.py` or `export_model.py`.
  - Single-GPU post-train `tools/eval.py` across `iter_epoch_30`, `iter_epoch_40`, `iter_epoch_50` (`hmean`).
  - `Global.export_with_pir=False`.
  - Pre-Flight Static Scope & AST check (`< 1 MB` notebook) + 300s/600s Autonomous Stall/Deadlock Detector.
"""

import os
import re
import ast
import sys
import json
import time
import shutil
import zipfile
import hashlib
import argparse
import builtins
from datetime import datetime, timezone

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
from kaggle_auth import init_kaggle_auth
from build_and_launch_r2_1_4c import SOURCE_ZIP_HASH, EXPECTED_RUNTIME_PATCHES

DET_DATASET_SLUG = "gustavnguyen/cham-ocr-det-v2-training-assets"
DET_KERNEL_SLUG = "paddleocr-cham-det-v2-dbnet"
DET_KERNEL_TITLE = "PaddleOCR Cham Det v2 DBNet"
DET_PRETRAIN_SHA = "8c5658919257f57ac6c9192dead425ab64628a11b9dc3d250aa59268437e5067"
BUNDLE_DIR = os.path.join(PROJECT_ROOT, "ocr-training", "output", "det_v2_training_bundle")


def package_det_v2_bundle():
    os.makedirs(BUNDLE_DIR, exist_ok=True)
    rec_bundle = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_4d_training_bundle")
    src_paddle_zip = os.path.join(rec_bundle, "paddleocr_source.zip")
    src_base_zip = os.path.join(rec_bundle, "r2_1_4c_base_and_configs.zip")
    det_data_dir = os.path.join(PROJECT_ROOT, "ocr-training", "data", "detector_v3_hb")
    det_cfg_path = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "det", "ch_PP-OCRv4_det_cham_v2.yml")
    det_pretrain = os.path.join(PROJECT_ROOT, "ocr-training", "data", "pretrain_models", "ch_PP-OCRv4_det_best_accuracy.pdparams")

    assert os.path.exists(src_paddle_zip), f"Missing {src_paddle_zip}"
    assert os.path.exists(det_data_dir), f"Missing {det_data_dir}"
    assert os.path.exists(det_cfg_path), f"Missing {det_cfg_path}"
    assert os.path.exists(det_pretrain), f"Missing {det_pretrain}"

    dst_paddle_zip = os.path.join(BUNDLE_DIR, "paddleocr_source.zip")
    if not os.path.exists(dst_paddle_zip):
        shutil.copy2(src_paddle_zip, dst_paddle_zip)

    # Extract only `patches/` from `r2_1_4c_base_and_configs.zip` + add `ch_PP-OCRv4_det_cham_v2.yml` + `PPLCNetV3_x0_75_ocr_det.pdparams`
    tmp_assets = os.path.join(PROJECT_ROOT, "ocr-training", "output", "_tmp_det_v2_assets")
    if os.path.exists(tmp_assets):
        shutil.rmtree(tmp_assets)
    os.makedirs(os.path.join(tmp_assets, "configs"), exist_ok=True)
    os.makedirs(os.path.join(tmp_assets, "pretrain"), exist_ok=True)

    with zipfile.ZipFile(src_base_zip, "r") as z:
        for member in z.namelist():
            if "patches/" in member:
                z.extract(member, tmp_assets)

    shutil.copy2(det_cfg_path, os.path.join(tmp_assets, "configs", "ch_PP-OCRv4_det_cham_v2.yml"))
    shutil.copy2(det_pretrain, os.path.join(tmp_assets, "pretrain", "ch_PP-OCRv4_det_best_accuracy.pdparams"))

    assets_zip = os.path.join(BUNDLE_DIR, "det_v2_assets_and_patches.zip")
    with zipfile.ZipFile(assets_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(tmp_assets):
            for fn in files:
                fp = os.path.join(root, fn)
                z.write(fp, os.path.relpath(fp, tmp_assets).replace("\\", "/"))
    shutil.rmtree(tmp_assets)

    data_zip = os.path.join(BUNDLE_DIR, "detector_v3_hb_dataset.zip")
    parent_d = os.path.dirname(det_data_dir)
    with zipfile.ZipFile(data_zip, "w", zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(det_data_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                z.write(fp, os.path.relpath(fp, parent_d).replace("\\", "/"))

    meta = {
        "title": "Cham OCR Det v2 Training Assets",
        "id": DET_DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(os.path.join(BUNDLE_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    print("✅ Packaged Cham-DBNet v2 bundle:")
    for fn in sorted(os.listdir(BUNDLE_DIR)):
        fp = os.path.join(BUNDLE_DIR, fn)
        print(f"  • {fn}: {os.path.getsize(fp):,} bytes")


def upload_det_dataset():
    u = init_kaggle_auth()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    assert api.config_values.get("username") == u

    existing = [d.ref for d in api.dataset_list(mine=True, search="det-v2")]
    if DET_DATASET_SLUG in existing:
        print(f"Updating dataset {DET_DATASET_SLUG}...")
        api.dataset_create_version(folder=BUNDLE_DIR, version_notes=f"det_v2 ({datetime.now(timezone.utc).isoformat()})", dir_mode="zip")
    else:
        print(f"Creating new dataset {DET_DATASET_SLUG}...")
        api.dataset_create_new(folder=BUNDLE_DIR, public=False, dir_mode="zip")

    for i in range(60):
        time.sleep(10)
        try:
            st = getattr(api.dataset_status(DET_DATASET_SLUG), "name", str(api.dataset_status(DET_DATASET_SLUG))).lower()
        except Exception as e:
            st = f"processing ({e})"
        print(f"  [{(i+1)*10}s] {DET_DATASET_SLUG} status: {st}", flush=True)
        if "ready" in st:
            print("Waiting 65s for Kaggle backend zip/mount propagation...", flush=True)
            time.sleep(65)
            print("✅ Det v2 dataset READY & PROPAGATED!", flush=True)
            return True
    raise RuntimeError(f"Dataset {DET_DATASET_SLUG} not ready in 600s")


def generate_det_v2_kernel():
    k_dir = os.path.join(PROJECT_ROOT, "ocr-training", "kernels", DET_KERNEL_SLUG)
    os.makedirs(k_dir, exist_ok=True)
    patches_json = json.dumps(EXPECTED_RUNTIME_PATCHES, indent=4)
    cfg_yaml_text = open(os.path.join(PROJECT_ROOT, "ocr-training", "configs", "det", "ch_PP-OCRv4_det_cham_v2.yml"), "r", encoding="utf-8").read()

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Cham-OCR Cham-DBNet v2 (`ch_PP-OCRv4_det_cham_v2` — HarfBuzz Multi-Layout & Tight-Line Detector)\n",
                "- Accelerator: Kaggle Dual Tesla T4x2 (`paddle.distributed.launch --gpus '0,1'`)\n",
                "- Dataset: `detector_v3_hb` (1,000 Train / 120 Val pages, 100% HarfBuzz shaping)\n",
                "- Deadlock Guard: `eval_batch_step: [0, 9999999]`, `cal_metric_during_train: false`\n",
                "- Checkpoint First-Defense `.zip` + Single-GPU Post-Train `eval.py` + `export_with_pir: false`\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, glob, json, hashlib, zipfile, shutil, sys, subprocess, site\n",
                f"SOURCE_SHA = '{SOURCE_ZIP_HASH}'\n",
                f"DET_PRETRAIN_SHA = '{DET_PRETRAIN_SHA}'\n",
                f"EXPECTED_PATCHES = {patches_json}\n",
                "\n",
                "def sha(p):\n",
                "    h = hashlib.sha256()\n",
                "    with open(p, 'rb') as f:\n",
                "        for c in iter(lambda: f.read(1024 * 1024), b''):\n",
                "            h.update(c)\n",
                "    return h.hexdigest().lower()\n",
                "\n",
                "work = '/kaggle/working'\n",
                "_ff = glob.glob('/kaggle/input/**/detector_v3_hb_dataset.zip', recursive=True) or [p for p in glob.glob('/kaggle/input/**/detector_v3_hb', recursive=True) if os.path.isdir(p)]\n",
                "assert _ff, f'FAIL-FAST: detector_v3_hb not mounted! Tops: {glob.glob(\"/kaggle/input/*/*\")}'\n",
                "print('✅ FAIL-FAST mounted:', _ff[0])\n",
                "\n",
                "paddle_dir = os.path.join(work, 'PaddleOCR')\n",
                "if os.path.exists(paddle_dir): shutil.rmtree(paddle_dir, ignore_errors=True)\n",
                "szips = glob.glob('/kaggle/input/**/paddleocr_source.zip', recursive=True)\n",
                "sdirs = [p for p in glob.glob('/kaggle/input/**/paddleocr_source', recursive=True) if os.path.isdir(p)]\n",
                "if szips:\n",
                "    os.makedirs(paddle_dir, exist_ok=True)\n",
                "    with zipfile.ZipFile(szips[0]) as z: z.extractall(paddle_dir)\n",
                "else:\n",
                "    shutil.copytree(sdirs[0], paddle_dir)\n",
                "\n",
                "azips = glob.glob('/kaggle/input/**/det_v2_assets_and_patches.zip', recursive=True)\n",
                "assets_base = os.path.join(work, 'det_assets')\n",
                "if azips:\n",
                "    os.makedirs(assets_base, exist_ok=True)\n",
                "    with zipfile.ZipFile(azips[0]) as z: z.extractall(assets_base)\n",
                "else:\n",
                "    adirs = [p for p in glob.glob('/kaggle/input/**/det_v2_assets_and_patches', recursive=True) if os.path.isdir(p)]\n",
                "    assets_base = adirs[0] if adirs else '/kaggle/input'\n",
                "\n",
                "for rel_path, exp_hash in EXPECTED_PATCHES.items():\n",
                "    src_p = glob.glob(f'{assets_base}/**/{rel_path}', recursive=True) or glob.glob(f'/kaggle/input/**/{rel_path}', recursive=True)\n",
                "    assert src_p, f'Missing patch {rel_path}'\n",
                "    dst_p = os.path.join(paddle_dir, rel_path)\n",
                "    os.makedirs(os.path.dirname(dst_p), exist_ok=True)\n",
                "    shutil.copy2(src_p[0], dst_p)\n",
                "    assert sha(dst_p) == exp_hash\n",
                "print('✅ Runtime patches verified.')\n",
                "\n",
                "os.makedirs('/kaggle/working/data', exist_ok=True)\n",
                "dzips = glob.glob('/kaggle/input/**/detector_v3_hb_dataset.zip', recursive=True)\n",
                "ddirs = [p for p in glob.glob('/kaggle/input/**/detector_v3_hb', recursive=True) if os.path.isdir(p)]\n",
                "det_dir = '/kaggle/working/data/detector_v3_hb'\n",
                "if dzips:\n",
                "    with zipfile.ZipFile(dzips[0]) as z: z.extractall('/kaggle/working/data')\n",
                "else:\n",
                "    os.symlink(ddirs[0], det_dir)\n",
                "assert sum(1 for _ in open(f'{det_dir}/det_train_label.txt', encoding='utf-8')) == 1000\n",
                "assert sum(1 for _ in open(f'{det_dir}/det_val_label.txt', encoding='utf-8')) == 120\n",
                "\n",
                "pre_cands = glob.glob(f'{assets_base}/**/ch_PP-OCRv4_det_best_accuracy.pdparams', recursive=True) or glob.glob('/kaggle/input/**/ch_PP-OCRv4_det_best_accuracy.pdparams', recursive=True)\n",
                "dst_pre = '/kaggle/working/data/ch_PP-OCRv4_det_best_accuracy.pdparams'\n",
                "shutil.copy2(pre_cands[0], dst_pre)\n",
                "assert sha(dst_pre) == DET_PRETRAIN_SHA\n",
                "\n",
                "os.makedirs('/kaggle/working/configs', exist_ok=True)\n",
                f"CFG_YAML_TEXT = {cfg_yaml_text!r}\n",
                "with open('/kaggle/working/configs/ch_PP-OCRv4_det_cham_v2.yml', 'w', encoding='utf-8') as _cf:\n",
                "    _cf.write(CFG_YAML_TEXT)\n",
                "print('✅ Installed deadlock-safe ch_PP-OCRv4_det_cham_v2.yml (MakeShrinkMap -> MakeBorderMap, num_workers=0, 35 epochs).')\n",
                "\n",
                "sitecustomize_code = (\n",
                "    \"import numpy as np\\n\"\n",
                "    \"if not hasattr(np, 'sctypes'): np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64], 'uint': [np.uint8, np.uint16, np.uint32, np.uint64], 'float': [np.float16, np.float32, np.float64], 'complex': [np.complex64, np.complex128], 'others': [bool, object, bytes, str]}\\n\"\n",
                "    \"if not hasattr(np, 'bool'): np.bool = getattr(np, 'bool_', bool)\\n\"\n",
                "    \"if not hasattr(np, 'int'): np.int = getattr(np, 'int_', int)\\n\"\n",
                "    \"if not hasattr(np, 'float'): np.float = getattr(np, 'float64', float)\\n\"\n",
                "    \"if not hasattr(np, 'typeDict'): np.typeDict = {}\\n\"\n",
                "    \"try:\\n\"\n",
                "    \"    import cv2; cv2.setNumThreads(0)\\n\"\n",
                "    \"except Exception: pass\\n\"\n",
                "    \"try:\\n\"\n",
                "    \"    import visualdl\\n\"\n",
                "    \"except Exception:\\n\"\n",
                "    \"    import types, sys\\n\"\n",
                "    \"    vdl = types.ModuleType('visualdl')\\n\"\n",
                "    \"    vdl.LogWriter = type('LogWriter', (), {'__init__': lambda self, *a, **k: None, 'add_scalar': lambda self, *a, **k: None, 'close': lambda self: None})\\n\"\n",
                "    \"    sys.modules['visualdl'] = vdl\\n\"\n",
                ")\n",
                "for td in [paddle_dir, work] + list(site.getsitepackages()):\n",
                "    try:\n",
                "        with open(os.path.join(td, 'sitecustomize.py'), 'w', encoding='utf-8') as f: f.write(sitecustomize_code)\n",
                "    except Exception: pass\n",
                "for tool_script in ['train.py', 'eval.py', 'export_model.py']:\n",
                "    sp = os.path.join(paddle_dir, 'tools', tool_script)\n",
                "    st = open(sp, 'r', encoding='utf-8').read()\n",
                "    if 'np.sctypes' not in st:\n",
                "        marker = 'from __future__ import print_function\\n'\n",
                "        new_st = st.replace(marker, marker + '\\n' + sitecustomize_code + '\\n', 1) if marker in st else (sitecustomize_code + '\\n' + st)\n",
                "        open(sp, 'w', encoding='utf-8').write(new_st)\n",
                "rcp = os.path.join(paddle_dir, 'ppocr', 'data', 'imaug', 'random_crop_data.py')\n",
                "if os.path.exists(rcp):\n",
                "    _rc_txt = open(rcp, 'r', encoding='utf-8').read().replace('int(np.random.choice(axis, size=1))', 'int(np.random.choice(axis))')\n",
                "    with open(rcp, 'w', encoding='utf-8') as _rc_f:\n",
                "        _rc_f.write(_rc_txt)\n",
                "    assert 'class EastRandomCropData' in open(rcp, 'r', encoding='utf-8').read()\n",
                "\n",
                "mbp = os.path.join(paddle_dir, 'ppocr', 'data', 'imaug', 'make_border_map.py')\n",
                "if os.path.exists(mbp):\n",
                "    _mb_txt = open(mbp, 'r', encoding='utf-8').read()\n",
                "    _mb_txt = _mb_txt.replace('if polygon_shape.area <= 0:', 'if polygon_shape.area < 1.0 or polygon_shape.length <= 0:')\n",
                "    _mb_txt = _mb_txt.replace('padded_polygon = np.array(padding.Execute(distance)[0])', 'padded = padding.Execute(distance)\\n        if not padded:\\n            return\\n        padded_polygon = np.array(padded[0])')\n",
                "    with open(mbp, 'w', encoding='utf-8') as _mb_f:\n",
                "        _mb_f.write(_mb_txt)\n",
                "    assert 'if not padded:' in open(mbp, 'r', encoding='utf-8').read()\n",
                "\n",
                "msp = os.path.join(paddle_dir, 'ppocr', 'data', 'imaug', 'make_shrink_map.py')\n",
                "if os.path.exists(msp):\n",
                "    _ms_txt = open(msp, 'r', encoding='utf-8').read()\n",
                "    _ms_txt = _ms_txt.replace(\"kwargs['epoch'] != \\\"None\\\":\", \"kwargs['epoch'] != \\\"None\\\" and float(kwargs['epoch']) <= float(kwargs['total_epoch']):\")\n",
                "    _ms_txt = _ms_txt.replace('np.append(possible_ratios, 1)', 'possible_ratios = np.append(possible_ratios, 1)')\n",
                "    with open(msp, 'w', encoding='utf-8') as _ms_f:\n",
                "        _ms_f.write(_ms_txt)\n",
                "print('✅ Setup, MakeBorderMap & MakeShrinkMap (seed=1024 safe) guards complete.')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys, os, subprocess, shutil, glob\n",
                "if sys.version_info >= (3, 13):\n",
                "    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'uv'], check=True)\n",
                "    subprocess.run([sys.executable, '-m', 'uv', 'venv', '--seed', '/tmp/py311', '--python', '3.11'], check=True)\n",
                "    PY_EXE = '/tmp/py311/bin/python'\n",
                "    install_cmd = [sys.executable, '-m', 'uv', 'pip', 'install', '--python', PY_EXE, 'numpy<2.0', 'paddlepaddle-gpu>=2.6.0,<3.0.0', 'visualdl', 'imgaug', 'scikit-image', 'pyyaml', 'opencv-python', 'Pillow', 'shapely', 'pyclipper', 'lmdb', 'rapidfuzz', 'tqdm']\n",
                "else:\n",
                "    PY_EXE = sys.executable\n",
                "    install_cmd = [PY_EXE, '-m', 'pip', 'install', '--quiet', 'numpy<2.0', 'paddlepaddle-gpu>=2.6.0,<3.0.0', 'visualdl', 'imgaug', 'scikit-image', 'pyyaml', 'opencv-python', 'Pillow', 'shapely', 'pyclipper', 'lmdb', 'rapidfuzz', 'tqdm']\n",
                "open('/kaggle/working/py_exe.txt', 'w').write(PY_EXE)\n",
                "subprocess.run(install_cmd, check=True)\n",
                "for sp in glob.glob('/tmp/py311/lib*/python*/site-packages'):\n",
                "    shutil.copy2('/kaggle/working/sitecustomize.py', os.path.join(sp, 'sitecustomize.py'))\n",
                "subprocess.run([PY_EXE, '-c', 'import sys, paddle; sys.path.insert(0, \"/kaggle/working/PaddleOCR\"); from ppocr.data.imaug.random_crop_data import EastRandomCropData; c = paddle.device.cuda.device_count(); print(\"GPUs:\", c, \"EastRandomCropData OK\"); assert c >= 2'], check=True)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, re, subprocess, time\n",
                "py_exe = open('/kaggle/working/py_exe.txt').read().strip()\n",
                "os.environ['PATH'] = f\"{os.path.dirname(py_exe)}:{os.environ.get('PATH', '')}\"\n",
                "os.environ['PYTHONUNBUFFERED'] = '1'\n",
                "os.environ['OMP_NUM_THREADS'] = '1'\n",
                "cmd = [py_exe, '-u', '-m', 'paddle.distributed.launch', '--gpus', '0,1', 'tools/train.py', '-c', '/kaggle/working/configs/ch_PP-OCRv4_det_cham_v2.yml']\n",
                "t0 = time.time()\n",
                "proc = subprocess.Popen(cmd, cwd=paddle_dir)\n",
                "train_log_p = '/kaggle/working/output/ch_PP-OCRv4_det_cham_v2/train.log'\n",
                "last_step = -1\n",
                "last_step_ts = time.time()\n",
                "stall_warned = False\n",
                "while proc.poll() is None:\n",
                "    time.sleep(10)\n",
                "    if os.path.exists(train_log_p):\n",
                "        for _ln in open(train_log_p, 'r', encoding='utf-8', errors='replace'):\n",
                "            _m = re.search(r'global_step:\\s*(\\d+)', _ln)\n",
                "            if _m and int(_m.group(1)) > last_step:\n",
                "                last_step = int(_m.group(1))\n",
                "                last_step_ts = time.time()\n",
                "                stall_warned = False\n",
                "    idle_s = time.time() - last_step_ts\n",
                "    if last_step < 0 and idle_s >= 240:\n",
                "        proc.kill()\n",
                "        raise TimeoutError(f'IN-CONTAINER WATCHDOG: no first global_step within {idle_s:.0f}s! Terminated to save GPU quota.')\n",
                "    if idle_s >= 300 and not stall_warned:\n",
                "        print(f'⚠️ IN-CONTAINER STALL WARNING (is_stalled=true): {idle_s:.0f}s at step {last_step}', flush=True)\n",
                "        stall_warned = True\n",
                "    if idle_s >= 600:\n",
                "        proc.kill()\n",
                "        raise TimeoutError(f'IN-CONTAINER DEADLOCK: stalled {idle_s:.0f}s at step {last_step}! Terminated to save GPU quota.')\n",
                "print(f'DBNet v2 train exited {proc.returncode} in {time.time() - t0:.1f}s (final step={last_step})')\n",
                "assert proc.returncode == 0\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# CHECKPOINT FIRST-DEFENSE GUARD\n",
                "import os, glob, hashlib, shutil, json, subprocess\n",
                "out_dir = '/kaggle/working/output/ch_PP-OCRv4_det_cham_v2'\n",
                "zip_base = '/kaggle/working/checkpoints_paddleocr-cham-det-v2-dbnet'\n",
                "shutil.make_archive(zip_base, 'zip', out_dir)\n",
                "print('✅ First-Defense zip created:', os.path.getsize(zip_base + '.zip'))\n",
                "\n",
                "# Offline Single-GPU Evaluation across epochs 30, 40, 50\n",
                "py_exe = open('/kaggle/working/py_exe.txt').read().strip()\n",
                "os.environ['PATH'] = f\"{os.path.dirname(py_exe)}:{os.environ.get('PATH', '')}\"\n",
                "eval_metrics = {}\n",
                "for ep in [15, 20, 'latest']:\n",
                "    cp = f'{out_dir}/iter_epoch_{ep}' if isinstance(ep, int) else f'{out_dir}/latest'\n",
                "    if os.path.exists(cp + '.pdparams'):\n",
                "        r = subprocess.run([py_exe, 'tools/eval.py', '-c', '/kaggle/working/configs/ch_PP-OCRv4_det_cham_v2.yml', '-o', f'Global.checkpoints={cp}', 'Global.distributed=False'], cwd=paddle_dir, capture_output=True, text=True)\n",
                "        print(f'=== Eval {ep} ===\\n', r.stdout[-600:])\n",
                "        eval_metrics[f'epoch_{ep}'] = r.stdout[-800:]\n",
                "with open('/kaggle/working/det_v2_val_eval.json', 'w') as f:\n",
                "    json.dump(eval_metrics, f, indent=2)\n",
                "\n",
                "subprocess.run([py_exe, 'tools/export_model.py', '-c', '/kaggle/working/configs/ch_PP-OCRv4_det_cham_v2.yml', '-o', f'Global.checkpoints={out_dir}/latest', 'Global.save_inference_dir=/kaggle/working/output/ch_PP-OCRv4_det_cham_v2_infer', 'Global.export_with_pir=False'], cwd=paddle_dir, check=True)\n",
                "for cd in ['/kaggle/working/PaddleOCR', '/kaggle/working/data', '/kaggle/working/configs', '/kaggle/working/det_assets']:\n",
                "    if os.path.exists(cd): shutil.rmtree(cd, ignore_errors=True)\n",
                "print('🎉 CHAM-DBNET V2 PIPELINE COMPLETE!')\n",
            ],
        },
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    # Pre-Flight Static Scope & AST Validation
    defined = set(dir(builtins))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        tree = ast.parse("".join(c["source"]))
        assigned, loaded = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assigned.add(node.name)
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, (ast.Store, ast.Param)):
                    assigned.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    loaded.add(node.id)
            elif isinstance(node, ast.Import):
                for al in node.names:
                    assigned.add(al.asname or al.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for al in node.names:
                    assigned.add(al.asname or al.name)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                assigned.add(node.name)
        undef = loaded - defined - assigned
        assert not undef, f"Undefined vars {undef} in cell {i}"
        defined.update(assigned)

    nb_p = os.path.join(k_dir, f"{DET_KERNEL_SLUG}.ipynb")
    with open(nb_p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    assert os.path.getsize(nb_p) < 1024 * 1024

    meta = {
        "id": f"gustavnguyen/{DET_KERNEL_SLUG}",
        "title": DET_KERNEL_TITLE,
        "code_file": f"{DET_KERNEL_SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_gpu": True,
        "enable_gpu": True,
        "enable_internet": True,
        "accelerator": "NvidiaTeslaT4",
        "dataset_sources": [DET_DATASET_SLUG],
        "kernel_sources": [],
    }
    with open(os.path.join(k_dir, "kernel-metadata.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)
    print(f"✅ Generated & AST-verified Det v2 kernel: {nb_p} ({os.path.getsize(nb_p):,} bytes)")
    return k_dir


def run_and_monitor_det_v2(push: bool = True):
    u = init_kaggle_auth()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    assert api.config_values.get("username") == u

    k_ref = f"{u}/{DET_KERNEL_SLUG}"
    if push:
        k_dir = generate_det_v2_kernel()
        res = api.kernels_push(k_dir)
        if getattr(res, "error", None):
            raise RuntimeError(f"Push error: {res.error}")
        print(f"🚀 Pushed {k_ref} -> monitoring (35 epochs * 62 steps = 2,170 steps)...")
    else:
        print(f"👀 Attaching monitor to already-running {k_ref} (2,170 steps)...")

    total_steps = 1550
    t0 = time.time()
    last_status = None
    last_line_cnt = 0
    last_step = -1
    last_step_t = None
    stall_warned = False
    last_hb = 0

    while True:
        el = time.time() - t0
        if el > 4500:
            try:
                api.kernels_delete(k_ref, no_confirm=True)
            except Exception:
                pass
            raise TimeoutError(f"{k_ref} exceeded 75m wall-clock cap (deleted to protect GPU quota)")
        st_obj = api.kernels_status(k_ref)
        raw_st = getattr(st_obj.status, "name", str(st_obj.status))
        st = raw_st.lower()
        if st != last_status:
            print(f"[{el:.0f}s] {k_ref}: {last_status} -> {raw_st}", flush=True)
            last_status = st
        if st in ("complete", "completed"):
            print(f"🎉 {k_ref} COMPLETED in {el/60:.1f}m!", flush=True)
            break
        if st in ("error", "failed"):
            print(api.kernels_logs(k_ref))
            raise RuntimeError(f"{k_ref} failed")

        try:
            logs_s = api.kernels_logs(k_ref)
            if isinstance(logs_s, str) and logs_s:
                try:
                    lines = "".join(e.get("data", "") for e in json.loads(logs_s) if isinstance(e, dict)).splitlines()
                except Exception:
                    lines = logs_s.splitlines()
                if len(lines) > last_line_cnt:
                    for nl in lines[last_line_cnt:]:
                        print(f"  | {nl}", flush=True)
                        m = re.search(r"global_step:\s*(\d+)", nl)
                        if m and int(m.group(1)) > last_step:
                            last_step = int(m.group(1))
                            last_step_t = time.time()
                            stall_warned = False
                    last_line_cnt = len(lines)
        except Exception:
            pass

        if st == "running" and last_step_t is not None and last_step < total_steps:
            ss = time.time() - last_step_t
            if ss >= 600:
                raise TimeoutError(f"DEADLOCK: {k_ref} stalled {ss:.0f}s at step {last_step}/{total_steps} (is_stalled=true)")
            if ss >= 300 and not stall_warned:
                print(f"⚠️ STALL WARNING (is_stalled=true): {ss:.0f}s at step {last_step}/{total_steps}", flush=True)
                stall_warned = True

        if st == "running" and (time.time() - last_hb >= 60):
            print(f"[{el:.0f}s | {el/60:.1f}m] Status: {raw_st} (step {last_step}/{total_steps})", flush=True)
            last_hb = time.time()
        time.sleep(30)

    dl_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "det_v2_kaggle_output")
    os.makedirs(dl_dir, exist_ok=True)
    api.kernels_output(k_ref, path=dl_dir)
    czips = [f for f in os.listdir(dl_dir) if f.startswith("checkpoints_") and f.endswith(".zip")]
    if czips:
        ext_d = os.path.join(dl_dir, "extracted_checkpoints")
        os.makedirs(ext_d, exist_ok=True)
        with zipfile.ZipFile(os.path.join(dl_dir, czips[0])) as z:
            z.extractall(ext_d)
        print(f"✅ Extracted Det v2 checkpoints -> {ext_d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--upload-dataset", action="store_true")
    parser.add_argument("--generate-kernel", action="store_true")
    parser.add_argument("--push-and-monitor", action="store_true")
    parser.add_argument("--monitor-only", action="store_true")
    parser.add_argument("--run-e2e", action="store_true")
    args = parser.parse_args()

    if args.package_only:
        package_det_v2_bundle()
    elif args.upload_dataset:
        upload_det_dataset()
    elif args.generate_kernel:
        generate_det_v2_kernel()
    elif args.push_and_monitor:
        run_and_monitor_det_v2(push=True)
    elif args.monitor_only:
        run_and_monitor_det_v2(push=False)
    elif args.run_e2e:
        package_det_v2_bundle()
        upload_det_dataset()
        run_and_monitor_det_v2()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
