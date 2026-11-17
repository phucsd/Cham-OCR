#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR Cham-DBNet v2.1 (`ch_PP-OCRv4_det_cham_v2_1` — Unified Wave + Multi-Layout) Kaggle Launcher
====================================================================================================
Warm-starts directly from `det_v2_checkpoints/latest.pdparams` (`905/905` unfused tensors, 100% H-Mean on Val-120)
and trains 20 epochs (`1,500` steps) on `detector_v3_1_hybrid` (`1,200` Train / `160` Val pages) on Kaggle Dual Tesla T4x2.
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
from build_and_launch_r2_1_4c import SOURCE_ZIP_HASH

DET_DATASET_SLUG = "gustavnguyen/cham-ocr-det-v2-training-assets"
DET_KERNEL_SLUG = "paddleocr-cham-det-v2-1-hybrid"
DET_KERNEL_TITLE = "PaddleOCR Cham Det v2 1 Hybrid"
BUNDLE_DIR = os.path.join(PROJECT_ROOT, "ocr-training", "output", "det_v2_1_training_bundle")


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def package_det_v2_1_bundle():
    if os.path.exists(BUNDLE_DIR):
        shutil.rmtree(BUNDLE_DIR)
    os.makedirs(BUNDLE_DIR, exist_ok=True)

    rec_bundle = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_4d_training_bundle")
    src_paddle_zip = os.path.join(rec_bundle, "paddleocr_source.zip")
    src_base_zip = os.path.join(rec_bundle, "r2_1_4c_base_and_configs.zip")
    det_data_dir = os.path.join(PROJECT_ROOT, "ocr-training", "data", "detector_v3_1_hybrid")
    det_cfg_path = os.path.join(PROJECT_ROOT, "ocr-training", "configs", "det", "ch_PP-OCRv4_det_cham_v2_1.yml")
    det_pretrain = os.path.join(PROJECT_ROOT, "ocr-training", "output", "det_v2_checkpoints", "latest.pdparams")

    assert all(os.path.exists(p) for p in (src_paddle_zip, det_data_dir, det_cfg_path, det_pretrain))
    shutil.copy2(src_paddle_zip, os.path.join(BUNDLE_DIR, "paddleocr_source.zip"))

    tmp_assets = os.path.join(PROJECT_ROOT, "ocr-training", "output", "_tmp_det_v2_1_assets")
    if os.path.exists(tmp_assets):
        shutil.rmtree(tmp_assets)
    os.makedirs(os.path.join(tmp_assets, "configs"), exist_ok=True)
    os.makedirs(os.path.join(tmp_assets, "pretrain"), exist_ok=True)

    with zipfile.ZipFile(src_base_zip, "r") as z:
        for member in z.namelist():
            if "patches/" in member:
                z.extract(member, tmp_assets)

    shutil.copy2(det_cfg_path, os.path.join(tmp_assets, "configs", "ch_PP-OCRv4_det_cham_v2_1.yml"))
    shutil.copy2(det_pretrain, os.path.join(tmp_assets, "pretrain", "cham_det_v2_best.pdparams"))

    assets_zip = os.path.join(BUNDLE_DIR, "det_v2_1_assets_and_patches.zip")
    with zipfile.ZipFile(assets_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(tmp_assets):
            for fn in files:
                fp = os.path.join(root, fn)
                z.write(fp, os.path.relpath(fp, tmp_assets).replace("\\", "/"))
    shutil.rmtree(tmp_assets)

    data_zip = os.path.join(BUNDLE_DIR, "detector_v3_1_hybrid_dataset.zip")
    parent_d = os.path.dirname(det_data_dir)
    with zipfile.ZipFile(data_zip, "w", zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(det_data_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                z.write(fp, os.path.relpath(fp, parent_d).replace("\\", "/"))

    meta = {"title": "Cham OCR Det v2 Training Assets", "id": DET_DATASET_SLUG, "licenses": [{"name": "CC0-1.0"}]}
    with open(os.path.join(BUNDLE_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)
    print("✅ Packaged Cham-DBNet v2.1 bundle ->", BUNDLE_DIR)


def upload_det_v2_1_dataset():
    u = init_kaggle_auth()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    assert api.config_values.get("username") == u

    api.dataset_create_version(folder=BUNDLE_DIR, version_notes=f"det_v2_1_hybrid ({datetime.now(timezone.utc).isoformat()})", dir_mode="zip")
    for i in range(60):
        time.sleep(10)
        try:
            st = getattr(api.dataset_status(DET_DATASET_SLUG), "name", str(api.dataset_status(DET_DATASET_SLUG))).lower()
        except Exception as e:
            st = f"processing ({e})"
        print(f"  [{(i+1)*10}s] {DET_DATASET_SLUG} status: {st}", flush=True)
        if "ready" in st:
            print("Waiting 70s for Kaggle mount propagation...", flush=True)
            time.sleep(70)
            return True
    raise RuntimeError("Dataset propagation timed out")


def generate_det_v2_1_kernel() -> str:
    k_dir = os.path.join(PROJECT_ROOT, "ocr-training", "kernels", DET_KERNEL_SLUG)
    os.makedirs(k_dir, exist_ok=True)
    pretrain_sha = sha256_file(os.path.join(PROJECT_ROOT, "ocr-training", "output", "det_v2_checkpoints", "latest.pdparams"))

    cells = [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, glob, json, hashlib, zipfile, shutil, sys\n",
                f"SOURCE_SHA = '{SOURCE_ZIP_HASH}'\n",
                f"PRETRAIN_SHA = '{pretrain_sha}'\n",
                "def sha(p):\n",
                "    h = hashlib.sha256()\n",
                "    with open(p, 'rb') as f:\n",
                "        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)\n",
                "    return h.hexdigest().lower()\n",
                "\n",
                "_ff = glob.glob('/kaggle/input/**/detector_v3_1_hybrid_dataset.zip', recursive=True) + [p for p in glob.glob('/kaggle/input/**/detector_v3_1_hybrid', recursive=True) if os.path.isdir(p)]\n",
                "if not _ff: raise FileNotFoundError('FAIL-FAST: detector_v3_1_hybrid not mounted yet!')\n",
                "print('✅ FAIL-FAST passed:', _ff[0])\n",
                "\n",
                "paddle_dir = '/kaggle/working/PaddleOCR'\n",
                "if os.path.exists(paddle_dir): shutil.rmtree(paddle_dir)\n",
                "szips = glob.glob('/kaggle/input/**/paddleocr_source.zip', recursive=True)\n",
                "sdirs = [p for p in glob.glob('/kaggle/input/**/paddleocr_source', recursive=True) if os.path.isdir(p)]\n",
                "if szips:\n",
                "    os.makedirs(paddle_dir, exist_ok=True)\n",
                "    with zipfile.ZipFile(szips[0]) as z: z.extractall(paddle_dir)\n",
                "else:\n",
                "    shutil.copytree(sdirs[0], paddle_dir)\n",
                "\n",
                "azips = glob.glob('/kaggle/input/**/det_v2_1_assets_and_patches.zip', recursive=True)\n",
                "adirs = [p for p in glob.glob('/kaggle/input/**/det_v2_1_assets_and_patches', recursive=True) if os.path.isdir(p)]\n",
                "assets_dir = '/kaggle/working/det_assets'\n",
                "if azips:\n",
                "    os.makedirs(assets_dir, exist_ok=True)\n",
                "    with zipfile.ZipFile(azips[0]) as z: z.extractall(assets_dir)\n",
                "else:\n",
                "    assets_dir = adirs[0] if adirs else '/kaggle/input'\n",
                "\n",
                "os.makedirs('/kaggle/working/data', exist_ok=True)\n",
                "dzips = glob.glob('/kaggle/input/**/detector_v3_1_hybrid_dataset.zip', recursive=True)\n",
                "ddirs = [p for p in glob.glob('/kaggle/input/**/detector_v3_1_hybrid', recursive=True) if os.path.isdir(p)]\n",
                "if dzips:\n",
                "    with zipfile.ZipFile(dzips[0]) as z: z.extractall('/kaggle/working/data')\n",
                "else:\n",
                "    os.symlink(ddirs[0], '/kaggle/working/data/detector_v3_1_hybrid')\n",
                "\n",
                "pt_src = glob.glob(f'{assets_dir}/**/cham_det_v2_best.pdparams', recursive=True) or glob.glob('/kaggle/input/**/cham_det_v2_best.pdparams', recursive=True)\n",
                "shutil.copy2(pt_src[0], '/kaggle/working/data/cham_det_v2_best.pdparams')\n",
                "assert sha('/kaggle/working/data/cham_det_v2_best.pdparams') == PRETRAIN_SHA\n",
                "\n",
                "cfg_src = glob.glob(f'{assets_dir}/**/ch_PP-OCRv4_det_cham_v2_1.yml', recursive=True) or glob.glob('/kaggle/input/**/ch_PP-OCRv4_det_cham_v2_1.yml', recursive=True)\n",
                "os.makedirs('/kaggle/working/configs', exist_ok=True)\n",
                "shutil.copy2(cfg_src[0], '/kaggle/working/configs/ch_PP-OCRv4_det_cham_v2_1.yml')\n",
                "\n",
                "# Apply NumPy/OpenCV sitecustomize + random_crop_data + make_border_map + make_shrink_map (seed=1024) patches\n",
                "sc_code = '''import numpy as np\nif not hasattr(np, 'sctypes'):\n    np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64], 'uint': [np.uint8, np.uint16, np.uint32, np.uint64], 'float': [np.float16, np.float32, np.float64], 'complex': [np.complex64, np.complex128], 'others': [bool, object, bytes, str]}\nif not hasattr(np, 'bool'): np.bool = bool\nif not hasattr(np, 'int'): np.int = int\nif not hasattr(np, 'float'): np.float = float\nif not hasattr(np, 'typeDict'): np.typeDict = {}\ntry:\n    import cv2; cv2.setNumThreads(0)\nexcept Exception: pass\n'''\n",
                "open('/kaggle/working/sitecustomize.py', 'w').write(sc_code)\n",
                "shutil.copy2('/kaggle/working/sitecustomize.py', os.path.join(paddle_dir, 'sitecustomize.py'))\n",
                "\n",
                "rcp = os.path.join(paddle_dir, 'ppocr', 'data', 'imaug', 'random_crop_data.py')\n",
                "_rc = open(rcp).read().replace('int(np.random.choice(axis, size=1))', 'int(np.random.choice(axis))')\n",
                "open(rcp, 'w').write(_rc)\n",
                "\n",
                "mbp = os.path.join(paddle_dir, 'ppocr', 'data', 'imaug', 'make_border_map.py')\n",
                "_mb = open(mbp).read().replace('if polygon_shape.area <= 0:', 'if polygon_shape.area < 1.0 or polygon_shape.length <= 0:')\n",
                "_mb = _mb.replace('padded_polygon = np.array(padding.Execute(distance)[0])', 'padded = padding.Execute(distance)\\n        if not padded:\\n            return\\n        padded_polygon = np.array(padded[0])')\n",
                "open(mbp, 'w').write(_mb)\n",
                "\n",
                "msp = os.path.join(paddle_dir, 'ppocr', 'data', 'imaug', 'make_shrink_map.py')\n",
                "_ms = open(msp).read().replace(\"kwargs['epoch'] != \\\"None\\\":\", \"kwargs['epoch'] != \\\"None\\\" and float(kwargs['epoch']) <= float(kwargs['total_epoch']):\")\n",
                "_ms = _ms.replace('np.append(possible_ratios, 1)', 'possible_ratios = np.append(possible_ratios, 1)')\n",
                "open(msp, 'w').write(_ms)\n",
                "print('✅ Cham-DBNet v2.1 setup & runtime patches verified.')\n",
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
                "subprocess.run([PY_EXE, '-c', 'import paddle; c = paddle.device.cuda.device_count(); print(\"GPUs:\", c); assert c >= 2'], check=True)\n",
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
                "cmd = [py_exe, '-u', '-m', 'paddle.distributed.launch', '--gpus', '0,1', 'tools/train.py', '-c', '/kaggle/working/configs/ch_PP-OCRv4_det_cham_v2_1.yml']\n",
                "t0 = time.time()\n",
                "proc = subprocess.Popen(cmd, cwd=paddle_dir)\n",
                "train_log_p = '/kaggle/working/output/ch_PP-OCRv4_det_cham_v2_1/train.log'\n",
                "last_step, last_step_ts, stall_warned = -1, time.time(), False\n",
                "while proc.poll() is None:\n",
                "    time.sleep(10)\n",
                "    if os.path.exists(train_log_p):\n",
                "        for _ln in open(train_log_p, 'r', encoding='utf-8', errors='replace'):\n",
                "            _m = re.search(r'global_step:\\s*(\\d+)', _ln)\n",
                "            if _m and int(_m.group(1)) > last_step:\n",
                "                last_step = int(_m.group(1)); last_step_ts = time.time(); stall_warned = False\n",
                "    idle_s = time.time() - last_step_ts\n",
                "    if last_step < 0 and idle_s >= 240:\n",
                "        proc.kill(); raise TimeoutError(f'Startup watchdog triggered ({idle_s:.0f}s)')\n",
                "    if idle_s >= 300 and not stall_warned:\n",
                "        print(f'⚠️ STALL WARNING (is_stalled=true): {idle_s:.0f}s at step {last_step}', flush=True); stall_warned = True\n",
                "    if idle_s >= 600:\n",
                "        proc.kill(); raise TimeoutError(f'DEADLOCK ({idle_s:.0f}s) at step {last_step}')\n",
                "print(f'DBNet v2.1 exited {proc.returncode} in {time.time() - t0:.1f}s (step={last_step})')\n",
                "assert proc.returncode == 0\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, shutil, json, subprocess\n",
                "out_dir = '/kaggle/working/output/ch_PP-OCRv4_det_cham_v2_1'\n",
                "zip_base = '/kaggle/working/checkpoints_paddleocr-cham-det-v2-1-hybrid'\n",
                "shutil.make_archive(zip_base, 'zip', out_dir)\n",
                "print('✅ First-Defense zip created:', os.path.getsize(zip_base + '.zip'))\n",
                "\n",
                "py_exe = open('/kaggle/working/py_exe.txt').read().strip()\n",
                "os.environ['PATH'] = f\"{os.path.dirname(py_exe)}:{os.environ.get('PATH', '')}\"\n",
                "eval_metrics = {}\n",
                "for ep in [10, 15, 'latest']:\n",
                "    cp = f'{out_dir}/iter_epoch_{ep}' if isinstance(ep, int) else f'{out_dir}/latest'\n",
                "    if os.path.exists(cp + '.pdparams'):\n",
                "        r = subprocess.run([py_exe, 'tools/eval.py', '-c', '/kaggle/working/configs/ch_PP-OCRv4_det_cham_v2_1.yml', '-o', f'Global.checkpoints={cp}', 'Global.distributed=False'], cwd=paddle_dir, capture_output=True, text=True)\n",
                "        eval_metrics[f'epoch_{ep}'] = r.stdout[-800:]\n",
                "with open('/kaggle/working/det_v2_1_val_eval.json', 'w') as f:\n",
                "    json.dump(eval_metrics, f, indent=2)\n",
                "subprocess.run([py_exe, 'tools/export_model.py', '-c', '/kaggle/working/configs/ch_PP-OCRv4_det_cham_v2_1.yml', '-o', f'Global.checkpoints={out_dir}/latest', 'Global.save_inference_dir=/kaggle/working/output/ch_PP-OCRv4_det_cham_v2_1_infer', 'Global.export_with_pir=False'], cwd=paddle_dir, check=True)\n",
                "for cd in ['/kaggle/working/PaddleOCR', '/kaggle/working/data', '/kaggle/working/configs', '/kaggle/working/det_assets']:\n",
                "    if os.path.exists(cd): shutil.rmtree(cd, ignore_errors=True)\n",
                "print('🎉 CHAM-DBNET V2.1 PIPELINE COMPLETE!')\n",
            ],
        },
    ]

    defined = set(dir(builtins))
    for i, c in enumerate(cells):
        tree = ast.parse("".join(c["source"]))
        assigned, loaded = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)): assigned.add(node.name)
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, (ast.Store, ast.Param)): assigned.add(node.id)
                elif isinstance(node.ctx, ast.Load): loaded.add(node.id)
            elif isinstance(node, ast.Import):
                for al in node.names: assigned.add(al.asname or al.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for al in node.names: assigned.add(al.asname or al.name)
            elif isinstance(node, ast.ExceptHandler) and node.name: assigned.add(node.name)
        undef = loaded - defined - assigned
        assert not undef, f"Undefined vars {undef} in cell {i}"
        defined.update(assigned)

    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.12"}}, "nbformat": 4, "nbformat_minor": 4}
    nb_p = os.path.join(k_dir, f"{DET_KERNEL_SLUG}.ipynb")
    with open(nb_p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    assert os.path.getsize(nb_p) < 1024 * 1024

    meta = {"id": f"gustavnguyen/{DET_KERNEL_SLUG}", "title": DET_KERNEL_TITLE, "code_file": f"{DET_KERNEL_SLUG}.ipynb", "language": "python", "kernel_type": "notebook", "is_gpu": True, "enable_gpu": True, "enable_internet": True, "accelerator": "NvidiaTeslaT4", "dataset_sources": [DET_DATASET_SLUG], "kernel_sources": []}
    with open(os.path.join(k_dir, "kernel-metadata.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)
    print(f"✅ AST-verified Det v2.1 kernel: {nb_p} ({os.path.getsize(nb_p):,} bytes)")
    return k_dir


def run_and_monitor_det_v2_1(push: bool = True):
    u = init_kaggle_auth()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    k_ref = f"{u}/{DET_KERNEL_SLUG}"

    if push:
        k_dir = generate_det_v2_1_kernel()
        res = api.kernels_push(k_dir)
        if getattr(res, "error", None):
            raise RuntimeError(f"Push error: {res.error}")
        print(f"🚀 Pushed {k_ref} (20 epochs * 75 steps = 1,500 steps)...")

    t0 = time.time()
    last_status = None
    last_hb = 0
    while True:
        el = time.time() - t0
        if el > 7200:
            raise TimeoutError(f"{k_ref} exceeded 120m monitor cap")
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
        if st == "running" and (time.time() - last_hb >= 60):
            print(f"[{el:.0f}s | {el/60:.1f}m] Status: {raw_st}", flush=True)
            last_hb = time.time()
        time.sleep(30)

    import socket
    socket.setdefaulttimeout(60)
    dl_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "det_v2_1_kaggle_output")
    for att in range(1, 6):
        try:
            if os.path.exists(dl_dir):
                shutil.rmtree(dl_dir)
            os.makedirs(dl_dir, exist_ok=True)
            print(f"[Attempt {att}/5] Downloading {k_ref} output -> {dl_dir}...", flush=True)
            api.kernels_output(k_ref, path=dl_dir)
            czips = [f for f in os.listdir(dl_dir) if f.startswith("checkpoints_") and f.endswith(".zip")]
            if czips and zipfile.is_zipfile(os.path.join(dl_dir, czips[0])):
                ext_d = os.path.join(dl_dir, "extracted_checkpoints")
                os.makedirs(ext_d, exist_ok=True)
                with zipfile.ZipFile(os.path.join(dl_dir, czips[0])) as z:
                    z.extractall(ext_d)
                print(f"✅ Extracted Det v2.1 checkpoints -> {ext_d}")
                break
        except Exception as exc:
            print(f"⚠️ Download attempt {att} failed: {exc}", flush=True)
            time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-only", action="store_true")
    ap.add_argument("--upload-only", action="store_true")
    ap.add_argument("--push-and-monitor", action="store_true")
    ap.add_argument("--run-e2e", action="store_true")
    ap.add_argument("--monitor-only", action="store_true")
    args = ap.parse_args()
    if args.package_only:
        package_det_v2_1_bundle()
    elif args.upload_only:
        upload_det_v2_1_dataset()
    elif args.push_and_monitor:
        run_and_monitor_det_v2_1(push=True)
    elif args.monitor_only:
        run_and_monitor_det_v2_1(push=False)
    elif args.run_e2e:
        package_det_v2_1_bundle()
        upload_det_v2_1_dataset()
        run_and_monitor_det_v2_1(push=True)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
