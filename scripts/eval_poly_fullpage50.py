#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Targeted FullPage-Frozen 50 comparison (`fullpage_frozen_50.json`, 50 pages / 239 GT lines)
using `run_v24_frozen_suite.run_suite_fullpage` with fixed recognizer `rec_cham_inference_v24_1_r4`
(`3, 48, 480`) across detector configurations in `DET_MODE='hybrid'`.

Reuses `RecognizerEvaluator` and `run_suite_fullpage` directly and merges results into
`REPORTS/cham_dbnet_poly_rectify_evaluation.json` & `REPORTS/CHAM_DBNET_POLY_RECTIFY_EVALUATION.md`.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime, timezone

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
if not hasattr(np, "sctypes"):
    np.sctypes = {"int": [np.int8, np.int16, np.int32, np.int64], "uint": [np.uint8, np.uint16, np.uint32, np.uint64], "float": [np.float16, np.float32, np.float64], "complex": [np.complex64, np.complex128], "others": [bool, object, bytes, str]}
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "typeDict"):
    np.typeDict = {}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIO_DIR = os.path.join(PROJECT_ROOT, "ocr-studio")
BENCH_SCRIPTS = os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts")
for _p in (os.path.join(PROJECT_ROOT, "scripts"), STUDIO_DIR, BENCH_SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PY313 = r"C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe"
if not os.path.exists(PY313):
    PY313 = sys.executable

PARTS_DIR = os.path.join(PROJECT_ROOT, "ocr-benchmark", "results", "cham_det_poly_rectify")
OUT_JSON = os.path.join(PROJECT_ROOT, "REPORTS", "cham_dbnet_poly_rectify_evaluation.json")
OUT_MD = os.path.join(PROJECT_ROOT, "REPORTS", "CHAM_DBNET_POLY_RECTIFY_EVALUATION.md")
DEFAULT_CONFIGS = ["v1_quad", "v1_poly_u1.8", "v21_poly_vpad0.4", "v21_poly_u2.0_vpad0.4"]


def run_worker(cfg_name: str) -> dict:
    import app as studio_app
    import evaluate_cham_det_poly_rectify as ev_mod
    from run_v24_frozen_suite import RecognizerEvaluator, run_suite_fullpage

    cfg = ev_mod.CONFIGS[cfg_name]
    rec_dir = os.path.join(STUDIO_DIR, "data", "output", "rec_cham_inference_v24_1_r4")
    dict_p = os.path.join(STUDIO_DIR, "data", "cham_dict_v24_1_r4.txt")

    studio_app.DET_MODE = "hybrid"
    studio_app.det_model = studio_app.get_generic_det_model()
    studio_app._cham_det_failed = False
    studio_app.CHAM_DET_DB_THRESH = float(cfg["db_thresh"])
    studio_app.CHAM_DET_BOX_THRESH = float(cfg["box_thresh"])
    studio_app.CHAM_DET_UNCLIP = float(cfg["unclip"])
    studio_app.CHAM_DET_POLY_VPAD = float(cfg["vpad"])
    studio_app.CHAM_DET_POLY_MASK = bool(cfg["mask"])
    studio_app.CHAM_DET_BOX_TYPE = cfg["box_type"]
    studio_app.cham_det_model = studio_app._build_text_detector(
        os.path.abspath(ev_mod._det_dir(cfg["det"])),
        cfg["db_thresh"],
        cfg["box_thresh"],
        cfg["unclip"],
        box_type=cfg["box_type"],
    )

    ev = RecognizerEvaluator(
        model_dir=rec_dir,
        dict_path=dict_p,
        det_dir=os.path.abspath(ev_mod._det_dir(cfg["det"])),
        rec_image_shape="3, 48, 480",
    )
    ev.paddle_full = None
    ev.det_model = studio_app.det_model
    ev.has_detector = True

    t0 = time.time()
    res = run_suite_fullpage(ev)
    assert res is not None, "run_suite_fullpage returned None"
    out = {
        "config": cfg_name,
        "params": cfg,
        "overall_recall_pct": round(float(res["overall_recall"]) * 100.0, 2),
        "overall_precision_pct": round(float(res["overall_precision"]) * 100.0, 2),
        "matched_line_cer_pct": round(float(res["matched_line_cer"]) * 100.0, 2),
        "end_to_end_cer_pct": round(float(res["end_to_end_cer"]) * 100.0, 2),
        "l1_to_l4_recall_pct": round(float(res["l1_to_l4_recall"]) * 100.0, 2),
        "l1_to_l4_matched_cer_pct": round(float(res["l1_to_l4_matched_cer"]) * 100.0, 2),
        "l5_recall_pct": round(float(res["l5_recall"]) * 100.0, 2),
        "l5_matched_cer_pct": round(float(res["l5_matched_cer"]) * 100.0, 2),
        "raw": res,
        "seconds": round(time.time() - t0, 1),
    }
    return out


def merge_into_reports(configs):
    fp_data = {}
    for c in configs:
        p = os.path.join(PARTS_DIR, f"fullpage50__{c}.json")
        if os.path.exists(p):
            fp_data[c] = json.load(open(p, "r", encoding="utf-8"))

    if os.path.exists(OUT_JSON):
        rep = json.load(open(OUT_JSON, "r", encoding="utf-8"))
        rep["fullpage_frozen_50"] = fp_data
        with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)

    if os.path.exists(OUT_MD) and fp_data:
        txt = open(OUT_MD, "r", encoding="utf-8").read()
        marker = "\n## 5. FullPage-Frozen 50"
        if marker in txt:
            txt = txt.split(marker, 1)[0].rstrip() + "\n"
        lines = [
            txt.rstrip(),
            "",
            "## 5. FullPage-Frozen 50 (`fullpage_frozen_50.json`, 50 trang / 239 dòng GT)",
            "| Config | Overall Recall | Overall Precision | L1-4 Recall / Matched CER | L5 Recall / Matched CER | Overall Matched CER | End-to-End CER |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for c in configs:
            if c not in fp_data:
                continue
            r = fp_data[c]
            lines.append(
                f"| `{c}` | {r['overall_recall_pct']:.2f}% | {r['overall_precision_pct']:.2f}% | "
                f"{r['l1_to_l4_recall_pct']:.2f}% / {r['l1_to_l4_matched_cer_pct']:.2f}% | "
                f"{r['l5_recall_pct']:.2f}% / {r['l5_matched_cer_pct']:.2f}% | "
                f"{r['matched_line_cer_pct']:.2f}% | {r['end_to_end_cer_pct']:.2f}% |"
            )
        with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    print(f"✅ Merged FullPage-Frozen 50 results ({len(fp_data)} configs) into {OUT_JSON} & {OUT_MD}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None)
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(PARTS_DIR, exist_ok=True)
    if args.worker:
        out = run_worker(args.worker)
        p = os.path.join(PARTS_DIR, f"fullpage50__{args.worker}.json")
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"PASS {args.worker}: Recall={out['overall_recall_pct']}% L1-4={out['l1_to_l4_recall_pct']}%/{out['l1_to_l4_matched_cer_pct']}% L5={out['l5_recall_pct']}%/{out['l5_matched_cer_pct']}% Matched={out['matched_line_cer_pct']}%")
        return

    queue = [c for c in args.configs if args.force or not os.path.exists(os.path.join(PARTS_DIR, f"fullpage50__{c}.json"))]
    running = []
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    while queue or running:
        while queue and len(running) < args.jobs:
            c = queue.pop(0)
            log_p = os.path.join(PARTS_DIR, f"log__fullpage50_{c}.txt")
            fh = open(log_p, "w", encoding="utf-8")
            p = subprocess.Popen([PY313, os.path.abspath(__file__), "--worker", c], cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
            running.append((p, c, fh))
            print(f"▶️  started fullpage50 --worker {c} (pid {p.pid})", flush=True)
        time.sleep(3)
        for item in list(running):
            p, c, fh = item
            if p.poll() is not None:
                fh.close()
                running.remove(item)
                assert p.returncode == 0, f"fullpage50 worker {c} failed with rc={p.returncode}"
                print(f"✅ finished fullpage50 --worker {c}", flush=True)
    merge_into_reports(args.configs)


if __name__ == "__main__":
    main()
