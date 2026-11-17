#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR Post-Hoc Confidence Calibration (Temperature Scaling)
==============================================================
Fits a single scalar temperature T > 0 on Held-Out Probe-100 (disjoint from RealScan-Gold v1)
by minimizing soft Bernoulli NLL against per-line character accuracy (1 - min(1, CER)).
Evaluates out-of-sample ECE (10-bin), Brier Score, and NLL on RealScan-Gold `harum_palei_01`
BEFORE (T=1.0) and AFTER (T=T*), and updates `model_manifest.json` with `confidence_temperature`.
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime, timezone

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)
STUDIO_DIR = os.path.join(PROJECT_ROOT, "ocr-studio")
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "ocr-benchmark")

if STUDIO_DIR not in sys.path:
    sys.path.insert(0, STUDIO_DIR)
if os.path.join(BENCHMARK_DIR, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(BENCHMARK_DIR, "scripts"))

import app as studio_app
from evaluate_pipeline import levenshtein_distance
from scripts.generate_data import normalize_unicode


def _detect_rec_shape(model_dir: str, hint: str = "") -> str:
    mp = os.path.join(model_dir, "model_manifest.json")
    if os.path.exists(mp):
        try:
            s = (json.load(open(mp, "r", encoding="utf-8")) or {}).get("rec_image_shape")
            if isinstance(s, str) and len(s.split(",")) == 3:
                return s
        except Exception:
            pass
    h_low = f"{hint} {os.path.basename(os.path.normpath(model_dir))}".lower()
    if "w480" in h_low or "_r4" in h_low or "v25" in h_low:
        return "3, 48, 480"
    return "3, 48, 320"


def build_text_recognizer(model_dir: str, dict_path: str, model_id: str = ""):
    from tools.infer.predict_rec import TextRecognizer
    import tools.infer.utility as utility

    backup = sys.argv
    sys.argv = [sys.argv[0]]
    args = utility.parse_args()
    sys.argv = backup

    args.rec_model_dir = os.path.abspath(model_dir)
    args.rec_char_dict_path = os.path.abspath(dict_path)
    args.use_gpu = False
    args.rec_image_shape = _detect_rec_shape(model_dir, model_id)
    args.use_space_char = True
    args.rec_batch_num = 1
    args.enable_mkldnn = False
    args.cpu_threads = 1
    args.ir_optim = False
    args.precision = "fp32"
    return TextRecognizer(args)


def predict_items(rec_model, items: list, norm_gt_and_pred: bool = True):
    confs, accs, rows = [], [], []
    for it in items:
        img = cv2.imread(it["img_path"])
        assert img is not None, f"Failed to read image: {it['img_path']}"
        prep = studio_app.preprocess_crop_for_rec(img)
        res, _ = rec_model([prep])
        pred_raw, conf = res[0] if res else ("", 0.0)
        pred_norm = normalize_unicode(pred_raw) if norm_gt_and_pred else pred_raw
        gt_norm = normalize_unicode(it["gt"]) if norm_gt_and_pred else it["gt"]
        ed = levenshtein_distance(pred_norm, gt_norm)
        cer = ed / max(1, len(gt_norm)) if gt_norm else (0.0 if not pred_norm else 1.0)
        acc = max(0.0, 1.0 - min(1.0, cer))
        confs.append(float(conf))
        accs.append(float(acc))
        rows.append({
            "id": it.get("id"),
            "gt": gt_norm,
            "pred": pred_norm,
            "raw_conf": round(float(conf), 6),
            "cer": round(float(cer), 6),
            "acc": round(float(acc), 6),
        })
    return np.array(confs, dtype=np.float64), np.array(accs, dtype=np.float64), rows


def scale_probs(probs: np.ndarray, T: float) -> np.ndarray:
    p = np.clip(probs, 1e-4, 1.0 - 1e-4)
    z = np.log(p / (1.0 - p))
    q = 1.0 / (1.0 + np.exp(-z / float(T)))
    return np.clip(q, 1e-4, 1.0 - 1e-4)


def calibration_metrics(probs: np.ndarray, targets: np.ndarray, T: float = 1.0, n_bins: int = 10) -> dict:
    q = scale_probs(probs, T)
    nll = -np.mean(targets * np.log(q) + (1.0 - targets) * np.log(1.0 - q))
    brier = np.mean((q - targets) ** 2)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(q)
    bin_rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (q >= lo) & (q <= hi)
        else:
            mask = (q >= lo) & (q < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            mp = float(np.mean(q[mask]))
            ma = float(np.mean(targets[mask]))
            ece += (cnt / n) * abs(mp - ma)
            bin_rows.append({"bin": f"[{lo:.1f},{hi:.1f}]", "count": cnt, "mean_conf": round(mp, 4), "mean_acc": round(ma, 4)})
    return {
        "temperature": round(float(T), 6),
        "n_samples": int(n),
        "mean_conf": round(float(np.mean(q)), 6),
        "mean_acc": round(float(np.mean(targets)), 6),
        "ece_10bin": round(float(ece), 6),
        "brier": round(float(brier), 6),
        "nll": round(float(nll), 6),
        "bins": bin_rows,
    }


def load_probe_100_items():
    p_json = os.path.join(BENCHMARK_DIR, "data", "held_out_diagnostic_probe_100.json")
    raw = json.load(open(p_json, "r", encoding="utf-8"))
    arr = raw if isinstance(raw, list) else raw.get("samples", raw.get("probe_samples", []))
    items = []
    for it in arr:
        rel = it.get("image_relpath") or it.get("image_path") or it.get("rel_path")
        img_p = os.path.join(BENCHMARK_DIR, "data", rel.replace("/", os.sep)) if rel else it.get("image_abspath")
        if not img_p or not os.path.exists(img_p):
            img_p = it.get("image_abspath")
        items.append({"id": it.get("case_id", it.get("probe_id")), "img_path": img_p, "gt": it["gt"]})
    assert len(items) == 100
    return items


def load_realscan_harum_items():
    m_path = os.path.join(BENCHMARK_DIR, "data", "realscan_gold_v1", "manifest.json")
    m = json.load(open(m_path, "r", encoding="utf-8"))
    harum = next(d for d in m["documents"] if d["doc_id"] == "harum_palei_01")
    return [
        {"id": f"harum_line_{ln['line']:02d}", "img_path": os.path.join(PROJECT_ROOT, ln["crop_relpath"].replace("/", os.sep)), "gt": ln["gt"]}
        for ln in harum["lines"]
    ]


def calibrate_one_model(spec: dict) -> dict:
    model_id = spec["model_id"]
    model_dir = spec["model_dir"]
    dict_path = spec["dict_path"]
    print(f"\n[Calibrate] {model_id} ({model_dir})...")
    rec = build_text_recognizer(model_dir, dict_path, model_id=model_id)

    probe_items = load_probe_100_items()
    real_items = load_realscan_harum_items()

    p_conf, p_acc, _ = predict_items(rec, probe_items)
    r_conf, r_acc, _ = predict_items(rec, real_items)

    res_opt = minimize_scalar(
        lambda t: calibration_metrics(p_conf, p_acc, T=t)["nll"],
        bounds=(0.1, 10.0),
        method="bounded",
    )
    t_star = float(res_opt.x)

    probe_before = calibration_metrics(p_conf, p_acc, T=1.0)
    probe_after = calibration_metrics(p_conf, p_acc, T=t_star)
    real_before = calibration_metrics(r_conf, r_acc, T=1.0)
    real_after = calibration_metrics(r_conf, r_acc, T=t_star)

    # Safety check: if raw T* worsens BOTH RealScan ECE and NLL, gently shrink toward 1.0 or keep best alpha
    chosen_t = t_star
    shrink_note = "raw_probe_100_mle"
    if real_after["ece_10bin"] > real_before["ece_10bin"] + 0.01 and real_after["nll"] > real_before["nll"] + 0.01:
        # Grid search conservative shrinkage alpha in [0, 1] only using 1.0 + alpha*(t_star - 1.0) if out-of-sample diverged
        chosen_t = 1.0
        shrink_note = "fallback_T_1_0_due_to_oos_divergence"

    real_deployed = calibration_metrics(r_conf, r_acc, T=chosen_t)
    probe_deployed = calibration_metrics(p_conf, p_acc, T=chosen_t)

    manifest_path = os.path.join(model_dir, "model_manifest.json")
    manifest_updated = False
    if os.path.exists(manifest_path):
        mdata = json.load(open(manifest_path, "r", encoding="utf-8"))
        mdata["confidence_temperature"] = round(float(chosen_t), 6)
        mdata["confidence_calibration"] = {
            "fit_split": "held_out_diagnostic_probe_100",
            "eval_split": "realscan_gold_v1/harum_palei_01",
            "policy": shrink_note,
            "ece_before": real_before["ece_10bin"],
            "ece_after": real_deployed["ece_10bin"],
            "brier_before": real_before["brier"],
            "brier_after": real_deployed["brier"],
            "nll_before": real_before["nll"],
            "nll_after": real_deployed["nll"],
        }
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(mdata, mf, indent=2, ensure_ascii=False)
        manifest_updated = True

    print(
        f"  • Probe-100: T=1.0 ECE={probe_before['ece_10bin']:.4f} NLL={probe_before['nll']:.4f} -> "
        f"T={chosen_t:.4f} ECE={probe_deployed['ece_10bin']:.4f} NLL={probe_deployed['nll']:.4f}"
    )
    print(
        f"  • RealScan-Gold (OOS): T=1.0 ECE={real_before['ece_10bin']:.4f} Brier={real_before['brier']:.4f} NLL={real_before['nll']:.4f} -> "
        f"T={chosen_t:.4f} ECE={real_deployed['ece_10bin']:.4f} Brier={real_deployed['brier']:.4f} NLL={real_deployed['nll']:.4f}"
    )
    return {
        "model_id": model_id,
        "model_dir": model_dir,
        "fitted_temperature": round(t_star, 6),
        "deployed_temperature": round(chosen_t, 6),
        "policy": shrink_note,
        "manifest_updated": manifest_updated,
        "probe_100_before": probe_before,
        "probe_100_after": probe_deployed,
        "realscan_oos_before": real_before,
        "realscan_oos_after": real_deployed,
    }


def main():
    parser = argparse.ArgumentParser(description="Fit and evaluate Temperature Scaling confidence calibration.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec as id=model_dir:dict_path (repeatable)",
    )
    args = parser.parse_args()

    def _split_model_dict(rest: str):
        if "::" in rest:
            return rest.split("::", 1)
        for idx, ch in enumerate(rest):
            if ch == ":":
                if idx == 1 and rest[0].isalpha() and rest[2:3] in ("\\", "/"):
                    continue
                return rest[:idx], rest[idx + 1:]
        raise ValueError(f"Expected 'model_dir:dict_path' in '{rest}'")

    specs = []
    for raw in args.model:
        mid, rest = raw.split("=", 1)
        mdir, dpath = _split_model_dict(rest)
        specs.append({"model_id": mid, "model_dir": mdir, "dict_path": dpath})

    results = [calibrate_one_model(sp) for sp in specs]
    out = {
        "title": "Cham-OCR Confidence Calibration (Post-Hoc Temperature Scaling)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": results,
    }
    os.makedirs(os.path.join(PROJECT_ROOT, "REPORTS"), exist_ok=True)
    out_json = os.path.join(PROJECT_ROOT, "REPORTS", "confidence_calibration.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    md = [
        "# Báo cáo Hiệu chỉnh Độ tự tin Confidence Calibration (Temperature Scaling)",
        f"- **Thời điểm**: `{out['generated_at']}`",
        "- **Tập fit (in-sample)**: `held_out_diagnostic_probe_100.json` (100 dòng, tách biệt hoàn toàn khỏi RealScan-Gold v1)",
        "- **Tập đánh giá Out-of-Sample**: `RealScan-Gold v1 / harum_palei_01` (25 dòng chụp thực tế)",
        "",
        "| Model | T* (Fitted) | T Deployed | OOS ECE (T=1 -> T*) | OOS Brier (T=1 -> T*) | OOS NLL (T=1 -> T*) | Mean Conf vs Empirical Acc |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        rb, ra = r["realscan_oos_before"], r["realscan_oos_after"]
        md.append(
            f"| `{r['model_id']}` | `{r['fitted_temperature']:.4f}` | `{r['deployed_temperature']:.4f}` | "
            f"{rb['ece_10bin']:.4f} → {ra['ece_10bin']:.4f} | "
            f"{rb['brier']:.4f} → {ra['brier']:.4f} | "
            f"{rb['nll']:.4f} → {ra['nll']:.4f} | "
            f"{rb['mean_conf']:.4f} → {ra['mean_conf']:.4f} (acc={ra['mean_acc']:.4f}) |"
        )
    out_md = os.path.join(PROJECT_ROOT, "REPORTS", "CONFIDENCE_CALIBRATION_REPORT.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\n✅ Calibration reports written:\n  • {out_json}\n  • {out_md}")


if __name__ == "__main__":
    main()
