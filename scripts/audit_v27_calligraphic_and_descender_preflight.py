#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4 (`Gate SC-4`): V27 Calligraphic / Tall-Descender Single-Factor Ablation & Adaptation Leak-Guard Audit
=============================================================================================================
1. Experiment 4A (Real Archival Tall-Descender Vertical Band Ablation):
   Tests 4 vertical crop ratios (`full_100`, `core_80`, `core_68`, `core_56`) on tall-descender lithographed
   (`aymonier1889_chronique_p108..p114`) and handwritten (`eap698_1_1_c15`, `eap698_4_1_c08`) lines from
   `RealScan-Gold v4` to separate vertical aspect-ratio compression from unseen scribal ductus.
2. Experiment 4B (Controlled Ground-Truth Calligraphic Perturbation Curve):
   Applies 4 isolated single-factor calligraphic perturbations (slant shear, 2.1x tall-descender vertical box,
   brush stroke thinning, and combined scribal transform) to 25 Ground-Truth lines from `clean_frozen_200_lines.json`
   and measures exact Strict CER on `v24_1_r4`.
3. Experiment 4C (Adversarial V27 Holdout & Pseudo-Label Leak-Guard Audit):
   Verifies that `verify_v27_training_disjoint_from_holdout` blocks all 4 leakage vectors (direct holdout `doc_id`,
   shared `EAP<project>-<series>` archive cousin, disguised `page_sha256`/`crop_sha256`, and unverified pseudo-labels)
   while accepting valid human-verified `adaptation_seed_pool` records.
"""

from datetime import datetime, timezone
import json
import os
import sys
import cv2
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-studio"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import app as studio_app
import ingest_studio_to_realscan_gold as ingest_v4


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1))
        prev = cur
    return prev[-1]


def crop_vertical_core_band(crop_bgr: np.ndarray, keep_frac: float) -> np.ndarray:
    """Crops symmetrically (biased slightly downward for sub-base vowels) around the densest horizontal ink core."""
    if keep_frac >= 0.999 or crop_bgr is None or crop_bgr.shape[0] < 24:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    bg = float(np.percentile(gray, 85))
    ink_row = np.maximum(0.0, bg - gray.astype(np.float32)).mean(axis=1)
    win = max(8, int(round(h * keep_frac)))
    if win >= h:
        return crop_bgr
    csum = np.cumsum(np.insert(ink_row, 0, 0.0))
    window_sums = csum[win:] - csum[:-win]
    best_y0 = int(np.argmax(window_sums))
    return crop_bgr[best_y0:best_y0 + win, :].copy()


def run_experiment_4a_archival_descenders(ocr_r4, v4_manifest: dict) -> dict:
    target_ids = [
        "aymonier1889_chronique_p108",
        "aymonier1889_chronique_p110",
        "aymonier1889_chronique_p112",
        "eap698_1_1_c15",
        "eap698_4_1_c08",
    ]
    docs_by_id = {d["doc_id"]: d for d in v4_manifest["documents"]}
    tall_crops = []
    for did in target_ids:
        d = docs_by_id.get(did)
        if not d:
            continue
        for ln in d.get("lines", []):
            cp = os.path.join(PROJECT_ROOT, ln["crop_relpath"].replace("/", os.sep))
            if not os.path.exists(cp):
                continue
            im = cv2.imread(cp)
            if im is not None and im.shape[0] >= 52 and im.shape[1] >= 180:
                tall_crops.append((did, ln["line"], im))

    band_specs = [("full_100", 1.00), ("core_80", 0.80), ("core_68", 0.68), ("core_56", 0.56)]
    results = {}
    for b_name, frac in band_specs:
        batch = [crop_vertical_core_band(im, frac) for _, _, im in tall_crops]
        preds, _ = ocr_r4(batch) if batch else ([], 0.0)
        confs = [float(c) for _, c in preds]
        cham_lens = [sum(1 for ch in (t or "") if "\uAA00" <= ch <= "\uAA5F") for t, _ in preds]
        sub_vowels = [sum(1 for ch in (t or "") if ch in "\uAA2D\uAA32\uAA33\uAA34\uAA35\uAA36") for t, _ in preds]
        results[b_name] = {
            "keep_vertical_fraction": frac,
            "num_tall_lines_tested": len(batch),
            "mean_crop_height_px": round(float(np.mean([b.shape[0] for b in batch])) if batch else 0.0, 1),
            "mean_confidence": round(float(np.mean(confs)) if confs else 0.0, 4),
            "high_conf_ge_075_count": sum(1 for c in confs if c >= 0.75),
            "mean_cham_chars_per_line": round(float(np.mean(cham_lens)) if cham_lens else 0.0, 2),
            "total_subjoined_medials_vowels": int(sum(sub_vowels)),
            "sample_predictions": [
                {"doc_id": tall_crops[k][0], "line": tall_crops[k][1], "h": batch[k].shape[0], "conf": round(confs[k], 4), "pred": preds[k][0][:65]}
                for k in range(min(5, len(batch)))
            ],
        }
    return results


def run_experiment_4b_calligraphic_gt_curve(ocr_r4) -> dict:
    fsuite = os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "frozen_suite")
    cf_path = os.path.join(fsuite, "clean_frozen_200_lines.json")
    rows = json.load(open(cf_path, "r", encoding="utf-8"))[:25]
    samples = []
    for r in rows:
        cp = os.path.join(fsuite, r["crop_file"])
        if os.path.exists(cp):
            im = cv2.imread(cp)
            if im is not None:
                samples.append((r["sample_id"], r["gt"].strip(), im))

    def _apply_slant(im, shear=0.22):
        h, w = im.shape[:2]
        shift = int(abs(shear) * h)
        M = np.float32([[1.0, -shear, shift], [0.0, 1.0, 0.0]])
        return cv2.warpAffine(im, M, (w + shift, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    def _apply_tall_descender_box(im, factor=2.1):
        h, w = im.shape[:2]
        nh = int(round(h * factor))
        pad_top = max(2, int(round((nh - h) * 0.25)))
        pad_bot = max(2, nh - h - pad_top)
        return cv2.copyMakeBorder(im, pad_top, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    def _apply_thin_brush_fade(im):
        # Dilate white paper by 2x2 to thin dark strokes + add mild uneven ink density
        thinned = cv2.dilate(im, np.ones((2, 2), np.uint8), iterations=1)
        return cv2.addWeighted(thinned, 0.85, np.full_like(thinned, 245), 0.15, 0)

    conditions = {
        "0_clean_printed_baseline": lambda im: im,
        "1_calligraphic_slant_12deg": _apply_slant,
        "2_tall_descender_2_1x_box": _apply_tall_descender_box,
        "3_thin_brush_stroke_fade": _apply_thin_brush_fade,
        "4_combined_scribal_perturbation": lambda im: _apply_thin_brush_fade(_apply_tall_descender_box(_apply_slant(im, 0.22), 1.85)),
    }

    curve = {}
    for cname, fn in conditions.items():
        imgs = [fn(im) for _, _, im in samples]
        preds, _ = ocr_r4(imgs)
        tot_ed, tot_ch, exact = 0, 0, 0
        confs = []
        for (_, gt, _), (pr, cf) in zip(samples, preds):
            pr_s = (pr or "").strip()
            ed = edit_distance(gt, pr_s)
            tot_ed += ed
            tot_ch += len(gt)
            exact += int(ed == 0)
            confs.append(float(cf))
        curve[cname] = {
            "n_lines": len(samples),
            "strict_cer_pct": round(100.0 * tot_ed / max(1, tot_ch), 2),
            "exact_line_acc_pct": round(100.0 * exact / max(1, len(samples)), 2),
            "mean_confidence": round(float(np.mean(confs)), 4),
        }
    return curve


def run_experiment_4c_leak_guard_audit(v4_manifest_path: str, v4_manifest: dict) -> dict:
    holdout_doc = next(d for d in v4_manifest["documents"] if d.get("benchmark_split") == "locked_test_holdout" and d.get("lines"))
    seed_doc = next(d for d in v4_manifest["documents"] if d.get("benchmark_split") == "adaptation_seed_pool" and d.get("lines"))

    blocked_vectors = {}
    attack_cases = {
        "attack_1_direct_holdout_doc_id": [{
            "doc_id": holdout_doc["doc_id"],
            "source_collection_id": holdout_doc["source_collection_id"],
            "benchmark_split": "adaptation_seed_pool",
            "training_forbidden": False,
            "verified_by_human": True,
            "page_sha256": "0" * 64,
            "crop_sha256": "1" * 64,
            "gt": "ꨀꨕꩅ ꨌꩌ",
        }],
        "attack_2_shared_eap_series_cousin": [{
            "doc_id": "eap531_1_99_fake_cousin",
            "source_collection_id": "EAP531-1-99",
            "benchmark_split": "adaptation_seed_pool",
            "training_forbidden": False,
            "verified_by_human": True,
            "page_sha256": "2" * 64,
            "crop_sha256": "3" * 64,
            "gt": "ꨀꨕꩅ ꨌꩌ",
        }],
        "attack_3_disguised_holdout_crop_sha256": [{
            "doc_id": seed_doc["doc_id"],
            "source_collection_id": seed_doc["source_collection_id"],
            "benchmark_split": "adaptation_seed_pool",
            "training_forbidden": False,
            "verified_by_human": True,
            "page_sha256": seed_doc["page_sha256"],
            "crop_sha256": holdout_doc["lines"][0]["crop_sha256"],
            "gt": "ꨀꨕꩅ ꨌꩌ",
        }],
        "attack_4_unverified_pseudo_label": [{
            "doc_id": seed_doc["doc_id"],
            "source_collection_id": seed_doc["source_collection_id"],
            "benchmark_split": "adaptation_seed_pool",
            "training_forbidden": False,
            "verified_by_human": False,
            "page_sha256": seed_doc["page_sha256"],
            "crop_sha256": seed_doc["lines"][0]["crop_sha256"],
            "gt": "ꨀꨕꩅ ꨌꩌ",
        }],
    }

    for atk_name, recs in attack_cases.items():
        try:
            ingest_v4.verify_v27_training_disjoint_from_holdout(recs, manifest_path=v4_manifest_path)
            raise AssertionError(f"SECURITY FAILURE: {atk_name} was NOT blocked!")
        except PermissionError as exc:
            blocked_vectors[atk_name] = {"blocked": True, "reason": str(exc)}

    valid_seed_probe = [{
        "doc_id": seed_doc["doc_id"],
        "source_collection_id": seed_doc["source_collection_id"],
        "benchmark_split": "adaptation_seed_pool",
        "training_forbidden": False,
        "verified_by_human": True,
        "page_sha256": seed_doc["page_sha256"],
        "crop_sha256": seed_doc["lines"][0]["crop_sha256"],
        "gt": "ꨀꨕꩅ ꨌꩌ",
    }]
    pass_res = ingest_v4.verify_v27_training_disjoint_from_holdout(valid_seed_probe, manifest_path=v4_manifest_path)
    return {"adversarial_attacks_blocked": blocked_vectors, "valid_human_verified_seed_probe": pass_res}


def main():
    v4_manifest_path = os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "realscan_gold_v4", "manifest.json")
    v4_manifest = json.load(open(v4_manifest_path, "r", encoding="utf-8"))
    ocr_r4 = studio_app.get_ocr_model("v24_1_r4")

    print("🔬 [Stage 4 / Gate SC-4] Running Experiment 4A (Archival Tall-Descender Vertical Band Ablation)...")
    exp4a = run_experiment_4a_archival_descenders(ocr_r4, v4_manifest)
    print("🔬 [Stage 4 / Gate SC-4] Running Experiment 4B (Controlled GT Calligraphic Perturbation Curve)...")
    exp4b = run_experiment_4b_calligraphic_gt_curve(ocr_r4)
    print("🔬 [Stage 4 / Gate SC-4] Running Experiment 4C (Adversarial V27 Leak-Guard Verification)...")
    exp4c = run_experiment_4c_leak_guard_audit(v4_manifest_path, v4_manifest)

    out_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_tested": "v24_1_r4",
        "experiment_4a_archival_tall_descender_bands": exp4a,
        "experiment_4b_controlled_gt_calligraphic_curve": exp4b,
        "experiment_4c_v27_leak_guard_audit": exp4c,
    }
    out_p = os.path.join(PROJECT_ROOT, "REPORTS", "v27_preflight_diagnostic_summary.json")
    with open(out_p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out_summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Stage 4 Pre-Flight Diagnostic Summary saved -> {out_p}")
    print(json.dumps({"exp4b_gt_curve": exp4b, "exp4a_conf_by_band": {k: v["mean_confidence"] for k, v in exp4a.items()}}, indent=2))


if __name__ == "__main__":
    main()
