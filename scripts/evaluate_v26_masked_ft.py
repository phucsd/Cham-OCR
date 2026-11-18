#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR V26 Masked-SVTR Fine-Tune: causal evaluation (masked vs identical unmasked control, plus
optional WiSE-FT blend with R4.1 WiSE-FT) and gated auto-promotion into
`ocr-studio/data/output/rec_cham_inference_v24_1_r4/`.

For each branch (masked / control), every epoch is exported and scored on Frozen-838 + Stress-82
(`run_v24_frozen_suite.py`), Held-Out Probe-100 (`W=480`), and `harum_palei_01` crops; release gates are
computed dynamically by `FrozenGatePolicy` (`valid_for_release` is never hard-coded). Production `R4.1 WiSE`
is also re-scored under the active `ocr-studio/app.py` detector so `FullPage-Frozen` & `RealScan-Gold v2`
comparisons remain strictly apples-to-apples. If no single `v26_masked` epoch hits 6/6 gates, a WiSE-FT
sweep (`alpha * R4.1_WiSE + (1 - alpha) * v26_masked_best` for `alpha in (0.15, 0.25, 0.35)`) is run.

Auto-promotion (user-approved policy, 2026-11-18): a candidate is promoted only if
  * valid_for_release == True (6/6 gates), and
  * no regression vs production: Clean <= +0.20 pt, Character <= +0.20 pt,
    harum crop CER <= +0.25 pt, OOD-180 crop CER <= +0.30 pt, OOD-12 E2E Doc CER <= +0.30 pt,
    Wikimedia crop CER <= +2.0 pt (7 lines only), Blank guard PASS.
Preference order among qualifying candidates: `masked` (epoch or WiSE-FT) > `control`. The previous
production directory is backed up to `rec_cham_inference_v24_1_r4_r41wise_backup` before packaging.
"""

import os
import sys
import json
import shutil
import argparse
import hashlib
import subprocess
from datetime import datetime, timezone

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIO_DIR = os.path.join(PROJECT_ROOT, "ocr-studio")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, STUDIO_DIR)
sys.path.insert(0, os.path.join(STUDIO_DIR, "PaddleOCR"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts"))

from evaluate_r2_1_6_causal import file_sha256, evaluate_probe_100  # noqa: E402
from evaluate_r2_1_7_causal import parse_train_log  # noqa: E402
from diagnostic_r2_1_4b_ctc_counterfactual import DICT_PATH  # noqa: E402
from frozen_gate_policy import FrozenGatePolicy  # noqa: E402

PY26 = os.path.join(PROJECT_ROOT, ".venv_paddle26", "Scripts", "python.exe")
PY313 = r"C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe"
if not os.path.exists(PY313):
    PY313 = sys.executable
PADDLE_DIR = os.path.join(STUDIO_DIR, "PaddleOCR")
SUITE = os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts", "run_v24_frozen_suite.py")
STUDIO_R4 = os.path.join(STUDIO_DIR, "data", "output", "rec_cham_inference_v24_1_r4")
R41_WISE_CKPT = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r4_1_wise83_ckpt", "best.pdparams")
EPOCHS = 6

BRANCHES = {
    "masked": {"b_key": "v26_masked_ft", "slug": "paddleocr-cham-v26-masked-svtr-ft", "cfg": "rec_cham_v26_masked_ft.yml", "mask": True},
    "control": {"b_key": "v26_control_ft", "slug": "paddleocr-cham-v26-control-unmasked-ft", "cfg": "rec_cham_v26_control_ft.yml", "mask": False},
}


def active_detector_signature() -> str:
    import app as studio_app
    det_params = os.path.join(studio_app.CHAM_DET_DIR, "inference.pdiparams")
    det_sha16 = file_sha256(det_params)[:16] if os.path.exists(det_params) else "none"
    raw = (
        f"{studio_app.DET_MODE}|{studio_app.CHAM_DET_BOX_TYPE}|{studio_app.CHAM_DET_UNCLIP}|"
        f"{studio_app.CHAM_DET_BOX_THRESH}|{studio_app.CHAM_DET_DB_THRESH}|"
        f"{studio_app.CHAM_DET_POLY_VPAD}|{studio_app.CHAM_DET_POLY_MASK}|{det_sha16}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _score_exported_model(tag: str, infer_dir: str, ckpt_file: str, use_mask: bool, policy: FrozenGatePolicy, det_sig: str) -> dict:
    from evaluate_r4_2_inktight import eval_harum_crops

    infer_sha = file_sha256(os.path.join(infer_dir, "inference.pdiparams"))
    out_dir = os.path.join(PROJECT_ROOT, "ocr-benchmark", "results", f"{tag}_raw")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "benchmark_summary.json")
    sig_file = os.path.join(out_dir, "detector_sig.txt")
    reuse = False
    if os.path.exists(out_json) and os.path.exists(sig_file):
        try:
            c = json.load(open(out_json, encoding="utf-8"))
            cached_sig = open(sig_file, encoding="utf-8").read().strip()
            reuse = (c.get("model_sha256") == infer_sha and cached_sig == det_sig and "fullpage_suite" in c)
        except Exception:
            reuse = False
    if not reuse:
        subprocess.run([PY313, SUITE, "--model-dir", infer_dir, "--dict-path", DICT_PATH,
                        "--rec-image-shape", "3, 48, 480", "--output", out_json], cwd=PROJECT_ROOT, check=True)
        with open(sig_file, "w", encoding="utf-8", newline="\n") as sf:
            sf.write(det_sig + "\n")

    s = json.load(open(out_json, encoding="utf-8"))
    probe_cache = os.path.join(out_dir, "probe_w480.json")
    if os.path.exists(probe_cache):
        probe = json.load(open(probe_cache, encoding="utf-8"))
    else:
        probe = evaluate_probe_100(ckpt_file, target_w=480, use_valid_mask=use_mask)
        json.dump(probe, open(probe_cache, "w", encoding="utf-8"), indent=2)
    harum = eval_harum_crops(infer_dir)
    failing = [k for k, v in s["symbol_suite"]["per_symbol"].items() if v["accuracy"] < policy.symbol_min_class_acc]
    ge = policy.evaluate_gates(
        clean_cer=s["clean_suite"]["norm_cer"], char_cer=s["character_suite"]["cer"],
        symbol_acc=s["symbol_suite"]["overall_accuracy"], failing_symbols=failing,
        hyphen_acc=s["hyphen_suite"]["overall_accuracy"], blank_passed=bool(s["blank_suite"]["passed"]),
        blank_white=str(s["blank_suite"].get("white_pred", "")), blank_gray=str(s["blank_suite"].get("gray_pred", "")),
        fullpage_cer=s["fullpage_suite"]["matched_line_cer"])
    gp = sum(1 for g in ge["gates"] if g["status"] == "PASS")
    sym_ok = sum(v["correct"] for v in s["symbol_suite"]["per_symbol"].values())
    return {
        "sha256": file_sha256(ckpt_file) if os.path.exists(ckpt_file) else None,
        "infer_dir": infer_dir,
        "infer_sha256": infer_sha,
        "clean_cer_pct": round(s["clean_suite"]["norm_cer"] * 100, 2),
        "char_cer_pct": round(s["character_suite"]["cer"] * 100, 2),
        "symbols_acc_pct": round(s["symbol_suite"]["overall_accuracy"] * 100, 2),
        "symbols_passed": f"{sym_ok}/280",
        "failing_symbols": failing,
        "hyphen_acc_pct": round(s["hyphen_suite"]["overall_accuracy"] * 100, 2),
        "blank_passed": bool(s["blank_suite"]["passed"]),
        "fullpage_cer_pct": round(s["fullpage_suite"]["matched_line_cer"] * 100, 2),
        "fullpage_recall_pct": round(s["fullpage_suite"]["overall_recall"] * 100, 2),
        "fullpage_l5_cer_pct": round(s["fullpage_suite"]["l5_matched_cer"] * 100, 2),
        "stress_82_cer_pct": round(s["stress_suite"]["overall_stress_cer"] * 100, 2),
        "probe_100_w480_cer_pct": probe["cer_pct"],
        "harum_crops_cer_pct": harum["crop_cer_pct"],
        "harum_exact_lines": harum["exact_lines"],
        "gates_passed": gp,
        "valid_for_release": ge["valid_for_release"],
        "gates": ge["gates"],
        "composite_score": round(float(s["composite_score"]), 4),
    }


def _export_with_py26(cfg_path: str, ckpt_prefix: str, infer_dir: str):
    if os.path.exists(infer_dir):
        shutil.rmtree(infer_dir)
    os.makedirs(infer_dir, exist_ok=True)
    subprocess.run([PY26, "tools/export_model.py", "-c", cfg_path, "-o", f"Global.checkpoints={ckpt_prefix}",
                    f"Global.save_inference_dir={infer_dir}", f"Global.character_dict_path={DICT_PATH}",
                    "Global.export_with_pir=False"], cwd=PADDLE_DIR, check=True)


def eval_branch(name: str, spec: dict, policy: FrozenGatePolicy, det_sig: str) -> dict:
    dl_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"r2_1_4c_{spec['b_key']}_kaggle_output")
    ext_dir = os.path.join(dl_dir, "extracted_checkpoints")
    ckpt_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"v26_{name}_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    for fn in os.listdir(ext_dir):
        fp = os.path.join(ext_dir, fn)
        if os.path.isfile(fp):
            shutil.copy2(fp, os.path.join(ckpt_dir, fn))
    accounting = parse_train_log(os.path.join(ckpt_dir, "train.log"), total_epochs=EPOCHS,
                                 kernel_log_path=os.path.join(dl_dir, f"{spec['slug']}.log"))
    cfg_path = os.path.join(PROJECT_ROOT, "ocr-training", "configs", spec["cfg"])

    results, best_ep, best_rank = {}, None, None
    for ep in range(1, EPOCHS + 1):
        prefix = os.path.join(ckpt_dir, f"iter_epoch_{ep}")
        ckpt_file = prefix + ".pdparams"
        if not os.path.exists(ckpt_file):
            print(f"  ! {name}: missing epoch {ep}, skipping")
            continue
        infer_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"v26_{name}_infer_e{ep}")
        if not os.path.exists(os.path.join(infer_dir, "inference.pdiparams")):
            _export_with_py26(cfg_path, prefix, infer_dir)
        r = _score_exported_model(f"v26_{name}_e{ep}", infer_dir, ckpt_file, spec["mask"], policy, det_sig)
        results[f"epoch_{ep}"] = r
        rank = (1 if r["valid_for_release"] else 0, r["gates_passed"], -r["char_cer_pct"] - r["clean_cer_pct"], -r["harum_crops_cer_pct"])
        if best_rank is None or rank > best_rank:
            best_rank, best_ep = rank, ep
        print(f"  -> {name} E{ep}: Clean={r['clean_cer_pct']}% Char={r['char_cer_pct']}% Sym={r['symbols_acc_pct']}% "
              f"(fail={r['failing_symbols']}) Hyp={r['hyphen_acc_pct']}% FP={r['fullpage_cer_pct']}% Stress={r['stress_82_cer_pct']}% "
              f"Probe={r['probe_100_w480_cer_pct']}% Harum={r['harum_crops_cer_pct']}% Gates={r['gates_passed']}/6 valid={r['valid_for_release']}")
    return {"accounting": accounting, "epochs": results, "best_epoch": best_ep, "ckpt_dir": ckpt_dir, "cfg_path": cfg_path}


def eval_wise_blend(base_ckpt: str, masked_ckpt: str, cfg_path: str, alphas, policy: FrozenGatePolicy, det_sig: str) -> dict:
    import paddle
    sd_base = paddle.load(base_ckpt)
    sd_mask = paddle.load(masked_ckpt)
    out = {}
    best_alpha, best_rank = None, None
    for alpha in alphas:
        tag_a = f"{int(round(alpha * 100)):02d}"
        w_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"v26_wise_a{tag_a}_ckpt")
        os.makedirs(w_dir, exist_ok=True)
        w_prefix = os.path.join(w_dir, "best")
        w_file = w_prefix + ".pdparams"
        blended = {k: float(alpha) * sd_base[k] + (1.0 - float(alpha)) * sd_mask[k] for k in sd_mask}
        paddle.save(blended, w_file)
        infer_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"v26_wise_a{tag_a}_infer")
        _export_with_py26(cfg_path, w_prefix, infer_dir)
        r = _score_exported_model(f"v26_wise_a{tag_a}", infer_dir, w_file, True, policy, det_sig)
        r["alpha_r41_wise"] = float(alpha)
        r["ckpt_path"] = w_file
        out[f"alpha_{tag_a}"] = r
        rank = (1 if r["valid_for_release"] else 0, r["gates_passed"], -r["char_cer_pct"] - r["clean_cer_pct"], -r["harum_crops_cer_pct"])
        if best_rank is None or rank > best_rank:
            best_rank, best_alpha = rank, f"alpha_{tag_a}"
        print(f"  -> WiSE alpha={alpha:.2f}: Clean={r['clean_cer_pct']}% Char={r['char_cer_pct']}% Sym={r['symbols_acc_pct']}% "
              f"(fail={r['failing_symbols']}) Hyp={r['hyphen_acc_pct']}% FP={r['fullpage_cer_pct']}% "
              f"Harum={r['harum_crops_cer_pct']}% Gates={r['gates_passed']}/6 valid={r['valid_for_release']}")
    return {"candidates": out, "best_key": best_alpha}


def eval_masked_soups(ckpt_dir: str, cfg_path: str, policy: FrozenGatePolicy, det_sig: str) -> dict:
    """In-basin checkpoint soups across V26 Masked epochs (all share use_valid_mask=True)."""
    import paddle

    recipes = {
        "soup_50e3_50e4": {3: 0.50, 4: 0.50},
        "soup_40e3_40e4_20e6": {3: 0.40, 4: 0.40, 6: 0.20},
        "soup_60e3_40e4": {3: 0.60, 4: 0.40},
    }
    loaded = {ep: paddle.load(os.path.join(ckpt_dir, f"iter_epoch_{ep}.pdparams")) for ep in (3, 4, 6)}
    out, best_key, best_rank = {}, None, None
    keys = list(loaded[3].keys())
    for sname, weights in recipes.items():
        s_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"v26_{sname}_ckpt")
        os.makedirs(s_dir, exist_ok=True)
        s_prefix = os.path.join(s_dir, "best")
        s_file = s_prefix + ".pdparams"
        blended = {}
        for k in keys:
            acc = None
            for ep, w in weights.items():
                term = float(w) * loaded[ep][k]
                acc = term if acc is None else (acc + term)
            blended[k] = acc
        paddle.save(blended, s_file)
        infer_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"v26_{sname}_infer")
        if not os.path.exists(os.path.join(infer_dir, "inference.pdiparams")):
            _export_with_py26(cfg_path, s_prefix, infer_dir)
        r = _score_exported_model(f"v26_{sname}", infer_dir, s_file, True, policy, det_sig)
        r["weights"] = {f"epoch_{ep}": w for ep, w in weights.items()}
        r["ckpt_path"] = s_file
        out[sname] = r
        rank = (1 if r["valid_for_release"] else 0, r["gates_passed"], -r["char_cer_pct"] - r["clean_cer_pct"], -r["harum_crops_cer_pct"])
        if best_rank is None or rank > best_rank:
            best_rank, best_key = rank, sname
        print(f"  -> Masked {sname}: Clean={r['clean_cer_pct']}% Char={r['char_cer_pct']}% Sym={r['symbols_acc_pct']}% "
              f"(fail={r['failing_symbols']}) Hyp={r['hyphen_acc_pct']}% FP={r['fullpage_cer_pct']}% Stress={r['stress_82_cer_pct']}% "
              f"Probe={r['probe_100_w480_cer_pct']}% Harum={r['harum_crops_cer_pct']}% Gates={r['gates_passed']}/6 valid={r['valid_for_release']}")
    return {"candidates": out, "best_key": best_key}


def realscan_v2(models: dict) -> dict:
    out_j = os.path.join(PROJECT_ROOT, "REPORTS", "v26_realscan_gold_v2_comparison.json")
    out_m = os.path.join(PROJECT_ROOT, "REPORTS", "v26_realscan_gold_v2_comparison.md")
    specs = [f"{alias}={d}:{DICT_PATH}" for alias, d in models.items()]
    subprocess.run([PY313, os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts", "evaluate_realscan_gold_v2.py"),
                    "--models", *specs, "--output-json", out_j, "--output-md", out_m], cwd=PROJECT_ROOT, check=True)
    return json.load(open(out_j, encoding="utf-8"))["models"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-promote", action="store_true")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--wise-alphas", nargs="*", type=float, default=[0.15, 0.25, 0.35])
    args = ap.parse_args()

    policy = FrozenGatePolicy()
    det_sig = active_detector_signature()
    prod_sha16 = file_sha256(os.path.join(STUDIO_R4, "inference.pdiparams"))[:16]

    # Snapshot current production R4.1 WiSE-FT model and score it on the EXACT same active detector
    prod_snapshot = os.path.join(PROJECT_ROOT, "ocr-training", "output", "v26_prod_r41wise_snapshot")
    if os.path.exists(prod_snapshot):
        shutil.rmtree(prod_snapshot)
    shutil.copytree(STUDIO_R4, prod_snapshot)
    print("=" * 80 + f"\n🔬 Scoring Production R4.1 WiSE-FT baseline (det_sig={det_sig})\n" + "=" * 80)
    prod_metrics = _score_exported_model("v26_prod_r41wise", prod_snapshot, R41_WISE_CKPT, False, policy, det_sig)
    if args.baseline_only:
        print("✅ Baseline cached:", json.dumps({k: prod_metrics[k] for k in ("clean_cer_pct", "char_cer_pct", "symbols_acc_pct", "hyphen_acc_pct", "fullpage_cer_pct", "fullpage_recall_pct", "fullpage_l5_cer_pct", "stress_82_cer_pct", "probe_100_w480_cer_pct", "harum_crops_cer_pct", "gates_passed", "valid_for_release")}, ensure_ascii=False))
        return

    branches = {}
    for name, spec in BRANCHES.items():
        print("=" * 80 + f"\n🔬 V26 branch `{name}` (det_sig={det_sig})\n" + "=" * 80)
        branches[name] = eval_branch(name, spec, policy, det_sig)

    wise_res, soup_res = None, None
    m_branch = branches.get("masked") or {}
    m_best_ep = m_branch.get("best_epoch")
    if m_best_ep is not None and not m_branch["epochs"][f"epoch_{m_best_ep}"]["valid_for_release"]:
        if args.wise_alphas:
            print("=" * 80 + "\n🧪 Running V26 Masked WiSE-FT sweep (no single masked epoch hit 6/6 gates)\n" + "=" * 80)
            masked_best_ckpt = os.path.join(m_branch["ckpt_dir"], f"iter_epoch_{m_best_ep}.pdparams")
            wise_res = eval_wise_blend(R41_WISE_CKPT, masked_best_ckpt, m_branch["cfg_path"], args.wise_alphas, policy, det_sig)
        print("=" * 80 + "\n🥣 Running In-Basin V26 Masked checkpoint soup sweep (E3/E4/E6)\n" + "=" * 80)
        soup_res = eval_masked_soups(m_branch["ckpt_dir"], m_branch["cfg_path"], policy, det_sig)

    # RealScan-Gold v2 (crops + OOD-12 end-to-end)
    rs_models = {"prod_r41wise": prod_snapshot}
    for name, b in branches.items():
        if b["best_epoch"] is not None:
            rs_models[f"v26_{name}"] = b["epochs"][f"epoch_{b['best_epoch']}"]["infer_dir"]
    if soup_res and soup_res.get("best_key"):
        sb_k = soup_res["best_key"]
        rs_models["v26_masked_soup"] = soup_res["candidates"][sb_k]["infer_dir"]
    if wise_res and wise_res.get("best_key"):
        wb_k = wise_res["best_key"]
        rs_models["v26_masked_wise"] = wise_res["candidates"][wb_k]["infer_dir"]
    rs = realscan_v2(rs_models)

    def rs_view(alias):
        m = rs[alias]
        return {"harum_crop_cer_pct": m["harum_palei_01_crops"]["norm_cer_pct"],
                "wikimedia_crop_cer_pct": m["wikimedia_specimens"]["norm_cer_pct"],
                "ood180_crop_cer_pct": m["ood_archival_overall"]["norm_cer_pct"],
                "ood12_e2e_doc_cer_pct": m["ood_archival_overall"]["e2e_doc_cer_pct"],
                "ood12_e2e_lines": m["ood_archival_overall"]["e2e_lines"]}

    def _check_candidate(r_metrics, v_rs):
        checks = {
            "valid_for_release_6_of_6": bool(r_metrics["valid_for_release"]),
            "clean_non_regression": r_metrics["clean_cer_pct"] <= float(prod_metrics["clean_cer_pct"]) + 0.20,
            "char_non_regression": r_metrics["char_cer_pct"] <= float(prod_metrics["char_cer_pct"]) + 0.20,
            "blank_guard": bool(r_metrics["blank_passed"]),
            "harum_crop_non_regression": v_rs["harum_crop_cer_pct"] <= prod_rs["harum_crop_cer_pct"] + 0.25,
            "ood180_crop_non_regression": v_rs["ood180_crop_cer_pct"] <= prod_rs["ood180_crop_cer_pct"] + 0.30,
            "ood12_e2e_non_regression": v_rs["ood12_e2e_doc_cer_pct"] <= prod_rs["ood12_e2e_doc_cer_pct"] + 0.30,
            "wikimedia_non_regression": v_rs["wikimedia_crop_cer_pct"] <= prod_rs["wikimedia_crop_cer_pct"] + 2.0,
        }
        return checks, all(checks.values())

    prod_rs = rs_view("prod_r41wise")
    decisions = {}
    for name, b in branches.items():
        if b["best_epoch"] is None:
            continue
        r = b["epochs"][f"epoch_{b['best_epoch']}"]
        v = rs_view(f"v26_{name}")
        checks, ok = _check_candidate(r, v)
        decisions[name] = {"best_epoch": b["best_epoch"], "realscan_v2": v, "checks": checks, "qualifies": ok}
    if soup_res and soup_res.get("best_key") and "v26_masked_soup" in rs:
        sb_k = soup_res["best_key"]
        sr = soup_res["candidates"][sb_k]
        sv = rs_view("v26_masked_soup")
        schk, sok = _check_candidate(sr, sv)
        decisions["masked_soup"] = {"best_key": sb_k, "weights": sr["weights"], "realscan_v2": sv, "checks": schk, "qualifies": sok}
    if wise_res and wise_res.get("best_key") and "v26_masked_wise" in rs:
        wb_k = wise_res["best_key"]
        wr = wise_res["candidates"][wb_k]
        wv = rs_view("v26_masked_wise")
        wchk, wok = _check_candidate(wr, wv)
        decisions["masked_wise"] = {"best_key": wb_k, "alpha": wr["alpha_r41_wise"], "realscan_v2": wv, "checks": wchk, "qualifies": wok}

    winner = None
    for cand_key in ("masked", "masked_soup", "masked_wise", "control"):
        if decisions.get(cand_key, {}).get("qualifies"):
            winner = cand_key
            break

    promoted = False
    backup_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "rec_cham_inference_v24_1_r4_r41wise_backup")
    if winner and not args.no_promote:
        if winner == "masked_soup":
            r = soup_res["candidates"][soup_res["best_key"]]
            ckpt_p = r["ckpt_path"]
            cfg_p = branches["masked"]["cfg_path"]
            ep_val = 4
            stage_str = f"V26_MASKED_{soup_res['best_key'].upper()}_FT_W480"
            src_kernel = "gustavnguyen/" + BRANCHES["masked"]["slug"]
            use_mask_flag = True
        elif winner == "masked_wise":
            r = wise_res["candidates"][wise_res["best_key"]]
            ckpt_p = r["ckpt_path"]
            cfg_p = branches["masked"]["cfg_path"]
            ep_val = int(m_best_ep)
            stage_str = f"V26_MASKED_WISE_A{int(round(r['alpha_r41_wise']*100)):02d}_FT_W480"
            src_kernel = "gustavnguyen/" + BRANCHES["masked"]["slug"]
            use_mask_flag = True
        else:
            b = branches[winner]
            ep_val = int(b["best_epoch"])
            r = b["epochs"][f"epoch_{ep_val}"]
            ckpt_p = os.path.join(b["ckpt_dir"], f"iter_epoch_{ep_val}.pdparams")
            cfg_p = b["cfg_path"]
            stage_str = f"V26_{winner.upper()}_FT_W480"
            src_kernel = "gustavnguyen/" + BRANCHES[winner]["slug"]
            use_mask_flag = BRANCHES[winner]["mask"]

        if not os.path.exists(backup_dir):
            shutil.copytree(STUDIO_R4, backup_dir)
        metrics = {
            "clean_cer_pct": r["clean_cer_pct"], "char_cer_pct": r["char_cer_pct"],
            "symbols_acc_pct": r["symbols_acc_pct"], "symbols_passed": r["symbols_passed"],
            "failing_symbols": r["failing_symbols"], "hyphen_acc_pct": r["hyphen_acc_pct"],
            "fullpage_cer_pct": r["fullpage_cer_pct"], "stress_82_cer_pct": r["stress_82_cer_pct"],
            "probe_100_w480_cer_pct": r["probe_100_w480_cer_pct"], "harum_crops_cer_pct": r["harum_crops_cer_pct"],
            "gates_passed": f"{r['gates_passed']}/6", "valid_for_release": r["valid_for_release"],
            "use_valid_mask": use_mask_flag,
        }
        subprocess.run([PY313, os.path.join(PROJECT_ROOT, "scripts", "package_v24_1_model.py"),
                        "--model-id", "v24_1_r4",
                        "--display-name", f"Cham-OCR V26 {'Masked-SVTR' if use_mask_flag else 'Control'} FT (W=480, {r['gates_passed']}/6 Gates)",
                        "--stage", stage_str, "--epoch", str(ep_val),
                        "--source-kernel", src_kernel,
                        "--ckpt-path", ckpt_p, "--config-path", cfg_p, "--rec-image-shape", "3, 48, 480",
                        "--metrics-json", json.dumps(metrics)], cwd=PROJECT_ROOT, check=True)
        promoted = True

    new_sha16 = file_sha256(os.path.join(STUDIO_R4, "inference.pdiparams"))[:16]
    report = {
        "title": "Cham-OCR V26 Masked-SVTR Fine-Tune: causal evaluation & gated promotion",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "detector_signature": det_sig,
        "production_before": {"sha256_16": prod_sha16, "metrics": prod_metrics, "realscan_v2": prod_rs},
        "branches": {n: {k: v for k, v in b.items() if k not in ("ckpt_dir", "cfg_path")} for n, b in branches.items()},
        "soup_sweep": soup_res,
        "wise_sweep": wise_res,
        "promotion_decisions": decisions,
        "winner": winner,
        "promoted": promoted,
        "production_after_sha256_16": new_sha16,
        "backup_dir": os.path.relpath(backup_dir, PROJECT_ROOT).replace("\\", "/") if promoted else None,
    }
    out_j = os.path.join(PROJECT_ROOT, "REPORTS", "v26_masked_ft_evaluation.json")
    with open(out_j, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    def row(label, key):
        cells = [str(prod_metrics.get(key, "—"))]
        for n in ("masked", "control"):
            b = branches[n]
            cells.append(str(b["epochs"][f"epoch_{b['best_epoch']}"][key]) if b["best_epoch"] else "—")
        if soup_res and soup_res.get("best_key"):
            cells.append(str(soup_res["candidates"][soup_res["best_key"]].get(key, "—")))
        if wise_res and wise_res.get("best_key"):
            cells.append(str(wise_res["candidates"][wise_res["best_key"]].get(key, "—")))
        return f"| {label} | " + " | ".join(cells) + " |"

    has_soup = bool(soup_res and soup_res.get("best_key"))
    has_wise = bool(wise_res and wise_res.get("best_key"))
    hdr = "| Metric | Production R4.1 WiSE | V26 masked | V26 control |" + (" V26 masked soup |" if has_soup else "") + (" V26 masked WiSE |" if has_wise else "")
    sep = "| :--- | :---: | :---: | :---: |" + (" :---: |" if has_soup else "") + (" :---: |" if has_wise else "")
    md = [
        "# Đánh giá V26 Masked-SVTR Fine-Tune (mask attention + SE theo padding) và nhánh đối chứng",
        f"- **Thời điểm**: `{report['evaluated_at']}`",
        f"- **Production trước**: `v24_1_r4` SHA256_16 `{prod_sha16}` (R4.1 WiSE-FT, `det_sig={det_sig}`)",
    ]
    for n in ("masked", "control"):
        a = branches[n]["accounting"]
        md.append(f"- **{n}**: epoch tốt nhất `{branches[n]['best_epoch']}`; Train Loop / Overhead / Wall-clock = "
                  f"`{a.get('train_loop_duration_seconds')}s` / `{a.get('pipeline_container_overhead_seconds')}s` / `{a.get('total_kernel_wall_clock_seconds')}s`")
    md += ["", "## 1. Frozen-838 và các bộ phụ (cùng cấu hình detector hiện hành)", hdr, sep,
           row("Clean CER (%)", "clean_cer_pct"), row("Character CER (%)", "char_cer_pct"),
           row("14 Symbols (%)", "symbols_acc_pct"), row("Hyphen (%)", "hyphen_acc_pct"),
           row("FullPage Matched CER (%)", "fullpage_cer_pct"), row("Stress-82 CER (%)", "stress_82_cer_pct"),
           row("Probe-100 W=480 CER (%)", "probe_100_w480_cer_pct"), row("harum crops CER (%)", "harum_crops_cer_pct"),
           "", "## 2. Theo từng epoch", "| Nhánh | Epoch | Clean | Char | Symbols | Hyphen | FullPage | Stress-82 | Probe | Harum | Gates |",
           "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for n in ("masked", "control"):
        for k, r in branches[n]["epochs"].items():
            md.append(f"| {n} | {k.split('_')[1]} | {r['clean_cer_pct']}% | {r['char_cer_pct']}% | {r['symbols_acc_pct']}% | "
                      f"{r['hyphen_acc_pct']}% | {r['fullpage_cer_pct']}% | {r['stress_82_cer_pct']}% | {r['probe_100_w480_cer_pct']}% | "
                      f"{r['harum_crops_cer_pct']}% | {r['gates_passed']}/6 |")
    if has_soup:
        for k, r in soup_res["candidates"].items():
            md.append(f"| masked_soup | {k} | {r['clean_cer_pct']}% | {r['char_cer_pct']}% | {r['symbols_acc_pct']}% | "
                      f"{r['hyphen_acc_pct']}% | {r['fullpage_cer_pct']}% | {r['stress_82_cer_pct']}% | {r['probe_100_w480_cer_pct']}% | "
                      f"{r['harum_crops_cer_pct']}% | {r['gates_passed']}/6 |")
    if has_wise:
        for k, r in wise_res["candidates"].items():
            md.append(f"| masked_wise | a={r['alpha_r41_wise']:.2f} | {r['clean_cer_pct']}% | {r['char_cer_pct']}% | {r['symbols_acc_pct']}% | "
                      f"{r['hyphen_acc_pct']}% | {r['fullpage_cer_pct']}% | {r['stress_82_cer_pct']}% | {r['probe_100_w480_cer_pct']}% | "
                      f"{r['harum_crops_cer_pct']}% | {r['gates_passed']}/6 |")
    md += ["", "## 3. RealScan-Gold v2", "| Model | harum crops | Wikimedia | OOD-180 crops | OOD-12 E2E Doc CER (dòng) |",
           "| :--- | :---: | :---: | :---: | :---: |",
           f"| Production | {prod_rs['harum_crop_cer_pct']}% | {prod_rs['wikimedia_crop_cer_pct']}% | {prod_rs['ood180_crop_cer_pct']}% | {prod_rs['ood12_e2e_doc_cer_pct']}% ({prod_rs['ood12_e2e_lines']}) |"]
    for n, d in decisions.items():
        v = d["realscan_v2"]
        md.append(f"| V26 {n} | {v['harum_crop_cer_pct']}% | {v['wikimedia_crop_cer_pct']}% | {v['ood180_crop_cer_pct']}% | {v['ood12_e2e_doc_cer_pct']}% ({v['ood12_e2e_lines']}) |")
    md += ["", "## 4. Quyết định thăng cấp tự động"]
    for n, d in decisions.items():
        failed = [k for k, ok in d["checks"].items() if not ok]
        md.append(f"- **{n}**: qualifies = `{d['qualifies']}`" + (f" (trượt: `{failed}`)" if failed else ""))
    md.append(f"- **Winner**: `{winner}`; **promoted**: `{promoted}`; production SHA256_16 sau: `{new_sha16}`")
    out_m = os.path.join(PROJECT_ROOT, "REPORTS", "V26_MASKED_FT_EVALUATION.md")
    with open(out_m, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print(f"✅ V26 evaluation -> {out_m} (winner={winner}, promoted={promoted})")


if __name__ == "__main__":
    main()
