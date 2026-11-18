#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR V26-Full (30 Epochs, 3 Stages, Unfrozen W=480) Evaluation & Gated Auto-Promotion.

Reuses the shared scoring, export, and gate infrastructure from `evaluate_v26_masked_ft.py`,
adds multi-stage checkpoint resolution (S1 E10, S2 E20, S3 E24..E30) plus optional Stage 3
in-basin checkpoint soups, computes 3-part Ground-Truth Training Accounting across Stages 1-3,
evaluates `RealScan-Gold v2`, writes `REPORTS/V26_FULL_EVALUATION.{md,json}`, and promotes any
qualifying `6/6`-gate non-regressing model into `ocr-studio/data/output/rec_cham_inference_v24_1_r4/`.
"""

import os
import sys
import json
import shutil
import argparse
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

from evaluate_r2_1_6_causal import file_sha256  # noqa: E402
from evaluate_r2_1_7_causal import parse_train_log  # noqa: E402
from diagnostic_r2_1_4b_ctc_counterfactual import DICT_PATH  # noqa: E402
from frozen_gate_policy import FrozenGatePolicy  # noqa: E402
from evaluate_v26_masked_ft import (  # noqa: E402
    PY313,
    STUDIO_R4,
    _export_with_py26,
    _score_exported_model,
    active_detector_signature,
)

STAGES = {
    1: {"b_key": "v26_full_s1", "slug": "paddleocr-cham-v26-full-stage1", "remote_slug": "paddleocr-cham-v26-full-stage-1-e1-10", "cfg": "rec_cham_v26_full_s1.yml", "epochs": list(range(1, 11))},
    2: {"b_key": "v26_full_s2", "slug": "paddleocr-cham-v26-full-stage2", "cfg": "rec_cham_v26_full_s2.yml", "epochs": list(range(11, 21))},
    3: {"b_key": "v26_full_s3", "slug": "paddleocr-cham-v26-full-stage3", "cfg": "rec_cham_v26_full_s3.yml", "epochs": list(range(21, 31))},
}


def _stage_ckpt_info(stage_num: int, epoch: int) -> tuple[str, str]:
    spec = STAGES[stage_num]
    dl_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"r2_1_4c_{spec['b_key']}_kaggle_output")
    ckpt_p = os.path.join(dl_dir, "extracted_checkpoints", f"iter_epoch_{epoch}.pdparams")
    cfg_p = os.path.join(PROJECT_ROOT, "ocr-training", "configs", spec["cfg"])
    return ckpt_p, cfg_p


def collect_stage_accounting() -> tuple[dict, dict]:
    per_stage = {}
    tot_train_s, tot_over_s, tot_wall_s = 0.0, 0.0, 0.0
    have_all_wall = True
    for s_num, spec in STAGES.items():
        dl_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"r2_1_4c_{spec['b_key']}_kaggle_output")
        t_log = os.path.join(dl_dir, "extracted_checkpoints", "train.log")
        k_log = os.path.join(dl_dir, f"{spec['slug']}.log")
        if os.path.exists(t_log):
            acc = parse_train_log(t_log, total_epochs=30, kernel_log_path=k_log)
            per_stage[f"stage_{s_num}"] = acc
            if isinstance(acc.get("train_loop_duration_seconds"), (int, float)):
                tot_train_s += float(acc["train_loop_duration_seconds"])
            if isinstance(acc.get("pipeline_container_overhead_seconds"), (int, float)):
                tot_over_s += float(acc["pipeline_container_overhead_seconds"])
            else:
                have_all_wall = False
            if isinstance(acc.get("total_kernel_wall_clock_seconds"), (int, float)):
                tot_wall_s += float(acc["total_kernel_wall_clock_seconds"])
            else:
                have_all_wall = False
    cumulative = {
        "train_loop_duration_seconds": round(tot_train_s, 2),
        "train_loop_duration_minutes": round(tot_train_s / 60.0, 2),
        "pipeline_container_overhead_seconds": round(tot_over_s, 2) if have_all_wall else None,
        "pipeline_container_overhead_minutes": round(tot_over_s / 60.0, 2) if have_all_wall else None,
        "total_kernel_wall_clock_seconds": round(tot_wall_s, 2) if have_all_wall else None,
        "total_kernel_wall_clock_minutes": round(tot_wall_s / 60.0, 2) if have_all_wall else None,
    }
    return per_stage, cumulative


def eval_stage3_soups(epoch_rows: dict, policy: FrozenGatePolicy, det_sig: str) -> dict:
    import paddle

    s3_eps = sorted(int(k.split("_")[1]) for k in epoch_rows if int(k.split("_")[1]) >= 21)
    if len(s3_eps) < 2:
        return {"candidates": {}, "best_key": None}
    ranked_s3 = sorted(
        s3_eps,
        key=lambda e: (
            1 if epoch_rows[f"epoch_{e}"]["valid_for_release"] else 0,
            epoch_rows[f"epoch_{e}"]["gates_passed"],
            -epoch_rows[f"epoch_{e}"]["char_cer_pct"] - epoch_rows[f"epoch_{e}"]["clean_cer_pct"],
            -epoch_rows[f"epoch_{e}"]["harum_crops_cer_pct"],
        ),
        reverse=True,
    )
    top2 = ranked_s3[:2]
    recipes = {f"soup_top2_e{top2[0]}_e{top2[1]}": {top2[0]: 0.5, top2[1]: 0.5}}
    if len(ranked_s3) >= 3:
        top3 = ranked_s3[:3]
        recipes[f"soup_top3_e{top3[0]}_e{top3[1]}_e{top3[2]}"] = {top3[0]: 0.4, top3[1]: 0.35, top3[2]: 0.25}
    if all(e in s3_eps for e in (28, 29, 30)):
        recipes["soup_last3_e28_e29_e30"] = {28: 0.30, 29: 0.35, 30: 0.35}

    needed_eps = sorted({ep for wmap in recipes.values() for ep in wmap})
    loaded = {ep: paddle.load(_stage_ckpt_info(3, ep)[0]) for ep in needed_eps}
    cfg_p = _stage_ckpt_info(3, 30)[1]
    keys = list(loaded[needed_eps[0]].keys())

    out, best_key, best_rank = {}, None, None
    for sname, wmap in recipes.items():
        s_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"v26_full_{sname}_ckpt")
        os.makedirs(s_dir, exist_ok=True)
        s_prefix = os.path.join(s_dir, "best")
        s_file = s_prefix + ".pdparams"
        blended = {}
        for k in keys:
            acc = None
            for ep, w in wmap.items():
                term = float(w) * loaded[ep][k]
                acc = term if acc is None else (acc + term)
            blended[k] = acc
        paddle.save(blended, s_file)
        infer_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"v26_full_{sname}_infer")
        if not os.path.exists(os.path.join(infer_dir, "inference.pdiparams")):
            _export_with_py26(cfg_p, s_prefix, infer_dir)
        r = _score_exported_model(f"v26_full_{sname}", infer_dir, s_file, True, policy, det_sig)
        r["weights"] = {f"epoch_{ep}": w for ep, w in wmap.items()}
        r["ckpt_path"] = s_file
        out[sname] = r
        rank = (1 if r["valid_for_release"] else 0, r["gates_passed"], -r["char_cer_pct"] - r["clean_cer_pct"], -r["harum_crops_cer_pct"])
        if best_rank is None or rank > best_rank:
            best_rank, best_key = rank, sname
        print(f"  -> V26-Full {sname}: Clean={r['clean_cer_pct']}% Char={r['char_cer_pct']}% Sym={r['symbols_acc_pct']}% "
              f"(fail={r['failing_symbols']}) Hyp={r['hyphen_acc_pct']}% FP={r['fullpage_cer_pct']}% Stress={r['stress_82_cer_pct']}% "
              f"Harum={r['harum_crops_cer_pct']}% Gates={r['gates_passed']}/6 valid={r['valid_for_release']}")
    return {"candidates": out, "best_key": best_key}


def eval_inbasin_wise_ft(policy: FrozenGatePolicy, det_sig: str) -> dict:
    """Sweep in-basin WiSE-FT blends between V26-Masked soup (frozen early backbone) and V26-Full S3 soup."""
    import paddle

    m_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "r2_1_4c_v26_masked_ft_kaggle_output", "extracted_checkpoints")
    m3_p, m4_p = os.path.join(m_dir, "iter_epoch_3.pdparams"), os.path.join(m_dir, "iter_epoch_4.pdparams")
    f_p = os.path.join(PROJECT_ROOT, "ocr-training", "output", "v26_full_soup_top3_e25_e27_e30_ckpt", "best.pdparams")
    if not (os.path.exists(m3_p) and os.path.exists(m4_p) and os.path.exists(f_p)):
        print("  ! Missing anchor checkpoints for in-basin WiSE-FT sweep, skipping")
        return {"candidates": {}}

    m3, m4, t1 = paddle.load(m3_p), paddle.load(m4_p), paddle.load(f_p)
    t0 = {k: 0.5 * m3[k] + 0.5 * m4[k] for k in t1.keys()}
    early_prefixes = ("backbone.conv1.", "backbone.blocks2.", "backbone.blocks3.")
    recipes = [
        ("inbasin_wise_a25", 0.25, 0.25),
        ("inbasin_wise_a35", 0.35, 0.35),
        ("inbasin_wise_a45", 0.45, 0.45),
        ("inbasin_wise_lw_15_45", 0.15, 0.45),
        ("inbasin_wise_lw_20_55", 0.20, 0.55),
    ]
    cfg_p = _stage_ckpt_info(3, 30)[1]
    out = {}
    for wname, a_early, a_rest in recipes:
        w_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"v26_full_{wname}_ckpt")
        os.makedirs(w_dir, exist_ok=True)
        w_prefix = os.path.join(w_dir, "best")
        w_file = w_prefix + ".pdparams"
        blended = {}
        for k in t1.keys():
            a = a_early if k.startswith(early_prefixes) else a_rest
            blended[k] = float(1.0 - a) * t0[k] + float(a) * t1[k]
        paddle.save(blended, w_file)
        infer_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"v26_full_{wname}_infer")
        if not os.path.exists(os.path.join(infer_dir, "inference.pdiparams")):
            _export_with_py26(cfg_p, w_prefix, infer_dir)
        r = _score_exported_model(f"v26_full_{wname}", infer_dir, w_file, True, policy, det_sig)
        r["alpha_early"] = a_early
        r["alpha_rest"] = a_rest
        r["ckpt_path"] = w_file
        out[wname] = r
        print(f"  -> V26-Full {wname} (a_early={a_early}, a_rest={a_rest}): Clean={r['clean_cer_pct']}% "
              f"Char={r['char_cer_pct']}% Sym={r['symbols_acc_pct']}% (fail={r['failing_symbols']}) "
              f"Hyp={r['hyphen_acc_pct']}% FP={r['fullpage_cer_pct']}% Stress={r['stress_82_cer_pct']}% "
              f"Harum={r['harum_crops_cer_pct']}% Gates={r['gates_passed']}/6 valid={r['valid_for_release']}")
    return {"candidates": out}


def realscan_v2_full(models: dict) -> dict:
    out_j = os.path.join(PROJECT_ROOT, "REPORTS", "v26_full_realscan_gold_v2_comparison.json")
    out_m = os.path.join(PROJECT_ROOT, "REPORTS", "v26_full_realscan_gold_v2_comparison.md")
    specs = [f"{alias}={d}:{DICT_PATH}" for alias, d in models.items()]
    subprocess.run([PY313, os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts", "evaluate_realscan_gold_v2.py"),
                    "--models", *specs, "--output-json", out_j, "--output-md", out_m], cwd=PROJECT_ROOT, check=True)
    return json.load(open(out_j, encoding="utf-8"))["models"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", nargs="*", type=int, default=[10, 20, 24, 25, 26, 27, 28, 29, 30])
    ap.add_argument("--no-promote", action="store_true")
    ap.add_argument("--reuse-cached", action="store_true")
    args = ap.parse_args()

    policy = FrozenGatePolicy()
    det_sig = active_detector_signature()
    prod_manifest = json.load(open(os.path.join(STUDIO_R4, "model_manifest.json"), "r", encoding="utf-8"))
    prod_ckpt = os.path.join(PROJECT_ROOT, *prod_manifest["source_checkpoint"].split("/"))
    prod_sha16 = file_sha256(os.path.join(STUDIO_R4, "inference.pdiparams"))[:16]
    prod_use_mask = bool((prod_manifest.get("benchmark_metrics") or {}).get("use_valid_mask", False))

    prod_snapshot = os.path.join(PROJECT_ROOT, "ocr-training", "output", "v26_full_prod_snapshot")
    if os.path.exists(prod_snapshot):
        shutil.rmtree(prod_snapshot)
    shutil.copytree(STUDIO_R4, prod_snapshot)

    per_stage_acc, cum_acc = collect_stage_accounting()
    out_j = os.path.join(PROJECT_ROOT, "REPORTS", "v26_full_evaluation.json")
    prev = None
    if args.reuse_cached and os.path.exists(out_j):
        cand_prev = json.load(open(out_j, "r", encoding="utf-8"))
        if cand_prev.get("detector_signature") == det_sig and (cand_prev.get("production_before") or {}).get("sha256_16") == prod_sha16 and cand_prev.get("epochs"):
            prev = cand_prev

    if prev is not None:
        print("=" * 80 + f"\n♻️ Reusing cached prod/epoch/soup scores from {out_j} ({prod_sha16}, det_sig={det_sig})\n" + "=" * 80)
        prod_metrics = prev["production_before"]["metrics"]
        epoch_results = prev["epochs"]
        best_ep = prev["best_epoch"]
        soup_res = prev.get("soup_sweep")
    else:
        print("=" * 80 + f"\n🔬 Scoring active `v24_1_r4` baseline ({prod_sha16}, det_sig={det_sig})\n" + "=" * 80)
        prod_metrics = _score_exported_model("v26_full_prod_active", prod_snapshot, prod_ckpt, prod_use_mask, policy, det_sig)

        epoch_results = {}
        best_ep, best_rank = None, None
        for ep in sorted(set(args.epochs)):
            s_num = 1 if ep <= 10 else (2 if ep <= 20 else 3)
            ckpt_p, cfg_p = _stage_ckpt_info(s_num, ep)
            if not os.path.exists(ckpt_p):
                print(f"  ! Missing V26-Full epoch {ep} ({ckpt_p}), skipping")
                continue
            infer_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"v26_full_infer_e{ep}")
            if not os.path.exists(os.path.join(infer_dir, "inference.pdiparams")):
                _export_with_py26(cfg_p, ckpt_p[:-9], infer_dir)
            r = _score_exported_model(f"v26_full_e{ep}", infer_dir, ckpt_p, True, policy, det_sig)
            r["stage"] = s_num
            epoch_results[f"epoch_{ep}"] = r
            rank = (1 if r["valid_for_release"] else 0, r["gates_passed"], -r["char_cer_pct"] - r["clean_cer_pct"], -r["harum_crops_cer_pct"])
            if best_rank is None or rank > best_rank:
                best_rank, best_ep = rank, ep
            print(f"  -> V26-Full S{s_num} E{ep}: Clean={r['clean_cer_pct']}% Char={r['char_cer_pct']}% Sym={r['symbols_acc_pct']}% "
                  f"(fail={r['failing_symbols']}) Hyp={r['hyphen_acc_pct']}% FP={r['fullpage_cer_pct']}% Stress={r['stress_82_cer_pct']}% "
                  f"Probe={r['probe_100_w480_cer_pct']}% Harum={r['harum_crops_cer_pct']}% Gates={r['gates_passed']}/6 valid={r['valid_for_release']}")

        soup_res = None
        if any(int(k.split("_")[1]) >= 21 for k in epoch_results):
            print("=" * 80 + "\n🥣 Running V26-Full Stage 3 in-basin checkpoint soups\n" + "=" * 80)
            soup_res = eval_stage3_soups(epoch_results, policy, det_sig)

    wise_res = None
    if soup_res:
        print("=" * 80 + "\n⚗️ Running V26-Full in-basin WiSE-FT sweep (V26-Masked soup <-> V26-Full S3 soup)\n" + "=" * 80)
        wise_res = eval_inbasin_wise_ft(policy, det_sig)

    rs_models = {"prod_active": prod_snapshot}
    if best_ep is not None:
        rs_models["v26_full_best"] = epoch_results[f"epoch_{best_ep}"]["infer_dir"]
    if soup_res and soup_res.get("best_key"):
        rs_models["v26_full_soup"] = soup_res["candidates"][soup_res["best_key"]]["infer_dir"]
    if wise_res and wise_res.get("candidates"):
        for wname, wr in wise_res["candidates"].items():
            if wr.get("valid_for_release"):
                rs_models[wname] = wr["infer_dir"]
    rs = realscan_v2_full(rs_models)

    def rs_view(alias):
        m = rs[alias]
        return {
            "harum_crop_cer_pct": m["harum_palei_01_crops"]["norm_cer_pct"],
            "wikimedia_crop_cer_pct": m["wikimedia_specimens"]["norm_cer_pct"],
            "ood180_crop_cer_pct": m["ood_archival_overall"]["norm_cer_pct"],
            "ood12_e2e_doc_cer_pct": m["ood_archival_overall"]["e2e_doc_cer_pct"],
            "ood12_e2e_lines": m["ood_archival_overall"]["e2e_lines"],
        }

    prod_rs = rs_view("prod_active")

    def _check(r_metrics, v_rs):
        checks = {
            "valid_for_release_6_of_6": bool(r_metrics["valid_for_release"]),
            "clean_non_regression": r_metrics["clean_cer_pct"] <= float(prod_metrics["clean_cer_pct"]) + 0.20,
            "char_non_regression": r_metrics["char_cer_pct"] <= float(prod_metrics["char_cer_pct"]) + 0.20,
            "blank_guard": bool(r_metrics["blank_passed"]),
            "harum_crop_non_regression": v_rs["harum_crop_cer_pct"] <= prod_rs["harum_crop_cer_pct"] + 0.25,
            "ood180_crop_non_regression": v_rs["ood180_crop_cer_pct"] <= prod_rs["ood180_crop_cer_pct"] + 0.30,
            "ood12_e2e_non_regression": v_rs["ood12_e2e_doc_cer_pct"] <= prod_rs["ood12_e2e_doc_cer_pct"] + 0.30,
            "wikimedia_non_regression": v_rs["wikimedia_crop_cer_pct"] <= prod_rs["wikimedia_crop_cer_pct"] + 2.0,
            "beats_or_ties_active_prod": (
                r_metrics["gates_passed"] > prod_metrics["gates_passed"]
                or (r_metrics["clean_cer_pct"] + r_metrics["char_cer_pct"] <= prod_metrics["clean_cer_pct"] + prod_metrics["char_cer_pct"] + 0.05)
            ),
        }
        return checks, all(checks.values())

    decisions = {}
    if best_ep is not None and "v26_full_best" in rs:
        r = epoch_results[f"epoch_{best_ep}"]
        v = rs_view("v26_full_best")
        chk, ok = _check(r, v)
        decisions["epoch_best"] = {"best_epoch": best_ep, "realscan_v2": v, "checks": chk, "qualifies": ok}
    if soup_res and soup_res.get("best_key") and "v26_full_soup" in rs:
        sk = soup_res["best_key"]
        sr = soup_res["candidates"][sk]
        sv = rs_view("v26_full_soup")
        schk, sok = _check(sr, sv)
        decisions["soup"] = {"best_key": sk, "weights": sr["weights"], "realscan_v2": sv, "checks": schk, "qualifies": sok}

    if wise_res and wise_res.get("candidates"):
        for wname, wr in wise_res["candidates"].items():
            if wname in rs:
                wv = rs_view(wname)
                wchk, wok = _check(wr, wv)
                decisions[wname] = {
                    "alpha_early": wr["alpha_early"],
                    "alpha_rest": wr["alpha_rest"],
                    "realscan_v2": wv,
                    "checks": wchk,
                    "qualifies": wok,
                }

    cand_scores = []
    if decisions.get("epoch_best", {}).get("qualifies"):
        r = epoch_results[f"epoch_{best_ep}"]
        v = decisions["epoch_best"]["realscan_v2"]
        cand_scores.append((r["char_cer_pct"] + r["clean_cer_pct"] + v["ood180_crop_cer_pct"], "epoch_best"))
    if decisions.get("soup", {}).get("qualifies"):
        sr = soup_res["candidates"][soup_res["best_key"]]
        sv = decisions["soup"]["realscan_v2"]
        cand_scores.append((sr["char_cer_pct"] + sr["clean_cer_pct"] + sv["ood180_crop_cer_pct"], "soup"))
    if wise_res and wise_res.get("candidates"):
        for wname, wr in wise_res["candidates"].items():
            if decisions.get(wname, {}).get("qualifies"):
                wv = decisions[wname]["realscan_v2"]
                cand_scores.append((wr["char_cer_pct"] + wr["clean_cer_pct"] + wv["ood180_crop_cer_pct"], wname))
    winner = min(cand_scores)[1] if cand_scores else None

    promoted = False
    backup_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "rec_cham_inference_v24_1_r4_r41wise_backup")
    if winner and not args.no_promote:
        if winner == "epoch_best":
            ep_val = int(best_ep)
            r = epoch_results[f"epoch_{ep_val}"]
            s_num = r["stage"]
            ckpt_p, cfg_p = _stage_ckpt_info(s_num, ep_val)
            stage_str = f"V26_FULL_30EP_S{s_num}_W480"
            src_kernel = "gustavnguyen/" + STAGES[s_num].get("remote_slug", STAGES[s_num]["slug"])
            disp_name = f"Cham-OCR V26-Full Masked-SVTR (30ep W=480, {r['gates_passed']}/6 Gates)"
        elif winner.startswith("inbasin_wise_"):
            r = wise_res["candidates"][winner]
            ckpt_p = r["ckpt_path"]
            cfg_p = _stage_ckpt_info(3, 30)[1]
            ep_val = 30
            stage_str = f"V26_FULL_30EP_{winner.upper()}_W480"
            src_kernel = "gustavnguyen/" + STAGES[3]["slug"]
            disp_name = f"Cham-OCR V26-Full Masked-SVTR WiSE-FT (30ep W=480, {r['gates_passed']}/6 Gates)"
        else:
            r = soup_res["candidates"][soup_res["best_key"]]
            ckpt_p = r["ckpt_path"]
            cfg_p = _stage_ckpt_info(3, 30)[1]
            ep_val = 30
            stage_str = f"V26_FULL_30EP_{soup_res['best_key'].upper()}_W480"
            src_kernel = "gustavnguyen/" + STAGES[3]["slug"]
            disp_name = f"Cham-OCR V26-Full Masked-SVTR (30ep W=480, {r['gates_passed']}/6 Gates)"

        if not os.path.exists(backup_dir):
            shutil.copytree(STUDIO_R4, backup_dir)
        metrics = {
            "clean_cer_pct": r["clean_cer_pct"], "char_cer_pct": r["char_cer_pct"],
            "symbols_acc_pct": r["symbols_acc_pct"], "symbols_passed": r["symbols_passed"],
            "failing_symbols": r["failing_symbols"], "hyphen_acc_pct": r["hyphen_acc_pct"],
            "fullpage_cer_pct": r["fullpage_cer_pct"], "stress_82_cer_pct": r["stress_82_cer_pct"],
            "probe_100_w480_cer_pct": r["probe_100_w480_cer_pct"], "harum_crops_cer_pct": r["harum_crops_cer_pct"],
            "gates_passed": f"{r['gates_passed']}/6", "valid_for_release": r["valid_for_release"],
            "use_valid_mask": True,
        }
        subprocess.run([PY313, os.path.join(PROJECT_ROOT, "scripts", "package_v24_1_model.py"),
                        "--model-id", "v24_1_r4",
                        "--display-name", disp_name,
                        "--stage", stage_str, "--epoch", str(ep_val),
                        "--source-kernel", src_kernel,
                        "--ckpt-path", ckpt_p, "--config-path", cfg_p, "--rec-image-shape", "3, 48, 480",
                        "--metrics-json", json.dumps(metrics)], cwd=PROJECT_ROOT, check=True)
        promoted = True

    if os.path.exists(prod_snapshot):
        shutil.rmtree(prod_snapshot, ignore_errors=True)

    new_sha16 = file_sha256(os.path.join(STUDIO_R4, "inference.pdiparams"))[:16]
    report = {
        "title": "Cham-OCR V26-Full (30 Epochs, 3 Stages Unfrozen W=480) Evaluation & Gated Promotion",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "detector_signature": det_sig,
        "production_before": {"sha256_16": prod_sha16, "metrics": prod_metrics, "realscan_v2": prod_rs},
        "stage_accounting": per_stage_acc,
        "cumulative_accounting": cum_acc,
        "epochs": epoch_results,
        "best_epoch": best_ep,
        "soup_sweep": soup_res,
        "inbasin_wise_sweep": wise_res,
        "promotion_decisions": decisions,
        "winner": winner,
        "promoted": promoted,
        "production_after_sha256_16": new_sha16,
    }
    with open(out_j, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md = [
        "# Đánh giá V26-Full (30 Epochs, 3 Chặng Unfrozen trên Kaggle Dual Tesla T4x2)",
        f"- **Thời điểm**: `{report['evaluated_at']}`",
        f"- **Production trước**: `v24_1_r4` SHA256_16 `{prod_sha16}` (`det_sig={det_sig}`)",
        f"- **Tổng kế toán 3 chặng**: Train Loop = `{cum_acc['train_loop_duration_seconds']}s` (`{cum_acc['train_loop_duration_minutes']}m`), "
        f"Overhead = `{cum_acc['pipeline_container_overhead_seconds']}s`, Wall-clock = `{cum_acc['total_kernel_wall_clock_seconds']}s`",
        "",
        "## 1. Kế toán theo từng chặng",
        "| Chặng | Train Loop (s) | Overhead (s) | Wall-clock (s) |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for s_key, a in per_stage_acc.items():
        md.append(f"| {s_key} | {a.get('train_loop_duration_seconds')} | {a.get('pipeline_container_overhead_seconds')} | {a.get('total_kernel_wall_clock_seconds')} |")

    md += ["", "## 2. Kết quả Benchmark theo Epoch (`W=480`)",
           "| Nguồn | Clean | Char | Symbols | Hyphen | FullPage | Stress-82 | Probe | Harum | Gates |",
           "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
           f"| Production trước | {prod_metrics['clean_cer_pct']}% | {prod_metrics['char_cer_pct']}% | {prod_metrics['symbols_acc_pct']}% | "
           f"{prod_metrics['hyphen_acc_pct']}% | {prod_metrics['fullpage_cer_pct']}% | {prod_metrics['stress_82_cer_pct']}% | "
           f"{prod_metrics['probe_100_w480_cer_pct']}% | {prod_metrics['harum_crops_cer_pct']}% | {prod_metrics['gates_passed']}/6 |"]
    for ek, r in epoch_results.items():
        md.append(f"| S{r['stage']} E{ek.split('_')[1]} | {r['clean_cer_pct']}% | {r['char_cer_pct']}% | {r['symbols_acc_pct']}% | "
                  f"{r['hyphen_acc_pct']}% | {r['fullpage_cer_pct']}% | {r['stress_82_cer_pct']}% | "
                  f"{r['probe_100_w480_cer_pct']}% | {r['harum_crops_cer_pct']}% | {r['gates_passed']}/6 |")
    if soup_res and soup_res.get("candidates"):
        for sk, r in soup_res["candidates"].items():
            md.append(f"| {sk} | {r['clean_cer_pct']}% | {r['char_cer_pct']}% | {r['symbols_acc_pct']}% | "
                      f"{r['hyphen_acc_pct']}% | {r['fullpage_cer_pct']}% | {r['stress_82_cer_pct']}% | "
                      f"{r['probe_100_w480_cer_pct']}% | {r['harum_crops_cer_pct']}% | {r['gates_passed']}/6 |")
    if wise_res and wise_res.get("candidates"):
        for wk, r in wise_res["candidates"].items():
            md.append(f"| {wk} | {r['clean_cer_pct']}% | {r['char_cer_pct']}% | {r['symbols_acc_pct']}% | "
                      f"{r['hyphen_acc_pct']}% | {r['fullpage_cer_pct']}% | {r['stress_82_cer_pct']}% | "
                      f"{r['probe_100_w480_cer_pct']}% | {r['harum_crops_cer_pct']}% | {r['gates_passed']}/6 |")

    md += ["", "## 3. RealScan-Gold v2",
           "| Model | harum crops | Wikimedia | OOD-180 crops | OOD-12 E2E Doc CER (dòng) |",
           "| :--- | :---: | :---: | :---: | :---: |",
           f"| Production trước | {prod_rs['harum_crop_cer_pct']}% | {prod_rs['wikimedia_crop_cer_pct']}% | {prod_rs['ood180_crop_cer_pct']}% | {prod_rs['ood12_e2e_doc_cer_pct']}% ({prod_rs['ood12_e2e_lines']}) |"]
    for dk, dv in decisions.items():
        v = dv["realscan_v2"]
        md.append(f"| V26-Full ({dk}) | {v['harum_crop_cer_pct']}% | {v['wikimedia_crop_cer_pct']}% | {v['ood180_crop_cer_pct']}% | {v['ood12_e2e_doc_cer_pct']}% ({v['ood12_e2e_lines']}) |")

    md += ["", "## 4. Quyết định thăng cấp tự động"]
    for dk, dv in decisions.items():
        failed = [k for k, ok in dv["checks"].items() if not ok]
        md.append(f"- **{dk}**: qualifies = `{dv['qualifies']}`" + (f" (trượt: `{failed}`)" if failed else ""))
    md.append(f"- **Winner**: `{winner}`; **promoted**: `{promoted}`; production SHA256_16 sau: `{new_sha16}`")

    out_m = os.path.join(PROJECT_ROOT, "REPORTS", "V26_FULL_EVALUATION.md")
    with open(out_m, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print(f"✅ V26-Full evaluation -> {out_m} (winner={winner}, promoted={promoted})")


if __name__ == "__main__":
    main()
