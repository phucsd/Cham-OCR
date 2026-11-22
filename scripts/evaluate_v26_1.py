#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR V26.1 Masked-SVTR (16 Epochs, 2 Stages, Early-Backbone Frozen + BN-Locked + Paleographic Aug)
Evaluation, In-Basin Soup/WiSE-FT Sweep, Gated Auto-Promotion & T* Confidence Calibration.
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
    1: {"b_key": "v26_1_s1", "slug": "paddleocr-cham-v26-1-stage1", "cfg": "rec_cham_v26_1_s1.yml", "epochs": list(range(1, 9))},
    2: {"b_key": "v26_1_s2", "slug": "paddleocr-cham-v26-1-stage2", "cfg": "rec_cham_v26_1_s2.yml", "epochs": list(range(9, 17))},
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
            acc = parse_train_log(t_log, total_epochs=16, kernel_log_path=k_log)
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


def eval_v26_1_soups(epoch_rows: dict, policy: FrozenGatePolicy, det_sig: str) -> dict:
    import paddle

    avail_eps = sorted(int(k.split("_")[1]) for k in epoch_rows)
    if len(avail_eps) < 2:
        return {"candidates": {}, "best_key": None}
    ranked = sorted(
        avail_eps,
        key=lambda e: (
            1 if epoch_rows[f"epoch_{e}"]["valid_for_release"] else 0,
            epoch_rows[f"epoch_{e}"]["gates_passed"],
            -epoch_rows[f"epoch_{e}"]["char_cer_pct"] - epoch_rows[f"epoch_{e}"]["clean_cer_pct"],
            -epoch_rows[f"epoch_{e}"]["harum_crops_cer_pct"],
        ),
        reverse=True,
    )
    top2 = ranked[:2]
    recipes = {f"soup_top2_e{top2[0]}_e{top2[1]}": {top2[0]: 0.5, top2[1]: 0.5}}
    if len(ranked) >= 3:
        top3 = ranked[:3]
        recipes[f"soup_top3_e{top3[0]}_e{top3[1]}_e{top3[2]}"] = {top3[0]: 0.4, top3[1]: 0.35, top3[2]: 0.25}
    if all(e in avail_eps for e in (14, 15, 16)):
        recipes["soup_last3_e14_e15_e16"] = {14: 0.30, 15: 0.35, 16: 0.35}

    needed_eps = sorted({ep for wmap in recipes.values() for ep in wmap})
    loaded = {ep: paddle.load(_stage_ckpt_info(1 if ep <= 8 else 2, ep)[0]) for ep in needed_eps}
    cfg_p = _stage_ckpt_info(2 if max(needed_eps) > 8 else 1, max(needed_eps))[1]
    keys = list(loaded[needed_eps[0]].keys())

    out, best_key, best_rank = {}, None, None
    for sname, wmap in recipes.items():
        s_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"v26_1_{sname}_ckpt")
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
        infer_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"v26_1_{sname}_infer")
        if not os.path.exists(os.path.join(infer_dir, "inference.pdiparams")):
            _export_with_py26(cfg_p, s_prefix, infer_dir)
        r = _score_exported_model(f"v26_1_{sname}", infer_dir, s_file, True, policy, det_sig)
        r["weights"] = {f"epoch_{ep}": w for ep, w in wmap.items()}
        r["ckpt_path"] = s_file
        out[sname] = r
        rank = (1 if r["valid_for_release"] else 0, r["gates_passed"], -r["char_cer_pct"] - r["clean_cer_pct"], -r["harum_crops_cer_pct"])
        if best_rank is None or rank > best_rank:
            best_rank, best_key = rank, sname
        print(f"  -> V26.1 {sname}: Clean={r['clean_cer_pct']}% Char={r['char_cer_pct']}% Sym={r['symbols_acc_pct']}% "
              f"(fail={r['failing_symbols']}) Hyp={r['hyphen_acc_pct']}% FP={r['fullpage_cer_pct']}% Stress={r['stress_82_cer_pct']}% "
              f"Harum={r['harum_crops_cer_pct']}% Gates={r['gates_passed']}/6 valid={r['valid_for_release']}")
    return {"candidates": out, "best_key": best_key}


def eval_true_inbasin_wise_ft(target_ckpt: str, cfg_p: str, policy: FrozenGatePolicy, det_sig: str, prefix: str = "wise_inbasin", alphas=(0.35, 0.50, 0.65, 0.80)) -> dict:
    """True in-basin WiSE-FT: both anchor (`v26_1_base`) and `V26.1` share 100% identical frozen conv1..blocks3 + BN stats!"""
    import paddle

    anchor_p = os.path.join(PROJECT_ROOT, "ocr-training", "output", "v26_1_base", "best.pdparams")
    if not (os.path.exists(anchor_p) and os.path.exists(target_ckpt)):
        return {"candidates": {}}

    t0 = paddle.load(anchor_p)
    t1 = paddle.load(target_ckpt)
    out = {}
    for alpha in alphas:
        wname = f"{prefix}_a{int(round(alpha * 100))}"
        w_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"v26_1_{wname}_ckpt")
        os.makedirs(w_dir, exist_ok=True)
        w_prefix = os.path.join(w_dir, "best")
        w_file = w_prefix + ".pdparams"
        blended = {k: float(1.0 - alpha) * t0[k] + float(alpha) * t1[k] for k in t1.keys()}
        paddle.save(blended, w_file)
        new_sha = file_sha256(w_file)
        infer_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"v26_1_{wname}_infer")
        sha_marker = os.path.join(infer_dir, ".ckpt_sha256")
        old_sha = open(sha_marker, encoding="utf-8").read().strip() if os.path.exists(sha_marker) else None
        if not os.path.exists(os.path.join(infer_dir, "inference.pdiparams")) or (old_sha is not None and old_sha != new_sha):
            _export_with_py26(cfg_p, w_prefix, infer_dir)
        with open(sha_marker, "w", encoding="utf-8") as mf:
            mf.write(new_sha + "\n")
        r = _score_exported_model(f"v26_1_{wname}", infer_dir, w_file, True, policy, det_sig)
        r["alpha"] = alpha
        r["ckpt_path"] = w_file
        out[wname] = r
        print(f"  -> V26.1 {wname} (alpha={alpha}): Clean={r['clean_cer_pct']}% Char={r['char_cer_pct']}% "
              f"Sym={r['symbols_acc_pct']}% (fail={r['failing_symbols']}) Hyp={r['hyphen_acc_pct']}% "
              f"FP={r['fullpage_cer_pct']}% Stress={r['stress_82_cer_pct']}% Harum={r['harum_crops_cer_pct']}% "
              f"Gates={r['gates_passed']}/6 valid={r['valid_for_release']}")
    return {"candidates": out}


def realscan_v2_eval(models: dict) -> dict:
    out_j = os.path.join(PROJECT_ROOT, "REPORTS", "v26_1_realscan_gold_v2_comparison.json")
    out_m = os.path.join(PROJECT_ROOT, "REPORTS", "v26_1_realscan_gold_v2_comparison.md")
    specs = [f"{alias}={d}:{DICT_PATH}" for alias, d in models.items()]
    subprocess.run([PY313, os.path.join(PROJECT_ROOT, "ocr-benchmark", "scripts", "evaluate_realscan_gold_v2.py"),
                    "--models", *specs, "--output-json", out_j, "--output-md", out_m], cwd=PROJECT_ROOT, check=True)
    return json.load(open(out_j, encoding="utf-8"))["models"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", nargs="*", type=int, default=[4, 6, 8, 12, 13, 14, 15, 16])
    ap.add_argument("--no-promote", action="store_true")
    args = ap.parse_args()

    policy = FrozenGatePolicy()
    det_sig = active_detector_signature()
    prod_manifest = json.load(open(os.path.join(STUDIO_R4, "model_manifest.json"), "r", encoding="utf-8"))
    prod_ckpt = os.path.join(PROJECT_ROOT, *prod_manifest["source_checkpoint"].split("/"))
    prod_sha16 = file_sha256(os.path.join(STUDIO_R4, "inference.pdiparams"))[:16]
    prod_use_mask = bool((prod_manifest.get("benchmark_metrics") or {}).get("use_valid_mask", False))

    prod_snapshot = os.path.join(PROJECT_ROOT, "ocr-training", "output", "v26_1_prod_snapshot")
    if os.path.exists(prod_snapshot):
        shutil.rmtree(prod_snapshot)
    shutil.copytree(STUDIO_R4, prod_snapshot)

    per_stage_acc, cum_acc = collect_stage_accounting()
    print("=" * 80 + f"\n🔬 Scoring active `v24_1_r4` baseline ({prod_sha16}, det_sig={det_sig})\n" + "=" * 80)
    prod_metrics = _score_exported_model("v26_1_prod_active", prod_snapshot, prod_ckpt, prod_use_mask, policy, det_sig)

    epoch_results = {}
    best_ep, best_rank = None, None
    for ep in sorted(set(args.epochs)):
        s_num = 1 if ep <= 8 else 2
        ckpt_p, cfg_p = _stage_ckpt_info(s_num, ep)
        if not os.path.exists(ckpt_p):
            print(f"  ! Missing V26.1 epoch {ep} ({ckpt_p}), skipping")
            continue
        infer_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", f"v26_1_infer_e{ep}")
        if not os.path.exists(os.path.join(infer_dir, "inference.pdiparams")):
            _export_with_py26(cfg_p, ckpt_p[:-9], infer_dir)
        r = _score_exported_model(f"v26_1_e{ep}", infer_dir, ckpt_p, True, policy, det_sig)
        r["stage"] = s_num
        epoch_results[f"epoch_{ep}"] = r
        rank = (1 if r["valid_for_release"] else 0, r["gates_passed"], -r["char_cer_pct"] - r["clean_cer_pct"], -r["harum_crops_cer_pct"])
        if best_rank is None or rank > best_rank:
            best_rank, best_ep = rank, ep
        print(f"  -> V26.1 S{s_num} E{ep}: Clean={r['clean_cer_pct']}% Char={r['char_cer_pct']}% Sym={r['symbols_acc_pct']}% "
              f"(fail={r['failing_symbols']}) Hyp={r['hyphen_acc_pct']}% FP={r['fullpage_cer_pct']}% Stress={r['stress_82_cer_pct']}% "
              f"Probe={r['probe_100_w480_cer_pct']}% Harum={r['harum_crops_cer_pct']}% Gates={r['gates_passed']}/6 valid={r['valid_for_release']}")

    print("=" * 80 + "\n🥣 Running V26.1 in-basin checkpoint soups\n" + "=" * 80)
    soup_res = eval_v26_1_soups(epoch_results, policy, det_sig)

    wise_target = None
    wise_cfg = _stage_ckpt_info(2 if (best_ep or 8) > 8 else 1, best_ep or 8)[1]
    if soup_res and soup_res.get("candidates") and "soup_last3_e14_e15_e16" in soup_res["candidates"]:
        wise_target = soup_res["candidates"]["soup_last3_e14_e15_e16"]["ckpt_path"]
    elif soup_res and soup_res.get("best_key"):
        wise_target = soup_res["candidates"][soup_res["best_key"]]["ckpt_path"]
    elif best_ep is not None:
        wise_target = _stage_ckpt_info(1 if best_ep <= 8 else 2, best_ep)[0]
    wise_res = {"candidates": {}}
    s1_soup_ckpt = os.path.join(PROJECT_ROOT, "ocr-training", "output", "v26_1_soup_top3_e8_e6_e4_ckpt", "best.pdparams")
    if os.path.exists(s1_soup_ckpt):
        print("=" * 80 + "\n⚗️ Scoring Stage-1 in-basin WiSE-FT blends (cached fast-path)\n" + "=" * 80)
        w_s1 = eval_true_inbasin_wise_ft(s1_soup_ckpt, _stage_ckpt_info(1, 8)[1], policy, det_sig, prefix="wise_inbasin", alphas=(0.35, 0.50))
        wise_res["candidates"].update(w_s1["candidates"])
    if wise_target and any(int(k.split("_")[1]) > 8 for k in epoch_results):
        print("=" * 80 + "\n⚗️ Running Stage-2 in-basin WiSE-FT sweep (locked conv1..blocks3 + BN stats)\n" + "=" * 80)
        w_s2 = eval_true_inbasin_wise_ft(wise_target, wise_cfg, policy, det_sig, prefix="wise_s2", alphas=(0.35, 0.50, 0.65, 0.80))
        wise_res["candidates"].update(w_s2["candidates"])
    elif wise_target and not wise_res["candidates"]:
        print("=" * 80 + "\n⚗️ Running V26.1 true in-basin WiSE-FT sweep (locked conv1..blocks3 + BN stats)\n" + "=" * 80)
        wise_res = eval_true_inbasin_wise_ft(wise_target, wise_cfg, policy, det_sig, prefix="wise_inbasin")

    # Evaluate ALL 6/6-gate candidates (plus best_ep & best_soup) on RealScan-Gold v2!
    rs_models = {"prod_active": prod_snapshot}
    cand_registry = {}
    for ek, r in epoch_results.items():
        ep_num = int(ek.split("_")[1])
        if r.get("valid_for_release") or ep_num == best_ep:
            alias = f"epoch_{ep_num}"
            rs_models[alias] = r["infer_dir"]
            s_num = r["stage"]
            ckpt_p, cfg_p = _stage_ckpt_info(s_num, ep_num)
            cand_registry[alias] = {"metrics": r, "ckpt_path": ckpt_p, "cfg_path": cfg_p, "epoch": ep_num, "stage_str": f"V26_1_16EP_S{s_num}_E{ep_num}_W480", "kernel": "gustavnguyen/" + STAGES[s_num]["slug"]}
    if soup_res and soup_res.get("candidates"):
        for sk, sr in soup_res["candidates"].items():
            if sr.get("valid_for_release") or sk == soup_res.get("best_key"):
                rs_models[sk] = sr["infer_dir"]
                cand_registry[sk] = {"metrics": sr, "ckpt_path": sr["ckpt_path"], "cfg_path": wise_cfg, "epoch": 16, "stage_str": f"V26_1_16EP_{sk.upper()}_W480", "kernel": "gustavnguyen/" + STAGES[2]["slug"]}
    if wise_res and wise_res.get("candidates"):
        for wk, wr in wise_res["candidates"].items():
            if wr.get("valid_for_release"):
                rs_models[wk] = wr["infer_dir"]
                cand_registry[wk] = {"metrics": wr, "ckpt_path": wr["ckpt_path"], "cfg_path": wise_cfg, "epoch": 16, "stage_str": f"V26_1_16EP_{wk.upper()}_W480", "kernel": "gustavnguyen/" + STAGES[2]["slug"]}

    rs = realscan_v2_eval(rs_models)

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
    cand_scores = []
    for alias, info in cand_registry.items():
        if alias not in rs:
            continue
        rm = info["metrics"]
        rv = rs_view(alias)
        chk, ok = _check(rm, rv)
        decisions[alias] = {"realscan_v2": rv, "checks": chk, "qualifies": ok}
        if ok:
            score_sum = rm["char_cer_pct"] + rm["clean_cer_pct"] + 0.5 * rv["ood180_crop_cer_pct"] + 0.5 * rv["ood12_e2e_doc_cer_pct"]
            cand_scores.append((score_sum, alias))

    winner = min(cand_scores)[1] if cand_scores else None
    promoted = False
    cal_report = None
    backup_dir = os.path.join(PROJECT_ROOT, "ocr-training", "output", "rec_cham_inference_v24_1_r4_pre_v26_1_backup")

    if winner and not args.no_promote:
        winfo = cand_registry[winner]
        r = winfo["metrics"]
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
            "realscan_v2": decisions[winner]["realscan_v2"],
        }
        disp_name = f"Cham-OCR V26.1 Masked-SVTR ({winner}, W=480, {r['gates_passed']}/6 Gates)"
        subprocess.run([PY313, os.path.join(PROJECT_ROOT, "scripts", "package_v24_1_model.py"),
                        "--model-id", "v24_1_r4",
                        "--display-name", disp_name,
                        "--stage", winfo["stage_str"], "--epoch", str(winfo["epoch"]),
                        "--source-kernel", winfo["kernel"],
                        "--ckpt-path", winfo["ckpt_path"], "--config-path", winfo["cfg_path"],
                        "--rec-image-shape", "3, 48, 480",
                        "--metrics-json", json.dumps(metrics)], cwd=PROJECT_ROOT, check=True)
        # Re-calibrate post-hoc temperature T* on Held-Out Probe-100 and update model_manifest.json!
        cal_script = os.path.join(PROJECT_ROOT, "ocr-training", "scripts", "calibrate_confidence.py")
        if os.path.exists(cal_script):
            cal_cmd = [PY313, cal_script]
            for mid in ("v24_1_r2_1_6", "v24_1_r3", "v24_1_r3_1", "v24_1_r4"):
                mdir_rel = f"ocr-studio/data/output/rec_cham_inference_{mid}"
                if os.path.exists(os.path.join(PROJECT_ROOT, *mdir_rel.split("/"))):
                    cal_cmd += ["--model", f"{mid}={mdir_rel}::ocr-studio/data/cham_dict_v24_1.txt"]
            subprocess.run(cal_cmd, cwd=PROJECT_ROOT, check=True)
            cal_json_p = os.path.join(PROJECT_ROOT, "REPORTS", "confidence_calibration.json")
            if os.path.exists(cal_json_p):
                cal_models = json.load(open(cal_json_p, "r", encoding="utf-8")).get("models") or []
                cal_report = next((m for m in cal_models if m.get("model_id") == "v24_1_r4"), None)
        promoted = True

    if os.path.exists(prod_snapshot):
        shutil.rmtree(prod_snapshot, ignore_errors=True)

    new_sha16 = file_sha256(os.path.join(STUDIO_R4, "inference.pdiparams"))[:16]
    report = {
        "title": "Cham-OCR V26.1 Masked-SVTR (16 Epochs, 2 Stages, Frozen Early Backbone + Paleographic Aug) Evaluation",
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
        "confidence_calibration": cal_report,
        "production_after_sha256_16": new_sha16,
    }
    out_j = os.path.join(PROJECT_ROOT, "REPORTS", "v26_1_evaluation.json")
    with open(out_j, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md = [
        "# Đánh giá V26.1 Masked-SVTR (16 Epochs, 2 Chặng, Early-Backbone Frozen + Paleographic Augmentation)",
        f"- **Thời điểm**: `{report['evaluated_at']}`",
        f"- **Production trước**: `v24_1_r4` SHA256_16 `{prod_sha16}` (`det_sig={det_sig}`)",
        f"- **Tổng kế toán 2 chặng**: Train Loop = `{cum_acc['train_loop_duration_seconds']}s` (`{cum_acc['train_loop_duration_minutes']}m`), "
        f"Overhead = `{cum_acc['pipeline_container_overhead_seconds']}s`, Wall-clock = `{cum_acc['total_kernel_wall_clock_seconds']}s`",
        "",
        "## 1. Kế toán theo từng chặng",
        "| Chặng | Train Loop (s) | Overhead (s) | Wall-clock (s) |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for s_key, a in per_stage_acc.items():
        md.append(f"| {s_key} | {a.get('train_loop_duration_seconds')} | {a.get('pipeline_container_overhead_seconds')} | {a.get('total_kernel_wall_clock_seconds')} |")

    md += ["", "## 2. Kết quả Benchmark Frozen-838 & Bộ Phụ (`W=480`)",
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

    md += ["", "## 3. RealScan-Gold v2 (`9/9` Promotion Checks)",
           "| Model | harum crops | Wikimedia | OOD-180 crops | OOD-12 E2E Doc CER (dòng) | Qualifies |",
           "| :--- | :---: | :---: | :---: | :---: | :---: |",
           f"| Production trước | {prod_rs['harum_crop_cer_pct']}% | {prod_rs['wikimedia_crop_cer_pct']}% | {prod_rs['ood180_crop_cer_pct']}% | {prod_rs['ood12_e2e_doc_cer_pct']}% ({prod_rs['ood12_e2e_lines']}) | — |"]
    for dk, dv in decisions.items():
        v = dv["realscan_v2"]
        md.append(f"| V26.1 ({dk}) | {v['harum_crop_cer_pct']}% | {v['wikimedia_crop_cer_pct']}% | {v['ood180_crop_cer_pct']}% | {v['ood12_e2e_doc_cer_pct']}% ({v['ood12_e2e_lines']}) | `{dv['qualifies']}` |")

    md += ["", "## 4. Quyết định thăng cấp tự động"]
    for dk, dv in decisions.items():
        failed = [k for k, ok in dv["checks"].items() if not ok]
        md.append(f"- **{dk}**: qualifies = `{dv['qualifies']}`" + (f" (trượt: `{failed}`)" if failed else " (**PASS `9/9`**)"))
    md.append(f"- **Winner**: `{winner}`; **promoted**: `{promoted}`; production SHA256_16 sau: `{new_sha16}`")

    out_m = os.path.join(PROJECT_ROOT, "REPORTS", "V26_1_EVALUATION.md")
    with open(out_m, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print(f"✅ V26.1 evaluation -> {out_m} (winner={winner}, promoted={promoted})")


if __name__ == "__main__":
    main()
