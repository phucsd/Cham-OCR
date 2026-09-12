#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Multi-GPU Benchmark & Profiling Orchestrator.
Discovers optimal num_workers, sweeps batch sizes, performs binary search on OOM,
and computes 40-epoch economic projections (Time, Cost, IPS/$).
"""

import os
import sys
import time
import json
import argparse
import subprocess
import shutil

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PRICING_TABLE = {
    "A100": 2.19,
    "A100-SXM4-40GB": 2.19,
    "NVIDIA A100": 2.19,
    "L4": 0.79,
    "NVIDIA L4": 0.79,
    "L40S": 2.14,
    "NVIDIA L40S": 2.14,
    "T4": 0.40,
    "NVIDIA T4": 0.40,
}

TOTAL_DATASET_SAMPLES = 250000
TOTAL_EPOCHS = 40
TOTAL_TRAIN_SAMPLES = TOTAL_DATASET_SAMPLES * TOTAL_EPOCHS  # 10,000,000


def detect_gpu_name():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=2.0
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "Unknown_GPU"


def round_to_multiple_of_32(val):
    return max(32, (int(val) // 32) * 32)


def round_to_multiple_of_16(val):
    return max(16, (int(val) // 16) * 16)


def run_single_trial(python_bin, worker_script, config_path, batch_size, num_workers, warmup_steps, measure_steps, output_dir):
    trial_json = os.path.join(output_dir, f"trial_b{batch_size}_w{num_workers}.json")
    if os.path.exists(trial_json):
        try:
            os.remove(trial_json)
        except Exception:
            pass

    cmd = [
        python_bin,
        worker_script,
        "-c", config_path,
        "-b", str(batch_size),
        "-w", str(num_workers),
        "--warmup_steps", str(warmup_steps),
        "--measure_steps", str(measure_steps),
        "-o", trial_json
    ]

    print(f"\n>>> [Trial] Launching subprocess: Batch={batch_size}, Workers={num_workers} (Warmup={warmup_steps}, Measure={measure_steps})", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0

    if os.path.exists(trial_json):
        try:
            with open(trial_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data["elapsed_wall_time"] = round(elapsed, 2)
                return data
        except Exception as e:
            print(f"Failed to read trial json: {e}")

    # Fallback if trial json was not written
    status = "OOM" if res.returncode in (137, -9) else "CRASH"
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "status": status,
        "ips": 0.0,
        "avg_reader_cost": 0.0,
        "avg_batch_cost": 0.0,
        "reader_ratio": 0.0,
        "peak_vram_mb": 0.0,
        "total_vram_mb": 0.0,
        "vram_pct": 0.0,
        "avg_gpu_util": 0.0,
        "avg_cpu_pct": 0.0,
        "error": f"Process exited with code {res.returncode}",
        "elapsed_wall_time": round(elapsed, 2)
    }


def compute_economics(trial, cost_per_hour):
    ips = trial.get("ips", 0.0)
    if ips > 0:
        est_hours = TOTAL_TRAIN_SAMPLES / (ips * 3600.0)
        est_cost = est_hours * cost_per_hour
        ips_per_dollar = ips / cost_per_hour
    else:
        est_hours = 0.0
        est_cost = 0.0
        ips_per_dollar = 0.0

    trial["est_40_epoch_hours"] = round(est_hours, 2)
    trial["est_40_epoch_cost"] = round(est_cost, 2)
    trial["ips_per_dollar"] = round(ips_per_dollar, 2)
    trial["cost_per_hour"] = cost_per_hour
    return trial


def run_benchmark_suite(config_path, gpu_name=None, cost_per_hour=None, python_bin=None, warmup_steps=100, measure_steps=300, output_dir="benchmark_results"):
    os.makedirs(output_dir, exist_ok=True)
    if not python_bin:
        python_bin = sys.executable

    script_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(script_dir, "benchmark_worker.py")
    if not os.path.exists(worker_script):
        raise FileNotFoundError(f"Worker script not found: {worker_script}")

    detected_gpu = detect_gpu_name()
    if not gpu_name:
        gpu_name = detected_gpu

    if cost_per_hour is None:
        cost_per_hour = 2.19
        for k, v in PRICING_TABLE.items():
            if k.lower() in gpu_name.lower():
                cost_per_hour = v
                break

    cpu_count = os.cpu_count() or 8
    print("=" * 80)
    print(f"🚀 PADDLEOCR AUTOMATED HARDWARE BENCHMARK SUITE")
    print(f"GPU: {gpu_name} (Detected: {detected_gpu}) | Cost: ${cost_per_hour:.2f}/h")
    print(f"CPU Cores: {cpu_count} | Python: {python_bin}")
    print(f"Config: {config_path}")
    print(f"Warmup: {warmup_steps} steps | Measure: {measure_steps} steps")
    print("=" * 80, flush=True)

    all_trials = []

    # ---------------------------------------------------------
    # PHASE 1: FIND OPTIMAL NUM_WORKERS AT BASE BATCH
    # ---------------------------------------------------------
    base_batch = 128
    initial_oom_batch = None

    print(f"\nTesting initial base batch {base_batch} with 4 workers...")
    test_init = run_single_trial(
        python_bin, worker_script, config_path,
        batch_size=base_batch, num_workers=4,
        warmup_steps=warmup_steps, measure_steps=measure_steps,
        output_dir=output_dir
    )
    test_init = compute_economics(test_init, cost_per_hour)
    all_trials.append(test_init)

    if test_init["status"] == "OOM":
        initial_oom_batch = base_batch
        print(f"⚠️ Base batch {base_batch} encountered OOM on {gpu_name}! Stepping down to locate viable batch...")
        found_base = False
        for fallback_b in [96, 64, 48, 32]:
            print(f"Testing fallback batch {fallback_b}...")
            t_fb = run_single_trial(
                python_bin, worker_script, config_path,
                batch_size=fallback_b, num_workers=4,
                warmup_steps=warmup_steps, measure_steps=measure_steps,
                output_dir=output_dir
            )
            t_fb = compute_economics(t_fb, cost_per_hour)
            all_trials.append(t_fb)
            if t_fb["status"] == "SUCCESS":
                base_batch = fallback_b
                found_base = True
                print(f"✅ Viable base batch found: {base_batch} (VRAM: {t_fb['vram_pct']:.1f}%)")
                break
        if not found_base:
            raise RuntimeError(f"Unable to find viable batch size even at batch 32 on {gpu_name}")

    print("\n" + "=" * 60)
    print(f"📍 [PHASE 1/3] SWEEPING NUM_WORKERS AT BATCH SIZE {base_batch}")
    print("=" * 60, flush=True)

    candidate_workers = [4, 8, 12, 16, 24]
    max_reasonable_workers = min(24, max(8, cpu_count * 2))
    candidate_workers = [w for w in candidate_workers if w <= max_reasonable_workers]
    if 4 not in candidate_workers:
        candidate_workers.insert(0, 4)

    best_worker = 4
    best_worker_ips = 0.0
    worker_trials = []

    for w in candidate_workers:
        # Check if already tested
        existing = [t for t in all_trials if t["batch_size"] == base_batch and t["num_workers"] == w]
        if existing:
            trial = existing[0]
        else:
            trial = run_single_trial(
                python_bin, worker_script, config_path,
                batch_size=base_batch, num_workers=w,
                warmup_steps=warmup_steps, measure_steps=measure_steps,
                output_dir=output_dir
            )
            trial = compute_economics(trial, cost_per_hour)
            all_trials.append(trial)

        worker_trials.append(trial)

        if trial["status"] == "SUCCESS":
            print(f"  -> Workers={w:2d} | IPS={trial['ips']:6.1f} | Reader={trial['avg_reader_cost']:.4f}s | Batch={trial['avg_batch_cost']:.4f}s | ReaderRatio={trial['reader_ratio']:.2f}")
            if trial["ips"] > best_worker_ips:
                best_worker_ips = trial["ips"]
                best_worker = w
        else:
            print(f"  -> Workers={w:2d} | FAILED ({trial['status']}): {trial.get('error')}")

    print(f"\n✅ Optimal num_workers selected: {best_worker} (IPS: {best_worker_ips:.1f} samples/s)")

    # ---------------------------------------------------------
    # PHASE 2: BATCH SIZE SCALING
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"📍 [PHASE 2/3] BATCH SIZE PROGRESSION (WORKERS = {best_worker})")
    print("=" * 60, flush=True)

    if base_batch < 128:
        # Fine-grained progression for smaller GPUs
        batch_progression = [base_batch, 80, 96, 112, 128, 144, 160, 192, 224, 256]
    else:
        batch_progression = [128, 160, 192, 224, 256, 320, 384, 512, 768, 1024]

    # Filter only batches >= base_batch
    batch_progression = sorted(list(set([b for b in batch_progression if b >= base_batch])))

    batch_trials = []
    last_safe_batch = base_batch
    last_safe_trial = next((t for t in worker_trials if t["batch_size"] == base_batch and t["num_workers"] == best_worker and t["status"] == "SUCCESS"), None)
    oom_batch = initial_oom_batch
    best_batch = base_batch
    best_batch_ips = best_worker_ips
    diminishing_returns_detected = False

    for b in batch_progression:
        # If batch 128 was already tested in Phase 1 with the same worker, reuse result
        existing = [t for t in worker_trials if t["batch_size"] == b and t["num_workers"] == best_worker]
        if existing:
            trial = existing[0]
        else:
            trial = run_single_trial(
                python_bin, worker_script, config_path,
                batch_size=b, num_workers=best_worker,
                warmup_steps=warmup_steps, measure_steps=measure_steps,
                output_dir=output_dir
            )
            trial = compute_economics(trial, cost_per_hour)
            all_trials.append(trial)

        batch_trials.append(trial)

        if trial["status"] == "SUCCESS":
            cur_ips = trial["ips"]
            vram_pct = trial["vram_pct"]
            reader_ratio = trial["reader_ratio"]

            # Calculate improvement over previous successful batch
            delta_ips_pct = 0.0
            if last_safe_trial and last_safe_trial["ips"] > 0:
                delta_ips_pct = ((cur_ips - last_safe_trial["ips"]) / last_safe_trial["ips"]) * 100

            print(f"  -> Batch={b:4d} | IPS={cur_ips:6.1f} ({delta_ips_pct:+5.1f}%) | VRAM={trial['peak_vram_mb']:5.0f}MB ({vram_pct:4.1f}%) | ReaderRatio={reader_ratio:.2f} | Est: {trial['est_40_epoch_hours']:.1f}h (${trial['est_40_epoch_cost']:.2f})")

            # Check for best throughput batch
            if cur_ips > best_batch_ips:
                best_batch_ips = cur_ips
                best_batch = b

            # Check diminishing returns: if gain < 3% compared to previous batch
            if last_safe_trial and delta_ips_pct < 3.0:
                print(f"  ⚠️ Diminishing returns detected at Batch={b} (Throughput gain {delta_ips_pct:.2f}% < 3.0%). Sweet spot identified.")
                diminishing_returns_detected = True

            last_safe_batch = b
            last_safe_trial = trial

            # Check if VRAM exceeds 92% (danger zone)
            if vram_pct >= 92.0:
                print(f"  ⚠️ VRAM usage reached {vram_pct:.1f}% (>= 92%). Halting batch scaling to avoid sudden hard OOM.")
                break

            # If diminishing returns detected and VRAM is already high, stop scaling further
            if diminishing_returns_detected and vram_pct >= 85.0:
                print(f"  🏁 Stopping batch progression: Throughput saturated and VRAM comfortably utilized.")
                break

        elif trial["status"] == "OOM":
            print(f"  💥 OOM encountered at Batch={b}!")
            oom_batch = b
            break
        else:
            print(f"  ❌ Trial failed at Batch={b} with status {trial['status']}: {trial.get('error')}")
            break

    # ---------------------------------------------------------
    # PHASE 3: BINARY SEARCH FOR EXACT MAX SAFE BATCH
    # ---------------------------------------------------------
    max_safe_batch = last_safe_batch
    if oom_batch is not None and last_safe_batch is not None:
        print("\n" + "=" * 60)
        print(f"📍 [PHASE 3/3] BINARY SEARCH BETWEEN {last_safe_batch} (SAFE) AND {oom_batch} (OOM)")
        print("=" * 60, flush=True)

        low = last_safe_batch
        high = oom_batch
        while (high - low) > 16:
            mid = round_to_multiple_of_16((low + high) / 2)
            if mid == low or mid == high:
                break

            print(f"\n[Binary Search] Testing midpoint Batch={mid} (range: [{low}, {high}])...")
            trial = run_single_trial(
                python_bin, worker_script, config_path,
                batch_size=mid, num_workers=best_worker,
                warmup_steps=warmup_steps, measure_steps=measure_steps,
                output_dir=output_dir
            )
            trial = compute_economics(trial, cost_per_hour)
            all_trials.append(trial)

            if trial["status"] == "SUCCESS":
                print(f"  -> Batch={mid} succeeded! IPS={trial['ips']:.1f}, VRAM={trial['vram_pct']:.1f}%")
                low = mid
                if trial["ips"] > best_batch_ips and not diminishing_returns_detected:
                    best_batch_ips = trial["ips"]
                    best_batch = mid
            else:
                print(f"  -> Batch={mid} failed with {trial['status']}.")
                high = mid

        max_safe_batch = low
        print(f"\n🎯 Binary Search Complete: Max Safe Batch = {max_safe_batch}")

    # ---------------------------------------------------------
    # PHASE 4: BOTTLENECK ANALYSIS & REPORT GENERATION
    # ---------------------------------------------------------
    best_trial = next((t for t in all_trials if t["batch_size"] == best_batch and t["num_workers"] == best_worker and t["status"] == "SUCCESS"), last_safe_trial)
    
    # Identify primary bottleneck
    bottleneck = "Balanced"
    if best_trial:
        if best_trial["vram_pct"] >= 90.0:
            bottleneck = "VRAM Capacity Limited"
        elif best_trial["reader_ratio"] >= 0.40:
            bottleneck = "DataLoader / CPU Bound"
        elif best_trial["avg_gpu_util"] >= 80.0 and best_trial["reader_ratio"] < 0.20:
            bottleneck = "GPU Compute Saturated (Optimal)"
        elif best_trial["avg_cpu_pct"] >= 85.0:
            bottleneck = "Host CPU Bound"
        else:
            bottleneck = "Memory Bandwidth / Pipeline Latency"

    summary = {
        "gpu_name": gpu_name,
        "cost_per_hour": cost_per_hour,
        "optimal_workers": best_worker,
        "best_batch": best_batch,
        "max_safe_batch": max_safe_batch,
        "best_ips": best_trial["ips"] if best_trial else 0.0,
        "primary_bottleneck": bottleneck,
        "best_trial": best_trial,
        "all_trials": all_trials
    }

    # Save JSON summary
    summary_file = os.path.join(output_dir, f"benchmark_{gpu_name.replace(' ', '_')}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Generate Markdown Report
    md_report = generate_markdown_report(summary)
    md_file = os.path.join(output_dir, f"benchmark_{gpu_name.replace(' ', '_')}.md")
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_report)

    print("\n" + "=" * 80)
    print("📊 BENCHMARK REPORT COMPLETED")
    print(f"JSON Output: {summary_file}")
    print(f"MD Output:   {md_file}")
    print("=" * 80)
    print(md_report)

    return summary


def generate_markdown_report(summary):
    gpu = summary["gpu_name"]
    cost = summary["cost_per_hour"]
    best_b = summary["best_batch"]
    max_b = summary["max_safe_batch"]
    workers = summary["optimal_workers"]
    bottleneck = summary["primary_bottleneck"]
    trials = summary["all_trials"]

    lines = []
    lines.append(f"# Benchmark & Hardware Limit Analysis: {gpu}")
    lines.append(f"**Cluster / GPU**: `{gpu}` | **Pricing**: `${cost:.2f}/h` | **Primary Bottleneck**: `{bottleneck}`\n")

    lines.append("## 1. Kết Quả Đo Lường Chi Tiết (Full Measurement Trials)")
    lines.append("| Batch | Workers | Status | IPS (mẫu/s) | GPU Util % | VRAM (MB) | VRAM % | Reader Cost | Batch Cost | Reader Ratio | Est 40 Epoch | Est Cost | IPS/$ |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for t in trials:
        b = t["batch_size"]
        w = t["num_workers"]
        st = t["status"]
        if st == "SUCCESS":
            ips = f"{t['ips']:.1f}"
            gpu_u = f"{t['avg_gpu_util']:.0f}%"
            vram_mb = f"{t['peak_vram_mb']:.0f}"
            vram_p = f"{t['vram_pct']:.1f}%"
            r_cost = f"{t['avg_reader_cost']:.4f}s"
            b_cost = f"{t['avg_batch_cost']:.4f}s"
            r_ratio = f"{t['reader_ratio']:.2f}"
            est_h = f"{t['est_40_epoch_hours']:.2f}h"
            est_c = f"${t['est_40_epoch_cost']:.2f}"
            ips_d = f"{t['ips_per_dollar']:.1f}"
        else:
            ips = "-"
            gpu_u = "-"
            vram_mb = "-"
            vram_p = "-"
            r_cost = "-"
            b_cost = "-"
            r_ratio = "-"
            est_h = "-"
            est_c = "-"
            ips_d = "-"

        # Highlight best and max
        tag = ""
        if b == best_b and w == workers and st == "SUCCESS":
            tag = " 🌟 *(Best)*"
        elif b == max_b and w == workers and st == "SUCCESS":
            tag = " 🛡️ *(Max Safe)*"
        elif st == "OOM":
            tag = " 💥 *(OOM)*"

        lines.append(f"| {b}{tag} | {w} | {st} | {ips} | {gpu_u} | {vram_mb} | {vram_p} | {r_cost} | {b_cost} | {r_ratio} | {est_h} | {est_c} | {ips_d} |")

    lines.append("\n## 2. Kết Luận & Đề Xuất Cấu Hình")
    best_t = summary.get("best_trial")
    if best_t:
        lines.append(f"- **Max Batch Chạy Được (Hardware Limit)**: `{max_b}` (ngưỡng tối đa trước khi tràn VRAM).")
        lines.append(f"- **Best Batch Nên Dùng (Sweet Spot)**: `{best_b}` (đạt throughput `{best_t['ips']:.1f}` mẫu/s, khai thác `{best_t['vram_pct']:.1f}%` VRAM).")
        lines.append(f"- **Workers Tối Ưu**: `{workers}` (giữ `reader_ratio = {best_t['reader_ratio']:.2f}`, đảm bảo GPU không bị đói dữ liệu).")
        lines.append(f"- **Điểm Nghẽn Chính (Bottleneck)**: `{bottleneck}`.")
        lines.append(f"- **Thời Gian Huấn Luyện Full 40 Epoch (10M samples)**: `{best_t['est_40_epoch_hours']:.2f} giờ` (~`{best_t['est_40_epoch_hours']*60:.0f} phút`).")
        lines.append(f"- **Chi Phí Ước Tính Full 40 Epoch**: `${best_t['est_40_epoch_cost']:.2f}`.")
        lines.append(f"- **Hiệu Suất Kinh Tế (Throughput / $)**: `{best_t['ips_per_dollar']:.1f} samples / $`.")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated PaddleOCR Benchmark Suite")
    parser.add_argument("-c", "--config_path", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--gpu_name", type=str, default=None, help="GPU identifier name")
    parser.add_argument("--cost_per_hour", type=float, default=None, help="Hourly cost in USD")
    parser.add_argument("--python_bin", type=str, default=None, help="Path to Python interpreter")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--measure_steps", type=int, default=300, help="Measure steps")
    parser.add_argument("-o", "--output_dir", type=str, default="benchmark_results", help="Directory to save results")
    args = parser.parse_args()

    run_benchmark_suite(
        config_path=args.config_path,
        gpu_name=args.gpu_name,
        cost_per_hour=args.cost_per_hour,
        python_bin=args.python_bin,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        output_dir=args.output_dir
    )
