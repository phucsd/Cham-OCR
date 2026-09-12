#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparative Report Generator for Cham-OCR Multi-GPU Benchmarks.
Aggregates trials from benchmark_*.json and formats the final Markdown comparison.
"""

import os
import sys
import glob
import json
import argparse

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

def generate_comparative_report(results_dir="benchmark_results", output_md="benchmark_results/COMPARATIVE_REPORT.md"):
    json_files = glob.glob(os.path.join(results_dir, "benchmark_*.json"))
    if not json_files:
        print(f"No benchmark JSON files found in {results_dir}")
        return

    summaries = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                summaries.append(data)
        except Exception as e:
            print(f"Error reading {jf}: {e}")

    lines = []
    lines.append("# Báo Cáo Đối Soánh Hiệu Năng & Chi Phí Huấn Luyện Cham-OCR Đa GPU")
    lines.append("## (PP-OCRv4 SVTR_LCNet - Pipeline Huấn Luyện Thật)\n")

    lines.append("### 1. Bảng Tổng Hợp So Sánh Các Cấu Hình Tối Ưu (Best & Max Batch)")
    lines.append("| GPU | Phân Loại | Batch | Workers | IPS (mẫu/s) | GPU% | VRAM% | Reader Time | Batch Time | $/h | Est 40 Epoch | Est Cost | IPS/$ |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    rankings = []

    for s in summaries:
        gpu = s["gpu_name"]
        cost = s["cost_per_hour"]
        workers = s["optimal_workers"]
        best_b = s["best_batch"]
        max_b = s["max_safe_batch"]
        trials = { (t["batch_size"], t["num_workers"]): t for t in s["all_trials"] if t.get("status") == "SUCCESS" }

        # Best Trial
        best_t = trials.get((best_b, workers))
        if best_t:
            lines.append(
                f"| **{gpu}** | 🌟 **Best** | {best_b} | {workers} | "
                f"**{best_t['ips']:.1f}** | {best_t['avg_gpu_util']:.0f}% | {best_t['vram_pct']:.1f}% | "
                f"{best_t['avg_reader_cost']:.4f}s | {best_t['avg_batch_cost']:.4f}s | "
                f"${cost:.2f} | {best_t['est_40_epoch_hours']:.1f}h | "
                f"**${best_t['est_40_epoch_cost']:.2f}** | **{best_t['ips_per_dollar']:.1f}** |"
            )
            rankings.append({
                "gpu": gpu,
                "type": "Best",
                "batch": best_b,
                "workers": workers,
                "ips": best_t["ips"],
                "cost": cost,
                "est_hours": best_t["est_40_epoch_hours"],
                "est_cost": best_t["est_40_epoch_cost"],
                "ips_per_dollar": best_t["ips_per_dollar"],
                "bottleneck": s["primary_bottleneck"]
            })

        # Max Safe Trial (if different from Best)
        if max_b != best_b:
            max_t = trials.get((max_b, workers))
            if max_t:
                lines.append(
                    f"| {gpu} | 🛡️ Max Safe | {max_b} | {workers} | "
                    f"{max_t['ips']:.1f} | {max_t['avg_gpu_util']:.0f}% | {max_t['vram_pct']:.1f}% | "
                    f"{max_t['avg_reader_cost']:.4f}s | {max_t['avg_batch_cost']:.4f}s | "
                    f"${cost:.2f} | {max_t['est_40_epoch_hours']:.1f}h | "
                    f"${max_t['est_40_epoch_cost']:.2f} | {max_t['ips_per_dollar']:.1f} |"
                )

    lines.append("\n### 2. Kết Luận Chi Tiết Cho Từng Machine")
    for s in summaries:
        gpu = s["gpu_name"]
        cost = s["cost_per_hour"]
        workers = s["optimal_workers"]
        best_b = s["best_batch"]
        max_b = s["max_safe_batch"]
        bottleneck = s["primary_bottleneck"]
        best_t = s.get("best_trial")

        lines.append(f"\n#### 🔹 Machine: **{gpu}** (${cost:.2f}/h)")
        lines.append(f"- **Max Batch Chạy Được**: `{max_b}` (ngưỡng giới hạn phần cứng trước khi OOM).")
        lines.append(f"- **Best Batch Nên Dùng**: `{best_b}` (điểm ngọt cho throughput cao nhất trước khi bão hòa).")
        lines.append(f"- **Num Workers Tối Ưu**: `{workers}` worker.")
        lines.append(f"- **Điểm Nghẽn Chính (Bottleneck)**: `{bottleneck}`.")
        if best_t:
            lines.append(f"- **Thời Gian Huấn Luyện Full 40 Epoch (10,000,000 mẫu)**: `{best_t['est_40_epoch_hours']:.2f} giờ` (~`{best_t['est_40_epoch_hours']*60:.0f} phút`).")
            lines.append(f"- **Chi Phí Ước Tính Full 40 Epoch**: `${best_t['est_40_epoch_cost']:.2f}`.")
            lines.append(f"- **Hiệu Năng / Chi Phí (P/P)**: `{best_t['ips_per_dollar']:.1f} mẫu / $`.")

    # Sort rankings by IPS/$ descending
    rankings.sort(key=lambda x: x["ips_per_dollar"], reverse=True)
    lines.append("\n### 3. Xếp Hạng Hiệu Quả Đầu Tư (Performance / Price Ranking)")
    lines.append("| Hạng | GPU | Best Batch | Tốc Độ (IPS) | Thời Gian Full 40 Epoch | Tổng Chi Phí ($) | Hiệu Suất Kinh Tế (IPS/$) |")
    lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|")

    for i, r in enumerate(rankings, 1):
        medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else f"#{i}"))
        lines.append(f"| {medal} | **{r['gpu']}** | {r['batch']} | {r['ips']:.1f} mẫu/s | {r['est_hours']:.1f}h | **${r['est_cost']:.2f}** | **{r['ips_per_dollar']:.1f}** |")

    if rankings:
        top = rankings[0]
        lines.append(f"\n> 🏆 **LỰA CHỌN TỐI ƯU NHẤT (P/P)**: **{top['gpu']}** là cỗ máy kinh tế nhất với **{top['ips_per_dollar']:.1f} mẫu / $**, tổng chi phí train full 40 epoch chỉ **${top['est_cost']:.2f}** trong **{top['est_hours']:.1f} giờ**.")

    report_content = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(output_md)), exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n✅ Comparative report written to: {output_md}")
    print("\n" + report_content)
    return report_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Comparative Benchmark Report")
    parser.add_argument("-d", "--results_dir", type=str, default="ocr-training/benchmark_results", help="Directory with benchmark JSONs")
    parser.add_argument("-o", "--output_md", type=str, default="ocr-training/benchmark_results/COMPARATIVE_REPORT.md", help="Output Markdown report path")
    args = parser.parse_args()

    generate_comparative_report(args.results_dir, args.output_md)
