import os
import sys
import time
import json
import re
import math
import statistics
import yaml
import subprocess
import shutil

# Ma trận thử nghiệm tập trung khảo sát điểm ngọt hiệu năng
TRIALS = [
    # Nhóm 1: Mốc chuẩn tham chiếu (Baseline)
    {"name": "T1 (Baseline FP32 BS64)", "bs": 64, "workers": 2, "amp": False, "amp_level": "O1", "nccl_tune": False, "epochs": 1, "desc": "Mốc chuẩn đối sánh FP32 (BS 64/card = 128 Global, Workers 2)"},
    
    # Nhóm 2: Đối sánh 1:1 trực tiếp với Baseline tại cùng BS 64
    {"name": "T2 (Tensor Cores BS64)", "bs": 64, "workers": 2, "amp": True, "amp_level": "O1", "nccl_tune": False, "epochs": 1, "desc": "Kích hoạt 640 Tensor Cores (BS 64/card = 128, AMP O1)"},
    
    # Nhóm 3: Quét tìm điểm ngọt Batch Size với AMP O1
    {"name": "T3 (AMP BS48)", "bs": 48, "workers": 2, "amp": True, "amp_level": "O1", "nccl_tune": False, "epochs": 1, "desc": "Đo độ trễ batch nhỏ hơn (BS 48/card = 96, AMP O1)"},
    {"name": "T4 (AMP BS72)", "bs": 72, "workers": 2, "amp": True, "amp_level": "O1", "nccl_tune": False, "epochs": 1, "desc": "Thăm dò tăng 12.5% batch (BS 72/card = 144, AMP O1)"},
    {"name": "T5 (AMP BS80)", "bs": 80, "workers": 2, "amp": True, "amp_level": "O1", "nccl_tune": False, "epochs": 1, "desc": "Thăm dò tăng 25% batch (BS 80/card = 160, AMP O1)"},
    {"name": "T6 (AMP BS96)", "bs": 96, "workers": 2, "amp": True, "amp_level": "O1", "nccl_tune": False, "epochs": 1, "desc": "Kiểm tra giới hạn trên VRAM (BS 96/card = 192, AMP O1)"},
    
    # Nhóm 4: Tối ưu hóa DDP Communication & CPU Workers
    {"name": "T7 (NCCL Tuned BS64)", "bs": 64, "workers": 2, "amp": True, "amp_level": "O1", "nccl_tune": True, "epochs": 1, "desc": "Tối ưu hóa bus PCIe (NCCL_BUFFSIZE=16M, NCCL_P2P_DISABLE=0)"},
    {"name": "T8 (Workers 3 BS64)", "bs": 64, "workers": 3, "amp": True, "amp_level": "O1", "nccl_tune": False, "epochs": 1, "desc": "Tận dụng 4 vCPU với 3 workers/card (Tổng 6 workers)"},
]

WARMUP_STEPS = 15  # Loại bỏ 15 bước đầu tiên để loại trừ nhiễu khởi tạo CUDA / Allocator / NCCL

def get_vram_usage():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0
        )
        if res.returncode == 0 and res.stdout.strip():
            lines = res.stdout.strip().split("\n")
            vrams = []
            for l in lines:
                parts = [float(x.strip()) for x in l.split(",")]
                vrams.append({"used_mb": parts[0], "total_mb": parts[1], "pct": round(parts[0]/parts[1]*100, 1)})
            return vrams
    except Exception:
        pass
    return []

def run_trial(trial_cfg, base_config_path, data_dir, dict_path, log_dir):
    name = trial_cfg["name"]
    bs = trial_cfg["bs"]
    workers = trial_cfg["workers"]
    amp = trial_cfg["amp"]
    amp_level = trial_cfg["amp_level"]
    epochs = trial_cfg["epochs"]
    nccl_tune = trial_cfg.get("nccl_tune", False)
    desc = trial_cfg.get("desc", "")
    
    print("\n" + "="*90)
    print(f"🔥 BẮT ĐẦU: {name}")
    print(f"   Tham số: BS/card={bs} (Global BS={bs*2}) | Workers={workers} | AMP={amp} ({amp_level}) | NCCL_Tune={nccl_tune}")
    print(f"   Mục tiêu: {desc}")
    print("="*90, flush=True)
    
    out_temp = os.path.join(log_dir, f"run_{name.replace(' ', '_').replace('(', '').replace(')', '').replace(':', '')}")
    os.makedirs(out_temp, exist_ok=True)
    log_file = os.path.join(out_temp, "train.log")
    
    # 1. Tạo tệp cấu hình YAML riêng
    with open(base_config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    cfg['Global']['save_model_dir'] = out_temp
    cfg['Global']['save_res_path'] = os.path.join(out_temp, "predicts.txt")
    cfg['Global']['character_dict_path'] = dict_path
    cfg['Global']['checkpoints'] = None
    cfg['Global']['pretrained_model'] = None
    cfg['Global']['epoch_num'] = epochs
    cfg['Global']['print_batch_step'] = 5  # Báo cáo mỗi 5 bước để thu thập nhiều điểm dữ liệu ổn định
    cfg['Global']['save_epoch_step'] = 999
    cfg['Global']['eval_batch_step'] = [0, 99999]
    cfg['Global']['cal_metric_during_train'] = False
    cfg['Global']['use_visualdl'] = False
    cfg['Global']['use_amp'] = amp
    cfg['Global']['amp_level'] = amp_level
    
    cfg['Train']['dataset']['data_dir'] = data_dir
    cfg['Train']['dataset']['label_file_list'] = [os.path.join(data_dir, "train_label.txt")]
    cfg['Train']['loader']['batch_size_per_card'] = bs
    cfg['Train']['loader']['num_workers'] = workers
    
    if 'Eval' in cfg and cfg['Eval'] and 'dataset' in cfg['Eval']:
        cfg['Eval']['dataset']['data_dir'] = data_dir
        cfg['Eval']['dataset']['label_file_list'] = [os.path.join(data_dir, "val_label.txt")]
        cfg['Eval']['loader']['batch_size_per_card'] = min(64, bs)
        cfg['Eval']['loader']['num_workers'] = min(2, workers)
    
    trial_yaml_path = os.path.join(out_temp, "config_trial.yml")
    with open(trial_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f)
        
    cmd = [
        "python3", "-m", "paddle.distributed.launch", "--gpus", "0,1",
        "tools/train.py",
        "-c", trial_yaml_path
    ]
    
    # Thiết lập biến môi trường NCCL nếu có cờ nccl_tune
    proc_env = os.environ.copy()
    proc_env["PYTHONUTF8"] = "1"
    if nccl_tune:
        proc_env["NCCL_BUFFSIZE"] = "16777216"
        proc_env["NCCL_P2P_DISABLE"] = "0"
    
    t_start = time.time()
    try:
        with open(log_file, "w", encoding="utf-8") as f_out:
            proc = subprocess.Popen(
                cmd,
                cwd="/kaggle/working/PaddleOCR",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=proc_env
            )
            
            warmup_ips = []
            steady_ips = []
            steady_rc = []
            steady_bc = []
            steady_loss = []
            steady_ctc = []
            steady_nrtr = []
            steady_mem_alloc = []
            steady_mem_res = []
            recent_lines = []
            
            for line in proc.stdout:
                f_out.write(line)
                f_out.flush()
                recent_lines.append(line)
                if len(recent_lines) > 25:
                    recent_lines.pop(0)
                    
                # Bắt log chi tiết từ PaddleOCR
                if "ips:" in line and "avg_batch_cost:" in line:
                    m_step = re.search(r"global_step:\s*(\d+)", line)
                    m_ips = re.search(r"ips:\s*([0-9.]+)\s*samples/s", line)
                    m_rc = re.search(r"avg_reader_cost:\s*([0-9.]+)\s*s", line)
                    m_bc = re.search(r"avg_batch_cost:\s*([0-9.]+)\s*s", line)
                    m_loss = re.search(r"loss:\s*([0-9.]+)", line)
                    m_ctc = re.search(r"CTCLoss:\s*([0-9.]+)", line)
                    m_nrtr = re.search(r"NRTRLoss:\s*([0-9.]+)", line)
                    m_mem_res = re.search(r"max_mem_reserved:\s*(\d+)\s*MB", line)
                    m_mem_alloc = re.search(r"max_mem_allocated:\s*(\d+)\s*MB", line)
                    
                    cur_step = int(m_step.group(1)) if m_step else 0
                    cur_ips = (float(m_ips.group(1)) * 2.0) if m_ips else 0.0  # Dual GPU = 2 * per-card
                    cur_rc = float(m_rc.group(1)) if m_rc else 0.0
                    cur_bc = float(m_bc.group(1)) if m_bc else 0.0
                    cur_loss = float(m_loss.group(1)) if m_loss else 0.0
                    cur_ctc = float(m_ctc.group(1)) if m_ctc else 0.0
                    cur_nrtr = float(m_nrtr.group(1)) if m_nrtr else 0.0
                    cur_alloc = int(m_mem_alloc.group(1)) if m_mem_alloc else 0
                    cur_res = int(m_mem_res.group(1)) if m_mem_res else 0
                    
                    if cur_step <= WARMUP_STEPS:
                        warmup_ips.append(cur_ips)
                        sys.stdout.write(f"  [Warmup Step {cur_step:02d}] Dual IPS: {cur_ips:5.1f} s/s | Cost: {cur_bc:.3f}s | VRAM: {cur_alloc} MB\n")
                    else:
                        steady_ips.append(cur_ips)
                        steady_rc.append(cur_rc)
                        steady_bc.append(cur_bc)
                        steady_loss.append(cur_loss)
                        steady_ctc.append(cur_ctc)
                        steady_nrtr.append(cur_nrtr)
                        if cur_alloc > 0: steady_mem_alloc.append(cur_alloc)
                        if cur_res > 0: steady_mem_res.append(cur_res)
                        sys.stdout.write(f"  ⚡ [Steady Step {cur_step:02d}] Dual IPS: {cur_ips:5.1f} s/s | Cost: {cur_bc:.3f}s (Reader: {cur_rc:.4f}s) | Loss: {cur_loss:6.2f} | VRAM: {cur_alloc} MB\n")
                    sys.stdout.flush()
            
            proc.wait()
            elapsed = time.time() - t_start
            vram_stats = get_vram_usage()
            
            if proc.returncode != 0:
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().lower()
                if "out of memory" in content or "bad_alloc" in content or "cudabadalloc" in content:
                    status = "OOM"
                    err_msg = "CUDA Out Of Memory"
                else:
                    status = "ERROR"
                    err_msg = f"Exit code {proc.returncode}"
                print(f"❌ {name} THẤT BẠI: {status} ({err_msg})")
                print("--- Ghi chú lỗi gần nhất ---")
                for r_l in recent_lines[-8:]:
                    print("  " + r_l.strip())
                print("----------------------------")
                return {
                    "name": name,
                    "batch_size_per_card": bs,
                    "global_batch_size": bs * 2,
                    "num_workers": workers,
                    "amp": amp,
                    "amp_level": amp_level,
                    "nccl_tune": nccl_tune,
                    "status": status,
                    "median_ips": 0.0,
                    "mean_ips": 0.0,
                    "std_ips": 0.0,
                    "cov_pct": 0.0,
                    "avg_batch_cost": 0.0,
                    "avg_reader_cost": 0.0,
                    "compute_cost": 0.0,
                    "reader_ratio": 0.0,
                    "peak_alloc_mb": 0,
                    "peak_res_mb": 0,
                    "loss_initial": 0.0,
                    "loss_final": 0.0,
                    "loss_status": "N/A",
                    "vram": vram_stats,
                    "error": err_msg,
                    "elapsed_sec": round(elapsed, 2)
                }
            
            # Tính toán thống kê chuẩn xác trên các bước Steady-State
            if steady_ips:
                med_ips = statistics.median(steady_ips)
                avg_ips = statistics.mean(steady_ips)
                sd_ips = statistics.stdev(steady_ips) if len(steady_ips) > 1 else 0.0
                cov = (sd_ips / avg_ips * 100.0) if avg_ips > 0 else 0.0
                avg_rc = statistics.mean(steady_rc)
                avg_bc = statistics.mean(steady_bc)
                avg_compute = avg_bc - avg_rc
                rr = (avg_rc / avg_bc) if avg_bc > 0 else 0.0
                p_alloc = max(steady_mem_alloc) if steady_mem_alloc else 0
                p_res = max(steady_mem_res) if steady_mem_res else 0
                l_init = steady_loss[0] if steady_loss else 0.0
                l_end = steady_loss[-1] if steady_loss else 0.0
                l_stat = "HEALTHY" if (not math.isnan(l_end) and not math.isinf(l_end)) else "UNSTABLE"
            else:
                # Trường hợp không đủ step sau warmup
                all_ips = warmup_ips
                med_ips = statistics.median(all_ips) if all_ips else 0.0
                avg_ips = med_ips
                sd_ips, cov, avg_rc, avg_bc, avg_compute, rr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                p_alloc, p_res, l_init, l_end, l_stat = 0, 0, 0.0, 0.0, "SHORT_RUN"
            
            print(f"✅ {name} HOÀN THÀNH: Median Dual IPS = {med_ips:.1f} s/s (±{cov:.1f}%) | BatchCost = {avg_bc:.4f}s (Compute: {avg_compute:.4f}s, Reader: {rr*100:.1f}%) | Peak VRAM: {p_alloc} MB ({p_alloc/15360*100:.1f}%) | Loss: {l_init:.2f} -> {l_end:.2f} ({l_stat})")
            
            return {
                "name": name,
                "batch_size_per_card": bs,
                "global_batch_size": bs * 2,
                "num_workers": workers,
                "amp": amp,
                "amp_level": amp_level,
                "nccl_tune": nccl_tune,
                "status": "SUCCESS",
                "median_ips": round(med_ips, 1),
                "mean_ips": round(avg_ips, 1),
                "std_ips": round(sd_ips, 2),
                "cov_pct": round(cov, 1),
                "avg_batch_cost": round(avg_bc, 4),
                "avg_reader_cost": round(avg_rc, 4),
                "compute_cost": round(avg_compute, 4),
                "reader_ratio": round(rr, 3),
                "peak_alloc_mb": p_alloc,
                "peak_res_mb": p_res,
                "loss_initial": round(l_init, 2),
                "loss_final": round(l_end, 2),
                "loss_status": l_stat,
                "vram": vram_stats,
                "error": None,
                "elapsed_sec": round(elapsed, 2)
            }
            
    except Exception as e:
        print(f"❌ Exception in trial {name}: {e}")
        return {
            "name": name,
            "batch_size_per_card": bs,
            "global_batch_size": bs * 2,
            "num_workers": workers,
            "amp": amp,
            "amp_level": amp_level,
            "nccl_tune": nccl_tune,
            "status": "CRASH",
            "median_ips": 0.0,
            "mean_ips": 0.0,
            "error": str(e),
            "elapsed_sec": round(time.time() - t_start, 2)
        }

def main():
    base_cfg = "/kaggle/working/configs/rec_cham_v24.yml"
    data_dir = "/kaggle/working/data/cham_synthetic_benchmark"
    dict_path = "/kaggle/working/data/cham_dict_v24.txt"
    log_dir = "/kaggle/working/benchmark_output"
    
    results = []
    print("="*90)
    print("🚀 BẮT ĐẦU CHẠY BENCHMARK V2 CHUẨN XÁC CAO TRÊN DUAL TESLA T4x2")
    print(f"Tổng số trials tập trung: {len(TRIALS)}")
    print("="*90, flush=True)
    
    for trial_cfg in TRIALS:
        res = run_trial(trial_cfg, base_cfg, data_dir, dict_path, log_dir)
        results.append(res)
        time.sleep(3) # Thu hồi tài nguyên và để GPU hạ nhiệt
        
    # Baseline Median IPS (Trial 1)
    baseline_ips = results[0]["median_ips"] if results and results[0]["status"] == "SUCCESS" else 1.0
    
    # Tính toán Speedup và thời gian 40 epochs (10M mẫu)
    for r in results:
        if r["status"] == "SUCCESS" and r["median_ips"] > 0:
            r["speedup_pct"] = round(((r["median_ips"] - baseline_ips) / baseline_ips) * 100, 1)
            # 250,000 mẫu x 40 epochs = 10,000,000 mẫu
            est_hours = 10000000.0 / (r["median_ips"] * 3600.0)
            r["est_40_epoch_hours"] = round(est_hours, 2)
        else:
            r["speedup_pct"] = 0.0
            r["est_40_epoch_hours"] = 0.0

    # Xuất JSON
    report_json = "/kaggle/working/benchmark_t4_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    # In bảng Markdown chuẩn mực
    print("\n" + "="*110)
    print("📊 BẢNG TỔNG KẾT BENCHMARK V2 DUAL GPU TESLA T4x2 (STEADY-STATE)")
    print("="*110)
    header = f"{'Trial Name':<24} | {'BS/GPU':<7} | {'TotalBS':<7} | {'W':<3} | {'Mode':<7} | {'Median IPS':<11} | {'Speedup':<8} | {'BatchCost':<10} | {'Compute':<9} | {'Peak VRAM':<10} | {'Status'}"
    print(header)
    print("-" * len(header))
    
    md_table = [
        "| Thí nghiệm | BS/GPU | Tổng BS | Workers | Chế độ | Median Dual IPS (mẫu/s) | Tăng tốc (%) | Batch Cost (s) | GPU Compute (s) | Peak VRAM | Trạng thái |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    for r in results:
        mode = f"AMP-{r.get('amp_level', 'O1')}" if r.get('amp') else "FP32"
        if r.get("nccl_tune"): mode += "+NCCL"
        v_ips = f"{r['median_ips']:.1f}" if r['status'] == "SUCCESS" else "N/A"
        v_speedup = f"+{r['speedup_pct']}%" if r['status'] == "SUCCESS" else "N/A"
        v_bc = f"{r['avg_batch_cost']:.4f}s" if r['status'] == "SUCCESS" else "N/A"
        v_cmp = f"{r.get('compute_cost', 0):.4f}s" if r['status'] == "SUCCESS" else "N/A"
        v_vram = f"{r.get('peak_alloc_mb', 0)} MB ({r.get('peak_alloc_mb', 0)/15360*100:.1f}%)" if r['status'] == "SUCCESS" else "N/A"
        
        row_str = f"{r['name']:<24} | {r['batch_size_per_card']:<7} | {r['global_batch_size']:<7} | {r['num_workers']:<3} | {mode:<7} | {v_ips:<11} | {v_speedup:<8} | {v_bc:<10} | {v_cmp:<9} | {v_vram:<10} | {r['status']}"
        print(row_str)
        
        md_table.append(f"| {r['name']} | {r['batch_size_per_card']} | {r['global_batch_size']} | {r['num_workers']} | {mode} | **{v_ips}** | **{v_speedup}** | {v_bc} | {v_cmp} | {v_vram} | {r['status']} |")
        
    print("="*110)
    
    # Tìm cấu hình quán quân
    successful_trials = [r for r in results if r["status"] == "SUCCESS"]
    if successful_trials:
        best_trial = max(successful_trials, key=lambda x: x["median_ips"])
        print(f"\n🏆 CẤU HÌNH TỐI ƯU NHẤT: {best_trial['name']}")
        print(f"🔥 Max Steady-State Dual IPS: {best_trial['median_ips']} mẫu/giây (Tăng tốc +{best_trial['speedup_pct']}% so với Baseline)")
        print(f"⚡ Thời gian hoàn thành 40 epochs: {best_trial['est_40_epoch_hours']} giờ (so với Baseline {results[0].get('est_40_epoch_hours', 'N/A')} giờ)")
        print(f"💾 Peak Memory tiêu thụ: {best_trial.get('peak_alloc_mb', 0)} MB / 15,360 MB ({best_trial.get('peak_alloc_mb', 0)/15360*100:.1f}%)")
        print(f"📈 Tính ổn định Loss: {best_trial.get('loss_initial', 0)} -> {best_trial.get('loss_final', 0)} ({best_trial.get('loss_status', 'N/A')})")
        
        md_summary = f"""\n\n## 🏆 Kết luận Cấu hình Tối ưu Tuyệt đối trên Dual GPU T4x2
- **Tên cấu hình**: `{best_trial['name']}`
- **Batch Size mỗi card**: `{best_trial['batch_size_per_card']}` (Tổng Global Batch Size: `{best_trial['global_batch_size']}`)
- **Số Workers**: `{best_trial['num_workers']}`
- **Chế độ tính toán**: `{'AMP ' + best_trial['amp_level'] if best_trial['amp'] else 'FP32'}`
- **Throughput Steady-State Dual IPS**: **`{best_trial['median_ips']} mẫu/s`** (Tăng **`+{best_trial['speedup_pct']}%`**)
- **Thời gian hoàn thành 40 epochs (10M mẫu)**: **`{best_trial['est_40_epoch_hours']} giờ`** (Tiết kiệm **`{results[0].get('est_40_epoch_hours', 0) - best_trial['est_40_epoch_hours']:.2f} giờ`**)
- **Mức tiêu thụ VRAM đỉnh**: **`{best_trial.get('peak_alloc_mb', 0)} MB`** (**`{best_trial.get('peak_alloc_mb', 0)/15360*100:.1f}%`** VRAM)
- **Kiểm định Loss**: `{best_trial.get('loss_initial', 0)}` -> `{best_trial.get('loss_final', 0)}` (`{best_trial.get('loss_status', 'N/A')}`)
"""
    else:
        md_summary = "\n\n❌ Không có trial nào thành công."

    md_report = "# Báo Cáo Benchmark V2 Chuẩn Xác Cao Trên Kaggle Dual GPU Tesla T4x2 (Cham-OCR V24)\n\n" + "\n".join(md_table) + md_summary
    with open("/kaggle/working/benchmark_t4_summary.md", "w", encoding="utf-8") as f:
        f.write(md_report)
        
    print(f"\n✅ Đã lưu kết quả tại: {report_json}")
    print(f"✅ Đã lưu tóm tắt tại: /kaggle/working/benchmark_t4_summary.md")

if __name__ == '__main__':
    main()
