#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cloud Benchmark Runner for Cham-OCR.
Runs benchmark orchestrator on Lightning Cloud (A100, L4, L40S) with:
- Detached execution (nohup) resilient to network disconnects
- Real-time log streaming
- Automatic result download
- Guaranteed auto-shutdown in finally: studio.stop() (Zero Credit Leakage).
"""

import os
import sys
import time
import json
import base64
import argparse
from lightning_sdk import Studio, Machine

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

MACHINE_CONFIGS = {
    "A100": {
        "studio_name": "cham-ocr-v25-a100",
        "machine": Machine.A100,
        "cost_per_hour": 2.19,
        "gpu_display": "NVIDIA A100 (40GB SXM4)",
    },
    "L4": {
        "studio_name": "cham-det-h100",
        "machine": Machine.L4,
        "cost_per_hour": 0.79,
        "gpu_display": "NVIDIA L4 (24GB)",
    },
    "L40S": {
        "studio_name": "cham-det-h100",
        "machine": Machine.L40S,
        "cost_per_hour": 2.14,
        "gpu_display": "NVIDIA L40S (48GB)",
    }
}


def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def push_file(studio, local_path, remote_path):
    log(f"📦 Đẩy file: {os.path.basename(local_path)} -> {remote_path}")
    with open(local_path, "rb") as f:
        content = f.read()

    b64_str = base64.b64encode(content).decode("ascii")
    chunk_size = 64 * 1024
    chunks = [b64_str[i:i + chunk_size] for i in range(0, len(b64_str), chunk_size)]

    studio.run(f"python3 -c \"import pathlib; p = pathlib.Path('{remote_path}'); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('')\"")
    for chunk in chunks:
        cmd = f"python3 -c \"import pathlib; p = pathlib.Path('{remote_path}'); open('{remote_path}.b64', 'a').write('{chunk}')\""
        studio.run(cmd)

    decode_cmd = f"python3 -c \"import base64, pathlib; data = base64.b64decode(pathlib.Path('{remote_path}.b64').read_text()); pathlib.Path('{remote_path}').write_bytes(data); pathlib.Path('{remote_path}.b64').unlink()\""
    studio.run(decode_cmd)


def pull_file(studio, remote_path, local_path):
    log(f"📥 Tải file: {remote_path} -> {local_path}")
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    res = studio.run(f"test -f '{remote_path}' && base64 '{remote_path}' || echo 'NOT_FOUND'").strip()
    if not res or res == "NOT_FOUND":
        log(f"⚠️ Remote file {remote_path} không tồn tại!")
        return False

    b64_data = "".join(res.split())
    data = base64.b64decode(b64_data)
    with open(local_path, "wb") as f:
        f.write(data)
    log(f"   ✅ Đã tải thành công ({len(data):,} bytes)")
    return True


def setup_studio_environment(studio, remote_base):
    log("🔧 Kiểm tra môi trường thực thi trên Studio...")
    check_py_gpu = studio.run("test -x /teamspace/studios/this_studio/py_gpu && echo 'EXISTS' || echo 'MISSING'").strip()
    if "MISSING" in check_py_gpu:
        log("⚙️ Tạo launcher py_gpu cho Studio...")
        script = """
        cat << 'EOF' > /teamspace/studios/this_studio/py_gpu
#!/usr/bin/env bash
if [ -d /teamspace/studios/this_studio/py310/lib/python3.10/site-packages/nvidia ]; then
    SITE_PKG=/teamspace/studios/this_studio/py310/lib/python3.10/site-packages
    NVIDIA_PATHS=$(find $SITE_PKG/nvidia -name "lib" -type d 2>/dev/null | tr '\n' ':' | sed 's/:$//')
    export LD_LIBRARY_PATH=$NVIDIA_PATHS:$LD_LIBRARY_PATH
    exec /teamspace/studios/this_studio/py310/bin/python "$@"
else
    exec python3 "$@"
fi
EOF
        chmod +x /teamspace/studios/this_studio/py_gpu
        """
        studio.run(script)
        studio.run("pip install -q psutil pyyaml tqdm")
    log("✅ Môi trường Python GPU đã sẵn sàng!")


def run_benchmark_on_target(target_name, warmup_steps=100, measure_steps=300):
    if target_name not in MACHINE_CONFIGS:
        raise ValueError(f"Target '{target_name}' không hợp lệ. Chọn một trong: {list(MACHINE_CONFIGS.keys())}")

    cfg = MACHINE_CONFIGS[target_name]
    log("=" * 75)
    log(f"🚀 BẮT ĐẦU BENCHMARK TRÊN {cfg['gpu_display']} (${cfg['cost_per_hour']}/h)")
    log(f"Studio: {cfg['studio_name']} | Machine: {cfg['machine'].name}")
    log("=" * 75)

    studio = Studio(cfg["studio_name"], teamspace="phucsd", org="phucsd-org")
    studio.show_progress = False

    try:
        while str(studio.status).lower() in ["stopping", "transitioning"]:
            log(f"Studio đang ở trạng thái '{studio.status}', đợi 5 giây...")
            time.sleep(5)

        target_machine = cfg["machine"]
        if str(studio.status).lower() != "running":
            log(f"Khởi động Studio trên {target_machine.name}...")
            studio.start(machine=target_machine)
        elif studio.machine != target_machine:
            log(f"Studio đang chạy trên {studio.machine.name}, chuyển sang {target_machine.name}...")
            studio.switch_machine(target_machine)

        log(f"✅ Studio đã chạy! Trạng thái: {studio.status} | Machine: {studio.machine}")

        remote_base = "/teamspace/studios/this_studio/Cham-OCR/ocr-training"
        setup_studio_environment(studio, remote_base)

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        local_worker = os.path.join(repo_root, "ocr-training", "scripts", "benchmark_worker.py")
        local_orch = os.path.join(repo_root, "ocr-training", "scripts", "benchmark_orchestrator.py")
        local_yml = os.path.join(repo_root, "ocr-training", "configs", "rec_cham_v25_lightning.yml")

        # Sync code
        push_file(studio, local_worker, f"{remote_base}/scripts/benchmark_worker.py")
        push_file(studio, local_orch, f"{remote_base}/scripts/benchmark_orchestrator.py")
        push_file(studio, local_yml, f"{remote_base}/configs/rec_cham_v25_lightning.yml")

        # Ensure dataset is present
        data_check = studio.run(f"test -f {remote_base}/data/cham_synthetic_v25/train_label.txt && echo 'EXISTS' || echo 'MISSING'").strip()
        if "MISSING" in data_check:
            log("📦 Dữ liệu chưa có trên Studio, đang giải nén từ /teamspace/lightning_storage/v25_shared_data/ocr_training.tar...")
            studio.run(f"mkdir -p /teamspace/studios/this_studio/Cham-OCR && cd /teamspace/studios/this_studio/Cham-OCR && tar -xf /teamspace/lightning_storage/v25_shared_data/ocr_training.tar")
            log("✅ Đã giải nén xong dữ liệu huấn luyện!")

        # Ensure PaddleOCR repo is present
        pocr_check = studio.run(f"test -d {remote_base}/PaddleOCR && echo 'EXISTS' || echo 'MISSING'").strip()
        if "MISSING" in pocr_check:
            log("📦 Clone PaddleOCR repository...")
            studio.run(f"cd {remote_base} && git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleOCR.git")
            log("✅ Đã clone PaddleOCR repository!")

        # Launch detached benchmark process
        log("🔥 Khởi chạy benchmark ngầm (nohup) với giám sát thời gian thực...")
        studio.run(f"pkill -f 'benchmark_orchestrator.py' 2>/dev/null || true")
        studio.run(f"pkill -f 'benchmark_worker.py' 2>/dev/null || true")
        studio.run(f"rm -f {remote_base}/benchmark_running.pid {remote_base}/benchmark_done.flag")

        launch_cmd = f"""
        cat << 'EOF' > {remote_base}/run_bench.sh
#!/usr/bin/env bash
cd {remote_base}
/teamspace/studios/this_studio/py_gpu scripts/benchmark_orchestrator.py \\
    -c configs/rec_cham_v25_lightning.yml \\
    --gpu_name "{target_name}" \\
    --cost_per_hour {cfg['cost_per_hour']} \\
    --python_bin /teamspace/studios/this_studio/py_gpu \\
    --warmup_steps {warmup_steps} \\
    --measure_steps {measure_steps} \\
    -o benchmark_results > {remote_base}/benchmark.log 2>&1
echo \\$? > {remote_base}/benchmark_done.flag
EOF
        chmod +x {remote_base}/run_bench.sh
        nohup {remote_base}/run_bench.sh >/dev/null 2>&1 &
        echo \\$! > {remote_base}/benchmark_running.pid
        """
        studio.run(launch_cmd)
        log("✅ Tiến trình benchmark đã khởi chạy thành công!")

        # Monitoring loop
        seen_lines = 0
        poll_interval = 15
        start_time = time.time()
        max_benchmark_time = 45 * 60  # 45 minutes safety timeout

        while time.time() - start_time < max_benchmark_time:
            time.sleep(poll_interval)
            
            # Check done flag
            done_check = studio.run(f"test -f {remote_base}/benchmark_done.flag && cat {remote_base}/benchmark_done.flag || echo 'RUNNING'").strip()
            
            # Stream recent logs
            log_tail = studio.run(f"tail -n 15 {remote_base}/benchmark.log 2>/dev/null || true").strip()
            if log_tail:
                log(f"--- [Log Update ({int(time.time() - start_time)}s elapsed)] ---\n{log_tail}")

            if done_check != "RUNNING":
                exit_code = int(done_check) if done_check.isdigit() else -1
                log(f"🏁 Benchmark đã hoàn thành với exit code: {exit_code}")
                break

        # Pull results
        local_results_dir = os.path.join(repo_root, "ocr-training", "benchmark_results")
        os.makedirs(local_results_dir, exist_ok=True)

        pull_file(studio, f"{remote_base}/benchmark_results/benchmark_{target_name}.json", os.path.join(local_results_dir, f"benchmark_{target_name}.json"))
        pull_file(studio, f"{remote_base}/benchmark_results/benchmark_{target_name}.md", os.path.join(local_results_dir, f"benchmark_{target_name}.md"))
        pull_file(studio, f"{remote_base}/benchmark.log", os.path.join(local_results_dir, f"benchmark_{target_name}.log"))

        log(f"🎉 Hoàn thành benchmark cho {target_name}!")

    finally:
        log("🛑 [BẢO TOÀN CREDIT] Đang chuyển Studio sang STOPPED...")
        try:
            studio.stop()
            log("✅ Studio đã STOPPED thành công! Zero rò rỉ credit.")
        except Exception as e:
            log(f"Lỗi khi dừng studio: {e}")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Benchmark Runner on Lightning Cloud")
    parser.add_argument("--target", type=str, default="A100", choices=["A100", "L4", "L40S", "ALL"], help="Target GPU machine")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps per trial")
    parser.add_argument("--measure_steps", type=int, default=300, help="Measure steps per trial")
    args = parser.parse_args()

    targets = ["A100", "L4", "L40S"] if args.target == "ALL" else [args.target]
    for t in targets:
        run_benchmark_on_target(t, warmup_steps=args.warmup_steps, measure_steps=args.measure_steps)


if __name__ == "__main__":
    main()
