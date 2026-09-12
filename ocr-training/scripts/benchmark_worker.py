#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolated Benchmark Worker for PaddleOCR Training.
Executes a single (batch_size, num_workers) trial in an isolated subprocess.
Exits cleanly on success (code 0), OOM (code 137), NaN loss (code 138), or error (code 1).
"""

import os
import sys
import time
import json
import argparse
import traceback
import threading

# 1. NumPy 2.x compatibility monkeypatch
import numpy as np
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str]
    }
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'typeDict'): np.typeDict = {}

# 2. Add PaddleOCR path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Check possible PaddleOCR locations
paddleocr_dirs = [
    os.path.join(current_dir, "..", "PaddleOCR"),
    os.path.join(current_dir, "..", "..", "ocr-studio", "PaddleOCR"),
    os.path.join(current_dir, "..", "..", "PaddleOCR"),
    "/teamspace/studios/this_studio/Cham-OCR/ocr-training/PaddleOCR",
]
for p in paddleocr_dirs:
    if os.path.isdir(p):
        sys.path.insert(0, os.path.abspath(p))
        break

import yaml
import paddle
import paddle.distributed as dist

from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model, apply_to_static
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger

try:
    import psutil
except ImportError:
    psutil = None

def to_float32(preds):
    if isinstance(preds, dict):
        for k in preds:
            if isinstance(preds[k], dict):
                for sub_k in preds[k]:
                    preds[k][sub_k] = preds[k][sub_k].astype(paddle.float32)
            else:
                preds[k] = preds[k].astype(paddle.float32)
    elif isinstance(preds, list):
        preds = [pred.astype(paddle.float32) for pred in preds]
    else:
        preds = preds.astype(paddle.float32)
    return preds


class GPUMonitor(threading.Thread):
    """Monitors GPU utilization and VRAM in a background thread."""
    def __init__(self, interval=0.3):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = True
        self.gpu_utils = []
        self.vram_used_mb = []
        self.total_vram_mb = 0.0
        self.cpu_percents = []

    def run(self):
        while self.running:
            try:
                # Query nvidia-smi
                import subprocess
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1.5
                )
                if res.returncode == 0 and res.stdout.strip():
                    parts = [float(x.strip()) for x in res.stdout.strip().split("\n")[0].split(",")]
                    self.gpu_utils.append(parts[0])
                    self.vram_used_mb.append(parts[1])
                    self.total_vram_mb = parts[2]
            except Exception:
                pass

            if psutil:
                try:
                    self.cpu_percents.append(psutil.cpu_percent(interval=None))
                except Exception:
                    pass

            time.sleep(self.interval)

    def stop(self):
        self.running = False


def run_benchmark(config_path, batch_size, num_workers, warmup_steps=100, measure_steps=300, output_json="trial_result.json"):
    result = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "status": "UNKNOWN",
        "ips": 0.0,
        "avg_reader_cost": 0.0,
        "avg_batch_cost": 0.0,
        "reader_ratio": 0.0,
        "peak_vram_mb": 0.0,
        "total_vram_mb": 0.0,
        "vram_pct": 0.0,
        "avg_gpu_util": 0.0,
        "avg_cpu_pct": 0.0,
        "error": None
    }

    monitor = GPUMonitor(interval=0.3)
    monitor.start()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Override configurations
        config['Global']['distributed'] = False
        config['Global']['use_gpu'] = True
        config['Global']['cal_metric_during_train'] = False
        config['Global']['epoch_num'] = 100
        config['Train']['loader']['batch_size_per_card'] = batch_size
        config['Train']['loader']['num_workers'] = num_workers
        config['Train']['loader']['drop_last'] = True
        config['Eval'] = None

        device = 'gpu'
        paddle.set_device('gpu:0')
        logger = get_logger(name="benchmark_worker")

        # Set signal handlers & build DataLoader
        set_signal_handlers()
        train_dataloader = build_dataloader(config, 'Train', device, logger, seed=42)
        if len(train_dataloader) == 0:
            raise RuntimeError("DataLoader is empty! Check dataset paths in config.")

        global_config = config['Global']
        post_process_class = build_post_process(config['PostProcess'], global_config)

        # MultiHead channel adjustments
        if hasattr(post_process_class, 'character'):
            char_num = len(getattr(post_process_class, 'character'))
            if config['Architecture']['Head']['name'] == 'MultiHead':
                if config['PostProcess']['name'] == 'NRTRLabelDecode':
                    char_num = char_num - 3
                out_channels_list = {'CTCLabelDecode': char_num}
                if len(config['Loss']['loss_config_list']) > 1 and list(config['Loss']['loss_config_list'][1].keys())[0] == 'NRTRLoss':
                    out_channels_list['NRTRLabelDecode'] = char_num + 3
                config['Architecture']['Head']['out_channels_list'] = out_channels_list
            else:
                config['Architecture']['Head']['out_channels'] = char_num

        model = build_model(config['Architecture'])
        model = apply_to_static(model, config, logger)
        model.train()
        loss_class = build_loss(config['Loss'])
        optimizer, lr_scheduler = build_optimizer(
            config['Optimizer'],
            epochs=config['Global']['epoch_num'],
            step_each_epoch=len(train_dataloader),
            model=model
        )

        use_amp = config['Global'].get('use_amp', False)
        amp_level = config['Global'].get('amp_level', 'O1')
        amp_dtype = config['Global'].get('amp_dtype', 'float16')
        scaler = paddle.amp.GradScaler() if use_amp else None

        extra_input_models = [
            "SRN", "NRTR", "SAR", "SEED", "SVTR", "SVTR_LCNet", "SPIN", "VisionLAN",
            "RobustScanner", "RFL", 'DRRG', 'SATRN', 'SVTR_HGNet', "ParseQ", "CPPD"
        ]
        extra_input = config['Architecture'].get('algorithm', '') in extra_input_models or config['Architecture']['Head'].get('name', '') == 'MultiHead'

        total_steps = warmup_steps + measure_steps
        step_idx = 0

        total_reader_time = 0.0
        total_batch_time = 0.0
        total_samples = 0

        data_iter = iter(train_dataloader)
        print(f"[Worker] Starting trial: Batch={batch_size}, Workers={num_workers}, Warmup={warmup_steps}, Measure={measure_steps}, ExtraInput={extra_input}", flush=True)

        reader_start = time.perf_counter()
        while step_idx < total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            t_read = time.perf_counter() - reader_start
            images = batch[0]
            batch_samples = len(images)

            # Forward + Backward
            if scaler:
                with paddle.amp.auto_cast(level=amp_level, dtype=amp_dtype):
                    if extra_input:
                        preds = model(images, data=batch[1:])
                    else:
                        preds = model(images)
                preds = to_float32(preds)
                loss = loss_class(preds, batch)
                avg_loss = loss['loss']
                loss_val = float(avg_loss)
                if np.isnan(loss_val) or np.isinf(loss_val):
                    raise ValueError(f"Loss exploded: {loss_val}")
                scaled_avg_loss = scaler.scale(avg_loss)
                scaled_avg_loss.backward()
                scaler.minimize(optimizer, scaled_avg_loss)
            else:
                if extra_input:
                    preds = model(images, data=batch[1:])
                else:
                    preds = model(images)
                loss = loss_class(preds, batch)
                avg_loss = loss['loss']
                loss_val = float(avg_loss)
                if np.isnan(loss_val) or np.isinf(loss_val):
                    raise ValueError(f"Loss exploded: {loss_val}")
                avg_loss.backward()
                optimizer.step()

            optimizer.clear_grad()
            if not isinstance(lr_scheduler, float):
                lr_scheduler.step()

            # Synchronize CUDA to get exact batch execution time
            try:
                paddle.device.cuda.synchronize()
            except Exception:
                pass
            t_batch = time.perf_counter() - reader_start

            if step_idx >= warmup_steps:
                total_reader_time += t_read
                total_batch_time += t_batch
                total_samples += batch_samples

            step_idx += 1
            if step_idx % 50 == 0:
                cur_ips = batch_samples / t_batch if t_batch > 0 else 0
                phase = "WARMUP" if step_idx <= warmup_steps else "MEASURE"
                print(f"[Worker] Step {step_idx}/{total_steps} [{phase}] - IPS: {cur_ips:.1f} - BatchTime: {t_batch:.4f}s - Reader: {t_read:.4f}s", flush=True)

            reader_start = time.perf_counter()

        # Stop monitoring
        monitor.stop()

        avg_reader_cost = total_reader_time / measure_steps if measure_steps > 0 else 0
        avg_batch_cost = total_batch_time / measure_steps if measure_steps > 0 else 0
        ips = total_samples / total_batch_time if total_batch_time > 0 else 0

        # Memory stats
        paddle_peak_bytes = paddle.device.cuda.max_memory_allocated()
        paddle_peak_mb = paddle_peak_bytes / (1024 * 1024)
        smi_peak_mb = max(monitor.vram_used_mb) if monitor.vram_used_mb else paddle_peak_mb
        peak_vram_mb = max(paddle_peak_mb, smi_peak_mb)
        total_vram_mb = monitor.total_vram_mb if monitor.total_vram_mb > 0 else 40960.0
        vram_pct = (peak_vram_mb / total_vram_mb) * 100 if total_vram_mb > 0 else 0

        avg_gpu = sum(monitor.gpu_utils) / len(monitor.gpu_utils) if monitor.gpu_utils else 0.0
        avg_cpu = sum(monitor.cpu_percents) / len(monitor.cpu_percents) if monitor.cpu_percents else 0.0

        result.update({
            "status": "SUCCESS",
            "ips": round(ips, 2),
            "avg_reader_cost": round(avg_reader_cost, 4),
            "avg_batch_cost": round(avg_batch_cost, 4),
            "reader_ratio": round(avg_reader_cost / avg_batch_cost, 3) if avg_batch_cost > 0 else 0,
            "peak_vram_mb": round(peak_vram_mb, 1),
            "total_vram_mb": round(total_vram_mb, 1),
            "vram_pct": round(vram_pct, 1),
            "avg_gpu_util": round(avg_gpu, 1),
            "avg_cpu_pct": round(avg_cpu, 1)
        })
        print(f"[Worker SUCCESS] Batch={batch_size} Workers={num_workers} => IPS={ips:.1f} samples/s, VRAM={vram_pct:.1f}%, ReaderRatio={result['reader_ratio']}", flush=True)
        exit_code = 0

    except Exception as e:
        monitor.stop()
        err_str = str(e).lower()
        full_tb = traceback.format_exc()
        print(f"[Worker ERROR] {e}\n{full_tb}", file=sys.stderr, flush=True)

        if "out of memory" in err_str or "bad_alloc" in err_str or "cudabadalloc" in err_str or "memoryerror" in err_str:
            result["status"] = "OOM"
            result["error"] = "CUDA Out Of Memory"
            exit_code = 137
        elif "loss exploded" in err_str or "nan" in err_str or "inf" in err_str:
            result["status"] = "NAN_LOSS"
            result["error"] = str(e)
            exit_code = 138
        else:
            result["status"] = "CRASH"
            result["error"] = str(e)
            exit_code = 1

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    sys.exit(exit_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated PaddleOCR Benchmark Worker")
    parser.add_argument("-c", "--config_path", type=str, required=True, help="Path to YAML training config")
    parser.add_argument("-b", "--batch_size", type=int, required=True, help="Batch size per card")
    parser.add_argument("-w", "--num_workers", type=int, required=True, help="Number of DataLoader workers")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps before measuring")
    parser.add_argument("--measure_steps", type=int, default=300, help="Measured steps")
    parser.add_argument("-o", "--output_json", type=str, default="trial_result.json", help="Path to write trial JSON")
    args = parser.parse_args()

    run_benchmark(
        config_path=args.config_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        output_json=args.output_json
    )
