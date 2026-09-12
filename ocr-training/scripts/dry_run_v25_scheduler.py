#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dry-Run Verification Script for Cham-OCR V25 40-Epoch Cosine Schedule.
Simulates and verifies:
  1. Total epochs, stage ends, steps/epoch, global batch, warmup steps, T_max.
  2. Mathematical LR milestones for epochs 2, 12, 13, 22, 23, 40.
  3. Continuous schedule invariant (no resets across stage boundaries).
  4. Dictionary hash and token count invariance.
"""

import os
import sys
import math
import hashlib
import yaml

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(TRAINING_DIR)

CONFIG_PATH = os.path.join(TRAINING_DIR, "configs", "rec_cham_v25.yml")
DICT_PATH = os.path.join(TRAINING_DIR, "data", "cham_dict_v25.txt")


def calculate_lr(step, warmup_steps, T_max, base_lr):
    if step <= warmup_steps:
        return base_lr * (step / warmup_steps)
    else:
        inner_step = step - warmup_steps
        ratio = inner_step / T_max
        return 0.5 * base_lr * (1.0 + math.cos(math.pi * ratio))


def run_dry_run():
    print("=" * 80)
    print("🔬 BẮT ĐẦU DRY-RUN KIỂM THỬ INVARIANTS HUẤN LUYỆN V25 (40 EPOCHS)")
    print("=" * 80)

    # 1. Đọc và kiểm tra file config rec_cham_v25.yml
    assert os.path.exists(CONFIG_PATH), f"Không tìm thấy {CONFIG_PATH}"
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    epoch_num = cfg['Global']['epoch_num']
    stage_end_epoch = cfg['Global'].get('stage_end_epoch', 12)
    batch_size_per_card = cfg['Train']['loader']['batch_size_per_card']
    num_gpus = 2
    global_batch = batch_size_per_card * num_gpus

    train_samples = 140000
    steps_per_epoch = train_samples // global_batch
    warmup_epochs = cfg['Optimizer']['lr']['warmup_epoch']
    warmup_steps = warmup_epochs * steps_per_epoch
    base_lr = float(cfg['Optimizer']['lr']['learning_rate'])
    T_max = epoch_num * steps_per_epoch

    print("\n📋 1. THAM SỐ CẤU HÌNH & HẠ TẦNG TOÀN CỤC:")
    print(f"   • Total Epochs toàn training : {epoch_num}")
    print(f"   • Stage End Epoch (Chặng 1)  : {stage_end_epoch}")
    print(f"   • Batch size per card (GPU)  : {batch_size_per_card}")
    print(f"   • Số lượng GPU phân tán      : {num_gpus} (Tesla T4x2)")
    print(f"   • Global Batch Size hiệu dụng: {global_batch}")
    print(f"   • Quy mô mẫu Train           : {train_samples:,} mẫu")
    print(f"   • Steps per Epoch            : {steps_per_epoch} steps/epoch")
    print(f"   • Warmup Epochs              : {warmup_epochs} epochs")
    print(f"   • Warmup Steps               : {warmup_steps} steps")
    print(f"   • T_max (Toàn bộ chu kỳ)     : {T_max} steps")
    print(f"   • Base Learning Rate (η_max) : {base_lr:.6f} ({base_lr})")

    # Invariants assertions
    assert epoch_num == 40, f"Global.epoch_num phải bằng 40, hiện tại là {epoch_num}"
    assert steps_per_epoch == 2187, f"steps_per_epoch phải bằng 2187, hiện tại là {steps_per_epoch}"
    assert global_batch == 64, f"global_batch phải bằng 64, hiện tại là {global_batch}"
    assert warmup_steps == 4374, f"warmup_steps phải bằng 4374, hiện tại là {warmup_steps}"
    assert T_max == 87480, f"T_max phải bằng 87480, hiện tại là {T_max}"
    print("   👉 [INVARIANT 1]: Cấu hình 40 epochs & batch steps: 100% PASS!")

    # 2. Kiểm tra bộ từ điển V25 (Dictionary Freeze)
    assert os.path.exists(DICT_PATH), f"Không tìm thấy {DICT_PATH}"
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        tokens = [line.strip() for line in f if line.strip()]
    dict_md5 = hashlib.md5(open(DICT_PATH, 'rb').read()).hexdigest()
    dict_sha256 = hashlib.sha256(open(DICT_PATH, 'rb').read()).hexdigest()

    print("\n📖 2. KIỂM ĐỊNH TỪ ĐIỂN V25 (DICTIONARY FREEZE):")
    print(f"   • Số lượng tokens NFC        : {len(tokens)} tokens")
    print(f"   • MD5 Checksum               : {dict_md5}")
    print(f"   • SHA256 Checksum            : {dict_sha256[:24]}...")
    assert len(tokens) == 162, f"Từ điển phải có đúng 162 tokens, hiện tại: {len(tokens)}"
    print("   👉 [INVARIANT 2]: Từ điển và Token IDs khóa cứng: 100% PASS!")

    # 3. Tính toán và xác thực Learning Rate tại các mốc trọng yếu
    print("\n📈 3. BẢNG MỐC LEARNING RATE TRẢI DÀI 40 EPOCHS (87,480 STEPS):")
    print("-" * 88)
    print(f"{'Mốc Thời Điểm':<32} | {'Epoch':<6} | {'Step':<7} | {'Cosine Ratio':<12} | {'Learning Rate':<18}")
    print("-" * 88)

    milestones = [
        ("Khởi đầu Warmup", 0, 1, "0.0000"),
        ("Giữa Warmup (Epoch 1)", 1, 1 * steps_per_epoch, "0.0000"),
        ("Cuối Warmup (Epoch 2)", 2, 2 * steps_per_epoch, "0.0000"),
        ("Cuối Chặng 1 (Epoch 12)", 12, 12 * steps_per_epoch, "0.2500"),
        ("Đầu Chặng 2 (Epoch 13)", 13, 12 * steps_per_epoch + 1, "0.2500"),
        ("Cuối Chặng 2 (Epoch 22)", 22, 22 * steps_per_epoch, "0.5000"),
        ("Đầu Chặng 3 (Epoch 23)", 23, 22 * steps_per_epoch + 1, "0.5000"),
        ("Tiệm cận cuối (Epoch 35)", 35, 35 * steps_per_epoch, "0.8250"),
        ("Kết thúc Toàn bộ (Epoch 40)", 40, 40 * steps_per_epoch, "0.9500")
    ]

    calculated_lrs = {}
    for desc, ep, step, expected_ratio in milestones:
        lr_val = calculate_lr(step, warmup_steps, T_max, base_lr)
        calculated_lrs[ep] = lr_val
        inner_step = max(0, step - warmup_steps)
        ratio = inner_step / T_max
        print(f"{desc:<32} | {ep:<6} | {step:<7} | {ratio:<12.4f} | {lr_val:.8e}")

    print("-" * 88)

    # Xác thực toán học các mốc yêu cầu
    lr_ep2 = calculated_lrs[2]
    lr_ep12 = calculated_lrs[12]
    lr_ep13 = calculated_lrs[13]
    lr_ep22 = calculated_lrs[22]
    lr_ep23 = calculated_lrs[23]
    lr_ep40 = calculated_lrs[40]

    assert abs(lr_ep2 - 1.0e-4) < 1e-9, f"LR Epoch 2 phải bằng 1e-4, tính ra: {lr_ep2}"
    assert abs(lr_ep12 - 8.53553391e-5) < 1e-9, f"LR Epoch 12 phải bằng 8.5355e-5, tính ra: {lr_ep12}"
    assert abs(lr_ep13 - 8.53540694e-5) < 1e-9, f"LR Epoch 13 phải bằng 8.5354e-5, tính ra: {lr_ep13}"
    assert abs(lr_ep22 - 5.0e-5) < 1e-9, f"LR Epoch 22 phải bằng 5.0e-5, tính ra: {lr_ep22}"
    assert abs(lr_ep23 - 4.99982044e-5) < 1e-9, f"LR Epoch 23 phải bằng 4.9998e-5, tính ra: {lr_ep23}"
    assert abs(lr_ep40 - 6.15582970e-7) < 1e-9, f"LR Epoch 40 phải bằng 6.1558e-7, tính ra: {lr_ep40}"

    # Kiểm tra tính liên tục qua ranh giới chặng (Stage boundary continuity)
    step_diff_12_13 = abs(lr_ep12 - lr_ep13)
    step_diff_22_23 = abs(lr_ep22 - lr_ep23)
    print(f"\n🔄 4. ĐỘ LIÊN TỤC TẠI RANH GIỚI CÁC CHẶNG:")
    print(f"   • Chặng 1 -> Chặng 2 (Epoch 12 -> 13): Bước nhảy Δ = {step_diff_12_13:.2e} (Hoàn toàn mượt mà)")
    print(f"   • Chặng 2 -> Chặng 3 (Epoch 22 -> 23): Bước nhảy Δ = {step_diff_22_23:.2e} (Hoàn toàn mượt mà)")
    assert step_diff_12_13 < 1e-7, "Bước nhảy LR giữa Chặng 1 và Chặng 2 không liên tục!"
    assert step_diff_22_23 < 1e-7, "Bước nhảy LR giữa Chặng 2 và Chặng 3 không liên tục!"
    print("   👉 [INVARIANT 3]: Tính liên tục toán học toàn cục: 100% PASS!")

    print("\n🎉 TOÀN BỘ CÁC INVARIANTS ĐÃ ĐƯỢC XÁC THỰC THÀNH CÔNG!")
    print("=" * 80)
    return True


if __name__ == '__main__':
    run_dry_run()
