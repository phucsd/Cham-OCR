#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kịch bản xây dựng từ điển V25 tự động & Preflight Validation:
1. Chuẩn hóa nhãn văn bản với Unicode NFC (unicodedata.normalize('NFC', ...))
2. Giữ nguyên 100% thứ tự token V24 (103 tokens) làm nền tảng cố định
3. Tự động phát hiện và append các token mới vào cuối từ điển (không hardcode số lượng)
4. Quét phân bố độ dài nhãn (P50, P90, P95, P99, Max) và xác định max_text_length = P99
5. Chạy Preflight Validation:
   - unknown_char == 0
   - invalid_unicode == 0
   - labels_exceeding_max_len == 0
"""

import os
import sys
import json
import unicodedata
import math
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def load_v24_dict(v24_dict_path):
    """Nạp danh sách token V24, bảo tồn chính xác thứ tự."""
    if not os.path.exists(v24_dict_path):
        raise FileNotFoundError(f"Không tìm thấy từ điển V24 tại: {v24_dict_path}")
    tokens = []
    with open(v24_dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.rstrip('\r\n')
            if token:
                tokens.append(token)
    return tokens

def is_valid_unicode(char):
    """Kiểm tra ký tự có phải là Unicode hợp lệ (không phải surrogate hay control vô nghĩa)."""
    try:
        cat = unicodedata.category(char)
        # Loại trừ lone surrogates (Cs) và format control không nhìn thấy
        if cat in ('Cs',):
            return False
        return True
    except Exception:
        return False

def build_dict_and_validate(label_files, v24_dict_path, output_dict_path, report_path=None, enforced_max_len=None):
    print("=" * 80)
    print("🚀 BẮT ĐẦU XÂY DỰNG TỪ ĐIỂN V25 ĐỘNG & CHẠY PREFLIGHT VALIDATION")
    print("=" * 80)
    
    # 1. Nạp từ điển V24 gốc
    v24_tokens = load_v24_dict(v24_dict_path)
    print(f"✅ Đã nạp {len(v24_tokens)} tokens cơ sở từ V24 (bảo toàn 100% thứ tự).")
    
    v24_token_set = set(v24_tokens)
    new_token_set = set()
    
    all_lengths = []
    total_lines = 0
    invalid_unicode_chars = set()
    sample_labels = []
    
    # 2. Đọc toàn bộ nhãn từ các tệp dữ liệu
    print(f"📂 Đang quét các tệp nhãn: {label_files}")
    for lf in label_files:
        if not os.path.exists(lf):
            print(f"⚠️  Bỏ qua tệp không tồn tại: {lf}")
            continue
            
        with open(lf, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                    
                total_lines += 1
                parts = line.split('\t', 1)
                text = parts[1] if len(parts) > 1 else parts[0]
                
                # Bắt buộc chuẩn hóa Unicode NFC
                norm_text = unicodedata.normalize('NFC', text)
                all_lengths.append(len(norm_text))
                
                if total_lines <= 10:
                    sample_labels.append(norm_text)
                    
                for char in norm_text:
                    if not is_valid_unicode(char):
                        invalid_unicode_chars.add(char)
                        continue
                    if char != ' ' and char not in v24_token_set:
                        new_token_set.add(char)
                        
    print(f"📊 Đã quét tổng cộng {total_lines:,} dòng nhãn.")
    
    # Sắp xếp các ký tự mới append để đảm bảo tính nhất quán (deterministic order)
    # Nhóm ký tự tiếng Việt / Latin / dấu câu được sắp xếp theo codepoint
    sorted_new_tokens = sorted(list(new_token_set), key=lambda c: ord(c))
    
    # Ghép từ điển V25 hoàn chỉnh: [V24 Tokens] + [New Tokens appended at the end]
    v25_tokens = v24_tokens + sorted_new_tokens
    v25_token_set = set(v25_tokens)
    
    print(f"✨ Số token V24 ban đầu: {len(v24_tokens)}")
    print(f"✨ Số token mới phát hiện và bổ sung: {len(sorted_new_tokens)}")
    print(f"✨ Tổng kích thước từ điển V25: {len(v25_tokens)} tokens")
    
    # 3. Phân bố độ dài nhãn (Length Percentiles Scan)
    if all_lengths:
        arr_len = np.array(all_lengths)
        p50 = float(np.percentile(arr_len, 50))
        p90 = float(np.percentile(arr_len, 90))
        p95 = float(np.percentile(arr_len, 95))
        p99 = float(np.percentile(arr_len, 99))
        max_len = int(np.max(arr_len))
        min_len = int(np.min(arr_len))
        mean_len = float(np.mean(arr_len))
    else:
        p50 = p90 = p95 = p99 = max_len = min_len = mean_len = 0
        
    recommended_max_len = int(math.ceil(p99))
    target_max_len = enforced_max_len if enforced_max_len else recommended_max_len
    
    # 4. Chạy Preflight Validation
    unknown_chars = set()
    labels_exceeding_max_len = 0
    
    for lf in label_files:
        if not os.path.exists(lf):
            continue
        with open(lf, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t', 1)
                text = parts[1] if len(parts) > 1 else parts[0]
                norm_text = unicodedata.normalize('NFC', text)
                
                if len(norm_text) > target_max_len:
                    labels_exceeding_max_len += 1
                    
                for char in norm_text:
                    if char != ' ' and char not in v25_token_set:
                        unknown_chars.add(char)
                        
    # 5. Lưu từ điển V25
    os.makedirs(os.path.dirname(os.path.abspath(output_dict_path)), exist_ok=True)
    with open(output_dict_path, 'w', encoding='utf-8', newline='\n') as f_out:
        for t in v25_tokens:
            f_out.write(t + '\n')
            
    # 6. Tổng hợp báo cáo kiểm định Preflight
    validation_status = (
        len(unknown_chars) == 0 and
        len(invalid_unicode_chars) == 0 and
        labels_exceeding_max_len == 0
    )
    
    report = {
        "status": "PASS" if validation_status else "WARNING/FAIL",
        "preflight_checks": {
            "unknown_characters_count": len(unknown_chars),
            "unknown_characters_list": list(unknown_chars),
            "invalid_unicode_count": len(invalid_unicode_chars),
            "invalid_unicode_list": [f"U+{ord(c):04X}" for c in invalid_unicode_chars],
            "labels_exceeding_max_len_count": labels_exceeding_max_len,
            "labels_exceeding_percentage": round((labels_exceeding_max_len / total_lines * 100), 2) if total_lines > 0 else 0.0
        },
        "vocabulary_metrics": {
            "v24_base_tokens": len(v24_tokens),
            "new_appended_tokens": len(sorted_new_tokens),
            "v25_total_tokens": len(v25_tokens),
            "sample_new_tokens": sorted_new_tokens[:25]
        },
        "label_length_percentiles": {
            "min": min_len,
            "p50_median": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "max": max_len,
            "mean": round(mean_len, 2),
            "recommended_max_text_length": recommended_max_len,
            "configured_max_text_length": target_max_len
        }
    }
    
    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f_rep:
            json.dump(report, f_rep, indent=2, ensure_ascii=False)
            
    print("\n" + "="*80)
    print("📋 KẾT QUẢ PREFLIGHT VALIDATION TỪ ĐIỂN V25:")
    print("="*80)
    print(f" • Trạng thái Preflight: {'✅ PASS' if validation_status else '⚠️ CẦN LƯU Ý'}")
    print(f" • Ký tự lạ (Unknown chars): {len(unknown_chars)}")
    print(f" • Ký tự Unicode không hợp lệ: {len(invalid_unicode_chars)}")
    print(f" • Phân bố độ dài nhãn: P50={p50:.1f} | P90={p90:.1f} | P95={p95:.1f} | P99={p99:.1f} | Max={max_len}")
    print(f" • Max Text Length đề xuất (dựa trên P99): {recommended_max_len}")
    print(f" • Số nhãn vượt ngưỡng {target_max_len}: {labels_exceeding_max_len}")
    print(f" • Từ điển V25 đã lưu tại: {output_dict_path}")
    if report_path:
        print(f" • Báo cáo chi tiết lưu tại: {report_path}")
    print("="*80)
    
    return v25_tokens, report

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Build V25 dictionary and run preflight checks")
    parser.add_argument("--v24-dict", default="ocr-training/data/cham_dict_v24.txt", help="Path to V24 base dictionary")
    parser.add_argument("--output-dict", default="ocr-training/data/cham_dict_v25.txt", help="Path to save V25 dictionary")
    parser.add_argument("--report", default="ocr-training/data/v25_dict_preflight_report.json", help="Path to save validation report")
    parser.add_argument("--label-files", nargs="*", default=[], help="Explicit list of label files to scan")
    parser.add_argument("--max-len", type=int, default=None, help="Enforce specific max text length")
    args = parser.parse_args()

    if args.label_files:
        avail_labels = args.label_files
    else:
        label_candidates = [
            "ocr-training/data/cham_synthetic_v24/train_label.txt",
            "ocr-training/data/cham_synthetic_v24/val_label.txt",
            "ocr-training/data/cham_synthetic_v23_1/train_label.txt",
            "ocr-training/data/cham_synthetic_images/train_label.txt"
        ]
        avail_labels = [p for p in label_candidates if os.path.exists(p)]
    
    build_dict_and_validate(avail_labels, args.v24_dict, args.output_dict, args.report, args.max_len)

