#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kịch bản Phẫu Thuật Trọng Số V24 -> V25 (Character-Mapped Weight Surgery):
1. Nạp checkpoint V24 (pdparams)
2. Đọc từ điển cham_dict_v24.txt và cham_dict_v25.txt
3. Tạo bảng ánh xạ Ký Tự (Character Mapping): char -> old_idx, char -> new_idx
4. Mở rộng kích thước ma trận cho toàn bộ các layer phụ thuộc vocabulary:
   - CTCHead:
     * ctc_head.fc.weight [hidden_dim, num_classes]
     * ctc_head.fc.bias   [num_classes]
   - NRTRHead:
     * nrtr_head.item_embedding.weight [num_classes, hidden_dim]
     * nrtr_head.fc.weight [hidden_dim, num_classes]
     * nrtr_head.fc.bias   [num_classes]
5. Kế thừa 100% trọng số của ký tự cũ theo Ký tự (không map mù theo index).
6. Khởi tạo ký tự mới (tiếng Việt, ký hiệu) bằng phân phối Gauss nhỏ N(0, 0.02) để bảo toàn không gian đặc trưng cũ.
7. Lưu checkpoint sẵn sàng cho V25 huấn luyện kế thừa.
"""

import os
import sys
import numpy as np
try:
    import paddle
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Monkeypatch NumPy 2.x nếu cần
if not hasattr(np, 'sctypes'):
    np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64], 'uint': [np.uint8, np.uint16, np.uint32, np.uint64], 'float': [np.float16, np.float32, np.float64], 'complex': [np.complex64, np.complex128], 'others': [bool, object, bytes, str]}
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float

def load_dict(path):
    tokens = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            t = line.rstrip('\r\n')
            if t:
                tokens.append(t)
    return tokens

def perform_weight_surgery(v24_model_path, v24_dict_path, v25_dict_path, output_model_path):
    print("=" * 80)
    print("🔬 BẮT ĐẦU PHẪU THUẬT TRỌNG SỐ (WEIGHT SURGERY) V24 -> V25")
    print(f" • Checkpoint nguồn V24: {v24_model_path}")
    print(f" • Từ điển V24: {v24_dict_path}")
    print(f" • Từ điển V25: {v25_dict_path}")
    print(f" • Checkpoint xuất V25: {output_model_path}")
    print("=" * 80)

    # 1. Nạp từ điển
    v24_chars = load_dict(v24_dict_path)
    v25_chars = load_dict(v25_dict_path)
    
    print(f"✅ V24 Tokens: {len(v24_chars)}")
    print(f"✅ V25 Tokens: {len(v25_chars)}")
    
    # PaddleOCR quy ước số class:
    # Với CTC: num_classes = len(dict) + 1 (blank token tại index 0)
    # Với NRTR: num_classes = len(dict) + 4 (các token đặc biệt: blank, unk, bos, eos)
    old_char2idx = {c: i for i, c in enumerate(v24_chars)}
    new_char2idx = {c: i for i, c in enumerate(v25_chars)}

    if not PADDLE_AVAILABLE:
        raise RuntimeError("❌ Cần môi trường có thư viện 'paddlepaddle' (ví dụ trên Kaggle/Linux/GPU) để nạp và lưu tệp trọng số .pdparams.")

    # 2. Nạp weights V24
    if v24_model_path.endswith('.pdparams'):
        v24_weights = paddle.load(v24_model_path)
    else:
        # Đường dẫn thư mục hoặc prefix
        param_file = v24_model_path + '.pdparams' if not v24_model_path.endswith('.pdparams') else v24_model_path
        if os.path.exists(param_file):
            v24_weights = paddle.load(param_file)
        else:
            raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {v24_model_path}")

    v25_weights = {}
    surgered_layers = []

    for k, v in v24_weights.items():
        val = v.numpy() if hasattr(v, 'numpy') else v
        
        # A. Xử lý CTC Head FC: Shape [hidden_dim, num_classes] hoặc [num_classes] cho bias
        if "ctc_head.fc.weight" in k:
            hidden_dim, old_classes = val.shape
            # Số class = len(dict) + 1 (blank ở vị trí 0)
            offset = old_classes - len(v24_chars)
            new_classes = len(v25_chars) + offset
            print(f"🛠️  Phẫu thuật {k}: [{hidden_dim}, {old_classes}] -> [{hidden_dim}, {new_classes}]")
            
            # Khởi tạo ma trận mới với phân phối ngẫu nhiên nhỏ N(0, 0.02)
            new_val = np.random.normal(0.0, 0.02, size=(hidden_dim, new_classes)).astype(val.dtype)
            
            # Copy blank token (thường ở index 0)
            for b in range(offset):
                new_val[:, b] = val[:, b]
                
            # Copy trọng số theo từng ký tự chính xác (character mapping)
            mapped_count = 0
            for c, old_i in old_char2idx.items():
                if c in new_char2idx:
                    new_i = new_char2idx[c]
                    new_val[:, new_i + offset] = val[:, old_i + offset]
                    mapped_count += 1
                    
            print(f"   -> Đã sao chép chính xác {mapped_count} vectors trọng số ký tự Chăm cũ.")
            v25_weights[k] = paddle.to_tensor(new_val)
            surgered_layers.append(k)

        elif "ctc_head.fc.bias" in k:
            old_classes = val.shape[0]
            offset = old_classes - len(v24_chars)
            new_classes = len(v25_chars) + offset
            print(f"🛠️  Phẫu thuật {k}: [{old_classes}] -> [{new_classes}]")
            
            new_val = np.zeros(shape=(new_classes,), dtype=val.dtype)
            for b in range(offset):
                new_val[b] = val[b]
            for c, old_i in old_char2idx.items():
                if c in new_char2idx:
                    new_i = new_char2idx[c]
                    new_val[new_i + offset] = val[old_i + offset]
                    
            v25_weights[k] = paddle.to_tensor(new_val)
            surgered_layers.append(k)

        # B. Xử lý NRTR Head Item Embedding: Shape [num_classes, hidden_dim]
        elif "nrtr_head.item_embedding.weight" in k:
            old_classes, hidden_dim = val.shape
            offset = old_classes - len(v24_chars)
            new_classes = len(v25_chars) + offset
            print(f"🛠️  Phẫu thuật {k}: [{old_classes}, {hidden_dim}] -> [{new_classes}, {hidden_dim}]")
            
            new_val = np.random.normal(0.0, 0.02, size=(new_classes, hidden_dim)).astype(val.dtype)
            for b in range(offset):
                new_val[b, :] = val[b, :]
            for c, old_i in old_char2idx.items():
                if c in new_char2idx:
                    new_i = new_char2idx[c]
                    new_val[new_i + offset, :] = val[old_i + offset, :]
                    
            v25_weights[k] = paddle.to_tensor(new_val)
            surgered_layers.append(k)

        # C. Xử lý NRTR Head FC: Shape [hidden_dim, num_classes] hoặc [num_classes]
        elif "nrtr_head.fc.weight" in k:
            hidden_dim, old_classes = val.shape
            offset = old_classes - len(v24_chars)
            new_classes = len(v25_chars) + offset
            print(f"🛠️  Phẫu thuật {k}: [{hidden_dim}, {old_classes}] -> [{hidden_dim}, {new_classes}]")
            
            new_val = np.random.normal(0.0, 0.02, size=(hidden_dim, new_classes)).astype(val.dtype)
            for b in range(offset):
                new_val[:, b] = val[:, b]
            for c, old_i in old_char2idx.items():
                if c in new_char2idx:
                    new_i = new_char2idx[c]
                    new_val[:, new_i + offset] = val[:, old_i + offset]
                    
            v25_weights[k] = paddle.to_tensor(new_val)
            surgered_layers.append(k)

        elif "nrtr_head.fc.bias" in k:
            old_classes = val.shape[0]
            offset = old_classes - len(v24_chars)
            new_classes = len(v25_chars) + offset
            print(f"🛠️  Phẫu thuật {k}: [{old_classes}] -> [{new_classes}]")
            
            new_val = np.zeros(shape=(new_classes,), dtype=val.dtype)
            for b in range(offset):
                new_val[b] = val[b]
            for c, old_i in old_char2idx.items():
                if c in new_char2idx:
                    new_i = new_char2idx[c]
                    new_val[new_i + offset] = val[old_i + offset]
                    
            v25_weights[k] = paddle.to_tensor(new_val)
            surgered_layers.append(k)

        # D. Tất cả các layer Backbone / SVTR Transformer không phụ thuộc vocab: Giữ nguyên 100%
        else:
            v25_weights[k] = v

    # 3. Lưu checkpoint đã phẫu thuật
    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)
    save_target = output_model_path if output_model_path.endswith('.pdparams') else output_model_path + '.pdparams'
    paddle.save(v25_weights, save_target)
    
    print("\n" + "=" * 80)
    print("🎉 HOÀN TẤT PHẪU THUẬT TRỌNG SỐ THÀNH CÔNG!")
    print(f" • Tổng số layer trong mô hình: {len(v25_weights)}")
    print(f" • Các layer đã được phẫu thuật ánh xạ ký tự: {len(surgered_layers)}")
    for l in surgered_layers:
        print(f"    - {l}")
    print(f" • Checkpoint phẫu thuật V25 đã lưu tại: {save_target}")
    print("=" * 80)
    return save_target

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Character-Mapped Weight Surgery V24 -> V25")
    parser.add_argument("--src_model", type=str, default=None, help="Đường dẫn file trọng số V24 (.pdparams)")
    parser.add_argument("--dst_model", type=str, default="/kaggle/working/output/rec_cham_v25_init/init_weights.pdparams", help="Đường dẫn lưu trọng số V25")
    parser.add_argument("--v24_dict", type=str, default="/kaggle/working/data/cham_dict_v24.txt", help="Đường dẫn từ điển V24")
    parser.add_argument("--v25_dict", type=str, default="/kaggle/working/data/cham_dict_v25.txt", help="Đường dẫn từ điển V25")
    args = parser.parse_args()

    # Tự động tìm kiếm checkpoint V24 nếu không chỉ định rõ
    src_model = args.src_model
    if not src_model:
        candidates = [
            "/kaggle/working/data/v24_checkpoint/best_accuracy",
            "/kaggle/working/output/rec_cham_v24_best/best_checkpoint/best_accuracy",
            "/kaggle/input/paddleocr-cham-finetune/rec_cham_v24_best/best_checkpoint/best_accuracy",
            "ocr-training/output/v24/extracted_v24/best_checkpoint/best_accuracy",
            "ocr-training/data/output_v24_temp/rec_cham_best_model/iter_epoch_200"
        ]
        for c in candidates:
            if os.path.exists(c + '.pdparams') or os.path.exists(c):
                src_model = c
                break

    if src_model and (os.path.exists(src_model + '.pdparams') or os.path.exists(src_model)):
        perform_weight_surgery(src_model, args.v24_dict, args.v25_dict, args.dst_model)
    else:
        print(f"ℹ️  Không tìm thấy checkpoint V24 tại '{src_model}'. Kịch bản sẵn sàng khi có đường dẫn cụ thể.")
