import os
import sys
import json
import paddle
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def perform_weight_surgery_v24(old_ckpt_path='ocr-training/data/cham-ocr-v5-assets/v23_checkpoint/iter_epoch_200.pdparams',
                               new_ckpt_dir='ocr-training/data/output_v24_temp/rec_cham_best_model',
                               report_path='ocr-training/output/v24_training/head_extension_report.json',
                               old_dict_path='ocr-training/data/cham_dict_v23.txt',
                               new_dict_path='ocr-training/data/cham_dict_v24.txt'):
    
    os.makedirs(new_ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    new_ckpt_path = os.path.join(new_ckpt_dir, 'iter_epoch_200.pdparams')
    
    # 1. Đếm số ký tự từ điển
    with open(old_dict_path, 'r', encoding='utf-8') as f:
        old_chars = [l for l in f.read().splitlines() if l]
    with open(new_dict_path, 'r', encoding='utf-8') as f:
        new_chars = [l for l in f.read().splitlines() if l]
        
    num_old = len(old_chars) # 82
    num_new = len(new_chars) # 103
    
    # CTC: blank (0) + characters (1..num) + space (num+1)
    old_ctc_classes = num_old + 2 # 84
    new_ctc_classes = num_new + 2 # 105
    
    # GTC: 6 special tokens + characters
    old_gtc_vocab = num_old + 6 # 88
    new_gtc_vocab = num_new + 6 # 109
    
    print(f"🔬 Khởi chạy Phẫu thuật Trọng số (Weight Surgery) V24:")
    print(f"   - Từ điển: {num_old} -> {num_new} ký tự")
    print(f"   - CTC Classes: {old_ctc_classes} -> {new_ctc_classes}")
    print(f"   - GTC Vocab: {old_gtc_vocab} -> {new_gtc_vocab}")
    
    old_state = paddle.load(old_ckpt_path)
    new_state = {}
    
    for k, v in old_state.items():
        if k == 'head.ctc_head.fc.weight':
            in_features, old_c = v.shape
            assert old_c == old_ctc_classes, f"Mismatch old CTC shape: {v.shape}"
            
            std = v.std().item()
            mean = v.mean().item()
            new_v_np = np.random.normal(mean, std, (in_features, new_ctc_classes)).astype('float32')
            
            # Bảo tồn trọng số cũ:
            # idx 0: CTC blank
            # idx 1..num_old: các ký tự cũ
            new_v_np[:, 0:num_old+1] = v.numpy()[:, 0:num_old+1]
            # idx space cũ chuyển sang idx space mới
            new_v_np[:, new_ctc_classes - 1] = v.numpy()[:, old_ctc_classes - 1]
            
            new_state[k] = paddle.to_tensor(new_v_np)
            print(f"   ✅ {k}: {v.shape} -> {new_state[k].shape}")
            
        elif k == 'head.ctc_head.fc.bias':
            assert v.shape[0] == old_ctc_classes
            new_v_np = np.zeros(new_ctc_classes, dtype='float32')
            new_v_np[0:num_old+1] = v.numpy()[0:num_old+1]
            new_v_np[new_ctc_classes - 1] = v.numpy()[old_ctc_classes - 1]
            
            new_state[k] = paddle.to_tensor(new_v_np)
            print(f"   ✅ {k}: {v.shape} -> {new_state[k].shape}")
            
        elif k == 'head.gtc_head.embedding.embedding.weight':
            old_v, embed_dim = v.shape
            assert old_v == old_gtc_vocab
            std = v.std().item()
            mean = v.mean().item()
            new_v_np = np.random.normal(mean, std, (new_gtc_vocab, embed_dim)).astype('float32')
            
            # idx 0..num_old+3: blank, unk, bos, eos + old characters
            # Trong GTC: 4 tokens đầu + num_old ký tự = num_old + 4
            split_point = num_old + 4
            new_v_np[0:split_point, :] = v.numpy()[0:split_point, :]
            
            # 2 tokens cuối: space và pad
            new_v_np[new_gtc_vocab - 2:new_gtc_vocab, :] = v.numpy()[old_gtc_vocab - 2:old_gtc_vocab, :]
            
            new_state[k] = paddle.to_tensor(new_v_np)
            print(f"   ✅ {k}: {v.shape} -> {new_state[k].shape}")
            
        elif k == 'head.gtc_head.tgt_word_prj.weight':
            embed_dim, old_v = v.shape
            assert old_v == old_gtc_vocab
            std = v.std().item()
            mean = v.mean().item()
            new_v_np = np.random.normal(mean, std, (embed_dim, new_gtc_vocab)).astype('float32')
            
            split_point = num_old + 4
            new_v_np[:, 0:split_point] = v.numpy()[:, 0:split_point]
            new_v_np[:, new_gtc_vocab - 2:new_gtc_vocab] = v.numpy()[:, old_gtc_vocab - 2:old_gtc_vocab]
            
            new_state[k] = paddle.to_tensor(new_v_np)
            print(f"   ✅ {k}: {v.shape} -> {new_state[k].shape}")
            
        else:
            new_state[k] = v
            
    # Lưu trọng số mở rộng mới
    paddle.save(new_state, new_ckpt_path)
    
    report = {
        "old_num_classes": old_ctc_classes,
        "new_num_classes": new_ctc_classes,
        "old_gtc_vocab": old_gtc_vocab,
        "new_gtc_vocab": new_gtc_vocab,
        "old_chars_preserved": True,
        "new_chars_initialized": True,
        "new_ckpt_path": new_ckpt_path
    }
    with open(report_path, 'w', encoding='utf-8') as f_rep:
        json.dump(report, f_rep, indent=2)
        
    print(f"🎉 Phẫu thuật thành công! Đã lưu checkpoint nền V24 tại: {new_ckpt_path}")
    return new_ckpt_path, report

def find_assets_dir():
    candidates = [
        "ocr-training/data/cham-ocr-v5-assets",
        "/kaggle/input/cham-ocr-v5-assets",
        "/kaggle/input/datasets/gustavnguyen/cham-ocr-v5-assets"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'v23_checkpoint' in dirs:
                return root
    return "ocr-training/data/cham-ocr-v5-assets"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets-dir', type=str, default=None)
    parser.add_argument('--old-ckpt', type=str, default=None)
    parser.add_argument('--new-ckpt-dir', type=str, default=None)
    args = parser.parse_args()
    
    assets_dir = args.assets_dir or find_assets_dir()
    old_ckpt = args.old_ckpt or os.path.join(assets_dir, 'v23_checkpoint', 'iter_epoch_200.pdparams')
    new_ckpt_dir = args.new_ckpt_dir or ('/kaggle/working/data/output_v24_temp/rec_cham_best_model' if os.path.exists('/kaggle') else 'ocr-training/data/output_v24_temp/rec_cham_best_model')
    report_path = '/kaggle/working/output/v24_training/head_extension_report.json' if os.path.exists('/kaggle') else 'ocr-training/output/v24_training/head_extension_report.json'
    old_dict = os.path.join(assets_dir, 'v23_checkpoint', 'cham_dict_v23.txt')
    if not os.path.exists(old_dict):
        old_dict = 'ocr-training/data/cham_dict_v23.txt'
    new_dict = '/kaggle/working/data/cham_dict_v24.txt' if os.path.exists('/kaggle') else 'ocr-training/data/cham_dict_v24.txt'
    
    perform_weight_surgery_v24(
        old_ckpt_path=old_ckpt,
        new_ckpt_dir=new_ckpt_dir,
        report_path=report_path,
        old_dict_path=old_dict,
        new_dict_path=new_dict
    )

if __name__ == '__main__':
    main()
