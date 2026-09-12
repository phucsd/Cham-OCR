import os
import sys
import json
import hashlib

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def generate_v24_dict(old_dict_path='ocr-training/data/cham_dict_v23.txt', 
                       new_dict_path='ocr-training/data/cham_dict_v24.txt',
                       report_path='ocr-training/output/v24_training/dict_extension_report.json'):
    
    os.makedirs(os.path.dirname(os.path.abspath(new_dict_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    
    # 1. Đọc từ điển cũ V23
    with open(old_dict_path, 'r', encoding='utf-8') as f:
        old_lines = f.read().splitlines()
    
    old_chars = [line for line in old_lines if line]
    old_dict_size = len(old_chars)
    old_dict_sha = sha256(old_dict_path)
    
    # 2. Các ký tự bổ sung cốt lõi cho V24:
    # - ꨲ (U+AA32): Dấu phụ Au/Ue bị thiếu ở V23
    # - ꩁ (U+AA41): Final G trong khối Unicode Chăm
    # - Chữ số Latin và ký hiệu ngắt câu tham chiếu
    required_additions = [
        'ꨲ', 'ꩁ',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '[', ']', '(', ')', '.', ',', ';', '-', '/'
    ]
    
    added_chars = []
    for c in required_additions:
        if c not in old_chars and c not in added_chars:
            added_chars.append(c)
            
    new_chars = old_chars + added_chars
    new_dict_size = len(new_chars)
    
    with open(new_dict_path, 'w', encoding='utf-8', newline='\n') as f_out:
        for c in new_chars:
            f_out.write(c + '\n')
            
    new_dict_sha = sha256(new_dict_path)

    report = {
        "old_dict_size": old_dict_size,
        "new_dict_size": new_dict_size,
        "added_chars": added_chars,
        "old_dict_sha256": old_dict_sha,
        "new_dict_sha256": new_dict_sha
    }
    with open(report_path, 'w', encoding='utf-8') as f_rep:
        json.dump(report, f_rep, indent=2, ensure_ascii=False)
        
    print(f"✅ Từ điển V24 đã tạo tại: {new_dict_path}")
    print(f"   Số lượng ký tự: {old_dict_size} -> {new_dict_size} (Thêm {len(added_chars)} ký tự)")
    print(f"   Ký tự bổ sung: {' '.join(added_chars)}")
    return new_dict_path, report

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
    parser.add_argument('--old-dict', type=str, default=None)
    parser.add_argument('--new-dict', type=str, default=None)
    args = parser.parse_args()
    
    assets_dir = args.assets_dir or find_assets_dir()
    old_dict = args.old_dict or os.path.join(assets_dir, 'v23_checkpoint', 'cham_dict_v23.txt')
    if not os.path.exists(old_dict):
        old_dict = 'ocr-training/data/cham_dict_v23.txt'
        
    new_dict = args.new_dict or ('/kaggle/working/data/cham_dict_v24.txt' if os.path.exists('/kaggle') else 'ocr-training/data/cham_dict_v24.txt')
    report_path = '/kaggle/working/output/v24_training/dict_extension_report.json' if os.path.exists('/kaggle') else 'ocr-training/output/v24_training/dict_extension_report.json'
    
    generate_v24_dict(old_dict_path=old_dict, new_dict_path=new_dict, report_path=report_path)

if __name__ == '__main__':
    main()
