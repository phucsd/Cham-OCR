import os
import json

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    # Read the local files to embed
    generate_data_code = read_file('scripts/generate_data.py')
    dict_extension_code = read_file('scripts/dict_extension_v23_1.py')
    weight_surgery_code = read_file('scripts/weight_surgery.py')
    generate_data_v23_1_code = read_file('scripts/generate_data_v23_1.py')
    v23_inventory_content = read_file('output/v23_1_training/v23_checkpoint_inventory.json')
    
    # Let's adjust paths in weight_surgery_code and dict_extension_code for Kaggle environment
    # On Kaggle:
    # - old dict path: '/kaggle/input/datasets/gustavnguyen/cham-ocr-v5-assets/v23_checkpoint/cham_dict_v23.txt'
    # - old ckpt path: '/kaggle/input/datasets/gustavnguyen/cham-ocr-v5-assets/v23_checkpoint/iter_epoch_200.pdparams'
    # - new dict path: '/kaggle/working/paddleocr_cham_finetune/data/cham_dict_v23_1.txt'
    # - new ckpt path: '/kaggle/working/paddleocr_cham_finetune/data/output_v23_1_temp/rec_cham_best_model/iter_epoch_200.pdparams'
    # - report paths under '/kaggle/working/paddleocr_cham_finetune/output/v23_1_training/'
    
    kaggle_dict_extension_code = dict_extension_code.replace(
        "old_dict_path = 'data/cham_dict_v23.txt'",
        "old_dict_path = '/kaggle/input/datasets/gustavnguyen/cham-ocr-v5-assets/v23_checkpoint/cham_dict_v23.txt'"
    ).replace(
        "new_dict_path = 'data/cham_dict_v23_1.txt'",
        "new_dict_path = '/kaggle/working/paddleocr_cham_finetune/data/cham_dict_v23_1.txt'"
    ).replace(
        "report_path = 'output/v23_1_training/dict_extension_report.json'",
        "report_path = '/kaggle/working/paddleocr_cham_finetune/output/v23_1_training/dict_extension_report.json'"
    )
    
    kaggle_weight_surgery_code = weight_surgery_code.replace(
        "old_ckpt_path = 'data/output_v23/rec_cham_best_model/iter_epoch_200.pdparams'",
        "old_ckpt_path = '/kaggle/input/datasets/gustavnguyen/cham-ocr-v5-assets/v23_checkpoint/iter_epoch_200.pdparams'"
    ).replace(
        "new_ckpt_dir = 'data/output_v23_1_temp/rec_cham_best_model'",
        "new_ckpt_dir = '/kaggle/working/paddleocr_cham_finetune/data/output_v23_1_temp/rec_cham_best_model'"
    ).replace(
        "report_path = 'output/v23_1_training/head_extension_report.json'",
        "report_path = '/kaggle/working/paddleocr_cham_finetune/output/v23_1_training/head_extension_report.json'"
    )
    
    kaggle_generate_data_v23_1_code = generate_data_v23_1_code.replace(
        "corpus_path = 'data/corpus/cham_text.txt'",
        "corpus_path = '/kaggle/working/paddleocr_cham_finetune/data/corpus/cham_text.txt'"
    ).replace(
        "output_dir = os.path.join(PROJECT_ROOT, 'data', 'cham_synthetic_v23_1')",
        "output_dir = '/kaggle/working/paddleocr_cham_finetune/data/cham_synthetic_v23_1'"
    ).replace(
        "with open('output/v23_1_training/train_label_stats.json', 'w', encoding='utf-8') as f:",
        "with open('/kaggle/working/paddleocr_cham_finetune/output/v23_1_training/train_label_stats.json', 'w', encoding='utf-8') as f:"
    ).replace(
        "with open('output/v23_1_training/punctuation_coverage_stats.json', 'w', encoding='utf-8') as f:",
        "with open('/kaggle/working/paddleocr_cham_finetune/output/v23_1_training/punctuation_coverage_stats.json', 'w', encoding='utf-8') as f:"
    )

    # Let's write the evaluation script that runs on Kaggle
    evaluate_script_code = """import os
import sys
import json
import cv2
import paddle
import numpy as np

# Config Python paths
sys.path.insert(0, '/kaggle/working/PaddleOCR')
sys.path.insert(0, '/kaggle/working/paddleocr_cham_finetune')

from tools.predict_rec import TextRecognizer
from scripts.generate_data import unicode_to_visual

class DummyArgs:
    def __init__(self, model_dir, dict_path):
        self.rec_algorithm = 'SVTR_LCNet'
        self.rec_model_dir = model_dir
        self.rec_char_dict_path = dict_path
        self.rec_image_shape = '3, 48, 320'
        self.rec_batch_num = 6
        self.max_text_length = 80
        self.use_space_char = True
        self.use_gpu = True
        self.gpu_mem = 500
        
        # Default flags
        self.use_tensorrt = False
        self.use_onnx = False
        self.ir_optim = True
        self.use_fp16 = False
        self.total_process_num = 1
        self.benchmark = False
        self.save_log_path = "./log_predicts.txt"

def calculate_cer(pred, gt):
    if not gt:
        return 1.0 if pred else 0.0
    # Levenshtein distance
    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == gt[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 1)
    return dp[m][n] / len(gt)

def evaluate_splits():
    # Model directories
    v23_model_dir = '/kaggle/input/datasets/gustavnguyen/cham-ocr-v5-assets/v23_inference'
    v23_dict_path = '/kaggle/input/datasets/gustavnguyen/cham-ocr-v5-assets/v23_checkpoint/cham_dict_v23.txt'
    
    v23_1_model_dir = '/kaggle/working/paddleocr_cham_finetune/output/rec_cham_inference_v23_1'
    v23_1_dict_path = '/kaggle/working/paddleocr_cham_finetune/data/cham_dict_v23_1.txt'
    
    # Initialize predictors
    print("Loading V23 Baseline Model...")
    v23_args = DummyArgs(v23_model_dir, v23_dict_path)
    v23_recognizer = TextRecognizer(v23_args)
    
    print("Loading V23.1 Candidate Model...")
    v23_1_args = DummyArgs(v23_1_model_dir, v23_1_dict_path)
    v23_1_recognizer = TextRecognizer(v23_1_args)
    
    splits = [
        'clean_cham_200',
        'hard_diacritic_200',
        'long_line_200',
        'punctuation_digit_300',
        'mixed_cham_punctuation_300'
    ]
    
    predictions = []
    summary = {}
    
    overall_cer_v23 = 0.0
    overall_cer_v23_1 = 0.0
    overall_count = 0
    
    cham_only_cer_v23 = 0.0
    cham_only_cer_v23_1 = 0.0
    cham_only_count = 0
    
    latin_digit_correct = 0
    latin_digit_total = 0
    punc_correct = 0
    punc_total = 0
    bracket_correct = 0
    bracket_total = 0
    
    # Required characters for verification
    required_latin_digits = set("0123456789")
    required_punctuation = set("[]().,;:/-")
    brackets = set("[]()")
    
    eval_root = '/kaggle/working/paddleocr_cham_finetune/data/cham_synthetic_v23_1'
    
    for split in splits:
        split_label_path = os.path.join(eval_root, f"{split}_label.txt")
        if not os.path.exists(split_label_path):
            print(f"Warning: split label path {split_label_path} does not exist.")
            continue
            
        with open(split_label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        split_cer_v23 = 0.0
        split_cer_v23_1 = 0.0
        split_acc_v23 = 0
        split_acc_v23_1 = 0
        split_count = len(lines)
        
        for line in lines:
            img_rel_path, gt = line.strip().split('\t')
            img_path = os.path.join(eval_root, img_rel_path)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Run inference
            # predict_rec expects list of images, returns list of tuples (text, confidence)
            res_v23 = v23_recognizer([img])[0][0]
            res_v23_1 = v23_1_recognizer([img])[0][0]
            
            # calculate CER
            cer_v23 = calculate_cer(res_v23, gt)
            cer_v23_1 = calculate_cer(res_v23_1, gt)
            
            split_cer_v23 += cer_v23
            split_cer_v23_1 += cer_v23_1
            
            if res_v23 == gt:
                split_acc_v23 += 1
            if res_v23_1 == gt:
                split_acc_v23_1 += 1
                
            # Track counts
            overall_cer_v23 += cer_v23
            overall_cer_v23_1 += cer_v23_1
            overall_count += 1
            
            # Cham-only evaluation (if gt has no latin digits and no english punctuation)
            has_latin_or_punc = any(c in required_latin_digits or c in required_punctuation for c in gt)
            if not has_latin_or_punc:
                cham_only_cer_v23 += cer_v23
                cham_only_cer_v23_1 += cer_v23_1
                cham_only_count += 1
                
            # Metric check for latin digits, punctuation, brackets
            for char in gt:
                if char in required_latin_digits:
                    latin_digit_total += 1
                    if char in res_v23_1:
                        latin_digit_correct += 1
                if char in required_punctuation:
                    punc_total += 1
                    if char in res_v23_1:
                        punc_correct += 1
                if char in brackets:
                    bracket_total += 1
                    if char in res_v23_1:
                        bracket_correct += 1
                        
            predictions.append({
                "split": split,
                "image_path": img_rel_path,
                "gt": gt,
                "pred_v23": res_v23,
                "pred_v23_1": res_v23_1,
                "cer_v23": round(cer_v23, 4),
                "cer_v23_1": round(cer_v23_1, 4),
                "important_cham_sign_changes": [],
                "latin_digit_correct": all(c in res_v23_1 for c in gt if c in required_latin_digits),
                "punctuation_correct": all(c in res_v23_1 for c in gt if c in required_punctuation)
            })
            
        summary[split] = {
            "cer_v23": round(split_cer_v23 / split_count, 4) if split_count > 0 else 0,
            "cer_v23_1": round(split_cer_v23_1 / split_count, 4) if split_count > 0 else 0,
            "acc_v23": round(split_acc_v23 / split_count, 4) if split_count > 0 else 0,
            "acc_v23_1": round(split_acc_v23_1 / split_count, 4) if split_count > 0 else 0
        }
        print(f"Split {split}: V23 CER={summary[split]['cer_v23']:.4f}, V23.1 CER={summary[split]['cer_v23_1']:.4f}, V23 ACC={summary[split]['acc_v23']:.4f}, V23.1 ACC={summary[split]['acc_v23_1']:.4f}")

    # Calculate overall metrics
    final_summary = {
        "splits": summary,
        "overall": {
            "cer_v23": round(overall_cer_v23 / overall_count, 4) if overall_count > 0 else 0,
            "cer_v23_1": round(overall_cer_v23_1 / overall_count, 4) if overall_count > 0 else 0,
            "cham_only_cer_v23": round(cham_only_cer_v23 / cham_only_count, 4) if cham_only_count > 0 else 0,
            "cham_only_cer_v23_1": round(cham_only_cer_v23_1 / cham_only_count, 4) if cham_only_count > 0 else 0,
            "latin_digit_accuracy": round(latin_digit_correct / latin_digit_total, 4) if latin_digit_total > 0 else 1.0,
            "punctuation_accuracy": round(punc_correct / punc_total, 4) if punc_total > 0 else 1.0,
            "bracket_accuracy": round(bracket_correct / bracket_total, 4) if bracket_total > 0 else 1.0,
            "cham_regression": round((cham_only_cer_v23_1 - cham_only_cer_v23) / cham_only_count, 4) if cham_only_count > 0 else 0.0
        }
    }
    
    # Save predictions and summary
    predictions_path = '/kaggle/working/paddleocr_cham_finetune/output/v23_1_training/predictions_v23_1.jsonl'
    summary_path = '/kaggle/working/paddleocr_cham_finetune/output/v23_1_training/eval_summary.json'
    
    with open(predictions_path, 'w', encoding='utf-8') as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
            
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2)
        
    print("✅ Evaluation complete.")
    print(f"Overall V23 CER: {final_summary['overall']['cer_v23']:.4f}")
    print(f"Overall V23.1 CER: {final_summary['overall']['cer_v23_1']:.4f}")
    print(f"Cham-only V23 CER: {final_summary['overall']['cham_only_cer_v23']:.4f}")
    print(f"Cham-only V23.1 CER: {final_summary['overall']['cham_only_cer_v23_1']:.4f}")
    print(f"Latin Digit Accuracy: {final_summary['overall']['latin_digit_accuracy']:.2%}")
    print(f"Punctuation Accuracy: {final_summary['overall']['punctuation_accuracy']:.2%}")
    print(f"Bracket Accuracy: {final_summary['overall']['bracket_accuracy']:.2%}")

    # Generate comparison markdown
    comp_path = '/kaggle/working/paddleocr_cham_finetune/output/v23_1_training/v23_vs_v23_1_comparison.md'
    with open(comp_path, 'w', encoding='utf-8') as f:
        f.write("# V23 vs V23.1 Performance Comparison\\n\\n")
        f.write("## Overall Metrics\\n\\n")
        f.write("| Metric | V23 Baseline | V23.1 Candidate | Delta |\\n")
        f.write("| --- | --- | --- | --- |\\n")
        f.write(f"| Overall CER | {final_summary['overall']['cer_v23']:.2%} | {final_summary['overall']['cer_v23_1']:.2%} | {final_summary['overall']['cer_v23_1'] - final_summary['overall']['cer_v23']:.2%} |\\n")
        f.write(f"| Cham-only CER | {final_summary['overall']['cham_only_cer_v23']:.2%} | {final_summary['overall']['cham_only_cer_v23_1']:.2%} | {final_summary['overall']['cham_only_cer_v23_1'] - final_summary['overall']['cham_only_cer_v23']:.2%} |\\n")
        f.write(f"| Latin Digit Acc | - | {final_summary['overall']['latin_digit_accuracy']:.2%} | - |\\n")
        f.write(f"| Punctuation Acc | - | {final_summary['overall']['punctuation_accuracy']:.2%} | - |\\n")
        f.write(f"| Bracket Acc | - | {final_summary['overall']['bracket_accuracy']:.2%} | - |\\n\\n")
        
        f.write("## Split-wise Detail\\n\\n")
        f.write("| Split | V23 CER | V23.1 CER | V23 Accuracy | V23.1 Accuracy |\\n")
        f.write("| --- | --- | --- | --- | --- |\\n")
        for split, stats in summary.items():
            f.write(f"| {split} | {stats['cer_v23']:.2%} | {stats['cer_v23_1']:.2%} | {stats['acc_v23']:.2%} | {stats['acc_v23_1']:.2%} |\\n")
            
    print("✅ Comparison markdown generated at:", comp_path)

if __name__ == '__main__':
    evaluate_splits()
"""
    
    # 4. Constructing notebook structure
    cells = []
    
    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + '\n' for line in text.splitlines()]
        })
        
    def add_code(code_str):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + '\n' for line in code_str.splitlines()]
        })

    add_md("# V23.1 OCR Recognizer Training — Latin Digits & Punctuation Patch")
    
    add_md("## Phase 1: Environment Setup")
    add_code("""# Install required libraries
!pip install paddlepaddle-gpu paddleocr opencv-python albumentations pyclipper shapely Pillow pyyaml fonttools rapidfuzz

# Verify GPU
import paddle
print("Paddle Version:", paddle.__version__)
print("Paddle compiled with CUDA:", paddle.is_compiled_with_cuda())
print("Available GPU Count:", paddle.device.cuda.device_count())
if paddle.device.cuda.device_count() > 0:
    paddle.utils.run_check()
""")

    add_md("## Phase 2: Clone PaddleOCR & Directory Structure")
    add_code("""# Clone repo
%cd /kaggle/working/
!git clone https://github.com/PaddlePaddle/PaddleOCR.git
%cd PaddleOCR
!pip install -r requirements.txt
%cd /kaggle/working/

# Create folders
!mkdir -p /kaggle/working/paddleocr_cham_finetune/data/fonts
!mkdir -p /kaggle/working/paddleocr_cham_finetune/data/corpus
!mkdir -p /kaggle/working/paddleocr_cham_finetune/data/cham_synthetic_v23_1
!mkdir -p /kaggle/working/paddleocr_cham_finetune/scripts
!mkdir -p /kaggle/working/paddleocr_cham_finetune/configs
!mkdir -p /kaggle/working/paddleocr_cham_finetune/output/v23_1_training
""")

    add_md("## Phase 3: Copy Assets & Write Base Scripts")
    add_code(f"""# Copy assets directly from the Kaggle mount
import os
import shutil

# 1. Copy fonts
fonts_src = "/kaggle/input/datasets/gustavnguyen/cham-ocr-v5-assets/fonts/fonts"
fonts_dest = "/kaggle/working/paddleocr_cham_finetune/data/fonts"
for f in os.listdir(fonts_src):
    if f.endswith(('.ttf', '.otf')):
        shutil.copy2(os.path.join(fonts_src, f), os.path.join(fonts_dest, f))

# 2. Copy corpus
os.makedirs("/kaggle/working/paddleocr_cham_finetune/data/corpus", exist_ok=True)
shutil.copy2(
    "/kaggle/input/datasets/gustavnguyen/cham-ocr-v5-assets/corpus/corpus/cham_text.txt",
    "/kaggle/working/paddleocr_cham_finetune/data/corpus/cham_text.txt"
)

print("Fonts copied:", os.listdir(fonts_dest))
print("Corpus copied:", os.listdir("/kaggle/working/paddleocr_cham_finetune/data/corpus"))
""")

    add_code(f"%%writefile /kaggle/working/paddleocr_cham_finetune/scripts/generate_data.py\n{generate_data_code}")
    add_code(f"%%writefile /kaggle/working/paddleocr_cham_finetune/scripts/dict_extension_v23_1.py\n{kaggle_dict_extension_code}")
    add_code(f"%%writefile /kaggle/working/paddleocr_cham_finetune/scripts/weight_surgery.py\n{kaggle_weight_surgery_code}")
    add_code(f"%%writefile /kaggle/working/paddleocr_cham_finetune/scripts/generate_data_v23_1.py\n{kaggle_generate_data_v23_1_code}")
    add_code(f"%%writefile /kaggle/working/paddleocr_cham_finetune/scripts/evaluate_v23_1.py\n{evaluate_script_code}")
    add_code(f"%%writefile /kaggle/working/paddleocr_cham_finetune/output/v23_1_training/v23_checkpoint_inventory.json\n{v23_inventory_content}")

    add_md("## Phase 4: Run Data Generation, Dict Extension & Weight Surgery")
    add_code("""# Run scripts inside the project directory to ensure Cwd and paths are correct
%cd /kaggle/working/paddleocr_cham_finetune
!python3 scripts/dict_extension_v23_1.py
!python3 scripts/weight_surgery.py
!python3 scripts/generate_data_v23_1.py
%cd /kaggle/working/
""")

    add_md("## Phase 5: Config & Training")
    # Read the config and write it dynamically
    config_code = read_file('configs/rec_cham_v23_1.yml')
    # Change config paths for Kaggle
    config_code_kaggle = config_code.replace(
        "e:/Phuc's Data/Github/Cham-OCR/data/cham_dict_v23_1.txt",
        "/kaggle/working/paddleocr_cham_finetune/data/cham_dict_v23_1.txt"
    ).replace(
        "e:/Phuc's Data/Github/Cham-OCR/data/output_v23_1_temp/rec_cham_best_model/iter_epoch_200",
        "/kaggle/working/paddleocr_cham_finetune/data/output_v23_1_temp/rec_cham_best_model/iter_epoch_200"
    ).replace(
        "e:/Phuc's Data/Github/Cham-OCR/data/output/rec_cham_v23_1",
        "/kaggle/working/paddleocr_cham_finetune/data/output/rec_cham_v23_1"
    ).replace(
        "e:/Phuc's Data/Github/Cham-OCR/data/cham_synthetic_v23_1",
        "/kaggle/working/paddleocr_cham_finetune/data/cham_synthetic_v23_1"
    )
    
    add_code(f"%%writefile /kaggle/working/paddleocr_cham_finetune/configs/rec_cham_v23_1.yml\n{config_code_kaggle}")
    
    add_code("""# Execute training
import paddle
gpu_count = paddle.device.cuda.device_count()
print("GPU Count:", gpu_count)

if gpu_count > 1:
    !cd /kaggle/working/PaddleOCR && python3 -m paddle.distributed.launch --gpus '0,1' tools/train.py -c /kaggle/working/paddleocr_cham_finetune/configs/rec_cham_v23_1.yml -o Global.use_gpu=True
else:
    !cd /kaggle/working/PaddleOCR && python3 tools/train.py -c /kaggle/working/paddleocr_cham_finetune/configs/rec_cham_v23_1.yml -o Global.use_gpu=True
""")

    add_md("## Phase 6: Export Inference Model")
    add_code("""# Export V23.1 model to inference format
!cd /kaggle/working/PaddleOCR && python3 tools/export_model.py -c /kaggle/working/paddleocr_cham_finetune/configs/rec_cham_v23_1.yml -o Global.pretrained_model=/kaggle/working/paddleocr_cham_finetune/data/output/rec_cham_v23_1/best_accuracy Global.save_inference_dir=/kaggle/working/paddleocr_cham_finetune/output/rec_cham_inference_v23_1
""")

    add_md("## Phase 7: Run Evaluation & Comparison")
    add_code("""# Run the evaluation script inside the project directory
%cd /kaggle/working/paddleocr_cham_finetune
!python3 scripts/evaluate_v23_1.py
%cd /kaggle/working/
""")

    add_md("## Phase 8: Package Results & Generate Model Card")
    # We will generate model card inside the notebook
    add_code("""import os, hashlib

# Read dict and checkpoint hash
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

dict_hash = sha256('/kaggle/working/paddleocr_cham_finetune/data/cham_dict_v23_1.txt')
model_hash = sha256('/kaggle/working/paddleocr_cham_finetune/output/rec_cham_inference_v23_1/inference.pdiparams')

# Load eval stats
with open('/kaggle/working/paddleocr_cham_finetune/output/v23_1_training/eval_summary.json', 'r') as f:
    eval_stats = json.load(f)

card_content = f\"\"\"# Model Card — rec_cham_inference_v23_1

## Model Overview
V23.1 OCR Recognizer patch model. Fine-tuned from V23 baseline to add Latin digits and common punctuation.

## Architecture
- Backbone: PPLCNetV3
- Neck: SVTR Neck
- Head: MultiHead (CTC + NRTR)
- Output size: 103 classes (CTC), 107 classes (GTC)

## Metadata
- Base Checkpoint: V23 best accuracy (iter_epoch_200)
- Dict Hash: {dict_hash}
- inference.pdiparams Hash: {model_hash}
- Training dataset size: 7200 images (70% normal Cham, 20% mixed, 10% hard examples)
- Epochs: 20 epochs

## Evaluation Summary
- Overall CER: {eval_stats['overall']['cer_v23_1']:.2%} (V23 Baseline: {eval_stats['overall']['cer_v23']:.2%})
- Cham-only CER: {eval_stats['overall']['cham_only_cer_v23_1']:.2%} (V23 Baseline: {eval_stats['overall']['cham_only_cer_v23']:.2%})
- Latin Digit Accuracy: {eval_stats['overall']['latin_digit_accuracy']:.2%}
- Punctuation Accuracy: {eval_stats['overall']['punctuation_accuracy']:.2%}
- Bracket Accuracy: {eval_stats['overall']['bracket_accuracy']:.2%}
- Regression vs V23: {eval_stats['overall']['cham_regression']:.2%}

## Added Characters
0 1 2 3 4 5 6 7 8 9 [ ] ( ) . , : ; - /
\"\"\"

with open('/kaggle/working/paddleocr_cham_finetune/output/v23_1_training/v23_1_model_card.md', 'w', encoding='utf-8') as f:
    f.write(card_content)
    
print("Model card generated successfully.")
""")

    # Zip outputs
    add_code("""import shutil
pack_dir = '/kaggle/working/v23_1_training_evidence'
os.makedirs(pack_dir, exist_ok=True)

# Copy files
output_src = '/kaggle/working/paddleocr_cham_finetune/output/v23_1_training'
shutil.copy2(os.path.join(output_src, 'v23_checkpoint_inventory.json'), os.path.join(pack_dir, 'v23_checkpoint_inventory.json'))
shutil.copy2(os.path.join(output_src, 'dict_extension_report.json'), os.path.join(pack_dir, 'dict_extension_report.json'))
shutil.copy2(os.path.join(output_src, 'head_extension_report.json'), os.path.join(pack_dir, 'head_extension_report.json'))
shutil.copy2(os.path.join(output_src, 'train_label_stats.json'), os.path.join(pack_dir, 'train_label_stats.json'))
shutil.copy2(os.path.join(output_src, 'punctuation_coverage_stats.json'), os.path.join(pack_dir, 'punctuation_coverage_stats.json'))
shutil.copy2(os.path.join(output_src, 'eval_summary.json'), os.path.join(pack_dir, 'eval_summary.json'))
shutil.copy2(os.path.join(output_src, 'v23_vs_v23_1_comparison.md'), os.path.join(pack_dir, 'v23_vs_v23_1_comparison.md'))
shutil.copy2(os.path.join(output_src, 'predictions_v23_1.jsonl'), os.path.join(pack_dir, 'predictions_v23_1.jsonl'))
shutil.copy2(os.path.join(output_src, 'v23_1_model_card.md'), os.path.join(pack_dir, 'v23_1_model_card.md'))
shutil.copy2('/kaggle/working/paddleocr_cham_finetune/configs/rec_cham_v23_1.yml', os.path.join(pack_dir, 'rec_cham_v23_1.yml'))

# Copy train logs
shutil.copy2('/kaggle/working/paddleocr_cham_finetune/data/output/rec_cham_v23_1/train.log', os.path.join(pack_dir, 'train_log.txt'))

# Make zip archive
shutil.make_archive('/kaggle/working/v23_1_training_evidence', 'zip', pack_dir)
# Copy zip to expected output location
os.makedirs('/kaggle/working/paddleocr_cham_finetune/output/v23_1_training', exist_ok=True)
shutil.copy2('/kaggle/working/v23_1_training_evidence.zip', '/kaggle/working/paddleocr_cham_finetune/output/v23_1_training/v23_1_training_evidence.zip')
print("✅ Output packaged successfully to /kaggle/working/paddleocr_cham_finetune/output/v23_1_training/v23_1_training_evidence.zip")
""")

    # Construct the final notebook dict
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    # Save notebook to disk
    with open('paddleocr_cham_finetune.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print("✅ Overwritten paddleocr_cham_finetune.ipynb with V23.1 patch training steps!")

if __name__ == '__main__':
    main()
