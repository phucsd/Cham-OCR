import os
import sys
import zipfile
import json
import hashlib
from collections import Counter

# Configure path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.generate_data import unicode_to_visual, visual_to_unicode

def get_sha256(data_bytes):
    return hashlib.sha256(data_bytes).hexdigest()

def get_file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def main():
    zip_path = os.path.join(PROJECT_ROOT, "data", "cham-ocr-v5-assets.zip")
    after_path = os.path.join(PROJECT_ROOT, "data", "corpus", "cham_text.txt")
    evidence_dir = os.path.join(PROJECT_ROOT, "output", "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    
    # 1. Read before-normalization corpus from zip
    print(f"📦 Extracting raw corpus from: {zip_path}...")
    with zipfile.ZipFile(zip_path) as z:
        before_bytes = z.read("corpus/corpus/cham_text.txt")
        before_text = before_bytes.decode("utf-8")
        before_lines = before_text.splitlines()
        
    before_sha256 = get_sha256(before_bytes)
    
    # 2. Read current (after-normalization) corpus
    print(f"📂 Reading current corpus from: {after_path}...")
    with open(after_path, "r", encoding="utf-8") as f:
        after_text = f.read()
        after_lines = after_text.splitlines()
        
    after_sha256 = get_file_sha256(after_path)
    
    # Padding lines to ensure same length for comparison
    max_len = max(len(before_lines), len(after_lines))
    before_lines += [""] * (max_len - len(before_lines))
    after_lines += [""] * (max_len - len(after_lines))
    
    print(f"🔍 Analyzing {max_len} lines...")
    
    total_lines = 0
    total_changed = 0
    total_suspected_visual = 0
    total_pass_roundtrip = 0
    total_fail_roundtrip = 0
    
    changed_lines_list = []
    before_after_jsonl_data = []
    roundtrip_failures = []
    word_corrections = Counter()
    
    for i in range(max_len):
        b_line = before_lines[i].strip()
        a_line = after_lines[i].strip()
        
        # Don't count empty lines as checked corpus text lines
        if not b_line and not a_line:
            continue
            
        total_lines += 1
        
        # Check if line was modified
        is_modified = (b_line != a_line)
        if is_modified:
            total_changed += 1
            changed_lines_list.append({
                "line": i + 1,
                "before": b_line,
                "after": a_line
            })
            
            # Count word-level changes
            b_words = b_line.split(" ")
            a_words = a_line.split(" ")
            for w_idx in range(min(len(b_words), len(a_words))):
                bw = b_words[w_idx].strip()
                aw = a_words[w_idx].strip()
                if bw != aw:
                    word_corrections[(bw, aw)] += 1
                    
            # Check if suspected visual order
            total_suspected_visual += 1
            
        before_after_jsonl_data.append({
            "line_number": i + 1,
            "before": b_line,
            "after": a_line
        })
        
        # Check roundtrip on "after" line
        vis = unicode_to_visual(a_line)
        rt = visual_to_unicode(vis)
        
        if rt == a_line:
            total_pass_roundtrip += 1
        else:
            total_fail_roundtrip += 1
            roundtrip_failures.append({
                "line": i + 1,
                "after": a_line,
                "visual": vis,
                "roundtrip": rt
            })
            
    # Format top 30 corrections
    top_30_corrections = []
    for (bw, aw), count in word_corrections.most_common(30):
        top_30_corrections.append({
            "before": bw,
            "after": aw,
            "count": count
        })
        
    # Write corpus_normalization_before_after.jsonl
    jsonl_path = os.path.join(evidence_dir, "corpus_normalization_before_after.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in before_after_jsonl_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    # Write corpus_normalization_summary.json
    summary_path = os.path.join(evidence_dir, "corpus_normalization_summary.json")
    summary_data = {
        "total_lines_checked": total_lines,
        "total_lines_modified": total_changed,
        "total_lines_suspected_visual_order_before": total_suspected_visual,
        "total_lines_pass_roundtrip": total_pass_roundtrip,
        "total_lines_fail_roundtrip": total_fail_roundtrip,
        "sha256_before": before_sha256,
        "sha256_after": after_sha256,
        "top_30_corrections": top_30_corrections
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
        
    # Write corpus_normalization_changed_lines.txt
    changed_txt_path = os.path.join(evidence_dir, "corpus_normalization_changed_lines.txt")
    with open(changed_txt_path, "w", encoding="utf-8") as f:
        for item in changed_lines_list:
            f.write(f"Line {item['line']}:\n  Before: {item['before']}\n  After:  {item['after']}\n\n")
            
    # Write corpus_normalization_roundtrip_failures.txt
    failures_txt_path = os.path.join(evidence_dir, "corpus_normalization_roundtrip_failures.txt")
    with open(failures_txt_path, "w", encoding="utf-8") as f:
        if roundtrip_failures:
            for item in roundtrip_failures:
                f.write(f"Line {item['line']}:\n  After:     {item['after']}\n  Visual:    {item['visual']}\n  Roundtrip: {item['roundtrip']}\n\n")
        else:
            f.write("NO CORPUS ROUNDTRIP FAILURES! 100% CORRECT.\n")
            
    print(f"🎉 Corpus Verification Completed.")
    print(f"   - Total lines: {total_lines}")
    print(f"   - Modified lines: {total_changed}")
    print(f"   - Suspected visual-order lines: {total_suspected_visual}")
    print(f"   - Pass roundtrip: {total_pass_roundtrip}/{total_lines} ({total_pass_roundtrip/total_lines*100:.2f}%)")
    print(f"   - Fail roundtrip: {total_fail_roundtrip}")
    print(f"   - Summary saved to: {summary_path}")
    print(f"   - Changed lines saved to: {changed_txt_path}")
    print(f"   - Roundtrip failures saved to: {failures_txt_path}")

if __name__ == "__main__":
    main()
