import os
import sys
import json
import hashlib

# Configure path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.generate_data import unicode_to_visual, visual_to_unicode

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def test_vol_roundtrip():
    # 1. Defined cases from user requirements
    test_cases = [
        "ꨆꨯ",
        "ꨆꨰ",
        "ꨆꨴ",
        "ꨆꨳꨯ",
        "ꨆꨵꨯ",
        "ꨆꨶꨯ",
        "ꨒꨴꨮꩌ",
        "ꨙꨶꨮꩄ",
        "ꨆ ꨯ",
        "ꨆ꩝ꨯ",
        "ꨆ꩜ꨯ",
        "ꨆ꩞ꨯ",
        "ꨆ꩟ꨯ",
        "꩐ ꩑ ꩒ ꩓ ꩔ ꩕ ꩖ ꩗ ꩘ ꩙",
        "ꨆꨴꨯ",
        "ꨆꨴꨰ",
        "ꨆꨵꨴꨯ",
        "ꨆꨶꨴꨯ",
        "ꨎꨳꨯꨮꩆ",
        "ꨆꨴꨯꨱꩃ",
        "ꨄꨯꩆ",
        "ꨃꩍ",
        "ꨅꨩ",
        "ꨁꨪꩆ",
        "ꨅꩃ",
        "ꨂꨩ"
    ]
    
    special_failures = []
    special_results = []
    
    print("🧪 Running special test cases...")
    for case in test_cases:
        visual = unicode_to_visual(case)
        roundtrip = visual_to_unicode(visual)
        passed = (roundtrip == case)
        
        special_results.append({
            "gt": case,
            "visual": visual,
            "roundtrip": roundtrip,
            "passed": passed
        })
        
        if not passed:
            special_failures.append({
                "gt": case,
                "visual": visual,
                "roundtrip": roundtrip
            })
            print(f"❌ FAIL: '{case}' -> visual: '{visual}' -> roundtrip: '{roundtrip}'")
        else:
            print(f"✅ PASS: '{case}' -> visual: '{visual}' -> roundtrip: '{roundtrip}'")
            
    # 2. Corpus verification (data/corpus/cham_text.txt)
    corpus_path = os.path.join(PROJECT_ROOT, "data", "corpus", "cham_text.txt")
    corpus_failures = []
    corpus_total = 0
    corpus_passed = 0
    
    if os.path.exists(corpus_path):
        print(f"\n📂 Verifying entire corpus file: {corpus_path}...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                corpus_total += 1
                clean_line = line.strip()
                if not clean_line:
                    corpus_passed += 1
                    continue
                
                # Check visual roundtrip
                visual = unicode_to_visual(clean_line)
                roundtrip = visual_to_unicode(visual)
                if roundtrip == clean_line:
                    corpus_passed += 1
                else:
                    corpus_failures.append({
                        "line_number": line_idx + 1,
                        "original": clean_line,
                        "visual": visual,
                        "roundtrip": roundtrip
                    })
    else:
        print(f"\n⚠️ Warning: Corpus file not found at {corpus_path}")
        
    # Write output directories
    evidence_dir = os.path.join(PROJECT_ROOT, "output", "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    
    # Write vol_roundtrip_failures.txt
    failures_txt_path = os.path.join(evidence_dir, "vol_roundtrip_failures.txt")
    with open(failures_txt_path, "w", encoding="utf-8") as f:
        if special_failures or corpus_failures:
            f.write("=== SPECIAL TEST CASES FAILURES ===\n")
            for fail in special_failures:
                f.write(f"Original:  {fail['gt']}\nVisual:    {fail['visual']}\nRoundtrip: {fail['roundtrip']}\n\n")
            
            f.write("\n=== CORPUS FAILURES ===\n")
            for fail in corpus_failures:
                f.write(f"Line {fail['line_number']}:\nOriginal:  {fail['original']}\nVisual:    {fail['visual']}\nRoundtrip: {fail['roundtrip']}\n\n")
        else:
            f.write("NO VOL ROUNDTRIP FAILURES FOUND! 100% SUCCESS.\n")
            
    # Write vol_roundtrip_report.json
    report_json_path = os.path.join(evidence_dir, "vol_roundtrip_report.json")
    report_data = {
        "special_cases": special_results,
        "special_cases_summary": {
            "total": len(test_cases),
            "passed": len(test_cases) - len(special_failures),
            "failed": len(special_failures)
        },
        "corpus_summary": {
            "total_lines": corpus_total,
            "passed_lines": corpus_passed,
            "failed_lines": len(corpus_failures),
            "success_rate_percent": (corpus_passed / corpus_total * 100) if corpus_total > 0 else 0
        }
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n📊 Roundtrip Test Completed.")
    print(f"   - Special Cases: {report_data['special_cases_summary']['passed']}/{report_data['special_cases_summary']['total']} Passed")
    print(f"   - Corpus Lines: {report_data['corpus_summary']['passed_lines']}/{report_data['corpus_summary']['total_lines']} Passed ({report_data['corpus_summary']['success_rate_percent']:.2f}%)")
    print(f"   - Report saved to: {report_json_path}")
    print(f"   - Failures saved to: {failures_txt_path}")

if __name__ == "__main__":
    test_vol_roundtrip()
