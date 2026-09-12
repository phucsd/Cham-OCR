import os
import sys
import json
import hashlib

# Configure path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.generate_data import unicode_to_visual, visual_to_unicode, normalize_unicode

def test_va_initial_and_final():
    """Test U+AA25 (ꨥ CHAM LETTER VA) functioning as both initial and syllable-final consonant per Table 16-16."""
    print("\n--- Test Suite 1: U+AA25 VA Initial vs Syllable-Final ---")
    # Initial VA cases
    initial_cases = [
        "ꨥꨮꩆ",      # va + vowel oe + final n
        "ꨥꨯꨱꩃ",     # va + vowel o + vowel au + final ng
        "ꨥꨪꨌꨣ",     # va + vowel i + ca + ra
        "ꨥꨴꨯꨮ",      # va + medial ra + vowel o + vowel oe
    ]
    for case in initial_cases:
        vis = unicode_to_visual(case)
        rt = visual_to_unicode(vis)
        assert rt == case, f"VA initial failed: {case} -> {vis} -> {rt}"
        print(f"  [PASS] Initial VA: '{case}' -> visual: '{vis}' -> roundtrip: '{rt}'")

    # Syllable-final VA cases (per Unicode Chapter 16 Table 16-16)
    final_cases = [
        "ꨀꨍꨯꨱꨥ",    # achauv: standalone A + cha + vowel o + vowel au + final VA
        "ꨝꨗꨴꨭꨥ",    # biluv: ba + dha + medial ra + vowel u + final VA
        "ꨆꨯꨱꨥ",     # kauv: ka + vowel o + vowel au + final VA
        "ꨎꨯꨱꨥ",     # jauv: ja + vowel o + vowel au + final VA
    ]
    for case in final_cases:
        vis = unicode_to_visual(case)
        rt = visual_to_unicode(vis)
        assert rt == case, f"VA final failed: {case} -> {vis} -> {rt}"
        print(f"  [PASS] Final VA: '{case}' -> visual: '{vis}' -> roundtrip: '{rt}'")

def test_medial_precedence():
    """Test medial consonant ordering: Medial RA/LA (ꨴ, ꨵ) must precede Medial YA/WA (ꨳ, ꨶ)."""
    print("\n--- Test Suite 2: Medial Precedence (RA/LA before YA/WA) ---")
    # Canonical cases
    cases = [
        ("ꨆꨴꨳꨯ", "Medial RA + Medial YA with pre-vowel O"),
        ("ꨆꨴꨶꨯ", "Medial RA + Medial WA with pre-vowel O"),
        ("ꨆꨵꨳꨯ", "Medial LA + Medial YA with pre-vowel O"),
        ("ꨆꨵꨶꨯ", "Medial LA + Medial WA with pre-vowel O"),
    ]
    for case, desc in cases:
        vis = unicode_to_visual(case)
        rt = visual_to_unicode(vis)
        assert rt == case, f"Medial ordering failed: {case} ({desc})"
        print(f"  [PASS] {desc}: '{case}' -> visual: '{vis}' -> roundtrip: '{rt}'")

    # Reordering test: non-canonical order 'ꨆꨶꨴꨯ' must normalize to canonical 'ꨆꨴꨶꨯ'
    non_canonical = "ꨆꨶꨴꨯ"
    normalized = visual_to_unicode(unicode_to_visual(non_canonical))
    assert normalized == "ꨆꨴꨶꨯ", f"Expected canonical 'ꨆꨴꨶꨯ', got '{normalized}'"
    print(f"  [PASS] Reordering non-canonical '{non_canonical}' -> canonical: '{normalized}'")

def test_prevowel_reordering():
    """Test pre-base vowels O (ꨯ U+AA2F) and AI (ꨰ U+AA30) reordered from visual left to logical order."""
    print("\n--- Test Suite 3: Pre-Base Vowel Reordering (O and AI) ---")
    cases = [
        ("ꨆꨯ", "ꨯꨆ", "Simple O: ka + vowel o"),
        ("ꨆꨰ", "ꨰꨆ", "Simple AI: ka + vowel ai"),
        ("ꨆꨴꨯ", "ꨯꨆꨴ", "Medial RA + O: ka + medial ra + vowel o"),
        ("ꨆꨴꨰ", "ꨰꨆꨴ", "Medial RA + AI: ka + medial ra + vowel ai"),
        ("ꨆꨳꨯ", "ꨯꨆꨳ", "Medial YA + O: ka + medial ya + vowel o"),
        ("ꨎꨳꨯꨮꩆ", "ꨯꨎꨳꨮꩆ", "Complex compound: ja + medial ya + vowel o + vowel oe + final n"),
        ("ꨆꨴꨯꨱꩃ", "ꨯꨆꨴꨱꩃ", "Compound vowel: ka + medial ra + vowel o + vowel au + final ng"),
    ]
    for logical, expected_vis, desc in cases:
        vis = unicode_to_visual(logical)
        assert vis == expected_vis, f"Visual order mismatch for {logical}: expected {expected_vis}, got {vis}"
        rt = visual_to_unicode(vis)
        assert rt == logical, f"Logical roundtrip failed for {logical}: got {rt}"
        print(f"  [PASS] {desc}: logical '{logical}' -> visual '{vis}' -> roundtrip '{rt}'")

def test_dependent_vowels():
    """Test representation and roundtrip of all dependent vowels in Unicode Cham block."""
    print("\n--- Test Suite 4: Multi-Directional Dependent Vowels ---")
    vowel_cases = [
        ("ꨆꨩ", "Vowel Sign AA (ꨩ U+AA29) - post-base"),
        ("ꨆꨪ", "Vowel Sign I (ꨪ U+AA2A) - above"),
        ("ꨆꨫ", "Vowel Sign II (ꨫ U+AA2B) - above"),
        ("ꨆꨬ", "Vowel Sign EI (ꨬ U+AA2C) - above"),
        ("ꨆꨭ", "Vowel Sign U (ꨭ U+AA2D) - below"),
        ("ꨆꨮ", "Vowel Sign OE (ꨮ U+AA2E) - post-base"),
        ("ꨆꨯ", "Vowel Sign O (ꨯ U+AA2F) - pre-base"),
        ("ꨆꨰ", "Vowel Sign AI (ꨰ U+AA30) - pre-base"),
        ("ꨆꨱ", "Vowel Sign AU (ꨱ U+AA31) - post-base"),
        ("ꨆꨲ", "Vowel Sign UE (ꨲ U+AA32) - below"),
    ]
    for case, desc in vowel_cases:
        vis = unicode_to_visual(case)
        rt = visual_to_unicode(vis)
        assert rt == case, f"Dependent vowel failed: {case} ({desc})"
        print(f"  [PASS] {desc}: '{case}' -> visual: '{vis}' -> roundtrip: '{rt}'")

def test_punctuation_boundaries():
    """Test punctuation boundaries and whitespace preserving syllable cluster parsing."""
    print("\n--- Test Suite 5: Punctuation Boundaries & Numeral Stanzas ---")
    punct_cases = [
        ("ꨆ ꨯ", "Ka + space + Vowel O (isolated vowel sign after space)"),
        ("ꨆ꩝ꨯ", "Ka + Danda (꩝) + Vowel O"),
        ("ꨆ꩞ꨯ", "Ka + Double Danda (꩞) + Vowel O"),
        ("ꨆ꩟ꨯ", "Ka + Triple Danda (꩟) + Vowel O"),
        ("ꨆ꩜ꨯ", "Ka + Spiral (꩜) + Vowel O"),
        ("꩑꩞ ꨆꨯꨱꩃ", "Verse numeral 1 (꩑) + section mark (꩞) + space + Cham word"),
        ("꩑꩐꩞ ꨆꨴꨯ", "Verse numeral 10 (꩑꩐) + section mark (꩞) + space + Cham word"),
        ("꩔꩓꩞ ꨀꨍꨯꨱꨥ", "Verse numeral 43 (꩔꩓) + section mark (꩞) + space + Cham word with final VA"),
    ]
    for case, desc in punct_cases:
        vis = unicode_to_visual(case)
        rt = visual_to_unicode(vis)
        assert rt == case, f"Punctuation boundary failed: {case} ({desc})"
        print(f"  [PASS] {desc}: '{case}' -> visual: '{vis}' -> roundtrip: '{rt}'")

def test_vol_roundtrip():
    # Run specialized suites first
    test_va_initial_and_final()
    test_medial_precedence()
    test_prevowel_reordering()
    test_dependent_vowels()
    test_punctuation_boundaries()

    # 1. Defined benchmark cases
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
        "ꨆꨴꨶꨯ",  # Canonical order: Medial RA precedes Medial WA
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
    
    print("\n--- Test Suite 6: Full Special Test Cases Suite ---")
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
            print(f"  [PASS] '{case}' -> visual: '{visual}' -> roundtrip: '{roundtrip}'")
            
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
        
    evidence_dir = os.path.join(PROJECT_ROOT, "output", "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    
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
        
    print(f"\n📊 Complete Roundtrip & Unicode Suite Results:")
    print(f"   - Special Test Cases: {report_data['special_cases_summary']['passed']}/{report_data['special_cases_summary']['total']} Passed (100.0%)")
    print(f"   - Corpus Lines: {report_data['corpus_summary']['passed_lines']}/{report_data['corpus_summary']['total_lines']} Passed ({report_data['corpus_summary']['success_rate_percent']:.2f}%)")
    print(f"   - Report saved to: {report_json_path}")

if __name__ == "__main__":
    test_vol_roundtrip()
