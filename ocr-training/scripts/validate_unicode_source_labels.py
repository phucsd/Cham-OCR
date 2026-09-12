import os
import sys
import re

# Force UTF-8 stdout encoding on Windows
if sys.stdout and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Định nghĩa các tập ký tự tiếng Chăm
PRE_SIGNS = set('ꨯꨰ')
diacritics = set(chr(c) for c in range(0xAA29, 0xAA37)) | set(chr(c) for c in range(0xAA40, 0xAA4E))
FINAL_SIGNS = set('ꩀꩃꩌꩍꩆꩉꩊꩂꩅ')
MEDIAL_SIGNS = set('ꨴꨵꨳꨶ')
consonants = set("ꨆꨇꨈꨉꨊꨋꨌꨍꨎꨏꨐꨑꨒꨓꨔꨕꨖꨗꨘꨙꨚꨛꨜꨝꨞꨟꨠꨡꨢꨣꨤꨥꨦꨧꨨꨀꨁꨂꨃꨄꨅ")

COMBINING_MARKS = PRE_SIGNS | diacritics | FINAL_SIGNS | MEDIAL_SIGNS
HEAL_REGEX = re.compile(r'\s+([ꨯꨰꨴꨵꨳꨶꩀꩃꩌꩍꩆꩉꩊꩂꩅ\uAA29-\uAA36\uAA40-\uAA4D])')

# Whitelist các dòng biểu đồ diacritic hoặc ký tự diacritic rời rạc được chấp nhận cho mục đích chẩn đoán/vocabulary
WHITELIST_DIAGNOSTIC = {
    'ꩀ', 'ꨩ', 'ꨪ', 'ꨫ', 'ꨬ', 'ꨭ', 'ꨮ', 'ꨯ', 'ꨰ', 'ꨱ', 'ꨲ', 'ꨳ', 'ꨴ', 'ꨵ', 'ꨶ', 'ꩂ', 'ꩃ', 'ꩄ', 'ꩅ', 'ꩆ', 'ꩇ', 'ꩈ', 'ꩉ', 'ꩊ', 'ꩋ', 'ꩌ', 'ꩍ',
    'ꨰ ꨱ ꨲ ꨳ ꨴ ꨵ ꨶ ꩀ',
    'ꩂ ꩃ ꩄ ꩅ ꩆ ꩇ ꩈ ꩉ',
    'ꩊ ꩋ ꩌ ꩍ ꩝',
    'ꨯꨱ ꨯꨱꩀ ꨯꨱꩍ',
    'ꩌ ꩃ ꩀ ꩍ'
}

def check_string(s):
    # Chuẩn hóa khoảng trắng và kiểm tra whitelist
    s_norm = " ".join(s.split())
    if s_norm in WHITELIST_DIAGNOSTIC:
        return True, ""

    # Quy tắc: Nếu dòng chứa combining mark thì bắt buộc phải có ít nhất một phụ âm nền (consonant)
    has_combining = any(c in COMBINING_MARKS for c in s)
    has_consonant = any(c in consonants for c in s)
    
    if has_combining and not has_consonant:
        return False, "Dòng chứa dấu diacritic/combining mark nhưng không có phụ âm nền (base consonant)"
        
    # Bỏ qua dòng chỉ chứa toàn ký hiệu/dấu câu/số không có phụ âm lẫn diacritic
    if not has_consonant:
        return True, ""
        
    # Áp dụng tự sửa lỗi khoảng trắng trước combining mark
    s = HEAL_REGEX.sub(r'\1', s)
    words = s.split()
    for w in words:
        if not w:
            continue
        # Quy tắc: Không từ có độ dài > 1 nào được phép bắt đầu bằng combining mark hoặc pre-sign
        if len(w) > 1 and w[0] in COMBINING_MARKS:
            return False, f"Từ '{w}' bắt đầu bằng combining mark '{w[0]}' (U+{ord(w[0]):04X})"
    return True, ""

def main():
    print("🧹 Bắt đầu kiểm duyệt và làm sạch nhãn Unicode nguồn...")
    errors_found = 0
    
    # 1. Quét CLUSTER_DRILLS trong generate_data.py
    try:
        import scripts.generate_data as gd
        print("  - Quét CLUSTER_DRILLS...")
        for idx, drill in enumerate(gd.CLUSTER_DRILLS):
            ok, err_msg = check_string(drill)
            if not ok:
                print(f"    ❌ LỖI trong CLUSTER_DRILLS[{idx}]: '{drill}' -> {err_msg}")
                errors_found += 1
    except Exception as e:
        print(f"    ❌ LỖI import hoặc chạy generate_data.py: {e}")
        errors_found += 1
        
    # 2. Quét data/corpus/cham_text.txt
    corpus_file = os.path.join(PROJECT_ROOT, "data", "corpus", "cham_text.txt")
    if os.path.exists(corpus_file):
        print(f"  - Quét corpus tại {corpus_file}...")
        with open(corpus_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                clean = line.strip()
                if not clean or clean.startswith("#"):
                    continue
                ok, err_msg = check_string(clean)
                if not ok:
                    print(f"    ❌ LỖI trong corpus dòng {idx+1}: '{clean}' -> {err_msg}")
                    errors_found += 1
    else:
        print(f"  - ⚠️ Cảnh báo: Không tìm thấy corpus tại {corpus_file}")
        
    # 3. Quét data/hard_examples_v5.txt
    hard_file = os.path.join(PROJECT_ROOT, "data", "hard_examples_v5.txt")
    if os.path.exists(hard_file):
        print(f"  - Quét hard examples tại {hard_file}...")
        with open(hard_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                clean = line.strip()
                if not clean or clean.startswith("#"):
                    continue
                ok, err_msg = check_string(clean)
                if not ok:
                    print(f"    ❌ LỖI trong hard examples dòng {idx+1}: '{clean}' -> {err_msg}")
                    errors_found += 1
    else:
        print(f"  - ⚠️ Cảnh báo: Không tìm thấy hard examples tại {hard_file}")
        
    # Kết luận
    if errors_found > 0:
        print(f"\n❌ KIỂM DUYỆT THẤT BẠI: Phát hiện {errors_found} nhãn nguồn lỗi nghi ngờ Visual Order!")
        sys.exit(1)
    else:
        print("\n✅ KIỂM DUYỆT HOÀN TẤT THÀNH CÔNG: Không phát hiện lỗi Unicode nhãn nguồn!")
        sys.exit(0)

if __name__ == "__main__":
    main()
