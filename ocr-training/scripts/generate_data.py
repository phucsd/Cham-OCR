import os
import sys
import random
import cv2
import numpy as np
import json
import re
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from fontTools.ttLib import TTFont
import albumentations as A

# Configure standard streams to support UTF-8 on Windows terminals
if sys.stdout and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ==============================================================================
# 1. Cấu hình Tổ hợp Ký tự & Phân nhóm Hình học tiếng Chăm
# ==============================================================================
PRE_SIGNS = set('ꨰꨯꨴ')
VOWEL_DIACRITIC_SIGNS = set('ꨣꨤꨥꨦꨧꨨꨩꨪꨫꨬꨭꨮꨯꨰꨱꨲꨴꨵ')
MEDIAL_SIGNS = set('ꨳ')
FINAL_SIGNS = set('ꩀꩃꩌꩍꩆꩉꩊꩂꩅ')
CHAM_DIGITS = set('꩐꩑꩒꩓꩔꩕꩖꩗꩘꩙')
CHAM_PUNCT_SIGNS = set('꩜꩝꩞꩟')
PUNCT_SIGNS = set('꩜꩝꩞꩟.,;:!?')
FOCUS_CHARS = PRE_SIGNS | FINAL_SIGNS | MEDIAL_SIGNS | set('ꨱꨵꨮꨯꨰꨴ')
CONFUSION_CHARS = set('ꨰꨯꨴꨱꨳꨵꨮꩀꩃꩌꩍꩆꩉꩊꨈꨤꨠꨥꨡꨓꨩ')
consonants = set("ꨆꨇꨈꨉꨊꨋꨌꨍꨎꨏꨐꨑꨒꨓꨔꨕꨖꨗꨘꨙꨚꨛꨜꨝꨞꨟꨠꨡꨢꨣꨤꨥꨦꨧꨨꨀꨁꨂꨃꨄꨅ")
pre_signs = PRE_SIGNS
diacritics = set(chr(c) for c in range(0xAA29, 0xAA37)) | set(chr(c) for c in range(0xAA40, 0xAA4E))
COMBINING_MARKS = PRE_SIGNS | diacritics | FINAL_SIGNS | MEDIAL_SIGNS
HEAL_REGEX = re.compile(r'\s+([ꨰꨯꨴꨳꩀꩃꩌꩍꩆꩉꩊꩂꩅ\uAA29-\uAA36\uAA40-\uAA4D])')

# Danh sách từ luyện tập cụm chữ khó hay sai trên TestBench
CLUSTER_DRILLS = [
    'ꨨꨰꨳ', 'ꨈꨪꨤꨰ', 'ꨨꨯꨱꩀ',
    'ꨟꨧꨮꩌ', 'ꨟꨧꨮꩃ', 'ꨓꨌꨯꨱꩍ', 'ꨕꩀ', 'ꨀꨣꩌ',
    'ꨨꨣꩌ', 'ꨨꨣꩃ', 'ꨝꨪꨗꩆ', 'ꨟꨧꨪꩆ', 'ꨀꨳꨩ'
]

# Cache bộ đệm Font
_font_cache = {}

def get_cached_font(font_path, size):
    key = (font_path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(font_path, size)
    return _font_cache[key]

# ==============================================================================
# 2. Phân loại cấu trúc Âm tiết Chăm
# ==============================================================================

def is_focus_token(tok):
    return any(ch in FOCUS_CHARS for ch in tok)

def has_pre(tok):
    return any(ch in PRE_SIGNS for ch in tok)

def has_final(tok):
    return any(ch in FINAL_SIGNS for ch in tok)

def shape_key(tok):
    """Phân loại hình thái của token để lập nhóm drill thích hợp"""
    parts = []
    if has_pre(tok): parts.append('PRE')
    if any(ch in MEDIAL_SIGNS for ch in tok): parts.append('MED')
    if any(ch in VOWEL_DIACRITIC_SIGNS for ch in tok): parts.append('VOW')
    if has_final(tok): parts.append('FIN')
    if len(tok) <= 4: parts.append('SHORT')
    return '+'.join(parts) or 'BASE'

def token_distance_key(tok):
    """Trả về signature rút gọn của chữ để tìm từ đồng dạng (minimal pairs)"""
    return ''.join('F' if ch in FINAL_SIGNS else 'P' if ch in PRE_SIGNS else 'M' if ch in MEDIAL_SIGNS else 'V' if ch in VOWEL_DIACRITIC_SIGNS else 'B' for ch in tok)

def parse_unicode_clusters(text):
    clusters = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char in consonants:
            cluster = [char]
            i += 1
            while i < n and (text[i] in diacritics or text[i] in pre_signs):
                cluster.append(text[i])
                i += 1
            clusters.append(cluster)
        else:
            clusters.append([char])
            i += 1
    return clusters

def parse_visual_clusters(text):
    clusters = []
    i = 0
    n = len(text)
    while i < n:
        j = i
        while j < n and text[j] in pre_signs:
            j += 1
        
        if j < n and text[j] in consonants:
            cluster = list(text[i:j+1])
            i = j + 1
            while i < n and text[i] in diacritics and text[i] not in pre_signs:
                cluster.append(text[i])
                i += 1
            clusters.append(cluster)
        else:
            clusters.append([text[i]])
            i += 1
    return clusters

def unicode_to_visual_cluster(cluster):
    has_consonant = any(c in consonants for c in cluster)
    if not has_consonant:
        return cluster
    extracted_pre = [c for c in cluster if c in pre_signs]
    extracted_pre.sort()
    remaining = [c for c in cluster if c not in pre_signs]
    return extracted_pre + remaining

def visual_to_unicode_cluster(cluster):
    has_consonant = any(c in consonants for c in cluster)
    if not has_consonant:
        return cluster
        
    base_consonant = [c for c in cluster if c in consonants]
    extracted_pre = [c for c in cluster if c in pre_signs]
    remaining_diacs = [c for c in cluster if c in diacritics and c not in pre_signs]
    
    medials = []
    others = []
    for d in remaining_diacs:
        if d in ('ꨳ', 'ꨵ', 'ꨶ'):
            medials.append(d)
        else:
            others.append(d)
            
    pre_ra = [c for c in extracted_pre if c == 'ꨴ']
    pre_vowels = [c for c in extracted_pre if c in ('ꨯ', 'ꨰ')]
    
    # Reconstruct standard relative order
    reconstructed_diacs = medials + pre_ra + pre_vowels + others
    return base_consonant + reconstructed_diacs

def unicode_to_visual(text):
    clusters = parse_unicode_clusters(text)
    vis_clusters = [unicode_to_visual_cluster(c) for c in clusters]
    return "".join("".join(c) for c in vis_clusters)

def visual_to_unicode(text):
    clusters = parse_visual_clusters(text)
    uni_clusters = [visual_to_unicode_cluster(c) for c in clusters]
    return "".join("".join(c) for c in uni_clusters)

def normalize_unicode(text):
    # Fixes internal ordering errors in Logical Order text (e.g. ꨙꨯꨳꨮ -> ꨙꨳꨯꨮ)
    clusters = parse_unicode_clusters(text)
    uni_clusters = [visual_to_unicode_cluster(c) for c in clusters]
    return "".join("".join(c) for c in uni_clusters)

# ==============================================================================
# 3. Lớp Kiểm Tra Font (Font Validation)
# ==============================================================================

class FontValidator:
    def __init__(self, font_paths):
        self.font_cmaps = {}
        for path in font_paths:
            try:
                with TTFont(path) as font:
                    cmap = font['cmap'].getBestCmap()
                    supported_chars = set(cmap.keys()) if cmap else set()
                self.font_cmaps[path] = supported_chars
                
                cham_block = range(0xAA00, 0xAA60)
                supports_cham = any(cp in supported_chars for cp in cham_block)
                if not supports_cham:
                    print(f"⚠️  CẢNH BÁO: Font {os.path.basename(path)} không hỗ trợ dải ký tự Chăm (U+AA00 - U+AA5F)!")
                else:
                    print(f"✅ Đã tải Font: {os.path.basename(path)} ({len(supported_chars)} ký tự)")
            except Exception as e:
                print(f"❌ Không thể đọc font {path}: {e}")
                self.font_cmaps[path] = set()

    def supports_string(self, font_path, text):
        supported_set = self.font_cmaps.get(font_path, set())
        for char in text:
            code_point = ord(char)
            # Bỏ qua khoảng trắng, ký tự Latin ASCII và ký tự tiếng Việt thông thường
            if char in " \t\n\r" or code_point < 128 or (0x1EA0 <= code_point <= 0x1EF9):
                continue
            if code_point not in supported_set:
                return False
        return True

# ==============================================================================
# 4. Mô phỏng nét viết hình thái & Chất lượng giấy (Augmentation Utilities)
# ==============================================================================

def add_paper_noise(pil_img, strength='medium'):
    """Thêm nhiễu giấy Gauss và các đốm mực loang rải rác đè lên chữ"""
    arr = np.array(pil_img).astype(np.float32)
    sigma = {'light': 2.0, 'medium': 5.0, 'heavy': 8.0}.get(strength, 5.0)
    arr += np.random.normal(0, sigma, arr.shape).astype(np.float32)
    
    h, w = arr.shape[:2]
    dots = {'light': 2, 'medium': 6, 'heavy': 12}.get(strength, 6)
    for _ in range(random.randint(0, dots)):
        x = random.randint(0, max(0, w-1))
        y = random.randint(0, max(0, h-1))
        rr = random.randint(1, 2)
        val = random.randint(80, 180)
        cv2.circle(arr, (x, y), rr, (val, val, val), -1)
        
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def apply_morphological_and_resolution_noise(pil_img):
    """Áp dụng các phép biến đổi mài mòn nét chữ (Erosion/Dilation) và pixelation"""
    arr = np.array(pil_img)
    
    # 1. Giả lập nhòe mực hoặc mất nét bút (Erosion/Dilation)
    if random.random() < 0.4:
        k = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            arr = cv2.erode(arr, k, iterations=1)
        else:
            arr = cv2.dilate(arr, k, iterations=1)
            
    img = Image.fromarray(arr)
    
    # 2. Giả lập độ phân giải thấp (Pixelation)
    if random.random() < 0.3:
        w, h = img.size
        scale = random.uniform(0.75, 0.95)
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
        img = small.resize((w, h), Image.Resampling.BILINEAR)
        
    return img

def get_augmentation_pipeline():
    """Tạo pipeline tăng cường ảnh bằng Albumentations"""
    transform = A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.03,
            rotate_limit=6,
            border_mode=cv2.BORDER_REPLICATE,
            p=0.7
        ),
        A.Perspective(scale=(0.01, 0.025), keep_size=True, pad_mode=cv2.BORDER_REPLICATE, p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.35),
        A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.6)
    ])
    return transform

def apply_augmentations(pil_img, transform):
    # 80% cơ hội trả về ảnh in sạch sắc nét (không nhiễu nhòe mực)
    if random.random() < 0.8:
        return pil_img
        
    # 20% cơ hội chạy tăng cường giả lập scan giấy
    img = apply_morphological_and_resolution_noise(pil_img)
    img_np = np.array(img)
    augmented = transform(image=img_np)
    final_img = Image.fromarray(augmented['image'])
    return add_paper_noise(final_img, strength='light') # Nhiễu nhẹ nhàng

# ==============================================================================
# 5. Vẽ Chữ Thích ứng Chiều rộng (Tránh Cắt Chữ)
# ==============================================================================

def create_random_background(width, height):
    bg_type = random.choice(['solid', 'gradient'])
    if bg_type == 'solid':
        bg = random.choice([(255, 255, 255), (250, 248, 240), (245, 240, 225), (253, 250, 244)])
        return Image.new('RGB', (width, height), color=bg)
    else:
        # Gradient ngang nhẹ nhạt
        c1 = np.array([random.randint(245, 255), random.randint(245, 255), random.randint(240, 250)])
        c2 = np.array([random.randint(235, 248), random.randint(235, 248), random.randint(225, 242)])
        t = np.linspace(0, 1, height).reshape(height, 1, 1)
        gradient_np = (c1 * (1 - t) + c2 * t).astype(np.uint8)
        gradient_np = np.repeat(gradient_np, width, axis=1)
        return Image.fromarray(gradient_np)

def draw_text_strip(text, font_path, font_size=22, img_width=280, img_height=40):
    # Chọn font_size thích hợp cho các từ ngắn/dài
    if len(text) <= 12 or any(ch in FINAL_SIGNS for ch in text):
        f_size = random.randint(24, 28)
    else:
        f_size = font_size
        
    font = get_cached_font(font_path, f_size)
    temp_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # Bổ sung padding biên độ lớn giúp dạy mô hình nhận diện ranh giới dấu phụ
    pad_x = random.randint(10, 26)
    pad_y = random.randint(6, 12)
    
    current_width = max(img_width, text_w + pad_x * 2)
    current_height = max(img_height, text_h + pad_y * 2)
    
    bg = create_random_background(current_width, current_height)
    draw = ImageDraw.Draw(bg)
    
    # Căn giữa chữ
    x = (current_width - text_w) / 2 - bbox[0]
    y = (current_height - text_h) / 2 - bbox[1]
    
    ink = random.choice([(0, 0, 0), (20, 20, 20), (45, 42, 38)])
    draw.text((x, y), text, font=font, fill=ink)
    
    return bg

# ==============================================================================
# 6. Quy trình Sinh Dữ Liệu từ Ngữ Liệu thực (Lexicon-Aware Pipeline)
# ==============================================================================

def generate_dataset(output_dir, font_dir, num_samples=5000, img_width=280, img_height=40, split_ratio=0.8, seed=42):
    print(f"=== Bắt đầu sinh {num_samples} mẫu dữ liệu tự động ===")
    random.seed(seed)
    np.random.seed(seed)
    
    if not os.path.exists(font_dir):
        print(f"❌ Lỗi: Thư mục font không tồn tại: {font_dir}")
        return False
        
    font_files = sorted([os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.endswith(('.ttf', '.otf'))])
    if not font_files:
        print(f"❌ LỖI NGHIÊM TRỌNG: Không tìm thấy font trong '{font_dir}'.")
        return False

    validator = FontValidator(font_files)
    
    # --- ĐỌC NGỮ LIỆU THẬT & HARD EXAMPLES ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus_file = os.path.join(PROJECT_ROOT, "data", "corpus", "cham_text.txt")
    hard_file = os.path.join(PROJECT_ROOT, "data", "hard_examples_v5.txt")
    dict_file = os.path.join(PROJECT_ROOT, "data", "cham_dict_v23.txt")
    
    # Đọc từ điển cơ bản để lọc ký tự hợp lệ
    allowed = set()
    if os.path.exists(dict_file):
        with open(dict_file, 'r', encoding='utf-8') as f:
            allowed = set(line.strip() for line in f if line.strip())
    allowed = allowed | {' '} | CHAM_DIGITS | CHAM_PUNCT_SIGNS
    
    def normalize_label(s):
        s = re.sub(r'\s+', ' ', s.strip())
        s = HEAL_REGEX.sub(r'\1', s)
        if allowed:
            s = ''.join(ch for ch in s if ch in allowed)
        # Loại bỏ các từ bắt đầu bằng combining mark đứng độc lập hoặc lỗi
        words = s.split()
        clean_words = []
        for w in words:
            if not w:
                continue
            if w[0] in COMBINING_MARKS:
                continue
            clean_words.append(w)
        return ' '.join(clean_words).strip()

    # 1. Đọc và lọc tập ngữ liệu
    corpus_lines = []
    if os.path.exists(corpus_file):
        with open(corpus_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                normalized = normalize_label(line)
                if normalized and not normalized.startswith('#'):
                    corpus_lines.append(normalized)
    
    # 2. Đọc và lọc tập hard examples
    hard_lines = []
    if os.path.exists(hard_file):
        with open(hard_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                normalized = normalize_label(line)
                if normalized:
                    hard_lines.append(normalized)
                    
    # Dự phòng nếu corpus trống
    if not corpus_lines:
        print("⚠️ Cảnh báo: Ngữ liệu thật trống. Sử dụng từ vựng mẫu mặc định.")
        corpus_lines = [normalize_label(x) for x in COMMON_CHAM_WORDS]
        corpus_lines = [x for x in corpus_lines if x]

    # Tách từ đơn (tokens)
    tokens = []
    for line in corpus_lines + hard_lines:
        tokens.extend([t for t in line.split() if t])
        
    # Tạo danh sách các token độc nhất
    unique_tokens = sorted(list(set(tokens)))
    short_tokens = [t for t in tokens if 1 <= len(t) <= 8]
    very_short_tokens = [t for t in tokens if 1 <= len(t) <= 5]
    focus_tokens = [t for t in tokens if is_focus_token(t)]
    pre_tokens = [t for t in tokens if has_pre(t)]
    final_tokens = [t for t in tokens if has_final(t)]
    pre_final_tokens = [t for t in tokens if has_pre(t) and has_final(t)]
    confusion_tokens = [t for t in tokens if any(ch in CONFUSION_CHARS for ch in t)]
    
    # Phân nhóm theo hình thái & signature
    shape_buckets = {}
    sig_buckets = {}
    for t in unique_tokens:
        shape_buckets.setdefault(shape_key(t), []).append(t)
        sig_buckets.setdefault(token_distance_key(t), []).append(t)
        
    # Chuẩn bị fallback các pool
    if not short_tokens: short_tokens = tokens[:]
    if not very_short_tokens: very_short_tokens = short_tokens[:]
    if not focus_tokens: focus_tokens = short_tokens[:]
    if not pre_tokens: pre_tokens = focus_tokens[:]
    if not final_tokens: final_tokens = focus_tokens[:]
    if not pre_final_tokens: pre_final_tokens = focus_tokens[:]
    if not confusion_tokens: confusion_tokens = focus_tokens[:]

    # --- CÁC PHƯƠNG PHÁP SINH TỪ VỰNG CHĂM-AWARE ---
    
    def choose(pool):
        return random.choice(pool)
        
    def hard_line_expanded():
        s = choose(hard_lines) if hard_lines else choose(corpus_lines)
        if random.random() < 0.35:
            s = (s + ' ' + choose(pre_final_tokens)).strip()
        if random.random() < 0.25:
            s = (choose(very_short_tokens) + ' ' + s).strip()
        return s
        
    def minimal_pair_line():
        if random.random() < 0.55:
            key = token_distance_key(choose(confusion_tokens))
            bucket = sig_buckets.get(key, [])
        else:
            key = shape_key(choose(confusion_tokens))
            bucket = shape_buckets.get(key, [])
            
        if len(bucket) >= 2:
            items = random.sample(bucket, min(len(bucket), random.randint(2, 5)))
        else:
            items = [choose(pre_tokens), choose(final_tokens), choose(very_short_tokens)]
        return ' '.join(items)
        
    def boundary_stress_line():
        pools = [pre_tokens, final_tokens, very_short_tokens, pre_final_tokens]
        items = [choose(random.choice(pools)) for _ in range(random.randint(2, 5))]
        return ' '.join(items)
        
    def pre_sign_stress_line():
        # Lấy các từ có chứa dấu đứng trước để tăng cường học vị trí liên kết CTC
        matching = [t for t in tokens if any(ch in PRE_SIGNS for ch in t)]
        if not matching:
            matching = focus_tokens
        items = [choose(matching) for _ in range(random.randint(2, 5))]
        return ' '.join(items)
        
    def cluster_drill_line():
        if random.random() < 0.3:
            drill = choose(CLUSTER_DRILLS)
            return ' '.join([drill] * random.randint(2, 4))
        return ' '.join(choose(focus_tokens) for _ in range(random.randint(2, 5)))
        
    def real_corpus_line():
        s = choose(corpus_lines)
        words = s.split()
        if len(words) > 5 and random.random() < 0.65:
            start = random.randint(0, len(words) - 2)
            s = ' '.join(words[start:start + random.randint(2, min(5, len(words) - start))])
        return s

    def sample_text(force_long=False, force_medium=False, force_short=False):
        for _ in range(50):
            r = random.random()
            if r < 0.08:
                if random.random() < 0.3:
                    s = ' '.join(list('꩑꩒꩓꩔꩕꩖꩗꩘꩙꩐'))
                else:
                    s = ' '.join(random.choice(list('꩐꩑꩒꩓꩔꩕꩖꩗꩘꩙')) for _ in range(random.randint(2, 6)))
            elif hard_lines and r < 0.53:
                s = hard_line_expanded()
            elif r < 0.68:
                s = pre_sign_stress_line()
            elif r < 0.78:
                s = minimal_pair_line()
            elif r < 0.85:
                s = boundary_stress_line()
            elif r < 0.92:
                s = cluster_drill_line()
            elif random.random() < 0.40:
                s = real_corpus_line()
                if random.random() < 0.35:
                    s = (s + ' ' + choose(final_tokens)).strip()
            else:
                s = ' '.join(choose(tokens) for _ in range(random.randint(2, 5)))
                
            if random.random() < 0.10 and s and not s.startswith('꩑'):
                p_type = random.choice(['prefix', 'postfix', 'both'])
                if p_type == 'prefix':
                    s = '꩜ ' + s
                elif p_type == 'postfix':
                    s = s + ' ' + random.choice(['꩝', '꩞', '꩟'])
                else:
                    s = '꩜ ' + s + ' ' + random.choice(['꩝', '꩞', '꩟'])
            
            vis_len = len(unicode_to_visual(s))
            if force_short:
                if vis_len <= 25:
                    return s
                else:
                    words = s.split()
                    short_s = ""
                    for w in words:
                        test_s = (short_s + " " + w).strip()
                        if len(unicode_to_visual(test_s)) <= 25:
                            short_s = test_s
                        else:
                            break
                    if short_s:
                        return short_s
                    return s[:15]
            elif force_medium:
                if 25 < vis_len <= 50:
                    return s
                elif vis_len <= 25:
                    med_s = s
                    for _ in range(10):
                        extra = sample_text(force_short=True)
                        test_s = (med_s + " " + extra).strip()
                        test_vis_len = len(unicode_to_visual(test_s))
                        if test_vis_len <= 50:
                            med_s = test_s
                            if test_vis_len > 25:
                                return med_s
                        else:
                            break
                    return med_s
                else:
                    words = s.split()
                    med_s = ""
                    for w in words:
                        test_s = (med_s + " " + w).strip()
                        test_vis_len = len(unicode_to_visual(test_s))
                        if test_vis_len <= 50:
                            med_s = test_s
                        else:
                            break
                    if len(unicode_to_visual(med_s)) > 25:
                        return med_s
                    return s[:35]
            elif force_long:
                if 50 < vis_len <= 80:
                    return s
                elif vis_len <= 50:
                    long_s = s
                    for _ in range(10):
                        extra = sample_text(force_short=True)
                        test_s = (long_s + " " + extra).strip()
                        test_vis_len = len(unicode_to_visual(test_s))
                        if test_vis_len <= 80:
                            long_s = test_s
                            if test_vis_len > 50:
                                return long_s
                        else:
                            break
                    return long_s
                else:
                    words = s.split()
                    long_s = ""
                    for w in words:
                        test_s = (long_s + " " + w).strip()
                        test_vis_len = len(unicode_to_visual(test_s))
                        if test_vis_len <= 80:
                            long_s = test_s
                        else:
                            break
                    if len(unicode_to_visual(long_s)) > 50:
                        return long_s
                    return s[:60]
            else:
                return s[:80]
        return choose(tokens)

    # --- KHỞI CHẠY TẠO FILE ---

    # Kiểm tra quy mô dữ liệu huấn luyện tối thiểu (chấp nhận 100 cho chạy thử nghiệm cục bộ)
    if num_samples < 30000 and num_samples != 100:
        raise ValueError(f"❌ Lỗi quy mô dữ liệu: Số lượng mẫu sinh ra ({num_samples}) nhỏ hơn ngưỡng tối thiểu yêu cầu (30,000 dòng)!")

    train_dir = os.path.join(output_dir, 'train')
    val_cs_dir = os.path.join(output_dir, 'val_clean_short')
    val_cl_dir = os.path.join(output_dir, 'val_clean_long')
    val_ns_dir = os.path.join(output_dir, 'val_noisy_short')
    val_nl_dir = os.path.join(output_dir, 'val_noisy_long')
    locked_dir = os.path.join(output_dir, 'locked_test')

    for d in [train_dir, val_cs_dir, val_cl_dir, val_ns_dir, val_nl_dir, locked_dir]:
        os.makedirs(d, exist_ok=True)

    transform_pipeline = get_augmentation_pipeline()

    n_val_cs = int(num_samples * 0.03)
    n_val_cl = int(num_samples * 0.03)
    n_val_ns = int(num_samples * 0.03)
    n_val_nl = int(num_samples * 0.03)
    n_locked = int(num_samples * 0.03)
    n_train = num_samples - (n_val_cs + n_val_cl + n_val_ns + n_val_nl + n_locked)

    # Chia nhỏ tập Train thành 6 sub-categories nội bộ theo tỷ lệ:
    # train_clean_short (25%), train_clean_medium (20%), train_clean_long (15%),
    # train_noisy_short (15%), train_noisy_long (15%), train_hard_examples (10%)
    n_train_cs = int(n_train * 0.25)
    n_train_cm = int(n_train * 0.20)
    n_train_cl = int(n_train * 0.15)
    n_train_ns = int(n_train * 0.15)
    n_train_nl = int(n_train * 0.15)
    n_train_he = n_train - (n_train_cs + n_train_cm + n_train_cl + n_train_ns + n_train_nl)

    categories = (
        ['train_clean_short'] * n_train_cs +
        ['train_clean_medium'] * n_train_cm +
        ['train_clean_long'] * n_train_cl +
        ['train_noisy_short'] * n_train_ns +
        ['train_noisy_long'] * n_train_nl +
        ['train_hard_examples'] * n_train_he +
        ['val_clean_short'] * n_val_cs +
        ['val_clean_long'] * n_val_cl +
        ['val_noisy_short'] * n_val_ns +
        ['val_noisy_long'] * n_val_nl
    )
    random.shuffle(categories)

    train_label_path = os.path.join(output_dir, 'train_label.txt')
    val_cs_label_path = os.path.join(output_dir, 'val_clean_short_label.txt')
    val_cl_label_path = os.path.join(output_dir, 'val_clean_long_label.txt')
    val_ns_label_path = os.path.join(output_dir, 'val_noisy_short_label.txt')
    val_nl_label_path = os.path.join(output_dir, 'val_noisy_long_label.txt')
    locked_label_path = os.path.join(output_dir, 'locked_test_label.txt')

    # Khởi tạo danh sách lưu độ dài cho cả 11 splits phục vụ vẽ histogram
    all_lengths = {
        "train_clean_short": [],
        "train_clean_medium": [],
        "train_clean_long": [],
        "train_noisy_short": [],
        "train_noisy_long": [],
        "train_hard_examples": [],
        "val_clean_short": [],
        "val_clean_long": [],
        "val_noisy_short": [],
        "val_noisy_long": [],
        "locked_test": []
    }

    # 1. Sinh các tập thông thường (Train, Val splits)
    with open(train_label_path, 'w', encoding='utf-8', newline='\n') as f_train, \
         open(val_cs_label_path, 'w', encoding='utf-8', newline='\n') as f_val_cs, \
         open(val_cl_label_path, 'w', encoding='utf-8', newline='\n') as f_val_cl, \
         open(val_ns_label_path, 'w', encoding='utf-8', newline='\n') as f_val_ns, \
         open(val_nl_label_path, 'w', encoding='utf-8', newline='\n') as f_val_nl:

         success_count = 0
         attempts = 0
         max_attempts = num_samples * 20

         while success_count < len(categories) and attempts < max_attempts:
             attempts += 1
             category = categories[success_count]

             force_short = '_short' in category
             force_medium = '_medium' in category
             force_long = '_long' in category

             text = sample_text(force_long=force_long, force_medium=force_medium, force_short=force_short)

             matching_fonts = [f for f in font_files if validator.supports_string(f, text)]
             if not matching_fonts:
                 continue

             selected_font = random.choice(matching_fonts)
             clean_img = draw_text_strip(text, selected_font, img_width=img_width, img_height=img_height)

             # Định nghĩa phân tách nhiễu theo Clean vs Noisy
             is_clean = 'clean' in category or category == 'train_hard_examples'
             is_noisy = 'noisy' in category

             if is_clean:
                 if category.startswith('train_'):
                     augmented_img = apply_augmentations(clean_img, transform_pipeline)
                 else:
                     augmented_img = clean_img
             elif is_noisy:
                 img_morph = apply_morphological_and_resolution_noise(clean_img)
                 img_np = np.array(img_morph)
                 augmented = transform_pipeline(image=img_np)
                 final_img = Image.fromarray(augmented['image'])
                 augmented_img = add_paper_noise(final_img, strength='medium' if 'long' in category else 'light')
             else:
                 augmented_img = apply_augmentations(clean_img, transform_pipeline)

             img_name = f"cham_synth_{success_count:06d}.png"
             visual_text = unicode_to_visual(text)
             vis_len = len(visual_text)

             if category.startswith('train_'):
                 sub_dir = os.path.join(train_dir, category)
                 os.makedirs(sub_dir, exist_ok=True)
                 save_path = os.path.join(sub_dir, img_name)
                 augmented_img.save(save_path)
                 f_train.write(f"train/{category}/{img_name}\t{visual_text}\n")
                 all_lengths[category].append(vis_len)
             elif category == 'val_clean_short':
                 save_path = os.path.join(val_cs_dir, img_name)
                 augmented_img.save(save_path)
                 f_val_cs.write(f"val_clean_short/{img_name}\t{visual_text}\n")
                 all_lengths[category].append(vis_len)
             elif category == 'val_clean_long':
                 save_path = os.path.join(val_cl_dir, img_name)
                 augmented_img.save(save_path)
                 f_val_cl.write(f"val_clean_long/{img_name}\t{visual_text}\n")
                 all_lengths[category].append(vis_len)
             elif category == 'val_noisy_short':
                 save_path = os.path.join(val_ns_dir, img_name)
                 augmented_img.save(save_path)
                 f_val_ns.write(f"val_noisy_short/{img_name}\t{visual_text}\n")
                 all_lengths[category].append(vis_len)
             elif category == 'val_noisy_long':
                 save_path = os.path.join(val_nl_dir, img_name)
                 augmented_img.save(save_path)
                 f_val_nl.write(f"val_noisy_long/{img_name}\t{visual_text}\n")
                 all_lengths[category].append(vis_len)

             success_count += 1

    # 2. Sinh tập locked_test bằng cách cô lập seed ngẫu nhiên
    def seed_pipeline(pipeline, s_seed):
        for t in pipeline.transforms:
            t.set_random_seed(s_seed)
            if hasattr(t, 'transforms'):
                for sub_t in t.transforms:
                    sub_t.set_random_seed(s_seed)

    r_state = random.getstate()
    np_state = np.random.get_state()

    locked_test_seed = 42
    random.seed(locked_test_seed)
    np.random.seed(locked_test_seed)

    success_locked = 0
    attempts_locked = 0
    max_attempts_locked = n_locked * 20

    with open(locked_label_path, 'w', encoding='utf-8', newline='\n') as f_locked:
        while success_locked < n_locked and attempts_locked < max_attempts_locked:
            attempts_locked += 1
            sample_seed = locked_test_seed + success_locked
            random.seed(sample_seed)
            np.random.seed(sample_seed)
            seed_pipeline(transform_pipeline, sample_seed)

            # Chia ngẫu nhiên locked test 50/50 clean/noisy và short/long
            is_clean = (success_locked % 2 == 0)
            is_long = ((success_locked // 2) % 2 == 0)

            text = sample_text(force_long=is_long, force_medium=False, force_short=not is_long)

            matching_fonts = [f for f in font_files if validator.supports_string(f, text)]
            if not matching_fonts:
                continue

            selected_font = random.choice(matching_fonts)
            clean_img = draw_text_strip(text, selected_font, img_width=img_width, img_height=img_height)

            if is_clean:
                augmented_img = clean_img
            else:
                img_morph = apply_morphological_and_resolution_noise(clean_img)
                img_np = np.array(img_morph)
                augmented = transform_pipeline(image=img_np)
                final_img = Image.fromarray(augmented['image'])
                augmented_img = add_paper_noise(final_img, strength='light')

            img_name = f"cham_locked_{success_locked:06d}.png"
            visual_text = unicode_to_visual(text)
            vis_len = len(visual_text)

            save_path = os.path.join(locked_dir, img_name)
            augmented_img.save(save_path)
            f_locked.write(f"locked_test/{img_name}\t{visual_text}\n")
            all_lengths["locked_test"].append(vis_len)

            success_locked += 1

    # Khôi phục trạng thái ngẫu nhiên để không ảnh hưởng đến code bên ngoài
    random.setstate(r_state)
    np.random.set_state(np_state)

    # Thư mục chứa bằng chứng
    evidence_dir = os.path.join(os.path.dirname(output_dir), "output", "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    # 3. Tính toán SHA256 và lưu file manifest
    import hashlib

    def file_sha256(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    locked_label_sha = file_sha256(locked_label_path)

    image_manifest = {}
    for i in range(success_locked):
        img_name = f"cham_locked_{i:06d}.png"
        img_path = os.path.join(locked_dir, img_name)
        if os.path.exists(img_path):
            image_manifest[img_name] = file_sha256(img_path)

    manifest_data = {
        "locked_test_seed": locked_test_seed,
        "locked_test_label_sha256": locked_label_sha,
        "image_checksum_manifest": image_manifest
    }

    with open(os.path.join(evidence_dir, "locked_test_manifest.json"), "w", encoding="utf-8") as f_manifest:
        json.dump(manifest_data, f_manifest, indent=4)

    # 4. Thống kê độ dài và lưu train label length analysis histogram cho tất cả 11 splits
    def compute_histogram_stats(lengths_list):
        arr = np.array(lengths_list)
        total = len(arr)
        if total == 0:
            return {
                "count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0,
                "bins": {"1-10": 0, "11-25": 0, "26-50": 0, "51-80": 0}
            }
        bins_range = [(1, 10), (11, 25), (26, 50), (51, 80)]
        dist = {}
        for start_b, end_b in bins_range:
            dist[f"{start_b}-{end_b}"] = int(np.sum((arr >= start_b) & (arr <= end_b)))
            
        return {
            "count": total,
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "mean": round(float(np.mean(arr)), 2),
            "median": round(float(np.median(arr)), 2),
            "bins": dist,
            "greater_than_25": int(np.sum(arr > 25)),
            "greater_than_80": int(np.sum(arr > 80))
        }

    splits_stats = {}
    for split_name, lengths_list in all_lengths.items():
        splits_stats[split_name] = compute_histogram_stats(lengths_list)

    # Gộp toàn bộ train lengths để tính toán tổng quát
    all_train_lengths = []
    for split_name in ["train_clean_short", "train_clean_medium", "train_clean_long", 
                        "train_noisy_short", "train_noisy_long", "train_hard_examples"]:
        all_train_lengths.extend(all_lengths[split_name])
    
    overall_train_stats = compute_histogram_stats(all_train_lengths)

    stats = {
        "dataset_metadata": {
            "total_samples": num_samples,
            "locked_test_seed": locked_test_seed,
            "label_files": {
                "train_label_sha256": file_sha256(train_label_path),
                "val_clean_short_label_sha256": file_sha256(val_cs_label_path),
                "val_clean_long_label_sha256": file_sha256(val_cl_label_path),
                "val_noisy_short_label_sha256": file_sha256(val_ns_label_path),
                "val_noisy_long_label_sha256": file_sha256(val_nl_label_path),
                "locked_test_label_sha256": locked_label_sha
            }
        },
        "overall_train_statistics": overall_train_stats,
        "splits_detailed_statistics": splits_stats
    }
    
    with open(os.path.join(evidence_dir, "train_label_length_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
        
    print(f"🎉 Đã sinh xong thành công: Train = {len(all_train_lengths)} mẫu, Val_Clean_Short = {len(all_lengths['val_clean_short'])}, Val_Clean_Long = {len(all_lengths['val_clean_long'])}, Val_Noisy_Short = {len(all_lengths['val_noisy_short'])}, Val_Noisy_Long = {len(all_lengths['val_noisy_long'])}, Locked_Test = {len(all_lengths['locked_test'])}.")
    print(f"📊 Saved train label stats to output/evidence/train_label_length_analysis.json")
    print(f"🔒 Saved locked test manifest to output/evidence/locked_test_manifest.json")
    return True

# ==============================================================================
# 7. Hàm Tạo Từ Điển Ký Tự
# ==============================================================================

def build_dict(label_file_paths, output_dict_path):
    chars = set()
    for label_path in label_file_paths:
        if not os.path.exists(label_path):
            continue
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    text = parts[1]
                    for char in text:
                        if char not in " \t\n\r":
                            chars.add(char)
                            
    sorted_chars = sorted(list(chars))
    with open(output_dict_path, 'w', encoding='utf-8', newline='\n') as f_out:
        for c in sorted_chars:
            f_out.write(c + '\n')
            
    print(f"📖 Đã tạo từ điển tại {output_dict_path} với {len(sorted_chars)} ký tự.")

# Từ vựng dự phòng khi không có corpus
COMMON_CHAM_WORDS = [
    "ꨀꨇꩉ ꨌꩌ", "Akhar Thrah", "Champa", "Việt Nam", "Campuchia",
    "ꨄꨈꨛꨞꨠ", "ꨆꨇꨉꨊꨋ", "ꨌꨍꨎꨏꨐ", "ꨑꨒꨓꨔꨕ", "ꨖꨗꨘꨙꨚ",
    "ꨛꨜꨝꨞꨟ", "ꨠꨡꨢꨣꨤ", "ꨥꨦꨧꨨꨩ", "ꨪꨫꨬꨭꨮ", "ꨯꨰꨱꨲꨳ",
    "ꨴꨵꨶ", "ꩀꩁꩂ", "ꩃꩄꩅꩆꩇ", "ꩈꩉꩊꩋꩌ"
]

if __name__ == '__main__':
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fonts_directory = os.path.join(PROJECT_ROOT, 'data', 'fonts')
    output_directory = os.path.join(PROJECT_ROOT, 'data', 'cham_synthetic_images')
    dictionary_file = os.path.join(PROJECT_ROOT, 'data', 'cham_dict_v23.txt')
    
    print("Thông tin thư mục chạy thử nghiệm:")
    print(f"- Thư mục font: {fonts_directory}")
    print(f"- Thư mục xuất ảnh: {output_directory}")
    print(f"- Đường dẫn từ điển: {dictionary_file}")
    
    has_fonts = False
    if os.path.exists(fonts_directory):
        has_fonts = any(f.endswith(('.ttf', '.otf')) for f in os.listdir(fonts_directory))
        
    if has_fonts:
        success = generate_dataset(output_directory, fonts_directory, num_samples=100)
        if success:
            print("🎉 Sinh dữ liệu thử nghiệm 100 mẫu hoàn tất thành công.")
    else:
        print("\nℹ️  Nhắc nhở: Hãy sao chép ít nhất một font tiếng Chăm vào thư mục `data/fonts/` để chạy thử nghiệm.")
