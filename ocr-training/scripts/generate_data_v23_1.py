import os
import sys
import json
import random
import re
import numpy as np
from PIL import Image

sys.path.insert(0, '.')
from scripts.generate_data import (
    draw_text_strip,
    apply_augmentations,
    unicode_to_visual,
    get_augmentation_pipeline,
    apply_morphological_and_resolution_noise,
    add_paper_noise,
    FontValidator
)

# 1. Setup constants
CHAM_CONSONANTS = "ꨆꨇꨈꨉꨊꨋꨌꨍꨎꨏꨐꨑꨒꨓꨔꨕꨖꨗꨘꨙꨚꨛꨜꨝꨞꨟꨠꨡꨢꨣꨤꨥꨦꨧꨨꨀꨁꨂꨃꨄꨅ"
CHAM_DIACRITICS = "ꨩꨪꨫꨬꨭꨮꨯꨰꨱꨲꨳꨴꨵꨶꩀꩂꩃꩄꩅꩆꩇꩈꩉꩊꩋꩌꩍ"
CHAM_PUNCT = "꩜꩝꩞꩟"
CHAM_DIGITS = "꩐꩑꩒꩓꩔꩕꩖꩗꩘꩙"

# Target Latin digits and punctuation
LATIN_DIGITS = "0123456789"
PUNCTUATION_CHARS = "[]().,;:/-"

HARD_EXAMPLES_TEMPLATES = [
    "ꨆꨴꨯꩃ (1)",
    "ꨟꨧꨮꩌ [2]",
    "ꨓꨴꩀ, ꨕꨠꩀ.",
    "ꨀꨣꩌ 123",
    "ꨌꨰꩀ (2024)",
    "[ꨆꨴꨯꩃ] ꨙꩃ",
    "ꨚꨴꨯꨱꩃ, ꨕꨫ."
]

def load_corpus_tokens():
    corpus_path = 'data/corpus/cham_text.txt'
    if not os.path.exists(corpus_path):
        # Fallback words
        return ["ꨆꨴꨯꩃ", "ꨟꨧꨮꩌ", "ꨓꨴꩀ", "ꨕꨠꩀ", "ꨀꨣꩌ", "ꨌꨰꩀ", "ꨙꩃ", "ꨚꨴꨯꨱꩃ", "ꨕꨫ"]
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split by whitespace
    tokens = text.split()
    # Filter to retain mostly Cham words
    tokens = [t for t in tokens if len(t) >= 1]
    return tokens if tokens else ["ꨆꨴꨯꩃ", "ꨟꨧꨮꩌ"]

def main():
    PROJECT_ROOT = '.'
    fonts_dir = os.path.join(PROJECT_ROOT, 'data', 'fonts')
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'cham_synthetic_v23_1')
    os.makedirs(output_dir, exist_ok=True)
    
    # Find fonts
    font_files = [os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir) if f.endswith(('.ttf', '.otf'))]
    if not font_files:
        print("❌ No fonts found in data/fonts/")
        return
        
    validator = FontValidator(font_files)
    tokens = load_corpus_tokens()
    
    # Helper to generate random Cham word/sentence
    def gen_cham_word():
        return random.choice(tokens)
        
    def gen_cham_sentence(n_words=3):
        return ' '.join(gen_cham_word() for _ in range(n_words))
        
    # Stats trackers
    stats = {
        "latin_digit_count": 0,
        "bracket_count": 0,
        "period_count": 0,
        "comma_count": 0,
        "colon_count": 0,
        "semicolon_count": 0,
        "slash_count": 0,
        "dash_count": 0
    }
    
    def track_stats(text):
        for char in text:
            if char in LATIN_DIGITS:
                stats["latin_digit_count"] += 1
            elif char in "[]()":
                stats["bracket_count"] += 1
            elif char == '.':
                stats["period_count"] += 1
            elif char == ',':
                stats["comma_count"] += 1
            elif char == ':':
                stats["colon_count"] += 1
            elif char == ';':
                stats["semicolon_count"] += 1
            elif char == '/':
                stats["slash_count"] += 1
            elif char == '-':
                stats["dash_count"] += 1
                
    # 2. Dataset sampling functions
    def sample_normal():
        # 70% normal Cham lines
        n_words = random.randint(2, 6)
        return gen_cham_sentence(n_words)
        
    def sample_mixed():
        # 20% mixed Cham + digits / punctuation
        sentence = gen_cham_sentence(random.randint(2, 4))
        r = random.random()
        if r < 0.3:
            # append number
            sentence += f" {random.randint(0, 999)}"
        elif r < 0.6:
            # insert dash or slash
            sep = random.choice([" - ", " / ", " (2026) ", " [ref] "])
            sentence = gen_cham_sentence(random.randint(1, 2)) + sep + gen_cham_sentence(random.randint(1, 2))
        else:
            # punctuation at end or between
            sentence += random.choice([".", ",", " (ref).", " [ref];", ":"])
        return sentence
        
    def sample_hard():
        # 10% hard examples with brackets / periods / commas / line references
        r = random.random()
        if r < 0.4:
            # Choose from templates
            return random.choice(HARD_EXAMPLES_TEMPLATES)
        else:
            # Generate custom hard
            w1 = gen_cham_word()
            w2 = gen_cham_word()
            digit = random.randint(0, 99)
            hard_type = random.choice([
                f"{w1} ({digit})",
                f"{w1} [{digit}]",
                f"[{w1}] {w2}",
                f"{w1}, {w2}.",
                f"{w1} - {w2}",
                f"{w1}/{w2}",
                f"{digit}/{random.randint(1,12)}/{random.randint(2000, 2030)}"
            ])
            return hard_type

    # Generation loop
    num_samples = 8000
    n_normal = int(num_samples * 0.70) # 5600
    n_mixed = int(num_samples * 0.20)  # 1600
    n_hard = num_samples - n_normal - n_mixed # 800
    
    categories = (
        ['normal'] * n_normal +
        ['mixed'] * n_mixed +
        ['hard'] * n_hard
    )
    random.shuffle(categories)
    
    # Setup directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    transform_pipeline = get_augmentation_pipeline()
    
    train_labels = []
    val_labels = []
    
    print("Generating train and val images...")
    # We split 8000 into 7200 train and 800 val
    for idx, cat in enumerate(categories):
        if cat == 'normal':
            text = sample_normal()
        elif cat == 'mixed':
            text = sample_mixed()
        else:
            text = sample_hard()
            
        track_stats(text)
        
        # Draw image
        selected_font = random.choice(font_files)
        clean_img = draw_text_strip(text, selected_font, img_width=320, img_height=48)
        
        # Apply augmentations (only train gets heavy augs, val gets light morph noise)
        is_val = (idx % 10 == 0)
        if not is_val:
            augmented_img = apply_augmentations(clean_img, transform_pipeline)
            save_subdir = train_dir
            label_prefix = 'train/'
        else:
            img_morph = apply_morphological_and_resolution_noise(clean_img)
            img_np = np.array(img_morph)
            augmented = transform_pipeline(image=img_np)
            final_img = Image.fromarray(augmented['image'])
            augmented_img = add_paper_noise(final_img, strength='light')
            save_subdir = val_dir
            label_prefix = 'val/'
            
        img_name = f"synth_{idx:06d}.png"
        save_path = os.path.join(save_subdir, img_name)
        augmented_img.save(save_path)
        
        visual_text = unicode_to_visual(text)
        label_line = f"{label_prefix}{img_name}\t{visual_text}\n"
        
        if not is_val:
            train_labels.append(label_line)
        else:
            val_labels.append(label_line)
            
    # Save label files
    with open(os.path.join(output_dir, 'train_label.txt'), 'w', encoding='utf-8') as f:
        f.writelines(train_labels)
    with open(os.path.join(output_dir, 'val_label.txt'), 'w', encoding='utf-8') as f:
        f.writelines(val_labels)
        
    # Write train label stats
    all_lengths = [len(unicode_to_visual(l.split('\t')[1].strip())) for l in train_labels]
    arr = np.array(all_lengths)
    
    bins_range = [(1, 10), (11, 25), (26, 50), (51, 80)]
    dist = {}
    for start_b, end_b in bins_range:
        dist[f"{start_b}-{end_b}"] = int(np.sum((arr >= start_b) & (arr <= end_b)))
        
    train_label_stats = {
        "count": len(all_lengths),
        "min": int(np.min(arr)) if len(arr) > 0 else 0,
        "max": int(np.max(arr)) if len(arr) > 0 else 0,
        "mean": round(float(np.mean(arr)), 2) if len(arr) > 0 else 0.0,
        "median": round(float(np.median(arr)), 2) if len(arr) > 0 else 0.0,
        "bins": dist
    }
    
    with open('output/v23_1_training/train_label_stats.json', 'w', encoding='utf-8') as f:
        json.dump(train_label_stats, f, indent=2)
        
    with open('output/v23_1_training/punctuation_coverage_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
        
    print("✅ Synthetic dataset generated successfully.")
    print("   Train size:", len(train_labels))
    print("   Val size:", len(val_labels))
    
    # 3. Generate locked evaluation splits
    # Splits:
    # clean_cham_200 (200)
    # hard_diacritic_200 (200)
    # long_line_200 (200)
    # punctuation_digit_300 (300)
    # mixed_cham_punctuation_300 (300)
    eval_dir = os.path.join(output_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    
    def generate_eval_split(split_name, size, sampler_fn, draw_fn=None):
        labels = []
        for i in range(size):
            text = sampler_fn()
            selected_font = random.choice(font_files)
            if draw_fn:
                img = draw_fn(text, selected_font)
            else:
                img = draw_text_strip(text, selected_font, img_width=320, img_height=48)
            img_name = f"{split_name}_{i:04d}.png"
            img.save(os.path.join(eval_dir, img_name))
            visual_text = unicode_to_visual(text)
            labels.append(f"eval/{img_name}\t{visual_text}\n")
            
        with open(os.path.join(output_dir, f"{split_name}_label.txt"), 'w', encoding='utf-8') as f:
            f.writelines(labels)
        print(f"   Generated evaluation split {split_name}: {size} images.")

    # samplers for eval splits
    def sampler_clean():
        return gen_cham_sentence(random.randint(2, 4))
        
    def sampler_hard_diacritic():
        # Inject hard cluster drills
        sentence = gen_cham_sentence(random.randint(1, 2)) + " " + random.choice(["ꨨꨰꨳ", "ꨟꨧꨮꩌ", "ꨓꨌꨯꨱꩍ", "ꨝꨪꨗꩆ"]) + " " + gen_cham_sentence(random.randint(1, 2))
        return sentence
        
    def sampler_long():
        return gen_cham_sentence(random.randint(7, 10))
        
    def sampler_punctuation_digit():
        # Mostly numbers, brackets, punctuation
        digit1 = random.randint(0, 999)
        digit2 = random.randint(2000, 2030)
        puncs = random.choice([
            f"({digit1})",
            f"[{digit1}]",
            f"{digit1}/{digit2}",
            f"{digit1} - {digit2}",
            ".,;:/-"
        ])
        return puncs
        
    def sampler_mixed_cham_punc():
        return sample_mixed()

    generate_eval_split('clean_cham_200', 200, sampler_clean)
    generate_eval_split('hard_diacritic_200', 200, sampler_hard_diacritic)
    generate_eval_split('long_line_200', 200, sampler_long)
    generate_eval_split('punctuation_digit_300', 300, sampler_punctuation_digit)
    generate_eval_split('mixed_cham_punctuation_300', 300, sampler_mixed_cham_punc)

if __name__ == '__main__':
    main()
