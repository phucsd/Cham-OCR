#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dictionary Generator for Cham OCR Model V25.
Expands cham_dict_v24.txt into a Unified Multilingual Dictionary (cham_dict_v25.txt)
including:
- Full Cham characters, diacritics, finals, digits, punctuation (84 tokens)
- En-dash '–', Em-dash '—', '?', '!', '"', '\''
- Latin alphabet (a-z, A-Z)
- Complete Vietnamese alphabet with all tone marks (both lower and upper case)
"""

import os
import sys

# Force UTF-8 encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
TRAINING_DIR = os.path.join(PROJECT_ROOT, "ocr-training")
STUDIO_DIR = os.path.join(PROJECT_ROOT, "ocr-studio")

V24_DICT_PATH = os.path.join(TRAINING_DIR, "data", "cham_dict_v24.txt")
V25_DICT_TRAIN = os.path.join(TRAINING_DIR, "data", "cham_dict_v25.txt")
V25_DICT_STUDIO = os.path.join(STUDIO_DIR, "data", "cham_dict_v25.txt")

with open(V24_DICT_PATH, "r", encoding="utf-8") as f:
    existing_lines = [line.strip("\r\n") for line in f if line.strip("\r\n") != ""]

viet_lower = "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
viet_upper = "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"
latin_lower = "abcdefghijklmnopqrstuvwxyz"
latin_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
extra_punct = ["–", "—", "?", "!", "\"", "'"]

ordered_tokens = []
seen = set()

# 1. Existing tokens from v24
for tok in existing_lines:
    if tok not in seen:
        seen.add(tok)
        ordered_tokens.append(tok)

# 2. Add extra punctuation
for tok in extra_punct:
    if tok not in seen:
        seen.add(tok)
        ordered_tokens.append(tok)

# 3. Add Latin characters
for tok in latin_lower + latin_upper:
    if tok not in seen:
        seen.add(tok)
        ordered_tokens.append(tok)

# 4. Add Vietnamese diacritic characters
for tok in viet_lower + viet_upper:
    if tok not in seen:
        seen.add(tok)
        ordered_tokens.append(tok)

# Write to both ocr-training and ocr-studio
for out_path in [V25_DICT_TRAIN, V25_DICT_STUDIO]:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for tok in ordered_tokens:
            f.write(tok + "\n")
    print(f"✅ Saved {len(ordered_tokens)} unified tokens to: {out_path}")

print(f"\n🎉 Unified Dictionary V25 created successfully with {len(ordered_tokens)} tokens!")
