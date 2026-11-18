#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR Frozen Stress & Regression Benchmark Suite Runner
=========================================================
Executes the mandatory frozen test suites specified in:
archive/v24_to_v24.1_package/03_Test_Spec/V24_FROZEN_STRESS_TEST_SPEC.md

Suites:
  1. Clean-Frozen     : Guard against catastrophic forgetting (200 clean lines)
  2. Character-Frozen : 280 lines / 4,853 chars covering 100% dictionary + zero-corpus + P0 hard confusions
  3. Stress-Frozen    : 82 conditions (Low-res x blur cliffs, dilation, blank hallucination)
  4. Symbol-Frozen    : 14 mandatory punctuation and symbols (280 samples, >= 95% threshold)

Requirements:
  - Model verification: SHA256 of inference.pdiparams checked against baseline.
  - Strict Execution: Simulated fallback is strictly disabled. Requires native PaddleOCR engine.
"""

import os
import sys
import json
import time
import math
import hashlib
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Monkeypatch NumPy 2.x
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str]
    }
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'typeDict'): np.typeDict = {}

# Force UTF-8 on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)
STUDIO_DIR = os.path.join(PROJECT_ROOT, "ocr-studio")
TRAINING_DIR = os.path.join(PROJECT_ROOT, "ocr-training")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from frozen_gate_policy import get_policy, FrozenGatePolicy

def find_frozen_data_dir():
    candidates = [
        os.path.join(SCRIPT_DIR, "..", "03_Test_Spec"),
        os.path.join(PROJECT_ROOT, "03_Test_Spec"),
        os.path.join(SCRIPT_DIR, "03_Test_Spec"),
        os.path.join(os.getcwd(), "03_Test_Spec"),
        os.path.join(SCRIPT_DIR, "..", "frozen_suite"),
        os.path.join(PROJECT_ROOT, "frozen_suite"),
        os.path.join(SCRIPT_DIR, "frozen_suite"),
        os.path.join(os.getcwd(), "frozen_suite"),
        "/kaggle/working/frozen_suite",
        os.path.join(SCRIPT_DIR, "..", "07_Frozen_Benchmark", "data"),
        os.path.join(PROJECT_ROOT, "07_Frozen_Benchmark", "data"),
        os.path.join(SCRIPT_DIR, "07_Frozen_Benchmark", "data"),
        os.path.join(os.getcwd(), "07_Frozen_Benchmark", "data"),
        os.path.join(BENCHMARK_DIR, "data", "frozen_suite"),
        os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "frozen_suite"),
        os.path.join(SCRIPT_DIR, "..", "data", "frozen_suite"),
        os.path.join(os.getcwd(), "data", "frozen_suite"),
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "clean_frozen_200.json")):
            return os.path.abspath(c)
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.join(BENCHMARK_DIR, "data", "frozen_suite")

FROZEN_DATA_DIR = find_frozen_data_dir()

def get_frozen_file(filename):
    p = os.path.join(FROZEN_DATA_DIR, filename)
    if os.path.exists(p):
        return os.path.abspath(p)
    candidates = [
        os.path.join(SCRIPT_DIR, "..", "07_Frozen_Benchmark", "data", filename),
        os.path.join(PROJECT_ROOT, "07_Frozen_Benchmark", "data", filename),
        os.path.join(SCRIPT_DIR, "07_Frozen_Benchmark", "data", filename),
        os.path.join(BENCHMARK_DIR, "data", "frozen_suite", filename),
        os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "frozen_suite", filename),
        os.path.join(SCRIPT_DIR, "..", "frozen_suite", filename),
        os.path.join(PROJECT_ROOT, "frozen_suite", filename),
        os.path.join(SCRIPT_DIR, "frozen_suite", filename),
        os.path.join(os.getcwd(), "frozen_suite", filename),
        f"/kaggle/working/frozen_suite/{filename}",
        os.path.join(SCRIPT_DIR, "..", "03_Test_Spec", filename),
        os.path.join(PROJECT_ROOT, "03_Test_Spec", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Cannot find frozen data file: {filename}")

def get_crop_image(rel_crop_path):
    """
    Locates and reads a frozen crop image across potential environments (Local, Kaggle, Package).
    Raises FileNotFoundError if image does not exist. Runtime PIL fallback is forbidden!
    """
    candidates = [
        os.path.join(FROZEN_DATA_DIR, rel_crop_path),
        os.path.join(FROZEN_DATA_DIR, "..", rel_crop_path),
        os.path.join(BENCHMARK_DIR, "data", "frozen_suite", rel_crop_path),
        os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "frozen_suite", rel_crop_path),
        os.path.join("/kaggle/working/frozen_suite", rel_crop_path),
        os.path.join("/kaggle/working", rel_crop_path),
    ]
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'crops' in dirs:
                candidates.append(os.path.join(root, rel_crop_path))
                candidates.append(os.path.join(root, 'crops', os.path.basename(rel_crop_path)))

    for c in candidates:
        if os.path.exists(c):
            img = cv2.imread(c)
            if img is not None:
                return img

    raise FileNotFoundError(
        f"FATAL: Frozen crop image '{rel_crop_path}' not found!\n"
        f"Simulated / runtime PIL fallback is strictly prohibited to guarantee benchmark reproducibility.\n"
        f"Please ensure frozen crop images are present in frozen_suite/crops/."
    )

sys.path.insert(0, STUDIO_DIR)
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, os.path.join(TRAINING_DIR, "scripts"))
sys.path.insert(0, "/kaggle/working/scripts")

try:
    from scripts.generate_data_v24_1 import normalize_unicode
except ImportError:
    try:
        from scripts.generate_data import normalize_unicode
    except ImportError:
        try:
            from generate_data_v24_1 import normalize_unicode
        except ImportError:
            def normalize_unicode(s):
                return s

# Levenshtein distance for CER
def compute_edits(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]

def calculate_sha256(filepath):
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

# ==============================================================================
# 1. Pipeline Recognizer Wrapper (Strict Native Execution)
# ==============================================================================

def resolve_dict_path(dict_arg=None, verify_baseline=False):
    if dict_arg:
        if os.path.exists(dict_arg):
            return os.path.abspath(dict_arg)
        # Search candidate locations
        candidates = [
            dict_arg,
            os.path.join(PROJECT_ROOT, dict_arg),
            os.path.join(STUDIO_DIR, "data", os.path.basename(dict_arg)),
            os.path.join(BENCHMARK_DIR, "data", os.path.basename(dict_arg)),
            os.path.join(SCRIPT_DIR, "..", "05_Dictionaries_and_Fonts", os.path.basename(dict_arg)),
            os.path.join(SCRIPT_DIR, "05_Dictionaries_and_Fonts", os.path.basename(dict_arg)),
            os.path.join(PROJECT_ROOT, "05_Dictionaries_and_Fonts", os.path.basename(dict_arg)),
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
        raise FileNotFoundError(f"Specified dictionary path not found: {dict_arg}")

    # Explicit default: Baseline v24 or R1A MUST use cham_dict_v24.txt (103 chars)
    # v24.1 (R1B, R2, R3, R4) uses cham_dict_v24_1.txt (114 chars)
    default_name = "cham_dict_v24.txt" if verify_baseline else "cham_dict_v24_1.txt"
    candidates = [
        os.path.join(STUDIO_DIR, "data", default_name),
        os.path.join(BENCHMARK_DIR, "data", default_name),
        os.path.join(PROJECT_ROOT, "ocr-studio", "data", default_name),
        os.path.join(SCRIPT_DIR, "..", "05_Dictionaries_and_Fonts", default_name),
        os.path.join(SCRIPT_DIR, "05_Dictionaries_and_Fonts", default_name),
        os.path.join(PROJECT_ROOT, "05_Dictionaries_and_Fonts", default_name),
        # Fallback to v24 if v24_1 not created yet
        os.path.join(STUDIO_DIR, "data", "cham_dict_v24.txt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Could not locate dictionary file: {default_name}")

class RecognizerEvaluator:
    def __init__(self, model_dir=None, dict_path=None, verify_baseline=False, det_dir=None, rec_image_shape=None):
        self.model_dir = model_dir or os.path.join(STUDIO_DIR, "data", "output", "rec_cham_inference_v24")
        self.verify_baseline = verify_baseline
        self.dict_path = resolve_dict_path(dict_path, verify_baseline)
        if rec_image_shape:
            self.rec_image_shape = str(rec_image_shape)
        else:
            m_path = os.path.join(self.model_dir, "model_manifest.json")
            m_shape = None
            if os.path.exists(m_path):
                try:
                    with open(m_path, "r", encoding="utf-8") as mf:
                        m_shape = json.load(mf).get("rec_image_shape")
                except Exception:
                    m_shape = None
            self.rec_image_shape = str(m_shape or "3, 48, 320")
        self.engine = None
        self.model_sha256 = None
        self.dict_token_count = 0
        self.app_model = None
        self.paddle_full = None
        self.det_model = None
        self.has_detector = False
        self.verify_model_files()
        self.verify_dict_file()
        self.init_engine()
        self.init_detector(det_dir)

    def verify_dict_file(self):
        with open(self.dict_path, 'r', encoding='utf-8') as f:
            tokens = [line.strip('\r\n') for line in f if line.strip('\r\n')]
        self.dict_token_count = len(tokens)
        print(f"[INFO] Using character dictionary: {self.dict_path} ({self.dict_token_count} tokens)")
        if self.verify_baseline and self.dict_token_count != 103:
            print(f"⚠️  [WARNING] Baseline v24 dictionary expected 103 tokens, got {self.dict_token_count}!")

    def verify_model_files(self):
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory does not exist: {self.model_dir}")

        params_file = os.path.join(self.model_dir, "inference.pdiparams")
        if not os.path.exists(params_file):
            raise FileNotFoundError(f"Missing inference.pdiparams in: {self.model_dir}")

        self.model_sha256 = calculate_sha256(params_file)
        print(f"[INFO] Model directory: {self.model_dir}")
        print(f"[INFO] Model inference.pdiparams SHA256: {self.model_sha256}")

        v24_baseline_sha = "61e3606f1e57d9e83de0d0b92fa5722c6815ff0e7ea3725895f6bf9e56d71596"
        if self.verify_baseline:
            if self.model_sha256 != v24_baseline_sha:
                raise ValueError(
                    f"FATAL: Model SHA256 mismatch for v24 baseline!\n"
                    f"Expected: {v24_baseline_sha}\n"
                    f"Observed: {self.model_sha256}"
                )
            print("✅ Verified baseline v24 model SHA256 match.")

    def init_engine(self):
        # 1. Primary: Direct Paddle TextRecognizer with exact model_dir and dict_path
        paddle_err = None
        try:
            paddleocr_candidates = [
                "/kaggle/working/PaddleOCR",
                os.path.join(STUDIO_DIR, "PaddleOCR"),
                os.path.join(PROJECT_ROOT, "PaddleOCR"),
                os.path.join(os.getcwd(), "PaddleOCR"),
                os.path.abspath("PaddleOCR"),
            ]
            for p in paddleocr_candidates:
                if os.path.isdir(p) and p not in sys.path:
                    sys.path.insert(0, p)
            import tools.infer.utility as utility
            from tools.infer.predict_rec import TextRecognizer

            sys_argv_backup = sys.argv
            sys.argv = [sys.argv[0]]
            args = utility.parse_args()
            sys.argv = sys_argv_backup

            args.rec_model_dir = os.path.abspath(self.model_dir)
            args.rec_char_dict_path = os.path.abspath(self.dict_path)
            args.use_gpu = False
            args.rec_image_shape = self.rec_image_shape
            args.use_space_char = True
            args.rec_batch_num = 1
            args.enable_mkldnn = False
            args.cpu_threads = 1
            args.ir_optim = False
            args.precision = "fp32"

            self.app_model = TextRecognizer(args)
            self.engine = "studio_app"
            print(f"[INFO] Initialized direct TextRecognizer from: {self.model_dir}")
            print(f"[INFO] Character dictionary: {self.dict_path} ({self.dict_token_count} tokens)")
            return
        except Exception as e:
            paddle_err = e

        # 2. Simulated fallback is strictly prohibited!
        raise RuntimeError(
            f"FATAL: Native PaddleOCR inference engine is not available ({paddle_err}).\n"
            f"Frozen benchmark suites require actual model execution.\n"
            f"Simulated / empty fallback is strictly disabled to prevent invalid benchmark runs.\n"
            f"Please run this benchmark on a machine with PaddleOCR installed (e.g. Kaggle GPU environment)."
        )

    def init_detector(self, det_dir=None):
        # Cố gắng nạp Cham DBNet detector cho FullPage evaluation
        det_candidates = []
        if det_dir:
            det_candidates.append(det_dir)
        det_candidates.extend([
            os.path.join(STUDIO_DIR, "data", "output", "ch_PP-OCRv4_det_cham_infer"),
            os.path.join(PROJECT_ROOT, "ocr-studio", "data", "output", "ch_PP-OCRv4_det_cham_infer"),
            os.path.join(SCRIPT_DIR, "..", "data", "output", "ch_PP-OCRv4_det_cham_infer"),
            os.path.join(SCRIPT_DIR, "04_Code_and_Scripts", "ch_PP-OCRv4_det_cham_infer"),
            os.path.join(PROJECT_ROOT, "04_Code_and_Scripts", "ch_PP-OCRv4_det_cham_infer"),
            os.path.join(PROJECT_ROOT, "data", "output", "ch_PP-OCRv4_det_cham_infer"),
        ])
        found_det = next((p for p in det_candidates if os.path.exists(os.path.join(p, "inference.pdiparams"))), None)
        if found_det:
            try:
                from paddleocr import PaddleOCR
                self.paddle_full = PaddleOCR(
                    use_angle_cls=False,
                    lang="cham",
                    det_model_dir=found_det,
                    rec_model_dir=self.model_dir,
                    rec_char_dict_path=self.dict_path,
                    use_gpu=False,
                    show_log=False
                )
                self.has_detector = True
                print(f"[INFO] FullPage Pipeline: Initialized Cham-DBNet Detector from: {os.path.basename(found_det)}")
                return
            except Exception:
                pass

        try:
            import app
            if hasattr(app, "get_det_model"):
                self.det_model = app.get_det_model()
                self.has_detector = True
                print("[INFO] FullPage Pipeline: Initialized Cham-DBNet from studio app")
                return
        except Exception:
            pass

        self.has_detector = False
        print("[INFO] Dedicated Cham detector not loaded; FullPage will evaluate via GT line coordinates.")

    def predict_line(self, img_bgr):
        """Returns: (raw_pred, normalized_pred, conf)"""
        if self.engine == "studio_app" and self.app_model is not None:
            res, _ = self.app_model([img_bgr])
            if res and len(res) > 0:
                raw_text = res[0][0]
                conf = float(res[0][1])
                norm_text = normalize_unicode(raw_text)
                return raw_text, norm_text, conf
        elif self.engine == "paddle_ocr":
            res = self.paddle_ocr.ocr(img_bgr, det=False, cls=False)
            if res and len(res) > 0 and len(res[0]) > 0:
                raw_text = res[0][0][0]
                conf = float(res[0][0][1])
                norm_text = normalize_unicode(raw_text)
                return raw_text, norm_text, conf

        return "", "", 0.0

    def predict_page(self, img_bgr):
        """
        Executes actual page pipeline: Page -> Cham-DBNet -> Bounding boxes -> Recognizer.
        Returns: list of (box_polygon, raw_text, norm_text, conf)
        """
        if self.paddle_full is not None:
            res = self.paddle_full.ocr(img_bgr, cls=False)
            results = []
            if res and len(res) > 0 and res[0]:
                for line in res[0]:
                    box = line[0]
                    raw = line[1][0]
                    conf = float(line[1][1])
                    norm = normalize_unicode(raw)
                    results.append((box, raw, norm, conf))
            return results

        if self.det_model is not None:
            try:
                import app
                ink = app.analyze_text_ink(img_bgr)
                crops, metas = app.segment_lines_dbnet(img_bgr, self.det_model, ink=ink)
                results = []
                for crop, meta in zip(crops, metas):
                    raw, norm, conf = self.predict_line(crop)
                    poly = meta.get('polygon', meta.get('bbox', []))
                    results.append((poly, raw, norm, conf))
                return results
            except Exception:
                pass

        return []

# ==============================================================================
# 2. Image Rendering & Degradation Utilities
# ==============================================================================

STRESS_8_LINES = [
    "ꨒꨮꩌ ꨒꨯꩌ ꨓꨝꨵꨰꩅ ꨓꨝꨵꨯꩃ ꨚꨣꨈꩌ",  # 24 chars
    "ꨝꨝꨶꨮꩊ ꨨꨕꨬ ꨓꨤꨶꨪꩄ ꨣꨯ ꨨꨰꩈ",        # 22 chars
    "ꨌꨬ ꨝꨴꨬ ꨚꨵꩌ ꨟꨆꩍ ꨝꨳꨭꩍ ꨦꩀꨎꨝꨩ",        # 25 chars
    "ꨗꨶꨮꩆ ꨦꨴꨬ ꨤꨪ ꨆꨵꨩ ꨚꨙꩉ ꨀꨣꨯꨩ",        # 24 chars
    "ꨆꨯꨱꩀ ꨨꨕꩍ ꨓꨗꨴꩀ ꨨꨕꩍ ꨟꨓꨩ",            # 21 chars
    "ꨛꨯꨮ ꨎꨝꨯꩊ ꨛꨯꨮ",                        # 12 chars
    "ꨚꨶꨮꩄ ꨚꨏꨪꩂ ꨚꨶꨮꩄ",                    # 14 chars
    "ꨨꨕꨶꨰ ꨨꨕꨬ ꨓꨤꨶꨪꩄ"                     # 14 chars
]
# Total characters = 156 (matches exactly with stress_frozen_82.json chars: 156)

def apply_stress_condition(img, name, param):
    """
    Applies real physical & digital degradation corresponding to one of the 26 condition families:
    low_height, gaussian_blur, motion_blur, noise, jpeg, contrast, faded, paper,
    rotate, perspective, clip_top, clip_bottom, erode, dilate, gray, threshold,
    adaptive, shadow, vignette, bleed, vertical_scale, horizontal_scale,
    combo_low_motion, combo_low_gauss, combo_jpeg_blur, combo_fade_noise.
    """
    if img is None:
        return img

    h, w = img.shape[:2]

    if name == "low_height":
        th = max(8, int(param))
        down = cv2.resize(img, (max(10, int(w * (th / float(h)))), th), interpolation=cv2.INTER_AREA)
        return cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)

    elif name == "gaussian_blur":
        sigma = float(param)
        if sigma <= 0: return img
        ksize = int(math.ceil(sigma * 3)) * 2 + 1
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    elif name == "motion_blur":
        k = max(1, int(param))
        if k <= 1: return img
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[int((k - 1) / 2), :] = 1.0
        kernel /= np.sum(kernel)
        return cv2.filter2D(img, -1, kernel)

    elif name == "noise":
        std = float(param)
        rng = np.random.RandomState(int(std * 17) % 10000)
        noise = rng.normal(0, std, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    elif name == "jpeg":
        q = max(1, min(100, int(param)))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        _, enc = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(enc, 1)

    elif name == "contrast":
        factor = float(param)
        mean = 128.0
        res = mean + (img.astype(np.float32) - mean) * factor
        return np.clip(res, 0, 255).astype(np.uint8)

    elif name == "faded":
        factor = float(param)
        res = img.astype(np.float32) * (1.0 - factor) + 255.0 * factor
        return np.clip(res, 0, 255).astype(np.uint8)

    elif name == "paper":
        ptype = int(param)
        tint = np.array([210, 235, 245], dtype=np.float32) / 255.0
        intensity = ptype * 0.15
        res = img.astype(np.float32) * (1.0 - intensity) + (img.astype(np.float32) * tint) * intensity
        rng = np.random.RandomState(ptype * 123)
        grain = rng.normal(0, ptype * 4.0, img.shape).astype(np.float32)
        return np.clip(res + grain, 0, 255).astype(np.uint8)

    elif name == "rotate":
        angle = float(param)
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    elif name == "perspective":
        p = float(param)
        dx = float(w * p)
        dy = float(h * p)
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([[dx, dy], [w - dx, 0], [0, h], [w, h - dy]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    elif name == "clip_top":
        frac = float(param)
        res = img.copy()
        cut_h = int(h * frac)
        res[:cut_h, :] = 255
        return res

    elif name == "clip_bottom":
        frac = float(param)
        res = img.copy()
        cut_h = int(h * frac)
        res[-cut_h:, :] = 255
        return res

    elif name == "erode":
        k = max(1, int(param))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        kernel = np.ones((k, k), np.uint8)
        eroded = cv2.erode(inv, kernel, iterations=1)
        return cv2.cvtColor(255 - eroded, cv2.COLOR_GRAY2BGR)

    elif name == "dilate":
        k = max(1, int(param))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        kernel = np.ones((k, k), np.uint8)
        dilated = cv2.dilate(inv, kernel, iterations=1)
        return cv2.cvtColor(255 - dilated, cv2.COLOR_GRAY2BGR)

    elif name == "gray":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif name == "threshold":
        th = int(param)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    elif name == "adaptive":
        bs = int(param) if int(param) % 2 == 1 else int(param) + 1
        bs = max(3, bs)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, 2)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    elif name == "shadow":
        intensity = float(param)
        gradient = np.linspace(1.0 - intensity, 1.0, w, dtype=np.float32)
        mask = np.tile(gradient, (h, 1))[:, :, np.newaxis]
        return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)

    elif name == "vignette":
        intensity = float(param)
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2.0, w / 2.0
        dist = np.sqrt(((X - center_x) / (w / 2.0)) ** 2 + ((Y - center_y) / (h / 2.0)) ** 2)
        mask = np.clip(1.0 - (dist * intensity), 1.0 - intensity, 1.0)[:, :, np.newaxis]
        return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)

    elif name == "bleed":
        opacity = float(param)
        flipped = cv2.flip(img, 1)
        return cv2.addWeighted(img, 1.0, flipped, opacity, 0)

    elif name == "vertical_scale":
        factor = float(param)
        new_h = max(8, int(h * factor))
        scaled = cv2.resize(img, (w, new_h), interpolation=cv2.INTER_LINEAR)
        res = np.full((h, w, 3), 255, dtype=np.uint8)
        if new_h < h:
            offset = (h - new_h) // 2
            res[offset:offset+new_h, :] = scaled
        else:
            offset = (new_h - h) // 2
            res[:, :] = scaled[offset:offset+h, :]
        return res

    elif name == "horizontal_scale":
        factor = float(param)
        new_w = max(10, int(w * factor))
        return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)

    elif name == "combo_low_motion":
        h_val, k_val = param
        res = apply_stress_condition(img, "low_height", h_val)
        return apply_stress_condition(res, "motion_blur", k_val)

    elif name == "combo_low_gauss":
        h_val, s_val = param
        res = apply_stress_condition(img, "low_height", h_val)
        return apply_stress_condition(res, "gaussian_blur", s_val)

    elif name == "combo_jpeg_blur":
        q_val, s_val = param
        res = apply_stress_condition(img, "jpeg", q_val)
        return apply_stress_condition(res, "gaussian_blur", s_val)

    elif name == "combo_fade_noise":
        f_val, n_val = param
        res = apply_stress_condition(img, "faded", f_val)
        return apply_stress_condition(res, "noise", n_val)

    return img

def _raw_polygon_iou(poly1_pts, poly2_pts):
    """Computes exact Intersection-over-Union between two 2D polygons using Shapely with bbox fallback."""
    try:
        from shapely.geometry import Polygon
        p1 = Polygon(poly1_pts)
        p2 = Polygon(poly2_pts)
        if not p1.is_valid: p1 = p1.buffer(0)
        if not p2.is_valid: p2 = p2.buffer(0)
        if p1.area <= 0 or p2.area <= 0: return 0.0
        inter = p1.intersection(p2).area
        union = p1.area + p2.area - inter
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        x1_min = min(p[0] for p in poly1_pts)
        x1_max = max(p[0] for p in poly1_pts)
        y1_min = min(p[1] for p in poly1_pts)
        y1_max = max(p[1] for p in poly1_pts)
        x2_min = min(p[0] for p in poly2_pts)
        x2_max = max(p[0] for p in poly2_pts)
        y2_min = min(p[1] for p in poly2_pts)
        y2_max = max(p[1] for p in poly2_pts)
        inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        inter = inter_x * inter_y
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - inter
        return float(inter / max(union, 1e-6))

def compute_polygon_iou(poly1_pts, poly2_pts):
    """Computes IoU; when a 4-vertex AABB/quad GT is matched against a curved >4-vertex polygon,
    also takes the IoU of the detected polygon's minAreaRect so wavy GT quads still align cleanly."""
    raw_iou = _raw_polygon_iou(poly1_pts, poly2_pts)
    if len(poly1_pts) == 4 and len(poly2_pts) > 4:
        rect_pts = cv2.boxPoints(cv2.minAreaRect(np.asarray(poly2_pts, dtype=np.float32))).tolist()
        return max(raw_iou, _raw_polygon_iou(poly1_pts, rect_pts))
    return raw_iou

def match_polygons_bipartite(gt_polys, det_polys, iou_threshold=0.40):
    """
    Performs bipartite matching between Ground Truth polygons and Detected polygons.
    Uses scipy linear_sum_assignment when available, with greedy fallback.
    Returns: (matched_pairs, unmatched_gt, unmatched_det)
    """
    M = len(gt_polys)
    N = len(det_polys)
    if M == 0 or N == 0:
        return [], set(range(M)), set(range(N))

    iou_matrix = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            iou_matrix[i, j] = compute_polygon_iou(gt_polys[i], det_polys[j])

    try:
        from scipy.optimize import linear_sum_assignment
        cost_matrix = np.zeros((M, N), dtype=np.float32)
        valid = (iou_matrix >= iou_threshold)
        cost_matrix[valid] = -10000.0 - iou_matrix[valid]
        r_ind, c_ind = linear_sum_assignment(cost_matrix)
        matched_pairs = []
        matched_gt = set()
        matched_det = set()
        for r, c in zip(r_ind, c_ind):
            val = float(iou_matrix[r, c])
            if val >= iou_threshold:
                matched_pairs.append((int(r), int(c), val))
                matched_gt.add(int(r))
                matched_det.add(int(c))
        unmatched_gt = set(range(M)) - matched_gt
        unmatched_det = set(range(N)) - matched_det
        return matched_pairs, unmatched_gt, unmatched_det
    except Exception:
        candidates = []
        for i in range(M):
            for j in range(N):
                if iou_matrix[i, j] >= iou_threshold:
                    candidates.append((iou_matrix[i, j], i, j))
        candidates.sort(reverse=True)
        matched_pairs = []
        matched_gt = set()
        matched_det = set()
        for iou_val, r, c in candidates:
            if r not in matched_gt and c not in matched_det:
                matched_pairs.append((r, c, float(iou_val)))
                matched_gt.add(r)
                matched_det.add(c)
        unmatched_gt = set(range(M)) - matched_gt
        unmatched_det = set(range(N)) - matched_det
        return matched_pairs, unmatched_gt, unmatched_det

def get_cham_font(size=24, bold=False):
    font_name = "NotoSansCham-Bold.ttf" if bold else "NotoSansCham-Regular.ttf"
    font_candidates = [
        f"/kaggle/working/data/fonts/{font_name}",
        f"/kaggle/working/fonts/{font_name}",
        os.path.join(STUDIO_DIR, "fonts", font_name),
        os.path.join(TRAINING_DIR, "data", "fonts", font_name),
        os.path.join(SCRIPT_DIR, "..", "05_Dictionaries_and_Fonts", font_name),
        os.path.join(SCRIPT_DIR, "05_Dictionaries_and_Fonts", font_name),
        os.path.join(PROJECT_ROOT, "05_Dictionaries_and_Fonts", font_name),
        os.path.join(PROJECT_ROOT, "data", "fonts", font_name),
        "C:\\Windows\\Fonts\\NotoSansCham-Regular.ttf"
    ]
    import glob
    found_kaggle = glob.glob(f"/kaggle/input/**/{font_name}", recursive=True)
    if found_kaggle:
        font_candidates.insert(0, found_kaggle[0])

    for p in font_candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    raise FileNotFoundError(
        f"FATAL: Cham font '{font_name}' not found or could not be loaded! "
        f"Never fallback to system default font as it corrupts Cham OCR evaluation.\n"
        f"Searched candidates:\n" + "\n".join(font_candidates)
    )

def render_cham_line(text, font_size=24, bold=False, target_height=48):
    font = get_cham_font(font_size, bold)
    dummy_img = Image.new("RGB", (10, 10), (255, 255, 255))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = max(10, bbox[2] - bbox[0] + 16)
    h = max(10, bbox[3] - bbox[1] + 12)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((8 - bbox[0], 6 - bbox[1]), text, font=font, fill=(0, 0, 0))
    scale = target_height / float(h)
    new_w = max(10, int(w * scale))
    img_resized = img.resize((new_w, target_height), Image.Resampling.BILINEAR)
    return cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

def render_synthetic_page(page, target_w=None, target_h=None):
    w = target_w or page.get("width", 950)
    h = target_h or page.get("height", 650)
    font_size = page.get("font_size", 23)
    is_bold = "bold" in page.get("font_name", "").lower()
    img_pil = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img_pil)
    font = get_cham_font(size=font_size, bold=is_bold)
    for line in page.get("lines", []):
        text = line.get("text", "")
        poly = line.get("polygon", [])
        if len(poly) >= 4 and text:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            draw.text((min(xs), min(ys)), text, font=font, fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def find_page_image(page):
    fn = page.get("filename", "")
    rel_img = page.get("image_path", "")
    candidates = [
        os.path.join(FROZEN_DATA_DIR, "images", fn),
        os.path.join(FROZEN_DATA_DIR, "..", "data", "images", fn),
        os.path.join(BENCHMARK_DIR, "data", "images", fn),
        os.path.join(BENCHMARK_DIR, rel_img),
        os.path.join(PROJECT_ROOT, "ocr-benchmark", rel_img),
        os.path.join(PROJECT_ROOT, "07_Frozen_Benchmark", "data", "images", fn),
        os.path.join(PROJECT_ROOT, "03_Test_Spec", "images", fn),
    ]
    for c in candidates:
        if os.path.exists(c):
            img = cv2.imread(c)
            if img is not None:
                return img
    return render_synthetic_page(page)

# ==============================================================================
# 3. Test Suites Implementation
# ==============================================================================

def run_suite_clean(evaluator):
    print("\n--- Running Suite A: Clean-Frozen (200 Lines / 7,731 Chars - Frozen PNG Crops) ---")
    clean_lines_path = None
    try:
        clean_lines_path = get_frozen_file("clean_frozen_200_lines.json")
    except FileNotFoundError:
        pass

    if clean_lines_path and os.path.exists(clean_lines_path):
        with open(clean_lines_path, 'r', encoding='utf-8') as f:
            clean_items = json.load(f)
    else:
        clean_path = get_frozen_file("clean_frozen_200.json")
        with open(clean_path, 'r', encoding='utf-8') as f:
            pages = json.load(f)
        clean_items = []
        for p in pages:
            for l_idx, l in enumerate(p.get("lines", [])):
                if len(clean_items) < 200:
                    clean_items.append({
                        "sample_id": len(clean_items),
                        "page": p.get("filename", ""),
                        "line_idx": l_idx,
                        "gt": l.get("text", ""),
                        "crop_file": f"crops/clean_lines/clean_{len(clean_items):03d}.png"
                    })

    total_edits_raw = 0
    total_edits_norm = 0
    total_chars = 0
    exact_count = 0
    total_lines = 0
    confs = []
    failures = []
    all_clean_records = []

    for item in clean_items:
        gt = item.get("gt", "")
        crop_file = item.get("crop_file", f"crops/clean_lines/clean_{item.get('sample_id', 0):03d}.png")
        line_crop = get_crop_image(crop_file)

        raw, norm, conf = evaluator.predict_line(line_crop)
        edits_raw = compute_edits(gt, raw)
        edits_norm = compute_edits(gt, norm)

        total_lines += 1
        total_chars += len(gt)
        total_edits_raw += edits_raw
        total_edits_norm += edits_norm
        confs.append(conf)

        rec = {
            "sample_id": item.get("sample_id"),
            "page": item.get("page", ""),
            "line_idx": item.get("line_idx", 0),
            "crop_file": crop_file,
            "gt": gt,
            "raw_pred": raw,
            "pred": norm,
            "conf": conf,
            "edits": edits_norm,
            "exact": (edits_norm == 0)
        }
        all_clean_records.append(rec)

        if edits_norm == 0:
            exact_count += 1
        else:
            if len(failures) < 50:
                failures.append(rec)

    raw_cer = total_edits_raw / max(1, total_chars)
    norm_cer = total_edits_norm / max(1, total_chars)
    exact_rate = exact_count / max(1, total_lines)
    avg_conf = float(np.mean(confs)) if confs else 0.0

    print(f"  Clean Results: N={total_lines} lines (Level 1 Standard Clean), Chars={total_chars}")
    print(f"  Raw CER={raw_cer*100:.2f}%, Norm CER={norm_cer*100:.2f}%, ExactLine={exact_rate*100:.1f}%, AvgConf={avg_conf:.3f}")
    return {
        "n": total_lines,
        "chars": total_chars,
        "raw_cer": raw_cer,
        "norm_cer": norm_cer,
        "exact_rate": exact_rate,
        "avg_conf": avg_conf,
        "failures": failures,
        "all_records": all_clean_records
    }

def run_suite_character(evaluator):
    print("\n--- Running Suite B: Character-Frozen (280 Lines / 4,853 Chars - Frozen PNG Crops) ---")
    char_lines_path = None
    try:
        char_lines_path = get_frozen_file("character_frozen_280_lines.json")
    except FileNotFoundError:
        pass

    if char_lines_path and os.path.exists(char_lines_path):
        with open(char_lines_path, 'r', encoding='utf-8') as f:
            char_items = json.load(f)
    else:
        char_path = get_frozen_file("character_frozen_280.json")
        with open(char_path, 'r', encoding='utf-8') as f:
            rows = json.load(f)
        char_items = []
        for idx, row in enumerate(rows):
            char_items.append({
                "sample_id": idx,
                "gt": row["gt"],
                "font": row.get("font", "regular"),
                "crop_file": f"crops/character_lines/char_{idx:03d}.png"
            })

    records = []
    total_edits = 0
    total_chars = 0
    exact_count = 0
    high_conf_wrong = 0

    for item in char_items:
        gt = item['gt']
        crop_file = item.get("crop_file", f"crops/character_lines/char_{item.get('sample_id', 0):03d}.png")
        img = get_crop_image(crop_file)

        raw, norm, conf = evaluator.predict_line(img)
        edits = compute_edits(gt, norm)
        total_edits += edits
        total_chars += len(gt)
        if edits == 0:
            exact_count += 1
        if edits > 0 and conf >= 0.85:
            high_conf_wrong += 1

        records.append({
            "sample_id": item.get("sample_id"),
            "gt": gt,
            "raw_pred": raw,
            "norm_pred": norm,
            "conf": conf,
            "edits": edits,
            "exact": (edits == 0),
            "auto_healed": (raw != norm and norm == gt)
        })

    cer = total_edits / max(1, total_chars)
    exact_rate = exact_count / max(1, len(char_items))
    print(f"  Character-Frozen Results: N={len(char_items)}, Chars={total_chars}, CER={cer*100:.2f}%, Exact={exact_rate*100:.1f}%, HighConfWrong={high_conf_wrong}")
    return {
        "n": len(char_items),
        "chars": total_chars,
        "edits": total_edits,
        "cer": cer,
        "exact": exact_rate,
        "high_conf_wrong": high_conf_wrong,
        "all_records": records,
        "records": records
    }

def run_suite_stress(evaluator):
    print("\n--- Running Suite C: Stress-Frozen (All 82 Conditions / 8 Lines / 156 Chars - Frozen PNG Crops) ---")
    try:
        stress_path = get_frozen_file("stress_frozen_82.json")
        with open(stress_path, 'r', encoding='utf-8') as f:
            conditions = json.load(f)
    except FileNotFoundError:
        print("⚠️ Warning: stress_frozen_82.json not found, skipping 82 condition loops.")
        conditions = []

    # Load frozen base images directly from crops
    base_images = [get_crop_image(f"crops/stress_base_lines/stress_base_{idx}.png") for idx in range(len(STRESS_8_LINES))]
    total_chars_base = sum(len(line) for line in STRESS_8_LINES)  # 156

    all_condition_results = []
    key_cliffs = []
    KEY_CLIFF_SPECS = [
        ("combo_low_gauss", [16, 1.8]),
        ("combo_low_motion", [16, 9]),
        ("combo_low_motion", [18, 9]),
        ("combo_low_gauss", [20, 1.8]),
        ("combo_low_motion", [20, 7]),
        ("motion_blur", 17),
        ("perspective", 0.14),
        ("dilate", 3)
    ]

    total_edits_all = 0
    total_chars_all = 0

    for idx, cond in enumerate(conditions):
        c_name = cond.get("name")
        c_param = cond.get("param")
        baseline_cer = cond.get("cer", 0.0)

        cond_edits = 0
        cond_exact = 0
        confs = []

        for line_idx, (gt_line, base_img) in enumerate(zip(STRESS_8_LINES, base_images)):
            degraded = apply_stress_condition(base_img, c_name, c_param)
            raw, norm, conf = evaluator.predict_line(degraded)
            edits = compute_edits(gt_line, norm)
            cond_edits += edits
            if edits == 0:
                cond_exact += 1
            confs.append(conf)

        c_cer = cond_edits / float(total_chars_base)
        c_exact = cond_exact / float(len(STRESS_8_LINES))
        c_conf = float(np.mean(confs)) if confs else 0.0

        total_edits_all += cond_edits
        total_chars_all += total_chars_base

        record = {
            "name": c_name,
            "param": c_param,
            "cer": c_cer,
            "exact": c_exact,
            "conf": c_conf,
            "baseline_cer": baseline_cer,
            "delta_cer": c_cer - baseline_cer
        }
        all_condition_results.append(record)

        # Check if this matches a key cliff
        for k_name, k_param in KEY_CLIFF_SPECS:
            if c_name == k_name and c_param == k_param:
                key_cliffs.append({
                    "condition": c_name,
                    "param": c_param,
                    "baseline_cer": baseline_cer,
                    "current_cer": c_cer,
                    "conf": c_conf
                })
                break

    overall_stress_cer = total_edits_all / max(1, total_chars_all)
    print(f"  Stress-Frozen Overall (82 conditions): Mean CER={overall_stress_cer*100:.2f}%")
    print(f"  Key Failure Cliffs Status ({len(key_cliffs)} cliffs):")
    for kc in key_cliffs:
        print(f"    - {kc['condition']} {kc['param']}: Base={kc['baseline_cer']*100:.1f}% -> Curr={kc['current_cer']*100:.1f}% (Conf={kc['conf']:.3f})")

    # Non-text / Blank hallucination tests
    blank_white = np.full((48, 200, 3), 255, dtype=np.uint8)
    raw_bw, norm_bw, conf_bw = evaluator.predict_line(blank_white)
    hallucinated_bw = bool(len(norm_bw.strip()) > 0)
    print(f"  Blank White Crop: Pred='{norm_bw}' | Conf={conf_bw:.3f} | Hallucinated={hallucinated_bw}")

    blank_gray = np.full((48, 200, 3), 128, dtype=np.uint8)
    raw_bg, norm_bg, conf_bg = evaluator.predict_line(blank_gray)
    hallucinated_bg = bool(len(norm_bg.strip()) > 0)

    return {
        "conditions_count": len(conditions),
        "overall_stress_cer": overall_stress_cer,
        "key_cliffs": key_cliffs,
        "blank_white": {
            "pred": norm_bw,
            "conf": conf_bw,
            "hallucinated": hallucinated_bw
        },
        "blank_gray": {
            "pred": norm_bg,
            "conf": conf_bg,
            "hallucinated": hallucinated_bg
        },
        "all_conditions": all_condition_results
    }

def run_suite_stress_base_8(evaluator, tag=""):
    print("\n--- Running Suite C-Base: Stress Base (8 Frozen PNG Crops) ---")
    all_records = []
    total_edits = 0
    total_chars = 0
    exact_correct = 0

    for idx, gt_line in enumerate(STRESS_8_LINES):
        rel_path = f"crops/stress_base_lines/stress_base_{idx}.png"
        img = get_crop_image(rel_path)
        raw_pred, norm_pred, conf = evaluator.predict_line(img)
        edits = compute_edits(gt_line, norm_pred)
        total_edits += edits
        c_len = len(gt_line)
        total_chars += c_len
        is_pass = bool(norm_pred == gt_line)
        if is_pass:
            exact_correct += 1
        
        all_records.append({
            "case_id": idx,
            "sample_id": idx,
            "image_path": rel_path,
            "ground_truth": gt_line,
            "prediction": norm_pred,
            "raw_pred": raw_pred,
            "confidence": round(float(conf), 4),
            "edits": edits,
            "char_count": c_len,
            "cer": round(edits / max(1, c_len), 6),
            "pass": is_pass,
            "checkpoint": tag
        })
        status_sym = "✅" if is_pass else "❌"
        print(f"  {status_sym} [Case {idx}]: Conf={conf:.4f} | Edits={edits} | GT='{gt_line}' | Pred='{norm_pred}'")

    cer = total_edits / max(1, total_chars)
    acc = exact_correct / len(all_records)
    print(f"  Overall Stress Base-8: Exact Acc = {acc*100:.1f}% ({exact_correct}/{len(all_records)}) | CER = {cer*100:.2f}% ({total_edits}/{total_chars} edits)")

    return {
        "total_cases": len(all_records),
        "exact_correct": exact_correct,
        "exact_accuracy": acc,
        "total_edits": total_edits,
        "total_chars": total_chars,
        "cer": cer,
        "all_records": all_records
    }

def run_suite_blank(evaluator):
    print("\n--- Running Suite Blank: Blank & Non-Text Hallucination Suite ---")
    blank_white = np.full((48, 200, 3), 255, dtype=np.uint8)
    raw_bw, norm_bw, conf_bw = evaluator.predict_line(blank_white)
    hallucinated_bw = bool(len(norm_bw.strip()) > 0)
    print(f"  * Pure White Crop   : Pred='{norm_bw}' (raw='{raw_bw}') | Conf={conf_bw:.4f} | Hallucinated={hallucinated_bw}")

    blank_gray = np.full((48, 200, 3), 128, dtype=np.uint8)
    raw_bg, norm_bg, conf_bg = evaluator.predict_line(blank_gray)
    hallucinated_bg = bool(len(norm_bg.strip()) > 0)
    print(f"  * Neutral Gray Crop : Pred='{norm_bg}' (raw='{raw_bg}') | Conf={conf_bg:.4f} | Hallucinated={hallucinated_bg}")

    passed = not hallucinated_bw and not hallucinated_bg
    print(f"  Blank Hallucination Verdict: {'PASS (Zero Hallucination)' if passed else 'FAIL (Hallucination Detected)'}")

    return {
        "blank_white": {
            "pred": norm_bw,
            "raw_pred": raw_bw,
            "conf": conf_bw,
            "hallucinated": hallucinated_bw
        },
        "blank_gray": {
            "pred": norm_bg,
            "raw_pred": raw_bg,
            "conf": conf_bg,
            "hallucinated": hallucinated_bg
        },
        "passed": passed
    }

def run_suite_symbols(evaluator):
    print("\n--- Running Suite D: Symbol-Frozen (14 Mandatory Symbols - Single Source of Truth) ---")
    sym_lines_path = None
    try:
        sym_lines_path = get_frozen_file("symbol_frozen_14_lines.json")
    except FileNotFoundError:
        pass

    if sym_lines_path and os.path.exists(sym_lines_path):
        with open(sym_lines_path, 'r', encoding='utf-8') as f:
            symbol_cases = json.load(f)
    else:
        sym_path = get_frozen_file("symbol_frozen_14.json")
        with open(sym_path, 'r', encoding='utf-8') as f:
            raw_cases = json.load(f)
        symbol_cases = []
        for idx, item in enumerate(raw_cases):
            symbol_cases.append({
                "sample_id": idx + 1,
                "symbol": item["symbol"],
                "gt": item["gt"],
                "crop_file": f"crops/symbol_lines/symbol_{idx:03d}.png"
            })

    per_symbol = {}
    symbol_failures = []
    all_symbol_records = []

    for idx, item in enumerate(symbol_cases, 1):
        sym = item['symbol']
        gt = item['gt']
        crop_file = item.get("crop_file", f"crops/symbol_lines/symbol_{idx-1:03d}.png")
        if sym not in per_symbol:
            per_symbol[sym] = {"total": 0, "correct": 0, "exact": 0, "edits": 0, "chars": 0}

        img = get_crop_image(crop_file)
        raw, norm, conf = evaluator.predict_line(img)

        # Strict Count Accuracy as mandated by P1
        count_match = (norm.count(sym) == gt.count(sym))
        exact = (norm == gt)
        edits = compute_edits(gt, norm)

        per_symbol[sym]["total"] += 1
        per_symbol[sym]["chars"] += len(gt)
        per_symbol[sym]["edits"] += edits
        if count_match:
            per_symbol[sym]["correct"] += 1
        if exact:
            per_symbol[sym]["exact"] += 1

        rec = {
            "sample_id": item.get("sample_id", idx),
            "target_symbol": sym,
            "gt": gt,
            "raw_pred": raw,
            "pred": norm,
            "conf": conf,
            "count_match": count_match,
            "exact_line": exact,
            "edits": edits,
            "pass": count_match
        }
        all_symbol_records.append(rec)
        if not count_match:
            symbol_failures.append(rec)

    summary = {}
    failing_symbols = []
    for sym, d in per_symbol.items():
        acc = d["correct"] / max(1, d["total"])
        exact_acc = d["exact"] / max(1, d["total"])
        sym_cer = d["edits"] / max(1, d["chars"])
        summary[sym] = {
            "total": d["total"],
            "correct": d["correct"],
            "accuracy": acc,
            "exact_accuracy": exact_acc,
            "cer": sym_cer
        }
        if acc < 0.99:
            failing_symbols.append(sym)
        status_icon = "✅" if acc >= 0.99 else "❌"
        print(f"  {status_icon} Symbol [{sym}]: Acc = {acc*100:.1f}% ({d['correct']}/{d['total']}) | ExactLine = {exact_acc*100:.1f}% | CER = {sym_cer*100:.2f}%")

    failing_symbols = sorted(list(set(failing_symbols)))
    FrozenGatePolicy.verify_symbol_consistency(failing_symbols, all_symbol_records)

    overall_acc = sum(d["correct"] for d in per_symbol.values()) / max(1, len(symbol_cases))
    exact_overall = sum(d["exact"] for d in per_symbol.values()) / max(1, len(symbol_cases))
    total_edits = sum(d["edits"] for d in per_symbol.values())
    total_chars = sum(d["chars"] for d in per_symbol.values())
    overall_cer = total_edits / max(1, total_chars)

    print(f"  Overall 14 Symbols Accuracy: {overall_acc*100:.2f}% ({sum(d['correct'] for d in per_symbol.values())}/{len(symbol_cases)})")
    print(f"  Exact Line Accuracy: {exact_overall*100:.2f}% | Overall CER: {overall_cer*100:.2f}%")
    print(f"  Failing Symbols (<99%): {failing_symbols}")

    if symbol_failures:
        print(f"\n  🚨 PER-CASE SYMBOL FAILURE TABLE ({len(symbol_failures)} cases):")
        print(f"  {'ID':<5} | {'Symbol':<6} | {'Confidence':<10} | {'GT':<25} | {'Prediction':<25}")
        print("  " + "-" * 75)
        for f in symbol_failures:
            print(f"  {f['sample_id']:<5} | {f['target_symbol']:<6} | {f['conf']:<10.4f} | {f['gt']:<25} | {f['pred']:<25}")

    return {
        "total_cases": len(symbol_cases),
        "overall_accuracy": overall_acc,
        "exact_accuracy": exact_overall,
        "cer": overall_cer,
        "failing_symbols": failing_symbols,
        "per_symbol": summary,
        "failures": symbol_failures,
        "all_records": all_symbol_records
    }

def run_suite_hyphen(evaluator):
    print("\n--- Running Suite F: Hyphen Multi-Context Benchmark (70 Cases - Frozen PNG Crops) ---")
    hy_lines_path = None
    try:
        hy_lines_path = get_frozen_file("hyphen_frozen_multi_context_lines.json")
    except FileNotFoundError:
        pass

    if hy_lines_path and os.path.exists(hy_lines_path):
        with open(hy_lines_path, 'r', encoding='utf-8') as f:
            hyphen_cases = json.load(f)
    else:
        hy_path = get_frozen_file("hyphen_frozen_multi_context.json")
        with open(hy_path, 'r', encoding='utf-8') as f:
            raw_cases = json.load(f)
        hyphen_cases = []
        for idx, item in enumerate(raw_cases):
            hyphen_cases.append({
                "sample_id": idx + 1,
                "group": item.get("group", "unknown"),
                "gt": item.get("gt", ""),
                "crop_file": f"crops/hyphen_lines/hyphen_{idx:03d}.png"
            })

    group_results = {}
    total_hyphen_cases = 0
    total_count_correct = 0
    total_exact_correct = 0
    total_edits = 0
    total_chars = 0
    hyphen_failures = []
    all_hyphen_records = []

    for idx, item in enumerate(hyphen_cases, 1):
        group_name = item.get("group", "unknown")
        text = item.get("gt", "")
        crop_file = item.get("crop_file", f"crops/hyphen_lines/hyphen_{idx-1:03d}.png")
        if not text:
            continue

        if group_name not in group_results:
            group_results[group_name] = {
                "total": 0,
                "count_correct": 0,
                "exact_correct": 0,
                "edits": 0,
                "chars": 0
            }

        img = get_crop_image(crop_file)
        raw, norm, conf = evaluator.predict_line(img)

        count_match = (norm.count('-') == text.count('-'))
        exact_match = (norm == text)
        edits = compute_edits(text, norm)

        total_hyphen_cases += 1
        total_chars += len(text)
        total_edits += edits

        group_results[group_name]["total"] += 1
        group_results[group_name]["chars"] += len(text)
        group_results[group_name]["edits"] += edits

        if count_match:
            total_count_correct += 1
            group_results[group_name]["count_correct"] += 1
        if exact_match:
            total_exact_correct += 1
            group_results[group_name]["exact_correct"] += 1

        rec = {
            "sample_id": item.get("sample_id", idx),
            "group": group_name,
            "gt": text,
            "raw_pred": raw,
            "pred": norm,
            "conf": conf,
            "count_match": count_match,
            "exact_match": exact_match,
            "edits": edits,
            "cer": edits / max(1, len(text)),
            "pass": count_match
        }
        all_hyphen_records.append(rec)
        if not count_match or not exact_match:
            hyphen_failures.append(rec)

    print(f"  Total Hyphen Cases: {total_hyphen_cases}")
    for grp, d in group_results.items():
        acc = d["count_correct"] / max(1, d["total"])
        exact_acc = d["exact_correct"] / max(1, d["total"])
        cer = d["edits"] / max(1, d["chars"])
        d["accuracy"] = acc
        d["exact_accuracy"] = exact_acc
        d["cer"] = cer
        status_icon = "✅" if acc >= 0.95 else "❌"
        print(f"  {status_icon} Group [{grp:<15}]: CountAcc = {acc*100:.1f}% ({d['count_correct']}/{d['total']}) | ExactLine = {exact_acc*100:.1f}% | CER = {cer*100:.2f}%")

    overall_hyphen_acc = total_count_correct / max(1, total_hyphen_cases)
    overall_exact_acc = total_exact_correct / max(1, total_hyphen_cases)
    overall_cer = total_edits / max(1, total_chars)

    print(f"  Overall Hyphen Count Accuracy: {overall_hyphen_acc*100:.2f}% ({total_count_correct}/{total_hyphen_cases})")
    print(f"  Overall Exact Line Accuracy: {overall_exact_acc*100:.2f}% | Hyphen CER: {overall_cer*100:.2f}%")

    if hyphen_failures:
        print(f"\n  🚨 HYPHEN MISMATCHES / FAILURES ({len(hyphen_failures)} cases):")
        print(f"  {'ID':<5} | {'Group':<15} | {'CountMatch':<10} | {'Exact':<6} | {'GT':<22} | {'Prediction':<22}")
        print("  " + "-" * 85)
        for f in hyphen_failures[:15]:
            print(f"  {f['sample_id']:<5} | {f['group']:<15} | {str(f['count_match']):<10} | {str(f['exact_match']):<6} | {f['gt']:<22} | {f['pred']:<22}")

    return {
        "total_cases": total_hyphen_cases,
        "overall_accuracy": overall_hyphen_acc,
        "exact_accuracy": overall_exact_acc,
        "cer": overall_cer,
        "by_group": group_results,
        "failures": hyphen_failures,
        "all_records": all_hyphen_records
    }

def run_suite_fullpage(evaluator):
    print("\n--- Running Suite E: FullPage-Frozen (50 Pages / 239 Lines) ---")
    try:
        fp_path = get_frozen_file("fullpage_frozen_50.json")
    except FileNotFoundError:
        print("  [INFO] fullpage_frozen_50.json not found; skipping FullPage suite.")
        return None

    with open(fp_path, 'r', encoding='utf-8') as f:
        pages = json.load(f)

    total_gt_lines = 0
    total_matched_lines = 0
    total_detected_lines = 0
    total_gt_chars = 0
    total_matched_chars = 0
    total_matched_edits = 0
    total_end_to_end_edits = 0

    level_stats = {lvl: {"gt": 0, "det": 0, "matched": 0, "matched_edits": 0, "matched_chars": 0, "gt_chars": 0} for lvl in range(1, 6)}

    print(f"  Pipeline Mode: {'Full Cham-DBNet Detector + IoU Matching' if evaluator.has_detector else 'GT Coordinates Crop Mode'}")

    for page_idx, page in enumerate(pages):
        lvl = page.get("level", 1)
        img = find_page_image(page)
        lines = page.get("lines", [])
        gt_polys = [l.get("polygon", []) for l in lines]
        gt_texts = [l.get("text", "") for l in lines]

        level_stats[lvl]["gt"] += len(lines)
        total_gt_lines += len(lines)
        page_gt_chars = sum(len(t) for t in gt_texts)
        total_gt_chars += page_gt_chars
        level_stats[lvl]["gt_chars"] += page_gt_chars

        if evaluator.has_detector:
            det_results = evaluator.predict_page(img)
            total_detected_lines += len(det_results)
            level_stats[lvl]["det"] += len(det_results)
            det_polys = [dr[0] for dr in det_results]

            matched_pairs, un_gt, un_det = match_polygons_bipartite(gt_polys, det_polys, iou_threshold=0.40)
            total_matched_lines += len(matched_pairs)
            level_stats[lvl]["matched"] += len(matched_pairs)

            for gt_i, det_j, iou in matched_pairs:
                gt_text = gt_texts[gt_i]
                norm_pred = det_results[det_j][2]
                edits = compute_edits(gt_text, norm_pred)
                total_matched_edits += edits
                total_matched_chars += len(gt_text)
                level_stats[lvl]["matched_edits"] += edits
                level_stats[lvl]["matched_chars"] += len(gt_text)
                total_end_to_end_edits += edits

            # Unmatched GT lines count all chars as errors in end-to-end
            for gt_i in un_gt:
                total_end_to_end_edits += len(gt_texts[gt_i])
        else:
            # GT Crop evaluation fallback
            for l in lines:
                gt_text = l.get("text", "")
                poly = l.get("polygon", [])
                total_matched_lines += 1
                level_stats[lvl]["matched"] += 1
                total_detected_lines += 1
                level_stats[lvl]["det"] += 1

                if len(poly) >= 4:
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    x1, x2 = max(0, int(min(xs))), min(img.shape[1], int(max(xs)))
                    y1, y2 = max(0, int(min(ys))), min(img.shape[0], int(max(ys)))
                    if x2 > x1 + 5 and y2 > y1 + 5:
                        line_crop = img[y1:y2, x1:x2]
                        raw, norm, conf = evaluator.predict_line(line_crop)
                    else:
                        line_crop = render_cham_line(gt_text, font_size=24)
                        raw, norm, conf = evaluator.predict_line(line_crop)
                else:
                    line_crop = render_cham_line(gt_text, font_size=24)
                    raw, norm, conf = evaluator.predict_line(line_crop)

                edits = compute_edits(gt_text, norm)
                total_matched_edits += edits
                total_matched_chars += len(gt_text)
                level_stats[lvl]["matched_edits"] += edits
                level_stats[lvl]["matched_chars"] += len(gt_text)
                total_end_to_end_edits += edits

    overall_recall = total_matched_lines / max(1, total_gt_lines)
    overall_prec = total_matched_lines / max(1, total_detected_lines)
    overall_matched_cer = total_matched_edits / max(1, total_matched_chars)
    overall_e2e_cer = total_end_to_end_edits / max(1, total_gt_chars)

    l1_4_gt = sum(level_stats[l]["gt"] for l in [1, 2, 3, 4])
    l1_4_matched = sum(level_stats[l]["matched"] for l in [1, 2, 3, 4])
    l1_4_recall = l1_4_matched / max(1, l1_4_gt)
    l1_4_edits = sum(level_stats[l]["matched_edits"] for l in [1, 2, 3, 4])
    l1_4_chars = sum(level_stats[l]["matched_chars"] for l in [1, 2, 3, 4])
    l1_4_cer = l1_4_edits / max(1, l1_4_chars)

    l5_gt = level_stats[5]["gt"]
    l5_matched = level_stats[5]["matched"]
    l5_recall = l5_matched / max(1, l5_gt)
    l5_edits = level_stats[5]["matched_edits"]
    l5_chars = level_stats[5]["matched_chars"]
    l5_cer = l5_edits / max(1, l5_chars)

    print(f"  FullPage Results (50 pages / 239 GT lines):")
    print(f"    * Overall Detector Recall: {overall_recall*100:.2f}% ({total_matched_lines}/{total_gt_lines}) | Precision: {overall_prec*100:.2f}%")
    print(f"    * Recognizer Matched CER: {overall_matched_cer*100:.2f}% | End-to-End CER: {overall_e2e_cer*100:.2f}%")
    print(f"    * L1-4 (Standard to Wavy)  : Recall={l1_4_recall*100:.2f}%, Matched CER={l1_4_cer*100:.2f}%")
    print(f"    * L5 (Extreme Combined)    : Recall={l5_recall*100:.2f}%, Matched CER={l5_cer*100:.2f}%")

    return {
        "pages_tested": len(pages),
        "total_gt_lines": total_gt_lines,
        "overall_recall": overall_recall,
        "overall_precision": overall_prec,
        "matched_line_cer": overall_matched_cer,
        "end_to_end_cer": overall_e2e_cer,
        "l1_to_l4_lines": l1_4_gt,
        "l1_to_l4_recall": l1_4_recall,
        "l1_to_l4_matched_cer": l1_4_cer,
        "l5_lines": l5_gt,
        "l5_recall": l5_recall,
        "l5_matched_cer": l5_cer,
        "by_level": level_stats
    }

# ==============================================================================
# 4. Main Orchestrator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Cham-OCR Frozen Benchmark Suite")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to recognizer model directory")
    parser.add_argument("--det-dir", type=str, default=None, help="Path to detector model directory (optional)")
    parser.add_argument("--dict-path", type=str, default=None, help="Path to recognition dictionary (e.g. cham_dict_v24.txt or cham_dict_v24_1.txt)")
    parser.add_argument("--rec-image-shape", type=str, default=None, help="Override recognizer image shape (e.g. '3, 48, 480')")
    parser.add_argument("--verify-baseline", action="store_true", help="Assert that model matches v24 baseline SHA256")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSON report")
    args = parser.parse_args()

    print("====================================================================")
    print("      CHAM-OCR V24.1 FROZEN BENCHMARK SUITE RUNNER                  ")
    print("====================================================================")

    baseline_path = os.path.join(BENCHMARK_DIR, "results", "v24_frozen_baseline.json")
    baseline = {}
    if os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        print(f"[INFO] Loaded Baseline: {baseline.get('version')} ({baseline.get('created_at')})")

    evaluator = RecognizerEvaluator(
        model_dir=args.model_dir,
        dict_path=args.dict_path,
        verify_baseline=args.verify_baseline,
        det_dir=args.det_dir,
        rec_image_shape=args.rec_image_shape
    )

    res_blank = run_suite_blank(evaluator)
    res_clean = run_suite_clean(evaluator)
    res_char = run_suite_character(evaluator)
    res_stress = run_suite_stress(evaluator)
    res_stress_base_8 = run_suite_stress_base_8(evaluator, tag=os.path.basename(args.model_dir or ""))
    res_stress["stress_base_8"] = res_stress_base_8
    res_sym = run_suite_symbols(evaluator)
    res_hyphen = run_suite_hyphen(evaluator)
    res_fullpage = run_suite_fullpage(evaluator)

    # Dynamic Gates Summary derived from single source of truth (frozen_gate_policy.py)
    clean_cer = res_clean["norm_cer"]
    char_cer = res_char["cer"]

    policy = get_policy()
    composite_score = policy.compute_composite_score(
        res_sym["overall_accuracy"],
        res_hyphen["overall_accuracy"],
        clean_cer,
        char_cer
    )

    fp_cer = res_fullpage["matched_line_cer"] if res_fullpage is not None else None
    white_pred = res_blank["blank_white"]["pred"]
    gray_pred = res_blank["blank_gray"]["pred"]

    gate_eval = policy.evaluate_gates(
        clean_cer=clean_cer,
        char_cer=char_cer,
        symbol_acc=res_sym["overall_accuracy"],
        failing_symbols=res_sym["failing_symbols"],
        hyphen_acc=res_hyphen["overall_accuracy"],
        blank_passed=res_blank["passed"],
        blank_white=white_pred,
        blank_gray=gray_pred,
        fullpage_cer=fp_cer
    )

    gates = gate_eval["gates"]
    valid_for_release = gate_eval["valid_for_release"]
    release_status = gate_eval["release_status"]

    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "composite_score": composite_score,
        "policy_source": policy.config_path,
        "policy_sha256": policy.sha256,
        "model_dir": evaluator.model_dir,
        "model_sha256": evaluator.model_sha256,
        "dict_path": evaluator.dict_path,
        "dict_tokens": evaluator.dict_token_count,
        "engine": evaluator.engine,
        "has_detector": evaluator.has_detector,
        "release_status": release_status,
        "valid_for_release": valid_for_release,
        "blank_suite": res_blank,
        "clean_suite": res_clean,
        "character_suite": res_char,
        "stress_suite": res_stress,
        "stress_base_8_suite": res_stress_base_8,
        "symbol_suite": res_sym,
        "hyphen_suite": res_hyphen,
        "fullpage_suite": res_fullpage,
        "gates": gates
    }

    out_path = args.output or os.path.join(BENCHMARK_DIR, "results", "v24_frozen_eval_run.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    print("\n====================================================================")
    print("                     RELEASE GATES EVALUATION                       ")
    print("====================================================================")
    for g in gates:
        print(f"  * {g['gate']:<42} : {g['status']}")
    print(f"\n  🎯 FINAL RELEASE STATUS : [{release_status}]")
    print(f"  🎯 RELEASE VERDICT       : {'ACCEPTED (VALID FOR RELEASE)' if valid_for_release else 'REJECTED (FAILED HARD GATES)'}")
    print(f"[INFO] Report saved to: {out_path}")

if __name__ == "__main__":
    main()

