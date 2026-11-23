import os
import sys

# Force single-threaded execution for OpenMP and MKL to prevent conflicts between OpenCV and PaddlePaddle on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import base64
import cv2
import re
import math
import hashlib
from datetime import datetime, timezone
import numpy as np

# Monkeypatch NumPy 2.x to support legacy symbols removed/deprecated for PaddleOCR and imgaug
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str]
    }
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'typeDict'):
    np.typeDict = {}

from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import threading
import urllib.parse

_INFERENCE_LOCK = threading.RLock()
_CORRECTION_LOCK = threading.Lock()


class StudioHTTPServer(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True
    allow_reuse_address = True

# Force UTF-8 stdout encoding on Windows
if sys.stdout and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
paddleocr_dir = os.path.join(PROJECT_ROOT, "PaddleOCR")
if paddleocr_dir not in sys.path:
    # Use append instead of insert(0) to prevent shadowing PyPI paddleocr
    sys.path.append(paddleocr_dir)

# Add ocr-training to sys.path so 'scripts.generate_data' can be imported
ocr_training_dir = os.path.join(os.path.dirname(PROJECT_ROOT), "ocr-training")
if not os.path.exists(ocr_training_dir):
    ocr_training_dir = os.path.join(os.path.dirname(PROJECT_ROOT), "ocr_training")
if ocr_training_dir not in sys.path:
    sys.path.insert(0, ocr_training_dir)

# Global variables for OCR models
ocr_models = {}
_PINNED_RESIDENT_MODELS = frozenset({"v24_1_r4", "v24_1_r3_1", "vi", "auto"})
_MAX_LEGACY_RESIDENT_MODELS = int(os.getenv("MAX_LEGACY_RESIDENT_MODELS", "2"))
_legacy_model_lru = []
det_model = None
generic_det_model = None
cham_det_model = None
_cham_det_failed = False
# Model provenance registry (populated at load time) for response traceability
ocr_model_paths = {}
det_model_info = {}
_fingerprint_cache = {}
_model_manifest_cache = {}

# ------------------------------------------------------------------------------
# Detection / quality-gate configuration (stress test 2026-11-04 fixes)
# ------------------------------------------------------------------------------
# DET_MODE: 'hybrid'  (default, Legacy-First): the legacy generic PP-OCRv4 DBNet stays primary, so clean
#                      pages keep exactly the previous output; the fine-tuned Cham-DBNet takes over the page
#                      only when the legacy detector leaves text-like ink uncovered (missed / merged lines).
#           'cham'    : Cham-DBNet only.   'generic': legacy detector only (pre-2026-11-04 behaviour).
DET_MODE = os.getenv("DET_MODE", "hybrid").lower()
DEFAULT_CHAM_MODEL = os.getenv("DEFAULT_CHAM_MODEL", "v24_1_r4")
ADVERTISED_MODEL_IDS = (
    "v24_1_r4",
    "auto",
    "v24_1_r3_1",
    "v24_1_r2_1_6",
    "vi",
    "v25",
    "v24",
    "v23",
)
ALLOWED_MODEL_IDS = (
    "v24_1_r4",
    "auto",
    "v24_1_r3_1",
    "v24_1_r3",
    "v24_1_r2_1_6",
    "v25",
    "v24",
    "v23",
    "v22",
    "vi",
    "en",
    "fr",
    "japan",
)
ALLOWED_SEG_METHODS = ("dbnet", "valley", "adaptive")
ALLOWED_READING_ORDERS = ("auto", "rows")


class UnknownModelError(ValueError):
    """Raised when a client requests a non-existent or uninstalled recognizer model ID."""
    pass
CHAM_DET_DB_THRESH = float(os.getenv("CHAM_DET_DB_THRESH", "0.3"))
CHAM_DET_BOX_THRESH = float(os.getenv("CHAM_DET_BOX_THRESH", "0.4"))
CHAM_DET_UNCLIP = float(os.getenv("CHAM_DET_UNCLIP", "1.8"))
# Lines below this recognition confidence are flagged for human review
LINE_REVIEW_CONF = 0.60
# Fraction of text-like ink outside every detected line that marks the page as possibly incomplete
UNCOVERED_INK_REVIEW_RATIO = 0.12
# Legacy-First gate: try the Cham detector only when the legacy detector leaves this much ink uncovered
LEGACY_GATE_UNCOVERED_RATIO = 0.05
# ... and switch to it only if it covers clearly more ink than the legacy detector
DET_SWITCH_MIN_GAIN = 0.03
# Legacy-First gate trigger #3: two horizontally overlapping legacy boxes whose vertical overlap
# exceeds this fraction of the smaller box are two tilted / wavy lines the legacy detector could not
# separate (or one box swallowing a neighbour). Clean pages stay <= 0.25 (lvl1-2) and lvl3 pages
# trigger only a comparison, never a switch, because Cham-DBNet then finds the same line count.
LEGACY_GATE_STACKED_OVERLAP = float(os.getenv("LEGACY_GATE_STACKED_OVERLAP", "0.45"))
# Optional vertical padding (fraction of line height) around Cham-DBNet quad boxes before recognition.
# Default 0: in quad mode, padding (0.15) and larger unclip raised CER on tightly spaced pages because
# rigid quad crops pick up neighbour-line ink; in poly mode (CHAM_DET_UNCLIP=1.8, CHAM_DET_POLY_VPAD=0.15,
# CHAM_DET_POLY_MASK=1), neighbour-line ink is masked out along the straightened ribbon.
CHAM_DET_CROP_PAD = float(os.getenv("CHAM_DET_CROP_PAD", "0.0"))
# Cham-DBNet model directory (default: the production Cham-DBNet v1 export). Point it at another export
# (e.g. ocr-training/output/ch_PP-OCRv4_det_cham_v2_1_dual_infer) to A/B test a detector without copying weights.
CHAM_DET_DIR = os.getenv("CHAM_DET_DIR", "").strip() or os.path.join(PROJECT_ROOT, "data", "output", "ch_PP-OCRv4_det_cham_infer")
# Cham-DBNet box type: 'poly' (production default v1_poly_u1.8: DBPostProcess polygon contours +
# curved-line "ribbon" rectification, promoted over quad after cutting Level-5 short:5/a4:5/FullPage-50
# E2E CER without L1-4 regression) or 'quad' (legacy cv2.minAreaRect boxes + perspective crop).
# Only the Cham-DBNet output is affected: legacy generic-detector quads are always cropped as before.
CHAM_DET_BOX_TYPE = os.getenv("CHAM_DET_BOX_TYPE", "poly").strip().lower()
if CHAM_DET_BOX_TYPE not in ("quad", "poly"):
    CHAM_DET_BOX_TYPE = "poly"
# Ribbon rectification (poly mode only): extra vertical padding beyond the detected polygon, as a fraction
# of the page's median line thickness, and whether pixels outside the (dilated) own polygon / inside
# neighbouring line polygons are blanked with the paper colour.
CHAM_DET_POLY_VPAD = float(os.getenv("CHAM_DET_POLY_VPAD", "0.15"))
CHAM_DET_POLY_MASK = os.getenv("CHAM_DET_POLY_MASK", "1").strip().lower() in ("1", "true", "yes", "on")
NORMALIZER_VERSION = "generate_data.normalize_unicode/parse_unicode_clusters"

class OfficialPaddleOCRWrapper:
    def __init__(self, lang):
        self.lang = lang
        self.ocr = None
        
    def _init_ocr(self):
        if self.ocr is None:
            print("DEBUG: [OfficialPaddleOCRWrapper] Importing paddleocr...")
            from paddleocr import PaddleOCR
            ocr_rec_batch_size = int(os.getenv("OCR_REC_BATCH_SIZE", "16"))
            
            import paddleocr
            version_str = getattr(paddleocr, "__version__", "2.0.0")
            print(f"DEBUG: [OfficialPaddleOCRWrapper] paddleocr version={version_str}")
            
            if version_str.startswith("3."):
                # PP-OCRv6 (paddleocr 3.x)
                print("DEBUG: [OfficialPaddleOCRWrapper] Instantiating PP-OCRv6...")
                self.ocr = PaddleOCR(
                    use_textline_orientation=False, 
                    lang=self.lang, 
                    text_recognition_batch_size=ocr_rec_batch_size
                )
            else:
                # PP-OCRv4 / PP-OCRv5 (paddleocr 2.x)
                print("DEBUG: [OfficialPaddleOCRWrapper] Instantiating PP-OCRv4...")
                self.ocr = PaddleOCR(
                    use_angle_cls=False, 
                    show_log=False, 
                    lang=self.lang, 
                    rec=True, 
                    det=False, 
                    rec_batch_num=ocr_rec_batch_size
                )
            print("DEBUG: [OfficialPaddleOCRWrapper] Instantiated successfully.")
            
    def __call__(self, img_list):
        print(f"DEBUG: [OfficialPaddleOCRWrapper] Received {len(img_list)} images")
        self._init_ocr()
        if not img_list:
            return [], 0.0
            
        try:
            import paddleocr
            version_str = getattr(paddleocr, "__version__", "2.0.0")
            if version_str.startswith("3."):
                print("DEBUG: [OfficialPaddleOCRWrapper] Calling PP-OCRv6 text_rec_model...")
                rec_res = self.ocr.paddlex_pipeline._pipeline.text_rec_model(img_list)
                print("DEBUG: [OfficialPaddleOCRWrapper] PP-OCRv6 prediction finished.")
                results = []
                for r in rec_res:
                    pred_text = r['rec_text']
                    conf = r['rec_score']
                    results.append((pred_text, float(conf)))
                return results, 0.0
            else:
                # PaddleOCR 2.x batching
                print("DEBUG: [OfficialPaddleOCRWrapper] Calling PP-OCRv4 text_recognizer...")
                rec_res, _ = self.ocr.text_recognizer(img_list)
                print("DEBUG: [OfficialPaddleOCRWrapper] PP-OCRv4 prediction finished.")
                results = []
                for res in rec_res:
                    pred_text, conf = res if res else ("", 0.0)
                    results.append((pred_text, float(conf)))
                return results, 0.0
        except Exception as e:
            print(f"DEBUG: [OfficialPaddleOCRWrapper] Batch failed, falling back to sequential: {e}")
            results = []
            for img in img_list:
                try:
                    res = self.ocr.ocr(img, det=False, cls=False)
                    if res and res[0] and res[0][0]:
                        text, conf = res[0][0]
                        results.append((text, conf))
                    else:
                        results.append(("", 0.0))
                except Exception:
                    results.append(("", 0.0))
            return results, 0.0

_ROUTER_ILLEGAL_MEDIAL_RE = re.compile(r'[\uAA33\uAA34]{2,}|[\uAA2F\uAA30]{2,}')
_ROUTER_EDITORIAL_SYMS = set('-/_*[]{}<>=+%@#')
_ROUTER_GIBBERISH_TOKEN_RE = re.compile(
    r'(?:[a-z\u00E0-\u1EF9][A-Z]'
    r'|[A-Z\u00C0-\u01AF]{2,}[a-z\u00E0-\u1EF9]'
    r'|[A-Za-z\u00C0-\u1EF9]+[0-9]+[A-Za-z0-9\u00C0-\u1EF9]*'
    r'|[0-9]+(?!(?:er|e|st|nd|rd|th)\b)[A-Za-z\u00C0-\u1EF9]+'
    r'|\bZ[BCDFGHJKLMNPQRSTVWXYZ\u00C0-\u1EF9]'
    r'|\bXX\)'
    r'|[BCDFGHJKLMNPQRSTVWXZbcdfghjklmnpqrstvwxz]{5,})'
)
_ROUTER_CHAM_SOUP_RE = re.compile(
    r'(?:[{}\[\]_\\]|/[0-9 ]|/[\uAA00-\uAA5F]|[\uAA00-\uAA5F]/'
    r'|[0-9][\uAA00-\uAA5F]|[\uAA00-\uAA5F][0-9]'
    r'|[\uAA00-\uAA4F][\uAA50-\uAA59]+|[\uAA50-\uAA59]+[\uAA00-\uAA4F])'
)
_ROUTER_LATIN_VOWELS = set('aeiouyàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵœæ')


def _cham_char_count(text):
    return sum(1 for ch in (text or '') if '\uAA00' <= ch <= '\uAA5F')


def _is_clean_ascii_numeral_span(text):
    s = (text or '').strip()
    if not s:
        return False
    ascii_digits = sum(1 for ch in s if ch.isascii() and ch.isdigit())
    if ascii_digits < 3:
        return False
    return all((ch.isascii() and (ch.isdigit() or ch in ' -–—./:,()')) for ch in s)


def _is_plausible_latin_span(text):
    """Rejects CTC script-mismatch gibberish (e.g. '2VfWKV)', 'g873986)') while accepting real Latin/Vi/Fr words."""
    s = (text or '').strip()
    if not s:
        return False
    alpha = [ch for ch in s if ('a' <= ch.lower() <= 'z') or ('\u00C0' <= ch <= '\u1EF9')]
    if len(alpha) < 2:
        return False
    if _ROUTER_GIBBERISH_TOKEN_RE.search(s):
        return False
    v_cnt = sum(1 for ch in alpha if ch.lower() in _ROUTER_LATIN_VOWELS)
    return (v_cnt / float(len(alpha))) >= 0.20


def _has_plausible_latin_word(text):
    """True when at least one whitespace token (>=2 letters) in `text` is a clean non-gibberish Latin/Vi/Fr word."""
    for tok in (text or '').split():
        cleaned = tok.strip('.,;:!?()[]{}"\'-–—/')
        if len(cleaned) >= 2 and _is_plausible_latin_span(cleaned):
            return True
    return False


def _find_coarse_phrase_spans(crop_img, max_chunks=6):
    """Segments a line crop into at most `max_chunks` phrase spans while preserving standalone Cham headwords."""
    spans = _find_horizontal_word_spans(crop_img)
    if len(spans) <= 1:
        return spans
    h, w = crop_img.shape[:2]
    min_chunk_w = max(16, int(round(0.55 * h)))

    # Merge tiny spans (< min_chunk_w) into their closest neighbour while keeping headword-width tokens intact
    merged = list(spans)
    while len(merged) > 2:
        widths = [b - a for a, b in merged]
        smallest_i = int(np.argmin(widths))
        if widths[smallest_i] >= min_chunk_w and len(merged) <= max_chunks:
            break
        if len(merged) <= max_chunks and all(wd >= min_chunk_w for wd in widths):
            break
        if widths[smallest_i] < min_chunk_w:
            if smallest_i == 0:
                pair_i = 0
            elif smallest_i == len(merged) - 1:
                pair_i = len(merged) - 2
            else:
                gap_l = merged[smallest_i][0] - merged[smallest_i - 1][1]
                gap_r = merged[smallest_i + 1][0] - merged[smallest_i][1]
                pair_i = smallest_i - 1 if gap_l <= gap_r else smallest_i
        else:
            gaps = [merged[k + 1][0] - merged[k][1] for k in range(len(merged) - 1)]
            pair_i = int(np.argmin(gaps))
        merged[pair_i] = (merged[pair_i][0], merged[pair_i + 1][1])
        merged.pop(pair_i + 1)
    return merged if len(merged) >= 2 else [(0, w)]


def _find_horizontal_word_spans(crop_img):
    """Segments a mixed-script line crop into word/phrase x-intervals [x0, x1] at clean vertical whitespace valleys."""
    if not isinstance(crop_img, np.ndarray) or crop_img.ndim < 2 or crop_img.size == 0:
        return []
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if crop_img.ndim == 3 else crop_img
    h, w = gray.shape[:2]
    if h < 10 or w < int(2.4 * h):
        return [(0, w)]
    bg = float(np.percentile(gray, 85))
    ink = (bg - gray.astype(np.float32)) > max(22.0, 0.14 * bg)
    y0, y1 = int(h * 0.12), max(int(h * 0.12) + 1, int(h * 0.88))
    is_gap = ink[y0:y1, :].sum(axis=0) == 0
    min_gap = max(5, int(round(0.16 * h)))

    spans = []
    in_word = False
    start = 0
    gap_run = 0
    for x in range(w):
        if not is_gap[x]:
            if not in_word:
                start = max(0, x - 3)
                in_word = True
            gap_run = 0
        elif in_word:
            gap_run += 1
            if gap_run >= min_gap:
                end = min(w, x - gap_run + 4)
                if end - start >= max(8, int(0.25 * h)):
                    spans.append((start, end))
                in_word = False
    if in_word:
        end = w
        if end - start >= max(8, int(0.25 * h)):
            spans.append((start, end))
    return spans if len(spans) >= 2 else [(0, w)]


def _cham_script_ratio(text):
    non_space = [ch for ch in (text or '') if not ch.isspace()]
    if not non_space:
        return 0.0
    cham_cnt = sum(1 for ch in non_space if '\uAA00' <= ch <= '\uAA5F')
    return cham_cnt / float(len(non_space))


def _latin_alpha_ratio(text):
    non_space = [ch for ch in (text or '') if not ch.isspace()]
    if not non_space:
        return 0.0
    lat_cnt = sum(1 for ch in non_space if ('a' <= ch.lower() <= 'z') or ('\u00C0' <= ch <= '\u1EF9'))
    return lat_cnt / float(len(non_space))


def _latin_alpha_count(text):
    return sum(1 for ch in (text or '') if ('a' <= ch.lower() <= 'z') or ('\u00C0' <= ch <= '\u1EF9'))


def _has_viet_diacritics(text):
    return any('\u00C0' <= ch <= '\u1EF9' for ch in (text or ''))


class AutoRoutingOCRWrapper:
    """Tri-specialist adaptive router combining W=480 Cham primary, W=320 archival Cham adjudicator, and Vi/Latin OCR."""

    def __init__(self, cham_model, viet_model, cham_w320_model=None, primary_ver=None, w320_ver='v24_1_r3_1'):
        self.cham_model = cham_model
        self.viet_model = viet_model
        self.cham_w320_model = cham_w320_model
        self.primary_ver = primary_ver or DEFAULT_CHAM_MODEL
        self.w320_ver = w320_ver
        self.last_selected_versions = []

    def predict_with_provenance(self, img_list):
        """Executes tri-specialist routing and returns (results, selected_versions, elapsed) directly."""
        if not img_list:
            self.last_selected_versions = []
            return [], [], 0.0

        try:
            from scripts.generate_data import normalize_unicode as _norm_u
        except Exception:
            _norm_u = lambda x: x

        # 1. Fast primary pass with W=480 Cham specialist
        cham_res, _ = self.cham_model(img_list)

        # 2. Selective W=320 Cham adjudication on medium-width lower-confidence textline crops
        w320_indices = []
        w320_imgs = []
        if self.cham_w320_model is not None:
            for i, (t480, c480) in enumerate(cham_res):
                im = img_list[i]
                h, w = im.shape[:2] if hasattr(im, 'shape') and len(im.shape) >= 2 else (48, 320)
                ar48 = float(w) * 48.0 / max(1.0, float(h))
                if float(c480) < 0.905 and 104.0 <= ar48 < 420.0:
                    w320_indices.append(i)
                    w320_imgs.append(im)

        w320_map = {}
        if w320_imgs:
            w320_batch, _ = self.cham_w320_model(w320_imgs)
            for idx_orig, res320 in zip(w320_indices, w320_batch):
                w320_map[idx_orig] = res320

        best_cham = []
        best_cham_ver = []
        for i, (raw_t480, c480) in enumerate(cham_res):
            c480 = float(c480)
            t480 = _norm_u(raw_t480) if any('\uAA00' <= ch <= '\uAA5F' for ch in (raw_t480 or '')) else (raw_t480 or '')
            chosen_txt, chosen_conf, chosen_ver = t480, c480, self.primary_ver

            if i in w320_map:
                raw_t320, c320 = w320_map[i]
                c320 = float(c320)
                t320 = _norm_u(raw_t320) if any('\uAA00' <= ch <= '\uAA5F' for ch in (raw_t320 or '')) else (raw_t320 or '')
                cal480 = calibrate_line_confidence(c480, self.primary_ver, t480)
                cal320 = calibrate_line_confidence(c320, self.w320_ver, t320)
                score480 = cal480 - (0.08 if _ROUTER_ILLEGAL_MEDIAL_RE.search(raw_t480 or '') else 0.0)
                score320 = cal320 - (0.08 if _ROUTER_ILLEGAL_MEDIAL_RE.search(raw_t320 or '') else 0.0)
                dropped_editorial = any(t480.count(s) > t320.count(s) for s in _ROUTER_EDITORIAL_SYMS) and c480 >= 0.70
                if (not dropped_editorial) and (score320 > score480 + 0.020):
                    chosen_txt, chosen_conf, chosen_ver = t320, c320, self.w320_ver

            best_cham.append((chosen_txt, chosen_conf))
            best_cham_ver.append(chosen_ver)

        # 3. Selective Vietnamese / Latin multiscript handoff (v2.3: also probes low-Cham-ratio lines & Cham-soup lines
        #    so lines like 'Năm 2000-2025', '1990-2000', and 'ꨂꨣꩃ ꨌꩌ (Người Chăm)' are routed/split accurately)
        viet_indices = []
        viet_imgs = []
        for i, (c_txt, c_conf) in enumerate(best_cham):
            c_ratio = _cham_script_ratio(c_txt)
            has_cham_soup = bool(_ROUTER_CHAM_SOUP_RE.search(c_txt or ''))
            if (c_ratio < 0.62 and c_conf < 0.85) or c_ratio < 0.35 or c_conf < 0.34 or (has_cham_soup and c_conf < 0.90):
                viet_indices.append(i)
                viet_imgs.append(img_list[i])

        viet_map = {}
        if viet_imgs:
            viet_batch, _ = self.viet_model(viet_imgs)
            for idx_orig, v_res in zip(viet_indices, viet_batch):
                viet_map[idx_orig] = v_res

        # 4. Script-aware adjudication + Intra-Line Horizontal Script-Span Splitting (Auto Router v2.3)
        final_results = [None] * len(img_list)
        selected_vers = []
        for i in range(len(img_list)):
            c_txt, c_conf = best_cham[i]
            c_ver = best_cham_ver[i]
            if i not in viet_map:
                final_results[i] = (c_txt, c_conf)
                selected_vers.append(c_ver)
                continue

            v_txt, v_conf = viet_map[i]
            v_conf = float(v_conf)
            c_ratio = _cham_script_ratio(c_txt)
            c_cnt = _cham_char_count(c_txt)
            v_alpha = _latin_alpha_ratio(v_txt)
            v_lat_cnt = _latin_alpha_count(v_txt)
            has_cham_soup = bool(_ROUTER_CHAM_SOUP_RE.search(c_txt or ''))

            take_viet = False
            if v_alpha >= 0.42:
                if c_ratio < 0.60:
                    # ASCII digit/symbol soup from feeding Latin/Vietnamese into Cham CTC -> route to Vi/Latin
                    take_viet = (v_conf >= 0.52 and v_conf >= c_conf - 0.08)
                else:
                    # Coherent Cham script present -> protect historical Cham folios unless Vi is overwhelmingly confident
                    take_viet = (c_conf < 0.50 and v_conf >= 0.88 and v_alpha >= 0.70)
            elif c_ratio < 0.35 and v_lat_cnt >= 2 and (v_alpha >= 0.18 or _has_viet_diacritics(v_txt)):
                # Digit/punctuation-heavy Vietnamese lines (e.g. "Năm 2000-2025", "Số trang: 0123456789")
                take_viet = (v_conf >= 0.55 and (v_conf >= c_conf - 0.05 or _has_viet_diacritics(v_txt)))
            elif c_ratio < 0.25 and v_lat_cnt == 0 and _is_clean_ascii_numeral_span(v_txt):
                # Pure ASCII numeral/year-range lines where Cham CTC mixed script ('1꩐0-200') or collapsed repeated '00'/'99' ('1990-200' vs '1990-2000');
                # clean non-collapsed numerals ('123 - 456', '7-8') stay on the Cham specialist.
                v_digits = sum(1 for ch in (v_txt or '') if ch.isascii() and ch.isdigit())
                c_ascii_digits = sum(1 for ch in (c_txt or '') if ch.isascii() and ch.isdigit())
                ctc_collapsed_rep = (('00' in (v_txt or '') or '99' in (v_txt or '')) and v_digits > c_ascii_digits)
                take_viet = (v_conf >= 0.84 and (has_cham_soup or ctc_collapsed_rep))
            elif c_conf < 0.25 and v_conf > c_conf + 0.25:
                take_viet = True

            # Auto Router v2.3: Symmetric Intra-Line Horizontal Script-Span Splitting
            # Fast-skip 100% pure high-confidence Latin/Fr/Vi lines and 100% pure high-confidence Cham lines
            is_clean_pure_latin = (
                take_viet
                and v_conf >= 0.86
                and v_alpha >= 0.62
                and not _ROUTER_GIBBERISH_TOKEN_RE.search(v_txt or '')
            )
            is_clean_pure_cham = (
                (not take_viet)
                and c_conf >= 0.84
                and c_ratio >= 0.75
                and not has_cham_soup
            )
            if (
                not is_clean_pure_latin
                and not is_clean_pure_cham
                and (c_cnt >= 2 or (c_cnt >= 1 and _ROUTER_GIBBERISH_TOKEN_RE.search(v_txt or '')))
                and v_lat_cnt >= 2
                and v_conf >= 0.55
                and _has_plausible_latin_word(v_txt)
            ):
                split_res = self._try_intraline_script_split(
                    img_list[i], c_ver, _norm_u, default_is_viet=take_viet
                )
                if split_res is not None:
                    final_results[i] = (split_res[0], split_res[1])
                    selected_vers.append(split_res[2])
                    continue

            if take_viet:
                final_results[i] = (v_txt, v_conf)
                selected_vers.append('vi')
            else:
                final_results[i] = (c_txt, c_conf)
                selected_vers.append(c_ver)

        self.last_selected_versions = list(selected_vers)
        return final_results, list(selected_vers), 0.0

    def _try_intraline_script_split(self, crop_img, cham_ver, norm_fn, default_is_viet=False):
        """Splits a mixed Cham+Latin line into <=6 coarse phrase spans with symmetric bidirectional gain gating."""
        spans = _find_coarse_phrase_spans(crop_img, max_chunks=6)
        if len(spans) < 2 or len(spans) > 6:
            return None
        sub_imgs = [crop_img[:, x0:x1] for x0, x1 in spans]
        try:
            c_sub_res, _ = self.cham_model(sub_imgs)
            v_sub_res, _ = self.viet_model(sub_imgs)
        except Exception:
            return None
        if len(c_sub_res) != len(spans) or len(v_sub_res) != len(spans):
            return None

        labels = []
        for (ct, cc), (vt, vc) in zip(c_sub_res, v_sub_res):
            cc, vc = float(cc), float(vc)
            cr = _cham_script_ratio(ct)
            cham_soup = bool(_ROUTER_CHAM_SOUP_RE.search(ct or ''))
            if default_is_viet:
                # Line defaulted to Vi/Fr: only flip a chunk to Cham if it is genuinely Akhar Thrah AND Vi failed/garbled it
                vi_garbled = (vc < 0.82) or bool(_ROUTER_GIBBERISH_TOKEN_RE.search(vt or '')) or (_latin_alpha_ratio(vt) < 0.45)
                if cr >= 0.55 and _cham_char_count(ct) >= 1 and cc >= max(0.55, vc - 0.08) and vi_garbled and not cham_soup:
                    labels.append('cham')
                else:
                    labels.append('vi')
            else:
                # Line defaulted to Cham: only flip a chunk to Vi if Cham failed/garbled AND Vi decoded plausible Latin words
                cham_garbled = (cr < 0.45) or (cc < 0.78) or cham_soup
                min_vc = 0.74 if cham_soup else max(0.74, cc + 0.06)
                if cham_garbled and vc >= min_vc and (_is_plausible_latin_span(vt) or _has_viet_diacritics(vt)):
                    labels.append('vi')
                else:
                    labels.append('cham')

        if len(set(labels)) < 2:
            return None

        # Merge adjacent same-script word spans so each recognizer receives maximal contiguous phrase context!
        grouped = []
        cur_lbl = labels[0]
        cur_x0, cur_x1 = spans[0]
        for lbl, (x0, x1) in zip(labels[1:], spans[1:]):
            if lbl == cur_lbl:
                cur_x1 = x1
            else:
                grouped.append((cur_lbl, cur_x0, cur_x1))
                cur_lbl, cur_x0, cur_x1 = lbl, x0, x1
        grouped.append((cur_lbl, cur_x0, cur_x1))

        cham_grp_imgs = [crop_img[:, x0:x1] for lbl, x0, x1 in grouped if lbl == 'cham']
        vi_grp_imgs = [crop_img[:, x0:x1] for lbl, x0, x1 in grouped if lbl == 'vi']
        c_grp_out = iter(self.cham_model(cham_grp_imgs)[0]) if cham_grp_imgs else iter(())
        v_grp_out = iter(self.viet_model(vi_grp_imgs)[0]) if vi_grp_imgs else iter(())

        parts = []
        weighted_conf = 0.0
        total_w = 0.0
        for lbl, x0, x1 in grouped:
            span_w = max(1.0, float(x1 - x0))
            if lbl == 'cham':
                gt_txt, gt_conf = next(c_grp_out, ('', 0.0))
                gt_txt = norm_fn(gt_txt) if any('\uAA00' <= c <= '\uAA5F' for c in (gt_txt or '')) else (gt_txt or '')
            else:
                gt_txt, gt_conf = next(v_grp_out, ('', 0.0))
            gt_txt = (gt_txt or '').strip()
            if gt_txt:
                parts.append(gt_txt)
                weighted_conf += float(gt_conf) * span_w
                total_w += span_w

        if len(parts) < 2 or total_w <= 0:
            return None
        merged_txt = ' '.join(parts)
        if _cham_char_count(merged_txt) < 1 or _latin_alpha_count(merged_txt) < 2:
            return None
        return merged_txt, round(weighted_conf / total_w, 6), f"{cham_ver}+vi"

    def __call__(self, img_list):
        results, _, elapsed = self.predict_with_provenance(img_list)
        return results, elapsed

def validate_requested_model(raw_model):
    """Validates a client-supplied model identifier against ALLOWED_MODEL_IDS and local installation."""
    if raw_model is None:
        ver = DEFAULT_CHAM_MODEL
    elif isinstance(raw_model, str):
        ver = raw_model.strip() or DEFAULT_CHAM_MODEL
    else:
        raise UnknownModelError("Invalid 'model' parameter: expected a model identifier string.")

    if ver not in ALLOWED_MODEL_IDS:
        raise UnknownModelError(
            f"Unknown model '{ver}'. Allowed models: {', '.join(ADVERTISED_MODEL_IDS)}"
        )

    if ver not in ('auto', 'vi', 'en', 'fr', 'japan'):
        mdir = os.path.join(PROJECT_ROOT, "data", "output", f"rec_cham_inference_{ver}")
        has_weights = os.path.exists(os.path.join(mdir, "inference.pdiparams")) and (
            os.path.exists(os.path.join(mdir, "inference.pdmodel"))
            or os.path.exists(os.path.join(mdir, "inference.json"))
        )
        if not os.path.isdir(mdir) or not has_weights:
            raise UnknownModelError(
                f"Model '{ver}' is not installed on this server. Allowed installed models: "
                f"{', '.join(m['id'] for m in list_available_models()['models'] if m.get('installed'))}"
            )
    return ver


def _register_loaded_model(version, model_obj):
    """Caches `model_obj` while bounding resident non-core legacy C++ predictors to prevent OOM on multi-model sweeps."""
    import gc
    with _INFERENCE_LOCK:
        pinned = set(_PINNED_RESIDENT_MODELS) | {DEFAULT_CHAM_MODEL}
        if version not in pinned:
            if version in _legacy_model_lru:
                _legacy_model_lru.remove(version)
            while len(_legacy_model_lru) >= max(1, _MAX_LEGACY_RESIDENT_MODELS):
                evict_ver = _legacy_model_lru.pop(0)
                if evict_ver not in pinned and evict_ver in ocr_models:
                    ocr_models.pop(evict_ver, None)
                    gc.collect()
            _legacy_model_lru.append(version)
        ocr_models[version] = model_obj
        return model_obj


def get_ocr_model(version):
    """Loads and caches the PaddleOCR model for the given version dynamically"""
    global ocr_models
    version = validate_requested_model(version)
    with _INFERENCE_LOCK:
        if version in ocr_models:
            if version in _legacy_model_lru:
                _legacy_model_lru.remove(version)
                _legacy_model_lru.append(version)
            return ocr_models[version]
        
    if version in ['en', 'fr', 'japan']:
        ocr_model = OfficialPaddleOCRWrapper(lang=version)
        ocr_models[version] = ocr_model
        return ocr_model
        
    if version == 'auto':
        cham_model = get_ocr_model(DEFAULT_CHAM_MODEL)
        w320_ver = 'v24_1_r3_1'
        w320_dir = os.path.join(PROJECT_ROOT, "data", "output", f"rec_cham_inference_{w320_ver}")
        cham_w320 = get_ocr_model(w320_ver) if (w320_ver != DEFAULT_CHAM_MODEL and os.path.exists(w320_dir)) else None
        viet_model = get_ocr_model('vi')
        ocr_model = AutoRoutingOCRWrapper(
            cham_model, viet_model, cham_w320_model=cham_w320, primary_ver=DEFAULT_CHAM_MODEL, w320_ver=w320_ver
        )
        ocr_models[version] = ocr_model
        return ocr_model
        
    from tools.infer.predict_rec import TextRecognizer
    import tools.infer.utility as utility
    
    if version == 'vi':
        model_dir = os.path.join(PROJECT_ROOT, "data", "output", "rec_vietnamese_infer")
        dict_path = os.path.join(PROJECT_ROOT, "data", "vietnamese_dict.txt")
        model_ready = os.path.exists(os.path.join(model_dir, "inference.pdiparams")) and os.path.exists(dict_path)
        if not model_ready:
            try:
                print(f"📥 Downloading dedicated Vietnamese OCR model to {model_dir}...")
                os.makedirs(model_dir, exist_ok=True)
                import urllib.request
                base_url = "https://huggingface.co/anbeo2004/paddleocr-v5-rec-vi-custom/resolve/main/"
                dl_files = {
                    "custom_dict.txt": dict_path,
                    "inference.json": os.path.join(model_dir, "inference.json"),
                    "inference.yml": os.path.join(model_dir, "inference.yml"),
                    "inference.pdiparams": os.path.join(model_dir, "inference.pdiparams")
                }
                for fname, ftarget in dl_files.items():
                    if not os.path.exists(ftarget):
                        urllib.request.urlretrieve(base_url + fname, ftarget)
                print("✅ Downloaded Vietnamese OCR model successfully.")
            except Exception as e:
                print(f"⚠️ Failed to auto-download Vietnamese model: {e}")
                ocr_model = OfficialPaddleOCRWrapper(lang='vi')
                ocr_models[version] = ocr_model
                return ocr_model
    else:
        model_dir = os.path.join(PROJECT_ROOT, "data", "output", f"rec_cham_inference_{version}")
        dict_path = os.path.join(PROJECT_ROOT, "data", f"cham_dict_{version}.txt")
        
        if not os.path.isdir(model_dir):
            raise UnknownModelError(
                f"Model directory missing for '{version}': expected rec_cham_inference_{version} (silent fallback is disabled)."
            )
        if not os.path.exists(dict_path):
            bundled_dict = os.path.join(model_dir, f"cham_dict_{version}.txt")
            if os.path.exists(bundled_dict):
                dict_path = bundled_dict
            elif str(version).startswith("v24_1"):
                dict_path = os.path.join(PROJECT_ROOT, "data", "cham_dict_v24_1.txt")
            elif str(version).startswith("v24"):
                dict_path = os.path.join(PROJECT_ROOT, "data", "cham_dict_v24.txt")
            else:
                dict_path = os.path.join(PROJECT_ROOT, "data", "cham_dict_v23.txt")

    import paddle
    has_rec_pdmodel = os.path.exists(os.path.join(model_dir, "inference.pdmodel"))
    has_rec_json = os.path.exists(os.path.join(model_dir, "inference.json"))
    if has_rec_json and not has_rec_pdmodel:
        try:
            paddle_major = int(str(getattr(paddle, "__version__", "0")).split(".")[0])
        except Exception:
            paddle_major = 0
        if paddle_major < 3:
            if version == 'vi':
                print("⚠️  rec_vietnamese_infer is PIR (.json) on Paddle 2.x; falling back to OfficialPaddleOCRWrapper(lang='vi').")
                ocr_model = OfficialPaddleOCRWrapper(lang='vi')
                ocr_models[version] = ocr_model
                return ocr_model
            raise RuntimeError(
                f"Model '{version}' is exported in Paddle 3 PIR format (inference.json) and requires Paddle >= 3.0 "
                f"(current runtime: Paddle {getattr(paddle, '__version__', 'unknown')}). Please select V24.1-R4 or launch Studio with Python 3.13 / Paddle 3.x."
            )
        
    sys_argv_backup = sys.argv
    sys.argv = [sys.argv[0]]
    args = utility.parse_args()
    sys.argv = sys_argv_backup
    
    args.rec_model_dir = model_dir
    args.rec_char_dict_path = dict_path
    
    args.use_gpu = paddle.is_compiled_with_cuda()
    manifest_for_shape = {}
    if version != 'vi':
        manifest_file = os.path.join(model_dir, "model_manifest.json")
        if os.path.exists(manifest_file):
            try:
                with open(manifest_file, 'r', encoding='utf-8') as mf:
                    manifest_for_shape = json.load(mf)
            except Exception:
                manifest_for_shape = {}
    default_rec_shape = "3, 48, 480" if (version == "v25" or str(version).startswith("v24_1_r4")) else "3, 48, 320"
    args.rec_image_shape = str(manifest_for_shape.get("rec_image_shape") or default_rec_shape)
    args.use_space_char = True
    
    # Batch size config: TextRecognizer pre-sorts crops by wh_ratio, and V26.1 uses use_valid_mask=True,
    # so CPU mini-batch=6 cuts 70-line 2-column latency ~2x (preventing Cloudflare HTTP 524) with zero padding drift
    env_batch = os.getenv("OCR_REC_BATCH_SIZE")
    if env_batch:
        ocr_rec_batch_size = int(env_batch)
    else:
        ocr_rec_batch_size = 6 if not paddle.is_compiled_with_cuda() else 16
        
    args.rec_batch_num = ocr_rec_batch_size
    
    if args.use_gpu:
        args.enable_mkldnn = False
        args.ir_optim = True
        args.precision = "fp32"
    else:
        # CPU Settings: Disable MKLDNN by default on Windows to prevent OneDNN deadlocks inside socket handlers
        env_mkldnn = os.getenv("ENABLE_MKLDNN", "False").lower() in ("true", "1")
        args.enable_mkldnn = env_mkldnn
        args.mkldnn_cache_capacity = 32
        
        env_threads = os.getenv("VI_CPU_THREADS") if version == "vi" else os.getenv("CPU_THREADS")
        if env_threads:
            args.cpu_threads = int(env_threads)
        elif version == "vi":
            # Heavy PP-OCRv5_server_rec benefits ~3.2x from multi-thread math without MKLDNN
            args.cpu_threads = min(8, max(1, os.cpu_count() or 2))
        else:
            # Default to 1 thread for lightweight SVTR_LCNet to prevent OpenMP contention
            args.cpu_threads = int(os.getenv("CPU_THREADS", "1"))
            
        args.ir_optim = (version == "vi") # Constant-fold PIR graph on heavy Vi model while MKLDNN stays disabled
        args.precision = "fp32"
        
    print(f"⚙️  Loading PaddleOCR Model {version} (GPU: {args.use_gpu}, MKLDNN: {args.enable_mkldnn}, Batch: {args.rec_batch_num}, Threads: {args.cpu_threads})...")
    ocr_model = TextRecognizer(args)
    ocr_model_paths[version] = {"model_dir": model_dir, "dict_path": dict_path}
    
    # Warmup the model with a dummy image to trigger compilation/allocator caching
    try:
        dummy_w = int(args.rec_image_shape.split(",")[-1].strip())
        dummy_img = np.zeros((48, dummy_w, 3), dtype=np.uint8)
        ocr_model([dummy_img])
        print(f"🔥 Model {version} warmed up successfully.")
    except Exception as e:
        print(f"Failed to warmup model: {e}")
        
    return _register_loaded_model(version, ocr_model)

def _ensure_generic_det_dir():
    """Returns the generic PP-OCRv4 detector directory, downloading it if missing."""
    det_model_dir = os.path.join(PROJECT_ROOT, "data", "output", "ch_PP-OCRv4_det_infer")
    model_file_exists = os.path.exists(os.path.join(det_model_dir, "inference.pdmodel")) or os.path.exists(os.path.join(det_model_dir, "inference.json"))
    if not model_file_exists:
        print(f"📥 Downloading ch_PP-OCRv4_det_infer to {det_model_dir}...")
        os.makedirs(det_model_dir, exist_ok=True)
        url = "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar"
        tar_path = os.path.join(det_model_dir, "det.tar")
        import urllib.request, tarfile, shutil
        urllib.request.urlretrieve(url, tar_path)
        with tarfile.open(tar_path) as tar:
            tar.extractall(det_model_dir)
        sub = os.path.join(det_model_dir, "ch_PP-OCRv4_det_infer")
        if os.path.exists(sub):
            for f in os.listdir(sub):
                shutil.move(os.path.join(sub, f), os.path.join(det_model_dir, f))
            os.rmdir(sub)
        if os.path.exists(tar_path):
            os.remove(tar_path)
        print("✅ Downloaded and extracted ch_PP-OCRv4_det_infer successfully.")
    return det_model_dir

def _clip_poly_det_res(dt_boxes, image_shape):
    """Poly-mode replacement for TextDetector.filter_tag_det_res_only_clip.

    The upstream helper ends with np.array(list_of_polygons): DBPostProcess(box_type='poly') returns
    polygons with different vertex counts, which is a ragged array (ValueError on NumPy >= 1.24).
    Returns a plain list of clipped float32 (N, 2) polygons instead.
    """
    img_h, img_w = image_shape[0:2]
    out = []
    for box in dt_boxes if dt_boxes is not None else []:
        p = np.array(box, dtype=np.float32).reshape(-1, 2)
        if len(p) < 3:
            continue
        p[:, 0] = np.clip(p[:, 0], 0, img_w - 1)
        p[:, 1] = np.clip(p[:, 1], 0, img_h - 1)
        out.append(p)
    return out

def _safe_polygons_from_bitmap(postprocess, pred, _bitmap, dest_width, dest_height):
    """DBPostProcess.polygons_from_bitmap with guards for degenerate contours (bound per instance in poly mode).

    Same algorithm as upstream (approxPolyDP contour -> box_score_fast -> unclip -> min-size filter ->
    rescale), but skips contours whose unclip result is empty / has < 3 points (upstream then calls
    cv2.minAreaRect on an empty array and raises), and keeps coordinates float32 for OpenCV.
    """
    bitmap = _bitmap
    height, width = bitmap.shape
    boxes, scores = [], []
    contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours[:postprocess.max_candidates]:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        points = cv2.approxPolyDP(contour, epsilon, True).reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = postprocess.box_score_fast(pred, points.reshape(-1, 2))
        if postprocess.box_thresh > score:
            continue
        try:
            box = postprocess.unclip(points, postprocess.unclip_ratio)
        except Exception:
            continue
        if len(box) != 1:
            continue
        box = np.array(box, dtype=np.float32).reshape(-1, 2)
        if box.shape[0] < 3:
            continue
        _, sside = postprocess.get_mini_boxes(box.reshape((-1, 1, 2)))
        if sside < postprocess.min_size + 2:
            continue
        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.tolist())
        scores.append(score)
    return boxes, scores

def _build_text_detector(det_model_dir, db_thresh, box_thresh, unclip_ratio, box_type='quad'):
    from tools.infer.predict_det import TextDetector
    import tools.infer.utility as utility

    sys_argv_backup = sys.argv
    sys.argv = [sys.argv[0]]
    args = utility.parse_args()
    sys.argv = sys_argv_backup

    args.use_gpu = False
    args.enable_mkldnn = False
    args.ir_optim = False
    args.det_algorithm = 'DB'
    args.det_model_dir = det_model_dir
    args.det_db_thresh = db_thresh
    args.det_db_box_thresh = box_thresh
    args.det_db_unclip_ratio = unclip_ratio
    if box_type == 'poly':
        args.det_box_type = 'poly'
    print(f"⚙️  Loading PaddleOCR DBNet Detector from {det_model_dir} (thresh={db_thresh}, box={box_thresh}, unclip={unclip_ratio}, box_type={box_type})...")
    detector = TextDetector(args)
    if box_type == 'poly':
        # Instance-level overrides (PaddleOCR sources stay untouched; quad detectors are unaffected)
        detector.filter_tag_det_res_only_clip = _clip_poly_det_res
        post = detector.postprocess_op
        if hasattr(post, 'polygons_from_bitmap') and hasattr(post, 'unclip'):
            post.polygons_from_bitmap = (lambda pred, bm, dw, dh, _p=post: _safe_polygons_from_bitmap(_p, pred, bm, dw, dh))
    print("🔥 PaddleOCR DBNet Detector loaded successfully.")
    return detector

def _is_poly_detector(detector):
    """True when the TextDetector emits variable-length polygons (det_box_type='poly')."""
    args = getattr(detector, 'args', None)
    return str(getattr(args, 'det_box_type', 'quad')).lower() == 'poly'

def get_generic_det_model():
    """Loads and caches the legacy generic PP-OCRv4 DBNet detector."""
    global generic_det_model
    if generic_det_model is None:
        generic_det_model = _build_text_detector(_ensure_generic_det_dir(), 0.25, 0.5, 1.8)
    return generic_det_model

def get_cham_det_model():
    """Loads and caches the fine-tuned Cham-DBNet detector; returns None if unavailable.

    The model is exported in the Paddle 3 PIR format (inference.json + inference.pdiparams).
    Earlier code only looked for the legacy inference.pdmodel file, so this detector was never
    used in production (stress test 2026-11-04: legacy detector returned 4/8, 2/8, 1/5, 3/14 lines
    on wavy / perspective / tightly spaced pages where Cham-DBNet finds 8/8, 7/8, 5/5, 14/14).
    """
    global cham_det_model, _cham_det_failed
    if cham_det_model is not None or _cham_det_failed:
        return cham_det_model
    cham_det_dir = CHAM_DET_DIR
    has_pdmodel = os.path.exists(os.path.join(cham_det_dir, "inference.pdmodel"))
    has_json = os.path.exists(os.path.join(cham_det_dir, "inference.json"))
    has_cham_model = os.path.exists(os.path.join(cham_det_dir, "inference.pdiparams")) and (has_pdmodel or has_json)
    if not has_cham_model:
        _cham_det_failed = True
        return None
    if not has_pdmodel and has_json:
        try:
            import paddle as _paddle
            major_ver = int(str(getattr(_paddle, "__version__", "0")).split(".")[0])
        except Exception:
            major_ver = 0
        if major_ver < 3:
            _cham_det_failed = True
            print("⚠️  Cham DBNet is PIR (.json) and requires Paddle >= 3.0; skipping on Paddle 2.x.")
            return None
    try:
        cham_det_model = _build_text_detector(cham_det_dir, CHAM_DET_DB_THRESH, CHAM_DET_BOX_THRESH, CHAM_DET_UNCLIP,
                                              box_type=CHAM_DET_BOX_TYPE)
        print(f"🌟 Cham Text Detection model ready: {cham_det_dir} (box_type={CHAM_DET_BOX_TYPE})")
    except Exception as e:
        _cham_det_failed = True
        print(f"⚠️  Failed to load Cham DBNet detector ({e}); legacy generic detector only.")
    return cham_det_model

def get_det_model():
    """Loads and caches the primary PaddleOCR TextDetector model for DBNet segmentation.

    DET_MODE='hybrid' (default) and 'generic' keep the legacy generic detector as primary;
    'cham' uses the Cham-DBNet detector as primary (falls back to generic if it cannot load).
    """
    global det_model
    if det_model is not None:
        return det_model
    if DET_MODE == 'cham' and get_cham_det_model() is not None:
        det_model = get_cham_det_model()
        det_model_info.update({"name": "cham_dbnet", "dir": CHAM_DET_DIR, "mode": DET_MODE})
    else:
        det_model = get_generic_det_model()
        det_model_info.update({"name": "generic_ppocrv4_dbnet", "dir": _ensure_generic_det_dir(), "mode": DET_MODE})
    if DET_MODE == 'hybrid':
        get_cham_det_model()  # preload so the first hard page doesn't pay the load cost
    return det_model

def _quad_centerline(pts):
    """(y-at-x function, perpendicular thickness) of a 4-point text quad (tl, tr, br, bl)."""
    p = np.array(pts, dtype=np.float32)
    if p.shape != (4, 2):
        x1, y1, x2, y2 = float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())
        return (lambda x, c=(y1 + y2) / 2.0: c), (y2 - y1)
    left, right = (p[0] + p[3]) / 2.0, (p[1] + p[2]) / 2.0
    if right[0] < left[0]:
        left, right = right, left
    dx = float(right[0] - left[0])
    slope = float(right[1] - left[1]) / dx if dx > 1e-3 else 0.0
    thickness = (float(np.linalg.norm(p[3] - p[0])) + float(np.linalg.norm(p[2] - p[1]))) / 2.0
    return (lambda x, l=left, s=slope: float(l[1]) + s * (x - float(l[0]))), thickness

def _stacked_by_centerline(pts1, pts2, ox1, ox2, ratio=0.5):
    """True when two horizontally overlapping quads are distinct stacked lines: their centerlines,
    evaluated at the middle of the horizontal overlap [ox1, ox2], are further apart than
    `ratio` x the thinner quad's perpendicular thickness."""
    f1, t1 = _quad_centerline(pts1)
    f2, t2 = _quad_centerline(pts2)
    t = min(t1, t2)
    if t <= 0:
        return False
    xm = (ox1 + ox2) / 2.0
    return abs(f1(xm) - f2(xm)) > ratio * t

def merge_line_boxes(boxes, img_w, img_h, col_divider_x=None):
    """
    Merges horizontally separated bounding boxes on the same line into a single unified line crop.
    Prevents splitting a single line into separate fragments across distant words, while never
    bridging across a detected vertical two-column divider (`col_divider_x`).
    """
    if boxes is None or len(boxes) == 0:
        return []
    rects = []
    for b in boxes:
        pts = np.array(b, dtype=np.float32)
        x1 = float(np.min(pts[:, 0]))
        y1 = float(np.min(pts[:, 1]))
        x2 = float(np.max(pts[:, 0]))
        y2 = float(np.max(pts[:, 1]))
        if (x2 - x1) <= 3 or (y2 - y1) <= 3:
            continue
        rects.append([x1, y1, x2, y2, pts])

    if not rects:
        return []
    median_h = float(np.median([r[3] - r[1] for r in rects]))
    rects = sorted(rects, key=lambda r: (r[1], r[0]))

    merged = True
    while merged:
        merged = False
        new_rects = []
        skip = set()
        for i in range(len(rects)):
            if i in skip:
                continue
            r1 = rects[i]
            for j in range(i + 1, len(rects)):
                if j in skip:
                    continue
                r2 = rects[j]
                h1 = r1[3] - r1[1]
                h2 = r2[3] - r2[1]
                w1 = r1[2] - r1[0]
                w2 = r2[2] - r2[0]
                min_h = min(h1, h2)
                min_w = min(w1, w2)
                if min_h <= 0 or min_w <= 0:
                    continue

                v_inter = max(0.0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
                v_overlap = v_inter / min_h
                h_inter = max(0.0, min(r1[2], r2[2]) - max(r1[0], r2[0]))
                h_overlap = h_inter / min_w

                if r1[2] <= r2[0]:
                    x_dist = r2[0] - r1[2]
                elif r2[2] <= r1[0]:
                    x_dist = r1[0] - r2[2]
                else:
                    x_dist = 0.0

                new_y1 = max(0.0, min(r1[1], r2[1]))
                new_y2 = min(float(img_h), max(r1[3], r2[3]))
                new_h = new_y2 - new_y1
                cy1 = (r1[1] + r1[3]) / 2.0
                cy2 = (r2[1] + r2[3]) / 2.0

                # Guard against merging two vertically stacked lines whose AABBs overlap due to wave/tilt
                # (stress test 2026-11-04: test_161_lvl5 lines 6 & 7 had v_overlap=0.57 and h_overlap=1.0)
                if h_overlap > 0.25 and (abs(cy1 - cy2) > 0.45 * min_h or new_h > 1.35 * max(h1, h2)):
                    continue
                if median_h > 0 and new_h > 1.75 * median_h and new_h > 1.25 * max(h1, h2):
                    continue
                # Tilt-aware guard: on perspective / rotated pages the AABB of each line is several
                # times taller than the text itself, so the AABB tests above cannot tell two stacked
                # tilted lines apart (benchmark test_384_a4_lvl5: Cham-DBNet found 19/20 lines, AABB
                # merging collapsed them to 14). Compare the quads' own centerlines at the middle of the
                # horizontal overlap against their true (perpendicular) thickness instead.
                if h_overlap > 0.25 and _stacked_by_centerline(r1[4], r2[4], max(r1[0], r2[0]), min(r1[2], r2[2])):
                    continue

                if col_divider_x is not None:
                    if min(r1[0], r2[0]) < col_divider_x - 3.0 and max(r1[2], r2[2]) > col_divider_x + 3.0:
                        continue

                if v_overlap >= 0.50 and x_dist <= max(40.0, 2.5 * min_h):
                    new_x1 = max(0.0, min(r1[0], r2[0]))
                    new_x2 = min(float(img_w), max(r1[2], r2[2]))
                    new_pts = np.array([
                        [new_x1, new_y1],
                        [new_x2, new_y1],
                        [new_x2, new_y2],
                        [new_x1, new_y2]
                    ], dtype=np.float32)
                    r1 = [new_x1, new_y1, new_x2, new_y2, new_pts]
                    skip.add(j)
                    merged = True
            new_rects.append(r1)
        rects = new_rects

    # Suppress detached micro-diacritic fragments (e.g., isolated upper/lower Cham vowel specks
    # ~10x9px duplicated inside or touching an adjacent full-height line on harum_palei_01)
    # while strictly preserving standalone short headers, stanza numbers, and page numbers.
    if len(rects) >= 2:
        post_median_h = float(np.median([r[3] - r[1] for r in rects]))
        if post_median_h >= 14.0:
            filtered_rects = []
            for idx_r, r_cand in enumerate(rects):
                bw = r_cand[2] - r_cand[0]
                bh = r_cand[3] - r_cand[1]
                if bh < 0.38 * post_median_h and bw < 0.55 * post_median_h:
                    is_diacritic_speck = False
                    for idx_m, r_main in enumerate(rects):
                        if idx_m == idx_r:
                            continue
                        mw = r_main[2] - r_main[0]
                        mh = r_main[3] - r_main[1]
                        if mh < 0.65 * post_median_h or mw < 2.0 * bw:
                            continue
                        h_inter = max(0.0, min(r_cand[2], r_main[2]) - max(r_cand[0], r_main[0]))
                        v_gap = max(0.0, max(r_cand[1] - r_main[3], r_main[1] - r_cand[3]))
                        if h_inter >= 0.70 * max(1.0, bw) and v_gap <= 0.25 * post_median_h:
                            is_diacritic_speck = True
                            break
                    if is_diacritic_speck:
                        continue
                filtered_rects.append(r_cand)
            rects = filtered_rects

    # Line sorting: sort by vertical center
    rects = sorted(rects, key=lambda r: (r[1] + r[3]) / 2.0)
    return [r[4] for r in rects]

# ==============================================================================
# Curved-line (polygon) support for Cham-DBNet box_type='poly' (CHAM_DET_BOX_TYPE=poly)
#   ribbon = per-column upper / lower boundary of a text-line polygon; it replaces the AABB height
#   (which on a wavy line spans several neighbouring lines) in merging, gating and rectification.
# ==============================================================================
def _odd(n, lo=3):
    n = int(max(lo, round(float(n))))
    return n if n % 2 == 1 else n + 1

def _smooth1d(a, k, mode='mean'):
    """Edge-padded moving mean / median of a 1-D signal (window k, forced odd and <= len)."""
    a = np.asarray(a, dtype=np.float32)
    n = len(a)
    if n < 3:
        return a.copy()
    k = min(int(k), n if n % 2 == 1 else n - 1)
    if k <= 1:
        return a.copy()
    k = k if k % 2 == 1 else k - 1
    r = k // 2
    pad = np.pad(a, r, mode='edge')
    if mode == 'median':
        return np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1).astype(np.float32)
    c = np.cumsum(np.insert(pad.astype(np.float64), 0, 0.0))
    return ((c[k:] - c[:-k]) / k).astype(np.float32)

def poly_ribbon(poly):
    """Per-column boundary of a (possibly curved, variable-length) text polygon.

    Returns (x0, top, bottom): float32 arrays where top[i] / bottom[i] are the polygon's upper edge and
    lower edge (exclusive) at page column x0 + i. Columns without polygon pixels are interpolated.
    """
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x0, y0 = int(np.floor(p[:, 0].min())), int(np.floor(p[:, 1].min()))
    x1, y1 = int(np.ceil(p[:, 0].max())), int(np.ceil(p[:, 1].max()))
    w, h = max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(p - np.float32([x0, y0])).astype(np.int32)], 1)
    has = m.any(axis=0)
    if not has.any():
        return x0, np.full(w, float(y0), np.float32), np.full(w, float(y1 + 1), np.float32)
    top = np.argmax(m, axis=0).astype(np.float32)
    bot = (h - np.argmax(m[::-1], axis=0)).astype(np.float32)
    if not has.all():
        xs = np.arange(w, dtype=np.float32)
        top = np.interp(xs, xs[has], top[has]).astype(np.float32)
        bot = np.interp(xs, xs[has], bot[has]).astype(np.float32)
    return x0, top + y0, bot + y0

def poly_thickness(poly):
    """Median perpendicular-ish thickness (ribbon height) of a text polygon (not its AABB height)."""
    _, top, bot = poly_ribbon(poly)
    return float(np.median(bot - top)) if len(top) else 0.0

def _line_height(box):
    """Line height used by the page gates: AABB height for 4-point quads (legacy behaviour, unchanged),
    ribbon thickness for curved polygons (a wavy line's AABB spans several neighbouring lines)."""
    pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    if len(pts) > 4:
        return poly_thickness(pts)
    return float(np.ptp(np.array(box)[:, 1]))

def _ribbon_info(pts):
    x0, top, bot = poly_ribbon(pts)
    c = (top + bot) / 2.0
    t = float(np.median(bot - top)) if len(top) else 1.0
    return {'pts': np.asarray(pts, dtype=np.float32).reshape(-1, 2), 'x0': x0, 'x1': x0 + len(top) - 1,
            'top': top, 'bot': bot, 'c': c, 't': max(1.0, t), 'cy': float(np.median(c)),
            'y1': float(np.min(pts[:, 1])), 'y2': float(np.max(pts[:, 1]))}

def _reading_rect(box):
    """[x1, y1, x2, y2] used for reading order: AABB for quads, a centerline band for polygons
    (AABBs of consecutive wavy lines overlap heavily and would scramble top-to-bottom order)."""
    pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    if len(pts) <= 4:
        return _aabb(box)
    r = _ribbon_info(pts)
    return float(r['x0']), r['cy'] - r['t'] / 2.0, float(r['x1'] + 1), r['cy'] + r['t'] / 2.0

def _ribbon_center_at(r, x, span):
    """Centerline y of ribbon r at page column x (linear extrapolation from the nearest `span` columns)."""
    n = len(r['c'])
    if r['x0'] < x < r['x1']:
        return float(r['c'][int(round(x)) - r['x0']])
    if x <= r['x0']:
        seg = r['c'][:min(n, span)]
        xs = r['x0'] + np.arange(len(seg), dtype=np.float64)
    else:
        seg = r['c'][max(0, n - span):]
        xs = r['x0'] + np.arange(n - len(seg), n, dtype=np.float64)
    if len(seg) < 3:
        return float(np.mean(seg))
    slope = float(np.polyfit(xs, seg.astype(np.float64), 1)[0])
    slope = float(np.clip(slope, -0.35, 0.35))
    return float(np.mean(seg)) + slope * (x - float(np.mean(xs)))

def _ribbons_same_line(a, b):
    """True when two ribbons are fragments of ONE text line (shared centerline), False for stacked lines."""
    t = min(a['t'], b['t'])
    ox1, ox2 = max(a['x0'], b['x0']), min(a['x1'], b['x1'])
    if ox2 >= ox1:
        ca = a['c'][ox1 - a['x0']: ox2 - a['x0'] + 1]
        cb = b['c'][ox1 - b['x0']: ox2 - b['x0'] + 1]
        return float(np.median(np.abs(ca - cb))) <= 0.5 * t
    gap = ox1 - ox2
    if gap > max(40.0, 2.5 * t):
        return False
    left, right = (a, b) if a['x1'] < b['x0'] else (b, a)
    xm = (left['x1'] + right['x0']) / 2.0
    span = int(max(6, 2 * t))
    return abs(_ribbon_center_at(left, xm, span) - _ribbon_center_at(right, xm, span)) <= 0.5 * t

def _merge_ribbon_polys(a, b):
    """Union of two fragments of one curved line as a single polygon that keeps the curvature
    (mask union + a bridge between the facing ends, then the outer contour) instead of an AABB."""
    allp = np.vstack([a['pts'], b['pts']])
    x0, y0 = int(np.floor(allp[:, 0].min())) - 2, int(np.floor(allp[:, 1].min())) - 2
    x1, y1 = int(np.ceil(allp[:, 0].max())) + 2, int(np.ceil(allp[:, 1].max())) + 2
    m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    off = np.float32([x0, y0])
    for r in (a, b):
        cv2.fillPoly(m, [np.round(r['pts'] - off).astype(np.int32)], 1)
    left, right = (a, b) if (a['x0'] + a['x1']) <= (b['x0'] + b['x1']) else (b, a)
    nl, nr = len(left['c']), len(right['c'])
    k = int(max(1, min(0.5 * min(left['t'], right['t']), nl // 2, nr // 2)))
    xl, xr = left['x1'] - k, right['x0'] + k
    if xr > xl:
        il, ir = nl - 1 - k, k
        bridge = np.float32([[xl, left['top'][il]], [xr, right['top'][ir]],
                             [xr, right['bot'][ir]], [xl, left['bot'][il]]])
        cv2.fillPoly(m, [np.round(bridge - off).astype(np.int32)], 1)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 1:
        kv = _odd(0.5 * min(a['t'], b['t']))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kv, kv)))
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 1:
        cnt = cv2.convexHull(np.vstack(contours))
    else:
        cnt = contours[0]
    cnt = cv2.approxPolyDP(cnt, 1.0, True).reshape(-1, 2).astype(np.float32) + off
    return _ribbon_info(cnt)

def merge_line_polygons(polys, img_w, img_h, col_divider_x=None):
    """Polygon counterpart of merge_line_boxes() for Cham-DBNet box_type='poly'.

    Fragments of one curved line (shared, extrapolated centerline) are merged into one curved polygon;
    vertically stacked wavy lines are never merged even though their AABBs overlap (the decision uses
    ribbon centerlines / thickness, not AABB heights), and fragments on opposite sides of a vertical
    two-column divider (`col_divider_x`) are never merged. Detached micro-diacritic specks next to a full line
    are suppressed as in merge_line_boxes. Returns polygons sorted by centerline y.
    """
    items = []
    for b in (polys if polys is not None else []):
        p = np.array(b, dtype=np.float32).reshape(-1, 2)
        if len(p) < 3 or np.ptp(p[:, 0]) <= 3 or np.ptp(p[:, 1]) <= 3:
            continue
        p[:, 0] = np.clip(p[:, 0], 0, max(0, img_w - 1))
        p[:, 1] = np.clip(p[:, 1], 0, max(0, img_h - 1))
        items.append(_ribbon_info(p))
    if not items:
        return []
    items.sort(key=lambda r: (r['cy'], r['x0']))

    merged = True
    while merged:
        merged = False
        out, used = [], [False] * len(items)
        for i in range(len(items)):
            if used[i]:
                continue
            r1 = items[i]
            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                if col_divider_x is not None:
                    if min(r1['x0'], items[j]['x0']) < col_divider_x - 3.0 and max(r1['x1'], items[j]['x1']) > col_divider_x + 3.0:
                        continue
                if _ribbons_same_line(r1, items[j]):
                    r1 = _merge_ribbon_polys(r1, items[j])
                    used[j] = True
                    merged = True
            out.append(r1)
        items = out

    if len(items) >= 2:
        med_t = float(np.median([r['t'] for r in items]))
        if med_t >= 14.0:
            kept = []
            for r in items:
                bw, bh = r['x1'] - r['x0'], r['y2'] - r['y1']
                speck = False
                if bh < 0.38 * med_t and bw < 0.55 * med_t:
                    for mr in items:
                        if mr is r or mr['t'] < 0.65 * med_t or (mr['x1'] - mr['x0']) < 2.0 * bw:
                            continue
                        ox1, ox2 = max(r['x0'], mr['x0']), min(r['x1'], mr['x1'])
                        if ox2 - ox1 < 0.70 * max(1.0, bw):
                            continue
                        band_t = float(mr['top'][ox1 - mr['x0']: ox2 - mr['x0'] + 1].min())
                        band_b = float(mr['bot'][ox1 - mr['x0']: ox2 - mr['x0'] + 1].max())
                        if max(0.0, r['y1'] - band_b, band_t - r['y2']) <= 0.25 * med_t:
                            speck = True
                            break
                if not speck:
                    kept.append(r)
            items = kept
    items.sort(key=lambda r: r['cy'])
    return [r['pts'] for r in items]

def rectify_polygon_line(img, poly, others=(), median_h=None, vpad=None, mask_outside=None, inverted=False):
    """Straightens a curved text-line polygon into a fixed-height horizontal strip ("ribbon" crop).

    1. The polygon is expressed in a frame rotated by its principal angle (only when |angle| >= 3 deg).
    2. Per column, the ribbon's upper / lower boundary gives a centerline, smoothed (median then mean,
       windows ~0.6x / 1.0x the line thickness) so glyph shapes are not warped locally.
    3. cv2.remap samples a strip centred on that centerline: height = polygon extent above / below the
       centerline (98th pct) + `vpad` x median page line thickness on each side (ascender / descender
       protection on top of the DBNet unclip margin).
    4. With mask_outside, pixels outside the own polygon (dilated by the padding) and inside neighbouring
       line polygons are replaced by the paper colour; ink components that belong mostly to the own line
       are kept whole, components belonging mostly to a neighbour line are removed (difference_update).
    Returns (strip_bgr, info).
    """
    vpad = CHAM_DET_POLY_VPAD if vpad is None else float(vpad)
    mask_outside = CHAM_DET_POLY_MASK if mask_outside is None else bool(mask_outside)
    H, W = img.shape[:2]
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    mom = cv2.moments(p.reshape(-1, 1, 2))
    if mom['m00'] > 1.0:
        c = np.float32([mom['m10'] / mom['m00'], mom['m01'] / mom['m00']])
        theta = 0.5 * float(np.arctan2(2.0 * mom['mu11'], mom['mu20'] - mom['mu02']))
    else:
        c, theta = p.mean(axis=0), 0.0
    if abs(np.degrees(theta)) < 3.0 or abs(np.degrees(theta)) > 45.0:
        theta = 0.0
    cs, sn = float(np.cos(theta)), float(np.sin(theta))
    d = p - c
    q = np.stack([d[:, 0] * cs + d[:, 1] * sn, -d[:, 0] * sn + d[:, 1] * cs], axis=1)
    qo = np.float32([np.floor(q[:, 0].min()), np.floor(q[:, 1].min())])
    rx0, top, bot = poly_ribbon(q - qo)
    top, bot = top + qo[1], bot + qo[1]
    n = len(top)
    t_med = max(2.0, float(np.median(bot - top)))
    mh = float(median_h) if median_h else t_med
    cen = _smooth1d((top + bot) / 2.0, _odd(0.6 * t_med), 'median')
    cen = _smooth1d(cen, _odd(1.0 * t_med), 'mean')
    up = max(1.0, float(np.percentile(cen - top, 98)))
    dn = max(1.0, float(np.percentile(bot - cen, 98)))
    pad = max(1.0, vpad * mh)
    hpad = int(max(2, round(0.08 * t_med)))
    up_h = up + pad
    Hs = int(max(8, round(up + dn + 2.0 * pad)))
    Ws = n + 2 * hpad
    idx = np.clip(np.arange(Ws) - hpad, 0, n - 1)
    xr = np.broadcast_to((qo[0] + rx0 - hpad + np.arange(Ws, dtype=np.float32))[None, :], (Hs, Ws))
    yr = cen[idx][None, :] - up_h + np.arange(Hs, dtype=np.float32)[:, None]
    mapx = (xr * cs - yr * sn + c[0]).astype(np.float32)
    mapy = (xr * sn + yr * cs + c[1]).astype(np.float32)
    strip = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    info = {'angle_deg': round(float(np.degrees(theta)), 2), 'thickness': round(t_med, 2),
            'strip_h': Hs, 'pad_px': round(pad, 2), 'masked': False}
    if not mask_outside:
        return strip, info

    bx0, by0 = int(max(0, np.floor(mapx.min()) - 2)), int(max(0, np.floor(mapy.min()) - 2))
    bx1, by1 = int(min(W, np.ceil(mapx.max()) + 3)), int(min(H, np.ceil(mapy.max()) + 3))
    if bx1 - bx0 < 2 or by1 - by0 < 2:
        return strip, info
    off = np.float32([bx0, by0])
    own = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
    cv2.fillPoly(own, [np.round(p - off).astype(np.int32)], 1)
    oth = np.zeros_like(own)
    for o in others or []:
        op = np.asarray(o, dtype=np.float32).reshape(-1, 2)
        if (op[:, 0].max() < bx0 or op[:, 0].min() > bx1 or op[:, 1].max() < by0 or op[:, 1].min() > by1):
            continue
        cv2.fillPoly(oth, [np.round(op - off).astype(np.int32)], 1)
    oth[own > 0] = 0  # the own polygon always wins (never erase own strokes)
    ry = int(max(1, round(pad + 0.10 * t_med)))
    rxk = int(max(1, round(0.10 * t_med)))
    keep = cv2.dilate(own, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rxk + 1, 2 * ry + 1)))
    keep[oth > 0] = 0
    roi = img[by0:by1, bx0:bx1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    if inverted:
        gray = 255 - gray
    kb = _odd(float(np.clip(0.6 * t_med, 7, 41)))
    bgk = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kb, kb)))
    ink = (cv2.subtract(bgk, gray) > 40).astype(np.uint8)
    n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n_cc > 1:
        a_own = np.bincount(lab[own > 0], minlength=n_cc)
        a_oth = np.bincount(lab[oth > 0], minlength=n_cc)
        glyph_sized = stats[:, cv2.CC_STAT_HEIGHT] <= 1.6 * t_med
        add = (a_own > 0) & (a_own >= a_oth) & glyph_sized
        rem = a_oth > a_own
        add[0] = rem[0] = False
        keep[(add.astype(np.uint8)[lab] > 0)] = 1
        keep[(rem.astype(np.uint8)[lab] > 0) & (own == 0)] = 0
    keep_f = cv2.GaussianBlur(keep.astype(np.float32), (3, 3), 0)
    km = cv2.remap(keep_f, mapx - off[0], mapy - off[1], cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sel = km > 0.5
    flat = strip[sel] if sel.sum() >= 16 else strip.reshape(-1, strip.shape[2]) if strip.ndim == 3 else strip.reshape(-1)
    bgc = np.median(flat, axis=0).astype(np.float32)
    kmc = km[..., None] if strip.ndim == 3 else km
    strip = np.clip(strip.astype(np.float32) * kmc + bgc * (1.0 - kmc), 0, 255).astype(np.uint8)
    info['masked'] = True
    return strip, info

# ==============================================================================
# Page-level quality gates & reading order (stress test 2026-11-04 fixes)
# ==============================================================================
def _file_fingerprint(path):
    """Short SHA-256 fingerprint of a model weight file (cached) for response traceability."""
    if not path or not os.path.exists(path):
        return None
    if path not in _fingerprint_cache:
        import hashlib
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        _fingerprint_cache[path] = h.hexdigest()[:16]
    return _fingerprint_cache[path]

def get_model_manifest(version):
    """Loads and caches model_manifest.json for the given recognizer version (returns {} if absent)."""
    if not version or version in ('auto', 'en', 'fr', 'japan'):
        return {}
    p = ocr_model_paths.get(version)
    model_dir = p["model_dir"] if p else os.path.join(PROJECT_ROOT, "data", "output", f"rec_cham_inference_{version}")
    manifest_path = os.path.join(model_dir, "model_manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    mtime = os.path.getmtime(manifest_path)
    cache_key = (manifest_path, mtime)
    if cache_key not in _model_manifest_cache:
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                _model_manifest_cache[cache_key] = json.load(f)
        except Exception:
            _model_manifest_cache[cache_key] = {}
    return _model_manifest_cache[cache_key]

def resolve_actual_model_version(model_ver):
    """Resolves canonical recognizer identifier from loaded on-disk paths / manifest (never 'auto')."""
    if not model_ver or model_ver == 'auto':
        return None
    p = ocr_model_paths.get(model_ver)
    if p and p.get("model_dir"):
        dir_base = os.path.basename(os.path.normpath(p["model_dir"]))
        inferred_ver = (
            "vi" if dir_base == "rec_vietnamese_infer"
            else (dir_base[len("rec_cham_inference_"):] if dir_base.startswith("rec_cham_inference_") else model_ver)
        )
        manifest = get_model_manifest(inferred_ver)
        return str(manifest.get("model_id") or inferred_ver)
    manifest = get_model_manifest(model_ver)
    if manifest.get("model_id"):
        return str(manifest["model_id"])
    return str(model_ver)


def invoke_recognizer_with_provenance(ocr_model, img_list, model_ver=None):
    """Invokes a recognizer/router and returns (results, selected_versions, elapsed) without post-OCR script guessing."""
    if not img_list:
        return [], [], 0.0
    if hasattr(ocr_model, "predict_with_provenance"):
        res, sel_vers, elapsed = ocr_model.predict_with_provenance(img_list)
        return res, list(sel_vers), elapsed
    if hasattr(ocr_model, "last_selected_versions"):
        ocr_model.last_selected_versions = []
    res, elapsed = ocr_model(img_list)
    attr_vers = list(getattr(ocr_model, "last_selected_versions", None) or [])
    if len(attr_vers) == len(res):
        return res, attr_vers, elapsed
    single_ver = resolve_actual_model_version(model_ver)
    return res, [single_ver] * len(res), elapsed


def calibrate_line_confidence(raw_conf, model_ver=None, text="", temperature=None):
    """Post-hoc logit temperature scaling for recognizer confidence (identity when T=1.0)."""
    val = float(raw_conf)
    if val <= 0.0:
        return 0.0
    if val >= 1.0:
        return 1.0
    if temperature is None:
        target_ver = None if model_ver in (None, 'auto') else model_ver
        manifest = get_model_manifest(target_ver)
        try:
            T = float(manifest.get("confidence_temperature", 1.0))
        except (TypeError, ValueError):
            T = 1.0
    else:
        T = float(temperature)
    if T <= 0.0 or abs(T - 1.0) < 1e-6:
        return round(val, 6)
    p = min(max(val, 1e-6), 1.0 - 1e-6)
    z = math.log(p / (1.0 - p))
    return round(1.0 / (1.0 + math.exp(-z / T)), 6)

def get_model_info(model_ver, executed_detector=None, detector_switched=False):
    """Returns provenance info (model id, weight fingerprints, executed detector, normalizer) for the response."""
    info = {"model": model_ver, "normalizer": None}
    rec_versions = [DEFAULT_CHAM_MODEL, 'v24_1_r3_1', 'vi'] if model_ver == 'auto' else [model_ver]
    recs = []
    for v in rec_versions:
        p = ocr_model_paths.get(v)
        if p:
            dir_base = os.path.basename(os.path.normpath(p["model_dir"]))
            inferred_ver = (
                "vi" if dir_base == "rec_vietnamese_infer"
                else (dir_base[len("rec_cham_inference_"):] if dir_base.startswith("rec_cham_inference_") else v)
            )
            manifest = get_model_manifest(inferred_ver)
            actual_ver = str(manifest.get("model_id") or inferred_ver)
            try:
                conf_temp = float(manifest.get("confidence_temperature", 1.0))
            except (TypeError, ValueError):
                conf_temp = 1.0
            rec_shape_info = manifest.get("rec_image_shape") or (
                "3, 48, 480" if (actual_ver == "v25" or str(actual_ver).startswith("v24_1_r4")) else "3, 48, 320"
            )
            recs.append({"version": actual_ver,
                         "model_dir": os.path.relpath(p["model_dir"], PROJECT_ROOT).replace("\\", "/"),
                         "weights_sha256_16": _file_fingerprint(os.path.join(p["model_dir"], "inference.pdiparams")),
                         "dict": os.path.basename(p["dict_path"]),
                         "dict_sha256_16": _file_fingerprint(p["dict_path"]),
                         "rec_image_shape": rec_shape_info,
                         "confidence_temperature": conf_temp})
        elif v in ocr_models and isinstance(ocr_models.get(v), OfficialPaddleOCRWrapper):
            recs.append({"version": v, "model_dir": None, "wrapper": "OfficialPaddleOCRWrapper"})
    info["recognizers"] = recs
    if not det_model_info and not executed_detector:
        info["detector"] = {"name": None, "mode": DET_MODE, "loaded": False}
    else:
        active_det_name = executed_detector or det_model_info.get("name")
        active_det_dir = CHAM_DET_DIR if active_det_name == "cham_dbnet" else det_model_info.get("dir", "")
        cham_sha16 = _file_fingerprint(os.path.join(CHAM_DET_DIR, "inference.pdiparams")) if (cham_det_model is not None or active_det_name == "cham_dbnet") else None
        info["detector"] = {"name": active_det_name, "mode": det_model_info.get("mode", DET_MODE), "loaded": True,
                            "detector_switched": bool(detector_switched),
                            "weights_sha256_16": _file_fingerprint(os.path.join(active_det_dir, "inference.pdiparams")) if active_det_dir else None,
                            # Exposed so a deployment can be verified: False means the fine-tuned Cham-DBNet
                            # failed to load (e.g. PIR model on an old Paddle) and only the legacy detector runs.
                            "cham_dbnet_available": cham_det_model is not None,
                            "cham_dbnet_sha256_16": cham_sha16,
                            "cham_dbnet_box_type": ("poly" if _is_poly_detector(cham_det_model) else "quad") if cham_det_model is not None else None}
    try:
        import paddle as _paddle
        info["paddle_version"] = _paddle.__version__
    except Exception:
        info["paddle_version"] = None
    if model_ver in ['auto', 'v24', 'v25', 'v26'] or str(model_ver).startswith('v24'):
        info["normalizer"] = NORMALIZER_VERSION
    return info

def analyze_text_ink(img):
    """Estimates text-like ink components against a locally estimated paper background.

    Background is estimated with a grey-level morphological closing (removes thin dark strokes),
    so paper texture, vignetting and soft shadows are not counted as ink.
    Returns a dict with component stats/centroids, total ink pixels and ink ratio.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    if check_inverted_polarity(img):
        gray = 255 - gray
    h, w = gray.shape[:2]
    k = int(max(15, min(41, round(min(h, w) / 40)))) | 1
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    diff = cv2.subtract(bg, gray)
    mask = (diff > 45).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = []
    for i in range(1, n):
        cw, ch, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if area >= 10 and ch >= 3 and cw >= 2 and ch < 0.5 * h and cw < 0.9 * w:
            keep.append(i)
    stats_k = stats[keep] if keep else np.zeros((0, 5), dtype=np.int32)
    cents_k = cents[keep] if keep else np.zeros((0, 2), dtype=np.float64)
    ink_px = int(stats_k[:, cv2.CC_STAT_AREA].sum()) if len(keep) else 0
    return {"stats": stats_k, "centroids": cents_k, "ink_px": ink_px,
            "ink_ratio": ink_px / float(max(1, h * w)), "num_components": len(keep), "shape": (h, w)}

def is_blank_page(ink):
    """True when the page has no text-like ink (blank / near-blank paper)."""
    return ink["num_components"] < 4 or ink["ink_ratio"] < 0.0003

def compute_uncovered_ink(ink, polygons, median_line_h=None):
    """Fraction of text-like ink (area-weighted) whose component centroid lies outside every detected line.

    Returns (ratio, uncovered_component_indices).
    """
    h, w = ink["shape"]
    stats, cents = ink["stats"], ink["centroids"]
    if len(stats) == 0:
        return 0.0, []
    cover = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(cover, [np.round(np.array(poly, dtype=np.float32)).astype(np.int32)], 1)
    # Anisotropic dilation: tight vertical pad (0.08 * median_line_h) prevents two neighbouring lines
    # from bridging vertically across a missed short line (e.g. synthetic_142 GT[1]), while preserving horizontal word margin
    pad_y = int(max(2, round(0.08 * median_line_h))) if median_line_h else 3
    pad_x = int(max(3, round(0.25 * median_line_h))) if median_line_h else 5
    cover = cv2.dilate(cover, np.ones((2 * pad_y + 1, 2 * pad_x + 1), np.uint8))
    total, uncovered, idxs = 0, 0, []
    for i in range(len(stats)):
        ch, area = stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if median_line_h and ch < 0.12 * median_line_h:
            continue  # dust / speckle, not a glyph
        total += area
        cx = int(min(w - 1, max(0, round(cents[i][0]))))
        cy = int(min(h - 1, max(0, round(cents[i][1]))))
        if cover[cy, cx] == 0:
            uncovered += area
            idxs.append(i)
    return (uncovered / float(total) if total else 0.0), idxs

def _split_columns(idx, rects, median_h, img_w, col_divider_x=None):
    """Splits a group of non-spanning line boxes into side-by-side columns using vertical gutters or divider x."""
    ivs = sorted(idx, key=lambda i: rects[i][0])
    runs = []  # [x_start, x_end, members]
    for i in ivs:
        x1, _, x2, _ = rects[i]
        crosses_div = (
            col_divider_x is not None
            and runs
            and runs[-1][1] <= col_divider_x + 6
            and x1 >= col_divider_x - 6
        )
        if runs and x1 <= runs[-1][1] and not crosses_div:
            runs[-1][1] = max(runs[-1][1], x2)
            runs[-1][2].append(i)
        else:
            runs.append([x1, x2, [i]])
    if len(runs) < 2:
        return [idx]
    min_gutter = max(1.5 * median_h, 0.025 * img_w)
    cols = [runs[0]]
    for r in runs[1:]:
        div_between = (
            col_divider_x is not None
            and cols[-1][1] <= col_divider_x + 8
            and r[0] >= col_divider_x - 8
        )
        if div_between or (r[0] - cols[-1][1] >= min_gutter):
            cols.append([r[0], r[1], list(r[2])])
        else:
            cols[-1][1] = max(cols[-1][1], r[1])
            cols[-1][2].extend(r[2])
    if len(cols) < 2 or any(len(c[2]) < 3 for c in cols):
        return [idx]
    # Columns must sit side by side (overlapping vertical extents), not be stacked blocks
    for a, b in zip(cols, cols[1:]):
        ay1, ay2 = min(rects[i][1] for i in a[2]), max(rects[i][3] for i in a[2])
        by1, by2 = min(rects[i][1] for i in b[2]), max(rects[i][3] for i in b[2])
        inter = max(0.0, min(ay2, by2) - max(ay1, by1))
        if inter < 0.5 * min(ay2 - ay1, by2 - by1):
            return [idx]
    return [c[2] for c in cols]

def order_lines_reading(rects, img_w, col_divider_x=None):
    """Column-aware reading order for line boxes [x1, y1, x2, y2].

    Full-width (spanning) lines AND vertically isolated top/bottom centered headers/footers split the page
    into sections; inside each section, side-by-side columns separated by a vertical gutter or `col_divider_x`
    are read top-to-bottom, left column first. Single-column pages keep the legacy top-to-bottom order.
    Returns (order, layout) where layout = {"reading_order": "rows"|"columns", "columns": k}.
    """
    n = len(rects)
    yc = lambda i: (rects[i][1] + rects[i][3]) / 2.0
    order_y = sorted(range(n), key=yc)
    layout = {"reading_order": "rows", "columns": 1}
    if n < 6:
        return order_y, layout
    median_h = float(np.median([r[3] - r[1] for r in rects]))
    x0 = min(r[0] for r in rects)
    x1 = max(r[2] for r in rects)
    text_w = max(1.0, x1 - x0)
    spanning = {i for i in range(n) if (rects[i][2] - rects[i][0]) > 0.6 * text_w}
    sections, cur = [], []
    for i in order_y:
        if i in spanning:
            if cur:
                sections.append(cur)
                cur = []
            sections.append([i])
        else:
            cur.append(i)
    if cur:
        sections.append(cur)
    order = []
    for sec in sections:
        cols = _split_columns(sec, rects, median_h, img_w, col_divider_x=col_divider_x) if len(sec) >= 6 else [sec]
        lead_hdrs, tail_ftrs = [], []
        if len(cols) == 1 and len(sec) >= 7:
            # Test minimum-peel combinations (1 top, 1 bottom, 1+1, 2 top, 2 bottom, 2+1, 1+2) of vertically
            # isolated header/footer lines (e.g. short centered title '* ... *' and/or centered page number '- 30 -')
            # ONLY if every peeled line has no vertical overlap with the remaining body AND the body splits cleanly.
            def _isolated_band(peeled, body):
                return all(
                    (min(rects[p][3], rects[b][3]) - max(rects[p][1], rects[b][1])) <= 0.15 * median_h
                    for p in peeled for b in body
                )

            for n_top, n_bot in ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)):
                if len(sec) - n_top - n_bot < 6:
                    continue
                cand_top = sec[:n_top]
                cand_bot = sec[len(sec) - n_bot:] if n_bot > 0 else []
                rem_body = sec[n_top:len(sec) - n_bot if n_bot > 0 else len(sec)]
                if _isolated_band(cand_top + cand_bot, rem_body):
                    test_cols = _split_columns(rem_body, rects, median_h, img_w, col_divider_x=col_divider_x)
                    if len(test_cols) > 1:
                        cols = test_cols
                        lead_hdrs = list(cand_top)
                        tail_ftrs = list(cand_bot)
                        break
        if len(cols) > 1:
            layout = {"reading_order": "columns", "columns": max(layout["columns"], len(cols))}
            order.extend(lead_hdrs)
            for col in cols:
                order.extend(sorted(col, key=yc))
            order.extend(tail_ftrs)
        else:
            order.extend(sorted(sec, key=yc))
    return order, layout

def apply_line_quality_gates(lines_metadata, final_crops, dropped=None):
    """Drops empty / background-noise lines and flags uncertain lines for human review (DBNet path).

    dropped: optional list; receives the metadata of dropped lines that look like real text lines
             (Cham text in a box at least ~half the page's typical line size). The caller must
             surface these at document level instead of silently losing them (benchmark
             test_393_a4_lvl5: 4/5 lines dropped while the page still reported 'ocr_done').
    """
    # Line height: AABB height, or the ribbon thickness for rectified curved polygons (whose AABB
    # spans neighbouring lines and would wrongly look like 'possible_merged_lines').
    _mh = lambda m: float(m['line_thickness']) if m.get('line_thickness') else (m['bbox'][3] - m['bbox'][1])
    all_w = [m['bbox'][2] - m['bbox'][0] for m in lines_metadata]
    all_h = [_mh(m) for m in lines_metadata]
    med_w = float(np.median(all_w)) if all_w else 0.0
    med_h = float(np.median(all_h)) if all_h else 0.0
    kept_meta, kept_crops = [], []
    for meta, crop in zip(lines_metadata, final_crops):
        cand = meta['candidates'][meta['selected_crop_type']]
        pred = (cand.get('prediction') or '').strip()
        conf = float(cand.get('confidence') or 0.0)
        has_cham = any('\uAA00' <= ch <= '\uAA5F' or '\uA800' <= ch <= '\uA87F' for ch in pred)
        has_alnum = any(ch.isalnum() for ch in pred)
        if not pred or conf < 0.15 or (conf < 0.35 and not has_cham and not has_alnum):
            bw, bh = meta['bbox'][2] - meta['bbox'][0], _mh(meta)
            if dropped is not None and has_cham and bw >= 0.5 * med_w and bh >= 0.5 * med_h:
                dropped.append(meta)
            continue
        kept_meta.append(meta)
        kept_crops.append(crop)

    heights = [_mh(m) for m in kept_meta]
    median_h = float(np.median(heights)) if heights else 0.0
    for meta in kept_meta:
        cand = meta['candidates'][meta['selected_crop_type']]
        conf = float(cand.get('confidence') or 0.0)
        reasons = list(meta.get('reason') or [])
        if conf < LINE_REVIEW_CONF:
            reasons.append('low_confidence')
        if len(kept_meta) >= 3 and median_h > 0 and _mh(meta) > 1.8 * median_h:
            reasons.append('possible_merged_lines')
        meta['reason'] = reasons
        meta['needs_review'] = len(reasons) > 0
        meta['crop_confidence'] = 'high' if conf > 0.8 else ('medium' if conf > 0.5 else 'low')
    return kept_meta, kept_crops

def build_document_status(results, reasons, extra=None):
    """Document-level OCR status. 'verified' is never set by the server: only a human can verify."""
    reasons = list(dict.fromkeys(reasons))
    if not results:
        status = 'no_text' if (not reasons or reasons == ['blank_page']) else 'needs_review'
    elif reasons or any(r['segmentation'].get('needs_review') for r in results):
        status = 'needs_review'
    else:
        status = 'ocr_done'
    doc = {"status": status, "needs_review": status == 'needs_review', "reasons": reasons, "line_count": len(results)}
    if extra:
        doc.update(extra)
    return doc

def check_inverted_polarity(img):
    """
    Morphological Opening Solid-Substrate vs Thin-Stroke Polarity Detector.
    Returns True iff >=95% of the morphologically opened background is dark (<85) AND bright foreground
    pixels (`p99.2`) vanish under opening (`diff > 45`), proving genuine light text strokes on a dark
    background while never falsely inverting archival photos on dark velvet tables or tan/brown palm-leaf/paper.
    """
    if img is None or img.size == 0:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return False
    if float(np.median(gray)) >= 130.0:
        return False

    roi_wide = gray[
        int(h * 0.04):max(int(h * 0.04) + 1, int(h * 0.96)),
        int(w * 0.04):max(int(w * 0.04) + 1, int(w * 0.96)),
    ]
    roi_mid = gray[int(h * 0.14):int(h * 0.68), int(w * 0.16):int(w * 0.84)] if (h >= 180 and w >= 240) else roi_wide

    def _eval_roi(roi_arr, allow_sparse=False):
        scale = min(1.0, 900.0 / float(max(roi_arr.shape[:2])))
        work = (
            cv2.resize(roi_arr, (max(8, int(roi_arr.shape[1] * scale)), max(8, int(roi_arr.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            if scale < 1.0 else roi_arr
        )
        wh, ww = work.shape[:2]
        k = int(max(7, min(21, round(min(wh, ww) / 8)))) | 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, se)

        op_p95 = float(np.percentile(opened, 95))
        fg_p992 = float(np.percentile(work, 99.2))
        if op_p95 < 85.0 and (fg_p992 - op_p95) > 45.0:
            return True
        if allow_sparse:
            # Sparse single-line white-on-black page (e.g. 0.1% white stroke pixels):
            # only fire when >=98.5% of opened background is genuinely dark (<65), so dark-framed folios
            # with a bright paper/palm-leaf interior (op_p985 >= 110) are never inverted.
            op_p985 = float(np.percentile(opened, 98.5))
            if op_p985 < 65.0:
                tophat = cv2.subtract(work, opened)
                if int((tophat > 60).sum()) >= 28 and float(np.percentile(tophat, 99.92)) > 55.0:
                    return True
        return False

    if _eval_roi(roi_mid, allow_sparse=False):
        return True
    return _eval_roi(roi_wide, allow_sparse=True)


def detect_vertical_column_divider(img):
    """Detects a continuous printed vertical column-divider rule in [0.38W, 0.62W] (e.g. 2-column dictionaries)."""
    if img is None or img.size == 0:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape[:2]
    if h < 300 or w < 300:
        return None
    y0, y1 = int(h * 0.14), int(h * 0.86)
    x0, x1 = int(w * 0.38), int(w * 0.62)
    band = gray[y0:y1, x0:x1]
    bh, bw = band.shape
    horiz_bg = cv2.GaussianBlur(band, (31, 1), 0)
    dark = (cv2.subtract(horiz_bg, band) > 22).astype(np.uint8)

    q_peaks = []
    for q in range(4):
        s0, s1 = (q * bh) // 4, ((q + 1) * bh) // 4
        sub = cv2.dilate(dark[s0:s1, :], np.ones((1, 3), np.uint8))
        dens = sub.mean(axis=0)
        pk = int(np.argmax(dens))
        val = float(dens[pk])
        lf = dens[max(0, pk - 22):max(0, pk - 5)]
        rf = dens[min(bw, pk + 6):min(bw, pk + 23)]
        if val >= 0.50 and len(lf) >= 5 and len(rf) >= 5 and float(lf.mean()) <= 0.12 and float(rf.mean()) <= 0.12:
            q_peaks.append((q, pk, val))

    if len(q_peaks) < 3:
        return None
    xs = [p[1] for p in q_peaks]
    if max(xs) - min(xs) > 16:
        return None
    return x0 + int(round(float(np.median(xs))))


def maybe_safe_deskew_page(img, min_abs_deg=9.0, min_var_gain=1.55, search_deg=16.0, step_deg=0.5):
    """
    Conditional Safe-Border Pre-Detection Deskew gated strictly at |theta| >= 9.0 deg AND projection
    variance gain >= 1.55x. Returns (deskewed_img, M_inv_2x3_or_None, applied_angle_deg).
    Guarantees 0% change on upright or mildly tilted pages (|theta| < 9 deg) and 0 border glyph clipping.
    """
    if img is None or img.size == 0:
        return img, None, 0.0
    h, w = img.shape[:2]
    if h < 140 or w < 180:
        return img, None, 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    if check_inverted_polarity(img):
        gray = 255 - gray
    roi = gray[int(h * 0.14):int(h * 0.84), int(w * 0.14):int(w * 0.86)]
    scale = min(1.0, 560.0 / float(max(roi.shape[:2])))
    small = cv2.resize(roi, (max(16, int(roi.shape[1] * scale)), max(16, int(roi.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    bh = cv2.morphologyEx(small, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    _, bw = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if int((bw > 0).sum()) < 80:
        return img, None, 0.0

    sh, sw = bw.shape
    cx, cy = sw / 2.0, sh / 2.0
    zero_score = max(1.0, float(np.var(bw.sum(axis=1).astype(np.float64))))
    best_angle = 0.0
    best_score = zero_score

    for ang in np.arange(-search_deg, search_deg + 1e-6, step_deg):
        if abs(float(ang)) < 1e-4:
            continue
        M_s = cv2.getRotationMatrix2D((cx, cy), float(ang), 1.0)
        rot = cv2.warpAffine(bw, M_s, (sw, sh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sc = float(np.var(rot.sum(axis=1).astype(np.float64)))
        if sc > best_score:
            best_score = sc
            best_angle = float(ang)

    if abs(best_angle) < min_abs_deg or (best_score / zero_score) < min_var_gain:
        return img, None, 0.0

    rad = math.radians(abs(best_angle))
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    new_w = int(math.ceil(w * cos_a + h * sin_a))
    new_h = int(math.ceil(h * cos_a + w * sin_a))
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), best_angle, 1.0)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    deskewed = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    M_inv = cv2.invertAffineTransform(M)
    return deskewed, M_inv, round(best_angle, 2)


def map_deskew_pts_to_original(pts, M_inv, orig_w, orig_h):
    """Maps 2D points from deskewed canvas space back to original image frame [0..orig_w-1, 0..orig_h-1]."""
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((len(arr), 1), dtype=np.float64)
    mapped = np.hstack([arr, ones]) @ np.asarray(M_inv, dtype=np.float64).T
    mapped[:, 0] = np.clip(mapped[:, 0], 0, max(0, orig_w - 1))
    mapped[:, 1] = np.clip(mapped[:, 1], 0, max(0, orig_h - 1))
    return mapped

def preprocess_crop_for_rec(crop_img):
    """
    Normalizes a crop for OCR recognition. If inverted (white text on dark background),
    binarizes via Otsu to ensure black text on clean white background for SVTR/CTC models.
    """
    if crop_img is None or crop_img.size == 0:
        return crop_img
    if check_inverted_polarity(crop_img):
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if len(crop_img.shape) == 3 else crop_img
        _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return cv2.cvtColor(bin_inv, cv2.COLOR_GRAY2BGR)
    return crop_img

def _aabb(poly):
    pts = np.array(poly, dtype=np.float32)
    return float(np.min(pts[:, 0])), float(np.min(pts[:, 1])), float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))

def _iom(a, b):
    """Intersection over the smaller box area for axis-aligned boxes (x1, y1, x2, y2)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    sa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    sb = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return (ix * iy) / min(sa, sb)

def expand_quad(pts, pad_v, pad_h, img_w, img_h):
    """Expands a (tl, tr, br, bl) quad along its own vertical / horizontal axes by a fraction of its height."""
    p = np.array(pts, dtype=np.float32).copy()
    if p.shape != (4, 2):
        return p
    v = ((p[3] + p[2]) - (p[0] + p[1])) / 2.0
    line_h = float(np.linalg.norm(v)) + 1e-6
    v /= line_h
    u = ((p[1] + p[2]) - (p[0] + p[3])) / 2.0
    u /= (float(np.linalg.norm(u)) + 1e-6)
    dv, dh = v * pad_v * line_h, u * pad_h * line_h
    p[0] += -dv - dh
    p[1] += -dv + dh
    p[2] += dv + dh
    p[3] += dv - dh
    p[:, 0] = np.clip(p[:, 0], 0, img_w - 1)
    p[:, 1] = np.clip(p[:, 1], 0, img_h - 1)
    return p

def max_stacked_overlap(boxes):
    """Max vertical overlap (fraction of the smaller AABB height) between two boxes that also
    overlap horizontally by more than 25% of the narrower one (i.e. stacked, not side by side).

    Curved polygons (> 4 points) are compared by their ribbons instead: median per-column overlap of
    the [top, bottom] bands over the shared columns / the thinner ribbon's thickness (the AABBs of two
    adjacent wavy lines overlap heavily even when the lines themselves are cleanly separated)."""
    r = [_aabb(b) for b in boxes]
    is_poly = [len(np.asarray(b).reshape(-1, 2)) > 4 for b in boxes]
    ribbons = {}

    def _rib(k):
        if k not in ribbons:
            ribbons[k] = _ribbon_info(np.asarray(boxes[k], dtype=np.float32).reshape(-1, 2))
        return ribbons[k]

    best = 0.0
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            a, b = r[i], r[j]
            hx = min(a[2], b[2]) - max(a[0], b[0])
            if hx <= 0.25 * min(a[2] - a[0], b[2] - b[0]):
                continue
            if is_poly[i] or is_poly[j]:
                ra, rb = _rib(i), _rib(j)
                ox1, ox2 = max(ra['x0'], rb['x0']), min(ra['x1'], rb['x1'])
                if ox2 < ox1:
                    continue
                sa, sb = slice(ox1 - ra['x0'], ox2 - ra['x0'] + 1), slice(ox1 - rb['x0'], ox2 - rb['x0'] + 1)
                inter = np.minimum(ra['bot'][sa], rb['bot'][sb]) - np.maximum(ra['top'][sa], rb['top'][sb])
                best = max(best, float(np.median(np.maximum(inter, 0.0))) / max(1.0, min(ra['t'], rb['t'])))
                continue
            vy = min(a[3], b[3]) - max(a[1], b[1])
            best = max(best, vy / max(1.0, min(a[3] - a[1], b[3] - b[1])))
    return best

def segment_lines_dbnet(img, det_model, ink=None, return_info=False, reading_order='auto'):
    """
    Performs deep-learning based text line segmentation using PaddleOCR DBNet.
    Args:
        ink: optional result of analyze_text_ink(img); enables the uncovered-ink check and
             (in DET_MODE='hybrid') the Legacy-First switch to Cham-DBNet for missed / merged lines.
        return_info: also return a dict with detector diagnostics and layout.
        reading_order: 'auto' (column-aware) or 'rows' (legacy top-to-bottom).
    Returns:
        final_crops: list of cropped BGR images corresponding to each detected line
        lines_metadata: list of metadata dicts matching the Studio response schema
        (info): only when return_info=True
    """
    import tools.infer.utility as utility

    info = {"raw_boxes": 0, "lines_detected": 0, "detector": None, "detector_switched": False,
            "uncovered_ink_ratio": None, "uncovered_ink_ratio_primary": None,
            "layout": {"reading_order": "rows", "columns": 1}}

    def _ret(crops, metas):
        return (crops, metas, info) if return_info else (crops, metas)

    is_inverted = check_inverted_polarity(img)
    det_img = (255 - img) if is_inverted else img
    h, w = img.shape[:2]
    col_divider_x = detect_vertical_column_divider(det_img) if reading_order != 'rows' else None
    if col_divider_x is not None:
        # Mask thin vertical 2-column rule on det_img so DBNet never bridges left & right columns across '|'
        det_img = det_img.copy()
        rx0, rx1 = max(0, col_divider_x - 3), min(w, col_divider_x + 4)
        bg_fill = int(np.median(det_img[int(h * 0.14):int(h * 0.86), max(0, rx0 - 12):min(w, rx1 + 12)]))
        det_img[int(h * 0.10):int(h * 0.90), rx0:rx1] = bg_fill
        info["column_divider_x"] = int(col_divider_x)

    def _detect(detector):
        boxes, _ = detector(det_img)
        if _is_poly_detector(detector):
            boxes = [] if boxes is None else [np.array(b, dtype=np.float32).reshape(-1, 2) for b in boxes]
            return merge_line_polygons(boxes, w, h, col_divider_x=col_divider_x)
        boxes = [] if boxes is None else [np.array(b, dtype=np.float32) for b in boxes]
        return merge_line_boxes(boxes, w, h, col_divider_x=col_divider_x)

    def _coverage(boxes):
        """(uncovered ink ratio, max box height / glyph height) for a candidate box set.
        Box height = AABB height for quads, ribbon thickness for curved polygons."""
        if not boxes:
            return 1.0, 0.0
        hs = [_line_height(b) for b in boxes]
        unc = compute_uncovered_ink(ink, boxes, float(np.median(hs)))[0] if ink is not None else 0.0
        return unc, (max(hs) / glyph_h if glyph_h > 0 else 0.0)

    glyph_h = 0.0
    if ink is not None and len(ink["stats"]):
        comp_h = ink["stats"][:, cv2.CC_STAT_HEIGHT]
        comp_h = comp_h[comp_h >= 6]
        glyph_h = float(np.median(comp_h)) if len(comp_h) else 0.0

    dt_boxes = _detect(det_model)
    info["raw_boxes"] = len(dt_boxes)
    info["detector"] = det_model_info.get("name", "primary")
    use_cham_crop_pad = det_model is cham_det_model
    use_poly = _is_poly_detector(det_model)

    # Legacy-First gate: keep the legacy detector result unless it leaves text-like ink uncovered
    # or produced a box tall enough to contain several merged lines; only then try Cham-DBNet and
    # switch the whole page to it if it clearly covers the text better.
    if ink is not None and DET_MODE == 'hybrid' and det_model is not cham_det_model:
        unc_p, tall_p = _coverage(dt_boxes)
        stacked_p = max_stacked_overlap(dt_boxes)
        info["uncovered_ink_ratio_primary"] = round(unc_p, 4)
        info["stacked_overlap_primary"] = round(stacked_p, 3)
        missed = unc_p >= LEGACY_GATE_UNCOVERED_RATIO
        stacked = stacked_p >= LEGACY_GATE_STACKED_OVERLAP
        if missed or stacked or tall_p >= 3.5:
            cham = get_cham_det_model()
            if cham is not None:
                cham_boxes = _detect(cham)
                unc_c, tall_c = _coverage(cham_boxes)
                better_cover = cham_boxes and unc_c + DET_SWITCH_MIN_GAIN <= unc_p
                splits_merged = (cham_boxes and tall_p >= 3.5 and tall_c < 3.5
                                 and len(cham_boxes) > len(dt_boxes) and unc_c <= unc_p + 0.01)
                # Legacy boxes visibly stacked / missing ink and Cham-DBNet separates more lines
                # without losing coverage (benchmark test_128 / test_148_lvl4: 5 -> 6 of 6 lines).
                more_lines = ((missed or stacked) and len(cham_boxes) > len(dt_boxes)
                              and unc_c <= unc_p + 0.01)
                if better_cover or splits_merged or more_lines:
                    print(f"🔀 Legacy-First gate: switching to Cham-DBNet ({len(dt_boxes)} -> {len(cham_boxes)} lines, "
                          f"uncovered {unc_p:.3f} -> {unc_c:.3f}, tallest box/glyph {tall_p:.1f} -> {tall_c:.1f})")
                    dt_boxes = cham_boxes
                    info["detector"] = "cham_dbnet"
                    info["detector_switched"] = True
                    use_cham_crop_pad = True
                    use_poly = _is_poly_detector(cham)

    info["lines_detected"] = len(dt_boxes)
    if len(dt_boxes) == 0:
        return _ret([], [])

    if ink is not None:
        info["uncovered_ink_ratio"] = round(_coverage(dt_boxes)[0], 4)

    # Column-aware reading order (fixes left/right interleaving on multi-column pages)
    if reading_order != 'rows' and len(dt_boxes) > 1:
        rects = [_reading_rect(b) for b in dt_boxes] if use_poly else [_aabb(b) for b in dt_boxes]
        order, layout = order_lines_reading(rects, w, col_divider_x=col_divider_x)
        dt_boxes = [dt_boxes[i] for i in order]
        info["layout"] = layout

    final_crops = []
    lines_metadata = []
    poly_median_h = float(np.median([_line_height(b) for b in dt_boxes])) if use_poly else None

    for idx, box in enumerate(dt_boxes):
        pts = np.array(box, dtype=np.float32)
        x_min = int(max(0, np.min(pts[:, 0])))
        x_max = int(min(w, np.max(pts[:, 0])))
        y_min = int(max(0, np.min(pts[:, 1])))
        y_max = int(min(h, np.max(pts[:, 1])))
        rect_info = None
        if use_poly:
            # Curved Cham-DBNet polygon: straighten along its centerline into a padded, neighbour-masked strip
            others = [b for k, b in enumerate(dt_boxes) if k != idx]
            crop_img, rect_info = rectify_polygon_line(img, pts, others, median_h=poly_median_h, inverted=is_inverted)
        else:
            # Get perspective crop (Cham-DBNet boxes are tighter than the legacy detector's: pad them
            # along the line's own vertical axis so stacked diacritics are not clipped)
            crop_pts = expand_quad(pts, CHAM_DET_CROP_PAD, 0.05, w, h) if use_cham_crop_pad and CHAM_DET_CROP_PAD > 0 else pts
            crop_img = utility.get_rotate_crop_image(img, crop_pts)
        final_crops.append(crop_img)

        meta = {
            'line_id': idx,
            'core_box': [y_min, y_max],
            'bbox': [x_min, y_min, x_max, y_max],
            'polygon': pts.tolist(),
            'overlap_prev_px': 0,
            'overlap_next_px': 0,
            'crop_confidence': 1.0,
            'selected_crop_type': 'dbnet',
            'crop_sanity_score': 1.0,
            'needs_review': False,
            'detector': info.get('detector'),
            'candidates': {
                'dbnet': {
                    'prediction': '',
                    'confidence': 0.0
                }
            }
        }
        if rect_info is not None:
            meta['crop_mode'] = 'ribbon_rectified'
            meta['line_thickness'] = rect_info['thickness']
            meta['rectify'] = rect_info
        lines_metadata.append(meta)

    return _ret(final_crops, lines_metadata)

# ==============================================================================
# Line Segmentation Algorithms
# ==============================================================================

def segment_lines_adaptive(img, threshold_pct=0.08, gap_threshold=12):
    """Adaptive Projection Profile line segmentation"""
    is_inverted = check_inverted_polarity(img)
    proc_img = (255 - img) if is_inverted else img
    gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    hist = np.sum(thresh, axis=1)
    
    max_val = np.max(hist)
    threshold = max_val * threshold_pct
    active_rows = np.where(hist >= threshold)[0]
    
    if len(active_rows) == 0:
        return [], []
        
    line_ranges = []
    start = active_rows[0]
    for i in range(1, len(active_rows)):
        if active_rows[i] - active_rows[i-1] > gap_threshold:
            line_ranges.append((start, active_rows[i-1]))
            start = active_rows[i]
    line_ranges.append((start, active_rows[-1]))
    
    # Return ranges and crop coordinates
    cropped_lines = []
    coords = []
    pad = 8
    h, w = gray.shape
    for s, e in line_ranges:
        s_pad = max(0, s - pad)
        e_pad = min(h, e + pad)
        cropped_lines.append(proc_img[s_pad:e_pad, :])
        coords.append((s_pad, e_pad))
    return cropped_lines, coords

def segment_lines_valleys(img, window_size=25, min_dist=35):
    """Valley Detection (local minima) line segmentation"""
    is_inverted = check_inverted_polarity(img)
    proc_img = (255 - img) if is_inverted else img
    gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    hist = np.sum(thresh, axis=1)
    
    smoothed = np.convolve(hist, np.ones(window_size)/window_size, mode='same')
    
    valleys = []
    radius = max(3, int(window_size / 2))
    for i in range(20, len(smoothed) - 20):
        if smoothed[i] == np.min(smoothed[i-radius:i+radius+1]):
            if not valleys or i - valleys[-1] > min_dist:
                valleys.append(i)
                
    split_points = [0] + valleys + [len(smoothed)]
    cropped_lines = []
    coords = []
    for idx in range(len(split_points) - 1):
        s = split_points[idx]
        e = split_points[idx + 1]
        if e - s < 15:
            continue
        cropped_lines.append(proc_img[s:e, :])
        coords.append((s, e))
    return cropped_lines, coords

def calculate_crop_sanity_score(conf, pred_text, median_len, crop_img, ctype):
    """Calculates sanity score for a crop prediction"""
    if not pred_text or crop_img is None or crop_img.size == 0:
        return -1.0
    
    score = conf
    # Penalize if prediction length is too far from median
    if median_len > 0:
        ratio = len(pred_text) / median_len
        if ratio > 1.4 or ratio < 0.6:
            score *= 0.8
            
    # Penalize if there is significant ink touching the top or bottom edges
    # We check the top 3 and bottom 3 rows
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    h, w = thresh.shape
    if h > 6:
        # Removed edge ink penalty because it falsely penalizes tight crops that perfectly bound the diacritics.
        pass
            
    # Boost safe slightly because it inherently suffers from resize-down confidence drops
    if ctype == "safe":
        score *= 1.15
    elif ctype in ["upper_rescue", "lower_rescue"]:
        score *= 1.10
    elif ctype == "loose":
        score *= 1.05
        
    return score

def compute_cer(pred, gt):
    if len(pred) == 0 and len(gt) == 0:
        return 0.0
    if len(gt) == 0:
        return 1.0
    import difflib
    similarity = difflib.SequenceMatcher(None, pred, gt).ratio()
    return 1.0 - similarity

def has_important_cham_sign_changes(pred1, pred2):
    if pred1 == pred2:
        return False
    import difflib
    diff = difflib.ndiff(pred1, pred2)
    for d in diff:
        if d.startswith('- ') or d.startswith('+ '):
            char = d[2]
            # Cham Unicode Block: U+AA00 to U+AA5F (with diacritics / vowels U+AA29 to U+AA4D)
            # Also support legacy ranges U+A800 to U+A87F
            if '\uAA00' <= char <= '\uAA5F' or '\uA800' <= char <= '\uA87F':
                return True
    return False

def segment_lines_advanced(img, base_coords, ocr_model, model_ver=None):
    """
    Advanced Multi-Crop Strategy with Risk-Gated Batched Inference
    """
    import time
    import cv2
    import numpy as np
    
    t_start = time.time()
    if not base_coords:
        return [], [], {
            "candidate_generation_sec": 0.0,
            "recognition_batch_pass1_sec": 0.0,
            "recognition_batch_pass2_sec": 0.0,
            "selection_sec": 0.0,
            "total_segmentation_sec": 0.0,
            "num_lines": 0,
            "num_candidates_pass1": 0,
            "num_candidates_pass2": 0,
            "ocr_model_calls": 0
        }
        
    if check_inverted_polarity(img):
        img = 255 - img
    img_h, img_w, _ = img.shape
    heights = [float(e - s) for s, e in base_coords]
    valid_heights = [h for h in heights if h <= 0.15 * img_h]
    median_line_height = float(np.median(valid_heights)) if valid_heights else (float(np.median(heights)) if heights else 30.0)
    
    # [Pre-compute legacy bounds, CCA, etc., identical to original until Phase 1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    line_components = [set() for _ in base_coords]
    ambiguous_components = set()
    cca_bounds = []
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        # Ignore tiny noise (< 25px area or < 5px dim) or massive image-wide background blobs
        if area < 25 or h < 5 or w < 5 or area > 0.25 * img_h * img_w or h > 0.6 * img_h:
            continue
        cy = y + h / 2
        assigned_lines = []
        for l_idx, (s, e) in enumerate(base_coords):
            if s - 5 <= cy <= e + 5:
                assigned_lines.append(l_idx)
                
        if len(assigned_lines) == 1:
            line_components[assigned_lines[0]].add(i)
        elif len(assigned_lines) > 1:
            ambiguous_components.add(i)
            for l_idx in assigned_lines:
                line_components[l_idx].add(i)
                
    valid_base_coords = []
    valid_line_components = []
    true_core_coords = []
    
    for idx, (s, e) in enumerate(base_coords):
        comps = line_components[idx]
        total_area = sum(stats[i][4] for i in comps) if comps else 0
        # If line has no text components or total ink area is below minimum threshold, skip empty background line
        if not comps or total_area < 80:
            continue
            
        core_s = min(stats[i][1] for i in comps)
        core_e = max(stats[i][1] + stats[i][3] for i in comps)
        
        valid_base_coords.append((s, e))
        valid_line_components.append(comps)
        true_core_coords.append((core_s, core_e))
        
    # Fallback safeguard: If all lines were filtered out, preserve original base_coords
    if not true_core_coords:
        valid_base_coords = list(base_coords)
        valid_line_components = list(line_components)
        true_core_coords = [(s, e) for s, e in base_coords]
        
    # Pre-Pass 1: Deduplicate heavily overlapping base line bands (IoM > 0.40)
    if len(true_core_coords) > 1:
        dedup_base = []
        dedup_comps = []
        dedup_core = []
        
        for (bs, be), comps, (cs, ce) in zip(valid_base_coords, valid_line_components, true_core_coords):
            h1 = max(1, ce - cs)
            keep = True
            for k_idx in range(len(dedup_base)):
                k_cs, k_ce = dedup_core[k_idx]
                h2 = max(1, k_ce - k_cs)
                inter = max(0, min(ce, k_ce) - max(cs, k_cs))
                iom = inter / min(h1, h2)
                if iom > 0.40:
                    if (ce - cs) < (k_ce - k_cs):
                        dedup_base[k_idx] = (bs, be)
                        dedup_comps[k_idx] = comps
                        dedup_core[k_idx] = (cs, ce)
                    keep = False
                    break
            if keep:
                dedup_base.append((bs, be))
                dedup_comps.append(comps)
                dedup_core.append((cs, ce))
                
        valid_base_coords = dedup_base
        valid_line_components = dedup_comps
        true_core_coords = dedup_core
        
    base_coords = valid_base_coords
    
    # Re-assign line_components based on true_core_coords to capture both top diacritics and main consonants
    line_components = [set() for _ in true_core_coords]
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 25 or h < 5 or w < 5 or area > 0.25 * img_h * img_w or h > 0.6 * img_h:
            continue
        cy = y + h / 2
        for l_idx, (cs, ce) in enumerate(true_core_coords):
            if cs - 8 <= cy <= ce + 20:
                line_components[l_idx].add(i)
            
    page_safe = True
    for idx in range(len(base_coords) - 1):
        if true_core_coords[idx+1][0] - true_core_coords[idx][1] < 0.15 * median_line_height:
            page_safe = False
            break
            
    # Phase 1: Candidate Generation (Pass 1)
    t_candidate = time.time()
    pass1_candidates = []
    lines_meta = []
    
    for idx, (s, e) in enumerate(true_core_coords):
        comps = line_components[idx]
        if comps:
            # Sort components horizontally by x position
            sorted_comps = sorted(list(comps), key=lambda i: stats[i][0])
            
            # Filter out non-text graphic background blobs (like scroll ends, clouds, background artwork)
            text_comps = []
            for i in sorted_comps:
                x, y, w, h, area = stats[i]
                if (area > 300 and y < 15) or (area > 500 and w / max(1, h) > 2.2 and y < 25) or (area > 1000 and y < 10):
                    continue
                text_comps.append(i)
                
            if not text_comps:
                text_comps = sorted_comps
                
            # Perform 1D Horizontal Adaptive Clustering
            max_cluster_gap = max(25, min(45, int(1.0 * median_line_height)))
            clusters = []
            curr_cluster = []
            
            for i in text_comps:
                if not curr_cluster:
                    curr_cluster.append(i)
                else:
                    prev_i = curr_cluster[-1]
                    prev_right = stats[prev_i][0] + stats[prev_i][2]
                    curr_left = stats[i][0]
                    if curr_left - prev_right <= max_cluster_gap:
                        curr_cluster.append(i)
                    else:
                        clusters.append(curr_cluster)
                        curr_cluster = [i]
            if curr_cluster:
                clusters.append(curr_cluster)
                
            # Identify primary text cluster with highest ink area
            if clusters:
                main_c_idx = max(range(len(clusters)), key=lambda idx: sum(stats[i][4] for i in clusters[idx]))
                main_c = clusters[main_c_idx]
                main_left = min(stats[i][0] for i in main_c)
                main_right = max(stats[i][0] + stats[i][2] for i in main_c)
                
                # Prune isolated satellite outlier clusters (separated by > 1.2 * line_height from main text block)
                outlier_gap = max(35, min(60, int(1.2 * median_line_height)))
                best_cluster = []
                for c in clusters:
                    c_left = min(stats[i][0] for i in c)
                    c_right = max(stats[i][0] + stats[i][2] for i in c)
                    if not (c_right < main_left - outlier_gap or c_left > main_right + outlier_gap):
                        best_cluster.extend(c)
            else:
                best_cluster = text_comps
            
            xmin = min(stats[i][0] for i in best_cluster)
            xmax = max(stats[i][0] + stats[i][2] for i in best_cluster)
            ymin = min(stats[i][1] for i in best_cluster)
            ymax = max(stats[i][1] + stats[i][3] for i in best_cluster)
        else:
            xmin, xmax = 0, img_w
            ymin, ymax = int(s), int(e)
            
        line_meta = {
            "line_id": idx,
            "core_box": [int(s), int(e)],
            "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
            "needs_review": False,
            "reason": [],
            "candidates": {},
            "overlap_prev_px": 0,
            "overlap_next_px": 0
        }
        
        core_top = true_core_coords[idx][0]
        core_bot = true_core_coords[idx][1]
        prev_bot = true_core_coords[idx-1][1] if idx > 0 else 0
        next_top = true_core_coords[idx+1][0] if idx < len(true_core_coords)-1 else img_h
        
        gap_prev = core_top - prev_bot if idx > 0 else 999999
        gap_next = next_top - core_bot if idx < len(true_core_coords)-1 else 999999
        
        median_ink_height = median_line_height
        is_gap_safe = min(gap_prev, gap_next) >= 0.35 * median_ink_height
        
        valley_s, valley_e = base_coords[idx]
        has_edge_ink_risk = False
        if int(valley_s) > 0 and core_top <= int(valley_s) + 2:
            has_edge_ink_risk = True
        if int(valley_e) < img_h and core_bot >= int(valley_e) - 2:
            has_edge_ink_risk = True
            
        close_risk = False
        if min(gap_prev, gap_next) < 0.25 * median_line_height:
            close_risk = True
            line_meta["reason"].append("very_close_lines_or_touching_diacritics")
            
        pad_tight_top = int(0.05 * median_line_height)
        pad_tight_bot = int(0.05 * median_line_height)
        
        safety_margin_px = max(2, int(0.05 * median_line_height))
        
        if close_risk or not page_safe:
            safe_top = ymin - safety_margin_px
            safe_bot = ymax + safety_margin_px
        else:
            safe_top = valley_s
            safe_bot = valley_e
            
        min_top_pad = int(0.40 * median_line_height)
        min_bot_pad = int(0.40 * median_line_height)
        
        safe_top = min(safe_top, core_top - min_top_pad)
        safe_bot = max(safe_bot, core_bot + min_bot_pad)
        
        if idx > 0:
            safe_top = max(safe_top, true_core_coords[idx-1][1] + 2)
        if idx < len(true_core_coords) - 1:
            safe_bot = min(safe_bot, true_core_coords[idx+1][0] - 2)
            
        pad_safe_top = int(core_top - safe_top)
        pad_safe_bot = int(safe_bot - core_bot)
        
        def clamp_crop(pad_t, pad_b):
            c_s = max(0, core_top - pad_t)
            c_e = min(img_h, core_bot + pad_b)
            ov_prev = max(0, prev_bot - c_s) if idx > 0 else 0
            ov_next = max(0, c_e - next_top) if idx < len(true_core_coords)-1 else 0
            return (int(c_s), int(c_e), int(ov_prev), int(ov_next))
            
        candidates_to_run = ["safe"]
            
        for ctype in candidates_to_run:
            if ctype == "safe":
                c_s, c_e, o_p, o_n = clamp_crop(pad_safe_top, pad_safe_bot)
                
            c_img = img[c_s:c_e, :].copy()
            if ctype == "safe" and (close_risk or not page_safe):
                components_to_erase = set()
                for other_idx in range(len(true_core_coords)):
                    if other_idx != idx:
                        components_to_erase.update(line_components[other_idx])
                components_to_erase.difference_update(line_components[idx])
                
                # Fast LUT lookup instead of np.isin (prevents CPU freeze on large label sets)
                erase_lut = np.zeros(num_labels, dtype=bool)
                if components_to_erase:
                    erase_lut[list(components_to_erase)] = True
                
                label_roi = labels[c_s:c_e, :]
                erase_mask = erase_lut[label_roi]
                c_img[erase_mask] = [255, 255, 255]
                
            cand_obj = {
                "line_id": idx,
                "crop_type": ctype,
                "crop_img": c_img,
                "crop_box": [c_s, c_e],
                "overlap_prev_px": o_p,
                "overlap_next_px": o_n,
                "prediction": "",
                "confidence": 0.0,
                "pred_length": 0
            }
            pass1_candidates.append(cand_obj)
            line_meta["candidates"][ctype] = cand_obj
            
        # Store metadata and flags for later
        line_meta["close_risk"] = close_risk
        line_meta["page_safe"] = page_safe
        line_meta["is_gap_safe"] = is_gap_safe
        line_meta["has_edge_ink_risk"] = has_edge_ink_risk
        line_meta["has_ambiguous_cc"] = any(c in ambiguous_components for c in line_components[idx])
        lines_meta.append(line_meta)
        
    candidate_generation_sec = time.time() - t_candidate
    
    # Phase 3: Pass 1 Batch Recognition
    t_pass1 = time.time()
    if pass1_candidates:
        imgs = [preprocess_crop_for_rec(c["crop_img"]) for c in pass1_candidates]
        # ocr_model accepts a list of images and returns list of (text, conf)
        # However, PaddleOCR's predict() might fail if empty list or invalid dims.
        valid_imgs = []
        valid_indices = []
        for i, m in enumerate(imgs):
            if m.size > 0 and m.shape[0] >= 8 and m.shape[1] >= 8:
                # Prevent aspect ratio blowup that causes CPU hang during inference resizing
                scaled_w = int(m.shape[1] * 48 / m.shape[0])
                if scaled_w > 2000:
                    # Downscale horizontally to protect the sequence processing layer
                    m = cv2.resize(m, (2000, m.shape[0]))
                valid_imgs.append(m)
                valid_indices.append(i)
                
        if valid_imgs:
            try:
                print(f"DEBUG: [segment_lines_advanced] Calling ocr_model for Pass 1 ({len(valid_imgs)} images)...")
                results, sel_vers_p1, _ = invoke_recognizer_with_provenance(
                    ocr_model, valid_imgs, model_ver=model_ver
                )
                print("DEBUG: [segment_lines_advanced] Pass 1 prediction finished.")
                for local_idx, (vi, res) in enumerate(zip(valid_indices, results)):
                    pred_text, conf = res if res else ("", 0.0)
                    pred_text = pred_text.strip()
                    pass1_candidates[vi]["prediction"] = pred_text
                    pass1_candidates[vi]["confidence"] = float(conf)
                    pass1_candidates[vi]["pred_length"] = len(pred_text)
                    if local_idx < len(sel_vers_p1) and sel_vers_p1[local_idx] is not None:
                        pass1_candidates[vi]["routed_version"] = sel_vers_p1[local_idx]
            except Exception as e:
                print(f"DEBUG: [segment_lines_advanced] Batch inference pass 1 failed: {e}")
                
    recognition_batch_pass1_sec = time.time() - t_pass1
    
    # Phase 6: Two-Pass Fallback
    t_pass2_gen = time.time()
    pass2_candidates = []
    
    for idx, line_meta in enumerate(lines_meta):
        # Determine if we need Pass 2
        cands = line_meta["candidates"]
        best_conf = max([c["confidence"] for c in cands.values()]) if cands else 0.0
        
        is_clean_line = line_meta["is_gap_safe"] and not line_meta["has_edge_ink_risk"]
        
        needs_pass2 = False
        if is_clean_line:
            # Clean line: only run Pass 2 if confidence is below acceptable threshold (0.75)
            if best_conf < 0.75:
                needs_pass2 = True
        else:
            # Risky line: run Pass 2 if confidence is below high safety threshold (0.90)
            if best_conf < 0.90:
                needs_pass2 = True
                
        if needs_pass2:
            core_top = true_core_coords[idx][0]
            core_bot = true_core_coords[idx][1]
            prev_bot = true_core_coords[idx-1][1] if idx > 0 else 0
            next_top = true_core_coords[idx+1][0] if idx < len(true_core_coords)-1 else img_h
            
            pad_safe_top = int(core_top - cands["safe"]["crop_box"][0]) if "safe" in cands else 0
            pad_safe_bot = int(cands["safe"]["crop_box"][1] - core_bot) if "safe" in cands else 0
            
            pad_loose_top = int(0.80 * median_line_height)
            pad_loose_bot = int(0.80 * median_line_height)
            
            pad_tight_top = int(0.05 * median_line_height)
            pad_tight_bot = int(0.05 * median_line_height)
            
            pad_upper_rescue_top = pad_loose_top
            pad_upper_rescue_bot = pad_safe_bot
            
            pad_lower_rescue_top = pad_safe_top
            pad_lower_rescue_bot = pad_loose_bot
            
            def clamp_crop2(pad_t, pad_b):
                c_s = max(0, core_top - pad_t)
                c_e = min(img_h, core_bot + pad_b)
                ov_prev = max(0, prev_bot - c_s) if idx > 0 else 0
                ov_next = max(0, c_e - next_top) if idx < len(true_core_coords)-1 else 0
                return (int(c_s), int(c_e), int(ov_prev), int(ov_next))
                
            # We need to generate tight crop now since it wasn't generated in Pass 1
            tight_s, tight_e, tight_op, tight_on = clamp_crop2(pad_tight_top, pad_tight_bot)
            tight_img = img[tight_s:tight_e, :].copy()
            tight_cand = {
                "line_id": idx,
                "crop_type": "tight",
                "crop_img": tight_img,
                "crop_box": [tight_s, tight_e],
                "overlap_prev_px": tight_op,
                "overlap_next_px": tight_on,
                "prediction": "",
                "confidence": 0.0,
                "pred_length": 0
            }
            pass2_candidates.append(tight_cand)
            line_meta["candidates"]["tight"] = tight_cand

            for ctype, pt, pb in [
                ("upper_rescue", pad_upper_rescue_top, pad_upper_rescue_bot),
                ("lower_rescue", pad_lower_rescue_top, pad_lower_rescue_bot),
                ("loose", pad_loose_top, pad_loose_bot)
            ]:
                c_s, c_e, o_p, o_n = clamp_crop2(pt, pb)
                c_img = img[c_s:c_e, :].copy()
                cand_obj = {
                    "line_id": idx,
                    "crop_type": ctype,
                    "crop_img": c_img,
                    "crop_box": [c_s, c_e],
                    "overlap_prev_px": o_p,
                    "overlap_next_px": o_n,
                    "prediction": "",
                    "confidence": 0.0,
                    "pred_length": 0
                }
                pass2_candidates.append(cand_obj)
                line_meta["candidates"][ctype] = cand_obj

    # Run Batch Pass 2 with coordinate-level deduplication (preserving exact recognizer provenance)
    t_pass2_rec = time.time()
    if pass2_candidates:
        # Cache for coordinate-level deduplication: (s, e) -> (prediction, confidence, routed_version)
        seen_boxes = {}
        pending_boxes = set()
        
        # Pre-populate seen_boxes with Pass 1 (safe) results to avoid re-running identical boxes
        for line_meta in lines_meta:
            if "safe" in line_meta["candidates"]:
                safe_c = line_meta["candidates"]["safe"]
                seen_boxes[tuple(safe_c["crop_box"])] = (
                    safe_c["prediction"],
                    safe_c["confidence"],
                    safe_c.get("routed_version"),
                )
                
        exec_candidates = [] # candidates that require actual OCR execution
        deferred_copies = [] # candidates that will copy results from an identical box
        
        for cand in pass2_candidates:
            c_box = tuple(cand["crop_box"])
            if c_box in seen_boxes:
                if c_box in pending_boxes:
                    deferred_copies.append((cand, c_box))
                else:
                    p_txt, p_conf, p_ver = seen_boxes[c_box]
                    cand["prediction"] = p_txt
                    cand["confidence"] = p_conf
                    cand["pred_length"] = len(p_txt)
                    if p_ver is not None:
                        cand["routed_version"] = p_ver
            else:
                exec_candidates.append(cand)
                seen_boxes[c_box] = ("", 0.0, None)
                pending_boxes.add(c_box)
                
        if exec_candidates:
            imgs = [preprocess_crop_for_rec(c["crop_img"]) for c in exec_candidates]
            valid_imgs = []
            valid_indices = []
            for i, m in enumerate(imgs):
                if m.size > 0 and m.shape[0] >= 8 and m.shape[1] >= 8:
                    # Prevent aspect ratio blowup that causes CPU hang during inference resizing
                    scaled_w = int(m.shape[1] * 48 / m.shape[0])
                    if scaled_w > 2000:
                        # Downscale horizontally to protect the sequence processing layer
                        m = cv2.resize(m, (2000, m.shape[0]))
                    valid_imgs.append(m)
                    valid_indices.append(i)
                    
            if valid_imgs:
                try:
                    print(f"DEBUG: [segment_lines_advanced] Calling ocr_model for Pass 2 ({len(valid_imgs)} images, deduplicated from {len(pass2_candidates)})...")
                    results, sel_vers_p2, _ = invoke_recognizer_with_provenance(
                        ocr_model, valid_imgs, model_ver=model_ver
                    )
                    print("DEBUG: [segment_lines_advanced] Pass 2 prediction finished.")
                    for local_idx, (vi, res) in enumerate(zip(valid_indices, results)):
                        pred_text, conf = res if res else ("", 0.0)
                        pred_text = pred_text.strip()
                        r_ver = sel_vers_p2[local_idx] if local_idx < len(sel_vers_p2) else None
                        
                        exec_cand = exec_candidates[vi]
                        exec_cand["prediction"] = pred_text
                        exec_cand["confidence"] = float(conf)
                        exec_cand["pred_length"] = len(pred_text)
                        if r_ver is not None:
                            exec_cand["routed_version"] = r_ver
                        
                        seen_boxes[tuple(exec_cand["crop_box"])] = (pred_text, float(conf), r_ver)
                except Exception as e:
                    print(f"DEBUG: [segment_lines_advanced] Batch inference pass 2 failed: {e}")
                    
        # Process deferred copies
        for cand, c_box in deferred_copies:
            pred_text, conf, r_ver = seen_boxes[c_box]
            cand["prediction"] = pred_text
            cand["confidence"] = conf
            cand["pred_length"] = len(pred_text)
            if r_ver is not None:
                cand["routed_version"] = r_ver
            
    recognition_batch_pass2_sec = time.time() - t_pass2_rec
    
    # Selection and Finalization
    t_select = time.time()
    final_crops = []
    
    for idx, line_meta in enumerate(lines_meta):
        cand_data = line_meta["candidates"]
        
        # Determine best candidate
        eval_order = ["safe", "tight"]
        for ctype in ["upper_rescue", "lower_rescue", "loose"]:
            if ctype in cand_data:
                eval_order.append(ctype)
                
        selected_ctype = None
        for ctype in ["safe", "tight"]:
            if ctype in cand_data and cand_data[ctype]["confidence"] >= 0.90 and cand_data[ctype]["pred_length"] > 0:
                selected_ctype = ctype
                break
                
        if selected_ctype is not None:
            best_score = cand_data[selected_ctype]["confidence"]
            cand_data[selected_ctype]["sanity_score"] = best_score
        else:
            run_lengths = [float(c["pred_length"]) for c in cand_data.values() if c["pred_length"] > 0]
            median_len = float(np.median(run_lengths)) if run_lengths else 0.0
            
            best_score = -1.0
            selected_ctype = "safe" if "safe" in cand_data else "tight"
            
            for ctype in eval_order:
                if ctype not in cand_data:
                    continue
                data = cand_data[ctype]
                score = calculate_crop_sanity_score(data["confidence"], data["prediction"], median_len, data["crop_img"], ctype)
                
                if ctype == "loose" and "safe" in cand_data:
                    safe_data = cand_data["safe"]
                    if safe_data["pred_length"] > 0:
                        ratio = data["pred_length"] / safe_data["pred_length"]
                        if ratio > 1.35:
                            score *= 0.3
                    if safe_data["confidence"] - data["confidence"] > 0.10:
                        score *= 0.4
                    tokens = data["prediction"].split()
                    single_chars = sum(1 for t in tokens if len(t) == 1)
                    if len(tokens) > 0 and single_chars / len(tokens) > 0.3:
                        score *= 0.5
                    if len(safe_data["prediction"]) > 0:
                        import difflib
                        similarity = difflib.SequenceMatcher(None, data["prediction"], safe_data["prediction"]).ratio()
                        if similarity < 0.5:
                            score *= 0.6
                            
                data["sanity_score"] = score
                if score > best_score:
                    best_score = score
                    selected_ctype = ctype
                    
            if line_meta["page_safe"] and selected_ctype not in ["safe", "tight"]:
                safe_score = cand_data.get("safe", {}).get("sanity_score", 0)
                if safe_score > best_score * 0.9:
                    selected_ctype = "safe"
                    best_score = safe_score
                    
        line_meta["selected_crop_type"] = selected_ctype
        line_meta["crop_sanity_score"] = best_score
        
        sel_data = cand_data[selected_ctype]
        if sel_data.get("routed_version") is not None:
            line_meta["routed_version"] = sel_data["routed_version"]
        
        # Flags
        needs_review_reasons = []
        if line_meta["has_ambiguous_cc"]:
            needs_review_reasons.append("ambiguous_component_assignment")
            
        if selected_ctype not in ["safe", "tight"]:
            if sel_data["overlap_prev_px"] > 5 or sel_data["overlap_next_px"] > 5:
                needs_review_reasons.append("adjacent_line_contamination")
                
        # Only flag needs_review for multi-crop disagreement if important Cham signs change
        # or if there is a significant confidence drop (>= 0.15) compared to safe crop
        if len(cand_data) > 1 and "safe" in cand_data:
            safe_pred = cand_data["safe"].get("prediction", "")
            safe_conf = cand_data["safe"].get("confidence", 0.0)
            
            has_cham_sign_diff = False
            for ctype, cand in cand_data.items():
                if ctype != "safe" and has_important_cham_sign_changes(safe_pred, cand.get("prediction", "")):
                    has_cham_sign_diff = True
                    break
                    
            if has_cham_sign_diff:
                needs_review_reasons.append("important_cham_sign_change")
                
            if safe_conf - sel_data["confidence"] >= 0.15:
                needs_review_reasons.append("significant_confidence_drop")
                
        line_meta["needs_review"] = len(needs_review_reasons) > 0
        line_meta["reason"] = needs_review_reasons
        
        if sel_data["confidence"] > 0.8:
            line_meta["crop_confidence"] = "high"
        elif sel_data["confidence"] > 0.5:
            line_meta["crop_confidence"] = "medium"
        else:
            line_meta["crop_confidence"] = "low"
            
        line_meta["overlap_prev_px"] = sel_data["overlap_prev_px"]
        line_meta["overlap_next_px"] = sel_data["overlap_next_px"]
        
        # Compatibility with downstream (app expects ["img"] in candidate)
        for c in cand_data.values():
            c["img"] = c["crop_img"]
            
        final_crops.append(sel_data["crop_img"])
        
    selection_sec = time.time() - t_select
    total_sec = time.time() - t_start
    
    print("\n[DEBUG lines_meta before NMS]:")
    for m in lines_meta:
        print(f"  Line {m['line_id']}: bbox={m['bbox']}, core_box={m['core_box']}")
    
    # Phase 7: Non-Text Background Line Pruning & 2D Vertical/Horizontal NMS
    filtered_lines_meta = []
    filtered_final_crops = []
    
    for meta, crop in zip(lines_meta, final_crops):
        sel_type = meta.get("selected_crop_type", "safe")
        cand_sel = meta["candidates"].get(sel_type, {})
        conf = cand_sel.get("confidence", 0.0)
        pred = cand_sel.get("prediction", "").strip()
        
        has_cham_char = any('\uAA00' <= char <= '\uAA5F' or '\uA800' <= char <= '\uA87F' for char in pred)
        has_alphanumeric = any(char.isalnum() for char in pred)
        
        # Strict filter out false positive lines from background artwork (dragons, elephants, ocean graphics)
        # Only treat as fake background artwork if confidence < 0.35 AND neither Cham nor alphanumeric characters are found
        is_fake_background = (conf < 0.35 and not has_cham_char and not has_alphanumeric)
        if not is_fake_background and conf >= 0.15:
            filtered_lines_meta.append(meta)
            filtered_final_crops.append(crop)
            
    # Safeguard: If all lines were filtered, preserve the single highest-confidence candidate line,
    # but only when it is real text. Blank / noise-only pages must return no lines instead of an
    # empty or hallucinated line (stress test 2026-11-04: blank page -> '' / 'R' marked VERIFIED).
    if not filtered_lines_meta and lines_meta:
        best_i = max(range(len(lines_meta)), key=lambda i: lines_meta[i]["candidates"].get(lines_meta[i].get("selected_crop_type", "safe"), {}).get("confidence", 0.0))
        best_cand = lines_meta[best_i]["candidates"].get(lines_meta[best_i].get("selected_crop_type", "safe"), {})
        if best_cand.get("confidence", 0.0) >= 0.5 and best_cand.get("prediction", "").strip():
            lines_meta[best_i]["needs_review"] = True
            lines_meta[best_i]["reason"] = list(lines_meta[best_i].get("reason") or []) + ["only_low_quality_candidates"]
            filtered_lines_meta = [lines_meta[best_i]]
            filtered_final_crops = [final_crops[best_i]]
        
    # Non-Maximum Suppression (NMS) with 2D IoM (checks both vertical and horizontal overlaps)
    if len(filtered_lines_meta) > 1:
        indexed_items = []
        for meta, crop in zip(filtered_lines_meta, filtered_final_crops):
            sel_type = meta.get("selected_crop_type", "safe")
            cand_sel = meta["candidates"].get(sel_type, {})
            conf = cand_sel.get("confidence", 0.0)
            box = meta["core_box"]
            bbox = meta.get("bbox", [0, box[0], img_w, box[1]])
            indexed_items.append((conf, box, bbox, meta, crop))
            
        indexed_items.sort(key=lambda x: x[0], reverse=True)
        nms_kept = []
        for conf, box, bbox, meta, crop in indexed_items:
            s1, e1 = box
            h1 = max(1, e1 - s1)
            x1_min, _, x1_max, _ = bbox
            w1 = max(1, x1_max - x1_min)
            
            keep = True
            for k_conf, k_box, k_bbox, _, _ in nms_kept:
                s2, e2 = k_box
                h2 = max(1, e2 - s2)
                inter_s = max(s1, s2)
                inter_e = min(e1, e2)
                inter_y = max(0, inter_e - inter_s)
                iom_y = inter_y / min(h1, h2)
                
                # Check horizontal overlap to avoid suppressing side-by-side columns
                x2_min, _, x2_max, _ = k_bbox
                w2 = max(1, x2_max - x2_min)
                inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
                iom_x = inter_x / min(w1, w2)
                
                # Only suppress if both vertical and horizontal overlaps are significant
                if iom_y > 0.40 and iom_x > 0.30:
                    keep = False
                    break
            if keep:
                nms_kept.append((conf, box, bbox, meta, crop))
                
        nms_kept.sort(key=lambda x: x[1][0])
        filtered_lines_meta = [x[3] for x in nms_kept]
        filtered_final_crops = [x[4] for x in nms_kept]

    # Relative confidence pruning against max document confidence
    if len(filtered_lines_meta) > 1:
        max_doc_conf = max(
            meta["candidates"].get(meta.get("selected_crop_type", "safe"), {}).get("confidence", 0.0)
            for meta in filtered_lines_meta
        )
        if max_doc_conf > 0.60:
            rel_filtered_meta = []
            rel_filtered_crops = []
            for meta, crop in zip(filtered_lines_meta, filtered_final_crops):
                sel_type = meta.get("selected_crop_type", "safe")
                cand_sel = meta["candidates"].get(sel_type, {})
                conf = cand_sel.get("confidence", 0.0)
                pred = cand_sel.get("prediction", "").strip()
                has_cham_char = any('\uAA00' <= char <= '\uAA5F' or '\uA800' <= char <= '\uA87F' for char in pred)
                
                # If a line has real Cham text, never prune it even if confidence is lower
                # Only prune pure low-confidence non-Cham noise blobs
                if conf < 0.35 and (max_doc_conf - conf > 0.35) and not has_cham_char:
                    continue
                rel_filtered_meta.append(meta)
                rel_filtered_crops.append(crop)
                
            if rel_filtered_meta:
                filtered_lines_meta = rel_filtered_meta
                filtered_final_crops = rel_filtered_crops

    lines_meta = filtered_lines_meta
    final_crops = filtered_final_crops

    # Phase 8: Profiler
    print(f"""
    ⏱️ Profiler [segment_lines_advanced]:
    - candidate_generation_sec: {candidate_generation_sec:.4f}s
    - recognition_batch_pass1_sec: {recognition_batch_pass1_sec:.4f}s
    - recognition_batch_pass2_sec: {recognition_batch_pass2_sec:.4f}s
    - selection_sec: {selection_sec:.4f}s
    - total_sec: {total_sec:.4f}s
    - num_lines: {len(lines_meta)}
    - num_candidates_pass1: {len(pass1_candidates)}
    - num_candidates_pass2: {len(pass2_candidates)}
    - ocr_model_calls: { (1 if pass1_candidates else 0) + (1 if pass2_candidates else 0) }
    """)
        
    profiler_dict = {
        "candidate_generation_sec": round(candidate_generation_sec, 4),
        "recognition_batch_pass1_sec": round(recognition_batch_pass1_sec, 4),
        "recognition_batch_pass2_sec": round(recognition_batch_pass2_sec, 4),
        "selection_sec": round(selection_sec, 4),
        "total_segmentation_sec": round(total_sec, 4),
        "num_lines": len(lines_meta),
        "num_candidates_pass1": len(pass1_candidates),
        "num_candidates_pass2": len(pass2_candidates),
        "ocr_model_calls": (1 if pass1_candidates else 0) + (1 if pass2_candidates else 0)
    }
        
    return final_crops, lines_meta, profiler_dict



# ==============================================================================
# OCR Pipeline (shared by the HTTP handler and offline benchmark scripts)
# ==============================================================================
def run_ocr_pipeline(img, model_ver=None, method='dbnet', threshold=0.05, gap=12, win=25,
                     reading_order='auto', t_start=None):
    """Runs segmentation + recognition + quality gates on a BGR image.

    Returns the /ocr JSON response dict: {'lines', 'profiler', 'document', 'model_info'}.
    """
    import time
    t_start = t_start or time.time()
    model_ver = model_ver or DEFAULT_CHAM_MODEL
    ocr = get_ocr_model(model_ver)

    t_seg = time.time()
    base_segmentation_sec = 0.0
    profiler_dict = {
        'method': method,
        'base_segmentation_sec': 0.0,
        'legacy_fast_pass_count': 0,
        'pass2_rescue_count': 0
    }
    doc_reasons = []
    doc_extra = {}
    executed_det_name = None
    executed_det_switched = False

    if method == 'dbnet':
        orig_h, orig_w = img.shape[:2]
        work_img, deskew_M_inv, deskew_deg = maybe_safe_deskew_page(img)
        if deskew_M_inv is not None:
            profiler_dict['deskew_angle_deg'] = deskew_deg
            doc_extra['deskew_angle_deg'] = deskew_deg
        ink = analyze_text_ink(work_img)
        doc_extra['ink_ratio'] = round(ink['ink_ratio'], 5)
        if is_blank_page(ink):
            # No text-like ink: never hallucinate a line on blank / near-blank paper
            print("📄 Blank page detected (no text-like ink); skipping detection and recognition.")
            final_crops, lines_metadata = [], []
            doc_reasons.append('blank_page')
            profiler_dict['blank_page'] = True
        else:
            print("🔍 Running PaddleOCR DBNet segmentation...")
            det_info = {}
            try:
                det = get_det_model()
                final_crops, lines_metadata, det_info = segment_lines_dbnet(
                    work_img, det, ink=ink, return_info=True, reading_order=reading_order)
            except Exception:
                import traceback
                print(f"❌ DBNet segmentation error: {traceback.format_exc()}")
                final_crops, lines_metadata = [], []

            if deskew_M_inv is not None and lines_metadata:
                for meta in lines_metadata:
                    poly = meta.get('polygon')
                    if not poly and meta.get('bbox') and len(meta['bbox']) == 4:
                        bx0, by0, bx1, by1 = meta['bbox']
                        poly = [[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]]
                    if poly:
                        orig_pts = map_deskew_pts_to_original(poly, deskew_M_inv, orig_w, orig_h)
                        meta['polygon'] = np.round(orig_pts, 1).tolist()
                        x0_m = int(round(float(orig_pts[:, 0].min())))
                        y0_m = int(round(float(orig_pts[:, 1].min())))
                        x1_m = int(round(float(orig_pts[:, 0].max())))
                        y1_m = int(round(float(orig_pts[:, 1].max())))
                        meta['bbox'] = [x0_m, y0_m, x1_m, y1_m]
                        meta['core_box'] = [y0_m, y1_m]

            base_segmentation_sec = time.time() - t_seg
            if det_info:
                executed_det_name = det_info.get('detector')
                executed_det_switched = bool(det_info.get('detector_switched'))
                doc_extra['layout'] = det_info.get('layout')
                doc_extra['uncovered_ink_ratio'] = det_info.get('uncovered_ink_ratio')
                profiler_dict['det_lines'] = det_info.get('lines_detected')
                profiler_dict['detector'] = executed_det_name
                profiler_dict['detector_switched'] = executed_det_switched
                profiler_dict['uncovered_ink_ratio_primary'] = det_info.get('uncovered_ink_ratio_primary')

            if not final_crops:
                try:
                    lab_f = cv2.cvtColor(work_img, cv2.COLOR_BGR2LAB)
                    l_ch, a_ch, b_ch = cv2.split(lab_f)
                    clahe_img = cv2.cvtColor(
                        cv2.merge([cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8)).apply(l_ch), a_ch, b_ch]),
                        cv2.COLOR_LAB2BGR,
                    )
                    det = get_det_model()
                    final_crops, lines_metadata, det_info = segment_lines_dbnet(
                        clahe_img, det, ink=analyze_text_ink(clahe_img), return_info=True, reading_order=reading_order
                    )
                    if final_crops:
                        # Re-slice final_crops from un-enhanced work_img when polygon/bbox exists, or keep CLAHE crops for faint ink
                        profiler_dict['clahe_faint_ink_rescue'] = True
                        if det_info:
                            executed_det_name = det_info.get('detector')
                            executed_det_switched = bool(det_info.get('detector_switched'))
                            profiler_dict['det_lines'] = det_info.get('lines_detected')
                            profiler_dict['detector'] = executed_det_name
                            profiler_dict['detector_switched'] = executed_det_switched
                        if deskew_M_inv is not None and lines_metadata:
                            for meta in lines_metadata:
                                poly = meta.get('polygon')
                                if poly:
                                    orig_pts = map_deskew_pts_to_original(poly, deskew_M_inv, orig_w, orig_h)
                                    meta['polygon'] = np.round(orig_pts, 1).tolist()
                                    x0_m, y0_m = int(round(float(orig_pts[:, 0].min()))), int(round(float(orig_pts[:, 1].min())))
                                    x1_m, y1_m = int(round(float(orig_pts[:, 0].max()))), int(round(float(orig_pts[:, 1].max())))
                                    meta['bbox'] = [x0_m, y0_m, x1_m, y1_m]
                                    meta['core_box'] = [y0_m, y1_m]
                except Exception as e_clahe:
                    print(f"⚠️ CLAHE faint-ink rescue skipped: {e_clahe}")

            if not final_crops:
                print("⚠️ DBNet found 0 text lines, falling back to Valley segmentation...")
                _, coords = segment_lines_valleys(img, window_size=win, min_dist=gap)
                final_crops, lines_metadata, adv_profiler = segment_lines_advanced(
                    img, coords, ocr, model_ver=model_ver
                )
                profiler_dict.update(adv_profiler)
                profiler_dict['fallback'] = 'valley'
                doc_reasons.append('detector_found_no_lines')
            else:
                t_rec = time.time()
                rec_crops = [preprocess_crop_for_rec(c) for c in final_crops]
                rec_res, sel_vers, _ = invoke_recognizer_with_provenance(ocr, rec_crops, model_ver=model_ver)
                rec_sec = time.time() - t_rec
                print(f"⏱️  DBNet batch recognition ({len(final_crops)} lines) took: {rec_sec:.4f}s")
                for idx, (pred_text, conf) in enumerate(rec_res):
                    lines_metadata[idx]['candidates']['dbnet']['prediction'] = pred_text
                    lines_metadata[idx]['candidates']['dbnet']['confidence'] = conf
                    if idx < len(sel_vers) and sel_vers[idx] is not None:
                        lines_metadata[idx]['routed_version'] = sel_vers[idx]
                        lines_metadata[idx]['candidates']['dbnet']['routed_version'] = sel_vers[idx]
                num_detected = len(lines_metadata)
                dropped_text_lines = []
                lines_metadata, final_crops = apply_line_quality_gates(lines_metadata, final_crops, dropped_text_lines)
                profiler_dict['dropped_empty_or_noise_lines'] = num_detected - len(lines_metadata)
                if dropped_text_lines:
                    doc_reasons.append('unreadable_lines_dropped')
                    doc_extra['unreadable_lines_dropped'] = len(dropped_text_lines)
                if not lines_metadata:
                    doc_reasons.append('no_reliable_text')
                unc = det_info.get('uncovered_ink_ratio') if det_info else None
                if unc is not None and unc >= UNCOVERED_INK_REVIEW_RATIO:
                    doc_reasons.append('possible_missed_lines')
                profiler_dict['base_segmentation_sec'] = round(base_segmentation_sec, 4)
                profiler_dict['rec_inference_sec'] = round(rec_sec, 4)
                profiler_dict['num_lines'] = len(final_crops)
                profiler_dict['total_segmentation_sec'] = round(base_segmentation_sec, 4)
                profiler_dict['ocr_model_calls'] = 1
    else:
        if method == 'valley':
            _, coords = segment_lines_valleys(img, window_size=win, min_dist=gap)
        else:
            _, coords = segment_lines_adaptive(img, threshold_pct=threshold, gap_threshold=gap)
        base_segmentation_sec = time.time() - t_seg
        print(f"⏱️  Base line segmentation took: {base_segmentation_sec:.4f}s")

        t_adv = time.time()
        # Apply Advanced Segmentation and Multi-Crop
        final_crops, lines_metadata, adv_profiler = segment_lines_advanced(
            img, coords, ocr, model_ver=model_ver
        )
        profiler_dict.update(adv_profiler)
        print(f"⏱️  segment_lines_advanced took: {time.time() - t_adv:.4f}s")

    t_resp = time.time()
    # Prepare backward-compatible response (provenance strictly from executed recognizer, never script-guessed)
    results = []
    for idx, (line_img, meta) in enumerate(zip(final_crops, lines_metadata)):
        cand_sel = meta['candidates'][meta['selected_crop_type']]
        pred_text = cand_sel['prediction']
        confidence = cand_sel['confidence']

        cal_ver = cand_sel.get('routed_version') or meta.get('routed_version')
        if not cal_ver and model_ver != 'auto':
            cal_ver = resolve_actual_model_version(model_ver)

        primary_rec_part = str(cal_ver).split('+', 1)[0] if cal_ver else ''
        if (cal_ver and cal_ver not in ('vi', 'en', 'fr', 'japan')) and (
            primary_rec_part in ('v24', 'v25', 'v26') or primary_rec_part.startswith('v24')
        ):
            if any('\uAA00' <= ch <= '\uAA5F' for ch in (pred_text or '')):
                try:
                    from scripts.generate_data import normalize_unicode
                    pred_text = normalize_unicode(pred_text)
                except Exception as e:
                    print(f"Warning: normalize_unicode failed: {e}")

        # Encode cropped line to base64 for frontend display
        _, buffer = cv2.imencode('.png', line_img)
        line_b64 = base64.b64encode(buffer).decode('utf-8')

        cal_conf = calibrate_line_confidence(confidence, primary_rec_part or cal_ver, pred_text)

        # Backward-compatible format with explicit per-line recognizer provenance
        results.append({
            'index': idx + 1,
            'text': pred_text,
            'confidence': float(confidence),
            'calibrated_confidence': cal_conf,
            'selected_recognizer': cal_ver,
            'coords': meta['core_box'],
            'bbox': meta.get('bbox', [0, meta['core_box'][0], img.shape[1], meta['core_box'][1]]),
            'polygon': meta.get('polygon'),
            'image': line_b64,
            'segmentation': {
                'overlap_prev_px': meta.get('overlap_prev_px', 0),
                'overlap_next_px': meta.get('overlap_next_px', 0),
                'crop_confidence': meta.get('crop_confidence', 1.0),
                'selected_crop_type': meta.get('selected_crop_type', 'standard'),
                'crop_sanity_score': meta.get('crop_sanity_score', 1.0),
                'needs_review': bool(meta.get('needs_review', False)),
                'reason': meta.get('reason', '')
            }
        })
        print(f"📥 [DO_POST RESULT] Line {idx+1}: text='{pred_text[:15]}...', conf={float(confidence):.4f}, bbox={meta.get('bbox')}")
    print(f"⏱️  Formatted response in: {time.time() - t_resp:.4f}s")

    profiler_dict['method'] = method
    profiler_dict['backend_total_sec'] = round(time.time() - t_start, 4)
    profiler_dict['base_segmentation_sec'] = round(base_segmentation_sec, 4)
    document = build_document_status(results, doc_reasons, doc_extra)
    return {
        'lines': results,
        'profiler': profiler_dict,
        'document': document,
        'model_info': get_model_info(
            model_ver,
            executed_detector=executed_det_name,
            detector_switched=executed_det_switched,
        ),
    }

def save_correction_entry(data, base_dir=None):
    """Persists a verified/corrected text line to legacy ocr_corrections.txt and Stage R4 active learning storage."""
    base_dir = base_dir or os.path.join(PROJECT_ROOT, "data")
    os.makedirs(base_dir, exist_ok=True)

    raw_text = (data.get('text') or '').strip()
    normalized_text = raw_text
    if any('\uAA00' <= ch <= '\uAA5F' for ch in raw_text):
        try:
            from scripts.generate_data import normalize_unicode
            normalized_text = normalize_unicode(raw_text)
        except Exception as e:
            print(f"Warning: normalize_unicode in save_correction_entry failed: {e}")

    dest_file = os.path.join(base_dir, "ocr_corrections.txt")
    with open(dest_file, 'a', encoding='utf-8', newline='\n') as f:
        f.write(normalized_text + '\n')

    al_dir = os.path.join(base_dir, "active_learning")
    crops_dir = os.path.join(al_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    crop_sha256 = None
    crop_relpath = None
    crop_b64 = data.get('crop_image') or ''
    if isinstance(crop_b64, str) and crop_b64.strip():
        if ',' in crop_b64:
            crop_b64 = crop_b64.split(',', 1)[1]
        missing_pad = len(crop_b64) % 4
        if missing_pad:
            crop_b64 += '=' * (4 - missing_pad)
        try:
            raw_bytes = base64.b64decode(crop_b64)
            nparr = np.frombuffer(raw_bytes, np.uint8)
            decoded_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if decoded_img is not None and decoded_img.size > 0:
                ok, png_buf = cv2.imencode('.png', decoded_img)
                if ok:
                    png_bytes = png_buf.tobytes()
                    crop_sha256 = hashlib.sha256(png_bytes).hexdigest()
                    crop_fname = f"{crop_sha256[:16]}.png"
                    crop_abs = os.path.join(crops_dir, crop_fname)
                    if not os.path.exists(crop_abs):
                        with open(crop_abs, 'wb') as cf:
                            cf.write(png_bytes)
                    crop_relpath = f"active_learning/crops/{crop_fname}"
        except Exception as e:
            print(f"Warning: failed to decode active learning crop image: {e}")

    jsonl_file = os.path.join(al_dir, "corrections.jsonl")
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "image_name": str(data.get("image_name") or "unknown.png"),
        "line_idx": data.get("line_idx"),
        "text": normalized_text,
        "raw_submitted_text": raw_text,
        "original_text": data.get("original_text"),
        "confidence": data.get("confidence"),
        "calibrated_confidence": data.get("calibrated_confidence"),
        "model": data.get("model"),
        "bbox": data.get("bbox"),
        "crop_sha256": crop_sha256,
        "crop_relpath": crop_relpath,
    }
    with open(jsonl_file, 'a', encoding='utf-8', newline='\n') as jf:
        jf.write(json.dumps(record, ensure_ascii=False) + '\n')

    return {
        'success': True,
        'path': dest_file,
        'jsonl_path': jsonl_file,
        'crop_relpath': crop_relpath,
        'normalized_text': normalized_text,
    }


# ==============================================================================
# Studio Workbench Helpers (Samples, Model Catalog, Paragraph Flow, ZIP Export)
# ==============================================================================

_SAMPLE_SPECS = [
    {
        "id": "agal_panduranga",
        "name": "Palm-Leaf Agal Folio — Panduranga",
        "filename": "ood_agal_01_panduranga.png",
        "relpath": "ocr-benchmark/data/realscan_gold_v2/pages/ood_agal_01_panduranga.png",
        "category": "Palm-Leaf (Agal)",
        "recommended_model": "v24_1_r4",
        "description": "Traditional Panduranga calligraphic hand on elongated palm-leaf surface with tight interlinear spacing."
    },
    {
        "id": "curved_panduranga",
        "name": "Curved Baseline Folio — Ribbon Rectification",
        "filename": "ood_v3_16_curved_panduranga.png",
        "relpath": "ocr-benchmark/data/realscan_gold_v3/pages/ood_v3_16_curved_panduranga.png",
        "category": "Curved Baseline",
        "recommended_model": "v24_1_r4",
        "description": "Undulating folio baseline demonstrating Cham-DBNet v1_poly_u1.8 polygon ribbon straightening."
    },
    {
        "id": "two_col_wijaya",
        "name": "Two-Column Manuscript Folio — Wijaya",
        "filename": "ood_folio_06_wijaya_2col.png",
        "relpath": "ocr-benchmark/data/realscan_gold_v2/pages/ood_folio_06_wijaya_2col.png",
        "category": "Multi-Column",
        "recommended_model": "v24_1_r4",
        "description": "Dual-column historical layout showcasing automatic column-wise reading order reconstruction."
    },
    {
        "id": "bilingual_campa_vi",
        "name": "Bilingual Cham–Vietnamese Glossary Folio",
        "filename": "ood_v3_21_bilingual_campa_vi.png",
        "relpath": "ocr-benchmark/data/realscan_gold_v3/pages/ood_v3_21_bilingual_campa_vi.png",
        "category": "Bilingual Multiscript",
        "recommended_model": "auto",
        "description": "Interleaved Akhar Thrah and Vietnamese glosses demonstrating the Tri-Specialist Auto-Router."
    },
    {
        "id": "printed_book_semibold",
        "name": "Printed Akhar Thrah Literary Page",
        "filename": "ood_book_10_semibold.png",
        "relpath": "ocr-benchmark/data/realscan_gold_v2/pages/ood_book_10_semibold.png",
        "category": "Printed Book",
        "recommended_model": "v24_1_r4",
        "description": "Modern printed Eastern Cham verse and prose with Danda and Double Danda stanza markers."
    },
    {
        "id": "classical_verse_clean",
        "name": "Classical Verse Stanza Folio (Clean)",
        "filename": "test_006_lvl1.png",
        "relpath": "ocr-benchmark/data/images/test_006_lvl1.png",
        "category": "Classical Verse",
        "recommended_model": "v24_1_r4",
        "description": "Clean baseline folio featuring Cham stanza numerals and multi-directional vowel clusters."
    }
]
_sample_thumb_cache = {}


def _resolve_sample_path(spec):
    bundled_p = os.path.join(PROJECT_ROOT, "static", "samples", spec["filename"])
    if os.path.isfile(bundled_p):
        return bundled_p
    repo_root = os.path.dirname(PROJECT_ROOT)
    fallback_p = os.path.join(repo_root, *spec["relpath"].split("/"))
    if os.path.isfile(fallback_p):
        return fallback_p
    return None


def list_curated_samples():
    available = []
    for spec in _SAMPLE_SPECS:
        abs_p = _resolve_sample_path(spec)
        if not abs_p:
            continue
        thumb_b64 = _sample_thumb_cache.get(spec["id"])
        dims = _sample_thumb_cache.get(spec["id"] + ":dims")
        if thumb_b64 is None:
            try:
                im = cv2.imread(abs_p, cv2.IMREAD_COLOR)
                if im is not None and im.size > 0:
                    ih, iw = im.shape[:2]
                    dims = [int(iw), int(ih)]
                    scale = min(220.0 / max(1, iw), 150.0 / max(1, ih), 1.0)
                    tw, th = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
                    resized = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                    if ok:
                        thumb_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                        _sample_thumb_cache[spec["id"]] = thumb_b64
                        _sample_thumb_cache[spec["id"] + ":dims"] = dims
            except Exception:
                thumb_b64 = ""
        available.append({
            "id": spec["id"],
            "name": spec["name"],
            "filename": spec["filename"],
            "category": spec["category"],
            "recommended_model": spec["recommended_model"],
            "description": spec["description"],
            "dimensions": dims or [0, 0],
            "thumb": thumb_b64 or ""
        })
    return available


def load_curated_sample(sample_id):
    spec = next((s for s in _SAMPLE_SPECS if s["id"] == sample_id), None)
    if not spec:
        return None
    abs_p = _resolve_sample_path(spec)
    if not abs_p:
        return None
    with open(abs_p, "rb") as f:
        raw_bytes = f.read()
    im = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    ih, iw = im.shape[:2] if im is not None else (0, 0)
    return {
        "id": spec["id"],
        "name": spec["name"],
        "filename": spec["filename"],
        "category": spec["category"],
        "recommended_model": spec["recommended_model"],
        "description": spec["description"],
        "width": int(iw),
        "height": int(ih),
        "image": base64.b64encode(raw_bytes).decode("ascii")
    }


def list_available_models():
    catalog = [
        {
            "id": "v24_1_r4",
            "label": "Cham V26.1 Masked-SVTR (Production SOTA — W=480, 6/6 Gates)",
            "short_label": "V26.1 SOTA",
            "group": "Paleographic Cham Specialists",
            "shape": "3, 48, 480",
            "summary": "Padding-masked wide-canvas pure-Cham recognizer with frozen early-backbone WiSE-FT (6/6 Gates, 2.07% real-camera CER, 22.65% OOD-12 E2E).",
            "folder": "rec_cham_inference_v24_1_r4"
        },
        {
            "id": "auto",
            "label": "Auto Tri-Specialist Router (Cham W480 + W320 + Vietnamese)",
            "short_label": "Auto Router",
            "group": "Multiscript & Bilingual",
            "shape": "Adaptive",
            "summary": "Combines W=480 Cham primary, W=320 archival adjudicator, and Vietnamese PP-OCRv5 handoff.",
            "folder": None
        },
        {
            "id": "v24_1_r3_1",
            "label": "Cham V24.1-R3.1 (W=320 Level-5 Degradation Specialist)",
            "short_label": "V24.1-R3.1",
            "group": "Paleographic Cham Specialists",
            "shape": "3, 48, 320",
            "summary": "Compact W=320 specialist tuned for narrow/medium degraded archival crops and gate closure.",
            "folder": "rec_cham_inference_v24_1_r3_1"
        },
        {
            "id": "v24_1_r2_1_6",
            "label": "Cham V24.1-R2.1.6 (Clean HarfBuzz & 14-Symbol Polish)",
            "short_label": "V24.1-R2.1.6",
            "group": "Paleographic Cham Specialists",
            "shape": "3, 48, 320",
            "summary": "100% accuracy across all 14 editorial symbols and multi-context hyphens.",
            "folder": "rec_cham_inference_v24_1_r2_1_6"
        },
        {
            "id": "v25",
            "label": "Cham V25 (40-Epoch Dual-T4x2 Bilingual — Legacy Unmasked)",
            "short_label": "V25 Bilingual (Legacy)",
            "group": "Historical Baselines",
            "shape": "3, 48, 480",
            "deprecated": True,
            "summary": "40-epoch Kaggle Dual-T4x2 162-token bilingual model (90.96% Seq Acc / 0.72% CER on 10K in-domain validation; superseded in E2E production by V26.1 + Auto Router due to unmasked W=480 padding & CTC Latin dispersion).",
            "folder": "rec_cham_inference_v25"
        },
        {
            "id": "v24",
            "label": "Cham V24 (Logical Order Baseline)",
            "short_label": "V24 Baseline",
            "group": "Historical Baselines",
            "shape": "3, 48, 320",
            "summary": "Foundational Logical Order recognizer with Cham numerals (꩐–꩙) and Danda punctuation.",
            "folder": "rec_cham_inference_v24"
        },
        {
            "id": "v23",
            "label": "Cham V23 (Legacy Baseline)",
            "short_label": "V23 Legacy",
            "group": "Historical Baselines",
            "shape": "3, 48, 320",
            "summary": "Early 93-token baseline preserved for longitudinal benchmark comparison.",
            "folder": "rec_cham_inference_v23"
        },
        {
            "id": "vi",
            "label": "Vietnamese (Fine-Tuned PP-OCRv5)",
            "short_label": "Vietnamese",
            "group": "Multiscript & Bilingual",
            "shape": "Dynamic",
            "summary": "High-accuracy Vietnamese & Latin recognizer for modern glosses, headers, and translations.",
            "folder": "rec_vietnamese_infer"
        }
    ]
    out_root = os.path.join(PROJECT_ROOT, "data", "output")
    try:
        import paddle as _pdl
        paddle_major = int(str(getattr(_pdl, "__version__", "0")).split(".")[0])
    except Exception:
        paddle_major = 0
    enriched = []
    for item in catalog:
        folder = item["folder"]
        if folder is None:
            installed = True
        else:
            mdir = os.path.join(out_root, folder)
            installed = os.path.isdir(mdir)
            if installed and item["id"] != "vi":
                has_pdm = os.path.exists(os.path.join(mdir, "inference.pdmodel"))
                has_jsn = os.path.exists(os.path.join(mdir, "inference.json"))
                if has_jsn and not has_pdm and 0 < paddle_major < 3:
                    installed = False
        entry = dict(item)
        entry["installed"] = installed
        entry["is_default"] = (item["id"] == DEFAULT_CHAM_MODEL)
        enriched.append(entry)
    return {"default_model": DEFAULT_CHAM_MODEL, "det_mode": DET_MODE, "models": enriched}


def build_corpus_export_zip(payload):
    import io
    import zipfile

    docs = payload.get("documents") or []
    try:
        from scripts.generate_data import normalize_unicode as _norm_u
    except Exception:
        _norm_u = lambda s: s

    mem_zip = io.BytesIO()
    train_pairs = []
    jsonl_records = []
    md_sections = [
        "# Cham OCR Studio — Paleographic Corpus Export",
        f"- **Generated (UTC)**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Total Folios**: {len(docs)}",
        "- **Service**: https://ocr.cham.asia",
        ""
    ]

    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for d_idx, doc in enumerate(docs, start=1):
            raw_name = str(doc.get("name") or f"folio_{d_idx:03d}.png")
            stem = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(raw_name)[0]).strip("_") or f"folio_{d_idx:03d}"
            lines = doc.get("lines") or []
            model_used = str(doc.get("model") or DEFAULT_CHAM_MODEL)
            status = str(doc.get("status") or "OCR done")

            plain_lines = []
            md_sections.append(f"## {d_idx}. {raw_name}")
            md_sections.append(f"- **Status**: {status} | **Model**: `{model_used}` | **Lines**: {len(lines)}")
            md_sections.append("")

            for l_idx, ln in enumerate(lines, start=1):
                txt = (ln.get("text") or "").strip()
                if any("\uAA00" <= ch <= "\uAA5F" for ch in txt):
                    try:
                        txt = _norm_u(txt)
                    except Exception:
                        pass
                plain_lines.append(txt)
                conf = float(ln.get("confidence") or 0.0)
                verified = bool(ln.get("human_verified"))
                md_sections.append(f"{l_idx}. `{conf * 100:.0f}%`{' [Verified]' if verified else ''} {txt}")

                crop_b64 = ln.get("image") or ""
                crop_rel = None
                if isinstance(crop_b64, str) and crop_b64.strip() and txt:
                    if "," in crop_b64:
                        crop_b64 = crop_b64.split(",", 1)[1]
                    pad = len(crop_b64) % 4
                    if pad:
                        crop_b64 += "=" * (4 - pad)
                    try:
                        crop_bytes = base64.b64decode(crop_b64)
                        crop_rel = f"crops/{stem}_L{l_idx:03d}.png"
                        zf.writestr(crop_rel, crop_bytes)
                        train_pairs.append(f"{crop_rel}\t{txt}")
                    except Exception:
                        crop_rel = None

                jsonl_records.append(json.dumps({
                    "image_name": raw_name,
                    "line_idx": l_idx,
                    "text": txt,
                    "confidence": conf,
                    "human_verified": verified,
                    "model": model_used,
                    "bbox": ln.get("bbox"),
                    "crop_relpath": crop_rel
                }, ensure_ascii=False))

            md_sections.append("")
            zf.writestr(f"transcriptions/{stem}.txt", "\n".join(plain_lines) + "\n")

        zf.writestr("transcriptions/combined_corpus.md", "\n".join(md_sections) + "\n")
        zf.writestr("train_list.txt", "\n".join(train_pairs) + ("\n" if train_pairs else ""))
        zf.writestr("corrections.jsonl", "\n".join(jsonl_records) + ("\n" if jsonl_records else ""))
        zf.writestr("corpus_manifest.json", json.dumps({
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "service": "https://ocr.cham.asia",
            "total_documents": len(docs),
            "total_training_crops": len(train_pairs),
            "documents": [
                {
                    "image_name": d.get("name"),
                    "status": d.get("status"),
                    "model": d.get("model"),
                    "lines": [
                        {
                            "index": ln.get("index", i + 1),
                            "text": ln.get("text", ""),
                            "confidence": ln.get("confidence", 0.0),
                            "bbox": ln.get("bbox"),
                            "coords": ln.get("coords"),
                            "human_verified": bool(ln.get("human_verified"))
                        }
                        for i, ln in enumerate(d.get("lines") or [])
                    ]
                }
                for d in docs
            ]
        }, ensure_ascii=False, indent=2))
        zf.writestr("README.txt", (
            "Cham OCR Studio — Archival & Active Learning Package\n"
            "====================================================\n"
            "Generated by https://ocr.cham.asia\n\n"
            "Contents:\n"
            "- transcriptions/<folio>.txt     : Canonical Unicode Logical Order plaintext per folio\n"
            "- transcriptions/combined_corpus.md : Consolidated scholarly markdown report\n"
            "- corpus_manifest.json           : Structured bounding-box & line metadata\n"
            "- corrections.jsonl              : Active-learning JSONL records\n"
            "- train_list.txt + crops/*.png   : Ready-to-train PaddleOCR tab-delimited pairs\n"
        ))

    return mem_zip.getvalue()


# ==============================================================================
# Web Server Request Handler
# ==============================================================================

_STATIC_MIME_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
}


class ChamOCRRequestHandler(BaseHTTPRequestHandler):
    def _send_html_file(self, filename, no_store=False, head_only=False):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(dir_path, filename)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                payload = f.read()
        else:
            payload = f"<h1>Error: {filename} not found!</h1>".encode('utf-8')
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        if no_store:
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
        else:
            self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        if not head_only:
            self.wfile.write(payload)

    def do_HEAD(self):
        self._handle_get_or_head(head_only=True)

    def do_GET(self):
        self._handle_get_or_head(head_only=False)

    def _handle_get_or_head(self, head_only=False):
        parsed_path = urllib.parse.urlparse(self.path).path
        if parsed_path in ('/', '/index.html'):
            self._send_html_file('index.html', no_store=True, head_only=head_only)
        elif parsed_path in ('/favicon.ico', '/favicon.svg'):
            svg_icon = (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
                '<rect width="64" height="64" rx="14" fill="#C96442"/>'
                '<text x="32" y="45" font-size="36" text-anchor="middle" fill="#FFFCF7" font-family="sans-serif">ꨌ</text>'
                '</svg>'
            ).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-type', 'image/svg+xml')
            self.send_header('Content-Length', str(len(svg_icon)))
            self.send_header('Cache-Control', 'public, max-age=86400')
            self.end_headers()
            if not head_only:
                self.wfile.write(svg_icon)
        elif parsed_path.startswith('/static/'):
            dir_path = os.path.dirname(os.path.abspath(__file__))
            static_root = os.path.realpath(os.path.join(dir_path, 'static'))
            rel_sub = parsed_path[len('/static/'):].lstrip('/\\')
            candidate = os.path.realpath(os.path.join(static_root, rel_sub))
            try:
                common = os.path.commonpath([static_root, candidate])
            except ValueError:
                common = ""
            if common != static_root or not os.path.isfile(candidate):
                self.send_error(404, 'Static Asset Not Found')
                return
            ext = os.path.splitext(candidate)[1].lower()
            mime = _STATIC_MIME_TYPES.get(ext, 'application/octet-stream')
            with open(candidate, 'rb') as sf:
                payload = sf.read()
            self.send_response(200)
            self.send_header('Content-type', mime)
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            if not head_only:
                self.wfile.write(payload)
        elif parsed_path == '/api/samples':
            self.send_json_response({'samples': list_curated_samples()}, head_only=head_only)
        elif parsed_path.startswith('/api/sample/'):
            sample_id = urllib.parse.unquote(parsed_path[len('/api/sample/'):]).strip('/')
            sample_data = load_curated_sample(sample_id)
            if sample_data is None:
                self.send_json_response({'error': f'Sample not found: {sample_id}'}, 404, head_only=head_only)
            else:
                self.send_json_response(sample_data, head_only=head_only)
        elif parsed_path == '/api/models':
            self.send_json_response(list_available_models(), head_only=head_only)
        elif parsed_path in ('/help', '/help.html', '/guide', '/manual', '/docs'):
            self._send_html_file('help.html', no_store=False, head_only=head_only)
        elif parsed_path in ('/research', '/research.html', '/methodology', '/paper'):
            self._send_html_file('research.html', no_store=False, head_only=head_only)
        elif parsed_path in ('/health', '/api/health'):
            body = b"OK"
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            if not head_only:
                self.wfile.write(body)
        else:
            self.send_error(404, 'File Not Found')
            
    def _read_json_object(self, allow_empty=False):
        try:
            content_length = int(self.headers.get('Content-Length') or 0)
        except (TypeError, ValueError):
            self.send_json_response({'error': 'Invalid Content-Length header'}, 400)
            return None
        if content_length <= 0:
            if allow_empty:
                return {}
            self.send_json_response({'error': 'Empty request body (expected JSON object)'}, 400)
            return None
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode('utf-8'))
        except Exception as e:
            self.send_json_response({'error': f'Invalid JSON payload: {e}'}, 400)
            return None
        if not isinstance(data, dict):
            self.send_json_response({'error': 'JSON payload must be an object'}, 400)
            return None
        return data

    def do_POST(self):
        import time
        t_start = time.time()
        print(f"📥 Received POST request: {self.path}...")
        if self.path == '/log':
            content_length = int(self.headers.get('Content-Length') or 0)
            post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'
            try:
                log_data = json.loads(post_data.decode('utf-8'))
                print(f"\n🚨 [FRONTEND LOG] {log_data.get('message')}\n")
            except Exception as e:
                print(f"Failed to parse log: {e}")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
            return

        if self.path == '/ocr':
            data = self._read_json_object()
            if data is None:
                return
            print(f"⏱️  Read request body in: {time.time() - t_start:.4f}s")
            t_parse = time.time()

            # 1. Validate model against strict allowlist (prevents silent fallback / fake provenance)
            try:
                model_ver = validate_requested_model(data.get('model'))
            except UnknownModelError as ume:
                self.send_json_response({'error': str(ume), 'allowed_models': list(ADVERTISED_MODEL_IDS)}, 400)
                return

            # 2. Validate segmentation method, reading order, and numeric parameters explicitly
            method_raw = data.get('method')
            method = 'dbnet' if method_raw is None else str(method_raw).strip().lower()
            if method not in ALLOWED_SEG_METHODS:
                self.send_json_response({
                    'error': f"Invalid segmentation method '{method_raw}'. Allowed methods: {', '.join(ALLOWED_SEG_METHODS)}",
                    'allowed_methods': list(ALLOWED_SEG_METHODS),
                }, 400)
                return

            order_raw = data.get('reading_order')
            reading_order = 'auto' if order_raw is None else str(order_raw).strip().lower()
            if reading_order not in ALLOWED_READING_ORDERS:
                self.send_json_response({
                    'error': f"Invalid reading_order '{order_raw}'. Allowed values: {', '.join(ALLOWED_READING_ORDERS)}",
                    'allowed_reading_orders': list(ALLOWED_READING_ORDERS),
                }, 400)
                return

            try:
                threshold = float(data.get('threshold', 0.05))
                gap = int(data.get('gap', 12))
                win = int(data.get('window', 25))
            except (TypeError, ValueError) as pe:
                self.send_json_response({'error': f'Invalid numeric segmentation parameter: {pe}'}, 400)
                return

            # 3. Validate image field presence before base64/OpenCV decoding (eliminates KeyError -> HTTP 502)
            if 'image' not in data or not isinstance(data.get('image'), str) or not data['image'].strip():
                self.send_json_response({'error': "Missing or empty 'image' field (expected base64-encoded image)"}, 400)
                return
            img_b64 = data['image'].strip()
            
            # Decode image
            if ',' in img_b64:
                img_b64 = img_b64.split(',', 1)[1]
            missing_padding = len(img_b64) % 4
            if missing_padding:
                img_b64 += '=' * (4 - missing_padding)
            try:
                img_data = base64.b64decode(img_b64, validate=False)
                if not img_data:
                    self.send_json_response({'error': 'Decoded image payload is empty'}, 400)
                    return
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"❌ Error decoding image: {e}")
                self.send_json_response({'error': f'Unable to decode image data: {e}'}, 400)
                return
                
            print(f"⏱️  Parsed parameters and decoded image in: {time.time() - t_parse:.4f}s")
            
            if img is None:
                self.send_json_response({'error': 'Unable to read image (unsupported format or corrupted base64 payload)'}, 400)
                return
                
            # Serialize Paddle C++ model loading & inference safely while keeping HTTP GET/health non-blocking
            with _INFERENCE_LOCK:
                t_model = time.time()
                try:
                    get_ocr_model(model_ver)
                except UnknownModelError as ume:
                    self.send_json_response({'error': str(ume), 'allowed_models': list(ADVERTISED_MODEL_IDS)}, 400)
                    return
                except Exception as e:
                    import traceback
                    print(f"❌ Error loading OCR model '{model_ver}': {traceback.format_exc()}")
                    self.send_json_response({'error': f'Model loading failure ({model_ver}): {e}'}, 500)
                    return
                print(f"⏱️  Loaded model in: {time.time() - t_model:.4f}s")

                try:
                    response = run_ocr_pipeline(img, model_ver=model_ver, method=method, threshold=threshold,
                                                gap=gap, win=win, reading_order=reading_order, t_start=t_start)
                except Exception as e:
                    import traceback
                    print(f"❌ Error during OCR pipeline: {traceback.format_exc()}")
                    self.send_json_response({'error': f'OCR transcription engine error: {e}'}, 500)
                    return

            t_send = time.time()
            self.send_json_response(response)
            print(f"⏱️  Sent response in: {time.time() - t_send:.4f}s")
            print(f"🎉 Total POST /ocr execution time: {response['profiler']['backend_total_sec']:.4f}s (document status: {response['document']['status']})")
            
        elif self.path == '/save_correction':
            data = self._read_json_object()
            if data is None:
                return
            with _CORRECTION_LOCK:
                result = save_correction_entry(data)
            self.send_json_response(result)
        elif self.path == '/ocr_crop':
            data = self._read_json_object()
            if data is None:
                return

            try:
                model_ver = validate_requested_model(data.get('model'))
            except UnknownModelError as ume:
                self.send_json_response({'error': str(ume), 'allowed_models': list(ADVERTISED_MODEL_IDS)}, 400)
                return

            if 'image' not in data or not isinstance(data.get('image'), str) or not data['image'].strip():
                self.send_json_response({'error': "Missing or empty 'image' field (expected base64-encoded image)"}, 400)
                return
            img_b64 = data['image'].strip()

            raw_bbox = data.get('bbox')
            if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
                self.send_json_response({'error': "Invalid 'bbox': expected a 4-element array [x1, y1, x2, y2]"}, 400)
                return
            try:
                rx1, ry1, rx2, ry2 = [float(v) for v in raw_bbox]
            except (TypeError, ValueError):
                self.send_json_response({'error': "Invalid 'bbox': coordinates must be numbers"}, 400)
                return
            if not (math.isfinite(rx1) and math.isfinite(ry1) and math.isfinite(rx2) and math.isfinite(ry2)):
                self.send_json_response({'error': "Invalid 'bbox': coordinates must be finite numbers"}, 400)
                return
            if rx2 <= rx1 or ry2 <= ry1:
                self.send_json_response({'error': f"Invalid 'bbox' {list(raw_bbox)}: must satisfy x2 > x1 and y2 > y1"}, 400)
                return
            
            if ',' in img_b64:
                img_b64 = img_b64.split(',', 1)[1]
            missing_padding = len(img_b64) % 4
            if missing_padding:
                img_b64 += '=' * (4 - missing_padding)
            try:
                img_data = base64.b64decode(img_b64, validate=False)
                if not img_data:
                    self.send_json_response({'error': 'Decoded image payload is empty'}, 400)
                    return
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                self.send_json_response({'error': f'Unable to decode image data: {e}'}, 400)
                return
                
            if img is None:
                self.send_json_response({'error': 'Unable to read image'}, 400)
                return
                
            h, w = img.shape[:2]
            x1 = max(0, int(math.floor(rx1)))
            y1 = max(0, int(math.floor(ry1)))
            x2 = min(w, int(math.ceil(rx2)))
            y2 = min(h, int(math.ceil(ry2)))
            if x2 <= x1 or y2 <= y1:
                self.send_json_response({
                    'error': f"Invalid 'bbox' {list(raw_bbox)}: region lies outside image bounds ({w}x{h})"
                }, 400)
                return
            
            crop_img = img[y1:y2, x1:x2]
            if crop_img.size == 0:
                self.send_json_response({'error': 'Empty cropped region'}, 400)
                return
                
            try:
                with _INFERENCE_LOCK:
                    ocr = get_ocr_model(model_ver)
                    res, sel_vers, _ = invoke_recognizer_with_provenance(
                        ocr, [preprocess_crop_for_rec(crop_img)], model_ver=model_ver
                    )
                pred_text, conf = res[0] if res else ("", 0.0)
            except UnknownModelError as ume:
                self.send_json_response({'error': str(ume), 'allowed_models': list(ADVERTISED_MODEL_IDS)}, 400)
                return
            except Exception as e:
                self.send_json_response({'error': f'Cropped OCR transcription error ({model_ver}): {e}'}, 500)
                return

            sel_ver = sel_vers[0] if sel_vers else (
                resolve_actual_model_version(model_ver) if model_ver != 'auto' else None
            )

            if (sel_ver and sel_ver not in ('vi', 'en', 'fr', 'japan')) and (
                sel_ver in ('v24', 'v25', 'v26') or str(sel_ver).startswith('v24')
            ):
                if any('\uAA00' <= ch <= '\uAA5F' for ch in (pred_text or '')):
                    try:
                        from scripts.generate_data import normalize_unicode
                        pred_text = normalize_unicode(pred_text)
                    except Exception as e:
                        print(f"Warning: normalize_unicode failed: {e}")
                    
            _, buffer = cv2.imencode('.png', crop_img)
            crop_b64 = base64.b64encode(buffer).decode('utf-8')
            
            self.send_json_response({
                'text': pred_text,
                'confidence': float(conf),
                'calibrated_confidence': calibrate_line_confidence(conf, sel_ver, pred_text),
                'selected_recognizer': sel_ver,
                'model_info': get_model_info(model_ver),
                'image': crop_b64,
                'crop_image': f"data:image/png;base64,{crop_b64}",
                'bbox': [x1, y1, x2, y2]
            })
        elif self.path == '/reconstruct_paragraphs':
            content_length = int(self.headers.get('Content-Length') or 0)
            post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'
            try:
                data = json.loads(post_data.decode('utf-8'))
                lines_in = data.get('lines') or []
                from paragraph_flow import reconstruct_paragraph_flow
                paragraphs, boundaries = reconstruct_paragraph_flow(lines_in)
                self.send_json_response({'paragraphs': paragraphs, 'boundaries': boundaries})
            except Exception as e:
                self.send_json_response({'error': f'Paragraph reconstruction failed: {e}', 'paragraphs': [], 'boundaries': []}, 200)
        elif self.path == '/normalize_text':
            content_length = int(self.headers.get('Content-Length') or 0)
            post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'
            try:
                data = json.loads(post_data.decode('utf-8'))
                raw_text = str(data.get('text') or '')
                norm_text = raw_text
                if any('\uAA00' <= ch <= '\uAA5F' for ch in raw_text):
                    from scripts.generate_data import normalize_unicode
                    norm_text = normalize_unicode(raw_text)
                self.send_json_response({'normalized_text': norm_text, 'changed': (norm_text != raw_text)})
            except Exception as e:
                self.send_json_response({'error': str(e), 'normalized_text': data.get('text', ''), 'changed': False}, 200)
        elif self.path == '/export_corpus_zip':
            content_length = int(self.headers.get('Content-Length') or 0)
            post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'
            try:
                data = json.loads(post_data.decode('utf-8'))
                zip_bytes = build_corpus_export_zip(data)
                self.send_response(200)
                self.send_header('Content-type', 'application/zip')
                self.send_header('Content-Disposition', 'attachment; filename="cham_ocr_corpus_export.zip"')
                self.send_header('Content-Length', str(len(zip_bytes)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(zip_bytes)
            except Exception as e:
                self.send_json_response({'error': f'Failed to build corpus ZIP: {e}'}, 500)
        else:
            self.send_error(404, 'Endpoint Not Found')
            
    def send_json_response(self, data, status=200, head_only=False):
        payload = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        if not head_only:
            self.wfile.write(payload)

# ==============================================================================
# HTML Dashboard Frontend Template
# ==============================================================================

HTML_TEMPLATE = ""
def run_server():
    env_port = int(os.getenv("PORT", "7860"))
    preferred_ports = [env_port, 7860, 8080, 8081, 8082, 8501, 8888, 0]
    # Remove duplicate ports while preserving order
    seen = set()
    preferred_ports = [p for p in preferred_ports if not (p in seen or seen.add(p))]
    
    httpd = None
    actual_port = None
    for port in preferred_ports:
        try:
            server_address = ('0.0.0.0', port)
            httpd = StudioHTTPServer(server_address, ChamOCRRequestHandler)
            actual_port = httpd.server_address[1]
            break
        except OSError:
            print(f"⚠️  Port {port} is occupied, trying next...")
            continue
            
    if httpd is None:
        print("❌ Could not bind to any port.")
        return
        
    print(f"======================================================================")
    print(f"🚀 Cham OCR Diagnostic Studio is running at: http://localhost:{actual_port}")
    print(f"📁 Corrections will be saved to: ocr-studio/data/ocr_corrections.txt")
    print(f"======================================================================")
    
    for _preload_ver in dict.fromkeys([DEFAULT_CHAM_MODEL, 'v24_1_r3_1']):
        try:
            get_ocr_model(_preload_ver)
        except Exception as e:
            print(f"⚠️  Could not pre-load model {_preload_ver}: {e}. It will load when requested.")
    try:
        get_det_model()  # in DET_MODE='hybrid' this also pre-loads Cham-DBNet
    except Exception as e:
        print(f"⚠️  Could not pre-load the DBNet detector: {e}. It will load when requested.")
    try:
        list_curated_samples()  # Pre-warm specimen thumbnails at startup
    except Exception:
        pass
        
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == '__main__':
    run_server()
