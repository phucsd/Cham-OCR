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

from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse

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
det_model = None

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

class AutoRoutingOCRWrapper:
    def __init__(self, cham_model, viet_model):
        self.cham_model = cham_model
        self.viet_model = viet_model
        
    def __call__(self, img_list):
        if not img_list:
            return [], 0.0
            
        # 1. Predict all images with Cham model in a single batch
        print(f"DEBUG: [AutoRouting] Calling Cham model for {len(img_list)} images...")
        cham_res, _ = self.cham_model(img_list)
        print("DEBUG: [AutoRouting] Cham model prediction finished.")
        
        # 2. Identify indices requiring Vietnamese OCR
        viet_indices = []
        viet_imgs = []
        for i, (cham_text, cham_conf) in enumerate(cham_res):
            if cham_conf < 0.65:
                viet_indices.append(i)
                viet_imgs.append(img_list[i])
                
        # 3. Predict those with Vietnamese model in a single batch
        viet_res = []
        if viet_imgs:
            print(f"DEBUG: [AutoRouting] Calling Viet model for {len(viet_imgs)} images...")
            viet_batch_res, _ = self.viet_model(viet_imgs)
            viet_res = viet_batch_res
            print("DEBUG: [AutoRouting] Viet model prediction finished.")
            
        # 4. Merge results keeping router logic
        final_results = [None] * len(img_list)
        viet_ptr = 0
        for i in range(len(img_list)):
            cham_text, cham_conf = cham_res[i]
            if cham_conf >= 0.65:
                final_results[i] = (cham_text, cham_conf)
            else:
                if viet_ptr < len(viet_res):
                    viet_text, viet_conf = viet_res[viet_ptr]
                else:
                    viet_text, viet_conf = ("", 0.0)
                viet_ptr += 1
                
                if cham_conf >= 0.40 and cham_conf > viet_conf - 0.15:
                    final_results[i] = (cham_text, cham_conf)
                else:
                    final_results[i] = (viet_text, viet_conf)
                    
        return final_results, 0.0

def get_ocr_model(version):
    """Loads and caches the PaddleOCR model for the given version dynamically"""
    global ocr_models
    if version in ocr_models:
        return ocr_models[version]
        
    if version in ['vi', 'en', 'fr', 'japan']:
        ocr_model = OfficialPaddleOCRWrapper(lang=version)
        ocr_models[version] = ocr_model
        return ocr_model
        
    if version == 'auto':
        cham_model = get_ocr_model('v24')
        viet_model = get_ocr_model('vi')
        ocr_model = AutoRoutingOCRWrapper(cham_model, viet_model)
        ocr_models[version] = ocr_model
        return ocr_model
        
    from tools.infer.predict_rec import TextRecognizer
    import tools.infer.utility as utility
    
    model_dir = os.path.join(PROJECT_ROOT, "data", "output", f"rec_cham_inference_{version}")
    dict_path = os.path.join(PROJECT_ROOT, "data", f"cham_dict_{version}.txt")
    
    if not os.path.exists(model_dir):
        # Fallback to Golden Baseline v23 if version-specific folder is missing
        model_dir = os.path.join(PROJECT_ROOT, "data", "output", "rec_cham_inference_v23")
    if not os.path.exists(dict_path):
        dict_path = os.path.join(PROJECT_ROOT, "data", "cham_dict_v23.txt")
        
    sys_argv_backup = sys.argv
    sys.argv = [sys.argv[0]]
    args = utility.parse_args()
    sys.argv = sys_argv_backup
    
    args.rec_model_dir = model_dir
    args.rec_char_dict_path = dict_path
    
    import paddle
    args.use_gpu = paddle.is_compiled_with_cuda()
    args.rec_image_shape = "3, 48, 320"
    args.use_space_char = True
    
    # Batch size config: default to 1 on CPU to prevent padding latency blowup from varying crop aspect ratios
    env_batch = os.getenv("OCR_REC_BATCH_SIZE")
    if env_batch:
        ocr_rec_batch_size = int(env_batch)
    else:
        ocr_rec_batch_size = 1 if not paddle.is_compiled_with_cuda() else 16
        
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
        
        env_threads = os.getenv("CPU_THREADS")
        if env_threads:
            args.cpu_threads = int(env_threads)
        else:
            # Default to 1 thread to prevent OpenMP deadlocks on Windows socket servers
            args.cpu_threads = 1
            
        args.ir_optim = False # Disable on CPU to prevent OneDNN crash
        args.precision = "fp32"
        
    print(f"⚙️  Loading PaddleOCR Model {version} (GPU: {args.use_gpu}, MKLDNN: {args.enable_mkldnn}, Batch: {args.rec_batch_num}, Threads: {args.cpu_threads})...")
    ocr_model = TextRecognizer(args)
    
    # Warmup the model with a dummy image to trigger compilation/allocator caching
    try:
        dummy_img = np.zeros((48, 320, 3), dtype=np.uint8)
        ocr_model([dummy_img])
        print(f"🔥 Model {version} warmed up successfully.")
    except Exception as e:
        print(f"Failed to warmup model: {e}")
        
    ocr_models[version] = ocr_model
    return ocr_model

def get_det_model():
    """Loads and caches the PaddleOCR TextDetector model for DBNet segmentation."""
    global det_model
    if det_model is not None:
        return det_model

    from tools.infer.predict_det import TextDetector
    import tools.infer.utility as utility

    cham_det_dir = os.path.join(PROJECT_ROOT, "data", "output", "ch_PP-OCRv4_det_cham_infer")
    has_cham_model = os.path.exists(os.path.join(cham_det_dir, "inference.pdmodel")) or os.path.exists(os.path.join(cham_det_dir, "inference.json"))
    if has_cham_model:
        det_model_dir = cham_det_dir
        print(f"🌟 Using dedicated Cham Text Detection model: {det_model_dir}")
    else:
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

    sys_argv_backup = sys.argv
    sys.argv = [sys.argv[0]]
    args = utility.parse_args()
    sys.argv = sys_argv_backup

    args.use_gpu = False
    args.enable_mkldnn = False
    args.ir_optim = False
    args.det_algorithm = 'DB'
    args.det_model_dir = det_model_dir
    args.det_db_thresh = 0.25
    args.det_db_box_thresh = 0.5
    args.det_db_unclip_ratio = 1.8

    print(f"⚙️  Loading PaddleOCR DBNet Detector from {det_model_dir}...")
    det_model = TextDetector(args)
    print("🔥 PaddleOCR DBNet Detector loaded successfully.")
    return det_model

def merge_line_boxes(boxes, img_w, img_h):
    """
    Merges horizontally separated bounding boxes on the same line into a single unified line crop.
    Prevents splitting a single line into separate fragments across distant words.
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
                min_h = min(h1, h2)
                if min_h <= 0:
                    continue

                v_inter = max(0.0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
                v_overlap = v_inter / min_h

                if r1[2] <= r2[0]:
                    x_dist = r2[0] - r1[2]
                elif r2[2] <= r1[0]:
                    x_dist = r1[0] - r2[2]
                else:
                    x_dist = 0.0

                if v_overlap >= 0.50 and x_dist <= max(40.0, 2.5 * min_h):
                    new_x1 = max(0.0, min(r1[0], r2[0]))
                    new_y1 = max(0.0, min(r1[1], r2[1]))
                    new_x2 = min(float(img_w), max(r1[2], r2[2]))
                    new_y2 = min(float(img_h), max(r1[3], r2[3]))
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

    # Line sorting: sort by vertical center
    rects = sorted(rects, key=lambda r: (r[1] + r[3]) / 2.0)
    return [r[4] for r in rects]

def segment_lines_dbnet(img, det_model):
    """
    Performs deep-learning based text line segmentation using PaddleOCR DBNet.
    Returns:
        final_crops: list of cropped BGR images corresponding to each detected line
        lines_metadata: list of metadata dicts matching the Studio response schema
    """
    import tools.infer.utility as utility
    from tools.infer.predict_system import sorted_boxes

    dt_boxes, elapse = det_model(img)
    if dt_boxes is None or len(dt_boxes) == 0:
        return [], []

    h, w = img.shape[:2]
    # Merge horizontal box fragments on the same line
    dt_boxes = merge_line_boxes(dt_boxes, w, h)

    final_crops = []
    lines_metadata = []

    for idx, box in enumerate(dt_boxes):
        pts = np.array(box, dtype=np.float32)
        x_min = int(max(0, np.min(pts[:, 0])))
        x_max = int(min(w, np.max(pts[:, 0])))
        y_min = int(max(0, np.min(pts[:, 1])))
        y_max = int(min(h, np.max(pts[:, 1])))

        # Get perspective crop
        crop_img = utility.get_rotate_crop_image(img, pts)
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
            'candidates': {
                'dbnet': {
                    'prediction': '',
                    'confidence': 0.0
                }
            }
        }
        lines_metadata.append(meta)

    return final_crops, lines_metadata

# ==============================================================================
# Line Segmentation Algorithms
# ==============================================================================

def segment_lines_adaptive(img, threshold_pct=0.08, gap_threshold=12):
    """Adaptive Projection Profile line segmentation"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        cropped_lines.append(img[s_pad:e_pad, :])
        coords.append((s_pad, e_pad))
    return cropped_lines, coords

def segment_lines_valleys(img, window_size=25, min_dist=35):
    """Valley Detection (local minima) line segmentation"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        cropped_lines.append(img[s:e, :])
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

def segment_lines_advanced(img, base_coords, ocr_model):
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
        imgs = [c["crop_img"] for c in pass1_candidates]
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
                results, _ = ocr_model(valid_imgs)
                print("DEBUG: [segment_lines_advanced] Pass 1 prediction finished.")
                for vi, res in zip(valid_indices, results):
                    pred_text, conf = res if res else ("", 0.0)
                    pred_text = pred_text.strip()
                    pass1_candidates[vi]["prediction"] = pred_text
                    pass1_candidates[vi]["confidence"] = float(conf)
                    pass1_candidates[vi]["pred_length"] = len(pred_text)
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

    # Run Batch Pass 2 with coordinate-level deduplication
    t_pass2_rec = time.time()
    if pass2_candidates:
        # Cache for coordinate-level deduplication: (s, e) -> (prediction, confidence)
        seen_boxes = {}
        
        # Pre-populate seen_boxes with Pass 1 (safe) results to avoid re-running identical boxes
        for line_meta in lines_meta:
            if "safe" in line_meta["candidates"]:
                safe_c = line_meta["candidates"]["safe"]
                seen_boxes[tuple(safe_c["crop_box"])] = (safe_c["prediction"], safe_c["confidence"])
                
        exec_candidates = [] # candidates that require actual OCR execution
        deferred_copies = [] # candidates that will copy results from an identical box
        
        for cand in pass2_candidates:
            c_box = tuple(cand["crop_box"])
            if c_box in seen_boxes:
                res = seen_boxes[c_box]
                if res[0] != "" or res[1] > 0.0:
                    cand["prediction"] = res[0]
                    cand["confidence"] = res[1]
                    cand["pred_length"] = len(res[0])
                else:
                    deferred_copies.append((cand, c_box))
            else:
                exec_candidates.append(cand)
                seen_boxes[c_box] = ("", 0.0) # mark as pending
                
        if exec_candidates:
            imgs = [c["crop_img"] for c in exec_candidates]
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
                    results, _ = ocr_model(valid_imgs)
                    print("DEBUG: [segment_lines_advanced] Pass 2 prediction finished.")
                    for vi, res in zip(valid_indices, results):
                        pred_text, conf = res if res else ("", 0.0)
                        pred_text = pred_text.strip()
                        
                        exec_cand = exec_candidates[vi]
                        exec_cand["prediction"] = pred_text
                        exec_cand["confidence"] = float(conf)
                        exec_cand["pred_length"] = len(pred_text)
                        
                        seen_boxes[tuple(exec_cand["crop_box"])] = (pred_text, float(conf))
                except Exception as e:
                    print(f"DEBUG: [segment_lines_advanced] Batch inference pass 2 failed: {e}")
                    
        # Process deferred copies
        for cand, c_box in deferred_copies:
            pred_text, conf = seen_boxes[c_box]
            cand["prediction"] = pred_text
            cand["confidence"] = conf
            cand["pred_length"] = len(pred_text)
            
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
            
    # Safeguard: If all lines were filtered, preserve the single highest-confidence candidate line
    if not filtered_lines_meta and lines_meta:
        best_i = max(range(len(lines_meta)), key=lambda i: lines_meta[i]["candidates"].get(lines_meta[i].get("selected_crop_type", "safe"), {}).get("confidence", 0.0))
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
# Web Server Request Handler
# ==============================================================================

class ChamOCRRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path).path
        if parsed_path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            
            dir_path = os.path.dirname(os.path.abspath(__file__))
            html_path = os.path.join(dir_path, 'index.html')
            if os.path.exists(html_path):
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            else:
                html_content = "<h1>Error: index.html not found!</h1>"
                
            self.wfile.write(html_content.encode('utf-8'))
        else:
            self.send_error(404, 'File Not Found')
            
    def do_POST(self):
        import time
        t_start = time.time()
        print(f"📥 Received POST request: {self.path}...")
        if self.path == '/log':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
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
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            print(f"⏱️  Read request body in: {time.time() - t_start:.4f}s")
            t_parse = time.time()
            data = json.loads(post_data.decode('utf-8'))
            
            # Extract parameters
            img_b64 = data['image']
            model_ver = data.get('model', 'v24')
            method = data.get('method', 'dbnet')
            threshold = float(data.get('threshold', 0.05))
            gap = int(data.get('gap', 12))
            win = int(data.get('window', 25))
            
            # Decode image
            if ',' in img_b64:
                img_b64 = img_b64.split(',', 1)[1]
            missing_padding = len(img_b64) % 4
            if missing_padding:
                img_b64 += '=' * (4 - missing_padding)
            try:
                img_data = base64.b64decode(img_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"❌ Error decoding image: {e}")
                self.send_json_response({'error': f'Unable to decode image data: {e}'}, 400)
                return
                
            print(f"⏱️  Parsed parameters and decoded image in: {time.time() - t_parse:.4f}s")
            
            if img is None:
                self.send_json_response({'error': 'Unable to read image (corrupted base64 payload)'}, 400)
                return
                
            # Perform segmentation
            t_model = time.time()
            try:
                ocr = get_ocr_model(model_ver)
            except Exception as e:
                import traceback
                print(f"❌ Error loading OCR model '{model_ver}': {traceback.format_exc()}")
                self.send_json_response({'error': f'Model loading failure ({model_ver}): {e}'}, 200)
                return
            print(f"⏱️  Loaded model in: {time.time() - t_model:.4f}s")
                
            t_seg = time.time()
            base_segmentation_sec = 0.0
            profiler_dict = {
                'method': method,
                'base_segmentation_sec': 0.0,
                'legacy_fast_pass_count': 0,
                'pass2_rescue_count': 0
            }
            
            if method == 'dbnet':
                print("🔍 Running PaddleOCR DBNet segmentation...")
                try:
                    det = get_det_model()
                    final_crops, lines_metadata = segment_lines_dbnet(img, det)
                except Exception as e:
                    import traceback
                    print(f"❌ DBNet segmentation error: {traceback.format_exc()}")
                    final_crops, lines_metadata = [], []
                
                base_segmentation_sec = time.time() - t_seg
                if not final_crops:
                    print("⚠️ DBNet found 0 text lines, falling back to Valley segmentation...")
                    _, coords = segment_lines_valleys(img, window_size=win, min_dist=gap)
                    final_crops, lines_metadata, profiler_dict = segment_lines_advanced(img, coords, ocr)
                else:
                    t_rec = time.time()
                    rec_res, _ = ocr(final_crops)
                    rec_sec = time.time() - t_rec
                    print(f"⏱️  DBNet batch recognition ({len(final_crops)} lines) took: {rec_sec:.4f}s")
                    for idx, (pred_text, conf) in enumerate(rec_res):
                        lines_metadata[idx]['candidates']['dbnet']['prediction'] = pred_text
                        lines_metadata[idx]['candidates']['dbnet']['confidence'] = conf
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
                final_crops, lines_metadata, profiler_dict = segment_lines_advanced(img, coords, ocr)
                print(f"⏱️  segment_lines_advanced took: {time.time() - t_adv:.4f}s")
                
            t_resp = time.time()
            # Prepare backward-compatible response
            results = []
            try:
                for idx, (line_img, meta) in enumerate(zip(final_crops, lines_metadata)):
                    pred_text = meta['candidates'][meta['selected_crop_type']]['prediction']
                    confidence = meta['candidates'][meta['selected_crop_type']]['confidence']
                    
                    if model_ver in ['v24', 'v26']:
                        try:
                            from scripts.generate_data import normalize_unicode
                            pred_text = normalize_unicode(pred_text)
                        except Exception as e:
                            print(f"Warning: normalize_unicode failed: {e}")
                        
                    # Encode cropped line to base64 for frontend display
                    _, buffer = cv2.imencode('.png', line_img)
                    line_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Backward-compatible format
                    print(f"📥 [DO_POST RAW META] line_id={meta.get('line_id')}, bbox={meta.get('bbox')}, keys={list(meta.keys())}")
                    results.append({
                        'index': idx + 1,
                        'text': pred_text,
                        'confidence': float(confidence),
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
                            'needs_review': meta.get('needs_review', False),
                            'reason': meta.get('reason', '')
                        }
                    })
                    print(f"📥 [DO_POST RESULT] Line {meta['line_id']+1}: text='{pred_text[:15]}...', conf={confidence:.4f}, bbox={meta.get('bbox')}")
            except Exception as e:
                import traceback
                print(f"❌ Error during OCR formatting: {traceback.format_exc()}")
                self.send_json_response({'error': f'OCR transcription engine error: {e}'}, 200)
                return
            print(f"⏱️  Formatted response in: {time.time() - t_resp:.4f}s")
            
            t_send = time.time()
            profiler_dict['backend_total_sec'] = round(time.time() - t_start, 4)
            profiler_dict['base_segmentation_sec'] = round(base_segmentation_sec, 4)
            self.send_json_response({'lines': results, 'profiler': profiler_dict})
            print(f"⏱️  Sent response in: {time.time() - t_send:.4f}s")
            print(f"🎉 Total POST /ocr execution time: {profiler_dict['backend_total_sec']:.4f}s")
            
        elif self.path == '/save_correction':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            text = data.get('text', '')
            dest_file = os.path.join(PROJECT_ROOT, "data", "ocr_corrections.txt")
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            
            with open(dest_file, 'a', encoding='utf-8', newline='\n') as f:
                f.write(text.strip() + '\n')
                
            self.send_json_response({'success': True, 'path': dest_file})
        elif self.path == '/ocr_crop':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            img_b64 = data.get('image', '')
            model_ver = data.get('model', 'v24')
            bbox = data.get('bbox', [0, 0, 0, 0])
            
            if ',' in img_b64:
                img_b64 = img_b64.split(',', 1)[1]
            missing_padding = len(img_b64) % 4
            if missing_padding:
                img_b64 += '=' * (4 - missing_padding)
            try:
                img_data = base64.b64decode(img_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                self.send_json_response({'error': f'Unable to decode image data: {e}'}, 400)
                return
                
            if img is None:
                self.send_json_response({'error': 'Unable to read image'}, 400)
                return
                
            h, w = img.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, min(w - 1, x1))
            x2 = max(x1 + 1, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(y1 + 1, min(h, y2))
            
            crop_img = img[y1:y2, x1:x2]
            if crop_img.size == 0:
                self.send_json_response({'error': 'Empty cropped region'}, 400)
                return
                
            try:
                ocr = get_ocr_model(model_ver)
                res, _ = ocr([crop_img])
                pred_text, conf = res[0] if res else ("", 0.0)
            except Exception as e:
                self.send_json_response({'error': f'Cropped OCR transcription error ({model_ver}): {e}'}, 200)
                return
                
            if model_ver in ['v24', 'v26']:
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
                'image': crop_b64,
                'bbox': [x1, y1, x2, y2]
            })
        else:
            self.send_error(404, 'Endpoint Not Found')
            
    def send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        # Allow Cross-Origin for flexibility
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

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
            httpd = HTTPServer(server_address, ChamOCRRequestHandler)
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
    
    try:
        get_ocr_model('v23')
    except Exception as e:
        print(f"⚠️  Could not pre-load model v23: {e}. It will load when requested.")
        
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == '__main__':
    run_server()
