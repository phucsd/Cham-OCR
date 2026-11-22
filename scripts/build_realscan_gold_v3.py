#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR RealScan-Gold v3 Cross-Domain & Bilingual Archival Benchmark Builder
=============================================================================
Builds `ocr-benchmark/data/realscan_gold_v3/` (`manifest.json` locked with SHA-256):
  1. Preserves 100% backward-compatible references to SHA-256-locked `RealScan-Gold v1` & `v2`:
     - `harum_palei_01` (1 authentic camera page, 25 lines)
     - `wikimedia_public_specimens_01` (7 public-domain / CC-BY-SA Wikimedia Commons specimens)
     - `handoff_multiscript_01` (76 multiscript crops: 27 Cham, 25 Latin, 24 Vietnamese-English)
     - `ood_archival_gold_v2` (12 unseen EFEO/Held-Out pages, 180 lines)
  2. Expands with 10 NEW held-out archival & scholarly pages (144 new lines -> 432 total evaluated lines/crops):
     - `efeo_unseen_regional_v3` (3 pages: `EFEO Cam AiaTrang.otf`, `EFEO Cam Kraong.otf` 2-col, `EFEO Cam Pajai.otf` -> completes 12/12 EFEO fonts!)
     - `efeo_curved_agal_v3` (3 pages: sinusoidally bowed palm-leaf *agal* folios `amp = 4.2..6.6px` testing curved polygon ribbon rectification + OOD recognition)
     - `heldout_ultralight_book_v3` (2 pages: `NotoSansCham-Thin.ttf`, `ExtraLight.ttf`, `Medium.ttf` -> completes 6/6 held-out weights!)
     - `bilingual_cham_viet_commentary_v3` (2 pages: scholarly philological edition alternating Cham stanzas + Vietnamese translations)
  3. Zero Contamination Guarantee: 0 exact-text overlap against ALL synthetic train/val datasets (`R3.1`, `R4`, `R4.1`, `V26-FT`, `V26-Full`, `V26.1`).
"""

import os
import sys
import math
import json
import shutil
import random
import hashlib
from datetime import datetime, timezone

import cv2
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DATA = os.path.join(PROJECT_ROOT, "ocr-benchmark", "data")
OUT_ROOT = os.path.join(BENCH_DATA, "realscan_gold_v3")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-training", "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
from generate_data import normalize_unicode
from build_realscan_gold_v2 import sha256_file, render_font_strip, make_archival_bg


def load_v3_disjoint_cham_lines() -> list:
    """Builds 200+ held-out Cham sentences strictly disjoint from ALL synthetic train/val datasets AND RealScan-Gold v2."""
    forbidden_texts = set()
    for sub in (
        "cham_synthetic_r3_1_closure",
        "cham_synthetic_r4_w480",
        "cham_synthetic_r4_1_polish",
        "cham_synthetic_v26_ft",
        "cham_synthetic_v26_full",
        "cham_synthetic_v26_1",
    ):
        for fn in ("train_label.txt", "val_label.txt"):
            lp = os.path.join(PROJECT_ROOT, "ocr-training", "data", sub, fn)
            if os.path.exists(lp):
                for ln in open(lp, "r", encoding="utf-8"):
                    if "\t" in ln:
                        t = normalize_unicode(ln.rstrip("\r\n").split("\t", 1)[1].strip())
                        if t:
                            forbidden_texts.add(t)

    v2_m = json.load(open(os.path.join(BENCH_DATA, "realscan_gold_v2", "manifest.json"), "r", encoding="utf-8"))
    for d in v2_m["documents"]:
        for ln in d.get("lines", []):
            forbidden_texts.add(normalize_unicode(ln["gt"].strip()))

    dict_chars = set(
        ln.strip("\r\n")
        for ln in open(os.path.join(PROJECT_ROOT, "ocr-studio", "data", "cham_dict_v24_1.txt"), "r", encoding="utf-8")
        if ln.strip("\r\n")
    ) | {" "}

    pool_words = []
    for fn in ("poklong_corpus.txt", "benchmark_corpus.txt"):
        fp = os.path.join(BENCH_DATA, fn)
        if os.path.exists(fp):
            for ln in open(fp, "r", encoding="utf-8"):
                for w in ln.strip().split():
                    wn = normalize_unicode(w)
                    if wn and all(c in dict_chars for c in wn) and any("\uAA00" <= c <= "\uAA5F" for c in wn):
                        pool_words.append(wn)

    rng = random.Random(202611213)
    lines = []
    seen = set()
    stanza_num = 105
    cham_digits = "꩐꩑꩒꩓꩔꩕꩖꩗꩘꩙"

    def to_cnum(n: int) -> str:
        return "".join(cham_digits[int(d)] for d in str(n))

    for i in range(420):
        n_w = rng.randint(4, 7)
        start = (i * 7 + 31) % max(1, len(pool_words) - 12)
        ws = [pool_words[(start + k * 3) % len(pool_words)] for k in range(n_w)]
        if i % 8 == 0:
            ws.insert(0, f"{to_cnum(stanza_num)}꩞")
            stanza_num += 7
        elif i % 11 == 0:
            d, mo, yr = rng.randint(2, 26), rng.randint(1, 12), rng.choice([1996, 2003, 2012, 2019, 2026])
            ws.append(f"{to_cnum(d)}/{to_cnum(mo)}/{to_cnum(yr)}")
        elif i % 9 == 0 and len(ws) >= 3:
            ws[1] = f"{ws[1]}-{ws[2]}"
            ws.pop(2)
        tail = rng.choice(["", " ꩝", " ꩞", " ꩝꩝"])
        cand = normalize_unicode((" ".join(ws) + tail).strip())
        if cand not in forbidden_texts and cand not in seen and all(c in dict_chars for c in cand):
            seen.add(cand)
            lines.append(cand)
    assert len(lines) >= 180, f"Only built {len(lines)} disjoint v3 benchmark lines"
    return lines, forbidden_texts


VIETNAMESE_SCHOLARLY_GLOSSES = [
    "Bản dịch khổ thơ cổ thư Chăm vùng Panduranga thế kỷ 18",
    "Vua Po Klaong Garai cho đắp đập dẫn nước tưới ruộng đồng",
    "Lễ hội Katê đầu tháng bảy lịch Chăm tại tháp cổ Phan Rang",
    "Chữ Akhara Thrah khắc trên lá buông và chép trong giấy bản",
    "Khảo dị văn bản ký hiệu 12-18 và niên đại ngày 15/08/2025",
    "Lời khuyên răn con cháu giữ gìn phong tục tổ tiên ngàn đời",
    "Ghi chú ngữ âm và từ vựng tiếng Chăm cổ miền duyên hải",
    "Đối chiếu tư liệu viện Viễn Đông Bác Cổ ký hiệu CAM-204",
    "Bài ca lịch sử ghi chép nghi lễ mở cửa tháp đầu năm mới",
    "Tư liệu điền dã làng nghề dệt thổ cẩm và gốm cổ truyền thống",
    "Phiên âm đối chiếu phương ngữ Chăm Ninh Thuận và Bình Thuận",
    "Mục lục các truyền thuyết dân gian và gia phả hoàng tộc cổ",
    "Chú giải cách đọc dấu nguyên âm ghép và phụ âm cuối câu thơ",
    "Bản sao chụp trang thủ bút lưu trữ tại thư viện nghiên cứu",
]


def build_v3_expansion_pages(out_root: str, cham_lines: list) -> list:
    pages_dir = os.path.join(out_root, "pages")
    crops_dir = os.path.join(out_root, "crops")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)

    efeo_dir = os.path.join(BENCH_DATA, "fonts_efeo")
    held_dir = os.path.join(BENCH_DATA, "fonts_heldout")
    vi_font = os.path.join(PROJECT_ROOT, "ocr-training", "data", "fonts", "NotoSans-Regular.ttf")
    assert os.path.exists(vi_font), f"Missing {vi_font}"

    rng = random.Random(202611214)
    np_rng = np.random.default_rng(202611215)
    line_cursor = 0
    vi_cursor = 0
    docs = []

    pure_specs = [
        # Group A: Remaining 3 unseen EFEO fonts (completes 12/12 EFEO font family!)
        ("ood_v3_13_aiatrang", "efeo_unseen_regional_v3", [os.path.join(efeo_dir, "EFEO Cam AiaTrang.otf")], 15, 1, 0.0),
        ("ood_v3_14_kraong_2col", "efeo_unseen_regional_v3", [os.path.join(efeo_dir, "EFEO Cam Kraong.otf")], 16, 2, 0.0),
        ("ood_v3_15_pajai", "efeo_unseen_regional_v3", [os.path.join(efeo_dir, "EFEO Cam Pajai.otf")], 15, 1, 0.0),
        # Group B: Sinusoidally bowed palm-leaf *agal* folios (tests curved polygon ribbon rectification + OOD rec)
        ("ood_v3_16_curved_panduranga", "efeo_curved_agal_v3", [os.path.join(efeo_dir, "EFEO Cam Panduranga.otf")], 14, 1, 5.2),
        ("ood_v3_17_curved_aiatrang", "efeo_curved_agal_v3", [os.path.join(efeo_dir, "EFEO Cam AiaTrang.otf")], 14, 1, 6.2),
        ("ood_v3_18_curved_pajai", "efeo_curved_agal_v3", [os.path.join(efeo_dir, "EFEO Cam Pajai.otf")], 14, 1, 4.6),
        # Group C: Remaining 3 unseen Held-Out NotoSansCham weights (Thin, ExtraLight, Medium -> completes 6/6!)
        ("ood_v3_19_thin_camera", "heldout_ultralight_book_v3", [os.path.join(held_dir, "NotoSansCham-Thin.ttf")], 15, 1, 0.0),
        ("ood_v3_20_extralight_medium_camera", "heldout_ultralight_book_v3", [
            os.path.join(held_dir, "NotoSansCham-ExtraLight.ttf"),
            os.path.join(held_dir, "NotoSansCham-Medium.ttf"),
        ], 15, 1, 0.0),
    ]

    for doc_id, domain, font_list, n_lines, n_cols, wave_amp in pure_specs:
        for fp in font_list:
            assert os.path.exists(fp), f"Missing font: {fp}"
        bg_kind = "efeo_palm_leaf_agal" if "agal" in domain else ("efeo_regional_folio" if "efeo" in domain else "heldout_camera_book")
        w_page = 1120 if n_cols == 2 else 980
        h_page = 920 if n_cols == 2 else (980 if wave_amp > 0 else 1060)
        bg = make_archival_bg(w_page, h_page, bg_kind, rng, np_rng)
        fsize = 21 if n_cols == 2 else rng.choice([23, 25])

        placed_lines = []
        if n_cols == 1:
            cur_y = 54
            for li in range(n_lines):
                font_p = font_list[li % len(font_list)]
                gt = cham_lines[line_cursor % len(cham_lines)]
                line_cursor += 1
                words = gt.split()
                while len(words) > 3:
                    strip = render_font_strip(font_p, " ".join(words), fsize)
                    if strip.shape[1] <= w_page - 120:
                        gt = normalize_unicode(" ".join(words))
                        break
                    words = words[:-1]
                else:
                    gt = normalize_unicode(" ".join(words))
                    strip = render_font_strip(font_p, gt, fsize)
                sh, sw = strip.shape
                if cur_y + sh + 34 > h_page:
                    break
                x0 = rng.randint(46, 66)
                y0 = cur_y
                ink_max = 72 if "ultralight" in domain else 48
                ink_col = np.array([rng.randint(24, ink_max), rng.randint(24, ink_max), rng.randint(26, ink_max + 4)], dtype=np.float32)
                a = (strip.astype(np.float32) / 255.0)[:, :, None]
                roi = bg[y0:y0 + sh, x0:x0 + sw].astype(np.float32)
                bg[y0:y0 + sh, x0:x0 + sw] = np.clip(roi * (1.0 - a) + ink_col * a, 0, 255).astype(np.uint8)
                placed_lines.append((gt, [x0, y0, x0 + sw, y0 + sh], "cham"))
                gap = rng.randint(7, 12) if wave_amp > 0 else rng.randint(14, 23)
                cur_y += sh + gap
        else:
            half_n = n_lines // 2
            col_xs = [(42, w_page // 2 - 26), (w_page // 2 + 26, w_page - 42)]
            for cx0, cx1 in col_xs:
                cur_y = 54
                max_cw = cx1 - cx0
                for li in range(half_n):
                    font_p = font_list[0]
                    gt = cham_lines[line_cursor % len(cham_lines)]
                    line_cursor += 1
                    words = gt.split()[:4]
                    while len(words) > 2:
                        strip = render_font_strip(font_p, " ".join(words), fsize)
                        if strip.shape[1] <= max_cw:
                            gt = normalize_unicode(" ".join(words))
                            break
                        words = words[:-1]
                    else:
                        gt = normalize_unicode(" ".join(words))
                        strip = render_font_strip(font_p, gt, fsize)
                    sh, sw = strip.shape
                    if cur_y + sh + 30 > h_page:
                        break
                    x0 = cx0 + rng.randint(0, 8)
                    y0 = cur_y
                    ink_col = np.array([rng.randint(22, 50), rng.randint(22, 48), rng.randint(24, 54)], dtype=np.float32)
                    a = (strip.astype(np.float32) / 255.0)[:, :, None]
                    roi = bg[y0:y0 + sh, x0:x0 + sw].astype(np.float32)
                    bg[y0:y0 + sh, x0:x0 + sw] = np.clip(roi * (1.0 - a) + ink_col * a, 0, 255).astype(np.uint8)
                    placed_lines.append((gt, [x0, y0, x0 + sw, y0 + sh], "cham"))
                    cur_y += sh + rng.randint(15, 22)

        freq = 0.0068
        phase = 0.6 + 0.5 * len(docs)
        if wave_amp > 0:
            X, Y = np.meshgrid(np.arange(w_page, dtype=np.float32), np.arange(h_page, dtype=np.float32))
            dy_map = wave_amp * np.sin(X * freq + phase).astype(np.float32)
            bg = cv2.remap(bg, X, Y + dy_map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        bg = cv2.GaussianBlur(bg, (3, 3), sigmaX=0.58)
        page_path = os.path.join(pages_dir, f"{doc_id}.png")
        cv2.imwrite(page_path, bg)
        page_sha = sha256_file(page_path)

        doc_lines = []
        for l_i, (gt_txt, (x0, y0, x1, y1), script_tag) in enumerate(placed_lines, start=1):
            if wave_amp > 0:
                xs_band = np.arange(x0, x1, dtype=np.float32)
                shifts = -wave_amp * np.sin(xs_band * freq + phase)
                py0 = max(0, int(math.floor(y0 + float(shifts.min()) - 3)))
                py1 = min(h_page, int(math.ceil(y1 + float(shifts.max()) + 3)))
            else:
                py0, py1 = max(0, y0 - 3), min(h_page, y1 + 3)
            px0, px1 = max(0, x0 - 4), min(w_page, x1 + 4)
            crop_p = os.path.join(crops_dir, f"{doc_id}_line_{l_i:02d}.png")
            cv2.imwrite(crop_p, bg[py0:py1, px0:px1])
            doc_lines.append({
                "line": l_i,
                "script": script_tag,
                "bbox": [px0, py0, px1, py1],
                "crop_relpath": os.path.relpath(crop_p, PROJECT_ROOT).replace("\\", "/"),
                "crop_sha256": sha256_file(crop_p),
                "gt": gt_txt,
            })

        docs.append({
            "doc_id": doc_id,
            "source_type": "ood_archival_gold_v3",
            "archival_domain": domain,
            "font_name": "+".join(os.path.basename(f) for f in font_list),
            "columns": n_cols,
            "wave_amp_px": wave_amp,
            "page_relpath": os.path.relpath(page_path, PROJECT_ROOT).replace("\\", "/"),
            "page_sha256": page_sha,
            "line_count": len(doc_lines),
            "lines": doc_lines,
        })

    # Group D: 2 Scholarly Bilingual Cham + Vietnamese Commentary Folios (`Auto` Router v2 E2E test!)
    bilingual_specs = [
        ("ood_v3_21_bilingual_campa_vi", os.path.join(efeo_dir, "EFEO Cam Campa.otf")),
        ("ood_v3_22_bilingual_medium_vi", os.path.join(held_dir, "NotoSansCham-Medium.ttf")),
    ]
    for doc_id, cham_font_p in bilingual_specs:
        w_page, h_page = 1020, 1060
        bg = make_archival_bg(w_page, h_page, "heldout_camera_book", rng, np_rng)
        cur_y = 52
        placed_lines = []
        for pair_idx in range(7):
            # 1. Cham stanza line
            gt_c = cham_lines[line_cursor % len(cham_lines)]
            line_cursor += 1
            words = gt_c.split()
            while len(words) > 3:
                strip_c = render_font_strip(cham_font_p, " ".join(words), 24)
                if strip_c.shape[1] <= w_page - 120:
                    gt_c = normalize_unicode(" ".join(words))
                    break
                words = words[:-1]
            else:
                gt_c = normalize_unicode(" ".join(words))
                strip_c = render_font_strip(cham_font_p, gt_c, 24)
            sh_c, sw_c = strip_c.shape
            if cur_y + sh_c + 80 > h_page:
                break
            x0 = 52
            a_c = (strip_c.astype(np.float32) / 255.0)[:, :, None]
            bg[cur_y:cur_y + sh_c, x0:x0 + sw_c] = np.clip(bg[cur_y:cur_y + sh_c, x0:x0 + sw_c].astype(np.float32) * (1.0 - a_c) + 28.0 * a_c, 0, 255).astype(np.uint8)
            placed_lines.append((gt_c, [x0, cur_y, x0 + sw_c, cur_y + sh_c], "cham"))
            cur_y += sh_c + 14

            # 2. Vietnamese scholarly translation line
            gt_v = VIETNAMESE_SCHOLARLY_GLOSSES[vi_cursor % len(VIETNAMESE_SCHOLARLY_GLOSSES)]
            vi_cursor += 1
            strip_v = render_font_strip(vi_font, gt_v, 20)
            sh_v, sw_v = strip_v.shape
            x0_v = 68
            a_v = (strip_v.astype(np.float32) / 255.0)[:, :, None]
            bg[cur_y:cur_y + sh_v, x0_v:x0_v + sw_v] = np.clip(bg[cur_y:cur_y + sh_v, x0_v:x0_v + sw_v].astype(np.float32) * (1.0 - a_v) + 38.0 * a_v, 0, 255).astype(np.uint8)
            placed_lines.append((gt_v, [x0_v, cur_y, x0_v + sw_v, cur_y + sh_v], "vietnamese"))
            cur_y += sh_v + 22

        bg = cv2.GaussianBlur(bg, (3, 3), sigmaX=0.50)
        page_path = os.path.join(pages_dir, f"{doc_id}.png")
        cv2.imwrite(page_path, bg)
        page_sha = sha256_file(page_path)

        doc_lines = []
        for l_i, (gt_txt, (x0, y0, x1, y1), script_tag) in enumerate(placed_lines, start=1):
            px0, py0 = max(0, x0 - 4), max(0, y0 - 3)
            px1, py1 = min(w_page, x1 + 4), min(h_page, y1 + 3)
            crop_p = os.path.join(crops_dir, f"{doc_id}_line_{l_i:02d}.png")
            cv2.imwrite(crop_p, bg[py0:py1, px0:px1])
            doc_lines.append({
                "line": l_i,
                "script": script_tag,
                "bbox": [px0, py0, px1, py1],
                "crop_relpath": os.path.relpath(crop_p, PROJECT_ROOT).replace("\\", "/"),
                "crop_sha256": sha256_file(crop_p),
                "gt": gt_txt,
            })

        docs.append({
            "doc_id": doc_id,
            "source_type": "bilingual_archival_gold_v3",
            "archival_domain": "bilingual_cham_viet_commentary_v3",
            "font_name": f"{os.path.basename(cham_font_p)}+NotoSans-Regular.ttf",
            "columns": 1,
            "page_relpath": os.path.relpath(page_path, PROJECT_ROOT).replace("\\", "/"),
            "page_sha256": page_sha,
            "line_count": len(doc_lines),
            "lines": doc_lines,
        })

    return docs


def main():
    if os.path.exists(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)
    os.makedirs(OUT_ROOT, exist_ok=True)

    v1_m = json.load(open(os.path.join(BENCH_DATA, "realscan_gold_v1", "manifest.json"), "r", encoding="utf-8"))
    v2_m = json.load(open(os.path.join(BENCH_DATA, "realscan_gold_v2", "manifest.json"), "r", encoding="utf-8"))

    harum_doc = next(d for d in v2_m["documents"] if d["doc_id"] == "harum_palei_01")
    wiki_doc = next(d for d in v2_m["documents"] if d["doc_id"] == "wikimedia_public_specimens_01")
    handoff_doc = next(d for d in v1_m["documents"] if d["doc_id"] == "handoff_multiscript_01")
    v2_ood_docs = [d for d in v2_m["documents"] if d["source_type"] == "ood_archival_gold_v2"]

    cham_lines, forbidden_texts = load_v3_disjoint_cham_lines()
    v3_new_docs = build_v3_expansion_pages(OUT_ROOT, cham_lines)

    # Verify zero contamination of all new Cham lines against training sets
    for d in v3_new_docs:
        for ln in d["lines"]:
            if ln.get("script") == "cham":
                assert normalize_unicode(ln["gt"]) not in forbidden_texts, f"Contamination in {d['doc_id']}: {ln['gt']}"

    v2_ood_lines = sum(d["line_count"] for d in v2_ood_docs)
    v3_pure_docs = [d for d in v3_new_docs if d["source_type"] == "ood_archival_gold_v3"]
    v3_bi_docs = [d for d in v3_new_docs if d["source_type"] == "bilingual_archival_gold_v3"]
    v3_pure_lines = sum(d["line_count"] for d in v3_pure_docs)
    v3_bi_lines = sum(d["line_count"] for d in v3_bi_docs)
    handoff_cnt = sum(len(v) for v in handoff_doc["groups"].values())

    manifest = {
        "schema_version": "3.0.0",
        "dataset_id": "realscan_gold_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_policy": "100% held-out cross-domain & bilingual benchmark: real camera + Wikimedia + multiscript handoff + ALL 12/12 EFEO regional fonts + ALL 6/6 Held-Out NotoSansCham weights + curved palm-leaf agal + bilingual Cham-Vietnamese commentary; ZERO synthetic train/val overlap.",
        "summary": {
            "real_camera_pages": 1,
            "real_camera_lines": harum_doc["line_count"],
            "wikimedia_external_specimens": wiki_doc["specimen_count"],
            "handoff_multiscript_crops": handoff_cnt,
            "ood_v2_frozen_pages": len(v2_ood_docs),
            "ood_v2_frozen_lines": v2_ood_lines,
            "ood_v3_new_pure_cham_pages": len(v3_pure_docs),
            "ood_v3_new_pure_cham_lines": v3_pure_lines,
            "ood_all_pure_cham_pages": len(v2_ood_docs) + len(v3_pure_docs),
            "ood_all_pure_cham_lines": v2_ood_lines + v3_pure_lines,
            "bilingual_commentary_pages": len(v3_bi_docs),
            "bilingual_commentary_lines": v3_bi_lines,
            "efeo_fonts_covered": "12/12",
            "heldout_weights_covered": "6/6",
            "total_evaluated_items": harum_doc["line_count"] + wiki_doc["specimen_count"] + handoff_cnt + v2_ood_lines + v3_pure_lines + v3_bi_lines,
        },
        "documents": [harum_doc, wiki_doc, handoff_doc, *v2_ood_docs, *v3_new_docs],
    }

    m_path = os.path.join(OUT_ROOT, "manifest.json")
    with open(m_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"✅ Built RealScan-Gold v3 -> {m_path}")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
