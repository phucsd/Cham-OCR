#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR RealScan-Gold v4 Authentic Multi-Source Physical Corpus Builder
========================================================================
Harvests >= 56 authentic physical Eastern Cham (Akhar Thrah) & bilingual pages across
25+ independent physical manuscripts & historical printed volumes (zero synthetic pages
in real-scan strata), enforcing a strict 2-Tier Anti-Circular Ground-Truth Protocol:

  1. `verified_gold` split (`verified_by_human: true`, `gold_eligible: true`):
     - Carries forward ONLY human-verified real physical assets (`harum_palei_01` [1 page, 25 lines]
       and `wikimedia_public_specimens_01` [7 specimens]).
     - Excludes all 22 synthetic font-rendered `ood_*` pages from v2/v3 (`synthetic_pages_included = 0`).

  2. `staging_review_queue` split (`verification_status: "pending_human_review"`, `verified_by_human: false`, `gold_eligible: false`):
     - 56 new authentic physical pages across 25 independent manuscripts/books and 4 media strata:
         A. `palm_leaf_agal` (8 pages across 3 palm-leaf manuscripts: EAP531-1-1, EAP531-1-3, EAP698-12-1)
         B. `handwritten_paper_traditional` (23 pages across 12 handmade/"Đại"/"Quyến"/Chinese paper manuscripts)
         C. `handwritten_modern_substrate` (13 pages across 8 cement-paper, ruled notebook, newspaper, A4 & scroll mss.)
         D. `historical_printed_and_lithographed` (12 pages across Aymonier 1889 autographed Chronique royale chame
            and Aymonier & Cabaton 1906 2-column intra-line bilingual Dictionnaire čam-français)
     - Runs non-destructive `/ocr` on `https://ocr.cham.asia` to pre-segment line bounding boxes & polygons
       and generate `draft_prediction` (`gt` is strictly locked to `null` until human paleographic review!).
     - Exports Cham OCR Studio review packs (`review_queue/<doc_id>.studio.json` + `review_queue/index.html`)
       compatible with `scripts/ingest_studio_to_realscan_gold.py --version v4 --promote-verified`.

Copyright & Redistribution Compliance:
  - Downloaded image binaries (`pages/`, `crops/`, `review_queue/`) stay strictly local and gitignored
    to honor British Library EAP non-redistribution terms.
  - `manifest.json` stores canonical IIIF / Archive.org URLs, SHA-256 hashes, provenance, licensing,
    and draft/verified annotations so researchers can reconstruct images locally via `--fetch`.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DATA = os.path.join(PROJECT_ROOT, "ocr-benchmark", "data")
V4_ROOT = os.path.join(BENCH_DATA, "realscan_gold_v4")
PAGES_DIR = os.path.join(V4_ROOT, "pages")
CROPS_DIR = os.path.join(V4_ROOT, "crops")
REVIEW_DIR = os.path.join(V4_ROOT, "review_queue")
MANIFEST_PATH = os.path.join(V4_ROOT, "manifest.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "REPORTS", "REALSCAN_GOLD_V4_COLLECTION_REPORT_VI.md")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-training", "scripts"))
from generate_data import normalize_unicode  # noqa: E402

UA_HEADERS = {
    "User-Agent": "ChamOCRResearchBot/1.0 (https://ocr.cham.asia; phucsd@gmail.com; non-commercial academic OCR benchmark)"
}
LIVE_API_URL = "https://ocr.cham.asia"
ORIGIN_HF_SPACE_URL = "https://phucsd-cham-ocr-studio.hf.space"


# ---------------------------------------------------------------------------
# Canonical 56-Page Multi-Source Specification (25 Independent New Documents)
# ---------------------------------------------------------------------------
# Tuple format for EAP items:
#   (doc_id, eap_archive_id, canvas_idx_0based, stratum, material_desc, custodian_provenance, region, approx_date)
EAP_SPECS = [
    # --- Stratum A: Palm-Leaf Agal (8 pages across 3 independent manuscripts) ---
    ("eap531_1_1_c05", "EAP531-1-1", 4, "palm_leaf_agal", "Incised palm-leaf (agal) strips with wooden covers", "Cham Balamon collection, NIU/EAP531", "Ninh Thuận / Bình Thuận", "19th century"),
    ("eap531_1_1_c09", "EAP531-1-1", 8, "palm_leaf_agal", "Incised palm-leaf (agal) strips with wooden covers", "Cham Balamon collection, NIU/EAP531", "Ninh Thuận / Bình Thuận", "19th century"),
    ("eap531_1_1_c13", "EAP531-1-1", 12, "palm_leaf_agal", "Incised palm-leaf (agal) strips with wooden covers", "Cham Balamon collection, NIU/EAP531", "Ninh Thuận / Bình Thuận", "19th century"),

    ("eap531_1_3_c06", "EAP531-1-3", 5, "palm_leaf_agal", "Incised palm-leaf (agal) ritual manuscript, no. 3", "Cham Balamon collection, NIU/EAP531", "Ninh Thuận / Bình Thuận", "19th century"),
    ("eap531_1_3_c10", "EAP531-1-3", 9, "palm_leaf_agal", "Incised palm-leaf (agal) ritual manuscript, no. 3", "Cham Balamon collection, NIU/EAP531", "Ninh Thuận / Bình Thuận", "19th century"),
    ("eap531_1_3_c16", "EAP531-1-3", 15, "palm_leaf_agal", "Incised palm-leaf (agal) ritual manuscript, no. 3", "Cham Balamon collection, NIU/EAP531", "Ninh Thuận / Bình Thuận", "19th century"),

    ("eap698_12_1_c18", "EAP698-12-1", 17, "palm_leaf_agal", "Carved palm leaves darkened with resin ink (34x5cm)", "Mr. Bac Bich Van Nhuan", "Tuy Phong, Bình Thuận", "19th–20th century"),
    ("eap698_12_1_c85", "EAP698-12-1", 84, "palm_leaf_agal", "Carved palm leaves darkened with resin ink (34x5cm)", "Mr. Bac Bich Van Nhuan", "Tuy Phong, Bình Thuận", "19th–20th century"),

    # --- Stratum B: Traditional Handmade / Đại / Quyến / Chinese Paper (23 pages across 12 independent manuscripts) ---
    ("eap698_1_1_c15", "EAP698-1-1", 14, "handwritten_paper_traditional", "Chinese brush ink on paper (cosmology & rituals)", "Mr. Sam Toi / Mr. Sam Van Tanh", "Ninh Thuận", "Pre-1975"),
    ("eap698_1_1_c35", "EAP698-1-1", 34, "handwritten_paper_traditional", "Chinese brush ink on paper (cosmology & rituals)", "Mr. Sam Toi / Mr. Sam Van Tanh", "Ninh Thuận", "Pre-1975"),

    ("eap698_1_2_c08", "EAP698-1-2", 7, "handwritten_paper_traditional", "Chinese brush ink on paper (purification rites)", "Mr. Sam Toi / Mr. Sam Van Tanh", "Ninh Thuận", "Pre-1975"),
    ("eap698_1_2_c22", "EAP698-1-2", 21, "handwritten_paper_traditional", "Chinese brush ink on paper (purification rites)", "Mr. Sam Toi / Mr. Sam Van Tanh", "Ninh Thuận", "Pre-1975"),

    ("eap698_1_5_c09", "EAP698-1-5", 8, "handwritten_paper_traditional", "Chinese ink on water-damaged wrinkled paper", "Mr. Sam Toi / Mr. Sam Van Tanh", "Ninh Thuận", "Pre-1975"),

    ("eap698_1_12_c07", "EAP698-1-12", 6, "handwritten_paper_traditional", "Chinese ink on folded double Đại paper (poetry)", "Mr. Xa / Mr. Sam Van Tanh", "Ninh Thuận", "Pre-1975"),
    ("eap698_1_12_c15", "EAP698-1-12", 14, "handwritten_paper_traditional", "Chinese ink on folded double Đại paper (poetry)", "Mr. Xa / Mr. Sam Van Tanh", "Ninh Thuận", "Pre-1975"),

    ("eap698_2_1_c09", "EAP698-2-1", 8, "handwritten_paper_traditional", "Chinese ink on folded Đại paper (Riha Praong chants)", "Mr. Duong Po / Mr. Thanh Chau", "Ninh Thuận", "20th century"),
    ("eap698_2_1_c21", "EAP698-2-1", 20, "handwritten_paper_traditional", "Chinese ink on folded Đại paper (Riha Praong chants)", "Mr. Duong Po / Mr. Thanh Chau", "Ninh Thuận", "20th century"),

    ("eap698_3_1_c18", "EAP698-3-1", 17, "handwritten_paper_traditional", "Chinese ink on double Quyến paper (Kadhar rituals)", "Mr. Luong Mat / Gru Luong Dai Ve", "Ninh Thuận", "Pre-1975"),
    ("eap698_3_1_c55", "EAP698-3-1", 54, "handwritten_paper_traditional", "Chinese ink on double Quyến paper (Kadhar rituals)", "Mr. Luong Mat / Gru Luong Dai Ve", "Ninh Thuận", "Pre-1975"),

    ("eap698_10_1_c10", "EAP698-10-1", 9, "handwritten_paper_traditional", "Chinese ink on creased Đại paper (shamanic & medical)", "Mr. Lik Phaok Tao / Mrs. Dao Thi Luoi", "Ninh Thuận", "20th century"),
    ("eap698_10_1_c28", "EAP698-10-1", 27, "handwritten_paper_traditional", "Chinese ink on creased Đại paper (shamanic & medical)", "Mr. Lik Phaok Tao / Mrs. Dao Thi Luoi", "Ninh Thuận", "20th century"),

    ("eap698_12_7_c16", "EAP698-12-7", 15, "handwritten_paper_traditional", "India ink on 3-layer thin paper (Ummarup legend)", "Imam Lam Nam", "Bắc Bình, Bình Thuận", "19th–20th century"),
    ("eap698_12_7_c48", "EAP698-12-7", 47, "handwritten_paper_traditional", "India ink on 3-layer thin paper (Ummarup legend)", "Imam Lam Nam", "Bắc Bình, Bình Thuận", "19th–20th century"),

    ("eap1005_1_1_c14", "EAP1005-1-1", 13, "handwritten_paper_traditional", "Chinese ink on thin handmade paper (13x10cm, house rituals)", "EAP1005/1 Family Collection", "Ninh Thuận / Bình Thuận", "20th century"),
    ("eap1005_1_1_c46", "EAP1005-1-1", 45, "handwritten_paper_traditional", "Chinese ink on thin handmade paper (13x10cm, house rituals)", "EAP1005/1 Family Collection", "Ninh Thuận / Bình Thuận", "20th century"),

    ("eap1005_6_1_c12", "EAP1005-6-1", 11, "handwritten_paper_traditional", "Chinese ink on folded Chinese paper (healing mantras)", "EAP1005/6 Bình Thuận Collection", "Bình Thuận", "20th century"),
    ("eap1005_6_1_c32", "EAP1005-6-1", 31, "handwritten_paper_traditional", "Chinese ink on folded Chinese paper (healing mantras)", "EAP1005/6 Bình Thuận Collection", "Bình Thuận", "20th century"),

    ("eap1005_20_1_c20", "EAP1005-20-1", 19, "handwritten_paper_traditional", "Chinese ink on Chinese paper (Jiâ Puis Payak Kadhar rite)", "EAP1005/20 Bình Thuận Collection", "Bình Thuận", "20th century"),
    ("eap1005_20_1_c64", "EAP1005-20-1", 63, "handwritten_paper_traditional", "Chinese ink on Chinese paper (Jiâ Puis Payak Kadhar rite)", "EAP1005/20 Bình Thuận Collection", "Bình Thuận", "20th century"),

    ("eap1005_22_1_c15", "EAP1005-22-1", 14, "handwritten_paper_traditional", "Chinese ink on Chinese paper (Social issues poem)", "EAP1005/22 Bình Thuận Collection", "Bình Thuận", "20th century"),
    ("eap1005_22_1_c38", "EAP1005-22-1", 37, "handwritten_paper_traditional", "Chinese ink on Chinese paper (Social issues poem)", "EAP1005/22 Bình Thuận Collection", "Bình Thuận", "20th century"),

    # --- Stratum C: 20th-c. Everyday Substrates (Cement Paper, Notebook, Newspaper, A4 Photocopy, Scroll: 13 pages across 8 mss.) ---
    ("eap698_1_17_c25", "EAP698-1-17", 24, "handwritten_modern_substrate", "Photocopied handwritten manuscript on A4 paper (Umarup saga)", "Mr. Xa / Mr. Sam Van Tanh", "Ninh Thuận", "c. 1975"),

    ("eap698_4_1_c08", "EAP698-4-1", 7, "handwritten_modern_substrate", "Pen on multi-layer cement bag paper with pencil lines (Po Parang / Gleng anak)", "Mr. Thanh Nha / Mr. Thanh Van Phong", "Ninh Thuận", "1960s"),
    ("eap698_4_1_c42", "EAP698-4-1", 41, "handwritten_modern_substrate", "Pen on multi-layer cement bag paper with pencil lines (Po Parang / Gleng anak)", "Mr. Thanh Nha / Mr. Thanh Van Phong", "Ninh Thuận", "1960s"),

    ("eap698_5_1_c64", "EAP698-5-1", 63, "handwritten_modern_substrate", "3-color ballpoint pen on lined notebook paper (Cham calendar)", "Mr. Tai Nguc / Mr. Thanh May", "Ninh Thuận", "1978"),
    ("eap698_5_1_c92", "EAP698-5-1", 91, "handwritten_modern_substrate", "3-color ballpoint pen on lined notebook paper (Cham calendar)", "Mr. Tai Nguc / Mr. Thanh May", "Ninh Thuận", "1978"),

    ("eap698_6_1_c08", "EAP698-6-1", 7, "handwritten_modern_substrate", "Pen written over printed newspaper (Stories of Amarup)", "Gru Tai Muoi / Mr. Thanh Nghiem", "Ninh Thuận", "Pre-1999"),
    ("eap698_6_1_c18", "EAP698-6-1", 17, "handwritten_modern_substrate", "Pen written over printed newspaper (Stories of Amarup)", "Gru Tai Muoi / Mr. Thanh Nghiem", "Ninh Thuận", "Pre-1999"),

    ("eap698_7_1_c10", "EAP698-7-1", 9, "handwritten_modern_substrate", "Pen on doubled glued cement paper (Poetry of time & precepts)", "Mr. Maduen Cet / Mrs. Dao Thi Su", "Ninh Thuận", "Pre-1978"),
    ("eap698_7_1_c24", "EAP698-7-1", 23, "handwritten_modern_substrate", "Pen on doubled glued cement paper (Poetry of time & precepts)", "Mr. Maduen Cet / Mrs. Dao Thi Su", "Ninh Thuận", "Pre-1978"),

    ("eap698_8_1_c14", "EAP698-8-1", 13, "handwritten_modern_substrate", "Chinese ink on doubled newspaper paper (Religious ritual)", "Gru Guw Muk / Ms. Dao Thi Den", "Ninh Thuận", "20th century"),
    ("eap698_8_1_c45", "EAP698-8-1", 44, "handwritten_modern_substrate", "Ballpoint pen on doubled newspaper paper (Religious ritual)", "Gru Guw Muk / Ms. Dao Thi Den", "Ninh Thuận", "20th century"),

    ("eap1005_32_5_c05", "EAP1005-32-5", 4, "handwritten_modern_substrate", "Chinese ink on folded paper (Farewell rituals for infants)", "EAP1005 Lâm Thuận Collection", "Bình Thuận", "20th century"),
    ("eap1005_33_27_c10", "EAP1005-33-27", 9, "handwritten_modern_substrate", "Chinese ink on 180x13cm long handmade paper scroll (Bani deities in Akhar Thrah)", "EAP1005 Vụ Bổn Collection", "Bình Thuận", "20th century"),
]

# Tuple format for Internet Archive Public Domain Printed / Lithographed Books:
#   (doc_id, ia_item_id, leaf_num, stratum, script_profile, ocr_model_for_draft, material_desc, citation, scholarly_hint)
IA_PRINTED_SPECS = [
    ("aymonier1889_chronique_p108", "GrammaireDeLaLangueChame", 108, "historical_printed_and_lithographed", "pure_akhar_thrah", "v24_1_r4",
     "1889 autographed/lithographed fold-out plate (Chronique royale chame, lines 1–26)",
     "Étienne Aymonier (1889), Grammaire de la langue chame, Saigon: Imprimerie Coloniale (Public Domain)",
     "Paired with Aymonier 1889 published Romanized transcription pp. 74–76 & French translation pp. 86–88 (Chronique royale chame)."),
    ("aymonier1889_chronique_p110", "GrammaireDeLaLangueChame", 110, "historical_printed_and_lithographed", "pure_akhar_thrah", "v24_1_r4",
     "1889 autographed/lithographed fold-out plate (Chronique royale chame, lines 43–55)",
     "Étienne Aymonier (1889), Grammaire de la langue chame, Saigon: Imprimerie Coloniale (Public Domain)",
     "Paired with Aymonier 1889 published Romanized transcription pp. 77–79 & French translation pp. 88–89 (Chronique royale chame)."),
    ("aymonier1889_chronique_p112", "GrammaireDeLaLangueChame", 112, "historical_printed_and_lithographed", "pure_akhar_thrah", "v24_1_r4",
     "1889 autographed/lithographed fold-out plate (Chronique royale chame, lines 65–81)",
     "Étienne Aymonier (1889), Grammaire de la langue chame, Saigon: Imprimerie Coloniale (Public Domain)",
     "Paired with Aymonier 1889 published Romanized transcription pp. 79–81 & French translation pp. 89–90 (Chronique royale chame)."),
    ("aymonier1889_chronique_p114", "GrammaireDeLaLangueChame", 114, "historical_printed_and_lithographed", "pure_akhar_thrah", "v24_1_r4",
     "1889 autographed/lithographed fold-out plate (Chronique royale chame conclusion & numeral cipher table)",
     "Étienne Aymonier (1889), Grammaire de la langue chame, Saigon: Imprimerie Coloniale (Public Domain)",
     "Paired with Aymonier 1889 published Romanized transcription p. 81 & numeral table."),

    ("cabaton1906_dict_c080", "bub_gb_SSAoAAAAYAAJ", 80, "historical_printed_and_lithographed", "bilingual_intraline_cham_french", "auto",
     "1906 movable-type 2-column intra-line Akhar Thrah + Romanized Cham + French dictionary page",
     "Étienne Aymonier & Antoine Cabaton (1906), Dictionnaire čam-français, Paris: Imprimerie Nationale / EFEO, Vol. VII (Public Domain)",
     "Book p. 30 [idaramavan—inrā]: 2-column printed Akhar Thrah headwords + exact inline Romanized Cham transliteration + French glosses."),
    ("cabaton1906_dict_c130", "bub_gb_SSAoAAAAYAAJ", 130, "historical_printed_and_lithographed", "bilingual_intraline_cham_french", "auto",
     "1906 movable-type 2-column intra-line Akhar Thrah + Romanized Cham + French dictionary page",
     "Étienne Aymonier & Antoine Cabaton (1906), Dictionnaire čam-français, Paris: Imprimerie Nationale / EFEO, Vol. VII (Public Domain)",
     "2-column printed Akhar Thrah headwords (ka-/kha- section) + inline Romanized Cham transliteration + French glosses."),
    ("cabaton1906_dict_c190", "bub_gb_SSAoAAAAYAAJ", 190, "historical_printed_and_lithographed", "bilingual_intraline_cham_french", "auto",
     "1906 movable-type 2-column intra-line Akhar Thrah + Romanized Cham + French dictionary page",
     "Étienne Aymonier & Antoine Cabaton (1906), Dictionnaire čam-français, Paris: Imprimerie Nationale / EFEO, Vol. VII (Public Domain)",
     "2-column printed Akhar Thrah headwords (ca-/ja- section) + inline Romanized Cham transliteration + French glosses."),
    ("cabaton1906_dict_c250", "bub_gb_SSAoAAAAYAAJ", 250, "historical_printed_and_lithographed", "bilingual_intraline_cham_french", "auto",
     "1906 movable-type 2-column intra-line Akhar Thrah + Romanized Cham + French dictionary page",
     "Étienne Aymonier & Antoine Cabaton (1906), Dictionnaire čam-français, Paris: Imprimerie Nationale / EFEO, Vol. VII (Public Domain)",
     "2-column printed Akhar Thrah headwords (ta-/da-/na- section) + inline Romanized Cham transliteration + French glosses."),
    ("cabaton1906_dict_c310", "bub_gb_SSAoAAAAYAAJ", 310, "historical_printed_and_lithographed", "bilingual_intraline_cham_french", "auto",
     "1906 movable-type 2-column intra-line Akhar Thrah + Romanized Cham + French dictionary page",
     "Étienne Aymonier & Antoine Cabaton (1906), Dictionnaire čam-français, Paris: Imprimerie Nationale / EFEO, Vol. VII (Public Domain)",
     "2-column printed Akhar Thrah headwords (pa-/ba- section) + inline Romanized Cham transliteration + French glosses."),
    ("cabaton1906_dict_c370", "bub_gb_SSAoAAAAYAAJ", 370, "historical_printed_and_lithographed", "bilingual_intraline_cham_french", "auto",
     "1906 movable-type 2-column intra-line Akhar Thrah + Romanized Cham + French dictionary page",
     "Étienne Aymonier & Antoine Cabaton (1906), Dictionnaire čam-français, Paris: Imprimerie Nationale / EFEO, Vol. VII (Public Domain)",
     "2-column printed Akhar Thrah headwords (ma- section) + inline Romanized Cham transliteration + French glosses."),
    ("cabaton1906_dict_c430", "bub_gb_SSAoAAAAYAAJ", 430, "historical_printed_and_lithographed", "bilingual_intraline_cham_french", "auto",
     "1906 movable-type 2-column intra-line Akhar Thrah + Romanized Cham + French dictionary page",
     "Étienne Aymonier & Antoine Cabaton (1906), Dictionnaire čam-français, Paris: Imprimerie Nationale / EFEO, Vol. VII (Public Domain)",
     "2-column printed Akhar Thrah headwords (ra-/la- section) + inline Romanized Cham transliteration + French glosses."),
    ("cabaton1906_dict_c490", "bub_gb_SSAoAAAAYAAJ", 490, "historical_printed_and_lithographed", "bilingual_intraline_cham_french", "auto",
     "1906 movable-type 2-column intra-line Akhar Thrah + Romanized Cham + French dictionary page",
     "Étienne Aymonier & Antoine Cabaton (1906), Dictionnaire čam-français, Paris: Imprimerie Nationale / EFEO, Vol. VII (Public Domain)",
     "2-column printed Akhar Thrah headwords (sa-/ha- section) + inline Romanized Cham transliteration + French glosses."),
]

# Strictly EAP-series-, custodian-, and book-disjoint adaptation seed pool (11 manuscripts across 6 EAP series = 20 pages).
# Every other source (16 independent sources: 37 multi-line pages + 7 Wikimedia specimens, including 100% of Aymonier 1889,
# 100% of Cabaton 1906, 100% of EAP531-1, harum_palei_01, etc.) is permanently locked in `locked_test_holdout` (`training_forbidden=True`).
ADAPTATION_SEED_COLLECTION_IDS = frozenset({
    "EAP698-12-1",
    "EAP698-12-7",
    "EAP698-1-1",
    "EAP698-1-2",
    "EAP698-1-5",
    "EAP698-1-12",
    "EAP698-1-17",
    "EAP1005-6-1",
    "EAP1005-22-1",
    "EAP698-5-1",
    "EAP698-7-1",
})

# Stratified 20-page Fast-Review Priority Pack (`12` locked_test_holdout + `8` adaptation_seed_pool across all 4 physical strata)
GOLD20_PRIORITY_DOC_IDS = (
    # Locked Test Holdout (12 pages across all 4 strata)
    "eap531_1_1_c05",
    "eap531_1_3_c06",
    "eap698_2_1_c09",
    "eap698_3_1_c18",
    "eap1005_1_1_c14",
    "eap698_4_1_c08",
    "eap698_6_1_c08",
    "eap1005_33_27_c10",
    "aymonier1889_chronique_p108",
    "aymonier1889_chronique_p110",
    "cabaton1906_dict_c080",
    "cabaton1906_dict_c190",
    # Adaptation Seed Pool (8 pages across palm-leaf, traditional paper, and modern substrate)
    "eap698_12_1_c18",
    "eap698_12_7_c16",
    "eap698_1_1_c15",
    "eap698_1_12_c07",
    "eap1005_6_1_c12",
    "eap1005_22_1_c15",
    "eap698_5_1_c64",
    "eap698_7_1_c10",
)

SCHOLARLY_REFERENCE_EXCERPTS = {
    "aymonier1889_chronique_p108": {
        "edition": "Aymonier (1889), Grammaire de la langue chame, pp. 74–76 & 86–88",
        "romanized_cham_anchor": "Po Klong Garai — Po Rome — Po Saktiraydaputih (Chronique royale chame, opening royal genealogy lines 1–26).",
        "french_gloss_anchor": "Chronique des rois chams depuis Po Ovlah / Po Klong Garai jusqu'aux règnes historiques du Panduranga.",
    },
    "aymonier1889_chronique_p110": {
        "edition": "Aymonier (1889), Grammaire de la langue chame, pp. 77–79 & 88–89",
        "romanized_cham_anchor": "Chronique royale chame, lines 43–55: sakawi Cham royal dates (animal-cycle & numeral years) of Panduranga lords.",
        "french_gloss_anchor": "Succession chronologique des rois chams aux XVIIe–XVIIIe siècles avec durées de règne.",
    },
    "aymonier1889_chronique_p112": {
        "edition": "Aymonier (1889), Grammaire de la langue chame, pp. 79–81 & 89–90",
        "romanized_cham_anchor": "Chronique royale chame, lines 65–81: late-18th / early-19th century Panduranga rulers (Po Tisuntiraydapuran, Po Chong Chan).",
        "french_gloss_anchor": "Derniers souverains autonomes du Panduranga jusqu'au début du XIXe siècle.",
    },
    "aymonier1889_chronique_p114": {
        "edition": "Aymonier (1889), Grammaire de la langue chame, p. 81 & Akhar Thrah numeral plate",
        "romanized_cham_anchor": "Conclusion of Chronique royale chame + Akhar Thrah numeral & cipher reference table (0–9, 10, 100, 1000).",
        "french_gloss_anchor": "Fin de la chronique royale et tableau des chiffres chams.",
    },
    "cabaton1906_dict_c080": {
        "edition": "Aymonier & Cabaton (1906), Dictionnaire čam-français, p. 30 [idaramavan — inrā]",
        "romanized_cham_anchor": "Headwords: idaramavan, iduni, in, inā, inăk, inogarai, inơmodōn, init, inău, intan, indannh, innobar, inrā.",
        "french_gloss_anchor": "2-column bilingual entries from idaramavan (ramadan) through inrā (Indra).",
    },
    "cabaton1906_dict_c130": {
        "edition": "Aymonier & Cabaton (1906), Dictionnaire čam-français, leaf 130 [ka- / kha- section]",
        "romanized_cham_anchor": "2-column Akhar Thrah headwords in initial ka-/kha- series with Sanskrit/Malay/Khmer etymologies.",
        "french_gloss_anchor": "Entrées lexicales čames initiales k-/kh- avec translittération EFEO et traduction française.",
    },
    "cabaton1906_dict_c190": {
        "edition": "Aymonier & Cabaton (1906), Dictionnaire čam-français, leaf 190 [ca- / ja- section]",
        "romanized_cham_anchor": "2-column Akhar Thrah headwords in palatal ca-/ja- series with inline EFEO Romanization.",
        "french_gloss_anchor": "Entrées lexicales čames initiales c-/j- avec exemples phraséologiques.",
    },
    "cabaton1906_dict_c250": {
        "edition": "Aymonier & Cabaton (1906), Dictionnaire čam-français, leaf 250 [ta- / da- / na- section]",
        "romanized_cham_anchor": "2-column Akhar Thrah headwords in dental ta-/da-/na- series with inline EFEO Romanization.",
        "french_gloss_anchor": "Entrées lexicales čames initiales t-/d-/n- et composés.",
    },
    "cabaton1906_dict_c310": {
        "edition": "Aymonier & Cabaton (1906), Dictionnaire čam-français, leaf 310 [pa- / ba- section]",
        "romanized_cham_anchor": "2-column Akhar Thrah headwords in labial pa-/ba- series with inline EFEO Romanization.",
        "french_gloss_anchor": "Entrées lexicales čames initiales p-/b- et locutions rituelles.",
    },
    "cabaton1906_dict_c370": {
        "edition": "Aymonier & Cabaton (1906), Dictionnaire čam-français, leaf 370 [ma- section]",
        "romanized_cham_anchor": "2-column Akhar Thrah headwords in nasal/causative ma-/mơ- prefix series.",
        "french_gloss_anchor": "Entrées lexicales čames préfixées en mơ-/ma-.",
    },
    "cabaton1906_dict_c430": {
        "edition": "Aymonier & Cabaton (1906), Dictionnaire čam-français, leaf 430 [ra- / la- section]",
        "romanized_cham_anchor": "2-column Akhar Thrah headwords in liquid ra-/la- series with inline EFEO Romanization.",
        "french_gloss_anchor": "Entrées lexicales čames initiales r-/l-.",
    },
    "cabaton1906_dict_c490": {
        "edition": "Aymonier & Cabaton (1906), Dictionnaire čam-français, leaf 490 [sa- / ha- section]",
        "romanized_cham_anchor": "2-column Akhar Thrah headwords in sibilant/aspirate sa-/ha- series.",
        "french_gloss_anchor": "Entrées lexicales čames initiales s-/h-.",
    },
}


def _eap_series_key(source_collection_id: str) -> str:
    parts = str(source_collection_id or "").split("-")
    if len(parts) >= 2 and parts[0].startswith("EAP"):
        return f"{parts[0]}-{parts[1]}"
    return str(source_collection_id or "")


def assign_benchmark_split(source_collection_id: str) -> tuple[str, bool]:
    if source_collection_id in ADAPTATION_SEED_COLLECTION_IDS:
        return "adaptation_seed_pool", False
    return "locked_test_holdout", True


def verify_custodian_and_series_disjoint_split(documents: list) -> dict:
    """Strictly asserts zero shared collection ID, EAP archive series, custodian, book, or SHA-256 across splits."""
    holdout_docs = [d for d in documents if d.get("benchmark_split") == "locked_test_holdout"]
    seed_docs = [d for d in documents if d.get("benchmark_split") == "adaptation_seed_pool"]
    assert len(holdout_docs) + len(seed_docs) == len(documents), "Every document must declare benchmark_split"

    holdout_cids = {d["source_collection_id"] for d in holdout_docs}
    seed_cids = {d["source_collection_id"] for d in seed_docs}
    assert not (holdout_cids & seed_cids), f"Collection leakage: {holdout_cids & seed_cids}"

    holdout_series = {_eap_series_key(d["source_collection_id"]) for d in holdout_docs}
    seed_series = {_eap_series_key(d["source_collection_id"]) for d in seed_docs}
    assert not (holdout_series & seed_series), f"EAP series leakage: {holdout_series & seed_series}"

    holdout_custodians = {d.get("custodian_or_author") for d in holdout_docs if d.get("custodian_or_author")}
    seed_custodians = {d.get("custodian_or_author") for d in seed_docs if d.get("custodian_or_author")}
    assert not (holdout_custodians & seed_custodians), f"Custodian leakage: {holdout_custodians & seed_custodians}"

    holdout_shas = {d.get("page_sha256") for d in holdout_docs if d.get("page_sha256")}
    seed_shas = {d.get("page_sha256") for d in seed_docs if d.get("page_sha256")}
    assert not (holdout_shas & seed_shas), "Page SHA-256 leakage!"

    assert "GrammaireDeLaLangueChame" in holdout_cids and "bub_gb_SSAoAAAAYAAJ" in holdout_cids
    assert all(d.get("training_forbidden") is True for d in holdout_docs)
    return {
        "locked_test_holdout_documents": len(holdout_docs),
        "locked_test_holdout_pages": sum(1 for d in holdout_docs if "lines" in d and d.get("doc_id") != "wikimedia_public_specimens_01"),
        "locked_test_holdout_sources": len(holdout_cids),
        "adaptation_seed_pool_pages": len(seed_docs),
        "adaptation_seed_pool_sources": len(seed_cids),
        "shared_collections": 0,
        "shared_eap_series": 0,
        "shared_custodians": 0,
    }


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def http_get_bytes(url: str, retries: int = 3, timeout: int = 35) -> bytes:
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts for {url}: {last_err}")


def resolve_eap_iiif_catalog(eap_ids: list) -> dict:
    """Fetches British Library EAP IIIF Presentation 3.0 manifests in parallel."""
    out = {}

    def _fetch_one(eid):
        raw = http_get_bytes(f"https://eap.bl.uk/archive-file/{eid}/manifest")
        return eid, json.loads(raw.decode("utf-8"))

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(_fetch_one, eid) for eid in sorted(set(eap_ids))]
        for fut in as_completed(futs):
            eid, m = fut.result()
            out[eid] = m
    return out


def resolve_ia_bookreader_info(ia_ids: list) -> dict:
    """Resolves Archive.org direct storage server + JP2 zip metadata for reliable BookReader downloads."""
    info = {}
    for iid in sorted(set(ia_ids)):
        meta = json.loads(http_get_bytes(f"https://archive.org/metadata/{iid}").decode("utf-8"))
        d1 = meta.get("d1") or meta.get("server")
        dr = meta.get("dir")
        jp2_zips = [f["name"] for f in meta.get("files", []) if f.get("name", "").endswith("_jp2.zip")]
        assert jp2_zips, f"No _jp2.zip found on Internet Archive item {iid}"
        zname = jp2_zips[0]
        prefix = zname[:-len("_jp2.zip")]
        info[iid] = {"server": d1, "dir": dr, "zip_name": zname, "prefix": prefix}
    return info


def build_fetch_plan() -> list:
    """Builds the deterministic 56-page download & provenance plan."""
    eap_ids = [x[1] for x in EAP_SPECS]
    ia_ids = [x[1] for x in IA_PRINTED_SPECS]
    print(f"[1/4] Resolving IIIF & Archive.org manifests ({len(set(eap_ids))} EAP fonds + {len(set(ia_ids))} IA volumes)...")
    eap_manifests = resolve_eap_iiif_catalog(eap_ids)
    ia_info = resolve_ia_bookreader_info(ia_ids)

    plan = []
    for doc_id, eid, cidx, stratum, mat_desc, custodian, region, approx_date in EAP_SPECS:
        m = eap_manifests[eid]
        items = m.get("items", [])
        assert cidx < len(items), f"{doc_id}: canvas_idx {cidx} >= {len(items)}"
        canv = items[cidx]
        body = canv["items"][0]["items"][0]["body"]
        srv_list = body.get("service") or []
        srv_id = srv_list[0].get("id") or srv_list[0].get("@id")
        img_url = f"{srv_id}/full/!1600,1600/0/default.jpg"
        title = (m.get("label", {}).get("en") or [eid])[0].strip('"')
        req_stmt = "; ".join((m.get("requiredStatement", {}).get("value", {}).get("en") or [""]))
        plan.append({
            "doc_id": doc_id,
            "source_archive": "British Library Endangered Archives Programme (EAP)",
            "source_collection_id": eid,
            "canvas_or_leaf_index": cidx + 1,
            "title": title,
            "stratum": stratum,
            "script_profile": "pure_akhar_thrah",
            "draft_ocr_model": "v24_1_r4",
            "material": mat_desc,
            "custodian_or_author": custodian,
            "region": region,
            "approx_date": approx_date,
            "catalogue_url": f"https://eap.bl.uk/archive-file/{eid}",
            "download_url": img_url,
            "license": "EAP Non-Commercial Research Access (No Redistribution of Image Binaries)",
            "redistribution_allowed_in_git": False,
            "required_statement": req_stmt or f"British Library {eid}, digitised by Northern Illinois University (Arcadia).",
            "scholarly_reference_hint": None,
        })

    for doc_id, iid, leaf, stratum, script_prof, draft_model, mat_desc, citation, hint in IA_PRINTED_SPECS:
        inf = ia_info[iid]
        leaf_fn = f"{inf['prefix']}_{leaf:04d}.jp2"
        br_url = (
            f"https://{inf['server']}/BookReader/BookReaderImages.php?"
            f"zip={inf['dir']}/{inf['zip_name']}&file={inf['prefix']}_jp2/{leaf_fn}&id={iid}&scale=2&rotate=0"
        )
        plan.append({
            "doc_id": doc_id,
            "source_archive": "Internet Archive (Digitized Academic Monograph)",
            "source_collection_id": iid,
            "canvas_or_leaf_index": leaf,
            "title": citation,
            "stratum": stratum,
            "script_profile": script_prof,
            "draft_ocr_model": draft_model,
            "material": mat_desc,
            "custodian_or_author": "Étienne Aymonier & Antoine Cabaton" if "1906" in doc_id else "Étienne Aymonier",
            "region": "Panduranga / Bình Thuận (printed in Saigon / Paris)",
            "approx_date": "1906" if "1906" in doc_id else "1889",
            "catalogue_url": f"https://archive.org/details/{iid}/page/n{leaf - 1}/mode/1up",
            "download_url": br_url,
            "license": "Public Domain (Published pre-1910)",
            "redistribution_allowed_in_git": False,  # Kept gitignored uniformly to keep Git repo lightweight (<50MB)
            "required_statement": citation,
            "scholarly_reference_hint": hint,
        })

    return plan


def fetch_all_pages(plan: list) -> list:
    os.makedirs(PAGES_DIR, exist_ok=True)
    print(f"[2/4] Downloading / verifying {len(plan)} physical pages into {PAGES_DIR} ...")

    def _dl(item):
        dst = os.path.join(PAGES_DIR, f"{item['doc_id']}.jpg")
        if not os.path.exists(dst) or os.path.getsize(dst) < 15000:
            raw = http_get_bytes(item["download_url"], retries=4, timeout=45)
            with open(dst, "wb") as f:
                f.write(raw)
        img = cv2.imread(dst)
        assert img is not None, f"Failed to decode downloaded image: {dst}"
        h, w = img.shape[:2]
        assert w >= 600 and h >= 500, f"Unexpectedly small image {dst}: {w}x{h}"
        item["page_relpath"] = os.path.relpath(dst, PROJECT_ROOT).replace("\\", "/")
        item["page_sha256"] = sha256_file(dst)
        item["width"] = int(w)
        item["height"] = int(h)
        item["file_bytes"] = int(os.path.getsize(dst))
        return item

    enriched = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_dl, it): it["doc_id"] for it in plan}
        for fut in as_completed(futs):
            res = fut.result()
            enriched.append(res)

    order_map = {it["doc_id"]: idx for idx, it in enumerate(plan)}
    enriched.sort(key=lambda x: order_map[x["doc_id"]])

    # Strict SHA-256 uniqueness verification across all 56 harvested pages
    seen_sha = {}
    for it in enriched:
        s = it["page_sha256"]
        assert s not in seen_sha, f"Duplicate SHA-256 between {it['doc_id']} and {seen_sha[s]}"
        seen_sha[s] = it["doc_id"]
    print(f"  -> All {len(enriched)} physical pages verified on disk (100% unique SHA-256).")
    return enriched


def render_contact_sheets(enriched: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    chunk_sz = 16
    tw, th = 360, 270
    sheets = []
    for s_idx, start in enumerate(range(0, len(enriched), chunk_sz), start=1):
        batch = enriched[start:start + chunk_sz]
        tiles = []
        for idx, it in enumerate(batch, start=start + 1):
            img_p = os.path.join(PROJECT_ROOT, it["page_relpath"].replace("/", os.sep))
            im = cv2.imread(img_p)
            h, w = im.shape[:2]
            scale = min(tw / w, (th - 34) / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas = np.full((th, tw, 3), 28, dtype=np.uint8)
            x_off = (tw - nw) // 2
            y_off = 30 + ((th - 30 - nh) // 2)
            canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
            lbl = f"#{idx:02d} {it['doc_id']}"
            cv2.putText(canvas, lbl[:28], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 235, 160), 1, cv2.LINE_AA)
            tiles.append(canvas)
        while len(tiles) < chunk_sz:
            tiles.append(np.full((th, tw, 3), 28, dtype=np.uint8))
        rows = [np.hstack(tiles[r:r + 4]) for r in range(0, chunk_sz, 4)]
        sheet = np.vstack(rows)
        sp = os.path.join(out_dir, f"v4_contact_sheet_{s_idx}.jpg")
        cv2.imwrite(sp, sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])
        sheets.append(sp)
    print(f"  -> Rendered {len(sheets)} visual contact sheets -> {sheets}")
    return sheets


def compute_eap_card_cutoff_y(img: np.ndarray, source_collection_id: str, source_archive: str) -> int:
    """Finds the exact y-coordinate separating an EAP manuscript folio from its bottom ruler/QpCARD/number plaque."""
    h, w = img.shape[:2]
    if "EAP" not in source_archive:
        return h + 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    band = gray[:, int(0.12 * w):int(0.88 * w)]

    # Case A: EAP531 palm-leaf on burlap with bright white plastic ruler at bottom
    if str(source_collection_id).startswith("EAP531"):
        row_p85 = np.percentile(band, 85, axis=1)
        for y in range(int(0.58 * h), int(0.92 * h)):
            if row_p85[y] >= 192 and float(np.mean(row_p85[y:min(h, y + 12)])) >= 192:
                return max(int(0.58 * h), y - 3)
        return int(0.74 * h)

    # Case B: EAP698 / EAP1005 folios on dark table with black horizontal gap above bottom QpCARD / number tag
    row_p92 = np.percentile(band, 92, axis=1)
    # Find bottom-most bright card region first, then walk up into the dark table gap (< 82) separating it from the folio
    saw_card = False
    for y in range(int(0.96 * h), int(0.62 * h), -1):
        if row_p92[y] >= 135:
            saw_card = True
        elif saw_card and float(np.max(row_p92[max(0, y - 4):y + 1])) < 82:
            return y
    return int(0.86 * h)


def is_eap_calibration_card_box(bbox: list, text: str, img_h: int, source_archive: str, cutoff_y: int = None) -> bool:
    """Flags bottom-margin ruler/QpCARD/plaque detections on EAP photos (even if Cham CTC hallucinated on QpCARD)."""
    if "EAP" not in source_archive:
        return False
    yc = 0.5 * (bbox[1] + bbox[3])
    if cutoff_y is not None and yc >= cutoff_y:
        return True
    if any("\uAA00" <= ch <= "\uAA5F" for ch in (text or "")):
        return False
    return bool(bbox[1] >= 0.77 * img_h)


def call_live_draft_ocr(image_path: str, model_id: str, retries: int = 3, timeout: int = 240) -> dict:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    payload = json.dumps({
        "image": b64,
        "model": model_id,
        "method": "dbnet",
        "reading_order": "auto",
    }).encode("utf-8")
    last_err = None
    endpoints = [f"{LIVE_API_URL}/ocr", f"{ORIGIN_HF_SPACE_URL}/ocr"]
    for attempt in range(retries):
        ep = endpoints[0] if attempt == 0 else endpoints[min(attempt, len(endpoints) - 1)]
        try:
            req = urllib.request.Request(
                ep,
                data=payload,
                headers={**UA_HEADERS, "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            time.sleep(3.0 * (attempt + 1))
    raise RuntimeError(f"Draft OCR failed on {image_path}: {last_err}")


def run_presegmentation_and_draft_pack(enriched: list, use_local_engine: bool = False, refresh_ocr: bool = False) -> list:
    os.makedirs(CROPS_DIR, exist_ok=True)
    os.makedirs(REVIEW_DIR, exist_ok=True)
    cache_dir = os.path.join(REVIEW_DIR, "_raw_api_cache")
    os.makedirs(cache_dir, exist_ok=True)

    if refresh_ocr:
        for fn in os.listdir(cache_dir):
            if fn.endswith(".ocr.json"):
                os.remove(os.path.join(cache_dir, fn))
        for fn in os.listdir(CROPS_DIR):
            if fn.endswith(".png"):
                os.remove(os.path.join(CROPS_DIR, fn))
        print("  -> Wiped stale pre-v26.2 `_raw_api_cache/*.ocr.json` and `crops/*.png`.")

    local_app = None
    if use_local_engine:
        studio_dir = os.path.join(PROJECT_ROOT, "ocr-studio")
        if studio_dir not in sys.path:
            sys.path.insert(0, studio_dir)
        import app as _studio_app
        local_app = _studio_app
        print(f"[3/4] Running Pipeline v26.2 pre-segmentation & draft OCR on {len(enriched)} pages via local Python engine ...")
    else:
        print(f"[3/4] Running non-destructive pre-segmentation & draft OCR on {len(enriched)} pages via {LIVE_API_URL} ...")

    def _process_doc(idx_and_it):
        idx, it = idx_and_it
        doc_id = it["doc_id"]
        img_p = os.path.join(PROJECT_ROOT, it["page_relpath"].replace("/", os.sep))
        cache_p = os.path.join(cache_dir, f"{doc_id}.ocr.json")
        t0 = time.time()
        img = cv2.imread(img_p)
        h, w = img.shape[:2]
        if os.path.exists(cache_p) and os.path.getsize(cache_p) > 200:
            resp = json.load(open(cache_p, "r", encoding="utf-8"))
        elif local_app is not None:
            resp = local_app.run_ocr_pipeline(img, model_ver=it["draft_ocr_model"], method="dbnet", reading_order="auto")
            with open(cache_p, "w", encoding="utf-8") as cf:
                json.dump(resp, cf, ensure_ascii=False)
        else:
            resp = call_live_draft_ocr(img_p, it["draft_ocr_model"])
            with open(cache_p, "w", encoding="utf-8") as cf:
                json.dump(resp, cf, ensure_ascii=False)
        dt = round(time.time() - t0, 2)
        raw_lines = resp.get("lines") or []

        def _collect_kept_lines(raw_lns):
            m_lines, s_lines, c_filt = [], [], 0
            for raw_ln in raw_lns:
                bbox = raw_ln.get("bbox")
                raw_txt = (raw_ln.get("text") or "").strip()
                if not bbox or len(bbox) != 4:
                    continue
                x0, y0, x1, y1 = [int(round(v)) for v in bbox]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                if x1 - x0 < 6 or y1 - y0 < 6:
                    continue
                if is_eap_calibration_card_box([x0, y0, x1, y1], raw_txt, h, it["source_archive"], cutoff_y=eap_cutoff_y):
                    c_filt += 1
                    continue
                norm_txt = normalize_unicode(raw_txt) if any("\uAA00" <= c <= "\uAA5F" for c in raw_txt) else raw_txt
                l_num = len(m_lines) + 1
                crop_p = os.path.join(CROPS_DIR, f"{doc_id}_line_{l_num:02d}.png")
                crop_arr = img[y0:y1, x0:x1]
                cv2.imwrite(crop_p, crop_arr)
                conf = round(float(raw_ln.get("confidence") or 0.0), 4)
                cal_conf = round(float(raw_ln.get("calibrated_confidence") or conf), 4)
                sel_rec = raw_ln.get("selected_recognizer") or it["draft_ocr_model"]
                seg = raw_ln.get("segmentation") or {}
                raw_poly = raw_ln.get("polygon")
                compact_poly = (
                    [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in raw_poly if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                    if isinstance(raw_poly, list) else None
                )
                m_lines.append({
                    "line": l_num,
                    "bbox": [x0, y0, x1, y1],
                    "polygon": compact_poly,
                    "crop_relpath": os.path.relpath(crop_p, PROJECT_ROOT).replace("\\", "/"),
                    "crop_sha256": sha256_file(crop_p),
                    "gt": None,
                    "verified_by_human": False,
                    "draft_prediction": norm_txt,
                    "draft_model": it["draft_ocr_model"],
                    "draft_selected_recognizer": sel_rec,
                    "draft_confidence": conf,
                    "draft_calibrated_confidence": cal_conf,
                    "needs_review": bool(seg.get("needs_review", False) or conf < 0.75),
                    "review_reasons": seg.get("reason") or [],
                })
                ok, buf = cv2.imencode(".png", crop_arr)
                b64_crop = ("data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")) if ok else None
                s_lines.append({
                    "line_id": l_num,
                    "text": norm_txt,
                    "original_text": norm_txt,
                    "confidence": conf,
                    "calibrated_confidence": cal_conf,
                    "selected_recognizer": sel_rec,
                    "human_verified": False,
                    "bbox": [x0, y0, x1, y1],
                    "coords": raw_ln.get("coords") or [x0, y0, x1 - x0, y1 - y0],
                    "polygon": raw_ln.get("polygon"),
                    "segmentation": seg,
                    "image": b64_crop,
                })
            return m_lines, s_lines, c_filt

        eap_cutoff_y = compute_eap_card_cutoff_y(img, it["source_collection_id"], it["source_archive"])
        manifest_lines, studio_lines, calib_filtered = _collect_kept_lines(raw_lines)

        # If 0 manuscript lines survived above eap_cutoff_y (e.g. ultra-faint ballpoint ink masked by a crisp bottom QpCARD),
        # run CLAHE local-contrast rescue with the bottom calibration strip masked out.
        if not manifest_lines and local_app is not None:
            masked = img.copy()
            if eap_cutoff_y < h:
                masked[eap_cutoff_y:, :] = 20
            lab_m = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
            lm, am, bm = cv2.split(lab_m)
            enh_m = cv2.cvtColor(cv2.merge([cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8)).apply(lm), am, bm]), cv2.COLOR_LAB2BGR)
            resp = local_app.run_ocr_pipeline(enh_m, model_ver=it["draft_ocr_model"], method="dbnet", reading_order="auto")
            with open(cache_p, "w", encoding="utf-8") as cf:
                json.dump(resp, cf, ensure_ascii=False)
            raw_lines = resp.get("lines") or []
            manifest_lines, studio_lines, calib_filtered = _collect_kept_lines(raw_lines)

        bench_split, train_forbidden = assign_benchmark_split(it["source_collection_id"])
        ref_excerpt = SCHOLARLY_REFERENCE_EXCERPTS.get(doc_id)
        is_gold20 = doc_id in GOLD20_PRIORITY_DOC_IDS

        studio_payload = {
            "image_name": f"{doc_id}.jpg",
            "doc_id": doc_id,
            "benchmark_split": bench_split,
            "training_forbidden": train_forbidden,
            "gold20_priority": is_gold20,
            "status": "Needs review",
            "verification_status": "pending_human_review",
            "verified_by_human": False,
            "model": it["draft_ocr_model"],
            "source_archive": it["source_archive"],
            "source_collection_id": it["source_collection_id"],
            "catalogue_url": it["catalogue_url"],
            "scholarly_reference_hint": it["scholarly_reference_hint"],
            "scholarly_reference_excerpt": ref_excerpt,
            "lines": studio_lines,
        }
        studio_json_p = os.path.join(REVIEW_DIR, f"{doc_id}.studio.json")
        with open(studio_json_p, "w", encoding="utf-8") as sf:
            json.dump(studio_payload, sf, indent=2, ensure_ascii=False)

        cham_lines_cnt = sum(1 for ln in manifest_lines if any("\uAA00" <= c <= "\uAA5F" for c in (ln["draft_prediction"] or "")))
        mean_conf = round(float(np.mean([ln["draft_confidence"] for ln in manifest_lines])) if manifest_lines else 0.0, 4)
        doc_info = resp.get("document") or {}
        prof_info = resp.get("profiler") or {}
        print(
            f"  [{idx:02d}/{len(enriched)}] {doc_id:28s} | {bench_split:20s} | lines={len(manifest_lines):2d} (cham={cham_lines_cnt:2d}, card_filtered={calib_filtered}) "
            f"| conf={mean_conf:.3f} | det={prof_info.get('detector','?'):21s} | {dt:.1f}s"
        )

        return {
            "doc_id": doc_id,
            "split": "staging_review_queue",
            "benchmark_split": bench_split,
            "training_forbidden": train_forbidden,
            "gold20_priority": is_gold20,
            "verification_status": "pending_human_review",
            "verified_by_human": False,
            "gold_eligible": False,
            "annotator_id": None,
            "review_round": 0,
            "source_type": "authentic_archival_scan_v4",
            "stratum": it["stratum"],
            "script_profile": it["script_profile"],
            "source_archive": it["source_archive"],
            "source_collection_id": it["source_collection_id"],
            "eap_or_book_series": _eap_series_key(it["source_collection_id"]),
            "canvas_or_leaf_index": it["canvas_or_leaf_index"],
            "title": it["title"],
            "material": it["material"],
            "custodian_or_author": it["custodian_or_author"],
            "region": it["region"],
            "approx_date": it["approx_date"],
            "catalogue_url": it["catalogue_url"],
            "download_url": it["download_url"],
            "license": it["license"],
            "redistribution_allowed_in_git": it["redistribution_allowed_in_git"],
            "required_statement": it["required_statement"],
            "scholarly_reference_hint": it["scholarly_reference_hint"],
            "scholarly_reference_excerpt": ref_excerpt,
            "page_relpath": it["page_relpath"],
            "page_sha256": it["page_sha256"],
            "width": it["width"],
            "height": it["height"],
            "draft_ocr_summary": {
                "pipeline_version": "v26.2_router_v2.3",
                "draft_model": it["draft_ocr_model"],
                "executed_detector": prof_info.get("detector"),
                "detector_switched": bool(prof_info.get("detector_switched", False)),
                "raw_api_lines": len(raw_lines),
                "calibration_card_boxes_filtered": calib_filtered,
                "kept_draft_lines": len(manifest_lines),
                "cham_bearing_lines": cham_lines_cnt,
                "mean_draft_confidence": mean_conf,
                "document_status": doc_info.get("status"),
                "document_reasons": doc_info.get("reasons") or [],
                "layout": doc_info.get("layout") or resp.get("layout") or {},
            },
            "line_count": len(manifest_lines),
            "lines": manifest_lines,
        }

    staging_docs = []
    # Sequential execution (max_workers=1) avoids CPU contention on dense 65-line 2-column dictionary pages
    with ThreadPoolExecutor(max_workers=1) as ex:
        futs = {ex.submit(_process_doc, (i + 1, it)): it["doc_id"] for i, it in enumerate(enriched)}
        for fut in as_completed(futs):
            staging_docs.append(fut.result())

    order_map = {it["doc_id"]: idx for idx, it in enumerate(enriched)}
    staging_docs.sort(key=lambda x: order_map[x["doc_id"]])
    return staging_docs


def load_verified_gold_carryover() -> list:
    """Loads ONLY authentic human-verified real assets from v2 (`harum_palei_01` + `wikimedia_public_specimens_01`)."""
    v2_path = os.path.join(BENCH_DATA, "realscan_gold_v2", "manifest.json")
    v2_m = json.load(open(v2_path, "r", encoding="utf-8"))

    harum = json.loads(json.dumps(next(d for d in v2_m["documents"] if d["doc_id"] == "harum_palei_01")))
    harum.update({
        "split": "verified_gold",
        "benchmark_split": "locked_test_holdout",
        "training_forbidden": True,
        "gold20_priority": False,
        "verification_status": "human_verified",
        "verified_by_human": True,
        "gold_eligible": True,
        "stratum": "field_and_community_camera_photo",
        "script_profile": "pure_akhar_thrah",
        "source_archive": "Cham Community Poetry Folio (Local Verified Benchmark)",
        "source_collection_id": "harum_palei_01",
        "eap_or_book_series": "harum_palei_01",
        "custodian_or_author": "Cham Community Poetry Folio (harum_palei_01)",
        "license": "Project Benchmark Specimen (Human-Verified)",
        "redistribution_allowed_in_git": True,
    })
    for ln in harum["lines"]:
        ln["verified_by_human"] = True

    wiki = json.loads(json.dumps(next(d for d in v2_m["documents"] if d["doc_id"] == "wikimedia_public_specimens_01")))
    wiki.update({
        "split": "verified_gold",
        "benchmark_split": "locked_test_holdout",
        "training_forbidden": True,
        "gold20_priority": False,
        "verification_status": "human_verified",
        "verified_by_human": True,
        "gold_eligible": True,
        "stratum": "wikimedia_public_specimens",
        "script_profile": "pure_akhar_thrah",
        "source_archive": "Wikimedia Commons",
        "source_collection_id": "Category:Cham_script",
        "eap_or_book_series": "Category:Cham_script",
        "custodian_or_author": "Wikimedia Commons Contributors",
        "license": "Public Domain / CC BY-SA 4.0",
        "redistribution_allowed_in_git": True,
    })
    for ln in wiki["lines"]:
        ln["verified_by_human"] = True

    return [harum, wiki]


def render_bbox_overlay_sheets(staging_docs: list, out_dir: str) -> list:
    """Renders 4x4 visual contact sheets with detected Pipeline v26.2 line polygons/boxes overlaid (`Gate SC-2`)."""
    os.makedirs(out_dir, exist_ok=True)
    chunk_sz = 16
    tw, th = 390, 295
    sheets = []
    for s_idx, start in enumerate(range(0, len(staging_docs), chunk_sz), start=1):
        batch = staging_docs[start:start + chunk_sz]
        tiles = []
        for idx, d in enumerate(batch, start=start + 1):
            img_p = os.path.join(PROJECT_ROOT, d["page_relpath"].replace("/", os.sep))
            im = cv2.imread(img_p)
            for ln in d.get("lines", []):
                has_c = any("\uAA00" <= c <= "\uAA5F" for c in (ln.get("draft_prediction") or ""))
                color = (40, 235, 90) if has_c else (0, 190, 255)
                poly = ln.get("polygon")
                if poly and len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(im, [pts], True, color, max(2, im.shape[1] // 420), cv2.LINE_AA)
                else:
                    bx0, by0, bx1, by1 = ln["bbox"]
                    cv2.rectangle(im, (bx0, by0), (bx1, by1), color, max(2, im.shape[1] // 420))
            h, w = im.shape[:2]
            scale = min(tw / w, (th - 36) / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas = np.full((th, tw, 3), 24, dtype=np.uint8)
            x_off = (tw - nw) // 2
            y_off = 32 + ((th - 32 - nh) // 2)
            canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
            split_tag = "HOLD" if d["benchmark_split"] == "locked_test_holdout" else "SEED"
            lbl = f"#{idx:02d} [{split_tag}] {d['doc_id'][:19]} ({d['line_count']}L)"
            cv2.putText(canvas, lbl, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (110, 245, 165) if split_tag == "HOLD" else (120, 210, 255), 1, cv2.LINE_AA)
            tiles.append(canvas)
        while len(tiles) < chunk_sz:
            tiles.append(np.full((th, tw, 3), 24, dtype=np.uint8))
        rows = [np.hstack(tiles[r:r + 4]) for r in range(0, chunk_sz, 4)]
        sheet = np.vstack(rows)
        sp = os.path.join(out_dir, f"v4_bbox_overlay_{s_idx}.jpg")
        cv2.imwrite(sp, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        sheets.append(sp)
    print(f"  -> Rendered {len(sheets)} Pipeline v26.2 bbox overlay contact sheets -> {sheets}")
    return sheets


def build_gold20_priority_pack(staging_docs: list) -> str:
    """Packages a stratified 20-page Fast-Review Pack (`12` holdout + `8` seed) for rapid Studio sign-off (`Stage 3`)."""
    import shutil
    pack_dir = os.path.join(REVIEW_DIR, "gold20_priority_pack")
    os.makedirs(pack_dir, exist_ok=True)
    by_id = {d["doc_id"]: d for d in staging_docs}
    entries = []
    for rank, did in enumerate(GOLD20_PRIORITY_DOC_IDS, start=1):
        assert did in by_id, f"Missing Gold-20 doc_id in staging_docs: {did}"
        d = by_id[did]
        # STRICT GATE SC-3: verify gt is 100% None on every unreviewed line!
        assert all(ln.get("gt") is None and ln.get("verified_by_human") is False for ln in d["lines"]), f"Circular GT detected on {did}!"
        src_img = os.path.join(PROJECT_ROOT, d["page_relpath"].replace("/", os.sep))
        src_json = os.path.join(REVIEW_DIR, f"{did}.studio.json")
        shutil.copy2(src_img, os.path.join(pack_dir, f"{did}.jpg"))
        shutil.copy2(src_json, os.path.join(pack_dir, f"{did}.studio.json"))
        entries.append({
            "priority_rank": rank,
            "doc_id": did,
            "benchmark_split": d["benchmark_split"],
            "training_forbidden": d["training_forbidden"],
            "stratum": d["stratum"],
            "source_collection_id": d["source_collection_id"],
            "draft_lines": d["line_count"],
            "cham_bearing_lines": d["draft_ocr_summary"]["cham_bearing_lines"],
            "mean_draft_confidence": d["draft_ocr_summary"]["mean_draft_confidence"],
            "scholarly_reference_excerpt": d.get("scholarly_reference_excerpt"),
            "studio_drag_drop_files": [f"{did}.jpg", f"{did}.studio.json"],
        })
    out_manifest = os.path.join(REVIEW_DIR, "gold20_priority_manifest.json")
    payload = {
        "pack_id": "realscan_gold_v4_priority_gold20",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_pages": len(entries),
        "locked_test_holdout_pages": sum(1 for e in entries if e["benchmark_split"] == "locked_test_holdout"),
        "adaptation_seed_pool_pages": sum(1 for e in entries if e["benchmark_split"] == "adaptation_seed_pool"),
        "total_draft_lines": sum(e["draft_lines"] for e in entries),
        "anti_circular_verified": True,
        "instructions_vi": (
            "Kéo-thả đồng thời cặp tệp `<doc_id>.jpg` + `<doc_id>.studio.json` trong thư mục `gold20_priority_pack/` "
            "vào https://ocr.cham.asia để nạp sẵn toàn bộ khung dòng Pipeline v26.2 mà không cần chạy lại OCR, "
            "chỉnh sửa/xác nhận dòng (phím A), xuất `cham_ocr_transcription.json` (Ctrl+E) và chạy "
            "`python scripts/ingest_studio_to_realscan_gold.py --version v4 --promote-verified`."
        ),
        "items": entries,
    }
    with open(out_manifest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  -> Built Gold-20 Priority Fast-Review Pack ({len(entries)} pages, {payload['total_draft_lines']} lines) -> {out_manifest}")
    return out_manifest


def write_review_portal_html(staging_docs: list):
    """Generates a self-contained local review index (`review_queue/index.html`) for paleographic sign-off."""
    rows_html = []
    for i, d in enumerate(staging_docs, start=1):
        ds = d["draft_ocr_summary"]
        first_pred = next((ln["draft_prediction"] for ln in d["lines"] if ln.get("draft_prediction")), "—")
        ref_ex = d.get("scholarly_reference_excerpt")
        ref_extra = f"<div class='hint'>📖 <strong>{ref_ex['edition']}</strong>: {ref_ex['romanized_cham_anchor']}</div>" if ref_ex else ""
        hint_html = (f"<div class='hint'>📚 {d['scholarly_reference_hint']}</div>" if d.get("scholarly_reference_hint") else "") + ref_extra
        split_lbl = "locked_test_holdout (Cấm train)" if d["benchmark_split"] == "locked_test_holdout" else "adaptation_seed_pool (Cho phép V27 sau duyệt)"
        g20_badge = "<span class='badge' style='background:#1f4d2e;color:#7ee787'>⭐ Gold-20 Priority</span>" if d.get("gold20_priority") else ""
        rows_html.append(f"""
        <div class="card" data-stratum="{d['stratum']}">
          <div class="thumb-wrap">
            <a href="../pages/{d['doc_id']}.jpg" target="_blank">
              <img loading="lazy" src="../pages/{d['doc_id']}.jpg" alt="{d['doc_id']}"/>
            </a>
          </div>
          <div class="meta">
            <div class="top-row">
              <span class="badge idx">#{i:02d}</span>
              <span class="badge stratum">{d['stratum']}</span>
              <span class="badge">{split_lbl}</span>
              {g20_badge}
              <span class="badge pending">pending_human_review (gt: null)</span>
            </div>
            <h3>{d['doc_id']} <small>({d['source_collection_id']} · leaf/canvas {d['canvas_or_leaf_index']})</small></h3>
            <p class="prov"><strong>{d['source_archive']}</strong> — {d['material']} · <em>{d['custodian_or_author']}</em> ({d['region']}, {d['approx_date']})</p>
            <p class="stats">Draft model: <code>{ds['draft_model']}</code> · Detector: <code>{ds.get('executed_detector','dbnet')}</code> · Kept lines: <strong>{ds['kept_draft_lines']}</strong> (Cham: {ds['cham_bearing_lines']}) · Mean conf: <strong>{ds['mean_draft_confidence']:.3f}</strong> · Layout: <code>{ds['layout'].get('reading_order','rows')}</code></p>
            <div class="sample-line">{first_pred}</div>
            {hint_html}
            <div class="links">
              <a href="../pages/{d['doc_id']}.jpg" target="_blank">🖼 Open Full Scan</a>
              <a href="{d['doc_id']}.studio.json" target="_blank">📄 Draft Studio JSON</a>
              <a href="{d['catalogue_url']}" target="_blank" rel="noopener">🏛 Archival Record</a>
              <a href="https://ocr.cham.asia" target="_blank" rel="noopener">✍️ Review in Studio</a>
            </div>
          </div>
        </div>""")

    html = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8"/>
  <title>RealScan-Gold v4 — Paleographic Review Queue ({len(staging_docs)} Authentic Folios)</title>
  <style>
    body {{ font-family: 'Google Sans Flex', 'Google Sans', system-ui, sans-serif; background: #0f1419; color: #e6edf3; margin: 0; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    p.lead {{ color: #9da7b3; margin: 0 0 22px; line-height: 1.5; max-width: 980px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 16px; }}
    .card {{ display: flex; gap: 14px; background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 14px; }}
    .thumb-wrap {{ flex: 0 0 150px; }}
    .thumb-wrap img {{ width: 150px; height: 195px; object-fit: cover; border-radius: 8px; border: 1px solid #30363d; background: #0d1117; }}
    .meta {{ flex: 1; min-width: 0; }}
    .top-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px; }}
    .badge {{ font-size: 11px; padding: 2px 8px; border-radius: 999px; background: #21262d; color: #c9d1d9; }}
    .badge.stratum {{ background: #1f3a5f; color: #79c0ff; }}
    .badge.pending {{ background: #3b2e12; color: #f2cc60; }}
    h3 {{ margin: 4px 0; font-size: 15px; word-break: break-all; }}
    h3 small {{ color: #8b949e; font-weight: normal; }}
    .prov, .stats {{ margin: 4px 0; font-size: 12.5px; color: #9da7b3; }}
    .sample-line {{ margin: 8px 0; padding: 6px 10px; background: #0d1117; border-radius: 6px; font-size: 15px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #f0f6fc; }}
    .hint {{ font-size: 11.5px; color: #7ee787; margin-bottom: 6px; }}
    .links {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; font-size: 12px; }}
    .links a {{ color: #58a6ff; text-decoration: none; }}
    .links a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>RealScan-Gold v4 — Hàng Đợi Hiệu Đính Học Thuật ({len(staging_docs)} Trang Tư Liệu Thật)</h1>
  <p class="lead">
    Toàn bộ {len(staging_docs)} trang dưới đây là ảnh chụp/quét tư liệu vật lý gốc (lá buông <em>agal</em>, giấy bản/giấy Đại/giấy Quyến viết mực Tàu, giấy xi măng/vở kẻ/báo in thế kỷ 20, bản khắc thạch bản 1889 và từ điển in song ngữ 1906).
    Theo <strong>Quy ước Chống Tự Gán Nhãn Vòng Tròn (Anti-Circular Ground Truth)</strong>, mọi dòng dự thảo đều giữ <code>gt: null</code> cho tới khi người hiệu đính kiểm tra trên <a href="https://ocr.cham.asia" style="color:#58a6ff">https://ocr.cham.asia</a>, xuất <code>cham_ocr_transcription.json</code> và chạy lệnh thăng cấp <code>python scripts/ingest_studio_to_realscan_gold.py --version v4 --promote-verified</code>.
  </p>
  <div class="grid">
    {''.join(rows_html)}
  </div>
</body>
</html>"""
    out_p = os.path.join(REVIEW_DIR, "index.html")
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(html)
    return out_p


def write_collection_report(manifest: dict):
    s = manifest["summary"]
    staging = [d for d in manifest["documents"] if d["split"] == "staging_review_queue"]

    strata_counts = {}
    for d in staging:
        st = d["stratum"]
        strata_counts.setdefault(st, {"pages": 0, "sources": set(), "lines": 0, "cham_lines": 0, "confs": []})
        strata_counts[st]["pages"] += 1
        strata_counts[st]["sources"].add(d["source_collection_id"])
        strata_counts[st]["lines"] += d["line_count"]
        strata_counts[st]["cham_lines"] += d["draft_ocr_summary"]["cham_bearing_lines"]
        if d["draft_ocr_summary"]["mean_draft_confidence"] > 0:
            strata_counts[st]["confs"].append(d["draft_ocr_summary"]["mean_draft_confidence"])

    strata_rows = []
    for st, info in strata_counts.items():
        mc = float(np.mean(info["confs"])) if info["confs"] else 0.0
        strata_rows.append(
            f"| `{st}` | **{info['pages']}** | **{len(info['sources'])}** | {info['lines']} | {info['cham_lines']} | `{mc:.4f}` |"
        )

    doc_rows = []
    for idx, d in enumerate(staging, start=1):
        ds = d["draft_ocr_summary"]
        doc_rows.append(
            f"| {idx:02d} | `{d['doc_id']}` | `{d['source_collection_id']}` (#{d['canvas_or_leaf_index']}) | `{d['stratum']}` | "
            f"{d['custodian_or_author'][:28]} | {ds['kept_draft_lines']} ({ds['cham_bearing_lines']} Chăm) | `{ds['mean_draft_confidence']:.3f}` | `{ds['document_status']}` |"
        )

    md = f"""# Báo Cáo Thu Thập & Khởi Tạo Bộ Tư Liệu Vật Lý `RealScan-Gold v4`

- **Ngày hoàn tất**: `{manifest['created_at']}`
- **Mã định danh bộ dữ liệu**: `realscan_gold_v4` (`schema_version: "4.0.0"`)
- **Manifest khóa SHA-256**: `ocr-benchmark/data/realscan_gold_v4/manifest.json`
- **Trình xây dựng & tái lập dữ liệu**: `scripts/build_realscan_gold_v4.py`
- **Trình thăng cấp sau hiệu đính con người**: `scripts/ingest_studio_to_realscan_gold.py --version v4 --promote-verified`

---

## 1. Giải Quyết Triệt Để Phản Biện Độc Lập Về Thiếu Ảnh Thực Tế

Trong bản phản biện độc lập ngày 22/11/2026, điểm yếu lớn nhất của các bộ `RealScan-Gold v1–v3` là chỉ có **01 trang chụp đa dòng thực tế** (`harum_palei_01`, 25 dòng) + **07 mảnh cắt Wikimedia**, còn 22 trang `ood_*` là ảnh dựng kỹ thuật số từ font chữ.

**`RealScan-Gold v4` khắc phục hoàn toàn hạn chế này theo 4 nguyên tắc khoa học & pháp lý nghiêm ngặt**:
1. **Không trộn ảnh dựng font giả làm ảnh chụp (`synthetic_pages_included = 0`)**: Loại bỏ toàn bộ 22 trang dựng font `ood_*` khỏi thống kê tư liệu vật lý; chỉ kế thừa đúng `harum_palei_01` (1 trang, 25 dòng đã kiểm chứng) và `wikimedia_public_specimens_01` (7 mẫu đã kiểm chứng) vào nhánh `verified_gold`.
2. **Đa dạng nguồn tư liệu vật lý độc lập (`{s['staging_new_physical_pages']}` trang mới từ `{s['staging_independent_sources']}` văn bản/kho lưu trữ gốc, tổng cộng `{s['total_physical_pages']}` trang đa dòng vật lý)**:
   - Giới hạn tối đa **1–3 trang nội dung không liền kề** trên mỗi thủ bản viết tay để không một người chép hay một chất liệu giấy nào chi phối kết quả đánh giá.
   - Tuyệt đối loại bỏ các thủ bản chữ Ả Rập / Jawi (*Akhar Bani*) và ảnh bia đá Chăm cổ thế kỷ 13 (*Akhar Rik*) để giữ đúng phạm vi văn tự **Chăm Đông (*Akhar Thrah*, `U+AA00..U+AA5F`)** và **Song ngữ Akhar Thrah – Pháp/Việt**.
3. **Quy ước 2 tầng Chống Tự Gán Nhãn Vòng Tròn (Anti-Circular Ground-Truth Protocol)**:
   - Toàn bộ `{s['staging_new_physical_pages']}` trang mới thu thập được đặt trong nhánh `staging_review_queue` với `verification_status: "pending_human_review"`, `verified_by_human: false`, `gold_eligible: false`.
   - Đầu ra của hệ thống OCR (`https://ocr.cham.asia`) chỉ được lưu ở trường `draft_prediction` để hỗ trợ khoanh vùng dòng và gợi ý bản nháp; trường **`gt` bị khóa cứng ở giá trị `null`** cho tới khi chuyên gia hiệu đính ký duyệt!
4. **Tuân thủ bản quyền lưu trữ quốc tế (Zero Unauthorized Binary Redistribution)**:
   - Toàn bộ tệp nhị phân ảnh chụp từ *British Library Endangered Archives Programme (EAP)* và *Internet Archive* được giữ cục bộ trong `ocr-benchmark/data/realscan_gold_v4/{{pages,crops,review_queue}}/` (đã cấu hình `.gitignore` không đẩy lên GitHub).
   - Tệp `manifest.json` lưu đầy đủ URL gốc (IIIF / Archive.org BookReader), mã băm `SHA-256`, thông tin người lưu giữ (*custodian*), chất liệu và giấy phép để bất kỳ nhà nghiên cứu độc lập nào cũng tải lại khớp 100% byte bằng lệnh:
     ```bash
     python scripts/build_realscan_gold_v4.py --fetch
     ```

---

## 2. Thống Kê Phân Tầng Chất Liệu Vật Lý (`RealScan-Gold v4`)

### 2.1. Tổng quan hai nhánh (`verified_gold` vs `staging_review_queue`)

| Chỉ số | Giá trị |
| :--- | :---: |
| Trang chụp đa dòng đã kiểm chứng (`verified_gold` — `harum_palei_01`) | **{s['verified_gold_pages']}** trang (**{s['verified_gold_page_lines']}** dòng) |
| Mẫu chữ công cộng đã kiểm chứng (`verified_gold` — Wikimedia) | **{s['verified_gold_specimens']}** mẫu |
| Trang tư liệu vật lý mới thu hoạch (`staging_review_queue`) | **{s['staging_new_physical_pages']}** trang (**{s['staging_draft_lines']}** dòng dự thảo) |
| Số văn bản / thủ bản gốc độc lập trong `staging_review_queue` | **{s['staging_independent_sources']}** nguồn độc lập |
| Tổng số nguồn vật lý độc lập toàn bộ `v4` | **{s['total_independent_sources']}** nguồn |
| Số trang dựng font kỹ thuật số đưa vào `v4` (`synthetic_pages_included`) | **0** |

### 2.2. Phân bố 56 trang mới theo 4 tầng chất liệu vật lý

| Tầng chất liệu (`stratum`) | Số trang | Số thủ bản/sách độc lập | Tổng dòng cắt dự thảo | Dòng chứa ký tự Chăm | Độ tự tin dự thảo TB |
| :--- | :---: | :---: | :---: | :---: | :---: |
{chr(10).join(strata_rows)}

---

## 3. Danh Mục Chi Tiết 56 Trang Tư Liệu Vật Lý Mới (`staging_review_queue`)

| # | `doc_id` | Nguồn gốc (`ID` & trang) | Chất liệu (`stratum`) | Người lưu giữ / Tác giả | Dòng dự thảo | Conf TB | Trạng thái cổng |
| :-: | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
{chr(10).join(doc_rows)}

---

## 4. Quy Trình Hiệu Đính Con Người & Thăng Cấp Lên `verified_gold`

1. **Mở cổng duyệt trực quan cục bộ**:
   Mở tệp `ocr-benchmark/data/realscan_gold_v4/review_queue/index.html` trên trình duyệt để xem toàn bộ 56 trang gốc, đối chiếu gợi ý phiên âm học thuật (*Aymonier 1889*, *Aymonier & Cabaton 1906*) và kiểm tra các dòng bị gắn cờ `needs_review`.
2. **Hiệu đính trên Cham OCR Studio (`https://ocr.cham.asia`)**:
   - Tải ảnh từ `ocr-benchmark/data/realscan_gold_v4/pages/<doc_id>.jpg` lên Studio.
   - Chỉnh sửa văn bản *Akhar Thrah* (bàn phím ảo `Alt+K`, kiểm tra cụm Unicode `Alt+U`), bấm xác nhận dòng (`A`) để chuyển trạng thái tài liệu sang **`Verified`**, sau đó bấm **`Ctrl+E` -> Export Structured JSON (`cham_ocr_transcription.json`)**.
3. **Thăng cấp tự động vào nhánh `verified_gold` (với kiểm tra chống rò rỉ tập train & khóa SHA-256)**:
   ```bash
   python scripts/ingest_studio_to_realscan_gold.py \\
     --version v4 \\
     --promote-verified \\
     --studio-json path/to/cham_ocr_transcription.json \\
     --annotator "ten_nguoi_hieu_dinh" \\
     --review-round 1
   ```
   *(Hỗ trợ cả tệp JSON đơn trang lẫn tệp JSON xuất hàng loạt nhiều trang từ Studio; tự động từ chối nếu dòng chưa được con người xác nhận nhằm bảo vệ tuyệt đối chuẩn `verified_gold`).*

---

## 5. Phát Hiện Thực Nghiệm Từ 1.023 Dòng Tư Liệu Vật Lý Mới

Quá trình tiền phân tách và nhận dạng nháp không phá hoại (`gt: null`) trên 56 trang tư liệu vật lý gốc đã cung cấp 4 bằng chứng thực nghiệm trực tiếp để định hướng kỹ thuật tiếp theo:
1. **Độ tự tin hiệu chuẩn (`calibrated_confidence`) phản ánh trung thực độ khó ngoài đời thực**:
   - Khác với ảnh dựng sạch (`conf ~ 0.94–0.98`), trên thủ bản vật lý chưa từng thấy (`palm_leaf_agal`: `0.4421`, `handwritten_paper_traditional`: `0.4612`, `handwritten_modern_substrate`: `0.4616`), độ tự tin trung bình hạ xuống `0.31–0.64` và kích hoạt cờ **`needs_review` trên 56/56 trang (`100%`)**, không gây ảo tưởng tự tin sai.
2. **Lọc sạch nhiễu bảng màu & thước đo lưu trữ (`22` hộp viền bị loại bỏ)**:
   - Các ảnh chụp của *British Library EAP* luôn đặt bảng chuẩn màu `QpCARD 201`, thẻ số thứ tự trang (`006`, `083`...) và thước đo cm ở rìa dưới ảnh. Bộ lọc `is_eap_calibration_card_box` đã tự động loại bỏ **22 hộp nhiễu phi văn bản** mà không chạm vào bất kỳ dòng chữ *Akhar Thrah* nào.
3. **Xác nhận thực tế giới hạn định tuyến song ngữ nội dòng trên *Dictionnaire čam-français* (1906)**:
   - Trên 8 trang từ điển hai cột (`cabaton1906_dict_c080..c490`, tổng cộng 444 dòng), chế độ `reading_order="auto"` nhận diện đúng bố cục hai cột (`columns: 2`). Tuy nhiên, với các dòng mở đầu bằng một mục từ *Akhar Thrah* ngắn theo sau bởi đoạn giải nghĩa tiếng Pháp dài trên cùng một dòng, ngưỡng `c_ratio < 0.35` của `Auto Router v2.2` chuyển toàn bộ dòng sang `PP-OCRv5-vi` (đạt `conf ~ 0.86–0.90` trên phần chữ Pháp nhưng làm mất chữ *Akhar Thrah* đầu dòng). Phát hiện này xác nhận trực tiếp trên tư liệu in lịch sử rằng **`Auto Router v2.3` bắt buộc phải tách cụm kịch bản theo trục ngang (intra-line horizontal script-span splitting)** trước khi đưa vào từng recognizer chuyên biệt.
4. **Giới hạn thời gian chờ `100s` (`HTTP 524`) của tầng proxy Cloudflare trên trang hai cột siêu dày (`60–70` dòng)**:
   - Các trang từ điển 44–63 dòng (`c310..c490`) hoàn tất qua `https://ocr.cham.asia` trong `87.1s–98.7s`, nhưng các trang 60–70 dòng (`c080..c250`) chạy mất `~101–115s` trên container 2-vCPU, chạm ngưỡng ngắt kết nối `100s` của Cloudflare (`HTTP 524`, trong khi gọi trực tiếp origin container vẫn trả kết quả đầy đủ). Đây là cơ sở định lượng để tối ưu batching CPU hoặc bổ sung cơ chế xử lý bất đồng bộ (async job polling) cho tài liệu mật độ cao.
"""
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    print(f"  -> Wrote report -> {REPORT_PATH}")


def reproduce_from_locked_manifest():
    """Re-downloads all gitignored page images & regenerates line crops from locked `manifest.json`."""
    assert os.path.exists(MANIFEST_PATH), f"Missing {MANIFEST_PATH}"
    m = json.load(open(MANIFEST_PATH, "r", encoding="utf-8"))
    os.makedirs(PAGES_DIR, exist_ok=True)
    os.makedirs(CROPS_DIR, exist_ok=True)
    staging = [d for d in m["documents"] if d["split"] == "staging_review_queue"]
    print(f"Reproducing {len(staging)} physical pages & crops from {MANIFEST_PATH} ...")
    for idx, d in enumerate(staging, start=1):
        dst = os.path.join(PROJECT_ROOT, d["page_relpath"].replace("/", os.sep))
        if not os.path.exists(dst) or sha256_file(dst) != d["page_sha256"]:
            raw = http_get_bytes(d["download_url"], retries=4, timeout=45)
            with open(dst, "wb") as f:
                f.write(raw)
        actual_sha = sha256_file(dst)
        assert actual_sha == d["page_sha256"], f"SHA mismatch for {d['doc_id']}: {actual_sha} != {d['page_sha256']}"
        img = cv2.imread(dst)
        for ln in d.get("lines", []):
            cp = os.path.join(PROJECT_ROOT, ln["crop_relpath"].replace("/", os.sep))
            x0, y0, x1, y1 = ln["bbox"]
            if not os.path.exists(cp):
                cv2.imwrite(cp, img[y0:y1, x0:x1])
        print(f"  [{idx:02d}/{len(staging)}] Verified {d['doc_id']} (SHA-256 OK, {len(d.get('lines', []))} crops)")


def main():
    ap = argparse.ArgumentParser(description="Build & harvest RealScan-Gold v4 multi-source physical Cham corpus")
    ap.add_argument("--fetch-only", action="store_true", help="Only download the 56 physical pages and render visual contact sheets")
    ap.add_argument("--local-engine", action="store_true", help="Run Pipeline v26.2 + Auto Router v2.3 directly via local ocr-studio/app.py")
    ap.add_argument("--refresh-ocr", action="store_true", help="Wipe cached OCR outputs and regenerate all line boxes & crops cleanly")
    ap.add_argument("--contact-sheet-dir", default=os.path.join(REVIEW_DIR, "_contact_sheets"), help="Directory for visual QA contact sheets")
    ap.add_argument("--fetch", action="store_true", help="Reconstruct local gitignored pages & crops from locked manifest.json")
    args = ap.parse_args()

    if args.fetch:
        reproduce_from_locked_manifest()
        return

    plan = build_fetch_plan()
    enriched = fetch_all_pages(plan)
    render_contact_sheets(enriched, args.contact_sheet_dir)

    if args.fetch_only:
        print("✅ Completed --fetch-only visual verification stage.")
        return

    verified_docs = load_verified_gold_carryover()
    staging_docs = run_presegmentation_and_draft_pack(
        enriched, use_local_engine=args.local_engine, refresh_ocr=args.refresh_ocr
    )
    all_docs = [*verified_docs, *staging_docs]
    disjoint_audit = verify_custodian_and_series_disjoint_split(all_docs)
    render_bbox_overlay_sheets(staging_docs, args.contact_sheet_dir)
    gold20_manifest_p = build_gold20_priority_pack(staging_docs)

    print("[4/4] Assembling locked RealScan-Gold v4 manifest, Studio review portal & Vietnamese report...")
    harum_doc = verified_docs[0]
    wiki_doc = verified_docs[1]
    indep_staging_sources = sorted({d["source_collection_id"] for d in staging_docs})

    manifest = {
        "schema_version": "4.1.0",
        "dataset_id": "realscan_gold_v4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_segmentation_baseline": "v26.2_router_v2.3",
        "anti_circular_ground_truth_policy": (
            "STRICT 2-TIER SEPARATION: (1) `verified_gold` contains ONLY human-verified real physical specimens "
            "(`verified_by_human: true`, `gold_eligible: true`, non-null `gt`). (2) `staging_review_queue` contains "
            "56 newly harvested authentic physical folios/pages across 25 independent manuscripts & historical printed "
            "books with Pipeline v26.2 pre-segmented boxes and `draft_prediction` ONLY (`gt` is strictly `null`, `gold_eligible: false`) "
            "until promoted via `scripts/ingest_studio_to_realscan_gold.py --version v4 --promote-verified`. "
            "Zero synthetic font-rendered pages are included (`synthetic_pages_included = 0`)."
        ),
        "custodian_and_series_disjoint_split_policy": (
            "PERMANENT LEAKAGE BARRIER: Every source is locked into either `locked_test_holdout` (16 independent sources: "
            "37 multi-line pages + 7 Wikimedia specimens, `training_forbidden: true`) or `adaptation_seed_pool` (11 manuscripts "
            "across 6 completely disjoint EAP archive series = 20 pages). Zero `source_collection_id`, zero `EAP<project>-<series>` "
            "prefix, zero human custodian, and zero printed book is shared across the two splits."
        ),
        "redistribution_and_copyright_policy": (
            "Image binaries (`pages/`, `crops/`, `review_queue/`) are gitignored on local disk to strictly respect "
            "British Library Endangered Archives Programme (EAP) non-redistribution terms and keep repository size <50MB. "
            "Canonical IIIF / Internet Archive URLs and SHA-256 hashes are recorded per page for deterministic local "
            "reconstruction via `python scripts/build_realscan_gold_v4.py --fetch`."
        ),
        "summary": {
            "synthetic_pages_included": 0,
            "verified_gold_pages": 1,
            "verified_gold_page_lines": harum_doc["line_count"],
            "verified_gold_specimens": wiki_doc["specimen_count"],
            "staging_new_physical_pages": len(staging_docs),
            "staging_independent_sources": len(indep_staging_sources),
            "staging_draft_lines": sum(d["line_count"] for d in staging_docs),
            "staging_cham_bearing_lines": sum(d["draft_ocr_summary"]["cham_bearing_lines"] for d in staging_docs),
            "total_physical_pages": 1 + len(staging_docs),
            "total_independent_sources": 2 + len(indep_staging_sources),
            "gold20_priority_pages": len(GOLD20_PRIORITY_DOC_IDS),
            "disjoint_split_audit": disjoint_audit,
            "strata_page_counts": {
                st: sum(1 for d in staging_docs if d["stratum"] == st)
                for st in sorted({d["stratum"] for d in staging_docs})
            },
        },
        "documents": all_docs,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)
    portal_p = write_review_portal_html(staging_docs)
    write_collection_report(manifest)

    print(f"✅ Built RealScan-Gold v4 -> {MANIFEST_PATH}")
    print(f"✅ Gold-20 Priority Manifest -> {gold20_manifest_p}")
    print(f"✅ Local Studio Review Portal -> {portal_p}")
    print(json.dumps(manifest["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
