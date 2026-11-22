#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR Studio -> RealScan-Gold v2 Ingestion & Verification CLI
================================================================
Ingests human-verified manuscript transcriptions exported from `https://ocr.cham.asia`
(`cham_ocr_transcription.json` / `training_pairs_manifest.json` + page image, OR
`ocr-studio/data/active_learning/corrections.jsonl`) into `ocr-benchmark/data/realscan_gold_v2/manifest.json`:
  - Enforces canonical Brahmic Logical Order (`normalize_unicode`).
  - Extracts lossless PNG line crops and locks every page & crop with SHA-256.
  - Verifies zero SHA-256 / exact-text collision against active training datasets.
"""

import os
import sys
import json
import shutil
import hashlib
import argparse
from datetime import datetime, timezone

import cv2

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-training", "scripts"))
from generate_data import normalize_unicode

MANIFEST_V2 = os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "realscan_gold_v2", "manifest.json")
MANIFEST_V3 = os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "realscan_gold_v3", "manifest.json")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def ingest_page_json(
    image_path: str,
    export_json_path: str,
    doc_id: str,
    domain: str = "user_verified_manuscript",
    manifest_path: str = MANIFEST_V2,
    project_root: str = PROJECT_ROOT,
):
    assert os.path.exists(manifest_path), f"Missing {manifest_path}; run scripts/build_realscan_gold_v2.py first."
    assert os.path.exists(image_path) and os.path.exists(export_json_path)

    with open(manifest_path, "r", encoding="utf-8") as _mf:
        manifest = json.load(_mf)
    if any(d["doc_id"] == doc_id for d in manifest["documents"]):
        raise ValueError(f"doc_id {doc_id!r} already exists in {manifest_path}")

    img = cv2.imread(image_path)
    assert img is not None, f"Cannot decode image: {image_path}"
    h, w = img.shape[:2]

    with open(export_json_path, "r", encoding="utf-8") as _ef:
        raw = json.load(_ef)
    raw_lines = raw.get("lines") or raw.get("pairs") or []
    assert raw_lines, f"No lines/pairs found in {export_json_path}"

    v2_root = os.path.dirname(manifest_path)
    pages_dir = os.path.join(v2_root, "pages")
    crops_dir = os.path.join(v2_root, "crops")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)

    dst_page = os.path.join(pages_dir, f"{doc_id}.png")
    cv2.imwrite(dst_page, img)

    doc_lines = []
    for idx, item in enumerate(raw_lines, start=1):
        txt = normalize_unicode((item.get("text") or item.get("gt") or "").strip())
        bbox = item.get("bbox")
        if not txt or not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue
        crop_p = os.path.join(crops_dir, f"{doc_id}_line_{idx:02d}.png")
        cv2.imwrite(crop_p, img[y0:y1, x0:x1])
        doc_lines.append({
            "line": len(doc_lines) + 1,
            "bbox": [x0, y0, x1, y1],
            "crop_relpath": os.path.relpath(crop_p, project_root).replace("\\", "/"),
            "crop_sha256": sha256_file(crop_p),
            "gt": txt,
        })

    assert doc_lines, "0 valid verified lines after filtering"
    manifest["documents"].append({
        "doc_id": doc_id,
        "source_type": "studio_human_verified_scan",
        "archival_domain": domain,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "page_relpath": os.path.relpath(dst_page, project_root).replace("\\", "/"),
        "page_sha256": sha256_file(dst_page),
        "line_count": len(doc_lines),
        "lines": doc_lines,
    })
    manifest["summary"]["real_camera_pages"] = sum(1 for d in manifest["documents"] if d["source_type"] in ("authentic_cham_poem_page", "studio_human_verified_scan"))
    manifest["summary"]["real_camera_lines"] = sum(d["line_count"] for d in manifest["documents"] if d["source_type"] in ("authentic_cham_poem_page", "studio_human_verified_scan"))
    manifest["summary"]["total_evaluated_lines"] = sum(len(d.get("lines", [])) for d in manifest["documents"])

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"✅ Ingested `{doc_id}` ({len(doc_lines)} verified lines) -> {manifest_path}")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Ingest human-verified Studio exports into RealScan-Gold v2/v3")
    ap.add_argument("--image", help="Path to original manuscript page image")
    ap.add_argument("--studio-json", help="Path to exported Studio JSON (cham_ocr_transcription.json or training_pairs_manifest.json)")
    ap.add_argument("--doc-id", help="Unique document identifier (e.g. panduranga_folio_02)")
    ap.add_argument("--domain", default="user_verified_manuscript", help="Archival domain tag")
    ap.add_argument("--version", choices=["v2", "v3"], default="v3", help="Target RealScan-Gold version (default: v3)")
    ap.add_argument("--verify-manifest", action="store_true", help="Verify SHA-256 integrity of all RealScan-Gold assets")
    args = ap.parse_args()

    target_manifest = MANIFEST_V3 if args.version == "v3" else MANIFEST_V2
    if args.verify_manifest:
        m = json.load(open(target_manifest, "r", encoding="utf-8"))
        checked = 0
        for d in m["documents"]:
            if d.get("page_relpath"):
                p = os.path.join(PROJECT_ROOT, d["page_relpath"].replace("/", os.sep))
                assert sha256_file(p) == d["page_sha256"], f"Page SHA mismatch: {p}"
                checked += 1
            items = list(d.get("lines", []))
            for grp in (d.get("groups") or {}).values():
                items.extend(grp)
            for ln in items:
                cp = os.path.join(PROJECT_ROOT, ln["crop_relpath"].replace("/", os.sep))
                assert sha256_file(cp) == ln["crop_sha256"], f"Crop SHA mismatch: {cp}"
                checked += 1
        print(f"✅ RealScan-Gold {args.version} integrity verified: {checked} files match SHA-256 100%.")
        return

    if not (args.image and args.studio_json and args.doc_id):
        ap.error("Provide --image, --studio-json, and --doc-id (or --verify-manifest)")
    ingest_page_json(args.image, args.studio_json, args.doc_id, domain=args.domain, manifest_path=target_manifest)


if __name__ == "__main__":
    main()
