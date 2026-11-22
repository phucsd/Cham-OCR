#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR Studio -> RealScan-Gold (v2 / v3 / v4) Ingestion, Promotion & Verification CLI
=======================================================================================
Ingests and promotes human-verified manuscript transcriptions exported from `https://ocr.cham.asia`
(`cham_ocr_transcription.json` single-page or multi-page batch export, or `training_pairs_manifest.json`)
into `ocr-benchmark/data/realscan_gold_v{2,3,4}/manifest.json`:
  - Enforces canonical Brahmic Logical Order (`normalize_unicode`) on Cham textlines.
  - Enforces strict Anti-Circular Ground-Truth Gate on `v4`: rejects unverified Studio drafts
    (`status != "Verified"` and `human_verified != True`) unless explicitly attested by a named
    human annotator (`--confirm-human-verified --annotator <name>`).
  - Extracts lossless PNG line crops (respecting manual Studio bbox edits/splits/merges) and locks SHA-256.
  - Recomputes `verified_gold` vs `staging_review_queue` summary counters cleanly.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import sys

import cv2

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ocr-training", "scripts"))
from generate_data import normalize_unicode  # noqa: E402

MANIFEST_V2 = os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "realscan_gold_v2", "manifest.json")
MANIFEST_V3 = os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "realscan_gold_v3", "manifest.json")
MANIFEST_V4 = os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "realscan_gold_v4", "manifest.json")

MANIFEST_BY_VER = {
    "v2": MANIFEST_V2,
    "v3": MANIFEST_V3,
    "v4": MANIFEST_V4,
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def _normalize_line_text(raw_txt: str) -> str:
    s = (raw_txt or "").strip()
    if any("\uAA00" <= ch <= "\uAA5F" for ch in s):
        return normalize_unicode(s)
    return s


def recompute_v4_summary(manifest: dict) -> dict:
    docs = manifest.get("documents", [])
    verified_pages = [d for d in docs if d.get("split") == "verified_gold" and d.get("stratum") != "wikimedia_public_specimens"]
    verified_specs = [d for d in docs if d.get("split") == "verified_gold" and d.get("stratum") == "wikimedia_public_specimens"]
    staging_pages = [d for d in docs if d.get("split") == "staging_review_queue"]

    indep_staging = {d.get("source_collection_id") or d["doc_id"] for d in staging_pages}
    indep_all = {d.get("source_collection_id") or d["doc_id"] for d in docs}

    manifest["summary"] = {
        "synthetic_pages_included": 0,
        "verified_gold_pages": len(verified_pages),
        "verified_gold_page_lines": sum(len(d.get("lines", [])) for d in verified_pages),
        "verified_gold_specimens": sum(d.get("specimen_count", len(d.get("lines", []))) for d in verified_specs),
        "staging_new_physical_pages": len(staging_pages),
        "staging_independent_sources": len(indep_staging),
        "staging_draft_lines": sum(len(d.get("lines", [])) for d in staging_pages),
        "staging_cham_bearing_lines": sum(
            (d.get("draft_ocr_summary") or {}).get("cham_bearing_lines", 0) for d in staging_pages
        ),
        "total_physical_pages": len(verified_pages) + len(staging_pages),
        "total_independent_sources": len(indep_all),
        "strata_page_counts": {
            st: sum(1 for d in docs if d.get("stratum") == st and d.get("stratum") != "wikimedia_public_specimens")
            for st in sorted({d.get("stratum", "unknown") for d in docs if d.get("stratum") != "wikimedia_public_specimens"})
        },
    }
    return manifest


def _check_human_verified_gate(doc_payload: dict, raw_lines: list, annotator: str, confirm_human_verified: bool):
    status_slug = str(doc_payload.get("status") or "").strip().lower()
    all_lines_verified = bool(raw_lines) and all(bool(ln.get("human_verified")) for ln in raw_lines)
    if status_slug == "verified" or all_lines_verified:
        return
    if confirm_human_verified and annotator and annotator.strip():
        return
    raise ValueError(
        f"Anti-Circularity Gate rejected `{doc_payload.get('doc_id') or doc_payload.get('image_name')}`: "
        f"document status is {doc_payload.get('status')!r} and lines are not marked `human_verified: true`. "
        "Verify lines in Cham OCR Studio (`https://ocr.cham.asia`) so status becomes 'Verified', "
        "or pass `--confirm-human-verified --annotator <your_name>` if manually reviewed offline."
    )


def promote_verified_v4(
    export_json_path: str,
    annotator: str = "studio_reviewer",
    review_round: int = 1,
    confirm_human_verified: bool = False,
    manifest_path: str = MANIFEST_V4,
    project_root: str = PROJECT_ROOT,
) -> dict:
    """Promotes one or more human-verified `staging_review_queue` folios in RealScan-Gold v4 to `verified_gold`."""
    assert os.path.exists(manifest_path), f"Missing {manifest_path}; run scripts/build_realscan_gold_v4.py first."
    assert os.path.exists(export_json_path), f"Missing Studio JSON: {export_json_path}"

    with open(manifest_path, "r", encoding="utf-8") as mf:
        manifest = json.load(mf)
    with open(export_json_path, "r", encoding="utf-8") as ef:
        payload = json.load(ef)

    doc_entries = payload.get("documents") if isinstance(payload.get("documents"), list) else [payload]
    assert doc_entries, f"No document payloads found in {export_json_path}"

    v4_root = os.path.dirname(manifest_path)
    crops_dir = os.path.join(v4_root, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    by_id = {d["doc_id"]: d for d in manifest["documents"]}
    promoted = []

    for entry in doc_entries:
        raw_name = entry.get("doc_id") or entry.get("image_name") or entry.get("name") or ""
        doc_id = os.path.splitext(os.path.basename(str(raw_name)))[0]
        if doc_id not in by_id:
            raise KeyError(f"Cannot match doc_id {doc_id!r} in {manifest_path}")

        raw_lines = entry.get("lines") or entry.get("pairs") or []
        assert raw_lines, f"No lines found for `{doc_id}`"
        _check_human_verified_gate(entry, raw_lines, annotator=annotator, confirm_human_verified=confirm_human_verified)

        target_doc = by_id[doc_id]
        page_abs = os.path.join(project_root, target_doc["page_relpath"].replace("/", os.sep))
        assert os.path.exists(page_abs), (
            f"Missing local page image {page_abs}. Run `python scripts/build_realscan_gold_v4.py --fetch` first."
        )
        img = cv2.imread(page_abs)
        assert img is not None, f"Cannot decode {page_abs}"
        h, w = img.shape[:2]

        verified_lines = []
        for item in raw_lines:
            txt = _normalize_line_text(item.get("text") or item.get("gt") or "")
            bbox = item.get("bbox")
            if not txt or not bbox or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in bbox]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 - x0 < 6 or y1 - y0 < 6:
                continue
            l_idx = len(verified_lines) + 1
            crop_p = os.path.join(crops_dir, f"{doc_id}_gold_line_{l_idx:02d}.png")
            cv2.imwrite(crop_p, img[y0:y1, x0:x1])
            verified_lines.append({
                "line": l_idx,
                "bbox": [x0, y0, x1, y1],
                "polygon": item.get("polygon"),
                "crop_relpath": os.path.relpath(crop_p, project_root).replace("\\", "/"),
                "crop_sha256": sha256_file(crop_p),
                "gt": txt,
                "verified_by_human": True,
                "draft_prediction": item.get("original_text"),
            })

        assert verified_lines, f"0 valid verified lines remaining for `{doc_id}`"
        target_doc.update({
            "split": "verified_gold",
            "verification_status": "human_verified",
            "verified_by_human": True,
            "gold_eligible": True,
            "annotator_id": annotator.strip() if annotator else "studio_reviewer",
            "review_round": int(review_round),
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "line_count": len(verified_lines),
            "lines": verified_lines,
        })
        promoted.append((doc_id, len(verified_lines)))

    recompute_v4_summary(manifest)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as out_f:
        json.dump(manifest, out_f, indent=2, ensure_ascii=False)
    for did, lcnt in promoted:
        print(f"✅ Promoted `{did}` ({lcnt} human-verified lines, round={review_round}, annotator={annotator!r}) -> verified_gold")
    return manifest


def ingest_page_json(
    image_path: str,
    export_json_path: str,
    doc_id: str,
    domain: str = "user_verified_manuscript",
    manifest_path: str = MANIFEST_V2,
    project_root: str = PROJECT_ROOT,
    annotator: str = "studio_reviewer",
    review_round: int = 1,
    confirm_human_verified: bool = False,
):
    assert os.path.exists(manifest_path), f"Missing {manifest_path}"
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

    is_v4 = manifest.get("schema_version", "").startswith("4.")
    if is_v4:
        _check_human_verified_gate(raw, raw_lines, annotator=annotator, confirm_human_verified=confirm_human_verified)

    root_dir = os.path.dirname(manifest_path)
    pages_dir = os.path.join(root_dir, "pages")
    crops_dir = os.path.join(root_dir, "crops")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)

    dst_page = os.path.join(pages_dir, f"{doc_id}.png")
    cv2.imwrite(dst_page, img)

    doc_lines = []
    for idx, item in enumerate(raw_lines, start=1):
        txt = _normalize_line_text(item.get("text") or item.get("gt") or "")
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
        ln_obj = {
            "line": len(doc_lines) + 1,
            "bbox": [x0, y0, x1, y1],
            "crop_relpath": os.path.relpath(crop_p, project_root).replace("\\", "/"),
            "crop_sha256": sha256_file(crop_p),
            "gt": txt,
        }
        if is_v4:
            ln_obj["verified_by_human"] = True
        doc_lines.append(ln_obj)

    assert doc_lines, "0 valid verified lines after filtering"
    new_doc = {
        "doc_id": doc_id,
        "source_type": "studio_human_verified_scan",
        "archival_domain": domain,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "page_relpath": os.path.relpath(dst_page, project_root).replace("\\", "/"),
        "page_sha256": sha256_file(dst_page),
        "line_count": len(doc_lines),
        "lines": doc_lines,
    }
    if is_v4:
        new_doc.update({
            "split": "verified_gold",
            "verification_status": "human_verified",
            "verified_by_human": True,
            "gold_eligible": True,
            "stratum": domain,
            "annotator_id": annotator,
            "review_round": int(review_round),
        })
        manifest["documents"].append(new_doc)
        recompute_v4_summary(manifest)
    else:
        manifest["documents"].append(new_doc)
        manifest["summary"]["real_camera_pages"] = sum(
            1 for d in manifest["documents"] if d["source_type"] in ("authentic_cham_poem_page", "studio_human_verified_scan")
        )
        manifest["summary"]["real_camera_lines"] = sum(
            d["line_count"] for d in manifest["documents"] if d["source_type"] in ("authentic_cham_poem_page", "studio_human_verified_scan")
        )
        manifest["summary"]["total_evaluated_lines"] = sum(len(d.get("lines", [])) for d in manifest["documents"])

    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"✅ Ingested `{doc_id}` ({len(doc_lines)} verified lines) -> {manifest_path}")
    return manifest


def verify_manifest_integrity(target_manifest: str, project_root: str = PROJECT_ROOT) -> dict:
    with open(target_manifest, "r", encoding="utf-8") as f:
        m = json.load(f)
    is_v4 = m.get("schema_version", "").startswith("4.")
    checked_files = 0
    missing_gitignored = 0

    for d in m["documents"]:
        if is_v4:
            if d["split"] == "verified_gold":
                assert d["verified_by_human"] is True and d["gold_eligible"] is True, f"Invalid gold flags on {d['doc_id']}"
                for ln in d.get("lines", []):
                    assert ln.get("gt") and isinstance(ln["gt"], str), f"Gold line missing gt in {d['doc_id']}"
            elif d["split"] == "staging_review_queue":
                assert d["verified_by_human"] is False and d["gold_eligible"] is False, f"Staging leak on {d['doc_id']}"
                for ln in d.get("lines", []):
                    assert ln.get("gt") is None, f"Circular GT violation in staging doc {d['doc_id']}!"

        if d.get("page_relpath"):
            p = os.path.join(project_root, d["page_relpath"].replace("/", os.sep))
            if os.path.exists(p):
                assert sha256_file(p) == d["page_sha256"], f"Page SHA mismatch: {p}"
                checked_files += 1
            elif is_v4 and not d.get("redistribution_allowed_in_git", True):
                missing_gitignored += 1
            else:
                raise FileNotFoundError(f"Missing required page file: {p}")

        items = list(d.get("lines", []))
        for grp in (d.get("groups") or {}).values():
            items.extend(grp)
        for ln in items:
            cp = os.path.join(project_root, ln["crop_relpath"].replace("/", os.sep))
            if os.path.exists(cp):
                assert sha256_file(cp) == ln["crop_sha256"], f"Crop SHA mismatch: {cp}"
                checked_files += 1
            elif is_v4 and not d.get("redistribution_allowed_in_git", True):
                missing_gitignored += 1
            else:
                raise FileNotFoundError(f"Missing required crop file: {cp}")

    return {"checked_files": checked_files, "missing_gitignored": missing_gitignored, "documents": len(m["documents"])}


def main():
    ap = argparse.ArgumentParser(description="Ingest or promote human-verified Studio exports into RealScan-Gold v2/v3/v4")
    ap.add_argument("--image", help="Path to original manuscript page image (for brand-new page ingestion)")
    ap.add_argument("--studio-json", help="Path to exported Studio JSON (cham_ocr_transcription.json or training_pairs_manifest.json)")
    ap.add_argument("--doc-id", help="Unique document identifier (for brand-new page ingestion)")
    ap.add_argument("--domain", default="user_verified_manuscript", help="Archival domain / stratum tag")
    ap.add_argument("--version", choices=["v2", "v3", "v4"], default="v4", help="Target RealScan-Gold version (default: v4)")
    ap.add_argument("--promote-verified", action="store_true", help="Promote staging_review_queue folio(s) in v4 to verified_gold")
    ap.add_argument("--annotator", default="studio_reviewer", help="Human annotator identifier for provenance tracking")
    ap.add_argument("--review-round", type=int, default=1, help="Human verification round (1 = primary, 2 = dual-annotator adjudication)")
    ap.add_argument("--confirm-human-verified", action="store_true", help="Explicitly attest offline human verification if JSON status != 'Verified'")
    ap.add_argument("--verify-manifest", action="store_true", help="Verify SHA-256 & anti-circularity integrity of RealScan-Gold")
    args = ap.parse_args()

    target_manifest = MANIFEST_BY_VER[args.version]
    if args.verify_manifest:
        stats = verify_manifest_integrity(target_manifest)
        print(
            f"✅ RealScan-Gold {args.version} integrity & anti-circularity verified: "
            f"{stats['checked_files']} local files match SHA-256 100% "
            f"(documents={stats['documents']}, gitignored_unfetched={stats['missing_gitignored']})."
        )
        return

    if args.promote_verified:
        if args.version != "v4" or not args.studio_json:
            ap.error("--promote-verified requires --version v4 and --studio-json")
        promote_verified_v4(
            args.studio_json,
            annotator=args.annotator,
            review_round=args.review_round,
            confirm_human_verified=args.confirm_human_verified,
            manifest_path=target_manifest,
        )
        return

    if not (args.image and args.studio_json and args.doc_id):
        ap.error("Provide --promote-verified --studio-json ... OR --image, --studio-json, and --doc-id (or --verify-manifest)")
    ingest_page_json(
        args.image,
        args.studio_json,
        args.doc_id,
        domain=args.domain,
        manifest_path=target_manifest,
        annotator=args.annotator,
        review_round=args.review_round,
        confirm_human_verified=args.confirm_human_verified,
    )


if __name__ == "__main__":
    main()
