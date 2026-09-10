#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paragraph & Sentence-Flow Reconstruction Module for Cham OCR.
Automatically distinguishes soft line continuations (câu xuống dòng thuộc câu trên)
from hard paragraph/stanza breaks based on Cham linguistic rules and layout geometry.
"""

import re
import numpy as np

# Cham sentence terminating punctuation
CHAM_TERMINATORS = ('꩞', '꩝', '?', '!')
CHAM_SECTION_MARK = '꩞'
CHAM_DANDA = '꩝'

# Regex for Cham stanza/section numbering at line start: e.g. ꩑꩞, ꩒꩞, ꩓꩞꩑꩞, 1., – , • 
STANZA_START_PATTERN = re.compile(r'^(?:[꩐-꩙0-9]+[꩞\.:\s–\-])|(?:^[–•\-])')

def is_hard_break(curr_line, next_line, layout_stats):
    """
    Decides whether the boundary between curr_line and next_line is a HARD BREAK (new paragraph/sentence)
    or a SOFT CONTINUATION (câu xuống dòng thuộc câu trên).
    
    Returns:
        bool: True if HARD BREAK (new paragraph), False if SOFT CONTINUATION (same sentence flow).
        str: Reason for the decision.
    """
    curr_text = curr_line.get("text", "").strip()
    next_text = next_line.get("text", "").strip()
    
    if not curr_text:
        return True, "empty_curr"
    if not next_text:
        return True, "empty_next"

    # 1. Punctuation rule: Line ends with Cham Section Mark (꩞) or Danda (꩝) or question mark (?)
    if curr_text.endswith(CHAM_TERMINATORS):
        return True, "cham_terminator"

    # 2. Line ends with colon ':' (e.g. heading or introductory list)
    if curr_text.endswith(':'):
        return True, "colon_intro"

    # 3. Next line starts with Stanza/Section number or list bullet (e.g. ꩑꩞, ꩒꩞, ꩓꩞꩑꩞, – )
    if STANZA_START_PATTERN.search(next_text):
        return True, "next_is_stanza_heading"

    # 4. Geometric rule: Line gap comparison
    curr_bbox = curr_line.get("bbox", [0, 0, 0, 0])
    next_bbox = next_line.get("bbox", [0, 0, 0, 0])
    
    curr_y2 = curr_bbox[3]
    next_y1 = next_bbox[1]
    curr_gap = next_y1 - curr_y2

    median_gap = layout_stats.get("median_gap", 15.0)
    if median_gap > 0 and curr_gap >= 1.35 * median_gap:
        return True, "large_inter_paragraph_gap"

    # 5. Geometric rule: Next line has significant paragraph indentation
    left_margin = layout_stats.get("left_margin", 0.0)
    next_x1 = next_bbox[0]
    if (next_x1 - left_margin) >= 25.0:
        return True, "next_line_indented"

    # 6. Geometric rule: Current line is significantly short (< 65% page content width)
    curr_w = curr_bbox[2] - curr_bbox[0]
    max_line_w = layout_stats.get("max_line_w", 800.0)
    if max_line_w > 0 and curr_w < 0.65 * max_line_w:
        # If short line and either ends with punctuation or gap is noticeably wider
        if curr_text[-1] in ('꩝', '꩞', '.', ':', ';', ',', '…'):
            return True, "short_ending_line_with_punct"
        if curr_gap >= 1.15 * median_gap:
            return True, "short_ending_line_wider_gap"

    # Otherwise: Soft continuation (sentence wraps to next line)
    return False, "soft_continuation"

def reconstruct_paragraph_flow(detected_lines):
    """
    Takes a list of detected lines and reconstructs them into coherent paragraphs / continuous sentences.
    
    Args:
        detected_lines (list): List of dicts, each having at least:
            - 'text': string
            - 'bbox': [x1, y1, x2, y2]
            - 'confidence': float (optional)
            
    Returns:
        paragraphs (list of dicts): Reconstructed paragraphs with merged text, bounding box, and source line indices.
        boundaries (list of dicts): Step-by-step decision log for each boundary.
    """
    if not detected_lines:
        return [], []

    # Sort lines vertically
    sorted_lines = sorted(detected_lines, key=lambda l: (l.get("bbox", [0, 0, 0, 0])[1] + l.get("bbox", [0, 0, 0, 0])[3]) / 2.0)
    n = len(sorted_lines)

    # Compute layout statistics
    heights = []
    gaps = []
    widths = []
    x_lefts = []

    for i in range(n):
        b = sorted_lines[i].get("bbox", [0, 0, 0, 0])
        w = max(0.0, b[2] - b[0])
        h = max(0.0, b[3] - b[1])
        heights.append(h)
        widths.append(w)
        x_lefts.append(b[0])
        if i < n - 1:
            next_b = sorted_lines[i+1].get("bbox", [0, 0, 0, 0])
            g = next_b[1] - b[3]
            if g > 0:
                gaps.append(g)

    layout_stats = {
        "median_h": float(np.median(heights)) if heights else 25.0,
        "median_gap": float(np.median(gaps)) if gaps else 15.0,
        "max_line_w": float(np.percentile(widths, 90)) if widths else 700.0,
        "left_margin": float(np.percentile(x_lefts, 15)) if x_lefts else 40.0
    }

    paragraphs = []
    boundaries = []

    current_para_lines = [sorted_lines[0]]
    current_line_indices = [0]

    for i in range(n - 1):
        curr_l = sorted_lines[i]
        next_l = sorted_lines[i + 1]

        is_break, reason = is_hard_break(curr_l, next_l, layout_stats)
        boundaries.append({
            "boundary_index": i,
            "curr_line_idx": i,
            "next_line_idx": i + 1,
            "is_hard_break": is_break,
            "is_continuation": not is_break,
            "reason": reason
        })

        if is_break:
            # Finalize current paragraph
            para_text = " ".join(l.get("text", "").strip() for l in current_para_lines if l.get("text", "").strip())
            all_bboxes = [l.get("bbox", [0, 0, 0, 0]) for l in current_para_lines]
            p_x1 = min(b[0] for b in all_bboxes)
            p_y1 = min(b[1] for b in all_bboxes)
            p_x2 = max(b[2] for b in all_bboxes)
            p_y2 = max(b[3] for b in all_bboxes)
            avg_conf = float(np.mean([l.get("confidence", 1.0) for l in current_para_lines]))

            paragraphs.append({
                "paragraph_id": len(paragraphs),
                "text": para_text,
                "line_indices": current_line_indices,
                "num_lines": len(current_para_lines),
                "bbox": [p_x1, p_y1, p_x2, p_y2],
                "confidence": round(avg_conf, 4)
            })

            # Start new paragraph
            current_para_lines = [next_l]
            current_line_indices = [i + 1]
        else:
            # Soft continuation: accumulate line into current paragraph
            current_para_lines.append(next_l)
            current_line_indices.append(i + 1)

    # Finalize last paragraph
    if current_para_lines:
        para_text = " ".join(l.get("text", "").strip() for l in current_para_lines if l.get("text", "").strip())
        all_bboxes = [l.get("bbox", [0, 0, 0, 0]) for l in current_para_lines]
        p_x1 = min(b[0] for b in all_bboxes)
        p_y1 = min(b[1] for b in all_bboxes)
        p_x2 = max(b[2] for b in all_bboxes)
        p_y2 = max(b[3] for b in all_bboxes)
        avg_conf = float(np.mean([l.get("confidence", 1.0) for l in current_para_lines]))

        paragraphs.append({
            "paragraph_id": len(paragraphs),
            "text": para_text,
            "line_indices": current_line_indices,
            "num_lines": len(current_para_lines),
            "bbox": [p_x1, p_y1, p_x2, p_y2],
            "confidence": round(avg_conf, 4)
        })

    return paragraphs, boundaries
