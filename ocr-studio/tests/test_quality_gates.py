"""Regression tests for the 2026-11-04 stress-test fixes (blank page, reading order, review gates).

Run:  python ocr-studio/tests/test_quality_gates.py
Only needs numpy + opencv (no PaddlePaddle models are loaded).
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402


def _meta(text, conf, bbox):
    return {'line_id': 0, 'bbox': bbox, 'core_box': [bbox[1], bbox[3]], 'selected_crop_type': 'dbnet',
            'candidates': {'dbnet': {'prediction': text, 'confidence': conf}}}


def test_blank_page_is_detected():
    blank = np.full((460, 1500, 3), 250, np.uint8)
    noise = np.clip(blank.astype(np.int16) + np.random.RandomState(0).randint(-6, 7, blank.shape), 0, 255).astype(np.uint8)
    assert app.is_blank_page(app.analyze_text_ink(blank))
    assert app.is_blank_page(app.analyze_text_ink(noise))
    page = blank.copy()
    for i in range(4):
        cv2.putText(page, 'Cham OCR line %d' % i, (40, 80 + 90 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 3)
    assert not app.is_blank_page(app.analyze_text_ink(page))


def test_uncovered_ink_ratio():
    page = np.full((400, 900, 3), 245, np.uint8)
    for i in range(4):
        cv2.putText(page, 'line number %d text' % i, (30, 70 + 90 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (10, 10, 10), 3)
    ink = app.analyze_text_ink(page)
    boxes = [np.array([[20, 30 + 90 * i], [880, 30 + 90 * i], [880, 85 + 90 * i], [20, 85 + 90 * i]], np.float32) for i in range(4)]
    full, _ = app.compute_uncovered_ink(ink, boxes, 55)
    half, _ = app.compute_uncovered_ink(ink, boxes[:2], 55)
    assert full < 0.02, full
    assert 0.35 < half < 0.65, half


def test_two_column_reading_order():
    rects = []
    for i in range(8):  # left column
        rects.append((60, 90 + 47 * i, 420, 125 + 47 * i))
    for i in range(8):  # right column, row-aligned with the left column
        rects.append((580, 90 + 47 * i, 890, 125 + 47 * i))
    order, layout = app.order_lines_reading(rects, 1100)
    assert layout == {'reading_order': 'columns', 'columns': 2}, layout
    assert order == list(range(16)), order


def test_title_then_two_columns():
    rects = [(60, 20, 890, 60)]  # full-width title
    rects += [(60, 90 + 47 * i, 420, 125 + 47 * i) for i in range(6)]
    rects += [(580, 90 + 47 * i, 890, 125 + 47 * i) for i in range(6)]
    order, layout = app.order_lines_reading(rects, 1000)
    assert order == list(range(13)), order
    assert layout['columns'] == 2


def test_single_column_keeps_top_to_bottom():
    rects = [(40 + (i % 3) * 30, 50 + 45 * i, 700 - (i % 2) * 200, 85 + 45 * i) for i in range(10)]
    order, layout = app.order_lines_reading(rects, 800)
    assert order == list(range(10))
    assert layout['reading_order'] == 'rows'


def test_line_quality_gates():
    metas = [_meta('ꨀꨇꨉ ꨆꨮꨭ', 0.95, [10, 10, 500, 50]),
             _meta('', 0.0, [10, 60, 500, 100]),          # empty -> dropped
             _meta('R', 0.12, [10, 110, 500, 150]),        # very low conf -> dropped
             _meta('ꨌꨯꨱꩃ', 0.45, [10, 160, 500, 200]),   # kept, low confidence -> review
             _meta('ꨨꨎꩆ ꨀꨊꨪꩆ', 0.91, [10, 210, 500, 330])]  # kept, merged-height -> review
    kept, crops = app.apply_line_quality_gates(metas, list(range(len(metas))))
    assert crops == [0, 3, 4], crops
    assert kept[0]['needs_review'] is False
    assert 'low_confidence' in kept[1]['reason']
    assert 'possible_merged_lines' in kept[2]['reason']


def test_document_status_never_auto_verified():
    ok_line = {'segmentation': {'needs_review': False}}
    bad_line = {'segmentation': {'needs_review': True}}
    assert app.build_document_status([], ['blank_page'])['status'] == 'no_text'
    assert app.build_document_status([], ['no_reliable_text'])['status'] == 'needs_review'
    assert app.build_document_status([ok_line], [])['status'] == 'ocr_done'
    assert app.build_document_status([ok_line, bad_line], [])['status'] == 'needs_review'
    assert app.build_document_status([ok_line], ['possible_missed_lines'])['status'] == 'needs_review'


def test_expand_quad_pads_along_line_axis():
    quad = np.array([[100, 100], [400, 100], [400, 140], [100, 140]], np.float32)
    out = app.expand_quad(quad, 0.25, 0.0, 1000, 1000)
    assert np.allclose(out[:, 1], [90, 90, 150, 150]), out


def test_merge_line_boxes_preserves_stacked_wavy_lines():
    # Exact raw boxes from test_161_lvl5 (lines 6 & 7 had AABB vertical overlap 0.57 due to wave/tilt)
    raw_boxes = [
        [[89.0, 226.0], [673.0, 226.0], [673.0, 263.0], [89.0, 263.0]],
        [[80.0, 191.0], [967.0, 194.0], [967.0, 243.0], [80.0, 239.0]],
        [[96.0, 178.0], [436.0, 182.0], [436.0, 208.0], [96.0, 204.0]],
        [[90.0, 154.0], [464.0, 155.0], [464.0, 181.0], [90.0, 179.0]],
        [[81.0, 125.0], [396.0, 132.0], [396.0, 158.0], [81.0, 151.0]],
        [[84.0, 102.0], [573.0, 102.0], [573.0, 134.0], [84.0, 134.0]],
        [[81.0, 73.0], [532.0, 79.0], [531.0, 112.0], [81.0, 106.0]],
        [[103.0, 50.0], [434.0, 54.0], [434.0, 80.0], [103.0, 76.0]],
    ]
    merged = app.merge_line_boxes(raw_boxes, 1050, 289)
    assert len(merged) == 8, len(merged)

    # Horizontally separated fragments on the same line should still be merged into one box
    fragments = [
        [[50.0, 100.0], [210.0, 100.0], [210.0, 134.0], [50.0, 134.0]],
        [[235.0, 102.0], [480.0, 102.0], [480.0, 135.0], [235.0, 135.0]],
    ]
    merged_frags = app.merge_line_boxes(fragments, 1000, 300)
    assert len(merged_frags) == 1, len(merged_frags)


def test_merge_line_boxes_keeps_tilted_stacked_lines_apart():
    # Two parallel lines tilted ~3.5 deg on a perspective page: each AABB is ~2.5x the text
    # thickness, so the AABBs overlap ~60% vertically although the lines are 24px apart
    # (pattern of benchmark test_384_a4_lvl5, where AABB merging collapsed 19 lines into 14).
    def tilted(y0, x1=80.0, x2=920.0, slope=0.06, t=22.0):
        return [[x1, y0], [x2, y0 + slope * (x2 - x1)], [x2, y0 + slope * (x2 - x1) + t], [x1, y0 + t]]
    lines = [tilted(140.0), tilted(164.0), tilted(188.0)]
    merged = app.merge_line_boxes(lines, 1000, 400)
    assert len(merged) == 3, len(merged)
    assert app.max_stacked_overlap(lines) > 0.45  # legacy-gate trigger fires on such pages

    # Overlapping fragments of the SAME tilted line (h_overlap > 0.25, shared centerline) must
    # still merge: the centerline guard only separates genuinely stacked lines.
    frags = [tilted(140.0, 80.0, 500.0), tilted(140.0 + 0.06 * 270.0, 350.0, 920.0)]
    assert len(app.merge_line_boxes(frags, 1000, 400)) == 1


def test_dropped_text_lines_are_reported():
    metas = [_meta('ꨀꨇꨉ ꨆꨮꨭ', 0.95, [10, 10, 500, 50]),
             _meta('ꨯꨮꨯꨮ ꨯꨱ', 0.11, [10, 60, 520, 100]),   # full-size unreadable Cham line
             _meta('ꨯ', 0.05, [10, 110, 30, 120]),          # tiny fragment -> plain noise
             _meta('ꨌꨯꨱꩃ ꨀꨊ', 0.90, [10, 160, 480, 200])]
    dropped = []
    kept, crops = app.apply_line_quality_gates(metas, list(range(len(metas))), dropped)
    assert crops == [0, 3], crops
    assert len(dropped) == 1 and dropped[0]['bbox'] == [10, 60, 520, 100]


def test_model_info_always_reports_detector_and_paddle():
    # Blank-page requests return before the detector is needed: model_info must still say so
    # explicitly instead of omitting the detector (post-deploy check 2026-11-16).
    saved = dict(app.det_model_info)
    try:
        app.det_model_info.clear()
        info = app.get_model_info('v24_1_r2_1_6')
        assert info['detector']['loaded'] is False and info['detector']['mode'] == app.DET_MODE
        assert 'paddle_version' in info
    finally:
        app.det_model_info.update(saved)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print('PASS', t.__name__)
    print(f'{len(tests)} tests passed')
