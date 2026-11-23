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


def test_default_cham_model_and_auto_info():
    assert app.DEFAULT_CHAM_MODEL.startswith('v24_1'), app.DEFAULT_CHAM_MODEL
    info = app.get_model_info('auto')
    # When ocr_model_paths is not yet populated in unit tests, ensure get_model_info runs cleanly;
    # when populated, recognizer #0 must match DEFAULT_CHAM_MODEL.
    recs = info.get('recognizers', [])
    if recs:
        assert recs[0]['version'] == app.DEFAULT_CHAM_MODEL, recs
    app.ocr_model_paths.setdefault(
        app.DEFAULT_CHAM_MODEL,
        {'model_dir': os.path.join(app.PROJECT_ROOT, 'data', 'output', f'rec_cham_inference_{app.DEFAULT_CHAM_MODEL}'),
         'dict_path': os.path.join(app.PROJECT_ROOT, 'data', 'cham_dict_v24_1.txt')}
    )
    info2 = app.get_model_info('auto')
    assert info2['recognizers'][0]['version'] == app.DEFAULT_CHAM_MODEL, info2


def test_confidence_calibration_temperature_scaling():
    # Identity at T = 1.0
    assert abs(app.calibrate_line_confidence(0.85, temperature=1.0) - 0.85) < 1e-6
    assert app.calibrate_line_confidence(0.0, temperature=2.0) == 0.0
    assert app.calibrate_line_confidence(1.0, temperature=2.0) == 1.0

    # T > 1.0 softens high confidence (0.95 -> closer to 0.5) and stays strictly monotone
    c_low = app.calibrate_line_confidence(0.60, temperature=1.8)
    c_high = app.calibrate_line_confidence(0.95, temperature=1.8)
    assert 0.50 < c_low < 0.60, c_low
    assert 0.50 < c_high < 0.95, c_high
    assert c_low < c_high

    # T < 1.0 sharpens high confidence (> 0.5 moves closer to 1.0)
    c_sharp = app.calibrate_line_confidence(0.80, temperature=0.5)
    assert c_sharp > 0.80, c_sharp


def test_active_learning_save_correction_and_export_r4():
    import base64
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepare 5 fake line crops across 2 synthetic documents
        imgs_b64 = []
        for idx in range(5):
            arr = np.full((32, 120, 3), 240 - idx * 10, dtype=np.uint8)
            cv2.putText(arr, f'T{idx}', (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
            ok, buf = cv2.imencode('.png', arr)
            assert ok
            imgs_b64.append('data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode('ascii'))

        # Verify Logical Order normalize_unicode on out-of-order Cham text (ꨙꨯꨳꨮ -> ꨙꨳꨯꨮ)
        samples = [
            ('doc_alpha.png', 1, 'ꨙꨯꨳꨮ', 'ꨙꨯꨳꨮ', imgs_b64[0]),
            ('doc_alpha.png', 2, 'ꨨꨣꨭꩌ ꨚꨤꨬ', 'ꨨꨣꨭꩌ', imgs_b64[1]),
            ('doc_beta.png', 1, 'ꨌꩌ ꨚꨤꨬ ꨣꩌ', 'ꨌꩌ ꨚꨤꨬ', imgs_b64[2]),
            ('doc_gamma.png', 1, '꩒꩐꩐꩞ ꨀꨇꩉ', '꩒꩐꩞', imgs_b64[3]),
            ('doc_delta.png', 1, 'ꨓꨶꨮꩉ ꨆꨭꨖꨩ', 'ꨓꨶꨮꩉ', imgs_b64[4]),
        ]
        res0 = None
        for doc_name, lidx, submitted, orig, b64 in samples:
            r = app.save_correction_entry({
                'text': submitted,
                'original_text': orig,
                'confidence': 0.87,
                'calibrated_confidence': 0.81,
                'model': 'v24_1_r2_1_6',
                'bbox': [10, 20, 130, 52],
                'image_name': doc_name,
                'line_idx': lidx,
                'crop_image': b64,
            }, base_dir=tmpdir)
            if res0 is None:
                res0 = r
            assert r['success'] is True

        assert res0['normalized_text'] == 'ꨙꨳꨯꨮ', res0
        txt_lines = open(os.path.join(tmpdir, 'ocr_corrections.txt'), encoding='utf-8').read().splitlines()
        assert len(txt_lines) == 5 and txt_lines[0] == 'ꨙꨳꨯꨮ', txt_lines
        jsonl_path = os.path.join(tmpdir, 'active_learning', 'corrections.jsonl')
        rows = [json.loads(l) for l in open(jsonl_path, encoding='utf-8') if l.strip()]
        assert len(rows) == 5
        assert rows[0]['text'] == 'ꨙꨳꨯꨮ'
        assert rows[0]['crop_relpath'].startswith('active_learning/crops/')
        assert os.path.exists(os.path.join(tmpdir, rows[0]['crop_relpath']))

        # Test export_active_learning_r4.py document-grouped 80/20 export
        training_scripts = os.path.join(os.path.dirname(app.PROJECT_ROOT), 'ocr-training', 'scripts')
        if training_scripts not in sys.path:
            sys.path.insert(0, training_scripts)
        from export_active_learning_r4 import export_active_learning_dataset
        out_r4 = os.path.join(tmpdir, 'export_r4')
        manifest = export_active_learning_dataset(jsonl_path, out_r4, val_ratio=0.25, seed=42, crops_root=tmpdir)
        assert manifest['train_samples'] + manifest['val_samples'] == 5, manifest
        assert manifest['train_samples'] >= 3 and manifest['val_samples'] >= 1, manifest
        tr_rel = open(manifest['train_list'], encoding='utf-8').read().splitlines()
        va_rel = open(manifest['val_list'], encoding='utf-8').read().splitlines()
        for entry in tr_rel + va_rel:
            img_rel, lbl = entry.split('\t', 1)
            assert os.path.exists(os.path.join(out_r4, img_rel)), img_rel
            assert len(lbl) > 0


def test_suppress_detached_diacritic_fragments_keeps_short_lines():
    def _q(x1, y1, x2, y2):
        return [[float(x1), float(y1)], [float(x2), float(y1)], [float(x2), float(y2)], [float(x1), float(y2)]]

    # Main full-height lines (~45px tall) + 3 real harum_palei_01 detached diacritic specks (~10x9px)
    # + 1 standalone short stanza header ('꩑꩞') + 1 standalone page number ('12')
    raw_boxes = [
        _q(30, 5, 85, 42),        # standalone short stanza header (h=37, w=55) -> MUST KEEP
        _q(21, 55, 780, 100),     # Line 1
        _q(235, 99, 245, 108),    # Speck #1 (w=10, h=9) touching Line 1/2 -> MUST DROP
        _q(17, 105, 788, 152),    # Line 2
        _q(18, 160, 683, 206),    # Line 3
        _q(16, 411, 656, 463),    # Line 4
        _q(67, 415, 78, 423),     # Speck #2 (w=11, h=8) inside Line 4 -> MUST DROP
        _q(22, 496, 585, 540),    # Line 5
        _q(295, 497, 307, 505),   # Speck #3 (w=12, h=8) inside Line 5 -> MUST DROP
        _q(480, 960, 498, 994),   # standalone page number (h=34, w=18, far below) -> MUST KEEP
    ]
    merged = app.merge_line_boxes(raw_boxes, 900, 1050)
    assert len(merged) == 7, f"Expected 7 valid lines (3 specks dropped), got {len(merged)}"
    widths = [float(np.ptp(np.array(b)[:, 0])) for b in merged]
    assert all(w >= 17.0 for w in widths), widths


def test_v24_1_r4_and_r3_1_manifest_and_shapes():
    assert app.DEFAULT_CHAM_MODEL == 'v24_1_r4', app.DEFAULT_CHAM_MODEL
    m_r4 = app.get_model_manifest('v24_1_r4')
    m_r31 = app.get_model_manifest('v24_1_r3_1')
    assert m_r4.get('rec_image_shape') == '3, 48, 480', m_r4
    assert m_r31.get('rec_image_shape') == '3, 48, 320', m_r31
    assert 0.80 <= float(m_r4.get('confidence_temperature', 0)) <= 1.20, m_r4
    assert 0.80 <= float(m_r31.get('confidence_temperature', 0)) <= 1.20, m_r31

    app.ocr_model_paths['v24_1_r4'] = {
        'model_dir': os.path.join(app.PROJECT_ROOT, 'data', 'output', 'rec_cham_inference_v24_1_r4'),
        'dict_path': os.path.join(app.PROJECT_ROOT, 'data', 'cham_dict_v24_1_r4.txt'),
    }
    app.ocr_model_paths['v24_1_r3_1'] = {
        'model_dir': os.path.join(app.PROJECT_ROOT, 'data', 'output', 'rec_cham_inference_v24_1_r3_1'),
        'dict_path': os.path.join(app.PROJECT_ROOT, 'data', 'cham_dict_v24_1_r3_1.txt'),
    }
    info_r4 = app.get_model_info('v24_1_r4')
    info_r31 = app.get_model_info('v24_1_r3_1')
    assert info_r4['recognizers'][0]['rec_image_shape'] == '3, 48, 480', info_r4
    assert info_r4['recognizers'][0]['weights_sha256_16'] == '1deee5f5ff351070', info_r4
    assert '\\' not in info_r4['recognizers'][0]['model_dir'], info_r4
    assert info_r31['recognizers'][0]['rec_image_shape'] == '3, 48, 320', info_r31
    assert info_r31['recognizers'][0]['weights_sha256_16'] == '5fb0d8e4d512b4c0', info_r31


def test_ingest_studio_to_realscan_gold_cli():
    import json
    import tempfile

    repo_root = os.path.dirname(app.PROJECT_ROOT)
    scripts_dir = os.path.join(repo_root, 'scripts')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from ingest_studio_to_realscan_gold import ingest_page_json, sha256_file

    with tempfile.TemporaryDirectory() as tmpdir:
        v2_dir = os.path.join(tmpdir, 'ocr-benchmark', 'data', 'realscan_gold_v2')
        os.makedirs(v2_dir, exist_ok=True)
        manifest_p = os.path.join(v2_dir, 'manifest.json')
        seed_manifest = {
            'benchmark_id': 'realscan_gold_v2_test',
            'summary': {'real_camera_pages': 0, 'real_camera_lines': 0, 'total_evaluated_lines': 0},
            'documents': [],
        }
        with open(manifest_p, 'w', encoding='utf-8') as f:
            json.dump(seed_manifest, f)

        # Create synthetic manuscript page and Studio JSON export with out-of-order Cham cluster
        page_img = np.full((200, 500, 3), 245, dtype=np.uint8)
        cv2.putText(page_img, 'Cham Folio', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25, 25, 25), 2)
        src_img_p = os.path.join(tmpdir, 'folio_01.png')
        cv2.imwrite(src_img_p, page_img)

        export_p = os.path.join(tmpdir, 'cham_ocr_transcription.json')
        with open(export_p, 'w', encoding='utf-8') as f:
            json.dump({
                'lines': [
                    {'text': 'ꨙꨯꨳꨮ ꨨꨣꨭꩌ', 'bbox': [20, 25, 420, 80]},
                    {'text': '꩒꩐꩐꩞ ꨚꨤꨬ', 'bbox': [20, 100, 410, 155]},
                ]
            }, f, ensure_ascii=False)

        updated = ingest_page_json(
            src_img_p,
            export_p,
            doc_id='panduranga_test_01',
            domain='user_verified_manuscript',
            manifest_path=manifest_p,
            project_root=tmpdir,
        )
        assert updated['summary']['real_camera_pages'] == 1
        assert updated['summary']['real_camera_lines'] == 2
        doc = updated['documents'][0]
        assert doc['lines'][0]['gt'] == 'ꨙꨳꨯꨮ ꨨꨣꨭꩌ', doc['lines'][0]
        assert '\\' not in doc['page_relpath']
        assert sha256_file(os.path.join(tmpdir, doc['page_relpath'])) == doc['page_sha256']
        for ln in doc['lines']:
            assert '\\' not in ln['crop_relpath']
            assert sha256_file(os.path.join(tmpdir, ln['crop_relpath'])) == ln['crop_sha256']


import unittest  # noqa: E402


# ------------------------------------------------------------------------------------------------
# Cham-DBNet box_type='poly' + curved-line (ribbon) rectification
# ------------------------------------------------------------------------------------------------
def _wave_y(x, y0, amp=14.0, period=420.0, phase=0.0):
    return y0 + amp * np.sin(2.0 * np.pi * x / period + phase)


def _wave_page(colors=((20, 20, 20), (20, 20, 200)), y0s=(80.0, 116.0), w=900, h=200):
    """Two stacked sinusoidal 'text lines' (glyph blocks + small diacritic dots) on paper."""
    page = np.full((h, w, 3), 236, np.uint8)
    rs = np.random.RandomState(3)
    for y0, col in zip(y0s, colors):
        for x in range(60, w - 60, 15):
            yc = _wave_y(x + 4, y0)
            gh = int(rs.randint(9, 14))
            cv2.rectangle(page, (x, int(round(yc - gh / 2))), (x + 8, int(round(yc + gh / 2))), col, -1)
            if x % 45 == 0:
                cv2.circle(page, (x + 4, int(round(yc - gh / 2 - 4))), 2, col, -1)
    return page


def _wave_poly(y0, x1=55.0, x2=845.0, half=11.0, n=24):
    xs = np.linspace(x1, x2, n)
    top = [[x, _wave_y(x, y0) - half] for x in xs]
    bot = [[x, _wave_y(x, y0) + half] for x in xs[::-1]]
    return np.array(top + bot, np.float32)


def test_merge_line_polygons_keeps_stacked_wavy_lines_and_curvature():
    a1, a2 = _wave_poly(80.0, 55.0, 430.0), _wave_poly(80.0, 455.0, 845.0)  # line A in 2 fragments
    b = _wave_poly(116.0)
    quads = [np.array([[p[:, 0].min(), p[:, 1].min()], [p[:, 0].max(), p[:, 1].min()],
                       [p[:, 0].max(), p[:, 1].max()], [p[:, 0].min(), p[:, 1].max()]], np.float32) for p in (_wave_poly(80.0), b)]
    # AABBs of the two wavy lines overlap (28% of their height), their ribbons do not overlap at all
    assert app.max_stacked_overlap(quads) > 0.25
    assert app.max_stacked_overlap([_wave_poly(80.0), b]) < 0.05
    merged = app.merge_line_polygons([b, a2, a1], 900, 200)
    assert len(merged) == 2, len(merged)
    top_line = merged[0]
    assert len(top_line) > 4  # still a curved polygon, not an AABB rectangle
    x0, t, bt = app.poly_ribbon(top_line)
    for x in (120, 300, 500, 700):
        c = (t[x - x0] + bt[x - x0]) / 2.0
        assert abs(c - _wave_y(x, 80.0)) < 4.0, (x, c, _wave_y(x, 80.0))
    assert abs(app.poly_thickness(top_line) - 22.0) < 4.0
    # Reading-order rects use the centerline band (non-overlapping), not the tall AABBs
    ra, rb = app._reading_rect(merged[0]), app._reading_rect(merged[1])
    assert ra[3] <= rb[1] + 1.0, (ra, rb)


def test_rectify_polygon_line_straightens_and_masks_neighbour():
    page = _wave_page()
    pa, pb = _wave_poly(80.0), _wave_poly(116.0)
    # Generous padding (1.0 x line thickness) so the padded strip reaches the neighbour line's ink
    strip, info = app.rectify_polygon_line(page, pa, [pb], median_h=22.0, vpad=1.0, mask_outside=True)
    assert info['masked'] and info['angle_deg'] == 0.0
    assert strip.shape[0] < 75 and strip.shape[1] > 780, strip.shape
    red = (strip[:, :, 2] > 120) & (strip[:, :, 0] < 90) & (strip[:, :, 1] < 90)
    dark = strip.max(axis=2) < 90
    assert red.sum() == 0, red.sum()          # neighbour line ink never intrudes
    # straight: per-column ink centroid of the own line stays on one row (unrectified: std ~ amp/sqrt2 = 10px)
    cols = [np.nonzero(dark[:, x])[0].mean() for x in range(strip.shape[1]) if dark[:, x].sum() >= 4]
    assert len(cols) > 200 and float(np.std(cols)) < 3.0, float(np.std(cols))
    # diacritic dots above the glyph blocks are preserved (ink above the glyph band)
    band = np.nonzero(dark.sum(axis=1) > 0.2 * dark.sum(axis=1).max())[0]
    assert dark[:band.min()].sum() > 0
    # without masking the padded strip would contain neighbour ink (the mask is what removes it)
    strip_nm, _ = app.rectify_polygon_line(page, pa, [pb], median_h=22.0, vpad=1.0, mask_outside=False)
    red_nm = (strip_nm[:, :, 2] > 120) & (strip_nm[:, :, 0] < 90) & (strip_nm[:, :, 1] < 90)
    assert red_nm.sum() > 0
    # The perspective quad crop of the same line's AABB (what box_type='quad' produced) swallows the neighbour
    x1, y1, x2, y2 = app._aabb(pa)
    aabb = page[int(y1):int(y2), int(x1):int(x2)]
    assert ((aabb[:, :, 2] > 120) & (aabb[:, :, 0] < 90)).sum() > 100


class _FakeDetector:
    def __init__(self, boxes, box_type):
        self.boxes = boxes
        self.args = type('A', (), {'det_box_type': box_type})()

    def __call__(self, img):
        return self.boxes, 0.0


def test_segment_lines_dbnet_poly_and_quad_paths():
    try:
        import tools.infer.utility as utility
    except Exception:  # PaddleOCR not importable in this environment
        return
    page = _wave_page(colors=((20, 20, 20), (20, 20, 20)))
    polys = [_wave_poly(80.0, 55.0, 430.0), _wave_poly(80.0, 455.0, 845.0), _wave_poly(116.0)]
    crops, metas = app.segment_lines_dbnet(page, _FakeDetector(polys, 'poly'), ink=None)
    assert len(crops) == 2, len(crops)
    assert all(m['crop_mode'] == 'ribbon_rectified' and len(m['polygon']) > 4 for m in metas)
    assert all(c.shape[0] < 50 for c in crops)
    # Quad path stays the legacy perspective crop, byte for byte
    quads = [np.array([[60, 40], [840, 40], [840, 70], [60, 70]], np.float32),
             np.array([[60, 120], [840, 124], [840, 150], [60, 146]], np.float32)]
    crops_q, metas_q = app.segment_lines_dbnet(page, _FakeDetector(quads, 'quad'), ink=None)
    assert len(crops_q) == 2
    for c, q in zip(crops_q, quads):
        ref = utility.get_rotate_crop_image(page, q.copy())
        assert c.shape == ref.shape and np.array_equal(c, ref)
    assert all('crop_mode' not in m and len(m['polygon']) == 4 for m in metas_q)


def test_poly_clip_and_line_height_helpers():
    ragged = [[[1, 2], [50, 2], [60, 9], [50, 20], [-5, 20]], [[0, 0], [10, 0], [10, 10]]]
    out = app._clip_poly_det_res(ragged, (15, 55, 3))
    assert isinstance(out, list) and [len(p) for p in out] == [5, 3]
    assert out[0][:, 0].min() >= 0 and out[0][:, 0].max() <= 54 and out[0][:, 1].max() <= 14
    quad = np.array([[0, 10], [100, 30], [100, 52], [0, 32]], np.float32)
    assert app._line_height(quad) == float(np.ptp(quad[:, 1]))  # legacy AABB height for quads
    assert abs(app._line_height(_wave_poly(80.0)) - 22.0) < 3.0   # ribbon thickness for curved polygons
    assert app._is_poly_detector(_FakeDetector([], 'poly')) and not app._is_poly_detector(_FakeDetector([], 'quad'))
    assert app.CHAM_DET_BOX_TYPE == os.getenv('CHAM_DET_BOX_TYPE', 'poly').strip().lower()
    assert abs(app.CHAM_DET_UNCLIP - float(os.getenv('CHAM_DET_UNCLIP', '1.8'))) < 1e-6


def test_auto_routing_v2_tri_specialist_adjudication():
    calls = {'w480': 0, 'w320': 0, 'vi': 0}

    def fake_w480(imgs):
        calls['w480'] += len(imgs)
        # [0]: high-conf clean Cham (fast-exit, skips w320 & vi)
        # [1]: medium-width unseen EFEO Cham folio (lower w480 conf, out-of-order cluster -> rescued by w320)
        # [2]: editorial double-hyphen '--' line where w480 preserves '--' and w320 collapses to '-'
        # [3]: degraded historical Cham folio (conf=0.58 -> w320=0.62, pure Cham -> skips vi completely!)
        # [4]: severe degraded Cham folio (conf=0.28 -> w320=0.33, pure Cham -> queries vi but rejects fake Vi 0.74!)
        # [5]: genuine Vietnamese gloss (ASCII soup in Cham CTC -> routes cleanly to Vi!)
        return [
            ('ꨀꨇꩉ ꨌꩌ', 0.985),
            ('ꨙꨯꨳꨮ ꨚꨤꨬ', 0.84),
            ('ꨌꩌ--ꨚꨤꨬ', 0.86),
            ('ꨨꨣꨭꩌ ꨚꨤꨬ ꨣꩌ', 0.58),
            ('ꨚꨯꨮ ꨆꨵꨯꨱꩃ ꨈꨣꨰ', 0.28),
            ('56꩝ {0] 2026꩞', 0.47),
        ], 0.0

    def fake_w320(imgs):
        calls['w320'] += len(imgs)
        # Called on [1], [2], [3], [4], [5] (NOT [0] because c480 >= 0.905!)
        return [
            ('ꨙꨳꨯꨮ ꨚꨤꨬ ꨣꩌ', 0.94),
            ('ꨌꩌ-ꨚꨤꨬ', 0.92),
            ('ꨨꨣꨭꩌ ꨚꨤꨬ ꨣꩌ', 0.62),
            ('ꨚꨯꨮ ꨆꨵꨯꨱꩃ ꨈꨣꨰ', 0.33),
            ('5꩒꩞ 06] 2026', 0.48),
        ], 0.0

    def fake_vi(imgs):
        calls['vi'] += len(imgs)
        # Called ONLY on [4] (c_conf=0.32 < 0.34) and [5] (_cham_script_ratio < 0.62) — NOT [3]!
        return [
            ('38158"o) gi', 0.74),
            ('Bản dịch cổ thư Chăm năm 2026', 0.97),
        ], 0.0

    med_img = np.zeros((48, 280, 3), dtype=np.uint8)
    router = app.AutoRoutingOCRWrapper(
        fake_w480, fake_vi, cham_w320_model=fake_w320, primary_ver='v24_1_r4', w320_ver='v24_1_r3_1'
    )
    out, _ = router([med_img] * 6)
    assert calls == {'w480': 6, 'w320': 5, 'vi': 2}, calls
    assert out[0][0] == 'ꨀꨇꩉ ꨌꩌ' and router.last_selected_versions[0] == 'v24_1_r4'
    assert out[1][0] == 'ꨙꨳꨯꨮ ꨚꨤꨬ ꨣꩌ' and router.last_selected_versions[1] == 'v24_1_r3_1'
    assert '--' in out[2][0] and router.last_selected_versions[2] == 'v24_1_r4'
    assert out[3][0] == 'ꨨꨣꨭꩌ ꨚꨤꨬ ꨣꩌ' and router.last_selected_versions[3] == 'v24_1_r3_1'
    assert out[4][0] == 'ꨚꨯꨮ ꨆꨵꨯꨱꩃ ꨈꨣꨰ' and router.last_selected_versions[4] == 'v24_1_r3_1'
    assert out[5][0] == 'Bản dịch cổ thư Chăm năm 2026' and router.last_selected_versions[5] == 'vi'
    info_auto = app.get_model_info('auto')
    assert info_auto['normalizer'] == app.NORMALIZER_VERSION


def test_realscan_gold_v3_manifest_integrity():
    import json
    repo_root = os.path.dirname(app.PROJECT_ROOT)
    m_path = os.path.join(repo_root, 'ocr-benchmark', 'data', 'realscan_gold_v3', 'manifest.json')
    assert os.path.exists(m_path), m_path
    m = json.load(open(m_path, 'r', encoding='utf-8'))
    assert m['schema_version'] == '3.0.0'
    assert m['summary']['efeo_fonts_covered'] == '12/12'
    assert m['summary']['heldout_weights_covered'] == '6/6'
    assert m['summary']['total_evaluated_items'] >= 430


def test_workbench_samples_models_and_corpus_zip_export():
    import base64
    import io
    import json
    import zipfile

    catalog = app.list_available_models()
    assert catalog['default_model'] == 'v24_1_r4'
    ids = {m['id']: m for m in catalog['models']}
    try:
        import paddle
        paddle_major = int(str(getattr(paddle, '__version__', '0')).split('.')[0])
    except Exception:
        paddle_major = 0
    assert 'v24_1_r4' in ids and ids['v24_1_r4']['installed'] is True
    assert 'v25' in ids and ids['v25']['installed'] is (not (0 < paddle_major < 3))
    assert 'auto' in ids and ids['auto']['installed'] is True

    samples = app.list_curated_samples()
    assert len(samples) >= 4, samples
    s0 = app.load_curated_sample(samples[0]['id'])
    assert s0 is not None and s0['width'] > 50 and len(s0['image']) > 100

    # Verify corpus ZIP export normalizes Cham Logical Order (ꨙꨯꨳꨮ -> ꨙꨳꨯꨮ) and packs crops
    dummy_crop = np.full((32, 100, 3), 240, dtype=np.uint8)
    ok, buf = cv2.imencode('.png', dummy_crop)
    assert ok
    crop_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

    zip_bytes = app.build_corpus_export_zip({
        'documents': [
            {
                'name': 'agal_folio_01.png',
                'status': 'Verified',
                'model': 'v24_1_r4',
                'lines': [
                    {'index': 1, 'text': 'ꨙꨯꨳꨮ ꨨꨣꨭꩌ', 'confidence': 1.0, 'bbox': [10, 10, 110, 42], 'image': crop_b64, 'human_verified': True}
                ]
            }
        ]
    })
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
        names = set(zf.namelist())
        assert 'train_list.txt' in names
        assert 'corpus_manifest.json' in names
        assert 'transcriptions/combined_corpus.md' in names
        train_line = zf.read('train_list.txt').decode('utf-8').strip()
        assert train_line.endswith('\tꨙꨳꨯꨮ ꨨꨣꨭꩌ'), train_line
        crop_rel = train_line.split('\t', 1)[0]
        assert crop_rel in names


def test_static_asset_separation_and_http_routes():
    import urllib.request
    import threading
    from http.server import HTTPServer

    static_dir = os.path.join(app.PROJECT_ROOT, 'static')
    for rel in ('css/studio.css', 'css/research.css', 'css/help.css', 'js/studio.js', 'js/cham-keyboard.js', 'js/transliteration.js', 'js/research.js', 'js/help.js'):
        assert os.path.isfile(os.path.join(static_dir, *rel.split('/'))), rel

    httpd = HTTPServer(('127.0.0.1', 0), app.ChamOCRRequestHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/') as r:
            html = r.read().decode('utf-8')
            assert '/static/css/studio.css' in html and '/static/js/studio.js' in html and 'href="/help"' in html
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/help') as r:
            h_html = r.read().decode('utf-8')
            assert '/static/css/help.css' in h_html and '/static/js/help.js' in h_html and 'shortcuts-table' in h_html
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/research') as r:
            r_html = r.read().decode('utf-8')
            assert '/static/css/research.css' in r_html and 'sec7-3' in r_html and 'href="/help"' in r_html
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/static/css/studio.css') as r:
            css = r.read().decode('utf-8')
            assert '#FAF7F2' in css and '#C96442' in css
        # Ensure directory traversal outside /static/ is strictly blocked with HTTP 404
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/static/../app.py')
            assert False, 'Expected 404 on path traversal'
        except urllib.error.HTTPError as he:
            assert he.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_v26_audit_api_contract_and_provenance_hardening():
    import base64
    import json
    import urllib.request
    import urllib.error
    import threading
    from http.server import HTTPServer

    # 1. Model catalog reclassifies V25 as Legacy/Deprecated under Historical Baselines
    catalog = app.list_available_models()
    by_id = {m['id']: m for m in catalog['models']}
    assert by_id['v25']['group'] == 'Historical Baselines', by_id['v25']
    assert by_id['v25'].get('deprecated') is True, by_id['v25']
    assert 'SOTA' not in by_id['v25']['short_label'], by_id['v25']

    # 2. Strict model allowlist rejects unknown IDs (no silent fallback to V23)
    for bad_id in ('nonexistent_model_2026', 'model_does_not_exist', '../v23', 12345):
        try:
            app.validate_requested_model(bad_id)
            assert False, f"Expected UnknownModelError for {bad_id!r}"
        except app.UnknownModelError:
            pass
    assert app.validate_requested_model(None) == app.DEFAULT_CHAM_MODEL
    assert app.validate_requested_model('') == app.DEFAULT_CHAM_MODEL
    assert app.validate_requested_model('v24_1_r4') == 'v24_1_r4'

    # 3. Auto Router v2.2 handles digit/punctuation-heavy Vietnamese lines without breaking Cham safety
    def fake_cham_w480(imgs):
        return [
            (' {889- 0128466780} ꨏ}51/', 0.7554),   # vietnamese_00 line 3: digit/symbol soup
            ('ꨚꨯꨮ ꨆꨵꨯꨱꩃ ꨈꨣꨰ', 0.31),               # degraded historical Cham line -> MUST stay Cham
        ], 0.0

    def fake_vi(imgs):
        return [
            ('Số trang: 0123456789, dấu ! ? : ; ( ) / -', 0.92),
            ('38158"o) gi', 0.74),
        ], 0.0

    router = app.AutoRoutingOCRWrapper(fake_cham_w480, fake_vi, cham_w320_model=None, primary_ver='v24_1_r4')
    dummy_crop = np.full((48, 320, 3), 245, dtype=np.uint8)
    routed_out, _ = router([dummy_crop, dummy_crop])
    assert routed_out[0][0].startswith('Số trang: 0123456789'), routed_out[0]
    assert router.last_selected_versions[0] == 'vi'
    assert routed_out[1][0] == 'ꨚꨯꨮ ꨆꨵꨯꨱꩃ ꨈꨣꨰ', routed_out[1]
    assert router.last_selected_versions[1] == 'v24_1_r4'

    # 4. HTTP 400 validation across /ocr and /ocr_crop (no HTTP 502 on missing image, no HTTP 200 on invalid params)
    valid_img = np.full((80, 240, 3), 250, dtype=np.uint8)
    ok, buf = cv2.imencode('.png', valid_img)
    assert ok
    valid_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

    httpd = HTTPServer(('127.0.0.1', 0), app.ChamOCRRequestHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    def post_json(path, payload):
        req = urllib.request.Request(
            f'http://127.0.0.1:{port}{path}',
            data=(json.dumps(payload).encode('utf-8') if isinstance(payload, (dict, list)) else payload),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as he:
            return he.code, json.loads(he.read().decode('utf-8'))

    try:
        for path, bad_payload in [
            ('/ocr', {'model': 'v24_1_r4'}),  # missing_image -> 400 JSON (was 502)
            ('/ocr', {'image': valid_b64, 'model': 'nonexistent_model_2026'}),  # invalid_model -> 400 (was 200 V23)
            ('/ocr', {'image': valid_b64, 'model': 'v24_1_r4', 'method': 'not_a_method'}),  # invalid_segmentation -> 400
            ('/ocr', {'image': valid_b64, 'model': 'v24_1_r4', 'method': 'dbnet', 'reading_order': 'invalid'}),  # -> 400
            ('/ocr', {'image': '', 'model': 'v24_1_r4'}),
            ('/ocr', b'not-valid-json{{{'),
            ('/ocr_crop', {'image': valid_b64, 'model': 'nonexistent_model_2026', 'bbox': [0, 0, 100, 40]}),
            ('/ocr_crop', {'image': valid_b64, 'model': 'v24_1_r4', 'bbox': [0, 0, 0, 0]}),  # invalid_bbox_zero -> 400
            ('/ocr_crop', {'image': valid_b64, 'model': 'v24_1_r4', 'bbox': [-200, -200, -1, -1]}),  # negative -> 400
            ('/ocr_crop', {'image': valid_b64, 'model': 'v24_1_r4', 'bbox': [500, 500, 700, 700]}),  # outside image -> 400
        ]:
            code, body = post_json(path, bad_payload)
            assert code == 400 and 'error' in body, (path, bad_payload, code, body)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_priority_a_pass2_provenance_and_zero_script_guessing():
    """Priority A regression: exact recognizer provenance across Pass 2 (valley/adaptive) & /ocr_crop without script guessing."""
    import base64
    from http.server import HTTPServer
    import json
    import threading
    import urllib.request

    # 1. calibrate_line_confidence must never infer recognizer temperature from output characters when model_ver='auto'
    assert app.calibrate_line_confidence(0.80, model_ver='auto', text='ꨀꨕꩅ') == 0.8
    assert app.calibrate_line_confidence(0.80, model_ver=None, text='ꨀꨕꩅ') == 0.8

    # 2. Construct a crisp synthetic line image that triggers Pass 1 (safe, taller) -> Pass 2 (tight, shorter)
    canvas = np.full((160, 360, 3), 255, dtype=np.uint8)
    for x in range(40, 310, 28):
        cv2.rectangle(canvas, (x, 66), (x + 14, 92), (15, 15, 15), -1)

    # Height of core ink is 26px (y=66..92).
    # Pass 1 'safe' crop expands >= 0.40 * h on top & bottom (~46px+ tall).
    # Pass 2 'tight' crop expands ~0.05 * h (~28px tall, ar48 = 360*48/28 is >420 unless cropped narrower;
    # make canvas width 200 so tight crop ~28x200 has ar48 = 200*48/28 = 342.8 in [104, 420) for W320!).
    canvas = canvas[:, :200].copy()

    def fake_w480(imgs):
        out = []
        for im in imgs:
            h = im.shape[0]
            if h >= 36:
                # Pass 1 safe/rescue/loose crops: low confidence (<0.75) to force Pass 2 execution
                out.append(('123 - 456', 0.62))
            else:
                # Pass 2 tight crop: sub-0.905 confidence so W=320 specialist adjudicates
                out.append(('123 - 456', 0.82))
        return out, 0.0

    def fake_w320(imgs):
        # Pass 1 safe/rescue/loose crops stay <0.75 so Pass 2 is forced; Pass 2 tight crop (h < 36) wins with 0.94!
        return [('123 - 456', 0.65 if im.shape[0] >= 36 else 0.94) for im in imgs], 0.0

    def fake_vi(imgs):
        # Pure numerals/punctuation: v_lat_cnt == 0 -> Auto Router must NOT hand off to 'vi'
        return [('123 - 456', 0.91) for _ in imgs], 0.0

    router = app.AutoRoutingOCRWrapper(
        fake_w480, fake_vi, cham_w320_model=fake_w320, primary_ver='v24_1_r4', w320_ver='v24_1_r3_1'
    )

    saved_auto = app.ocr_models.get('auto')
    app.ocr_models['auto'] = router
    try:
        for seg_method in ('valley', 'adaptive'):
            resp = app.run_ocr_pipeline(canvas, model_ver='auto', method=seg_method)
            assert len(resp['lines']) == 1, (seg_method, resp)
            line = resp['lines'][0]
            assert line['segmentation']['selected_crop_type'] == 'tight', (seg_method, line)
            assert line['text'] == '123 - 456', (seg_method, line)
            # Crucial assertion: old post-OCR character check guessed 'vi' because '123 - 456' has no U+AA00..U+AA5F!
            # Exact Pass-2 provenance must report 'v24_1_r3_1'.
            assert line['selected_recognizer'] == 'v24_1_r3_1', (seg_method, line)

        # 3. Verify /ocr_crop with model='auto' on pure ASCII digits/hyphen ('7-8') reports 'v24_1_r4', never guesses 'vi'
        def crop_w480(imgs):
            return [('7-8', 0.95) for _ in imgs], 0.0

        def crop_vi(imgs):
            return [('7-8', 0.96) for _ in imgs], 0.0

        app.ocr_models['auto'] = app.AutoRoutingOCRWrapper(
            crop_w480, crop_vi, cham_w320_model=None, primary_ver='v24_1_r4'
        )
        ok, buf = cv2.imencode('.png', canvas)
        assert ok
        b64_img = base64.b64encode(buf.tobytes()).decode('ascii')

        httpd = HTTPServer(('127.0.0.1', 0), app.ChamOCRRequestHandler)
        port = httpd.server_address[1]
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        try:
            req = urllib.request.Request(
                f'http://127.0.0.1:{port}/ocr_crop',
                data=json.dumps({'image': b64_img, 'model': 'auto', 'bbox': [20, 55, 190, 105]}).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                crop_body = json.loads(r.read().decode('utf-8'))
            assert crop_body['text'] == '7-8', crop_body
            assert crop_body['selected_recognizer'] == 'v24_1_r4', crop_body
        finally:
            httpd.shutdown()
            httpd.server_close()
    finally:
        if saved_auto is None:
            app.ocr_models.pop('auto', None)
        else:
            app.ocr_models['auto'] = saved_auto


def test_realscan_gold_v4_anti_circular_and_promotion_gates():
    """RealScan-Gold v4: enforces gt=None on staging_review_queue, blocks unverified Studio drafts, promotes Verified pages."""
    import json
    import tempfile

    repo_root = os.path.dirname(app.PROJECT_ROOT)
    scripts_dir = os.path.join(repo_root, 'scripts')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import ingest_studio_to_realscan_gold as ingest_v4

    with tempfile.TemporaryDirectory() as tmpdir:
        v4_dir = os.path.join(tmpdir, 'ocr-benchmark', 'data', 'realscan_gold_v4')
        pages_dir = os.path.join(v4_dir, 'pages')
        os.makedirs(pages_dir, exist_ok=True)
        page_p = os.path.join(pages_dir, 'eap_test_folio_01.jpg')
        img = np.full((240, 600, 3), 235, dtype=np.uint8)
        cv2.rectangle(img, (40, 50), (520, 92), (25, 25, 25), -1)
        cv2.rectangle(img, (40, 120), (490, 162), (25, 25, 25), -1)
        cv2.imwrite(page_p, img)

        manifest_p = os.path.join(v4_dir, 'manifest.json')
        seed_manifest = {
            'schema_version': '4.0.0',
            'dataset_id': 'realscan_gold_v4',
            'summary': {},
            'documents': [{
                'doc_id': 'eap_test_folio_01',
                'split': 'staging_review_queue',
                'verification_status': 'pending_human_review',
                'verified_by_human': False,
                'gold_eligible': False,
                'stratum': 'handwritten_paper_traditional',
                'source_collection_id': 'EAP698-TEST',
                'redistribution_allowed_in_git': False,
                'page_relpath': 'ocr-benchmark/data/realscan_gold_v4/pages/eap_test_folio_01.jpg',
                'page_sha256': ingest_v4.sha256_file(page_p),
                'draft_ocr_summary': {'cham_bearing_lines': 2},
                'line_count': 2,
                'lines': [
                    {'line': 1, 'bbox': [40, 50, 520, 92], 'crop_relpath': 'ocr-benchmark/data/realscan_gold_v4/crops/x1.png',
                     'crop_sha256': '0' * 64, 'gt': None, 'verified_by_human': False, 'draft_prediction': 'ꨙꨳꨯꨮ'},
                    {'line': 2, 'bbox': [40, 120, 490, 162], 'crop_relpath': 'ocr-benchmark/data/realscan_gold_v4/crops/x2.png',
                     'crop_sha256': '0' * 64, 'gt': None, 'verified_by_human': False, 'draft_prediction': 'ꨌꩌ'},
                ],
            }],
        }
        with open(manifest_p, 'w', encoding='utf-8') as f:
            json.dump(seed_manifest, f)

        # 1. Unverified draft Studio JSON MUST be rejected by Anti-Circularity Gate
        unverified_json = os.path.join(tmpdir, 'draft_unverified.json')
        with open(unverified_json, 'w', encoding='utf-8') as f:
            json.dump({
                'image_name': 'eap_test_folio_01.jpg',
                'status': 'Needs review',
                'lines': [
                    {'bbox': [40, 50, 520, 92], 'text': 'ꨙꨯꨳꨮ', 'human_verified': False},
                    {'bbox': [40, 120, 490, 162], 'text': 'ꨌꩌ', 'human_verified': False},
                ],
            }, f)
        rejected = False
        try:
            ingest_v4.promote_verified_v4(unverified_json, manifest_path=manifest_p, project_root=tmpdir)
        except ValueError as exc:
            rejected = 'Anti-Circularity Gate' in str(exc)
        assert rejected, 'Expected ValueError from Anti-Circularity Gate on unverified Studio draft!'

        # 2. Human-verified Studio JSON promotes cleanly & enforces Logical Order normalize_unicode ('ꨙꨯꨳꨮ' -> 'ꨙꨳꨯꨮ')
        verified_json = os.path.join(tmpdir, 'human_verified.json')
        with open(verified_json, 'w', encoding='utf-8') as f:
            json.dump({
                'image_name': 'eap_test_folio_01.jpg',
                'status': 'Verified',
                'lines': [
                    {'bbox': [40, 50, 520, 92], 'text': 'ꨙꨯꨳꨮ ꨚꨤꨬ', 'original_text': 'ꨙꨳꨯꨮ', 'human_verified': True},
                    {'bbox': [40, 120, 490, 162], 'text': 'ꨌꩌ ꨣꩌ', 'original_text': 'ꨌꩌ', 'human_verified': True},
                ],
            }, f)
        updated = ingest_v4.promote_verified_v4(
            verified_json, annotator='paleographer_a', review_round=2, manifest_path=manifest_p, project_root=tmpdir
        )
        doc = updated['documents'][0]
        assert doc['split'] == 'verified_gold' and doc['verified_by_human'] is True and doc['gold_eligible'] is True
        assert doc['annotator_id'] == 'paleographer_a' and doc['review_round'] == 2
        assert doc['lines'][0]['gt'] == 'ꨙꨳꨯꨮ ꨚꨤꨬ', doc['lines'][0]
        assert updated['summary']['verified_gold_pages'] == 1
        assert updated['summary']['staging_new_physical_pages'] == 0
        stats = ingest_v4.verify_manifest_integrity(manifest_p, project_root=tmpdir)
        assert stats['checked_files'] == 3, stats

    # 3. Verify production RealScan-Gold v4 manifest on disk if already built
    prod_v4 = os.path.join(repo_root, 'ocr-benchmark', 'data', 'realscan_gold_v4', 'manifest.json')
    if os.path.exists(prod_v4):
        m4 = json.load(open(prod_v4, encoding='utf-8'))
        assert m4['summary']['synthetic_pages_included'] == 0
        assert m4['summary']['total_physical_pages'] >= 50
        prod_stats = ingest_v4.verify_manifest_integrity(prod_v4, project_root=repo_root)
        assert prod_stats['documents'] >= 58, prod_stats


def test_stage1_v26_2_polarity_divider_deskew_provenance_and_router_v23():
    """Adversarial regression gate for Pipeline v26.2 + Auto Router v2.3 (Gate SC-1)."""
    # 1. check_inverted_polarity: must NOT invert dark-framed archival photo (bright paper interior),
    #    and MUST invert genuine dark-substrate white-stroke negative/dark-mode image (even 1 line).
    archival_dark_frame = np.full((600, 800, 3), 24, dtype=np.uint8)
    archival_dark_frame[90:510, 120:680] = 215  # bright paper folio inside dark table frame
    for y in range(140, 460, 48):
        cv2.line(archival_dark_frame, (160, y), (640, y), (30, 30, 30), 3)
    assert app.check_inverted_polarity(archival_dark_frame) is False, "False positive inversion on dark-framed archival photo!"

    dark_mode_1line = np.full((240, 720, 3), 22, dtype=np.uint8)
    cv2.putText(dark_mode_1line, "CHAM NEGATIVE TEST", (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (242, 242, 242), 2)
    assert app.check_inverted_polarity(dark_mode_1line) is True, "Missed genuine 1-line dark-substrate white-stroke image!"

    # 2. detect_vertical_column_divider + cross-column barrier in merge_line_boxes
    two_col_page = np.full((900, 1200, 3), 242, dtype=np.uint8)
    cv2.line(two_col_page, (600, 140), (600, 820), (35, 35, 35), 2)
    div_x = app.detect_vertical_column_divider(two_col_page)
    assert div_x is not None and 595 <= div_x <= 605, f"Expected divider near x=600, got {div_x}"
    q_left = [[140, 300], [594, 300], [594, 336], [140, 336]]
    q_right = [[606, 301], [1080, 301], [1080, 337], [606, 337]]
    assert len(app.merge_line_boxes([q_left, q_right], 1200, 900, col_divider_x=None)) == 1
    merged_boxes = app.merge_line_boxes([q_left, q_right], 1200, 900, col_divider_x=div_x)
    assert len(merged_boxes) == 2, f"Cross-column merge leaked across col_divider_x={div_x}: {merged_boxes}"

    single_col_page = np.full((900, 1200, 3), 242, dtype=np.uint8)
    for y in range(160, 780, 52):
        cv2.line(single_col_page, (150, y), (1050, y), (30, 30, 30), 3)
    assert app.detect_vertical_column_divider(single_col_page) is None, "False column divider on single-column page!"

    # 3. get_model_info detector provenance when detector_switched=True
    mi_switched = app.get_model_info('v24_1_r4', executed_detector='cham_dbnet', detector_switched=True)
    assert mi_switched['detector']['name'] == 'cham_dbnet'
    assert mi_switched['detector']['detector_switched'] is True
    assert mi_switched['detector']['weights_sha256_16'] == mi_switched['detector']['cham_dbnet_sha256_16']

    # 4. maybe_safe_deskew_page: 0-deg untouched; -12-deg counter-rotated and inverse-mapped cleanly
    straight = np.full((520, 760, 3), 248, dtype=np.uint8)
    for y in range(90, 440, 46):
        cv2.line(straight, (90, y), (670, y), (20, 20, 20), 4)
    _, M0, ang0 = app.maybe_safe_deskew_page(straight)
    assert M0 is None and ang0 == 0.0, f"Unexpected deskew on 0-deg image: {ang0}"

    M_rot = cv2.getRotationMatrix2D((380.0, 260.0), -12.0, 1.0)
    tilted = cv2.warpAffine(straight, M_rot, (760, 520), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    dsk_im, M_inv, ang_est = app.maybe_safe_deskew_page(tilted)
    assert M_inv is not None and 10.5 <= ang_est <= 13.5, f"Expected ~+12 deg deskew, got {ang_est}"
    mapped_pt = app.map_deskew_pts_to_original([[100.0, 100.0], [200.0, 100.0], [200.0, 130.0], [100.0, 130.0]], M_inv, 760, 520)
    assert len(mapped_pt) == 4 and all(0.0 <= px <= 760.0 and 0.0 <= py <= 520.0 for px, py in mapped_pt)

    # 5. Auto Router v2.3: 'Năm 2000-2025', pure ASCII year '1990-2000', genuine Cham digits, and intra-line split
    dummy_mixed = np.full((48, 360, 3), 250, dtype=np.uint8)
    cv2.rectangle(dummy_mixed, (20, 12), (120, 36), (20, 20, 20), -1)   # span 0 (Cham headword)
    cv2.rectangle(dummy_mixed, (165, 12), (335, 36), (20, 20, 20), -1)  # span 1 (Vietnamese/French gloss)

    def _mock_cham(imgs):
        outs = []
        for im in imgs:
            w = im.shape[1]
            if w > 300:
                outs.append(('ꨂꨣꩃ ꨌꩌ {61ꨯꨮꩀ', 0.78))
            elif w < 140:
                outs.append(('ꨂꨣꩃ ꨌꩌ', 0.96))
            else:
                outs.append(('{61_48}', 0.41))
        return outs, 0.01

    def _mock_viet(imgs):
        outs = []
        for im in imgs:
            w = im.shape[1]
            if w > 300:
                outs.append(('2ZwY Người Chăm', 0.89))
            elif w < 140:
                outs.append(('2ZwY', 0.62))
            else:
                outs.append(('(Người Chăm)', 0.96))
        return outs, 0.01

    router_split = app.AutoRoutingOCRWrapper(_mock_cham, _mock_viet, cham_w320_model=None, primary_ver='v24_1_r4')
    res_s, vers_s, _ = router_split.predict_with_provenance([dummy_mixed])
    assert vers_s == ['v24_1_r4+vi'], f"Expected intra-line split v24_1_r4+vi, got {vers_s}: {res_s}"
    assert 'ꨂꨣꩃ ꨌꩌ' in res_s[0][0] and '(Người Chăm)' in res_s[0][0], res_s

    router_yr = app.AutoRoutingOCRWrapper(
        lambda imgs: ([('꩒) 200-2026', 0.87), ('1꩐0-200', 0.81), ('꩒꩐꩒꩖', 0.97)], 0.01),
        lambda imgs: ([('Năm 2000-2025', 0.98), ('1990-2000', 0.98), ('2026', 0.70)], 0.01),
        cham_w320_model=None,
        primary_ver='v24_1_r4',
    )
    blank = np.full((48, 240, 3), 250, dtype=np.uint8)
    res_y, vers_y, _ = router_yr.predict_with_provenance([blank, blank, blank])
    assert res_y[0][0] == 'Năm 2000-2025' and vers_y[0] == 'vi', (res_y[0], vers_y[0])
    assert res_y[1][0] == '1990-2000' and vers_y[1] == 'vi', (res_y[1], vers_y[1])
    assert res_y[2][0] == '꩒꩐꩒꩖' and vers_y[2] == 'v24_1_r4', (res_y[2], vers_y[2])


def test_stage2_to_4_disjoint_split_and_v27_leak_guard():
    """Adversarial regression gate for Stage 2–4 Custodian/Series-Disjoint Split & V27 Holdout Leak-Guard."""
    import json
    repo_root = os.path.dirname(app.PROJECT_ROOT)
    scripts_dir = os.path.join(repo_root, 'scripts')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import ingest_studio_to_realscan_gold as ingest_v4

    prod_v4 = os.path.join(repo_root, 'ocr-benchmark', 'data', 'realscan_gold_v4', 'manifest.json')
    if not os.path.exists(prod_v4):
        return
    m4 = json.load(open(prod_v4, encoding='utf-8'))
    if m4.get('schema_version', '') < '4.1.0':
        return

    # 1. Verify manifest integrity + zero collection/series/custodian leakage
    stats = ingest_v4.verify_manifest_integrity(prod_v4, project_root=repo_root)
    assert stats['documents'] == 58, stats

    holdout_doc = next(d for d in m4['documents'] if d.get('benchmark_split') == 'locked_test_holdout' and d.get('lines'))
    seed_doc = next(d for d in m4['documents'] if d.get('benchmark_split') == 'adaptation_seed_pool' and d.get('lines'))

    # 2. Must block holdout doc_id, shared EAP series cousin, disguised holdout crop SHA, and unverified pseudo-label
    for bad_rec in [
        {'doc_id': holdout_doc['doc_id'], 'source_collection_id': holdout_doc['source_collection_id'], 'verified_by_human': True, 'gt': 'ꨀ'},
        {'doc_id': 'cousin_leak', 'source_collection_id': 'EAP531-1-99', 'verified_by_human': True, 'gt': 'ꨀ'},
        {'doc_id': seed_doc['doc_id'], 'source_collection_id': seed_doc['source_collection_id'], 'crop_sha256': holdout_doc['lines'][0]['crop_sha256'], 'verified_by_human': True, 'gt': 'ꨀ'},
        {'doc_id': seed_doc['doc_id'], 'source_collection_id': seed_doc['source_collection_id'], 'verified_by_human': False, 'gt': 'ꨀ'},
    ]:
        blocked = False
        try:
            ingest_v4.verify_v27_training_disjoint_from_holdout([bad_rec], manifest_path=prod_v4)
        except PermissionError:
            blocked = True
        assert blocked, f"Failed to block leakage record: {bad_rec}"

    # 3. Must accept genuinely human-verified adaptation_seed_pool record
    ok_res = ingest_v4.verify_v27_training_disjoint_from_holdout([{
        'doc_id': seed_doc['doc_id'],
        'source_collection_id': seed_doc['source_collection_id'],
        'benchmark_split': 'adaptation_seed_pool',
        'training_forbidden': False,
        'verified_by_human': True,
        'page_sha256': seed_doc['page_sha256'],
        'crop_sha256': seed_doc['lines'][0]['crop_sha256'],
        'gt': 'ꨀꨕꩅ ꨌꩌ',
    }], manifest_path=prod_v4)
    assert ok_res['accepted_records'] == 1


def test_stage_v26_3_qa_p1_p2_regressions():
    """Adversarial regression gate for all 4 P1 bugs and 2 P2 issues from Independent QA (2026-11-23)."""
    # P1-1: Studio JS active-folio JSON re-import must update state.lines before dbPutFolio and pass skipSyncActive: true
    js_p = os.path.join(app.PROJECT_ROOT, 'static', 'js', 'studio.js')
    js_src = open(js_p, 'r', encoding='utf-8').read()
    assert 'selectFolio(folio.id, { skipSyncActive: true })' in js_src
    assert 'if (!options.skipSyncActive)' in js_src

    # P1-2: Two-column page with a SHORT centered top header (and optional centered bottom footer)
    rects_2col_short_title = [
        [489, 47, 633, 83],     # 0: short centered title (< 0.6 * text_w!)
        [95, 159, 246, 195],    # 1: left 1
        [758, 159, 905, 197],   # 2: right 1
        [759, 299, 917, 336],   # 3: right 2
        [96, 305, 215, 336],    # 4: left 2
        [759, 439, 900, 477],   # 5: right 3
        [94, 441, 201, 475],    # 6: left 3
    ]
    order_open, layout_open = app.order_lines_reading(rects_2col_short_title, 1250, col_divider_x=None)
    assert layout_open == {'reading_order': 'columns', 'columns': 2}, layout_open
    assert order_open == [0, 1, 4, 6, 2, 3, 5], order_open

    order_div, layout_div = app.order_lines_reading(rects_2col_short_title, 1250, col_divider_x=637)
    assert layout_div == {'reading_order': 'columns', 'columns': 2}, layout_div
    assert order_div == [0, 1, 4, 6, 2, 3, 5], order_div

    # P1-3: Sparse single-line white-on-black negative (~0.14% bright stroke pixels at left x=70..275)
    sparse_neg = np.zeros((220, 1050, 3), dtype=np.uint8)
    for x in range(72, 260, 22):
        cv2.line(sparse_neg, (x, 96), (x + 12, 122), (248, 248, 248), 2)
    assert app.check_inverted_polarity(sparse_neg) is True, "Missed sparse 0.14%-ink white-on-black negative!"
    assert app.check_inverted_polarity(np.zeros((220, 1050, 3), dtype=np.uint8)) is False, "False positive on pure black blank!"

    # P1-4: Leading-Vietnamese + trailing-Cham inline split where Cham CTC outputs glued Cham-digit soup ('ꨚ꩗꩐꩐꩖ ꨏ/ꨮ;')
    mixed_lead_vi = np.full((64, 440, 3), 252, dtype=np.uint8)
    cv2.rectangle(mixed_lead_vi, (24, 16), (170, 48), (18, 18, 18), -1)   # 'Người Chăm:'
    cv2.rectangle(mixed_lead_vi, (215, 16), (410, 48), (18, 18, 18), -1)  # 'ꨚꨗꩍ ꨓꨶꨮꩉ ?'

    def _mock_cham_lead(imgs):
        outs = []
        for im in imgs:
            w = im.shape[1]
            if w > 380:
                outs.append(('ꨚ꩗꩐꩐꩖ ꨏ/ꨮ; ꨚꨗꩍ ꨓꨶꨮꩉ ?', 0.74))
            elif w < 190:
                outs.append(('ꨚ꩗꩐꩐꩖ ꨏ/ꨮ;', 0.83))
            else:
                outs.append(('ꨚꨗꩍ ꨓꨶꨮꩉ ?', 0.96))
        return outs, 0.01

    def _mock_vi_lead(imgs):
        outs = []
        for im in imgs:
            w = im.shape[1]
            if w > 380:
                outs.append(('Người Chăm: 2ZwY ?', 0.88))
            elif w < 190:
                outs.append(('Người Chăm:', 0.97))
            else:
                outs.append(('2ZwY ?', 0.58))
        return outs, 0.01

    router_bi = app.AutoRoutingOCRWrapper(_mock_cham_lead, _mock_vi_lead, cham_w320_model=None, primary_ver='v24_1_r4')
    res_bi, vers_bi, _ = router_bi.predict_with_provenance([mixed_lead_vi])
    assert vers_bi == ['v24_1_r4+vi'], (res_bi, vers_bi)
    assert res_bi[0][0] == 'Người Chăm: ꨚꨗꩍ ꨓꨶꨮꩉ ?', res_bi

    # P2-1: Bounded LRU cache for non-pinned legacy models
    saved_models = dict(app.ocr_models)
    saved_lru = list(app._legacy_model_lru)
    try:
        app.ocr_models.clear()
        app._legacy_model_lru.clear()
        app._register_loaded_model('v24_1_r4', 'core_r4')
        app._register_loaded_model('v23', 'leg_23')
        app._register_loaded_model('v24', 'leg_24')
        app._register_loaded_model('v25', 'leg_25')
        assert 'v24_1_r4' in app.ocr_models
        assert 'v23' not in app.ocr_models
        assert app._legacy_model_lru == ['v24', 'v25']
    finally:
        app.ocr_models.clear()
        app.ocr_models.update(saved_models)
        app._legacy_model_lru[:] = saved_lru


class TestQualityGates(unittest.TestCase):
    pass


for _fn_name, _fn_obj in list(globals().items()):
    if _fn_name.startswith('test_') and callable(_fn_obj):
        setattr(TestQualityGates, _fn_name, staticmethod(_fn_obj))


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print('PASS', t.__name__)
    print(f'{len(tests)} tests passed')
