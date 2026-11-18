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
    assert info_r4['recognizers'][0]['weights_sha256_16'] == '85525b0ca091acaf', info_r4
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
