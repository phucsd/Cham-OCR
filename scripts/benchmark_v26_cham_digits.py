#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cham digits 0..9 and Latin digits, three NotoSansCham held-out fonts, /ocr_crop.

Methodological fix (2026-11-22):
- Inverts grayscale (`ImageOps.invert(image.convert('L')).getbbox()`) before computing the ink
  bounding box (`Image.getbbox()` on a white RGB canvas otherwise returns the full 1100x100 canvas,
  horizontally compressing ~75% blank white margin into W=480).
- Crops tightly around ink with realistic DBNet textline margins (pad_x=12, pad_y=8).
- Computes `digit_correct` via minimum-edit Levenshtein alignment rather than positional `zip()`.
- Writes to a separate `--out-dir` (`digits_tight_retest` by default) so historical `2026-11-22/digits`
  artifacts remain untouched.
"""
import argparse,collections,concurrent.futures,hashlib,json,statistics,sys
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps,features
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_all_live_v26 import MODELS as ALL_MODELS
MODELS=['v24_1_r4','v25','auto','v24_1_r3_1']
from live_symbol_crop_benchmark import call,norm
sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
DEFAULT_OUT=ROOT/'REPORTS/live_v26_audit_2026_11_22/digits_tight_retest'
fonts=['NotoSansCham-Light.ttf','NotoSansCham-Medium.ttf','NotoSansCham-SemiBold.ttf']
base=ROOT/'ocr-benchmark/data/fonts_heldout'
digits=''.join(chr(0xAA50+i) for i in range(10))


def aligned_digit_correct(gt: str, pred: str) -> int:
    """Counts matched digit characters along a minimum-edit Levenshtein alignment."""
    a, b = norm(gt), norm(pred)
    m, n = len(a), len(b)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
            )
    hits = 0
    i, j = m, n
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] == b[j - 1] and (0xAA50 <= ord(a[i - 1]) <= 0xAA59 or a[i - 1].isdigit()):
                hits += 1
            i -= 1
            j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return hits


def render(font, gt, name, image_dir, pad_x=12, pad_y=8):
    ft = ImageFont.truetype(str(base / font), 46)
    canvas = Image.new('RGB', (1200, 120), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 24), gt, font=ft, fill='black')
    ink_bb = ImageOps.invert(canvas.convert('L')).getbbox()
    if ink_bb:
        x0 = max(0, ink_bb[0] - pad_x)
        y0 = max(0, ink_bb[1] - pad_y)
        x1 = min(canvas.width, ink_bb[2] + pad_x)
        y1 = min(canvas.height, ink_bb[3] + pad_y)
        image = canvas.crop((x0, y0, x1, y1))
    else:
        image = canvas
    file = image_dir / (name + '.png')
    image.save(file)
    return dict(
        id=name,
        kind='digit',
        group='Cham 0-9' if all(0xAA50 <= ord(c) <= 0xAA59 or c == ' ' for c in gt) else 'mixed digit',
        gt=gt,
        path=file,
        w=image.width,
        h=image.height,
        font=font,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='https://ocr.cham.asia')
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--out-dir', default=str(DEFAULT_OUT), help='Output directory (defaults to digits_tight_retest to preserve historical 2026-11-22/digits)')
    ap.add_argument('--fresh', action='store_true', help='Ignore cached results.jsonl and re-run cleanly')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    image_dir = out_dir / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(',') if m.strip()]

    texts = [digits, ' '.join(digits), '꩒꩐꩒꩖ ꩐꩙ ꩑꩘', '꩙꩘꩗꩖꩕ ꩔꩓꩒꩑꩐']
    cases = [render(f, gt, f.replace('.ttf', '') + '_' + str(i), image_dir) for f in fonts for i, gt in enumerate(texts)]
    case_sha = {c['id']: hashlib.sha256(c['path'].read_bytes()).hexdigest() for c in cases}

    j = out_dir / 'results.jsonl'
    old = {}
    if j.exists() and not args.fresh:
        for x in j.read_text(encoding='utf-8').splitlines():
            try:
                r = json.loads(x)
                # Invalidate stale records if image crop SHA256 changed (e.g. uncropped 1100x100 bug)
                if r.get('sha256') == case_sha.get(r.get('id')):
                    old[(r['id'], r['model'])] = r
            except ValueError:
                pass
    elif args.fresh and j.exists():
        j.unlink()

    todo = [(c, m) for c in cases for m in models if (c['id'], m) not in old]
    print('DIGITS_START', len(cases), 'images', len(todo), 'calls, RAQM', features.check('raqm'), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = {ex.submit(call, c, m, args.url, 90): (c, m) for c, m in todo}
        for idx, f in enumerate(concurrent.futures.as_completed(fut), 1):
            c, m = fut[f]
            r = f.result()
            r['font'] = c['font']
            r['crop_size'] = [c['w'], c['h']]
            r['digit_correct'] = aligned_digit_correct(r.get('gt', ''), r.get('pred', '')) if not r.get('error') else None
            old[(c['id'], m)] = r
            with j.open('a', encoding='utf-8') as h:
                h.write(json.dumps(r, ensure_ascii=False) + '\n')
            if idx % 12 == 0 or r.get('error'):
                print('DIGITS_PROGRESS', idx, '/', len(todo), 'model', m, 'CER', r.get('cer'), 'ERR', r.get('error'), flush=True)

    rows = list(old.values())
    summary = {}
    for m in models:
        a = [r for r in rows if r['model'] == m]
        good = [r for r in a if not r.get('error')]
        by = {}
        for group in set(r['group'] for r in good):
            g = [r for r in good if r['group'] == group]
            by[group] = {
                'n': len(g),
                'cer_pct': round(100 * sum(r['edits'] for r in g) / max(1, sum(r['gt_chars'] for r in g)), 2),
                'exact_pct': round(100 * sum(r['exact'] for r in g) / len(g), 2),
            }
        summary[m] = {'n': len(good), 'errors': len(a) - len(good), 'groups': by}
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DIGITS_DONE', len(rows), json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
