import sys
import requests
import json
import base64

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

with open('ocr-training/output/evidence/page_ocr/page_0.png', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')

print("=== TEST 1: Model V24 + Valley Segmentation ===")
payload1 = {
    'image': 'data:image/png;base64,' + img_b64,
    'model': 'v24',
    'method': 'valley',
    'threshold': 0.05,
    'gap': 12,
    'window': 25
}

resp1 = requests.post('https://ocr.cham.asia/ocr', json=payload1, timeout=60)
data1 = resp1.json()

if 'error' in data1:
    print("❌ Error:", data1['error'])
else:
    lines = data1.get('lines', [])
    print(f"✅ Thành công! Số dòng nhận diện: {len(lines)}")
    for idx, l in enumerate(lines):
        print(f"  [Dòng {idx+1}] Độ tin cậy: {l['confidence']:.4f} | BBox: {l['bbox']} | Text: {l['text']}")
    print("⏱️ Profiler:", json.dumps(data1.get('profiler'), indent=2))

print("\n=== TEST 2: Model V24 + DBNet Segmentation ===")
payload2 = {
    'image': 'data:image/png;base64,' + img_b64,
    'model': 'v24',
    'method': 'dbnet'
}

resp2 = requests.post('https://ocr.cham.asia/ocr', json=payload2, timeout=60)
data2 = resp2.json()

if 'error' in data2:
    print("❌ Error:", data2['error'])
else:
    lines = data2.get('lines', [])
    print(f"✅ Thành công! Số dòng nhận diện: {len(lines)}")
    for idx, l in enumerate(lines):
        print(f"  [Dòng {idx+1}] Độ tin cậy: {l['confidence']:.4f} | BBox: {l['bbox']} | Text: {l['text']}")
    print("⏱️ Profiler:", json.dumps(data2.get('profiler'), indent=2))
