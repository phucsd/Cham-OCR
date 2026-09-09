import sys
import requests
import json
import base64

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("=== KIỂM THỬ GIAO DIỆN & API TINH GIẢN HÓA (CHỈ CẦN ẢNH & MÔ HÌNH) ===")

# 1. Kiểm tra HTML đã gỡ bỏ hoàn toàn khối cấu hình thủ công
html_resp = requests.get('https://ocr.cham.asia', timeout=30)
html_text = html_resp.text
has_adv = 'advanced-settings-content' in html_text
has_method_sel = 'method-select' in html_text
has_thresh = 'threshold-input' in html_text
print(f"1. Kiểm tra HTML giao diện Web:")
print(f"   - Mã phản hồi HTTP: {html_resp.status_code}")
print(f"   - Còn chứa khối 'Cấu hình nâng cao': {has_adv}")
print(f"   - Còn chứa menu chọn thuật toán: {has_method_sel}")
print(f"   - Còn chứa input ngưỡng nhị phân: {has_thresh}")

if not has_adv and not has_method_sel and not has_thresh:
    print("   👉 Giao diện Web đã được tinh giản hoàn hảo!")
else:
    print("   ⚠️ Vẫn còn sót thành phần cũ trên giao diện.")

# 2. Kiểm tra API POST /ocr với payload tối giản
with open('ocr-training/output/evidence/page_ocr/page_0.png', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')

print(f"\n2. Gửi yêu cầu OCR với Payload tối giản (không gửi method/threshold):")
payload = {
    'image': 'data:image/png;base64,' + img_b64,
    'model': 'v24'
}

resp = requests.post('https://ocr.cham.asia/ocr', json=payload, timeout=60)
data = resp.json()

if 'error' in data:
    print("❌ Lỗi:", data['error'])
else:
    lines = data.get('lines', [])
    profiler = data.get('profiler', {})
    print(f"✅ Thành công 100%! Phát hiện {len(lines)} dòng chữ:")
    print(f"   - Thuật toán phân đoạn tự động kích hoạt: {profiler.get('method')}")
    print(f"   - Tổng thời gian xử lý: {profiler.get('backend_total_sec')}s")
    for idx, l in enumerate(lines):
        print(f"   [Dòng {idx+1}] Độ tin cậy: {l['confidence']:.4f} | BBox: {l['bbox']} | Text: {l['text']}")
