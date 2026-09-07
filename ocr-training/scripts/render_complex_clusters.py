import os
import sys
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fonts_dir = os.path.join(PROJECT_ROOT, "data", "fonts")
output_dir = os.path.join(PROJECT_ROOT, "data", "output", "evidence", "cluster_renders")
os.makedirs(output_dir, exist_ok=True)

CLUSTERS = [
    ('ꨆꨴꨯ', 'cluster_1'),
    ('ꨆꨴꨰ', 'cluster_2'),
    ('ꨆꨵꨴꨯ', 'cluster_3'),
    ('ꨆꨶꨴꨯ', 'cluster_4'),
    ('ꨆꨴꨯꨱꩃ', 'cluster_5')
]

FONTS = [
    'NotoSansCham-Regular.ttf',
    'NotoSansCham-Bold.ttf',
    'NotoSansCham-Black.ttf'
]

def render_cluster(text, font_path, output_path):
    font_size = 32
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        return False
        
    # Lấy kích thước hộp giới hạn của chữ
    try:
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        w = max(120, text_w + 40)
        h = 48
    except Exception:
        w, h = 180, 48
        
    # Tạo ảnh nền trắng
    img = Image.new('RGB', (w, h), color='white')
    draw = ImageDraw.Draw(img)
    
    # Vẽ chữ ở giữa ảnh theo chiều đứng
    draw.text((20, 2), text, font=font, fill='black')
    
    # Vẽ viền xám mỏng bao quanh dải ảnh
    draw.rectangle([(0, 0), (w-1, h-1)], outline='gray', width=1)
    
    img.save(output_path)
    return True

def main():
    print("🎨 Bắt đầu sinh ảnh bằng chứng cụm diacritic phức tạp...")
    success_count = 0
    for font_name in FONTS:
        font_path = os.path.join(fonts_dir, font_name)
        if not os.path.exists(font_path):
            print(f"  ❌ Không tìm thấy font: {font_path}")
            continue
            
        print(f"  - Đang vẽ trên font: {font_name}...")
        for text, c_id in CLUSTERS:
            output_name = f"{font_name.replace('.ttf', '')}_{c_id}.png"
            output_path = os.path.join(output_dir, output_name)
            
            ok = render_cluster(text, font_path, output_path)
            if ok:
                print(f"    ✅ Đã sinh: {output_name}")
                success_count += 1
            else:
                print(f"    ❌ Lỗi vẽ: {output_name}")
                
    print(f"\n✅ Đã hoàn thành sinh {success_count} ảnh bằng chứng cụm chữ diacritic khó!")

if __name__ == "__main__":
    main()
