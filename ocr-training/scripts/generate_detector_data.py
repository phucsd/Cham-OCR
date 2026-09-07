import os
import sys
import cv2
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

# Thêm thư mục scratch của artifacts để tải các helper
artifacts_scratch = "C:/Users/admin/.gemini/antigravity/brain/3b1501b2-bb18-4c92-a78b-3d27867b4bce/scratch"
sys.path.insert(0, artifacts_scratch)

from generate_data import unicode_to_visual, get_cached_font
from robustness_stress_test import apply_old_paper_background

def warp_points(pts, matrix, amp, freq, phase):
    """
    Áp dụng cùng phép biến đổi wave và perspective lên các điểm tọa độ ground-truth.
    """
    pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    # 1. Perspective transformation
    warped_persp = cv2.perspectiveTransform(pts_arr, matrix)
    
    # 2. Wave deformation (y_w = y + dy)
    final_pts = []
    for pt in warped_persp:
        x, y = pt[0]
        dy = amp * np.sin(x * freq + phase)
        final_pts.append([float(x), float(y + dy)])
        
    return final_pts

def draw_distorted_page_with_labels(lines, font_path, width=800, height=650):
    font_size = 22
    font = get_cached_font(font_path, font_size)
    bg = apply_old_paper_background(width, height)
    draw = ImageDraw.Draw(bg)
    
    y_start = 70
    y_spacing = 80
    
    # Tính toán tọa độ hộp bao sạch
    line_boxes = []
    for idx, text in enumerate(lines):
        y_pos = y_start + idx * y_spacing
        x_pos = 60
        draw.text((x_pos, y_pos), text, font=font, fill=(35, 30, 25))
        
        # Lấy kích thước text
        try:
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = len(text) * 12, font_size
            
        # 4 điểm góc của hộp bao dòng chữ
        box = [
            [x_pos, y_pos],
            [x_pos + text_w + 10, y_pos],
            [x_pos + text_w + 10, y_pos + text_h + 10],
            [x_pos, y_pos + text_h + 10]
        ]
        line_boxes.append((text, box))
        
    img_np = np.array(bg)
    h, w = img_np.shape[:2]
    
    # 1. Wave distortion parameters
    amp = random.uniform(8.0, 14.0)
    freq = random.uniform(0.005, 0.012)
    phase = random.uniform(0, 2 * np.pi)
    
    # 2. Perspective distortion parameters
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    tx1 = random.uniform(0, w * 0.06)
    ty1 = random.uniform(0, h * 0.06)
    tx2 = random.uniform(0, w * 0.06)
    ty2 = random.uniform(0, h * 0.06)
    tx3 = random.uniform(0, w * 0.06)
    ty3 = random.uniform(0, h * 0.06)
    tx4 = random.uniform(0, w * 0.06)
    ty4 = random.uniform(0, h * 0.06)
    
    pts2 = np.float32([
        [0 + tx1, 0 + ty1],
        [w - tx2, 0 + ty2],
        [0 + tx3, h - ty3],
        [w - tx4, h - ty4]
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    
    # Thực hiện warp ảnh
    # Wave
    x_idx = np.arange(w)
    y_idx = np.arange(h)
    X, Y = np.meshgrid(x_idx, y_idx)
    dy = amp * np.sin(X * freq + phase)
    map_x = X.astype(np.float32)
    map_y = (Y + dy).astype(np.float32)
    img_wave = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # Perspective
    img_warped = cv2.warpPerspective(img_wave, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Áp dụng warp cho các điểm tọa độ nhãn dòng
    warped_labels = []
    for text, box in line_boxes:
        warped_box = warp_points(box, matrix, amp, freq, phase)
        warped_labels.append({
            "transcription": unicode_to_visual(text),
            "points": warped_box
        })
        
    return Image.fromarray(img_warped), warped_labels

def main():
    print("📂 Khởi tạo dữ liệu huấn luyện và validation cho DBNet...")
    
    # Tải các dòng corpus mẫu
    corpus_file = os.path.join(PROJECT_ROOT, "data", "corpus", "cham_text.txt")
    if not os.path.exists(corpus_file):
        print("❌ Không tìm thấy corpus!")
        return
        
    with open(corpus_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        
    font_path = os.path.join(PROJECT_ROOT, "data", "fonts", "NotoSansCham-Regular.ttf")
    
    # Thiết lập thư mục detector
    det_root = os.path.join(PROJECT_ROOT, "data", "detector")
    train_img_dir = os.path.join(det_root, "train_images")
    val_img_dir = os.path.join(det_root, "val_images")
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    
    # Thiết lập chạy thử nghiệm quy mô nhỏ 5 train / 3 val, nâng lên 100/30 khi chạy full
    num_train = 5
    num_val = 3
    
    print(f"  - Sinh {num_train} trang train_pages...")
    train_label_entries = []
    for i in range(num_train):
        num_lines = random.randint(5, 8)
        lines = random.sample(sentences, num_lines)
        img, labels = draw_distorted_page_with_labels(lines, font_path)
        
        img_name = f"det_train_{i:04d}.png"
        img.save(os.path.join(train_img_dir, img_name))
        train_label_entries.append(f"train_images/{img_name}\t{json.dumps(labels, ensure_ascii=False)}")
        
    with open(os.path.join(det_root, "det_train_label.txt"), "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write("\n".join(train_label_entries) + "\n")
        
    print(f"  - Sinh {num_val} trang val_pages...")
    val_label_entries = []
    for i in range(num_val):
        num_lines = random.randint(5, 8)
        lines = random.sample(sentences, num_lines)
        img, labels = draw_distorted_page_with_labels(lines, font_path)
        
        img_name = f"det_val_{i:04d}.png"
        img.save(os.path.join(val_img_dir, img_name))
        val_label_entries.append(f"val_images/{img_name}\t{json.dumps(labels, ensure_ascii=False)}")
        
    with open(os.path.join(det_root, "det_val_label.txt"), "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write("\n".join(val_label_entries) + "\n")
        
    print("🎉 Sinh dữ liệu huấn luyện và kiểm chứng Detector thành công!")

if __name__ == '__main__':
    main()
