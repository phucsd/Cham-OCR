import os
import sys
import yaml
from PIL import Image

# Configure UTF-8 stdout to support printing Cham characters to the terminal on Windows
if sys.stdout and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def run_integrity_check(model_dir):
    """
    Kiểm tra sự tồn tại và tính hợp lệ vật lý của các file checkpoint hoặc inference.
    """
    print("🔍 1. Kiểm tra tính toàn vẹn của tệp tin mô hình...")
    if not os.path.exists(model_dir):
        print(f"❌ Lỗi: Thư mục chứa mô hình không tồn tại tại: {model_dir}")
        return False
        
    found_files = os.listdir(model_dir)
    
    # Kiểm tra xem đây có phải là thư mục mô hình inference không
    if "inference.pdiparams" in found_files:
        print("✅ Tìm thấy bộ mô hình Inference ('inference')")
        file_path = os.path.join(model_dir, "inference.pdiparams")
        sz_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"   - inference.pdiparams ({sz_mb:.2f} MB)")
        return True
        
    required_extensions = [".pdparams", ".pdopt", ".states"]
    prefixes = ["best_accuracy", "latest"]
    
    passed = True
    for prefix in prefixes:
        has_prefix = any(f.startswith(prefix) for f in found_files)
        if has_prefix:
            print(f"✅ Tìm thấy bộ checkpoint loại: '{prefix}'")
            # Kiểm tra xem có đủ các file cần thiết không
            for ext in required_extensions:
                expected_file = f"{prefix}{ext}"
                file_path = os.path.join(model_dir, expected_file)
                if expected_file not in found_files:
                    print(f"⚠️  Thiếu tệp: {expected_file}")
                    passed = False
                else:
                    sz_mb = os.path.getsize(file_path) / (1024 * 1024)
                    if sz_mb == 0:
                        print(f"❌ Tệp tin bị rỗng (0 bytes): {expected_file}")
                        passed = False
                    else:
                        print(f"   - {expected_file} ({sz_mb:.2f} MB)")
            break
    else:
        print("❌ Không tìm thấy tệp checkpoint hoặc inference hợp lệ.")
        passed = False
        
    return passed

def calculate_cer(predicted, target):
    """Tính toán Levenshtein Distance để tìm Character Error Rate (CER)"""
    # Mảng 2D cho Levenshtein distance
    d = [[0] * (len(target) + 1) for _ in range(len(predicted) + 1)]
    for i in range(len(predicted) + 1):
        d[i][0] = i
    for j in range(len(target) + 1):
        d[0][j] = j
        
    for i in range(1, len(predicted) + 1):
        for j in range(1, len(target) + 1):
            if predicted[i - 1] == target[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # Deletion
                d[i][j - 1] + 1,        # Insertion
                d[i - 1][j - 1] + cost  # Substitution
            )
            
    if not target:
        return 0.0 if not predicted else 1.0
    return d[len(predicted)][len(target)] / len(target)

def run_inference_qa(model_dir, dict_path, data_dir, num_test_samples=20):
    """
    Tải mô hình và chạy nhận diện thử nghiệm để đánh giá chất lượng (Accuracy, CER).
    """
    print("\n🔮 2. Khởi chạy kiểm tra độ chính xác nhận dạng mô hình...")
    
    # 1. Kiểm tra tập dữ liệu validation
    val_label_path = os.path.join(data_dir, "val_label.txt")
    if not os.path.exists(val_label_path):
        print(f"❌ Không tìm thấy nhãn tập Val: {val_label_path}")
        return None
        
    # 2. Đọc các mẫu kiểm thử từ file nhãn
    test_samples = []
    with open(val_label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                test_samples.append((parts[0], parts[1]))
                
    if not test_samples:
        print("❌ Tập nhãn Val rỗng, không có dữ liệu để chạy thử.")
        return None
        
    # Lấy ngẫu nhiên các mẫu để chạy QA nhanh
    import random
    random.seed(42)  # Cố định ngẫu nhiên
    if len(test_samples) > num_test_samples:
        test_samples = random.sample(test_samples, num_test_samples)
        
    # 3. Khởi tạo PaddleOCR Recognition engine sử dụng TextRecognizer cấp thấp để nạp checkpoint hoặc inference trực tiếp
    try:
        import sys
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        paddleocr_dir = os.path.join(PROJECT_ROOT, "PaddleOCR")
        if paddleocr_dir not in sys.path:
            sys.path.insert(0, paddleocr_dir)
            
        from tools.infer.predict_rec import TextRecognizer
        import tools.infer.utility as utility
        
        best_model_prefix = None
        if "inference.pdiparams" in os.listdir(model_dir):
            best_model_prefix = model_dir
        else:
            for f in os.listdir(model_dir):
                if f.startswith("best_accuracy") and f.endswith(".pdparams"):
                    best_model_prefix = os.path.join(model_dir, "best_accuracy")
                    break
                elif f.startswith("latest") and f.endswith(".pdparams"):
                    best_model_prefix = os.path.join(model_dir, "latest")
                    break
                
        if not best_model_prefix:
            print("❌ Không tìm thấy tiền tố file pdparams để load.")
            return None
            
        print(f"🔄 Đang nạp mô hình từ: {best_model_prefix}")
        
        # Backup sys.argv và xóa tạm thời để tránh argparse lỗi đối số chưa định nghĩa
        sys_argv_backup = sys.argv
        sys.argv = [sys.argv[0]]
        args = utility.parse_args()
        sys.argv = sys_argv_backup
        
        args.rec_model_dir = best_model_prefix
        args.rec_char_dict_path = dict_path
        
        # Cấu hình sử dụng CPU hoặc GPU bằng paddle
        import paddle
        args.use_gpu = paddle.is_compiled_with_cuda()
        args.rec_image_shape = "3, 48, 320"
        args.use_space_char = True
        
        ocr = TextRecognizer(args)
    except Exception as e:
        print(f"❌ Không thể tải bộ thư viện PaddleOCR hoặc trọng số mô hình: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    # 4. Chạy kiểm thử từng ảnh
    total_cer = 0.0
    correct_predictions = 0
    
    print("\n--- Báo cáo Chi tiết Nhận dạng ---")
    for i, (rel_img_path, ground_truth) in enumerate(test_samples):
        img_path = os.path.join(data_dir, rel_img_path)
        if not os.path.exists(img_path):
            continue
            
        try:
            # Chạy nhận diện bằng cv2.imread + TextRecognizer
            import cv2
            img = cv2.imread(img_path)
            if img is not None:
                rec_res, elapse = ocr([img])
                if rec_res and rec_res[0]:
                    pred_text, confidence = rec_res[0]
                else:
                    pred_text, confidence = "", 0.0
            else:
                pred_text, confidence = "", 0.0
                
            cer = calculate_cer(pred_text, ground_truth)
            total_cer += cer
            
            is_correct = (pred_text == ground_truth)
            if is_correct:
                correct_predictions += 1
                
            status_symbol = "✅ Đúng" if is_correct else f"❌ Sai (CER: {cer * 100:.1f}%)"
            print(f"[{i+1:02d}] Ảnh: {os.path.basename(rel_img_path)}")
            print(f"     Nhãn gốc:  '{ground_truth}'")
            print(f"     Nhận diện: '{pred_text}' (Độ tin cậy: {confidence:.2f})")
            print(f"     Kết quả:   {status_symbol}")
            
        except Exception as e:
            print(f"⚠️  Lỗi khi chạy ảnh {rel_img_path}: {e}")
            
    # 5. Tổng kết độ chính xác
    avg_cer = total_cer / len(test_samples)
    word_accuracy = correct_predictions / len(test_samples)
    
    print("\n--- KẾT QUẢ ĐÁNH GIÁ TRỰC QUAN ---")
    print(f"📈 Word-level Accuracy (Độ chính xác từ): {word_accuracy * 100:.2f}%")
    print(f"📉 Average Character Error Rate (CER trung bình): {avg_cer * 100:.2f}%")
    
    # Tiêu chuẩn thông qua QA: CER < 5%
    qa_status = "PASS" if avg_cer < 0.05 else "WARNING" if avg_cer < 0.15 else "FAIL"
    print(f"🏆 Trạng thái chất lượng mô hình: {qa_status.upper()}")
    
    return {
        "word_accuracy": word_accuracy,
        "avg_cer": avg_cer,
        "qa_status": qa_status
    }

if __name__ == '__main__':
    # Đường dẫn kiểm thử mặc định
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_model = os.path.join(PROJECT_ROOT, "data", "output", "rec_cham", "rec_ppocr_v4") # Hoặc đường dẫn đã giải nén
    default_dict = os.path.join(PROJECT_ROOT, "data", "cham_dict.txt")
    default_data = os.path.join(PROJECT_ROOT, "data", "cham_synthetic_images")
    
    if len(sys.argv) > 1:
        default_model = sys.argv[1]
    if len(sys.argv) > 2:
        default_dict = sys.argv[2]
    if len(sys.argv) > 3:
        default_data = sys.argv[3]
        
    print("=== Khởi chạy Kiểm tra Chất lượng Mô hình ===")
    integrity_ok = run_integrity_check(default_model)
    if integrity_ok:
        run_inference_qa(default_model, default_dict, default_data)
    else:
        print("\n❌ Thất bại: Không thể thực hiện kiểm tra độ chính xác do tệp tin checkpoints thiếu hoặc lỗi.")
