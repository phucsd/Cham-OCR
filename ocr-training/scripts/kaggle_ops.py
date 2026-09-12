import os
import sys
import time
import zipfile
import socket
from kaggle.api.kaggle_api_extended import KaggleApi

# Thiết lập timeout cho socket để tránh bị treo kết nối vô hạn trên Windows
socket.setdefaulttimeout(30)

# Configure UTF-8 standard output for Windows console stability
if sys.stdout and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def get_authenticated_api():
    """Khởi tạo và xác thực với Kaggle API"""
    try:
        try:
            from scripts.kaggle_auth import init_kaggle_auth
        except ImportError:
            from kaggle_auth import init_kaggle_auth
        init_kaggle_auth()

        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        print(f"❌ Lỗi xác thực Kaggle API: {e}")
        print("Vui lòng đảm bảo thông tin xác thực Kaggle hợp lệ.")
        return None

def monitor_kernel(user, slug, poll_interval_seconds=60):
    """
    Theo dõi liên tục trạng thái chạy của Kaggle notebook.
    Trả về True nếu hoàn thành thành công, False nếu thất bại.
    """
    api = get_authenticated_api()
    if not api:
        return False
        
    print(f"👁️  Bắt đầu theo dõi Kernel: {user}/{slug}")
    print("⏱️  Đợi 15 giây để máy chủ Kaggle cập nhật trạng thái mới sau khi push...")
    time.sleep(15)
    start_time = time.time()
    
    last_status = None
    while True:
        try:
            res = api.kernels_status(f"{user}/{slug}")
            if hasattr(res, 'status') and hasattr(res.status, 'name'):
                status = res.status.name.lower()
            else:
                status = 'unknown'
            
            # Chỉ in trạng thái khi có sự thay đổi để tránh tràn màn hình log
            if status != last_status:
                elapsed_minutes = int((time.time() - start_time) / 60)
                print(f"⏱️  [Sau {elapsed_minutes} phút] Trạng thái Kernel: {status.upper()}")
                last_status = status
                
            if status == 'complete':
                print("🎉 Tiến trình máy ảo Kaggle đã kết thúc chạy code.")
                return True
            elif status in ['error', 'cancel', 'aborted']:
                print(f"❌ Kernel kết thúc với trạng thái lỗi từ hệ thống: {status.upper()}")
                return False
                
        except Exception as e:
            print(f"⚠️  Lỗi khi kiểm tra trạng thái Kernel: {e}")
            
        time.sleep(poll_interval_seconds)

def download_kernel_outputs(user, slug, download_dir):
    """
    Tải toàn bộ file đầu ra (ZIP checkpoint) từ Kaggle và giải nén cục bộ.
    """
    api = get_authenticated_api()
    if not api:
        return False
        
    os.makedirs(download_dir, exist_ok=True)
    print(f"📥 Đang tải các tệp đầu ra từ Kaggle {user}/{slug} về thư mục: {download_dir}...")
    
    # Tăng default timeout khi tải để tránh lỗi timeout do khối lượng file lớn (ảnh + PaddleOCR)
    import socket
    socket.setdefaulttimeout(600)
    
    try:
        # Tải đầu ra thông qua Kaggle API, chỉ lọc các file .zip để tải về cực nhanh
        api.kernels_output(f"{user}/{slug}", path=download_dir, file_pattern=r'.*\.zip$', page_size=100)
        print("✅ Tải hoàn tất.")
        
        # Tìm file .zip để giải nén tự động
        zip_files = [f for f in os.listdir(download_dir) if f.endswith('.zip')]
        if not zip_files:
            print("❌ LỖI: Không tìm thấy bất kỳ tệp kết quả .zip nào trong thư mục tải về!")
            print("Có thể tiến trình huấn luyện đã bị sập sớm trên Kaggle nên không xuất được tệp trọng số.")
            return False
            
        for zip_f in zip_files:
            zip_path = os.path.join(download_dir, zip_f)
            # Kiểm tra kích thước file zip
            if os.path.getsize(zip_path) < 1024: # Dưới 1KB là file rỗng/lỗi
                print(f"❌ LỖI: Tệp {zip_f} có dung lượng quá nhỏ ({os.path.getsize(zip_path)} bytes), có thể bị lỗi.")
                return False
                
            extract_path = os.path.join(download_dir, zip_f.replace('.zip', ''))
            os.makedirs(extract_path, exist_ok=True)
            
            print(f"📦 Đang giải nén {zip_f} vào {extract_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"✅ Giải nén xong: {extract_path}")
            
        return True
    except Exception as e:
        print(f"❌ Lỗi khi tải hoặc giải nén kết quả: {e}")
        return False

def upload_dataset_version(dataset_id, folder_path, version_notes="New model version from local training"):
    """
    Tải thư mục lên Kaggle dưới dạng một phiên bản mới của Dataset.
    dataset_id: định dạng 'username/dataset-slug'
    """
    api = get_authenticated_api()
    if not api:
        return False
        
    if not os.path.exists(folder_path):
        print(f"❌ Thư mục không tồn tại: {folder_path}")
        return False
        
    print(f"📤 Đang tạo phiên bản mới cho Dataset: {dataset_id} từ: {folder_path}...")
    try:
        api.dataset_create_version(folder_path, version_notes, dir_mode='zip')
        print("✅ Đã đẩy lên Kaggle Dataset thành công!")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi cập nhật Dataset Kaggle: {e}")
        return False

def push_kernel(folder_path="."):
    """
    Đẩy kernel lên Kaggle sử dụng CLI và luôn luôn ép buộc bộ tăng tốc GPU tối ưu (NvidiaTeslaT4)
    để cấp phát GPU T4x2 song song.
    """
    import subprocess
    print("🚀 Đang chạy lệnh đẩy Notebook lên Kaggle...")
    print("🔥 Bắt buộc sử dụng bộ tăng tốc: NvidiaTeslaT4 (GPU T4x2 phân tán)...")
    cmd = "kaggle kernels push -p . --accelerator NvidiaTeslaT4"
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)
        print("✅ Thành công!")
        print(res.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi đẩy kernel: {e}")
        print(f"Stderr: {e.stderr}")
        return False

if __name__ == '__main__':
    # Hướng dẫn chạy thử nghiệm
    if len(sys.argv) < 2:
        print("Sử dụng:")
        print("  python scripts/kaggle_ops.py push")
        print("  python scripts/kaggle_ops.py monitor <username> <slug>")
        print("  python scripts/kaggle_ops.py download <username> <slug> <output_dir>")
        sys.exit(1)
        
    action = sys.argv[1]
    if action == 'push':
        folder = sys.argv[2] if len(sys.argv) > 2 else "."
        push_kernel(folder)
    elif action == 'monitor':
        if len(sys.argv) < 4:
            print("Thiếu tham số: python scripts/kaggle_ops.py monitor <username> <slug>")
            sys.exit(1)
        monitor_kernel(sys.argv[2], sys.argv[3])
    elif action == 'download':
        if len(sys.argv) < 5:
            print("Thiếu tham số: python scripts/kaggle_ops.py download <username> <slug> <output_dir>")
            sys.exit(1)
        download_kernel_outputs(sys.argv[2], sys.argv[3], sys.argv[4])
