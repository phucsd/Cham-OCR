import os
import sys
import json
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))
import kaggle_auth
kaggle_auth.init_kaggle_auth()
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "test_mount"))
    os.makedirs(work_dir, exist_ok=True)
    
    nb = {
        'cells': [{
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                "import os\n",
                "print('=== ROOT /kaggle/input ===')\n",
                "if os.path.exists('/kaggle/input'):\n",
                "    print('Direct children:', os.listdir('/kaggle/input'))\n",
                "    for root, dirs, files in os.walk('/kaggle/input'):\n",
                "        print('DIR:', root, '| FILES:', files[:10])\n",
                "else:\n",
                "    print('/kaggle/input does not exist!')\n"
            ]
        }],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'nbformat': 4,
        'nbformat_minor': 4
    }
    
    with open(os.path.join(work_dir, 'test_mount.ipynb'), 'w', encoding='utf-8') as f:
        json.dump(nb, f)
        
    meta = {
        'id': 'gustavnguyen/test-stage1-mount',
        'title': 'test-stage1-mount',
        'code_file': 'test_mount.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': 'true',
        'enable_gpu': 'false',
        'enable_internet': 'true',
        'dataset_sources': ['gustavnguyen/cham-ocr-v5-assets'],
        'kernel_sources': ['gustavnguyen/paddleocr-cham-v25-stage1']
    }
    
    with open(os.path.join(work_dir, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f)
        
    api = KaggleApi()
    api.authenticate()
    print("🚀 Đang đẩy CPU kernel thăm dò lên Kaggle...")
    api.kernels_push(work_dir)
    print("✅ Đã đẩy kernel! Đang chờ thực thi và lấy log...")
    
    for _ in range(30):
        time.sleep(5)
        st = api.kernels_status('gustavnguyen/test-stage1-mount')
        status = st.get('status')
        print(f"📡 Trạng thái: {status}")
        if status in ('COMPLETE', 'ERROR'):
            logs = api.kernels_logs('gustavnguyen/test-stage1-mount')
            print("=" * 60)
            print("📋 LOGS TỪ CPU TEST KERNEL:")
            print(logs)
            print("=" * 60)
            break

if __name__ == '__main__':
    main()
