import os
import sys
import time
import threading
import queue

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.environ['KAGGLE_USERNAME'] = 'gustavnguyen'
os.environ['KAGGLE_KEY'] = '6bf56db7e5c0fa7895d157167961d92b'

from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()
    
    q = queue.Queue()
    stop_event = threading.Event()
    
    def worker():
        try:
            for event in api.kernels_logs_stream('gustavnguyen/paddleocr-cham-finetune'):
                if stop_event.is_set():
                    break
                data = event.get('data', '')
                if data:
                    q.put(data)
        except Exception:
            pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    
    raw_text = []
    # Thu thập tất cả log đã có (cho đến khi không còn event mới trong 3 giây)
    t_start = time.time()
    last_recv = time.time()
    
    while time.time() - t_start < 20:
        try:
            chunk = q.get(timeout=2.0)
            raw_text.append(chunk)
            last_recv = time.time()
        except queue.Empty:
            if raw_text and (time.time() - last_recv > 3.0):
                # Đã đọc hết log hiện có
                break
                
    stop_event.set()
    
    full_str = "".join(raw_text)
    clean_lines = []
    for line in full_str.splitlines():
        l = line.strip()
        if l and 'MB/s' not in l and 'â”' not in l and not l.startswith('\x1b['):
            clean_lines.append(l)
            
    print(f"📊 Thu thập được {len(clean_lines)} dòng log từ Kaggle:")
    print("="*70)
    for l in clean_lines[-40:]:
        print(l)
    print("="*70)

if __name__ == '__main__':
    main()
