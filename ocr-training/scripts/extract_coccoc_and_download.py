import os
import sys
import json
import sqlite3
import shutil
import base64
import requests
import win32crypt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def get_coccoc_kaggle_cookies():
    user_data = os.path.expandvars(r'%LOCALAPPDATA%\CocCoc\Browser\User Data')
    local_state_path = os.path.join(user_data, 'Local State')
    
    if not os.path.exists(local_state_path):
        print("❌ Không tìm thấy thư mục Cốc Cốc User Data!")
        return {}
        
    with open(local_state_path, 'r', encoding='utf-8') as f:
        local_state = json.load(f)
        
    encrypted_key = base64.b64decode(local_state['os_crypt']['encrypted_key'])[5:]
    master_key = win32crypt.CryptUnprotectData(encrypted_key, None, None, None, 0)[1]
    
    cookies_db = os.path.join(user_data, 'Default', 'Network', 'Cookies')
    temp_db = 'temp_coccoc_cookies.db'
    shutil.copy2(cookies_db, temp_db)
    
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name, encrypted_value FROM cookies WHERE host_key LIKE '%kaggle.com%'")
    
    kaggle_cookies = {}
    for name, enc_val in cursor.fetchall():
        try:
            if enc_val[:3] == b'v10':
                nonce = enc_val[3:15]
                ciphertext = enc_val[15:-16]
                tag = enc_val[-16:]
                cipher = Cipher(algorithms.AES(master_key), modes.GCM(nonce, tag))
                decryptor = cipher.decryptor()
                val = decryptor.update(ciphertext) + decryptor.finalize()
                kaggle_cookies[name] = val.decode('utf-8', errors='ignore')
        except Exception:
            pass
            
    conn.close()
    if os.path.exists(temp_db):
        os.remove(temp_db)
        
    return kaggle_cookies

def main():
    print("🔑 Đang trích xuất phiên đăng nhập Kaggle từ trình duyệt Cốc Cốc...")
    cookies = get_coccoc_kaggle_cookies()
    print(f"✅ Đã tìm thấy {len(cookies)} cookies của kaggle.com:")
    for k in cookies:
        print(f"   - {k}")
        
    session = requests.Session()
    session.cookies.update(cookies)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.kaggle.com/code/gustavnguyen/paddleocr-cham-finetune/output"
    })
    
    # Kiểm tra xác thực
    res = session.get("https://www.kaggle.com/api/i/users.UserService/GetCurrentUser")
    print(f"Current User API status: {res.status_code}")
    if res.status_code == 200:
        print("User Info:", res.text[:200])

if __name__ == '__main__':
    main()
