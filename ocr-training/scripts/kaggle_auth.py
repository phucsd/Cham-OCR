#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hàm tiện ích nạp Kaggle credentials an toàn từ biến môi trường hoặc tệp .env (không hardcode key).
"""

import os
import sys

def init_kaggle_auth():
    """Tự động nạp Kaggle credentials an toàn từ .env hoặc biến môi trường."""
    os.environ["PYTHONUTF8"] = "1"
    # 1. Kiểm tra nếu đã có trong môi trường
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"]
        
    # 2. Tìm tệp .env trong cây thư mục
    cur = os.path.abspath(os.path.dirname(__file__))
    for _ in range(4):
        env_path = os.path.join(cur, ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k in ["KAGGLE_USERNAME", "KAGGLE_KEY"] and k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass
            break
        cur = os.path.dirname(cur)
        
    if "KAGGLE_USERNAME" not in os.environ:
        os.environ["KAGGLE_USERNAME"] = "gustavnguyen"
        
    return os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")

if __name__ == "__main__":
    u, k = init_kaggle_auth()
    print(f"Kaggle User: {u}, Key loaded: {'Yes' if k else 'No'}")
