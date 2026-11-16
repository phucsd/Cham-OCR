#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle Authentication Helper for Cham-OCR
=========================================
Ensures Kaggle API calls use the project's dedicated Kaggle account (gustavnguyen)
read securely from the local .env file, and prevents kagglesdk 2.2.x from
hijacking authentication via ~/.kaggle/access_token or ~/.kaggle/kaggle.json
belonging to other projects/accounts on the host machine.
"""

import os


def init_kaggle_auth():
    """Load KAGGLE_USERNAME and KAGGLE_KEY from .env into os.environ and isolate kagglesdk auth."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
                        os.environ[key] = val

    # Remove any global KAGGLE_API_TOKEN that would override legacy username/key
    os.environ.pop("KAGGLE_API_TOKEN", None)

    username = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    if not username or not key:
        raise RuntimeError("Kaggle credentials missing from environment or .env file.")
    if username != "gustavnguyen":
        raise RuntimeError(f"Active Kaggle username is '{username}', expected 'gustavnguyen'.")

    # Patch kagglesdk in-memory so ~/.kaggle/access_token and ~/.kaggle/kaggle.json
    # from other accounts (e.g. phcnguynhukendykerry) cannot hijack gustavnguyen's session.
    try:
        import kagglesdk.kaggle_env as k_env
        import kagglesdk.kaggle_http_client as k_http
        k_env._get_access_token_from_file = lambda path: None
        k_env.get_access_token_from_env = lambda: (None, None)
        k_http.get_access_token_from_env = lambda: (None, None)
        k_http._get_apikey_creds = lambda: (username, key)
    except Exception:
        pass

    return username


if __name__ == "__main__":
    u = init_kaggle_auth()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    actual_user = api.config_values.get("username")
    assert actual_user == "gustavnguyen", f"Auth mismatch: {actual_user} != gustavnguyen"
    print(f"Kaggle authentication initialized and verified for user: {actual_user}")
