import subprocess
import sys
import os

# Get path of webapp-ui directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(PROJECT_ROOT, "app.py")

# Environment settings to prevent OpenMP deadlocks on Windows
env = os.environ.copy()
env["CPU_THREADS"] = "1"
env["ENABLE_MKLDNN"] = "False"
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Starting Cham OCR Studio in a new clean console window...")
subprocess.Popen(
    [sys.executable, "-u", app_path],
    cwd=os.path.dirname(PROJECT_ROOT), # Run from workspace root to match relative paths
    env=env,
    creationflags=subprocess.CREATE_NEW_CONSOLE
)
print("Cham OCR Studio launched successfully in a separate desktop window.")
