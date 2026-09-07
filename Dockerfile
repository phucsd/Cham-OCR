FROM python:3.11-slim
# Rebuild timestamp: 2026-08-01T09:26:20Z - Re-assign components using true_core_coords for full line bbox height [30..78]

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for OpenCV, PaddlePaddle, and OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Force cache bust for updated app code
RUN echo "Cache bust 2026-08-03T10:03:00Z"

# Copy application files
COPY . /app

# Environment variables for CPU optimization & HF Spaces port
ENV PORT=7860
ENV CPU_THREADS=1
ENV ENABLE_MKLDNN=False
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV PYTHONUNBUFFERED=1

# Expose Hugging Face Space default port
EXPOSE 7860

# Run Cham OCR Studio
CMD ["python", "-u", "webapp-ui/app.py"]
