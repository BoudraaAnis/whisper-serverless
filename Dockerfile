# ==============================================================================
# 🚀 Dockerfile: TurboScribe Ultra Audit (Pure C++ + CUDA GPU on RunPod)
# ==============================================================================
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# 1. تثبيت أدوات البناء و ffmpeg و Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ffmpeg \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 2. تثبيت مكتبة RunPod Serverless الرسمية
RUN pip3 install --no-cache-dir runpod

# 3. نسخ ملفات المشروع والبناء
COPY CMakeLists.txt whisper_ultra_engine.cpp handler.py /workspace/

# 4. بناء محرك C++ بالكامل عبر CMake مع تسريع CUDA لكروت RTX 3090 و RTX 4090
RUN cmake -B /workspace/build -S /workspace && \
    cmake --build /workspace/build --config Release -j2 && \
    find /workspace/build -name "whisper_ultra_engine" -type f -exec cp {} /workspace/whisper_ultra_engine \; && \
    chmod +x /workspace/whisper_ultra_engine && \
    rm -rf /workspace/build

# 5. تحميل نموذج Large-v3-Turbo المحسن مسبقاً داخل الحاوية
RUN mkdir -p /workspace/models && \
    wget -q -O /workspace/models/ggml-large-v3-turbo.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin

# 6. تشغيل المعالج السحابي عند الطلب
CMD ["python3", "-u", "/workspace/handler.py"]
