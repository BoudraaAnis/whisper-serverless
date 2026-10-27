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

# 3. تحميل وبناء مكتبة whisper.cpp بـ C++ مع تفعيل CUDA GPU
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git /workspace/whisper.cpp && \
    mkdir -p /workspace/whisper.cpp/build && \
    cd /workspace/whisper.cpp/build && \
    cmake -DGGML_CUDA=1 -DBUILD_SHARED_LIBS=ON .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# 4. نسخ ملفات المحرك
COPY whisper_ultra_engine.cpp /workspace/
COPY handler.py /workspace/

# 5. تجميع محرك whisper_ultra_engine بلغة C++
RUN g++ -O3 -std=c++17 /workspace/whisper_ultra_engine.cpp \
    -I/workspace/whisper.cpp/include \
    -I/workspace/whisper.cpp/ggml/include \
    -L/workspace/whisper.cpp/build \
    -lwhisper -lpthread -ldl \
    -o /workspace/whisper_ultra_engine

# 6. تحميل نموذج Large-v3-Turbo المحسن مسبقاً داخل الحاوية
RUN mkdir -p /workspace/models && \
    wget -q -O /workspace/models/ggml-large-v3-turbo.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin

# ضبط مسار المكتبات المشتركة
ENV LD_LIBRARY_PATH="/workspace/whisper.cpp/build:${LD_LIBRARY_PATH}"

# تشغيل المعالج
CMD ["python3", "-u", "/workspace/handler.py"]
