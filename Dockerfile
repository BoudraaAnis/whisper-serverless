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

# 3. نسخ ملفات المحرك والمعالج
COPY whisper_ultra_engine.cpp /workspace/
COPY handler.py /workspace/

# 4. بناء محرك C++ ومكتبة whisper عبر CMake مع تسريع CUDA GPU لكروت RTX 3090 و RTX 4090
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git /workspace/whisper_src && \
    cp /workspace/whisper_ultra_engine.cpp /workspace/whisper_src/ && \
    printf "\nadd_executable(whisper_ultra_engine whisper_ultra_engine.cpp)\ntarget_include_directories(whisper_ultra_engine PRIVATE \${CMAKE_CURRENT_SOURCE_DIR}/include \${CMAKE_CURRENT_SOURCE_DIR}/ggml/include)\ntarget_link_libraries(whisper_ultra_engine PRIVATE whisper)\n" >> /workspace/whisper_src/CMakeLists.txt && \
    mkdir -p /workspace/whisper_src/build && \
    cd /workspace/whisper_src/build && \
    cmake -DGGML_CUDA=1 -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES="80;86;89" .. && \
    cmake --build . --config Release --target whisper_ultra_engine -j2 && \
    find . -name "whisper_ultra_engine" -type f -exec cp {} /workspace/whisper_ultra_engine \; && \
    chmod +x /workspace/whisper_ultra_engine && \
    rm -rf /workspace/whisper_src

# 5. تحميل نموذج Large-v3-Turbo المحسن مسبقاً داخل الحاوية
RUN mkdir -p /workspace/models && \
    wget -q -O /workspace/models/ggml-large-v3-turbo.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin

# 6. تشغيل المعالج السحابي عند الطلب
CMD ["python3", "-u", "/workspace/handler.py"]
