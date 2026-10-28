# ==============================================================================
# 🚀 Dockerfile: TurboScribe Ultra Serverless (NVIDIA GPU Powered)
# ==============================================================================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# تثبيت الحزم ومكتبات الصوت
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# تثبيت محرك Faster-Whisper و RunPod Serverless
RUN pip3 install --no-cache-dir runpod faster-whisper

# تحميل نموذج Large-v3-Turbo مسبقاً داخل الحاوية ليكون الإقلاع فورياً
RUN python3 -c 'from faster_whisper import WhisperModel; WhisperModel("large-v3-turbo", download_root="/workspace/models")'

COPY handler.py /workspace/

CMD ["python3", "-u", "/workspace/handler.py"]
