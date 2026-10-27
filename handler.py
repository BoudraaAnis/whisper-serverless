import os
import sys
import json
import base64
import tempfile
import subprocess
import runpod

# مسار النموذج المترجم ومسار محرك C++
ENGINE_BIN = "/workspace/whisper_ultra_engine"
MODEL_TURBO = "/workspace/models/ggml-large-v3-turbo.bin"
MODEL_LARGE = "/workspace/models/ggml-large-v3.bin"

def handler(job):
    """
    معالج RunPod Serverless:
    يستقبل طلب برنامج C++ المكتبي، وينفذ المعالجة عبر محرك C++ و CUDA GPU
    """
    job_input = job.get("input", {})
    
    # 1. استخراج ملف الصوت من Base64
    audio_b64 = job_input.get("audio_base64", "")
    if not audio_b64:
        return {"error": "Missing audio_base64 in payload"}
    
    model_choice = job_input.get("model", "large-v3-turbo")
    merge_fractions = "1" if job_input.get("merge_fractions", True) else "0"
    use_librosa = job_input.get("use_librosa", True)
    custom_dict = job_input.get("custom_dict", "")

    # اختيار مسار النموذج
    model_path = MODEL_TURBO if "turbo" in model_choice else MODEL_LARGE
    if not os.path.exists(model_path):
        model_path = MODEL_TURBO

    with tempfile.TemporaryDirectory() as tmpdir:
        input_audio_path = os.path.join(tmpdir, "input_raw")
        wav_16k_path = os.path.join(tmpdir, "input_16k.wav")
        dict_path = os.path.join(tmpdir, "custom_dict.txt")

        # حفظ ملف الصوت الأولي
        with open(input_audio_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))

        # حفظ القاموس المخصص
        with open(dict_path, "w", encoding="utf-8") as f:
            f.write(custom_dict)

        # 2. تحويل الصوت وتصفيته إلى WAV 16kHz Mono
        # استخدام ffmpeg عالي السرعة مع فلترة الصوت
        if use_librosa:
            # تنقية وعزل الصوت مع فلتر إزالة التشويش
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", input_audio_path,
                "-af", "highpass=f=100,lowpass=f=7500,afftdn=nf=-25",
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                wav_16k_path
            ]
        else:
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", input_audio_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                wav_16k_path
            ]

        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # 3. تشغيل محرك C++ الصافي مع الـ GPU
        engine_cmd = [
            ENGINE_BIN,
            model_path,
            wav_16k_path,
            merge_fractions,
            dict_path
        ]

        proc = subprocess.run(engine_cmd, capture_output=True, text=True, check=True)
        output_stdout = proc.stdout

        # استخراج مخرجات الـ JSON من C++
        if "---JSON_START---" in output_stdout and "---JSON_END---" in output_stdout:
            json_str = output_stdout.split("---JSON_START---")[1].split("---JSON_END---")[0].strip()
            result_data = json.loads(json_str)
            return {
                "text": result_data.get("text", ""),
                "execution_time": result_data.get("execution_time", 0.0),
                "engine": "C++ Whisper.cpp CUDA"
            }
        else:
            return {
                "text": output_stdout.strip(),
                "engine": "C++ Whisper.cpp Direct"
            }

if __name__ == "__main__":
    # بدء خادم RunPod Serverless
    runpod.serverless.start({"handler": handler})
