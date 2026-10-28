import os, sys, json, base64, tempfile, subprocess, re
import runpod
from faster_whisper import WhisperModel

# تحميل النموذج مسبقاً في الذاكرة لسرعة الإقلاع الفوري (FlashBoot)
MODEL_DIR = "/workspace/models"
print("🚀 Loading Whisper Large-v3-Turbo Model on GPU...", flush=True)
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16", download_root=MODEL_DIR)
print("✅ Whisper Ultra Engine is READY on GPU!", flush=True)

def format_time(seconds):
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

def apply_ultra_rules(words_list, custom_dict_str):
    # 1. دمج الأرقام والكسور (مثل 4 و 5 ⬅️ 4.5)
    merged = []
    i = 0
    while i < len(words_list):
        if i + 1 < len(words_list):
            w1, t1 = words_list[i]
            w2, t2 = words_list[i+1]
            if w1.isdigit() and w2.isdigit() and len(w2) == 1 and abs(t2 - t1) <= 0.5:
                merged.append((f"{w1}.{w2}", t1))
                i += 2
                continue
        merged.append(words_list[i])
        i += 1

    # 2. القواعد التقنية الشائعة
    tech_rules = [
        (r"\b(كات\s*اوف|كاتوف|cut\s*off)\b", "Cut-off"),
        (r"\b(ساتوريشن|saturation)\b", "Saturation"),
        (r"\b(ياما)\b", "يا إما"),
        (r"\b(ترانزستور|ترانزيستور)\b", "Transistor"),
        (r"\b(كولكتور|كوليكتور)\b", "Collector"),
        (r"\b(اميتر|إميتر)\b", "Emitter"),
        (r"\b(بيز|بيس)\b", "Base"),
        (r"\b(في\s*اوت|v\s*out)\b", "V output"),
        (r"\b(تو\s*سيجنالز)\b", "Two signals"),
    ]

    # 3. القاموس المخصص للمستخدم
    custom_rules = []
    if custom_dict_str:
        for line in custom_dict_str.splitlines():
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() and v.strip():
                    custom_rules.append((k.strip(), v.strip()))

    # بناء الأسطر بالتنسيق المطلوب: "00:00:00 الكلمة"
    output_lines = []
    for w, t in merged:
        for pat, rep in tech_rules:
            w = re.sub(pat, rep, w, flags=re.IGNORECASE)
        for k, v in custom_rules:
            w = w.replace(k, v)
        output_lines.append(f"{format_time(t)} {w}")

    return "\n".join(output_lines)

def handler(job):
    try:
        job_input = job.get("input", {})
        audio_b64 = job_input.get("audio_base64", "")
        if not audio_b64:
            return {"error": "No audio_base64 provided"}

        use_librosa = job_input.get("use_librosa", True)
        custom_dict = job_input.get("custom_dict", "")

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "input_raw")
            wav_path = os.path.join(tmpdir, "input_16k.wav")

            with open(raw_path, "wb") as f:
                f.write(base64.b64decode(audio_b64))

            # تحويل الصوت بـ FFmpeg عالي السرعة
            ff_cmd = ["ffmpeg", "-y", "-i", raw_path]
            if use_librosa:
                ff_cmd += ["-af", "highpass=f=100,lowpass=f=7500,afftdn=nf=-25"]
            ff_cmd += ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path]
            subprocess.run(ff_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            # تفريغ الصوت العصبي بدقة كلمة بكلمة
            segments, info = model.transcribe(wav_path, word_timestamps=True, vad_filter=True)

            words_list = []
            for seg in segments:
                if seg.words:
                    for w in seg.words:
                        clean_w = w.word.strip()
                        if clean_w:
                            words_list.append((clean_w, w.start))

            result_text = apply_ultra_rules(words_list, custom_dict)
            return {"text": result_text}

    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
