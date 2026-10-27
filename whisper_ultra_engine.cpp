#include "whisper.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <cstring>

// ==============================================================================
// 🎙️ محرك التدقيق فائق السرعة والدقة (Whisper Large-v3 + Ultra Audit C++)
// التنسيق المطلوب: "00:00:00 الكلمة" (كلمة كاملة مدققة لكل سطر مع التوقيت)
// ==============================================================================

struct WordItem {
    std::string word;
    int64_t t0;
};

std::string format_time(int64_t t_ms) {
    if (t_ms < 0) t_ms = 0;
    int total_sec = static_cast<int>(t_ms / 100);
    int h = total_sec / 3600;
    int m = (total_sec % 3600) / 60;
    int s = total_sec % 60;
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << h << ":"
        << std::setfill('0') << std::setw(2) << m << ":"
        << std::setfill('0') << std::setw(2) << s;
    return oss.str();
}

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// Read WAV (16-bit PCM mono 16kHz)
bool read_wav(const std::string& filename, std::vector<float>& pcmf32) {
    std::ifstream wav(filename, std::ios::binary);
    if (!wav.is_open()) return false;

    char riff[4];
    wav.read(riff, 4);
    if (wav.gcount() < 4 || std::string(riff, 4) != "RIFF") return false;

    wav.seekg(8);
    char wave[4];
    wav.read(wave, 4);
    if (wav.gcount() < 4 || std::string(wave, 4) != "WAVE") return false;

    char chunk_id[4];
    uint32_t chunk_size = 0;
    bool found_data = false;
    while (wav.read(chunk_id, 4) && wav.read(reinterpret_cast<char*>(&chunk_size), 4)) {
        if (std::string(chunk_id, 4) == "data") {
            found_data = true;
            break;
        }
        wav.seekg(chunk_size, std::ios::cur);
    }

    if (!found_data || chunk_size == 0) return false;

    size_t num_samples = chunk_size / sizeof(int16_t);
    std::vector<int16_t> samples(num_samples);
    wav.read(reinterpret_cast<char*>(samples.data()), chunk_size);

    pcmf32.resize(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        pcmf32[i] = static_cast<float>(samples[i]) / 32768.0f;
    }
    return true;
}

std::string replace_all(std::string str, const std::string& from, const std::string& to) {
    if (from.empty()) return str;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

// استخراج وتجميع الكلمات الكاملة مع الطابع الزمني وتجاوز الرموز الخاصة
std::vector<WordItem> extract_words(struct whisper_context* ctx) {
    std::vector<WordItem> result;
    const int n_segments = whisper_full_n_segments(ctx);

    for (int i = 0; i < n_segments; ++i) {
        const int n_tokens = whisper_full_n_tokens(ctx, i);
        int64_t seg_t0 = whisper_full_get_segment_t0(ctx, i);
        int64_t seg_t1 = whisper_full_get_segment_t1(ctx, i);

        std::string current_word = "";
        int64_t current_t0 = seg_t0;

        for (int j = 0; j < n_tokens; ++j) {
            std::string t_text = whisper_full_get_token_text(ctx, i, j);
            int64_t t0 = whisper_full_get_token_t0(ctx, i, j);

            // استبعاد الرموز الداخلية مثل [_BEG_] و [_TT_xxx] و <|...|>
            if (t_text.empty()) continue;
            if (t_text.find("[_") != std::string::npos || t_text.find("<|") != std::string::npos) {
                continue;
            }

            // فحص بداية الكلمة الجديدة (تبدأ بمسافة في ترميز Whisper BPE)
            bool is_new_word = (!t_text.empty() && (t_text[0] == ' ' || t_text[0] == '\t' || t_text[0] == '\n'));
            std::string clean_piece = trim(t_text);
            if (clean_piece.empty()) continue;

            if (is_new_word && !current_word.empty()) {
                result.push_back({current_word, current_t0});
                current_word = clean_piece;
                current_t0 = (t0 >= 0) ? t0 : seg_t0;
            } else {
                if (current_word.empty()) {
                    current_word = clean_piece;
                    current_t0 = (t0 >= 0) ? t0 : seg_t0;
                } else {
                    current_word += clean_piece;
                }
            }
        }

        if (!current_word.empty()) {
            result.push_back({current_word, current_t0});
        }
    }

    // صمام أمان: في حال عدم توفر التوكنات، يتم تجزئة نص الجملة تلقائياً
    if (result.empty()) {
        for (int i = 0; i < n_segments; ++i) {
            const char* st = whisper_full_get_segment_text(ctx, i);
            if (!st) continue;
            std::string seg_str = trim(st);
            if (seg_str.empty()) continue;

            int64_t seg_t0 = whisper_full_get_segment_t0(ctx, i);
            int64_t seg_t1 = whisper_full_get_segment_t1(ctx, i);

            std::istringstream iss(seg_str);
            std::vector<std::string> words;
            std::string w;
            while (iss >> w) {
                words.push_back(w);
            }
            if (words.empty()) continue;

            int64_t dt = (words.size() > 1) ? (seg_t1 - seg_t0) / words.size() : 0;
            for (size_t k = 0; k < words.size(); ++k) {
                result.push_back({words[k], seg_t0 + static_cast<int64_t>(k * dt)});
            }
        }
    }

    return result;
}

// تطبيق التدقيق الفائق وتنسيق الإخراج: 00:00:00 الكلمة
std::string apply_ultra_audit(const std::vector<WordItem>& raw_words, bool merge_fractions, const std::string& custom_dict_str) {
    std::vector<WordItem> words = raw_words;

    // 1. دمج الأرقام والكسور (مثل 4 00:01:35 و 5 00:01:35 -> 4.5 00:01:35)
    if (merge_fractions && words.size() >= 2) {
        std::vector<WordItem> merged;
        for (size_t i = 0; i < words.size(); ++i) {
            if (i + 1 < words.size()) {
                if (!words[i].word.empty() && std::isdigit(static_cast<unsigned char>(words[i].word[0])) &&
                    !words[i+1].word.empty() && std::isdigit(static_cast<unsigned char>(words[i+1].word[0]))) {
                    int64_t diff = std::abs(words[i+1].t0 - words[i].t0);
                    if (diff <= 50) { // خلال نصف ثانية
                        merged.push_back({words[i].word + "." + words[i+1].word, words[i].t0});
                        i++;
                        continue;
                    }
                }
            }
            merged.push_back(words[i]);
        }
        words = merged;
    }

    // 2. قواعد القاموس المخصص
    std::vector<std::pair<std::string, std::string>> custom_rules;
    if (!custom_dict_str.empty()) {
        std::istringstream stream(custom_dict_str);
        std::string line;
        while (std::getline(stream, line)) {
            line = trim(line);
            if (line.empty()) continue;
            size_t eq = line.find('=');
            if (eq != std::string::npos) {
                std::string from = trim(line.substr(0, eq));
                std::string to = trim(line.substr(eq + 1));
                if (!from.empty() && !to.empty()) custom_rules.push_back({from, to});
            }
        }
    }

    // 3. تدقيق المصطلحات الهندسية وحالات الترانزستور
    std::vector<std::pair<std::string, std::string>> rules = {
        {"100 حالة", "100% حالة"},
        {"100 التشغيل", "100% التشغيل"},
        {"كاتوف", "Cut-off"},
        {"كات اوف", "Cut-off"},
        {"كات-اوف", "Cut-off"},
        {"ساتوريشن", "Saturation"},
        {"ياما", "يا إما"},
        {"كولكتور", "Collector"},
        {"اميتر", "Emitter"},
        {"إميتر", "Emitter"},
        {"بيز", "Base"},
        {"ترانزستور", "Transistor"},
        {"في اوت", "V output"},
        {"تو سيجنالز", "Two signals"}
    };

    // 4. بناء النص بالتنسيق المطلوب بالضبط: "00:00:00 الكلمة"
    std::ostringstream full_text;
    for (size_t i = 0; i < words.size(); ++i) {
        std::string w = words[i].word;
        for (const auto& r : rules) {
            w = replace_all(w, r.first, r.second);
        }
        for (const auto& r : custom_rules) {
            w = replace_all(w, r.first, r.second);
        }

        full_text << format_time(words[i].t0) << " " << w << "\n";
    }

    return full_text.str();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <audio_wav_path> [merge_fractions:0/1] [custom_dict_file]\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string audio_path = argv[2];
    bool merge_fractions = (argc >= 4) ? (std::string(argv[3]) == "1") : true;
    std::string custom_dict = "";
    if (argc >= 5) {
        std::ifstream df(argv[4]);
        if (df.is_open()) {
            std::stringstream buffer;
            buffer << df.rdbuf();
            custom_dict = buffer.str();
        }
    }

    std::vector<float> pcmf32;
    if (!read_wav(audio_path, pcmf32)) {
        std::cerr << "Error: Failed to read WAV file: " << audio_path << "\n";
        return 1;
    }

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;

    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (!ctx) {
        std::cerr << "Error: Failed to initialize whisper context from: " << model_path << "\n";
        return 1;
    }

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress   = false;
    wparams.print_realtime   = false;
    wparams.print_timestamps = false;
    wparams.translate        = false;
    wparams.language         = "ar";
    wparams.n_threads        = 4;
    wparams.token_timestamps = true;
    wparams.suppress_blank   = true;
    wparams.suppress_nst     = true;
    wparams.temperature      = 0.0f;
    wparams.temperature_inc  = 0.2f;
    wparams.no_speech_thold  = 0.6f;

    auto t_start = std::chrono::high_resolution_clock::now();

    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        std::cerr << "Error: Transcription failed\n";
        whisper_free(ctx);
        return 1;
    }

    // استخراج الكلمات المكتملة بدون تقطيع الحروف
    std::vector<WordItem> words = extract_words(ctx);

    whisper_free(ctx);

    auto t_end = std::chrono::high_resolution_clock::now();
    double exec_sec = std::chrono::duration<double>(t_end - t_start).count();

    // تطبيق التدقيق والتنسيق 00:00:00 الكلمة
    std::string audited_text = apply_ultra_audit(words, merge_fractions, custom_dict);

    // إخراج JSON
    std::cout << "---JSON_START---\n";
    std::cout << "{\n";
    std::cout << "  \"success\": true,\n";
    std::cout << "  \"execution_time\": " << exec_sec << ",\n";
    std::cout << "  \"text\": \"";

    for (char c : audited_text) {
        if (c == '"') std::cout << "\\\"";
        else if (c == '\\') std::cout << "\\\\";
        else if (c == '\n') std::cout << "\\n";
        else if (c == '\r') std::cout << "\\r";
        else if (c == '\t') std::cout << "\\t";
        else std::cout << c;
    }
    std::cout << "\"\n";
    std::cout << "}\n";
    std::cout << "---JSON_END---\n";

    return 0;
}
