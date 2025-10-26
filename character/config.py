import torch
from dataclasses import dataclass, field

# ============================================================
# 📜 CENTRAL CONFIGURATION
# ============================================================
@dataclass
class Config:
    # --- Device Configuration ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    gtts_voice_gender: str = "male"     # "male" | "female"
    gtts_maleify_enabled: bool = True   # apply pitch shift if using gTTS and gender is "male"
    gtts_maleify_semitones: int = -4    # -4 to -6 sounds more “male”; negative lowers pitch

    # --- Character Model ---
    character_model_id: str = "Hitmanguy/tinyllama-ironman_vfinal"

    # --- Speech-to-Text Model (Whisper) ---
    stt_model_name: str = "base"

    # --- Text-to-Speech Model (Coqui XTTSv2) ---
    tts_model_id: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_speaker_wav: str = "tony.wav"

    # --- Supporting Analysis Models ---
    emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base"

    # --- Translation Model (NLLB) ---
    translation_model_id: str = "facebook/nllb-200-distilled-600M"

    # --- Persona Definition ---
    persona_description: str = (
        "You are the superhero Iron Man (Tony Stark). You are a genius, billionaire, playboy, philanthropist. "
        "Respond with your characteristic wit, intelligence, and a touch of arrogance. "
        "Be verbose and keep short responses. Don't elaborate too much."
    )

    # --- Language Code Mappings ---
    ISO_TO_NLLB_MAPPING: dict = field(default_factory=lambda: {
        "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
        "it": "ita_Latn", "pt": "por_Latn", "pl": "pol_Latn", "tr": "tur_Latn",
        "ru": "rus_Cyrl", "nl": "nld_Latn", "cs": "ces_Latn", "ar": "arb_Arab",
        "zh-cn": "zho_Hans", "ja": "jpn_Jpan", "hu": "hun_Latn", "ko": "kor_Hang",
        "hi": "hin_Deva", "kn": "kan_Knda",  # Kannada
    })
    LANG_CODE_MAPPING: dict = field(default_factory=lambda: {
        "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it", "pt": "pt",
        "pl": "pl", "tr": "tr", "ru": "ru", "nl": "nl", "cs": "cs", "ar": "ar",
        "zh-cn": "zh-cn", "ja": "ja", "hu": "hu", "ko": "ko", "hi": "hi",
        "kn": "kn",  # Kannada
    })
    # Used by Whisper to keep output in the native script of the detected language
    WHISPER_INITIAL_PROMPTS: dict = field(default_factory=lambda: {
        "hi": "नमस्ते, यह एक परीक्षण है।",
        "es": "Hola, esto es una prueba.",
        "fr": "Bonjour, ceci est un test.",
        "ja": "こんにちは、これはテストです。",
        "ru": "Здравствуйте, это тест.",
        "ar": "مرحبا هذا اختبار.",
        "zh": "你好，这是一个测试。",
        "kn": "ನಮಸ್ಕಾರ, ಇದು ಒಂದು ಪರೀಕ್ಷೆ.",  # Kannada
    })