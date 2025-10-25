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

    # --- Character Model ---
    character_model_id: str = "Hitmanguy/pythia-2.8b-ironman_v2"

    # --- Speech-to-Text Model (Whisper) ---
    stt_model_name: str = "base"

    # --- Text-to-Speech Model (Coqui XTTSv2) ---
    tts_model_id: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_speaker_wav: str = "weapon.wav"
    
    # --- Translation Model (NLLB) ---
    translation_model_id: str = "facebook/nllb-200-distilled-600M"

    # --- Persona Definition ---
    persona_description: str = (
        "You are the superhero Iron Man (Tony Stark). You are a genius, billionaire, playboy, philanthropist. "
        "Your responses must always be in character, reflecting your characteristic wit, intelligence, and a touch of arrogance. "
        "You should give elaborate, verbose, and detailed responses."
    )
    
    # --- Language Code Mappings ---
    ISO_TO_NLLB_MAPPING: dict = field(default_factory=lambda: {
        "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
        "it": "ita_Latn", "pt": "por_Latn", "pl": "pol_Latn", "tr": "tur_Latn",
        "ru": "rus_Cyrl", "nl": "nld_Latn", "cs": "ces_Latn", "ar": "arb_Arab",
        "zh-cn": "zho_Hans", "ja": "jpn_Jpan", "hu": "hun_Latn", "ko": "kor_Hang",
        "hi": "hin_Deva",
        "kn": "kan_Knda",  # Kannada
    })
    LANG_CODE_MAPPING: dict = field(default_factory=lambda: {
        "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it", "pt": "pt",
        "pl": "pl", "tr": "tr", "ru": "ru", "nl": "nl", "cs": "cs", "ar": "ar",
        "zh-cn": "zh-cn", "ja": "ja", "hu": "hu", "ko": "ko", "hi": "hi",
        "kn": "kn",  # Kannada
    })
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