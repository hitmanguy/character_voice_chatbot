from TTS.api import TTS
from config import Config
import logging
import os
from gtts import gTTS
from pydub import AudioSegment
import tempfile

class TextToSpeechService:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        if not os.path.exists(config.tts_speaker_wav):
            raise FileNotFoundError(f"Speaker WAV file not found at {config.tts_speaker_wav}.")
        logging.info("Loading TTS model (Coqui XTTSv2)...")
        self.model = TTS(config.tts_model_id).to(self.device)
        logging.info("✅ TTS model loaded.")

    def synthesize_with_gtts(self, text: str, language: str, output_path: str) -> str:
        try:
            gtts_lang = self.config.LANG_CODE_MAPPING.get(language, language or "en")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp_mp3 = tmp.name
            gTTS(text, lang=gtts_lang).save(tmp_mp3)
            AudioSegment.from_mp3(tmp_mp3).export(output_path, format="wav")
            os.remove(tmp_mp3)
            logging.info(f"✅ Fallback gTTS synthesized to {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"gTTS fallback failed: {e}")
            return ""

    def synthesize(self, text: str, language: str, output_path: str = "output.wav") -> str:
        """
        Try XTTS first; for languages like 'ja' (requires MeCab on Windows) or on error, fallback to gTTS.
        """
        logging.info(f"Synthesizing speech in language '{language}' for: '{text}'")
        tts_lang_code = self.config.LANG_CODE_MAPPING.get(language, "en")

        # Force fallback for Japanese to avoid MeCab issues on Windows
        if tts_lang_code in {"ja"}:
            logging.warning("Using gTTS fallback for 'ja' (MeCab not available).")
            return self.synthesize_with_gtts(text, language, output_path)

        try:
            self.model.tts_to_file(
                text=text,
                speaker_wav=self.config.tts_speaker_wav,
                language=tts_lang_code,
                file_path=output_path,
            )
            logging.info(f"✅ Speech synthesized successfully to {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Failed to synthesize with XTTS: {e}. Falling back to gTTS.")
            return self.synthesize_with_gtts(text, language, output_path)