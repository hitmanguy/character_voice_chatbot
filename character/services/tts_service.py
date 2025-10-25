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

    def _maleify_wav_inplace(self, wav_path: str, semitones: int) -> None:
        """
        Lower pitch by N semitones. Uses librosa if available (formant-preserving),
        otherwise a simple pydub frame-rate trick (affects tempo slightly).
        """
        try:
            import librosa, soundfile as sf
            y, sr = librosa.load(wav_path, sr=None, mono=True)
            y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
            sf.write(wav_path, y_shift, sr)
            logging.info(f"Applied librosa pitch shift ({semitones} st) to {wav_path}")
        except Exception as e:
            logging.warning(f"librosa maleify failed or not installed: {e}. Falling back to pydub rate change.")
            try:
                audio = AudioSegment.from_wav(wav_path)
                # Change playback rate to shift pitch, then re-set standard rate
                rate_factor = 2 ** (semitones / 12.0)
                new_rate = max(8000, int(audio.frame_rate * rate_factor))
                shifted = audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate}).set_frame_rate(audio.frame_rate)
                shifted.export(wav_path, format="wav")
                logging.info(f"Applied pydub frame-rate pitch shift ({semitones} st) to {wav_path}")
            except Exception as e2:
                logging.error(f"Pydub maleify fallback failed: {e2}")

    def synthesize_with_gtts(self, text: str, language: str, output_path: str) -> str:
        try:
            gtts_lang = self.config.LANG_CODE_MAPPING.get(language, language or "en")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp_mp3 = tmp.name
            gTTS(text, lang=gtts_lang).save(tmp_mp3)
            AudioSegment.from_mp3(tmp_mp3).export(output_path, format="wav")
            os.remove(tmp_mp3)

            # Optional “male” shaping
            if self.config.gtts_maleify_enabled and (self.config.gtts_voice_gender.lower() == "male"):
                self._maleify_wav_inplace(output_path, self.config.gtts_maleify_semitones)

            logging.info(f"✅ Fallback gTTS synthesized to {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"gTTS fallback failed: {e}")
            return ""

    def synthesize(self, text: str, language: str, output_path: str = "output.wav") -> str:
        logging.info(f"Synthesizing speech in language '{language}' for: '{text}'")
        tts_lang_code = self.config.LANG_CODE_MAPPING.get(language, "en")

        # Force fallback for languages that need platform-specific phonemizers or where XTTS is unreliable on Windows
        if tts_lang_code in {"ja", "kn", "bn"}:  # add others as needed
            logging.warning("Using gTTS fallback for language '%s'.", tts_lang_code)
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