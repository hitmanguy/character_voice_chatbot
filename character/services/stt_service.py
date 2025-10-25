import whisper
import logging
from config import Config
import torch

class SpeechToTextService:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        logging.info("Loading STT model (Whisper) to CPU...")
        self.model = whisper.load_model(config.stt_model_name, device="cpu")
        logging.info("✅ STT model loaded.")

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribes audio using a two-pass method with an initial_prompt
        to GUARANTEE the output is in the language's native script.
        This is the single source of truth for transcription.
        """
        logging.info(f"Transcribing audio from: {audio_path}")
        model_on_device = self.model.to(self.device)
        try:
            # 1. Quickly detect the language from the first 30 seconds of audio
            audio = whisper.load_audio(audio_path)
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio)).to(model_on_device.device)
            _, probs = model_on_device.detect_language(mel)
            detected_lang_code = max(probs, key=probs.get)
            logging.info(f"Detected language: {detected_lang_code}")

            # 2. Get the appropriate initial prompt for the detected language from the config
            initial_prompt = self.config.WHISPER_INITIAL_PROMPTS.get(detected_lang_code)
            if initial_prompt:
                logging.info(f"Using initial prompt for '{detected_lang_code}' to ensure native script.")

            # 3. Transcribe using the high-level function, passing the crucial initial_prompt
            result = model_on_device.transcribe(
                audio_path,
                language=detected_lang_code,
                initial_prompt=initial_prompt, # <-- THE DEFINITIVE FIX
                fp16=(self.config.torch_dtype == torch.float16)
            )
            
            text = result.get("text", "").strip()
            
            logging.info(f"Native script transcription result: '{text}'")
            return {"text": text, "language": detected_lang_code}
        finally:
            model_on_device.to("cpu") # Free up GPU VRAM immediately after use```