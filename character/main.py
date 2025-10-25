import torch
import logging
import os
import time
from gtts import gTTS
from pydub import AudioSegment

# ============================================================
# ⚙️ PROFESSIONAL SETUP
# ============================================================
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
    logging.info("PyTorch serialization patched for Coqui TTS compatibility.")
except Exception as e:
    logging.error(f"Failed to apply PyTorch compatibility patch: {e}")

from config import Config
from services.stt_service import SpeechToTextService
from services.nlu_service import NaturalLanguageUnderstandingService
from services.character_service import CharacterResponseService
from services.tts_service import TextToSpeechService

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def play_audio(file_path):
    try:
        from IPython.display import Audio, display
        print(f"\n▶️ Playing audio from {file_path}...")
        display(Audio(file_path, autoplay=True))
    except ImportError:
        logging.warning("IPython not found. Cannot automatically play audio.")
        print(f"--> Please play the output audio file manually: {file_path}")

# ============================================================
# 🚀 THE SPEECH-TO-SPEECH PIPELINE ORCHESTRATOR
# ============================================================
class SpeechToSpeechPipeline:
    def __init__(self, config: Config):
        self.config = config
        logging.info("Initializing all services for the S2S pipeline...")
        self.stt_service = SpeechToTextService(config)
        self.nlu_service = NaturalLanguageUnderstandingService(config)
        self.character_service = CharacterResponseService(config)
        self.tts_service = TextToSpeechService(config)
        logging.info("✅ Speech-to-speech pipeline is ready.")

    def run(self, audio_input_path: str, output_audio_path: str = "final_response.wav") -> dict: # <-- MODIFIED: Return type is now a dictionary
        """
        Executes the full, end-to-end speech-to-speech pipeline.

        Returns:
            dict: A dictionary containing 'audio_path' and 'text', or empty strings on failure.
        """
        try:
            start_time = time.time()
            
            # 1. STT
            stt_result = self.stt_service.transcribe(audio_input_path)
            native_transcription, source_lang = stt_result["text"], stt_result["language"]
            if not native_transcription: raise ValueError("STT failed.")

            # 2. Translation to English
            if source_lang != "en":
                clean_intent_en = self.nlu_service.translate(native_transcription, src_lang=source_lang, tgt_lang="en")
            else:
                clean_intent_en = native_transcription
            if not clean_intent_en: raise ValueError("Translation to English failed.")

            # 3. Character Generation
            character_response_en = self.character_service.generate_response(clean_intent_en)
            if not character_response_en: raise ValueError("Character generation failed.")
            
            # 4. Response Cleanup
            final_text_en = self.nlu_service.cleanup_character_response(character_response_en)
            if not final_text_en: raise ValueError("Character response cleanup failed.")
            
            # 5. Translation back to Native Language
            if source_lang != "en":
                final_text_native = self.nlu_service.translate(final_text_en, src_lang="en", tgt_lang=source_lang)
            else:
                final_text_native = final_text_en
            if not final_text_native: raise ValueError("Translation back to native language failed.")
            
            # 6. TTS
            final_output_path = self.tts_service.synthesize(final_text_native, source_lang, output_audio_path)
            if not final_output_path: raise ValueError("TTS synthesis failed.")

            end_time = time.time()
            logging.info(f"🚀 Full pipeline executed in {end_time - start_time:.2f} seconds.")
            
            # <-- MODIFIED: Return a dictionary with both outputs
            return {"audio_path": final_output_path, "text": final_text_native, "input_text": native_transcription}

        except Exception as e:
            logging.error(f"An error occurred in the pipeline: {e}")
            # <-- MODIFIED: Return a dictionary on failure as well
            return {"audio_path": "", "text": ""}

# ============================================================
# 🚀 EXAMPLE USAGE & DEMONSTRATION
# ============================================================
if __name__ == '__main__':
    setup_logging()

    print("\n--- Creating a Hindi test audio input file ('test_input_hindi.wav')... ---")
    try:
        kn_text = "ನಮಸ್ಕಾರ ಟೋನಿ ಸ್ಟಾರ್ಕ್, ಇಂದು ನಿಮ್ಮ ಯೋಜನೆ ಏನು?"
        tts = gTTS(kn_text, lang='kn')
        tts.save("test_input_hindi.mp3")
        AudioSegment.from_mp3("test_input_hindi.mp3").export("test_input_hindi.wav", format="wav")
        os.remove("test_input_hindi.mp3")
        print("✅ Hindi test audio created successfully.")
    except Exception as e:
        logging.error(f"Failed to create test audio. Error: {e}")
        exit()

    try:
        config = Config()
        pipeline = SpeechToSpeechPipeline(config)
        
        # <-- MODIFIED: The result is now a dictionary
        result = pipeline.run("test_input_hindi.wav", "final_response_hindi.wav")
        
        # <-- MODIFIED: Check for the audio_path in the result dictionary
        if result and result["audio_path"]:
            # <-- MODIFIED: Print the final text to the console
            print("\n" + "="*50)
            print(f"💬 Final Generated Text: {result['text']}")
            print("="*50)
            
            play_audio(result["audio_path"])
        else:
            logging.error("Pipeline execution failed. No output was generated.")

    except FileNotFoundError as e:
        logging.error(f"A required model or file was not found: {e}")
    except Exception as e:
        logging.error(f"A critical error occurred: {e}")