import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment

from config import Config
from main import SpeechToSpeechPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="S2S Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

config = Config()
pipeline = SpeechToSpeechPipeline(config)

@app.post("/s2s")
async def s2s(file: UploadFile = File(...)):
    try:
        # Save uploaded
        in_ext = os.path.splitext(file.filename or "")[1].lower() or ".webm"
        temp_input_path = os.path.join(INPUT_DIR, f"upload_{uuid.uuid4().hex}{in_ext}")
        with open(temp_input_path, "wb") as f:
            f.write(await file.read())

        # Convert to wav
        input_wav_path = os.path.join(INPUT_DIR, f"in_{uuid.uuid4().hex}.wav")
        AudioSegment.from_file(temp_input_path).export(input_wav_path, format="wav")
        try: os.remove(temp_input_path)
        except Exception: pass

        # Output file path
        out_name = f"out_{uuid.uuid4().hex}.wav"
        output_wav_path = os.path.join(OUTPUT_DIR, out_name)

        # Run pipeline
        result = pipeline.run(input_wav_path, output_wav_path)
        if not isinstance(result, dict):
            return {"ok": False, "error": "Unexpected pipeline result.", "output_url": None, "text": "", "input_text": ""}

        audio_path = result.get("audio_path") or ""
        text = result.get("text") or ""
        input_text = result.get("input_text") or ""

        output_url = None
        if audio_path and os.path.exists(audio_path):
            output_url = f"/outputs/{os.path.basename(audio_path)}"

        # Partial success supported: return text even if audio is missing
        ok = bool(text or output_url)
        return {"ok": ok, "output_url": output_url, "text": text, "input_text": input_text, "error": "" if ok else "Pipeline produced no output."}

    except Exception as e:
        logging.exception("S2S processing failed")
        return {"ok": False, "error": str(e), "output_url": None, "text": "", "input_text": ""}