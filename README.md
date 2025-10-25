# Iron Man Character Voice Chatbot

Talk to Iron Man (Tony Stark) with your voice. The app records your speech in the browser, sends it to a Python backend that:

- Transcribes speech (Whisper)
- Understands intent and generates an in-character Iron Man reply (LLM)
- Translates across languages (NLLB)
- Speaks the response back (Coqui XTTSv2 with safe gTTS fallback)
- Returns both the reply text and the synthesized audio

Frontend shows your transcribed input, Iron Man’s reply text, and plays the response audio.

---

## Features

- Voice in, voice out (speech-to-speech)
- Iron Man persona (witty, confident, technical)
- Multilingual:
  - STT via Whisper
  - Translation via NLLB (ISO/NLLB mapping in config)
  - TTS via Coqui XTTSv2 or gTTS fallback (for environments where phonemizers are unavailable)
- Resilient pipeline: returns best-effort text even if TTS fails
- Local-first: run entirely on your machine (gTTS fallback requires internet)

---

## Architecture

- Frontend (React + TypeScript)
  - Records audio with MediaRecorder
  - Calls POST /s2s with the recorded blob
  - Renders your transcription and Iron Man’s reply, and plays the audio
- Backend (FastAPI + Python)
  - Endpoint: POST /s2s (multipart/form-data with file)
  - Saves input audio, converts to WAV (pydub/ffmpeg)
  - Runs SpeechToSpeechPipeline.run(input_wav, output_wav)
  - Serves generated audio under /outputs

Key files:

- App.tsx – main React app
- hooks/useSpeechPipeline.ts – API client
- character/server.py – FastAPI server
- character/main.py – Orchestrator pipeline
- character/config.py – Central configuration (persona, models, language maps)
- character/services/\* – STT, NLU, Character, TTS services

---

## Requirements

- OS: Windows 10/11 (tested), macOS/Linux should also work with tweaks
- Python: 3.10+ recommended
- Node.js: 18+ (LTS recommended) for the frontend
- ffmpeg on PATH (required by pydub)
- Optional GPU: CUDA-enabled PyTorch for faster models
- Internet for first-time model downloads (and for gTTS fallback)

Models (downloaded automatically on first run):

- Whisper (base)
- NLLB 200 distilled 600M
- Character LLM: Hitmanguy/pythia-2.8b-ironman_v2
- Coqui XTTSv2: tts_models/multilingual/multi-dataset/xtts_v2

Audio assets:

- A speaker reference WAV at character/config.py -> tts_speaker_wav (default: weapon.wav). Place your WAV file at the repo root or adjust the path.

---

## Installation (Windows)

1. Clone

```powershell
git clone https://github.com/your-org/character_voice_chatbot.git
cd character_voice_chatbot
```

2. Python environment

- Using Conda (recommended):

```powershell
conda create -n ai-env python=3.10 -y
conda activate ai-env
```

3. Python dependencies

```powershell
pip install --upgrade pip
pip install fastapi uvicorn pydub gTTS
pip install transformers sentencepiece torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install TTS  # Coqui TTS
pip install bitsandbytes
pip install librosa
# Optional (Japanese XTTS phonemizer). We use gTTS fallback by default, so this is optional:
# pip install "fugashi[unidic-lite]"
```

4. ffmpeg

- Install via winget or choco, or download binary:

```powershell
winget install --id Gyan.FFmpeg -e
# or
choco install ffmpeg -y
```

Ensure ffmpeg is on PATH. Verify:

```powershell
ffmpeg -version
```

5. Frontend dependencies

```powershell
# From the repo root where package.json resides
npm install
```

6. Speaker WAV

- Ensure the file configured in character/config.py exists (default: weapon.wav in repo root).
- You can change tts_speaker_wav in config.py to your own reference WAV.

---

## Running

Open two terminals (both in the repo root).

A) Backend API (FastAPI)

```powershell
conda activate ai-env
<<<<<<< HEAD
cd character
python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000
=======
python -m uvicorn character.server:app --reload --host 127.0.0.1 --port 8000
>>>>>>> 62acd1e3edf719fd1f9d5ad9ad084b37416ce203
```

You should see “Application startup complete.”  
Generated audio files are served under: http://127.0.0.1:8000/outputs/<filename>.wav

Quick test without the frontend:

```powershell
# If you don’t have a test input, create a 1s tone:
python - << 'PY'
import wave, struct, math, os
path = r'.\character\test_input.wav'
sr=16000; f=440.0; dur=1.0
wf=wave.open(path,'w'); wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
for i in range(int(sr*dur)):
    v=int(32767*0.3*math.sin(2*math.pi*f*i/sr))
    wf.writeframes(struct.pack('<h', v))
wf.close(); print('Wrote', os.path.abspath(path))
PY

# PowerShell's curl is Invoke-WebRequest; use curl.exe or Invoke-RestMethod:
curl.exe -F "file=@.\character\test_input.wav;type=audio/wav" http://127.0.0.1:8000/s2s

# Or PowerShell native:
$resp = Invoke-RestMethod -Uri http://127.0.0.1:8000/s2s -Method Post -Form @{ file = Get-Item .\character\test_input.wav }
$resp
if ($resp.output_url) { start "" ("http://127.0.0.1:8000" + $resp.output_url) }
```

B) Frontend (React)

```powershell
npm run dev
```

Open the printed local URL (often http://localhost:5173).  
Tap the mic to record. The app will:

- Show your transcribed text (input_text)
- Show Iron Man’s reply (text)
- Play the synthesized audio (output_url)

---

## API

Endpoint:

- POST /s2s (multipart/form-data)
  - form field: file (the audio to process; any browser-recorded format is fine, server converts to WAV)
    Response (JSON):

```json
{
  "ok": true,
  "output_url": "/outputs/out_xxx.wav", // may be null if TTS failed
  "text": "Iron Man's reply text",
  "input_text": "Your transcribed input",
  "error": "" // present if ok=false
}
```

Notes:

- Static files under /outputs serve generated audio.
- On partial success (e.g., TTS failed), server returns ok=true with text and output_url=null so the app can still display text.

---

## Configuration

File: character/config.py

- Persona:
  - persona_description models Iron Man’s style.
- Models:
  - character_model_id: Hitmanguy/pythia-2.8b-ironman_v2
  - stt_model_name: Whisper “base”
  - translation_model_id: NLLB distilled 600M
  - tts_model_id: Coqui XTTSv2
  - tts_speaker_wav: speaker reference WAV (default: weapon.wav)
- Language maps:
  - ISO_TO_NLLB_MAPPING
  - LANG_CODE_MAPPING
  - WHISPER_INITIAL_PROMPTS
- Kannada added as "kn"; Japanese is supported via fallback (gTTS) by default on Windows.

TTS fallback:

- character/services/tts_service.py forces gTTS fallback for Japanese (and optionally Kannada) to avoid MeCab issues on Windows.
- gTTS requires internet access.

---

## Running the CLI demo (optional)

You can also run the pipeline directly:

```powershell
cd character
python main.py
```

- Will create a test Hindi input, run the pipeline, print the final text, and save the audio.
- If imports fail in script vs package mode, ensure you run from the character folder.

---

## Project Structure

```
character_voice_chatbot/
├─ App.tsx
├─ hooks/
│  ├─ useSpeechPipeline.ts
│  └─ useAutoScroll.ts
├─ components/
│  ├─ Header.tsx
│  ├─ MessageList.tsx
│  └─ VoiceInput.tsx
├─ character/
│  ├─ server.py               # FastAPI server
│  ├─ main.py                 # Pipeline orchestrator
│  ├─ config.py               # Central config (persona, models, languages)
│  ├─ services/
│  │  ├─ stt_service.py
│  │  ├─ nlu_service.py
│  │  ├─ character_service.py
│  │  └─ tts_service.py
│  ├─ inputs/                 # Stored inputs (created at runtime)
│  └─ outputs/                # Generated outputs (served statically)
├─ weapon.wav                 # Speaker reference (example)
└─ README.md
```

---

## Troubleshooting

- ffmpeg not found:
  - Install ffmpeg and ensure it’s on PATH. Verify with `ffmpeg -version`.
- PowerShell curl errors:
  - Use `curl.exe` or `Invoke-RestMethod` with `-Form`. `curl` alias is `Invoke-WebRequest` which doesn’t support `-F`.
- 500 errors from pipeline:
  - The pipeline is resilient and should return text even if TTS fails. Check server logs for “Character generation failed” or “TTS synthesis failed”.
- Japanese TTS errors (MeCab/fugashi):
  - We default to gTTS fallback on Windows. To enable full XTTS Japanese:
    - Install MeCab and fugashi, set MECABRC. Otherwise keep fallback.
- Large model downloads slow:
  - First run downloads can be several GB. Pre-download with a stable connection.
- CUDA/VRAM errors:
  - Switch to CPU (set device in config.py), or use smaller models. Ensure the PyTorch CUDA wheel matches your GPU/CUDA version.

---

## Security and Production Notes

- In development, CORS is set to allow all origins. Restrict `allow_origins` before deployment.
- Consider persisting inputs/outputs outside the repo and cleaning old files.
- Set up logging to files and structured logs in production.

---

## License

This project integrates third-party models and libraries (Hugging Face, Coqui TTS, gTTS, etc.) which have their own licenses. Ensure compliance when redistributing or deploying.

---
