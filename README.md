# 🎙️ Talk-To-AI (Voice Conversational AI Assistant)

This project enables a **two-way voice conversation** between a user and an AI assistant using:
- Speech-to-Text (ASR)
- Text-based LLM response
- Text-to-Speech (TTS)
- Real-time voice playback

It consists of two components:
1. **Server (`talk_to_me_server.py`)** — handles speech recognition, LLM text generation, and voice synthesis.
2. **Client (`talk_to_me_client.py`)** — records your voice, sends it to the server, and plays back the AI’s voice response.

------

## 🧠 How It Works
1. The client records your audio (using `sounddevice`). 
2. It sends the recorded WAV file to the FastAPI server endpoint (`/converse`).
3. The server:
   - Transcribes audio using Whisper (`openai/whisper-large-v3`) via Hugging Face API.
   - Generates an AI reply using a local model (`google/flan-t5-small`).
   - Synthesizes speech from the reply using Coqui TTS (`tts_models/en/ljspeech/tacotron2-DDC`).
4. The client receives text of conversation and then plays back the AI’s voice response.

## Key observations

1. The client currently runs on the laptop as remote server doesnot come with speaker or a mic
2. Hugging face models are downloaded and run locally to avoid token limit and endpoints becoming unreachable
3. The model has latency as it is runing on a very small server. A large CPU would speed up the response.
   ---
   
## 🧩 Tech Stack
- **FastAPI** — API framework for handling voice requests.
- **Transformers** — For local text generation model.
- **Coqui TTS** — Text-to-Speech synthesis.
- **Whisper (Hugging Face)** — Speech-to-Text model.
- **SoundDevice / SoundFile** — Audio recording and playback.

---

## ⚙️ Setup Instructions
### 1. Clone the repo
```bash
git clone https://github.com/vikasosca/talk_to_AI.git
cd talk_to_AI

