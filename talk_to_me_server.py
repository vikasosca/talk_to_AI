
#import sounddevice as sd
#import soundfile as sf
import numpy as np
import requests
import tempfile
import time
from huggingface_hub import InferenceClient
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import os

# 1. Load the SAME model used during ingestion
global HF_TOKEN
HF_TOKEN = 'hf_Txxxxxxxxxxxxx' # Get Hugging Face Token from huggingface portal
client = InferenceClient(api_key=HF_TOKEN)
#model = SentenceTransformer("mistralai/Mixtral-8x7B-Instruct-v0.1")
# ==== CONFIG ====
ASR_MODEL = "openai/whisper-large-v3"
LLM_MODEL = "distilgpt2"  # Supported conversational model
TTS_MODEL = "suno/bark"  # Supported TTS model
SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # seconds per recording chunk
# ================
#  Using local model

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# load once at startup
LLM_MODEL = "google/flan-t5-small"
print(f"Loading local LLM model: {LLM_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
from TTS.api import TTS
import tempfile

# Load model once at startup
# Use a lightweight English model
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
tts = TTS(model_name=TTS_MODEL, progress_bar=False, gpu=False)


HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "audio/wav"}
history = []
app = FastAPI()

def record_audio(duration = CHUNK_DURATION):
    print(f"\n 🎤 you can speak now\n")
    audio =NULL # sd.rec(int(duration * SAMPLE_RATE), samplerate = SAMPLE_RATE, channel=1 , dtype="float32")
    #sd.wait()
    audio = np.squeeze(audio)
    tmp = tempfile.NamedTemporaryFile(delete = false, suffix =".wav")
    #sf.write(tmp.name,audio,SAMPLE_RATE)
    return tmp.name


def transcribe(audio_path):
    print(f"Transcribing\n")
    # ✅  Check file exists and not empty
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1024:
       print("⚠ Skipping: audio file is empty or too small.")
       return ""

    with open(audio_path,"rb") as f:
        response = requests.post(f"https://api-inference.huggingface.co/models/{ASR_MODEL}", headers=HEADERS, data=f,)
        print(f"Response: {response.status_code} {response}")
        response.raise_for_status()

        if response.ok:
           return response.json().get("text", "")
        else:
           print("Error transcribe:", response.text)
           return ""

def query_llm(history, user_text):
    print(f"\n querying LLM ")
    try:
        print('History : ', history)

        context = ("You are a helpful AI Indian female assistant with a warm sensous tone.\n")
        conversation =""
        for u, a in history[-1:]:
            conversation += f"User: {u}AI\n: {a}\n"
        conversation += f"User:{user_text}\nAI"
        payload = context + conversation
        # Double-check type before tokenizing
        if not isinstance(payload, str):
           payload= str(payload)

        print(f"payload : {payload}")

        '''
        payload =  {"inputs": context+ conversation}
        response = requests.post(f"https://api-inference.huggingface.co/models/{LLM_MODEL}",
                            headers=HEADERS, json=payload, timeout=60,)
        print("My respons is: ", response)
        response.raise_for_status()
        data = response.json
        if isinstance(data, list) and len(data) and "generated_text" in data[0]:
                    text = data[0]["generated_text"].split("AI:")[-1].strip()
                    return text
        return str(data)
        '''

        inputs = tokenizer(payload, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True,         # allows some variety
                    temperature=0.8)        # optional: controls randomness)
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return reply

    except Exception as e:
        print("Exception occured in LLM_QUERY ",e)

def synthesize_speech(text):
    print(f"\n speech synthesizer ")
    '''
    payload = {"inputs": text}
    response = requests.post(
                    f"https://api-inference.huggingface.co/models/{TTS_MODEL}", headers=HEADERS, json=payload, timeout=60,)

    response.raise_for_status()
    '''
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tts.tts_to_file(text=text, file_path=tmp.name)

    #with open(tmp.name,"wb") as f:
    #    f.write(response.content)
    return tmp.name

def play_audio(path):
    print(f"Playing Audio now \n")
    #data, sr= sf.read(path)
    ##sd.play(data,sr)
    #sd.wait()


def play_audio():
    print(f"Voice Assistant ready to converse naoe, press Ctr+C to exit anytime")
    history=[]
    try:
        while True:
            audio_path = record_audio()
            user_text = transcribe(audio_path)
            print(f"You said: {user_text} : \n")
            if not user_text.strip():
                continue
            reply = query_llm(history,user_text)
            print(f" Reply: {reply}")
            history.append(user_text,reply)
            tts_path = synthesize_speech(reply)
            play_audio(tts_path)
            time.sleep(0.5)
    except KeyboardInterrupt:
            print("\ Inturrupted while play_audio, bye for now")

@app.post("/converse")
async def converse(file:UploadFile=File(...)):
        """ Takes a .wav file as Input and returns .wav file after processing"""
        try:
        #Save uploaded file
            tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

            tmp_in.write(await file.read())
            tmp_in.close()

            user_text = transcribe(tmp_in.name)
            print(f"you said - {user_text}")
            if not user_text.strip():
                print(f"Nothing was said")
                return JSONResponse({"error":"Empty Transcriprion"},status_code=400)

            reply = query_llm(history , user_text)
            print(f" AI Reply : {reply}")
            history.append((user_text,reply))
            tts_path =  synthesize_speech(reply)
            print(f"Returning transcribed text reply \n")
            return FileResponse(tts_path, media_type="audio/wav", filename="reply.wav")
        except Exception as e:
            print(f"Error occured during converse:{e}")
            return JSONResponse({"error":str(e)},status_code=500)


if __name__ == "__main__":
        main()
