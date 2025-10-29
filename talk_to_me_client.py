import requests, sounddevice as sd, soundfile as sf, tempfile
server_URL = "http://161.118.169.134:8000/converse"
sample_rate = 16000

def record_audio(duration=4):
    print("You can speak for 4 seconds")
    audio = sd.rec(int(duration*sample_rate),samplerate=sample_rate,channels=1,dtype="float32")
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".wav")
    sf.write(tmp.name,audio,sample_rate)
    return tmp.name

def play_audio(path):
    print('Audio playing now: ', path)
    try:
        data, sr = sf.read(path)
        sd.play(data,sr)
        sd.wait()
        print('Audio played now')
    except Exception as e:
        print(f"Error playing audio: {e}")
        
def main():
    try:
        while True:
            audio_path = record_audio()
            print('Audio Recorded ')
            with open(audio_path,"rb") as f:
                response = requests.post(server_URL,files={"file":f})
            if response.status_code != 200:
                #print("Server error: ", response.text, " - ", response.status_code)
                print("Server error: ", response.status_code)
                break #continue
            # Save audio response to temp file
            tmp_out = tempfile.NamedTemporaryFile(delete=False,suffix=".wav")
            tmp_out.write(response.content)  # ← THIS LINE IS CRITICAL
            tmp_out.close()
            print('Playing Audio  from: ', tmp_out.name)
            play_audio(tmp_out.name)
            
            print('End of Main')
    except KeyboardInterrupt:
            print("\n Inturrupted, bye for now")    

if __name__ == "__main__":
	main()