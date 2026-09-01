from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue
import json

model = Model("vosk-model-small-en-us-0.15")
rec = KaldiRecognizer(model, 16000)
q = queue.Queue()

def callback(indata, frames, time, status):
    q.put(bytes(indata))

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                        channels=1, callback=callback):
    print("🎤 Speak now...")
    collected = b''
    while len(collected) < 5 * 16000 * 2:
        data = q.get()
        collected += data
        if rec.AcceptWaveform(data):
            break
    print(json.loads(rec.FinalResult()))
    print("🎤 Voice input processed.")
    
