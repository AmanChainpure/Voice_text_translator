from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import json
import sounddevice as sd
import numpy as np
import wave
from transformers import pipeline

FRAME_RATE = 16000
CHANNELS = 1
DTYPE = np.int16
FILENAME = "captured_audio.wav"


# Function to record audio and save it as a .wav file
def record_and_save():
    print("Recording... Press Ctrl+C to stop.")
    try:
        # Record 5 seconds of audio
        audio_data = sd.rec(int(FRAME_RATE * 5), samplerate=FRAME_RATE, channels=CHANNELS, dtype=DTYPE)
        sd.wait()

        # Save the recorded audio as a .wav file
        with wave.open(FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio_data.dtype.itemsize)
            wf.setframerate(FRAME_RATE)
            wf.writeframes(audio_data.tobytes())

        print(f"Audio saved as {FILENAME}")
    except KeyboardInterrupt:
        print("\nRecording stopped.")


# Function to translate English text to Romanian
def translate_text(input_text):
    t5_small_pipeline = pipeline(
        task="text2text-generation",
        model="t5-small",
        max_length=50,
        model_kwargs={"cache_dir": 't5_small'},
    )
    translated_text = t5_small_pipeline(f"translate English to Romanian: {input_text}")
    return translated_text[0]['generated_text']


# Call the function to record and save audio
record_and_save()

# Load the saved .wav file
with wave.open(FILENAME, 'rb') as wf:
    audio_data = wf.readframes(wf.getnframes())

# Perform voice-to-text conversion
model = Model(model_name="vosk-model-en-us-0.22")
rec = KaldiRecognizer(model, FRAME_RATE)
rec.SetWords(True)

rec.AcceptWaveform(audio_data)
result = rec.Result()

text = json.loads(result)["text"]

# Translate the text to Romanian
translated_text = translate_text(text)

print(f"Original Text: {text}")
print(f"Translated Text: {translated_text}")