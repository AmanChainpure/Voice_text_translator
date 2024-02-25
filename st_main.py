import streamlit as st
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import json
import sounddevice as sd
import numpy as np
import wave
from transformers import pipeline


# Function to record audio and save it as a .wav file
def record_and_save(filename):
    st.text("Recording... Press the 'Stop' button to finish.")
    try:
        # Record 5 seconds of audio
        audio_data = sd.rec(int(FRAME_RATE * 5), samplerate=FRAME_RATE, channels=CHANNELS, dtype=DTYPE)
        sd.wait()

        # Save the recorded audio as a .wav file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio_data.dtype.itemsize)
            wf.setframerate(FRAME_RATE)
            wf.writeframes(audio_data.tobytes())

        st.text(f"Audio saved as {filename}")
    except KeyboardInterrupt:
        st.text("\nRecording stopped.")


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


# Streamlit UI
st.title("Voice-to-Text and Text Translation")

# Constants
FRAME_RATE = 16000
CHANNELS = 1
DTYPE = np.int16
FILENAME = "captured_audio.wav"

# Button to start recording
if st.button("Start Recording"):
    record_and_save(FILENAME)

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

# Display results
st.subheader("Original Text:")
st.text(text)

st.subheader("Translated Text:")
st.text(translated_text)