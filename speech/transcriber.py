import whisper
import os

# Load Whisper model once
model = whisper.load_model("base")

def transcribe_audio(audio_path):
    if not os.path.exists(audio_path):
        return "Audio file not found"
    result = model.transcribe(audio_path)
    return result["text"]
