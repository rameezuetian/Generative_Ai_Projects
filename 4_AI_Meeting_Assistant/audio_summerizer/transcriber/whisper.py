from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    chunk_length_s=30
)

def transcribe_audio(audio_file):
    return pipe(audio_file, batch_size=8)["text"]
