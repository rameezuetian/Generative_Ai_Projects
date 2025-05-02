from llm.llama2_llm import llm
from prompts.template import prompt_to_LLAMA2
from transcriber.whisper_asr import transcribe_audio
import gradio as gr

def transcript_audio(audio_file):
    transcript_txt = transcribe_audio(audio_file)
    result = prompt_to_LLAMA2.run(transcript_txt)
    return result

iface = gr.Interface(fn=transcript_audio,
                     inputs=gr.Audio(sources="upload", type="filepath"),
                     outputs=gr.Textbox(),
                     title="Audio Summarization App",
                     description="Upload an audio file for transcription and summarization.")
iface.launch(server_name="0.0.0.0", server_port=7860)
