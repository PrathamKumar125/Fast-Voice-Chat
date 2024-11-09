import gradio as gr
import asyncio
import sys
import os
from pydub import AudioSegment

sys.path.append('voice_chat') 
from voice_chat import respond

def process_audio(audio):
    if audio is None:
        return None
    
    if isinstance(audio, tuple):  # Gradio returns a tuple (path, sampling_rate)
        audio_path = audio[0]
    else:
        audio_path = audio
    
    # Ensure the file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Convert audio to wav if it's not already
    if not audio_path.endswith('.wav'):
        audio = AudioSegment.from_file(audio_path)
        wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
        audio.export(wav_path, format="wav")
        audio_path = wav_path
    
    return audio_path

async def voice_chat(audio):
    try:
        processed_audio = process_audio(audio)
        if processed_audio is None:
            return None
        
        response_audio = await respond(processed_audio)
        return response_audio
    except Exception as e:
        print(f"Error in voice_chat: {str(e)}")
        return None

interface = gr.Interface(
    fn=lambda audio: asyncio.run(voice_chat(audio)),
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Audio(),
    title="Voice Chat with AI Assistant",
    description="Upload an audio file or record using the microphone, and the AI will respond with voice.",
    examples=None,
    theme="default"
)

if __name__ == "__main__":
    interface.launch(debug=True)