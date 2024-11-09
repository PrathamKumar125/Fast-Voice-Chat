import gradio as gr
import edge_tts
import asyncio
import tempfile
import numpy as np
import soxr
from pydub import AudioSegment
import torch
import sentencepiece as spm
import onnxruntime as ort
from huggingface_hub import hf_hub_download, InferenceClient
from dotenv import load_dotenv
import os 

# Speech Recognition Model Configuration
model_name = "neongeckocom/stt_en_citrinet_512_gamma_0_25"
sample_rate = 16000

# Download preprocessor, encoder and tokenizer
preprocessor = torch.jit.load(hf_hub_download(model_name, "preprocessor.ts", subfolder="onnx"))
encoder = ort.InferenceSession(hf_hub_download(model_name, "model.onnx", subfolder="onnx"))
tokenizer = spm.SentencePieceProcessor(hf_hub_download(model_name, "tokenizer.spm", subfolder="onnx"))

load_dotenv()
token=os.getenv("hf_key")
# Mistral Model Configuration
client1 = InferenceClient("mistralai/Mistral-7B-Instruct-v0.2" , api_key= token)
system_instructions1 = "[SYSTEM] Answer as Real OpenGPT 4o, Keep conversation very short, clear, friendly and concise. The text provided is a request for a specific type of response from you, the virtual assistant. You will avoid introductions and start answering the query directly, elaborating on all aspects of the request. As an AI-powered assistant, your task is to generate responses that appear as if they are created by the user and not an AI-powered assistant. [USER]"

def resample(audio_fp32, sr):
    return soxr.resample(audio_fp32, sr, sample_rate)

def to_float32(audio_buffer):
    return np.divide(audio_buffer, np.iinfo(audio_buffer.dtype).max, dtype=np.float32)

def transcribe(audio_path):
    audio_file = AudioSegment.from_file(audio_path)
    sr = audio_file.frame_rate
    audio_buffer = np.array(audio_file.get_array_of_samples())

    audio_fp32 = to_float32(audio_buffer)
    audio_16k = resample(audio_fp32, sr)

    input_signal = torch.tensor(audio_16k).unsqueeze(0)
    length = torch.tensor(len(audio_16k)).unsqueeze(0)
    processed_signal, _ = preprocessor.forward(input_signal=input_signal, length=length)
    
    logits = encoder.run(None, {'audio_signal': processed_signal.numpy(), 'length': length.numpy()})[0][0]

    blank_id = tokenizer.vocab_size()
    decoded_prediction = [p for p in logits.argmax(axis=1).tolist() if p != blank_id]
    text = tokenizer.decode_ids(decoded_prediction)

    return text

def model(text):
    formatted_prompt = system_instructions1 + text + "[OpenGPT 4o]"
    stream = client1.text_generation(formatted_prompt, max_new_tokens=300)
    return stream[:-4]

async def respond(audio):
    user = transcribe(audio)
    reply = model(user)
    communicate = edge_tts.Communicate(reply)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_path = tmp_file.name
        await communicate.save(tmp_path)
    return tmp_path