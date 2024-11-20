import torch
import gradio as gr
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoTokenizer, AutoModelForCausalLM
import pyttsx3
import numpy as np
from scipy.signal import resample
import soundfile as sf

# Initialize devices and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# STT Model
stt_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
stt_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)

# Language Model
lm_name = "distilgpt2"
lm_tokenizer = AutoTokenizer.from_pretrained(lm_name)
lm_model = AutoModelForCausalLM.from_pretrained(lm_name).to(device)

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Function to transcribe audio
def transcribe(audio):
    if not isinstance(audio, tuple) or not audio:
        return "Invalid audio input"
    
    try:
        sample_rate, audio_data = audio
        audio_data = np.array(audio_data, dtype=np.float32)

        # Resample to 16kHz if needed
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            num_samples = int(len(audio_data) * target_sample_rate / sample_rate)
            audio_data = resample(audio_data, num_samples)
        
        # Normalize audio
        audio_data = audio_data / np.abs(audio_data).max() if np.abs(audio_data).max() > 0 else audio_data
        
        # Process audio input and transcribe
        inputs = stt_processor(audio_data, sampling_rate=target_sample_rate, return_tensors="pt", padding="longest").input_values
        inputs = inputs.to(device)
        with torch.no_grad():
            logits = stt_model(inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = stt_processor.batch_decode(predicted_ids)[0]
        return transcription.strip().capitalize()
    except Exception as e:
        print(f"Error in audio input: {e}")
        return "Invalid audio format"

# Function to generate a response using distilgpt2
def generate_response(text):
    responses = {
        "hello": "Hello! How can I help you today?",
        "hi": "Hi there! What can I do for you?",
        "hey": "Hey! How can I assist you?",
        "how are you": "I'm just a program, but I'm here to help! How can I assist you?",
        "how are you doing": "I'm just a program, but I'm here to help! How can I assist you?",
        "what is your name": "I am your AI assistant. How can I help you today?",
        "what is the time": "I'm unable to provide real-time information. Can I help you with something else?",
        "tell me a joke": "Why don't scientists trust atoms? Because they make up everything!"
    }
    
    # Check for common phrases
    lower_text = text.lower()
    if lower_text in responses:
        return responses[lower_text]
    
    # Controlled prompt for other inputs
    prompt = f"Q: {text}\nA:"
    inputs = lm_tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(inputs).to(device)
    with torch.no_grad():
        outputs = lm_model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=50,  # Increased response length
            no_repeat_ngram_size=3,  # Avoid repeating trigrams
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=lm_tokenizer.eos_token_id
        )
    response = lm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

# Function for Text-to-Speech using pyttsx3
def synthesize_speech(text):
    engine.save_to_file(text, "response.wav")
    engine.runAndWait()
    return "response.wav"

# Function for Gradio Interface
def gradio_interface(audio):
    try:
        transcription = transcribe(audio)
        if transcription.lower() == "invalid audio input":
            return transcription, "No valid response due to invalid audio input.", None
        
        response = generate_response(transcription)
        tts_output = synthesize_speech(response)
        return transcription, response, tts_output
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return error_msg, "An error occurred.", None

# Create Gradio Interface
interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Audio(type="numpy"),
    outputs=[
        gr.Textbox(label="Transcribed Text"),
        gr.Textbox(label="Generated Response"),
        gr.Audio(label="TTS Output", type="filepath")
    ],
    title="Lightweight STT, NLP, and TTS System",
    description="Record your voice, get transcribed text, AI-generated response, and TTS output."
)

if __name__ == "__main__":
    interface.launch(share=True)
