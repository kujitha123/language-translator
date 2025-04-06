import os
import torch
import gradio as gr
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from deep_translator import GoogleTranslator  # Replaced googletrans with deep-translator
from gtts import gTTS
import librosa
import tempfile
import soundfile as sf
from huggingface_hub import login

class RealTimeTranslator:
    def __init__(self):
        # Initialize Whisper model for speech recognition (using tiny model for lower resource usage)
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        # Supported languages
        self.languages = {
            'en': 'English',
            'fr': 'French',
            'hi': 'Hindi',
            'es': 'Spanish',
            'de': 'German',
            'ja': 'Japanese',
            'te': 'Telugu'
        }

    def speech_to_text(self, audio_path, source_lang):
        """Convert speech to text using Whisper"""
        try:
            # Load and preprocess audio
            audio, _ = librosa.load(audio_path, sr=16000)
            input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(self.device)

            # Generate token ids
            predicted_ids = self.model.generate(input_features)

            # Decode token ids to text
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            return transcription[0]
        except Exception as e:
            return f"Error in speech-to-text: {str(e)}"

    def translate_text(self, text, source_lang, target_lang):
        """Translate text using Google Translate"""
        try:
            translation = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
            return translation
        except Exception as e:
            return f"Error in translation: {str(e)}"

    def text_to_speech(self, text, target_lang):
        """Convert text to speech using gTTS"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
                tts = gTTS(text=text, lang=target_lang)
                tts.save(fp.name)
                return fp.name
        except Exception as e:
            return f"Error in text-to-speech: {str(e)}"

    def process_input(self, input_type, input_data, source_lang, target_lang, *args):
        """Process input based on the selected type (audio or text)"""
        try:
            if input_type == "Audio":
                if input_data is None:
                    return None, "No audio input received", "Please provide audio input"

                # Save input audio temporarily
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
                    sf.write(fp.name, input_data[1], input_data[0])
                    audio_path = fp.name

                # Speech to text
                text = self.speech_to_text(audio_path, source_lang)
                if "Error" in text:
                    return None, text, ""

                # Translate text
                translated_text = self.translate_text(text, source_lang, target_lang)
                if "Error" in translated_text:
                    return None, text, translated_text

                return None, text, translated_text

            elif input_type == "Text":
                if not input_data:
                    return None, "No text input received", "Please provide text input"

                # Translate text
                translated_text = self.translate_text(input_data, source_lang, target_lang)
                if "Error" in translated_text:
                    return None, input_data, translated_text

                # Text to speech
                output_audio_path = self.text_to_speech(translated_text, target_lang)
                if "Error" in output_audio_path:
                    return None, input_data, translated_text

                # Load the generated audio
                output_audio, sr = librosa.load(output_audio_path)

                # Clean up temporary files
                os.unlink(output_audio_path)

                return (sr, output_audio), input_data, translated_text

        except Exception as e:
            return None, f"Error: {str(e)}", f"Error: {str(e)}"

def create_gradio_interface():
    translator = RealTimeTranslator()

    # Create the Gradio interface
    demo = gr.Interface(
        fn=translator.process_input,
        inputs=[
            gr.Radio(choices=["Audio", "Text"], value="Audio", label="Input Type"),
            gr.Audio(sources=["microphone"], type="numpy", label="Input Audio"),
            gr.Textbox(label="Input Text"),
            gr.Dropdown(choices=list(translator.languages.keys()), value="en", label="Source Language"),
            gr.Dropdown(choices=list(translator.languages.keys()), value="fr", label="Target Language")
        ],
        outputs=[
            gr.Audio(label="Translated Audio"),
            gr.Textbox(label="Original Input"),
            gr.Textbox(label="Translated Text")
        ],
        title="Real-time Language Translator",
        description="Choose between audio or text input. If you select audio, the output will be the translated text. If you select text, the output will be the translated audio.",
        examples=[
            ["Audio", None, "", "en", "fr"],
            ["Text", None, "Hello, how are you?", "en", "fr"]
        ]
    )
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True, debug=True)
    
