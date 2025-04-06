import os
import torch
import gradio as gr
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from deep_translator import GoogleTranslator
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

        # Supported languages with correct Google Translate codes
        self.languages = {
            'en': 'English',
            'fr': 'French',
            'hi': 'Hindi',
            'es': 'Spanish',
            'de': 'German',
            'ja': 'Japanese',
            'te': 'Telugu'
        }
        
        # Mapping from our language codes to Google Translator codes
        self.google_codes = {
            'en': 'en',
            'fr': 'fr',
            'hi': 'hi',
            'es': 'es',
            'de': 'de',
            'ja': 'ja',
            'te': 'te'
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
            # Convert our language codes to Google Translator codes
            google_source = self.google_codes.get(source_lang, source_lang)
            google_target = self.google_codes.get(target_lang, target_lang)
            
            # Print for debugging
            print(f"Translating from {google_source} to {google_target}: '{text}'")
            
            translation = GoogleTranslator(source=google_source, target=google_target).translate(text)
            return translation
        except Exception as e:
            return f"Error in translation: {str(e)}"

    def text_to_speech(self, text, target_lang):
        """Convert text to speech using gTTS"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
                # gTTS uses the same language codes as Google Translate
                tts_lang = self.google_codes.get(target_lang, target_lang)
                tts = gTTS(text=text, lang=tts_lang)
                tts.save(fp.name)
                return fp.name
        except Exception as e:
            return f"Error in text-to-speech: {str(e)}"

    def process_input(self, input_type, audio_input, text_input, source_lang, target_lang):
        """Process input based on the selected type (audio or text)"""
        try:
            if input_type == "Audio":
                if audio_input is None:
                    return None, "No audio input received", "Please provide audio input"

                # Save input audio temporarily
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
                    sf.write(fp.name, audio_input[1], audio_input[0])
                    audio_path = fp.name

                # Speech to text
                text = self.speech_to_text(audio_path, source_lang)
                if text.startswith("Error"):
                    return None, text, ""

                # Translate text
                translated_text = self.translate_text(text, source_lang, target_lang)
                if translated_text.startswith("Error"):
                    return None, text, translated_text

                # Text to speech (output audio)
                output_audio_path = self.text_to_speech(translated_text, target_lang)
                if output_audio_path.startswith("Error"):
                    return None, text, translated_text

                # Load the generated audio
                output_audio, sr = librosa.load(output_audio_path)

                # Clean up temporary files
                os.unlink(output_audio_path)
                os.unlink(audio_path)  # Also clean up input audio

                return (sr, output_audio), text, translated_text

            elif input_type == "Text":
                if not text_input:
                    return None, "No text input received", "Please provide text input"

                # Translate text
                translated_text = self.translate_text(text_input, source_lang, target_lang)
                if translated_text.startswith("Error"):
                    return None, text_input, translated_text

                # Text to speech
                output_audio_path = self.text_to_speech(translated_text, target_lang)
                if output_audio_path.startswith("Error"):
                    return None, text_input, translated_text

                # Load the generated audio
                output_audio, sr = librosa.load(output_audio_path)

                # Clean up temporary files
                os.unlink(output_audio_path)

                return (sr, output_audio), text_input, translated_text

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)  # Print error for debugging
            return None, error_msg, error_msg

def create_gradio_interface():
    translator = RealTimeTranslator()

    # Create the Gradio interface with improved input handling
    with gr.Blocks(title="Real-time Language Translator") as demo:
        gr.Markdown("# Real-time Language Translator")
        gr.Markdown("Choose between audio or text input. The system will translate and provide both text and audio output.")
        
        input_type = gr.Radio(choices=["Audio", "Text"], value="Text", label="Input Type")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Input Audio", visible=False)
                text_input = gr.Textbox(label="Input Text", value="Hello, how are you?")
                
                language_choices = [f"{code} - {name}" for code, name in translator.languages.items()]
                source_lang = gr.Dropdown(choices=list(translator.languages.keys()), 
                                         value="en", 
                                         label="Source Language")
                target_lang = gr.Dropdown(choices=list(translator.languages.keys()), 
                                         value="fr", 
                                         label="Target Language")
                translate_button = gr.Button("Translate")
            
            with gr.Column():
                output_audio = gr.Audio(label="Translated Audio")
                original_text = gr.Textbox(label="Original Text")
                translated_text = gr.Textbox(label="Translated Text")
        
        # Show/hide input elements based on selection
        def update_visibility(input_choice):
            if input_choice == "Audio":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
        
        input_type.change(fn=update_visibility, inputs=input_type, outputs=[audio_input, text_input])
        
        # Trigger translation when button is clicked
        translate_button.click(
            fn=translator.process_input,
            inputs=[input_type, audio_input, text_input, source_lang, target_lang],
            outputs=[output_audio, original_text, translated_text]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["Text", None, "Hello, how are you?", "en", "fr"],
                ["Text", None, "I love learning new languages", "en", "es"],
                ["Text", None, "This is a great translation tool", "en", "de"]
            ],
            inputs=[input_type, audio_input, text_input, source_lang, target_lang]
        )
        
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True, debug=True)