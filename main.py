from __future__ import annotations
import os
import json
import base64
import tempfile
import time
from typing import Optional

import streamlit as st
from langdetect import detect, DetectorFactory
from streamlit_mic_recorder import mic_recorder

import google.generativeai as genai
import google.cloud.speech
from google.cloud import texttospeech

DetectorFactory.seed = 0

# =================== GOOGLE CREDENTIALS ===================
temp_file_path = None

try:
    creds_b64 = st.secrets["GOOGLE_CREDENTIALS"]
    creds_bytes = base64.b64decode(creds_b64)
    creds_dict = json.loads(creds_bytes)

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(creds_dict, f)
        temp_file_path = f.name

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path

except Exception as e:
    st.error(f"Credential hatası: {e}")
    st.stop()

# =================== GEMINI ===================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

PERSONA = """
Sen Şerafettin'sin. Sarkastik, zeki, hafif alaycı konuşursun.
Her zaman 'sen' diye konuş.
Kullanıcının dilinde cevap ver.
"""

# =================== SESSION ===================
@st.cache_resource
def init_chat():
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=PERSONA
    )
    return model.start_chat()

chat = init_chat()

# =================== RATE LIMIT SAFE CALL ===================
def safe_llm(user_input: str) -> str:
    for attempt in range(3):
        try:
            time.sleep(1)  # RATE LIMIT
            response = chat.send_message(user_input)
            return response.text

        except Exception as e:
            if "429" in str(e):
                time.sleep(2)
            else:
                return f"Hata: {e}"

    return "Kota doldu. Biraz nefes al gel."

# =================== LANGUAGE ===================
def detect_lang(text: str):
    try:
        return detect(text)
    except:
        return "tr"

# =================== TTS ===================
def tts(text, lang_code):
    try:
        client = texttospeech.TextToSpeechClient()

        voice_map = {
            "tr": ("tr-TR", "tr-TR-Standard-B"),
            "en": ("en-US", "en-US-Standard-D"),
        }

        lang, voice = voice_map.get(lang_code, voice_map["tr"])

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice_params = texttospeech.VoiceSelectionParams(
            language_code=lang,
            name=voice
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config
        )

        return response.audio_content

    except Exception as e:
        st.error(f"TTS hata: {e}")
        return None

# =================== STT ===================
def transcribe(audio_bytes):
    try:
        client = google.cloud.speech.SpeechClient()

        audio = google.cloud.speech.RecognitionAudio(content=audio_bytes)

        config = google.cloud.speech.RecognitionConfig(
            encoding=google.cloud.speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code="tr-TR"
        )

        response = client.recognize(config=config, audio=audio)

        if response.results:
            return response.results[0].alternatives[0].transcript

        return ""

    except Exception as e:
        st.error(f"STT hata: {e}")
        return ""

# =================== UI ===================
st.title("💀 Şerafettin v3 (Akıllandı biraz)")

audio = mic_recorder(start_prompt="Konuş", stop_prompt="Dur", key="mic")

text_input = st.text_input("Yaz veya konuş:")

user_input = None

# PRIORITY: AUDIO > TEXT
if audio and audio.get("bytes"):
    st.info("Dinliyorum...")
    user_input = transcribe(audio["bytes"])

elif text_input:
    user_input = text_input

# =================== DUPLICATE GUARD ===================
if "last_input" not in st.session_state:
    st.session_state.last_input = ""

if user_input and user_input != st.session_state.last_input:

    st.session_state.last_input = user_input

    st.write(f"**Neslihan:** {user_input}")

    # LLM
    answer = safe_llm(user_input)

    st.write(f"**Şerafettin:** {answer}")

    # TTS
    lang = detect_lang(answer)
    audio_bytes = tts(answer, lang)

    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio autoplay controls>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

# =================== CLEANUP ===================
if temp_file_path and os.path.exists(temp_file_path):
    os.remove(temp_file_path)
