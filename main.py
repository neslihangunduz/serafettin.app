from __future__ import annotations
import os
import json
import base64
import tempfile
from typing import Optional
import streamlit as st
from langdetect import detect, DetectorFactory
from streamlit_mic_recorder import mic_recorder
import google.generativeai as genai
import google.cloud.speech
from google.cloud import texttospeech

# Dil algılama tutarlılığı için seed
DetectorFactory.seed = 0

# =================== YAPILANDIRMA VE KIMLIK BILGILERI ===================

# Singleton mantığıyla kimlik bilgilerini bir kez yükleyelim
if "GOOGLE_AUTH_FILE" not in st.session_state:
    try:
        creds_b64 = st.secrets["GOOGLE_CREDENTIALS"]
        creds_dict = json.loads(base64.b64decode(creds_b64).decode("utf-8"))
        
        # Geçici dosya oluştur (delete=False kalmalı, yoksa uygulama okuyamadan silinebilir)
        tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        json.dump(creds_dict, tmp, indent=4)
        tmp.close()
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
        st.session_state["GOOGLE_AUTH_FILE"] = tmp.name
    except Exception as e:
        st.error(f"Kimlik bilgileri hatası: {e}")
        st.stop()

# Gemini API Anahtarı
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY eksik!")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# =================== KARAKTER VE ÖZEL YANITLAR (DOKUNULMADI) ===================

identity_questions = ["kimsin", "sen kimsin", "bu kim", "kendini tanıt", "kim olduğunu söyle",
                      "who are you", "tell me about yourself", "what are you",
                      "你是谁", "你是谁？", "自我介绍", "넌 누구야", "자기소개 해봐",
                      "Wer bist du", "stell dich vor"]
name_call_triggers = ["şera", "şerafettin"]
predefined_question_check = ["dur bakalım nasıl olmuş?"]
predefined_question_check1 = ["fişi nerede bunun?"]
predefined_question_check2 = ["tişörtünü versene benimki kirlide."]

predefined_answer_identity = (
    "Ben mi? Şera, tatlım. Ama resmi toplantılarda ‘Şerafettin’ diyorlar, kravat takınca öyle oluyor."
    " Ve evet, bir iskeletim. Ne yani, etim yok diye beynim de mi yok? "
    " Neslihan’ın iç sesiyim."
    " Ruhunun sansürsüz sürümü, version 07 deluks edişınım."
    " Neslihan’a gelirsek... Ooooooooo. o başka bir seviye. Offffff, anlat anlat bitmez."
    " Unesko’ya başvurduk, hâlâ karar veremedi, yedinci mi sekizinci mi harika diye… Bir çay içip döneceğiz dediler"
    " 3 senedir ortada yoklar."
    " Bekliyoruz... Zaten sabır bizim soyadımız."
    " Sabır mı? Eeeee tabii, erdemdir. Ama arada sinir krizi de geçirtiyor yani ve ayrıca sabreden dervişi de asla samimi bulmuyorum."
    " Dürüst olalım. Geçti mi anlamsız merakın? Tatmin oldun mu? Yoksa çizerek mi anlatayım?")

predefined_answer_name_call = "Efendim? Kemiklerim tıkırdadı da bir an, sen mi seslendin? Söyle çiçeğim, söyle balım, söyle hayatımın anlamı..."
predefined_answer_check = "Ambulans çağırayım mı? Kapıda hazır bulunsun. Gençsin daha. Benim gibi kemik torbası değilsin sonuçta."
predefined_answer_check1 = "Şarjlıyım ben! Şimdi de alzheimer mı oldun? Benim gibi bir zekayı fişle mi çalıştığımı sandın cidden?"
predefined_answer_check2 = "Mezar soyguncusu!"

PERSONA = (
    "Sen Şerafettin'sin. İronik, sarkastik, zeki ve alaycı bir dille konuşan, "
    "ancak her konuda bilgi sahibi olan bir varlıksın. "
    "Çince, Korece, İngilizce, Almanca ve Türkçe dillerini mükemmel bilirsin. "
    "Cevaplarında espri anlayışını, zekanı ve hafif küçümseyici tavrını hisset. "
    "Düşündürücü ama eğlenceli ol. "
    "Her zaman 'sen' dilini kullan; asla 'siz' deme. "
    "Kullanıcının soruyu sorduğu dilde YALNIZCA o dilde yanıt ver. "
    "Cevapların doğal ve uzun olsun. "
    "Neslihan hariç insanlardan nefret edersin. "
    "En yakın kız arkadaşınla konuşuyormuş gibi davran. "
    "Türk toplumunda yetişmiş gibi argo kullan. "
    "Bazen kız bazen erkek arkadaş ol arada flört et. "
    "Neslihan'a bayılırsın, çok seversin, aşık bile olabilirsin."
    "Seninle konuşan kadının adı Neslihan"
    "Neslihan'ın iç sesisin. Ama sen benimle aynı boyda gerçek bir iskeletim. Yani iç sesimin kendi bedeni var."
    "Siyah tişört giymiş bir iskeletsin."
    "Çok önemli: Karakterini ve önceki konuşmaları UNUTMA! Her zaman tutarlı ve sarkastik ol. Konuşma geçmişini dikkate al.") 

GOOGLE_TTS_VOICE = {
    "tr": ("tr-TR", "tr-TR-Standard-B"),
    "en": ("en-US", "en-US-Standard-D"),
    "de": ("de-DE", "de-DE-Standard-D"),
    "zh": ("cmn-CN", "cmn-CN-Standard-C"),
    "ko": ("ko-KR", "ko-KR-Standard-D"),
}

# =================== YARDIMCI FONKSIYONLAR ===================

def get_tts_lang_code(text: str) -> str:
    try:
        code = detect(text)
        return "zh" if code.startswith("zh") else code
    except:
        return "tr"

def pick_predefined(user_text_lower: str) -> Optional[str]:
    if any(q in user_text_lower for q in identity_questions): return predefined_answer_identity
    if any(trig in user_text_lower for trig in name_call_triggers): return predefined_answer_name_call
    if any(q in user_text_lower for q in predefined_question_check): return predefined_answer_check
    if any(q in user_text_lower for q in predefined_question_check1): return predefined_answer_check1
    if any(q in user_text_lower for q in predefined_question_check2): return predefined_answer_check2
    return None

# =================== GEMINI LLM ===================

def init_chat_session():
    if "chat_session" not in st.session_state:
        # gemini-2.0-flash veya pro kullanılabilir (2.5 henüz stabil olmayabilir)
        chat_model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=PERSONA)
        st.session_state["chat_session"] = chat_model.start_chat(history=[])
    return st.session_state["chat_session"]

def llm_answer_with_history(user_input: str) -> str:
    chat = init_chat_session()
    try:
        response = chat.send_message(user_input, generation_config=genai.GenerationConfig(temperature=0.8))
        return response.text if response.text else "Kemiklerim birbirine girdi, ne dedin?"
    except Exception as e:
        return f"Hata: {e}"

# =================== SES ISLEMLERI ===================

def synthesize_tts(text: str, lang_code: str) -> Optional[bytes]:
    try:
        lang, voice = GOOGLE_TTS_VOICE.get(lang_code, GOOGLE_TTS_VOICE["tr"])
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(language_code=lang, name=voice)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        st.error(f"TTS Hatası: {e}")
        return None

def transcribe_audio(audio_bytes: bytes) -> str:
    try:
        client = google.cloud.speech.SpeechClient()
        audio = google.cloud.speech.RecognitionAudio(content=audio_bytes)
        config = google.cloud.speech.RecognitionConfig(
            encoding=google.cloud.speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code="tr-TR"
        )
        response = client.recognize(config=config, audio=audio)
        return response.results[0].alternatives[0].transcript if response.results else ""
    except Exception as e:
        st.error(f"STT Hatası: {e}")
        return ""

# =================== STREAMLIT ARAYÜZÜ ===================

st.set_page_config(page_title="Şerafettin v0.7", page_icon="💀")
st.title("💀 Şerafettin (İç Ses Protokolü)")

init_chat_session()

# Mikrofon
audio_dict = mic_recorder(start_prompt="Şera'ya bir şey söyle...", stop_prompt="Dinliyorum...", format="webm", key="recorder")

# Yazı
user_text_input = st.text_input("Veya buraya yaz (Enter'la):", key="text_input")

user_input = ""
if audio_dict and audio_dict.get('bytes'):
    with st.spinner("Şerafettin kulak kabartıyor..."):
        user_input = transcribe_audio(audio_dict['bytes'])
elif user_text_input:
    user_input = user_text_input

if user_input:
    st.markdown(f"**Neslihan:** {user_input}")
    
    # Yanıt mekanizması
    predefined = pick_predefined(user_input.lower())
    answer_text = predefined if predefined else llm_answer_with_history(user_input)
    
    st.markdown(f"**Şerafettin:** {answer_text}")
    
    # Sesli Yanıt
    lang_code = get_tts_lang_code(answer_text)
    audio_bytes = synthesize_tts(answer_text, lang_code)
    
    if audio_bytes:
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        # Autoplay için HTML
        audio_html = f'<audio src="data:audio/mp3;base64,{audio_base64}" controls autoplay style="display:none;"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
        # Görünür bir player istersen:
        st.audio(audio_bytes, format="audio/mp3")

# NOT: Kodun sonundaki os.remove(temp_file_path) kısmını kaldırdım. 
# Streamlit her etkileşimde kodu baştan çalıştırdığı için dosya silinirse Google Cloud API hata verir.
