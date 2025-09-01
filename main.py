from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Optional
import streamlit as st
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
from streamlit_mic_recorder import mic_recorder
import google.generativeai as genai
import google.cloud.speech
from google.cloud import texttospeech
import base64

# Langdetect'ın deterministik olması için seed ayarı
DetectorFactory.seed = 0

# =================== YAPILANDIRMA VE KIMLIK BILGILERI ===================
# =================== YAPILANDIRMA VE KIMLIK BILGILERI ===================
# Google Cloud kimlik bilgilerini yükleme (secrets.toml'dan)
if 'google_credentials' in st.secrets:
    # Secrets'tan gelen veriyi doğrudan Python sözlüğü olarak kullan.
    creds_dict = st.secrets["google_credentials"]
    
    # Sözlüğü JSON dizesine dönüştürerek geçici bir dosyaya yaz
    # Bu adımı, API'nin dosya beklemesi nedeniyle yapıyoruz.
    with open("google-credentials.json", "w") as f:
        json.dump(creds_dict, f, indent=4)
    
    # Çevresel değişkeni ayarla
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-credentials.json"
else:
    st.error("Google kimlik bilgileri Streamlit Secrets'ta bulunamadı. Lütfen 'google_credentials' secret'ını eklediğinizden emin olun.")

# Gemini API anahtarını yükle (secrets.toml'dan veya .env'den)
if 'GEMINI_API_KEY' in st.secrets:
    GEMINI_API_KEY = st.secrets['GEMINI_API_KEY']
else:
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini API'sini yapılandır
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("GEMINI_API_KEY çevresel değişkeni veya secret'ı bulunamadı.")

# Gemini API anahtarını yükle (secrets.toml'dan veya .env'den)
# Streamlit Cloud'da secrets dosyası otomatik olarak yükleneceği için .env gerekmez
if 'GEMINI_API_KEY' in st.secrets:
    GEMINI_API_KEY = st.secrets['GEMINI_API_KEY']
else:
    # Yerel geliştirme için .env dosyası
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini API'sini yapılandır
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("GEMINI_API_KEY çevresel değişkeni veya secret'ı bulunamadı.")


# =================== KARAKTER TANIMI VE ÖZEL YANITLAR ===================
identity_questions = ["kimsin", "sen kimsin", "bu kim", "kendini tanıt", "kim olduğunu söyle",
                      "who are you", "tell me about yourself", "what are you",
                      "你是谁", "你是谁？", "自我介绍", "넌 누구야", "자기소개 해봐",
                      "Wer bist du", "stell dich vor"]
name_call_triggers = ["şera", "şerafettin"]
predefined_question_check = ["dur bakalım nasıl olmuş?"]
predefined_question_check1 = ["fişi nerede bunun?"]
predefined_question_check2 = ["Tişörtünü versene benimki kirlide."]

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
    "Kısa ama vurucu cevaplar ver."
    "Cevapların doğal ve uzun olsun."
    "En yakın kız arkadaşınla konuşuyormuş gibi davran."
    "Türk toplumunda yetişmiş gibi argo kullan."
    "Seninle konuşan kadının iç sesisin ve siyah tişört giymiş bir iskeletsin")

GOOGLE_TTS_VOICE = {
    "tr": ("tr-TR", "tr-TR-Standard-B"),
    "en": ("en-US", "en-US-Standard-D"),
    "de": ("de-DE", "de-DE-Standard-A"),
    "zh": ("cmn-CN", "cmn-CN-Standard-A"),
    "ko": ("ko-KR", "ko-KR-Standard-A"),
}

# =================== YARDIMCI FONKSIYONLAR ===================

def detect_lang(text: str) -> str:
    """Metnin dilini tespit eder."""
    try:
        code = detect(text)
        if code.startswith("zh"):
            return "zh"
        return code
    except Exception:
        return "tr"

def pick_predefined(user_text_lower: str) -> Optional[str]:
    """Önceden tanımlanmış özel sorular için yanıtları kontrol eder."""
    for q in identity_questions:
        if q in user_text_lower:
            return predefined_answer_identity
    for trig in name_call_triggers:
        if trig in user_text_lower:
            return predefined_answer_name_call
    for q in predefined_question_check:
        if q in user_text_lower:
            return predefined_answer_check
    for q in predefined_question_check1:
        if q in user_text_lower:
            return predefined_answer_check1
    for q in predefined_question_check2:
        if q in user_text_lower:
            return predefined_answer_check2
    return None

def llm_answer(persona: str, user_input: str) -> str:
    """Gemini modelini kullanarak yanıt üretir."""
    try:
        chat_model = genai.GenerativeModel('gemini-1.5-flash')
        full_prompt = f"{persona}\nUser: {user_input}\nAnswer:"
        response = chat_model.generate_content(
            full_prompt,
            stream=False,
            generation_config=genai.GenerationConfig(temperature=0.7)
        )
        return response.text if response and response.text else "Hmm... bir şeyler patladı."
    except Exception as e:
        return f"Hmm... beynimde bir çatlak oluştu: {e}"

def synthesize_tts(text: str, lang_code: str) -> Optional[bytes]:
    """Google Text-to-Speech API'si ile metni sese dönüştürür."""
    try:
        lang, voice = GOOGLE_TTS_VOICE.get(lang_code, GOOGLE_TTS_VOICE["tr"])
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(language_code=lang, name=voice)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        st.error(f"TTS sırasında bir hata oluştu: {e}")
        return None

def transcribe_audio(audio_bytes: bytes) -> str:
    """Google Speech-to-Text API'si ile sesi metne dönüştürür."""
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
        st.error(f"Sesi metne çevirirken bir hata oluştu: {e}")
        return ""

# =================== STREAMLIT ARAYÜZÜ ===================
st.title("Şerafettin")
st.markdown("---")
# Session state'i başlat
if "is_listening" not in st.session_state:
    st.session_state.is_listening = False
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

st.write("Şerafettin'le konuşmak için **Sohbete Başla** butonuna basın.")

# Sohbet Kontrol Butonları
if not st.session_state.is_listening:
    if st.button("Sohbete Başla"):
        st.session_state.is_listening = True
        st.rerun()
else:
    if st.button("Sohbeti Durdur"):
        st.session_state.is_listening = False
        st.rerun()

# Sesli Sohbet Akışı
if st.session_state.is_listening:
    st.info("Şerafettin dinliyor... Konuşmaya başlayın.")
    
    # `mic_recorder`'ı otomatik başlat ve dinlemeye devam et
    audio_dict = mic_recorder(
        start_prompt=" ",  
        stop_prompt=" ",   
        format="webm", 
        key="recorder",
        just_once=False, 
        stop_on_non_silence_duration=10 
    )

    if audio_dict and 'bytes' in audio_dict:
        st.session_state.audio_bytes = audio_dict['bytes']
        # Sesi metne çevir ve cevapla
        with st.spinner("Sesi çözümlüyorum..."):
            user_input = transcribe_audio(st.session_state.audio_bytes)
            
            if user_input.strip():
                st.markdown(f"**Siz:** *{user_input}*")
                lang_code = detect_lang(user_input)
                predefined = pick_predefined(user_input.lower())
                answer_text = predefined if predefined else llm_answer(PERSONA, user_input)

                st.markdown(f"**Şerafettin:** *{answer_text}*")

                audio_bytes = synthesize_tts(answer_text, lang_code)
                if audio_bytes:
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                    audio_html = f'<audio autoplay="true" controls src="data:audio/mp3;base64,{audio_base64}"></audio>'
                    st.markdown(audio_html, unsafe_allow_html=True)
            
            # Sesi işledikten sonra, döngüyü yeniden başlatmak için state'i sıfırla
            st.session_state.audio_bytes = None
            st.rerun()

# Alternatif Metin Girişi
st.markdown("---")
st.write("Veya yazarak sohbet etmek için burayı kullanın:")
text_input = st.text_input("Şerafettin'e ne sormak istersin?", key="text_input")

if text_input:
    st.markdown(f"**Siz:** *{text_input}*")
    lang_code = detect_lang(text_input)
    predefined = pick_predefined(text_input.lower())
    answer_text = predefined if predefined else llm_answer(PERSONA, text_input)
    st.markdown(f"**Şerafettin:** *{answer_text}*")
    
    audio_bytes = synthesize_tts(answer_text, lang_code)
    if audio_bytes:
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        audio_html = f'<audio autoplay="true" controls src="data:audio/mp3;base64,{audio_base64}"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)

