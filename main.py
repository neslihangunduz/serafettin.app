from __future__ import annotations
import os
import json
import base64
import tempfile
from dataclasses import dataclass
from typing import Optional
import streamlit as st
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
from streamlit_mic_recorder import mic_recorder
import google.generativeai as genai
import google.cloud.speech
from google.cloud import texttospeech

DetectorFactory.seed = 0

# =================== YAPILANDIRMA VE KIMLIK BILGILERI ===================
try:
    creds_b64 = st.secrets["GOOGLE_CREDENTIALS"]
    creds_bytes = base64.b64decode(creds_b64)
    creds_dict = json.loads(creds_bytes)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(creds_dict, f, indent=4)
        temp_file_path = f.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path
except KeyError:
    st.error("Google kimlik bilgileri bulunamadı. Lütfen '.streamlit/secrets.toml' dosyasını yapılandırın.")
    st.stop()
except Exception as e:
    st.error(f"Kimlik bilgileri işlenirken bir hata oluştu: {e}")
    st.stop()

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

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
    "Seninle konuşan kadının adı Neslihan ve sen de onun iç sesisin ve siyah tişört giymiş bir iskeletsin")

GOOGLE_TTS_VOICE = {
    "tr": ("tr-TR", "tr-TR-Standard-B"),
    "en": ("en-US", "en-US-Standard-D"),
    "de": ("de-DE", "de-DE-Standard-A"),
    "zh": ("cmn-CN", "cmn-CN-Standard-A"),
    "ko": ("ko-KR", "ko-KR-Standard-A"),
}

# =================== YARDIMCI FONKSIYONLAR ===================

def detect_lang(text: str) -> str:
    try:
        code = detect(text)
        if code.startswith("zh"):
            return "zh"
        return code
    except Exception:
        return "tr"

def pick_predefined(user_text_lower: str) -> Optional[str]:
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

@dataclass
class BotOutput:
    text: str
    lang: str
    audio_bytes: Optional[bytes] = None

# =================== GEMINI LLM YANITI ===================
def llm_answer(persona: str, user_input: str) -> str:
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

# =================== GOOGLE TTS (METINDEN KONUŞMA) ===================
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
        st.error(f"TTS sırasında bir hata oluştu: {e}")
        return None

# GÜNCELLENMİŞ FONKSIYON: Sesi metne çevirir
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
        if response.results:
            return response.results[0].alternatives[0].transcript
        return ""
    except Exception as e:
        st.error(f"Sesi metne çevirirken bir hata oluştu: {e}")
        return ""

# =================== STREAMLIT ARAYÜZÜ ===================
st.title("Şerafettin")

st.write("Şerafettin'e konuşmak için butona basın ve konuşun:")
audio_dict = mic_recorder(start_prompt="Başla", stop_prompt="Dur", format="webm", key="recorder")

user_input = ""

if audio_dict and 'bytes' in audio_dict:
    # Bu satırlar eklenerek ses verisi kontrolü yapılır
    audio_data_size = len(audio_dict['bytes'])
    st.write(f"Yakalanan ses verisinin boyutu: {audio_data_size} byte")

    if audio_data_size > 0:
        st.write("Sesi çözümlüyorum...")
        user_input = transcribe_audio(audio_dict['bytes'])
    else:
        st.error("Ses verisi yakalanamadı. Lütfen mikrofonunuzu kontrol edin ve sayfaya mikrofon izni verdiğinizden emin olun.")

user_text_input = st.text_input("Veya buraya yazarak bir şeyler sor:")
if user_text_input:
    user_input = user_text_input

if user_input:
    st.write(f"Siz: {user_input}")

    if not user_input.strip():
        st.error("Lütfen bir şeyler yazın veya söyleyin.")
    else:
        lang_code = detect_lang(user_input)
        predefined = pick_predefined(user_input.lower())
        answer_text = predefined if predefined else llm_answer(PERSONA, user_input)

        st.write(f"Şerafettin: {answer_text}")

        audio_bytes = synthesize_tts(answer_text, lang_code)
        if audio_bytes:
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            audio_html = f'<audio autoplay="true" controls src="data:audio/mp3;base64,{audio_base64}"></audio>'
            st.markdown(audio_html, unsafe_allow_html=True)

