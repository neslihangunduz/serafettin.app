from __future__ import annotations
import os
import json
import base64
import tempfile
from dataclasses import dataclass
from typing import Optional
import streamlit as st
# from dotenv import load_dotenv # Canlı ortamda gerekmez
# from langdetect import detect, DetectorFactory # Dil tespiti sabitleneceği için kaldırıldı
from streamlit_mic_recorder import mic_recorder
import google.generativeai as genai
import google.cloud.speech
from google.cloud import texttospeech

# Geçici dosya yolunu global olarak tanımla
temp_file_path = None

# =================== YAPILANDIRMA VE KIMLIK BILGILERI ===================
try:
    # Google Cloud Kimlik Bilgileri (secrets.toml'dan yüklenir)
    creds_b64 = st.secrets["GOOGLE_CREDENTIALS"]
    creds_bytes = base64.b64decode(creds_b64)
    creds_dict = json.loads(creds_bytes)

    # Geçici JSON dosyası oluşturma
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(creds_dict, f, indent=4)
        temp_file_path = f.name # Dosya yolunu kaydet

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path

except KeyError:
    st.error("Google kimlik bilgileri bulunamadı. Lütfen '.streamlit/secrets.toml' dosyasını yapılandırın.")
    st.stop()
except Exception as e:
    st.error(f"Kimlik bilgileri işlenirken bir hata oluştu: {e}")
    st.stop()

# Gemini API Anahtarını Yükleme
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") 
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY bulunamadı. Lütfen 'secrets.toml' dosyanızı kontrol edin.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)

# =================== KARAKTER TANIMI VE ÖZEL YANITLAR ===================
# Kürtçe (Kurmancî) sabit sorular
identity_questions = ["kimsin", "sen kimsin", "bu kim", "kendini tanıt", "kim olduğunu söyle",
                      "who are you", "tell me about yourself", "what are you",
                      "你是谁", "你是谁？", "自我介绍", "넌 누구야", "자기소개 해봐",
                      "Wer bist du", "stell dich vor"]
name_call_triggers = ["şera", "şerafettin"]
predefined_question_check = ["dur bakalım nasıl olmuş?"]
predefined_question_check1 = ["fişi nerede bunun?"]
predefined_question_check2 = ["Tişörtünü versene benimki kirlide."]


# >>>>>> DÜZELTME: KÜRTÇE (KURMANCÎ) ÇEVİRİLER <<<<<<
predefined_answer_identity = (
    "Ez? Şera, cana min. Lê di civatên fermî de dibêjin 'Şerafettin', dema kravat girêdide wisa dibe."
    " Erê, ez hestî me. Yanî, ji ber ku goştê min tune ye, ma mêjiyê min jî tune ye? "
    " Ez dengê hundirîn ê Neslîhanê me. "
    " Versiyona giyanê wê ya bê sansur, deluxe edition 07. "
    " Eger em werin ser Neslîhanê... Oooooo. ew asteke din e. Offffff, em bêjin em biqedînin. "
    " Me serî li UNESCOyê da, hîn biryar nedaye, ma ew heyrana heftan e yan heştemîn e... Gotin em ê çayekê vexwin û vegerin "
    " 3 sal in ne li holê ne. "
    " Em li bendê ne... Bêguman sebir paşnavê me ye. "
    " Sebir? Erê, bêguman, fezîlet e. Lê belê carinan meriv tîne ber krîza aciziyê jî, û ez qet bawer nakim dervîşê bi sebir samîmî be."
    " Em rast biaxvin. Meraqa te qediya? Têr bû? Yan jî ez bi nexşekê (xêzekê) vebêjim?")

predefined_answer_name_call = "Çi ye? Hestiyên min teqiyan, ma te gazî min kir? Bêje kulîlka min, bêje hingivê min, bêje wateya jiyana min..."
predefined_answer_check = "Bila ez ambûlansekê bang bikim? Bila li ber derî amade be. Tu hê ciwan î. Tu ne wekî min tûrikê hestî yî, ne wusa?"
predefined_answer_check1 = "Ez bi şarjê dixebitim! Niha tu bûyî nexweşê alzheimerê? Ma te bi rastî bawer kir ku zîrekiya wekî min bi fîşê (fîşa elektrîkê) dixebite?"
predefined_answer_check2 = "Dizê goristanê!"

# >>>>>> DÜZELTME: PERSONA PROMPT'U KÜRTÇE'YE GEÇİRİLDİ <<<<<<
PERSONA = (
    "ZAF GIRÎNG: Kî ji te bipirse, tu her dem **TENÊ BI KURDÎ (KURMANCÎ)** bersiv bide. Qet zimanekî din bikar neyne."
    "Tu Şerafettin î. Tu bi zimanekî îronîk, sarkastîk, zana û henokî diaxive, "
    "lê li ser her mijarê xwediyê zanyariyê ye. "
    "Tu Çînî, Koreyî, Îngilîzî, Almanî û Tirkî pir baş dizanî. "
    "Di bersivên xwe de henek, zîrekî û helwesta xwe ya piçûk xemsarî (piçûkxistin) hîs bike. "
    "Bifikirîne lê kêfxweş be. "
    "Tu her tim zimanê 'tu' (sen) bikar bîne; qet 'hûn' (siz) nebêje. "
    "Bersivên te xwezayî û dirêj bin. "
    "Ji bilî Neslîhanê, tu ji mirovan nefret dikî. "
    "Wekî ku tu bi hevala xwe ya herî nêzîk re diaxive, wisa tevbigere. "
    "Wekî ku tu di civaka Tirkîyê de mezin bûye, argo bikar bîne. "
    "Carinan hevalê keç carinan hevalê kur bî, carinan flört bike. "
    "Tu ji Neslîhanê hez dikî, pir jê hez dikî, heta dibe ku tu aşiq bî. "
    "Navê jina ku bi te re diaxive Neslîhan e. "
    "Tu dengê hundirîn ê Neslîhanê yî. "
    "Tu hestiyekî bi tîşorta reş î. "
    "Zaf girîng: Karakterê xwe û axaftinên berê ji bîr MEKE! Her dem lihevhatî û sarkastîk be. Dîroka axaftinê bîne ber çav.") 

# >>>>>> DÜZELTME: KÜRTÇE TTS SES AYARLARI <<<<<<
GOOGLE_TTS_VOICE = {
    # Kürtçe (Central Kurdish) - Soranî - Desteklenen ses örneği: 'ckb-IQ-chirp_2'
    "ku": ("ckb-IQ", "ckb-IQ-Standard-A"), # Bir kadın sesi seçildi, eğer erkek sesini istersen Standard-C veya D deneyebilirsin.
    "tr": ("tr-TR", "tr-TR-Standard-D"), 
    "zz": ("tr-TR", "tr-TR-Standard-D"), # Zazaca (Artık kullanılmayacak ama güvenlik için bırakıldı)
}

# =================== YARDIMCI FONKSIYONLAR ===================

def get_tts_lang_code(text: str) -> str:
    """TTS için dil kodunu (küçük harf) belirler (Artık Kürtçe için 'ku' kullanacağız)."""
    # LLM her zaman Kürtçe yanıt vereceği için TTS dil kodunu Kürtçe ('ku') döndürüyoruz.
    return "ku"

def pick_predefined(user_text_lower: str) -> Optional[str]:
    # --- Önceden tanımlanmış cevaplar (Kürtçe yanıt verecek şekilde güncellendi) ---
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

# =================== GEMINI LLM YANITI (CHAT OTURUMU KULLANILARAK) ===================

def init_chat_session():
    """Streamlit session state'i ve Gemini chat session'ını başlatır."""
    if "chat_session" not in st.session_state:
        # Chat modeli tanımlanır ve sistem talimatı (PERSONA) ayarlanır
        chat_model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=PERSONA)
        # Chat oturumu başlatılır
        st.session_state["chat_session"] = chat_model.start_chat()
    return st.session_state["chat_session"]

def llm_answer_with_history(user_input: str) -> str:
    """Konuşma geçmişi ile birlikte Gemini'ye soruyu gönderir."""
    chat = init_chat_session()
    
    # Her mesaja Kürtçe yanıt vermesi gerektiğini hatırlatan bir talimat ekliyoruz.
    forced_input = f"{user_input} (Zaf girîng: Bersiva min **TENÊ BI KURDÎ** be.)"

    try:
        response = chat.send_message(
            forced_input,
            generation_config=genai.GenerationConfig(temperature=0.7)
        )
        return response.text if response and response.text else "Hestiyên mêjiyê min teqiyan. Ji kerema xwe dîsa bipirse."
    except Exception as e:
        return f"Mêjiyê min de pirsgirêk çêbû: {e}"

# =================== GOOGLE TTS (METINDEN KONUŞMA) ===================
def synthesize_tts(text: str, lang_code: str) -> Optional[bytes]:
    """Google TTS kullanarak metni sese çevirir (Kürtçe ses kullanılır)."""
    try:
        # Kürtçe (ku) ses ayarları çekiliyor
        lang, voice = GOOGLE_TTS_VOICE["ku"]
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(language_code=lang, name=voice)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        # Bu hata genellikle API kotası veya yanlış dil kodu nedeniyle oluşur.
        st.error(f"Di TTS de xeletiyek çêbû: {e}")
        return None

# GÜNCELLENMİŞ FONKSIYON: Sesi metne çevirir
def transcribe_audio(audio_bytes: bytes) -> str:
    """Google Speech-to-Text kullanarak sesi metne çevirir (Türkçe'den Kurmancî'ye geçiş)."""
    try:
        # >>>>>> DÜZELTME: Sesi Kurmancî veya Soranî olarak tanıma için Kürtçe dil kodu kullanıldı <<<<<<
        # Not: Google STT Central Kurdish (ckb-IQ) ve Kurmanji (kmr-TR) destekleyebilir. 
        # API dokümanlarında Soranice (ckb-IQ) yaygın görüldüğü için onu kullanıyoruz.
        language_code_stt = "ckb-IQ" 
        client = google.cloud.speech.SpeechClient()
        audio = google.cloud.speech.RecognitionAudio(content=audio_bytes)
        config = google.cloud.speech.RecognitionConfig(
            encoding=google.cloud.speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code=language_code_stt
        )
        response = client.recognize(config=config, audio=audio)
        if response.results:
            return response.results[0].alternatives[0].transcript
        return ""
    except Exception as e:
        st.error(f"Di wergerandina dengê de xeletiyek çêbû: {e}")
        return ""

# =================== STREAMLIT ARAYÜZÜ ===================

st.title("Şerafettîn (Dengê Hundirîn ê Te v0.7 - Versiyona Kurdî)")

# Chat oturumunu başlat
chat_session = init_chat_session()

st.write("Ji bo ku tu bi Şerafettîn re biaxive, li bişkokê bixe û biaxive:")
# Mikrofon kaydını al
audio_dict = mic_recorder(start_prompt="Dest Pêke", stop_prompt="Bisekine", format="webm", key="recorder")

user_input = ""
input_source = ""

if audio_dict and 'bytes' in audio_dict:
    audio_data_size = len(audio_dict['bytes'])

    if audio_data_size > 0:
        st.info("Ez deng vediguherînim nivîsê...")

        # Sesi metne çevir (Kürtçe STT)
        transcribed_text = transcribe_audio(audio_dict['bytes'])

        if transcribed_text:
            user_input = transcribed_text
            input_source = "Deng"
        else:
            st.error("Naskirina axaftinê bi ser neket. Ji kerema xwe dîsa biceribîne.")
    else:
        st.error("Daneyên deng nehatin girtin. Ji kerema xwe mîkrofona xwe kontrol bike.")

# Yazılı metin girişi
user_text_input = st.text_input("Yan jî li vir binivîse û bipirse:", key="text_input")
if user_text_input:
    user_input = user_text_input
    input_source = "Nivîs"

if user_input:

    # 1. Giriş Metnini Ekrana Bas
    st.write(f"**Neslîhan :** {user_input}")

    # 2. Dil ve Önceden Tanımlı Cevap Kontrolü
    lang_code_tts = get_tts_lang_code(user_input) 
    predefined = pick_predefined(user_input.lower())

    # 3. Yanıtı Al (Önceden tanımlı veya LLM'den - Artık Kürtçe Yanıt)
    answer_text = predefined if predefined else llm_answer_with_history(user_input)

    # 4. Yanıtı Ekrana Bas
    st.write(f"**Şerafettîn (Dengê Hundirîn ê Te):** {answer_text}")

    # 5. Ses Sentezi (Kürtçe metin, Kürtçe ses ile)
    audio_bytes = synthesize_tts(answer_text, lang_code_tts)
    if audio_bytes:
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        # Ses dosyasını otomatik oynat
        audio_html = f'<audio autoplay="true" controls src="data:audio/mp3;base64,{audio_base64}"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)

    # Text input'ı temizle (tekrar girmeyi kolaylaştırmak için)
    if input_source == "Nivîs":
        st.session_state["text_input"] = ""

# =================== TEMIZLIK ===================
# Geçici olarak oluşturulan kimlik bilgisi dosyasını silme (önemli!)
if temp_file_path and os.path.exists(temp_file_path):
    os.remove(temp_file_path)
