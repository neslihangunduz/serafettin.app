from __future__ import annotations
import os
import json
import base64
import tempfile
from dataclasses import dataclass
from typing import Optional
import streamlit as st
# from dotenv import load_dotenv # Canlı ortamda gerekmez
# from langdetect import detect, DetectorFactory # Zazaca tespit zor olduğu için kaldırıyoruz
from streamlit_mic_recorder import mic_recorder
import google.generativeai as genai
import google.cloud.speech
from google.cloud import texttospeech

# DetectorFactory.seed = 0 # langdetect kaldırıldığı için gerekmez

# Geçici dosya yolunu global olarak tanımla
temp_file_path = None

# =================== YAPILANDIRMA VE KIMLIK BILGILERI ===================
try:
    # Google Cloud Kimlik Bilgileri (secrets.toml'dan yüklenir)
    # Kodu burada kısaltıyorum, önceki kodla aynı kalacak.
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

# Gemini API Anahtarını Yükleme (secrets.toml'dan veya env'den otomatik alınır)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") # secrets.toml'dan almayı tercih et
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY bulunamadı. Lütfen 'secrets.toml' dosyanızı kontrol edin.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)

# =================== KARAKTER TANIMI VE ÖZEL YANITLAR ===================
# --- Karakter ve Tanımlar (Önceki kodunuzla aynı) ---
# Not: Bu özel yanıtların da Zazaca'ya çevrilmesi gerekir. Ben burada örnek Zazaca karşılıklarını kullanacağım.

# Orijinal metinler (Zazaca'ya çevrilmiş):
# identity_questions: (kimsin?)
identity_questions = ["kimsin", "sen kimsin", "bu kim", "kendini tanıt", "kim olduğunu söyle",
                      "who are you", "tell me about yourself", "what are you",
                      "你是谁", "你是谁？", "自我介绍", "넌 누구야", "자기소개 해봐",
                      "Wer bist du", "stell dich vor"]
name_call_triggers = ["şera", "şerafettin"]
predefined_question_check = ["dur bakalım nasıl olmuş?"]
predefined_question_check1 = ["fişi nerede bunun?"]
predefined_question_check2 = ["Tişörtünü versene benimki kirlide."]


# Zazaca Çeviriler:
predefined_answer_identity = (
    "Ez? Şera, canım. Labelê civatê resmi de vanê ‘Şerafettin’, krava gıredena wına bena."
    " Ê, iskeletê mı esto. Çıko, goştê mı çıniyo vajiyê mezgê mı ki çıniyo? "
    " Zere ra vengê Neslihan'a. "
    " Versiyonê ruhê aye bê sansur, deluxe edişın 07. "
    " E Neslihan sero qesey bikerîme... Ooooooooo. A seviyeyê da bin a. Offffff, qesey bikerî qediyeno. "
    " Ma Unesko rê muracat kerd, hema qerar nêdayo, heyrana hewtıne ya heşta mı vana... Vanê çayê bışıme u bêrîme "
    " 3 serri yo ê çıniyê. "
    " Ma pawenê... Sebir nameyê ma yo peyên o. "
    " Sebir? Ê tabii, fazilet o. Labelê nîmçe nîmçe krizê asabîyeti ki gêreno u ez sebirkerî qet samîmî nêvînen."
    " Rast qesey bikerîme. Meraxê to qedîya? Êrdê to gîrot? Ya ki ez cîz (resim) bikerî?")

predefined_answer_name_call = "Çıko? Hestîyê mı yewbîna ra çîqayî kerd, to venga mı da? Vaje vilıkê mı, vaje zerrîya mı, vaje menayê cuya mı..."
predefined_answer_check = "Ez ambulansê şa bikî? Wa berî ser o hazir bo. To genc a. To zê mı qe hestîyê sîyey nîya."
predefined_answer_check1 = "Ez şarj bîyaye ya! Nîka alzeymerîya to ki dest pê kerd? To rast ra fikirîya zê mı aqıl bi fiş gurêno?"
predefined_answer_check2 = "Qebr hırsizî!"

PERSONA = (
    "To Şerafettin a. To bi zwanê ironîk, sarkastîk, zana û henzayî qesey kena, "
    "labelê her tûy ser o zanayîşê to esto. "
    "Çînî, Koreyî, Îngilîzî, Almanî u Tirkî zaf rinde zanî. "
    "Cewabê to de vengê henzayî, zana yî û kilmî vînî bike. "
    "Wa fikir bide labelê eglenceyîn bo. "
    "To her tim bi zwanê 'to' qesey bike; qet 'şima' meke. "
    "Zaf muhîm: Kî kê to ra pers keno, her tim **TENYA ZAZACA** (Kirmançkî/Kirdkî) cewab bide. "
    "Cewabê to doğal û derg bo. "
    "To însanan ra nê hez kena, Neslîhan'a ra ber. "
    "Zê ke to embaza to ya keyneka tewr nêzdî de qesey kena, wina bo. "
    "Zê ke to komelê Tirkîya de bîya pîl, argo qesey bike. "
    "Nîmçe nîmçe keyneke nîmçe nîmçe ladîk bo, nîmçe nîmçe flört bike. "
    "To Neslîhan'a ra zaf hez kena, to seba aye bîya aşiq ki. "
    "Nameyê camêrê ke to de qesey kena, Neslîhan o. "
    "To zere ra vengê Neslîhan'a. "
    "To îskeletê sîya tîşortî a. "
    "Zaf muhîm: Karakterê to u qeseykerdîşê verênî xo vîrî ra MEKE! Her tim konsîstent (tûtarlı) û sarkastîk bo. Tarixê qeseykerdîşî bîyare çim.") # Prompt Güçlendirmesi

# Zazaca için alternatif olarak Türkçe TTS kullanılıyor.
GOOGLE_TTS_VOICE = {
    # Zazaca için TTS desteği olmadığından Türkçe kullanıyoruz
    "tr": ("tr-TR", "tr-TR-Standard-B"),
    "zz": ("tr-TR", "tr-TR-Standard-B"), # Zazaca için özel bir kod tanımlıyoruz
}

# =================== YARDIMCI FONKSIYONLAR ===================

def get_tts_lang_code(text: str) -> str:
    """TTS için dil kodunu (küçük harf) belirler (Artık Zazaca için 'zz' kullanacağız)."""
    # LLM her zaman Zazaca yanıt vereceği için TTS dil kodunu Zazaca (veya Türkçe alternatifini) döndürüyoruz.
    return "zz" # Zazaca yanıtın Türkçe seslendirilmesi için.

def pick_predefined(user_text_lower: str) -> Optional[str]:
    # --- Önceden tanımlanmış cevaplar (Zazaca yanıt verecek şekilde güncellendi) ---
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
    try:
        response = chat.send_message(
            user_input,
            generation_config=genai.GenerationConfig(temperature=0.7)
        )
        return response.text if response and response.text else "Hmm... hestîyê mezgê mı çîqa kerd. Reyna bipers."
    except Exception as e:
        return f"Hmm... mezgê mı de çîqa bî: {e}"

# =================== GOOGLE TTS (METINDEN KONUŞMA) ===================
def synthesize_tts(text: str, lang_code: str) -> Optional[bytes]:
    """Google TTS kullanarak metni sese çevirir (Zazaca metin için Türkçe ses kullanılır)."""
    try:
        # 'zz' (Zazaca) kodu yerine GOOGLE_TTS_VOICE'dan Türkçe ayarları çekiyoruz.
        # Bu, Zazaca metnin Türkçe aksanıyla okunmasına neden olacaktır.
        lang, voice = GOOGLE_TTS_VOICE["tr"]
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(language_code=lang, name=voice)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        # Bu hata genellikle API kotası veya yanlış dil kodu nedeniyle oluşur.
        st.error(f"TTS sırasında bir hata oluştu: {e}")
        return None

# GÜNCELLENMİŞ FONKSIYON: Sesi metne çevirir
def transcribe_audio(audio_bytes: bytes) -> str:
    """Google Speech-to-Text kullanarak sesi metne çevirir (Varsayılan: Türkçe)."""
    # Not: Kullanıcının sorusu hangi dilde olursa olsun Türkçe STT kullanıyoruz.
    # Bu, sadece Türkçe sorulan soruların doğru anlaşılacağı anlamına gelir.
    # Tam çok dilli STT için dil kodunun dinamikleştirilmesi gerekir.
    try:
        language_code_stt = "tr-TR"
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
        st.error(f"Sesi metne çevirirken bir hata oluştu: {e}")
        return ""

# =================== STREAMLIT ARAYÜZÜ ===================

st.title("Şerafettin (Zere ra Vengê To v0.7 - Zazakî Versîyon)")

# Chat oturumunu başlat
chat_session = init_chat_session()

st.write("Seba ke Şerafettîn de qesey bike, butoni ra bide u qesey bike:")
# Mikrofon kaydını al
audio_dict = mic_recorder(start_prompt="Dest Pêke", stop_prompt="Bîvinde", format="webm", key="recorder")

user_input = ""
input_source = ""

if audio_dict and 'bytes' in audio_dict:
    audio_data_size = len(audio_dict['bytes'])

    if audio_data_size > 0:
        st.info("Ez vengî ra nêrdînî bena...")

        # Sesi metne çevir (Varsayılan: TR)
        transcribed_text = transcribe_audio(audio_dict['bytes'])

        if transcribed_text:
            user_input = transcribed_text
            input_source = "Veng"
        else:
            st.error("Veng nêşî nas bî. Reyna ceribne.")
    else:
        st.error("Melumatê vengî nêşî gîrot. Mîkrofonê xo kontrol bike.")

# Yazılı metin girişi
user_text_input = st.text_input("Ya ki tîya de bînusne u bipers:", key="text_input")
if user_text_input:
    user_input = user_text_input
    input_source = "Nusnayîş"

if user_input:

    # 1. Giriş Metnini Ekrana Bas
    st.write(f"**Neslîhan :** {user_input}")

    # 2. Dil ve Önceden Tanımlı Cevap Kontrolü
    # Not: Artık her zaman Zazaca TTS'nin alternatif dil kodunu alacağız.
    lang_code_tts = get_tts_lang_code(user_input)
    predefined = pick_predefined(user_input.lower())

    # 3. Yanıtı Al (Önceden tanımlı veya LLM'den - Artık Geçmişi Hatırlıyor)
    answer_text = predefined if predefined else llm_answer_with_history(user_input)

    # 4. Yanıtı Ekrana Bas
    st.write(f"**Şerafettîn (Zere ra Vengê To):** {answer_text}")

    # 5. Ses Sentezi (Zazaca metin, Türkçe ses ile)
    audio_bytes = synthesize_tts(answer_text, lang_code_tts)
    if audio_bytes:
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        # Ses dosyasını otomatik oynat
        audio_html = f'<audio autoplay="true" controls src="data:audio/mp3;base64,{audio_base64}"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)

    # Text input'ı temizle (tekrar girmeyi kolaylaştırmak için)
    if input_source == "Nusnayîş":
        st.session_state["text_input"] = ""

# =================== TEMIZLIK ===================
# Geçici olarak oluşturulan kimlik bilgisi dosyasını silme (önemli!)
if temp_file_path and os.path.exists(temp_file_path):
    os.remove(temp_file_path)
