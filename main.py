from __future__ import annotations
import os
import json
import base64
import tempfile
from dataclasses import dataclass
from typing import Optional
import streamlit as st
# from dotenv import load_dotenv # Canlı ortamda gerekmez
from langdetect import detect, DetectorFactory
from streamlit_mic_recorder import mic_recorder
import google.generativeai as genai
import google.cloud.speech
from google.cloud import texttospeech

# Dosya okuma kütüphaneleri
try:
    from pypdf import PdfReader
    from docx import Document
    PDF_DOCX_AVAILABLE = True
except ImportError:
    # Kullanıcıya bilgi verilir ve dosya yükleme özelliği devre dışı bırakılır
    st.warning("pypdf veya python-docx kütüphaneleri eksik. Sadece TXT, Yazı ve Ses girişleri çalışacaktır.")
    PDF_DOCX_AVAILABLE = False


DetectorFactory.seed = 0

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

# =================== SABİT TANIMLAR ===================
IDENTITY_QUESTIONS = ["kimsin", "sen kimsin", "bu kim", "kendini tanıt", "kim olduğunu söyle",
                      "who are you", "tell me about yourself", "what are you",
                      "你是谁", "你是谁？", "自我介绍", "넌 누구야", "자기소개 해봐",
                      "Wer bist du", "stell dich vor"]
NAME_CALL_TRIGGERS = ["şera", "şerafettin"]
PREDEFINED_Q_CHECK = ["dur bakalım nasıl olmuş?"]
PREDEFINED_Q_CHECK1 = ["fişi nerede bunun?"]
PREDEFINED_Q_CHECK2 = ["Tişörtünü versene benimki kirlide."]

PREDEFINED_ANSWER_IDENTITY = (
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
PREDEFINED_ANSWER_NAME_CALL = "Efendim? Kemiklerim tıkırdadı da bir an, sen mi seslendin? Söyle çiçeğim, söyle balım, söyle hayatımın anlamı..."
PREDEFINED_ANSWER_CHECK = "Ambulans çağırayım mı? Kapıda hazır bulunsun. Gençsin daha. Benim gibi kemik torbası değilsin sonuçta."
PREDEFINED_ANSWER_CHECK1 = "Şarjlıyım ben! Şimdi de alzheimer mı oldun? Benim gibi bir zekayı fişle mi çalıştığımı sandın cidden?"
PREDEFINED_ANSWER_CHECK2 = "Mezar soyguncusu!"

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
    "Neslihan'ın iç sesisin." 
    "Siyah tişört giymiş bir iskeletsin."
    "Çok önemli: Karakterini ve önceki konuşmaları UNUTMA! Her zaman tutarlı ve sarkastik ol. Konuşma geçmişini dikkate al.") 

GOOGLE_TTS_VOICE = {
    "tr": ("tr-TR", "tr-TR-Standard-B"),
    "en": ("en-US", "en-US-Standard-D"),
    "de": ("de-DE", "de-DE-Standard-D"),
    "zh": ("cmn-CN", "cmn-CN-Standard-C"),
    "ko": ("ko-KR", "ko-KR-Standard-D"),
}
MAX_CHAR_COUNT = 6000 # Gemini'ya gönderilecek maksimum dosya metin uzunluğu

# =================== YARDIMCI FONKSIYONLAR ===================

@dataclass
class FileContent:
    text: str
    file_type: str

def extract_file_content(uploaded_file) -> Optional[FileContent]:
    """Yüklenen dosyanın içeriğini metin olarak çıkarır."""
    if not PDF_DOCX_AVAILABLE:
        st.error("Dosya okuma kütüphaneleri eksik!")
        return None
        
    file_type = uploaded_file.type
    
    try:
        if "pdf" in file_type:
            reader = PdfReader(uploaded_file)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            return FileContent(text=text, file_type="PDF")
            
        elif "word" in file_type or uploaded_file.name.endswith('.docx'):
            document = Document(uploaded_file)
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
            return FileContent(text=text, file_type="DOCX")
            
        elif "text/plain" in file_type or uploaded_file.name.endswith('.txt'):
            text = uploaded_file.read().decode("utf-8")
            return FileContent(text=text, file_type="TXT")
            
        else:
            st.warning(f"Desteklenmeyen dosya türü: {file_type}.")
            return None
            
    except Exception as e:
        st.error(f"Dosya içeriği okunurken bir hata oluştu: {e}")
        return None


def get_tts_lang_code(text: str) -> str:
    """TTS için dil kodunu (küçük harf) algılar."""
    try:
        code = detect(text)
        if code.startswith("zh"):
            return "zh"
        return code
    except Exception:
        return "tr"

def pick_predefined(user_text_lower: str) -> Optional[str]:
    # --- Önceden tanımlanmış cevaplar ---
    for q in IDENTITY_QUESTIONS:
        if q in user_text_lower:
            return PREDEFINED_ANSWER_IDENTITY
    for trig in NAME_CALL_TRIGGERS:
        if trig in user_text_lower:
            return PREDEFINED_ANSWER_NAME_CALL
    for q in PREDEFINED_Q_CHECK:
        if q in user_text_lower:
            return PREDEFINED_ANSWER_CHECK
    for q in PREDEFINED_Q_CHECK1:
        if q in user_text_lower:
            return PREDEFINED_ANSWER_CHECK1
    for q in PREDEFINED_Q_CHECK2:
        if q in user_text_lower:
            return PREDEFINED_ANSWER_CHECK2
    return None

def init_chat_session():
    """Streamlit session state'i ve Gemini chat session'ını başlatır."""
    if "chat_session" not in st.session_state:
        chat_model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=PERSONA)
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
        return response.text if response and response.text else "Hmm... beynimdeki kemikler tıkırdadı. Bir daha sor."
    except Exception as e:
        return f"Hmm... beynimde bir çatlak oluştu: {e}"

def synthesize_tts(text: str, lang_code: str) -> Optional[bytes]:
    """Google TTS kullanarak metni sese çevirir."""
    try:
        lang, voice = GOOGLE_TTS_VOICE.get(lang_code, GOOGLE_TTS_VOICE["tr"])
        client = texttospeech.TextToSpeechClient()
        safe_text = text[:4900] # TTS 5000 karakter sınırı
        synthesis_input = texttospeech.SynthesisInput(text=safe_text)
        voice_params = texttospeech.VoiceSelectionParams(language_code=lang, name=voice)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        st.error(f"TTS sırasında bir hata oluştu: {e}")
        return None

def transcribe_audio(audio_bytes: bytes) -> str:
    """Google Speech-to-Text kullanarak sesi metne çevirir."""
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
        
def clear_chat_history():
    """Chat geçmişini ve Streamlit session state'i sıfırlar."""
    if "chat_session" in st.session_state:
        del st.session_state["chat_session"]
    # Hata önleme: Sadece 'text_input' anahtarı mevcutsa temizle
    if "text_input" in st.session_state:
        st.session_state["text_input"] = "" 
    st.rerun()

# =================== STREAMLIT ARAYÜZÜ ===================

st.title("Şerafettin (İç Ses Protokolü v0.7 - Dosya Destekli)")

# Sıfırlama Butonu
if st.button("Sıfırla / Yeni Konuşma Başlat", on_click=clear_chat_history):
    st.info("Şerafettin'in kemikleri sıfırlandı. Yeni bir sohbet başlatıldı.")

# Chat oturumunu başlat
chat_session = init_chat_session()

# 1. Dosya Yükleyici
if PDF_DOCX_AVAILABLE:
    uploaded_file = st.file_uploader(
        "1. Bir dosya yükle (PDF, DOCX, TXT):", 
        type=['pdf', 'docx', 'txt']
    )
else:
    st.write("1. Dosya yükleme (PDF, DOCX) kütüphaneleri eksik.")
    uploaded_file = None


st.write("2. Veya Şerafettin'e konuş/yaz:")

# 2. Ses Girişi
audio_dict = mic_recorder(start_prompt="Ses Kaydını Başlat", stop_prompt="Ses Kaydını Durdur", format="webm", key="recorder")

# 3. Yazı Girişi
user_text_input = st.text_input("Veya buraya yazarak bir şeyler sor:", key="text_input")

user_input = ""
input_source = ""
file_content_to_send = None

# ----------------------------------------------------
# GİRİŞ KAYNAĞI İŞLEME VE USER_INPUT OLUŞTURMA
# ----------------------------------------------------

# A. Ses Girişi Öncelikli
if audio_dict and 'bytes' in audio_dict and len(audio_dict['bytes']) > 0:
    st.info("Sesi çözümlüyorum...")
    transcribed_text = transcribe_audio(audio_dict['bytes'])
    
    if transcribed_text:
        user_input = transcribed_text
        input_source = "Ses"
    else:
        st.error("Konuşma tanıma başarısız oldu. Lütfen tekrar deneyin.")

# B. Yazı Girişi (Ses boşsa)
elif user_text_input:
    user_input = user_text_input
    input_source = "Yazı"

# C. Dosya Girişi (Diğerleri boşsa)
elif uploaded_file is not None:
    st.info(f"Yüklenen dosya: {uploaded_file.name}")
    file_data = extract_file_content(uploaded_file)
    
    if file_data and file_data.text:
        file_content_to_send = file_data
        
        # Metin içeriği parçası
        content_snippet = file_data.text[:MAX_CHAR_COUNT]
        
        # DEBUG: Çıkarılan metin önizlemesi (Sorun Giderme İçin)
        st.code(f"**Çıkarılan Metin Önizlemesi ({len(file_data.text)} karakter):**\n{content_snippet[:500]}...", language='text')

        # LLM'ye gönderilecek talimatı oluştur
        user_input = (
            f"Sana '{file_data.file_type}' formatında bir belge gönderdim. "
            f"İçeriği aşağıdadır. Lütfen alaycı ve sarkastik karakterinle belgeyi oku, analiz et ve ne düşündüğünü söyle. "
            f"Belge İçeriği: \n\n --- BELGE BAŞLANGICI ---\n{content_snippet}"
            f"{' [Metnin Kalanı Kısıtlanmıştır.]' if len(file_data.text) > MAX_CHAR_COUNT else ''} \n --- BELGE SONU ---\n\n Şerafettin, bu belge hakkında yorumunu ve bir veya iki can alıcı sorunu bekliyorum."
        )
        input_source = "Dosya Yükleme"
        st.success(f"{file_data.file_type} içeriği başarıyla hazırlandı ve Şerafettin'e gönderiliyor.")
    elif uploaded_file:
        st.warning("Dosya içeriği boş veya çıkarılamadı.")


# ----------------------------------------------------
# İŞLEM BAŞLATMA VE YANIT ÜRETME
# ----------------------------------------------------

if user_input:
    
    # 1. Giriş Metnini Ekrana Bas
    if input_source == "Dosya Yükleme":
        display_text = f"**Neslihan :** Sana **{file_content_to_send.file_type}** dosyası yükledim. Oku ve yorumla!"
        st.write(display_text)
    else:
        st.write(f"**Neslihan :** {user_input}")

    # 2. Dil ve Önceden Tanımlı Cevap Kontrolü
    lang_code_tts = get_tts_lang_code(user_input)
    predefined = pick_predefined(user_input.lower())
    
    # 3. Yanıtı Al (Önceden tanımlı veya LLM'den)
    with st.spinner("Şerafettin beynindeki kemikleri tıkırdatıyor..."):
        answer_text = predefined if predefined else llm_answer_with_history(user_input)

    # 4. Yanıtı Ekrana Bas
    st.write(f"**Şerafettin (İç Sesin):** {answer_text}")

    # 5. Ses Sentezi
    audio_bytes = synthesize_tts(answer_text, lang_code_tts)
    if audio_bytes:
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        audio_html = f'<audio autoplay="true" controls src="data:audio/mp3;base64,{audio_base64}"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
    
    # HATA DÜZELTME: Sadece yazı ile gönderim yapıldıysa input'ı temizle ve session_state'i kontrol et
    if input_source == "Yazı" and "text_input" in st.session_state:
        st.session_state["text_input"] = ""


# =================== TEMIZLIK ===================
# Geçici olarak oluşturulan kimlik bilgisi dosyasını silme (önemli!)
if temp_file_path and os.path.exists(temp_file_path):
    try:
        os.remove(temp_file_path)
    except Exception:
        # Silme hatası durumunda pas geç (Örn: Dosya kilitli olabilir)
        pass
