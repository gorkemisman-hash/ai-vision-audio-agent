import os
import google.generativeai as genai
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# .env dosyasındaki değişkenleri yükler
load_dotenv()

# Gemini API anahtarını ayarlar
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# FastAPI uygulamasını başlatır
app = FastAPI(title="Multimodal AI Agent API (Vision, Audio, Docs)")

# --- Veri Modelleri ---
class ImageAgentRequest(BaseModel):
    image_url: str
    prompt: str

class AudioAgentRequest(BaseModel):
    audio_url: str
    prompt: str

class DocumentAgentRequest(BaseModel):
    doc_url: str
    prompt: str

# --- GÜVENLİK FİLTRELERİ (Estetik/Tıbbi Görseller İçin Kapatıldı) ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- API Endpoint'leri ---

@app.post("/analyze-image")
def analyze_image(request: ImageAgentRequest):
    """Görsel URL'sini alıp Gemini ile analiz eder."""
    image_url = request.image_url
    user_prompt = request.prompt

    if not image_url or not user_prompt:
        raise HTTPException(status_code=400, detail="image_url ve prompt alanları zorunludur.")

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        image_response = requests.get(image_url, headers=headers)
        image_response.raise_for_status()
        image_bytes = image_response.content

        model = genai.GenerativeModel('gemini-2.5-flash')
        # MIME type otomatik alınır, alınamazsa jpeg varsayılır
        mime_type = image_response.headers.get('Content-Type', 'image/jpeg')
        image_part = {'mime_type': mime_type, 'data': image_bytes}
        
        # Güvenlik ayarları ile modeli çağırıyoruz
        response = model.generate_content(
            [user_prompt, image_part],
            safety_settings=safety_settings
        )

        # Gemini inat edip görseli yine de bloklarsa sistem çökmesin diye kontrol ediyoruz
        if not response.candidates or not response.parts:
            return {"response": "Yapay zeka sistemi, bu görseli tıbbi güvenlik filtreleri nedeniyle analiz edemedi. Lütfen işlemi manuel olarak devam ettirin."}

        return {"response": response.text}

    except Exception as e:
        print(f"HATA (GÖRSEL ANALİZİ): {e}")
        # Sunucu 500 hatası vermek yerine Make.com'a kontrollü bir hata döner
        return {"response": "Görsel analiz edilemedi. Sistem görseli işleyemedi veya sunucu hatasına takıldı."}

@app.post("/analyze-audio")
def analyze_audio(request: AudioAgentRequest):
    """Ses URL'sini alıp Gemini ile analiz eder."""
    audio_url = request.audio_url
    user_prompt = request.prompt

    if not audio_url or not user_prompt:
        raise HTTPException(status_code=400, detail="audio_url ve prompt alanları zorunludur.")

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        audio_response = requests.get(audio_url, headers=headers)
        audio_response.raise_for_status()
        audio_bytes = audio_response.content

        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Instagram'dan mp4, WhatsApp'tan ogg gelebilir. Dinamik MIME type algılama eklendi.
        mime_type = audio_response.headers.get('Content-Type', 'audio/ogg')
        audio_part = {'mime_type': mime_type, 'data': audio_bytes}

        response = model.generate_content([user_prompt, audio_part])
        return {"response": response.text}

    except Exception as e:
        print(f"HATA (SES ANALİZİ): {e}")
        # 500 verip çökmek yerine asistanın "anlayamadım" demesi için kontrollü dönüş
        return {"response": "[DİL]: Belirsiz [DEŞİFRE]: (Ses kaydı teknik bir nedenden dolayı işlenemedi) [ÖZET]: Ses kaydı açılamadı."}

@app.post("/analyze-document")
def analyze_document(request: DocumentAgentRequest):
    """Doküman URL'sini alıp Gemini ile analiz eder."""
    doc_url = request.doc_url
    user_prompt = request.prompt

    if not doc_url or not user_prompt:
        raise HTTPException(status_code=400, detail="doc_url ve prompt alanları zorunludur.")

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        doc_response = requests.get(doc_url, headers=headers)
        doc_response.raise_for_status()
        doc_bytes = doc_response.content

        model = genai.GenerativeModel('gemini-2.5-flash')
        document_part = {'mime_type': doc_response.headers.get('Content-Type', 'application/pdf'), 'data': doc_bytes}

        response = model.generate_content([user_prompt, document_part])
        return {"response": response.text}

    except Exception as e:
        print(f"HATA (DOKÜMAN ANALİZİ): {e}")
        return {"response": "Doküman analiz edilemedi, lütfen tekrar deneyin."}