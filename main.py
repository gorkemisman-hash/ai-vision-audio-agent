import os
import google.generativeai as genai
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

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
        image_part = {'mime_type': image_response.headers.get('Content-Type', 'image/jpeg'), 'data': image_bytes}
        
        response = model.generate_content([user_prompt, image_part])
        return {"response": response.text}

    except Exception as e:
        print(f"HATA (GÖRSEL ANALİZİ): {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        audio_part = {'mime_type': audio_response.headers.get('Content-Type', 'audio/ogg'), 'data': audio_bytes}

        response = model.generate_content([user_prompt, audio_part])
        return {"response": response.text}

    except Exception as e:
        print(f"HATA (SES ANALİZİ): {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))

