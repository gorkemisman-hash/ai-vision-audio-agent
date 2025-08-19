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
app = FastAPI(title="AI Vision & Audio Agent API")

# Dışarıdan gelecek görsel isteğinin modelini tanımlar
class AgentRequest(BaseModel):
    image_url: str
    prompt: str

# Dışarıdan gelecek ses isteğinin modelini tanımlar
class AudioAgentRequest(BaseModel):
    audio_url: str
    prompt: str

# /analyze-image adresine POST isteği geldiğinde bu fonksiyon çalışır
@app.post("/analyze-image")
def analyze_image(request: AgentRequest):
    image_url = request.image_url
    user_prompt = request.prompt

    if not image_url or not user_prompt:
        raise HTTPException(status_code=400, detail="image_url ve prompt alanları zorunludur.")

    try:
        # Kendimizi normal bir tarayıcı gibi tanıtmak için headers ekliyoruz
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        # isteği bu header ile birlikte gönderiyoruz
        image_response = requests.get(image_url, headers=headers)
        image_response.raise_for_status() # Hata varsa yakala (4xx, 5xx)
        image_bytes = image_response.content

        # Gemini Vision modelini hazırlar
        model = genai.GenerativeModel('gemini-1.5-pro-latest')

        # Gemini'a göndereceğimiz içeriği oluşturur
        content = [
            user_prompt,
            {
                'mime_type': image_response.headers.get('Content-Type', 'image/jpeg'),
                'data': image_bytes
            }
        ]
        
        # API'ye isteği gönderir
        response = model.generate_content(content)
        
        # Sonucu JSON olarak geri döndürür
        return {"response": response.text}

    except requests.exceptions.RequestException as e:
        print(f"HATA OLUŞTU (GÖRSEL İNDİRME): {e}") 
        raise HTTPException(status_code=500, detail=f"Görsel indirilirken hata oluştu: {e}")
    except Exception as e:
        print(f"HATA OLUŞTU (GENEL): {e}") 
        raise HTTPException(status_code=500, detail=f"Gemini API hatası: {e}")


# --- YENİ EKLENEN SES ANALİZİ FONKSİYONU ---
@app.post("/analyze-audio")
def analyze_audio(request: AudioAgentRequest):
    audio_url = request.audio_url
    user_prompt = request.prompt

    if not audio_url or not user_prompt:
        raise HTTPException(status_code=400, detail="audio_url ve prompt alanları zorunludur.")

    try:
        # Sesi URL'den indir
        print("Ses dosyası indiriliyor...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        audio_response = requests.get(audio_url, headers=headers)
        audio_response.raise_for_status()
        audio_bytes = audio_response.content
        print("İndirme tamamlandı.")

        # Gemini modelini hazırla
        model = genai.GenerativeModel('gemini-1.5-pro')

        # Gemini'a gönderilecek içeriği oluştur
        audio_part = {
            # WhatsApp sesli notları genellikle .ogg formatındadır, ama ManyChat'in linki farklı olabilir.
            # MIME type'ı dinamik olarak almak en iyisidir, ancak şimdilik genel bir varsayım yapabiliriz.
            'mime_type': audio_response.headers.get('Content-Type', 'audio/ogg'),
            'data': audio_bytes
        }

        print("Gemini'a ses analizi isteği gönderiliyor...")
        response = model.generate_content([user_prompt, audio_part])
        print("Cevap alındı.")

        return {"response": response.text}

    except Exception as e:
        print(f"HATA OLUŞTU (SES ANALİZİ): {e}")
        raise HTTPException(status_code=500, detail=f"Ses analizi sırasında hata oluştu: {e}")
