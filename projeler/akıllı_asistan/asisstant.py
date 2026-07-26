import os #ortam değişkenleri ve dosya yolu
import requests #http istekleri yapmak için
from dotenv import load_dotenv #ortam değişkenlerini yüklemek


#.env dosyasından ortam değişkenlerini yükleme
load_dotenv()

api_key= os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY .env dosyasında tanımlı değil.")


#gemini 2.0 flash modeline ait api url
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={api_key}"

#api çağrısı için gerekli http başlıkları
headers= {
    "Content-Type": "application/json", #json formatında veri gönderilecek
    "X-Goog-Api-Key": api_key #yetkilendirme için api anahtarı
}


#gemini api'sine prompt gönderip yanıt alan fonksiyon
def get_gemini_response(prompt: str) -> str:

    #api'ye gönderilecek json yapısı
    payload= {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt #kullanıcıdan gelen mesajı içeren bölüm
                    } 
                ]
            }
        ]
    }

    #gemini api ye http post isteği gönderme
    response= requests.post(url, headers= headers, json= payload)

    #istek başarılı ise (http 200)
    if response.status_code == 200:
        try:
            result= response.json() #json formatındaki yanıtı sözlüğe çevir
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"yanıt hatası: {e}"
    else:
        return f"api hatasi {response.status_code}: {response.text}"


def detect_intent():
    pass


if __name__ == "__main__":
    user_input= input("Kullanıcı Sorusu: ") #terminal üzerinden girdi almak
    yanit= get_gemini_response(user_input)
    print(f"Akıllı Asistan Yanıtı: {yanit}")



