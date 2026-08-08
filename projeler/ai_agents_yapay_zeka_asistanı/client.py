"""
*fast api üzerinden çalışan gemini çok araçlı ai agent için istemci oluşturulur.
*yapay zeka ajanını kullanmak için istemci oluşturma
"""

import requests
import json

API_URL= "http://127.0.0.1:8000/ask"

#user_id tanımla, aynı kullanıcıya sorgu atmak için
USER_ID= "nur"

def send_message(message: str):
    payload= {
        "user_id": USER_ID,
        "message": message
    }

    try:
        response= requests.post(API_URL, json= payload)
        response.raise_for_status() #hata varsa (status != 200)
        data= response.json()
        print(f"soru: {message}")
        print(f"cevap: {data.get("response")}")

    except Exception as e:
        print(f"hata oluştu: {e}")


if __name__ == "__main__":
    print("AI ajanımız başladı.")

    while True:
        user_input= input("siz: ")
        send_message(user_input)



# uvicorn fast_api:app --reload