"""
terminal üzerinden fast api web sunucu ile sohbet gerçekleştir
(post request atarak)
"""

import requests #http istekleri yapmak için kullanılan kütüphane


#api adresi
API_URL= "http://127.0.0.1:8000/chat" #fast api sunucusunun çalıştığı adres ve endpoint

#başlangıçta kullanılan bilgileri al
name= input("isim: ")
age= input("yaş: ")

print("sohbet başladı. çıkmak için q ya basın")

#kullanıcıdan mesajı alıp sunucuya gönderen döngü
while True:

    user_msg= input(f"{name}: ") #kullanıcıdan mesajı al

    if user_msg.lower == "q":
        print("program sonlandırıldı.")
        break

    #API ye gönderilecek veri paketi
    payload= {
        "name": name,
        "age": age,
        "message": user_msg
    }


    try:
        #fast api sunucusuna post request atma, timeout=bekleme süresi
        response= requests.post(API_URL, json=payload, timeout=2)

        if response.status_code == 200: #eğer istek başarılıysa
            print(f"doktor asistanı: {response.json()["response"]}")
        else:
            print("hata", response.status_code, response.text)
    
    except requests.exceptions.RequestException as e:
        print("bağlantı hatası")


