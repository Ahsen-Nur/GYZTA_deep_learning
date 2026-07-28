"""
amaç:
    *Gemini ile akıllı asistan projesi: notlar ve etkinlikler için akıllı asistan kullanılacak
    *Google Gemini API kullanılacak
    *Kural tabanlı notlar ve etkinlikler oluşturulacak
    *Doğal dilde notlar ve ve etkinlikler ile konuşabilme(chatbot)
    *Kısaca asistan notlara ve etkinliklere erişim sağlayarak özetleme, bilgi çıkarma veya takvim oluşturma gibi görevleri gerçekleştirecek

model:
    *google gemini-3.1-flash modeli

planlama:
    *assistant.py: gemini ile chatbot oluşturma
    *database.py: sqlite içinde notlar ve etkinlikler depolanacak
    *main.py: bileşenleri bir araya getirme

pip install requests python-dotenv
"""

#assistant.py dosyasından gemini api yanıtını alan fonk. çağır
from asisstant import get_gemini_response, detect_intent

#database.py dosyasından veritabanı işlemleri için gerekli fonk. çağır
from database import initialize_db, add_note, add_event, get_notes, get_events

initialize_db()

print("Akıllı Asistana Hoş Geldiniz.")
print("Komutlar: not ekle | etkinlik ekle | notları göster| etkinlikleri göster | sohbet et| çıkış")

#kullanıcıdan sürekli komut almak için sonsuz döngü
while True:
    komut= input("Komut girin: ").strip().lower() #komutu al, boşlukarı kırp, küçük harfe çevir

    if komut == "not ekle":
        content= input("Not içeriği nedir?")
        add_note(content)
        print("not başarıyla kaydedildi.")

    elif komut == "etkinlik ekle":
        event= input("Etkinlik içeriği nedir?")
        event_date= input("Etkinlik tarihi nedir?")
        add_event(event, event_date)

    elif komut == "notları göster":
        notes= get_notes()
        if notes:
            print("Kaydedilmiş notlar: ")
            for content, created_at in notes:
                print(f"\t- [{created_at}]{content}")
        else:
            print("Henüz hiçbir not eklenmedi")

    elif komut == "etkinlikleri göster":
        events= get_events()
        if events:
            print("Etkinlikler: ")
            for events, event_date in events:
                print(f"\t- {event_date}:{event}")
        else:
            print("Henüz hiçbir etkinlik eklenmedi")

    elif komut == "sohbet et":
        message= input("Kullanıcı sorusu: ").strip() #kullanıcıdan serbest metin alma işlemi
        intent= detect_intent(message) #input("Kullanıcı niyeti: ") #kullanıcı niyeti (not özeti veya etkinlik özeti veya günlük konuşma)

        if intent == "not_ozet":
            notes= get_notes()
            if not notes:
                print("Henüz özetlenecek bir not bulunamadı.")
                continue

            all_notes_text= "\n".join([f"- {note[0]}" for note in notes]) #tüm notları birleştir ve text haline getir
            prompt= f"Aşağıda bulunan notlar doğrultusunda kullanıcı sorusunu yanıtlar mısın? Eğer notlarda kullanıcı sorusuna cevap yoksa bilmediğini kibarca belirt. notlar: {all_notes_text}, kullanıcı sorusu: {message}"
            response= get_gemini_response(prompt)

            print("Notlar hakkında: ")
            print(response)

        elif intent == "etkinlik_ozet":
            events= get_events()
            if not events:
                print("Henüz özetlenecek bir etkinlik bulunamadı.")
                continue

            all_events_text= "\n".join([f"- {event[1]}: {event[0]}" for event in events]) 
            prompt= f"Aşağıdaki takvime göre kullanıcı sorusunu yanıtlar mısın? Takvim: {all_events_text}, kullanıcı sorusu: {message}"
            response= get_gemini_response(prompt)

            print("Etkinlikler hakkında: ")
            print(response)

        else: #normal
            reply= get_gemini_response(message)
            print(f"Akıllı asistan: {reply}")

    elif komut == "çıkış":
        break

    else:
        print("hatalı komut")
    