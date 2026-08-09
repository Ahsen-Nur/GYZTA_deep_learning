import os
from dotenv import load_dotenv
from google.generativeai.generative_models import GenerativeModel

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

model = GenerativeModel(model_name="gemini-1.5-flash")

#soru sorma ajanı
def generate_followup_questions(person, task, current_time, previous_responses=None):
    """
    gemini kişi, görev, zaman ve geçmiş yanıtları alarak en uygun soruyu üretir

    person: ekip üyesi
    task: görev
    current_time: şuanki zaman
    previous_responces: kişinin bu göreve daha önce verdiği yanıtlar
    """

    history = ""
    if previous_responses:
        for item in previous_responses:
            history += f"saat {item['time']}: {item['response']}\n"

    #gemini gönderilecek prompt
    prompt = f"""
                şu anda saat {current_time}.
                sen bir proje yöneticisisin.

                görev: "{task}"
                kişi: {person}

                bu kişiye bu görev daha önce verildi.
                şimdiye kadar verdiği cevaplar:
                {history if history else "henüz cevap yok"}

                lütfen {person}'a doğrudan hitap ederek görevle ilgili ne durumda olduğunu soran
                net ve kısa bir soru yaz.

                soru şunları içermeli:
                    -kişinin ismiyle hitap et
                    -görevin ne olduğu açıkça tekrar et
                    -görevin tamamlanma durumu ya da üzerinde çalışılıp çalışılmadığı sorgulansın
                    -sadece doğrudan bir soru cümlesi döndür, başka açıklama yazma

    """

    response = model.generate_content(prompt, generation_config={"temperature": 0.7})
    return response.text.strip()

#ekip üyesi cevabına göre taskların tamamlanıp tamamlanmadığına karar verir
def is_task_completed(person, task, responces, current_time):
    """
    AI yöneticisi görevin tamamlanıp tamamlanmadığını anlar
    yalnızca 3 cevaptan (tamamlandı/ devam ediyor/ yapılmadı) birini return eder
    """

#rooter yapısı: kullanıcının vermiş olduğu cevaplar doğrultusunda belirlenen etiketleri seçen bir sınıflandırma mekanizaması
    history = ""
    for item in responces:
        history += f"saat: {item['time']}: {item['response']}\n"

    #prompt engineering
    prompt = f"""
            saat: {current_time}
            kişi: {person}
            görev: {task}

            bu görevle ilgili şimdiye kadar {person} tarafından verilen cevaplar:
            {history}

            lütfen sadece tek bir kelime ile cevap ver:
            -tamamlandı
            -devam ediyor
            -yapılmadı

            yalnızca bu 3 kelimeden birini döndür. açıklama yapma

    """

    responce = model.generate_content(prompt, generation_config={"temperature": 0})
    return responce.text.strip().lower()


if __name__ == "__main__":

    example_history = [
        {"time": "12:02", "response": "başladım ama eksik bir şeyler var."},
        {"time": "12:04", "response": "veritabanı bağlantısını henüz kurmadım."}
    ]

    soru = generate_followup_questions(
        person="Yılmaz",
        task="Veritabanı bağlantısını ayağa kaldır ve temel kullanıcı (user) tablosunu oluştur.",
        current_time="1900-01-01 12:02:00",
        previous_responses=example_history
    )

    print(f"AI proje yöneticisinin sorusu: {soru}")

    durum = is_task_completed(
        person= "Yılmaz",
        task= "Veritabanı bağlantısını ayağa kaldır ve temel kullanıcı (user) tablosunu oluştur.",
        responces=[{"time": "12.02", "response": "ben bu taskı tamamladım, bence gayet güzel oldu, testleri de yaptım, çalışıyor."}],
        current_time= "1900-01-01 12:02:00"
    )

    print(f"AI proje yöneticisi durum değerlendirmesi. Task: {durum}")

