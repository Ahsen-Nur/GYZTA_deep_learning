"""
problem tanımı:
    *akıllı dr asistanı: kullanıcının sağlıkla ilgili sorularını anlayan ve yanıtlayabilir
    LLM tabanlı doktor asistanı (chatbot)
    *LLM: Google Gemini API
    *kişiselleştirme: kullanıcının adını ve yaşını bilerek ona göre cevap üretmeli
    *hafıza(memory): mesaj geçmişini hatırlayarak diyaloğu ona göre sürdürmeli

çalışma ortamı:
    *ilk olarak terminal üzerinden çalıştır: doktor_asistani_terminal.py
        -test edebilmek için terminal üzerinden sorgu oluştur
    *FastApi ile bir web servisi oluştur: doktor_asistani_api.py
        -client.py dosyası ile test senaryosu belirle
        -swagger üzerinden test et

veri seti:
    *RAG yapılmayacak, bunun yerine prompt engineering yapılacak

model tanımı:
    *Google Gemini 2.5 flash
    *API üzerinden iletişim kurulacak ve gerçek zamanlı sağlık önerileri alınacak

kütüphaneler:
    *langchain: llm kütüphanesi, prompt yönetimi, memory, chain yapısı
    *fastapi: web api geliştirmek için bir framework
    *uvicorn: fastapi geliştirmek için bir sunucu

kurulumlar:
    pip install langchain-google-genai python-dotenv langchain fastapi uvicorn

"""

import os #os ile .env içindeki api key almak için iletişim
from dotenv import load_dotenv #.env dosyasından api key okumak 
from langchain_google_genai import ChatGoogleGenerativeAI #gemini modelleri için langchain ara birimi
from langchain.memory import ConversationBufferMemory #hafıza sohbet geçmişini saklamak için
from langchain.chains import ConversationChain #hafıza + llm zinciri

import warnings
warnings.filterwarnings("ignore")

#ortam değişkenlerini yükle(api key)
load_dotenv()
api_key= os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("Google api key bulunamadı.")


#llm tanımla
llm= ChatGoogleGenerativeAI(
    model= "gemini-flash-latest",
    temperature= 0.7, #cevap çeşitliliği (0: kesin | 1:yaratıcı)
    google_api_key= api_key

)


#memory tanımla(hafıza)
memory= ConversationBufferMemory(return_message= True)


#llm+memory= zincir oluştur
conversation= ConversationChain(
    llm= llm,
    memory= memory,
    verbose= True #arka plandaki işlem akışını terminalde göster
)


#kullanıcı bilgileri tanımla(kişiselleştirme)
name= input("Adınız: ")
age= input("Yaşınız: ")
history= input("Hasta geçmişi: öksürük, yüksek ateş, bulantı ve kusma.")


#prompt tanımlama
intro= (f"""
    sen bir doktor asistanısın. 
    hasta {name}, {age} yaşında. 
    sağlık geçmişi şu şekilde {history}.

    sağlık sorunları hakkında konuşmak istiyor. 
    yaşına uygun, sağlık geçmişini dikkate alarak dikkatli ve nazik tavsiyeler ver; 
    ismiyle hitap et.
"""
)


#başlangıç mesajını hafızaya kaydet
memory.chat_memory.add_user_message(intro)
print("Merhaba ben bir doktor asistanıyım. Size nasıl yardımcı olabilirim")


#chatbot diyalog döngüsü
while True:
    #kullanıcı mesajı al
    user_msg= input(f"{name}:")

    if user_msg.lower() in ["quit", "q"]:
        print("sana yardımcı olabildiysem ne mutlu bana, görüşmek üzere")
        break

    #llm (chatbot) cevabı
    #modelden (gemini) cevap al
    reply= conversation.predict(input= user_msg)

    #doktor asistanı cevabı yazdır
    print(f"Doktor Asistanı: {reply}")

