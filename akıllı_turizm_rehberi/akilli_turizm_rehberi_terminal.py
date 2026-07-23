"""
problem:
    *akıllı turizm rehberi(LLM): kullanıcı Türkiye turizmi ile ilgili soru sorar, örneğin 
    Gaziantep'de nereler gezilir, ne yenir?
    *LLM akıllı bir şekilde doğal dilde cevap üretir
    *Google Gemma3 modelini Ollama frameworku Üzerinden kullanma
    *kullanıcı arayüzü(UI) oluşturulacak/streamlit

model tanımı:
    *Google Gemma3: 4b parametreli model
    *local de yani on prem çalışır(bulut ortamına bilgiler gitmez)
    
kütüphanler:
    pip install langchain langchain-community streamlit pydantic requests

ollama:
    *açık kaynaklı bir platform ve büyük dil modellerini yerelde çalıştırma imkanı verir.
    (gerçek hayatta dil modeli kullanılacaksa, eş zamanlı kullanıcı sayısı çok değilse ve gerçek hayatta çalışmasına gerek yoksa tercih edilebilir.)
    *örneğin gemini api bulut üzerinden kullanılır ancak ollama gemma yerelde kullanımı sağlar.
    *llama, mistral, gemma, qwen, deep seek...
    *https://ollama.com/library/gemma3

"""

from langchain.chat_models import ChatOllama #ollama llm arayüzü
from langchain. schema import SystemMessage, HumanMessage #chat mesaj sınıflandırıcı
from langchain.memory import ConversationBufferMemory #konuşma geçmişi için basit hafıza


#llm modeli tanımla
llm= ChatOllama(model= "gemma3:4b")

#memory: konuşma geçmişi takip etmek için
memory= ConversationBufferMemory(return_messages= True)# return_messages= True -> mesajları formatlı dönmesi

#karşılama mesajı (welcome message)
print("Akıllı Turizm Rehberine Hoş Geldiniz")
print("Size gezilecek yerler, tatil önerileri ve ulaşım bilgileri gibi konularda yardımcı olabilirim.")


#terminal üzerinden input al ve llm modeline gönder
#llm modelinin doğal dilde cevap üretmesi
while True:
    user_input= input("Siz: ")

    if user_input.lower() in ["q", "quit", "bye"]:
        print("Program sonlandırıldı.")
        break

    #kullanıcı mesajlarını hafızaya kaydet
    memory.chat_memory.add_user_message(user_input)

    #model için gerekli olan tüm mesajları oluştur: sistem mesajı, memory, human mesajı
    messages= [
        SystemMessage(content= "sen bir turizm rehberisin"
                      "kullanıcılara Türkiye'de ki şehirler, tarihi yerler, yöresel yemekler, ulaşım ve tatil soruları hakkında yardımcı ol")
    ] + memory.load_memory_variables({})["history"] + [HumanMessage(content= user_input)]

    #modelden yanıt al
    response= llm(messages)

    #modein cevabını hafızaya ekle
    memory.chat_memory.add_ai_message(response.content)
    print(f"Akıllı Turizm Rehberi: {response.content}")


#ollama serve