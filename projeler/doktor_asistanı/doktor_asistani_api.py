"""
Fast API ile gemini 2.5 flash doktor asistanını bir web servise çevir.
Her kullanıcı için ayrı bir memory yani sohbet geçmişi tutulacak.
"""

import os
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel #istek ve yanıt şemaları
from dotenv import load_dotenv #.env den google api key okumak için
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory #sohbet geçmişi
from langchain.chains import ConversationChain #hafıza+model zinciri


#ortam değişkenini yükle
load_dotenv()
api_key= os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("api key bulunamadı")


#fast api uygulaması başlat
app= FastAPI(title= "Gemini 2.5 Doktor Asistanı API")


#llm tanımla
llm= ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash",
    temperature= 0.7, #cevap çeşitliliği (0: kesin | 1:yaratıcı)
    google_api_key= api_key

)


#her kullanıcı için memory yapılandırması
user_memories:Dict[str, ConversationBufferMemory]= {}


#istek ve yanıt şemaları oluştur
class ChatRequest(BaseModel): #kullanıcının gönderdiği mesaj
    name: str
    age: int
    message: str

class ChatResponse(BaseModel): #modelin dönderiği yanıt
    response: str


#sohbet endpoint'i oluştur
@app.post("/chat", response_model= ChatResponse)
async def chat_with_doctor(request: ChatRequest):
    try:
        #kullanıcıya özel hafıza oluştur
        if request.name not in user_memories: #eğer hafızada bu kullanıcı yoksa yeni hafıza oluştur
            user_memories[request.name]= ConversationBufferMemory(return_messages= True)

        memory= user_memories[request.name]#zaten hafızada bu kullanıcı varsa kullanıcıyı getir

        #ilk konuşma(eğer memory boşsa giriş bağlamı ekle)
        if len(memory.chat_memory.messages) == 0:
            intro= (f"""
                        sen bir doktor asistanısın. 
                        hasta {request.name}, {request.age} yaşında. 

                        sağlık sorunları hakkında konuşmak istiyor. 
                        yaşına uygun, sağlık geçmişini dikkate alarak dikkatli ve 
                        nazik tavsiyeler ver; ismiyle hitap et.
                    """
                    )
            
            memory.chat_memory.add_user_message(intro)

        #hafıza ve modeli birleştir -> sohbet zinciri oluştur
        conversation= ConversationChain(
            llm= llm,
            memory= memory,
            verbose= True #arka plandaki işlem akışını terminalde göster
        )

        #modelden yanıt al
        reply= conversation.predict(input= request.message)

        #terminale hafızayı yazdır
        print(f"memory for{request.name}")
        for idx, m in enumerate(memory.chat_memory.messages, start=1):
            print(f"{idx:02d}. {m.type.upper()}: {m.content}")
        print("----------------------------------------------------")

        #api yanıtını return et
        return ChatResponse(response=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=(e))


#uvicorn doktor_asistani_api:app --reload


    
