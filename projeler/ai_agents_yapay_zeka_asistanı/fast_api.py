"""
fast api ile /ask endpoint'i oluşturma
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

#langchain modülleri içeri aktar
from langchain_classic.agents import initialize_agent, AgentType #ajan tanımlamak
from langchain_community.utilities import SerpAPIWrapper #web search
from langchain_community.tools import Tool
from langchain_classic.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.calculator_tool import calculator_tool
from tools.custom_discount_tool import discount_tool
from tools.rag_tool import create_rag_tool


#ortam değişkenlerinin yüklenmeesi
load_dotenv()
GOOGLE_API_KEY= os.getenv("GOOGLE_API_KEY")
SERP_API_KEY= os.getenv("SERP_API_KEY")

#LLM yapılandırması
llm= ChatGoogleGenerativeAI(
    model= "gemini-3.5-flash-lite",
    temperature=0.7,
    google_api_key= GOOGLE_API_KEY
)

#----------------TOOLS-----------------------
#Tool-1: SerpAPI (genel web araması)
search= SerpAPIWrapper(serpapi_api_key= SERP_API_KEY)

#Tool-2: matematiksel işlemler
#Tool-3: custom indirim hesaplayıcı
#Tool-4: RAG
rag_tool= create_rag_tool("data/sss.pdf", llm)

#Tüm tool ları agent'a ekleme
tools=[
    Tool(
        name= "SearchTool",
        func= search.run,
        description= "Google araması yani web araması yapar."
    ),
    calculator_tool,
    discount_tool,
    rag_tool
]


#memory kullanımı

user_memories= {}
def get_user_memory(user_id: str):
    if user_id not in user_memories:
        user_memories[user_id]= ConversationBufferMemory(memory_key= "chat_history", return_messages= True)
    return user_memories[user_id]

#fast api yapılandırması
app= FastAPI(title= "Gemini Çok Araçlı AI Agent API", version= "1.0")

class UserRequest(BaseModel):
    user_id: str
    message: str

#api endpoint
@app.post("/ask")
async def ask_agent(request: UserRequest):

    user_id= request.user_id
    user_input= request.message

    #kullanıcı hafızasını al
    memory= get_user_memory(user_id)

    #geçmiş konuşmayı string haline getir
    chat_history= "\n".join([
        f"Kullanıcı: {msg.content}" if msg.type == "human" else f"Asistan: {msg.content}" for msg in memory.chat_memory.messages
    ])

    prompt_with_memory= f"{chat_history}\nKullanıcı: {user_input}\nAsistan: "

    #ajanı oluştur
    agent= initialize_agent(
        tools= tools,
        llm=llm,
        agent_type= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose= True,
        handle_parsing_error= True
    )

    #agent'tan cevap almaya çalışma
    try:
        response= agent.run(prompt_with_memory)
    except Exception as e:
        raise HTTPException(status_code=500, detail= str(e))

    #soru ve yanıtı hafızaya ekle
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(response)

    return {"user_id":user_id, "response":response}

