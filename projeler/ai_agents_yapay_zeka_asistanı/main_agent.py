"""
problem tanımı:
    *projenin amacı, Google gemini modeli kullanarak  çok araçlı (multi-tool) bir 
    yapay zeka aracı geliştirmek.
    *ajan, kullanıcıdan gelen doğal dil girdilerini anlayıp, uygun aracı seçerek(tool call)
    görevleri otonom şekilde yerine getirecek.
    *ajan, langchain altyapısını kullanacak ve gerçek dünya senaryolarını simüle eden 5 farklı
    yeteneğe(tools) sahip olur
        *RAG: belge ile konuşma
        *calculator: matematiksel hesaplamalar
        *discount tool: indirim hesaplamaları
        *web search(serpAPI): internet üzerinden bilgi getirme
        *memory: konuşma geçmişi hatırlama
    *bu sistem, kullanıcı mesajlarını analiz eder, uuygun aracı seçer, görev sonucunu üretir
    ve yanıtı doğal dilde oluşturur.
    *aynı zamanda geçiş ve geçmiş konuşmalarını hatırlayarak bağlamlı bir sohbet deneyimi sağlar.

kullanılacak teknolojiler:
    *LLM(büyük dil modeli): google gemini (hızlı, düşük maliyetli, metin tabanlı, tool çağırabilen)
    *framework: langchain (ajan yapısı, memory yönetimi, tool entegrasyonu)
    *vektör veritabanı: FAISS (pdf içeriğini vektörleştirerek hızlı benzerlik araması)
    *embedding modeli: LaBSE(çok dilli metin vektörleştirme)
    *API ve Arayüz: fastAPI ve streamlit
    *diğer:
        *web search: serpAPI
        *dotenv
        *requests(client isteği gönderme)

kullanılıcak olan toolar:
    *RAG tool
        *amaç: yüklü pdf belgesinden (sss.pdf) bilgi getirme
        *çalışma: belge FAISS veritabanına çevrilir, kullanıcı sorusu embedding'e çevrilir, 
        en benzer parçalar geri çağrılarak llm modeline bağlam olarak sunulur
    *Discount tool
        *ürün fiyatına %x indirim uygular
        *metinden sayısal fiyat bilgisi çekilir ve 0.x katsayısı ile çarpılır yani indirim uygulanmış olur
    *calculator
        *amaç: temel matematiksel ifadeleri çözmek
        *çalışma: python eval ile yapılacak (langchain kütüphanesinde de bunu yapan hazır bir tool var)
    *web search tool
        *amaç: internet üzerinden bilgi getirme
        *çalışma: serpAPIWrapper ile google üzerinden sorgu yapılır
    *memory
        *amaç: kullanıcının önceki mesajlarını hatırlamak
        *çalışma: ConversationBufferMemory ile önceki geçmiş konuşmaları kaydet

planlama:
    *başlangıç: api anahtarlarının oluştulup okunması
    *tooların hazırlanması
        *her bir tool için ayrı .py dosyalarının tanımlanması
    *ajan oluşturma
        *ajanı tanımla, tooları ajana yani llm'e bağla
    *hafıza yönetimi
        *kullanıcıya özel memory nesnesi oluştur
        *her mesaj sonunda hafıza güncellenir
    *Fast API katmanı
        * /ask endpointi üzerinden json mesaj alınır
    *istemci katmanı(client.py)
        *request modülünü kullanarak /ask endpointine ister atarak test yapılır
    *arayüz katmanı (streamlit)

sistem çalışma akışı(nihai olarak ortaya çıkacak olan sistem özeti):

1) kullanıcı streamlit üzerinden sorgu yapar
2) FastAPI /ask url
3) Agent -> tool seçimi
4) Langchain -> tool çağırısı + LLM reasoning
5) Gemini modeli -> cevap üretimi
6) Memory -> geçmişi saklama
7) yanıt -> fastapi -> streamlit ile kullanıcıya gösterilir

sonuç:
    *bu proje, üretken yapay zeka ajanlarının nasıl "düşünebilen" sistemler haline geldiğini gösteren bir projedir.
    *langchain ve gemini entegrasyonu sayesinde
        *çok araçlı (multi tool)
        *hafızalı (memory)
        *belge tabanlı (RAG)
        *etkileşimli (api arayüz destekli)

        ##mlops devops (docker deployment)
        ##monitor
        ##gpu balancing
    bir akıllı sistem ortaya çıkarılmış olur.

    
pip install langchain langchain-google-genai google-generativeai langchain-community faiss-cpu python-dotenv serpapi streamlit google-search-results pypdf sentence-transformers fastapi uvicorn requests
"""

import os
from dotenv import load_dotenv

from langchain_classic.agents import initialize_agent, AgentType #ajan tanımlamak
from langchain_community.utilities import SerpAPIWrapper #web search
from langchain_community.tools import Tool
from langchain_classic.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.calculator_tool import calculator_tool
from tools.custom_discount_tool import discount_tool
from tools.rag_tool import create_rag_tool


#.env dosyasından API anahtarlarını al
load_dotenv()
GOOGLE_API_KEY= os.getenv("GOOGLE_API_KEY")
SERP_API_KEY= os.getenv("SERP_API_KEY")

#LLM yapılandırması
llm= ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash",
    temperature=0.7,
    google_api_key= GOOGLE_API_KEY
)

#----------------TOOL tanımlama-----------------------
#Tool-1: SerpAPI (genel web araması)
search= SerpAPIWrapper(serpapi_api_key= SERP_API_KEY)

#Tool-2: Memory
memory= ConversationBufferMemory(memory_key= "chat_history", return_messages= True)

#Tool-3: matematiksel işlemler
#şuradan geliyor -> from tools.calculator_tool import calculator_tool

#Tool-4: custom indirim hesaplayıcı

#Tool-5: RAG
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

#ajan oluşturma (zero shot ReAct)
agent= initialize_agent(
    tools= tools, #araçlar
    llm= llm, #beyin
    agent_type= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose= True,
    memory= memory,
    handle_parsing_errors= True
)


#örnek kullanım
if __name__ == "__main__":
    print("Ajan hazır! Konuşmaya başlayabilirsiniz.")

    while True:

        user_input= input("Siz: ")
        if user_input.lower() in ["q"]:
            break

        chat_history= "\n".join([
            f"Kullanıcı: {msg.content}" if msg.type == "human" else f"Asistan: {msg.content}" for msg in memory.chat_memory.messages
        ])

        #geçmiş ve kullanıcı sorusu verilir ve cevap beklenir
        prompt_with_memory= f"{chat_history}\nKullanıcı: {user_input}\nAsistan: "
        response= agent.run(prompt_with_memory)

        #yanıtı hafızaya kaydet
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response)
        print(f"Ajan: {response}\n")



