"""
problem tanımı: 
    *akıllı müşteri destek sistemi: sık sorulan sorulara yanıt verecek, belgeye dayalı yanıt sistemi kurulacak
    *müşteriler sık sık benzer sorular sorarlar
        *şifremi unuttum
        *faturamı nereden alabilirim
        *iade süresi kaç gün
        *yurt dışına gönderim yapıyor musunuz
    *çözüm
        *pdf(db, text, json, ...) dosyası formatında sıkça sorulan sorular vektör veri tabanına dönüştürülür.
        *kullanıcıdan gelen sorular veri tabanında sorgulanır ve gemma(llm) türkçe cevaplar üretir
teknolojiler:
    *langchain: rag mimari kurmak için
    *faiss: embeddingleri saklamak için
    *ollama: gemma3:4b soru-cevap llm için
    *streamlit: web arayüzü, son kullanıcı ile interaktif kullanıcı deneyimi
veri seti:
    *gemini dan faydalanılacak
plan:
    *sss içeren bir pdf oluşturulur (kullanıcı bu dosyayı arayüzden yükler)
    *pdf metni chunklara ayrılır, embeddingler çıkarılır
    *kullanıcı soru sorduğunda vektör db'den benzer içerikler getirlir, gemma ile cevap oluşturulur
    *memory ile konuşma geçmişi saklanır ve sonraki yanıtlara bağlam oluşturulur.

pip install langchain langchain-community sentence-transformers faiss-cpu pypdf ollama streamlit
    
"""

from langchain_ollama import ChatOllama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings 
import os


#embedding modelini başlat (text->vektör)
embedding= HuggingFaceEmbeddings(model_name= "sentence-transformers/LaBSE")

#vektör db yükle
vectordb= FAISS.load_local(
    "sss_store",
    embedding,
    allow_dangerous_deserialization= True #pkl formatında veri yüklerken hata oluşmasını engeller
)

#memory
memory= ConversationBufferMemory(
    memory_key= "chat_history",
    return_messages= True
)

#llm
llm= ChatOllama(
    model= "gemma3:4b",
    temperature=0.2 #garantici, yaratıcı değil
)

qa_chain= ConversationalRetrievalChain.from_llm(
    llm= llm,
    retriever= vectordb.as_retriever(search_kwargs= {"k":3}), #db den 3 bilgi getirilir
    memory= memory,
    verbose= True
)

#test
print("Müşteri Destek Botuna Hoş Geldiniz.")
while True:
    user_input= input("siz: ")
    if user_input.lower() == "q":
        break
    response= qa_chain.run(user_input)
    print(f"Müşteri Destek Botu: {response}")
