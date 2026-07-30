"""
problem tanımı: sözleşme inceleme asistanı
    *kullanıcının yüklediği bir sözleşme dokümanından bilgi çıkarımını gerçekleştirme
    *içeriği vektörel olarak temsil etme(embedding)
    *faiss ile hızlı arama yapabilen bir vektör veri tabanı oluşturma
    *kullanıcıdan soruları alır, db'den bilgileri alıp getirir. Bu bilgi ve kullanıcı sorusu doğrultusunda gemini cevap üretir

teknolojiler:
    *embedding: metni vektörleştirme
    *faiss: hızlı benzerlik araması için veritabanı
    *gemini(llm): dil modeli için gemini-3.1-flash

RAG(Retrieval Augmented Generation): dil modellerine bilgi sağlayan teknik
    *kullanıcı sorusunu alır, ilgili bilgiyi veri tabanından çeker, gemini ile cevap üretir
    *retrieval:
        *kullanıcı sorusunu sorar -> embedding ile vektörleştirilir
        *faiss(db) üzerinden en alakalı içerik(chunk(metin parçaları)) getirilir
    *augmentation: zenginleştirme, kullanıcı sorusu + prompt + getirilen bilgi
    *generation: dil modeli bilgiler ile mantıklı yanıt üretir

planlama:
    *db işlemleri: build_vector_db.py
        *sözleşme belgesi hazırlama
        *belgeyi okuma, metin çıkarma, parçalama(chunk), embedding ve faiss db'de depolama
    *soru-cevap sistemi: main.py
        *kullanıcı sorusunu sorar, embedding yapılır, RAG yapılır

pip install google-generativeai python-dotenv sentence-transformers faiss-cpu numpy PyMuPDF
"""

import os
import pickle
import faiss #vektör db
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google.generativeai.generative_models import GenerativeModel
import google.generativeai as genai


load_dotenv()

api_key= os.getenv("GOOGLE_API_KEY")
genai.configure(api_key= api_key)

#initialize gemini model
model_gemini= GenerativeModel("gemini-3.0-flash")

embedding_model= SentenceTransformer("all-MiniLM-L6-v2")

#faiss index dosyasını yükle(önceden oluşturulmuş vektör veri tabanı)
index= faiss.read_index("./data/sozlesme_ornek.faiss")

#chunklanmış metin verisini yukle
with open("data/sozlesme_ornek.pkl", "rb") as f:
    chunks= pickle.load(f)


#kullanıcıdan soru al
while True:

    question= input("sorunuzu girin(eng): ")

    if question.lower() in ["q"]:
        print("çıkış yapılıyor.")
        break

    #sorulara embedding uygula(vektörleştir)
    question_embedding= embedding_model.encode([question])

    #faiss veri tabanından en yakın 3 chunk aranır ve getirilir
    k=3
    distances, indices= index.search(np.array(question_embedding), k)

    #bulunan chunkları birleştir ve context(bağlam) oluştur
    retrieved_chunks= [chunks[i] for i in indices[0]]
    context= "\n --- \n".join(retrieved_chunks)

    #llm'e gönderilecek sistem prompt
    prompt= f"""
            you are a contract lawyer AI assistant. based on the contract below,
            answer the user's question clearly.

            context:
            {context}

            question:
            {question}

            answer:
            """

    response= model_gemini.generate_content(prompt)
    print(f"AI: {response.text.strip()}")



"""
Q1: Who pays for the Gemini API token costs during production?
A1: The CLIENT is responsible for setup and ongoing operational Gemini API costs, while the CONTRACTOR is only responsible for integration.

Q2: Who is liable if the AI chatbot provides inaccurate or harmful responses (hallucinations)?
A2: Due to the probabilistic nature of LLMs, the CONTRACTOR cannot be held liable for unexpected AI outputs, provided the RAG architecture is integrated correctly.

Q3: Is post-launch maintenance included in the 5,000 USD budget?
A3: No, the budget covers initial development and delivery only. Post-delivery maintenance and SLA support require a separate agreement.

Q4: What happens if Gemini API updates require changes to the system architecture?
A4: Any major refactoring caused by external API structural changes after project delivery will be billed separately as additional scope.
"""





