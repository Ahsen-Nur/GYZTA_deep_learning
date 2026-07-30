"""
*db işlemleri: build_vector_db.py
    *sözleşme belgesi hazırlama
    *belgeyi okuma, metin çıkarma, parçalama(chunk), embedding ve faiss db oluşturma
"""

import os
import fitz
from sentence_transformers import SentenceTransformer #embedding
import faiss #vektör veri tabanı
import numpy as np
import pickle #vector db'yi kaydetmek için


#.pdf'den metin dönüşümü yapılır
def extract_text_from_pdf(pdf_path):
    """
    pdf dosyasından metin çıkarma
    """

    doc= fitz.open(pdf_path)
    text= ""
    for page in doc:
        text += page.get_text("text")

    return text

#uzun metni daha küçük parçalara böl
def chunk_text(text, max_length=500):
    """
    metni belirtilen karakter uzunluğuna böl
    """
    chunks= []
    current= ""
    for line in text.split("\n"): #Metni satır sonlarından (\n) bölerek bir liste oluşturur.
        if len(current) + len(line) < max_length:
            current += " " + line.strip() #line.strip() Satırın başındaki ve sonundaki boşlukları, \t, \n gibi karakterleri temizler.
        else:
            chunks.append(current.strip())
            current= line.strip()

    if current:
        chunks.append(current.strip())

    print(len(chunks))
    return chunks


#text= extract_text_from_pdf("./data/sozlesme_ornek.pdf")
#print(chunk_text(text))

pdf_file_path= "./data/sozlesme_ornek.pdf"

#pdf den metin çıkarma
text= extract_text_from_pdf(pdf_file_path)

#metni chunklara böl
chunks= chunk_text(text)

model= SentenceTransformer("all-MiniLM-L6-v2")
embeddings= model.encode(chunks)

#print(embeddings.shape)

#faiss index oluştur
dimension= embeddings.shape[1] # 1= embedding vektör boyutu
index= faiss.IndexFlatL2(dimension) #l2 norm (euclidean distance) kullanarak benzerlik arama
index.add(np.array(embeddings)) #embeddingleri indexe kaydeder

#faiss indexi ve chunkları kaydet
faiss.write_index(index, "data/sozlesme_ornek.faiss")
with open("data/sozlesme_ornek.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("faiss index ve chunklar kaydedildi.")