"""
*kütüphaneler içeri aktarılacak
*sss pdf oluştur ve yükle
*chunkları oluştur
*embedding uygula
*vektör db oluştur
*vektör db'yi kaydet
"""


from langchain_community.embeddings import HuggingFaceEmbeddings  # mevcut langchain kurulumunda çalışan import yolu
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


loader= PyPDFLoader("sss.pdf")
documents= loader.load() #langchain document objesi oluştur
#print(documents)

#metin parçalama
#splitter(metni anlamlı parçalara ayırırken cümle ve paragraf bütünlüğünü korumaya çalışan bir yöntem)
text_splitter= RecursiveCharacterTextSplitter(
    chunk_size= 500,
    chunk_overlap= 50
)

#chunkları oluştur
docs= text_splitter.split_documents(documents)

#türkçe için labse embedding yöntemi
embedding= HuggingFaceEmbeddings(
    model_name= "sentence-transformers/LaBSE"
)

#parçalara ayrılmış metni embedding ile vektörleştir, indeks oluştur ve faiss de depola
vectordb= FAISS.from_documents(docs, embedding)

#oluşturulan vektör db'yi yerel diske kaydet
vectordb.save_local("sss_store")
print("embedding ve vektör veri tabanı oluştuurldu.")
