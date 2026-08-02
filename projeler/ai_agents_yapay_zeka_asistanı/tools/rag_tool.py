"""
RAG: bilgi getirme yapılır, sss.pdf kullanılır
"""

import os
from langchain_community.document_loaders import PyPDFLoader #pdf dosyası yükleme
from langchain_community.tools import Tool
from langchain_text_splitters import RecursiveCharacterTextSplitter #metni chunklara(parça) ayırır
from langchain_community.embeddings import HuggingFaceEmbeddings #embedding oluşturma
from langchain_community.vectorstores import FAISS #vektör db
from langchain_community.chains import ConversationalRetrievalChain #zincir oluşturma


def create_rag_tool(pdf_path: str, llm):
    """
    Belirtilen PDF belgesini yükleyip FAISS ile arama yapılabilir hale getirir. 
    """

    print("RAG başlatıldı.")

    #pdf yükle
    loader= PyPDFLoader(pdf_path)
    documents= loader.load()

    #metni paarçalara ayır(chunks)
    splitter= RecursiveCharacterTextSplitter(
        chunk_size= 1000,
        chunk_overlap= 150
    )
    texts= splitter.split_documents(documents)

    #embedding
    embeddings= HuggingFaceEmbeddings(model_name= "sentence-transformers/LaBSE")

    #FAISS vektör veritabanı oluştur
    db= FAISS.from_documents(texts, embeddings)
    retriever= db.as_retriever(search_kwargs= {"k": 3})

    #rag zinciri oluşturma
    rag_chain= ConversationalRetrievalChain(
        llm= llm,
        retriever= retriever,
        return_source_documents= False
    )

    #tool fonksiyonu
    def rag_query(query: str):
        """
        soruya göre PDF'ten en alakalı bilgiyi getirir.
        """
        
        response= rag_chain.run({"question": query, "chat_history": []})
        return f"Belgede bulunan bilgi: {response}"

    return Tool(
        name=  "RAGTool",
        func= rag_query,
        description= "Müşteri destek SSS belgesinden bilgi araması yapar."
    )

#def create_rag_tool ve def rag_query ayrı dosyalarda da yapılabilir