import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter #chunk oluşturma
from langchain_classic.chains import ConversationalRetrievalChain #chain= llm+memory+vektordb
from langchain_classic.memory import ConversationBufferMemory
import os
import tempfile #geçici dosya işlemleri için


#streamlit ile sayfa başlığı ve ikon 
st.set_page_config(page_title= "Müşeteri Destek Botu", page_icon= "📄")
st.title("Müşteri Destek Botu (RAG + Memory)")
st.write("Bir PDF yükleyin, içeriğine dair sorular sorun. Türkçe desteklidir.")


uploaded_file= st.file_uploader("PDF dosyanızı yükleyin.", type= "pdf", key= "pdf_uploader")

#eğer kullanıcı yeni bir pdf yüklediyse ve daha önceki yüklenen ile aynı değilse
if uploaded_file is not None:
    if "last_uploaded_name" not in st.session_state or uploaded_file.name != st.session_state.last_uploaded_name:
        #kullanıcıya işleniyor bilgisi gönder
        with st.spinner("pdf işleniyor..."):
            #yüklenen pdf i geçici bir dosyaya yazıdr
            with tempfile.NamedTemporaryFile(delete= False, suffix= ".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path= tmp.name #geçici dosyanın yolu

            #pdf yükle
            loader= PyPDFLoader(tmp_path)
            documents= loader.load()

            #metinleri chunklara böl
            splitter= RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap= 50)
            docs= splitter.split_documents(documents)

            #LaBSE embedding
            embedding= HuggingFaceEmbeddings(model_name= "sentence-transformers/LaBSE")

            #faiis ile vektör db 
            vectordb= FAISS.from_documents(docs, embedding)

            #memory + llm
            memory= ConversationBufferMemory(memory_key= "chat_history", return_messages= True)
            llm= ChatOllama(model= "gemma3:4b", temperature= 0.2)

            #rag+memory chain
            qa_chain= ConversationalRetrievalChain(
                llm= llm,
                retriever= vectordb.as_retriever(search_kwargs= {"k":3}),
                memory= memory
            )

            st.session_state.qa_chain= qa_chain
            st.session_state.chat_history= []
            st.session_state.last_uploaded_name= uploaded_file.name
        st.success("PDF başarıyla işlendi.")


if "qa_chain" in st.session_state: #eğer pdf işelndiyse
    #kullanıcı sorusunu al
    user_question= st.text_input("sorunuzu yazın: ")

    if user_question:
        response= st.session_state.qa_chain.invoke({"question": user_question}) #langchain zincirine soruyu gönder
        st.session_state.chat_history.append(("🧑🏻", user_question)) #kullanıcı mesajını geçmişe ekleme
        st.session_state.chat_history.append(("🤖", response["answer"])) #model/bot cevabını geçmişe ekleme

    if st.session_state.chat_history:
        st.subheader("sohbet geçmişi")
        for sender, msg in st.session_state.chat_history:
            st.markdown(f"**{sender}**: {msg}")




#streamlit run atreamlit_app.py