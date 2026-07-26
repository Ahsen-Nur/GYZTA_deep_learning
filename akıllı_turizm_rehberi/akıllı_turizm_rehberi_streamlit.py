"""
web üzerinde çalışan chatbot ekranı geliştirme
streamlit freamwork
"""

import akıllı_turizm_rehberi_streamlit as st
from langchain.chat_models import ChatOllama #ollama üzerinden gemma çağırmak için
from langchain.schema import SystemMessage, HumanMessage #sobet mesajları
from langchain.memory import ConversationBufferMemory 


#başlık ve açıklamalar 
st.set_page_config(page_title= "Akıllı Turizm Rehberi", page_icon= "🌍")
st.title("🌍 Akıllı Turizm Rehberi")
st.markdown("Türkiye'nin dört bir yanındaki turistik yerler hakkında bilgi edinmek için sorular sorabilirsiniz")


#session state (streamlit de kullanıcı geçmişini tutar)
if "memory" not in st.session_state:
    st.session_state.memory= ConversationBufferMemory(return_messages= True)


#ollama ile gemma3:4b parametreli modelini yükle
llm= ChatOllama(model= "gemma3:4b")


#mesaj kutusu: kullanıcıdan gelen mesaj
user_input= st.chat_input("Bir şehir, mekan ya da yemek aktivitesi sorabilirsiniz.")

if user_input:

    #yeni gelen kullanıcı mesajını ilk olarak memory e yükle
    st. session_state.memory.chat_memory.add_user_message(user_input)

    #tüm konuşmayı modele verecek şekilde mesajları oluştur: sistem mesajı + memory + human message
    messages= [
        SystemMessage(content= "sen bir turizm rehberisin"
                      "kullanıcıların Türkiye'deki şehirler, tarihi yerler yöresel yemekler, ulaşım ve tatil önerileri yap")
    ] + st.session_state.memory.load_memory_variables({})["history"] + [HumanMessage(content= user_input)]

    #modelden yanıt al
    response= llm(messages)

    #yanıtı hafızaya kaydet
    st.session_state.memory.chat_memory.add_ai_message(response.content)


#sohbet geçmişini UI da göster
for msg in st.session_state.memory.chat_memory.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("Kullanıcı: "):
            st.markdown(msg.content)

    else:
        with st.chat_message("Akıllı Rehber: "):
            st.markdown(msg.content)





#streamlit run akıllı_turizm_rehberi_streamlit.py