import streamlit as st
import requests

#fast api bağlantı ayarları
API_URL= "http://127.0.0.1:8000/ask"
USER_ID= "nur"

#streamlit ile sayfa ayarları
st.set_page_config(page_title= "Gemini AI Agent Chat", page_icon="🤖", layout="centered")
st.title("🤖 Gemini Çok Amaçlı AI Agent ")
st.caption("Gemini modeli + RAG + Math + Discount + Memory + Web Search")

#session state (sohbet geçmişi)
if "messages" not in st.session_state:
    st.session_state["messages"]= []

#mesaj gönderme fonksiyonu
def send_message_to_api(message: str):
    payload= {"user_id": USER_ID, "message": message} #API'ye gönderilecek json verisi

    try:
        response= requests.post(API_URL, json= payload)
        if response.status_code == 200:
            data= response.json()
            return data.get("response", "Sunucudan yanıt alınamadı.")
        else:
            return f"hata ({response.status_code}): {response.text}"
    except Exception as e:
        return f"bağlantı hatası: {e}"


#streamlit ile sohbet arayüzü
user_input= st.chat_input("bir mesaj yazın...")

if user_input:
    #kullanıcının mesajını session state'e ekle
    st.session_state["messages"].append({"role":"user", "content": user_input})

    #API'ye gönder
    with st.spinner("ajan düşünüyor..."):
        response= send_message_to_api(user_input)

    #ajan cevabını ui üzerine ekle
    st.session_state["messages"].append({"role":"assistant", "content": response})


#mesajları görüntüle
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**siz:** {msg["content"]}")

    #asistan(ai)
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**ajan:** {msg["content"]}")


#streamlit run app_streamlit.py