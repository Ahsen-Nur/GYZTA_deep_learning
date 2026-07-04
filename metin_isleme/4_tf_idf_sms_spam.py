"""
SMS spam veri seti üzerinden tf-idf analizi

* csv dosyasından veri okuma
* tf-idf vektorizer ile sms verisini sayısal vektörlere dönüştür
* her kelimenin ortalama tf-idf değerini hesapla
* sonucları df e aktar ve en yüksek skora sahip 10 kelimeyi listele
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


veri = pd.read_csv("sms_spam.csv")
print(veri.head())

mesajlar = veri["text"]

#tf-idf vektörizer nesnesi oluştur
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# mesajları sayısal vektörlere dönüştür
mesaj_vektorleri = tfidf_vectorizer.fit_transform(mesajlar)

#vocabulary
kelime_kümesi = tfidf_vectorizer.get_feature_names_out()

#her kelimenin ortalama tf-idf değeri
ortalama_tfidf = mesaj_vektorleri.mean(axis=0).A1

#ortalama tf-idf değerlerini df e aktar
df_tfidf = pd.DataFrame({"kelime": kelime_kümesi, "ortalama_tfidf": ortalama_tfidf})
df_tfidf_sirali = df_tfidf.sort_values(by="ortalama_tfidf", ascending=False)

print(df_tfidf_sirali.head(10))