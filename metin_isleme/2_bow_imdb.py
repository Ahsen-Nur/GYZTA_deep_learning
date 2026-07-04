"""
- csv'den veri okuma
- text cleaning(küçük harfe çevirme, rakam-özel karakterler kaldırma, stop word'leri kaldırma)
- BoW vektörleştirme
- kelime frekanslarını hesapla ve en sık kullanılan 5 kelimeyi listele
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


data = pd.read_csv("IMDB Dataset.csv")
print(data.head())

yorumlar = data["review"]
etiketler = data["sentiment"]

def temizle(text):
    # Küçük harfe çevirme
    text = text.lower()
    #rakamları kaldırma
    text = re.sub(r"\d+", "", text)
    #özel karakterleri kaldırma
    text = re.sub(r"[^\w\s]", "", text)
    #kısa kelimeleri kaldırma(2 harften kısa olanlar)
    text = " ".join([word for word in text.split() if len(word) > 2])
    #stop word'leri kaldırma
    text = " ".join(
        [word for word in text.split()
         if word not in ENGLISH_STOP_WORDS]
    )

    return text

temizlenmis_yorumlar = [temizle(y) for y in yorumlar]

#BoW
vectorizer = CountVectorizer()

#ilk 75 yorumu sayısal vektörlere dönüştür
yorum_vektorleri = vectorizer.fit_transform(temizlenmis_yorumlar[:75])

#vocabulary
kelime_kümesi = vectorizer.get_feature_names_out()

#to narray
vektor_temsili = yorum_vektorleri.toarray()
print(f"vektör temsili: {vektor_temsili}")

#vektör temsillerini dataframe'e çevirme
df_bow = pd.DataFrame(vektor_temsili, columns=kelime_kümesi)
print(df_bow.head())

#kelime frekanslarını hesaplama
kelime_sayilari = yorum_vektorleri.sum(axis=0).A1
kelime_frekansi = dict(zip(kelime_kümesi, kelime_sayilari))

#en sık kullanılan 5 kelime
en_sik_kelimeler = Counter(kelime_frekansi).most_common(5)
print(f"En sık kullanılan 5 kelime: {en_sik_kelimeler}")

